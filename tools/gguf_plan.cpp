// tools/gguf_plan.cpp
//
// gguf_tool plan: a read-only sizing advisor for a GGUF model + the local
// machine, so a user can pick dense_gb / moe_cache_gb / pinned_vram_capacity
// / pin_topk_per_layer for a NEW model without trial-and-error real runs.
// Reports:
//   - dense (always-VRAM-resident) tensor requirement, by category, with a
//     worst-case (nothing BF16-converted) and best-case (attn_output fully
//     BF16-converted) GB range -- mirrors core/main.cpp's REAL categorization
//     (is_dense_offload_candidate / is_attn_output / categorize_dense_tensor)
//     byte-for-byte, so these numbers agree with what a real run's
//     [VRAM-BREAKDOWN] log would show for the same model.
//   - MoE routed-expert requirement: per-expert slab bytes (gate+up+down),
//     computed the same way as engine.hpp's compute_expert_slab_bytes, plus
//     a GB->slots quick-reference table.
//   - detected system config: GPU name + total VRAM per device (via
//     `nvidia-smi`, best-effort -- absent on macOS / no-NVIDIA boxes, not an
//     error) and total system RAM (/proc/meminfo on Linux, `sysctl` on
//     macOS).
//   - .moestream_usage summary, if present: how many (layer,expert) pairs
//     have been seen, and per-layer top-K concentration (how much of a
//     layer's traffic its K hottest experts cover) -- the same metric
//     core/main.cpp's [PIN] log already computes for a live run, surfaced
//     here BEFORE a run exists (or from a prior one) to size pin_topk_per_layer.
//
// Deliberately advisory only -- prints numbers, does not write a config
// file. Different models (this tool's whole reason to exist) need
// different dense_gb/moe_cache_gb; a generated TOML would need to guess at
// things this tool has no way to know (how many GPUs the user wants,
// which role goes where, KV cache target context) -- see CLAUDE.md's GPU
// role-assignment findings for why that decision needs a human anyway.
// KV cache sizing is intentionally NOT estimated here: it depends on
// architecture constants (kv_lora/qk_rope for MLA, or head_dim/n_head_kv for
// dense attention) that live in GGUF metadata KVs this lightweight,
// name-pattern-only tool doesn't parse (see graph_initializer.cpp, which is
// engine-side and intentionally not linked into this zero-dependency
// binary) -- check a real run's own "[GRAPH] KV cache: ... GB" log line
// instead.
//
// Build: see gguf_common.hpp (compiled together with the other gguf_*.cpp).
#include "gguf_common.hpp"
#include <algorithm>
#include <array>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace gguf_tool {

namespace {

// ---- dense-tensor categorization, copied verbatim (same predicates, same
// category names) from core/main.cpp so these numbers agree with a real
// run's [VRAM-BREAKDOWN] log for the same model. Keep these three in sync
// with main.cpp if the engine's own categorization ever changes. ----

bool is_dense_offload_candidate(const std::string &name) {
  // attn_output_a (DeepSeek-V4's grouped output projection, first stage)
  // gets the SAME treatment as attn_k_b/attn_v_b in the real engine
  // (dsv4_attention_gguf, attention_layer.cpp): dequant_row'd once into a
  // static per-layer cache and consumed via raw CPU matmul(), never
  // through dense_proj/device_ptr -- excluded for the identical reason.
  if (name.find("attn_k_b.weight") != std::string::npos ||
      name.find("attn_v_b.weight") != std::string::npos ||
      name.find("attn_output_a.weight") != std::string::npos) {
    return false;
  }
  // DeepSeek-V4 Pro CSA/HCA compressor and Lightning Indexer weights are
  // resident tensors (graph_initializer.cpp's compressed-attention
  // manifest). Compressor names do not contain "attn", so include them
  // explicitly. This does not alter the unrelated GLM DSA indexer policy:
  // its attn tensors are still excluded upstream from the manifest.
  if (name.find("compressor_") != std::string::npos ||
      name.find("indexer.proj.weight") != std::string::npos) {
    return true;
  }
  // engram_q/engram_k/engram_wkv (DeepSeek-V4.1's Engram gate/projection
  // weights, NOT engram_embd -- that giant hash-embedding table is
  // deliberately kept mmap'd on NVMe, see deepseek41_engram.cpp) -- same
  // gap fixed in core/main.cpp's own predicate, found investigating a real
  // CPU-fallback cost these small resident tensors were paying.
  if (name.find("engram_wkv.weight") != std::string::npos ||
      name.find("engram_q.weight") != std::string::npos ||
      name.find("engram_k.weight") != std::string::npos) {
    return true;
  }
  // hc_ catches DeepSeek-V4's Hyper-Connections weights (6 per-layer +
  // 3 global output_hc_*) -- see core/main.cpp's own comment on this same
  // predicate for the full rationale.
  if (name.find("hc_") != std::string::npos) return true;
  // ssm_/ple_ catch Qwen4Exp's GDN/PLE resident tensors -- see main.cpp's
  // own comment on this same predicate.
  if (name.find("ssm_") != std::string::npos ||
      name.find("ple_") != std::string::npos) return true;
  return name.find("attn") != std::string::npos ||
         name.find("norm") != std::string::npos ||
         name.find("gate_inp") != std::string::npos ||
         name.find("shexp") != std::string::npos ||
         name.find("ffn_gate.weight") != std::string::npos ||
         name.find("ffn_up.weight") != std::string::npos ||
         name.find("ffn_down.weight") != std::string::npos ||
         name == "token_embd.weight" || name == "output.weight" ||
         name == "output_norm.weight";
}

bool is_attn_output(const std::string &name) {
  // attn_output_b is V4's second-stage (single dense matmul) analog of
  // attn_output.weight -- attn_output_a never reaches this check (excluded
  // above).
  return name.find("attn_output.weight") != std::string::npos ||
         name.find("attn_output_b.weight") != std::string::npos;
}

const char *categorize_dense_tensor(const std::string &name) {
  if (name.find("attn_output.weight") != std::string::npos ||
      name.find("attn_output_b.weight") != std::string::npos)
    return "attn_output";
  if (name.find("hc_") != std::string::npos)
    return "hyper_connections (hc_*)";
  if (name.find("compressor_") != std::string::npos ||
      name.find("indexer.") != std::string::npos)
    return "compressed_attention (CSA/HCA/indexer)";
  if (name.find("engram_wkv.weight") != std::string::npos ||
      name.find("engram_q.weight") != std::string::npos ||
      name.find("engram_k.weight") != std::string::npos)
    return "indexer/engram (DeepSeek-V4.1 gate projections)";
  if (name.find("shexp") != std::string::npos)
    return "shared_expert (shexp)";
  if (name.find("ffn_gate.weight") != std::string::npos ||
      name.find("ffn_up.weight") != std::string::npos ||
      name.find("ffn_down.weight") != std::string::npos)
    return "leading_dense_ffn (pre-MoE layers)";
  if (name == "token_embd.weight" || name == "output.weight" ||
      name == "output_norm.weight")
    return "embedding + LM head";
  if (name.find("gate_inp") != std::string::npos)
    return "moe_router (gate_inp)";
  if (name.find("norm") != std::string::npos)
    return "norms";
  if (name.find("attn") != std::string::npos)
    return "other_attention (q/k/v/kv_a projections)";
  return "other";
}

struct CategoryTotal {
  size_t bytes = 0;
  int count = 0;
};

// Run a shell command and capture stdout. Used only for best-effort system
// detection (nvidia-smi / sysctl) -- absence or failure is expected on many
// boxes (no NVIDIA GPU, non-macOS-non-Linux), never treated as an error.
std::string run_capture(const std::string &cmd) {
  std::string out;
  FILE *p = popen((cmd + " 2>/dev/null").c_str(), "r");
  if (!p)
    return out;
  char buf[4096];
  size_t n;
  while ((n = fread(buf, 1, sizeof(buf), p)) > 0)
    out.append(buf, n);
  pclose(p);
  return out;
}

std::vector<std::string> split_csv(const std::string &line) {
  std::vector<std::string> out;
  std::stringstream ss(line);
  std::string field;
  while (std::getline(ss, field, ',')) {
    size_t a = field.find_first_not_of(" \t\r");
    size_t b = field.find_last_not_of(" \t\r");
    out.push_back(a == std::string::npos ? "" : field.substr(a, b - a + 1));
  }
  return out;
}

void print_detected_gpus() {
  std::string out = run_capture(
      "nvidia-smi --query-gpu=index,name,memory.total "
      "--format=csv,noheader,nounits");
  if (out.empty()) {
    printf("  nvidia-smi not found or no GPUs reported -- run this on the "
           "target machine, or size manually from known card specs.\n");
    return;
  }
  std::stringstream ss(out);
  std::string line;
  while (std::getline(ss, line)) {
    if (line.empty())
      continue;
    auto f = split_csv(line);
    if (f.size() < 3)
      continue;
    double mib = std::atof(f[2].c_str());
    printf("  GPU %s: %-30s %.1f GB total VRAM\n", f[0].c_str(),
           f[1].c_str(), mib / 1024.0);
  }
}

void print_detected_ram() {
  // Linux: /proc/meminfo's MemTotal is in kB.
  std::ifstream meminfo("/proc/meminfo");
  if (meminfo) {
    std::string line;
    while (std::getline(meminfo, line)) {
      if (line.rfind("MemTotal:", 0) == 0) {
        unsigned long long kb = 0;
        std::sscanf(line.c_str(), "MemTotal: %llu kB", &kb);
        if (kb > 0) {
          printf("  System RAM: %.1f GB total (from /proc/meminfo)\n",
                 kb / 1e6);
          return;
        }
      }
    }
  }
  // macOS fallback: sysctl hw.memsize is in bytes.
  std::string out = run_capture("sysctl -n hw.memsize");
  if (!out.empty()) {
    double bytes = std::atof(out.c_str());
    if (bytes > 0) {
      printf("  System RAM: %.1f GB total (from sysctl hw.memsize)\n",
             bytes / 1e9);
      return;
    }
  }
  printf("  System RAM: could not detect (neither /proc/meminfo nor sysctl "
         "worked) -- check manually (free -h / top).\n");
}

// .moestream_usage is a plain "layer expert count" text file (one line per
// ever-selected (layer,expert) pair, written by StreamingEngine::dump_usage
// in core/engine.hpp). Re-implemented here rather than including
// engine.hpp directly, to keep this tool decoupled from the engine's own
// (CUDA-linked) headers -- same reasoning as gguf_verify.cpp's own
// from-scratch completeness check.
struct UsageEntry {
  int layer;
  int expert;
  unsigned long long count;
};

std::vector<UsageEntry> load_usage_file(const std::string &path) {
  std::vector<UsageEntry> out;
  std::ifstream f(path);
  if (!f)
    return out;
  std::string line;
  while (std::getline(f, line)) {
    int layer, expert;
    unsigned long long cnt;
    if (std::sscanf(line.c_str(), "%d %d %llu", &layer, &expert, &cnt) == 3)
      out.push_back({layer, expert, cnt});
  }
  return out;
}

void print_usage_summary(const std::string &path) {
  auto entries = load_usage_file(path);
  if (entries.empty()) {
    printf("  No usage history at '%s' yet (first run, or pin_file unset) -- "
           "pinning has nothing to prioritize until at least one real run "
           "writes this file.\n",
           path.c_str());
    return;
  }
  std::map<int, std::vector<unsigned long long>> per_layer;
  unsigned long long global_max = 0;
  for (auto &e : entries) {
    per_layer[e.layer].push_back(e.count);
    global_max = std::max(global_max, e.count);
  }
  printf("  %zu (layer,expert) pairs known, %zu distinct layer(s), max "
         "single-expert hit count %llu.\n",
         entries.size(), per_layer.size(), global_max);

  // Top-K coverage: for each layer, what fraction of that layer's TOTAL
  // recorded hits its K hottest experts account for. Same concept as
  // core/main.cpp's [PIN] concentration log, computed here so it's visible
  // before pin_topk_per_layer is even chosen for a first real run.
  for (int k : {8, 11, 16}) {
    double min_cov = 1.0, max_cov = 0.0, sum_cov = 0.0;
    int n_layers_considered = 0;
    for (auto &[layer, counts] : per_layer) {
      std::vector<unsigned long long> sorted = counts;
      std::sort(sorted.rbegin(), sorted.rend());
      unsigned long long total = 0;
      for (auto c : sorted)
        total += c;
      if (total == 0)
        continue;
      unsigned long long top_sum = 0;
      for (int i = 0; i < k && i < (int)sorted.size(); ++i)
        top_sum += sorted[i];
      double cov = (double)top_sum / (double)total;
      min_cov = std::min(min_cov, cov);
      max_cov = std::max(max_cov, cov);
      sum_cov += cov;
      ++n_layers_considered;
    }
    if (n_layers_considered == 0)
      continue;
    printf("  top-%d/layer coverage of recorded traffic: min %.1f%%, avg "
           "%.1f%%, max %.1f%% (across %d layer(s))\n",
           k, min_cov * 100.0, 100.0 * sum_cov / n_layers_considered,
           max_cov * 100.0, n_layers_considered);
  }
  printf("  Low coverage at your chosen K means a flat/wide routing "
         "distribution -- pinning more per layer won't help much until this "
         "grows richer with more real traffic. High coverage means a small "
         "top-K genuinely captures most of what that layer needs.\n");
}

} // namespace

int cmd_plan(int argc, char **argv) {
  if (argc < 2) {
    fprintf(stderr,
            "Usage: %s <model_file_or_dir> [--usage-file <path>]\n"
            "\n"
            "Read-only sizing advisor: prints dense-tensor VRAM requirement\n"
            "(by category, worst/best-case BF16 range), MoE expert slab\n"
            "size + a GB->slots table, this machine's detected GPU(s) and\n"
            "system RAM, and (if present) a .moestream_usage summary --\n"
            "everything needed to pick dense_gb / moe_cache_gb /\n"
            "pinned_vram_capacity / pin_topk_per_layer for a NEW model\n"
            "without trial-and-error real runs. Does not write a config\n"
            "file -- prints numbers only.\n",
            argv[0]);
    return 2;
  }
  std::string model = argv[1];
  std::string usage_file = ".moestream_usage";
  for (int i = 2; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--usage-file" && i + 1 < argc)
      usage_file = argv[++i];
    else {
      fprintf(stderr, "[plan] unknown arg: %s\n", a.c_str());
      return 2;
    }
  }

  GGUFFile M;
  if (!M.load_header(model)) {
    fprintf(stderr, "[plan] FAIL: cannot load model %s\n", model.c_str());
    return 1;
  }
  printf("[plan] Loaded %zu tensor(s) across %zu shard(s) from %s\n\n",
         M.tensors.size(), M.shard_paths.size(), model.c_str());

  // ---- layer range + expert format (same detection style as verify) ----
  int min_layer = INT_MAX, max_layer = -1;
  std::set<int> layers_with_experts;
  int n_experts = -1;
  std::map<int, size_t> layer_slab_bytes; // layer -> gate+up+down / n_experts

  for (auto &kv : M.tensors) {
    int L;
    if (!parse_blk_layer(kv.first, L))
      continue;
    min_layer = std::min(min_layer, L);
    max_layer = std::max(max_layer, L);
  }
  if (max_layer < 0) {
    printf("[plan] No blk.* tensors found -- not a transformer GGUF, or "
           "wrong file/dir.\n");
    return 1;
  }

  // NextN/MTP draft-block exclusion -- mirrors graph_initializer.cpp's own
  // c.n_layers computation EXACTLY (block_count - nextn_predict_layers),
  // via the SAME get_config_int KV lookup GGUFFile already provides. Found
  // by hand (2026-08-16) chasing a real ~1.4GB gap between this tool's
  // reported dense total and a live engine run's own [VRAM-BREAKDOWN] for
  // the SAME GLM-5.2 checkpoint: this tool was naively including the raw
  // max blk.N found in the file (79 blocks, blk.0..blk.78) as if all were
  // real transformer layers, while the live engine excludes the trailing
  // NextN/MTP draft block(s) (nextn_predict_layers=1 for this checkpoint)
  // from n_layers entirely -- it doesn't implement speculative decoding, so
  // that block's tensors are loaded but never touched. Without this fix,
  // every downstream number here (dense GB, MoE layer count, per-layer
  // slab table) silently disagreed with what a real run actually allocates
  // -- exactly the invariant this tool's own header comment promises.
  // Falls back to the naive max_layer+1 (old behavior) if block_count is
  // absent from KV metadata, matching graph_initializer.cpp's own fallback.
  int block_count = M.get_config_int("block_count", 0);
  int nextn_layers = M.get_config_int("nextn_predict_layers", 0);
  int n_layers_real = (block_count > 0) ? (block_count - nextn_layers)
                                         : (max_layer - min_layer + 1);
  int last_real_layer = min_layer + n_layers_real - 1;
  if (nextn_layers > 0) {
    printf("[plan] block_count=%d nextn_predict_layers=%d -- excluding "
           "blk.%d..blk.%d (NextN/MTP draft block(s), never touched by the "
           "live engine's main forward pass) from every total below.\n",
           block_count, nextn_layers, last_real_layer + 1, max_layer);
  }

  for (int L = min_layer; L <= last_real_layer; ++L) {
    std::string base = "blk." + std::to_string(L) + ".";
    auto gE = M.tensors.find(base + "ffn_gate_exps.weight");
    auto uE = M.tensors.find(base + "ffn_up_exps.weight");
    auto dE = M.tensors.find(base + "ffn_down_exps.weight");
    if (gE != M.tensors.end() && uE != M.tensors.end() &&
        dE != M.tensors.end() && !gE->second.shape.empty()) {
      int ne = (int)gE->second.shape.back();
      if (ne > 0) {
        if (n_experts < 0)
          n_experts = ne;
        layers_with_experts.insert(L);
        layer_slab_bytes[L] = gE->second.size_bytes / ne +
                               uE->second.size_bytes / ne +
                               dE->second.size_bytes / ne;
      }
    }
  }

  int total_layers = n_layers_real;
  int n_moe_layers = (int)layers_with_experts.size();
  size_t max_slab_bytes = 0;
  for (auto &kv : layer_slab_bytes)
    max_slab_bytes = std::max(max_slab_bytes, kv.second);

  printf("Layers blk.%d..blk.%d (%d total), %d with routed experts, %d "
         "leading dense-only.\n",
         min_layer, last_real_layer, total_layers, n_moe_layers,
         total_layers - n_moe_layers);
  if (n_experts > 0)
    printf("Experts per MoE layer: %d\n", n_experts);
  printf("\n");

  // ---- dense (persistent VRAM) requirement ----
  std::map<std::string, CategoryTotal> categories;
  size_t attn_output_quantized_total = 0, attn_output_bf16_estimate = 0;
  int attn_output_count = 0;
  for (auto &kv : M.tensors) {
    const std::string &name = kv.first;
    const GGUFTensor &t = kv.second;
    // Same NextN/MTP exclusion as the layer-range detection above -- a
    // blk.N tensor past last_real_layer is real data in the file but never
    // loaded/dispatched by the live engine's dense-offload pass, so it
    // must not count toward the dense VRAM total either.
    int L;
    if (parse_blk_layer(name, L) && (L < min_layer || L > last_real_layer))
      continue;
    if (!is_dense_offload_candidate(name))
      continue;
    auto &cat = categories[categorize_dense_tensor(name)];
    cat.bytes += t.size_bytes;
    cat.count += 1;
    if (is_attn_output(name) && t.shape.size() >= 2) {
      attn_output_quantized_total += t.size_bytes;
      attn_output_bf16_estimate += t.shape[0] * t.shape[1] * 2; // BF16 = 2 bytes/elem
      ++attn_output_count;
    }
  }

  printf("Dense (persistent, always-VRAM-resident) tensor requirement:\n");
  std::vector<std::pair<std::string, CategoryTotal>> sorted_cats(
      categories.begin(), categories.end());
  std::sort(sorted_cats.begin(), sorted_cats.end(),
            [](auto &a, auto &b) { return a.second.bytes > b.second.bytes; });
  size_t dense_total_native = 0;
  for (auto &[name, cat] : sorted_cats) {
    printf("  %-45s %8.3f GB  (%d tensor(s))\n", name.c_str(),
           cat.bytes / 1e9, cat.count);
    dense_total_native += cat.bytes;
  }
  printf("  %-45s %8.3f GB\n", "TOTAL (native quant, nothing BF16-converted)",
         dense_total_native / 1e9);
  if (attn_output_count > 0) {
    size_t best_case_total =
        dense_total_native - attn_output_quantized_total + attn_output_bf16_estimate;
    printf("  If all %d attn_output layer(s) BF16-convert (engine's "
           "matched-type cuBLAS fast-path, VRAM-availability permitting): "
           "%.3f GB quantized -> %.3f GB BF16, TOTAL becomes ~%.3f GB\n",
           attn_output_count, attn_output_quantized_total / 1e9,
           attn_output_bf16_estimate / 1e9, best_case_total / 1e9);
    printf("  -> size dense_gb somewhere in [%.1f, %.1f] GB depending on how "
           "much VRAM headroom is left for that fast-path once moe_cache_gb "
           "is also decided (see CLAUDE.md: it's a real capacity ceiling on "
           "a small card, not a config knob).\n",
           std::min(dense_total_native, best_case_total) / 1e9 + 0.5,
           std::max(dense_total_native, best_case_total) / 1e9 + 0.5);
  }
  printf("  KV cache is NOT estimated here (needs architecture constants "
         "this lightweight tool doesn't parse) -- check a real run's own "
         "\"[GRAPH] KV cache: ... GB\" log line for your target context/"
         "kv_bits.\n\n");

  // ---- MoE / routed-expert requirement ----
  if (max_slab_bytes > 0) {
    printf("MoE routed-expert requirement:\n");
    printf("  One expert's slab (gate+up+down, largest across layers): "
           "%.4f GB (%.2f MB)\n",
           max_slab_bytes / 1e9, max_slab_bytes / 1e6);
    size_t total_expert_slots = (size_t)n_moe_layers * (size_t)n_experts;
    printf("  Full residency (every expert, every MoE layer, all at once): "
           "%zu slots = %.2f GB\n",
           total_expert_slots, total_expert_slots * max_slab_bytes / 1e9);
    printf("  GB -> slots quick reference (this model's slab size):\n");
    for (double gb : {2.0, 4.0, 8.0, 12.0, 16.0, 20.0, 24.0}) {
      size_t slots = (size_t)((gb * 1e9) / max_slab_bytes);
      printf("    %5.1f GB -> %6zu slots (%.2f/layer avg over %d MoE "
             "layer(s))\n",
             gb, slots, (double)slots / n_moe_layers, n_moe_layers);
    }
    printf("  -> moe_cache_gb translates directly via this table; "
           "pin_topk_per_layer x n_moe_layers should stay at roughly 50-60%%\n"
           "     of the slot count your chosen moe_cache_gb yields (see "
           "CLAUDE.md -- overshooting this floods \"evict_lru found NO "
           "victim\" for the whole run).\n\n");
  } else {
    printf("MoE routed-expert requirement: no packed *_exps tensors found "
           "(unpacked/per-expert layout, or a dense-only model) -- expert "
           "slab sizing not available from this quick scan.\n\n");
  }

  // ---- detected system config ----
  printf("Detected system config (this machine, not necessarily where the "
         "engine will run):\n");
  print_detected_gpus();
  print_detected_ram();
  printf("\n");

  // ---- .moestream_usage summary ----
  printf(".moestream_usage summary (%s):\n", usage_file.c_str());
  print_usage_summary(usage_file);

  return 0;
}

} // namespace gguf_tool
