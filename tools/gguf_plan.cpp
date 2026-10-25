// tools/gguf_plan.cpp
//
// gguf_tool plan: size a GGUF against local RAM/VRAM and optionally write a
// reviewable starting TOML profile. Dense weight totals mirror the engine's
// tensor categorization; MoE bytes are summed per layer; KV is a conservative
// F32 estimate at the requested context. The generated GPU profile reserves
// dense/KV/workspace budgets first, then assigns the remaining safe VRAM to
// the expert pool. Runtime pin_all_experts_if_fit fills that pool completely
// when the actual per-layer regions can hold every expert; otherwise it uses
// .zeromoe_usage as a hot-set ranking. All estimates are advisory and the
// runtime's startup report remains authoritative.
//
// Build: see gguf_common.hpp (compiled together with the gguf_*.cpp files).
// Build: see gguf_common.hpp (compiled together with the other gguf_*.cpp).
#include "gguf_common.hpp"
#include "../memory/resident_memory_policy.hpp"
#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
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

bool is_dense_offload_candidate(const std::string &name, bool embedding_gpu) {
  if (name == "token_embd.weight" && !embedding_gpu) return false;
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

struct DetectedGpu {
  int index = -1;
  std::string name;
  double total_gb = 0.0;
  double free_gb = 0.0;
};

std::vector<DetectedGpu> detect_gpus() {
  const std::string output = run_capture(
      "nvidia-smi --query-gpu=index,name,memory.total,memory.free "
      "--format=csv,noheader,nounits");
  std::vector<DetectedGpu> result;
  std::stringstream rows(output);
  std::string line;
  while (std::getline(rows, line)) {
    const auto fields = split_csv(line);
    if (fields.size() < 4)
      continue;
    DetectedGpu gpu;
    gpu.index = std::atoi(fields[0].c_str());
    gpu.name = fields[1];
    // nvidia-smi reports MiB. Its memory.total/free values track the
    // driver's usable budget closely enough for a draft profile; the engine
    // re-queries CUDA properties and runs its normal startup preflight.
    gpu.total_gb = std::atof(fields[2].c_str()) / 1000.0;
    gpu.free_gb = std::atof(fields[3].c_str()) / 1000.0;
    result.push_back(std::move(gpu));
  }
  return result;
}

struct DetectedRam {
  double total_gb = 0.0;
  double available_gb = 0.0;
};

DetectedRam detect_ram() {
  DetectedRam result;
  std::ifstream meminfo("/proc/meminfo");
  if (meminfo) {
    std::string line;
    while (std::getline(meminfo, line)) {
      unsigned long long kb = 0;
      if (line.rfind("MemTotal:", 0) == 0 &&
          std::sscanf(line.c_str(), "MemTotal: %llu kB", &kb) == 1)
        result.total_gb = (double)kb / 1e6;
      else if (line.rfind("MemAvailable:", 0) == 0 &&
               std::sscanf(line.c_str(), "MemAvailable: %llu kB", &kb) == 1)
        result.available_gb = (double)kb / 1e6;
    }
    if (result.total_gb > 0.0)
      return result;
  }
  const std::string output = run_capture("sysctl -n hw.memsize");
  if (!output.empty()) {
    const double bytes = std::atof(output.c_str());
    if (bytes > 0.0) {
      result.total_gb = bytes / 1e9;
      // sysctl exposes total RAM, not current free RAM. Leave a conservative
      // quarter for macOS and applications already running.
      result.available_gb = result.total_gb * 0.75;
    }
  }
  return result;
}

void print_detected_gpus() {
  const auto gpus = detect_gpus();
  if (gpus.empty()) {
    printf("  nvidia-smi not found or no GPUs reported -- run this on "
           "the target machine, or size manually from known card specs.\n");
    return;
  }
  for (const auto &gpu : gpus)
    printf("  GPU %d: %-30s %.1f GB total, %.1f GB free VRAM\n",
           gpu.index, gpu.name.c_str(), gpu.total_gb, gpu.free_gb);
}

void print_detected_ram() {
  const DetectedRam ram = detect_ram();
  if (ram.total_gb > 0.0) {
    printf("  System RAM: %.1f GB total, %.1f GB available\n",
           ram.total_gb, ram.available_gb);
    return;
  }
  printf("  System RAM: could not detect (neither /proc/meminfo nor sysctl "
         "worked) -- check manually (free -h / top).\n");
}

size_t estimate_kv_bytes(const GGUFFile &model, int context_size) {
  const int layers = std::max(0, model.get_config_int("block_count", 0) -
                                      model.get_config_int("nextn_predict_layers", 0));
  if (layers <= 0 || context_size <= 0) return 0;
  const int kv_lora = model.get_config_int("attention.kv_lora_rank", 0);
  if (kv_lora > 0) {
    const int rope = model.get_config_int(
        "attention.qk_rope_head_dim",
        model.get_config_int("rope.dimension_count", 0));
    return (size_t)layers * (size_t)context_size *
           (size_t)(kv_lora + std::max(0, rope)) * sizeof(float);
  }
  int full_layers = layers;
  const auto recurrent = model.get_config_int_array("attention.recurrent_layers");
  if (recurrent.size() >= (size_t)layers) {
    full_layers = 0;
    for (int i = 0; i < layers; ++i)
      if (recurrent[(size_t)i] == 0) ++full_layers;
  } else if (model.get_config_int("ssm.state_size", 0) > 0) {
    const int interval = model.get_config_int("full_attention_interval", 0);
    if (interval > 0) full_layers = layers / interval;
  }
  const int heads = model.get_config_int(
      "attention.head_count_kv",
      model.get_config_int("attention.head_count", 1));
  const int head_count = std::max(1, model.get_config_int("attention.head_count", 1));
  const int key_dim = model.get_config_int(
      "attention.key_length",
      std::max(1, model.get_config_int("embedding_length", 1) / head_count));
  const int value_dim = model.get_config_int("attention.value_length", key_dim);
  return (size_t)full_layers * (size_t)context_size * (size_t)std::max(1, heads) *
         (size_t)(std::max(1, key_dim) + std::max(1, value_dim)) * sizeof(float);
}

std::string toml_quote(const std::string &value) {
  std::string escaped;
  escaped.reserve(value.size() + 2);
  escaped.push_back('"');
  for (char c : value) {
    if (c == '\\' || c == '"') escaped.push_back('\\');
    if (c == '\n' || c == '\r') escaped.push_back(' ');
    else escaped.push_back(c);
  }
  escaped.push_back('"');
  return escaped;
}

// .zeromoe_usage is a plain "layer expert count" text file (one line per
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
            "Usage: %s <model_file_or_dir> [--usage-file <path>] [--embedding-device auto|cpu|gpu]\n"
"       [--write-config <path>] [--gpu-id <n>|--cpu-only]\n"
"       [--context-size <n>] [--max-new-tokens <n>]\n"
"\n"
"Sizes GGUF dense and per-layer MoE tensors against detected RAM/VRAM.\n"
"With --write-config, emits a reviewable TOML profile sized from actual\n"
"tensor bytes; it enables full expert preload automatically when the set\n"
"fits the selected GPU's configured expert pool.\n",
argv[0]);
    return 2;
  }
  std::string model = argv[1];
  std::string usage_file = ".zeromoe_usage";
  std::string embedding_device = "auto";
  std::string write_config_path;
  int selected_gpu_id = -1;
  int context_size = 4096;
  int max_new_tokens = 256;
  bool force_cpu_only = false;
  for (int i = 2; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--usage-file" && i + 1 < argc)
      usage_file = argv[++i];
    else if (a == "--embedding-device" && i + 1 < argc)
      embedding_device = argv[++i];
    else if (a == "--write-config" && i + 1 < argc)
      write_config_path = argv[++i];
    else if (a == "--gpu-id" && i + 1 < argc)
      selected_gpu_id = std::atoi(argv[++i]);
    else if (a == "--context-size" && i + 1 < argc)
      context_size = std::atoi(argv[++i]);
    else if (a == "--max-new-tokens" && i + 1 < argc)
      max_new_tokens = std::atoi(argv[++i]);
    else if (a == "--cpu-only")
      force_cpu_only = true;
    else {
      fprintf(stderr, "[plan] unknown arg: %s\n", a.c_str());
      return 2;
    }
  }

  GGUFFile M;
  if (!zeromoe::memory::valid_embedding_device(embedding_device)) {
    fprintf(stderr, "[plan] embedding device must be auto, cpu or gpu\n");
    return 2;
  }
  const bool embedding_gpu = zeromoe::memory::embedding_on_gpu(embedding_device, false);
  if (!M.load_header(model)) {
    fprintf(stderr, "[plan] FAIL: cannot load model %s\n", model.c_str());
    return 1;
  }
  if (context_size <= 0 || max_new_tokens < 0) {
    fprintf(stderr, "[plan] context-size must be positive and max-new-tokens nonnegative\n");
    return 2;
  }
  size_t model_total_bytes = 0;
  for (const auto &tensor : M.tensors)
    model_total_bytes += tensor.second.size_bytes;
  printf("[plan] Loaded %zu tensor(s) across %zu shard(s) from %s\n\n",
         M.tensors.size(), M.shard_paths.size(), model.c_str());

  // ---- layer range + expert format (same detection style as verify) ----
  int min_layer = INT_MAX, max_layer = -1;
  std::set<int> layers_with_experts;
  int n_experts = -1;
  std::map<int, size_t> layer_slab_bytes; // layer -> gate+up+down / n_experts
std::map<int, int> layer_expert_counts;

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
layer_expert_counts[L] = ne;
      }
    }
  }

  int total_layers = n_layers_real;
  int n_moe_layers = (int)layers_with_experts.size();
  size_t max_slab_bytes = 0;
  for (auto &kv : layer_slab_bytes)
    max_slab_bytes = std::max(max_slab_bytes, kv.second);
size_t full_expert_slots = 0;
size_t full_expert_bytes = 0;
for (const auto &[layer, slab_bytes] : layer_slab_bytes) {
const size_t count = (size_t)layer_expert_counts.at(layer);
full_expert_slots += count;
full_expert_bytes += slab_bytes * count;
}

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
    if (!is_dense_offload_candidate(name, embedding_gpu))
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
  size_t dense_total_best = dense_total_native;
  printf("  %-45s %8.3f GB\n", "TOTAL (native quant, nothing BF16-converted)",
         dense_total_native / 1e9);
  if (attn_output_count > 0) {
    size_t best_case_total =
        dense_total_native - attn_output_quantized_total + attn_output_bf16_estimate;
    dense_total_best = best_case_total;
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
  printf("  KV estimate: %.3f GB for the requested context (conservative F32 "
         "cache estimate; check the runtime's [GRAPH] KV cache line).\n\n",
         (double)estimate_kv_bytes(M, context_size) / 1e9);

  // ---- MoE / routed-expert requirement ----
  const auto embedding = M.tensors.find("token_embd.weight");
  const size_t embedding_bytes = embedding == M.tensors.end() ? 0 : embedding->second.size_bytes;
  const size_t reclaimed_bytes = embedding_gpu ? 0 : embedding_bytes;
  printf("[EMBEDDING-PLAN] device=%s host_bytes=%zu reclaimed_vram_bytes=%zu additional_expert_slots=%zu\n",
         embedding_gpu ? "gpu" : "cpu", embedding_bytes, reclaimed_bytes,
         max_slab_bytes > 0 ? reclaimed_bytes / max_slab_bytes : 0);
  printf("  Compare the exact full-residency byte sum with the expert-pool "
         "budget left after dense weights, KV, and workspace. A conservative "
         "largest-slab bound is also reported.\n\n");
  if (max_slab_bytes > 0) {
    printf("MoE routed-expert requirement:\n");
    printf("  One expert's slab (gate+up+down, largest across layers): "
           "%.4f GB (%.2f MB)\n",
           max_slab_bytes / 1e9, max_slab_bytes / 1e6);
    printf("  Full residency (exact sum of each MoE layer's packed expert "
           "bytes): %zu slots = %.2f GB\n",
           full_expert_slots, full_expert_bytes / 1e9);
    printf("  Conservative bound using the largest expert slab in every "
           "layer: %.2f GB\n",
           full_expert_slots * max_slab_bytes / 1e9);
    printf("  GB -> slots quick reference (this model's slab size):\n");
    for (double gb : {2.0, 4.0, 8.0, 12.0, 16.0, 20.0, 24.0}) {
      size_t slots = (size_t)((gb * 1e9) / max_slab_bytes);
      printf("    %5.1f GB -> %6zu slots (%.2f/layer avg over %d MoE "
             "layer(s))\n",
             gb, slots, (double)slots / n_moe_layers, n_moe_layers);
    }
    printf("  Onboarding: compare the exact full-residency size with the "
           "selected expert-pool budget after dense weights, KV, and "
           "workspace. With pin_all_experts_if_fit=true and "
           "pin_topk_per_layer=0 (default), startup checks actual per-layer "
           "capacity and loads every expert when they fit; otherwise it uses "
           ".zeromoe_usage to rank the resident hot set.\n\n");
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

  // ---- .zeromoe_usage summary ----
  printf(".zeromoe_usage summary (%s):\n", usage_file.c_str());
  print_usage_summary(usage_file);

  if (!write_config_path.empty()) {
    const DetectedRam ram = detect_ram();
    const auto detected = detect_gpus();
    const DetectedGpu *gpu = nullptr;
    if (!force_cpu_only && !detected.empty()) {
      if (selected_gpu_id >= 0) {
        for (const auto &candidate : detected)
          if (candidate.index == selected_gpu_id)
            gpu = &candidate;
        if (!gpu) {
          fprintf(stderr, "[plan] requested GPU id %d was not detected\n",
                  selected_gpu_id);
          return 2;
        }
      } else {
        gpu = &*std::max_element(
            detected.begin(), detected.end(),
            [](const DetectedGpu &a, const DetectedGpu &b) {
              return a.free_gb < b.free_gb;
            });
      }
    } else if (selected_gpu_id >= 0) {
      fprintf(stderr, "[plan] --gpu-id cannot be used without a detected NVIDIA GPU\n");
      return 2;
    }

    const double model_gb = (double)model_total_bytes / 1e9;
    double host_ram_gb = 0.0;
    if (ram.available_gb > 0.0) {
      if (gpu) {
        // Keep the engine's Linux startup-preflight margin (~21.5 GB) and
        // cap the expert host tier so a fresh config does not claim all DDR.
        host_ram_gb = std::max(1.0, std::min(32.0, ram.available_gb - 22.0));
      } else {
        // CPU inference needs a dense host copy plus an expert working set.
        // Leave a few GB for the OS and let the user raise this on large-RAM
        // machines if they want a bigger CPU expert cache.
        host_ram_gb =
            std::max(1.0, std::min(32.0, ram.available_gb - model_gb - 4.0));
      }
    }

    const double dense_required_gb =
        (double)std::max(dense_total_native, dense_total_best) / 1e9;
    const auto ceil_tenth = [](double x) {
      return std::ceil(x * 10.0) / 10.0;
    };
    const double dense_gb = ceil_tenth(dense_required_gb + 0.5);
    const double kv_required_gb =
        (double)estimate_kv_bytes(M, context_size) / 1e9;
    const double kv_cache_gb =
        ceil_tenth(std::max(0.25, kv_required_gb * 1.25 + 0.25));
    double moe_cache_gb = 0.0;
    double usable_vram_gb = 0.0;
    double pool_budget_gb = 0.0;
    if (gpu && n_moe_layers > 0) {
      // Account for other current GPU users, the driver, and graph/scratch
      // workspace before assigning the remaining budget to expert slots.
      usable_vram_gb =
          std::max(0.0, std::min(gpu->total_gb - 1.0, gpu->free_gb - 0.5));
      pool_budget_gb =
          std::max(0.0, usable_vram_gb - dense_gb - kv_cache_gb - 1.0);
      const double full_expert_gb = (double)full_expert_bytes / 1e9;
      if (pool_budget_gb >= full_expert_gb + 0.25)
        moe_cache_gb = ceil_tenth(full_expert_gb + 0.25);
      else
        moe_cache_gb = std::max(0.1, std::floor(pool_budget_gb * 10.0) / 10.0);
    }

    std::ofstream config_out(write_config_path);
    if (!config_out) {
      fprintf(stderr, "[plan] could not write config: %s\n",
              write_config_path.c_str());
      return 1;
    }
    config_out << "# Generated by gguf_tool plan. Review context and memory budgets.\n"
               << "model_path = " << toml_quote(model) << "\n"
               << "context_size = " << context_size << "\n"
               << "max_new_tokens = " << max_new_tokens << "\n"
               << "temperature = 0.7\n"
               << "top_p = 0.95\n"
               << "raw_prompt = false\n"
               << "enable_thinking = false\n"
               << "host_ram_gb = " << std::fixed << std::setprecision(1)
               << host_ram_gb << "\n";
    if (!gpu)
      config_out << "# CPU-only profile; run with zeromoe_cpu --config.\n";
    if (gpu) {
      config_out << "pin_all_experts_if_fit = true\n"
                 << "pin_topk_per_layer = 0\n";
    }
    config_out << "pin_file = \".zeromoe_usage\"\n";
    if (gpu) {
      config_out << "# VRAM sizing uses the free-memory snapshot from planning time.\n"
                 << "\n[[gpus]]\n"
                 << "device_id = " << gpu->index << "\n"
                 << "name = " << toml_quote(gpu->name) << "\n"
                 << "role = \"dense+moe\"\n"
                 << "vram_gb = " << std::fixed << std::setprecision(1)
                 << gpu->total_gb << "\n"
                 << "dense_gb = " << dense_gb << "\n"
                 << "kv_cache_gb = " << kv_cache_gb << "\n"
                 << "moe_cache_gb = " << moe_cache_gb << "\n"
                 << "kv_bits = 4\n";
    }
    config_out.close();
    printf("[plan] wrote %s profile %s\n",
           gpu ? "CUDA" : "CPU", write_config_path.c_str());
    printf("[plan] estimated dense=%.1f GB, KV=%.1f GB, MoE pool=%.1f GB, "
           "host RAM budget=%.1f GB",
           dense_gb, kv_cache_gb, moe_cache_gb, host_ram_gb);
    if (gpu) {
      printf(" on GPU %d (%s; %.1f GB free at planning time)",
             gpu->index, gpu->name.c_str(), gpu->free_gb);
      if (full_expert_bytes > 0 &&
          moe_cache_gb * 1e9 >= (double)full_expert_bytes)
        printf("; full MoE set fits");
      else if (full_expert_bytes > 0)
        printf("; usage-ranked partial MoE residency");
    }
    printf("\n");
    if (gpu && n_moe_layers > 0 &&
        pool_budget_gb + 1e-6 < (double)full_expert_bytes / 1e9)
      printf("[plan] note: full MoE residency exceeds this GPU pool budget; "
             "the generated config will use the usage-ranked hot set.\n");
    if (gpu && ram.available_gb > 0.0 &&
        host_ram_gb + 21.5 > ram.available_gb)
      printf("[plan] warning: host RAM headroom may fail ZeroMoE startup "
             "preflight; lower host_ram_gb or free DDR first.\n");
    if (!gpu && ram.available_gb > 0.0 &&
        host_ram_gb + model_gb > ram.available_gb)
      printf("[plan] warning: CPU model plus host expert-cache budget exceeds "
             "detected available RAM; lower host_ram_gb or use a GPU profile.\n");
  }

  return 0;
}

} // namespace gguf_tool
