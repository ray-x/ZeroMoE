// tools/gguf_mix.cpp
//
// gguf_tool mix: builds a real per-expert MIXED-PRECISION MoE model from
// TWO separately-quantized source models (a "hot"/baseline model and a
// "cold" model whose gate/up/down are quantized smaller) plus a real
// .zeromoe_usage-style per-(layer,expert) traffic survey. For every
// eligible layer (one where the cold source actually provides its own
// ffn_gate_exps/ffn_up_exps/ffn_down_exps), the coldest N% of that
// layer's experts (ranked by real traffic in --usage-file, ascending)
// are sourced from the cold model; the rest stay on the baseline model's
// own bytes. A layer the cold model doesn't cover (or whose types don't
// match --require-cold-types, if given) is left untouched entirely --
// never a fabricated downgrade.
//
// This does NOT rewrite the whole model. It writes ONE SMALL OVERRIDE
// SHARD containing only the touched layers, fully unpacked into individual
// blk.<L>.ffn_{gate,up,down}.<E>.weight tensors -- engine.hpp's
// sanity_check_unpacked_format requires a genuinely-unpacked layer to be
// COMPLETE (all n_exp experts present); there is no partial/fallback-to-
// packed WITHIN one layer, only whole layers are packed-or-unpacked (see
// engine.hpp's layer_is_unpacked comment). Every OTHER layer, and every
// non-MoE tensor (attention, dense, embeddings, norms, shared expert), is
// left completely untouched in the baseline model -- --symlink-baseline
// populates the output directory with symlinks to the baseline model's
// own shards rather than copying them, so unaffected data costs zero new
// disk bytes. The override shard's filename is prefixed "zzz_mix" so
// gguf_loader.hpp's directory loader (sorts filenames, later shard wins
// on tensor-NAME collision) loads it last -- it never actually collides
// by name with anything in the baseline shards (this tool only emits
// blk.<L>.ffn_{gate,up,down}.<E>.weight names, which packed baseline
// shards don't have at all), so both simply coexist in the final loaded
// tensor table, exactly as engine.hpp's own layer_is_unpacked comment
// describes for a real multi-source build.
//
// Pure byte-copy tool -- no requantization, no value is ever decoded or
// reinterpreted. Every expert's bytes come verbatim from whichever source
// model (hot or cold) the survey-driven selection picked for it.
//
// Usage:
//   gguf_tool mix <baseline_dir> <cold_dir> <output_dir> --usage-file <path>
//       [--cold-fraction 0.30] [--align N] [--max-shard-gb N]
//       [--require-cold-types GATE,UP,DOWN] [--layers L1,L2,...]
//       [--symlink-baseline] [--dry-run]
#include "gguf_common.hpp"
#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <unistd.h>
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;

namespace {

// ============================================================
// Minimal raw-KV writer -- this shard only ever needs to assert its own
// general.alignment; every other KV key (architecture, tokenizer, hparams)
// is already set by whichever baseline shard the loader reads first, and
// gguf_loader.hpp's per-shard KV loop only WRITES keys it actually finds
// in THAT shard's own header (kv_uint etc. are never reset/cleared per
// shard) -- so a later, KV-minimal shard cannot blank out anything an
// earlier shard already established. See gguf_loader.hpp's own "kv_uint
// is a GGUFFile-wide member map that persists across every shard" comment
// for the real incident that confirmed this.
// ============================================================
struct RawKV {
  std::string key;
  uint32_t type;
  std::vector<uint8_t> value_bytes;
};

RawKV make_uint32_kv(const std::string &key, uint32_t value) {
  RawKV kv;
  kv.key = key;
  kv.type = 4; // UINT32
  kv.value_bytes.resize(4);
  std::memcpy(kv.value_bytes.data(), &value, 4);
  return kv;
}

void write_raw_kv(std::ofstream &out, const RawKV &kv) {
  uint64_t klen = kv.key.size();
  out.write(reinterpret_cast<const char *>(&klen), 8);
  out.write(kv.key.data(), (std::streamsize)klen);
  out.write(reinterpret_cast<const char *>(&kv.type), 4);
  out.write(reinterpret_cast<const char *>(kv.value_bytes.data()),
            (std::streamsize)kv.value_bytes.size());
}

// Only the handful of types this tool is ever likely to see/compare
// against --require-cold-types. Falls back to "TYPE<n>" rather than
// guessing, so an unrecognized type still compares predictably (never
// silently matches a --require-cold-types filter it shouldn't).
std::string gguf_type_name(GGUFType t) {
  switch (t) {
  case GGUFType::F32: return "F32";
  case GGUFType::F16: return "F16";
  case GGUFType::Q4_0: return "Q4_0";
  case GGUFType::Q4_1: return "Q4_1";
  case GGUFType::Q5_0: return "Q5_0";
  case GGUFType::Q5_1: return "Q5_1";
  case GGUFType::Q8_0: return "Q8_0";
  case GGUFType::Q8_1: return "Q8_1";
  case GGUFType::Q2_K: return "Q2_K";
  case GGUFType::Q3_K: return "Q3_K";
  case GGUFType::Q4_K: return "Q4_K";
  case GGUFType::Q5_K: return "Q5_K";
  case GGUFType::Q6_K: return "Q6_K";
  case GGUFType::Q8_K: return "Q8_K";
  case GGUFType::IQ2_XXS: return "IQ2_XXS";
  case GGUFType::IQ2_XS: return "IQ2_XS";
  case GGUFType::IQ3_XXS: return "IQ3_XXS";
  case GGUFType::IQ1_S: return "IQ1_S";
  case GGUFType::IQ4_NL: return "IQ4_NL";
  case GGUFType::IQ3_S: return "IQ3_S";
  case GGUFType::IQ2_S: return "IQ2_S";
  case GGUFType::IQ4_XS: return "IQ4_XS";
  case GGUFType::IQ1_M: return "IQ1_M";
  case GGUFType::BF16: return "BF16";
  case GGUFType::Q2_0: return "Q2_0";
  default: return "TYPE" + std::to_string((uint32_t)t);
  }
}

// .zeromoe_usage-style "layer expert count" text file, one line per
// ever-selected (layer,expert) pair. Same format/parser as gguf_plan.cpp's
// own load_usage_file -- re-implemented here rather than shared, matching
// this project's existing per-subcommand-local-helper convention
// (gguf_common.hpp's own header comment).
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

// Packed *_exps tensor geometry for one (layer, projection): the whole
// tensor plus the per-expert byte stride (asserted to divide evenly --
// every real quantized MoE tensor this tool has been run against does,
// since block-quant row sizes are themselves powers of two for the hidden
// dims in use here; a non-dividing tensor is refused rather than silently
// truncated).
struct PackedProj {
  const GGUFTensor *t = nullptr;
  uint64_t per_expert_bytes = 0;
  uint64_t n_experts = 0;
};

bool find_packed_proj(const GGUFFile &f, int layer, const char *kind,
                       PackedProj &out) {
  std::string name =
      "blk." + std::to_string(layer) + ".ffn_" + kind + "_exps.weight";
  auto it = f.tensors.find(name);
  if (it == f.tensors.end())
    return false;
  const GGUFTensor &t = it->second;
  if (t.shape.empty())
    return false;
  uint64_t n_exp = t.shape.back(); // GGUF ne order: [in, out, n_experts]
  if (n_exp == 0 || t.size_bytes % n_exp != 0)
    return false;
  out.t = &t;
  out.n_experts = n_exp;
  out.per_expert_bytes = t.size_bytes / n_exp;
  return true;
}

struct PlannedTensor {
  std::string name;
  GGUFType type;
  std::vector<uint64_t> shape; // 2D: [in, out]
  bool from_cold = false;      // which GGUFFile src_shard_idx indexes into
  int src_shard_idx;
  uint64_t src_offset; // absolute byte offset in its own shard file
  uint64_t size_bytes;
  uint64_t out_offset = 0; // relative to this output shard's own data_start
};

} // namespace

namespace gguf_tool {

int cmd_mix(int argc, char **argv) {
  if (argc < 4) {
    fprintf(stderr,
            "Usage: %s <baseline_dir> <cold_dir> <output_dir> "
            "--usage-file <path>\n"
            "    [--cold-fraction 0.30] [--align N=4096] "
            "[--max-shard-gb N=40]\n"
            "    [--require-cold-types GATE,UP,DOWN] "
            "[--layers L1,L2,...]\n"
            "    [--exclude-layers L1,L2,...] "
            "[--symlink-baseline] [--dry-run]\n"
            "\n"
            "Builds a real per-expert mixed-precision override shard: for\n"
            "every layer where <cold_dir> genuinely provides its own\n"
            "ffn_{gate,up,down}_exps tensors, the coldest N%% of that\n"
            "layer's experts (ranked by real traffic in --usage-file,\n"
            "ascending) are sourced from <cold_dir>; the rest stay on\n"
            "<baseline_dir>'s own bytes. A layer the cold model doesn't\n"
            "cover (or whose types don't match --require-cold-types, if\n"
            "given, or that's named in --exclude-layers) is left\n"
            "untouched -- never a fabricated downgrade.\n"
            "--symlink-baseline also populates <output_dir> with symlinks\n"
            "to every *.gguf in <baseline_dir>, so the result is a\n"
            "directly loadable model directory at zero extra disk cost\n"
            "for the untouched data. Pure byte-copy -- no requantization.\n",
            argv[0]);
    return 2;
  }

  std::string baseline_dir = argv[1];
  std::string cold_dir = argv[2];
  std::string output_dir = argv[3];
  std::string usage_file;
  double cold_fraction = 0.30;
  uint64_t alignment = 4096;
  uint64_t max_shard_bytes = 40ULL * 1024 * 1024 * 1024;
  std::string require_gate, require_up, require_down;
  std::set<int> layer_filter;   // --layers: empty = no include-filter
  std::set<int> layer_exclude;  // --exclude-layers: always applied on top
  bool symlink_baseline = false;
  bool dry_run = false;

  for (int i = 4; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--usage-file" && i + 1 < argc)
      usage_file = argv[++i];
    else if (a == "--cold-fraction" && i + 1 < argc)
      cold_fraction = std::stod(argv[++i]);
    else if (a == "--align" && i + 1 < argc)
      alignment = std::stoull(argv[++i]);
    else if (a == "--max-shard-gb" && i + 1 < argc)
      max_shard_bytes = (uint64_t)(std::stod(argv[++i]) * 1e9);
    else if (a == "--require-cold-types" && i + 1 < argc) {
      std::string spec = argv[++i];
      size_t p1 = spec.find(',');
      size_t p2 = spec.find(',', p1 == std::string::npos ? p1 : p1 + 1);
      if (p1 == std::string::npos || p2 == std::string::npos) {
        fprintf(stderr, "[mix] FATAL: --require-cold-types needs "
                        "GATE,UP,DOWN (e.g. IQ1_S,IQ1_S,IQ3_XXS)\n");
        return 2;
      }
      require_gate = spec.substr(0, p1);
      require_up = spec.substr(p1 + 1, p2 - p1 - 1);
      require_down = spec.substr(p2 + 1);
    } else if (a == "--layers" && i + 1 < argc) {
      std::stringstream ss(argv[++i]);
      std::string tok;
      while (std::getline(ss, tok, ','))
        if (!tok.empty())
          layer_filter.insert(std::stoi(tok));
    } else if (a == "--exclude-layers" && i + 1 < argc) {
      std::stringstream ss(argv[++i]);
      std::string tok;
      while (std::getline(ss, tok, ','))
        if (!tok.empty())
          layer_exclude.insert(std::stoi(tok));
    } else if (a == "--symlink-baseline") {
      symlink_baseline = true;
    } else if (a == "--dry-run") {
      dry_run = true;
    } else {
      fprintf(stderr, "[mix] FATAL: unknown option '%s'\n", a.c_str());
      return 2;
    }
  }
  if (usage_file.empty()) {
    fprintf(stderr, "[mix] FATAL: --usage-file is required (a real "
                    "per-(layer,expert) traffic survey -- refusing to "
                    "pick cold experts blind)\n");
    return 2;
  }
  if (cold_fraction <= 0.0 || cold_fraction >= 1.0) {
    fprintf(stderr, "[mix] FATAL: --cold-fraction must be in (0,1), got %g\n",
            cold_fraction);
    return 2;
  }

  // ---- Load both source models (real, proven multi-shard loader). ----
  GGUFFile base, cold;
  if (!base.load_header(baseline_dir)) {
    fprintf(stderr, "[mix] FATAL: failed to load baseline model at %s\n",
            baseline_dir.c_str());
    return 1;
  }
  if (!cold.load_header(cold_dir)) {
    fprintf(stderr, "[mix] FATAL: failed to load cold model at %s\n",
            cold_dir.c_str());
    return 1;
  }
  printf("[mix] baseline: %zu tensor(s) across %zu shard(s)\n",
         base.tensors.size(), base.shard_paths.size());
  printf("[mix] cold:     %zu tensor(s) across %zu shard(s)\n",
         cold.tensors.size(), cold.shard_paths.size());

  auto usage = load_usage_file(usage_file);
  if (usage.empty()) {
    fprintf(stderr,
            "[mix] FATAL: --usage-file '%s' has no parseable entries -- "
            "refusing to pick cold experts blind\n",
            usage_file.c_str());
    return 1;
  }
  std::map<std::pair<int, int>, unsigned long long> hits;
  for (auto &e : usage)
    hits[{e.layer, e.expert}] = e.count;
  printf("[mix] usage survey: %zu real (layer,expert) entries from %s\n",
         hits.size(), usage_file.c_str());

  // ---- Discover every baseline MoE layer, and which are cold-eligible. ----
  std::vector<int> base_layers;
  for (auto &[name, t] : base.tensors) {
    int l;
    if (parse_blk_layer(name, l) &&
        name.rfind(".ffn_gate_exps.weight") ==
            name.size() - std::string(".ffn_gate_exps.weight").size())
      base_layers.push_back(l);
  }
  std::sort(base_layers.begin(), base_layers.end());

  struct LayerPlan {
    int layer;
    PackedProj hg, hu, hd; // hot/baseline gate,up,down
    PackedProj cg, cu, cd; // cold gate,up,down
    uint64_t n_experts;
    std::vector<int> cold_ids; // this layer's real selected-cold experts
  };
  std::vector<LayerPlan> plans;
  int n_skipped_no_cold = 0, n_skipped_type_mismatch = 0, n_skipped_filter = 0,
      n_skipped_excluded = 0;

  for (int l : base_layers) {
    if (!layer_filter.empty() && !layer_filter.count(l)) {
      ++n_skipped_filter;
      continue;
    }
    if (layer_exclude.count(l)) {
      ++n_skipped_excluded;
      continue;
    }
    LayerPlan p{};
    p.layer = l;
    if (!find_packed_proj(base, l, "gate", p.hg) ||
        !find_packed_proj(base, l, "up", p.hu) ||
        !find_packed_proj(base, l, "down", p.hd)) {
      fprintf(stderr,
              "[mix] WARNING: layer %d has ffn_gate_exps but not a "
              "complete gate/up/down set in baseline -- skipping\n",
              l);
      continue;
    }
    if (!find_packed_proj(cold, l, "gate", p.cg) ||
        !find_packed_proj(cold, l, "up", p.cu) ||
        !find_packed_proj(cold, l, "down", p.cd)) {
      ++n_skipped_no_cold;
      continue;
    }
    if (!require_gate.empty()) {
      if (gguf_type_name(p.cg.t->type) != require_gate ||
          gguf_type_name(p.cu.t->type) != require_up ||
          gguf_type_name(p.cd.t->type) != require_down) {
        ++n_skipped_type_mismatch;
        continue;
      }
    }
    if (p.hg.n_experts != p.hu.n_experts || p.hg.n_experts != p.hd.n_experts ||
        p.hg.n_experts != p.cg.n_experts || p.cg.n_experts != p.cu.n_experts ||
        p.cg.n_experts != p.cd.n_experts) {
      fprintf(stderr,
              "[mix] WARNING: layer %d expert-count mismatch across "
              "gate/up/down/hot/cold -- skipping\n",
              l);
      continue;
    }
    p.n_experts = p.hg.n_experts;

    // ---- Rank this layer's real experts by traffic, ascending; take the
    // coldest cold_fraction as the cold set. Ties (very common at low
    // counts, e.g. many experts tied at 0) break on expert id ascending,
    // for a fully deterministic, reproducible selection. ----
    std::vector<std::pair<unsigned long long, int>> ranked;
    ranked.reserve(p.n_experts);
    for (uint64_t e = 0; e < p.n_experts; ++e) {
      auto it = hits.find({l, (int)e});
      ranked.push_back({it == hits.end() ? 0ULL : it->second, (int)e});
    }
    std::sort(ranked.begin(), ranked.end());
    int n_cold = (int)std::llround(cold_fraction * (double)p.n_experts);
    n_cold = std::clamp(n_cold, 0, (int)p.n_experts);
    for (int i = 0; i < n_cold; ++i)
      p.cold_ids.push_back(ranked[i].second);
    std::sort(p.cold_ids.begin(), p.cold_ids.end());
    plans.push_back(std::move(p));
  }

  printf("[mix] eligible layers: %zu (skipped: %d no cold data, %d type "
         "mismatch, %d filtered out, %d excluded)\n",
         plans.size(), n_skipped_no_cold, n_skipped_type_mismatch,
         n_skipped_filter, n_skipped_excluded);
  if (plans.empty()) {
    fprintf(stderr, "[mix] FATAL: no eligible layers -- nothing to build\n");
    return 1;
  }

  // ---- Build the output tensor plan: per eligible layer, ALL n_experts
  // experts, fully unpacked (engine.hpp requires per-layer completeness --
  // see this file's header comment). ----
  uint64_t total_hot_bytes = 0, total_cold_bytes = 0, total_would_be_all_hot = 0;
  std::vector<PlannedTensor> plan;
  for (auto &p : plans) {
    std::set<int> cold_set(p.cold_ids.begin(), p.cold_ids.end());
    uint64_t layer_actual = 0;
    for (uint64_t e = 0; e < p.n_experts; ++e) {
      bool is_cold = cold_set.count((int)e) != 0;
      const PackedProj &g = is_cold ? p.cg : p.hg;
      const PackedProj &u = is_cold ? p.cu : p.hu;
      const PackedProj &d = is_cold ? p.cd : p.hd;
      uint64_t triple = g.per_expert_bytes + u.per_expert_bytes + d.per_expert_bytes;
      (is_cold ? total_cold_bytes : total_hot_bytes) += triple;
      layer_actual += triple;
      std::string es = std::to_string(e);
      auto add = [&](const char *kind, const PackedProj &proj) {
        PlannedTensor pt;
        pt.name = "blk." + std::to_string(p.layer) + ".ffn_" + kind + "." +
                  es + ".weight";
        pt.type = proj.t->type;
        // 2D per-expert shape: the packed tensor's own shape minus the
        // trailing n_experts dim.
        pt.shape.assign(proj.t->shape.begin(), proj.t->shape.end() - 1);
        pt.from_cold = is_cold;
        pt.src_shard_idx = proj.t->shard_idx;
        pt.src_offset = proj.t->offset + e * proj.per_expert_bytes;
        pt.size_bytes = proj.per_expert_bytes;
        plan.push_back(std::move(pt));
      };
      // gate -> up -> down, back-to-back with zero gap (engine.hpp's
      // unpacked_expert_is_tight requires this for the single-DMA-per-
      // expert read path). Alignment padding, if any, only ever happens
      // BEFORE gate (the expert's own start), never between the three.
      add("gate", g);
      add("up", u);
      add("down", d);
    }
    total_would_be_all_hot +=
        p.n_experts * (p.hg.per_expert_bytes + p.hu.per_expert_bytes +
                        p.hd.per_expert_bytes);
    printf("  layer %3d: %4zu/%llu cold  (%.3f GB -> %.3f GB, -%.1f%%)\n",
           p.layer, p.cold_ids.size(), (unsigned long long)p.n_experts,
           (p.n_experts * (p.hg.per_expert_bytes + p.hu.per_expert_bytes +
                            p.hd.per_expert_bytes)) /
               1e9,
           layer_actual / 1e9,
           100.0 * (1.0 - (double)layer_actual /
                              (double)(p.n_experts *
                                       (p.hg.per_expert_bytes +
                                        p.hu.per_expert_bytes +
                                        p.hd.per_expert_bytes))));
  }
  uint64_t total_bytes = total_hot_bytes + total_cold_bytes;
  printf("[mix] TOTAL: %zu layer(s), %zu tensor(s), %.2f GB override shard "
         "(%.2f GB hot passthrough + %.2f GB cold) vs %.2f GB if those "
         "layers stayed all-hot -- %.2f GB saved\n",
         plans.size(), plan.size(), total_bytes / 1e9, total_hot_bytes / 1e9,
         total_cold_bytes / 1e9, total_would_be_all_hot / 1e9,
         (total_would_be_all_hot - total_bytes) / 1e9);

  if (dry_run) {
    printf("[mix] --dry-run: stopping before any write.\n");
    return 0;
  }

  std::error_code ec;
  fs::create_directories(output_dir, ec);
  if (ec) {
    fprintf(stderr, "[mix] FATAL: cannot create output directory %s: %s\n",
            output_dir.c_str(), ec.message().c_str());
    return 1;
  }

  if (symlink_baseline) {
    int n_linked = 0;
    for (auto &sp : base.shard_paths) {
      fs::path target = fs::absolute(sp);
      fs::path link = fs::path(output_dir) / fs::path(sp).filename();
      std::error_code lec;
      fs::remove(link, lec); // fine if it doesn't exist yet
      fs::create_symlink(target, link, lec);
      if (lec) {
        fprintf(stderr, "[mix] WARNING: could not symlink %s -> %s: %s\n",
                link.c_str(), target.c_str(), lec.message().c_str());
      } else {
        ++n_linked;
      }
    }
    printf("[mix] symlinked %d/%zu baseline shard(s) into %s\n", n_linked,
           base.shard_paths.size(), output_dir.c_str());
  }

  // ---- Assign to shard(s) by size, in plan order (already grouped by
  // layer, which is fine -- no cross-layer adjacency requirement). ----
  std::vector<std::vector<size_t>> shard_idxs;
  {
    uint64_t running = 0;
    shard_idxs.emplace_back();
    for (size_t i = 0; i < plan.size(); ++i) {
      if (running > 0 && running + plan[i].size_bytes > max_shard_bytes) {
        shard_idxs.emplace_back();
        running = 0;
      }
      shard_idxs.back().push_back(i);
      running += plan[i].size_bytes;
    }
  }
  int n_out_shards = (int)shard_idxs.size();

  std::vector<uint8_t> copy_buf(64ULL * 1024 * 1024);
  uint64_t total_written = 0;
  for (int s = 0; s < n_out_shards; ++s) {
    std::ostringstream fname;
    fname << "zzz_mix-part" << std::setw(5) << std::setfill('0') << (s + 1)
          << "-of" << std::setw(5) << std::setfill('0') << n_out_shards
          << ".gguf";
    std::string out_path = (fs::path(output_dir) / fname.str()).string();
    std::ofstream out(out_path, std::ios::binary | std::ios::trunc);
    if (!out.is_open()) {
      fprintf(stderr, "[mix] FATAL: cannot create output shard %s\n",
              out_path.c_str());
      return 1;
    }
    out.exceptions(std::ofstream::failbit | std::ofstream::badbit);

    std::vector<RawKV> kvs;
    kvs.push_back(make_uint32_kv("general.alignment", (uint32_t)alignment));
    uint64_t tensor_count = shard_idxs[s].size();
    uint64_t kv_count = kvs.size();

    // Plan this shard's own layout (out_offset relative to its own
    // data_start) before writing anything. Only the FIRST tensor of each
    // expert triple (gate) is aligned; up/down are forced tight right
    // after it (detected by name substring, since plan order already
    // guarantees gate->up->down adjacency per expert).
    uint64_t running = 0;
    for (size_t idx : shard_idxs[s]) {
      auto &t = plan[idx];
      bool is_gate = t.name.find(".ffn_gate.") != std::string::npos;
      if (is_gate)
        running = align_up(running, alignment);
      t.out_offset = running;
      running += t.size_bytes;
    }

    uint64_t written_so_far = 0;
    uint64_t last_progress_gb = 0;
    try {
      out.write("GGUF", 4);
      uint32_t version = 3;
      out.write(reinterpret_cast<const char *>(&version), 4);
      out.write(reinterpret_cast<const char *>(&tensor_count), 8);
      out.write(reinterpret_cast<const char *>(&kv_count), 8);
      for (auto &kv : kvs)
        write_raw_kv(out, kv);

      for (size_t idx : shard_idxs[s]) {
        auto &t = plan[idx];
        uint64_t klen = t.name.size();
        out.write(reinterpret_cast<const char *>(&klen), 8);
        out.write(t.name.data(), (std::streamsize)klen);
        uint32_t n_dims = (uint32_t)t.shape.size();
        out.write(reinterpret_cast<const char *>(&n_dims), 4);
        out.write(reinterpret_cast<const char *>(t.shape.data()),
                  (std::streamsize)(t.shape.size() * 8));
        uint32_t type_u32 = (uint32_t)t.type;
        out.write(reinterpret_cast<const char *>(&type_u32), 4);
        out.write(reinterpret_cast<const char *>(&t.out_offset), 8);
      }

      uint64_t pos_after_table = (uint64_t)out.tellp();
      uint64_t data_start = align_up(pos_after_table, alignment);
      if (data_start > pos_after_table) {
        std::vector<char> pad(data_start - pos_after_table, 0);
        out.write(pad.data(), (std::streamsize)pad.size());
      }

      for (size_t idx : shard_idxs[s]) {
        auto &t = plan[idx];
        if (t.out_offset > written_so_far) {
          std::vector<char> pad(t.out_offset - written_so_far, 0);
          out.write(pad.data(), (std::streamsize)pad.size());
        }
        const GGUFFile &src = t.from_cold ? cold : base;
        int src_fd = src.shard_fds[t.src_shard_idx];
        uint64_t remaining = t.size_bytes;
        uint64_t src_off = t.src_offset;
        while (remaining > 0) {
          size_t chunk = (size_t)std::min<uint64_t>(remaining, copy_buf.size());
          ssize_t r = pread(src_fd, copy_buf.data(), chunk, (off_t)src_off);
          if (r <= 0 || (size_t)r != chunk) {
            fprintf(stderr,
                    "[mix] FATAL: short/failed pread for %s (wanted %zu, "
                    "got %zd) at src_offset=%llu\n",
                    t.name.c_str(), chunk, (ssize_t)r,
                    (unsigned long long)src_off);
            return 1;
          }
          out.write(reinterpret_cast<const char *>(copy_buf.data()),
                    (std::streamsize)chunk);
          remaining -= chunk;
          src_off += chunk;
        }
        written_so_far = t.out_offset + t.size_bytes;
        total_written += t.size_bytes;
        uint64_t gb_now = total_written / (1ULL << 30);
        if (gb_now != last_progress_gb) {
          last_progress_gb = gb_now;
          printf("\r[mix] written %llu GB...", (unsigned long long)gb_now);
          fflush(stdout);
        }
      }
      out.close();
    } catch (const std::ios_base::failure &) {
      int err = errno;
      fprintf(stderr,
              "\n[mix] FATAL: write to output shard %s failed after %.2f "
              "GB of this shard's data (errno=%d: %s) -- likely ENOSPC. "
              "This output file is INCOMPLETE -- do not use it.\n",
              out_path.c_str(), (double)written_so_far / 1e9, err,
              strerror(err));
      return 1;
    }
    printf("\n[mix] Wrote shard %d/%d: %s (%llu tensors, %.2f GB)\n", s + 1,
           n_out_shards, out_path.c_str(),
           (unsigned long long)tensor_count,
           (double)written_so_far / 1e9);
  }
  printf("[mix] Done. Total override bytes written: %.2f GB across %d "
         "shard(s).\n",
         total_written / 1e9, n_out_shards);
  printf("[mix] Next: gguf_tool verify %s   (then a real smoke-test run)\n",
         output_dir.c_str());
  return 0;
}

} // namespace gguf_tool
