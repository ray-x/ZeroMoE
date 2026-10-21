// GLM5Next full graph: Process state, environment-driven settings and feature gates.
//
// Split out of the former single-file ggml_qwen4exp_graph.cpp; shared
// declarations live in ggml_glm5next_graph_impl.hpp.

#include "ggml_glm5next_graph_impl.hpp"

#if defined(MOESTREAM_USE_GGML_CUDA)

using namespace glm5next_graph;

namespace glm5next_graph {

bool env_enabled() {
  static const bool enabled = [] {
    // Full graph is the GLM5NEXT execution path; MOESTREAM_GLM5NEXT_FULL_GRAPH=0 opts out (legacy layered path).
    const char *v = moestream::graphcfg::env("MOESTREAM_GLM5NEXT_FULL_GRAPH");
    return !v || !v[0] || v[0] != '0';
  }();
  return enabled;
}

bool graph_enabled_for(const StreamingEngine &engine) {
  if (engine.family_ops.family == ModelFamily::BailingMoE3)
    return ling_full_graph_enabled();
  return glm5next_graph::glm_family(engine.family_ops.family) && env_enabled();
}

// MOESTREAM_GLM5NEXT_OFFLOAD_ASYNC=1: the arena lock covers only bookkeeping; expert reads run outside it (see
// ExpertResidency::begin/finish), so a background prefetch and a demand miss overlap instead of serializing on I/O.
bool offload_async() {
  static const bool on = [] {
    return moestream::graphcfg::flag_on("MOESTREAM_LING_OFFLOAD_ASYNC") ||
           moestream::graphcfg::flag_on("MOESTREAM_GLM5NEXT_OFFLOAD_ASYNC");
  }();
  return on;
}

int offload_first_layer() {
  static const int first = [] {
    const char *ling = moestream::graphcfg::env("MOESTREAM_LING_OFFLOAD_FIRST_LAYER");
    return ling && ling[0]
               ? std::atoi(ling)
               : moestream::graphcfg::atoi_or(
                     "MOESTREAM_GLM5NEXT_OFFLOAD_FIRST_LAYER", -1);
  }();
  return first;
}

bool offload_layer(int layer) {
  const int f = offload_first_layer();
  return f >= 0 && layer >= f;
}

std::string offload_env(const char *name, const char *dflt)  {
  constexpr const char *prefix = "MOESTREAM_GLM5NEXT_";
  if (std::strncmp(name, prefix, std::strlen(prefix)) == 0) {
    const std::string ling_name =
        std::string("MOESTREAM_LING_") + (name + std::strlen(prefix));
    const char *ling = moestream::graphcfg::env(ling_name.c_str());
    if (ling && ling[0])
      return ling;
  }
  const char *v = moestream::graphcfg::env(name);
  return v && v[0] ? v : dflt;
}

// ---------------------------------------------------------------------------
// Pilot staging (MOESTREAM_GLM5NEXT_PILOT_STAGE=K): in decode, layer l's graph predicts the K most likely COLD experts of layer l+1
// (layer l+1's router applied to layer l's FFN input, hot experts masked out), a worker copies them DDR -> pinned -> VRAM ring while
// layer l's FFN and layer l+1's attention run, and layer l+1 computes them on the GPU; the CPU cold branch skips whatever was staged.
// A staged expert that was not ready or not routed simply stays on the CPU branch, so correctness never depends on timing.
// ---------------------------------------------------------------------------
int pilot_stage_k() {
  static const int k = [] {
    const char *v = moestream::graphcfg::env("MOESTREAM_GLM5NEXT_PILOT_STAGE");
    return v && v[0] ? std::max(0, std::min(8, std::atoi(v))) : 0;
  }();
  return k;
}

// Number of next-layer experts the pilot ranks (>= the staged K). Predicted experts that are not resident in DDR are read from the SSD ahead
// of the layer that needs them; only the first pilot_stage_k() resident ones are copied into the VRAM ring.
bool pilot_prefetch_on() {
  static const bool on = [] {
    return moestream::graphcfg::atoi_or("MOESTREAM_GLM5NEXT_PILOT_PREFETCH", 0) > 0;
  }();
  return on;
}

// MOESTREAM_GLM5NEXT_PILOT_COPY=0: prefetch-only mode. The pilot reads predicted experts into DDR in the background; nothing is copied to the
// VRAM ring and no layer waits for the prefetch, so the CPU computes every cold route as without staging.
bool pilot_copy_on() {
  static const bool on = [] {
    return moestream::graphcfg::flag_unless_off("MOESTREAM_GLM5NEXT_PILOT_COPY");
  }();
  return on;
}

int pilot_prefetch_k() {
  static const int k = [] {
    const char *v = moestream::graphcfg::env("MOESTREAM_GLM5NEXT_PILOT_PREFETCH");
    return std::max(pilot_stage_k(), v && v[0] ? std::max(0, std::min(8, std::atoi(v))) : 0);
  }();
  return k;
}

// MOESTREAM_GLM5NEXT_PILOT_LOOKAHEAD=k (prefetch-only mode): layer l predicts the cold experts of layer l+k (using layer l+k's
// router on layer l's FFN input) so the read has k layers of lead time. Staging copies keep k=1 (the ring has two sets).
int pilot_lookahead() {
  static const int k = [] {
    const char *v = moestream::graphcfg::env("MOESTREAM_GLM5NEXT_PILOT_LOOKAHEAD");
    int want = v && v[0] ? std::max(1, std::min(4, std::atoi(v))) : 1;
    if (want > 1) {
      const char *cv = moestream::graphcfg::env("MOESTREAM_GLM5NEXT_PILOT_COPY");
      if (!(cv && cv[0] == '0'))
        want = 1;
    }
    return want;
  }();
  return k;
}

// MOESTREAM_GLM5NEXT_PILOT_MIN_SCORE=x: only prefetch pilot candidates whose router score is >= x.
float pilot_min_score() {
  static const float v = [] {
    return (float)moestream::graphcfg::atof_or("MOESTREAM_GLM5NEXT_PILOT_MIN_SCORE", -1e30f);
  }();
  return v;
}

bool dynamic_cache_enabled() {
  static const bool enabled = [] {
    return moestream::graphcfg::flag_on("MOESTREAM_GLM5NEXT_DYNAMIC_CACHE");
  }();
  return enabled;
}

bool route_stats_enabled() {
  static const bool enabled = [] {
    const char *v = moestream::graphcfg::env("MOESTREAM_GLM5NEXT_CACHE_STATS");
    return (v && v[0] == '1') || dynamic_cache_enabled();
  }();
  return enabled;
}

int env_device_expert_layers() {
  const char *v = moestream::graphcfg::env("MOESTREAM_GLM5NEXT_GRAPH_PRELOAD_LAYERS");
  if (!v || !v[0])
    return 0;
  char *end = nullptr;
  long n = std::strtol(v, &end, 10);
  if (end == v || *end != '\0' || n <= 0)
    return 0;
  return static_cast<int>(std::min<long>(n, 45));
}

bool type_ok(GGUFType type) {
  return static_cast<uint32_t>(type) < static_cast<uint32_t>(GGML_TYPE_COUNT);
}

bool layer_timing_env() {
  static const bool on = [] {
    return moestream::graphcfg::flag_on("MOESTREAM_GLM5NEXT_GRAPH_LAYER_TIMING");
  }();
  return on;
}

int dynamic_upload_budget_override() {
  const char *value =
      moestream::graphcfg::env("MOESTREAM_GLM5NEXT_DYNAMIC_CACHE_UPLOADS_PER_TOKEN");
  if (!value || !value[0])
    return -1;
  char *end = nullptr;
  const long parsed = std::strtol(value, &end, 10);
  if (end == value || *end != '\0' || parsed < 0)
    return -1;
  return (int)std::min<long>(parsed, 128);
}

double dynamic_upload_fraction() {
  const char *value =
      moestream::graphcfg::env("MOESTREAM_GLM5NEXT_DYNAMIC_CACHE_SWAP_FRAC");
  if (!value || !value[0])
    return 0.25;
  char *end = nullptr;
  const double parsed = std::strtod(value, &end);
  return end != value && *end == '\0' && parsed >= 0.0 && parsed <= 1.0
             ? parsed
             : 0.25;
}

// Run each layer's cold (CPU) experts on a private lane that starts as soon as the
// routed ids reach the host, instead of after the layer's GPU work. Decode only.
bool early_ready_wanted(bool has_overlap_work) {
  // The accepted lane gain was measured with offload / GPU-hot work to overlap.
  // A fully RAM-resident CPU-only branch runs faster in the ordinary graph.
  // Preserve the explicit flag's legacy semantics; an unset flag selects per layer.
  static const int override_mode = [] {
    const char *value = moestream::graphcfg::env("MOESTREAM_GLM5NEXT_EARLY_READY");
    return value ? (value[0] == '0' ? 0 : 1) : -1;
  }();
  return override_mode < 0 ? has_overlap_work : override_mode != 0;
}

// Overlap a layer's SSD miss reads with the CPU compute of its already-resident
// cold experts (lane only). Bit-exact: the two partial outputs are disjoint routes.
// On by default (order-balanced A/B: 4.29 -> 4.92 tok/s, logits and tokens identical);
// MOESTREAM_GLM5NEXT_SPLIT_LOAD=0 opts out.
bool split_load_wanted() {
  static const bool on = moestream::graphcfg::flag_unless_off("MOESTREAM_GLM5NEXT_SPLIT_LOAD");
  return on;
}

bool lane_with_gpu1_wanted() {
  static const bool on = moestream::graphcfg::flag_unless_off("MOESTREAM_GLM5NEXT_LANE_WITH_GPU1");
  return on;
}

// Up to N resident cold experts per layer run on a second CUDA stream while the CPU computes the rest. On by
// default (3): 58 GiB pool 8.1 -> 9.4-9.8 tok/s, dual GPU 78.3 -> 65.2 ms/token. Numerics differ from the CPU
// kernels by the same amount as moving an expert between valid placements (teacher-forced 48 tokens: top-1 92-96 %,
// mean KL 0.03-0.05 for assist AND for plain 3090-only vs dual-GPU). MOESTREAM_GLM5NEXT_GPU_ASSIST=0 disables it
// and gives bit-identical CPU-lane output.
// Optional second assist device (its own ring and PCIe link): MOESTREAM_GLM5NEXT_ASSIST2_DEVICE=<cuda device> (default: device 1 on a
// dual-GPU run with a second-GPU hot set, else off) and MOESTREAM_GLM5NEXT_ASSIST2_SLOTS=<n> (default 1).
int second_gpu_device(int core_device) { return core_device == 0 ? 1 : 0; }

int gpu_assist2_device(int core_device) {
  // Unset: use the non-core CUDA device when two devices are visible and the second GPU already holds a hot set; -1 turns it off.
  static const int d = [core_device] {
    const char *v = moestream::graphcfg::env("MOESTREAM_GLM5NEXT_ASSIST2_DEVICE");
    if (v && v[0])
      return std::atoi(v);
    int n = 0;
    const char *hot1 = moestream::graphcfg::env("MOESTREAM_GLM5NEXT_GPU1_HOT_GIB");
    return cudaGetDeviceCount(&n) == cudaSuccess && n >= 2 && hot1 && hot1[0] ? second_gpu_device(core_device) : -1;
  }();
  return d;
}
int gpu_assist2_slots() {
  static const int n = std::max(0, std::min(8, moestream::graphcfg::atoi_or("MOESTREAM_GLM5NEXT_ASSIST2_SLOTS", 1)));
  return n;
}

int gpu_assist_slots() {
  static const int n = std::max(0, std::min(8, moestream::graphcfg::atoi_or("MOESTREAM_GLM5NEXT_GPU_ASSIST", 3)));
  return n;
}

// Which expert types are worth assisting: the codebook types (IQ1/IQ2/IQ3, Q2_0) are compute-bound on the CPU (11-26 GB/s of
// weights) and small, so a PCIe copy costs about what the CPU saves. The K- and legacy 4-8 bit types stream at 50-60 GB/s on the
// CPU, so the copy of their ~1.3-2x larger experts costs 2.8-3x the CPU time and assisting them slows the layer down (GSQ-RCO
// Q2_K: assist 3 = 88 ms/token vs 70.6 without). MOESTREAM_GLM5NEXT_GPU_ASSIST_ANY_TYPE=1 lifts the filter.
static bool gpu_assist_type_ok(int type) {
  switch (type) {
  case GGML_TYPE_IQ1_S:
  case GGML_TYPE_IQ1_M:
  case GGML_TYPE_IQ2_XXS:
  case GGML_TYPE_IQ2_XS:
  case GGML_TYPE_IQ2_S:
  case GGML_TYPE_IQ3_XXS:
  case GGML_TYPE_IQ3_S:
  case GGML_TYPE_Q2_0:
    return true;
  default:
    return false;
  }
}

bool gpu_assist_layer_ok(const FullGraph &g, int layer) {
  static const bool any = moestream::graphcfg::flag_on("MOESTREAM_GLM5NEXT_GPU_ASSIST_ANY_TYPE");
  if (any)
    return true;
  for (const char *stem : {"ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"}) {
    const GGUFTensor *m = model_tensor(*g.engine, "blk." + std::to_string(layer) + "." + stem);
    if (!m || !gpu_assist_type_ok(static_cast<int>(m->type)))
      return false;
  }
  return true;
}

} // namespace glm5next_graph

bool ling_full_graph_enabled() {
  return moestream::graphcfg::flag_on("MOESTREAM_LING_FULL_GRAPH");
}

#endif // MOESTREAM_USE_GGML_CUDA
