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

// MOESTREAM_GLM5NEXT_OFFLOAD_ASYNC=1: the arena lock covers only bookkeeping; expert reads run outside it (see
// ExpertResidency::begin/finish), so a background prefetch and a demand miss overlap instead of serializing on I/O.
bool offload_async() {
  static const bool on = [] {
    return moestream::graphcfg::flag_on("MOESTREAM_GLM5NEXT_OFFLOAD_ASYNC");
  }();
  return on;
}

int offload_first_layer() {
  static const int first = [] {
    return moestream::graphcfg::atoi_or("MOESTREAM_GLM5NEXT_OFFLOAD_FIRST_LAYER", -1);
  }();
  return first;
}

bool offload_layer(int layer) {
  const int f = offload_first_layer();
  return f >= 0 && layer >= f;
}

std::string offload_env(const char *name, const char *dflt)  {
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
bool early_ready_wanted() {
  // On by default (4-pair A/B: +1.9% decode, logits identical);
  // MOESTREAM_GLM5NEXT_EARLY_READY=0 opts out.
  static const bool on = moestream::graphcfg::flag_unless_off("MOESTREAM_GLM5NEXT_EARLY_READY");
  return on;
}

// Overlap a layer's SSD miss reads with the CPU compute of its already-resident
// cold experts (lane only). Bit-exact: the two partial outputs are disjoint routes.
// On by default (order-balanced A/B: 4.29 -> 4.92 tok/s, logits and tokens identical);
// MOESTREAM_GLM5NEXT_SPLIT_LOAD=0 opts out.
bool split_load_wanted() {
  static const bool on = moestream::graphcfg::flag_unless_off("MOESTREAM_GLM5NEXT_SPLIT_LOAD");
  return on;
}

int gpu_assist_slots() {
  static const int n = std::max(0, std::min(8, moestream::graphcfg::atoi_or("MOESTREAM_GLM5NEXT_GPU_ASSIST", 0)));
  return n;
}

} // namespace glm5next_graph

#endif // MOESTREAM_USE_GGML_CUDA
