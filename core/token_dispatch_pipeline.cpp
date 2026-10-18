#include "active_layer_upload_workers.hpp"
#include "async_prefetcher.hpp"
#include "cpu_expert_executor.hpp"
#include "gpu_expert_executor.hpp"
#include "dense_dispatch.hpp"
#include "engine.hpp"
#include "expert_activation.hpp"
#include "ggml_gdn_proj_backend.hpp"
#include "ggml_glm5next_graph.hpp"
#include "ggml_hc_backend.hpp"
#include "ggml_moe_backend.hpp"
#include "ggml_qwen_shared_runner.hpp"
#include "ggml_qwen4exp_graph.hpp"
#include "kernels_cpu.hpp"
#include "kquant_dequant.hpp"
#include "logger.hpp"
#include "mtp_speculative.hpp"
#include "moe_pipeline.hpp"
#include "moe_router.hpp"
#include "perf_timer.hpp"
#include "prefill_expert_stream.hpp"
#include "qwen_cpu_offload_ring.hpp"
#include "ram_manager.hpp"
#include "scratch_buffers.hpp"
#include "tensor_trace.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <immintrin.h>
#include <limits>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

extern void mla_attention_absorbed_gguf(StreamingEngine &engine, int layer_idx,
                                        const float *Q_roped, int S,
                                        int pos_base, float *out);
extern void dense_attention_gguf(StreamingEngine &engine, int layer_idx,
                                 const float *hidden_states, int S,
                                 int pos_base, float *out);
extern bool qwen4exp_hc_device_live();
extern bool qwen4exp_hc_post_combine_device(StreamingEngine &engine,
                                            int layer_idx, int sub_block, int S,
                                            int D,
                                            const void *device_sub_block_out,
                                            const void *producer_event);
extern bool qwen4exp_hc_post_combine_device_raw(StreamingEngine &engine,
                                                int layer_idx, int sub_block,
                                                int S, int D,
                                                const void *device_data);
extern bool qwen4exp_hc_prepare_host_layer_input(StreamingEngine &engine,
                                                 int layer_idx, int S);
extern bool qwen4exp_hc_download_streams(StreamingEngine &engine, int S);
extern bool qwen4exp_hc_commit_host_layer_input(StreamingEngine &engine,
                                                int layer_idx, int S);
extern bool qwen4exp_copy_gdn_device_output_to_host(const void *device_output,
                                                    float *host, int S);

#include "moe_pipeline_internal.hpp"
#include "shared_expert_executor.hpp"
#include "token_dispatch_pipeline.hpp"

static bool expert_ffn_cpu_qwen_grouped_quantized(
    const float *input, float *output,
    const std::vector<MoEExpertBlock *> &experts, const float *weights, int D,
    int moe_inter, float swiglu_clamp, int layer_idx) {
  const int n = static_cast<int>(experts.size());
  if (!input || !output || !weights || n <= 0 || D <= 0 || moe_inter <= 0)
    return false;
  std::vector<const uint8_t *> gate_w((size_t)n);
  std::vector<const uint8_t *> up_w((size_t)n);
  std::vector<const uint8_t *> down_w((size_t)n);
  std::vector<float *> gate((size_t)n);
  std::vector<float *> up((size_t)n);
  std::vector<float *> down((size_t)n);
  std::vector<float> gate_storage((size_t)n * moe_inter);
  std::vector<float> up_storage((size_t)n * moe_inter);
  std::vector<float> down_storage((size_t)n * D);
  GGUFType gate_type = GGUFType::F32;
  GGUFType up_type = GGUFType::F32;
  GGUFType down_type = GGUFType::F32;
  bool types_initialized = false;
  for (int i = 0; i < n; ++i) {
    MoEExpertBlock *e = experts[(size_t)i];
    if (!e || e->vram_shadow || !e->gate_proj.host_ptr ||
        !e->up_proj.host_ptr || !e->down_proj.host_ptr)
      return false;
    if (!types_initialized) {
      gate_type = e->gate_proj.meta.type;
      up_type = e->up_proj.meta.type;
      down_type = e->down_proj.meta.type;
      types_initialized = true;
    }
    if (e->gate_proj.meta.type != gate_type ||
        e->up_proj.meta.type != up_type || e->down_proj.meta.type != down_type)
      return false;
    gate_w[(size_t)i] = e->gate_proj.host_ptr;
    up_w[(size_t)i] = e->up_proj.host_ptr;
    down_w[(size_t)i] = e->down_proj.host_ptr;
    gate[(size_t)i] = gate_storage.data() + (size_t)i * moe_inter;
    up[(size_t)i] = up_storage.data() + (size_t)i * moe_inter;
    down[(size_t)i] = down_storage.data() + (size_t)i * D;
  }

  PERF_START(qwen_cpu_group);
  // Prefer the persistent GGML CPU MUL_MAT_ID graph.  It groups all selected
  // experts and all three projections into one CPU backend submission, which
  // is the important difference from the older per-projection OpenMP helper
  // below.  Keep the old implementation as a correctness-preserving
  // fallback for unsupported/mixed tensor layouts or non-GGML builds.
  // The persistent graph currently models plain SwiGLU.  Do not route a
  // clamped Qwen variant through it until its clamp semantics are represented
  // in the graph; the established CPU path remains the safe fallback.
  static const int cpu_graph_mode = [] {
    const char *env = std::getenv("MOESTREAM_QWEN_CPU_GRAPH");
    if (env && env[0] == '0')
      return 0; // force the established path
    if (env && env[0] == '2')
      return 2; // validate, then use established path
    return 1;   // graph, unless it cannot represent it
  }();
  std::vector<float> graph_reference;
  std::vector<float> output_before;
  if (cpu_graph_mode == 2)
    output_before.assign(output, output + D);
  if (cpu_graph_mode != 0 && GgmlQwenCpuMoeRunner::compiled_in() &&
      swiglu_clamp <= 0.0f) {
    std::vector<GgmlQwenCpuExpertRef> refs((size_t)n);
    for (int i = 0; i < n; ++i) {
      MoEExpertBlock *e = experts[(size_t)i];
      refs[(size_t)i] = {e->gate_proj.host_ptr,
                         e->up_proj.host_ptr,
                         e->down_proj.host_ptr,
                         e->gate_proj.meta.size_bytes,
                         e->up_proj.meta.size_bytes,
                         e->down_proj.meta.size_bytes,
                         static_cast<uint32_t>(e->gate_proj.meta.type),
                         static_cast<uint32_t>(e->up_proj.meta.type),
                         static_cast<uint32_t>(e->down_proj.meta.type)};
    }
    std::vector<float> graph_output((size_t)D, 0.0f);
    const bool cpu_diag = [] {
      const char *env = std::getenv("MOESTREAM_QWEN4EXP_DIAG_LAYER0_IQ2");
      return env && env[0] == '1';
    }();
    if (cpu_diag)
      qwen_cpu_moe_diag_set_layer(layer_idx);
    const bool graph_ok = GgmlQwenCpuMoeRunner::instance().run(
        input, graph_output.data(), refs.data(), n, weights, D, moe_inter, n);
    if (cpu_diag)
      qwen_cpu_moe_diag_set_layer(-1);
    if (graph_ok) {
      static std::atomic<bool> logged_cpu_graph{false};
      if (!logged_cpu_graph.exchange(true, std::memory_order_relaxed))
        LOG_STREAM_INFO()
            << "[CPU-MOE] Qwen persistent GGML MUL_MAT_ID CPU graph active "
               "(topk="
            << n << ", threads=" << omp_get_max_threads() << ")\n";
      if (cpu_graph_mode == 1) {
        for (int d = 0; d < D; ++d)
          output[d] += graph_output[(size_t)d];
        PERF_END(qwen_cpu_group, t_cpu_expert);
        perf().n_expert_cpu_calls.fetch_add((uint64_t)n,
                                            std::memory_order_relaxed);
        (void)layer_idx;
        return true;
      }
      graph_reference = std::move(graph_output);
    }
  }

  auto grouped_gate_up = [&](float *const *dst, const uint8_t *const *weights,
                             GGUFType type) {
    switch (type) {
    case GGUFType::IQ1_M:
      return matmul_iq1_m_ggml_q8k_grouped(dst, input, weights, n, 1, D,
                                           moe_inter);
    case GGUFType::IQ2_XXS:
      // Qwen's grouped CPU path uses GGML's production kernel directly,
      // matching Llama.cpp's MUL_MAT_ID implementation.  Keep the existing
      // AVX2-prototype gate for all non-Qwen callers; this switch is reached
      // only by the Qwen4Exp-specific helper below.
      return matmul_iq2_xxs_ggml_q8k_grouped(dst, input, weights, n, 1, D,
                                             moe_inter);
    case GGUFType::IQ3_XXS:
      return matmul_iq3_xxs_ggml_q8k_grouped(dst, input, weights, n, 1, D,
                                             moe_inter);
    default:
      return false;
    }
  };
  if (!grouped_gate_up(gate.data(), gate_w.data(), gate_type) ||
      !grouped_gate_up(up.data(), up_w.data(), up_type)) {
    PERF_END(qwen_cpu_group, t_cpu_expert);
    return false;
  }
  for (int i = 0; i < n; ++i) {
    if (swiglu_clamp_fused()) {
      swiglu_clamp_fused_inplace(gate[(size_t)i], up[(size_t)i], moe_inter,
                                 swiglu_clamp);
    } else {
      silu_inplace(gate[(size_t)i], moe_inter);
      elem_mul(gate[(size_t)i], gate[(size_t)i], up[(size_t)i], moe_inter);
      swiglu_clamp_inplace(gate[(size_t)i], moe_inter, swiglu_clamp);
    }
  }
  bool down_grouped = false;
  // The IQ4_NL q8_0 kernel is kept behind an explicit diagnostic switch until
  // its numerical output is checked against cpu_quant_matmul on this model's
  // exact GGUF rows.  The validated default remains the old down projection;
  // gate/up grouping is independent and remains enabled below.
  static const bool enable_grouped_iq4_nl = [] {
    const char *env = std::getenv("MOESTREAM_QWEN4EXP_GROUPED_IQ4_NL");
    return env && env[0] == '1';
  }();
  if (enable_grouped_iq4_nl && down_type == GGUFType::IQ4_NL) {
    std::vector<const float *> act((size_t)n);
    for (int i = 0; i < n; ++i)
      act[(size_t)i] = gate[(size_t)i];
    down_grouped = matmul_iq4_nl_ggml_q8_0_grouped_inputs(
        down.data(), act.data(), down_w.data(), n, 1, moe_inter, D);
  }
  if (!down_grouped) {
    for (int i = 0; i < n; ++i) {
      if (!cpu_quant_matmul(down_type, down_w[(size_t)i], gate[(size_t)i],
                            down[(size_t)i], 1, moe_inter, D)) {
        PERF_END(qwen_cpu_group, t_cpu_expert);
        return false;
      }
    }
  }
  for (int d = 0; d < D; ++d) {
    float sum = 0.0f;
    for (int i = 0; i < n; ++i)
      sum += weights[i] * down[(size_t)i][d];
    output[d] += sum;
  }
  if (!graph_reference.empty()) {
    float max_abs = 0.0f;
    float max_rel = 0.0f;
    for (int d = 0; d < D; ++d) {
      const float legacy = output[d] - output_before[(size_t)d];
      const float graph = graph_reference[(size_t)d];
      max_abs = std::max(max_abs, std::abs(legacy - graph));
      max_rel = std::max(max_rel, std::abs(legacy - graph) /
                                      std::max(1.0f, std::abs(legacy)));
    }
    static std::atomic<bool> logged_graph_compare{false};
    if (!logged_graph_compare.exchange(true, std::memory_order_relaxed))
      LOG_STREAM_INFO()
          << "[CPU-MOE-COMPARE] Qwen persistent GGML graph vs established "
             "CPU path: layer="
          << layer_idx << " topk=" << n << " max_abs=" << max_abs
          << " max_rel=" << max_rel << " (graph result not used)\n";
  }
  PERF_END(qwen_cpu_group, t_cpu_expert);
  perf().n_expert_cpu_calls.fetch_add((uint64_t)n, std::memory_order_relaxed);
  (void)layer_idx;
  return true;
}
bool qwen4exp_ram_cpu_moe_layer(const StreamingEngine &engine,
                                int layer_idx, int n_experts,
                                       QwenRamCpuMoeLayer &out) {
  // Projection-compact/unpacked layers are exactly the layout represented by
  // graph_ram_arenas for the Qwen CPU/GPU parity path.  Do not reject them
  // here: the validation below supplies the three plane bases and strides
  // consumed by the CPU runner.  The old layer_is_unpacked() guard made the
  // full graph scheduler candidate decline before construction, silently
  // sending Layer 0 back through the legacy host shared-expert path.
  if (layer_idx < 0 || n_experts <= 0 ||
      !engine.graph_ram_layer_ready(layer_idx))
    return false;

  const auto *arena = engine.graph_ram_arena(layer_idx);
  if (!arena || arena->n_experts != n_experts || !arena->base ||
      arena->gate_bytes == 0 || arena->up_bytes == 0 ||
      arena->down_bytes == 0 ||
      (!arena->projection_compact && arena->stride == 0))
    return false;

  const auto &slots = engine.layers_experts[(size_t)layer_idx];
  if (slots.size() < static_cast<size_t>(n_experts))
    return false;
  const MoEExpertBlock &first = slots[0];
  // Projection-compact arenas deliberately expose three independent planes:
  // their per-expert slab_buffer/stride fields are empty by design.  The
  // strided CPU runner consumes those plane bases directly, so validate the
  // representation that was actually allocated instead of requiring the
  // interleaved-slab metadata used by the legacy cache.
  if (arena->projection_compact) {
    if (arena->gate_stride == 0 || arena->up_stride == 0 ||
        arena->down_stride == 0)
      return false;
  } else if (!first.slab_buffer || first.slab_capacity < arena->stride) {
    return false;
  }
  if (!first.gate_proj.host_ptr || !first.up_proj.host_ptr ||
      !first.down_proj.host_ptr)
    return false;
  // graph_ram_layer_ready() performed the complete 512-slot validation and
  // caches it for the lifetime of this immutable shared arena. Repeating
  // that walk here made every decode token pay another O(n_experts) check for
  // every CPU layer; graph-backed slices cannot be reclaimed or resized.

  // Use the arena's explicit plane offsets.  Projection-compact arenas may
  // insert alignment padding between gate/up/down; deriving the addresses
  // from the raw plane byte counts makes the CPU graph read from the wrong
  // projection whenever that padding is present.
  out.meta = {arena->base + arena->gate_offset,
              arena->base + arena->up_offset,
              arena->base + arena->down_offset,
              arena->gate_bytes,
              arena->up_bytes,
              arena->down_bytes,
              static_cast<uint32_t>(first.gate_proj.meta.type),
              static_cast<uint32_t>(first.up_proj.meta.type),
              static_cast<uint32_t>(first.down_proj.meta.type)};
  out.gate_plane = arena->base + arena->gate_offset;
  out.up_plane = arena->base + arena->up_offset;
  out.down_plane = arena->base + arena->down_offset;
  out.expert_stride = arena->stride;
  out.projection_compact = arena->projection_compact;
  out.gate_stride = arena->projection_compact ? arena->gate_stride
                                               : arena->stride;
  out.up_stride = arena->projection_compact ? arena->up_stride
                                             : arena->stride;
  out.down_stride = arena->projection_compact ? arena->down_stride
                                               : arena->stride;
  if (const char *debug = std::getenv("MOESTREAM_QWEN_DECODE_SCHED_DEBUG");
      debug && debug[0] == '1' && layer_idx == 0) {
    LOG_WARN("[QWEN-CPU-ARENA] layer=%d compact=%d arena_stride=%zu "
             "planes=%p/%p/%p strides=%zu/%zu/%zu\n",
             layer_idx, out.projection_compact ? 1 : 0, out.expert_stride,
             out.gate_plane, out.up_plane, out.down_plane, out.gate_stride,
             out.up_stride, out.down_stride);
  }
  return true;
}

static bool qwen_cpu_compact_graph_enabled() {
  static const bool enabled = [] {
    const char *env = std::getenv("MOESTREAM_QWEN_CPU_COMPACT_GRAPH");
    return env && env[0] == '1';
  }();
  return enabled;
}

static bool qwen_cpu_direct_gpu_overlap_probe() {
  static const bool enabled = [] {
    const char *env = std::getenv("MOESTREAM_QWEN_CPU_DIRECT_GPU_OVERLAP");
    return env && env[0] == '1';
  }();
  return enabled;
}

static bool qwen_cpu_strided_grouped_enabled(const StreamingEngine &engine) {
  static const int env_override = [] {
    const char *env = std::getenv("MOESTREAM_QWEN_CPU_STRIDED_GROUPED");
    if (!env || !env[0])
      return -1;
    return env[0] == '0' ? 0 : (env[0] == '1' ? 1 : -1);
  }();
  if (env_override == 0)
    return false;
  if (env_override == 1)
    return true;
  return engine.qwen4exp_cpu_strided_grouped;
}

static bool qwen_cpu_gpu_combine_enabled() {
  static const bool enabled = [] {
    const char *env = std::getenv("MOESTREAM_QWEN_CPU_GPU_COMBINE");
    return env && env[0] == '1';
  }();
  return enabled;
}

// Qwen-only direct strided CPU path.  The persistent MUL_MAT_ID graph above
// is the safe default, but its per-layer graph submission still carries
// generic GGML scheduling overhead.  For the common UD-IQ1_M layout, the
// selected experts can be addressed directly in the already-populated DDR
// arena and grouped into one activation-quantization/dot pass per
// projection.  Keep this behind an environment switch until a real hardware
// A/B proves it; no other model family can reach this helper.
static bool qwen_cpu_stage_timing_enabled() {
  static const bool enabled = [] {
    const char *env = std::getenv("MOESTREAM_QWEN_CPU_STAGE_TIMING");
    return env && env[0] == '1';
  }();
  return enabled;
}

static bool qwen_cpu_fused_iq1_iq4_enabled() {
  static const bool enabled = [] {
    const char *env = std::getenv("MOESTREAM_QWEN_CPU_FUSED_IQ1_IQ4");
    return env && env[0] == '1';
  }();
  return enabled;
}

static bool qwen_cpu_fused_grouped_enabled(const StreamingEngine &engine) {
  static const int env_override = [] {
    const char *env = std::getenv("MOESTREAM_QWEN_CPU_FUSED_GROUPED");
    if (!env || !env[0])
      return -1;
    return env[0] == '0' ? 0 : (env[0] == '1' ? 1 : -1);
  }();
  return env_override >= 0 ? env_override == 1
                           : engine.qwen4exp_cpu_fused_grouped;
}

static bool qwen_cpu_fused_iq2_iq4_enabled() {
  static const bool enabled = [] {
    const char *env = std::getenv("MOESTREAM_QWEN_CPU_FUSED_IQ2_IQ4");
    return env && env[0] == '1';
  }();
  return enabled;
}

static bool qwen_cpu_fused_q4_0_enabled() {
  static const bool enabled = [] {
    const char *env = std::getenv("MOESTREAM_QWEN_CPU_FUSED_Q4_0");
    return env && env[0] == '1';
  }();
  return enabled;
}

static bool qwen4exp_cpu_strided_grouped_quantized(
    const StreamingEngine &engine, const float *input, float *output,
    const QwenRamCpuMoeLayer &layer,
    const int *expert_ids, const float *router_weights, int n_experts,
    int topk, int D, int moe_inter, float swiglu_clamp) {
  if (!input || !output || !expert_ids || !router_weights ||
      !layer.gate_plane || !layer.up_plane || !layer.down_plane ||
      layer.expert_stride == 0 || n_experts <= 0 || topk <= 0 ||
      topk > n_experts || D <= 0 || moe_inter <= 0 || swiglu_clamp != 0.0f)
    return false;

  const GGUFType gate_type = static_cast<GGUFType>(layer.meta.gate_type);
  const GGUFType up_type = static_cast<GGUFType>(layer.meta.up_type);
  const GGUFType down_type = static_cast<GGUFType>(layer.meta.down_type);
  const bool is_q4_0_combo = gate_type == GGUFType::Q4_0 &&
                             up_type == GGUFType::Q4_0 &&
                             down_type == GGUFType::Q4_0;
  if (!is_q4_0_combo) {
    if (down_type != GGUFType::IQ4_NL)
      return false;

    auto grouped_gate_up_supported = [](GGUFType type) {
      return type == GGUFType::IQ1_M || type == GGUFType::IQ2_XXS ||
             type == GGUFType::IQ3_XXS;
    };
    if (!grouped_gate_up_supported(gate_type) ||
        !grouped_gate_up_supported(up_type))
      return false;
  }

  // Decode has one caller at a time.  Retain capacity across layers/tokens
  // so the fast path does not turn its own dispatch overhead into repeated
  // heap traffic.
  struct Scratch {
    std::vector<const uint8_t *> gate_w, up_w, down_w;
    std::vector<const float *> down_x;
    std::vector<float *> gate, up, down;
    std::vector<float> gate_storage, up_storage, down_storage;
  };
  static thread_local Scratch scratch;
  scratch.gate_w.resize(static_cast<size_t>(topk));
  scratch.up_w.resize(static_cast<size_t>(topk));
  scratch.down_w.resize(static_cast<size_t>(topk));
  scratch.down_x.resize(static_cast<size_t>(topk));
  scratch.gate.resize(static_cast<size_t>(topk));
  scratch.up.resize(static_cast<size_t>(topk));
  scratch.down.resize(static_cast<size_t>(topk));
  scratch.gate_storage.resize(static_cast<size_t>(topk) * moe_inter);
  scratch.up_storage.resize(static_cast<size_t>(topk) * moe_inter);
  scratch.down_storage.resize(static_cast<size_t>(topk) * D);

  const bool stage_timing = qwen_cpu_stage_timing_enabled();
  using StageClock = std::chrono::steady_clock;
  auto add_stage_time = [](std::atomic<double> &counter,
                           StageClock::time_point start) {
    PerfCounters::atomic_add(
        counter, std::chrono::duration<double>(StageClock::now() - start)
                     .count());
  };

  const auto gate_base = static_cast<const uint8_t *>(layer.gate_plane);
  const auto up_base = static_cast<const uint8_t *>(layer.up_plane);
  const auto down_base = static_cast<const uint8_t *>(layer.down_plane);
  for (int k = 0; k < topk; ++k) {
    const int expert = expert_ids[k];
    if (expert < 0 || expert >= n_experts)
      return false;
    const size_t offset = static_cast<size_t>(expert) * layer.expert_stride;
    scratch.gate_w[static_cast<size_t>(k)] = gate_base + offset;
    scratch.up_w[static_cast<size_t>(k)] = up_base + offset;
    scratch.down_w[static_cast<size_t>(k)] = down_base + offset;
    scratch.gate[static_cast<size_t>(k)] =
        scratch.gate_storage.data() + static_cast<size_t>(k) * moe_inter;
    scratch.up[static_cast<size_t>(k)] =
        scratch.up_storage.data() + static_cast<size_t>(k) * moe_inter;
    scratch.down[static_cast<size_t>(k)] =
        scratch.down_storage.data() + static_cast<size_t>(k) * D;
  }

  const bool fused_iq1_iq4 =
      (qwen_cpu_fused_grouped_enabled(engine) ||
       qwen_cpu_fused_iq1_iq4_enabled()) &&
      gate_type == GGUFType::IQ1_M &&
      up_type == GGUFType::IQ1_M;
  const bool fused_iq2_iq4 =
      (qwen_cpu_fused_grouped_enabled(engine) ||
       qwen_cpu_fused_iq2_iq4_enabled()) &&
      gate_type == GGUFType::IQ2_XXS &&
      up_type == GGUFType::IQ2_XXS;
  bool fused_done = false;
  if (fused_iq1_iq4) {
    fused_done = matmul_qwen_iq1_m_iq4_nl_fused(
        scratch.gate.data(), scratch.up.data(), scratch.down.data(), input,
        scratch.gate_w.data(), scratch.up_w.data(), scratch.down_w.data(),
        topk, D, moe_inter, D);
    static std::atomic<bool> logged_fused_iq1_iq4{false};
    if (fused_done &&
        !logged_fused_iq1_iq4.exchange(true, std::memory_order_relaxed))
      LOG_INFO("[CPU-MOE] Qwen fused IQ1_M gate/up + SwiGLU + IQ4_NL down "
               "probe active (one OpenMP team)\n");
  }
  if (!fused_done && fused_iq2_iq4) {
    fused_done = matmul_qwen_iq2_xxs_iq4_nl_fused(
        scratch.gate.data(), scratch.up.data(), scratch.down.data(), input,
        scratch.gate_w.data(), scratch.up_w.data(), scratch.down_w.data(),
        topk, D, moe_inter, D);
    static std::atomic<bool> logged_fused_iq2_iq4{false};
    if (fused_done &&
        !logged_fused_iq2_iq4.exchange(true, std::memory_order_relaxed))
      LOG_INFO("[CPU-MOE] Qwen fused IQ2_XXS gate/up + SwiGLU + IQ4_NL "
               "down probe active (one OpenMP team)\n");
  }
  const bool fused_q4_0 =
      (qwen_cpu_fused_grouped_enabled(engine) ||
       qwen_cpu_fused_q4_0_enabled()) &&
      is_q4_0_combo;
  if (!fused_done && fused_q4_0) {
    fused_done = matmul_qwen_q4_0_q4_0_fused(
        scratch.gate.data(), scratch.up.data(), scratch.down.data(), input,
        scratch.gate_w.data(), scratch.up_w.data(), scratch.down_w.data(),
        topk, D, moe_inter, D);
    static std::atomic<bool> logged_fused_q4_0{false};
    if (fused_done &&
        !logged_fused_q4_0.exchange(true, std::memory_order_relaxed))
      LOG_INFO("[CPU-MOE] Qwen fused Q4_0 gate/up + SwiGLU + Q4_0 down "
               "probe active (one OpenMP team)\n");
  }

  StageClock::time_point stage_start;
  if (!fused_done) {
    if (stage_timing)
      stage_start = StageClock::now();
    if (gate_type == GGUFType::IQ1_M && up_type == GGUFType::IQ1_M &&
        matmul_iq1_m_ggml_q8k_grouped_pair(
            scratch.gate.data(), scratch.up.data(), input,
            scratch.gate_w.data(), scratch.up_w.data(), topk, 1, D,
            moe_inter)) {
      // The common Qwen IQ1_M gate/up layout shares the activation. The pair
      // wrapper quantizes Q8_K once and avoids a second OpenMP team launch.
    } else {
    auto grouped_gate_up = [&](std::vector<float *> &dst,
                               const std::vector<const uint8_t *> &weights,
                               GGUFType type) {
    if (type == GGUFType::IQ1_M)
      return matmul_iq1_m_ggml_q8k_grouped(
          dst.data(), input, weights.data(), topk, 1, D, moe_inter);
    if (type == GGUFType::IQ2_XXS)
      return matmul_iq2_xxs_ggml_q8k_grouped(
          dst.data(), input, weights.data(), topk, 1, D, moe_inter);
    if (type == GGUFType::IQ3_XXS)
      return matmul_iq3_xxs_ggml_q8k_grouped(
          dst.data(), input, weights.data(), topk, 1, D, moe_inter);
      return false;
    };

    if (!grouped_gate_up(scratch.gate, scratch.gate_w, gate_type) ||
        !grouped_gate_up(scratch.up, scratch.up_w, up_type))
      return false;
    }
    if (stage_timing)
      add_stage_time(perf().t_qwen_cpu_grouped_gate_up, stage_start);
  }

  if (!fused_done) {
    if (stage_timing)
      stage_start = StageClock::now();
    for (int k = 0; k < topk; ++k) {
      float *gate = scratch.gate[static_cast<size_t>(k)];
      float *up = scratch.up[static_cast<size_t>(k)];
      if (swiglu_clamp_fused())
        swiglu_clamp_fused_inplace(gate, up, moe_inter, swiglu_clamp);
      else {
        silu_inplace(gate, moe_inter);
        elem_mul(gate, gate, up, moe_inter);
      }
      scratch.down_x[static_cast<size_t>(k)] = gate;
    }
    if (stage_timing)
      add_stage_time(perf().t_qwen_cpu_grouped_swiglu, stage_start);

    if (stage_timing)
      stage_start = StageClock::now();
    if (!matmul_iq4_nl_ggml_q8_0_grouped_inputs(
            scratch.down.data(), scratch.down_x.data(), scratch.down_w.data(),
            topk, 1, moe_inter, D))
      return false;
    if (stage_timing)
      add_stage_time(perf().t_qwen_cpu_grouped_down, stage_start);
  }

  if (stage_timing)
    stage_start = StageClock::now();
  std::fill(output, output + D, 0.0f);
  for (int k = 0; k < topk; ++k) {
    const float weight = router_weights[k];
    const float *down = scratch.down[static_cast<size_t>(k)];
    for (int d = 0; d < D; ++d)
      output[d] += weight * down[d];
  }
  if (stage_timing)
    add_stage_time(perf().t_qwen_cpu_grouped_combine, stage_start);
  return true;
}

void TokenDispatchPipeline::run(MoePipelineContext &context) {
  auto &engine = context.engine;
  auto &ram_mgr = context.ram_mgr;
  auto &async_io_queue = context.async_io_queue;
  const int layer_idx = context.layer_idx;
  const float *hidden_states = context.hidden_states;
  const int S = context.S;
  float *output = context.output;
  auto &scratch = context.scratch;
  const int *token_ids = context.token_ids;
  auto &c = context.c;
  const int D = context.D;
  const int topk = context.topk;
  const int n_experts = context.n_experts;
  const int moe_inter = context.moe_inter;
  const float clamp_exp = context.clamp_exp;
  const float clamp_shexp = context.clamp_shexp;
  auto &gpu_router_handled = context.gpu_router_handled;
  auto &gpu_route_device = context.gpu_route_device;
  auto &gpu_route_ids_device = context.gpu_route_ids_device;
  auto &gpu_route_weights_device = context.gpu_route_weights_device;
  auto &device_router = context.device_router;
  auto &gpu_route_ids_ks = context.gpu_route_ids_ks;
  auto &gpu_route_weights_ks = context.gpu_route_weights_ks;
  auto &batched_handled = context.batched_handled;
  // Option B is deliberately a legacy-dispatch experiment: only the target
  // decode layer is split into the startup-pinned hot set and the DDR/CPU
  // cold set.  Keeping this predicate here leaves every other layer on the
  // normal master dispatcher and prevents the Phase-A graph/staging path from
  // being selected accidentally.
  const bool option_b_target =
      !batched_handled && S == 1 && topk > 0 && n_experts > 0 &&
      engine.qwen4exp_option_b_committed &&
      engine.family_ops.family == ModelFamily::Qwen4Exp &&
      engine.qwen4exp_option_b_layer_enabled(layer_idx);
  const bool cpu_offload_target =
      !batched_handled && S == 1 && topk > 0 && n_experts > 0 &&
      engine.family_ops.family == ModelFamily::Qwen4Exp &&
      engine.qwen4exp_cpu_offload_layer_enabled(layer_idx);
  const bool mixed_cpu_gpu_target = option_b_target || cpu_offload_target;
  bool option_b_shared_started = false;
  const void *option_b_shared_only_output = nullptr;
  if (mixed_cpu_gpu_target && engine.qwen4exp_cpu_gpu_overlap &&
      engine.qwen4exp_device_hc_ffn_input && qwen4exp_hc_device_live() &&
      c.n_expert_shared > 0 && moe_ggml_shared_expert() &&
      engine.resident_layers.count("blk." + std::to_string(layer_idx) +
                                   ".ffn_gate_shexp.weight")) {
    option_b_shared_started = run_qwen4exp_shared_device(
        engine, layer_idx, S, D, moe_inter,
        engine.qwen4exp_device_hc_ffn_input, nullptr, c,
        &option_b_shared_only_output, nullptr, true, false);
    static std::atomic<bool> logged_option_b{false};
    if (option_b_shared_started &&
        !logged_option_b.exchange(true, std::memory_order_relaxed))
      LOG_STREAM_INFO()
          << "[OPTION-B] Layer " << layer_idx
          << " legacy mixed dispatch active: hot experts on GPU, cold experts on DDR/CPU, shared FFN submitted first\n";
  }
  // Qwen CPU-affinity layers can use the same source-backed workflow as
  // Llama.cpp: the complete packed expert layer stays in the already-loaded
  // DDR arena, the CPU GGML graph receives the real global route IDs, and
  // only the [D] input and [D] result cross the CPU/GPU boundary. This branch
  // is intentionally
  // Qwen+decode+CPU-affinity gated; all other families and the existing RAM
  // slab fallback retain their established dispatch.
  if (!batched_handled && S == 1 && topk > 0 && n_experts > 0 &&
      engine.qwen4exp_cpu_ggml_moe &&
      engine.family_ops.family == ModelFamily::Qwen4Exp &&
      engine.family_ops.uses_hyper_connections &&
      is_cpu_affinity_layer(layer_idx) && clamp_exp == 0.0f) {
    QwenRamCpuMoeLayer ram_layer;
    if (qwen4exp_ram_cpu_moe_layer(engine, layer_idx, n_experts, ram_layer)) {
      PERF_START(qwen_cpu_ram);
      // A projection-compact arena has three independent expert planes.  The
      // generic strided MUL_MAT_ID graph requires one non-zero common expert
      // stride and therefore cannot represent this layout safely.  Use the
      // per-expert reference graph for this representation (the same path
      // used by the proven compact A/B replay); keep the strided runner for
      // legacy interleaved arenas.
      const bool use_compact_refs =
          qwen_cpu_compact_graph_enabled() || ram_layer.projection_compact;
      GgmlQwenCpuMoeRunner &cpu_runner = use_compact_refs
                                              ? GgmlQwenCpuMoeRunner::instance()
                                              : GgmlQwenCpuMoeRunner::instance_for_strided(
                                                    ram_layer.meta, n_experts,
                                                    D, moe_inter, topk,
                                                    ram_layer.expert_stride,
                                                    false);
      const bool shared_available =
          c.n_expert_shared > 0 &&
          engine.resident_layers.count("blk." + std::to_string(layer_idx) +
                                       ".ffn_gate_shexp.weight");
      // The old overlap experiment ran the shared expert on the host while a
      // worker computed the routed result.  That still paid three GPU dense
      // launches, a host SiLU/mul, and a host accumulation for every CPU
      // layer.  The useful split is narrower: keep the routed projections on
      // the CPU, move only the top-k weighting/reduction to CUDA, and let the
      // existing device graph do shared FFN + routed/shared addition.  This
      // is Qwen-only and opt-in so other model families retain their
      // established CPU path.
      int cpu_affinity_count = 0;
      for (bool selected : cpu_affinity_layer_flags())
        cpu_affinity_count += selected ? 1 : 0;
      // The asynchronous overlap worker currently submits the persistent
      // strided GGML graph.  When the Qwen-only direct DDR-strided grouped
      // kernel is enabled, selecting that worker would silently replace the
      // measured fast path with the much slower graph path (and lose the
      // direct IQ1_M/IQ4_NL dispatch).  Keep the direct path authoritative;
      // the ordinary device handoff below still keeps the shared expert on
      // the GPU with one pinned H2D and no host shared projections, without
      // regressing this model-specific optimization.
      const bool qwen_direct_grouped_cpu =
          !qwen_cpu_compact_graph_enabled() &&
          qwen_cpu_strided_grouped_enabled(engine);
      const bool try_device_shared =
          engine.qwen4exp_cpu_gpu_overlap && shared_available &&
          qwen4exp_hc_device_live() && engine.qwen4exp_device_hc_ffn_input &&
          moe_ggml_shared_expert() &&
          (!qwen_direct_grouped_cpu || qwen_cpu_direct_gpu_overlap_probe()) &&
          (engine.qwen4exp_cpu_gpu_overlap_max_layers <= 0 ||
           cpu_affinity_count <= engine.qwen4exp_cpu_gpu_overlap_max_layers);
      static std::atomic<bool> logged_direct_overlap_guard{false};
      if (engine.qwen4exp_cpu_gpu_overlap && qwen_direct_grouped_cpu &&
          !qwen_cpu_direct_gpu_overlap_probe() &&
          !logged_direct_overlap_guard.exchange(true,
                                                 std::memory_order_relaxed)) {
        LOG_INFO("[CPU-MOE] Qwen CPU/GPU overlap worker bypassed: direct "
                 "DDR-strided grouped kernel is active; retaining the "
                 "faster device-handoff path\n");
      }
      const bool try_device_handoff =
          !try_device_shared && engine.qwen4exp_cpu_shared_device &&
          shared_available && qwen4exp_hc_device_live() &&
          engine.qwen4exp_device_hc_ffn_input && moe_ggml_shared_expert();
      static std::atomic<bool> logged_overlap_limit{false};
      if (engine.qwen4exp_cpu_gpu_overlap && !try_device_shared &&
          shared_available && cpu_affinity_count > 0 &&
          engine.qwen4exp_cpu_gpu_overlap_max_layers > 0 &&
          cpu_affinity_count > engine.qwen4exp_cpu_gpu_overlap_max_layers &&
          !logged_overlap_limit.exchange(true, std::memory_order_relaxed)) {
        LOG_STREAM_INFO()
            << "[CPU-MOE] Qwen CPU/GPU overlap disabled for this run: "
               "cpu-affinity set has "
            << cpu_affinity_count << " layers, limit is "
            << engine.qwen4exp_cpu_gpu_overlap_max_layers
            << " (the overlap path submits a shared-only graph plus a "
               "host-routed-add graph per CPU layer)\n";
      }
      bool qwen_cpu_direct_grouped = false;

      // The compact A/B keeps the source in the preloaded DDR slabs but gives
      // GGML the same per-projection contiguous expert layout that llama.cpp uses.
      // It is intentionally constructed from the exact routed IDs, so this
      // does not copy the other 502 experts in a Qwen layer.
      auto run_qwen_cpu_graph = [&](float *result) {
        if (!use_compact_refs) {
          if (qwen_cpu_strided_grouped_enabled(engine) &&
              qwen4exp_cpu_strided_grouped_quantized(
                  engine, scratch.normed.data(), result, ram_layer,
                  scratch.tok_topk_ids.data(),
                  scratch.tok_topk_weights.data(), n_experts, topk, D,
                  moe_inter, clamp_exp)) {
            static std::atomic<bool> logged_strided_grouped{false};
            if (!logged_strided_grouped.exchange(true,
                                                 std::memory_order_relaxed))
              LOG_INFO("[CPU-MOE] Qwen direct strided grouped quantized path "
                       "active (DDR arena, IQ1_M/Q8_K + IQ4_NL/Q8_0)\n");
            qwen_cpu_direct_grouped = true;
            return true;
          }
          return cpu_runner.run_strided(
              scratch.normed.data(), result, ram_layer.meta,
              ram_layer.gate_plane, ram_layer.up_plane, ram_layer.down_plane,
              scratch.tok_topk_ids.data(), n_experts,
              scratch.tok_topk_weights.data(), D, moe_inter, topk,
              ram_layer.expert_stride);
        }

        std::vector<GgmlQwenCpuExpertRef> refs((size_t)topk);
        const auto &slots = engine.layers_experts[(size_t)layer_idx];
        for (int k = 0; k < topk; ++k) {
          const int expert_id = scratch.tok_topk_ids[(size_t)k];
          if (expert_id < 0 || expert_id >= n_experts ||
              (size_t)expert_id >= slots.size()) {
            return false;
          }
          const MoEExpertBlock &expert = slots[(size_t)expert_id];
          refs[(size_t)k] = {
              expert.gate_proj.host_ptr,
              expert.up_proj.host_ptr,
              expert.down_proj.host_ptr,
              expert.gate_proj.meta.size_bytes,
              expert.up_proj.meta.size_bytes,
              expert.down_proj.meta.size_bytes,
              static_cast<uint32_t>(expert.gate_proj.meta.type),
              static_cast<uint32_t>(expert.up_proj.meta.type),
              static_cast<uint32_t>(expert.down_proj.meta.type)};
        }
        return cpu_runner.run(
            scratch.normed.data(), result, refs.data(), topk,
            scratch.tok_topk_weights.data(), D, moe_inter, topk);
      };

      // Keep the actual DDR-backed routed expert work separate from the
      // device shared-FFN enqueue and HC event handoff.  The ordinary
      // t_moe_expert bucket intentionally includes both, which is useful for
      // request throughput but cannot explain the gap to llama.cpp's CPU split.
      auto run_qwen_cpu_graph_timed = [&](float *result) {
        const auto start = std::chrono::high_resolution_clock::now();
        // Keep the first non-finite value visible at the CPU/GPU MoE seam.
        // The downstream HC/device graph can otherwise turn one bad routed
        // value into a much less useful all-layer failure report.
        check_finite("Qwen CPU route input", layer_idx, -1,
                     scratch.normed.data(), static_cast<size_t>(D));
        const bool cpu_diag = [] {
          const char *env =
              std::getenv("MOESTREAM_QWEN4EXP_DIAG_LAYER0_IQ2");
          return env && env[0] == '1';
        }();
        if (cpu_diag)
          qwen_cpu_moe_diag_set_layer(layer_idx);
        const bool ok = run_qwen_cpu_graph(result);
        if (cpu_diag)
          qwen_cpu_moe_diag_set_layer(-1);
        if (ok)
          check_finite("Qwen CPU route output", layer_idx, -1, result,
                       static_cast<size_t>(D));
        if (S == 1)
          moe_pipeline_internal::add_decode_qwen_cpu_graph_ms(
              std::chrono::duration<double, std::milli>(
                  std::chrono::high_resolution_clock::now() - start)
                  .count());
        return ok;
      };

      auto run_qwen_cpu_down_graph = [&](float *result) {
        GgmlQwenCpuMoeRunner &down_runner =
            GgmlQwenCpuMoeRunner::instance_for_strided(
                ram_layer.meta, n_experts, D, moe_inter, topk,
                ram_layer.expert_stride, true);
        return down_runner.run_strided_down(
            scratch.normed.data(), result, ram_layer.meta,
            ram_layer.gate_plane, ram_layer.up_plane, ram_layer.down_plane,
            scratch.tok_topk_ids.data(), n_experts, D, moe_inter, topk,
            ram_layer.expert_stride);
      };

      bool handled = false;
      bool device_shared_handled = false;
      if (try_device_handoff &&
          scratch.qwen_cpu_moe_out.resize((size_t)D, "qwen_cpu_moe_out")) {
        // This is the simple, non-overlapped device handoff.  The CPU still
        // owns all routed projections and finishes the weighted [D] result,
        // but the always-on shared expert consumes that result through the
        // device graph.  Unlike qwen4exp_cpu_gpu_overlap, this submits one
        // shared graph after the CPU result is ready; it does not submit a
        // shared-only graph plus a second host-routed add graph.
        handled = run_qwen_cpu_graph_timed(scratch.qwen_cpu_moe_out.data());
        if (handled) {
          const void *combined = nullptr;
          device_shared_handled = run_qwen4exp_shared_device(
              engine, layer_idx, S, D, moe_inter,
              engine.qwen4exp_device_hc_ffn_input, nullptr, c, &combined,
              scratch.qwen_cpu_moe_out.data());
          if (device_shared_handled && combined) {
            scratch.moe_device_output = combined;
            scratch.moe_shared_device_output = true;
            scratch.moe_device_output_ready = true;
            static std::atomic<bool> logged_qwen_cpu_device_handoff{false};
            if (!logged_qwen_cpu_device_handoff.exchange(
                    true, std::memory_order_relaxed))
              LOG_INFO("[CPU-MOE] Qwen CPU routed graph handed directly to "
                       "the device shared FFN (one host->device routed "
                       "handoff, no shared readback)\n");
          } else {
            // Preserve the established host-facing path if the optional
            // device graph cannot allocate or execute on a particular
            // layer/device.  The CPU result is already valid.
            std::memcpy(output, scratch.qwen_cpu_moe_out.data(),
                        (size_t)D * sizeof(float));
            run_qwen4exp_shared_host(engine, layer_idx, S, D, moe_inter,
                                     scratch.normed.data(), output, scratch,
                                     c);
          }
        }
      }

      if (!handled && try_device_shared) {
        // Start the shared GPU graph before running the DDR-backed routed
        // graph.  The CUDA backend is asynchronous, so the existing decode
        // thread can submit the shared-only graph and then execute the
        // proven persistent CPU graph while CUDA is busy.  Avoid a second
        // CPU worker here: the previous worker experiment made GGML's CPU
        // graph contend with its own threadpool and was substantially slower
        // on the real Qwen IQ1_M/IQ2_XXS checkpoint.
        if (!qwen_cpu_compact_graph_enabled() &&
            scratch.qwen_cpu_moe_out.resize((size_t)D,
                                            "qwen_cpu_moe_out")) {
          const void *shared_only = nullptr;
          const bool shared_started = run_qwen4exp_shared_device(
              engine, layer_idx, S, D, moe_inter,
              engine.qwen4exp_device_hc_ffn_input, nullptr, c, &shared_only,
              nullptr, true, false);
          const bool cpu_finished = run_qwen_cpu_graph_timed(
              scratch.qwen_cpu_moe_out.data());
          if (shared_started && cpu_finished) {
              const void *combined = nullptr;
              device_shared_handled = run_qwen4exp_shared_device(
                  engine, layer_idx, S, D, moe_inter, nullptr, nullptr, c,
                  &combined, scratch.qwen_cpu_moe_out.data(), false, true);
              if (device_shared_handled && combined) {
                handled = true;
                scratch.moe_device_output = combined;
                scratch.moe_shared_device_output = true;
                scratch.moe_device_output_ready = true;
                static std::atomic<bool> logged_qwen_cpu_gpu_overlap{false};
                if (!logged_qwen_cpu_gpu_overlap.exchange(
                        true, std::memory_order_relaxed))
                  LOG_INFO(
                      "[CPU-MOE] Qwen CPU routed graph overlapped with "
                      "shared GPU FFN on the decode thread; final host-route "
                      "add queued on device\n");
              }
          }
          // If the shared-only/add graph is unavailable, the CPU result is
          // still valid. Use the established host-routed shared graph so a
          // failed optional overlap allocation cannot change numerics.
          if (!handled && cpu_finished) {
            const void *combined = nullptr;
            device_shared_handled = run_qwen4exp_shared_device(
                engine, layer_idx, S, D, moe_inter,
                engine.qwen4exp_device_hc_ffn_input, nullptr, c, &combined,
                scratch.qwen_cpu_moe_out.data());
            if (device_shared_handled && combined) {
              handled = true;
              scratch.moe_device_output = combined;
              scratch.moe_shared_device_output = true;
              scratch.moe_device_output_ready = true;
            } else {
              std::memcpy(output, scratch.qwen_cpu_moe_out.data(),
                          (size_t)D * sizeof(float));
              run_qwen4exp_shared_host(engine, layer_idx, S, D, moe_inter,
                                       scratch.normed.data(), output, scratch,
                                       c);
              handled = true;
            }
          }
        }

        if (!handled && qwen_cpu_gpu_combine_enabled() &&
            scratch.qwen_cpu_moe_out.resize(
                (size_t)D * (size_t)topk, "qwen_cpu_moe_down_out")) {
          handled = run_qwen_cpu_down_graph(scratch.qwen_cpu_moe_out.data());
          if (handled) {
            const int combine_device =
                moe_ggml_device() >= 0 ? moe_ggml_device() : 0;
            const void *device_routed = nullptr;
            auto &combine = GgmlQwenDeviceCombineRunner::instance();
            if (combine.ensure(combine_device, D, topk) &&
                combine.run(scratch.qwen_cpu_moe_out.data(),
                            scratch.tok_topk_weights.data(),
                            &device_routed) &&
                device_routed) {
              const void *combined = nullptr;
              device_shared_handled = run_qwen4exp_shared_device(
                  engine, layer_idx, S, D, moe_inter,
                  engine.qwen4exp_device_hc_ffn_input, device_routed, c,
                  &combined, nullptr);
              if (device_shared_handled && combined) {
                scratch.moe_device_output = combined;
                scratch.moe_shared_device_output = true;
                scratch.moe_device_output_ready = true;
              }
            }
          }
        }

        if (!device_shared_handled) {
          // Safe numerical fallback: retain the established CPU weighted
          // result and host/device shared-expert behavior if the new
          // post-down GPU combine cannot be built or accepted.
          if (scratch.qwen_cpu_moe_out.resize((size_t)D,
                                               "qwen_cpu_moe_out")) {
            handled = run_qwen_cpu_graph_timed(scratch.qwen_cpu_moe_out.data());
            if (handled) {
              const void *combined = nullptr;
              device_shared_handled = run_qwen4exp_shared_device(
                  engine, layer_idx, S, D, moe_inter,
                  engine.qwen4exp_device_hc_ffn_input, nullptr, c, &combined,
                  scratch.qwen_cpu_moe_out.data());
              if (device_shared_handled && combined) {
                scratch.moe_device_output = combined;
                scratch.moe_shared_device_output = true;
                scratch.moe_device_output_ready = true;
              } else {
                std::memcpy(output, scratch.qwen_cpu_moe_out.data(),
                            (size_t)D * sizeof(float));
                run_qwen4exp_shared_host(engine, layer_idx, S, D, moe_inter,
                                         scratch.normed.data(), output, scratch,
                                         c);
              }
            }
          }
        }
      }
      if (!handled && !try_device_shared) {
        handled = run_qwen_cpu_graph_timed(output);
        if (handled && shared_available)
          run_qwen4exp_shared_host(engine, layer_idx, S, D, moe_inter,
                                   scratch.normed.data(), output, scratch, c);
      }
      PERF_END(qwen_cpu_ram, t_cpu_expert);
      if (handled) {
        perf().n_expert_cpu_calls.fetch_add((uint64_t)topk,
                                            std::memory_order_relaxed);
        batched_handled = true;
        static std::atomic<bool> logged_qwen_ram_cpu{false};
        if (!logged_qwen_ram_cpu.exchange(true, std::memory_order_relaxed))
          LOG_STREAM_INFO()
              << "[CPU-MOE] Qwen DDR-RAM-strided CPU graph active "
                 "(source=preloaded-DDR-arena, file-mmap=not-used, full expert "
                 "layer resident, layout="
              << (qwen_cpu_compact_graph_enabled() ? "compact-routed" :
                  (qwen_cpu_direct_grouped ? "strided-grouped-quantized"
                                           : "strided-slab"))
              << ", post-down-combine="
              << (device_shared_handled ? "GPU" : "CPU")
              << ", global route IDs, threads="
              << (qwen_cpu_direct_grouped ? qwen_cpu_grouped_threads()
                                          : cpu_runner.thread_count())
              << " (OMP max=" << omp_get_max_threads()
              << "), shared expert remains GPU)\n";
        if (device_shared_handled) {
          static std::atomic<bool> logged_qwen_cpu_device_handoff{false};
          if (!logged_qwen_cpu_device_handoff.exchange(
                  true, std::memory_order_relaxed))
            LOG_STREAM_INFO()
                << "[CPU-MOE] Qwen CPU routed result handed to device "
                   "shared FFN (one pinned H2D, no host shared projections)\n";
        }
      }
    }
  }

  // --- Phase 3: execute each token's already-selected experts ---
  for (int s = 0; !batched_handled && s < S; ++s) {
    // Diagnostic upper bound on this whole per-token dispatch loop --
    // see t_moe_phase3_dispatch's own comment (perf_timer.hpp) for why
    // this is DELIBERATELY not a clean partition against gpu=/cpu=/
    // ram_scan= (all nested inside this same loop body); its own real
    // per-loop-iteration cost accumulates correctly across repeated
    // scope entry/exit (ScopedTimer's destructor ADDS elapsed, doesn't
    // overwrite), matching the S>1 prefill case too.
    PERF_SCOPE(t_moe_phase3_dispatch);
    // FIX: both routed and shared expert FFNs must consume the SAME
    // ffn_norm'd input the router scored (matching the reference: llama.cpp
    // builds one normed `cur` tensor and feeds it to both the router and
    // build_moe_ffn's per-expert matmuls). This previously fed the raw,
    // un-normalized hidden_states into every expert FFN — router weights
    // were correct but the FFN itself was operating on the wrong input.
    const float *token_hidden = scratch.normed.data() + s * D;
    float *token_output = output + s * D;
    // Cache placement must not change DeepSeek's reduction tree. Summing
    // separately on each GPU changes the F32 result whenever an expert moves
    // between devices, and tiny differences can flip a nearly tied router.
    // Keep per-route activation rows, download once per GPU, and apply the
    // native weighted FMA in k order after every expert has completed.
    const bool ordered_routes =
        engine.family_ops.family == ModelFamily::DeepSeek_V41;
    bool collecting_ordered = ordered_routes;
    static thread_local std::vector<float> ordered_route_outputs;
    if (ordered_routes)
      ordered_route_outputs.assign((size_t)topk * D, 0.0f);

    // Weighted accumulation + VERBOSE trace, shared by both the CPU path
    // (dispatch_expert_k's own synchronous fallback) and the GPU path
    // (Pass B, accumulates after finish()) -- same logic either way, just
    // called at a different point in the pipeline depending on path.
    // Moved up (was previously defined between the loading block and the
    // old Pass A loop) so dispatch_expert_k below can capture it.
    auto accumulate = [&](int k, MoEExpertBlock *exp,
                          const float *down_buf_ptr) {
      PERF_SCOPE(t_accumulate_total);
      // Per-expert output norm, VERBOSE-gated: distinguishes "expert_ffn
      // itself computed all-zero" from "it computed something real but
      // the weighted accumulation below never applied it" -- check_finite
      // only catches NaN/Inf, never a suspiciously-exact zero, which is
      // exactly the symptom under investigation (moe_out|=0 every MoE
      // layer during prefill)
      if (trace_gate("TRACE-EXPERT-NORM")) {
        double ss = 0.0;
        for (int d = 0; d < D; ++d)
          ss += (double)down_buf_ptr[d] * down_buf_ptr[d];
        LOG_STREAM_VERBOSE()
            << "[TRACE l=" << layer_idx << "] expert k=" << k
            << " id=" << scratch.tok_topk_ids[s * topk + k]
            << " requested_id=" << scratch.tok_topk_ids[s * topk + k]
            << " |down_buf|=" << std::sqrt(ss) << "\n";
      }

      if (collecting_ordered) {
        std::memcpy(ordered_route_outputs.data() + (size_t)k * D,
                    down_buf_ptr, (size_t)D * sizeof(float));
        return;
      }
      // Weighted accumulation: output += weight * expert_output
      float w = scratch.tok_topk_weights[s * topk + k];
      if (std::isnan(w) || std::isinf(w)) {
        LOG_ERROR("[EXPERT-GPU] router weight itself is non-finite: layer=%d "
                  "token=%d k=%d "
                  "expert=%d w=%f -- expert_ffn's own output can be perfectly "
                  "clean and this "
                  "still poisons token_output; check "
                  "router_logits/normalization above.\n",
                  layer_idx, s, k, scratch.tok_topk_ids[s * topk + k], w);
      }
      __m512 v_w = _mm512_set1_ps(w);
      int i = 0;
      for (; i + 16 <= D; i += 16) {
        __m512 v_out = _mm512_loadu_ps(token_output + i);
        __m512 v_exp = _mm512_loadu_ps(down_buf_ptr + i);
        _mm512_storeu_ps(token_output + i, _mm512_fmadd_ps(v_w, v_exp, v_out));
      }
      for (; i < D; ++i)
        token_output[i] += w * down_buf_ptr[i];
    };

    // Issues expert k's GPU kernels (async, no sync) against whatever
    // scratch.expert_ptrs[k] already holds, falling back to synchronous
    // CPU compute if GPU dispatch isn't possible. Pulled out of the old
    // single "Pass A" loop (which used to run only AFTER get_experts_
    // batch) so it can also be called INLINE for shadowed (already
    // VRAM-resident) experts below, BEFORE get_experts_batch's blocking
    // disk read for this layer's genuine misses -- expert_ffn_gpu_issue
    // never syncs, so those kernels execute on the GPU stream WHILE this
    // thread blocks on io_uring for the rest of the batch, instead of
    // the two running strictly back-to-back as before (real compute/IO
    // overlap, not just async-within-GPU overlap).
    // EngineConfig::cpu_offload_max_experts: per-k flag for THIS token/
    // layer iteration only, read by dispatch_expert_k below (via gpu_
    // ctx.force_cpu_offload) and populated right after get_experts_
    // batch resolves this layer/token's genuine misses (further down).
    // Reset to all-false HERE, fresh every token/layer iteration, since
    // a shadowed (already VRAM-resident) k dispatches inline through
    // this same lambda BEFORE that later assignment ever runs -- and a
    // shadowed k must never be forced to CPU, it was never a disk-cold
    // miss to begin with.
    static thread_local std::vector<bool> force_cpu_offload_k;
    if ((int)force_cpu_offload_k.size() < topk)
      force_cpu_offload_k.resize(topk);
    std::fill(force_cpu_offload_k.begin(), force_cpu_offload_k.begin() + topk,
              false);

    // CPUOffload Milestone A: classify the selected route slots before any
    // GPU dispatch.  The input is already in the pinned `normed` buffer, and
    // the complete DDR arena is immutable for the lifetime of this request,
    // so the persistent CPU worker can start while the hot GPU work is
    // submitted below.  If worker setup fails, leave every slot on the
    // established dispatcher so this experiment cannot drop a contribution.
    GgmlQwenCpuMoeRunner *cpu_offload_runner = nullptr;
    QwenRamCpuMoeLayer cpu_offload_ram_layer;
    QwenCpuOffloadRing *cpu_offload_ring = nullptr;
    int cpu_offload_ring_slot = -1;
    bool cpu_offload_event_fence_active = false;
    bool cpu_offload_worker_started = false;
    int cpu_offload_cold_count = 0;
    auto cpu_offload_start = std::chrono::steady_clock::now();
    if (cpu_offload_target) {
      const int forced_count = std::min(
          std::max(0, engine.qwen4exp_cpu_offload_force_cold_count), topk);
      for (int k = 0; k < topk; ++k) {
        const int expert_id = scratch.tok_topk_ids[s * topk + k];
        bool cold = false;
        // If Option B has already built a usage-ranked mask for this layer,
        // reuse it as the hot/cold policy.  The dedicated CPUOffload switch
        // remains independent: with no mask, only the deterministic forced
        // slots below are cold.
        if (engine.qwen4exp_option_b_layer_enabled(layer_idx) &&
            !engine.qwen4exp_option_b_hot_expert(layer_idx, expert_id))
          cold = true;
        if (k < forced_count)
          cold = true;
        if (engine.qwen4exp_cpu_offload_force_expert >= 0 &&
            expert_id == engine.qwen4exp_cpu_offload_force_expert)
          cold = true;
        scratch.qwen_cpu_offload_cold_slots[(size_t)k] = cold ? 1 : 0;
        if (cold) {
          scratch.qwen_cpu_offload_cold_ids[(size_t)cpu_offload_cold_count] =
              expert_id;
          scratch.qwen_cpu_offload_cold_weights[
              (size_t)cpu_offload_cold_count] =
              scratch.tok_topk_weights[s * topk + k];
          ++cpu_offload_cold_count;
        }
      }

      float *cpu_offload_output = nullptr;
      if (cpu_offload_cold_count > 0 &&
          qwen4exp_ram_cpu_moe_layer(engine, layer_idx, n_experts,
                                     cpu_offload_ram_layer)) {
        // Milestone B uses a dedicated pinned host/device result slot.  The
        // shared graph has already been submitted above, so its backend
        // stream is the exact compute stream that must consume the slot.
        if (engine.qwen4exp_cpu_offload_event_fence &&
            option_b_shared_started) {
          auto &shared_runner =
              GgmlQwenSharedRunner::instance_for_layer(layer_idx);
          const int ring_device =
              moe_ggml_device() >= 0 ? moe_ggml_device() : 0;
          auto &ring = QwenCpuOffloadRing::instance();
          if (ring.ensure(ring_device, D,
                          engine.qwen4exp_cpu_offload_ring_slots)) {
            double reuse_wait_ms = 0.0;
            const double h2d_before = ring.h2d_total_ms();
            if (ring.acquire(cpu_offload_ring_slot, reuse_wait_ms) &&
                ring.prepare_slot(cpu_offload_ring_slot,
                                  shared_runner.backend_stream())) {
              cpu_offload_ring = &ring;
              cpu_offload_event_fence_active = true;
              cpu_offload_output = ring.host_output(cpu_offload_ring_slot);
              if (reuse_wait_ms > 0.0)
                moe_pipeline_internal::add_decode_qwen_cpu_offload_ring_reuse_wait_ms(
                    reuse_wait_ms);
              const double h2d_after = ring.h2d_total_ms();
              if (h2d_after > h2d_before)
                moe_pipeline_internal::add_decode_qwen_cpu_offload_h2d_ms(
                    h2d_after - h2d_before);
            }
          }
        }
        if (!cpu_offload_output &&
            scratch.qwen_cpu_moe_out.resize((size_t)D,
                                            "qwen_cpu_offload_result"))
          cpu_offload_output = scratch.qwen_cpu_moe_out.data();
      }

      if (cpu_offload_cold_count > 0 && cpu_offload_output) {
        cpu_offload_start = std::chrono::steady_clock::now();
        cpu_offload_runner =
            &GgmlQwenCpuMoeRunner::instance_for_strided(
                cpu_offload_ram_layer.meta, n_experts, D, moe_inter,
                cpu_offload_cold_count,
                cpu_offload_ram_layer.expert_stride, false);
        if (cpu_offload_event_fence_active)
          cpu_offload_worker_started = cpu_offload_runner->start_strided_async(
            scratch.normed.data() + s * D, cpu_offload_output,
            cpu_offload_ram_layer.meta, cpu_offload_ram_layer.gate_plane,
            cpu_offload_ram_layer.up_plane, cpu_offload_ram_layer.down_plane,
            scratch.qwen_cpu_offload_cold_ids.data(), n_experts,
            scratch.qwen_cpu_offload_cold_weights.data(), D, moe_inter,
            cpu_offload_cold_count, cpu_offload_ram_layer.expert_stride,
            QwenCpuOffloadRing::completion_callback,
            cpu_offload_ring->completion_user(cpu_offload_ring_slot));
        else
          cpu_offload_worker_started = cpu_offload_runner->start_strided_async(
              scratch.normed.data() + s * D, cpu_offload_output,
              cpu_offload_ram_layer.meta, cpu_offload_ram_layer.gate_plane,
              cpu_offload_ram_layer.up_plane,
              cpu_offload_ram_layer.down_plane,
              scratch.qwen_cpu_offload_cold_ids.data(), n_experts,
              scratch.qwen_cpu_offload_cold_weights.data(), D, moe_inter,
              cpu_offload_cold_count, cpu_offload_ram_layer.expert_stride);
        if (cpu_offload_worker_started)
          moe_pipeline_internal::add_decode_qwen_cpu_offload_worker_call();
        else if (cpu_offload_event_fence_active)
          cpu_offload_ring->abandon(cpu_offload_ring_slot);
      }

      if (cpu_offload_cold_count > 0)
        moe_pipeline_internal::add_decode_qwen_cpu_offload_cold_slots(
            static_cast<uint64_t>(cpu_offload_cold_count));

      if (!cpu_offload_worker_started) {
        std::fill(scratch.qwen_cpu_offload_cold_slots.begin(),
                  scratch.qwen_cpu_offload_cold_slots.begin() + topk, 0);
        cpu_offload_cold_count = 0;
        cpu_offload_runner = nullptr;
        cpu_offload_ring = nullptr;
        cpu_offload_ring_slot = -1;
        cpu_offload_event_fence_active = false;
        static std::atomic<bool> logged_cpu_offload_fallback{false};
        if (!logged_cpu_offload_fallback.exchange(true,
                                                  std::memory_order_relaxed))
          LOG_STREAM_WARN()
              << "[CPU-OFFLOAD] Layer " << layer_idx
              << " Milestone A worker setup failed; retaining legacy dispatch\n";
      }
    }

    // Decode normally routes one token through each selected expert.  Reuse
    // the same device-side scatter accumulator as the grouped prefill path so
    // result is downloaded once per layer/GPU instead of once per expert.
    // DeepSeek retains separate k rows until its fixed-order host reduction.
    // The route table is tiny at S==1, but keep it in a
    // thread-local grow-only vector so this does not allocate on every token.
    const bool decode_gpu_accum =
        S == 1 && topk > 0 && moe_gpu_accum() && ram_mgr.vram_cache;
    static thread_local std::vector<int> decode_accum_tok;
    static thread_local std::vector<float> decode_accum_w;
    bool decode_accum_begun[PerfCounters::PERF_MAX_GPU_SLOTS] = {};
    if (decode_gpu_accum) {
      decode_accum_tok.resize((size_t)topk);
      decode_accum_w.resize((size_t)topk);
      for (int k = 0; k < topk; ++k) {
        decode_accum_tok[(size_t)k] = ordered_routes ? k : 0;
        decode_accum_w[(size_t)k] =
            ordered_routes ? 1.0f : scratch.tok_topk_weights[s * topk + k];
      }
    }

    auto dispatch_expert_k = [&](int k) {
      if (cpu_offload_worker_started && k >= 0 && k < topk &&
          scratch.qwen_cpu_offload_cold_slots[(size_t)k])
        return;
      // 2026-08-29, "Fifth round" of the phase3_loop-residual
      // investigation (see PerfCounters::t_shadow_precheck_total's
      // own comment) -- an upper bound over this WHOLE lambda body,
      // same "measure the whole thing, subtract known nested pieces"
      // discipline as t_moe_phase3_dispatch itself. Nests
      // t_resolve_gpu_loc_total/t_gpu_issue_total/t_gpu_finish_total
      // (already separately timed) plus this lambda's own genuinely
      // untimed bookkeeping (the gpu_ctx.loc fallback find_expert()
      // call below, ram_pin(), error-path branching).
      PERF_SCOPE(t_dispatch_expert_k_total);
      MoEExpertBlock *exp = scratch.expert_ptrs[k];
      // A RAM block can be MOVE-promoted into the main VRAM cache while it
      // is loaded or resolved, which clears exp->expert_id while leaving the
      // block's projection pointers valid. Always resolve/hold by the
      // immutable routed ID for this token, never by that mutable slot field.
      const int routed_expert_id = scratch.tok_topk_ids[s * topk + k];
      // A shadow block (see the loading loop below) legitimately has
      // null host_ptr by design -- it's metadata-only, GPU dispatch
      // reads weight bytes from ExpertLocation::device_ptr, never
      // from here. Only a REAL (non-shadow) block missing host_ptr
      // is the genuine "something went wrong" case this check exists
      // to catch.
      if (!exp || (!exp->vram_shadow &&
                   (!exp->gate_proj.host_ptr || !exp->down_proj.host_ptr))) {
        // Previously a completely silent skip -- if this fires for
        // EVERY k this would exactly explain moe_out staying at
        // exactly 0.0 for a whole layer (not just NaN-poisoned,
        // which check_finite already covers) with zero prior
        // diagnostic pointing at it.
        LOG_ERROR(
            "[EXPERT-GPU] layer=%d token=%d k=%d expert_id=%d: expert_ptrs[k] "
            "is "
            "%s -- skipping this expert's contribution entirely (exp=%p).\n",
            layer_idx, s, k, scratch.tok_topk_ids[s * topk + k],
            !exp ? "NULL"
                 : (!exp->gate_proj.host_ptr ? "missing gate_proj.host_ptr"
                                             : "missing down_proj.host_ptr"),
            (void *)exp);
        return;
      }

      // EngineConfig::moe_decode_batched (default OFF, PROTOTYPE) --
      // decode-time (S==1) fast path. Skips the VRAM-cache resolve/lock/
      // GPU-pool machinery below entirely and goes straight to the same
      // CPU dequant-matmul call this lambda's own GPU-dispatch-failure
      // branch already uses further down (ram_pin -> expert_ffn_cpu ->
      // ram_unpin -> accumulate) -- see dense_dispatch.hpp's
      // moe_decode_batched() for the full writeup, including the real
      // trade-off (this expert can no longer be newly promoted into VRAM
      // from a decode-time hit while this path is taken for it). Never
      // engages for a vram_shadow block -- it has no host bytes to read
      // without the synchronous reload the code below already handles;
      // falls through unchanged for that case.
      if (S == 1 && !exp->vram_shadow && moe_decode_batched()) {
        ram_mgr.ram_pin(*exp);
        expert_ffn_cpu(
            token_hidden, scratch.down_buf.data(), exp->gate_proj.host_ptr,
            exp->gate_proj.meta, exp->up_proj.host_ptr, exp->up_proj.meta,
            exp->down_proj.host_ptr, exp->down_proj.meta,
            scratch.gate_buf.data(), scratch.up_buf.data(), 1, D, moe_inter,
            clamp_exp, layer_idx, routed_expert_id);
        ram_mgr.ram_unpin(*exp);
        accumulate(k, exp, scratch.down_buf.data());
        return;
      }

      ExpertGpuContext gpu_ctx;
      gpu_ctx.topo = ram_mgr.vram_cache;
      gpu_ctx.layer_idx = layer_idx;
      gpu_ctx.expert_id = routed_expert_id;
      gpu_ctx.slab_buffer = exp->slab_buffer;
      // Real total, not slab_capacity -- same bug/reasoning as
      // decode_async_prewarm's own fix above (see its comment for the
      // full incident): slab_capacity is this RAM slot's grow-only
      // allocation, which can legitimately outlive whatever smaller
      // expert currently occupies it after an LRU reuse. This is the
      // field every synchronous cold-miss GPU upload below reads its
      // byte count from, so getting it right here covers all of them.
      gpu_ctx.slab_capacity = exp->gate_proj.meta.size_bytes +
                              exp->up_proj.meta.size_bytes +
                              exp->down_proj.meta.size_bytes;
      // Null for a shadow block (exp->vram_shadow) -- correct: a
      // shadow's slab_buffer is already null too, so none of the
      // synchronous-upload branches below (which all require
      // gpu_ctx->slab_buffer) can ever fire for it anyway; also_in_vram
      // marking is only relevant for a REAL RAM-tier block being
      // promoted, never a metadata-only shadow.
      gpu_ctx.exp = exp;
      gpu_ctx.force_cpu_offload =
          ((k >= 0 && k < (int)force_cpu_offload_k.size()) &&
           force_cpu_offload_k[k]) ||
          (option_b_target &&
           !engine.qwen4exp_option_b_hot_expert(
               layer_idx, scratch.tok_topk_ids[s * topk + k]));
      // Reuse the shadow pre-check's find_expert() result instead of
      // calling it a second time -- a real, measured cost (Cache
      // mutex wait roughly doubled, 4.185s vs 1.4-2.0s, from this
      // exact redundant lookup on a real run). Safe: hold_for_compute
      // (below, inside expert_ffn_resolve_gpu_loc) does its own fresh
      // identity-based re-scan regardless of which pointer is passed
      // in, so a stale hint here just makes the hold correctly fail,
      // exactly as a stale result from a second find_expert() call
      // would. nullptr (not shadowed, or shadow lookup found nothing)
      // falls back to the original single find_expert() call.
      gpu_ctx.loc = (exp->vram_shadow && scratch.vram_loc_hint[k])
                        ? scratch.vram_loc_hint[k]
                        : (ram_mgr.vram_cache ? ram_mgr.vram_cache->find_expert(
                                                    layer_idx,
                                                    routed_expert_id)
                                              : nullptr);

      bool needs_release = false;
      bool loc_from_active_layer = false;
      const ExpertLocation *resolved = expert_ffn_resolve_gpu_loc(
          &gpu_ctx, 1, exp->gate_proj.meta, exp->up_proj.meta,
          exp->down_proj.meta, &needs_release, &loc_from_active_layer, ram_mgr,
          engine);

      bool gpu_issued = false;
      if (resolved) {
        if (scratch.pending_gpu.size() >= scratch.gpu_pool.size()) {
          LOG_STREAM_ERROR()
              << "[DIAG-OOB] gpu_pool OVERFLOW: pending_gpu.size()="
              << scratch.pending_gpu.size()
              << " gpu_pool.size()=" << scratch.gpu_pool.size()
              << " layer=" << layer_idx << " k=" << k
              << " expert=" << routed_expert_id << " topk=" << topk << "\n";
        }
        GpuExpertScratch &gs = scratch.gpu_pool[scratch.pending_gpu.size()];
        gpu_issued = expert_ffn_gpu_issue(
            gs, resolved, token_hidden, exp->gate_proj.meta, exp->up_proj.meta,
            1, D, moe_inter, gpu_ctx.topo);
        if (gpu_issued) {
          scratch.pending_gpu.push_back({k, exp, resolved, needs_release,
                                         loc_from_active_layer,
                                         gpu_ctx.expert_id});
          // Protect this RAM slot from being reclaimed by ANY
          // RAM-tier victim-selection (get_expert/get_experts_
          // batch/demote_from_vram) until Pass B below has
          // finished with this k -- see MoEExpertBlock::
          // ram_temp_pin_count's own comment for the real
          // release_hold-identity-mismatch bug this closes.
          // Harmless no-op if exp is a shadow scratch block
          // (never a RAM-tier victim-selection candidate). Goes
          // through RAMSafetyManager::ram_pin (not a direct field
          // increment) -- a LockFreePrefetcher worker thread's own
          // claim_or_wait victim-selection scan now reads this
          // exact field too, under ram_mutex; an unlocked write
          // here would race that locked read. See ram_pin's own
          // comment (ram_manager.hpp).
          ram_mgr.ram_pin(*exp);
        } else {
          // gpu_scratch.ensure's cudaMalloc genuinely failed (VRAM
          // exhaustion) -- this expert never got Pass B queued, so
          // release any hold taken above HERE rather than leaving
          // it dangling forever (expert_ffn_gpu_finish, the only
          // other place that releases it, will never run for this
          // expert). Falls through to the CPU path below like any
          // other GPU-dispatch failure.
          //
          // Uses gpu_ctx.expert_id (captured before
          // expert_ffn_resolve_gpu_loc ran), NOT exp->expert_id --
          // that call can promote this exact `exp` into the main
          // VRAM cache and, on a RAM-tier reclaim, clear
          // exp->expert_id to -1 before we ever get here. Same
          // "capture, don't re-read late" fix as PendingGpuExpert's
          // own expert_id field.
          if (needs_release && gpu_ctx.topo) {
            gpu_ctx.topo->release_hold(layer_idx, gpu_ctx.expert_id,
                                       resolved->device_ptr,
                                       loc_from_active_layer);
          }
          LOG_ERROR("[EXPERT-GPU] layer=%d expert=%d: GPU issue failed "
                    "(scratch buffer "
                    "grow), falling back to CPU for this call\n",
                    layer_idx, gpu_ctx.expert_id);
        }
      }
      if (!gpu_issued && exp->vram_shadow) {
        // Rare: a shadow-fast-pathed expert (confirmed VRAM-resident
        // moments ago, in the loading loop below) failed to actually
        // dispatch on GPU after all -- either a genuine race
        // (evicted in the narrow window between that check and
        // here) or expert_ffn_gpu_issue's scratch buffer grow
        // failing. Either way we never loaded real RAM data for it,
        // since the VRAM fast path was expected to succeed -- load
        // it now, synchronously, on this rare/slow path, so the CPU
        // fallback below has real bytes instead of the shadow's
        // null host_ptr.
        MoEExpertBlock *real =
            ram_mgr.get_expert(engine, layer_idx, routed_expert_id);
        if (real && real->gate_proj.host_ptr && real->down_proj.host_ptr) {
          perf().n_shadow_race_reload_success.fetch_add(
              1, std::memory_order_relaxed);
          exp = real;
          scratch.expert_ptrs[k] = real;
        } else {
          perf().n_shadow_race_reload_hard_fail.fetch_add(
              1, std::memory_order_relaxed);
          LOG_ERROR(
              "[EXPERT-GPU] layer=%d expert=%d: VRAM fast-path failed AND "
              "emergency "
              "RAM reload also failed -- skipping this expert's contribution "
              "entirely rather than read a null/stale pointer\n",
              layer_idx, routed_expert_id);
          return;
        }
      }
      if (!gpu_issued) {
        // 2026-08-29: real, hardware-confirmed race found via
        // bisection of a live SIGSEGV (multiple OMP worker threads
        // faulting at the identical instruction inside
        // dot_iq2_xxs_f32_avx512, reproducible at normal execution
        // speed, absent under gdb's slower scheduling -- classic
        // race signature, not a deterministic bug). This is the
        // ONLY reader of a RAM-tier block's raw host_ptr bytes in
        // this whole file that did NOT bracket its read with
        // ram_pin/ram_unpin (grepped every other call site: the
        // GPU-dispatch Pass A/B hold, decode_async_prewarm's
        // candidates, the speculative-promotion claim all do) --
        // without it, a concurrent prefetch-worker or active-layer-
        // upload-worker thread's own RAM-tier victim-selection scan
        // (get_expert/get_experts_batch/demote_from_vram, all of
        // which check ram_temp_pin_count) could reclaim/realloc
        // this exact slab WHILE cpu_quant_matmul's OMP threads are
        // still mid-read of gate/up/down_data below -- exactly
        // explaining every observed symptom (only on a longer
        // prefill with more CPU fallbacks, multiple threads
        // faulting at once, timing-sensitive). exp is guaranteed a
        // real RAM-tier block here (never the vram_shadow's null
        // host_ptr -- the branch above either reloads it via
        // ram_mgr.get_expert or returns early).
        ram_mgr.ram_pin(*exp);
        // 2026-08-30: a same-day diagnostic here (now removed) was
        // built while chasing a real, reproducible S=100 crash in
        // this exact dequant path, and briefly compared exp->
        // expert_id against scratch.tok_topk_ids[s*topk+k] right
        // after ram_pin -- but this comparison is WRONG in general,
        // not just here: a Hit result from claim_or_wait can be a
        // MOVE-promoted slot (move_on_promotion, do_promote_ram_
        // slot_to_vram's allow_move branch, ram_manager.hpp) --
        // expert_id is deliberately reset to -1 as part of that
        // promotion's own bookkeeping, while gate_proj/up_proj/
        // down_proj/slab_buffer remain the CORRECT, still-valid
        // bytes for exactly this expert (the RAM copy just became
        // "redundant, VRAM has it now," not wrong) -- see
        // get_experts_batch's own "claimed_this_batch" comment,
        // which already documents and relies on this exact
        // behavior. The diagnostic's own mismatch check fired on
        // this completely normal, expected case, and its early
        // return was silently discarding valid experts' contributions
        // (confirmed on real hardware: expert_calls=.../0CPU for a
        // whole request, cpu=0.00s -- every CPU-dispatched expert
        // that hit this false positive got dropped, not computed).
        // The REAL race (see claim_or_wait's own pin_on_hit
        // parameter comment) was a separate, genuine unlock-then-
        // relock gap between claim_or_wait's own return and a
        // caller pinning the slot in a SEPARATE, later lock
        // acquisition -- fixed by pinning atomically inside claim_
        // or_wait itself (get_experts_batch's Phase A/C, ram_
        // manager.hpp) before ever releasing its lock. Confirmed
        // fixed on real hardware: the same S=100 crashing config
        // that reliably segfaulted (or, without external
        // interruption, ran to completion with 187K+ NaN-cascade
        // errors) now completes cleanly, 0 errors, real hardware,
        // once ONLY the atomic pin_on_hit fix is applied (i.e.
        // without this now-removed, overly-strict identity check).
        expert_ffn_cpu(
            token_hidden, scratch.down_buf.data(), exp->gate_proj.host_ptr,
            exp->gate_proj.meta, exp->up_proj.host_ptr, exp->up_proj.meta,
            exp->down_proj.host_ptr, exp->down_proj.meta,
            scratch.gate_buf.data(), scratch.up_buf.data(), 1, D, moe_inter,
            clamp_exp, layer_idx, scratch.tok_topk_ids[s * topk + k]);
        ram_mgr.ram_unpin(*exp);
        accumulate(k, exp, scratch.down_buf.data());
      }
    };
    scratch.pending_gpu.clear();

    // --- Load all needed experts (LRU cache handles hits/misses; most
    // will already be warm here thanks to Phase 1/2 above). Batched:
    // get_experts_batch parallelizes the disk reads for whichever of
    // this layer's topk picks are genuine misses (not already staged
    // via prefetch), instead of the old one-pread-at-a-time sequential
    // loop
    //
    // FIX (real, measured "[TIER-OVERLAP]" bug -- see CLAUDE.md): this
    // used to call get_experts_batch UNCONDITIONALLY for the entire
    // topk list, even for experts already GPU-resident -- wasting a
    // real disk read and a real RAM slot on data whose BYTES the GPU
    // path never reads (it reads weight bytes from
    // ExpertLocation::device_ptr, and the per-projection TYPE from the
    // packed base tensor, uniform across every expert in a layer --
    // neither needs a RAM copy). A real run showed 66% of everything
    // VRAM-resident was ALSO RAM-resident as a direct result, 12+ GB
    // double-counted. Fixed by checking VRAM residency first (a real
    // find_expert lookup, same one dispatch_expert_k above already does
    // for its own purposes) and building a metadata-only "shadow" block
    // (populate_vram_shadow, zero I/O) for anything already resident +
    // GPU-type-supported, instead of loading it into RAM at all. Only
    // the remainder -- not yet VRAM-resident, or an unsupported type --
    // still goes through the real get_experts_batch path below,
    // batched exactly as before.
    //
    // COMPUTE/IO OVERLAP: a shadowed expert's kernels are issued via
    // dispatch_expert_k IMMEDIATELY below, inline in this same k-loop --
    // NOT deferred until after get_experts_batch runs, as they used to
    // be -- so they run on the GPU stream while this thread goes on to
    // block on this layer's disk-bound misses instead of the two
    // running strictly sequentially.
    {
      size_t shadow_gb = 0, shadow_ub = 0, shadow_db = 0;
      const GGUFTensor *shadow_gM = nullptr, *shadow_uM = nullptr,
                       *shadow_dM = nullptr;
      // FIX (real, high-impact bug found while chasing a persistent
      // [TIER-OVERLAP] baseline that didn't respond to any promotion-
      // reclaim fix): this used to call engine.packed_slab_geometry(...)
      // here, which only recognizes the PACKED tensor convention. For
      // an unpacked_expert_format=true model (this project's own
      // recommended format for the single-DMA-per-expert read path),
      // packed_slab_geometry returns false for EVERY layer -- meaning
      // layer_gpu_capable was false everywhere, the entire shadow-
      // fast-path below (the mechanism whose whole job is "skip the
      // RAM-tier lookup for anything already VRAM-resident") never
      // fired even once, and EVERY routed expert on EVERY token --
      // including all of pinned_vram_capacity's PERMANENTLY VRAM-
      // resident experts, which never go through any promotion event
      // and therefore have no other path to ever getting their RAM
      // copy reclaimed -- fell through to the unconditional RAM-tier
      // load below. This is almost certainly why [TIER-OVERLAP]
      // stayed flat (~520-720) across four rounds of active-layer-
      // promotion-reclaim fixes: those fixes only address experts
      // freshly PROMOTED into VRAM, never the much larger set that's
      // ALREADY VRAM-resident (pinned or otherwise) and gets
      // redundantly RAM-loaded anyway because this gate never lets
      // them skip that load. The fix already exists a few lines
      // below for a DIFFERENT feature (cuFile eligibility) that hit
      // the exact same packed-only limitation and was updated to use
      // expert_layer_geometry (the unified dispatcher supporting
      // both formats) -- that fix was never carried back to this
      // older variable. Now uses the same unified dispatcher here too.
      bool layer_gpu_capable =
          !cpu_only_mode() && !is_cpu_affinity_layer(layer_idx) &&
          ram_mgr.vram_cache &&
          engine.expert_layer_geometry(layer_idx, shadow_gb, shadow_ub,
                                       shadow_db, shadow_gM, shadow_uM,
                                       shadow_dM) &&
          gpu_type_supported(shadow_gM->type) &&
          gpu_type_supported(shadow_uM->type) &&
          gpu_type_supported(shadow_dM->type);

      // cuFile (GPUDirect Storage) direct disk->VRAM eligibility for
      // this layer -- kept as a SEPARATE check from layer_gpu_capable
      // above (not merged into one, even though both now use
      // expert_layer_geometry) since the two conditions are otherwise
      // independent (cuFile also needs cuda_cufile_active) and
      // merging them would make a future change to one silently
      // affect the other. Layer-level pre-filter only (cheap,
      // tensor-table lookups) -- the actual per-expert absolute file
      // offset is resolved again inside the k-loop below, where the
      // specific expert_id is known.
      bool cufile_layer_eligible = !option_b_target &&
                                   engine.cuda_cufile_active &&
                                   !is_cpu_affinity_layer(layer_idx) &&
                                   ram_mgr.vram_cache;
      if (cufile_layer_eligible) {
        size_t cf_gb = 0, cf_ub = 0, cf_db = 0;
        const GGUFTensor *cf_gM = nullptr, *cf_uM = nullptr, *cf_dM = nullptr;
        cufile_layer_eligible =
            engine.expert_layer_geometry(layer_idx, cf_gb, cf_ub, cf_db, cf_gM,
                                         cf_uM, cf_dM) &&
            gpu_type_supported(cf_gM->type) &&
            gpu_type_supported(cf_uM->type) && gpu_type_supported(cf_dM->type);
      }

      static thread_local std::vector<int> ram_needed_ids;
      static thread_local std::vector<int> ram_needed_k;
      static thread_local std::vector<MoEExpertBlock *> ram_needed_ptrs;
      // std::vector<bool> is a bitset specialization with no .data()
      // -- use a fixed-size std::array instead (real per-layer topk
      // is always small in every model this engine supports; 64 is
      // generous headroom, matching kFetchHistogramBuckets/the 0-8
      // cpu_offload_max_experts range's own "small" assumption).
      static thread_local std::array<bool, 64> ram_was_real_fetch{};
      ram_needed_ids.clear();
      ram_needed_k.clear();
      ram_needed_ids.reserve(topk);
      ram_needed_k.reserve(topk);

      for (int k = 0; k < topk; ++k) {
        // 2026-08-29, same "Fifth round" investigation as
        // t_dispatch_expert_k_total above -- an upper bound over
        // this whole shadow-fast-path pre-check iteration
        // (find_expert() + populate_vram_shadow() for the shadow
        // check itself, plus dispatch_expert_k's own now-timed
        // body when this k IS shadowed and dispatched inline
        // below, before get_experts_batch's blocking disk read).
        PERF_SCOPE(t_shadow_precheck_total);
        int expert_id = scratch.tok_topk_ids[s * topk + k];
        bool shadowed = false;
        scratch.vram_loc_hint[k] =
            nullptr; // explicit every layer/token -- resize()'s
                     // one-time clear only covers the FIRST call
        if (layer_gpu_capable &&
            (!option_b_target ||
             engine.qwen4exp_option_b_hot_expert(layer_idx, expert_id))) {
          const ExpertLocation *loc =
              ram_mgr.vram_cache->find_expert(layer_idx, expert_id);
          if (loc && loc->device_ptr) {
            // Measurement only (expert_fetch_tracker.hpp) -- this
            // is a genuine VRAM-tier HIT via the shadow fast
            // path, resolved WITHOUT ever calling consume_staged.
            // Was this exact expert ALSO recently touched by the
            // async prefetcher?
            record_hit_if_recently_fetched(layer_idx, expert_id,
                                           /*is_vram_tier=*/true);
            MoEExpertBlock &shadow = scratch.vram_shadow_blocks[k];
            engine.populate_vram_shadow(shadow, layer_idx, expert_id);
            scratch.expert_ptrs[k] = &shadow;
            scratch.vram_loc_hint[k] = loc;
            shadowed = true;
          }
        }
        // cuFile direct disk->VRAM: only ever for a GENUINE double
        // miss (not shadowed above -- i.e. not already VRAM-
        // resident -- AND not already RAM-resident). Re-reading
        // from disk via cuFile when a perfectly good copy already
        // sits in host RAM would waste real NVMe bandwidth for a
        // slower net result than the existing RAM->VRAM promotion
        // path, which is exactly what this check exists to prevent
        // -- see RAMSafetyManager::is_ram_resident's own comment.
        if (!shadowed && cufile_layer_eligible) {
          if (ram_mgr.is_ram_resident(layer_idx, expert_id)) {
            perf().n_cufile_skipped_ram_resident.fetch_add(
                1, std::memory_order_relaxed);
          } else {
            // Resolve THIS expert's absolute file offsets --
            // packed format needs expert_id*per_expert_size
            // added to the shared _exps tensor's base offset;
            // unpacked format's per-expert tensor offset IS
            // already absolute (mirrors build_expert_read_
            // requests' identical packed-vs-unpacked branching).
            // layer_is_unpacked, not the global flag -- see its own
            // comment in engine.hpp (2026-09, Phase 5: per-layer mixed
            // packed/unpacked models). Cached once per layer here since
            // it's consulted 4 times below, all needing the same answer.
            bool this_layer_unpacked = engine.layer_is_unpacked(layer_idx);
            size_t eg = 0, eu = 0, ed = 0;
            const GGUFTensor *egM = nullptr, *euM = nullptr, *edM = nullptr;
            bool have_geom =
                this_layer_unpacked
                    ? engine.unpacked_expert_geometry(layer_idx, expert_id, eg,
                                                      eu, ed, egM, euM, edM)
                    : engine.packed_slab_geometry(layer_idx, eg, eu, ed, egM,
                                                  euM, edM);
            if (have_geom) {
              uint64_t gate_off =
                  egM->offset +
                  (this_layer_unpacked ? 0 : (uint64_t)expert_id * eg);
              uint64_t up_off =
                  euM->offset +
                  (this_layer_unpacked ? 0 : (uint64_t)expert_id * eu);
              uint64_t down_off =
                  edM->offset +
                  (this_layer_unpacked ? 0 : (uint64_t)expert_id * ed);
              const cufile_io::FileHandle *h =
                  engine.model_file.cufile_handle_for_tensor(*egM);
              if (h && h->valid) {
                const ExpertLocation *cf_loc =
                    ram_mgr.vram_cache->upload_expert_cufile(
                        layer_idx, expert_id, *h, gate_off, eg, up_off, eu,
                        down_off, ed);
                if (cf_loc) {
                  MoEExpertBlock &shadow = scratch.vram_shadow_blocks[k];
                  engine.populate_vram_shadow(shadow, layer_idx, expert_id);
                  scratch.expert_ptrs[k] = &shadow;
                  scratch.vram_loc_hint[k] = cf_loc;
                  shadowed = true;
                }
              }
            }
          }
        }
        if (!shadowed) {
          ram_needed_ids.push_back(expert_id);
          ram_needed_k.push_back(k);
        } else {
          // Already VRAM-resident (or cuFile-loaded moments ago
          // above) -- issue its GPU kernels now instead of waiting
          // for the blocking get_experts_batch call below, so they
          // run on the GPU stream concurrently with that call's
          // disk I/O for this layer's genuine misses.
          dispatch_expert_k(k);
        }
      }

      // EngineConfig::prefill_dispatch_lookahead_tokens (prefill-only,
      // additive) -- see that field's own comment (engine_config.hpp)
      // for the full motivation. Widens the combined get_experts_batch
      // call below to also opportunistically RAM-warm upcoming tokens'
      // not-yet-resident picks. A sentinel k of -1 marks a "lookahead
      // only" entry in ram_needed_k -- it has no (token,k) slot of its
      // own to write into or dispatch, so every one of the 4 real
      // consumers of ram_needed_k below (expert_ptrs write-back, the
      // cpu_offload_max_experts forcing loop, decode_async_prewarm's
      // candidate collection, and the final dispatch loop) must skip
      // it. Real risk this deliberately avoids: NOT promoting a
      // lookahead entry into the active-layer VRAM pool via decode_
      // async_prewarm -- only warming the RAM tier -- since this
      // codebase's own history (CLAUDE.md) has repeatedly found
      // over-eager speculative VRAM promotion to be a real regression
      // (churn/eviction), while RAM-tier warming alone has no such
      // downside (a claim_or_wait slot that's never consumed just sits
      // resident, exactly like an ordinary background prefetch hit).
      if (S > 1 && engine.prefill_dispatch_lookahead_tokens > 0 &&
          layer_gpu_capable) {
        int lookahead_end =
            std::min(S, s + 1 + engine.prefill_dispatch_lookahead_tokens);
        for (int s2 = s + 1; s2 < lookahead_end; ++s2) {
          for (int k2 = 0; k2 < topk; ++k2) {
            int expert_id2 = scratch.tok_topk_ids[s2 * topk + k2];
            if (ram_mgr.vram_cache->find_expert(layer_idx, expert_id2)) {
              perf().n_prefill_lookahead_skipped_vram.fetch_add(
                  1, std::memory_order_relaxed);
              continue;
            }
            if (ram_mgr.is_ram_resident(layer_idx, expert_id2)) {
              perf().n_prefill_lookahead_skipped_ram.fetch_add(
                  1, std::memory_order_relaxed);
              continue;
            }
            bool dup = std::find(ram_needed_ids.begin(), ram_needed_ids.end(),
                                 expert_id2) != ram_needed_ids.end();
            if (dup) {
              perf().n_prefill_lookahead_skipped_dup.fetch_add(
                  1, std::memory_order_relaxed);
              continue;
            }
            ram_needed_ids.push_back(expert_id2);
            ram_needed_k.push_back(
                -1); // sentinel: lookahead-only, no (token,k) to fill
            perf().n_prefill_lookahead_added.fetch_add(
                1, std::memory_order_relaxed);
          }
        }
      }

      if (!ram_needed_ids.empty()) {
        ram_needed_ptrs.resize(ram_needed_ids.size());
        // nullptr (rather than overflow the fixed-size array) on the
        // extremely-unlikely chance this layer's real per-layer topk
        // exceeds ram_was_real_fetch's capacity -- cpu_offload_max_
        // experts just can't engage for this one call, same
        // graceful-degrade discipline as every other GPU/best-effort
        // fast path in this codebase; every OTHER field this call
        // populates is unaffected.
        bool real_fetch_overflow =
            ram_needed_ids.size() > ram_was_real_fetch.size();
        // ram_needed_k.data() doubles as the caller_k parameter --
        // its own -1 sentinel (prefill_dispatch_lookahead_tokens
        // additions) is exactly what get_experts_batch needs to tag
        // MoEExpertBlock::loaded_by_lookahead correctly, no separate
        // array required.
        if (S == 1 && engine.family_ops.family == ModelFamily::Qwen4Exp)
          moe_pipeline_internal::qwen4exp_record_pilot_fetch_start(layer_idx);
        ram_mgr.get_experts_batch(
            engine, layer_idx, ram_needed_ids.data(),
            (int)ram_needed_ids.size(), ram_needed_ptrs.data(),
            real_fetch_overflow ? nullptr : ram_was_real_fetch.data(),
            ram_needed_k.data());
        // 2026-08-30: real, hardware-confirmed bug found while
        // debugging a S=100 crash/NaN-corruption with speculative_
        // promote_to_active_layer + active_layer_upload_worker_
        // threads both enabled -- see dispatch_expert_k's own
        // RAM-IDENTITY diagnostic for the full investigation.
        // get_experts_batch's Phase A (ram_manager.hpp) now pins
        // every out[k] it resolves (Hit and Claimed) at the MOMENT
        // of resolution -- see that function's own comment for why
        // "pin right before use" (the 2026-08-29 fix, dispatch_
        // expert_k's own CPU-path pin below) is insufficient: the
        // slot can already be reclaimed before that later pin ever
        // runs, including DURING get_experts_batch's own Phase B/C
        // execution (real blocking I/O, real time for another
        // thread's victim-selection scan to run). This loop is
        // that pin's matching release:
        //   - a REAL entry (ram_needed_k[i]>=0) is held through
        //     THIS token's own full dispatch instead -- released
        //     right after dispatch_expert_k(k) returns, in the
        //     ram_needed_k dispatch loop below (covers both the CPU
        //     path's synchronous compute and the GPU path's
        //     synchronous upload_expert/upload_expert_active call).
        //   - a LOOKAHEAD-only entry (ram_needed_k[i]<0,
        //     prefill_dispatch_lookahead_tokens) belongs to a
        //     LATER token, never dispatched by this one -- once
        //     get_experts_batch returns it's fully published
        //     (Ready, not just Loading), safe to treat exactly like
        //     any other resident expert from here on (a later
        //     token's own Phase A will find+use it normally); held
        //     no longer than necessary, released right here.
        for (size_t i = 0; i < ram_needed_ids.size(); ++i) {
          if (ram_needed_k[i] < 0) {
            ram_mgr.ram_unpin(*ram_needed_ptrs[i]);
            continue;
          }
          scratch.expert_ptrs[ram_needed_k[i]] = ram_needed_ptrs[i];
        }

        // Qwen-only CPU-affinity grouping. At this point every selected
        // expert has a published RAM block and get_experts_batch has already
        // pinned it for the dispatch that follows. Consume the whole token in
        // one grouped gate/up pass when the IQ1_M geometry permits it, then
        // clear the normal per-expert work lists so the legacy active-layer
        // prewarm/upload/dispatch code becomes a no-op for this token. Other
        // families, mixed quant signatures, and non-affinity layers retain
        // the established path byte-for-byte.
        if (S == 1 && engine.qwen4exp_cpu_ggml_moe &&
            engine.family_ops.family == ModelFamily::Qwen4Exp &&
            engine.family_ops.uses_hyper_connections &&
            is_cpu_affinity_layer(layer_idx) &&
            ram_needed_k.size() == (size_t)topk) {
          std::vector<MoEExpertBlock *> grouped_experts((size_t)topk);
          bool complete = true;
          for (int k = 0; k < topk; ++k) {
            grouped_experts[(size_t)k] = scratch.expert_ptrs[k];
            if (!grouped_experts[(size_t)k] ||
                grouped_experts[(size_t)k]->vram_shadow)
              complete = false;
          }
          const float *grouped_weights =
              scratch.tok_topk_weights.data() + (size_t)s * topk;
          if (complete &&
              expert_ffn_cpu_qwen_grouped_quantized(
                  token_hidden, token_output, grouped_experts, grouped_weights,
                  D, moe_inter, clamp_exp, layer_idx)) {
            static std::atomic<bool> logged_grouped_cpu{false};
            if (!logged_grouped_cpu.exchange(true, std::memory_order_relaxed))
              LOG_STREAM_INFO()
                  << "[CPU-MOE] Qwen grouped IQ1_M CPU path active at layer "
                  << layer_idx << " (topk=" << topk
                  << ", shared activation quantization)\n";
            for (MoEExpertBlock *block : ram_needed_ptrs)
              if (block)
                ram_mgr.ram_unpin(*block);
            ram_needed_ids.clear();
            ram_needed_k.clear();
            ram_needed_ptrs.clear();
          }
        }

        // EngineConfig::cpu_offload_max_experts: of THIS token/
        // layer's genuine real-fetch misses (ram_was_real_fetch[i],
        // set by get_experts_batch's own Phase A -- the exact
        // population [[LAYER-MISS]]'s data was gathered from, see
        // that field's own comment), redirect up to the configured
        // cap to CPU compute (force_cpu_offload_k[k], read by
        // dispatch_expert_k below) instead of letting them continue
        // to a VRAM upload. Scoped OUT of cpu_affinity_layers layers
        // (already 100% CPU by a standing config, nothing to gain by
        // also computing this) and OUT of cpu_only_mode() (the
        // diagnostic kill-switch already forces everything to CPU;
        // running this loop there would just waste cycles double-
        // deciding an outcome that's already fixed). A Hit/Loading
        // entry (ram_was_real_fetch[i]==false) never touched disk
        // this call and is never eligible -- forcing it to CPU would
        // throw away a cache hit for no reason.
        if (engine.cpu_offload_max_experts > 0 && !cpu_only_mode() &&
            !is_cpu_affinity_layer(layer_idx) && !real_fetch_overflow) {
          int budget = engine.cpu_offload_max_experts;
          for (size_t i = 0; i < ram_needed_ids.size(); ++i) {
            // A lookahead-only entry (ram_needed_k[i]==-1) isn't
            // this token's own real fetch to force to CPU or
            // not -- skip before any k-indexed access.
            if (ram_needed_k[i] < 0)
              continue;
            if (!ram_was_real_fetch[i])
              continue;
            if (budget > 0) {
              force_cpu_offload_k[ram_needed_k[i]] = true;
              --budget;
              perf().n_cpu_offload_forced.fetch_add(1,
                                                    std::memory_order_relaxed);
            } else {
              perf().n_cpu_offload_left_uncapped.fetch_add(
                  1, std::memory_order_relaxed);
            }
          }
        }

        // Batch-async pre-warm this token's own RAM->VRAM active-
        // layer misses BEFORE dispatching them one at a time below --
        // mirrors prefill Phase 2's own upload_expert_active_async +
        // sync_active_layer_batch pattern (gpu_topology.cpp),
        // applied to this TOKEN's real ram_needed_k set instead of a
        // whole layer's predicted candidates. Real, measured DMA-level
        // win on both GPUs in this box via tests/bench_dma_sync_vs_
        // async.cu -- see that file's header comment.
        //
        // DEFAULT OFF (engine.decode_async_prewarm) and CHUNKED
        // (engine.decode_async_prewarm_max_batch, default 3), NOT
        // the simple "batch everything in one shot" shape this
        // started as -- that unbounded version caused a real
        // production incident (host RAM exhaustion / swap thrashing
        // / freeze) on a config running host_ram_gb close to
        // physical RAM. Processing in small chunks, each with its
        // own full pin -> issue -> sync -> unpin cycle before the
        // next chunk starts, bounds how many RAM slots are
        // simultaneously pinned and how many active-layer slots are
        // simultaneously mid-DMA at any one instant -- trading some
        // batching benefit for tokens with more misses than
        // max_batch, in exchange for a hard cap on peak simultaneous
        // footprint. See EngineConfig::decode_async_prewarm's own
        // comment (engine_config.hpp) for the full incident writeup.
        //
        // Only ever targets the ACTIVE-LAYER pool (the one pool with
        // an existing async+batch-sync primitive) -- a model without
        // one allocated (use_active_layer_area=false, e.g. DeepSeek-
        // V4) makes upload_expert_active_async a guaranteed no-op
        // nullptr return for every candidate, so this whole block
        // degrades to zero-cost no-op and dispatch_expert_k's
        // existing main-cache (upload_expert) sync path runs
        // completely unchanged for those models.
        // 2026-08-19: a dispatch-reordering fix was TRIED and
        // MEASURED here (issue dispatch_expert_k as soon as each
        // piece of data is ready -- immediately for non-prewarm
        // candidates, right after each chunk's own sync for
        // prewarmed ones -- instead of deferring every ram_needed
        // dispatch to one final loop after ALL decode_async_prewarm
        // chunks finish), on the theory that queuing GPU compute
        // earlier would give later chunks' H2D DMA real, already-
        // in-flight work to overlap with (motivated by n_active_
        // prewarm_saw_compute_stream_busy showing 0% busy at this
        // call site -- see that counter's own comment, perf_timer.hpp).
        // REVERTED after real-hardware A/B: the SAME diagnostic still
        // read compute_stream busy 0/15092 (0.0%) even with compute
        // issued earlier -- i.e. this decode-time model's per-token
        // GPU kernels are small/fast enough that they finish before
        // the CPU-side orchestration (locks, identity checks, the
        // next chunk's own pin/issue calls) loops back around to
        // check again, so there was never any slow GPU work to hide
        // DMA behind in the first place. Decode throughput was
        // unchanged (0.86 tok/s, same band as before), output stayed
        // coherent, full test suite passed -- a clean, well-
        // instrumented negative result, not a bug, but real added
        // control-flow complexity in this exact historically
        // incident-prone area for zero measured benefit, so reverted
        // rather than kept. This is now the THIRD independent real-
        // hardware confirmation (after demote_from_vram's and
        // upload_expert_active's own stream-busy findings, both
        // documented in CLAUDE.md) of the same underlying fact: this
        // hardware's decode-time dispatch loop is genuinely CPU-
        // orchestration-bound per call, not schedulable-around by
        // reordering when work gets issued. See CLAUDE.md's
        // 2026-08-19 entry for the full writeup.
        if (engine.decode_async_prewarm && !option_b_target &&
            !moe_pipeline_detail::qwen4exp_decode_ggml_path(engine) &&
            !cpu_only_mode() && !is_cpu_affinity_layer(layer_idx) &&
            ram_mgr.vram_cache) {
          static thread_local std::vector<size_t> decode_async_candidates;
          decode_async_candidates.clear();
          for (size_t i = 0; i < ram_needed_ids.size(); ++i) {
            // A lookahead-only entry (ram_needed_k[i]==-1,
            // prefill_dispatch_lookahead_tokens) is deliberately
            // NEVER a candidate for active-layer VRAM pre-warm --
            // only for RAM-tier warming above. Promoting a not-
            // yet-needed expert into the scarce active-layer pool
            // this early is exactly the over-eager speculative-
            // promotion regression this codebase has repeatedly
            // found and reverted (see CLAUDE.md) -- and reading
            // force_cpu_offload_k[-1] just below would be OOB.
            if (ram_needed_k[i] < 0)
              continue;
            MoEExpertBlock *blk = ram_needed_ptrs[i];
            // Cheap pre-filter only, NOT the safety-critical check
            // (that one is re-done immediately adjacent to
            // ram_pin below, right before use -- see that
            // comment for why re-checking there, not trusting
            // this earlier read, is what actually matters).
            // Still worth filtering here too, to avoid adding a
            // definitely-invalid candidate to the list at all.
            if (!blk || !blk->slab_buffer || blk->slab_capacity == 0)
              continue;
            // Already capped to CPU compute this token (see
            // force_cpu_offload_k above) -- pre-warming its VRAM
            // slot would pay for exactly the DMA this feature
            // exists to skip, for an expert dispatch_expert_k is
            // about to force off the GPU path anyway.
            if (force_cpu_offload_k[ram_needed_k[i]])
              continue;
            int expert_id = ram_needed_ids[i];
            // Side-effect-free peek (active_layer_has, NOT
            // find_active_expert -- the real lookup would bump
            // hit_count/could promote as a side effect of a mere
            // pre-check, corrupting the exact state this batch is
            // simultaneously choosing victims from). Already-
            // resident candidates need no upload -- dispatch_
            // expert_k's own find_active_expert call will hit
            // them directly.
            if (ram_mgr.vram_cache->active_layer_has(layer_idx, expert_id))
              continue;
            bool gpu_capable = gpu_type_supported(blk->gate_proj.meta.type) &&
                               gpu_type_supported(blk->up_proj.meta.type) &&
                               gpu_type_supported(blk->down_proj.meta.type);
            if (!gpu_capable)
              continue;
            decode_async_candidates.push_back(i);
          }

          static thread_local std::vector<ExpertLocation *>
              decode_async_victims;
          static thread_local std::vector<MoEExpertBlock *>
              decode_async_pinned_blocks;
          size_t max_batch =
              (size_t)std::max(1, engine.decode_async_prewarm_max_batch);
          for (size_t chunk_start = 0;
               chunk_start < decode_async_candidates.size();
               chunk_start += max_batch) {
            size_t chunk_end = std::min(decode_async_candidates.size(),
                                        chunk_start + max_batch);
            decode_async_victims.clear();
            decode_async_pinned_blocks.clear();
            for (size_t ci = chunk_start; ci < chunk_end; ++ci) {
              size_t i = decode_async_candidates[ci];
              MoEExpertBlock *blk = ram_needed_ptrs[i];
              int expert_id = ram_needed_ids[i];
              // FIX (real bug behind a real production
              // incident's "[ERROR] [GPU] Invalid upload_
              // expert_active_async parameters" log line): the
              // FIRST version of this code checked slab_buffer
              // once in the filter pass above, then did MORE
              // lock-taking work (active_layer_has, gpu_
              // capable) before actually pinning and using it
              // -- a real TOCTOU window in which a concurrent
              // prefetch-worker thread's own ORDINARY (not
              // even the buggy "fallback to slot 0") victim-
              // selection could legitimately reclaim this
              // exact, still-unprotected RAM slot before this
              // code got around to protecting it. The fix is
              // this exact ordering: re-check freshly, pin,
              // THEN re-read slab_buffer/slab_capacity AGAIN
              // (not the filter pass's now-possibly-stale
              // values) immediately after pinning -- ram_pin
              // is what actually excludes this slot from
              // claim_or_wait's victim-selection from this
              // point on (never_pick checks ram_temp_pin_
              // count>0 under the same lock), so only a value
              // read AFTER the pin is guaranteed protected.
              // Nothing else (no lock-taking calls) sits
              // between this check and the pin below.
              if (!blk || !blk->slab_buffer || blk->slab_capacity == 0)
                continue;
              ram_mgr.ram_pin(*blk);
              // Identity check, not just non-null/non-zero:
              // if this slot was reclaimed AND already
              // repopulated for a DIFFERENT expert in the
              // window before the pin above, slab_buffer/
              // slab_capacity could both look perfectly valid
              // while actually belonging to someone else's
              // data -- a silent wrong-bytes-uploaded-under-
              // the-wrong-label bug, worse than a clean error.
              // Confirms this slot still actually holds the
              // expert we intend before trusting its bytes.
              if (blk->expert_id != expert_id) {
                ram_mgr.ram_unpin(*blk);
                perf().n_decode_async_prewarm_stale_caught.fetch_add(
                    1, std::memory_order_relaxed);
                continue;
              }
              const uint8_t *host_data = blk->slab_buffer;
              // real total, not slab_capacity -- same bug/reasoning as
              // ram_manager.hpp's promote_ram_to_active_layer (see its own
              // comment for the full incident): slab_capacity is this
              // slot's grow-only allocation, which can legitimately
              // outlive whatever smaller expert currently occupies it
              // after an LRU reuse -- uploading it copies real data
              // followed by a previous occupant's stale tail bytes.
              size_t host_bytes = blk->gate_proj.meta.size_bytes +
                                  blk->up_proj.meta.size_bytes +
                                  blk->down_proj.meta.size_bytes;
              if (!host_data || host_bytes == 0) {
                // Reclaimed (freed, not yet repopulated) in
                // the now-negligible window between the
                // check above and the pin -- extremely
                // unlikely (no lock-taking work in between
                // anymore) but cheap to guard against rather
                // than assume can't happen.
                ram_mgr.ram_unpin(*blk);
                perf().n_decode_async_prewarm_stale_caught.fetch_add(
                    1, std::memory_order_relaxed);
                continue;
              }
              // The async DMA below keeps reading host_data in
              // the background until sync_active_layer_batch's
              // cudaStreamSynchronize confirms it landed --
              // ram_pin above is what protects that window
              // (bounded further by chunking); unpinned again
              // immediately below once this CHUNK's sync
              // confirms completion.
              // 2026-08-27: real coverage measurement -- see
              // n_active_layer_upload_src_pinned's own comment
              // (perf_timer.hpp).
              if (blk->slab_pinned)
                perf().n_active_layer_upload_src_pinned.fetch_add(
                    1, std::memory_order_relaxed);
              else {
                perf().n_active_layer_upload_src_unpinned.fetch_add(
                    1, std::memory_order_relaxed);
                perf().mark_unpinned_touch(layer_idx, expert_id);
              }
              // 2026-08-28: multi-GPU routing decision -- ci
              // (this candidate's position within decode_
              // async_candidates) preserves ram_needed_ids'
              // own top-K priority order (the filter pass
              // that built decode_async_candidates only
              // ever skips entries, never reorders), so it's
              // a real, if not perfectly 1:1, priority
              // signal for the router. See route_active_
              // layer_gpu's own comment (dense_dispatch.hpp).
              // total_candidates = decode_async_candidates.
              // size() -- the REAL count for THIS transition,
              // not a fixed nominal 8 (which real hardware
              // showed made the tail unreachable, since this
              // list is typically much shorter than 8 on a
              // well-cached run).
              int routed_gpu =
                  active_layer_multi_gpu_enabled()
                      ? route_active_layer_gpu(
                            (int)ci, (int)decode_async_candidates.size())
                      : -1;
              if (routed_gpu >= 0) {
                perf()
                    .n_active_layer_routed_by_gpu
                        [routed_gpu < PerfCounters::PERF_MAX_GPU_SLOTS
                             ? routed_gpu
                             : 0]
                    .fetch_add(1, std::memory_order_relaxed);
              }
              // 2026-08-28: an UNPINNED source's inline call
              // secretly blocks this (decode) thread for
              // ~1.7ms (a driver-internal staged copy -- see
              // ActiveLayerUploadWorkers' own header comment
              // for the full evidence chain) -- route those
              // through the worker pool instead when it's
              // enabled. A pinned source's inline call is
              // already near-free, so it's left exactly as-is
              // regardless of the pool's availability.
              bool use_worker_pool = !blk->slab_pinned &&
                                     ram_mgr.upload_workers &&
                                     ram_mgr.upload_workers->enabled();
              if (use_worker_pool) {
                ExpertLocation *claimed =
                    ram_mgr.vram_cache->claim_active_layer_slot_for_worker(
                        layer_idx, expert_id, host_bytes, /*speculative=*/false,
                        routed_gpu);
                if (claimed && ram_mgr.upload_workers->try_submit(
                                   ActiveLayerUploadWorkItem{
                                       claimed, blk, host_data, host_bytes})) {
                  // Worker owns the RAM-source unpin (via
                  // ram_mark_pending_copy_and_unpin, once it
                  // genuinely dispatches) -- this thread's
                  // own ram_pin above must NOT be released
                  // here.
                  perf().n_decode_async_prewarm_issued.fetch_add(
                      1, std::memory_order_relaxed);
                } else {
                  // Either the claim itself failed (pool
                  // full) or try_submit rejected it (queue
                  // full) -- the latter needs an explicit
                  // abandon to release the claim's own
                  // hold; the former has nothing to
                  // abandon.
                  if (claimed)
                    ram_mgr.vram_cache->abandon_worker_claim(claimed);
                  ram_mgr.ram_unpin(*blk);
                  perf().n_decode_async_prewarm_no_slot.fetch_add(
                      1, std::memory_order_relaxed);
                }
              } else {
                ExpertLocation *v =
                    ram_mgr.vram_cache->upload_expert_active_async(
                        layer_idx, expert_id, host_data, host_bytes,
                        /*speculative=*/false, routed_gpu);
                if (v) {
                  if (engine.decode_async_prewarm_nonblocking) {
                    // 2026-08-28: no batch sync at all for
                    // this k -- hand the RAM-source
                    // protection off to pending_copy_event
                    // (self-healing, polled) right now
                    // instead of holding ram_temp_pin_count
                    // until a later blocking wait. The
                    // active-layer DESTINATION side needs no
                    // change: find_active_expert's own poll-
                    // before-trust check (gpu_topology.cpp)
                    // already treats "not landed yet" as a
                    // plain miss for ANY active-layer slot,
                    // not just speculative ones. See
                    // EngineConfig::decode_async_prewarm_
                    // nonblocking's own comment.
                    ram_mgr.ram_mark_pending_copy_and_unpin(*blk,
                                                            v->upload_event);
                  } else {
                    decode_async_victims.push_back(v);
                    decode_async_pinned_blocks.push_back(blk);
                  }
                  perf().n_decode_async_prewarm_issued.fetch_add(
                      1, std::memory_order_relaxed);
                } else {
                  // Pool full/alloc failed -- no async copy was
                  // actually issued, nothing to protect; this k
                  // falls through to dispatch_expert_k's
                  // existing synchronous fallback path.
                  ram_mgr.ram_unpin(*blk);
                  perf().n_decode_async_prewarm_no_slot.fetch_add(
                      1, std::memory_order_relaxed);
                }
              }
            }
            if (!decode_async_victims.empty()) {
              // Only ever populated when decode_async_prewarm_
              // nonblocking is false (see the branch above) --
              // the nonblocking path never pushes into this
              // vector at all, so this whole block is a clean
              // no-op under the new mode, not merely unused.
              ram_mgr.vram_cache->sync_active_layer_batch(decode_async_victims);
              for (auto *blk : decode_async_pinned_blocks)
                ram_mgr.ram_unpin(*blk);
              perf().n_decode_async_prewarm_synced.fetch_add(
                  decode_async_victims.size(), std::memory_order_relaxed);
              perf().n_decode_async_prewarm_chunks.fetch_add(
                  1, std::memory_order_relaxed);
            }
          }
        }

        // Dispatch whatever get_experts_batch (and the pre-warm pass
        // above, if enabled) just resolved. The shadowed experts
        // above already had their kernels issued inline, before the
        // blocking get_experts_batch call ran -- only the genuine
        // misses are dispatched here, after their data is ready. Any
        // expert pre-warmed above is now active-layer-resident with
        // upload_ready=true, so expert_ffn_resolve_gpu_loc's find_
        // active_expert call inside dispatch_expert_k will hit it
        // directly instead of paying for its own synchronous upload
        // -- everything else (not pre-warmed: feature off, active-
        // layer pool full, unsupported type, or no active-layer area
        // on this model) takes the original synchronous path exactly
        // as before.
        {
          // Same "Fifth round" investigation as shadow_precheck
          // above -- this is the OTHER (mutually exclusive)
          // half of every dispatch_expert_k call this token/layer:
          // shadow_precheck covers shadowed k's (dispatched
          // inline, before get_experts_batch); this covers the
          // genuine RAM-tier misses (dispatched after). Together
          // they should account for ~all of t_dispatch_expert_k_
          // total's own aggregate, letting phase3_loop's residual
          // be decomposed additively: shadow_precheck + ram_scan
          // (already reported, covers get_experts_batch in
          // between the two) + this + gpu_finish/accumulate
          // (Pass B, below) + shared_expert + layer_cleanup +
          // active_layer_prewarm_issue/sync should sum close to
          // phase3_loop's own total, with whatever's left being
          // the genuinely still-unexplained residual.
          PERF_SCOPE(t_ram_needed_dispatch_total);
          for (size_t i = 0; i < ram_needed_k.size(); ++i) {
            // A lookahead-only entry (-1) belongs to a later
            // token -- never dispatched by this one; it's now
            // sitting RAM-resident, ready for a cheap resolve
            // when its own token's turn comes up. (Its own pin
            // was already released right after get_experts_
            // batch returned -- see that call site's own
            // comment.)
            if (ram_needed_k[i] < 0)
              continue;
            dispatch_expert_k(ram_needed_k[i]);
            // 2026-08-30: matching release for get_experts_
            // batch's pin-at-resolution fix (ram_manager.hpp) --
            // this k's dispatch (CPU synchronous compute, or
            // GPU's synchronous upload_expert/upload_expert_
            // active at this S>=GPU_COLD_BATCH_THRESHOLD prefill
            // scale) has now fully consumed the RAM-tier bytes,
            // so it's safe to drop this reference-lifetime pin.
            // ram_needed_ptrs[i] is the exact slot get_experts_
            // batch pinned for this same i (identical index
            // space -- both arrays are built together above).
            ram_mgr.ram_unpin(*ram_needed_ptrs[i]);
          }
        }
      }
    }

    // --- Pass B, batched (2026-08-29): finish every GPU-issued expert
    // queued above, grouped by loc->gpu_id rather than synchronizing per
    // expert.  With device SwiGLU and moe_gpu_accum enabled (the normal
    // qwen4exp decode configuration), stage1 and the scatter accumulate are
    // both queued on the legacy stream and the accumulator's one final wait
    // per GPU confirms the whole chain.  A first wait remains only when a
    // host gate/up read is required; a second wait remains only for a legacy
    // per-expert host-output fallback.  This is the same cache-safety
    // invariant as the old two-round implementation, but it avoids waits
    // that no longer guard a host-visible dependency.
    //
    // gate_buf/up_buf/down_buf are SHARED host buffers (scratch's own,
    // one copy, not one per pending expert) -- stage1 is safe to call
    // back-to-back for every pending expert because each call fully
    // consumes gate_buf/up_buf (uploads them to that expert's OWN
    // per-slot d_prod) before returning, so the next stage1 call
    // overwriting them is never a race. Stage2+accumulate, by
    // contrast, MUST stay paired per-expert in the same iteration --
    // down_buf can only hold one expert's output at a time, so
    // accumulate() has to consume it before the next stage2 call
    // overwrites it. This mirrors, not changes, a constraint the
    // original single-function design already had implicitly (its own
    // per-expert loop already fully finished and accumulated one
    // expert before starting the next).
    if (!scratch.pending_gpu.empty()) {
      // With device SwiGLU there is no host read between the gate/up issue
      // and stage1: stage1 only queues more work on the same legacy stream.
      // Avoid the first batch wait in that case.  If the GPU activation
      // scratch cannot be prepared (or tracing forces the host activation
      // path), retain the old wait before stage1's blocking gate/up copies.
      const bool activation_requested =
          expert_gpu_activation() && !trace_gate("SILU-TRACE-GPU") &&
          !trace_gate("SWIGLU-TRACE-GPU") && !trace_gate("EXPERT-GPU") &&
          !trace_gate("INPUT-TRACE-GPU-PAIRED");
      bool all_device_activation = activation_requested;
      if (all_device_activation) {
        for (size_t pi = 0; pi < scratch.pending_gpu.size(); ++pi) {
          if (!scratch.gpu_pool[pi].ensure_bad_counter()) {
            all_device_activation = false;
            break;
          }
        }
      }

      bool synced[PerfCounters::PERF_MAX_GPU_SLOTS] = {};
      if (!all_device_activation) {
        PERF_START(gpu_batch_sync1);
        for (auto &p : scratch.pending_gpu) {
          int gid = (p.loc->gpu_id >= 0 &&
                     p.loc->gpu_id < PerfCounters::PERF_MAX_GPU_SLOTS)
                        ? p.loc->gpu_id
                        : 0;
          if (!synced[gid]) {
            cudaSetDevice(p.loc->gpu_id);
            cudaStreamSynchronize(0);
            synced[gid] = true;
            perf().n_gpu_batch_sync_calls.fetch_add(1,
                                                    std::memory_order_relaxed);
          }
        }
        PERF_END(gpu_batch_sync1, t_gpu_batch_sync);
      }

      // The decode route table has one row per selected expert.  Upload and
      // clear the accumulator once per GPU before stage1 starts, so each
      // expert can scatter its d_down directly into the final token result.
      if (decode_gpu_accum) {
        for (auto &p : scratch.pending_gpu) {
          const int gid = p.loc->gpu_id;
          if (gid < 0 || gid >= PerfCounters::PERF_MAX_GPU_SLOTS ||
              decode_accum_begun[gid])
            continue;
          decode_accum_begun[gid] = engine.gpu->expert_accum_begin(
              gid, ordered_routes ? topk : 1, D,
              decode_accum_tok.data(), decode_accum_w.data(), topk);
        }
      }

      for (size_t pi = 0; pi < scratch.pending_gpu.size(); ++pi) {
        PendingGpuExpert &p = scratch.pending_gpu[pi];
        GpuExpertScratch &gs = scratch.gpu_pool[pi];
        expert_ffn_gpu_finish_stage1(
            gs, p.loc, scratch.gate_buf.data(), scratch.up_buf.data(),
            p.exp->gate_proj.meta, p.exp->down_proj.meta, layer_idx,
            p.expert_id, 1, D, moe_inter, clamp_exp, ram_mgr.vram_cache);
      }

      // Queue the device scatter immediately after each expert's down kernel.
      // The default stream orders down -> scatter, so a second batch wait is
      // needed only for experts that still require the legacy host download.
      bool need_sync2[PerfCounters::PERF_MAX_GPU_SLOTS] = {};
      if (decode_gpu_accum) {
        for (size_t pi = 0; pi < scratch.pending_gpu.size(); ++pi) {
          PendingGpuExpert &p = scratch.pending_gpu[pi];
          const int gid = p.loc->gpu_id;
          if (gid >= 0 && gid < PerfCounters::PERF_MAX_GPU_SLOTS &&
              decode_accum_begun[gid]) {
            p.accumulated_on_device = engine.gpu->expert_accum_add(
                gid, scratch.gpu_pool[pi].d_down, p.k, 1, D,
                ordered_routes ? topk : 1);
            if (p.accumulated_on_device)
              perf().n_expert_gpu_calls.fetch_add(1, std::memory_order_relaxed);
          }
          if (!p.accumulated_on_device) {
            const int sync_gid =
                (gid >= 0 && gid < PerfCounters::PERF_MAX_GPU_SLOTS) ? gid : 0;
            need_sync2[sync_gid] = true;
          }
        }
      } else {
        for (auto &p : scratch.pending_gpu) {
          const int gid = (p.loc->gpu_id >= 0 &&
                           p.loc->gpu_id < PerfCounters::PERF_MAX_GPU_SLOTS)
                              ? p.loc->gpu_id
                              : 0;
          need_sync2[gid] = true;
        }
      }

      std::fill(std::begin(synced), std::end(synced), false);
      bool any_sync2 = false;
      for (bool needed : need_sync2)
        any_sync2 |= needed;
      if (any_sync2) {
        PERF_START(gpu_batch_sync2);
        for (auto &p : scratch.pending_gpu) {
          int gid = (p.loc->gpu_id >= 0 &&
                     p.loc->gpu_id < PerfCounters::PERF_MAX_GPU_SLOTS)
                        ? p.loc->gpu_id
                        : 0;
          if (need_sync2[gid] && !synced[gid]) {
            cudaSetDevice(p.loc->gpu_id);
            cudaStreamSynchronize(0);
            synced[gid] = true;
            perf().n_gpu_batch_sync_calls.fetch_add(1,
                                                    std::memory_order_relaxed);
          }
        }
        PERF_END(gpu_batch_sync2, t_gpu_batch_sync);
      }

      for (size_t pi = 0; pi < scratch.pending_gpu.size(); ++pi) {
        PendingGpuExpert &p = scratch.pending_gpu[pi];
        GpuExpertScratch &gs = scratch.gpu_pool[pi];
        if (p.accumulated_on_device)
          continue;
        // This sync round just confirmed EVERY kernel that reads
        // this slab (gate, up, AND down -- the last of the three)
        // has actually finished executing, not just that launches
        // completed -- so it's safe to release the temporary hold
        // now, exactly the same invariant the original single-
        // function design relied on, just satisfied by the shared
        // batch sync above instead of a per-call one.
        expert_ffn_gpu_finish_stage2(
            p.loc, scratch.down_buf.data(), layer_idx, p.expert_id, 1, D,
            ram_mgr.vram_cache, p.needs_release, p.loc_from_active_layer, gs);
        // Matches Pass A's increment -- release the RAM-tier protection
        // now that expert_ffn_gpu_finish_stage2's own release_hold()
        // call (which uses p.expert_id, captured at Pass A issue time
        // -- see PendingGpuExpert's own comment for why NOT
        // p.exp->expert_id) has already happened. Must be the LAST
        // thing done with p.exp's identity in this iteration; this
        // slot becomes a valid victim-selection candidate again
        // immediately after this decrement. Goes through
        // RAMSafetyManager::ram_unpin -- see Pass A's matching
        // ram_pin call for why this is no longer a direct field
        // decrement.
        ram_mgr.ram_unpin(*p.exp);
        accumulate(p.k, p.exp, scratch.down_buf.data());
      }

      // One download per active GPU: [topk,D] disjoint rows for DeepSeek's
      // ordered reduction, [D] summed output otherwise. A GPU with a fallback
      // was already synchronized above, so avoid paying a duplicate wait.
      for (int gid = 0; gid < PerfCounters::PERF_MAX_GPU_SLOTS; ++gid) {
        if (!decode_accum_begun[gid])
          continue;
        const bool already_synced = need_sync2[gid] && synced[gid];
        if (!engine.gpu->expert_accum_end(
                gid, ordered_routes ? ordered_route_outputs.data() : token_output,
                ordered_routes ? topk : 1, D, already_synced))
          perf().n_moe_gpu_accum_fallback.fetch_add(1,
                                                    std::memory_order_relaxed);
      }
      for (auto &p : scratch.pending_gpu) {
        if (!p.accumulated_on_device)
          continue;
        // The accumulator end above has waited for down and scatter, so the
        // temporary VRAM hold can now be released without reopening the
        // eviction-during-compute race.  The RAM pin has the same lifetime.
        if (p.needs_release && ram_mgr.vram_cache)
          ram_mgr.vram_cache->release_hold(layer_idx, p.expert_id,
                                           p.loc->device_ptr,
                                           p.loc_from_active_layer);
        ram_mgr.ram_unpin(*p.exp);
      }
    }

    if (ordered_routes) {
      collecting_ordered = false;
      for (int k = 0; k < topk; ++k)
        accumulate(k, scratch.expert_ptrs[k],
                    ordered_route_outputs.data() + (size_t)k * D);
    }

    bool cpu_offload_event_fenced_handled = false;
    bool cpu_offload_event_fenced_finished = false;
    if (cpu_offload_worker_started && cpu_offload_runner &&
        cpu_offload_event_fence_active && cpu_offload_ring) {
      // The worker publishes the copy/event/fence after its CPU graph
      // finishes.  Waiting here is only a queue-publication handshake; the
      // PCIe copy itself is consumed by the already-fenced GGML stream.
      double fence_wait_ms = 0.0;
      bool cpu_ok = false;
      const bool fence_queued = cpu_offload_ring->wait_fence_enqueued(
          cpu_offload_ring_slot, fence_wait_ms, cpu_ok);
      moe_pipeline_internal::add_decode_qwen_cpu_offload_fence_wait_ms(
          fence_wait_ms);
      cpu_offload_event_fenced_finished = true;
      if (fence_queued) {
        moe_pipeline_internal::add_decode_qwen_cpu_offload_fenced_call();
        moe_pipeline_internal::add_decode_qwen_cpu_offload_worker_success();
        const void *hot_combined = nullptr;
        const bool hot_added = run_qwen4exp_shared_device(
            engine, layer_idx, S, D, moe_inter,
            engine.qwen4exp_device_hc_ffn_input, nullptr, c, &hot_combined,
            token_output, false, true);
        const void *combined = nullptr;
        auto &shared_runner =
            GgmlQwenSharedRunner::instance_for_layer(layer_idx);
        const bool device_added =
            hot_added && hot_combined &&
            shared_runner.run_device_routed_add(
                layer_idx, hot_combined,
                cpu_offload_ring->device_output(cpu_offload_ring_slot),
                &combined);
        if (device_added && combined) {
          const bool consumed_marked = cpu_offload_ring->mark_consumed(
              cpu_offload_ring_slot, shared_runner.backend_stream());
          if (!consumed_marked)
            moe_pipeline_internal::add_decode_qwen_cpu_offload_fence_failure();
          scratch.moe_device_output = combined;
          scratch.moe_shared_device_output = true;
          scratch.moe_device_output_ready = true;
          cpu_offload_event_fenced_handled = true;
        } else {
          // The CPU result is valid even if the optional device add cannot be
          // built.  Retain it on the host and let the established host-routed
          // add below publish a complete result.
          moe_pipeline_internal::add_decode_qwen_cpu_offload_fence_failure();
          for (int d = 0; d < D; ++d)
            token_output[d] +=
                cpu_offload_ring->host_output(cpu_offload_ring_slot)[d];
          cpu_offload_ring->mark_consumed(
              cpu_offload_ring_slot, shared_runner.backend_stream());
        }
      } else {
        moe_pipeline_internal::add_decode_qwen_cpu_offload_fence_failure();
        moe_pipeline_internal::add_decode_qwen_cpu_offload_worker_failure();
        // The callback did not publish a usable device result.  Recompute the
        // cold subset synchronously into the ordinary pinned scratch buffer;
        // the final host-routed add remains the correctness fallback.
        if (scratch.qwen_cpu_moe_out.resize((size_t)D,
                                            "qwen_cpu_offload_result")) {
          const bool fallback_ok = cpu_offload_runner->run_strided(
              scratch.normed.data() + s * D,
              scratch.qwen_cpu_moe_out.data(), cpu_offload_ram_layer.meta,
              cpu_offload_ram_layer.gate_plane,
              cpu_offload_ram_layer.up_plane,
              cpu_offload_ram_layer.down_plane,
              scratch.qwen_cpu_offload_cold_ids.data(), n_experts,
              scratch.qwen_cpu_offload_cold_weights.data(), D, moe_inter,
              cpu_offload_cold_count, cpu_offload_ram_layer.expert_stride);
          if (fallback_ok)
            for (int d = 0; d < D; ++d)
              token_output[d] += scratch.qwen_cpu_moe_out[(size_t)d];
        }
      }
      moe_pipeline_internal::add_decode_qwen_cpu_offload_total_ms(
          std::chrono::duration<double, std::milli>(
              std::chrono::steady_clock::now() - cpu_offload_start)
              .count());
    }

    if (cpu_offload_worker_started && cpu_offload_runner &&
        !cpu_offload_event_fenced_finished) {
      const auto cpu_wait_start = std::chrono::steady_clock::now();
      const bool cpu_ok = cpu_offload_runner->wait_strided_async();
      const double wait_ms = std::chrono::duration<double, std::milli>(
                                 std::chrono::steady_clock::now() -
                                 cpu_wait_start)
                                 .count();
      moe_pipeline_internal::add_decode_qwen_cpu_offload_wait_ms(wait_ms);
      if (cpu_ok)
        moe_pipeline_internal::add_decode_qwen_cpu_offload_worker_success();
      else
        moe_pipeline_internal::add_decode_qwen_cpu_offload_worker_failure();
      if (cpu_ok) {
        for (int d = 0; d < D; ++d)
          token_output[d] += scratch.qwen_cpu_moe_out[(size_t)d];
      } else {
        // The worker owns no model state beyond the persistent runner. If it
        // fails, recompute the same cold subset synchronously before the
        // result is published; correctness takes precedence over the A/B
        // timing path.
        const bool fallback_ok = cpu_offload_runner->run_strided(
            scratch.normed.data() + s * D, scratch.qwen_cpu_moe_out.data(),
            cpu_offload_ram_layer.meta, cpu_offload_ram_layer.gate_plane,
            cpu_offload_ram_layer.up_plane, cpu_offload_ram_layer.down_plane,
            scratch.qwen_cpu_offload_cold_ids.data(), n_experts,
            scratch.qwen_cpu_offload_cold_weights.data(), D, moe_inter,
            cpu_offload_cold_count, cpu_offload_ram_layer.expert_stride);
        if (fallback_ok) {
          for (int d = 0; d < D; ++d)
            token_output[d] += scratch.qwen_cpu_moe_out[(size_t)d];
        } else {
          LOG_STREAM_ERROR()
              << "[CPU-OFFLOAD] Layer " << layer_idx
              << " cold worker and synchronous fallback both failed\n";
        }
      }
      // Milestone A total is measured from worker submission through
      // completion. The actual GPU-side handoff is submitted by the shared
      // graph below and remains asynchronous.
      moe_pipeline_internal::add_decode_qwen_cpu_offload_total_ms(
          std::chrono::duration<double, std::milli>(
              std::chrono::steady_clock::now() - cpu_offload_start)
              .count());
    }

    check_finite("token_output (post routed experts, pre-shared)", layer_idx,
                 -1, token_output, (size_t)D);

    // Option B's shared-only graph was submitted before the routed work, so
    // the CPU cold tail could execute while CUDA processed the always-on
    // shared FFN.  Add the completed routed result to that graph output now;
    // the normal execution-layer HC post-combine then consumes one device
    // result instead of running the host shared-expert implementation again.
    if (option_b_shared_started && !cpu_offload_event_fenced_handled) {
      const void *combined = nullptr;
      const bool added = run_qwen4exp_shared_device(
          engine, layer_idx, S, D, moe_inter, nullptr, nullptr, c, &combined,
          token_output, false, true);
      if (added && combined) {
        scratch.moe_device_output = combined;
        scratch.moe_shared_device_output = true;
        scratch.moe_device_output_ready = true;
      } else {
        // The shared-only graph has already been submitted.  Preserve
        // numerical correctness if the add graph cannot be built by waiting
        // for and downloading that result, then use the established host
        // accumulation path for the routed result.
        auto &shared_runner =
            GgmlQwenSharedRunner::instance_for_layer(layer_idx);
        if (!option_b_shared_only_output ||
            !shared_runner.copy_output_to_host(option_b_shared_only_output,
                                               scratch.down_buf.data())) {
          LOG_STREAM_ERROR()
              << "[OPTION-B] Layer " << layer_idx
              << " shared add failed and shared-only output recovery failed;"
                 " falling back to host shared FFN\n";
          run_qwen4exp_shared_host(engine, layer_idx, S, D, moe_inter,
                                   token_hidden, token_output, scratch, c);
        } else {
          for (int i = 0; i < D; ++i)
            token_output[i] += scratch.down_buf[i];
        }
      }
    }

    // Shared expert (always-active) — GLM-5.2/DeepSeek-V3/Hunyuan MoE
    // layers can have n_expert_shared always-on experts whose combined
    // FFN (intermediate width n_expert_shared * moe_inter — NOT just
    // moe_inter, confirmed against deepseek2.cpp's shexp tensor shapes)
    // is added to every token's routed-expert output. Unscaled: only the
    // routed weights get expert_weights_scale (see build_moe_ffn — w_scale
    // is applied to the gathered top-k weights, never to the shared path).
    std::string sp = "blk." + std::to_string(layer_idx) + ".";
    if (!option_b_shared_started && c.n_expert_shared > 0 &&
        engine.resident_layers.count(sp + "ffn_gate_shexp.weight")) {
      PERF_SCOPE(t_shared_expert_total);
      int shared_inter = moe_inter * c.n_expert_shared;
      dense_proj(engine, sp + "ffn_gate_shexp.weight", token_hidden,
                 scratch.gate_buf.data(), 1, D, shared_inter);
      dense_proj(engine, sp + "ffn_up_shexp.weight", token_hidden,
                 scratch.up_buf.data(), 1, D, shared_inter);
      if (swiglu_clamp_fused()) {
        // GLM5Next: ggml_swiglu_clamp semantics (fused min/clamp/silu/mul).
        swiglu_clamp_fused_inplace(scratch.gate_buf.data(),
                                   scratch.up_buf.data(), shared_inter,
                                   clamp_shexp);
      } else {
        silu_inplace(scratch.gate_buf.data(), shared_inter);
        elem_mul(scratch.gate_buf.data(), scratch.gate_buf.data(),
                 scratch.up_buf.data(), shared_inter);
        swiglu_clamp_inplace(scratch.gate_buf.data(), shared_inter,
                             clamp_shexp);
      }
      dense_proj(engine, sp + "ffn_down_shexp.weight", scratch.gate_buf.data(),
                 scratch.down_buf.data(), 1, shared_inter, D);
      // Previously completely uninstrumented -- the shared-expert path
      // has no equivalent of expert_ffn's check_finite calls, so a bug
      // here (e.g. a dense-tensor GPU dispatch through the same fused
      // K-quant kernels that turned out to have the Q3_K struct-layout
      // bug) would silently poison every token's output with no log
      // ever pointing at it.
      check_finite("shexp down_buf (pre-accumulate)", layer_idx, -1,
                   scratch.down_buf.data(), (size_t)D);
      // Qwen4Exp (Qwen3.8-Flash-Next): the shared expert has its OWN
      // per-token sigmoid gate (ffn_gate_inp_shexp, a [D] vector -> one
      // scalar per token -- qwen4exp.cpp's build_layer_ffn). Gated on
      // tensor PRESENCE (the established "optional small tensor" idiom),
      // not family, so GLM/DeepSeek's gateless shexp is untouched.
      float shexp_gate = 1.0f;
      if (engine.resident_layers.count(sp + "ffn_gate_inp_shexp.weight")) {
        float g = 0.0f;
        dense_proj(engine, sp + "ffn_gate_inp_shexp.weight", token_hidden, &g,
                   1, D, 1);
        shexp_gate = 1.0f / (1.0f + std::exp(-g));
      }
      for (int i = 0; i < D; ++i)
        token_output[i] += shexp_gate * scratch.down_buf[i];
      check_finite("token_output (post shared expert)", layer_idx, -1,
                   token_output, (size_t)D);
    }
  }

}
