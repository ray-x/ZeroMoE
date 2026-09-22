#include "active_layer_upload_workers.hpp"
#include "async_prefetcher.hpp"
#include "cpu_expert_executor.hpp"
#include "gpu_expert_executor.hpp"
#include "dense_dispatch.hpp"
#include "engine.hpp"
#include "expert_activation.hpp"
#include "ggml_deepseek_v4_graph.hpp"
#include "ggml_gdn_proj_backend.hpp"
#include "ggml_glm5next_graph.hpp"
#include "ggml_hc_backend.hpp"
#include "ggml_moe_backend.hpp"
#include "ggml_qwen4exp_graph.hpp"
#include "kernels_cpu.hpp"
#include "kquant_dequant.hpp"
#include "logger.hpp"
#include "mtp_speculative.hpp"
#include "moe_pipeline.hpp"
#include "moe_router.hpp"
#include "perf_timer.hpp"
#include "prefill_expert_stream.hpp"
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

#include "batched_dispatch_pipeline.hpp"
#include "moe_pipeline_internal.hpp"
#include "moe_pipeline_orchestrator.hpp"
#include "moe_router_service.hpp"
#include "prefill_warmup_manager.hpp"
#include "token_dispatch_pipeline.hpp"

namespace {
double g_diag_decode_qwen_cpu_graph_ms = 0.0;
double g_diag_decode_qwen_shared_submit_ms = 0.0;
double g_diag_decode_qwen_cpu_offload_wait_ms = 0.0;
double g_diag_decode_qwen_cpu_offload_total_ms = 0.0;
double g_diag_decode_qwen_cpu_offload_fence_wait_ms = 0.0;
double g_diag_decode_qwen_cpu_offload_h2d_ms = 0.0;
double g_diag_decode_qwen_cpu_offload_ring_reuse_wait_ms = 0.0;
uint64_t g_diag_decode_qwen_cpu_offload_cold_slots = 0;
uint64_t g_diag_decode_qwen_cpu_offload_worker_calls = 0;
uint64_t g_diag_decode_qwen_cpu_offload_worker_successes = 0;
uint64_t g_diag_decode_qwen_cpu_offload_worker_failures = 0;
uint64_t g_diag_decode_qwen_cpu_offload_fenced_calls = 0;
uint64_t g_diag_decode_qwen_cpu_offload_fence_failures = 0;
}

namespace moe_pipeline_internal {
void add_decode_qwen_cpu_graph_ms(double elapsed_ms) {
  g_diag_decode_qwen_cpu_graph_ms += elapsed_ms;
}
void add_decode_qwen_shared_submit_ms(double elapsed_ms) {
  g_diag_decode_qwen_shared_submit_ms += elapsed_ms;
}
void add_decode_qwen_cpu_offload_wait_ms(double elapsed_ms) {
  g_diag_decode_qwen_cpu_offload_wait_ms += elapsed_ms;
}
void add_decode_qwen_cpu_offload_total_ms(double elapsed_ms) {
  g_diag_decode_qwen_cpu_offload_total_ms += elapsed_ms;
}
void add_decode_qwen_cpu_offload_fence_wait_ms(double elapsed_ms) {
  g_diag_decode_qwen_cpu_offload_fence_wait_ms += elapsed_ms;
}
void add_decode_qwen_cpu_offload_h2d_ms(double elapsed_ms) {
  g_diag_decode_qwen_cpu_offload_h2d_ms += elapsed_ms;
}
void add_decode_qwen_cpu_offload_ring_reuse_wait_ms(double elapsed_ms) {
  g_diag_decode_qwen_cpu_offload_ring_reuse_wait_ms += elapsed_ms;
}
void add_decode_qwen_cpu_offload_cold_slots(uint64_t count) {
  g_diag_decode_qwen_cpu_offload_cold_slots += count;
}
void add_decode_qwen_cpu_offload_worker_call() {
  ++g_diag_decode_qwen_cpu_offload_worker_calls;
}
void add_decode_qwen_cpu_offload_worker_success() {
  ++g_diag_decode_qwen_cpu_offload_worker_successes;
}
void add_decode_qwen_cpu_offload_worker_failure() {
  ++g_diag_decode_qwen_cpu_offload_worker_failures;
}
void add_decode_qwen_cpu_offload_fenced_call() {
  ++g_diag_decode_qwen_cpu_offload_fenced_calls;
}
void add_decode_qwen_cpu_offload_fence_failure() {
  ++g_diag_decode_qwen_cpu_offload_fence_failures;
}
} // namespace moe_pipeline_internal

void moe_pipeline_detail::reset_diagnostics() {
  g_diag_decode_qwen_cpu_graph_ms = 0.0;
  g_diag_decode_qwen_shared_submit_ms = 0.0;
  g_diag_decode_qwen_cpu_offload_wait_ms = 0.0;
  g_diag_decode_qwen_cpu_offload_total_ms = 0.0;
  g_diag_decode_qwen_cpu_offload_fence_wait_ms = 0.0;
  g_diag_decode_qwen_cpu_offload_h2d_ms = 0.0;
  g_diag_decode_qwen_cpu_offload_ring_reuse_wait_ms = 0.0;
  g_diag_decode_qwen_cpu_offload_cold_slots = 0;
  g_diag_decode_qwen_cpu_offload_worker_calls = 0;
  g_diag_decode_qwen_cpu_offload_worker_successes = 0;
  g_diag_decode_qwen_cpu_offload_worker_failures = 0;
  g_diag_decode_qwen_cpu_offload_fenced_calls = 0;
  g_diag_decode_qwen_cpu_offload_fence_failures = 0;
}

double diag_get_decode_qwen_cpu_graph_ms() {
  return g_diag_decode_qwen_cpu_graph_ms;
}

double diag_get_decode_qwen_shared_submit_ms() {
  return g_diag_decode_qwen_shared_submit_ms;
}

double diag_get_decode_qwen_cpu_offload_wait_ms() {
  return g_diag_decode_qwen_cpu_offload_wait_ms;
}

double diag_get_decode_qwen_cpu_offload_total_ms() {
  return g_diag_decode_qwen_cpu_offload_total_ms;
}
double diag_get_decode_qwen_cpu_offload_fence_wait_ms() {
  return g_diag_decode_qwen_cpu_offload_fence_wait_ms;
}
double diag_get_decode_qwen_cpu_offload_h2d_ms() {
  return g_diag_decode_qwen_cpu_offload_h2d_ms;
}
double diag_get_decode_qwen_cpu_offload_ring_reuse_wait_ms() {
  return g_diag_decode_qwen_cpu_offload_ring_reuse_wait_ms;
}
uint64_t diag_get_decode_qwen_cpu_offload_cold_slots() {
  return g_diag_decode_qwen_cpu_offload_cold_slots;
}
uint64_t diag_get_decode_qwen_cpu_offload_worker_calls() {
  return g_diag_decode_qwen_cpu_offload_worker_calls;
}
uint64_t diag_get_decode_qwen_cpu_offload_worker_successes() {
  return g_diag_decode_qwen_cpu_offload_worker_successes;
}
uint64_t diag_get_decode_qwen_cpu_offload_worker_failures() {
  return g_diag_decode_qwen_cpu_offload_worker_failures;
}
uint64_t diag_get_decode_qwen_cpu_offload_fenced_calls() {
  return g_diag_decode_qwen_cpu_offload_fenced_calls;
}
uint64_t diag_get_decode_qwen_cpu_offload_fence_failures() {
  return g_diag_decode_qwen_cpu_offload_fence_failures;
}

// Per-layer decode timing is intentionally opt-in.  It updates several
// counters once per layer call, so leaving it enabled in a production decode
// would measure the instrumentation itself (and can materially lower the
// single-token throughput).  Enable it only for a diagnostic run with
// MOESTREAM_QWEN_DECODE_LAYER_TIMING=1.
static bool qwen4exp_decode_layer_timing_enabled() {
  static const bool enabled = [] {
    const char *e = std::getenv("MOESTREAM_QWEN_DECODE_LAYER_TIMING");
    return e && e[0] == '1';
  }();
  return enabled;
}
// Decode-only, per-layer wall timer.  The existing aggregate counters tell
// us that the request is slow, but they cannot answer which layers are worth
// moving to CPU: a CPU-affinity layer includes a different cache/transfer
// path than a GPU layer.  Keep this diagnostic local to Qwen4Exp so no other
// model family changes its accounting or behavior.
struct QwenDecodeLayerTimer {
  int layer = -1;
  bool cpu_path = false;
  bool active = false;
  std::chrono::high_resolution_clock::time_point start;

  QwenDecodeLayerTimer(int layer_idx, bool is_cpu, bool enabled)
      : layer(layer_idx), cpu_path(is_cpu), active(enabled) {
    if (active)
      start = std::chrono::high_resolution_clock::now();
  }

  ~QwenDecodeLayerTimer() {
    if (!active || layer < 0 ||
        layer >= PerfCounters::kMaxTrackedLayersForMiss)
      return;
    const auto end = std::chrono::high_resolution_clock::now();
    const double elapsed =
        std::chrono::duration<double>(end - start).count();
    auto &p = perf();
    p.n_qwen_decode_layer[layer].fetch_add(1, std::memory_order_relaxed);
    PerfCounters::atomic_add(p.t_qwen_decode_layer[layer], elapsed);
    PerfCounters::atomic_add(
        cpu_path ? p.t_qwen_decode_cpu_layer[layer]
                 : p.t_qwen_decode_gpu_layer[layer],
        elapsed);
  }
};

void MoePipelineOrchestrator::forward(
    StreamingEngine &engine, RAMSafetyManager &ram_mgr,
    LockFreePrefetcher &async_io_queue, int layer_idx,
    const float *hidden_states, int S, float *output, LayerScratch &scratch,
    const int *token_ids) {
  scratch.moe_device_output = nullptr;
  scratch.moe_device_runner = nullptr;
  scratch.moe_shared_device_output = false;
  scratch.moe_device_output_ready = false;
  auto &c = engine.model_file.architecture_config;
  const int D = c.hidden;
  const int topk = c.topk;
  const int n_experts = c.n_experts;
  const int moe_inter = c.moe_inter;

  const bool qwen_layer_timing =
      S == 1 && engine.family_ops.family == ModelFamily::Qwen4Exp &&
      qwen4exp_decode_layer_timing_enabled();
  QwenDecodeLayerTimer qwen_decode_layer_timer(
      layer_idx, qwen_layer_timing && is_cpu_affinity_layer(layer_idx),
      qwen_layer_timing);

  if (n_experts <= 0 || moe_inter <= 0) {
    std::fill(output, output + S * D, 0.0f);
    return;
  }

  const float clamp_exp =
      layer_idx < static_cast<int>(c.swiglu_clamp_exp.size())
          ? c.swiglu_clamp_exp[layer_idx]
          : 0.0f;
  const float clamp_shexp =
      layer_idx < static_cast<int>(c.swiglu_clamp_shexp.size())
          ? c.swiglu_clamp_shexp[layer_idx]
          : 0.0f;
  MoePipelineContext context{
      engine, ram_mgr, async_io_queue, layer_idx, hidden_states, S, output,
      scratch, token_ids, c, D, topk, n_experts, moe_inter, clamp_exp,
      clamp_shexp};

  if (MoeRouterService::route(context))
    return;
  PrefillWarmupManager::prepare(context);
  const BatchedDispatchResult result = BatchedDispatchPipeline::run(context);
  if (result == BatchedDispatchResult::Abort)
    return;
  if (result == BatchedDispatchResult::Continue)
    TokenDispatchPipeline::run(context);
  PrefillWarmupManager::finish(context);
}

void moe_forward_gguf(
    StreamingEngine &engine, RAMSafetyManager &ram_mgr,
    LockFreePrefetcher &async_io_queue, int layer_idx,
    const float *hidden_states, int S, float *output, LayerScratch &scratch,
    const int *token_ids) {
  MoePipelineOrchestrator::forward(engine, ram_mgr, async_io_queue, layer_idx,
                                   hidden_states, S, output, scratch,
                                   token_ids);
}
