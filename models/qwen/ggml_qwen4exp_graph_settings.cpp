// Qwen4Exp full graph: Environment-driven feature gates, decode capacity,
// and scheduler policy.
//
// Split out of the former single-file ggml_qwen4exp_graph.cpp; shared
// declarations live in ggml_qwen4exp_graph_internal.hpp.

#include "ggml_qwen4exp_graph_internal.hpp"
#include "graph_settings_global.hpp"
#include <sched.h>
#include <vector>
#include <string>
#include <algorithm>

#if defined(ZEROMOE_USE_GGML_CUDA)

using namespace qwen4exp_graph;

namespace qwen4exp_graph {

bool prefill_state_handoff_enabled() {
  return g_config_prefill_state_handoff;
}

bool prefill_device_state_inputs_enabled(const StreamingEngine &engine) {
  return zeromoe::graphsettings::active_settings().tun.prefill_device_state.value != 0 &&
         prefill_state_handoff_enabled() &&
         !engine.prompt_cache && !engine.mtp_enabled &&
         engine.qwen4exp_gpu_stage_layers.empty();
}

bool qwen4exp_prefill_op_offload_enabled() {
  const auto &settings = zeromoe::graphsettings::settings_for(
      zeromoe::graphsettings::Family::Qwen);
  return settings.tun.prefill_op_offload.value != 0;
}

int env_device_expert_layers() {
  const auto &s = zeromoe::graphsettings::settings_for(zeromoe::graphsettings::Family::Qwen);
  long n = std::clamp(s.tun.device_expert_layers.value, 0L, 48L);
  // 0 is the explicit "no whole-layer block" sentinel (the global-pool recipe
  // uses it).  Any enabled block needs at least two complete layers: one layer
  // cannot amortize the per-layer device arena and made the 5060 Ti decline the
  // dense-arena path (docs/handoff-2026-10-03-option-b-global-pool.md).
  if (n == 1) {
    static bool warned = false;
    if (!warned) {
      LOG_WARN("[GRAPH-SETTINGS] device_expert_layers=1 raised to 2; "
               "use 0 to disable the whole-layer block\n");
      warned = true;
    }
    n = 2;
  }
  return static_cast<int>(n);
}

// Placement side of device_expert_layers.  false (default) keeps ZeroMoE's
// historical leading-N block; true matches llama.cpp's `-ncmoe N` convention
// and puts the same number of complete layers at the END of the network.
// It has no effect when device_expert_layers is 0.
// See FullGraph::device_expert_trailing's own comment.
bool env_device_expert_trailing() {
  return zeromoe::graphsettings::settings_for(zeromoe::graphsettings::Family::Qwen)
      .tun.device_expert_trailing.value != 0;
}

// Detailed graph timing follows the common DEBUG verbosity threshold.
bool env_layer_timing_enabled() {
  return Logger::instance().would_log(LogLevel::DEBUG);
}

// Phase 4 host-node integration is deliberately a separate opt-in from the
// ordinary full graph.  It is only safe when the graph builder replaces the
// configured CPU MoE nodes with the fused CPY markers; an unset variable must
// retain the proven scheduler-split path byte-for-byte.
bool env_qwen4exp_host_node_enabled() {
  return zeromoe::graphsettings::active_settings().tun.host_node.value != 0;
}

// One-shot diagnostic (2026-09-19): dumps EVERY node the scheduler visits
// (both CPU- and CUDA-backend, which layer_timing_eval_callback's normal
// path skips entirely) for the FIRST decode token only, so the real
// node/backend sequence around a CPU-dispatched MoE layer's boundary is
// visible directly instead of guessed at from the undifferentiated
// layer_timing_gap_ms_total bucket. See memory:
// qwen4exp-phase2-gpu-branch-slower-than-cpu.md's own "not yet diagnosed"
// note and the live investigation this answers.
bool env_layer_timing_verbose() {
  return Logger::instance().would_log(LogLevel::DEBUG);
}

// A storage miss must not hold decode indefinitely. The DDR/CPU suffix is an
// exact fallback while the worker continues its read; the worker retires the
// timed-out generation when it finishes so repeated slow reads cannot strand
// all bounded service slots. The staged path is opt-in, so this deadline does
// not affect the production early-ready profile.
std::chrono::milliseconds env_storage_wait_timeout() {
  constexpr long kMaximumMs = 30000;
  const long ms = zeromoe::graphsettings::active_settings().tun.storage_timeout_ms.value;
  return std::chrono::milliseconds(std::clamp(ms, 0L, kMaximumMs));
}

bool env_adaptive_completion_policy_enabled() {
  return zeromoe::graphsettings::active_settings().tun.adaptive_policy.value != 0;
}

int env_decode_kv_capacity(int max_t, int required) {
  const char *v = zeromoe::graphcfg::env("ZEROMOE_QWEN4EXP_FULL_GRAPH_DECODE_KV_CAPACITY");
  int requested = 2048;
  if (v && v[0]) {
    char *end = nullptr;
    const long parsed = std::strtol(v, &end, 10);
    if (end != v && *end == '\0' && parsed > 0)
      requested = static_cast<int>(std::min<long>(parsed, max_t));
  }
  return std::min(max_t, std::max(required, requested));
}

// Host-node mode runs the CPU MoE inside a CUDA host callback, so a CUDA driver thread
// leads the OpenMP team. Left to the scheduler, the team's workers land on arbitrary logical
// CPUs and the identical graph computes ~33% slower than it does from the main thread
// (standalone repro, tests/option_b_split_reverify_howto.md section 19). Pinning thread i to
// the i-th allowed CPU recovers most of it: host-node 22.75 -> 25.37 tok/s. Applied only in
// host-node mode (binding hurts the plain scheduler-split modes).
void pin_host_node_threads(ggml_threadpool_params &p, int n_threads) {
  if (!env_qwen4exp_host_node_enabled())
    return;
  cpu_set_t allowed;
  CPU_ZERO(&allowed);
  if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0)
    return;
  bool mask[GGML_MAX_N_THREADS] = {};
  int got = 0;
  for (int cpu = 0; cpu < CPU_SETSIZE && cpu < GGML_MAX_N_THREADS &&
                    got < n_threads;
       ++cpu) {
    if (CPU_ISSET(cpu, &allowed)) {
      mask[cpu] = true;
      ++got;
    }
  }
  if (got < n_threads)
    return; // not enough distinct CPUs: leave placement to the scheduler
  std::memcpy(p.cpumask, mask, sizeof(mask));
  p.strict_cpu = true;
  LOG_INFO("[GGML-QWEN-FULL] host-node mode: pinning %d CPU worker threads "
           "to distinct logical CPUs\n",
           n_threads);
}

// REVERTED (2026-09-17): a prior commit defaulted this to false, reasoning
// "the reference llama.cpp scheduler uses pipeline_parallel=false for this
// graph... so the event/four-copy mode adds bookkeeping without useful
// overlap." That reasoning does not hold up against llama.cpp's own source
// (llama-context.cpp): `pipeline_parallel` there is an UNRELATED feature --
// it requires `model.n_devices() > 1` (multi-GPU tensor split) AND
// `!model.has_tensor_overrides()`, and llama.cpp's own `-ncmoe` flag sets tensor
// overrides for exactly the CPU-offloaded MoE tensors this graph mirrors.
// So llama.cpp's `pipeline_parallel=false` in every comparison this project has
// ever run is not a deliberate choice about CPU/GPU MoE handoff at all --
// it is llama.cpp's own multi-GPU pipelining feature being auto-disabled by
// `-ncmoe`, for reasons having nothing to do with this graph's performance.
// Confirmed via direct measurement this graph's OWN parallel=false default
// was a real regression (11.76 vs 12.20-12.86 tok/s, N=22 mixed decode) --
// this is the fix from qwen4exp-mixed-gpu-decode-gap.md
// (ggml_backend_sched_new's parallel=false leaves CUDA sync events NULL,
// forcing a blocking stream-sync every CPU/GPU transition instead of a
// llama.cpp gives the CPU scheduler slot the CUDA device's host buffer type. That
// makes graph-owned CPU activations pinned host allocations, so the many
// CPU<->CUDA boundary copies do not fall through the pageable-host staging
// path. This is the fixed full-graph scheduler policy.
bool graph_cuda_host_buffers() {
  return true;
}

bool graph_match_llama_cpp_topology(const FullGraph &graph) {
  // Match the reference scheduler topology when experts span host and device;
  // use the native all-device graph when every expert layer is resident.
  return device_expert_resident_layer_count(graph) < graph.n_layers;
}

// Emit one-time tensor metadata for CPU MoE comparisons.  The same
// LLAMA_CPP_GGML_PROFILE_LAYOUT switch is consumed by llama.cpp's GGML scheduler so the
// two implementations can be compared without changing their execution.
bool graph_layout_debug() {
  return zeromoe::graphcfg::flag_on("LLAMA_CPP_GGML_PROFILE_LAYOUT");
}

bool has_ple_layer(const ArchitectureConfig &c) {
  return !c.ple_layers.empty() && c.ple_n_heads > 0 &&
         c.ple_n_embd_per_layer > 0;
}

bool type_ok(GGUFType type) {
  return static_cast<uint32_t>(type) < static_cast<uint32_t>(GGML_TYPE_COUNT);
}

} // namespace qwen4exp_graph

#endif // ZEROMOE_USE_GGML_CUDA
