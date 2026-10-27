// Qwen4Exp full graph: Environment-driven settings of the full graph: feature gates, decode
// capacity, CPU thread / scheduler policy, partial-layer knobs.
//
// Split out of the former single-file ggml_qwen4exp_graph.cpp; shared
// declarations live in ggml_qwen4exp_graph_internal.hpp.

#include "ggml_qwen4exp_graph_internal.hpp"
#include "graph_settings_global.hpp"

#if defined(ZEROMOE_USE_GGML_CUDA)

using namespace qwen4exp_graph;

namespace qwen4exp_graph {

bool env_prefill_state_handoff_enabled() {
  const char *value =
      zeromoe::graphcfg::env("ZEROMOE_QWEN4EXP_PREFILL_STATE_HANDOFF");
  if (value && value[0])
    return value[0] == '1';
  return g_config_prefill_state_handoff;
}

bool prefill_device_state_inputs_enabled(const StreamingEngine &engine) {
  const char *value = zeromoe::graphcfg::env("ZEROMOE_QWEN4EXP_PREFILL_DEVICE_STATE");
  const char *phase2 =
      zeromoe::graphcfg::env("ZEROMOE_QWEN4EXP_PARTIAL_LAYER_PHASE2");
  const bool phase2_enabled = phase2 && phase2[0] == '1';
  return !(value && value[0] == '0') && env_prefill_state_handoff_enabled() &&
         !engine.prompt_cache && !engine.mtp_enabled &&
         (engine.qwen4exp_partial_layer < 0 || phase2_enabled) &&
         engine.qwen4exp_gpu_stage_layers.empty();
}

bool env_decode_borrow_prefill_arenas() {
  const char *v =
      zeromoe::graphcfg::env("ZEROMOE_QWEN4EXP_DECODE_RECREATE_GRAPH_ARENA");
  return !(v && std::strcmp(v, "1") == 0);
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

// See FullGraph::layer_timing_enabled's own comment.
bool env_layer_timing_enabled() {
  return zeromoe::graphcfg::flag_on("ZEROMOE_QWEN4EXP_GRAPH_LAYER_TIMING");
}

// Decode phase ledger for the Phase-2/master-graph A/B.  This is deliberately
// separate from the per-CPU-node callback timing above: it brackets the
// scheduler submission, route decision/readback, suffix submission, output
// readback, and (for the master path) the actual device graph span.
bool env_phase_timing_enabled() {
  return zeromoe::graphcfg::flag_on("ZEROMOE_QWEN4EXP_GRAPH_PHASE_TIMING");
}

// Phase 4 host-node integration is deliberately a separate opt-in from the
// ordinary full graph.  It is only safe when the graph builder replaces the
// configured CPU MoE nodes with the fused CPY markers; an unset variable must
// retain the proven scheduler-split path byte-for-byte.
bool env_qwen4exp_host_node_enabled() {
  return zeromoe::graphcfg::flag_on("ZEROMOE_QWEN4EXP_HOST_NODE");
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
  return zeromoe::graphcfg::flag_on("ZEROMOE_QWEN4EXP_GRAPH_LAYER_TIMING_VERBOSE");
}

// Opt-in ledger for the CUDA graph replay boundary and ordinary GGML CPU
// backend work.  Keep this disabled by default because it adds steady_clock
// reads to the replay path.
bool env_graph_replay_timing_enabled() {
  return zeromoe::graphcfg::flag_on("ZEROMOE_QWEN4EXP_GRAPH_REPLAY_TIMING");
}

// The merged gate/up representation is intentionally runtime-switchable while
// its CUDA and CPU kernel cost is being compared with the established
// three-projection graph.  It is opt-in until the doubled-row MUL_MAT_ID path
// matches the separate projection path; setting this to 1 enables it without
// changing placement or routing.
bool env_merged_gate_up_enabled() {
  return zeromoe::graphcfg::flag_on("ZEROMOE_QWEN4EXP_GRAPH_MERGED_GATE_UP");
}

// Host-node transport diagnostic. The layer is selected explicitly so the
// same pre-H2D/post-H2D proof used for the historical Layer-0 IQ2 check can be
// applied to Layer 13 without enabling diagnostics for every callback.
int env_host_node_diag_layer() {
  const char *v = zeromoe::graphcfg::env("ZEROMOE_QWEN4EXP_HOST_NODE_DIAG_LAYER");
  if (!v || !v[0])
    return -1;
  char *end = nullptr;
  const long layer = std::strtol(v, &end, 10);
  if (end == v || *end != '\0' || layer < 0 || layer > 47)
    return -1;
  return static_cast<int>(layer);
}

bool env_host_node_transport_diag_enabled(int layer) {
  return layer == env_host_node_diag_layer();
}

// Phase 1 "all-or-nothing" partial-layer GPU dispatch (see
// engine.hpp's qwen4exp_partial_layer field group for the full contract).
int env_partial_layer() {
  const char *v = zeromoe::graphcfg::env("ZEROMOE_QWEN4EXP_PARTIAL_LAYER");
  if (!v || !v[0])
    return -1;
  char *end = nullptr;
  const long n = std::strtol(v, &end, 10);
  if (end == v || *end != '\0' || n < 0)
    return -1;
  return static_cast<int>(n);
}

int env_partial_layer_top_n() {
  const char *v = zeromoe::graphcfg::env("ZEROMOE_QWEN4EXP_PARTIAL_LAYER_TOP_N");
  if (!v || !v[0])
    return 256;
  char *end = nullptr;
  const long n = std::strtol(v, &end, 10);
  if (end == v || *end != '\0' || n <= 0)
    return 256;
  return static_cast<int>(n);
}

// Required prefill-routing coverage (fraction of this prompt's actual
// distinct layer-13-style routing already inside the top-N pinned set)
// before decode commits to the GPU-only path for the session. Default 1.0
// (100%) is deliberately strict: Phase 1 has no per-token fallback once
// committed, so an out-of-set expert mid-session is silently dropped by
// GGML_MUL_MAT_ID's negative-id contract rather than mis-routed -- see
// build_moe_partial_gpu's own comment. Lower this only to experiment; it
// trades a small, measured accuracy risk for a higher commit rate.
float env_partial_layer_min_coverage() {
  const char *v = zeromoe::graphcfg::env("ZEROMOE_QWEN4EXP_PARTIAL_LAYER_MIN_COVERAGE");
  if (!v || !v[0])
    return 1.0f;
  char *end = nullptr;
  const double n = std::strtod(v, &end);
  if (end == v || n < 0.0 || n > 1.0)
    return 1.0f;
  return static_cast<float>(n);
}

// Phase 2: true per-token dynamic GPU/CPU dispatch for the partial layer
// (see FullGraph::Phase2Mode's own comment for the 3-graph-instance
// mechanism). Off by default -- Phase 1's static session-level commit
// (env_partial_layer_min_coverage) remains the default behavior even when
// qwen4exp_partial_layer is configured; this flag opts into the dynamic
// per-token variant instead for a direct A/B against Phase 1.
bool env_partial_layer_phase2_enabled() {
  return zeromoe::graphcfg::flag_on("ZEROMOE_QWEN4EXP_PARTIAL_LAYER_PHASE2");
}

// The payload-backed staged child is a separate opt-in from the validated
// resident/CPU Phase-2 A/B. It requires the host storage boundary and a
// persistent CompletionGraph, and remains disabled in the production profile.
// Whether the PREFILL graph should capture the partial layer's router outputs
// (ids/probs/bias) for the Phase-1 coverage gate. Those tensors have to be marked
// ggml_set_output, which keeps them out of allocator reuse and backend fusion for the
// whole graph and so perturbs prefill numerics from a later layer onward (diag: every
// position diverges from early-ready from layer ~35, RMS 2.4e-2 at layer 47, while
// skipping the capture is bit-identical). Phase 2 routes per token in decode and never
// consumes the Phase-1 gate, so under Phase 2 the capture is skipped. Set
// ZEROMOE_QWEN4EXP_PARTIAL_LAYER_PREFILL_CAPTURE=1 to force the old behaviour.
bool partial_prefill_capture_enabled() {
  const char *v = zeromoe::graphcfg::env("ZEROMOE_QWEN4EXP_PARTIAL_LAYER_PREFILL_CAPTURE");
  if (v && v[0])
    return v[0] == '1';
  return !env_partial_layer_phase2_enabled();
}

// A storage miss must not hold decode indefinitely. The DDR/CPU suffix is an
// exact fallback while the worker continues its read; the worker retires the
// timed-out generation when it finishes so repeated slow reads cannot strand
// all bounded service slots. The staged path is opt-in, so this deadline does
// not affect the production early-ready profile.
std::chrono::milliseconds env_storage_wait_timeout() {
  constexpr long kDefaultMs = 100;
  constexpr long kMaximumMs = 30000;
  const char *value =
      zeromoe::graphcfg::env("ZEROMOE_QWEN4EXP_STORAGE_TIMEOUT_MS");
  if (!value || !value[0])
    return std::chrono::milliseconds(kDefaultMs);
  char *end = nullptr;
  const long parsed = std::strtol(value, &end, 10);
  if (end == value || *end != '\0' || parsed < 0)
    return std::chrono::milliseconds(kDefaultMs);
  return std::chrono::milliseconds(std::min(parsed, kMaximumMs));
}

bool env_adaptive_completion_policy_enabled() {
  return zeromoe::graphcfg::flag_on("ZEROMOE_QWEN4EXP_ADAPTIVE_POLICY");
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
// host-node mode (binding hurts the plain scheduler-split modes);
// ZEROMOE_QWEN4EXP_HOST_NODE_PIN=0 disables it.
void pin_host_node_threads(ggml_threadpool_params &p, int n_threads) {
  if (!env_qwen4exp_host_node_enabled())
    return;
  const char *v = zeromoe::graphcfg::env("ZEROMOE_QWEN4EXP_HOST_NODE_PIN");
  if (v && v[0] == '0')
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

int graph_cpu_threads() {
#if defined(_OPENMP)
  const int available = std::max(1, omp_get_max_threads());
#else
  // The native ggml CPU backend has its own persistent worker pool when
  // OpenMP is disabled.  Using one thread here would make the non-OpenMP
  // build an accidental serial baseline and would also defeat the host-node
  // runner's shared-pool experiment.
  const unsigned hw_threads = std::thread::hardware_concurrency();
  const int available = std::max(
      1, static_cast<int>(hw_threads == 0 ? 1 : hw_threads));
#endif
  const long cap = zeromoe::graphsettings::settings_for(zeromoe::graphsettings::Family::Qwen)
                       .tun.cpu_threads_decode.value;
  if (cap > 0) return static_cast<int>(std::min<long>(available, cap));
  return available;
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
// cheap event wait), previously verified at ~15.7->18.5-18.9 tok/s and
// accidentally undone by that commit. Default back to true; set
// ZEROMOE_QWEN4EXP_GRAPH_PARALLEL=0 for the blocking-sync A/B.
bool graph_scheduler_parallel() {
  return zeromoe::graphcfg::flag_unless_off("ZEROMOE_QWEN4EXP_GRAPH_PARALLEL");
}

// llama.cpp gives the CPU scheduler slot the CUDA device's host buffer type. That
// makes graph-owned CPU activations pinned host allocations, so the many
// CPU<->CUDA boundary copies do not fall through the pageable-host staging
// path. This is the normal full-graph mode; set the environment variable to
// 0 for an explicit pageable-buffer A/B.
bool graph_cuda_host_buffers() {
  return zeromoe::graphcfg::flag_unless_off("ZEROMOE_QWEN4EXP_GRAPH_CUDA_HOST_BUFFERS");
}

bool graph_match_llama_cpp_topology(const FullGraph &graph) {
  const char *v =
      zeromoe::graphcfg::env("ZEROMOE_QWEN4EXP_GRAPH_MATCH_LLAMA_CPP_TOPOLOGY");
  if (v && v[0])
    return v[0] != '0';
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
