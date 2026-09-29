// Persistent CPU GGML routed-expert graph for Qwen4Exp.
//
// This file is intentionally independent of the CUDA MoE runner. It is used
// only by execution_layer.cpp's Qwen CPU-affinity path. It supports both the
// original compact fallback (selected experts copied into a temporary expert
// axis) and a Llama.cpp strided mode that points a persistent
// MUL_MAT_ID graph at complete host-RAM expert arenas.

#include "ggml_moe_backend.hpp"

#include "kernels_cpu.hpp"
#include "logger.hpp"
#include "thread_profile.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <fstream>
#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif
#if defined(__linux__)
#include <pthread.h>
#include <sys/resource.h>
#include <sched.h>
#endif
#include <new>
#include <sstream>
#include <vector>

#if defined(MOESTREAM_USE_GGML)

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

// Direct Qwen execution reuses the same production quantized dots as the
// GGML MUL_MAT_ID path, but calls them from a persistent worker team instead
// of entering ggml_graph_compute().
struct qwen_direct_q8_k_block {
  float d;
  int8_t qs[256];
  int16_t bsums[16];
};
static_assert(sizeof(qwen_direct_q8_k_block) == 292);
extern "C" void quantize_row_q8_K(const float *, void *, int64_t);
extern "C" void quantize_row_q8_0(const float *, void *, int64_t);
extern "C" void ggml_vec_dot_iq1_m_q8_K(int, float *, size_t, const void *,
                                         size_t, const void *, size_t, int);
extern "C" void ggml_vec_dot_iq2_xxs_q8_K(int, float *, size_t, const void *,
                                           size_t, const void *, size_t, int);
extern "C" void ggml_vec_dot_iq3_xxs_q8_K(int, float *, size_t, const void *,
                                           size_t, const void *, size_t, int);
extern "C" void ggml_vec_dot_iq4_nl_q8_0(int, float *, size_t, const void *,
                                          size_t, const void *, size_t, int);

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace {

thread_local int qwen_cpu_diag_layer = -1;

bool qwen_cpu_diag_enabled() {
  const char *v = std::getenv("MOESTREAM_QWEN4EXP_DIAG_LAYER0_IQ2");
  return v && v[0] == '1' && qwen_cpu_diag_layer == 0;
}

void dump_qwen_cpu_diag_tensor(const char *role, const ggml_tensor *tensor) {
  if (!tensor || !qwen_cpu_diag_enabled())
    return;
  const size_t n = static_cast<size_t>(ggml_nelements(tensor));
  if (n == 0)
    return;
  std::vector<float> values(n, 0.0f);
  ggml_backend_tensor_get(tensor, values.data(), 0, n * sizeof(float));
  float min_v = INFINITY;
  float max_v = -INFINITY;
  double sum = 0.0;
  std::ostringstream first;
  for (size_t i = 0; i < n; ++i) {
    min_v = std::min(min_v, values[i]);
    max_v = std::max(max_v, values[i]);
    sum += values[i];
    if (i < 8)
      first << (i ? "," : "") << std::setprecision(9) << values[i];
  }
  LOG_INFO("[QWEN4EXP-LAYER0-DIAG] %s CPU ne=[%lld,%lld,%lld,%lld] "
           "nb=[%zu,%zu,%zu,%zu] values=%zu first=[%s] min=%.9g "
           "max=%.9g sum=%.9g data=%p\n",
           role, static_cast<long long>(tensor->ne[0]),
           static_cast<long long>(tensor->ne[1]),
           static_cast<long long>(tensor->ne[2]),
           static_cast<long long>(tensor->ne[3]), tensor->nb[0], tensor->nb[1],
           tensor->nb[2], tensor->nb[3], n, first.str().c_str(), min_v, max_v,
           sum, tensor->data);

  // Optional one-shot binary snapshots make the CPU graph boundary directly
  // comparable with llama.cpp's per-expert dumps.  Keep this opt-in and write only
  // the first instance of each role: a diagnostic prefill may execute many
  // S=1 positions, but position 0 is the parity boundary we are auditing.
  const char *dump_dir = std::getenv("MOESTREAM_QWEN4EXP_DIAG_DUMP_DIR");
  if (dump_dir && dump_dir[0]) {
    static std::atomic<bool> dumped_down_input{false};
    static std::atomic<bool> dumped_down_output{false};
    static std::atomic<bool> dumped_cpu_output{false};
    std::atomic<bool> *done = nullptr;
    if (std::strcmp(role, "down_input_swiglu") == 0)
      done = &dumped_down_input;
    else if (std::strcmp(role, "down_output_raw") == 0)
      done = &dumped_down_output;
    else if (std::strcmp(role, "cpu_output_pre_h2d") == 0)
      done = &dumped_cpu_output;
    if (done && !done->exchange(true, std::memory_order_relaxed)) {
      std::string path = std::string(dump_dir) + "/moe_cpu_" + role + ".f32";
      std::ofstream out(path, std::ios::binary | std::ios::trunc);
      if (out)
        out.write(reinterpret_cast<const char *>(values.data()),
                  static_cast<std::streamsize>(values.size() * sizeof(float)));
      LOG_INFO("[QWEN4EXP-LAYER0-DIAG] %s snapshot=%s bytes=%zu\n", role,
               path.c_str(), values.size() * sizeof(float));
    }
  }
}

constexpr size_t kWeightContextBytes = 64ull * 1024ull;
constexpr size_t kGraphContextBytes = 2ull * 1024ull * 1024ull;

int default_cpu_threads() {
#if defined(_OPENMP)
  const int available = std::max(1, omp_get_max_threads());
#else
  // The native ggml threadpool is available when OpenMP is disabled.  Do not
  // collapse the Qwen host-node runner to one worker merely because the
  // translation unit has no _OPENMP macro; that would make the non-OpenMP
  // scheduling experiment measure serial CPU math instead of a paused,
  // persistent worker team.  The runtime cap below still lets the caller
  // choose the physical-core budget explicitly.
  const unsigned hw_threads = std::thread::hardware_concurrency();
  const int available = std::max(
      1, static_cast<int>(hw_threads == 0 ? 1 : hw_threads));
#endif

  // Qwen decode's MUL_MAT_ID graphs are small enough that SMT is not always
  // beneficial: each selected expert performs irregular IQ unpacking and
  // competes for the same LLC.  Keep the existing process-wide default, but
  // allow a Qwen-only A/B override without changing any other model family or
  // the application's global OpenMP setting.
  const char *env = std::getenv("MOESTREAM_QWEN_CPU_THREADS");
  if (env && env[0]) {
    char *end = nullptr;
    const long requested = std::strtol(env, &end, 10);
    if (end != env && *end == '\0' && requested > 0)
      return std::min(available, static_cast<int>(requested));
  }

  // The host-node runner is a second GGML CPU execution context that may
  // overlap the scheduler's ordinary CPU backend.  Keep its cap separate
  // from MOESTREAM_QWEN_CPU_THREADS so contention can be diagnosed without
  // changing the A/B baseline CPU split configuration.
  const char *host_env = std::getenv("MOESTREAM_QWEN_HOST_NODE_THREADS");
  if (host_env && host_env[0]) {
    char *end = nullptr;
    const long requested = std::strtol(host_env, &end, 10);
    if (end != host_env && *end == '\0' && requested > 0)
      return std::min(available, static_cast<int>(requested));
  }
  return available;
}

bool type_is_valid(uint32_t type) {
  return type < static_cast<uint32_t>(GGML_TYPE_COUNT);
}

// Optional decode-only timing for the persistent Qwen CPU graph.  The normal
// production path does not pay for extra clocks or atomics.  Set
// MOESTREAM_QWEN_CPU_GRAPH_TIMING_REPORT_CALLS to the expected number of
// decode graph calls (for example 3840 for 48 CPU layers x 80 tokens) to get
// one aggregate split at the end of that window.  This is deliberately kept
// in the Qwen runner instead of the generic PerfCounters so other model
// families and the ordinary CPU fallback remain untouched.
struct QwenCpuGraphTiming {
  bool enabled = false;
  uint64_t report_calls = 0;
  std::atomic<uint64_t> calls{0};
  std::atomic<double> input_s{0.0};
  std::atomic<double> metadata_s{0.0};
  std::atomic<double> compute_s{0.0};
  std::atomic<double> output_s{0.0};
  std::atomic<double> total_s{0.0};
  std::atomic<bool> reported{false};
};

QwenCpuGraphTiming &qwen_cpu_graph_timing() {
  static QwenCpuGraphTiming state;
  static std::once_flag once;
  std::call_once(once, [] {
    const char *env =
        std::getenv("MOESTREAM_QWEN_CPU_GRAPH_TIMING_REPORT_CALLS");
    if (!env || !env[0])
      return;
    char *end = nullptr;
    const unsigned long long parsed = std::strtoull(env, &end, 10);
    if (end != env && *end == '\0' && parsed > 0) {
      // The state object is function-static and already constructed before
      // call_once invokes this initializer.
      state.enabled = true;
      state.report_calls = static_cast<uint64_t>(parsed);
    }
  });
  return state;
}

void qwen_cpu_graph_timing_add(std::atomic<double> &target, double seconds) {
  double old = target.load(std::memory_order_relaxed);
  while (!target.compare_exchange_weak(old, old + seconds,
                                       std::memory_order_relaxed,
                                       std::memory_order_relaxed)) {
  }
}

void qwen_cpu_graph_timing_report(QwenCpuGraphTiming &state,
                                  uint64_t calls) {
  if (!state.enabled || calls != state.report_calls ||
      state.reported.exchange(true, std::memory_order_relaxed))
    return;
  const double input = state.input_s.load(std::memory_order_relaxed);
  const double metadata = state.metadata_s.load(std::memory_order_relaxed);
  const double compute = state.compute_s.load(std::memory_order_relaxed);
  const double output = state.output_s.load(std::memory_order_relaxed);
  const double total = state.total_s.load(std::memory_order_relaxed);
  LOG_STREAM_INFO()
      << "[CPU-MOE-GRAPH-TIMING] calls=" << calls
      << " input_copy=" << input << "s (" << input * 1000.0 / calls
      << "ms/call) metadata=" << metadata << "s ("
      << metadata * 1000.0 / calls << "ms/call) compute=" << compute << "s ("
      << compute * 1000.0 / calls << "ms/call) output_copy=" << output
      << "s (" << output * 1000.0 / calls << "ms/call) total=" << total
      << "s (" << total * 1000.0 / calls << "ms/call)\n";
}

// The production Qwen graph-RAM arena is interleaved per expert as
// [gate|up|down].  Some focused tests and legacy callers provide three
// independent strided planes instead; those must keep the separate gate/up
// graph because their up pointer is not adjacent to the gate plane.
bool can_merge_gate_up(const GgmlQwenCpuExpertRef &ref,
                       const void *gate_plane, const void *up_plane) {
  static const bool disabled = [] {
    const char *env = std::getenv("MOESTREAM_QWEN_CPU_DISABLE_MERGED_GATE_UP");
    return env && env[0] == '1';
  }();
  if (disabled)
    return false;
  return ref.gate_type == ref.up_type && ref.gate_bytes == ref.up_bytes &&
         gate_plane && up_plane &&
         static_cast<const uint8_t *>(up_plane) ==
             static_cast<const uint8_t *>(gate_plane) + ref.gate_bytes;
}

bool qwen_cpu_direct_pool_env_enabled() {
  const char *env = std::getenv("MOESTREAM_QWEN_CPU_DIRECT_POOL");
  return env && env[0] == '1';
}

bool qwen_cpu_direct_ephemeral_env_enabled() {
  const char *env = std::getenv("MOESTREAM_QWEN_CPU_DIRECT_EPHEMERAL");
  return env && env[0] == '1';
}

bool qwen_cpu_direct_timing_enabled() {
  const char *env = std::getenv("MOESTREAM_QWEN_CPU_DIRECT_TIMING");
  return env && env[0] == '1';
}

bool qwen_cpu_direct_fused_enabled() {
  const char *env = std::getenv("MOESTREAM_QWEN_CPU_DIRECT_FUSED");
  return env && env[0] == '1';
}

// Direct workers must remain scheduler-free while a request is active, but a
// permanently hot team also steals every core from GGML's ordinary CPU
// splits during the rest of the token.  Park only the idle state; the active
// input/gate-up/down/reduce phases still use the generation counter and
// _mm_pause() exclusively.
bool qwen_cpu_direct_idle_wait_enabled() {
  const char *env = std::getenv("MOESTREAM_QWEN_CPU_DIRECT_IDLE_WAIT");
  return !(env && env[0] == '0');
}

bool qwen_cpu_direct_caller_worker_enabled() {
  const char *env = std::getenv("MOESTREAM_QWEN_CPU_DIRECT_CALLER_WORKER");
  return env && env[0] == '1';
}

int qwen_cpu_direct_worker_count() {
  const int grouped = std::max(1, qwen_cpu_grouped_threads());
  // Direct workers are parked while idle, and the host-node callback is
  // serialized against ordinary GGML CPU splits by the execution gate.  The
  // old grouped-1 cap was retained from the first always-spinning pool and
  // unnecessarily left one physical core unused during the callback itself.
  // Allow the direct team to use the full configured CPU group; the callback
  // thread may still run on an SMT sibling, but it no longer competes with a
  // permanently runnable worker while the pool is idle.
  const int max_workers = grouped;
  const char *env = std::getenv("MOESTREAM_QWEN_CPU_DIRECT_THREADS");
  if (env && env[0]) {
    char *end = nullptr;
    const long requested = std::strtol(env, &end, 10);
    if (end != env && *end == '\0' && requested > 0)
      return std::min(max_workers, static_cast<int>(requested));
  }

  // This machine has six physical cores / twelve logical CPUs.  The direct
  // team is idle-waiting between requests, so use all physical cores during
  // the callback instead of preserving the old five-worker reservation.
  const unsigned logical = std::thread::hardware_concurrency();
  const int physical_guess = logical >= 2 ? static_cast<int>((logical + 1) / 2)
                                         : static_cast<int>(logical);
  const int default_workers = physical_guess > 1 ? physical_guess - 1 : 1;
  return std::min(max_workers, std::max(1, default_workers));
}

bool qwen_cpu_direct_prefetch_enabled() {
  const char *env = std::getenv("MOESTREAM_QWEN_CPU_DIRECT_PREFETCH");
  return env && env[0] == '1';
}

bool qwen_cpu_direct_diag_enabled() {
  const char *env = std::getenv("MOESTREAM_QWEN_CPU_DIRECT_DIAG");
  return env && env[0] == '1';
}

enum : int {
  QWEN_DIRECT_IDLE = 0,
  QWEN_DIRECT_INPUT = 1,
  QWEN_DIRECT_GATE_UP = 2,
  QWEN_DIRECT_DOWN = 3,
  QWEN_DIRECT_REDUCE = 4,
};

// The low byte is the command; the upper bits are a monotonically increasing
// phase generation.  A single atomic publication prevents workers from
// pairing a newly observed generation with a command from the previous phase.
constexpr uint64_t kDirectPhaseCommandMask = 0xffu;
constexpr unsigned kDirectPhaseGenerationShift = 8;

inline uint64_t direct_phase_generation(uint64_t phase) {
  return phase >> kDirectPhaseGenerationShift;
}

inline int direct_phase_command(uint64_t phase) {
  return static_cast<int>(phase & kDirectPhaseCommandMask);
}

inline uint64_t direct_next_phase(uint64_t phase, int command) {
  return ((direct_phase_generation(phase) + 1u)
          << kDirectPhaseGenerationShift) |
         (static_cast<uint64_t>(command) & kDirectPhaseCommandMask);
}

bool qwen_cpu_direct_gate_type(uint32_t type) {
  return type == GGML_TYPE_IQ1_M || type == GGML_TYPE_IQ2_XXS ||
         type == GGML_TYPE_IQ3_XXS || type == GGML_TYPE_Q4_0 ||
         type == GGML_TYPE_Q2_0;
}

size_t qwen_cpu_direct_gate_row_bytes(uint32_t type, int d_model) {
  // Q2_0 uses 64 weights per 18-byte block.
  if (type == GGML_TYPE_Q2_0)
    return (d_model > 0 && d_model % 64 == 0)
               ? static_cast<size_t>(d_model / 64) * 18
               : 0;
  // Q4_0 has a 32-element block (not the 256-element Q8_K-style block the
  // IQ1_M/IQ2_XXS/IQ3_XXS cases below use), so it needs its own precondition
  // rather than sharing the "d_model % 256" guard.
  if (type == GGML_TYPE_Q4_0)
    return (d_model > 0 && d_model % 32 == 0)
               ? static_cast<size_t>(d_model / 32) * sizeof(block_q4_0)
               : 0;
  if (d_model <= 0 || d_model % 256 != 0)
    return 0;
  const size_t blocks = static_cast<size_t>(d_model / 256);
  switch (type) {
  case GGML_TYPE_IQ1_M:
    return blocks * 56;
  case GGML_TYPE_IQ2_XXS:
    return blocks * 66;
  case GGML_TYPE_IQ3_XXS:
    return blocks * 98;
  default:
    return 0;
  }
}

bool qwen_cpu_direct_is_iq4_nl(uint32_t type, int d_ff) {
  return type == GGML_TYPE_IQ4_NL && d_ff > 0 && d_ff % 32 == 0;
}

bool qwen_cpu_direct_is_q4_0(uint32_t type, int d_ff) {
  return type == GGML_TYPE_Q4_0 && d_ff > 0 && d_ff % 32 == 0;
}

bool qwen_cpu_direct_is_q2_0(uint32_t type, int d_ff) {
  return type == GGML_TYPE_Q2_0 && d_ff > 0 && d_ff % 64 == 0;
}

inline void qwen_cpu_direct_pause() {
#if defined(__x86_64__) || defined(__i386__)
  _mm_pause();
#else
  std::this_thread::yield();
#endif
}

inline void qwen_cpu_direct_prefetch_row(const uint8_t *row,
                                         size_t row_bytes) {
#if defined(__x86_64__) || defined(__i386__)
  // Seed only the first cache line of the next quantized row.  The production
  // IQ kernels still own all decoding and remain nrc=1; this is deliberately
  // a light-touch experiment because prefetching several lines per row was
  // slower in the first A/B run.
  _mm_prefetch(reinterpret_cast<const char *>(row), _MM_HINT_T0);
  (void)row_bytes;
#else
  (void)row;
  (void)row_bytes;
#endif
}

bool qwen_cpu_direct_thread_diag_enabled() {
  return qwen_cpu_direct_diag_enabled();
}

int64_t qwen_cpu_direct_thread_cpu_ns() {
#if defined(__linux__)
  timespec ts{};
  if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts) == 0)
    return static_cast<int64_t>(ts.tv_sec) * 1000000000ll + ts.tv_nsec;
#endif
  return 0;
}

void qwen_cpu_direct_sample_begin(GgmlQwenDirectWorkerSample &sample,
                                  bool enabled) {
  if (!enabled)
    return;
  sample.wall_start_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                             std::chrono::steady_clock::now().time_since_epoch())
                             .count();
  sample.cpu_start_ns = qwen_cpu_direct_thread_cpu_ns();
#if defined(__linux__)
  sample.cpu_start = sched_getcpu();
  rusage usage{};
  if (getrusage(RUSAGE_THREAD, &usage) == 0) {
    sample.voluntary_start = usage.ru_nvcsw;
    sample.involuntary_start = usage.ru_nivcsw;
  }
#endif
}

void qwen_cpu_direct_sample_end(GgmlQwenDirectWorkerSample &sample,
                                bool enabled) {
  if (!enabled)
    return;
  sample.wall_end_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                           std::chrono::steady_clock::now().time_since_epoch())
                           .count();
  sample.cpu_end_ns = qwen_cpu_direct_thread_cpu_ns();
#if defined(__linux__)
  sample.cpu_end = sched_getcpu();
  rusage usage{};
  if (getrusage(RUSAGE_THREAD, &usage) == 0) {
    sample.voluntary_end = usage.ru_nvcsw;
    sample.involuntary_end = usage.ru_nivcsw;
  }
#endif
}

void qwen_cpu_direct_pin_thread(int worker_index) {
#if defined(__linux__)
  if (moestream_thread_profile_enabled()) {
    configure_current_thread({
        .name = "MOE-WORKER-" + std::to_string(worker_index),
        .cpu_ids = moestream_thread_profile_physical_core(worker_index),
        .use_rt = false,
        .rt_priority = 0,
        .nice_value = -15,
        .apply_nice = false,
    });
    return;
  }
  const char *env = std::getenv("MOESTREAM_QWEN_CPU_DIRECT_PIN");
  if (!env || env[0] != '1')
    return;

  cpu_set_t allowed;
  CPU_ZERO(&allowed);
  if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0)
    return;
  int target = -1;
  for (int cpu = 0, seen = 0; cpu < CPU_SETSIZE; ++cpu) {
    if (!CPU_ISSET(cpu, &allowed))
      continue;
    if (seen++ == worker_index) {
      target = cpu;
      break;
    }
  }
  if (target < 0)
    return;
  cpu_set_t selected;
  CPU_ZERO(&selected);
  CPU_SET(target, &selected);
  (void)pthread_setaffinity_np(pthread_self(), sizeof(selected), &selected);
#else
  (void)worker_index;
#endif
}

} // namespace

void qwen_cpu_moe_diag_set_layer(int layer) { qwen_cpu_diag_layer = layer; }

GgmlQwenCpuMoeRunner &GgmlQwenCpuMoeRunner::instance() {
  static GgmlQwenCpuMoeRunner runner;
  return runner;
}

GgmlQwenCpuMoeRunner &GgmlQwenCpuMoeRunner::instance_for_strided(
    const GgmlQwenCpuExpertRef &layer, int n_experts, int d_model, int d_ff,
    int topk, size_t expert_stride, bool output_per_expert) {
  // The current Qwen4Exp checkpoint needs two entries (IQ1_M and IQ2_XXS),
  // while the extra slots make this resilient to a future third quantized
  // layer family without falling back to per-layer graph destruction.  These
  // runners are used serially by one decode request, but each keeps its own
  // persistent GGML graph and worker pool.
  static std::array<GgmlQwenCpuMoeRunner, 8> runners;
  for (auto &runner : runners) {
    if (runner.has_strided_signature(layer, n_experts, d_model, d_ff, topk,
                                     expert_stride, output_per_expert))
      return runner;
  }
  for (auto &runner : runners) {
    if (!runner.graph_ && !runner.direct_ready_)
      return runner;
  }
  // More signatures than the reserved cache is unexpected.  Reusing a slot
  // retains the old safe behavior (ensure_graph() will rebuild it), while all
  // normal Qwen signatures take one of the persistent slots above.
  return runners[0];
}

GgmlQwenCpuMoeRunner &GgmlQwenCpuMoeRunner::instance_for_strided_planes(
    const GgmlQwenCpuExpertRef &layer, int n_experts, int d_model, int d_ff,
    int topk, size_t gate_stride, size_t up_stride, size_t down_stride,
    bool output_per_expert) {
  // Keep compact-plane runners separate from the legacy single-stride cache:
  // their GGML tensor descriptors have genuinely different nb[2] values.
  static std::array<GgmlQwenCpuMoeRunner, 8> runners;
  for (auto &runner : runners) {
    if (runner.has_strided_planes_signature(
            layer, n_experts, d_model, d_ff, topk, gate_stride, up_stride,
            down_stride, output_per_expert))
      return runner;
  }
  for (auto &runner : runners) {
    if (!runner.graph_ && !runner.direct_ready_)
      return runner;
  }
  return runners[0];
}

bool GgmlQwenCpuMoeRunner::compiled_in() { return true; }

void GgmlQwenCpuMoeRunner::pause_workers() {
  // A borrowed pool belongs to the full graph and remains available to its
  // ordinary CPU splits.  Pausing it here would suspend unrelated work.
  if (cpu_threadpool_ && !cpu_threadpool_borrowed_)
    ggml_threadpool_pause(static_cast<ggml_threadpool_t>(cpu_threadpool_));
}

void GgmlQwenCpuMoeRunner::resume_workers() {
  if (cpu_threadpool_ && !cpu_threadpool_borrowed_)
    ggml_threadpool_resume(static_cast<ggml_threadpool_t>(cpu_threadpool_));
}

void GgmlQwenCpuMoeRunner::bind_external_threadpool(void *threadpool,
                                                     int n_threads) {
  if (!threadpool || n_threads <= 0)
    return;

  if (cpu_threadpool_ == threadpool && cpu_threadpool_borrowed_) {
    n_threads_ = n_threads;
    return;
  }

  // No callback may be using the runner while a graph is being rebuilt or
  // rebound: host-node construction is performed before capture, and replay
  // is synchronous with respect to graph teardown.
  stop_direct_pool();
  free_graph();

  if (cpu_backend_ && cpu_threadpool_ && !cpu_threadpool_borrowed_)
    ggml_backend_cpu_set_threadpool(
        static_cast<ggml_backend_t>(cpu_backend_), nullptr);

  // A previously borrowed pool is owned by its caller; detach only.  An
  // owned pool is released here because the runner is switching to the
  // graph-owned pool.
  if (cpu_threadpool_ && !cpu_threadpool_borrowed_)
    ggml_threadpool_free(static_cast<ggml_threadpool_t>(cpu_threadpool_));

  cpu_threadpool_ = threadpool;
  cpu_threadpool_borrowed_ = true;
  n_threads_ = n_threads;
  if (cpu_backend_) {
    // ggml's OpenMP build does not consume the threadpool pointer, but it
    // still takes the worker count from the CPU backend context when creating
    // the graph plan.  Keep both backends consistent so borrowing the
    // FullGraph pool cannot leave this runner at its old host-node cap.
    ggml_backend_cpu_set_n_threads(
        static_cast<ggml_backend_t>(cpu_backend_), n_threads_);
    ggml_backend_cpu_set_threadpool(
        static_cast<ggml_backend_t>(cpu_backend_),
        static_cast<ggml_threadpool_t>(cpu_threadpool_));
  }
}

void GgmlQwenCpuMoeRunner::unbind_external_threadpool(void *threadpool) {
  if (!threadpool || !cpu_threadpool_borrowed_ ||
      cpu_threadpool_ != threadpool)
    return;

  free_graph();
  if (cpu_backend_)
    ggml_backend_cpu_set_threadpool(
        static_cast<ggml_backend_t>(cpu_backend_), nullptr);
  cpu_threadpool_ = nullptr;
  cpu_threadpool_borrowed_ = false;
  n_threads_ = 0;
}

bool GgmlQwenCpuMoeRunner::has_strided_signature(
    const GgmlQwenCpuExpertRef &layer, int n_experts, int d_model, int d_ff,
    int topk, size_t expert_stride, bool output_per_expert) const {
  return has_strided_planes_signature(
      layer, n_experts, d_model, d_ff, topk, expert_stride, expert_stride,
      expert_stride, output_per_expert);
}

bool GgmlQwenCpuMoeRunner::has_strided_planes_signature(
    const GgmlQwenCpuExpertRef &layer, int n_experts, int d_model, int d_ff,
    int topk, size_t gate_stride, size_t up_stride, size_t down_stride,
    bool output_per_expert) const {
  return (graph_ || direct_ready_) && strided_mode_ &&
         n_experts_ == n_experts &&
         d_model_ == d_model && d_ff_ == d_ff && topk_ == topk &&
         gate_type_ == layer.gate_type && up_type_ == layer.up_type &&
         down_type_ == layer.down_type &&
         output_per_expert_ == output_per_expert &&
         expert_stride_ == gate_stride &&
         up_expert_stride_ == up_stride &&
         down_expert_stride_ == down_stride;
}

GgmlQwenCpuMoeRunner::~GgmlQwenCpuMoeRunner() { shutdown(); }

void GgmlQwenCpuMoeRunner::free_graph() {
  // All graph executions are synchronous, so no CPU worker can still be
  // touching these objects when a shape/type change reaches this function.
  if (graph_plan_ && cpu_backend_) {
    ggml_backend_graph_plan_free(
        static_cast<ggml_backend_t>(cpu_backend_),
        static_cast<ggml_backend_graph_plan_t>(graph_plan_));
    graph_plan_ = nullptr;
  }
  if (galloc_) {
    ggml_gallocr_free(static_cast<ggml_gallocr_t>(galloc_));
    galloc_ = nullptr;
  }
  if (graph_ctx_) {
    ggml_free(static_cast<ggml_context *>(graph_ctx_));
    graph_ctx_ = nullptr;
  }
  graph_ = nullptr;
  x_ = ids_ = weights_ = output_ = nullptr;
  act_ = down_output_ = nullptr;

  if (weight_buf_) {
    ggml_backend_buffer_free(static_cast<ggml_backend_buffer_t>(weight_buf_));
    weight_buf_ = nullptr;
  }
  for (void *buf : external_weight_bufs_) {
    if (buf)
      ggml_backend_buffer_free(static_cast<ggml_backend_buffer_t>(buf));
  }
  external_weight_bufs_.clear();
  if (weight_ctx_) {
    ggml_free(static_cast<ggml_context *>(weight_ctx_));
    weight_ctx_ = nullptr;
  }
  gate_ = up_ = gate_up_ = down_ = nullptr;
  if (weight_storage_)
    std::free(weight_storage_);
  weight_storage_ = nullptr;
  weight_storage_bytes_ = 0;
  n_experts_ = d_model_ = d_ff_ = topk_ = 0;
  gate_type_ = up_type_ = down_type_ = 0;
  strided_mode_ = false;
  merged_gate_up_ = false;
  output_per_expert_ = false;
  expert_stride_ = 0;
  up_expert_stride_ = 0;
  down_expert_stride_ = 0;
  cached_sources_.clear();
  direct_ready_ = false;
  direct_armed_.store(false, std::memory_order_release);
  direct_threads_ = 0;
  direct_input_q8k_.clear();
  direct_down_q8_0_.clear();
  direct_gate_output_.clear();
  direct_up_output_.clear();
  direct_down_output_.clear();
  direct_gate_rows_.clear();
  direct_up_rows_.clear();
  direct_down_rows_.clear();
  direct_gate_samples_.clear();
  direct_down_samples_.clear();
  direct_reduce_samples_.clear();
}

void GgmlQwenCpuMoeRunner::shutdown() {
  if (!direct_workers_.empty()) {
    std::unique_lock<std::mutex> lock(async_mutex_);
    async_cv_.wait(lock, [this] { return async_request_done_; });
  }
  stop_direct_pool();
  if (async_worker_.joinable()) {
    {
      std::unique_lock<std::mutex> lock(async_mutex_);
      // A caller may shut the runner down from the request teardown path.
      // Finish the one in-flight graph before freeing its GGML objects.
      async_cv_.wait(lock, [this] { return async_request_done_; });
      async_stop_ = true;
    }
    async_cv_.notify_all();
    async_worker_.join();
  }
  free_graph();
  if (cpu_backend_ && cpu_threadpool_ && !cpu_threadpool_borrowed_) {
    // Detach before freeing the pool so the backend never retains a dangling
    // threadpool pointer during teardown.
    ggml_backend_cpu_set_threadpool(
        static_cast<ggml_backend_t>(cpu_backend_), nullptr);
  }
  if (cpu_threadpool_ && !cpu_threadpool_borrowed_) {
    ggml_threadpool_free(static_cast<ggml_threadpool_t>(cpu_threadpool_));
  }
  cpu_threadpool_ = nullptr;
  cpu_threadpool_borrowed_ = false;
  if (cpu_backend_) {
    ggml_backend_free(static_cast<ggml_backend_t>(cpu_backend_));
    cpu_backend_ = nullptr;
  }
  n_threads_ = 0;
}

bool GgmlQwenCpuMoeRunner::direct_pool_enabled() const {
  return qwen_cpu_direct_pool_env_enabled();
}

void GgmlQwenCpuMoeRunner::ensure_direct_workers() {
  if (!direct_ready_ || !direct_workers_.empty())
    return;
  direct_stop_.store(false, std::memory_order_release);
  direct_workers_.reserve(static_cast<size_t>(direct_threads_));
  for (int i = 0; i < direct_threads_; ++i)
    direct_workers_.emplace_back(&GgmlQwenCpuMoeRunner::direct_worker_loop,
                                 this, i);
}

void GgmlQwenCpuMoeRunner::prepare_direct_workers() {
  if (direct_pool_enabled())
    ensure_direct_workers();
}

void GgmlQwenCpuMoeRunner::arm_direct_workers() {
  if (!direct_pool_enabled() || !direct_ready_)
    return;
  ensure_direct_workers();
  direct_armed_.store(true, std::memory_order_release);
  direct_phase_.notify_all();
}

bool GgmlQwenCpuMoeRunner::direct_signature_supported(
    const GgmlQwenCpuExpertRef &ref, int d_model, int d_ff, int topk,
    size_t gate_stride, size_t up_stride, size_t down_stride) const {
  const size_t gate_row_bytes =
      qwen_cpu_direct_gate_row_bytes(ref.gate_type, d_model);
  const size_t up_row_bytes =
      qwen_cpu_direct_gate_row_bytes(ref.up_type, d_model);
  const bool q2_0_pair = ref.gate_type == GGML_TYPE_Q2_0 &&
                         ref.up_type == GGML_TYPE_Q2_0 &&
                         ref.down_type == GGML_TYPE_Q2_0;
  const bool has_q2_0 = ref.gate_type == GGML_TYPE_Q2_0 ||
                        ref.up_type == GGML_TYPE_Q2_0 ||
                        ref.down_type == GGML_TYPE_Q2_0;
  const size_t down_row_bytes = qwen_cpu_direct_is_q2_0(ref.down_type, d_ff)
                                    ? static_cast<size_t>(d_ff / 64) * 18
                                    : (d_ff > 0 && d_ff % 32 == 0
                                           ? static_cast<size_t>(d_ff / 32) * 18
                                           : 0);
  const bool gate_type_ok = qwen_cpu_direct_gate_type(ref.gate_type);
  const bool up_type_ok = qwen_cpu_direct_gate_type(ref.up_type);
  const bool down_type_ok = qwen_cpu_direct_is_iq4_nl(ref.down_type, d_ff) ||
                            qwen_cpu_direct_is_q4_0(ref.down_type, d_ff) ||
                            qwen_cpu_direct_is_q2_0(ref.down_type, d_ff);
  const bool dims_ok = d_model > 0 && d_ff > 0 && topk > 0;
  const bool row_bytes_ok = gate_row_bytes != 0 && up_row_bytes != 0 &&
                            down_row_bytes != 0;
  const bool stride_ok = row_bytes_ok &&
                         gate_stride >= gate_row_bytes * d_ff &&
                         up_stride >= up_row_bytes * d_ff &&
                         down_stride >= down_row_bytes * d_model;
  const bool source_bytes_ok = row_bytes_ok &&
                               ref.gate_bytes >= gate_row_bytes * d_ff &&
                               ref.up_bytes >= up_row_bytes * d_ff &&
                               ref.down_bytes >= down_row_bytes * d_model;
  const bool supported = gate_type_ok && up_type_ok && down_type_ok &&
                         dims_ok && stride_ok && source_bytes_ok;
  // Q2_0 bypasses the worker-pool graph only through its complete fused
  // implementation; otherwise the regular GGML graph remains the fallback.
  const bool q2_0_route_ok =
      !has_q2_0 || (q2_0_pair && qwen_cpu_direct_fused_enabled());
  if (!supported && qwen_cpu_direct_diag_enabled()) {
    static std::atomic<int> diagnostic_count{0};
    if (diagnostic_count.fetch_add(1, std::memory_order_relaxed) < 64) {
      const auto type_name = [](uint32_t type) {
        return type < static_cast<uint32_t>(GGML_TYPE_COUNT)
                   ? ggml_type_name(static_cast<ggml_type>(type))
                   : "invalid";
      };
      LOG_WARN(
          "[DIRECT-MOE-REJECT] gate=%s(%u) up=%s(%u) down=%s(%u) "
          "D=%d FF=%d topk=%d gate_stride=%zu up_stride=%zu "
          "down_stride=%zu gate_bytes=%zu up_bytes=%zu down_bytes=%zu "
          "checks gate_type=%d up_type=%d down_type=%d dims=%d rows=%d "
          "strides=%d source_bytes=%d\n",
          type_name(ref.gate_type), ref.gate_type, type_name(ref.up_type),
          ref.up_type, type_name(ref.down_type), ref.down_type, d_model, d_ff,
          topk, gate_stride, up_stride, down_stride, ref.gate_bytes,
          ref.up_bytes, ref.down_bytes, gate_type_ok, up_type_ok, down_type_ok,
          dims_ok, row_bytes_ok, stride_ok, source_bytes_ok);
    }
  }
  return supported && q2_0_route_ok;
}

bool GgmlQwenCpuMoeRunner::ensure_direct_signature(
    const GgmlQwenCpuExpertRef &ref, int n_experts, int d_model, int d_ff,
    int topk, size_t gate_stride, size_t up_stride, size_t down_stride) {
  if (n_experts <= 0 || topk <= 0 || topk > n_experts ||
      !direct_signature_supported(ref, d_model, d_ff, topk, gate_stride,
                                  up_stride, down_stride))
    return false;

  const bool matches =
      direct_ready_ && strided_mode_ && n_experts_ == n_experts &&
      d_model_ == d_model && d_ff_ == d_ff && topk_ == topk &&
      gate_type_ == ref.gate_type && up_type_ == ref.up_type &&
      down_type_ == ref.down_type && expert_stride_ == gate_stride &&
      up_expert_stride_ == up_stride && down_expert_stride_ == down_stride &&
      !output_per_expert_;
  if (matches)
    return true;

  // A runner is normally selected by signature.  If the small signature
  // cache is exhausted, it may be reused; no request is allowed to be in
  // flight at this point, so retiring its direct team is safe.
  stop_direct_pool();
  free_graph();

  n_experts_ = n_experts;
  d_model_ = d_model;
  d_ff_ = d_ff;
  topk_ = topk;
  gate_type_ = ref.gate_type;
  up_type_ = ref.up_type;
  down_type_ = ref.down_type;
  strided_mode_ = true;
  merged_gate_up_ = false;
  output_per_expert_ = false;
  expert_stride_ = gate_stride;
  up_expert_stride_ = up_stride;
  down_expert_stride_ = down_stride;

  const size_t input_blocks = static_cast<size_t>(d_model / 256);
  const size_t down_blocks = static_cast<size_t>(d_ff / 32);
  direct_input_q8k_.resize(input_blocks * sizeof(qwen_direct_q8_k_block));
  direct_down_q8_0_.resize(static_cast<size_t>(topk_) * down_blocks *
                           sizeof(block_q8_0));
  direct_gate_output_.resize(static_cast<size_t>(topk_) * d_ff_);
  direct_up_output_.resize(static_cast<size_t>(topk_) * d_ff_);
  direct_down_output_.resize(static_cast<size_t>(topk_) * d_model_);
  direct_fused_gate_output_ptrs_.resize(static_cast<size_t>(topk_));
  direct_fused_up_output_ptrs_.resize(static_cast<size_t>(topk_));
  direct_fused_down_output_ptrs_.resize(static_cast<size_t>(topk_));
  direct_fused_gate_weight_ptrs_.resize(static_cast<size_t>(topk_));
  direct_fused_up_weight_ptrs_.resize(static_cast<size_t>(topk_));
  direct_fused_down_weight_ptrs_.resize(static_cast<size_t>(topk_));
  direct_gate_rows_.resize(static_cast<size_t>(topk_) * d_ff_);
  direct_up_rows_.resize(static_cast<size_t>(topk_) * d_ff_);
  direct_down_rows_.resize(static_cast<size_t>(topk_) * d_model_);
  // Synchronous host-node callers spin while the team runs. Leave one
  // logical worker slot for that caller instead of oversubscribing the CPU
  // with N spinning workers plus one spinning callback thread.
  direct_threads_ = qwen_cpu_direct_worker_count();
  direct_gate_samples_.assign(static_cast<size_t>(direct_threads_), {});
  direct_down_samples_.assign(static_cast<size_t>(direct_threads_), {});
  direct_reduce_samples_.assign(static_cast<size_t>(direct_threads_), {});
  direct_ready_ = true;
  LOG_INFO("[QWEN-CPU-DIRECT] enabled: d_model=%d d_ff=%d topk=%d "
           "gate=%u up=%u down=%u threads=%d\n",
           d_model_, d_ff_, topk_, gate_type_, up_type_, down_type_,
           direct_threads_);
  return true;
}

bool GgmlQwenCpuMoeRunner::run_direct_fused_request(
    const AsyncStridedRequest &request) {
  const bool is_iq_iq4_nl_combo =
      (request.layer.gate_type == GGML_TYPE_IQ1_M ||
       request.layer.gate_type == GGML_TYPE_IQ2_XXS) &&
      request.layer.up_type == request.layer.gate_type &&
      request.layer.down_type == GGML_TYPE_IQ4_NL;
  const bool is_q4_0_combo =
      request.layer.gate_type == GGML_TYPE_Q4_0 &&
      request.layer.up_type == GGML_TYPE_Q4_0 &&
      request.layer.down_type == GGML_TYPE_Q4_0;
  const bool is_q2_0_combo =
      request.layer.gate_type == GGML_TYPE_Q2_0 &&
      request.layer.up_type == GGML_TYPE_Q2_0 &&
      request.layer.down_type == GGML_TYPE_Q2_0;
  if (!qwen_cpu_direct_fused_enabled() ||
      (!is_iq_iq4_nl_combo && !is_q4_0_combo && !is_q2_0_combo) ||
      !direct_ready_ || request.topk <= 0 ||
      request.topk > static_cast<int>(direct_fused_gate_output_ptrs_.size()))
    return false;

  const auto start = std::chrono::steady_clock::now();
  for (int m = 0; m < request.topk; ++m) {
    const int expert = request.expert_ids[m];
    if (expert < 0 || expert >= request.n_experts)
      return false;
    direct_fused_gate_output_ptrs_[static_cast<size_t>(m)] =
        direct_gate_output_.data() + static_cast<size_t>(m) * request.d_ff;
    direct_fused_up_output_ptrs_[static_cast<size_t>(m)] =
        direct_up_output_.data() + static_cast<size_t>(m) * request.d_ff;
    direct_fused_down_output_ptrs_[static_cast<size_t>(m)] =
        direct_down_output_.data() + static_cast<size_t>(m) * request.d_model;
    direct_fused_gate_weight_ptrs_[static_cast<size_t>(m)] =
        static_cast<const uint8_t *>(request.gate_plane) +
        static_cast<size_t>(expert) * request.expert_stride;
    direct_fused_up_weight_ptrs_[static_cast<size_t>(m)] =
        static_cast<const uint8_t *>(request.up_plane) +
        static_cast<size_t>(expert) * request.up_expert_stride;
    direct_fused_down_weight_ptrs_[static_cast<size_t>(m)] =
        static_cast<const uint8_t *>(request.down_plane) +
        static_cast<size_t>(expert) * request.down_expert_stride;
  }

  const int team_threads = std::max(1, direct_threads_);
  bool ok = false;
  if (request.layer.gate_type == GGML_TYPE_Q2_0) {
    ok = matmul_qwen_q2_0_q2_0_fused(
        direct_fused_gate_output_ptrs_.data(),
        direct_fused_up_output_ptrs_.data(),
        direct_fused_down_output_ptrs_.data(), request.input,
        direct_fused_gate_weight_ptrs_.data(),
        direct_fused_up_weight_ptrs_.data(),
        direct_fused_down_weight_ptrs_.data(), request.topk, request.d_model,
        request.d_ff, request.d_model, team_threads);
  } else if (request.layer.gate_type == GGML_TYPE_Q4_0) {
    ok = matmul_qwen_q4_0_q4_0_fused(
        direct_fused_gate_output_ptrs_.data(),
        direct_fused_up_output_ptrs_.data(),
        direct_fused_down_output_ptrs_.data(), request.input,
        direct_fused_gate_weight_ptrs_.data(),
        direct_fused_up_weight_ptrs_.data(),
        direct_fused_down_weight_ptrs_.data(), request.topk, request.d_model,
        request.d_ff, request.d_model, team_threads);
  } else if (request.layer.gate_type == GGML_TYPE_IQ1_M) {
    ok = matmul_qwen_iq1_m_iq4_nl_fused(
        direct_fused_gate_output_ptrs_.data(),
        direct_fused_up_output_ptrs_.data(),
        direct_fused_down_output_ptrs_.data(), request.input,
        direct_fused_gate_weight_ptrs_.data(),
        direct_fused_up_weight_ptrs_.data(),
        direct_fused_down_weight_ptrs_.data(), request.topk, request.d_model,
        request.d_ff, request.d_model, team_threads);
  } else {
    ok = matmul_qwen_iq2_xxs_iq4_nl_fused(
        direct_fused_gate_output_ptrs_.data(),
        direct_fused_up_output_ptrs_.data(),
        direct_fused_down_output_ptrs_.data(), request.input,
        direct_fused_gate_weight_ptrs_.data(),
        direct_fused_up_weight_ptrs_.data(),
        direct_fused_down_weight_ptrs_.data(), request.topk, request.d_model,
        request.d_ff, request.d_model, team_threads);
  }
  if (!ok)
    return false;

  const auto math_end = std::chrono::steady_clock::now();
  std::fill_n(request.output, request.d_model, 0.0f);
  for (int m = 0; m < request.topk; ++m) {
    const float weight = request.router_weights[m];
    const float *down = direct_fused_down_output_ptrs_[static_cast<size_t>(m)];
    for (int d = 0; d < request.d_model; ++d)
      request.output[d] += weight * down[d];
  }
  const auto end = std::chrono::steady_clock::now();
  if (qwen_cpu_direct_timing_enabled()) {
    const double math_ms = std::chrono::duration<double, std::milli>(
                               math_end - start)
                               .count();
    const double reduce_ms = std::chrono::duration<double, std::milli>(
                                 end - math_end)
                                 .count();
    const double total_ms = std::chrono::duration<double, std::milli>(
                                end - start)
                                .count();
    LOG_INFO("[QWEN-CPU-DIRECT-FUSED] math=%.3f reduce=%.3f total=%.3f "
             "threads=%d type=%s topk=%d\n",
             math_ms, reduce_ms, total_ms, team_threads,
             request.layer.gate_type == GGML_TYPE_Q2_0
                 ? "Q2_0"
                 : (request.layer.gate_type == GGML_TYPE_Q4_0 ? "Q4_0"
                 : (request.layer.gate_type == GGML_TYPE_IQ1_M ? "IQ1_M"
                                                                : "IQ2_XXS")),
             request.topk);
  }
  if (request.completion)
    request.completion(request.completion_user, true);
  return true;
}

bool GgmlQwenCpuMoeRunner::submit_direct_request(
    const AsyncStridedRequest &request) {
  if (!direct_ready_)
    return false;
  ensure_direct_workers();
  direct_armed_.store(true, std::memory_order_release);

  if (!direct_finished_.load(std::memory_order_acquire))
    return false;
  {
    std::lock_guard<std::mutex> lock(async_mutex_);
    if (async_request_pending_)
      return false;
    async_request_ = request;
    async_result_ = false;
    async_request_done_ = false;
    async_request_pending_ = true;
    direct_request_ok_.store(true, std::memory_order_relaxed);
    direct_result_.store(false, std::memory_order_relaxed);
    direct_finished_.store(false, std::memory_order_release);
    direct_completed_.store(0, std::memory_order_relaxed);
  }

  // Flatten the routed rows once per request.  The workers then perform one
  // indexed load per dot instead of recomputing expert and row offsets inside
  // the hot projection loops.  These vectors were sized at signature setup,
  // so this preparation performs no decode-time allocation.
  const size_t gate_row_bytes =
      qwen_cpu_direct_gate_row_bytes(request.layer.gate_type, request.d_model);
  const size_t up_row_bytes =
      qwen_cpu_direct_gate_row_bytes(request.layer.up_type, request.d_model);
  const size_t down_row_bytes =
      request.layer.down_type == GGML_TYPE_Q2_0
          ? static_cast<size_t>(request.d_ff / 64) * 18
          : static_cast<size_t>(request.d_ff / 32) * 18;
  for (int m = 0; m < request.topk; ++m) {
    const int expert_id = request.expert_ids[m];
    const uint8_t *gate_base = static_cast<const uint8_t *>(request.gate_plane) +
                               static_cast<size_t>(expert_id) *
                                   request.expert_stride;
    const uint8_t *up_base = static_cast<const uint8_t *>(request.up_plane) +
                             static_cast<size_t>(expert_id) *
                                 request.up_expert_stride;
    const uint8_t *down_base =
        static_cast<const uint8_t *>(request.down_plane) +
        static_cast<size_t>(expert_id) * request.down_expert_stride;
    for (int o = 0; o < request.d_ff; ++o) {
      const size_t index = static_cast<size_t>(m) * request.d_ff + o;
      direct_gate_rows_[index] =
          gate_base + static_cast<size_t>(o) * gate_row_bytes;
      direct_up_rows_[index] =
          up_base + static_cast<size_t>(o) * up_row_bytes;
    }
    for (int d = 0; d < request.d_model; ++d)
      direct_down_rows_[static_cast<size_t>(m) * request.d_model + d] =
          down_base + static_cast<size_t>(d) * down_row_bytes;
  }

  // The input quantization is only one 2.5K-element row.  Do it on the
  // callback thread before publishing the first worker phase instead of
  // waking worker 0 for a serial five-microsecond task and then publishing
  // Gate/Up from that worker.  The workers now start directly on the first
  // parallel phase, which removes an avoidable wake/phase handoff from every
  // host-node replay.
  const auto input_start = std::chrono::steady_clock::now();
  quantize_row_q8_K(request.input,
                    reinterpret_cast<qwen_direct_q8_k_block *>(
                        direct_input_q8k_.data()),
                    request.d_model);
  direct_timing_input_ns_.store(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now() - input_start)
          .count(),
      std::memory_order_relaxed);
  const uint64_t phase = direct_phase_.load(std::memory_order_relaxed);
  direct_phase_.store(direct_next_phase(phase, QWEN_DIRECT_GATE_UP),
                      std::memory_order_release);
  direct_phase_.notify_all();
  return true;
}

void GgmlQwenCpuMoeRunner::run_direct_caller_phases(
    const AsyncStridedRequest &request) {
  // The CUDA host-function thread is already occupied until this callback
  // returns.  Let it own participant 0 of the direct team instead of making
  // it spin while all worker threads execute the whole request.  The other
  // participants keep the same phase protocol as the worker-only path; the
  // caller is simply the final participant that advances each phase.
  const int nth = direct_threads_ + 1;
  const int participant = 0;
  const size_t gate_row_bytes =
      qwen_cpu_direct_gate_row_bytes(request.layer.gate_type,
                                     request.d_model);
  const size_t up_row_bytes =
      qwen_cpu_direct_gate_row_bytes(request.layer.up_type,
                                     request.d_model);
  const size_t down_row_bytes = static_cast<size_t>(request.d_ff / 32) * 18;
  auto *input_q = reinterpret_cast<qwen_direct_q8_k_block *>(
      direct_input_q8k_.data());
  auto *down_q = reinterpret_cast<block_q8_0 *>(direct_down_q8_0_.data());
  const bool report_timing = qwen_cpu_direct_timing_enabled();
  const bool report_phase_diag = qwen_cpu_direct_diag_enabled();
  const bool prefetch_rows = qwen_cpu_direct_prefetch_enabled();
  const auto now_ns = [] {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
  };

  auto wait_and_advance = [&](int command, int next_command, bool final) {
    // In caller-worker mode the callback owns participant 0, but it is also
    // the phase coordinator.  Count only the native workers here; letting the
    // callback increment the same counter makes a worker-side early advance
    // indistinguishable from a completed five-way team and can leave the
    // callback waiting on a phase that has already been reset.
    const int expected_workers = direct_threads_;
    const int completed =
        direct_completed_.load(std::memory_order_acquire);
    if (report_phase_diag)
      LOG_INFO("[QWEN-CPU-DIRECT-CALLER] command=%d worker_done=%d/%d\n",
               command, completed, expected_workers);
    while (direct_completed_.load(std::memory_order_acquire) <
           expected_workers)
      qwen_cpu_direct_pause();

    direct_completed_.store(0, std::memory_order_relaxed);
    const uint64_t phase = direct_phase_.load(std::memory_order_acquire);
    if (direct_phase_command(phase) != command)
      return;

    if (report_timing && command == QWEN_DIRECT_GATE_UP) {
      const int64_t start =
          direct_timing_phase1_start_ns_.load(std::memory_order_relaxed);
      direct_timing_phase1_ns_.store(now_ns() - start,
                                     std::memory_order_relaxed);
    } else if (report_timing && command == QWEN_DIRECT_DOWN) {
      const int64_t start =
          direct_timing_down_start_ns_.load(std::memory_order_relaxed);
      direct_timing_down_ns_.store(now_ns() - start,
                                   std::memory_order_relaxed);
    }

    direct_phase_.store(direct_next_phase(phase, next_command),
                        std::memory_order_release);
    direct_phase_.notify_all();
    if (report_phase_diag)
      LOG_INFO("[QWEN-CPU-DIRECT-CALLER] command=%d advance=%d final=%d\n",
               command, next_command, final ? 1 : 0);
    if (final) {
      if (report_timing) {
        const int64_t start =
            direct_timing_reduce_start_ns_.load(std::memory_order_relaxed);
        direct_timing_reduce_ns_.store(now_ns() - start,
                                       std::memory_order_relaxed);
      }
      const bool result = direct_request_ok_.load(std::memory_order_acquire);
      if (request.completion)
        request.completion(request.completion_user, result);
      {
        std::lock_guard<std::mutex> lock(async_mutex_);
        async_result_ = result;
        async_request_pending_ = false;
        async_request_done_ = true;
      }
      direct_result_.store(result, std::memory_order_release);
      direct_phase_.store(direct_next_phase(phase, QWEN_DIRECT_IDLE),
                          std::memory_order_release);
      direct_finished_.store(true, std::memory_order_release);
      direct_phase_.notify_all();
      async_cv_.notify_all();
    }
    (void) completed;
  };

  while (direct_phase_command(direct_phase_.load(std::memory_order_acquire)) !=
         QWEN_DIRECT_GATE_UP)
    qwen_cpu_direct_pause();
  if (report_phase_diag)
    LOG_INFO("[QWEN-CPU-DIRECT-CALLER] gate begin workers=%d\n",
             direct_threads_);
  if (report_timing)
    direct_timing_phase1_start_ns_.store(now_ns(),
                                         std::memory_order_relaxed);

  const int64_t hidden_blocks =
      static_cast<int64_t>(request.topk) * (request.d_ff / 32);
  const int64_t blocks_per_participant =
      (hidden_blocks + nth - 1) / nth;
  const int64_t first_block =
      std::min<int64_t>(hidden_blocks, participant * blocks_per_participant);
  const int64_t last_block = std::min<int64_t>(
      hidden_blocks, first_block + blocks_per_participant);
  for (int64_t block = first_block; block < last_block; ++block) {
    const int m = static_cast<int>(block / (request.d_ff / 32));
    const int o0 = static_cast<int>(block % (request.d_ff / 32)) * 32;
    for (int j = 0; j < 32; ++j) {
      const int o = o0 + j;
      const size_t row_index = static_cast<size_t>(m) * request.d_ff + o;
      const uint8_t *gate_row = direct_gate_rows_[row_index];
      const uint8_t *up_row = direct_up_rows_[row_index];
      if (prefetch_rows && row_index + 2 < direct_gate_rows_.size()) {
        qwen_cpu_direct_prefetch_row(direct_gate_rows_[row_index + 2],
                                     gate_row_bytes);
        qwen_cpu_direct_prefetch_row(direct_up_rows_[row_index + 2],
                                     up_row_bytes);
      }
      float *gate_out = direct_gate_output_.data() + row_index;
      float *up_out = direct_up_output_.data() + row_index;
      switch (request.layer.gate_type) {
      case GGML_TYPE_IQ1_M:
        ggml_vec_dot_iq1_m_q8_K(request.d_model, gate_out, 0, gate_row, 0,
                                input_q, 0, 1);
        break;
      case GGML_TYPE_IQ2_XXS:
        ggml_vec_dot_iq2_xxs_q8_K(request.d_model, gate_out, 0, gate_row, 0,
                                  input_q, 0, 1);
        break;
      case GGML_TYPE_IQ3_XXS:
        ggml_vec_dot_iq3_xxs_q8_K(request.d_model, gate_out, 0, gate_row, 0,
                                  input_q, 0, 1);
        break;
      default:
        direct_request_ok_.store(false, std::memory_order_release);
        break;
      }
      switch (request.layer.up_type) {
      case GGML_TYPE_IQ1_M:
        ggml_vec_dot_iq1_m_q8_K(request.d_model, up_out, 0, up_row, 0,
                                input_q, 0, 1);
        break;
      case GGML_TYPE_IQ2_XXS:
        ggml_vec_dot_iq2_xxs_q8_K(request.d_model, up_out, 0, up_row, 0,
                                  input_q, 0, 1);
        break;
      case GGML_TYPE_IQ3_XXS:
        ggml_vec_dot_iq3_xxs_q8_K(request.d_model, up_out, 0, up_row, 0,
                                  input_q, 0, 1);
        break;
      default:
        direct_request_ok_.store(false, std::memory_order_release);
        break;
      }
    }
    float *hidden = direct_gate_output_.data() +
                    static_cast<size_t>(m) * request.d_ff + o0;
    const float *up = direct_up_output_.data() +
                      static_cast<size_t>(m) * request.d_ff + o0;
    silu_inplace(hidden, 32);
    elem_mul(hidden, hidden, up, 32);
    quantize_row_q8_0(hidden, down_q + block, 32);
  }
  wait_and_advance(QWEN_DIRECT_GATE_UP, QWEN_DIRECT_DOWN, false);

  while (direct_phase_command(direct_phase_.load(std::memory_order_acquire)) !=
         QWEN_DIRECT_DOWN)
    qwen_cpu_direct_pause();
  if (report_phase_diag)
    LOG_INFO("[QWEN-CPU-DIRECT-CALLER] down begin\n");
  if (report_timing)
    direct_timing_down_start_ns_.store(now_ns(), std::memory_order_relaxed);
  const int rows_per_participant =
      (request.d_model + nth - 1) / nth;
  const int first_down =
      std::min(request.d_model, participant * rows_per_participant);
  const int last_down = std::min(request.d_model,
                                 first_down + rows_per_participant);
  const int local_rows = std::max(0, last_down - first_down);
  float *local_sums = static_cast<float *>(
      alloca(static_cast<size_t>(local_rows) * sizeof(float)));
  std::memset(local_sums, 0, static_cast<size_t>(local_rows) * sizeof(float));
  for (int m = 0; m < request.topk; ++m) {
    const float route_weight = request.router_weights[m];
    const size_t row_base = static_cast<size_t>(m) * request.d_model;
    const auto *expert_hidden =
        down_q + static_cast<size_t>(m) * (request.d_ff / 32);
    for (int d = first_down; d < last_down; ++d) {
      if (prefetch_rows && d + 3 < last_down)
        qwen_cpu_direct_prefetch_row(
            direct_down_rows_[row_base + static_cast<size_t>(d + 3)],
            down_row_bytes);
      const uint8_t *down_row =
          direct_down_rows_[row_base + static_cast<size_t>(d)];
      float down_val = 0.0f;
      ggml_vec_dot_iq4_nl_q8_0(request.d_ff, &down_val, 0, down_row, 0,
                                expert_hidden, 0, 1);
      local_sums[d - first_down] += down_val * route_weight;
    }
  }
  for (int d = first_down; d < last_down; ++d)
    direct_down_output_[d] = local_sums[d - first_down];
  wait_and_advance(QWEN_DIRECT_DOWN, QWEN_DIRECT_REDUCE, false);

  while (direct_phase_command(direct_phase_.load(std::memory_order_acquire)) !=
         QWEN_DIRECT_REDUCE)
    qwen_cpu_direct_pause();
  if (report_phase_diag)
    LOG_INFO("[QWEN-CPU-DIRECT-CALLER] reduce begin\n");
  if (report_timing)
    direct_timing_reduce_start_ns_.store(now_ns(),
                                         std::memory_order_relaxed);
  const int reduce_per_participant =
      (request.d_model + nth - 1) / nth;
  const int first_reduce =
      std::min(request.d_model, participant * reduce_per_participant);
  const int last_reduce = std::min(request.d_model,
                                   first_reduce + reduce_per_participant);
  for (int d = first_reduce; d < last_reduce; ++d)
    request.output[d] = direct_down_output_[d];
  wait_and_advance(QWEN_DIRECT_REDUCE, QWEN_DIRECT_IDLE, true);
}

bool GgmlQwenCpuMoeRunner::run_direct_request(
    const AsyncStridedRequest &request) {
  const bool fused_eligible =
      request.layer.gate_type == request.layer.up_type &&
      ((request.layer.gate_type == GGML_TYPE_Q4_0 &&
        request.layer.down_type == GGML_TYPE_Q4_0) ||
       (request.layer.gate_type == GGML_TYPE_Q2_0 &&
        request.layer.down_type == GGML_TYPE_Q2_0) ||
       ((request.layer.gate_type == GGML_TYPE_IQ1_M ||
         request.layer.gate_type == GGML_TYPE_IQ2_XXS) &&
        request.layer.down_type == GGML_TYPE_IQ4_NL));
  if (qwen_cpu_direct_fused_enabled() && fused_eligible) {
    if (run_direct_fused_request(request))
      return true;
  }
  const auto wall_start = std::chrono::steady_clock::now();
  const bool caller_worker = qwen_cpu_direct_caller_worker_enabled();
  if (!submit_direct_request(request))
    return false;
  const auto submit_end = std::chrono::steady_clock::now();
  if (caller_worker)
    run_direct_caller_phases(request);
  while (!direct_finished_.load(std::memory_order_acquire))
    qwen_cpu_direct_pause();
  const auto finish_end = std::chrono::steady_clock::now();
  const bool result = direct_result_.load(std::memory_order_acquire);

  // Stop the active window before emitting the optional per-worker dump.  The
  // dump can contain many log calls; leaving the team armed while it runs
  // would keep workers spinning and would contaminate the teardown timing.
  direct_armed_.store(false, std::memory_order_release);
  direct_phase_.notify_all();
  if (qwen_cpu_direct_ephemeral_env_enabled())
    stop_direct_pool();
  const auto teardown_end = std::chrono::steady_clock::now();

  if (qwen_cpu_direct_thread_diag_enabled()) {
    const auto ms = [](int64_t start, int64_t end) {
      return static_cast<double>(end - start) / 1000000.0;
    };
    const auto span_ms = [](const auto &samples,
                            int64_t GgmlQwenDirectWorkerSample::*field) {
      int64_t minimum = INT64_MAX;
      int64_t maximum = INT64_MIN;
      for (const auto &sample : samples) {
        const int64_t value = sample.*field;
        if (value == 0)
          continue;
        minimum = std::min(minimum, value);
        maximum = std::max(maximum, value);
      }
      return minimum == INT64_MAX
                 ? 0.0
                 : static_cast<double>(maximum - minimum) / 1000000.0;
    };
    LOG_INFO(
        "[QWEN-CPU-DIRECT-PHASE-SKEW] gate_start_span=%.3fms "
        "gate_end_span=%.3fms down_start_span=%.3fms "
        "down_end_span=%.3fms reduce_start_span=%.3fms "
        "reduce_end_span=%.3fms\n",
        span_ms(direct_gate_samples_,
                &GgmlQwenDirectWorkerSample::wall_start_ns),
        span_ms(direct_gate_samples_,
                &GgmlQwenDirectWorkerSample::wall_end_ns),
        span_ms(direct_down_samples_,
                &GgmlQwenDirectWorkerSample::wall_start_ns),
        span_ms(direct_down_samples_,
                &GgmlQwenDirectWorkerSample::wall_end_ns),
        span_ms(direct_reduce_samples_,
                &GgmlQwenDirectWorkerSample::wall_start_ns),
        span_ms(direct_reduce_samples_,
                &GgmlQwenDirectWorkerSample::wall_end_ns));
    for (int worker = 0; worker < direct_threads_; ++worker) {
      const auto &gate = direct_gate_samples_[static_cast<size_t>(worker)];
      const auto &down = direct_down_samples_[static_cast<size_t>(worker)];
      const auto &reduce =
          direct_reduce_samples_[static_cast<size_t>(worker)];
      LOG_INFO(
          "[QWEN-CPU-DIRECT-THREAD] worker=%d "
          "gate_wall=%.3f gate_cpu=%.3f gate_cs=%ld/%ld gate_cpu_id=%d->%d "
          "down_wall=%.3f down_cpu=%.3f down_cs=%ld/%ld down_cpu_id=%d->%d "
          "reduce_wall=%.3f reduce_cpu=%.3f reduce_cs=%ld/%ld "
          "reduce_cpu_id=%d->%d\n",
          worker, ms(gate.wall_start_ns, gate.wall_end_ns),
          ms(gate.cpu_start_ns, gate.cpu_end_ns),
          gate.voluntary_end - gate.voluntary_start,
          gate.involuntary_end - gate.involuntary_start, gate.cpu_start,
          gate.cpu_end, ms(down.wall_start_ns, down.wall_end_ns),
          ms(down.cpu_start_ns, down.cpu_end_ns),
          down.voluntary_end - down.voluntary_start,
          down.involuntary_end - down.involuntary_start, down.cpu_start,
          down.cpu_end, ms(reduce.wall_start_ns, reduce.wall_end_ns),
          ms(reduce.cpu_start_ns, reduce.cpu_end_ns),
          reduce.voluntary_end - reduce.voluntary_start,
          reduce.involuntary_end - reduce.involuntary_start,
          reduce.cpu_start, reduce.cpu_end);
    }
  }
  if (qwen_cpu_direct_timing_enabled()) {
    const double submit_ms = std::chrono::duration<double, std::milli>(
                                 submit_end - wall_start)
                                 .count();
    const double spin_wait_ms = std::chrono::duration<double, std::milli>(
                                    finish_end - submit_end)
                                    .count();
    const double teardown_ms = std::chrono::duration<double, std::milli>(
                                   teardown_end - finish_end)
                                   .count();
    const double total_ms = std::chrono::duration<double, std::milli>(
                                teardown_end - wall_start)
                                .count();
    LOG_INFO("[DIAG-CALLBACK] submit=%.3f ms, spin_wait=%.3f ms, "
             "teardown=%.3f ms, total=%.3f ms, threads=%d caller_worker=%d "
             "result=%d\n",
             submit_ms, spin_wait_ms, teardown_ms, total_ms, direct_threads_,
             caller_worker ? 1 : 0, result ? 1 : 0);
    // Keep the older tag for scripts that already consume the direct-pool
    // ledger.  The new fields above are the precise queue/math/teardown
    // partition requested for the oversubscription sweep.
    LOG_INFO("[QWEN-CPU-DIRECT-WALL] prepare_submit=%.3f "
             "worker_wait=%.3f teardown=%.3f total=%.3f threads=%d result=%d\n",
             submit_ms, spin_wait_ms, teardown_ms, total_ms, direct_threads_,
             result ? 1 : 0);
  }
  return result;
}

void GgmlQwenCpuMoeRunner::direct_worker_loop(int worker_index) {
  qwen_cpu_direct_pin_thread(worker_index);
  uint64_t seen_phase = 0;
  for (;;) {
    while (!direct_stop_.load(std::memory_order_acquire)) {
      const uint64_t phase = direct_phase_.load(std::memory_order_acquire);
      if (phase != seen_phase)
        break;
      // Park only between requests.  During an active request, a fast worker
      // may reach the phase boundary while the last worker is still finishing
      // its chunk.  Waiting on a futex here turns every gate->down and
      // down->reduce handoff into an OS wake-up and was measured as the
      // missing milliseconds in the host callback.  Active phases therefore
      // remain a pure pause loop; the idle state can still use an efficient
      // atomic wait so the team does not steal cores between tokens.
      if (qwen_cpu_direct_idle_wait_enabled() &&
          !direct_armed_.load(std::memory_order_acquire) &&
          direct_phase_command(phase) == QWEN_DIRECT_IDLE) {
        direct_phase_.wait(seen_phase, std::memory_order_acquire);
      } else {
        qwen_cpu_direct_pause();
      }
    }
    if (direct_stop_.load(std::memory_order_acquire))
      return;

    const uint64_t phase = direct_phase_.load(std::memory_order_acquire);
    seen_phase = phase;
    const AsyncStridedRequest request = async_request_;
    // Caller-worker mode is a process-wide, opt-in execution mode.  Read the
    // immutable configuration here instead of sampling the lifecycle flag
    // while a request is in flight.  The lifecycle flag is cleared as soon as
    // the callback's final participant has finished; a worker that observes
    // that clear while it is transitioning out of the final phase could then
    // derive nth=direct_threads_ instead of nth=direct_threads_+1 and advance
    // the phase early.  The callback would subsequently wait forever for a
    // participant that no longer exists.
    const bool caller_worker = qwen_cpu_direct_caller_worker_enabled();
    const int nth = direct_threads_ + (caller_worker ? 1 : 0);
    const int participant = worker_index + (caller_worker ? 1 : 0);
    const size_t gate_row_bytes =
        qwen_cpu_direct_gate_row_bytes(request.layer.gate_type,
                                       request.d_model);
    const size_t up_row_bytes =
        qwen_cpu_direct_gate_row_bytes(request.layer.up_type,
                                       request.d_model);
    const size_t down_row_bytes =
        static_cast<size_t>(request.d_ff / 32) * 18;
    const int command = direct_phase_command(phase);
    auto *input_q = reinterpret_cast<qwen_direct_q8_k_block *>(
        direct_input_q8k_.data());
    auto *down_q = reinterpret_cast<block_q8_0 *>(direct_down_q8_0_.data());
    const bool report_timing = qwen_cpu_direct_timing_enabled();
    const bool report_worker_diag = qwen_cpu_direct_thread_diag_enabled();
    const bool prefetch_rows = qwen_cpu_direct_prefetch_enabled();
    const auto now_ns = [] {
      return std::chrono::duration_cast<std::chrono::nanoseconds>(
                 std::chrono::steady_clock::now().time_since_epoch())
          .count();
    };

    if (command == QWEN_DIRECT_INPUT) {
      if (worker_index == 0) {
        const int64_t start = now_ns();
        quantize_row_q8_K(request.input, input_q, request.d_model);
        direct_timing_input_ns_.store(now_ns() - start,
                                      std::memory_order_relaxed);
        direct_request_ok_.store(true, std::memory_order_release);
        direct_completed_.store(0, std::memory_order_relaxed);
        direct_phase_.store(direct_next_phase(phase, QWEN_DIRECT_GATE_UP),
                            std::memory_order_release);
        direct_phase_.notify_all();
      }
      continue;
    }

    if (command == QWEN_DIRECT_GATE_UP) {
      if (worker_index < static_cast<int>(direct_gate_samples_.size()))
        qwen_cpu_direct_sample_begin(direct_gate_samples_[worker_index],
                                     report_worker_diag);
      if (worker_index == 0 && report_timing)
        direct_timing_phase1_start_ns_.store(now_ns(),
                                             std::memory_order_relaxed);

      // Assign complete Q8_0 blocks to workers. Each worker computes gate/up,
      // applies the AVX-512 SwiGLU, and quantizes its own hidden blocks.
      const int64_t hidden_blocks = static_cast<int64_t>(request.topk) *
                                    (request.d_ff / 32);
      const int64_t blocks_per_worker =
          (hidden_blocks + nth - 1) / nth;
      const int64_t first_block =
          std::min<int64_t>(hidden_blocks, participant * blocks_per_worker);
      const int64_t last_block =
          std::min<int64_t>(hidden_blocks, first_block + blocks_per_worker);
      for (int64_t block = first_block; block < last_block; ++block) {
        const int m = static_cast<int>(block / (request.d_ff / 32));
        const int o0 = static_cast<int>(block % (request.d_ff / 32)) * 32;
        for (int j = 0; j < 32; ++j) {
          const int o = o0 + j;
          const size_t row_index = static_cast<size_t>(m) * request.d_ff + o;
          const uint8_t *gate_row = direct_gate_rows_[row_index];
          const uint8_t *up_row = direct_up_rows_[row_index];
          if (prefetch_rows && row_index + 2 < direct_gate_rows_.size()) {
            qwen_cpu_direct_prefetch_row(direct_gate_rows_[row_index + 2],
                                         gate_row_bytes);
            qwen_cpu_direct_prefetch_row(direct_up_rows_[row_index + 2],
                                         up_row_bytes);
          }
          float *gate_out = direct_gate_output_.data() + row_index;
          float *up_out = direct_up_output_.data() + row_index;
          switch (request.layer.gate_type) {
          case GGML_TYPE_IQ1_M:
            ggml_vec_dot_iq1_m_q8_K(request.d_model, gate_out, 0, gate_row,
                                    0, input_q, 0, 1);
            break;
          case GGML_TYPE_IQ2_XXS:
            ggml_vec_dot_iq2_xxs_q8_K(request.d_model, gate_out, 0, gate_row,
                                      0, input_q, 0, 1);
            break;
          case GGML_TYPE_IQ3_XXS:
            ggml_vec_dot_iq3_xxs_q8_K(request.d_model, gate_out, 0, gate_row,
                                      0, input_q, 0, 1);
            break;
          default:
            direct_request_ok_.store(false, std::memory_order_release);
            break;
          }
          switch (request.layer.up_type) {
          case GGML_TYPE_IQ1_M:
            ggml_vec_dot_iq1_m_q8_K(request.d_model, up_out, 0, up_row, 0,
                                    input_q, 0, 1);
            break;
          case GGML_TYPE_IQ2_XXS:
            ggml_vec_dot_iq2_xxs_q8_K(request.d_model, up_out, 0, up_row, 0,
                                      input_q, 0, 1);
            break;
          case GGML_TYPE_IQ3_XXS:
            ggml_vec_dot_iq3_xxs_q8_K(request.d_model, up_out, 0, up_row, 0,
                                      input_q, 0, 1);
            break;
          default:
            direct_request_ok_.store(false, std::memory_order_release);
            break;
          }
        }
        float *hidden = direct_gate_output_.data() +
                        static_cast<size_t>(m) * request.d_ff + o0;
        const float *up = direct_up_output_.data() +
                          static_cast<size_t>(m) * request.d_ff + o0;
        silu_inplace(hidden, 32);
        elem_mul(hidden, hidden, up, 32);
        quantize_row_q8_0(hidden, down_q + block, 32);
      }

      if (worker_index < static_cast<int>(direct_gate_samples_.size()))
        qwen_cpu_direct_sample_end(direct_gate_samples_[worker_index],
                                   report_worker_diag);

      const int completed =
          direct_completed_.fetch_add(1, std::memory_order_acq_rel) + 1;
      if (!caller_worker && completed == nth) {
        if (report_timing) {
          const int64_t start = direct_timing_phase1_start_ns_.load(
              std::memory_order_relaxed);
          direct_timing_phase1_ns_.store(now_ns() - start,
                                         std::memory_order_relaxed);
        }
        direct_completed_.store(0, std::memory_order_relaxed);
        direct_phase_.store(direct_next_phase(phase, QWEN_DIRECT_DOWN),
                            std::memory_order_release);
        direct_phase_.notify_all();
      }
      continue;
    }

    if (command == QWEN_DIRECT_DOWN) {
      if (worker_index < static_cast<int>(direct_down_samples_.size()))
        qwen_cpu_direct_sample_begin(direct_down_samples_[worker_index],
                                     report_worker_diag);
      if (worker_index == 0 && report_timing)
        direct_timing_down_start_ns_.store(now_ns(),
                                           std::memory_order_relaxed);
      const int rows_per_worker =
          (request.d_model + nth - 1) / nth;
      const int first_down =
          std::min(request.d_model, participant * rows_per_worker);
      const int last_down =
          std::min(request.d_model, first_down + rows_per_worker);

      // Keep the reduction in a worker-local L1-sized buffer.  The previous
      // implementation wrote one raw result per expert and performed a
      // second top-k reduction pass, which added both memory traffic and a
      // separate phase over the same output range.
      const int local_rows = std::max(0, last_down - first_down);
      float *local_sums = static_cast<float *>(alloca(
          static_cast<size_t>(local_rows) * sizeof(float)));
      std::memset(local_sums, 0,
                  static_cast<size_t>(local_rows) * sizeof(float));

      for (int m = 0; m < request.topk; ++m) {
        const float route_weight = request.router_weights[m];
        const size_t row_base = static_cast<size_t>(m) * request.d_model;
        const auto *expert_hidden =
            down_q + static_cast<size_t>(m) * (request.d_ff / 32);
        for (int d = first_down; d < last_down; ++d) {
          if (prefetch_rows && d + 3 < last_down)
            qwen_cpu_direct_prefetch_row(
                direct_down_rows_[row_base + static_cast<size_t>(d + 3)],
                down_row_bytes);
          const uint8_t *down_row =
              direct_down_rows_[row_base + static_cast<size_t>(d)];
          float down_val = 0.0f;
          ggml_vec_dot_iq4_nl_q8_0(
              request.d_ff, &down_val, 0, down_row, 0, expert_hidden, 0, 1);
          local_sums[d - first_down] += down_val * route_weight;
        }
      }
      for (int d = first_down; d < last_down; ++d)
        direct_down_output_[d] = local_sums[d - first_down];
      if (worker_index < static_cast<int>(direct_down_samples_.size()))
        qwen_cpu_direct_sample_end(direct_down_samples_[worker_index],
                                   report_worker_diag);
      const int completed =
          direct_completed_.fetch_add(1, std::memory_order_acq_rel) + 1;
      if (!caller_worker && completed == nth) {
        if (report_timing) {
          const int64_t start = direct_timing_down_start_ns_.load(
              std::memory_order_relaxed);
          direct_timing_down_ns_.store(now_ns() - start,
                                       std::memory_order_relaxed);
        }
        direct_completed_.store(0, std::memory_order_relaxed);
        direct_phase_.store(direct_next_phase(phase, QWEN_DIRECT_REDUCE),
                            std::memory_order_release);
        direct_phase_.notify_all();
      }
      continue;
    }

    if (command == QWEN_DIRECT_REDUCE) {
      if (worker_index < static_cast<int>(direct_reduce_samples_.size()))
        qwen_cpu_direct_sample_begin(direct_reduce_samples_[worker_index],
                                     report_worker_diag);
      if (worker_index == 0 && report_timing)
        direct_timing_reduce_start_ns_.store(now_ns(),
                                             std::memory_order_relaxed);
      const int reduce_per_worker = (request.d_model + nth - 1) / nth;
      const int first_reduce =
          std::min(request.d_model, participant * reduce_per_worker);
      const int last_reduce =
          std::min(request.d_model, first_reduce + reduce_per_worker);
      for (int d = first_reduce; d < last_reduce; ++d)
        request.output[d] = direct_down_output_[d];

      if (worker_index < static_cast<int>(direct_reduce_samples_.size()))
        qwen_cpu_direct_sample_end(direct_reduce_samples_[worker_index],
                                   report_worker_diag);

      const int completed =
          direct_completed_.fetch_add(1, std::memory_order_acq_rel) + 1;
      if (!caller_worker && completed == nth) {
        if (report_timing) {
          const int64_t start = direct_timing_reduce_start_ns_.load(
              std::memory_order_relaxed);
          direct_timing_reduce_ns_.store(now_ns() - start,
                                         std::memory_order_relaxed);
          const auto ms = [](int64_t ns) {
            return static_cast<double>(ns) / 1000000.0;
          };
          LOG_INFO("[QWEN-CPU-DIRECT-TIMING] input_q8k=%.3f "
                   "gate_up_swiglu_q8_0=%.3f down=%.3f reduce_store=%.3f "
                   "total=%.3f d_model=%d d_ff=%d topk=%d\n",
                   ms(direct_timing_input_ns_.load()),
                   ms(direct_timing_phase1_ns_.load()),
                   ms(direct_timing_down_ns_.load()),
                   ms(direct_timing_reduce_ns_.load()),
                   ms(direct_timing_input_ns_.load() +
                      direct_timing_phase1_ns_.load() +
                      direct_timing_down_ns_.load() +
                      direct_timing_reduce_ns_.load()),
                   request.d_model, request.d_ff, request.topk);
        }
        const bool result = direct_request_ok_.load(std::memory_order_acquire);
        AsyncCompletionFn completion = request.completion;
        void *completion_user = request.completion_user;
        if (completion)
          completion(completion_user, result);
        {
          std::lock_guard<std::mutex> lock(async_mutex_);
          async_result_ = result;
          async_request_pending_ = false;
          async_request_done_ = true;
        }
        direct_result_.store(result, std::memory_order_release);
        direct_phase_.store(direct_next_phase(phase, QWEN_DIRECT_IDLE),
                            std::memory_order_release);
        direct_finished_.store(true, std::memory_order_release);
        direct_phase_.notify_all();
        async_cv_.notify_all();
      }
      continue;
    }
  }
}

void GgmlQwenCpuMoeRunner::stop_direct_pool() {
  if (direct_workers_.empty())
    return;
  direct_stop_.store(true, std::memory_order_release);
  direct_armed_.store(false, std::memory_order_release);
  const uint64_t phase = direct_phase_.load(std::memory_order_relaxed);
  direct_phase_.store(direct_next_phase(phase, QWEN_DIRECT_IDLE),
                      std::memory_order_release);
  direct_phase_.notify_all();
  for (auto &worker : direct_workers_)
    if (worker.joinable())
      worker.join();
  direct_workers_.clear();
  direct_phase_.store(0, std::memory_order_relaxed);
  direct_completed_.store(0, std::memory_order_relaxed);
  direct_finished_.store(true, std::memory_order_relaxed);
  direct_stop_.store(false, std::memory_order_relaxed);
}

bool GgmlQwenCpuMoeRunner::ensure_graph(
    const GgmlQwenCpuExpertRef &ref, int n_experts, int d_model, int d_ff,
    int topk, bool strided, const void *gate_plane, const void *up_plane,
    const void *down_plane, size_t expert_stride, bool output_per_expert,
    bool merged_gate_up, size_t up_expert_stride,
    size_t down_expert_stride) {
  const size_t effective_up_stride =
      up_expert_stride == 0 ? expert_stride : up_expert_stride;
  const size_t effective_down_stride =
      down_expert_stride == 0 ? expert_stride : down_expert_stride;
  if (!type_is_valid(ref.gate_type) || !type_is_valid(ref.up_type) ||
      !type_is_valid(ref.down_type) || n_experts <= 0 || d_model <= 0 ||
      d_ff <= 0 || topk <= 0 || topk > n_experts ||
      (strided &&
       (!gate_plane || !up_plane || !down_plane || expert_stride == 0 ||
        effective_up_stride == 0 || effective_down_stride == 0)) ||
      (!strided && topk != n_experts) ||
      (merged_gate_up &&
       (ref.gate_type != ref.up_type || ref.gate_bytes != ref.up_bytes ||
        (strided && effective_up_stride != expert_stride))))
    return false;

  const bool shape_matches =
      graph_ && n_experts_ == n_experts && d_model_ == d_model &&
      d_ff_ == d_ff && topk_ == topk && gate_type_ == ref.gate_type &&
      up_type_ == ref.up_type && down_type_ == ref.down_type &&
      strided_mode_ == strided && output_per_expert_ == output_per_expert &&
      merged_gate_up_ == merged_gate_up &&
      (!strided || (expert_stride_ == expert_stride &&
                    up_expert_stride_ == effective_up_stride &&
                    down_expert_stride_ == effective_down_stride));
  if (shape_matches)
    return true;

  free_graph();
  if (!cpu_backend_) {
    cpu_backend_ = ggml_backend_cpu_init();
    if (!cpu_backend_)
      return false;
    if (!cpu_threadpool_borrowed_)
      n_threads_ = default_cpu_threads();
    ggml_backend_cpu_set_n_threads(static_cast<ggml_backend_t>(cpu_backend_),
                                   n_threads_);
  }
  if (cpu_backend_ && cpu_threadpool_)
    ggml_backend_cpu_set_threadpool(
        static_cast<ggml_backend_t>(cpu_backend_),
        static_cast<ggml_threadpool_t>(cpu_threadpool_));
  if (!cpu_threadpool_) {
    if (n_threads_ <= 0) {
      n_threads_ = default_cpu_threads();
      ggml_backend_cpu_set_n_threads(
          static_cast<ggml_backend_t>(cpu_backend_), n_threads_);
    }
    ggml_threadpool_params params =
        ggml_threadpool_params_default(n_threads_);
    cpu_threadpool_ = ggml_threadpool_new(&params);
    if (!cpu_threadpool_)
      return false;
    ggml_backend_cpu_set_threadpool(
        static_cast<ggml_backend_t>(cpu_backend_),
        static_cast<ggml_threadpool_t>(cpu_threadpool_));
  }

  // Make the effective ISA visible in the runtime log.  The Qwen IQ expert
  // path is intentionally using GGML's production quantized dot kernels:
  // IQ1_M/IQ2_XXS/IQ3_XXS select the x86 __AVX2__ implementations in
  // arch/x86/quants.c.  AVX-512/VNNI remains available to other GGML ops and
  // repacking paths, but those IQ kernels are not VNNI dot products.
  static std::atomic<bool> logged_cpu_isa{false};
  if (!logged_cpu_isa.exchange(true, std::memory_order_relaxed)) {
    LOG_STREAM_INFO()
        << "[CPU-ISA] Qwen GGML CPU graph: threads=" << n_threads_
        << ", avx2=" << ggml_cpu_has_avx2()
        << ", avx512=" << ggml_cpu_has_avx512()
        << ", avx512_vnni=" << ggml_cpu_has_avx512_vnni()
        << ", bmi2=" << ggml_cpu_has_bmi2()
        << "; IQ1_M/IQ2_XXS/IQ3_XXS dot kernels=GGML AVX2 (not VNNI)\n";
  }

  ggml_init_params wip = {};
  wip.mem_size = kWeightContextBytes + ggml_tensor_overhead() * 8;
  wip.no_alloc = true;
  ggml_context *wctx = ggml_init(wip);
  if (!wctx)
    return false;

  const ggml_type gate_type = static_cast<ggml_type>(ref.gate_type);
  const ggml_type up_type = static_cast<ggml_type>(ref.up_type);
  const ggml_type down_type = static_cast<ggml_type>(ref.down_type);
  ggml_tensor *gate = nullptr;
  ggml_tensor *up = nullptr;
  ggml_tensor *gate_up = nullptr;
  if (merged_gate_up) {
    // The Qwen DDR arena stores each expert as [gate|up|down].  When gate and
    // up have the same quantization/layout, the first two planes are already
    // one contiguous [d_model, 2*d_ff] matrix per expert.  This mirrors
    // llama.cpp's merged gate_up MUL_MAT_ID without copying or changing the
    // arena ownership contract.
    gate_up = ggml_new_tensor_3d(wctx, gate_type, d_model,
                                 static_cast<int64_t>(d_ff) * 2, n_experts);
  } else {
    gate = ggml_new_tensor_3d(wctx, gate_type, d_model, d_ff, n_experts);
    up = ggml_new_tensor_3d(wctx, up_type, d_model, d_ff, n_experts);
  }
  ggml_tensor *down =
      ggml_new_tensor_3d(wctx, down_type, d_ff, d_model, n_experts);
  if ((!merged_gate_up && (!gate || !up)) ||
      (merged_gate_up && !gate_up) || !down) {
    ggml_free(wctx);
    return false;
  }

  const size_t gate_bytes_per_expert =
      merged_gate_up
          ? (ggml_nbytes(gate_up) / static_cast<size_t>(n_experts)) / 2
          : ggml_nbytes(gate) / static_cast<size_t>(n_experts);
  const size_t up_bytes_per_expert =
      merged_gate_up
          ? gate_bytes_per_expert
          : ggml_nbytes(up) / static_cast<size_t>(n_experts);
  const size_t down_bytes_per_expert =
      ggml_nbytes(down) / static_cast<size_t>(n_experts);
  if (gate_bytes_per_expert == 0 || up_bytes_per_expert == 0 ||
      down_bytes_per_expert == 0 ||
      gate_bytes_per_expert > SIZE_MAX / static_cast<size_t>(n_experts) ||
      up_bytes_per_expert > SIZE_MAX / static_cast<size_t>(n_experts) ||
      down_bytes_per_expert > SIZE_MAX / static_cast<size_t>(n_experts) ||
      (strided &&
       (expert_stride < gate_bytes_per_expert ||
        effective_up_stride < up_bytes_per_expert ||
        effective_down_stride < down_bytes_per_expert ||
        (n_experts > 1 &&
         (expert_stride > SIZE_MAX / static_cast<size_t>(n_experts - 1) ||
          effective_up_stride > SIZE_MAX / static_cast<size_t>(n_experts - 1) ||
          effective_down_stride >
              SIZE_MAX / static_cast<size_t>(n_experts - 1)))))) {
    ggml_free(wctx);
    return false;
  }
  const size_t gate_bytes =
      gate_bytes_per_expert * static_cast<size_t>(n_experts);
  const size_t up_bytes = up_bytes_per_expert * static_cast<size_t>(n_experts);
  const size_t down_bytes =
      down_bytes_per_expert * static_cast<size_t>(n_experts);
  if (gate_bytes > SIZE_MAX - up_bytes ||
      gate_bytes + up_bytes > SIZE_MAX - down_bytes) {
    ggml_free(wctx);
    return false;
  }

  void *storage = nullptr;
  ggml_backend_buffer_t wbuf = nullptr;
  if (!strided) {
    const size_t gate_up_total =
        merged_gate_up ? gate_bytes + up_bytes : gate_bytes;
    const size_t storage_bytes =
        gate_up_total + (merged_gate_up ? down_bytes : up_bytes + down_bytes);
    if (posix_memalign(&storage, 64, storage_bytes) != 0 || !storage) {
      ggml_free(wctx);
      return false;
    }
    weight_storage_ = storage;
    weight_storage_bytes_ = storage_bytes;
    wbuf = ggml_backend_cpu_buffer_from_ptr(weight_storage_,
                                            weight_storage_bytes_);
    const bool weight_alloc_ok =
        wbuf &&
        (merged_gate_up
             ? (ggml_backend_tensor_alloc(wbuf, gate_up, weight_storage_) ==
                    GGML_STATUS_SUCCESS &&
                ggml_backend_tensor_alloc(
                    wbuf, down,
                    static_cast<uint8_t *>(weight_storage_) + gate_up_total) ==
                    GGML_STATUS_SUCCESS)
             : (ggml_backend_tensor_alloc(wbuf, gate, weight_storage_) ==
                    GGML_STATUS_SUCCESS &&
                ggml_backend_tensor_alloc(
                    wbuf, up,
                    static_cast<uint8_t *>(weight_storage_) + gate_bytes) ==
                    GGML_STATUS_SUCCESS &&
                ggml_backend_tensor_alloc(
                    wbuf, down,
                    static_cast<uint8_t *>(weight_storage_) + gate_bytes +
                        up_bytes) == GGML_STATUS_SUCCESS));
    if (!weight_alloc_ok) {
      if (wbuf)
        ggml_backend_buffer_free(wbuf);
      ggml_free(wctx);
      std::free(weight_storage_);
      weight_storage_ = nullptr;
      weight_storage_bytes_ = 0;
      return false;
    }
  } else {
    // The three pointers refer to the first expert in one host-RAM arena.
    // GGML's expert-axis stride is the arena's fixed slab stride, so the
    // projection views skip each expert's other two projections and padding.
    if (merged_gate_up) {
      gate_up->nb[2] = expert_stride;
      gate_up->nb[3] = expert_stride * static_cast<size_t>(n_experts);
    } else {
      gate->nb[2] = expert_stride;
      up->nb[2] = effective_up_stride;
      gate->nb[3] = expert_stride * static_cast<size_t>(n_experts);
      up->nb[3] = effective_up_stride * static_cast<size_t>(n_experts);
    }
    down->nb[2] = effective_down_stride;
    down->nb[3] = effective_down_stride * static_cast<size_t>(n_experts);

    const void *planes[3] = {gate_plane, up_plane, down_plane};
    const size_t strides[3] = {expert_stride, effective_up_stride,
                               effective_down_stride};
    size_t plane_bytes[3] = {
        merged_gate_up ? gate_bytes_per_expert + up_bytes_per_expert
                       : gate_bytes_per_expert,
        up_bytes_per_expert, down_bytes_per_expert};
    ggml_tensor *tensors[3] = {merged_gate_up ? gate_up : gate, up, down};
    const int n_planes = merged_gate_up ? 2 : 3;
    if (merged_gate_up) {
      planes[1] = down_plane;
      plane_bytes[1] = down_bytes_per_expert;
      tensors[1] = down;
    }
    for (int i = 0; i < n_planes; ++i) {
      const size_t view_bytes =
          strides[i] * static_cast<size_t>(n_experts - 1) + plane_bytes[i];
      const void *plane = planes[i];
      ggml_backend_buffer_t mbuf = ggml_backend_cpu_buffer_from_ptr(
          const_cast<void *>(plane), view_bytes);
      if (!mbuf || ggml_backend_tensor_alloc(mbuf, tensors[i],
                                             const_cast<void *>(plane)) !=
                       GGML_STATUS_SUCCESS) {
        if (mbuf)
          ggml_backend_buffer_free(mbuf);
        for (void *old : external_weight_bufs_)
          if (old)
            ggml_backend_buffer_free(static_cast<ggml_backend_buffer_t>(old));
        external_weight_bufs_.clear();
        ggml_free(wctx);
        return false;
      }
      external_weight_bufs_.push_back(mbuf);
    }
  }

  auto discard_weight_storage = [&]() {
    if (wbuf) {
      ggml_backend_buffer_free(wbuf);
      wbuf = nullptr;
    }
    for (void *buf : external_weight_bufs_) {
      if (buf)
        ggml_backend_buffer_free(static_cast<ggml_backend_buffer_t>(buf));
    }
    external_weight_bufs_.clear();
    if (weight_storage_) {
      std::free(weight_storage_);
      weight_storage_ = nullptr;
      weight_storage_bytes_ = 0;
    }
  };

  ggml_init_params gip = {};
  gip.mem_size =
      kGraphContextBytes + ggml_graph_overhead() + ggml_tensor_overhead() * 256;
  gip.no_alloc = true;
  ggml_context *gctx = ggml_init(gip);
  if (!gctx) {
    discard_weight_storage();
    ggml_free(wctx);
    return false;
  }
  ggml_cgraph *graph = ggml_new_graph(gctx);
  ggml_tensor *x = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, d_model, 1);
  ggml_tensor *ids = ggml_new_tensor_2d(gctx, GGML_TYPE_I32, topk, 1);
  ggml_tensor *weights = output_per_expert
                             ? nullptr
                             : ggml_new_tensor_3d(gctx, GGML_TYPE_F32, 1, topk,
                                                  1);
  if (!graph || !x || !ids || (!output_per_expert && !weights)) {
    ggml_free(gctx);
    discard_weight_storage();
    ggml_free(wctx);
    return false;
  }
  ggml_set_input(x);
  ggml_set_input(ids);
  if (weights)
    ggml_set_input(weights);

  ggml_tensor *input3 = ggml_reshape_3d(gctx, x, d_model, 1, 1);
  ggml_tensor *gate_out = nullptr;
  ggml_tensor *up_out = nullptr;
  if (merged_gate_up) {
    ggml_tensor *gate_up_out = ggml_mul_mat_id(gctx, gate_up, input3, ids);
    if (gate_up_out)
      ggml_set_name(gate_up_out, "qwen4exp_cpu_moe_gate_up");
    if (gate_up_out) {
      gate_out = ggml_view_3d(gctx, gate_up_out, d_ff, gate_up_out->ne[1],
                              gate_up_out->ne[2], gate_up_out->nb[1],
                              gate_up_out->nb[2], 0);
      up_out = ggml_view_3d(
          gctx, gate_up_out, d_ff, gate_up_out->ne[1], gate_up_out->ne[2],
          gate_up_out->nb[1], gate_up_out->nb[2],
          static_cast<size_t>(d_ff) * gate_up_out->nb[0]);
    }
  } else {
    gate_out = ggml_mul_mat_id(gctx, gate, input3, ids);
    up_out = ggml_mul_mat_id(gctx, up, input3, ids);
    if (gate_out)
      ggml_set_name(gate_out, "qwen4exp_cpu_moe_gate");
    if (up_out)
      ggml_set_name(up_out, "qwen4exp_cpu_moe_up");
  }
  ggml_tensor *act = ggml_swiglu_split(gctx, gate_out, up_out);
  if (act)
    ggml_set_name(act, "qwen4exp_cpu_moe_swiglu");
  ggml_tensor *down_out = ggml_mul_mat_id(gctx, down, act, ids);
  if (down_out)
    ggml_set_name(down_out, "qwen4exp_cpu_moe_down");
  ggml_tensor *weighted = output_per_expert
                              ? nullptr
                              : ggml_mul(gctx, down_out, weights);
  if (weighted)
    ggml_set_name(weighted, "qwen4exp_cpu_moe_weighted");
  if (!input3 || !gate_out || !up_out || !act || !down_out ||
      (!output_per_expert && !weighted)) {
    ggml_free(gctx);
    discard_weight_storage();
    ggml_free(wctx);
    return false;
  }

  ggml_tensor *result = nullptr;
  if (output_per_expert) {
    // Preserve the expert axis. The GPU combine graph consumes this as
    // [d_model, topk, 1], so no CPU weighting or accumulation is performed.
    result = down_out;
    ggml_set_name(result, "qwen4exp_cpu_moe_down_out");
  } else {
    ggml_tensor *sum = nullptr;
    for (int k = 0; k < topk; ++k) {
      ggml_tensor *expert =
          ggml_view_2d(gctx, weighted, d_model, 1, weighted->nb[2],
                       static_cast<size_t>(k) * weighted->nb[1]);
      sum = k == 0 ? expert : ggml_add(gctx, sum, expert);
      if (sum && k > 0)
        ggml_set_name(sum, "qwen4exp_cpu_moe_reduce_add");
    }
    if (!sum) {
      ggml_free(gctx);
      discard_weight_storage();
      ggml_free(wctx);
      return false;
    }
    if (topk > 1)
      sum = ggml_cont(gctx, sum);
    result = sum;
    if (result)
      ggml_set_name(result, "qwen4exp_cpu_moe_reduce");
  }
  ggml_set_output(result);
  ggml_build_forward_expand(graph, result);

  ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
  if (!ga || !ggml_gallocr_alloc_graph(ga, graph)) {
    if (ga)
      ggml_gallocr_free(ga);
    ggml_free(gctx);
    discard_weight_storage();
    ggml_free(wctx);
    return false;
  }

  // Keep the CPU execution plan alongside the persistent graph.  The graph's
  // tensor data pointers are retargeted between Qwen layers, but its shape,
  // node topology, threadpool, and workspace requirements remain fixed.  A
  // plan is therefore safe to reuse and removes the per-call graph-plan scan
  // and work-buffer setup that ggml_backend_graph_compute() otherwise repeats.
  // If a backend build cannot create a plan, retain the ordinary compute path
  // as a correctness-preserving fallback.
  graph_plan_ = ggml_backend_graph_plan_create(
      static_cast<ggml_backend_t>(cpu_backend_), graph);

  weight_ctx_ = wctx;
  weight_buf_ = wbuf;
  gate_ = gate;
  up_ = up;
  gate_up_ = gate_up;
  down_ = down;
  graph_ctx_ = gctx;
  galloc_ = ga;
  graph_ = graph;
  x_ = x;
  ids_ = ids;
  weights_ = weights;
  output_ = result;
  act_ = act;
  down_output_ = down_out;
  n_experts_ = n_experts;
  d_model_ = d_model;
  d_ff_ = d_ff;
  topk_ = topk;
  gate_type_ = ref.gate_type;
  up_type_ = ref.up_type;
  down_type_ = ref.down_type;
  strided_mode_ = strided;
  merged_gate_up_ = merged_gate_up;
  output_per_expert_ = output_per_expert;
  expert_stride_ = strided ? expert_stride : 0;
  up_expert_stride_ = strided ? effective_up_stride : 0;
  down_expert_stride_ = strided ? effective_down_stride : 0;
  return true;
}

bool GgmlQwenCpuMoeRunner::compute_graph() {
  if (!cpu_backend_ || !graph_)
    return false;
  const ggml_status status =
      graph_plan_
          ? ggml_backend_graph_plan_compute(
                static_cast<ggml_backend_t>(cpu_backend_),
                static_cast<ggml_backend_graph_plan_t>(graph_plan_))
          : ggml_backend_graph_compute(static_cast<ggml_backend_t>(cpu_backend_),
                                       static_cast<ggml_cgraph *>(graph_));
  return status == GGML_STATUS_SUCCESS;
}

bool GgmlQwenCpuMoeRunner::run(const float *input, float *output,
                               const GgmlQwenCpuExpertRef *experts,
                               int n_experts, const float *router_weights,
                               int d_model, int d_ff, int topk) {
  if (!input || !output || !experts || !router_weights || n_experts <= 0 ||
      n_experts != topk || d_model <= 0 || d_ff <= 0)
    return false;

  const GgmlQwenCpuExpertRef &first = experts[0];
  if (!first.gate || !first.up || !first.down || first.gate_bytes == 0 ||
      first.up_bytes == 0 || first.down_bytes == 0 ||
      !type_is_valid(first.gate_type) || !type_is_valid(first.up_type) ||
      !type_is_valid(first.down_type))
    return false;
  for (int i = 1; i < n_experts; ++i) {
    const GgmlQwenCpuExpertRef &e = experts[i];
    if (!e.gate || !e.up || !e.down || e.gate_bytes != first.gate_bytes ||
        e.up_bytes != first.up_bytes || e.down_bytes != first.down_bytes ||
        e.gate_type != first.gate_type || e.up_type != first.up_type ||
        e.down_type != first.down_type)
      return false;
  }
  const bool merged_gate_up = first.gate_type == first.up_type &&
                              first.gate_bytes == first.up_bytes;
  if (!ensure_graph(first, n_experts, d_model, d_ff, topk, false, nullptr,
                    nullptr, nullptr, 0, false, merged_gate_up))
    return false;

  const size_t gate_plane =
      weight_storage_ == nullptr
          ? 0
          : (merged_gate_up
                 ? (ggml_nbytes(static_cast<ggml_tensor *>(gate_up_)) /
                    static_cast<size_t>(n_experts_)) /
                       2
                 : ggml_nbytes(static_cast<ggml_tensor *>(gate_)) /
                       static_cast<size_t>(n_experts_));
  const size_t up_plane =
      merged_gate_up
          ? gate_plane
          : ggml_nbytes(static_cast<ggml_tensor *>(up_)) /
                static_cast<size_t>(n_experts_);
  const size_t down_plane = ggml_nbytes(static_cast<ggml_tensor *>(down_)) /
                            static_cast<size_t>(n_experts_);
  if (gate_plane != first.gate_bytes || up_plane != first.up_bytes ||
      down_plane != first.down_bytes)
    return false;

  uint8_t *storage = static_cast<uint8_t *>(weight_storage_);
  const size_t gate_total = gate_plane * static_cast<size_t>(n_experts_);
  const size_t up_total = up_plane * static_cast<size_t>(n_experts_);
  const size_t gate_up_total = gate_total + up_total;
  if (cached_sources_.size() != static_cast<size_t>(n_experts_))
    cached_sources_.assign(static_cast<size_t>(n_experts_), {});
  for (int i = 0; i < n_experts_; ++i) {
    const GgmlQwenCpuExpertRef &expert = experts[i];
    auto &cached = cached_sources_[static_cast<size_t>(i)];
    const bool unchanged =
        cached.gate == expert.gate && cached.up == expert.up &&
        cached.down == expert.down && cached.gate_bytes == gate_plane &&
        cached.up_bytes == up_plane && cached.down_bytes == down_plane;
    if (!unchanged) {
      if (merged_gate_up) {
        const size_t expert_gate_up =
            static_cast<size_t>(i) * (gate_plane + up_plane);
        std::memcpy(storage + expert_gate_up, expert.gate, gate_plane);
        std::memcpy(storage + expert_gate_up + gate_plane, expert.up,
                    up_plane);
      } else {
        std::memcpy(storage + static_cast<size_t>(i) * gate_plane,
                    expert.gate, gate_plane);
        std::memcpy(storage + gate_total +
                        static_cast<size_t>(i) * up_plane,
                    expert.up, up_plane);
      }
      std::memcpy(storage + (merged_gate_up ? gate_up_total
                                             : gate_total + up_total) +
                      static_cast<size_t>(i) * down_plane,
                  expert.down, down_plane);
      cached.gate = expert.gate;
      cached.up = expert.up;
      cached.down = expert.down;
      cached.gate_bytes = gate_plane;
      cached.up_bytes = up_plane;
      cached.down_bytes = down_plane;
    }
  }

  std::memcpy(static_cast<ggml_tensor *>(x_)->data, input,
              static_cast<size_t>(d_model_) * sizeof(float));
  auto *ids = static_cast<int32_t *>(static_cast<ggml_tensor *>(ids_)->data);
  for (int i = 0; i < topk_; ++i)
    ids[i] = i;
  std::memcpy(static_cast<ggml_tensor *>(weights_)->data, router_weights,
              static_cast<size_t>(topk_) * sizeof(float));

  if (!compute_graph())
    return false;
  dump_qwen_cpu_diag_tensor("down_input_swiglu",
                            static_cast<const ggml_tensor *>(act_));
  dump_qwen_cpu_diag_tensor("down_output_raw",
                            static_cast<const ggml_tensor *>(down_output_));
  dump_qwen_cpu_diag_tensor("cpu_output_pre_h2d",
                            static_cast<const ggml_tensor *>(output_));
  std::memcpy(output, static_cast<ggml_tensor *>(output_)->data,
              static_cast<size_t>(d_model_) * sizeof(float));
  return true;
}

bool GgmlQwenCpuMoeRunner::run_strided(
    const float *input, float *output, const GgmlQwenCpuExpertRef &layer,
    const void *gate_plane, const void *up_plane, const void *down_plane,
    const int *expert_ids, int n_experts, const float *router_weights,
    int d_model, int d_ff, int topk, size_t expert_stride) {
  if (!input || !output || !gate_plane || !up_plane || !down_plane ||
      !expert_ids || !router_weights || n_experts <= 0 || topk <= 0 ||
      topk > n_experts || d_model <= 0 || d_ff <= 0 || layer.gate_bytes == 0 ||
      layer.up_bytes == 0 || layer.down_bytes == 0 || expert_stride == 0 ||
      !type_is_valid(layer.gate_type) || !type_is_valid(layer.up_type) ||
      !type_is_valid(layer.down_type))
    return false;
  for (int k = 0; k < topk; ++k) {
    if (expert_ids[k] < 0 || expert_ids[k] >= n_experts)
      return false;
  }

  if (direct_pool_enabled() &&
      direct_signature_supported(layer, d_model, d_ff, topk, expert_stride,
                                 expert_stride, expert_stride)) {
    if (!ensure_direct_signature(layer, n_experts, d_model, d_ff, topk,
                                 expert_stride, expert_stride, expert_stride))
      return false;
    AsyncStridedRequest request{input,
                                output,
                                layer,
                                gate_plane,
                                up_plane,
                                down_plane,
                                expert_ids,
                                n_experts,
                                router_weights,
                                d_model,
                                d_ff,
                                topk,
                                expert_stride,
                                expert_stride,
                                expert_stride,
                                nullptr,
                                nullptr};
    return run_direct_request(request);
  }

  if (!ensure_graph(layer, n_experts, d_model, d_ff, topk, true, gate_plane,
                    up_plane, down_plane, expert_stride, false,
                    can_merge_gate_up(layer, gate_plane, up_plane)))
    return false;

  // The graph and its allocator are persistent, but the source planes can
  // belong to a different layer on the next call. GGML's CPU backend reads
  // tensor->data directly, so retarget the three external views without
  // allocating or copying the model weights.
  if (merged_gate_up_) {
    static_cast<ggml_tensor *>(gate_up_)->data = const_cast<void *>(gate_plane);
  } else {
    static_cast<ggml_tensor *>(gate_)->data = const_cast<void *>(gate_plane);
    static_cast<ggml_tensor *>(up_)->data = const_cast<void *>(up_plane);
  }
  static_cast<ggml_tensor *>(down_)->data = const_cast<void *>(down_plane);

  QwenCpuGraphTiming &timing = qwen_cpu_graph_timing();
  const auto total_start = std::chrono::steady_clock::now();
  const auto input_start = total_start;
  std::memcpy(static_cast<ggml_tensor *>(x_)->data, input,
              static_cast<size_t>(d_model_) * sizeof(float));
  if (timing.enabled)
    qwen_cpu_graph_timing_add(
        timing.input_s,
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      input_start)
            .count());

  const auto metadata_start = std::chrono::steady_clock::now();
  auto *ids = static_cast<int32_t *>(static_cast<ggml_tensor *>(ids_)->data);
  for (int k = 0; k < topk_; ++k)
    ids[k] = static_cast<int32_t>(expert_ids[k]);
  std::memcpy(static_cast<ggml_tensor *>(weights_)->data, router_weights,
              static_cast<size_t>(topk_) * sizeof(float));

  if (timing.enabled)
    qwen_cpu_graph_timing_add(
        timing.metadata_s,
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      metadata_start)
            .count());
  const auto compute_start = std::chrono::steady_clock::now();
  if (!compute_graph())
    return false;
  dump_qwen_cpu_diag_tensor("down_input_swiglu",
                            static_cast<const ggml_tensor *>(act_));
  dump_qwen_cpu_diag_tensor("down_output_raw",
                            static_cast<const ggml_tensor *>(down_output_));
  dump_qwen_cpu_diag_tensor("cpu_output_pre_h2d",
                            static_cast<const ggml_tensor *>(output_));
  if (timing.enabled)
    qwen_cpu_graph_timing_add(
        timing.compute_s,
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      compute_start)
            .count());
  const auto output_start = std::chrono::steady_clock::now();
  std::memcpy(output, static_cast<ggml_tensor *>(output_)->data,
              static_cast<size_t>(d_model_) * sizeof(float));
  if (timing.enabled) {
    qwen_cpu_graph_timing_add(
        timing.output_s,
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      output_start)
            .count());
    qwen_cpu_graph_timing_add(
        timing.total_s,
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      total_start)
            .count());
    const uint64_t calls =
        timing.calls.fetch_add(1, std::memory_order_relaxed) + 1;
    qwen_cpu_graph_timing_report(timing, calls);
  }
  return true;
}

bool GgmlQwenCpuMoeRunner::run_strided_planes(
    const float *input, float *output, const GgmlQwenCpuExpertRef &layer,
    const void *gate_plane, const void *up_plane, const void *down_plane,
    const int *expert_ids, int n_experts, const float *router_weights,
    int d_model, int d_ff, int topk, size_t gate_stride, size_t up_stride,
    size_t down_stride) {
  if (!input || !output || !gate_plane || !up_plane || !down_plane ||
      !expert_ids || !router_weights || n_experts <= 0 || topk <= 0 ||
      topk > n_experts || d_model <= 0 || d_ff <= 0 ||
      layer.gate_bytes == 0 || layer.up_bytes == 0 ||
      layer.down_bytes == 0 || gate_stride == 0 || up_stride == 0 ||
      down_stride == 0 || !type_is_valid(layer.gate_type) ||
      !type_is_valid(layer.up_type) || !type_is_valid(layer.down_type))
    return false;
  for (int k = 0; k < topk; ++k) {
    if (expert_ids[k] < 0 || expert_ids[k] >= n_experts)
      return false;
  }

  if (direct_pool_enabled() &&
      direct_signature_supported(layer, d_model, d_ff, topk, gate_stride,
                                 up_stride, down_stride)) {
    if (!ensure_direct_signature(layer, n_experts, d_model, d_ff, topk,
                                 gate_stride, up_stride, down_stride))
      return false;
    AsyncStridedRequest request{input,
                                output,
                                layer,
                                gate_plane,
                                up_plane,
                                down_plane,
                                expert_ids,
                                n_experts,
                                router_weights,
                                d_model,
                                d_ff,
                                topk,
                                gate_stride,
                                up_stride,
                                down_stride,
                                nullptr,
                                nullptr};
    return run_direct_request(request);
  }

  // Compact projection planes cannot use the merged gate/up view: the next
  // expert's gate bytes follow the current expert in the gate plane, not the
  // current expert's up bytes. The CPU graph therefore receives three
  // independent expert-axis strides here.
  if (!ensure_graph(layer, n_experts, d_model, d_ff, topk, true,
                    gate_plane, up_plane, down_plane, gate_stride, false,
                    false, up_stride, down_stride)) {
    const char *sched_debug = std::getenv("MOESTREAM_QWEN_DECODE_SCHED_DEBUG");
    if (sched_debug && sched_debug[0] == '1')
      LOG_WARN("[QWEN-CPU-PLANES] ensure_graph failed types=%u/%u/%u "
               "bytes=%zu/%zu/%zu strides=%zu/%zu/%zu n=%d D=%d FF=%d K=%d\n",
               layer.gate_type, layer.up_type, layer.down_type,
               layer.gate_bytes, layer.up_bytes, layer.down_bytes,
               gate_stride, up_stride, down_stride, n_experts, d_model, d_ff,
               topk);
    return false;
  }

  static_cast<ggml_tensor *>(gate_)->data = const_cast<void *>(gate_plane);
  static_cast<ggml_tensor *>(up_)->data = const_cast<void *>(up_plane);
  static_cast<ggml_tensor *>(down_)->data = const_cast<void *>(down_plane);

  QwenCpuGraphTiming &timing = qwen_cpu_graph_timing();
  const auto total_start = std::chrono::steady_clock::now();
  const auto input_start = total_start;
  std::memcpy(static_cast<ggml_tensor *>(x_)->data, input,
              static_cast<size_t>(d_model_) * sizeof(float));
  if (timing.enabled)
    qwen_cpu_graph_timing_add(
        timing.input_s,
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      input_start)
            .count());

  const auto metadata_start = std::chrono::steady_clock::now();
  auto *ids = static_cast<int32_t *>(static_cast<ggml_tensor *>(ids_)->data);
  for (int k = 0; k < topk_; ++k)
    ids[k] = static_cast<int32_t>(expert_ids[k]);
  std::memcpy(static_cast<ggml_tensor *>(weights_)->data, router_weights,
              static_cast<size_t>(topk_) * sizeof(float));
  if (timing.enabled)
    qwen_cpu_graph_timing_add(
        timing.metadata_s,
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      metadata_start)
            .count());

  const auto compute_start = std::chrono::steady_clock::now();
  if (!compute_graph())
    return false;
  if (timing.enabled)
    qwen_cpu_graph_timing_add(
        timing.compute_s,
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      compute_start)
            .count());
  const auto output_start = std::chrono::steady_clock::now();
  std::memcpy(output, static_cast<ggml_tensor *>(output_)->data,
              static_cast<size_t>(d_model_) * sizeof(float));
  if (timing.enabled) {
    qwen_cpu_graph_timing_add(
        timing.output_s,
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      output_start)
            .count());
    qwen_cpu_graph_timing_add(
        timing.total_s,
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      total_start)
            .count());
    const uint64_t calls =
        timing.calls.fetch_add(1, std::memory_order_relaxed) + 1;
    qwen_cpu_graph_timing_report(timing, calls);
  }
  return true;
}

bool GgmlQwenCpuMoeRunner::run_strided_down(
    const float *input, float *expert_output,
    const GgmlQwenCpuExpertRef &layer, const void *gate_plane,
    const void *up_plane, const void *down_plane, const int *expert_ids,
    int n_experts, int d_model, int d_ff, int topk, size_t expert_stride) {
  if (!input || !expert_output || !gate_plane || !up_plane || !down_plane ||
      !expert_ids || n_experts <= 0 || topk <= 0 || topk > n_experts ||
      d_model <= 0 || d_ff <= 0 || layer.gate_bytes == 0 ||
      layer.up_bytes == 0 || layer.down_bytes == 0 || expert_stride == 0 ||
      !type_is_valid(layer.gate_type) || !type_is_valid(layer.up_type) ||
      !type_is_valid(layer.down_type))
    return false;
  for (int k = 0; k < topk; ++k) {
    if (expert_ids[k] < 0 || expert_ids[k] >= n_experts)
      return false;
  }

  if (!ensure_graph(layer, n_experts, d_model, d_ff, topk, true, gate_plane,
                    up_plane, down_plane, expert_stride, true,
                    can_merge_gate_up(layer, gate_plane, up_plane)))
    return false;

  if (merged_gate_up_) {
    static_cast<ggml_tensor *>(gate_up_)->data = const_cast<void *>(gate_plane);
  } else {
    static_cast<ggml_tensor *>(gate_)->data = const_cast<void *>(gate_plane);
    static_cast<ggml_tensor *>(up_)->data = const_cast<void *>(up_plane);
  }
  static_cast<ggml_tensor *>(down_)->data = const_cast<void *>(down_plane);

  std::memcpy(static_cast<ggml_tensor *>(x_)->data, input,
              static_cast<size_t>(d_model_) * sizeof(float));
  auto *ids = static_cast<int32_t *>(static_cast<ggml_tensor *>(ids_)->data);
  for (int k = 0; k < topk_; ++k)
    ids[k] = static_cast<int32_t>(expert_ids[k]);

  if (!compute_graph())
    return false;

  const ggml_tensor *result = static_cast<const ggml_tensor *>(output_);
  if (!result || result->ne[0] != d_model_ || result->ne[1] != topk_ ||
      result->ne[2] != 1 || result->ne[3] != 1 ||
      result->nb[0] != sizeof(float) ||
      result->nb[1] != static_cast<size_t>(d_model_) * sizeof(float))
    return false;
  std::memcpy(expert_output, result->data,
              static_cast<size_t>(d_model_) * static_cast<size_t>(topk_) *
                  sizeof(float));
  return true;
}

void GgmlQwenCpuMoeRunner::async_worker_loop() {
  for (;;) {
    AsyncStridedRequest request;
    {
      std::unique_lock<std::mutex> lock(async_mutex_);
      async_cv_.wait(lock, [this] {
        return async_stop_ || async_request_pending_;
      });
      if (async_stop_ && !async_request_pending_)
        return;
      request = async_request_;
      async_request_pending_ = false;
    }

    const bool result = run_strided(
        request.input, request.output, request.layer, request.gate_plane,
        request.up_plane, request.down_plane, request.expert_ids,
        request.n_experts, request.router_weights, request.d_model,
        request.d_ff, request.topk, request.expert_stride);

    {
      std::lock_guard<std::mutex> lock(async_mutex_);
      async_result_ = result;
      async_request_done_ = true;
    }
    async_cv_.notify_all();
    // The CPU graph is complete before this callback runs.  It may enqueue a
    // CUDA H2D copy and its stream fence while the persistent runner is
    // already available for the next request; ring-slot lifetime is managed
    // by the callback owner, not by this runner.
    if (request.completion)
      request.completion(request.completion_user, result);
  }
}

bool GgmlQwenCpuMoeRunner::start_strided_async(
    const float *input, float *output, const GgmlQwenCpuExpertRef &layer,
    const void *gate_plane, const void *up_plane, const void *down_plane,
    const int *expert_ids, int n_experts, const float *router_weights,
    int d_model, int d_ff, int topk, size_t expert_stride) {
  return start_strided_async(
      input, output, layer, gate_plane, up_plane, down_plane, expert_ids,
      n_experts, router_weights, d_model, d_ff, topk, expert_stride, nullptr,
      nullptr);
}

bool GgmlQwenCpuMoeRunner::start_strided_async(
    const float *input, float *output, const GgmlQwenCpuExpertRef &layer,
    const void *gate_plane, const void *up_plane, const void *down_plane,
    const int *expert_ids, int n_experts, const float *router_weights,
    int d_model, int d_ff, int topk, size_t expert_stride,
    AsyncCompletionFn completion, void *completion_user) {
  if (direct_pool_enabled() &&
      direct_signature_supported(layer, d_model, d_ff, topk, expert_stride,
                                 expert_stride, expert_stride)) {
    if (!ensure_direct_signature(layer, n_experts, d_model, d_ff, topk,
                                 expert_stride, expert_stride, expert_stride))
      return false;
    AsyncStridedRequest request{input,
                                output,
                                layer,
                                gate_plane,
                                up_plane,
                                down_plane,
                                expert_ids,
                                n_experts,
                                router_weights,
                                d_model,
                                d_ff,
                                topk,
                                expert_stride,
                                expert_stride,
                                expert_stride,
                                completion,
                                completion_user};
    return submit_direct_request(request);
  }

  std::lock_guard<std::mutex> lock(async_mutex_);
  if (async_request_pending_ || !async_request_done_)
    return false;
  if (!async_worker_.joinable()) {
    async_stop_ = false;
    async_worker_ = std::thread(&GgmlQwenCpuMoeRunner::async_worker_loop, this);
  }
  async_request_ = {input,       output,       layer,
                    gate_plane, up_plane,     down_plane,
                    expert_ids,  n_experts,    router_weights,
                    d_model,    d_ff,         topk,
                    expert_stride, expert_stride, expert_stride,
                    completion, completion_user};
  async_result_ = false;
  async_request_done_ = false;
  async_request_pending_ = true;
  async_cv_.notify_one();
  return true;
}

bool GgmlQwenCpuMoeRunner::wait_strided_async() {
  if (direct_ready_ && !direct_workers_.empty()) {
    while (!direct_finished_.load(std::memory_order_acquire))
      qwen_cpu_direct_pause();
    return direct_result_.load(std::memory_order_acquire);
  }
  std::unique_lock<std::mutex> lock(async_mutex_);
  if (!async_worker_.joinable() || async_request_done_)
    return async_worker_.joinable() ? async_result_ : false;
  async_cv_.wait(lock, [this] { return async_request_done_; });
  return async_result_;
}

#else

void qwen_cpu_moe_diag_set_layer(int) {}

GgmlQwenCpuMoeRunner &GgmlQwenCpuMoeRunner::instance() {
  static GgmlQwenCpuMoeRunner runner;
  return runner;
}

GgmlQwenCpuMoeRunner &GgmlQwenCpuMoeRunner::instance_for_strided(
    const GgmlQwenCpuExpertRef &, int, int, int, int, size_t, bool) {
  return instance();
}
GgmlQwenCpuMoeRunner &GgmlQwenCpuMoeRunner::instance_for_strided_planes(
    const GgmlQwenCpuExpertRef &, int, int, int, int, size_t, size_t, size_t,
    bool) {
  return instance();
}

bool GgmlQwenCpuMoeRunner::compiled_in() { return false; }

void GgmlQwenCpuMoeRunner::pause_workers() {}
void GgmlQwenCpuMoeRunner::resume_workers() {}

GgmlQwenCpuMoeRunner::~GgmlQwenCpuMoeRunner() = default;
void GgmlQwenCpuMoeRunner::shutdown() {}
bool GgmlQwenCpuMoeRunner::run(const float *, float *,
                               const GgmlQwenCpuExpertRef *, int, const float *,
                               int, int, int) {
  return false;
}
bool GgmlQwenCpuMoeRunner::run_strided(const float *, float *,
                                       const GgmlQwenCpuExpertRef &,
                                       const void *, const void *, const void *,
                                       const int *, int, const float *, int,
                                       int, int, size_t) {
  return false;
}
bool GgmlQwenCpuMoeRunner::run_strided_planes(
    const float *, float *, const GgmlQwenCpuExpertRef &, const void *,
    const void *, const void *, const int *, int, const float *, int, int, int,
    size_t, size_t, size_t) {
  return false;
}
bool GgmlQwenCpuMoeRunner::run_strided_down(
    const float *, float *, const GgmlQwenCpuExpertRef &, const void *,
    const void *, const void *, const int *, int, int, int, int, size_t) {
  return false;
}
bool GgmlQwenCpuMoeRunner::start_strided_async(
    const float *, float *, const GgmlQwenCpuExpertRef &, const void *,
    const void *, const void *, const int *, int, const float *, int, int, int,
    size_t) {
  return false;
}
bool GgmlQwenCpuMoeRunner::start_strided_async(
    const float *, float *, const GgmlQwenCpuExpertRef &, const void *,
    const void *, const void *, const int *, int, const float *, int, int, int,
    size_t, AsyncCompletionFn, void *) {
  return false;
}
bool GgmlQwenCpuMoeRunner::wait_strided_async() { return false; }

#endif
