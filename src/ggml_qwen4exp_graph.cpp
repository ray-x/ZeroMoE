// Real multi-layer Qwen4Exp GGML graph.
//
// This is intentionally a separate bridge rather than a rewrite of the
// established execution path.  Llama.cpp's important property is that the
// C++ caller constructs one dependency graph for the entire trunk and then
// gives it to ggml_backend_sched_graph_compute_async().  The scheduler may
// split that graph internally around host-backed expert weights, but the
// caller does not return to C++ after every layer.  This file reproduces that
// shape for MoEStream's model representation.

#include "ggml_qwen4exp_graph.hpp"

#include "dense_dispatch.hpp" // is_cpu_affinity_layer (cpu_affinity_layers config)
#include "cuda_manager.hpp"
#include "direct_io.hpp"
#include "engine.hpp"
#include "ggml_moe_backend.hpp"
#include "kquant_dequant.hpp"
#include "logger.hpp"
#include "memory_guard.hpp"
#include "ple_hash.hpp"
#include "thread_profile.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#if defined(MOESTREAM_USE_GGML_CUDA)

#include <cuda_runtime.h>

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#if defined(_OPENMP)
#include <omp.h>
#endif

// A CPY node carrying this context is the graph-visible marker for a fused
// CPU-offloaded MoE layer.  The CUDA backend sees only the opaque `extra`
// pointer and calls the bridge below; the context itself stays owned by the
// FullGraph instance for the lifetime of the captured graph.
struct Qwen4ExpCudaHostNodeContext {
  static constexpr uint64_t kMagic = 0x514f46464c4f4144ull; // QOFFLOAD
  uint64_t magic = kMagic;
  GgmlQwenCpuExpertRef layer{};
  const void *gate_plane = nullptr;
  const void *up_plane = nullptr;
  const void *down_plane = nullptr;
  int n_experts = 0;
  int d_model = 0;
  int d_ff = 0;
  int topk = 0;
  size_t expert_stride = 0;
  size_t gate_stride = 0;
  size_t up_stride = 0;
  size_t down_stride = 0;
  bool projection_compact = false;
  // The full graph owns this pool. The host-node runner borrows it for the
  // callback and never frees or pauses it.
  ggml_threadpool_t shared_cpu_threadpool = nullptr;
  int shared_cpu_threads = 0;
  GgmlQwenCpuMoeRunner *cpu_runner = nullptr;
  float *host_input = nullptr;
  float *host_output = nullptr;
  int32_t *host_ids = nullptr;
  float *host_weights = nullptr;
  size_t input_bytes = 0;
  size_t output_bytes = 0;
  size_t ids_bytes = 0;
  size_t weights_bytes = 0;
  void *device_output = nullptr;
  int layer_index = -1;
  // Immutable static hot set for a planned per-expert split.  This is
  // diagnostic-only for the current full-CPU host marker, but keeping it in
  // the captured context makes it possible to prove when the marker would
  // otherwise bypass a planned GPU branch.
  std::vector<int> gpu_resident_experts;
  bool transport_diag = false;
  float *host_output_gpu_check = nullptr;
  cudaEvent_t d2h_start = nullptr;
  cudaEvent_t d2h_end = nullptr;
  cudaEvent_t h2d_start = nullptr;
  cudaEvent_t h2d_end = nullptr;
  cudaEvent_t post_h2d_check_end = nullptr;
  std::atomic<bool> last_ok{false};
  std::atomic<double> last_cpu_ms{0.0};
  std::atomic<double> last_callback_total_ms{0.0};
  std::atomic<double> last_callback_guard_wait_ms{0.0};
  std::atomic<double> last_callback_body_ms{0.0};
  std::atomic<uint64_t> last_cpu_hash{0};
  std::atomic<uint64_t> last_gpu_hash{0};
  std::atomic<bool> last_cpu_hash_valid{false};
};

// A CPY node carrying THIS context is the graph-visible marker for a
// generalized GPU-staged decode layer (see engine.qwen4exp_gpu_stage_*).
// Unlike Qwen4ExpCudaHostNodeContext above, the callback body here does a
// pure host-memory GATHER (no CPU FFN compute, no CUDA API calls -- a hard
// requirement inside a cudaLaunchHostFunc callback) -- it copies each of
// this token's real routed experts' gate/up/down bytes, in router-output
// order, out of the preloaded DDR arena into one compact pinned bounce
// buffer, which the SAME enqueue call then H2D-copies into three fixed
// compact device destinations (gate_dst/up_dst/down_dst). Slot k of every
// destination always holds real routed expert ids[k]'s bytes, which is why
// the downstream MUL_MAT_ID nodes can use a CONSTANT [0..topk-1] ids
// tensor instead of any real per-token id remap -- see build_gpu_stage_node
// and build_moe's staged-layer branch for the full invariant.
struct Qwen4ExpGpuStageContext {
  static constexpr uint64_t kMagic = 0x51474d53544147ull; // "QGMSTAG" (trunc)
  uint64_t magic = kMagic;
  int layer_index = -1;
  int n_experts = 0;
  int topk = 0;
  // Source planes/strides into the persistent, engine-owned, independently
  // pinnable DDR arena (StreamingEngine::graph_ram_arenas[layer]) -- the
  // SAME arena load_full_weight()'s general host-resident path and Option
  // B's layer-13 staging already read from. Borrowed, not owned.
  const uint8_t *gate_plane = nullptr;
  const uint8_t *up_plane = nullptr;
  const uint8_t *down_plane = nullptr;
  size_t gate_stride = 0;
  size_t up_stride = 0;
  size_t down_stride = 0;
  size_t gate_bytes = 0; // bytes for ONE expert's gate plane
  size_t up_bytes = 0;
  size_t down_bytes = 0;
  // Pinned host buffers, sized for exactly `topk` slots each.
  int32_t *host_ids = nullptr;        // topk * sizeof(int32_t)
  uint8_t *host_gate_bounce = nullptr; // topk * gate_bytes
  uint8_t *host_up_bounce = nullptr;   // topk * up_bytes
  uint8_t *host_down_bounce = nullptr; // topk * down_bytes
  size_t ids_bytes = 0;
  // Fixed device destinations (gate_dst/up_dst/down_dst's ->data), bound
  // once via phase2_bind_or_allocate_expert -- never reallocated.
  void *device_gate_dst = nullptr;
  void *device_up_dst = nullptr;
  void *device_down_dst = nullptr;
  // The `ids` device tensor this layer's real routing produced this build
  // (ggml_cont'd, so a flat D2H read is safe -- decode is always S==1, so
  // the argsort-view-stride pitfall documented at build_moe's ids_out site
  // does not apply here either way, but ggml_cont keeps this robust).
  cudaEvent_t d2h_start = nullptr;
  cudaEvent_t d2h_end = nullptr;
  cudaEvent_t h2d_start = nullptr;
  cudaEvent_t h2d_end = nullptr;
  std::atomic<bool> last_ok{false};
  std::atomic<double> last_callback_ms{0.0};
};

static void CUDART_CB
qwen4exp_cuda_gpu_stage_callback(void *user_data) {
  auto *ctx = static_cast<Qwen4ExpGpuStageContext *>(user_data);
  if (!ctx || ctx->magic != Qwen4ExpGpuStageContext::kMagic)
    return;
  const auto start = std::chrono::steady_clock::now();
  bool ok = true;
  for (int k = 0; k < ctx->topk; ++k) {
    const int32_t expert = ctx->host_ids[k];
    if (expert < 0 || expert >= ctx->n_experts) {
      // Defensive only -- real routing always produces valid ids. Zero-fill
      // this slot and keep going rather than read out of bounds, mirroring
      // env_host_node_skip_cpu()'s existing zero-fill escape hatch.
      std::memset(ctx->host_gate_bounce +
                      static_cast<size_t>(k) * ctx->gate_bytes,
                  0, ctx->gate_bytes);
      std::memset(ctx->host_up_bounce +
                      static_cast<size_t>(k) * ctx->up_bytes,
                  0, ctx->up_bytes);
      std::memset(ctx->host_down_bounce +
                      static_cast<size_t>(k) * ctx->down_bytes,
                  0, ctx->down_bytes);
      ok = false;
      continue;
    }
    std::memcpy(ctx->host_gate_bounce + static_cast<size_t>(k) * ctx->gate_bytes,
               ctx->gate_plane + static_cast<size_t>(expert) * ctx->gate_stride,
               ctx->gate_bytes);
    std::memcpy(ctx->host_up_bounce + static_cast<size_t>(k) * ctx->up_bytes,
               ctx->up_plane + static_cast<size_t>(expert) * ctx->up_stride,
               ctx->up_bytes);
    std::memcpy(ctx->host_down_bounce + static_cast<size_t>(k) * ctx->down_bytes,
               ctx->down_plane + static_cast<size_t>(expert) * ctx->down_stride,
               ctx->down_bytes);
  }
  ctx->last_ok.store(ok, std::memory_order_release);
  const double callback_ms = std::chrono::duration<double, std::milli>(
                                 std::chrono::steady_clock::now() - start)
                                 .count();
  ctx->last_callback_ms.store(callback_ms, std::memory_order_release);
  if (std::getenv("MOESTREAM_QWEN4EXP_GPU_STAGE_TIMING"))
    LOG_INFO("[QWEN4EXP-GPU-STAGE-TIMING] layer=%d callback_body_ms=%.4f\n",
             ctx->layer_index, callback_ms);
  if (!ok)
    LOG_ERROR("[QWEN4EXP-GPU-STAGE] layer=%d one or more routed ids were "
              "out of range [0,%d); affected slot(s) zero-filled\n",
              ctx->layer_index, ctx->n_experts);
}

// Enqueues the D2H/callback/H2D sequence for a GPU-stage marker while a CUDA
// stream is being captured -- same technique as
// moestream_qwen4exp_cuda_host_node_enqueue below, called from the SAME weak
// hook (dispatch is by ctx->magic, see that function's own top).
static bool qwen4exp_cuda_gpu_stage_node_enqueue(const ggml_tensor *node,
                                                 cudaStream_t stream) {
  if (!node || node->op != GGML_OP_CPY || !node->extra || !stream)
    return false;
  auto *ctx = static_cast<Qwen4ExpGpuStageContext *>(node->extra);
  if (ctx->magic != Qwen4ExpGpuStageContext::kMagic || !node->src[2] ||
      !node->src[2]->data || !ctx->host_ids || !ctx->host_gate_bounce ||
      !ctx->host_up_bounce || !ctx->host_down_bounce ||
      !ctx->device_gate_dst || !ctx->device_up_dst || !ctx->device_down_dst ||
      !ctx->d2h_start || !ctx->d2h_end || !ctx->h2d_start || !ctx->h2d_end)
    return false;

  const size_t gate_total = static_cast<size_t>(ctx->topk) * ctx->gate_bytes;
  const size_t up_total = static_cast<size_t>(ctx->topk) * ctx->up_bytes;
  const size_t down_total = static_cast<size_t>(ctx->topk) * ctx->down_bytes;
  if (cudaEventRecord(ctx->d2h_start, stream) != cudaSuccess ||
      cudaMemcpyAsync(ctx->host_ids, node->src[2]->data, ctx->ids_bytes,
                      cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
      cudaEventRecord(ctx->d2h_end, stream) != cudaSuccess ||
      cudaLaunchHostFunc(stream, qwen4exp_cuda_gpu_stage_callback, ctx) !=
          cudaSuccess ||
      cudaEventRecord(ctx->h2d_start, stream) != cudaSuccess ||
      cudaMemcpyAsync(ctx->device_gate_dst, ctx->host_gate_bounce, gate_total,
                      cudaMemcpyHostToDevice, stream) != cudaSuccess ||
      cudaMemcpyAsync(ctx->device_up_dst, ctx->host_up_bounce, up_total,
                      cudaMemcpyHostToDevice, stream) != cudaSuccess ||
      cudaMemcpyAsync(ctx->device_down_dst, ctx->host_down_bounce, down_total,
                      cudaMemcpyHostToDevice, stream) != cudaSuccess ||
      cudaEventRecord(ctx->h2d_end, stream) != cudaSuccess)
    return false;
  return true;
}

// Decode-only router-usage tap (engine.qwen4exp_track_router_usage): every
// layer's real routed `ids` is appended (via an ordinary ggml_concat node --
// a ususal graph op, no host bridge, no manual backend pin needed since it
// inherits CUDA placement from its argsort_top_k ancestor) into ONE
// persistent [topk * n_layers] I32 device accumulator
// (FullGraph::usage_track_ids), in layer order. A SINGLE marker after the
// whole layer loop D2H-copies that one buffer and bumps
// StreamingEngine::route_freq for every (layer, expert) entry in one host
// callback -- one host-bridge dispatch per token instead of n_layers, unlike
// a naive per-layer tap. See engine.qwen4exp_track_router_usage's own
// comment for why this is decode-only (S==1): the argsort-view-stride
// pitfall documented at build_moe's ids_out site does not apply here since
// there is exactly one row per layer.
struct Qwen4ExpUsageTrackContext {
  static constexpr uint64_t kMagic = 0x515552544b475431ull; // "QURTKGT1" trunc
  uint64_t magic = kMagic;
  StreamingEngine *engine = nullptr;
  int n_layers = 0;
  int n_experts = 0;
  int topk = 0;
  int32_t *host_ids = nullptr; // pinned, n_layers * topk entries
  size_t ids_bytes = 0;
  std::atomic<double> last_callback_ms{0.0};
};

static void CUDART_CB qwen4exp_cuda_usage_track_callback(void *user_data) {
  auto *ctx = static_cast<Qwen4ExpUsageTrackContext *>(user_data);
  if (!ctx || ctx->magic != Qwen4ExpUsageTrackContext::kMagic || !ctx->engine ||
      !ctx->host_ids)
    return;
  const auto start = std::chrono::steady_clock::now();
  for (int l = 0; l < ctx->n_layers; ++l) {
    for (int k = 0; k < ctx->topk; ++k) {
      const int32_t expert = ctx->host_ids[static_cast<size_t>(l) * ctx->topk + k];
      if (expert < 0 || expert >= ctx->n_experts)
        continue; // defensive only -- real routing always produces valid ids
      ctx->engine->bump_route_freq(l, expert);
    }
  }
  const double ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - start)
                        .count();
  ctx->last_callback_ms.store(ms, std::memory_order_release);
  if (std::getenv("MOESTREAM_QWEN4EXP_USAGE_TRACK_TIMING"))
    LOG_INFO("[QWEN4EXP-USAGE-TRACK-TIMING] callback_body_ms=%.4f\n", ms);
}

// Enqueues the single D2H + host-callback sequence for the usage-track flush
// marker while a CUDA stream is being captured -- same technique as
// moestream_qwen4exp_cuda_host_node_enqueue below, dispatched by magic. No
// H2D write-back: this marker is a pure read-only observer of already-real
// routing, it never feeds back into the compute graph.
static bool qwen4exp_cuda_usage_track_node_enqueue(const ggml_tensor *node,
                                                    cudaStream_t stream) {
  if (!node || node->op != GGML_OP_CPY || !node->extra || !stream)
    return false;
  auto *ctx = static_cast<Qwen4ExpUsageTrackContext *>(node->extra);
  if (ctx->magic != Qwen4ExpUsageTrackContext::kMagic || !node->src[0] ||
      !node->src[0]->data || !ctx->host_ids)
    return false;
  if (cudaMemcpyAsync(ctx->host_ids, node->src[0]->data, ctx->ids_bytes,
                      cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
      cudaLaunchHostFunc(stream, qwen4exp_cuda_usage_track_callback, ctx) !=
          cudaSuccess)
    return false;
  return true;
}

static bool env_host_node_skip_cpu() {
  const char *v = std::getenv("MOESTREAM_QWEN4EXP_HOST_NODE_SKIP_CPU");
  return v && v[0] == '1';
}

// Diagnostic topology switch: keep the initial CPU->GPU contiguous handoff
// on the CPU backend while retaining the CUDA-owned host-node markers.  This
// lets A/B runs separate the prologue/backend-edge cost from host-callback
// CPU contention without changing the default host-node topology.
static bool env_host_node_keep_cpu_prologue() {
  const char *v =
      std::getenv("MOESTREAM_QWEN4EXP_HOST_NODE_KEEP_CPU_PROLOGUE");
  return v && v[0] == '1';
}

// The CUDA host callback and the ordinary GGML CPU scheduler otherwise use
// independent execution contexts.  During the callback's CPU critical
// section, starting another CPU split causes the two pools to compete for
// the same cores and memory bandwidth.  The scheduler-side weak hook (added
// by patches/ggml-backend-host-node-gate.patch) waits on this counter.  It is
// deliberately process-global because a replay has one host callback stream
// per active MoE marker, and the guarded region is short.
static std::atomic<unsigned> g_qwen4exp_host_node_cpu_active{0};
static std::atomic<uint64_t> g_qwen4exp_host_node_cpu_gate_wait_ns{0};
static std::atomic<unsigned> g_qwen4exp_ordinary_cpu_active{0};
static std::atomic<uint64_t> g_qwen4exp_host_node_ordinary_cpu_wait_ns{0};
static std::atomic<uint64_t> g_qwen4exp_cpu_split_enter_wait_ns{0};
static std::mutex g_qwen4exp_cpu_execution_mutex;

static bool qwen4exp_host_node_timing_enabled() {
  const char *env = std::getenv("MOESTREAM_QWEN4EXP_HOST_NODE_TIMING");
  return env && env[0] == '1';
}

extern "C" bool moestream_qwen4exp_cpu_offload_coordination_enabled() {
  const char *v = std::getenv("MOESTREAM_QWEN4EXP_HOST_NODE");
  return v && v[0] == '1';
}

extern "C" void moestream_qwen4exp_host_node_cpu_wait() {
  if (g_qwen4exp_host_node_cpu_active.load(std::memory_order_acquire) == 0)
    return;

  const auto start = std::chrono::steady_clock::now();
  while (g_qwen4exp_host_node_cpu_active.load(std::memory_order_acquire) !=
         0) {
    // The callback normally lasts only a few milliseconds. Yielding here
    // gives its worker pool a runnable core without creating another busy
    // spinning CPU pool. This hook is only called before ordinary CPU splits.
    std::this_thread::yield();
  }
  const uint64_t wait_ns = static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now() - start)
          .count());
  g_qwen4exp_host_node_cpu_gate_wait_ns.fetch_add(
      wait_ns, std::memory_order_relaxed);
  if (wait_ns >= 50000 && qwen4exp_host_node_timing_enabled()) {
    LOG_INFO("[DIAG-SCHEDULER] CPU split blocked for %.3f ms "
             "waiting on host-node\n",
             static_cast<double>(wait_ns) / 1000000.0);
  }
}

// Admission protocol for the two independent CPU execution contexts.  The
// retry after increment closes the race where a scheduler split observes an
// idle host callback immediately before the callback publishes itself active.
// This keeps the callback worker pool from competing with an already-running
// ordinary GGML CPU split without taking a mutex on the normal path.
extern "C" void moestream_qwen4exp_cpu_split_enter() {
  const auto wait_start = std::chrono::steady_clock::now();
  g_qwen4exp_cpu_execution_mutex.lock();
  g_qwen4exp_cpu_split_enter_wait_ns.fetch_add(
      static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now() - wait_start)
                                .count()),
      std::memory_order_relaxed);
  g_qwen4exp_ordinary_cpu_active.fetch_add(1, std::memory_order_acq_rel);
}

extern "C" void moestream_qwen4exp_cpu_split_exit() {
  g_qwen4exp_ordinary_cpu_active.fetch_sub(1, std::memory_order_acq_rel);
  g_qwen4exp_cpu_execution_mutex.unlock();
}

struct Qwen4ExpHostNodeCpuExecutionGuard {
  const std::chrono::steady_clock::time_point start =
      std::chrono::steady_clock::now();
  std::unique_lock<std::mutex> lock;
  explicit Qwen4ExpHostNodeCpuExecutionGuard()
      : lock(g_qwen4exp_cpu_execution_mutex) {
    const uint64_t wait_ns = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - start)
            .count());
    g_qwen4exp_host_node_ordinary_cpu_wait_ns.fetch_add(
        wait_ns, std::memory_order_relaxed);
  }
};

static double qwen4exp_host_node_ordinary_cpu_wait_ms_since_last() {
  const uint64_t wait_ns = g_qwen4exp_host_node_ordinary_cpu_wait_ns.exchange(
      0, std::memory_order_acq_rel);
  return static_cast<double>(wait_ns) / 1000000.0;
}

static double qwen4exp_host_node_gate_wait_ms_since_last() {
  const uint64_t wait_ns = g_qwen4exp_host_node_cpu_gate_wait_ns.exchange(
      0, std::memory_order_acq_rel);
  return static_cast<double>(wait_ns) / 1000000.0;
}

static double qwen4exp_cpu_split_enter_wait_ms_since_last() {
  const uint64_t wait_ns = g_qwen4exp_cpu_split_enter_wait_ns.exchange(
      0, std::memory_order_acq_rel);
  return static_cast<double>(wait_ns) / 1000000.0;
}

struct Qwen4ExpHostNodeCpuActivityGuard {
  Qwen4ExpHostNodeCpuActivityGuard() {
    g_qwen4exp_host_node_cpu_active.fetch_add(1, std::memory_order_acq_rel);
  }
  ~Qwen4ExpHostNodeCpuActivityGuard() {
    g_qwen4exp_host_node_cpu_active.fetch_sub(1, std::memory_order_acq_rel);
  }
};

static uint64_t qwen4exp_fnv64(const void *data, size_t bytes) {
  constexpr uint64_t kOffset = 0xcbf29ce484222325ull;
  constexpr uint64_t kPrime = 0x100000001b3ull;
  uint64_t hash = kOffset;
  const auto *p = static_cast<const unsigned char *>(data);
  for (size_t i = 0; i < bytes; ++i) {
    hash ^= p[i];
    hash *= kPrime;
  }
  return hash;
}

static const char *env_layer13_moe_dump_path() {
  const char *path =
      std::getenv("MOESTREAM_QWEN4EXP_LAYER13_MOE_DUMP");
  return path && path[0] ? path : nullptr;
}

static const char *env_layer13_moe_input_dump_path() {
  const char *path =
      std::getenv("MOESTREAM_QWEN4EXP_LAYER13_MOE_INPUT_DUMP");
  return path && path[0] ? path : nullptr;
}

static const char *env_layer13_moe_host_input_dump_path() {
  const char *path =
      std::getenv("MOESTREAM_QWEN4EXP_LAYER13_MOE_HOST_INPUT_DUMP");
  return path && path[0] ? path : env_layer13_moe_input_dump_path();
}

static const char *env_layer13_moe_ordinary_input_dump_path() {
  const char *path =
      std::getenv("MOESTREAM_QWEN4EXP_LAYER13_MOE_ORDINARY_INPUT_DUMP");
  return path && path[0] ? path : env_layer13_moe_input_dump_path();
}

static bool qwen4exp_dump_f32(const char *path, const float *data,
                              size_t elements) {
  if (!path || !data)
    return false;
  std::FILE *file = std::fopen(path, "wb");
  if (!file)
    return false;
  const size_t written =
      std::fwrite(data, sizeof(float), elements, file);
  const bool closed = std::fclose(file) == 0;
  return written == elements && closed;
}

static void CUDART_CB qwen4exp_cuda_host_node_callback(void *user_data) {
  static thread_local bool thread_configured = false;
  if (!thread_configured) {
    configure_current_thread({
        .name = "QWEN-CUDA-CB",
        // Leave the driver callback unpinned.  A fixed orchestration core
        // contended with the ordinary GGML CPU pool in the first A/B run.
        .cpu_ids = {},
        .use_rt = true,
        .rt_priority = 85,
        .nice_value = -20,
        .apply_nice = false,
    });
    thread_configured = true;
  }
  auto *ctx = static_cast<Qwen4ExpCudaHostNodeContext *>(user_data);
  if (!ctx || ctx->magic != Qwen4ExpCudaHostNodeContext::kMagic)
    return;

  const auto callback_entry = std::chrono::steady_clock::now();
  Qwen4ExpHostNodeCpuActivityGuard activity_guard;
  const auto guard_wait_start = std::chrono::steady_clock::now();
  Qwen4ExpHostNodeCpuExecutionGuard execution_guard;
  const double guard_wait_ms = std::chrono::duration<double, std::milli>(
                                   std::chrono::steady_clock::now() -
                                   guard_wait_start)
                                   .count();

  // A callback context is replayed across tokens. Invalidate the previous
  // token's diagnostic result before doing this token's CPU work so a failed
  // callback can never be reported with a stale hash.
  if (ctx->transport_diag)
    ctx->last_cpu_hash_valid.store(false, std::memory_order_release);

  const auto start = std::chrono::steady_clock::now();
  if (qwen4exp_host_node_timing_enabled()) {
    int resident = 0;
    for (int k = 0; k < ctx->topk; ++k) {
      const int expert = ctx->host_ids[k];
      if (std::find(ctx->gpu_resident_experts.begin(),
                    ctx->gpu_resident_experts.end(), expert) !=
          ctx->gpu_resident_experts.end())
        ++resident;
    }
    LOG_INFO("[QWEN4EXP-HOST-NODE-ROUTES] layer=%d routed=%d resident=%d "
             "cold=%d static_hot=%zu callback_mode=full-cpu\n",
             ctx->layer_index, ctx->topk, resident, ctx->topk - resident,
             ctx->gpu_resident_experts.size());
    if (resident > 0) {
      LOG_WARN("[QWEN4EXP-HOST-NODE-ROUTES] layer=%d host marker is "
               "computing routes that have a static GPU copy; this marker "
               "does not yet compact the cold subset\n",
               ctx->layer_index);
    }
  }
  qwen_cpu_moe_diag_set_layer(ctx->layer_index);
  bool ok = false;
  GgmlQwenCpuMoeRunner *runner = nullptr;
  if (env_host_node_skip_cpu()) {
    std::fill_n(ctx->host_output, ctx->output_bytes / sizeof(float), 0.0f);
    ok = true;
  } else if (ctx->projection_compact) {
    runner = ctx->cpu_runner;
    if (!runner)
      runner = &GgmlQwenCpuMoeRunner::instance_for_strided_planes(
          ctx->layer, ctx->n_experts, ctx->d_model, ctx->d_ff, ctx->topk,
          ctx->gate_stride, ctx->up_stride, ctx->down_stride, false);
    if (!runner->uses_external_threadpool())
      runner->resume_workers();
    ok = runner->run_strided_planes(
        ctx->host_input, ctx->host_output, ctx->layer, ctx->gate_plane,
        ctx->up_plane, ctx->down_plane, ctx->host_ids, ctx->n_experts,
        ctx->host_weights, ctx->d_model, ctx->d_ff, ctx->topk,
        ctx->gate_stride, ctx->up_stride, ctx->down_stride);
  } else {
    runner = ctx->cpu_runner;
    if (!runner)
      runner = &GgmlQwenCpuMoeRunner::instance_for_strided(
          ctx->layer, ctx->n_experts, ctx->d_model, ctx->d_ff, ctx->topk,
          ctx->expert_stride, false);
    if (!runner->uses_external_threadpool())
      runner->resume_workers();
    ok = runner->run_strided(
        ctx->host_input, ctx->host_output, ctx->layer, ctx->gate_plane,
        ctx->up_plane, ctx->down_plane, ctx->host_ids, ctx->n_experts,
        ctx->host_weights, ctx->d_model, ctx->d_ff, ctx->topk,
        ctx->expert_stride);
  }
  if (runner && !runner->uses_external_threadpool())
    runner->pause_workers();
  const double cpu_ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - start)
                            .count();
  ctx->last_callback_guard_wait_ms.store(guard_wait_ms,
                                         std::memory_order_release);
  ctx->last_callback_body_ms.store(cpu_ms, std::memory_order_release);
  if (ctx->transport_diag && ctx->host_output && ctx->output_bytes != 0) {
    // This is the exact byte sequence supplied to the subsequent H2D copy.
    // Keep this inside the callback, immediately after the CPU runner, so a
    // later consumer cannot be mistaken for a transport diagnostic.
    ctx->last_cpu_hash.store(qwen4exp_fnv64(ctx->host_output,
                                            ctx->output_bytes),
                             std::memory_order_release);
    ctx->last_cpu_hash_valid.store(true, std::memory_order_release);
    if (ctx->layer_index == 13) {
      const uint64_t input_hash =
          qwen4exp_fnv64(ctx->host_input, ctx->input_bytes);
      const uint64_t ids_hash =
          qwen4exp_fnv64(ctx->host_ids, ctx->ids_bytes);
      const uint64_t weights_hash =
          qwen4exp_fnv64(ctx->host_weights, ctx->weights_bytes);
      LOG_INFO("[QWEN4EXP-LAYER13-ROUTE-DIAG] host input_fnv64=0x%016llx "
               "ids_fnv64=0x%016llx weights_fnv64=0x%016llx "
               "ids=[%d,%d,%d,%d,%d,%d,%d,%d,%d,%d]\n",
               static_cast<unsigned long long>(input_hash),
               static_cast<unsigned long long>(ids_hash),
               static_cast<unsigned long long>(weights_hash),
               ctx->host_ids[0], ctx->host_ids[1], ctx->host_ids[2],
               ctx->host_ids[3], ctx->host_ids[4], ctx->host_ids[5],
               ctx->host_ids[6], ctx->host_ids[7], ctx->host_ids[8],
               ctx->host_ids[9]);
      const char *input_dump = env_layer13_moe_host_input_dump_path();
      if (input_dump) {
        const bool dumped = qwen4exp_dump_f32(
            input_dump, ctx->host_input,
            ctx->input_bytes / sizeof(float));
        LOG_INFO("[QWEN4EXP-LAYER13-INPUT-DIAG] path=host-node "
                 "dump=%s file=%s elements=%zu\n",
                 dumped ? "ok" : "failed", input_dump,
                 ctx->input_bytes / sizeof(float));
      }
    }
  }
  qwen_cpu_moe_diag_set_layer(-1);
  ctx->last_ok.store(ok, std::memory_order_release);
  ctx->last_cpu_ms.store(cpu_ms, std::memory_order_release);
  const double callback_total_ms =
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - callback_entry)
          .count();
  ctx->last_callback_total_ms.store(callback_total_ms,
                                    std::memory_order_release);
  if (qwen4exp_host_node_timing_enabled()) {
    LOG_INFO("[DIAG-CUDA-CB] layer=%d total=%.3f ms body=%.3f ms "
             "guard_wait=%.3f ms ok=%d\n",
             ctx->layer_index, callback_total_ms, cpu_ms, guard_wait_ms,
             ok ? 1 : 0);
  }
}

// Runs immediately before the marker's D2H copies on replay.  The direct
// worker team was created during graph construction; this callback only arms
// its short active window so workers do not pay an idle futex wake while the
// activation/route metadata reaches the main CPU callback.
static void CUDART_CB qwen4exp_cuda_host_node_arm_direct_workers(
    void *user_data) {
  auto *ctx = static_cast<Qwen4ExpCudaHostNodeContext *>(user_data);
  if (!ctx || ctx->magic != Qwen4ExpCudaHostNodeContext::kMagic ||
      !ctx->cpu_runner)
    return;
  ctx->cpu_runner->arm_direct_workers();
}

// Called by the opt-in CUDA backend bridge while a CUDA stream is being
// captured.  All CUDA work is enqueued here, outside the callback itself;
// the callback only runs the synchronous CPU GGML MoE and returns.
extern "C" bool moestream_qwen4exp_cuda_host_node_enqueue(
    const ggml_tensor *node, cudaStream_t stream) {
  if (!node || node->op != GGML_OP_CPY || !node->extra || !stream)
    return false;
  // Two distinct marker context types share this one ggml-cuda weak-symbol
  // hook (see patches/ggml-backend-host-node-*.patch's own wiring) rather
  // than each needing its own patch to ggml-cuda.cu: the CPU-offload bridge
  // below (Qwen4ExpCudaHostNodeContext), and the generalized GPU-staging
  // bridge (Qwen4ExpGpuStageContext, see its own struct comment). Both put
  // a uint64_t `magic` as their first member, so a peek at that field alone
  // -- before committing to either concrete type -- is safe.
  {
    const auto *maybe_stage =
        static_cast<const Qwen4ExpGpuStageContext *>(node->extra);
    if (maybe_stage->magic == Qwen4ExpGpuStageContext::kMagic)
      return qwen4exp_cuda_gpu_stage_node_enqueue(node, stream);
  }
  {
    const auto *maybe_usage =
        static_cast<const Qwen4ExpUsageTrackContext *>(node->extra);
    if (maybe_usage->magic == Qwen4ExpUsageTrackContext::kMagic)
      return qwen4exp_cuda_usage_track_node_enqueue(node, stream);
  }
  auto *ctx = static_cast<Qwen4ExpCudaHostNodeContext *>(node->extra);
  if (ctx->magic != Qwen4ExpCudaHostNodeContext::kMagic ||
      !node->data || !node->src[0] || !node->src[2] || !node->src[3] ||
      !node->src[0]->data || !node->src[2]->data || !node->src[3]->data ||
      !ctx->host_input || !ctx->host_output || !ctx->host_ids ||
      !ctx->host_weights || !ctx->d2h_start || !ctx->d2h_end ||
      !ctx->h2d_start || !ctx->h2d_end)
    return false;

  ctx->device_output = node->data;
  const char *direct_pool_env =
      std::getenv("MOESTREAM_QWEN_CPU_DIRECT_POOL");
  const char *direct_fused_env =
      std::getenv("MOESTREAM_QWEN_CPU_DIRECT_FUSED");
  const bool direct_pool_requested =
      direct_pool_env && direct_pool_env[0] == '1' &&
      !(direct_fused_env && direct_fused_env[0] == '1');
  if (direct_pool_requested &&
      cudaLaunchHostFunc(stream, qwen4exp_cuda_host_node_arm_direct_workers,
                         ctx) != cudaSuccess)
    return false;
  if (cudaEventRecord(ctx->d2h_start, stream) != cudaSuccess ||
      cudaMemcpyAsync(ctx->host_input, node->src[0]->data, ctx->input_bytes,
                      cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
      cudaMemcpyAsync(ctx->host_ids, node->src[2]->data, ctx->ids_bytes,
                      cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
      cudaMemcpyAsync(ctx->host_weights, node->src[3]->data,
                      ctx->weights_bytes, cudaMemcpyDeviceToHost, stream) !=
          cudaSuccess ||
      cudaEventRecord(ctx->d2h_end, stream) != cudaSuccess ||
      cudaLaunchHostFunc(stream, qwen4exp_cuda_host_node_callback, ctx) !=
          cudaSuccess ||
      cudaEventRecord(ctx->h2d_start, stream) != cudaSuccess ||
      cudaMemcpyAsync(node->data, ctx->host_output, ctx->output_bytes,
                      cudaMemcpyHostToDevice, stream) != cudaSuccess ||
      cudaEventRecord(ctx->h2d_end, stream) != cudaSuccess)
    return false;
  if (ctx->transport_diag) {
    if (!ctx->host_output_gpu_check || !ctx->post_h2d_check_end ||
        cudaMemcpyAsync(ctx->host_output_gpu_check, node->data,
                        ctx->output_bytes, cudaMemcpyDeviceToHost, stream) !=
            cudaSuccess ||
        cudaEventRecord(ctx->post_h2d_check_end, stream) != cudaSuccess)
      return false;
  }
  return true;
}

namespace {

extern "C" cudaStream_t moestream_ggml_backend_cuda_get_stream(
    ggml_backend_t backend);
extern "C" cudaGraph_t moestream_ggml_cuda_get_captured_graph(
    ggml_backend_t backend, const ggml_tensor *first_node);
extern "C" void *moestream_qwen4exp_master_predicate_kernel_ptr();

constexpr int kGraphSize = 50000;
constexpr size_t kTensorContextBytes = 128ull * 1024ull * 1024ull;

struct HostAllocation {
  void *ptr = nullptr;
  size_t bytes = 0;
  bool pinned = false;
  size_t reservation_bytes = 0;
};

struct FullGraph {
  StreamingEngine *engine = nullptr;
  int device = -1;
  int S = 0;
  int D = 0;
  int HC = 0;
  int n_layers = 0;
  int H = 0;
  int Hkv = 0;
  int hd = 0;
  int Hk_gdn = 0;
  int Hv_gdn = 0;
  int gdn_hd = 0;
  int key_dim = 0;
  int value_dim = 0;
  int conv_dim = 0;
  int d_conv = 0;
  bool decode_mode = false;
  int kv_capacity = 0;
  int decode_last_pos = -1;

  ggml_backend_t cuda = nullptr;
  ggml_backend_t cpu = nullptr;
  ggml_threadpool_t cpu_threadpool = nullptr;
  int cpu_threads = 0;
  ggml_backend_sched_t sched = nullptr;
  // Per-layer CPU timing instrumentation (env: MOESTREAM_QWEN4EXP_GRAPH_
  // LAYER_TIMING=1). Uses ggml_backend_sched_set_eval_callback, which fires
  // synchronously around each CPU-backend node's real compute (ggml-cpu has
  // no async concept) -- gives real per-layer/per-op CPU time that neither
  // nsys nor the legacy per-token DEBUG counters can see for full-graph
  // decode. See memory: full-graph-per-layer-timing-instrumentation-plan.md.
  // All empty/false and zero-cost when the env var is unset.
  bool layer_timing_enabled = false;
  std::vector<double> layer_timing_cpu_ms;      // indexed by layer
  std::unordered_map<std::string, double> layer_timing_op_ms; // keyed by stem
  std::unordered_map<const ggml_tensor *,
                     std::chrono::steady_clock::time_point>
      layer_timing_t0;
  long layer_timing_tokens_since_emit = 0;
  // Gap between consecutive CPU-backend nodes' ask=false and the next
  // ask=true -- the dispatch/barrier/scheduling overhead the op-bracket
  // timing above cannot see (it only times INSIDE one node's bracket).
  std::chrono::steady_clock::time_point layer_timing_last_ask_false{};
  bool layer_timing_have_last_ask_false = false;
  double layer_timing_gap_ms_total = 0.0;
  long layer_timing_gap_count = 0;
  // One-shot verbose node/backend dump (env: MOESTREAM_QWEN4EXP_GRAPH_
  // LAYER_TIMING_VERBOSE=1) -- see env_layer_timing_verbose()'s comment.
  bool layer_timing_verbose_enabled = false;
  bool layer_timing_verbose_dumped = false;
  int layer_timing_verbose_tokens_seen = 0;
  std::chrono::steady_clock::time_point layer_timing_verbose_last{};
  bool layer_timing_verbose_have_last = false;
  // Op result tensors (mul_mat_id/swiglu outputs inside build_moe()) get
  // ggml's auto-generated "node_NNNN" name, not a "blk.N.stem" name --
  // only the persistent WEIGHT leaf tensors are explicitly named by
  // expert_weight()/etc. Recorded once at graph-build time (the graph is
  // reused across tokens, so this never needs updating per-token) so the
  // eval callback can attribute a result tensor to its layer without
  // relying on tensor names at all.
  std::unordered_map<const ggml_tensor *, std::pair<int, const char *>>
      layer_timing_layer_of; // tensor -> (layer, op stem label)
  // Opt-in Layer-0 quantization-path diagnostic. These graph-owned
  // intermediates are retained only so the first compute can dump the
  // actual gate/up values; the normal path never populates or reads them.
  bool layer0_diag_enabled = false;
  bool layer0_diag_dumped = false;
  ggml_tensor *layer0_diag_gate = nullptr;
  ggml_tensor *layer0_diag_up = nullptr;
  ggml_tensor *layer0_diag_act = nullptr;
  ggml_tensor *layer0_diag_down = nullptr;
  ggml_tensor *layer0_diag_gate_up = nullptr;
  ggml_tensor *layer0_diag_residual_before = nullptr;
  ggml_tensor *layer0_diag_moe_block = nullptr;
  ggml_tensor *layer0_diag_shared = nullptr;
  ggml_tensor *layer0_diag_moe_plus_shared = nullptr;
  ggml_tensor *layer0_diag_residual_after = nullptr;
  ggml_tensor *layer0_diag_next_consumer = nullptr;
  ggml_tensor *layer0_diag_residual_before_snapshot = nullptr;
  ggml_tensor *layer0_diag_moe_plus_shared_snapshot = nullptr;
  ggml_tensor *layer0_diag_residual_after_snapshot = nullptr;
  ggml_tensor *layer0_diag_next_consumer_snapshot = nullptr;
  // Actual graph-owned outputs for the two-layer S=1 diagnostic. These are
  // deliberately not independent cpy leaves: each tensor remains on the
  // Layer-0 -> Layer-1 dependency chain and is marked as a graph output.
  bool two_layer_diag = false;
  ggml_tensor *two_layer_l0_post_ffn = nullptr;
  ggml_tensor *two_layer_l1_pre_attn = nullptr;
  ggml_tensor *two_layer_l1_post_attn = nullptr;
  ggml_tensor *two_layer_l1_post_ffn = nullptr;
  // Optional exact hidden-state snapshots at the same semantic boundary as
  // beellama's l_last-0/l_last-1 callbacks: after each layer's attention and
  // FFN residual combine.  These are intentionally limited to layers 0/1.
  ggml_tensor *hidden01_snapshot[2] = {nullptr, nullptr};
  bool hidden01_dumped = false;
  std::vector<std::pair<std::string, ggml_tensor *>> layer0_split_snapshots;
  bool layer0_split_dumped = false;
  // Optional one-shot raw routed-MoE output dump for Layer 13. This is
  // deliberately before the shared expert and residual combine so the
  // ordinary GGML CPU path and the host-node callback are compared at the
  // same semantic boundary.
  bool layer13_moe_diag_enabled = false;
  bool layer13_moe_diag_dumped = false;
  ggml_tensor *layer13_moe_diag = nullptr;
  ggml_tensor *layer13_moe_input_diag = nullptr;
  ggml_tensor *layer13_moe_ids_diag = nullptr;
  ggml_tensor *layer13_moe_weights_diag = nullptr;
  ggml_backend_buffer_t device_expert_buffer = nullptr;
  void *device_expert_base = nullptr;
  size_t device_expert_capacity = 0;
  size_t device_expert_used = 0;
  ggml_backend_buffer_t device_dense_buffer = nullptr;
  void *device_dense_base = nullptr;
  size_t device_dense_capacity = 0;
  size_t device_dense_used = 0;
  int device_dense_tensors = 0;
  // Diagnostic (2026-09-17): cumulative time inside resident_tensor()'s own
  // cudaMemcpy H2D calls, across this ONE build_graph() call -- see memory:
  // full-graph-decode-graph-construction-5s-latency.md. Reset to 0 at the
  // top of build_graph(), read+logged at the bottom, alongside the
  // node_construction/sched_alloc_graph split already there.
  double dense_upload_ms_total = 0.0;
  // Same idea, for bind_device_expert()'s own cudaMemcpy (the 17.68 GiB,
  // 22-GPU-resident-layer upload -- the other real candidate for where
  // node_construction's time goes, besides dense_upload_ms_total above).
  double expert_upload_ms_total = 0.0;
  long expert_upload_calls = 0;
  // Diagnostic (2026-09-17): further split of node_construction beyond the
  // cudaMemcpy timers above, which only accounted for ~28-44% of the total.
  // Buckets the REST of the per-layer node-construction work (host-side
  // ggml_tensor/op building, not any device transfer) into attention-block
  // vs MoE-block, to find where the remaining ~55-72% actually goes.
  double attn_construction_ms_total = 0.0;
  double moe_construction_ms_total = 0.0;
  // Finer split of moe_construction_ms_total: time inside the weight-tensor
  // lookup calls (expert_weight_split/expert_weight/expert_gate_up_weight --
  // model_tensor lookup + preloaded_ram_expert_source + bind_host/
  // bind_device_expert*) vs build_moe()'s own ggml graph-node creation.
  double moe_weight_lookup_ms_total = 0.0;
  ggml_backend_buffer_t device_state_buffer = nullptr;
  void *device_state_base = nullptr;
  size_t device_state_capacity = 0;
  size_t device_state_used = 0;
  int device_expert_layers = 0;
  bool device_expert_partial_gate_up = false;
  // When true, the device_expert_layers-sized GPU-resident block sits at
  // the END of the model (layers [n_layers-N, n_layers)) instead of the
  // START (layers [0, N)). Added specifically to match beellama's own
  // `-ncmoe N` convention: llama-bench forces layers 0..N-1 to CPU via
  // llm_ffn_block_regex(i, ...) for i in [0,N), which leaves the LAST
  // n_layers-N layers GPU-resident -- the opposite end from MoEStream's
  // historical leading-N convention. Every same-N comparison run before
  // this flag existed compared GPU residency on genuinely different
  // layers between the two engines, not just a different count. See
  // [[full-graph-mixed-decode-root-cause-cpu-backend-no-events]]'s FIFTH
  // ROUND for the discovery. device_expert_partial_gate_up's boundary-
  // layer optimization is forced off whenever this is set (see its own
  // setter) rather than generalizing its own boundary math -- v1 scope.
  bool device_expert_trailing = false;
  // Per-layer plan produced once by plan_expert_vram_split(), BEFORE
  // init_device_expert_arena(): layer -> absolute expert ids chosen for
  // VRAM residency (the hottest-by-usage subset a fixed VRAM budget could
  // afford), for layers cpu_affinity_layers/device_expert_layers did not
  // already claim. Empty/absent entry = no split for that layer (today's
  // behavior, unchanged). v1 scope: decode only (S==1) -- see build_moe.
  std::unordered_map<int, std::vector<int>> split_gpu_expert_ids;
  // Phase 2 suffix graphs borrow the immutable VRAM expert arena from the
  // prefix owner.  Expert tensors are keyed by their source layout because
  // the same logical layer can be represented as a full, strided, gathered,
  // or compact gate/up tensor depending on the graph path.
  FullGraph *phase2_expert_source_graph = nullptr;
  std::unordered_map<std::string, void *> phase2_expert_addrs;
  // Layer-13 staging proof. The graph keeps fixed compact GPU tensors; each
  // decode token refreshes one or more cold expert records from the
  // active-layer pool before graph replay.
  bool stage_enabled = false;
  bool stage_route_logged = false;
  int stage_layer = -1;
  int stage_expert = -1;
  int stage_local_index = -1;
  struct StageBinding {
    int expert = -1;
    int local_index = -1;
    const uint8_t *src_gate = nullptr;
    const uint8_t *src_up = nullptr;
    const uint8_t *src_down = nullptr;
    size_t gate_bytes = 0;
    size_t up_bytes = 0;
    size_t down_bytes = 0;
    uint8_t *dst_gate = nullptr;
    uint8_t *dst_up = nullptr;
    uint8_t *dst_down = nullptr;
    ExpertLocation *slot = nullptr;
  };
  std::vector<int> stage_experts;
  std::vector<StageBinding> stage_bindings;
  std::vector<ExpertLocation *> stage_slots;
  // Stable per-layer contexts for the opt-in CUDA host-node bridge. These
  // must outlive graph capture and every replay; they are cleared only after
  // the scheduler/graph objects have been torn down.
  std::vector<std::unique_ptr<Qwen4ExpCudaHostNodeContext>> host_node_contexts;
  // Generalized GPU staging for host-resident decode layers (see
  // engine.qwen4exp_gpu_stage_enabled/_layers). One binding per staged
  // layer: a fixed, topk-sized compact gate/up/down destination in the
  // shared device_expert_buffer bump arena (see device_expert_bytes()'s own
  // staging-bytes block), refreshed every decode token via the same
  // CUDA-graph host-node-callback technique the CPU-offload bridge above
  // uses -- but the callback here does a pure host-memory GATHER (real
  // routed experts' bytes copied into a compact pinned bounce buffer) in
  // place of CPU FFN compute, feeding ordinary GPU MUL_MAT_ID nodes instead
  // of producing the final answer directly. See build_gpu_stage_node().
  struct GpuStageBinding {
    int layer = -1;
    // [topk, S] I32, value = 0..topk-1, uploaded once at graph-build time,
    // never touched again -- see build_gpu_stage_node's own comment for why
    // a real per-token id remap table is unnecessary.
    ggml_tensor *ids_dispatch_const = nullptr;
    // Fixed compact device tensors, shape [cols, rows, topk] (topk, not
    // n_experts), bound once via phase2_bind_or_allocate_expert into
    // device_expert_buffer -- never re-bound, only their CONTENTS change
    // per token.
    ggml_tensor *gate_dst = nullptr;
    ggml_tensor *up_dst = nullptr;
    ggml_tensor *down_dst = nullptr;
    // The GGML_OP_CPY marker node that refreshes gate/up/down_dst's
    // contents each replay. build_moe attaches this as an extra src[] on
    // every MUL_MAT_ID call that reads them, purely to force correct
    // dependency ordering -- see build_gpu_stage_node's own comment.
    ggml_tensor *gpu_stage_marker_dep = nullptr;
  };
  std::vector<GpuStageBinding> gpu_stage_bindings;
  std::vector<std::unique_ptr<struct Qwen4ExpGpuStageContext>>
      gpu_stage_contexts;
  // Decode-only router-usage tap (engine.qwen4exp_track_router_usage) -- see
  // Qwen4ExpUsageTrackContext's own comment. usage_track_ids accumulates
  // every layer's real routed ids via ordinary ggml_concat nodes across the
  // layer loop; usage_track_context owns the single flush marker's pinned
  // buffer/state, built once after the loop.
  ggml_tensor *usage_track_ids = nullptr;
  std::unique_ptr<struct Qwen4ExpUsageTrackContext> usage_track_context;
  const uint8_t *stage_src_gate = nullptr;
  const uint8_t *stage_src_up = nullptr;
  const uint8_t *stage_src_down = nullptr;
  size_t stage_gate_bytes = 0;
  size_t stage_up_bytes = 0;
  size_t stage_down_bytes = 0;
  uint8_t *stage_dst_gate = nullptr;
  uint8_t *stage_dst_up = nullptr;
  uint8_t *stage_dst_down = nullptr;
  ExpertLocation *stage_slot = nullptr;
  cudaEvent_t stage_wait_done_event = nullptr;
  cudaEvent_t stage_wait_start_event = nullptr;
  cudaEvent_t stage_d2d_start_event = nullptr;
  cudaEvent_t stage_d2d_end_event = nullptr;
  cudaEvent_t stage_compute_start_event = nullptr;
  cudaEvent_t stage_compute_end_event = nullptr;
  ggml_tensor *stage_route_ids = nullptr;
  // Phase 1 partial-layer dispatch: prefill-only, non-null exactly when
  // this graph is building engine->qwen4exp_partial_layer's MoE block
  // during a non-decode (S>1) build. Marked as a graph output so
  // qwen4exp_full_graph_run can read back this prompt's real routing for
  // that layer and decide, once, whether decode may commit to the
  // GPU-only path. Shape [topk, S] I32, same layout ggml_argsort_top_k
  // itself produces.
  ggml_tensor *partial_layer_ids_out = nullptr;
  // Diagnostic sibling of partial_layer_ids_out: the partial layer's raw
  // post-softmax router distribution ([n_experts, S]), for checking
  // whether routing is genuinely peaked or effectively near-uniform noise.
  ggml_tensor *partial_layer_probs_out = nullptr;
  // Root-cause diagnostic (2026-09-18): the raw exp_probs_b.bias tensor
  // for the partial layer, exposed as a graph output so
  // qwen4exp_full_graph_run can dump its actual values -- see
  // [[qwen4exp-legacy-vs-fullgraph-routing-divergence]].
  ggml_tensor *partial_layer_bias_out = nullptr;

  // Phase 2: true per-token GPU/CPU dispatch for the partial layer,
  // validated via a standalone spike test (splitting a computation across
  // two separately-submitted ggml graphs with a persistent device-tensor
  // handoff -- see qwen4exp-full-graph-decode-benchmark-baseline memory's
  // follow-on note). Off (Full) is byte-identical to today's single-graph
  // decode for every other config; Prefix/SuffixGpu/SuffixCpu are three
  // DISTINCT FullGraph instances (own ctx/graph/sched) built once at
  // decode-prepare time, chosen per token by a cheap host-side readback of
  // the prefix's own layer routing.
  enum class Phase2Mode { Off, Prefix, SuffixGpu, SuffixCpu };
  Phase2Mode phase2_mode = Phase2Mode::Off;
  int phase2_split_layer = -1;
  // Prefix-only: the persistent handoff buffer this instance writes `res`
  // (the residual right after the split layer's attention) into. Suffix
  // instances read the prefix instance's own phase2_handoff_buf/addr
  // directly (the prefix instance IS g_graph, phase2_mode==Prefix -- see
  // the g_graph global's own comment) and bind a fresh leaf tensor to the
  // SAME address in their own ctx -- never touch this pair themselves.
  ggml_backend_buffer_t phase2_handoff_buf = nullptr;
  void *phase2_handoff_addr = nullptr;
  // SuffixGpu and SuffixCpu build the SAME layer range [split_layer+1,
  // n_layers) and therefore issue new_device_state_tensor() calls in an
  // IDENTICAL sequence (GDN/dense-attn/PLE classification per layer is a
  // static model property, not decode-time data) -- but only ONE of them
  // runs per token, so their GDN/KV/PLE state for those shared layers
  // must be the SAME physical buffer, or it would silently diverge the
  // moment dispatch alternates between them. The instance built FIRST
  // (SuffixGpu) allocates for real and records each address in
  // phase2_state_addrs, in call order; the instance built SECOND
  // (SuffixCpu, via phase2_state_source_graph pointing at the first) replays
  // that list instead of allocating its own arena slots, binding a fresh
  // leaf in ITS OWN ctx to each recorded address in turn.
  FullGraph *phase2_state_source_graph = nullptr;
  std::vector<void *> phase2_state_addrs;
  size_t phase2_state_replay_cursor = 0;
  // Phase 2 sibling graphs use the same read-only dense-weight arena.  The
  // source graph owns the CUDA buffer; borrowers bind their own ggml tensor
  // handles to the source addresses.  Unlike the state arena above, dense
  // weights are keyed by model tensor name because prefix and suffix graphs
  // do not materialize the same layer range.
  FullGraph *phase2_dense_source_graph = nullptr;
  std::unordered_map<std::string, void *> phase2_dense_addrs;
  ggml_context *ctx = nullptr;
  ggml_cgraph *graph = nullptr;

  ggml_tensor *input = nullptr;         // [D, HC, S], CPU input
  ggml_tensor *topology_cpu_prologue = nullptr;
  ggml_tensor *output = nullptr;        // [D, S], final HC output
  ggml_tensor *logits = nullptr;        // [vocab, S], decode-only output
  ggml_tensor *final_streams = nullptr; // [D, HC, S], for state/debug writeback
  ggml_tensor *positions = nullptr;     // [S], I32
  ggml_tensor *attn_mask = nullptr;     // [S, S], F16 (flash-attn contract)
  ggml_tensor *decode_indices = nullptr; // [Hkv], I32 row indices for KV append
  ggml_tensor *ple_embedding = nullptr;
  ggml_tensor *ple_hist_out = nullptr;
  ggml_tensor *ple_hist_storage = nullptr;
  ggml_tensor *ple_hist_update = nullptr;

  std::vector<ggml_tensor *> kv_k;
  std::vector<ggml_tensor *> kv_v;
  std::vector<int> kv_layers;
  std::vector<ggml_tensor *> gdn_state_out;
  std::vector<ggml_tensor *> gdn_hist_out;
  std::vector<ggml_tensor *> kv_state_k;
  std::vector<ggml_tensor *> kv_state_v;
  std::vector<ggml_tensor *> gdn_state_storage;
  std::vector<ggml_tensor *> gdn_hist_storage;
  std::vector<ggml_tensor *> gdn_state_updates;
  std::vector<ggml_tensor *> gdn_hist_updates;
  std::vector<int> gdn_layers;
  std::vector<float *> gdn_state_inputs;
  std::vector<float *> gdn_hist_inputs;
  int ple_layer = -1;
  float *ple_hist_input = nullptr;

  std::vector<HostAllocation> owned;
  std::vector<ggml_backend_buffer_t> host_buffers;
  std::vector<void *> host_inputs;

  size_t graph_bytes = 0;
  size_t host_weight_bytes = 0;
  // Bytes/views borrowed from StreamingEngine's already-populated graph-RAM
  // arenas.  These are not owned by this graph and must not be counted as a
  // second host-side expert copy.
  size_t host_ram_alias_bytes = 0;
  size_t host_ram_alias_tensors = 0;
  size_t host_pinned_bytes = 0;
  size_t host_pin_failures = 0;
  size_t host_reservation_bytes = 0;
  // ggml_init(no_alloc=true) still allocates a host-side context arena. Keep
  // that arena in the process-wide admission ledger as well, otherwise graph
  // construction can exceed host_ram_gb between explicit host-buffer
  // allocations.
  size_t context_reservation_bytes = 0;
  // The source admission is made once before graph construction, while this
  // graph materializes that source as many individual host allocations.  This
  // remainder prevents alloc_host() from charging those same bytes twice.
  size_t host_precharged_bytes = 0;
  bool ready = false;

  ~FullGraph() { destroy(); }

  void destroy() {
    if (sched) {
      // qwen4exp_full_graph_run synchronizes before returning.  Avoid calling
      // into the borrowed CUDA backend during process teardown: the shared
      // backend can already be in CUDA runtime shutdown at this point.
      ggml_backend_sched_free(sched);
      sched = nullptr;
    }
    // Host-node CPU runners borrow this graph-owned pool.  Detach them after
    // the scheduler is gone, but before the pool itself is destroyed.  The
    // runner's graph plan is synchronous, so no callback can still be using
    // it once graph teardown begins.
    for (auto &host_node : host_node_contexts) {
      if (host_node && host_node->cpu_runner &&
          host_node->shared_cpu_threadpool) {
        host_node->cpu_runner->unbind_external_threadpool(
            host_node->shared_cpu_threadpool);
        host_node->cpu_runner = nullptr;
      }
    }
    if (stage_wait_done_event || stage_wait_start_event ||
        stage_d2d_start_event ||
        stage_d2d_end_event || stage_compute_start_event ||
        stage_compute_end_event) {
      cudaSetDevice(device);
      if (stage_wait_done_event)
        cudaEventDestroy(stage_wait_done_event);
      if (stage_wait_start_event)
        cudaEventDestroy(stage_wait_start_event);
      if (stage_d2d_start_event)
        cudaEventDestroy(stage_d2d_start_event);
      if (stage_d2d_end_event)
        cudaEventDestroy(stage_d2d_end_event);
      if (stage_compute_start_event)
        cudaEventDestroy(stage_compute_start_event);
      if (stage_compute_end_event)
        cudaEventDestroy(stage_compute_end_event);
    }
    stage_wait_done_event = nullptr;
    stage_wait_start_event = nullptr;
    stage_d2d_start_event = nullptr;
    stage_d2d_end_event = nullptr;
    stage_compute_start_event = nullptr;
    stage_compute_end_event = nullptr;
    stage_slot = nullptr;
    stage_slots.clear();
    stage_bindings.clear();
    stage_experts.clear();
    stage_route_ids = nullptr;
    stage_enabled = false;
    if (cpu && cpu_threadpool)
      ggml_backend_cpu_set_threadpool(cpu, nullptr);
    if (cpu_threadpool) {
      ggml_threadpool_free(cpu_threadpool);
      cpu_threadpool = nullptr;
    }
    if (cpu) {
      ggml_backend_free(cpu);
      cpu = nullptr;
    }
    if (device_expert_buffer && !phase2_expert_source_graph) {
      ggml_backend_buffer_free(device_expert_buffer);
    }
    device_expert_buffer = nullptr;
    device_expert_base = nullptr;
    device_expert_capacity = 0;
    device_expert_used = 0;
    if (device_dense_buffer && !phase2_dense_source_graph) {
      ggml_backend_buffer_free(device_dense_buffer);
    }
    device_dense_buffer = nullptr;
    device_dense_base = nullptr;
    device_dense_capacity = 0;
    device_dense_used = 0;
    device_dense_tensors = 0;
    if (device_state_buffer) {
      ggml_backend_buffer_free(device_state_buffer);
      device_state_buffer = nullptr;
    }
    device_state_base = nullptr;
    device_state_capacity = 0;
    device_state_used = 0;
    device_expert_layers = 0;
    device_expert_partial_gate_up = false;
    split_gpu_expert_ids.clear();
    phase2_expert_source_graph = nullptr;
    phase2_expert_addrs.clear();
    partial_layer_ids_out = nullptr;
    partial_layer_probs_out = nullptr;
    // Only the Prefix instance owns phase2_handoff_buf (a standalone
    // buffer, independent of ctx/sched -- see the field's own comment).
    // Suffix instances only ever bind a leaf to it, never allocate or
    // free it themselves.
    if (phase2_mode == Phase2Mode::Prefix && phase2_handoff_buf) {
      ggml_backend_buffer_free(phase2_handoff_buf);
    }
    phase2_handoff_buf = nullptr;
    phase2_handoff_addr = nullptr;
    phase2_mode = Phase2Mode::Off;
    phase2_split_layer = -1;
    phase2_state_source_graph = nullptr;
    phase2_state_addrs.clear();
    phase2_state_replay_cursor = 0;
    phase2_dense_source_graph = nullptr;
    phase2_dense_addrs.clear();
    for (auto &host_node : host_node_contexts) {
      if (!host_node)
        continue;
      if (host_node->d2h_start)
        cudaEventDestroy(host_node->d2h_start);
      if (host_node->d2h_end)
        cudaEventDestroy(host_node->d2h_end);
      if (host_node->h2d_start)
        cudaEventDestroy(host_node->h2d_start);
      if (host_node->h2d_end)
        cudaEventDestroy(host_node->h2d_end);
      if (host_node->post_h2d_check_end)
        cudaEventDestroy(host_node->post_h2d_check_end);
      host_node->d2h_start = nullptr;
      host_node->d2h_end = nullptr;
      host_node->h2d_start = nullptr;
      host_node->h2d_end = nullptr;
      host_node->post_h2d_check_end = nullptr;
    }
    host_node_contexts.clear();
    for (ggml_backend_buffer_t b : host_buffers) {
      if (b)
        ggml_backend_buffer_free(b);
    }
    host_buffers.clear();
    if (ctx) {
      ggml_free(ctx);
      ctx = nullptr;
    }
    for (HostAllocation &a : owned) {
      if (a.pinned)
        cudaHostUnregister(a.ptr);
      if (a.ptr)
        std::free(a.ptr);
      MemoryGuard::release_host_bytes(a.reservation_bytes);
      a.ptr = nullptr;
      a.pinned = false;
    }
    owned.clear();
    host_inputs.clear();
    cuda = nullptr;
    cpu = nullptr;
    cpu_threads = 0;
    graph = nullptr;
    input = output = logits = final_streams = positions = attn_mask =
        decode_indices = ple_embedding = nullptr;
    two_layer_diag = false;
    two_layer_l0_post_ffn = nullptr;
    two_layer_l1_pre_attn = nullptr;
    two_layer_l1_post_attn = nullptr;
    two_layer_l1_post_ffn = nullptr;
    ple_hist_out = nullptr;
    ple_hist_storage = nullptr;
    ple_hist_update = nullptr;
    kv_k.clear();
    kv_v.clear();
    kv_layers.clear();
    gdn_state_out.clear();
    gdn_hist_out.clear();
    kv_state_k.clear();
    kv_state_v.clear();
    gdn_state_storage.clear();
    gdn_hist_storage.clear();
    gdn_state_updates.clear();
    gdn_hist_updates.clear();
    gdn_layers.clear();
    gdn_state_inputs.clear();
    gdn_hist_inputs.clear();
    ple_layer = -1;
    ple_hist_input = nullptr;
    ready = false;
    graph_bytes = 0;
    host_weight_bytes = 0;
    host_ram_alias_bytes = 0;
    host_ram_alias_tensors = 0;
    host_pinned_bytes = 0;
    host_pin_failures = 0;
    MemoryGuard::release_host_bytes(host_reservation_bytes);
    host_reservation_bytes = 0;
    MemoryGuard::release_host_bytes(context_reservation_bytes);
    context_reservation_bytes = 0;
    host_precharged_bytes = 0;
    engine = nullptr;
    S = 0;
    decode_mode = false;
    kv_capacity = 0;
    decode_last_pos = -1;
  }
};

std::unique_ptr<FullGraph> g_graph;

// Standalone two-layer diagnostic owner. It is intentionally separate from
// g_graph so enabling the diagnostic cannot alter the production full-graph
// cache or its phase-2 sibling ownership.
std::unique_ptr<FullGraph> g_two_layer_diag_graph;
bool g_two_layer_diag_done = false;

// Phase 2 per-token dynamic dispatch: three separately-built/submitted
// FullGraph instances, own ctx/graph/sched each (see FullGraph::Phase2Mode's
// comment). Only ever populated when qwen4exp_full_graph_decode_prepare
// commits to Phase 2 for this session (real per-token routing at
// phase2_split_layer decided the static Phase 1 coverage gate was not met,
// but the layer is still worth a dynamic per-token GPU/CPU choice). Globals,
// not FullGraph fields, because a suffix instance's build_graph() must reach
// the SIBLING prefix instance's handoff buffer/address while its own `g` is
// a completely different object.
// The Prefix instance is just g_graph itself with phase2_mode==Prefix (its
// construction is otherwise identical to the ordinary single-graph decode
// path -- see qwen4exp_full_graph_decode_prepare's build_one lambda) --
// no separate global needed for it.
std::unique_ptr<FullGraph> g_graph_p2_suffix_gpu;
std::unique_ptr<FullGraph> g_graph_p2_suffix_cpu;

bool g_config_full_graph = false;
bool g_config_full_graph_decode = false;

bool env_enabled() {
  const char *v = std::getenv("MOESTREAM_QWEN4EXP_FULL_GRAPH");
  if (v && v[0])
    return v[0] == '1';
  return g_config_full_graph;
}

int env_device_expert_layers() {
  const char *v = std::getenv("MOESTREAM_QWEN4EXP_GRAPH_PRELOAD_LAYERS");
  if (!v || !v[0])
    return 0;
  char *end = nullptr;
  long n = std::strtol(v, &end, 10);
  if (end == v || *end != '\0' || n <= 0)
    return 0;
  return static_cast<int>(std::min<long>(n, 48));
}

// Matches beellama's `-ncmoe N` convention (GPU-resident block at the END
// of the model) instead of MoEStream's historical leading-N convention.
// See FullGraph::device_expert_trailing's own comment.
bool env_device_expert_trailing() {
  const char *v = std::getenv("MOESTREAM_QWEN4EXP_GRAPH_PRELOAD_TRAILING");
  return v && v[0] == '1';
}

// See FullGraph::layer_timing_enabled's own comment.
bool env_layer_timing_enabled() {
  const char *v = std::getenv("MOESTREAM_QWEN4EXP_GRAPH_LAYER_TIMING");
  return v && v[0] == '1';
}

// Decode phase ledger for the Phase-2/master-graph A/B.  This is deliberately
// separate from the per-CPU-node callback timing above: it brackets the
// scheduler submission, route decision/readback, suffix submission, output
// readback, and (for the master path) the actual device graph span.
bool env_phase_timing_enabled() {
  const char *v = std::getenv("MOESTREAM_QWEN4EXP_GRAPH_PHASE_TIMING");
  return v && v[0] == '1';
}

// Phase 4 host-node integration is deliberately a separate opt-in from the
// ordinary full graph.  It is only safe when the graph builder replaces the
// configured CPU MoE nodes with the fused CPY markers; an unset variable must
// retain the proven scheduler-split path byte-for-byte.
bool env_qwen4exp_host_node_enabled() {
  const char *v = std::getenv("MOESTREAM_QWEN4EXP_HOST_NODE");
  return v && v[0] == '1';
}

// Runtime guard (2026-09-19): CMakeLists.txt already documents that a plain
// MOESTREAM_USE_GGML_CUDA build can silently compile out ggml-cuda's CUDA
// graph capture/replay path (GGML_CUDA_USE_GRAPHS never defined) -- this
// exact failure mode cost an entire investigation before it was found once
// already (full-graph decode 20 tok/s -> 8 tok/s, mistaken for a
// correctness/scheduling bug). A build-time default is not enough: it can
// still silently drift (wrong CMake cache value, a stale build directory,
// an override left on from debugging). This checks the ACTUAL compiled
// binary's feature list via ggml's own backend-registry API, every single
// run, and logs impossible-to-miss output either way -- no one should have
// to remember to ask "is this still on" again.
static void verify_cuda_graphs_compiled_in(ggml_backend_t cuda_backend) {
  static bool checked = false;
  if (checked || !cuda_backend)
    return;
  checked = true;
  ggml_backend_dev_t dev = ggml_backend_get_device(cuda_backend);
  ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
  auto get_features = reg ? reinterpret_cast<ggml_backend_get_features_t>(
                                ggml_backend_reg_get_proc_address(
                                    reg, "ggml_backend_get_features"))
                          : nullptr;
  bool has_graphs = false;
  if (get_features) {
    for (const ggml_backend_feature *f = get_features(reg);
        f && f->name; ++f) {
      if (std::strcmp(f->name, "USE_GRAPHS") == 0) {
        has_graphs = true;
        break;
      }
    }
  }
  if (has_graphs) {
    LOG_INFO("[CUDA-GRAPHS-CHECK] compiled in: yes (GGML_CUDA_USE_GRAPHS "
             "active in this binary)\n");
  } else {
    LOG_WARN("[CUDA-GRAPHS-CHECK] compiled in: NO -- this build silently "
             "lacks CUDA graph capture/replay (GGML_CUDA_USE_GRAPHS was not "
             "defined when ggml-cuda was compiled). Full-graph decode will "
             "run every node as an individual cudaLaunchKernel instead of "
             "one batched cudaGraphLaunch per token -- previously measured "
             "cost: ~20 tok/s -> ~8 tok/s on this same decode path. Rebuild "
             "with -DMOESTREAM_ENABLE_CUDA_GRAPHS=ON (the CMakeLists.txt "
             "default) and verify with a fresh 'cmake -B <dir>' configure, "
             "not just an incremental build in a stale build directory.\n");
  }
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
  const char *v = std::getenv("MOESTREAM_QWEN4EXP_GRAPH_LAYER_TIMING_VERBOSE");
  return v && v[0] == '1';
}

// Opt-in ledger for the CUDA graph replay boundary and ordinary GGML CPU
// backend work.  Keep this disabled by default because it adds steady_clock
// reads to the replay path.
bool env_graph_replay_timing_enabled() {
  const char *v = std::getenv("MOESTREAM_QWEN4EXP_GRAPH_REPLAY_TIMING");
  return v && v[0] == '1';
}

// The merged gate/up representation is intentionally runtime-switchable while
// its CUDA and CPU kernel cost is being compared with the established
// three-projection graph.  It is opt-in until the doubled-row MUL_MAT_ID path
// matches the separate projection path; setting this to 1 enables it without
// changing placement or routing.
bool env_merged_gate_up_enabled() {
  const char *v = std::getenv("MOESTREAM_QWEN4EXP_GRAPH_MERGED_GATE_UP");
  return v && v[0] == '1';
}

bool env_layer0_iq2_diag_enabled() {
  const char *v = std::getenv("MOESTREAM_QWEN4EXP_DIAG_LAYER0_IQ2");
  return v && v[0] == '1';
}

static const char * env_hidden01_dump_path() {
  const char *v = std::getenv("MOESTREAM_QWEN4EXP_HIDDEN01_DUMP");
  return v && v[0] ? v : nullptr;
}

static const char * env_layer0_split_dump_path() {
  const char *v = std::getenv("MOESTREAM_QWEN4EXP_LAYER0_SPLIT_DUMP");
  return v && v[0] ? v : nullptr;
}

// Host-node transport diagnostic. The layer is selected explicitly so the
// same pre-H2D/post-H2D proof used for the historical Layer-0 IQ2 check can be
// applied to Layer 13 without enabling diagnostics for every callback.
int env_host_node_diag_layer() {
  const char *v = std::getenv("MOESTREAM_QWEN4EXP_HOST_NODE_DIAG_LAYER");
  if (!v || !v[0])
    return -1;
  char *end = nullptr;
  const long layer = std::strtol(v, &end, 10);
  if (end == v || *end != '\0' || layer < 0 || layer > 47)
    return -1;
  return static_cast<int>(layer);
}

bool env_host_node_transport_diag_enabled(int layer) {
  return (layer == 0 && env_layer0_iq2_diag_enabled()) ||
         layer == env_host_node_diag_layer();
}

// Narrow laboratory mode for the Layer-0 host-node residual/consumer
// diagnostic. The ordinary full graph deliberately requires a complete DDR
// arena for every layer when ram_preload_all_experts is enabled. That is the
// right production safety rule, but it makes a one-layer diagnostic depend on
// having enough RAM for the whole model. In this mode only Layer 0's callback
// boundary is required to use a complete arena; other layers keep the normal
// graph-owned source path.
bool env_layer0_narrow_graph_diag_enabled() {
  const char *v = std::getenv("MOESTREAM_QWEN4EXP_DIAG_LAYER0_RESIDUAL");
  return v && v[0] == '1';
}

bool env_two_layer_s1_diag_enabled() {
  const char *v = std::getenv("MOESTREAM_QWEN4EXP_TWO_LAYER_S1_DIAG");
  return v && v[0] == '1';
}

const char *env_two_layer_s1_diag_dump_path() {
  const char *v = std::getenv("MOESTREAM_QWEN4EXP_TWO_LAYER_S1_DUMP");
  return v && v[0] ? v : nullptr;
}

// Controlled QKV replay input for the two-layer S=1 diagnostic.  The file
// must contain exactly one F32 Layer-0 attention input row ([D] elements),
// typically extracted from Bee's hc_mixed dump.  This replaces only the
// diagnostic graph's Layer-0 hc_mix result; production execution is never
// affected.  It lets us distinguish a small upstream HC rounding difference
// from a QKV/MMQ implementation difference using the same bytes at the GEMM
// boundary.
const char *env_two_layer_s1_common_attn_input_path() {
  const char *v = std::getenv(
      "MOESTREAM_QWEN4EXP_TWO_LAYER_S1_COMMON_ATTN_INPUT");
  return v && v[0] ? v : nullptr;
}

// Optional earlier seam for the same controlled replay.  This file contains
// the post-RMSNorm*HC-norm activation ([D*HC] for S=1), allowing the HC
// down/up/gate projection path to be tested independently of its input
// normalization and upstream hidden-state construction.
const char *env_two_layer_s1_common_hc_norm_path() {
  const char *v = std::getenv(
      "MOESTREAM_QWEN4EXP_TWO_LAYER_S1_COMMON_HC_NORM");
  return v && v[0] ? v : nullptr;
}

const char *env_two_layer_s1_common_hc_down_silu_path() {
  const char *v = std::getenv(
      "MOESTREAM_QWEN4EXP_TWO_LAYER_S1_COMMON_HC_DOWN_SILU");
  return v && v[0] ? v : nullptr;
}

const char *env_layer0_common_attn_input_path() {
  const char *v =
      std::getenv("MOESTREAM_QWEN4EXP_LAYER0_COMMON_ATTN_INPUT");
  return v && v[0] ? v : nullptr;
}

bool load_exact_f32_file(const char *path, float *dst, size_t elements) {
  if (!path || !dst || elements == 0)
    return false;
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in)
    return false;
  const std::streamoff expected = static_cast<std::streamoff>(
      elements * sizeof(float));
  if (in.tellg() != expected)
    return false;
  in.seekg(0, std::ios::beg);
  in.read(reinterpret_cast<char *>(dst), expected);
  return in.good() || in.eof() && in.gcount() == expected;
}

void log_layer0_diag_tensor(const char *role, const ggml_tensor *t) {
  if (!t) {
    LOG_INFO("[QWEN4EXP-LAYER0-DIAG] %s=<null>\n", role);
    return;
  }
  const char *buft = "unbound";
  const char *dev = "unbound";
  if (t->buffer) {
    const auto type = ggml_backend_buffer_get_type(t->buffer);
    buft = ggml_backend_buft_name(type);
    if (const auto device = ggml_backend_buft_get_device(type))
      dev = ggml_backend_dev_name(device);
  }
  LOG_INFO(
      "[QWEN4EXP-LAYER0-DIAG] %s type=%s backend=%s device=%s "
      "ne=[%lld,%lld,%lld,%lld] nb=[%zu,%zu,%zu,%zu] data=%p\n",
      role, ggml_type_name(t->type), buft, dev,
      static_cast<long long>(t->ne[0]), static_cast<long long>(t->ne[1]),
      static_cast<long long>(t->ne[2]), static_cast<long long>(t->ne[3]),
      t->nb[0], t->nb[1], t->nb[2], t->nb[3], t->data);
}

void log_layer0_diag_host_values(const char *role, const float *values,
                                 size_t n) {
  if (!values || n == 0)
    return;
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
  LOG_INFO("[QWEN4EXP-LAYER0-DIAG] %s values=%zu first=[%s] min=%.9g "
           "max=%.9g sum=%.9g data=%p\n",
           role, n, first.str().c_str(), min_v, max_v, sum,
           static_cast<const void *>(values));
}

void dump_layer0_diag_values(FullGraph &g) {
  const char *split_dump_path = env_layer0_split_dump_path();
  if (split_dump_path && g.decode_mode && !g.layer0_split_dumped) {
    g.layer0_split_dumped = true;
    std::unordered_map<std::string, size_t> snapshot_occurrences;
    for (const auto &entry : g.layer0_split_snapshots) {
      const size_t occurrence = snapshot_occurrences[entry.first]++;
      // The side diagnostic graph runs immediately before production decode
      // and uses the same snapshot labels. Keep its files separate so the
      // production replay cannot overwrite controlled side-graph evidence.
      const char *graph_tag = g.two_layer_diag ? ".two_layer." : ".";
      std::string path = std::string(split_dump_path) + graph_tag +
                         entry.first;
      if (occurrence != 0)
        path += "." + std::to_string(occurrence);
      path += ".f32";
      ggml_tensor *t = entry.second;
      if (!t || !t->data)
        continue;
      const size_t n = static_cast<size_t>(ggml_nelements(t));
      std::vector<float> values(n);
      ggml_backend_tensor_get(t, values.data(), 0, n * sizeof(float));
      std::ofstream out(path, std::ios::binary | std::ios::trunc);
      if (!out) {
        LOG_WARN("[QWEN4EXP-LAYER0-SPLIT] cannot open %s\n", path.c_str());
        continue;
      }
      out.write(reinterpret_cast<const char *>(values.data()),
                static_cast<std::streamsize>(n * sizeof(float)));
      float min_v = INFINITY;
      float max_v = -INFINITY;
      double sum = 0.0;
      double sum2 = 0.0;
      for (float v : values) {
        min_v = std::min(min_v, v);
        max_v = std::max(max_v, v);
        sum += v;
        sum2 += static_cast<double>(v) * v;
      }
      LOG_INFO("[QWEN4EXP-LAYER0-SPLIT] boundary=%s elements=%zu "
               "fnv64=0x%016llx min=%.9g max=%.9g mean=%.9g rms=%.9g "
               "file=%s\n",
               entry.first.c_str(), n,
               static_cast<unsigned long long>(qwen4exp_fnv64(
                   values.data(), n * sizeof(float))),
               min_v, max_v, n ? sum / static_cast<double>(n) : 0.0,
               n ? std::sqrt(sum2 / static_cast<double>(n)) : 0.0,
               path.c_str());
    }
  }
  const bool hidden_dump = env_hidden01_dump_path() && g.decode_mode &&
                           !g.hidden01_dumped;
  if (!g.layer0_diag_enabled ||
      (g.layer0_diag_dumped && !hidden_dump))
    return;
  if (!g.layer0_diag_dumped) {
    g.layer0_diag_dumped = true;
    if (!g.layer0_diag_gate || !g.layer0_diag_up)
      LOG_INFO("[QWEN4EXP-LAYER0-DIAG] numeric gate/up outputs unavailable "
               "for this graph (host-node or no separate projection result)\n");
    auto dump = [](const char *role, const ggml_tensor *t) {
      if (!t)
        return;
      const size_t n = static_cast<size_t>(ggml_nelements(t));
      std::vector<float> values(n, 0.0f);
      ggml_backend_tensor_get(t, values.data(), 0, n * sizeof(float));
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
      LOG_INFO("[QWEN4EXP-LAYER0-DIAG] %s values=%zu first=[%s] min=%.9g "
               "max=%.9g sum=%.9g\n",
               role, n, first.str().c_str(), min_v, max_v, sum);
    };
    if (g.layer0_diag_gate)
      dump("gate_output", g.layer0_diag_gate);
    if (g.layer0_diag_up)
      dump("up_output", g.layer0_diag_up);
    dump("down_input_swiglu", g.layer0_diag_act);
    dump("down_output_raw", g.layer0_diag_down);
    // The following graph-owned tensors may alias later allocations by the
    // time the whole graph has retired. Their exact values are reported from
    // the host snapshots below instead.
    auto dump_snapshot = [](const char *role, const ggml_tensor *t) {
      if (!t || !t->data)
        return;
      log_layer0_diag_host_values(role, static_cast<const float *>(t->data),
                                  static_cast<size_t>(ggml_nelements(t)));
    };
    dump_snapshot("residual_before_moe_snapshot",
                  g.layer0_diag_residual_before_snapshot);
    dump_snapshot("moe_plus_shared_snapshot",
                  g.layer0_diag_moe_plus_shared_snapshot);
    dump_snapshot("residual_after_moe_add_snapshot",
                  g.layer0_diag_residual_after_snapshot);
    dump_snapshot("layer1_first_consumer_snapshot",
                  g.layer0_diag_next_consumer_snapshot);
  }

  const char *dump_path = env_hidden01_dump_path();
  if (!dump_path || !hidden_dump)
    return;
  g.hidden01_dumped = true;
  for (int layer = 0; layer < 2; ++layer) {
    ggml_tensor *t = g.hidden01_snapshot[layer];
    if (!t || !t->data)
      continue;
    const size_t n = static_cast<size_t>(ggml_nelements(t));
    const auto *values = static_cast<const float *>(t->data);
    std::ostringstream suffix;
    suffix << dump_path << ".decode.layer" << layer << ".f32";
    std::ofstream out(suffix.str(), std::ios::binary | std::ios::trunc);
    if (!out) {
      LOG_WARN("[QWEN4EXP-HIDDEN01] cannot open %s\n", suffix.str().c_str());
      continue;
    }
    out.write(reinterpret_cast<const char *>(values),
              static_cast<std::streamsize>(n * sizeof(float)));
    float min_v = INFINITY;
    float max_v = -INFINITY;
    double sum = 0.0;
    double sum2 = 0.0;
    for (size_t i = 0; i < n; ++i) {
      min_v = std::min(min_v, values[i]);
      max_v = std::max(max_v, values[i]);
      sum += values[i];
      sum2 += static_cast<double>(values[i]) * values[i];
    }
    LOG_INFO("[QWEN4EXP-HIDDEN01] layer=%d elements=%zu fnv64=0x%016llx "
             "min=%.9g max=%.9g mean=%.9g rms=%.9g file=%s\n",
             layer, n,
             static_cast<unsigned long long>(qwen4exp_fnv64(
                 values, n * sizeof(float))),
             min_v, max_v, n ? sum / static_cast<double>(n) : 0.0,
             n ? std::sqrt(sum2 / static_cast<double>(n)) : 0.0,
             suffix.str().c_str());
  }
}

// Phase 1 "all-or-nothing" partial-layer GPU dispatch (see
// engine.hpp's qwen4exp_partial_layer field group for the full contract).
int env_partial_layer() {
  const char *v = std::getenv("MOESTREAM_QWEN4EXP_PARTIAL_LAYER");
  if (!v || !v[0])
    return -1;
  char *end = nullptr;
  const long n = std::strtol(v, &end, 10);
  if (end == v || *end != '\0' || n < 0)
    return -1;
  return static_cast<int>(n);
}

int env_partial_layer_top_n() {
  const char *v = std::getenv("MOESTREAM_QWEN4EXP_PARTIAL_LAYER_TOP_N");
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
  const char *v = std::getenv("MOESTREAM_QWEN4EXP_PARTIAL_LAYER_MIN_COVERAGE");
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
  const char *v = std::getenv("MOESTREAM_QWEN4EXP_PARTIAL_LAYER_PHASE2");
  return v && v[0] == '1';
}

int layer_timing_parse_layer(const char *name);

// Parses the "blk.<N>." prefix ggml_qwen4exp_graph.cpp's own tensor names
// always use (see expert_weight/dense_weight/etc.) -- returns -1 if the
// name doesn't match (dense/global tensors with no per-layer index, e.g.
// the embedding or output head). Mirrors load_full_weight's own parsing.
int layer_timing_parse_layer(const char *name) {
  if (!name || std::strncmp(name, "blk.", 4) != 0)
    return -1;
  char *end = nullptr;
  const long n = std::strtol(name + 4, &end, 10);
  if (end == name + 4 || !end || *end != '.')
    return -1;
  return static_cast<int>(n);
}

// ggml_backend_sched_eval_callback: fires with ask=true immediately before
// a node runs, ask=false immediately after. For CPU-backend nodes this
// synchronously brackets real compute time (ggml-cpu has no async
// concept); for CUDA-backend nodes it would only bracket the async launch,
// so those are skipped entirely -- GPU-side timing already exists from the
// nsys work elsewhere in this investigation. See FullGraph::layer_timing_*
// fields and the memory note this feature's plan came from.
bool layer_timing_eval_callback(struct ggml_tensor *t, bool ask,
                                void *user_data) {
  FullGraph *g = static_cast<FullGraph *>(user_data);
  if (!g || !t)
    return true;
  if (g->layer_timing_verbose_enabled && !g->layer_timing_verbose_dumped) {
    // One-shot: log EVERY node (both backends), first decode token only,
    // so the real node/backend sequence bordering a CPU-dispatched MoE
    // layer is visible directly instead of inferred from the undiferen-
    // tiated gap bucket. ask=true is logged with the gap since the last
    // ask=false of ANY node (not just CPU ones, unlike the timing above)
    // so a GPU-backend D2H/H2D copy sitting in that gap shows up as real
    // elapsed time next to its own name/op/backend.
    ggml_backend_t node_be = ggml_backend_sched_get_tensor_backend(g->sched, t);
    const char *be_name = node_be == g->cpu ? "CPU"
                          : node_be == g->cuda ? "CUDA"
                                               : "?";
    const auto vnow = std::chrono::steady_clock::now();
    double gap_ms = 0.0;
    if (g->layer_timing_verbose_have_last)
      gap_ms = std::chrono::duration<double, std::milli>(
                   vnow - g->layer_timing_verbose_last)
                   .count();
    auto vtag_it = g->layer_timing_layer_of.find(t);
    char tag_buf[32] = "-";
    if (vtag_it != g->layer_timing_layer_of.end())
      std::snprintf(tag_buf, sizeof(tag_buf), "L%d/%s",
                    vtag_it->second.first, vtag_it->second.second);
    LOG_INFO("[GGML-QWEN-FULL-LAYER-TIMING-VERBOSE] ask=%d name=%s op=%s "
             "backend=%s tag=%s since_last=%.4fms\n",
             ask ? 1 : 0, ggml_get_name(t), ggml_op_name(t->op), be_name,
             tag_buf, gap_ms);
    g->layer_timing_verbose_last = vnow;
    g->layer_timing_verbose_have_last = true;
    if (!ask && g->n_layers > 0) {
      // End-of-graph marker: the LAST node's ask=false has fired once we
      // see the final layer's down-projection result. Token 1 is reliably
      // JIT/warm-up-contaminated (confirmed: ~4ms MoE ops vs ~0.3-0.4ms
      // steady-state) -- capture several tokens, not just one, so a real
      // steady-state sample is available alongside the warm-up outlier.
      if (t == g->output || t == g->logits) {
        ++g->layer_timing_verbose_tokens_seen;
        if (g->layer_timing_verbose_tokens_seen >= 10)
          g->layer_timing_verbose_dumped = true;
      }
    }
  }
  if (ggml_backend_sched_get_tensor_backend(g->sched, t) != g->cpu)
    return true; // GPU node or unassigned -- not timeable this way, skip
  const auto now = std::chrono::steady_clock::now();
  if (ask) {
    // Gap since the LAST CPU node's ask=false -- dispatch/barrier/
    // scheduling overhead the op-bracket timing below cannot see, since
    // it only covers what happens INSIDE one node's own bracket.
    if (g->layer_timing_have_last_ask_false) {
      g->layer_timing_gap_ms_total +=
          std::chrono::duration<double, std::milli>(
              now - g->layer_timing_last_ask_false)
              .count();
      ++g->layer_timing_gap_count;
    }
    g->layer_timing_t0[t] = now;
    return true;
  }
  g->layer_timing_last_ask_false = now;
  g->layer_timing_have_last_ask_false = true;
  auto it = g->layer_timing_t0.find(t);
  if (it == g->layer_timing_t0.end())
    return true; // no matching ask=true (shouldn't happen, be defensive)
  const double ms =
      std::chrono::duration<double, std::milli>(now - it->second).count();
  g->layer_timing_t0.erase(it);
  // Op result tensors (mul_mat_id/swiglu outputs) get ggml's auto "node_
  // NNNN" name, not a "blk.N.stem" one -- only build_moe()'s own explicit
  // layer_timing_layer_of recording (populated at graph-build time, see
  // its own comment) can attribute these correctly. Fall back to name
  // parsing for anything build_moe() didn't tag (e.g. weight leaves, if
  // this callback ever fires on one) so the op-level map still groups
  // something meaningful rather than silently dropping it.
  auto tag_it = g->layer_timing_layer_of.find(t);
  const bool tagged = tag_it != g->layer_timing_layer_of.end();
  const int layer = tagged ? tag_it->second.first
                           : layer_timing_parse_layer(ggml_get_name(t));
  if (layer >= 0 && layer < static_cast<int>(g->layer_timing_cpu_ms.size()))
    g->layer_timing_cpu_ms[static_cast<size_t>(layer)] += ms;
  std::string stem;
  if (tagged) {
    stem = tag_it->second.second;
  } else {
    const char *name = ggml_get_name(t);
    stem = name;
    if (layer >= 0) {
      // Strip the "blk.<N>." prefix so the op-level map groups by stem
      // across all layers, e.g. "ffn_down_exps.weight" regardless of
      // which layer -- same convention expert_weight() itself uses.
      const char *dot = std::strchr(name + 4, '.');
      if (dot && dot[1] != '\0')
        stem = dot + 1;
    }
  }
  g->layer_timing_op_ms[stem] += ms;
  return true;
}

bool env_decode_enabled() {
  const char *v = std::getenv("MOESTREAM_QWEN4EXP_FULL_GRAPH_DECODE");
  if (v && v[0])
    return v[0] == '1';
  return g_config_full_graph_decode;
}

int env_decode_kv_capacity(int max_t, int required) {
  const char *v = std::getenv("MOESTREAM_QWEN4EXP_FULL_GRAPH_DECODE_KV_CAPACITY");
  int requested = 2048;
  if (v && v[0]) {
    char *end = nullptr;
    const long parsed = std::strtol(v, &end, 10);
    if (end != v && *end == '\0' && parsed > 0)
      requested = static_cast<int>(std::min<long>(parsed, max_t));
  }
  return std::min(max_t, std::max(required, requested));
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
  const char *v = std::getenv("MOESTREAM_QWEN_CPU_THREADS");
  if (v && v[0]) {
    char *end = nullptr;
    const long parsed = std::strtol(v, &end, 10);
    if (end != v && *end == '\0' && parsed > 0)
      return std::min(available, static_cast<int>(parsed));
  }
  return available;
}

// REVERTED (2026-09-17): a prior commit defaulted this to false, reasoning
// "the reference Bee scheduler uses pipeline_parallel=false for this
// graph... so the event/four-copy mode adds bookkeeping without useful
// overlap." That reasoning does not hold up against Bee's own source
// (llama-context.cpp): `pipeline_parallel` there is an UNRELATED feature --
// it requires `model.n_devices() > 1` (multi-GPU tensor split) AND
// `!model.has_tensor_overrides()`, and Bee's own `-ncmoe` flag sets tensor
// overrides for exactly the CPU-offloaded MoE tensors this graph mirrors.
// So Bee's `pipeline_parallel=false` in every comparison this project has
// ever run is not a deliberate choice about CPU/GPU MoE handoff at all --
// it is bee's own multi-GPU pipelining feature being auto-disabled by
// `-ncmoe`, for reasons having nothing to do with this graph's performance.
// Confirmed via direct measurement this graph's OWN parallel=false default
// was a real regression (11.76 vs 12.20-12.86 tok/s, N=22 mixed decode) --
// this is the fix from qwen4exp-mixed-gpu-decode-gap.md
// (ggml_backend_sched_new's parallel=false leaves CUDA sync events NULL,
// forcing a blocking stream-sync every CPU/GPU transition instead of a
// cheap event wait), previously verified at ~15.7->18.5-18.9 tok/s and
// accidentally undone by that commit. Default back to true; set
// MOESTREAM_QWEN4EXP_GRAPH_PARALLEL=0 for the blocking-sync A/B.
bool graph_scheduler_parallel() {
  const char *v = std::getenv("MOESTREAM_QWEN4EXP_GRAPH_PARALLEL");
  return !(v && v[0] == '0');
}

// Bee gives the CPU scheduler slot the CUDA device's host buffer type. That
// makes graph-owned CPU activations pinned host allocations, so the many
// CPU<->CUDA boundary copies do not fall through the pageable-host staging
// path. This is the normal full-graph mode; set the environment variable to
// 0 for an explicit pageable-buffer A/B.
bool graph_cuda_host_buffers() {
  const char *v = std::getenv("MOESTREAM_QWEN4EXP_GRAPH_CUDA_HOST_BUFFERS");
  return !(v && v[0] == '0');
}

// Bee's full graph begins with the CPU token-embedding GET_ROWS node, then
// crosses to CUDA for the first transformer block.  MoEStream receives the
// already-expanded hidden state from the engine, so it cannot include that
// embedding operation without duplicating model work.  Keep the same
// CPU->GPU graph boundary with one CPU contiguous staging node instead.  This
// is a topology-compatibility node, not a numerical transformation; set the
// variable to 0 for the original MoEStream graph shape when doing an A/B.
bool graph_match_bee_topology() {
  const char *v =
      std::getenv("MOESTREAM_QWEN4EXP_GRAPH_MATCH_BEE_TOPOLOGY");
  return !(v && v[0] == '0');
}

// Emit one-time tensor metadata for CPU MoE comparisons.  The same
// BEE_GGML_PROFILE_LAYOUT switch is consumed by Bee's GGML scheduler so the
// two implementations can be compared without changing their execution.
bool graph_layout_debug() {
  const char *v = std::getenv("BEE_GGML_PROFILE_LAYOUT");
  return v && v[0] == '1';
}

bool has_ple_layer(const ArchitectureConfig &c) {
  return !c.ple_layers.empty() && c.ple_n_heads > 0 &&
         c.ple_n_embd_per_layer > 0;
}

bool type_ok(GGUFType type) {
  return static_cast<uint32_t>(type) < static_cast<uint32_t>(GGML_TYPE_COUNT);
}

ggml_tensor *new_tensor(ggml_context *ctx, ggml_type type,
                        const std::vector<int64_t> &shape) {
  switch (shape.size()) {
  case 1:
    return ggml_new_tensor_1d(ctx, type, shape[0]);
  case 2:
    return ggml_new_tensor_2d(ctx, type, shape[0], shape[1]);
  case 3:
    return ggml_new_tensor_3d(ctx, type, shape[0], shape[1], shape[2]);
  case 4:
    return ggml_new_tensor_4d(ctx, type, shape[0], shape[1], shape[2],
                              shape[3]);
  default:
    return nullptr;
  }
}

ggml_backend_buffer_t bind_host(FullGraph &g, ggml_tensor *t, void *ptr,
                                size_t bytes, ggml_backend_buffer_usage usage) {
  if (!t || !ptr || bytes < ggml_nbytes(t))
    return nullptr;
  ggml_backend_buffer_t b = ggml_backend_cpu_buffer_from_ptr(ptr, bytes);
  if (!b)
    return nullptr;
  ggml_backend_buffer_set_usage(b, usage);
  if (ggml_backend_tensor_alloc(b, t, ptr) != GGML_STATUS_SUCCESS) {
    ggml_backend_buffer_free(b);
    return nullptr;
  }
  g.host_buffers.push_back(b);
  return b;
}

// A preallocated host buffer and a scheduler backend assignment are separate
// pieces of state in ggml_backend_sched.  The former keeps gallocr from
// allocating storage for the tensor; the latter keeps the operation that
// consumes the tensor on the CPU backend.  This matters for full-graph
// Qwen4Exp builds: without the explicit assignment, a host-backed expert
// weight can be reconsidered during scheduler splitting and its CPU-layer
// MoE subtree can be represented as a CUDA allocation request.  At 22 GPU /
// 26 CPU layers that request is approximately the full 26-layer expert
// footprint (~88 GiB), instead of a small activation workspace.
void bind_host_expert_scheduler_backend(FullGraph &g, ggml_tensor *t) {
  static int moestream_bind_diag = -1;
  if (moestream_bind_diag < 0) {
    const char *e = std::getenv("MOESTREAM_BIND_HOST_DIAG");
    moestream_bind_diag = (e && e[0] == '1') ? 1 : 0;
  }
  if (moestream_bind_diag) {
    LOG_WARN(
        "[MOESTREAM-BIND-DIAG] name=%s sched=%p cpu=%p t=%p t->buffer=%p "
        "is_host=%d\n",
        t ? ggml_get_name(t) : "(null)", (void *)g.sched, (void *)g.cpu,
        (void *)t, t ? (void *)t->buffer : nullptr,
        (t && t->buffer) ? (int)ggml_backend_buffer_is_host(t->buffer) : -1);
  }
  if (!g.sched || !g.cpu || !t || !t->buffer ||
      !ggml_backend_buffer_is_host(t->buffer)) {
    if (moestream_bind_diag)
      LOG_WARN("[MOESTREAM-BIND-DIAG] SKIPPED pin for name=%s\n",
               t ? ggml_get_name(t) : "(null)");
    return;
  }
  ggml_backend_sched_set_tensor_backend(g.sched, t, g.cpu);
}

void *alloc_host(FullGraph &g, size_t bytes) {
  void *p = nullptr;
  if (bytes == 0)
    return nullptr;
  size_t reservation = 0;
  if (g.host_precharged_bytes >= bytes) {
    g.host_precharged_bytes -= bytes;
  } else {
    reservation = bytes - g.host_precharged_bytes;
    g.host_precharged_bytes = 0;
    if (!MemoryGuard::try_reserve_host_bytes(reservation)) {
      LOG_WARN("[GGML-QWEN-FULL-MEM] host graph allocation refused: %.2f MB "
               "(reserved=%.2f GB, budget=%.2f GB)\n",
               bytes / 1e6, MemoryGuard::reserved_host_bytes() / 1e9,
               MemoryGuard::host_budget_bytes() / 1e9);
      return nullptr;
    }
  }
  const int err = posix_memalign(&p, 4096, bytes);
  if (err != 0 || !p) {
    MemoryGuard::release_host_bytes(reservation);
    return nullptr;
  }
  g.owned.push_back({p, bytes, false, reservation});
  return p;
}

ggml_tensor *host_input(FullGraph &g, ggml_context *ctx, ggml_type type,
                        const std::vector<int64_t> &shape, size_t bytes,
                        const char *name) {
  ggml_tensor *t = new_tensor(ctx, type, shape);
  if (!t)
    return nullptr;
  void *p = alloc_host(g, std::max(bytes, ggml_nbytes(t)));
  if (!p || !bind_host(g, t, p, std::max(bytes, ggml_nbytes(t)),
                       GGML_BACKEND_BUFFER_USAGE_COMPUTE))
    return nullptr;
  ggml_set_input(t);
  ggml_set_name(t, name);
  g.host_inputs.push_back(p);
  return t;
}

// Diagnostic-only snapshot for tensors whose GGML allocator storage may be
// reused later in the graph.  Reading such a tensor after the full graph has
// retired can observe a later consumer's data, so materialize the value at
// the exact graph position into a persistent host buffer.
ggml_tensor *layer0_diag_snapshot(FullGraph &g, ggml_context *ctx,
                                   ggml_tensor *src, const char *name) {
  if (!g.layer0_diag_enabled || !src || !g.graph)
    return nullptr;
  const size_t bytes = ggml_nbytes(src);
  const std::vector<int64_t> shape = {src->ne[0], src->ne[1], src->ne[2],
                                      src->ne[3]};
  ggml_tensor *dst = host_input(g, ctx, GGML_TYPE_F32, shape, bytes, name);
  if (!dst)
    return nullptr;
  ggml_tensor *copy = ggml_cpy(ctx, src, dst);
  if (!copy)
    return nullptr;
  ggml_set_name(copy, name);
  ggml_build_forward_expand(g.graph, copy);
  return dst;
}

ggml_tensor *layer13_moe_diag_snapshot(
    FullGraph &g, ggml_context *ctx, ggml_tensor *src, ggml_type type,
    const std::vector<int64_t> &shape, const char *name) {
  if (!g.layer13_moe_diag_enabled || !src || !g.graph)
    return nullptr;
  size_t elements = 1;
  for (const int64_t dim : shape) {
    if (dim <= 0 || elements > SIZE_MAX / static_cast<size_t>(dim))
      return nullptr;
    elements *= static_cast<size_t>(dim);
  }
  ggml_tensor *dst = host_input(
      g, ctx, type, shape, elements * ggml_type_size(type), name);
  if (!dst)
    return nullptr;
  ggml_tensor *copy = ggml_cpy(ctx, src, dst);
  if (!copy)
    return nullptr;
  ggml_set_name(copy, name);
  ggml_build_forward_expand(g.graph, copy);
  return dst;
}

const QuantizedTensor *resident(const StreamingEngine &engine,
                                const std::string &name) {
  auto it = engine.resident_layers.find(name);
  if (it == engine.resident_layers.end() || !it->second.host_ptr)
    return nullptr;
  return &it->second;
}

bool graph_dense_weight_name(const std::string &name);

ggml_tensor *resident_tensor(
    FullGraph &g, ggml_context *ctx, const std::string &name,
    ggml_backend_buffer_usage usage = GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
  const QuantizedTensor *q = resident(*g.engine, name);
  if (!q || !type_ok(q->meta.type))
    return nullptr;
  std::vector<int64_t> shape;
  shape.reserve(q->meta.shape.size());
  for (uint64_t d : q->meta.shape)
    shape.push_back(static_cast<int64_t>(d));
  ggml_tensor *t = new_tensor(ctx, static_cast<ggml_type>(q->meta.type), shape);
  if (!t || q->meta.size_bytes < ggml_nbytes(t))
    return nullptr;
  if (g.device_dense_buffer && usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
      graph_dense_weight_name(name)) {
    // Prefix/suffix Phase 2 graphs are mutually exclusive at runtime and
    // only read these weights.  Reuse the prefix-owned buffer rather than
    // allocating and uploading a full dense arena for every graph instance.
    // A suffix may request a name the prefix never materialized; in that
    // case append it to the owner's pre-sized arena and record the address
    // for the remaining siblings.
    FullGraph &owner = g.phase2_dense_source_graph
                           ? *g.phase2_dense_source_graph
                           : g;
    auto existing = owner.phase2_dense_addrs.find(name);
    void *addr = nullptr;
    if (existing != owner.phase2_dense_addrs.end()) {
      addr = existing->second;
      if (ggml_backend_tensor_alloc(owner.device_dense_buffer, t, addr) !=
          GGML_STATUS_SUCCESS)
        return nullptr;
    } else {
      const ggml_backend_buffer_type_t buft =
          ggml_backend_buffer_get_type(owner.device_dense_buffer);
      const size_t alloc_size = ggml_backend_buft_get_alloc_size(buft, t);
      const size_t alignment =
          std::max<size_t>(256, ggml_backend_buft_get_alignment(buft));
      const size_t aligned =
          (owner.device_dense_used + alignment - 1) & ~(alignment - 1);
      if (aligned > owner.device_dense_capacity ||
          alloc_size > owner.device_dense_capacity - aligned)
        return nullptr;
      addr = static_cast<uint8_t *>(owner.device_dense_base) + aligned;
      if (ggml_backend_tensor_alloc(owner.device_dense_buffer, t, addr) !=
          GGML_STATUS_SUCCESS)
        return nullptr;
      const auto t_memcpy_start = std::chrono::steady_clock::now();
      const cudaError_t err = cudaMemcpy(
          addr, q->host_ptr, q->meta.size_bytes, cudaMemcpyHostToDevice);
      owner.dense_upload_ms_total += std::chrono::duration<double, std::milli>(
                                        std::chrono::steady_clock::now() -
                                        t_memcpy_start)
                                        .count();
      if (err != cudaSuccess) {
        LOG_ERROR("[GGML-QWEN-FULL] dense device upload failed for %s: %s\n",
                  name.c_str(), cudaGetErrorString(err));
        return nullptr;
      }
      owner.device_dense_used = aligned + alloc_size;
      owner.phase2_dense_addrs.emplace(name, addr);
    }
    ++g.device_dense_tensors;
  } else if (!bind_host(g, t, q->host_ptr, q->meta.size_bytes, usage)) {
    return nullptr;
  }
  ggml_set_name(t, name.c_str());
  return t;
}

bool graph_dense_weight_name(const std::string &name) {
  // Keep this list aligned with the names consumed by build_graph(). Routed
  // *_exps tensors have their own residency policy; indexer/NextN tensors are
  // not part of this v1 graph, and absorbed-MLA tensors are unused here.
  if (name.find("_exps.weight") != std::string::npos ||
      name.find("indexer") != std::string::npos ||
      name.find("nextn") != std::string::npos ||
      name.find("attn_k_b.weight") != std::string::npos ||
      name.find("attn_v_b.weight") != std::string::npos ||
      name.find("attn_output_a.weight") != std::string::npos)
    return false;
  return name.find("attn") != std::string::npos ||
         name.find("hc_") != std::string::npos ||
         name.find("ssm_") != std::string::npos ||
         name.find("ple_") != std::string::npos ||
         name.find("norm") != std::string::npos ||
         name.find("gate_inp") != std::string::npos ||
         name.find("shexp") != std::string::npos ||
         name.find("ffn_gate.weight") != std::string::npos ||
         name.find("ffn_up.weight") != std::string::npos ||
         name.find("ffn_down.weight") != std::string::npos ||
         name.find("exp_probs_b.bias") != std::string::npos ||
         name == "output_hc_norm.weight" || name == "output_hc_down.weight" ||
         name == "output_hc_up.weight" || name == "output.weight" ||
         name == "output_norm.weight";
}

bool init_device_dense_arena(FullGraph &g) {
  if (g.phase2_dense_source_graph) {
    FullGraph &source = *g.phase2_dense_source_graph;
    if (!source.device_dense_buffer || !source.device_dense_base ||
        source.device_dense_capacity == 0)
      return false;
    g.device_dense_buffer = source.device_dense_buffer;
    g.device_dense_base = source.device_dense_base;
    g.device_dense_capacity = source.device_dense_capacity;
    g.device_dense_used = 0;
    g.device_dense_tensors = 0;
    LOG_INFO("[GGML-QWEN-FULL] dense device arena borrowed from Phase 2 "
             "source: %.2f GiB\n",
             static_cast<double>(g.device_dense_capacity) /
                 (1024.0 * 1024.0 * 1024.0));
    return true;
  }
  size_t bytes = 0;
  int tensors = 0;
  for (const auto &[name, q] : g.engine->resident_layers) {
    if (!q.host_ptr || !graph_dense_weight_name(name))
      continue;
    if (g.two_layer_diag) {
      // Reserve only the dense weights consumed by blocks 0 and 1 plus the
      // final HC projection. A two-layer diagnostic must not claim the
      // production model's complete dense arena.
      int layer = -1;
      if (name.rfind("blk.", 0) == 0) {
        char *end = nullptr;
        const long parsed = std::strtol(name.c_str() + 4, &end, 10);
        if (end != name.c_str() + 4 && end && *end == '.')
          layer = static_cast<int>(parsed);
      }
      const bool final_hc = name == "output_hc_norm.weight" ||
                            name == "output_hc_down.weight" ||
                            name == "output_hc_up.weight";
      if (!final_hc && (layer < 0 || layer >= g.n_layers))
        continue;
    }
    constexpr size_t kTensorSlack = 4096;
    if (bytes > SIZE_MAX - q.meta.size_bytes - kTensorSlack)
      return false;
    bytes += q.meta.size_bytes + kTensorSlack;
    ++tensors;
  }
  if (bytes == 0)
    return false;
  constexpr size_t kArenaSlack = 1ull << 20;
  if (bytes > SIZE_MAX - kArenaSlack)
    return false;
  const size_t capacity = bytes + kArenaSlack;
  const ggml_backend_buffer_type_t buft =
      ggml_backend_get_default_buffer_type(g.cuda);
  g.device_dense_buffer = ggml_backend_buft_alloc_buffer(buft, capacity);
  if (!g.device_dense_buffer) {
    LOG_WARN(
        "[GGML-QWEN-FULL] dense device arena allocation declined: %.2f GiB\n",
        static_cast<double>(capacity) / (1024.0 * 1024.0 * 1024.0));
    return false;
  }
  ggml_backend_buffer_set_usage(g.device_dense_buffer,
                                GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
  g.device_dense_base = ggml_backend_buffer_get_base(g.device_dense_buffer);
  g.device_dense_capacity = ggml_backend_buffer_get_size(g.device_dense_buffer);
  g.device_dense_used = 0;
  g.device_dense_tensors = 0;
  LOG_INFO("[GGML-QWEN-FULL] dense device arena=%.2f GiB for %d graph weight "
           "tensors\n",
           static_cast<double>(g.device_dense_capacity) /
               (1024.0 * 1024.0 * 1024.0),
           tensors);
  return true;
}

bool init_device_state_arena(FullGraph &g) {
  if (!g.decode_mode || g.kv_capacity <= 0)
    return true;
  const auto &c = g.engine->model_file.architecture_config;
  size_t bytes = 0;
  auto add = [&](size_t n) {
    constexpr size_t kSlack = 4096;
    if (n > SIZE_MAX - kSlack || bytes > SIZE_MAX - n - kSlack)
      return false;
    bytes += n + kSlack;
    return true;
  };
  for (int l = 0; l < g.n_layers; ++l) {
    if (!qwen4exp_is_gdn_layer(c, l)) {
      const size_t kv = static_cast<size_t>(g.kv_capacity) * g.Hkv * g.hd *
                        sizeof(float);
      if (!add(kv) || !add(kv))
        return false;
    } else {
      const size_t state = static_cast<size_t>(g.gdn_hd) * g.gdn_hd *
                           g.Hv_gdn * sizeof(float);
      const size_t hist = static_cast<size_t>(g.d_conv - 1) * g.conv_dim *
                          sizeof(float);
      if (!add(state) || !add(hist))
        return false;
    }
    if (qwen4exp_is_ple_layer(c, l)) {
      const size_t ple_hist = static_cast<size_t>(c.ple_conv_kernel - 1) *
                              c.ple_ngram_size * g.D * g.HC * sizeof(float);
      if (!add(ple_hist))
        return false;
    }
  }
  constexpr size_t kArenaSlack = 1ull << 20;
  if (bytes > SIZE_MAX - kArenaSlack)
    return false;
  const size_t capacity = bytes + kArenaSlack;
  const ggml_backend_buffer_type_t buft =
      ggml_backend_get_default_buffer_type(g.cuda);
  g.device_state_buffer = ggml_backend_buft_alloc_buffer(buft, capacity);
  if (!g.device_state_buffer) {
    LOG_WARN("[GGML-QWEN-FULL-DECODE] device state arena declined: %.2f GiB\n",
             static_cast<double>(capacity) / (1024.0 * 1024.0 * 1024.0));
    return false;
  }
  ggml_backend_buffer_set_usage(g.device_state_buffer,
                                GGML_BACKEND_BUFFER_USAGE_COMPUTE);
  g.device_state_base = ggml_backend_buffer_get_base(g.device_state_buffer);
  g.device_state_capacity = ggml_backend_buffer_get_size(g.device_state_buffer);
  g.device_state_used = 0;
  LOG_INFO("[GGML-QWEN-FULL-DECODE] device state arena=%.2f GiB, KV capacity=%d\n",
           static_cast<double>(g.device_state_capacity) /
               (1024.0 * 1024.0 * 1024.0),
           g.kv_capacity);
  return true;
}

ggml_tensor *new_device_state_tensor(FullGraph &g, ggml_type type,
                                     const std::vector<int64_t> &shape,
                                     const char *name) {
  ggml_tensor *t = new_tensor(g.ctx, type, shape);
  if (!t)
    return nullptr;
  // Phase 2 SuffixCpu (or any future sibling): bind to the SAME physical
  // state a sibling instance already allocated for this exact call
  // position, instead of allocating a fresh, independent buffer -- see
  // FullGraph::phase2_state_source_graph's own comment.
  if (g.phase2_state_source_graph) {
    FullGraph &src = *g.phase2_state_source_graph;
    if (!src.device_state_buffer ||
        g.phase2_state_replay_cursor >= src.phase2_state_addrs.size())
      return nullptr;
    void *addr = src.phase2_state_addrs[g.phase2_state_replay_cursor++];
    if (ggml_backend_tensor_alloc(src.device_state_buffer, t, addr) !=
        GGML_STATUS_SUCCESS)
      return nullptr;
    ggml_set_name(t, name);
    return t;
  }
  if (!g.device_state_buffer)
    return nullptr;
  const ggml_backend_buffer_type_t buft =
      ggml_backend_buffer_get_type(g.device_state_buffer);
  const size_t alloc_size = ggml_backend_buft_get_alloc_size(buft, t);
  const size_t alignment =
      std::max<size_t>(256, ggml_backend_buft_get_alignment(buft));
  const size_t aligned =
      (g.device_state_used + alignment - 1) & ~(alignment - 1);
  if (aligned > g.device_state_capacity ||
      alloc_size > g.device_state_capacity - aligned)
    return nullptr;
  void *addr = static_cast<uint8_t *>(g.device_state_base) + aligned;
  if (ggml_backend_tensor_alloc(g.device_state_buffer, t, addr) !=
      GGML_STATUS_SUCCESS)
    return nullptr;
  g.device_state_used = aligned + alloc_size;
  // Record for a potential Phase 2 sibling built right after this one
  // (harmless no-op cost otherwise -- one pointer per call, freed with
  // the rest of this instance).
  g.phase2_state_addrs.push_back(addr);
  ggml_set_name(t, name);
  return t;
}

// True for a layer inside the device_expert_layers-sized GPU-resident
// block, wherever that block sits (leading, the historical default, or
// trailing, matching beellama's `-ncmoe` convention -- see
// FullGraph::device_expert_trailing's own comment). Single source of
// truth for the block's bounds so device_expert_bytes() and
// plan_expert_vram_split()'s eligibility filter can't drift from
// graph_expert_projection_resident()'s own notion of which layers are
// covered.
bool device_expert_block_covers(const FullGraph &g, int layer) {
  if (g.device_expert_layers <= 0)
    return false;
  const int n_layers = g.engine->model_file.architecture_config.n_layers;
  if (g.device_expert_trailing)
    return layer >= n_layers - g.device_expert_layers;
  return layer < g.device_expert_layers;
}

bool graph_expert_projection_resident(const FullGraph &g, int layer,
                                      const char *stem) {
  // cpu_affinity_layers (config): force this layer's routed-expert weights
  // to stay CPU-resident regardless of device_expert_layers, mirroring the
  // legacy dispatch path's own is_cpu_affinity_layer() contract ("never
  // pin or pre-warm VRAM for these layers"). Checked first so it overrides
  // the leading-N device_expert_layers count below even if this layer would
  // otherwise have qualified for VRAM residency.
  if (is_cpu_affinity_layer(layer))
    return false;
  if (layer >= 0 && device_expert_block_covers(g, layer))
    return true;
  // device_expert_partial_gate_up's boundary-layer optimization is only
  // meaningful for the leading convention (the boundary is the first CPU
  // layer just after the leading GPU block, layer==device_expert_layers);
  // it is forced off in trailing mode at the config-set call site instead
  // of generalizing this boundary math, so this line never fires there.
  return g.device_expert_partial_gate_up && layer == g.device_expert_layers &&
         std::strcmp(stem, "ffn_down_exps.weight") != 0;
}

// Per-expert VRAM split planner (qwen4exp_graph_expert_vram_budget_gb).
// Generalizes cpu_affinity_layers/device_expert_layers' whole-layer choice:
// instead of an all-or-nothing residency decision per layer, spend a fixed
// VRAM budget on the globally hottest routed experts (by real usage, same
// ranking data pin_topk_per_layer/global_topn already consult), across any
// layer NOT already claimed by cpu_affinity_layers or device_expert_layers/
// device_expert_partial_gate_up -- those two mechanisms are checked FIRST
// and always win, so this planner never double-places a layer they already
// resolved. Must run before init_device_expert_arena() so the arena can
// size itself for the resulting split_gpu_expert_ids. v1 scope: decode only
// (g.S==1) -- see build_moe()'s own S==1 guard for why; a config with the
// budget set during prefill (S>1) simply plans nothing here, no error.
void plan_expert_vram_split(FullGraph &g) {
  g.split_gpu_expert_ids.clear();
  if (g.S != 1)
    return;

  // Option B is a deterministic usage-driven plan: a static hot set per
  // selected layer, with complementary experts left in the fixed-stride DDR
  // arena.  In concentrated-only mode the selected layers were chosen from
  // the usage histogram by the startup preload pass.  The graph repeats the
  // eligibility checks here because this is the final owner of the CUDA
  // arena layout and must never split a layer already covered by the complete
  // baseline GPU block.
  const int option_layer = g.engine->qwen4exp_option_b_layer;
  const int option_top_n = g.engine->qwen4exp_option_b_top_n;
  const bool option_b_auto =
      g.engine->qwen4exp_option_b_concentrated_only && option_layer < 0;
  if (option_top_n > 0 &&
      (option_layer >= 0 || option_b_auto ||
       !g.engine->qwen4exp_option_b_layers.empty())) {
    const auto &c = g.engine->model_file.architecture_config;

    const std::string path = g.engine->qwen4exp_graph_pin_file.empty()
                                 ? std::string(".moestream_usage")
                                 : g.engine->qwen4exp_graph_pin_file;
    auto [entries, maxc] = StreamingEngine::load_usage(path);
    (void)maxc;
    if (entries.empty()) {
      LOG_WARN("[OPTION-B] no usage data at %s; Layer %d remains CPU/DDR\n",
               path.c_str(), option_layer);
      return;
    }

    std::vector<int> option_layers = g.engine->qwen4exp_option_b_layers;
    if (option_layer >= 0)
      option_layers = {option_layer};
    else if (option_layers.empty() && option_b_auto) {
      option_layers.reserve(static_cast<size_t>(c.n_layers));
      for (int layer = 0; layer < c.n_layers; ++layer)
        option_layers.push_back(layer);
    }

    int committed_layers = 0;
    for (int layer : option_layers) {
      if (layer < 0 || layer >= c.n_layers || c.n_experts <= 0) {
        LOG_WARN("[OPTION-B] invalid target layer=%d (layers=%d); skipping\n",
                 layer, c.n_layers);
        continue;
      }
      if (is_cpu_affinity_layer(layer) || device_expert_block_covers(g, layer))
        continue;

      // Usage files normally contain one row per expert, but aggregate rather
      // than trusting that format. Zero-count experts are included so a
      // top-N request really means N out of the model's complete expert
      // domain, not merely N of the experts seen in an earlier run.
      std::vector<uint64_t> counts(static_cast<size_t>(c.n_experts), 0);
      for (const auto &e : entries) {
        if (e.layer == layer && e.expert >= 0 && e.expert < c.n_experts)
          counts[static_cast<size_t>(e.expert)] += e.count;
      }
      uint64_t total_count = 0;
      for (uint64_t count : counts)
        total_count += count;
      if (total_count == 0)
        continue;

      std::vector<int> ranked(static_cast<size_t>(c.n_experts));
      for (int expert = 0; expert < c.n_experts; ++expert)
        ranked[static_cast<size_t>(expert)] = expert;
      std::stable_sort(ranked.begin(), ranked.end(), [&](int a, int b) {
        if (counts[static_cast<size_t>(a)] != counts[static_cast<size_t>(b)])
          return counts[static_cast<size_t>(a)] > counts[static_cast<size_t>(b)];
        return a < b;
      });

      const int hot_count = std::min(option_top_n, c.n_experts);
      std::vector<int> hot_ids(ranked.begin(), ranked.begin() + hot_count);
      uint64_t hot_route_count = 0;
      for (int expert : hot_ids)
        hot_route_count += counts[static_cast<size_t>(expert)];
      if (g.engine->qwen4exp_option_b_fill_vram) {
        // Start with the requested coverage floor, then let the arena
        // planner trim the least-used tail if the remaining VRAM cannot hold
        // every expert from every selected layer.
        hot_ids.insert(hot_ids.end(), ranked.begin() + hot_count,
                       ranked.end());
      }
      const double coverage = static_cast<double>(hot_route_count) /
                              static_cast<double>(total_count);
      if (coverage < g.engine->qwen4exp_option_b_min_coverage) {
        LOG_INFO("[OPTION-B] Layer %d remains CPU/DDR: top-%d coverage=%.2f%% "
                 "is below %.2f%%\n",
                 layer, hot_count, coverage * 100.0,
                 g.engine->qwen4exp_option_b_min_coverage * 100.0);
        continue;
      }

      std::vector<int> staged_experts;
      // The staging proof is intentionally still Layer-13-only. Automatic
      // concentrated-layer selection must not silently create active-layer
      // slots or forced routes for every selected layer.
      if (layer == option_layer) {
        int stage_seed = g.engine->qwen4exp_option_b_stage_expert;
        const int stage_count = std::min(
            std::max(1, g.engine->qwen4exp_option_b_stage_count), c.topk);
        if (stage_seed >= 0 && stage_seed < c.n_experts) {
          for (int candidate = stage_seed;
               candidate < c.n_experts &&
               static_cast<int>(staged_experts.size()) < stage_count;
               ++candidate) {
            if (std::find(hot_ids.begin(), hot_ids.end(), candidate) !=
                hot_ids.end())
              continue;
            staged_experts.push_back(candidate);
          }
          for (int candidate = 0;
               candidate < stage_seed &&
               static_cast<int>(staged_experts.size()) < stage_count;
               ++candidate) {
            if (std::find(hot_ids.begin(), hot_ids.end(), candidate) !=
                hot_ids.end())
              continue;
            staged_experts.push_back(candidate);
          }
        }
      }
      if (!staged_experts.empty())
        hot_ids.insert(hot_ids.end(), staged_experts.begin(),
                       staged_experts.end());
      if (!staged_experts.empty()) {
        g.stage_experts = staged_experts;
        g.stage_layer = layer;
        g.stage_expert = staged_experts.front();
      }

      const size_t placed_hot = hot_ids.size();
      const size_t per_expert = g.engine->compute_expert_slab_bytes(layer);
      size_t bytes = 0;
      if (per_expert != 0 && hot_ids.size() <= SIZE_MAX / per_expert)
        bytes = per_expert * hot_ids.size();
      g.split_gpu_expert_ids[layer] = std::move(hot_ids);
      ++committed_layers;
      LOG_INFO("[OPTION-B] Layer %d static hot/cold split: hot=%zu cold=%d "
               "staged=%d coverage=%.2f%% threshold=%.2f%% VRAM=%.3f GiB "
               "source=%s\n",
               layer, placed_hot,
               c.n_experts - static_cast<int>(placed_hot) -
                   static_cast<int>(staged_experts.size()),
               static_cast<int>(staged_experts.size()), coverage * 100.0,
               g.engine->qwen4exp_option_b_min_coverage * 100.0,
               static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0),
               path.c_str());
    }
    if (option_b_auto)
      LOG_INFO("[OPTION-B] concentrated-only graph plan committed %d CPU-side "
               "layer(s); baseline GPU block and flat layers were skipped\n",
               committed_layers);
    return;
  }

  const double budget_gb = g.engine->qwen4exp_graph_expert_vram_budget_gb;
  if (budget_gb <= 0.0)
    return; // feature off -- identical to today, zero planning cost

  const std::string path = g.engine->qwen4exp_graph_pin_file.empty()
                               ? std::string(".moestream_usage")
                               : g.engine->qwen4exp_graph_pin_file;
  auto [entries, maxc] = StreamingEngine::load_usage(path);
  (void)maxc;
  if (entries.empty()) {
    LOG_INFO("[GGML-QWEN-FULL-SPLIT] no usage data at %s -- per-expert VRAM "
             "split stays off this run.\n",
             path.c_str());
    return;
  }

  const auto &c = g.engine->model_file.architecture_config;
  const int n_layers = c.n_layers;
  const int n_experts = c.n_experts;

  // Bounds-check + drop any (layer,expert) already resolved by
  // cpu_affinity_layers/device_expert_layers/device_expert_partial_gate_up --
  // same stale-usage-file tolerance as main.cpp's sanitize_usage_entries.
  std::vector<StreamingEngine::UsageEntry> eligible;
  eligible.reserve(entries.size());
  for (const auto &e : entries) {
    if (e.layer < 0 || e.layer >= n_layers || e.expert < 0 ||
        e.expert >= n_experts)
      continue;
    if (is_cpu_affinity_layer(e.layer))
      continue;
    if (device_expert_block_covers(g, e.layer))
      continue;
    if (g.device_expert_partial_gate_up && e.layer == g.device_expert_layers)
      continue; // boundary layer's own asymmetric gate/up-only residency
    eligible.push_back(e);
  }
  if (eligible.empty()) {
    LOG_INFO("[GGML-QWEN-FULL-SPLIT] no eligible layers (all claimed by "
             "cpu_affinity_layers/device_expert_layers) -- per-expert VRAM "
             "split stays off this run.\n");
    return;
  }

  // Single global priority queue over all (layer, expert, count) triples --
  // spends the fixed budget on the model's genuinely hottest experts first,
  // rather than splitting it evenly per layer. Stable tie-break by
  // (layer, expert) for determinism, same reasoning as main.cpp's [PIN]
  // block's own stable_sort.
  std::stable_sort(eligible.begin(), eligible.end(),
                   [](const StreamingEngine::UsageEntry &a,
                      const StreamingEngine::UsageEntry &b) {
                     if (a.count != b.count)
                       return a.count > b.count;
                     if (a.layer != b.layer)
                       return a.layer < b.layer;
                     return a.expert < b.expert;
                   });

  size_t remaining_budget =
      static_cast<size_t>(budget_gb * 1024.0 * 1024.0 * 1024.0);
  std::unordered_map<int, std::unordered_set<int>> chosen;
  size_t experts_placed = 0;
  size_t bytes_spent = 0;
  for (const auto &e : eligible) {
    if (remaining_budget == 0)
      break;
    auto &layer_set = chosen[e.layer];
    if (layer_set.count(e.expert))
      continue; // usage file is expected unique per (layer,expert); don't trust that blindly
    const size_t cost = g.engine->compute_expert_slab_bytes(e.layer);
    if (cost == 0 || cost > remaining_budget)
      continue; // doesn't fit -- skip, keep scanning cheaper/later entries
    layer_set.insert(e.expert);
    remaining_budget -= cost;
    bytes_spent += cost;
    ++experts_placed;
  }

  for (auto &[layer, ids_set] : chosen) {
    if (ids_set.empty())
      continue;
    std::vector<int> &dst = g.split_gpu_expert_ids[layer];
    dst.assign(ids_set.begin(), ids_set.end());
    std::sort(dst.begin(), dst.end()); // deterministic local-index order
  }

  LOG_INFO("[GGML-QWEN-FULL-SPLIT] budget=%.3f GiB spent=%.3f GiB across %zu "
           "layer(s), %zu expert(s) total\n",
           budget_gb, bytes_spent / (1024.0 * 1024.0 * 1024.0),
           g.split_gpu_expert_ids.size(), experts_placed);
}

// Phase 1 partial-layer planner: ranks ONE specific layer's experts by real
// usage (same pin_file the startup [PIN] preload block and
// plan_expert_vram_split already read) and keeps the top N. Idempotent --
// only does real work once per process (engine.qwen4exp_partial_layer_planned
// guards it), since a layer's usage ranking doesn't change mid-process and
// this must survive the prefill->decode FullGraph object swap (see
// engine.hpp's own comment on why this state is engine-owned).
void plan_partial_layer(StreamingEngine &engine) {
  engine.qwen4exp_partial_layer = env_partial_layer();
  if (engine.qwen4exp_partial_layer < 0)
    return;
  engine.qwen4exp_partial_layer_top_n = env_partial_layer_top_n();
  engine.qwen4exp_partial_layer_min_coverage = env_partial_layer_min_coverage();
  if (engine.qwen4exp_partial_layer_planned)
    return;

  const std::string path = engine.qwen4exp_graph_pin_file.empty()
                               ? std::string(".moestream_usage")
                               : engine.qwen4exp_graph_pin_file;
  auto [entries, maxc] = StreamingEngine::load_usage(path);
  (void)maxc;

  std::vector<StreamingEngine::UsageEntry> layer_entries;
  for (const auto &e : entries)
    if (e.layer == engine.qwen4exp_partial_layer)
      layer_entries.push_back(e);
  std::stable_sort(layer_entries.begin(), layer_entries.end(),
                   [](const auto &a, const auto &b) {
                     return a.count > b.count;
                   });

  engine.qwen4exp_partial_layer_gpu_ids.clear();
  for (size_t i = 0; i < layer_entries.size() &&
                     static_cast<int>(i) < engine.qwen4exp_partial_layer_top_n;
       ++i)
    engine.qwen4exp_partial_layer_gpu_ids.push_back(layer_entries[i].expert);
  engine.qwen4exp_partial_layer_planned = true;

  LOG_INFO("[PARTIAL-LAYER] layer=%d top_n=%d ranked from %s: %zu known "
           "expert(s) for this layer, using top %zu for the GPU-resident "
           "compact set\n",
           engine.qwen4exp_partial_layer, engine.qwen4exp_partial_layer_top_n,
           path.c_str(), layer_entries.size(),
           engine.qwen4exp_partial_layer_gpu_ids.size());
}

const GGUFTensor *model_tensor(const StreamingEngine &engine,
                               const std::string &name) {
  auto it = engine.model_file.tensors.find(name);
  return it == engine.model_file.tensors.end() ? nullptr : &it->second;
}

size_t full_expert_source_bytes(const StreamingEngine &engine,
                                int layer_limit = -1) {
  const auto &c = engine.model_file.architecture_config;
  size_t total = 0;
  const int limit = layer_limit > 0 ? std::min(layer_limit, c.n_layers)
                                    : c.n_layers;
  for (int layer = 0; layer < limit; ++layer) {
    // When main.cpp prepared a complete graph-RAM arena, the three packed
    // expert planes for this layer are already resident in DDR.  The full
    // graph aliases those bytes below; do not pre-charge or allocate a second
    // graph-owned source set for the same layer.
    if (engine.graph_ram_layer_ready(layer))
      continue;
    for (const char *stem : {"ffn_gate_exps.weight", "ffn_up_exps.weight",
                             "ffn_down_exps.weight"}) {
      const GGUFTensor *m =
          model_tensor(engine, "blk." + std::to_string(layer) + "." + stem);
      if (!m || m->size_bytes == 0)
        return 0;
      if (static_cast<size_t>(m->size_bytes) >
          std::numeric_limits<size_t>::max() - total)
        return 0;
      total += static_cast<size_t>(m->size_bytes);
    }
  }
  return total;
}

// Resolve one packed Qwen expert projection to the complete, preloaded
// graph-RAM arena. This is deliberately Qwen-specific. The arena may use the
// legacy [gate|up|down] per-expert layout or the projection-compact layout
// (three separate expert-axis planes); in both cases this returns a strided
// view without making a graph-owned copy.
bool preloaded_ram_expert_source(FullGraph &g, const std::string &name,
                                 const GGUFTensor &meta, void *&ptr,
                                 size_t &bytes, size_t &expert_stride,
                                 size_t &arena_bytes) {
  const std::string prefix = "blk.";
  if (name.compare(0, prefix.size(), prefix) != 0)
    return false;

  char *end = nullptr;
  const long parsed_layer = std::strtol(name.c_str() + prefix.size(), &end, 10);
  if (end == name.c_str() + prefix.size() || parsed_layer < 0 || *end != '.')
    return false;
  const int layer = static_cast<int>(parsed_layer);
  const std::string suffix(end + 1);
  const auto *arena = g.engine->graph_ram_arena(layer);
  if (!arena)
    return false;
  size_t plane_offset = 0;
  size_t plane_bytes = 0;
  size_t plane_stride = 0;
  if (suffix == "ffn_gate_exps.weight") {
    plane_offset = arena->gate_offset;
    plane_bytes = arena->gate_bytes;
    plane_stride = arena->gate_stride;
  } else if (suffix == "ffn_up_exps.weight") {
    plane_offset = arena->projection_compact ? arena->up_offset
                                             : arena->gate_bytes;
    plane_bytes = arena->up_bytes;
    plane_stride = arena->up_stride;
  } else if (suffix == "ffn_down_exps.weight") {
    plane_offset = arena->projection_compact
                       ? arena->down_offset
                       : arena->gate_bytes + arena->up_bytes;
    plane_bytes = arena->down_bytes;
    plane_stride = arena->down_stride;
  } else {
    return false;
  }

  if (!g.engine->graph_ram_layer_ready(layer))
    return false;
  const int n_experts = g.engine->model_file.architecture_config.n_experts;
  const size_t n_exp = static_cast<size_t>(std::max(0, n_experts));
  if (!arena || !arena->base || n_experts <= 0 ||
      arena->n_experts != n_experts || plane_bytes == 0 ||
      plane_stride == 0 ||
      plane_offset > arena->bytes ||
      plane_bytes > arena->bytes - plane_offset ||
      n_exp > SIZE_MAX / plane_bytes ||
      meta.size_bytes != plane_bytes * n_exp || plane_stride < plane_bytes ||
      static_cast<size_t>(n_experts - 1) >
          (SIZE_MAX - plane_bytes) / plane_stride)
    return false;

  const size_t view_bytes =
      plane_stride * static_cast<size_t>(n_experts - 1) + plane_bytes;
  if (plane_offset > arena->bytes || view_bytes > arena->bytes - plane_offset)
    return false;

  ptr = arena->base + plane_offset;
  bytes = view_bytes;
  expert_stride = plane_stride;
  arena_bytes = arena->bytes;
  return true;
}

size_t device_expert_bytes(const FullGraph &g) {
  const auto &engine = *g.engine;
  const auto &c = engine.model_file.architecture_config;
  const int limit = std::min(std::max(0, g.device_expert_layers), c.n_layers);
  // Block bounds mirror device_expert_block_covers() exactly: leading mode
  // walks [0,limit), trailing mode walks [n_layers-limit,n_layers). Kept as
  // an explicit loop (not a device_expert_block_covers() call per layer)
  // since this function also needs the leading-only boundary-layer index
  // (`limit`) below, unchanged from before trailing mode existed.
  const int block_start = g.device_expert_trailing ? c.n_layers - limit : 0;
  const int block_end = g.device_expert_trailing ? c.n_layers : limit;
  size_t total = 0;
  for (int layer = block_start; layer < block_end; ++layer) {
    for (const char *stem : {"ffn_gate_exps.weight", "ffn_up_exps.weight",
                             "ffn_down_exps.weight"}) {
      const GGUFTensor *m =
          model_tensor(engine, "blk." + std::to_string(layer) + "." + stem);
      if (!m || m->size_bytes == 0 ||
          total > SIZE_MAX - static_cast<size_t>(m->size_bytes))
        return 0;
      total += static_cast<size_t>(m->size_bytes);
    }
  }
  // Boundary-layer (partial_gate_up) accounting is leading-only by
  // construction -- device_expert_partial_gate_up is forced off whenever
  // device_expert_trailing is set (see that setter), so limit < c.n_layers
  // here always refers to the leading block's own boundary.
  if (g.device_expert_partial_gate_up && limit < c.n_layers) {
    const int layer = limit;
    for (const char *stem : {"ffn_gate_exps.weight", "ffn_up_exps.weight"}) {
      const GGUFTensor *m =
          model_tensor(engine, "blk." + std::to_string(layer) + "." + stem);
      if (!m || m->size_bytes == 0 ||
          total > SIZE_MAX - static_cast<size_t>(m->size_bytes))
        return 0;
      total += static_cast<size_t>(m->size_bytes);
    }
  }
  // plan_expert_vram_split()'s per-layer split tensors: M hot experts'
  // combined gate+up+down bytes per split layer. These layers are disjoint
  // from the leading-N/boundary accounting above by construction (the
  // planner never places a layer device_expert_layers/
  // device_expert_partial_gate_up already claimed).
  for (const auto &[layer, ids] : g.split_gpu_expert_ids) {
    if (ids.empty())
      continue;
    const size_t per_expert = engine.compute_expert_slab_bytes(layer);
    if (per_expert == 0)
      continue;
    if (ids.size() > SIZE_MAX / per_expert)
      return 0;
    const size_t layer_bytes = per_expert * ids.size();
    if (total > SIZE_MAX - layer_bytes)
      return 0;
    total += layer_bytes;
  }
  // Phase 1/2 partial-layer compact GPU tensor (build_moe_partial_gpu):
  // qwen4exp_partial_layer_gpu_ids.size() experts' gate+up+down for
  // engine.qwen4exp_partial_layer, on every decode instance that could
  // possibly build it -- Phase 1's g_graph when it might commit, and
  // ALL THREE Phase 2 instances (only SuffixGpu actually builds the
  // tensor, but reserving on Prefix/SuffixCpu too is a small, bounded,
  // always-safe no-op there). Without this, g.device_expert_buffer is
  // never allocated at all when device_expert_layers<=0 and
  // split_gpu_expert_ids is empty (this feature's only two OTHER
  // consumers) -- build_split_gpu_tensor then silently fails via
  // bind_device_expert_gather's own `!g.device_expert_buffer` guard,
  // and the whole compact path falls back to ordinary dispatch for
  // that layer (found via real hardware testing, 2026-09-18: the
  // [PARTIAL-LAYER] "compact GPU tensor construction failed... RAM
  // arena not ready?" warning).
  if (g.decode_mode && !engine.qwen4exp_partial_layer_gpu_ids.empty() &&
      engine.qwen4exp_partial_layer >= 0) {
    const size_t per_expert =
        engine.compute_expert_slab_bytes(engine.qwen4exp_partial_layer);
    const size_t n = engine.qwen4exp_partial_layer_gpu_ids.size();
    if (per_expert != 0 && n <= SIZE_MAX / per_expert) {
      const size_t layer_bytes = per_expert * n;
      if (total <= SIZE_MAX - layer_bytes)
        total += layer_bytes;
    }
  }
  // Generalized GPU staging (build_gpu_stage_node): a fixed compact
  // gate+up+down destination per staged host-resident layer, sized for
  // `topk` slots (not n_experts) -- see that function's own comment for
  // why this is safe to size this small. A layer already covered by the
  // static device_expert_layers block doesn't need a separate staging
  // allocation (it's already fully device-resident); skip those to avoid
  // double-counting.
  if (g.decode_mode && engine.qwen4exp_gpu_stage_enabled) {
    for (int layer : engine.qwen4exp_gpu_stage_layers) {
      if (device_expert_block_covers(g, layer))
        continue;
      const size_t per_expert = engine.compute_expert_slab_bytes(layer);
      const size_t topk = static_cast<size_t>(std::max(0, c.topk));
      if (per_expert == 0 || topk == 0 || topk > SIZE_MAX / per_expert)
        continue;
      const size_t layer_bytes = per_expert * topk;
      if (total > SIZE_MAX - layer_bytes)
        return 0;
      total += layer_bytes;
    }
  }
  return total;
}

bool init_device_expert_arena(FullGraph &g) {
  if (g.phase2_expert_source_graph) {
    FullGraph &source = *g.phase2_expert_source_graph;
    if (!source.device_expert_buffer || !source.device_expert_base ||
        source.device_expert_capacity == 0)
      return false;
    // The owner may have reduced a whole-layer request to fit the actual
    // device budget.  Reuse its final placement/layout, not this borrower's
    // original request, so every graph binds the same physical records.
    g.device_expert_layers = source.device_expert_layers;
    g.device_expert_partial_gate_up = source.device_expert_partial_gate_up;
    g.split_gpu_expert_ids = source.split_gpu_expert_ids;
    g.device_expert_buffer = source.device_expert_buffer;
    g.device_expert_base = source.device_expert_base;
    g.device_expert_capacity = source.device_expert_capacity;
    g.device_expert_used = 0;
    LOG_INFO("[GGML-QWEN-FULL] device expert arena borrowed from Phase 2 "
             "source: %.2f GiB\n",
             static_cast<double>(g.device_expert_capacity) /
                 (1024.0 * 1024.0 * 1024.0));
    return true;
  }
  const bool split_active = !g.split_gpu_expert_ids.empty();
  // See device_expert_bytes's own comment: the Phase 1/2 partial-layer
  // compact tensor needs this same arena, on every decode instance that
  // could possibly build it, even when neither of the other two features
  // that normally drive this arena's sizing is active.
  const bool partial_active =
      g.decode_mode && g.engine &&
      !g.engine->qwen4exp_partial_layer_gpu_ids.empty() &&
      g.engine->qwen4exp_partial_layer >= 0 &&
      (!g.two_layer_diag || g.engine->qwen4exp_partial_layer < g.n_layers);
  if (g.device_expert_layers <= 0 && !split_active && !partial_active)
    return true;
  const ggml_backend_buffer_type_t buft =
      ggml_backend_get_default_buffer_type(g.cuda);
  const int requested_layers = g.device_expert_layers;
  // A request for all 48 routed layers is valid as a placement preference but
  // cannot fit on the 24-GiB card.  Clamp only whole-layer residency requests
  // to the largest arena that actually allocates; the graph then keeps the
  // remaining layers on the DDR/CPU backend instead of abandoning the entire
  // full graph. Per-expert split allocations (and the partial-layer compact
  // tensor, same rationale) remain strict until their own budget is made
  // independently retryable.
  const bool allow_layer_clamp =
      requested_layers > 0 && !split_active && !partial_active;

  auto sync_option_b_hot_masks = [&]() {
    if (!g.engine || !g.engine->qwen4exp_option_b_fill_vram)
      return;
    const int n_experts = g.engine->model_file.architecture_config.n_experts;
    for (const auto &[layer, ids] : g.split_gpu_expert_ids) {
      std::vector<uint8_t> mask(static_cast<size_t>(std::max(0, n_experts)),
                                0);
      for (int expert : ids) {
        if (expert >= 0 && expert < n_experts)
          mask[static_cast<size_t>(expert)] = 1;
      }
      g.engine->qwen4exp_option_b_hot_masks[layer] = std::move(mask);
      if (layer == g.engine->qwen4exp_option_b_layer)
        g.engine->qwen4exp_option_b_hot_mask =
            g.engine->qwen4exp_option_b_hot_masks[layer];
    }
  };

  if (split_active && g.engine && g.engine->qwen4exp_option_b_fill_vram) {
    const auto &c = g.engine->model_file.architecture_config;
    const int minimum = std::min(g.engine->qwen4exp_option_b_top_n,
                                 c.n_experts);
    size_t free_bytes = 0, total_bytes = 0;
    int previous_device = -1;
    cudaGetDevice(&previous_device);
    cudaSetDevice(g.device);
    const cudaError_t meminfo = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (previous_device >= 0)
      cudaSetDevice(previous_device);
    constexpr size_t kGraphFreeFloor = 1ull * 1024ull * 1024ull * 1024ull;
    constexpr size_t kArenaSlack = 1ull << 20;
    const size_t tensor_slack = g.split_gpu_expert_ids.size() * 3 * 256;
    const size_t reserve = kGraphFreeFloor + tensor_slack + kArenaSlack;
    const size_t max_data_bytes =
        meminfo == cudaSuccess && free_bytes > reserve
            ? free_bytes - reserve
            : 0;
    std::unordered_map<int, std::vector<int>> removed_experts;
    size_t bytes = device_expert_bytes(g);
    while (minimum > 0 && bytes > max_data_bytes) {
      int trim_layer = -1;
      size_t trim_per_expert = 0;
      for (const auto &[layer, ids] : g.split_gpu_expert_ids) {
        if (static_cast<int>(ids.size()) <= minimum)
          continue;
        const size_t per_expert = g.engine->compute_expert_slab_bytes(layer);
        if (per_expert > trim_per_expert) {
          trim_layer = layer;
          trim_per_expert = per_expert;
        }
      }
      if (trim_layer >= 0 && trim_per_expert != 0) {
        auto &ids = g.split_gpu_expert_ids[trim_layer];
        removed_experts[trim_layer].push_back(ids.back());
        ids.pop_back(); // ranked least-used expert in this layer
        bytes = device_expert_bytes(g);
        continue;
      }

      // If every selected layer is at the requested 50% floor and the
      // aggregate still does not fit, remove the lowest-coverage layer. This
      // preserves the user's layer-ranking rule instead of silently falling
      // back to the legacy dispatcher for the entire graph.
      int drop_layer = -1;
      double drop_coverage = std::numeric_limits<double>::infinity();
      for (const auto &[layer, ids] : g.split_gpu_expert_ids) {
        (void)ids;
        const auto it = g.engine->qwen4exp_option_b_layer_coverage.find(layer);
        const double coverage =
            it == g.engine->qwen4exp_option_b_layer_coverage.end()
                ? 0.0
                : it->second;
        if (coverage < drop_coverage ||
            (coverage == drop_coverage && layer > drop_layer)) {
          drop_layer = layer;
          drop_coverage = coverage;
        }
      }
      if (drop_layer < 0)
        break;
      LOG_WARN("[OPTION-B] VRAM capacity dropped layer %d at its %d-expert "
               "floor (coverage=%.3f)\n",
               drop_layer, minimum, drop_coverage * 100.0);
      g.split_gpu_expert_ids.erase(drop_layer);
      removed_experts.erase(drop_layer);
      g.engine->qwen4exp_option_b_hot_masks.erase(drop_layer);
      g.engine->qwen4exp_option_b_layer_coverage.erase(drop_layer);
      g.engine->qwen4exp_option_b_layers.erase(
          std::remove(g.engine->qwen4exp_option_b_layers.begin(),
                      g.engine->qwen4exp_option_b_layers.end(), drop_layer),
          g.engine->qwen4exp_option_b_layers.end());
      bytes = device_expert_bytes(g);
    }

    // After dropping whole layers, put back as many removed tail experts as
    // the remaining capacity allows. This makes the final placement as full
    // as possible rather than stopping at an unnecessarily conservative 50%.
    bool added = true;
    while (added) {
      added = false;
      for (auto &[layer, removed] : removed_experts) {
        auto it = g.split_gpu_expert_ids.find(layer);
        if (it == g.split_gpu_expert_ids.end() || removed.empty())
          continue;
        const size_t per_expert = g.engine->compute_expert_slab_bytes(layer);
        if (per_expert == 0 || bytes + per_expert > max_data_bytes)
          continue;
        it->second.push_back(removed.back());
        removed.pop_back();
        bytes = device_expert_bytes(g);
        added = true;
      }
    }
    sync_option_b_hot_masks();
    size_t final_experts = 0;
    for (const auto &[layer, ids] : g.split_gpu_expert_ids)
      final_experts += ids.size();
    LOG_INFO("[OPTION-B] VRAM fill: %.2f/%.2f GiB expert data, %zu expert "
             "records across %zu split layer(s), floor=%.2f GiB\n",
             static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0),
             static_cast<double>(max_data_bytes) /
                 (1024.0 * 1024.0 * 1024.0),
             final_experts, g.split_gpu_expert_ids.size(),
             static_cast<double>(kGraphFreeFloor) /
                 (1024.0 * 1024.0 * 1024.0));
  }

  for (;;) {
    size_t bytes = device_expert_bytes(g);
    if (bytes == 0) {
      if (g.device_expert_layers <= 0 && !split_active && !partial_active) {
        g.split_gpu_expert_ids.clear();
        return true;
      }
      if (allow_layer_clamp && g.device_expert_layers > 1) {
        --g.device_expert_layers;
        continue;
      }
      if (allow_layer_clamp && g.device_expert_layers == 1) {
        g.device_expert_layers = 0;
        g.device_expert_partial_gate_up = false;
        LOG_WARN("[GGML-QWEN-FULL] no whole expert layer fits in the "
                 "device arena; keeping all routed experts in DDR/CPU\n");
        return true;
      }
      return false;
    }
    constexpr size_t kAlignment = 256;
    size_t tensor_slack =
        static_cast<size_t>(g.device_expert_layers) * 3 * kAlignment;
    // Each split layer allocates 3 more small tensors (gate/up/down) into the
    // same bump arena -- same per-tensor alignment slack accounting as above.
    tensor_slack += g.split_gpu_expert_ids.size() * 3 * kAlignment;
    // The partial-layer compact tensor is also 3 tensors (gate/up/down).
    if (partial_active)
      tensor_slack += 3 * kAlignment;
    constexpr size_t kArenaSlack = 1ull << 20;
    if (bytes > SIZE_MAX - tensor_slack - kArenaSlack) {
      if (allow_layer_clamp && g.device_expert_layers > 1) {
        --g.device_expert_layers;
        continue;
      }
      return false;
    }
    const size_t capacity = bytes + tensor_slack + kArenaSlack;
    g.device_expert_buffer = ggml_backend_buft_alloc_buffer(buft, capacity);
    if (g.device_expert_buffer)
      {
        // build_graph() applies the same one-GiB free-VRAM floor before the
        // scheduler reserves its workspace.  Apply it here too, while the
        // whole-layer request is still retryable; otherwise a 24-layer arena
        // can allocate successfully and only fail later during graph
        // construction, returning the entire request to the legacy path.
        size_t free_bytes = 0, total_bytes = 0;
        const cudaError_t meminfo =
            cudaMemGetInfo(&free_bytes, &total_bytes);
        constexpr size_t kGraphFreeFloor = 1ull * 1024ull * 1024ull * 1024ull;
        if (meminfo == cudaSuccess && free_bytes < kGraphFreeFloor &&
            allow_layer_clamp && g.device_expert_layers > 1) {
          ggml_backend_buffer_free(g.device_expert_buffer);
          g.device_expert_buffer = nullptr;
          g.device_expert_base = nullptr;
          g.device_expert_capacity = 0;
          g.device_expert_used = 0;
          --g.device_expert_layers;
          continue;
        }
        break;
      }
    if (allow_layer_clamp && g.device_expert_layers > 1) {
      --g.device_expert_layers;
      continue;
    }
    if (allow_layer_clamp && g.device_expert_layers == 1) {
      g.device_expert_layers = 0;
      g.device_expert_partial_gate_up = false;
      LOG_WARN("[GGML-QWEN-FULL] device expert arena could not fit even "
               "one whole layer; keeping all routed experts in DDR/CPU\n");
      return true;
    }
    LOG_WARN(
        "[GGML-QWEN-FULL] device expert arena allocation declined: %.2f GiB\n",
        static_cast<double>(capacity) / (1024.0 * 1024.0 * 1024.0));
    return false;
  }
  if (g.device_expert_layers != requested_layers)
    LOG_WARN("[GGML-QWEN-FULL] requested %d device expert layers, but only "
             "%d whole layers fit; remaining layers stay in DDR/CPU\n",
             requested_layers, g.device_expert_layers);
  ggml_backend_buffer_set_usage(g.device_expert_buffer,
                                GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
  g.device_expert_base = ggml_backend_buffer_get_base(g.device_expert_buffer);
  g.device_expert_capacity =
      ggml_backend_buffer_get_size(g.device_expert_buffer);
  g.device_expert_used = 0;
  {
    const int n_layers = g.engine->model_file.architecture_config.n_layers;
    const int block_start =
        g.device_expert_trailing ? n_layers - g.device_expert_layers : 0;
    const int block_end =
        g.device_expert_trailing ? n_layers : g.device_expert_layers;
    LOG_INFO(
        "[GGML-QWEN-FULL] device expert arena=%.2f GiB for layers [%d,%d)%s%s "
        "split_layers=%zu\n",
        static_cast<double>(g.device_expert_capacity) /
            (1024.0 * 1024.0 * 1024.0),
        block_start, block_end,
        g.device_expert_partial_gate_up ? " + next-layer gate/up" : "",
        split_active ? " + per-expert split" : "",
        g.split_gpu_expert_ids.size());
  }
  return true;
}

std::string phase2_expert_ptr_text(const void *ptr) {
  return std::to_string(reinterpret_cast<uintptr_t>(ptr));
}

// Allocate one immutable expert tensor in the owner arena, or bind a sibling
// graph's tensor handle to the already allocated address.  The owner arena is
// sized before any graph is built; this helper only advances its bump cursor
// within that union-sized buffer and never allocates a second CUDA buffer.
bool phase2_bind_or_allocate_expert(FullGraph &g, ggml_tensor *t,
                                    const std::string &key, void **addr_out,
                                    bool *new_allocation) {
  if (!g.device_expert_buffer || !t || !addr_out || !new_allocation)
    return false;
  FullGraph &owner = g.phase2_expert_source_graph
                         ? *g.phase2_expert_source_graph
                         : g;
  auto existing = owner.phase2_expert_addrs.find(key);
  if (existing != owner.phase2_expert_addrs.end()) {
    *addr_out = existing->second;
    *new_allocation = false;
    return ggml_backend_tensor_alloc(owner.device_expert_buffer, t,
                                     *addr_out) == GGML_STATUS_SUCCESS;
  }

  const ggml_backend_buffer_type_t buft =
      ggml_backend_buffer_get_type(owner.device_expert_buffer);
  const size_t alloc_size = ggml_backend_buft_get_alloc_size(buft, t);
  constexpr size_t kAlignment = 256;
  const size_t alignment =
      std::max<size_t>(kAlignment, ggml_backend_buft_get_alignment(buft));
  const size_t aligned =
      (owner.device_expert_used + alignment - 1) & ~(alignment - 1);
  if (aligned > owner.device_expert_capacity ||
      alloc_size > owner.device_expert_capacity - aligned)
    return false;
  void *addr = static_cast<uint8_t *>(owner.device_expert_base) + aligned;
  if (ggml_backend_tensor_alloc(owner.device_expert_buffer, t, addr) !=
      GGML_STATUS_SUCCESS)
    return false;
  owner.device_expert_used = aligned + alloc_size;
  owner.phase2_expert_addrs.emplace(key, addr);
  *addr_out = addr;
  *new_allocation = true;
  return true;
}

bool bind_device_expert(FullGraph &g, ggml_tensor *t, const void *host,
                        size_t bytes) {
  if (!g.device_expert_buffer || !t || !host || bytes < ggml_nbytes(t))
    return false;
  const std::string key = "single:" + phase2_expert_ptr_text(host) + ":" +
                          std::to_string(bytes);
  void *addr = nullptr;
  bool new_allocation = false;
  if (!phase2_bind_or_allocate_expert(g, t, key, &addr, &new_allocation))
    return false;
  if (!new_allocation)
    return true;
  const auto t_memcpy_start = std::chrono::steady_clock::now();
  const cudaError_t err = cudaMemcpy(addr, host, bytes, cudaMemcpyHostToDevice);
  g.expert_upload_ms_total += std::chrono::duration<double, std::milli>(
                                  std::chrono::steady_clock::now() -
                                  t_memcpy_start)
                                  .count();
  ++g.expert_upload_calls;
  if (err != cudaSuccess) {
    LOG_ERROR("[GGML-QWEN-FULL] device expert upload failed: %s\n",
              cudaGetErrorString(err));
    return false;
  }
  return true;
}

// Same bump-allocation contract as bind_device_expert() above, but the
// source is one uniform-stride run of n_experts planes inside a RAM arena
// (base + e*src_stride for e in [0,n_experts)) rather than one contiguous
// host buffer -- the arena interleaves gate|up|down per expert, so a single
// plane's bytes across experts are NOT contiguous (see load_full_weight()'s
// device_resident arena path). A per-expert cudaMemcpy loop (like
// bind_device_expert_gather() below) works but pays real per-call CUDA API
// overhead n_experts times (measured: made node_construction WORSE, not
// better, at this model's expert count). cudaMemcpy2D expresses the whole
// strided transfer as ONE driver call (width=per_expert_bytes,
// height=n_experts, spitch=src_stride, dpitch=per_expert_bytes since the
// destination is freshly bump-allocated and therefore compact).
bool bind_device_expert_strided(FullGraph &g, ggml_tensor *t, const void *base,
                                size_t src_stride, size_t per_expert_bytes,
                                int n_experts) {
  if (!g.device_expert_buffer || !t || !base || src_stride == 0 ||
      per_expert_bytes == 0 || n_experts <= 0 || src_stride < per_expert_bytes)
    return false;
  if (static_cast<size_t>(n_experts) > SIZE_MAX / per_expert_bytes)
    return false;
  const size_t total_bytes =
      per_expert_bytes * static_cast<size_t>(n_experts);
  if (total_bytes < ggml_nbytes(t))
    return false;
  const std::string key =
      "strided:" + phase2_expert_ptr_text(base) + ":" +
      std::to_string(src_stride) + ":" + std::to_string(per_expert_bytes) +
      ":" + std::to_string(n_experts);
  void *addr = nullptr;
  bool new_allocation = false;
  if (!phase2_bind_or_allocate_expert(g, t, key, &addr, &new_allocation))
    return false;
  if (!new_allocation)
    return true;
  const auto t_memcpy_start = std::chrono::steady_clock::now();
  const cudaError_t err =
      cudaMemcpy2D(addr, per_expert_bytes, base, src_stride, per_expert_bytes,
                  static_cast<size_t>(n_experts), cudaMemcpyHostToDevice);
  g.expert_upload_ms_total += std::chrono::duration<double, std::milli>(
                                  std::chrono::steady_clock::now() -
                                  t_memcpy_start)
                                  .count();
  ++g.expert_upload_calls;
  if (err != cudaSuccess) {
    LOG_ERROR("[GGML-QWEN-FULL] device expert strided upload failed: %s\n",
              cudaGetErrorString(err));
    return false;
  }
  return true;
}

// Same bump-allocation contract as bind_device_expert() above, but the
// destination tensor `t` (shape [cols,rows,M], M == src_ptrs.size()) is
// filled from M separately-sourced host planes instead of one contiguous
// range -- the M hot experts plan_expert_vram_split() chose are not
// contiguous in the original expert axis. The destination IS contiguous
// (freshly allocated here), so only the source side needs per-expert
// copies; local index k in `t`'s expert axis holds src_ptrs[k]'s bytes.
bool bind_device_expert_gather(FullGraph &g, ggml_tensor *t,
                               const std::vector<const void *> &src_ptrs,
                               size_t per_expert_bytes) {
  if (!g.device_expert_buffer || !t || src_ptrs.empty() ||
      per_expert_bytes == 0)
    return false;
  if (src_ptrs.size() > SIZE_MAX / per_expert_bytes)
    return false;
  const size_t total_bytes = per_expert_bytes * src_ptrs.size();
  if (total_bytes < ggml_nbytes(t))
    return false;
  std::string key = "gather:" + std::to_string(per_expert_bytes);
  for (const void *src : src_ptrs)
    key += ":" + phase2_expert_ptr_text(src);
  void *addr = nullptr;
  bool new_allocation = false;
  if (!phase2_bind_or_allocate_expert(g, t, key, &addr, &new_allocation))
    return false;
  if (!new_allocation)
    return true;
  for (size_t k = 0; k < src_ptrs.size(); ++k) {
    void *dst = static_cast<uint8_t *>(addr) + k * per_expert_bytes;
    const cudaError_t err =
        cudaMemcpy(dst, src_ptrs[k], per_expert_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
      LOG_ERROR(
          "[GGML-QWEN-FULL-SPLIT] device expert gather-copy failed at "
          "local index %zu/%zu: %s\n",
          k, src_ptrs.size(), cudaGetErrorString(err));
      // Bytes already copied for k'<k stay allocated (arena is a bump
      // allocator with no partial free) but the caller treats a false
      // return as "this tensor is unusable" and falls back to CPU-only for
      // this layer -- the wasted arena bytes are the same bounded, one-time
      // cost bind_device_expert()'s own failure path already accepts.
      return false;
    }
  }
  return true;
}

// Projection-compact counterpart of bind_device_expert_gather(). Gate and up
// are separate host planes, so copy the two source slices into each compact
// VRAM expert record. This is a startup-only graph construction operation; it
// avoids rebuilding a large interleaved DDR copy merely to preserve the GPU
// gate/up fusion.
// Uniform-stride counterpart to bind_device_expert_gather_gate_up() below --
// same interleaved-destination contract (dst holds [gate_e|up_e] per expert,
// compact), but the two sources are each one uniform-stride run
// (gate_base+e*gate_stride, up_base+e*up_stride) rather than arbitrary
// per-expert pointers, so each plane's whole transfer is one cudaMemcpy2D
// call instead of n_experts separate ones (same rationale as
// bind_device_expert_strided() above -- measured real win on this model's
// expert count).
bool bind_device_expert_strided_gate_up(FullGraph &g, ggml_tensor *t,
                                        const void *gate_base,
                                        size_t gate_stride, size_t gate_bytes,
                                        const void *up_base, size_t up_stride,
                                        size_t up_bytes, int n_experts) {
  if (!g.device_expert_buffer || !t || !gate_base || !up_base ||
      gate_bytes == 0 || up_bytes == 0 || n_experts <= 0 ||
      gate_stride < gate_bytes || up_stride < up_bytes ||
      gate_bytes > SIZE_MAX - up_bytes)
    return false;
  const size_t per_expert_bytes = gate_bytes + up_bytes;
  if (static_cast<size_t>(n_experts) > SIZE_MAX / per_expert_bytes)
    return false;
  const size_t total_bytes =
      per_expert_bytes * static_cast<size_t>(n_experts);
  if (total_bytes < ggml_nbytes(t))
    return false;
  const std::string key =
      "strided_gate_up:" + phase2_expert_ptr_text(gate_base) + ":" +
      std::to_string(gate_stride) + ":" + std::to_string(gate_bytes) + ":" +
      phase2_expert_ptr_text(up_base) + ":" + std::to_string(up_stride) +
      ":" + std::to_string(up_bytes) + ":" + std::to_string(n_experts);
  void *addr = nullptr;
  bool new_allocation = false;
  if (!phase2_bind_or_allocate_expert(g, t, key, &addr, &new_allocation))
    return false;
  if (!new_allocation)
    return true;
  const auto t_memcpy_start = std::chrono::steady_clock::now();
  cudaError_t err =
      cudaMemcpy2D(addr, per_expert_bytes, gate_base, gate_stride, gate_bytes,
                  static_cast<size_t>(n_experts), cudaMemcpyHostToDevice);
  if (err == cudaSuccess)
    err = cudaMemcpy2D(static_cast<uint8_t *>(addr) + gate_bytes,
                       per_expert_bytes, up_base, up_stride, up_bytes,
                       static_cast<size_t>(n_experts), cudaMemcpyHostToDevice);
  g.expert_upload_ms_total += std::chrono::duration<double, std::milli>(
                                  std::chrono::steady_clock::now() -
                                  t_memcpy_start)
                                  .count();
  ++g.expert_upload_calls;
  if (err != cudaSuccess) {
    LOG_ERROR("[GGML-QWEN-FULL] compact gate/up strided upload failed: %s\n",
              cudaGetErrorString(err));
    return false;
  }
  return true;
}


ggml_tensor *load_full_weight(FullGraph &g, ggml_context *ctx,
                              const std::string &name,
                              bool device_resident = false) {
  const GGUFTensor *m = model_tensor(*g.engine, name);
  if (!m || !type_ok(m->type) || m->shape.size() < 2)
    return nullptr;
  std::vector<int64_t> shape;
  for (uint64_t d : m->shape)
    shape.push_back(static_cast<int64_t>(d));
  ggml_tensor *t = new_tensor(ctx, static_cast<ggml_type>(m->type), shape);
  if (!t)
    return nullptr;
  const size_t bytes = static_cast<size_t>(m->size_bytes);
  const size_t required = std::max(bytes, ggml_nbytes(t));

  // A Qwen full graph must consume the same complete DDR expert planes that
  // the RAM tier already loaded.  In that configuration the source is a
  // fixed-stride arena, so adjust only the expert-axis stride; no selected
  // weights are copied and no GGUF mapping is consulted.
  //
  // 2026-09-17: this arena source used to be consulted ONLY for
  // !device_resident (the host-alias path just below). A device_resident
  // caller (device_expert_layers' whole-layer VRAM block) fell through
  // unconditionally to the pread_direct_aware() fallback further down --
  // a REAL re-read from disk/page-cache of the entire device-resident expert
  // block, on every single build_graph() call, even though the RAM tier had
  // already loaded and validated (graph_ram_layer_ready()) these exact
  // bytes moments earlier. Root-caused via a chrono split of node_construction
  // (see memory: full-graph-decode-graph-construction-5s-latency.md) that
  // isolated ~3.1s of the ~5.3s decode-build cost to this function, entirely
  // OUTSIDE the already-measured dense/expert cudaMemcpy timers -- consistent
  // with device_expert_bytes() (~17.68 GiB at device_expert_layers=22) moving
  // through a page-cache-speed host memcpy that no counter was watching.
  // Fixed by transferring straight from the arena into VRAM via
  // bind_device_expert_strided() (one cudaMemcpy2D per tensor, since the
  // arena's per-plane expert stride is uniform), skipping the host
  // round-trip entirely instead of just skipping the disk read. A first cut
  // used a per-expert cudaMemcpy loop (matching expert_gate_up_weight()'s
  // pre-existing bind_device_expert_gather() mechanism) but measured WORSE
  // than the disk-read baseline at this model's expert count -- n_experts
  // separate CUDA API calls cost more than the read they replaced;
  // expert_gate_up_weight()'s own device_resident branch had the same
  // per-call-overhead problem and was converted to
  // bind_device_expert_strided()/bind_device_expert_strided_gate_up() too.
  {
    void *ram_ptr = nullptr;
    size_t ram_bytes = 0;
    size_t ram_stride = 0;
    size_t ram_arena_bytes = 0;
    if (preloaded_ram_expert_source(g, name, *m, ram_ptr, ram_bytes, ram_stride,
                                    ram_arena_bytes)) {
      if (shape.size() < 3 ||
          t->ne[2] != g.engine->model_file.architecture_config.n_experts ||
          ram_stride < t->nb[2])
        return nullptr;
      if (device_resident) {
        const int n_experts = static_cast<int>(t->ne[2]);
        if (n_experts <= 0 || bytes % static_cast<size_t>(n_experts) != 0)
          return nullptr;
        const size_t per_expert_bytes =
            bytes / static_cast<size_t>(n_experts);
        if (!bind_device_expert_strided(g, t, ram_ptr, ram_stride,
                                        per_expert_bytes, n_experts))
          return nullptr;
        ggml_set_name(t, name.c_str());
        return t;
      }

      t->nb[2] = ram_stride;
      t->nb[3] = ram_stride * static_cast<size_t>(t->ne[2]);
      if (!bind_host(g, t, ram_ptr, ram_bytes,
                     GGML_BACKEND_BUFFER_USAGE_WEIGHTS))
        return nullptr;
      // The three projection views overlap the same one-layer arena. Count
      // the backing arena once, on the gate view, rather than summing the
      // three strided view spans and reporting a fictitious 3x allocation.
      if (name.find("ffn_gate_exps.weight") != std::string::npos)
        g.host_ram_alias_bytes += ram_arena_bytes;
      ++g.host_ram_alias_tensors;
      ggml_set_name(t, name.c_str());
      bind_host_expert_scheduler_backend(g, t);
      return t;
    }

    // A graph-RAM arena is an explicit ownership decision: once it exists,
    // this Qwen layer is supposed to use the already-preloaded DDR bytes.
    // Do not quietly turn an incomplete arena into a second graph-owned
    // startup read from the GGUF file.  Rejecting the optional graph keeps
    // the normal RAM/cache dispatcher as the correctness fallback and makes
    // the source policy visible instead of hiding an SSD read behind this
    // path.  This guard is deliberately Qwen-full-graph-only; other model
    // families have their own graph source ownership rules.
    int graph_ram_layer = -1;
    if (name.compare(0, 4, "blk.") == 0) {
      char *end = nullptr;
      const long parsed = std::strtol(name.c_str() + 4, &end, 10);
      if (end != name.c_str() + 4 && end && *end == '.')
        graph_ram_layer = static_cast<int>(parsed);
    }
    const bool narrow_layer0_diag = env_layer0_narrow_graph_diag_enabled();
    const bool require_preloaded_ram =
        g.engine->qwen_ram_preload_all_experts &&
        (!narrow_layer0_diag || graph_ram_layer == 0);
    if (graph_ram_layer >= 0 && require_preloaded_ram) {
      LOG_WARN("[GGML-QWEN-FULL] refusing file-backed expert source for "
               "layer=%d: ram_preload_all_experts requires the preloaded "
               "DDR arena\n",
               graph_ram_layer);
      return nullptr;
    }
    if (graph_ram_layer >= 0 &&
        g.engine->graph_ram_layer_allocated(graph_ram_layer) &&
        !g.engine->graph_ram_layer_ready(graph_ram_layer)) {
      LOG_WARN("[GGML-QWEN-FULL] refusing file-backed expert source for "
               "layer=%d: preloaded DDR arena is allocated but not ready\n",
               graph_ram_layer);
      return nullptr;
    }
  }

  void *p = alloc_host(g, required);
  if (!p)
    return nullptr;
  // Fallback for graph configurations without a complete preloaded RAM
  // arena. This is a startup copy into graph-owned DDR via pread; it is not
  // the decode-time CPU source used by the DDR-affinity path above.
  const int fd = g.engine->model_file.fd_for_tensor(*m);
  const ssize_t got = direct_io::pread_direct_aware(
      fd, g.engine->model_file.direct_io_active_for(*m),
      g.engine->model_file.direct_io_alignment, p, bytes, m->offset);
  if (got < 0 || static_cast<size_t>(got) != bytes) {
    LOG_ERROR("[GGML-QWEN-FULL] short read for %s: %zd/%zu\n", name.c_str(),
              got, bytes);
    return nullptr;
  }
  if (device_resident) {
    if (!bind_device_expert(g, t, p, bytes))
      return nullptr;
  } else if (!bind_host(g, t, p, required, GGML_BACKEND_BUFFER_USAGE_WEIGHTS)) {
    return nullptr;
  } else {
    // GGML's CUDA scheduler uses set_tensor_async for host-backed expert
    // weights. Register each graph-owned source once, after it has been
    // faulted in by the read above, so those copies can use DMA directly.
    // Registration is outside the measured prefill window and is best effort:
    // this machine has a finite aggregate CUDA host-registration ceiling.
    const size_t owned_index = g.owned.size() - 1;
    const cudaError_t pin_err =
        cudaHostRegister(p, required, cudaHostRegisterPortable);
    if (pin_err == cudaSuccess) {
      g.owned[owned_index].pinned = true;
      g.host_pinned_bytes += required;
    } else {
      ++g.host_pin_failures;
      cudaGetLastError();
      if (g.host_pin_failures == 1) {
        LOG_WARN(
            "[GGML-QWEN-FULL] host expert pinning declined at %.2f GiB: %s; "
            "remaining sources stay pageable\n",
            static_cast<double>(g.host_pinned_bytes) /
                (1024.0 * 1024.0 * 1024.0),
            cudaGetErrorString(pin_err));
      }
    }
  }
  g.host_weight_bytes += bytes;
  ggml_set_name(t, name.c_str());
  if (!device_resident)
    bind_host_expert_scheduler_backend(g, t);
  return t;
}

ggml_tensor *expert_weight(FullGraph &g, ggml_context *ctx, int layer,
                           const char *stem) {
  return load_full_weight(g, ctx, "blk." + std::to_string(layer) + "." + stem,
                          graph_expert_projection_resident(g, layer, stem));
}

// When gate and up have the same quantized row layout, expose them as one
// [cols, 2*rows, n_expert] MUL_MAT_ID weight. GPU graph layers retain this
// merge by gathering the two compact host planes into their final VRAM tensor.
// CPU graph layers use separate views: gate and up are separate planes in the
// projection-compact arena, and making a merged host tensor would recreate the
// large duplicate this layout is intended to remove.
ggml_tensor *expert_gate_up_weight(FullGraph &g, ggml_context *ctx, int layer,
                                   bool device_resident) {
  const std::string base = "blk." + std::to_string(layer) + ".";
  const GGUFTensor *gate = model_tensor(
      *g.engine, base + "ffn_gate_exps.weight");
  const GGUFTensor *up =
      model_tensor(*g.engine, base + "ffn_up_exps.weight");
  const int n_experts = g.engine->model_file.architecture_config.n_experts;
  if (!gate || !up || n_experts <= 0 || gate->type != up->type ||
      gate->shape.size() != 3 || up->shape != gate->shape ||
      gate->shape[2] != static_cast<uint64_t>(n_experts) ||
      gate->size_bytes == 0 || gate->size_bytes != up->size_bytes ||
      gate->size_bytes % static_cast<uint64_t>(n_experts) != 0)
    return nullptr;

  const auto *arena = g.engine->graph_ram_arena(layer);
  if (!arena || !g.engine->graph_ram_layer_ready(layer) || !arena->base ||
      arena->n_experts != n_experts || arena->gate_bytes == 0 ||
      arena->gate_bytes != arena->up_bytes ||
      arena->gate_bytes !=
          static_cast<size_t>(gate->size_bytes) /
              static_cast<size_t>(n_experts) ||
      arena->up_bytes !=
              static_cast<size_t>(up->size_bytes) /
              static_cast<size_t>(n_experts) ||
      arena->gate_bytes > SIZE_MAX - arena->up_bytes)
    return nullptr;

  if (!device_resident && arena->projection_compact)
    return nullptr;

  const size_t plane_bytes = arena->gate_bytes + arena->up_bytes;
  if (plane_bytes == 0)
    return nullptr;
  const size_t legacy_view_bytes =
      arena->stride != 0 &&
              static_cast<size_t>(n_experts - 1) <=
                  (SIZE_MAX - plane_bytes) / arena->stride
          ? arena->stride * static_cast<size_t>(n_experts - 1) + plane_bytes
          : 0;
  if (!arena->projection_compact &&
      (arena->stride < plane_bytes || legacy_view_bytes == 0 ||
       legacy_view_bytes > arena->bytes))
    return nullptr;

  std::vector<int64_t> shape;
  shape.reserve(gate->shape.size());
  for (uint64_t d : gate->shape)
    shape.push_back(static_cast<int64_t>(d));
  if (shape[1] > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) /
                       2)
    return nullptr;
  shape[1] *= 2;
  ggml_tensor *t = new_tensor(ctx, static_cast<ggml_type>(gate->type), shape);
  if (!t || t->ne[0] != static_cast<int64_t>(gate->shape[0]) ||
      t->ne[1] != static_cast<int64_t>(gate->shape[1]) * 2 ||
      t->ne[2] != n_experts)
    return nullptr;

  if (device_resident) {
    // Keep ggml's default compact expert-axis strides here; using a DDR
    // arena stride on this descriptor would make ggml_nbytes(t) include the
    // skipped down plane/padding and reject the valid VRAM allocation.
    if (arena->projection_compact) {
      if (!bind_device_expert_strided_gate_up(
              g, t, arena->base + arena->gate_offset, arena->gate_stride,
              arena->gate_bytes, arena->base + arena->up_offset,
              arena->up_stride, arena->up_bytes, n_experts))
        return nullptr;
    } else {
      // Uniform expert-axis stride -- same shape bind_device_expert_strided()
      // (load_full_weight()'s own device_resident arena path) already
      // handles with one cudaMemcpy2D instead of an n_experts-call loop.
      if (!bind_device_expert_strided(g, t, arena->base, arena->stride,
                                      plane_bytes, n_experts))
        return nullptr;
    }
  } else {
    // The host view aliases the interleaved DDR arena, so its expert-axis
    // stride must skip the following down projection and any alignment
    // padding in each source slab.
    t->nb[2] = arena->stride;
    t->nb[3] = arena->stride * static_cast<size_t>(n_experts);
    if (!bind_host(g, t, arena->base, legacy_view_bytes,
                   GGML_BACKEND_BUFFER_USAGE_WEIGHTS))
      return nullptr;
    g.host_ram_alias_bytes += arena->bytes;
    ++g.host_ram_alias_tensors;
  }
  ggml_set_name(t, (base + "ffn_gate_up_exps.weight").c_str());
  if (!device_resident)
    bind_host_expert_scheduler_backend(g, t);
  return t;
}

// Builds the [cols,rows,M] VRAM-resident tensor for a genuinely split layer
// (0<M<n_expert), gathering the M hot experts named by `hot_ids` straight
// out of the SAME preloaded RAM arena bytes the (unchanged) full-width CPU
// tensor for this same (layer,stem) already aliases -- reuses
// preloaded_ram_expert_source()'s existing base+stride resolution, no new
// stride math. Returns nullptr (never a hard error -- caller falls back to
// CPU-only for this layer) whenever the arena isn't ready, e.g. during
// prefill/non-decode graph builds that don't unconditionally require it.
ggml_tensor *build_split_gpu_tensor(FullGraph &g, ggml_context *ctx,
                                    const std::string &name,
                                    const GGUFTensor &m,
                                    const std::vector<int> &hot_ids) {
  void *ptr = nullptr;
  size_t bytes = 0, stride = 0, arena_bytes = 0;
  if (!preloaded_ram_expert_source(g, name, m, ptr, bytes, stride,
                                   arena_bytes))
    return nullptr;
  const int n_experts = g.engine->model_file.architecture_config.n_experts;
  if (n_experts <= 0 || m.size_bytes % static_cast<uint64_t>(n_experts) != 0)
    return nullptr;
  const size_t plane_bytes =
      static_cast<size_t>(m.size_bytes) / static_cast<size_t>(n_experts);
  if (plane_bytes == 0)
    return nullptr;

  std::vector<int64_t> shape;
  shape.reserve(m.shape.size());
  for (uint64_t d : m.shape)
    shape.push_back(static_cast<int64_t>(d));
  if (shape.size() < 3 || shape[2] != n_experts)
    return nullptr;
  shape[2] = static_cast<int64_t>(hot_ids.size());
  ggml_tensor *t = new_tensor(ctx, static_cast<ggml_type>(m.type), shape);
  if (!t)
    return nullptr;

  std::vector<const void *> src_ptrs;
  src_ptrs.reserve(hot_ids.size());
  for (int e : hot_ids) {
    if (e < 0 || e >= n_experts)
      return nullptr;
    src_ptrs.push_back(static_cast<const uint8_t *>(ptr) +
                       static_cast<size_t>(e) * stride);
  }
  if (!bind_device_expert_gather(g, t, src_ptrs, plane_bytes))
    return nullptr;
  const int stage_layer = g.engine->qwen4exp_option_b_layer;
  if (g.decode_mode && !g.stage_experts.empty() && stage_layer >= 0) {
    const auto *arena = g.engine->graph_ram_arena(stage_layer);
    if (arena && g.engine->graph_ram_layer_ready(stage_layer)) {
      if (g.stage_bindings.empty()) {
        for (int expert : g.stage_experts) {
          const auto it = std::find(hot_ids.begin(), hot_ids.end(), expert);
          if (it == hot_ids.end())
            continue;
          FullGraph::StageBinding binding;
          binding.expert = expert;
          binding.local_index = static_cast<int>(it - hot_ids.begin());
          g.stage_bindings.push_back(binding);
        }
      }
      const std::string gate_stem = "ffn_gate_exps.weight";
      const std::string up_stem = "ffn_up_exps.weight";
      const std::string down_stem = "ffn_down_exps.weight";
      g.stage_layer = stage_layer;
      for (auto &binding : g.stage_bindings) {
        const size_t dst_offset =
            static_cast<size_t>(binding.local_index) * plane_bytes;
        const uint8_t *src = static_cast<const uint8_t *>(ptr) +
                             static_cast<size_t>(binding.expert) * stride;
        if (arena->projection_compact) {
          if (name.find(gate_stem) != std::string::npos)
            binding.src_gate = arena->base + arena->gate_offset +
                               static_cast<size_t>(binding.expert) *
                                   arena->gate_stride;
          else if (name.find(up_stem) != std::string::npos)
            binding.src_up = arena->base + arena->up_offset +
                             static_cast<size_t>(binding.expert) *
                                 arena->up_stride;
          else if (name.find(down_stem) != std::string::npos)
            binding.src_down = arena->base + arena->down_offset +
                               static_cast<size_t>(binding.expert) *
                                   arena->down_stride;
        } else {
          if (name.find(gate_stem) != std::string::npos)
            binding.src_gate = src;
          else if (name.find(up_stem) != std::string::npos)
            binding.src_up = src + arena->gate_bytes;
          else if (name.find(down_stem) != std::string::npos)
            binding.src_down = src + arena->gate_bytes + arena->up_bytes;
        }
        if (name.find(gate_stem) != std::string::npos) {
          binding.dst_gate = static_cast<uint8_t *>(t->data) + dst_offset;
          binding.gate_bytes = plane_bytes;
        } else if (name.find(up_stem) != std::string::npos) {
          binding.dst_up = static_cast<uint8_t *>(t->data) + dst_offset;
          binding.up_bytes = plane_bytes;
        } else if (name.find(down_stem) != std::string::npos) {
          binding.dst_down = static_cast<uint8_t *>(t->data) + dst_offset;
          binding.down_bytes = plane_bytes;
        }
      }
      bool complete = !g.stage_bindings.empty();
      for (const auto &binding : g.stage_bindings) {
        complete = complete && binding.src_gate && binding.src_up &&
                   binding.src_down && binding.dst_gate && binding.dst_up &&
                   binding.dst_down && binding.gate_bytes && binding.up_bytes &&
                   binding.down_bytes;
      }
      if (complete) {
        g.stage_enabled = true;
        g.stage_expert = g.stage_bindings.front().expert;
        g.stage_local_index = g.stage_bindings.front().local_index;
        g.stage_src_gate = g.stage_bindings.front().src_gate;
        g.stage_src_up = g.stage_bindings.front().src_up;
        g.stage_src_down = g.stage_bindings.front().src_down;
        g.stage_dst_gate = g.stage_bindings.front().dst_gate;
        g.stage_dst_up = g.stage_bindings.front().dst_up;
        g.stage_dst_down = g.stage_bindings.front().dst_down;
        g.stage_gate_bytes = g.stage_bindings.front().gate_bytes;
        g.stage_up_bytes = g.stage_bindings.front().up_bytes;
        g.stage_down_bytes = g.stage_bindings.front().down_bytes;
      }
    }
  }
  ggml_set_name(t, (name + ".split_gpu").c_str());
  return t;
}

struct SplitExpertWeight {
  ggml_tensor *cpu = nullptr; // full [cols,rows,n_expert] -- today's tensor,
                              // unchanged, used either as-is (no split) or
                              // as build_moe's "CPU branch" input (split).
  ggml_tensor *gpu = nullptr; // [cols,rows,M], only for a genuine split.
};

// Split-aware replacement for expert_weight() at build_graph()'s MoE call
// site. Behaves EXACTLY like expert_weight() (returns only `.cpu`, `.gpu`
// stays null) for every layer graph_expert_projection_resident() already
// resolves (whole-layer CPU, or whole-layer/boundary VRAM-resident) --
// those paths are untouched. Only consults g.split_gpu_expert_ids, and
// only when g.S==1 (decode; see plan_expert_vram_split's own S==1 guard),
// so a split config never affects prefill's tensor construction at all.
SplitExpertWeight expert_weight_split(FullGraph &g, ggml_context *ctx,
                                      int layer, const char *stem) {
  SplitExpertWeight r;
  const std::string name = "blk." + std::to_string(layer) + "." + stem;
  if (graph_expert_projection_resident(g, layer, stem)) {
    r.cpu = load_full_weight(g, ctx, name, /*device_resident=*/true);
    return r; // today's whole-layer/boundary VRAM path, unchanged
  }
  r.cpu = load_full_weight(g, ctx, name, /*device_resident=*/false);
  if (!r.cpu || g.S != 1)
    return r; // today's whole-layer CPU path, unchanged; or non-decode build
  auto it = g.split_gpu_expert_ids.find(layer);
  if (it == g.split_gpu_expert_ids.end() || it->second.empty())
    return r; // no split planned for this layer -- today's path, unchanged
  const GGUFTensor *m = model_tensor(*g.engine, name);
  if (!m)
    return r;
  r.gpu = build_split_gpu_tensor(g, ctx, name, *m, it->second);
  if (!r.gpu)
    LOG_WARN("[GGML-QWEN-FULL-SPLIT] layer=%d %s: planned split unavailable "
             "this run (RAM arena not ready) -- falling back to CPU-only "
             "for this layer\n",
             layer, stem);
  return r;
}

ggml_tensor *dense_weight(FullGraph &g, ggml_context *ctx, int layer,
                          const char *stem) {
  return resident_tensor(g, ctx, "blk." + std::to_string(layer) + "." + stem);
}

ggml_tensor *hc_mix(FullGraph &g, ggml_context *ctx, ggml_tensor *x,
                    ggml_tensor *norm, ggml_tensor *down, ggml_tensor *up,
                    ggml_tensor *inject, ggml_tensor **inject_out,
                    int diag_layer = -1, const char *diag_tag = nullptr) {
  ggml_tensor *xn =
      ggml_rms_norm(ctx, x, g.engine->model_file.architecture_config.eps);
  xn = ggml_reshape_2d(ctx, xn, g.D * g.HC, g.S);
  xn = ggml_mul(ctx, xn, norm);
  if (g.two_layer_diag && diag_layer == 0 && diag_tag &&
      !std::strcmp(diag_tag, "attn")) {
    if (const char *common_path = env_two_layer_s1_common_hc_norm_path()) {
      const size_t common_n = static_cast<size_t>(g.D) * g.HC * g.S;
      ggml_tensor *common = host_input(
          g, ctx, GGML_TYPE_F32, {g.D * g.HC, g.S},
          common_n * sizeof(float), "qwen_two_layer_s1_common_hc_norm");
      if (!common || !load_exact_f32_file(
                           common_path, static_cast<float *>(common->data),
                           common_n)) {
        LOG_WARN(
            "[QWEN4EXP-TWO-LAYER-S1] common HC norm rejected: path=%s "
            "expected_bytes=%zu\n",
            common_path, common_n * sizeof(float));
      } else {
        const uint64_t hash = qwen4exp_fnv64(
            common->data, common_n * sizeof(float));
        LOG_INFO(
            "[QWEN4EXP-TWO-LAYER-S1] forcing Layer-0 HC projection input "
            "from %s elements=%zu fnv64=0x%016llx\n",
            common_path, common_n, static_cast<unsigned long long>(hash));
        xn = common;
      }
    }
  }
  const bool capture_projection_diag =
      diag_layer >= 0 && diag_layer < 2 && diag_tag && g.layer0_diag_enabled &&
      env_layer0_split_dump_path();
  if (capture_projection_diag) {
    const std::string down_role = std::string(diag_tag) +
                                  "_hc_down_weight";
    const std::string up_role = std::string(diag_tag) + "_hc_up_weight";
    log_layer0_diag_tensor(down_role.c_str(), down);
    log_layer0_diag_tensor(up_role.c_str(), up);
    const std::string down_name = std::string(diag_tag) +
                                  "_hc_down_weight";
    const std::string up_name = std::string(diag_tag) + "_hc_up_weight";
    g.layer0_split_snapshots.emplace_back(
        down_name, layer0_diag_snapshot(g, ctx, down, down_name.c_str()));
    g.layer0_split_snapshots.emplace_back(
        up_name, layer0_diag_snapshot(g, ctx, up, up_name.c_str()));
  }
  if (capture_projection_diag) {
    const std::string name = std::string("qwen_layer0_split_") + diag_tag +
                             "_hc_norm";
    g.layer0_split_snapshots.emplace_back(
        std::string(diag_tag) + "_hc_norm",
        layer0_diag_snapshot(g, ctx, xn, name.c_str()));
    const std::string weight_name = std::string("qwen_layer0_split_") +
                                    diag_tag + "_hc_norm_weight";
    g.layer0_split_snapshots.emplace_back(
        std::string(diag_tag) + "_hc_norm_weight",
        layer0_diag_snapshot(g, ctx, norm, weight_name.c_str()));
  }
  ggml_tensor *lo = ggml_mul_mat(ctx, down, xn);
  if (capture_projection_diag) {
    const std::string name = std::string("qwen_layer0_split_") + diag_tag +
                             "_hc_down_raw";
    g.layer0_split_snapshots.emplace_back(
        std::string(diag_tag) + "_hc_down_raw",
        layer0_diag_snapshot(g, ctx, lo, name.c_str()));
  }
  lo = ggml_silu(ctx, ggml_scale(ctx, lo, 1.0f / static_cast<float>(g.HC)));
  if (g.two_layer_diag && diag_layer == 0 && diag_tag &&
      !std::strcmp(diag_tag, "attn")) {
    if (const char *common_path =
            env_two_layer_s1_common_hc_down_silu_path()) {
      const size_t common_n = static_cast<size_t>(ggml_nelements(lo));
      ggml_tensor *common = host_input(
          g, ctx, GGML_TYPE_F32, {lo->ne[0], lo->ne[1]},
          common_n * sizeof(float), "qwen_two_layer_s1_common_hc_down_silu");
      if (!common || !load_exact_f32_file(
                           common_path, static_cast<float *>(common->data),
                           common_n)) {
        LOG_WARN(
            "[QWEN4EXP-TWO-LAYER-S1] common HC down-SiLU rejected: "
            "path=%s expected_bytes=%zu\n",
            common_path, common_n * sizeof(float));
      } else {
        const uint64_t hash = qwen4exp_fnv64(
            common->data, common_n * sizeof(float));
        LOG_INFO(
            "[QWEN4EXP-TWO-LAYER-S1] forcing Layer-0 HC up input from %s "
            "elements=%zu fnv64=0x%016llx\n",
            common_path, common_n, static_cast<unsigned long long>(hash));
        lo = common;
      }
    }
  }
  if (capture_projection_diag) {
    const std::string name = std::string("qwen_layer0_split_") + diag_tag +
                             "_hc_down_silu";
    g.layer0_split_snapshots.emplace_back(
        std::string(diag_tag) + "_hc_down_silu",
        layer0_diag_snapshot(g, ctx, lo, name.c_str()));
  }
  ggml_tensor *gate_logits = ggml_mul_mat(ctx, up, lo);
  if (capture_projection_diag) {
    const std::string name = std::string("qwen_layer0_split_") + diag_tag +
                             "_hc_up_logits";
    g.layer0_split_snapshots.emplace_back(
        std::string(diag_tag) + "_hc_up_logits",
        layer0_diag_snapshot(g, ctx, gate_logits, name.c_str()));
  }
  ggml_tensor *gate = ggml_sigmoid(ctx, gate_logits);
  if (capture_projection_diag) {
    const std::string name = std::string("qwen_layer0_split_") + diag_tag +
                             "_hc_gate";
    g.layer0_split_snapshots.emplace_back(
        std::string(diag_tag) + "_hc_gate",
        layer0_diag_snapshot(g, ctx, gate, name.c_str()));
  }
  ggml_tensor *gated = ggml_mul(ctx, xn, gate);
  ggml_tensor *g3 = ggml_reshape_3d(ctx, gated, g.D, g.HC, g.S);

  ggml_tensor *mixed =
      ggml_cont(ctx, ggml_view_2d(ctx, g3, g.D, g.S,
                                  ggml_row_size(g3->type, g.D) * g.HC, 0));
  for (int h = 1; h < g.HC; ++h) {
    ggml_tensor *s =
        ggml_view_2d(ctx, g3, g.D, g.S, ggml_row_size(g3->type, g.D) * g.HC,
                     ggml_row_size(g3->type, g.D) * h);
    mixed = ggml_add(ctx, mixed, s);
  }
  mixed = ggml_scale(ctx, mixed, 1.0f / static_cast<float>(g.HC));
  if (inject_out) {
    *inject_out = inject ? ggml_mul_mat(ctx, inject, xn) : nullptr;
    if (diag_layer >= 0 && diag_layer < 2 && diag_tag &&
        g.layer0_diag_enabled &&
        env_layer0_split_dump_path()) {
      const std::string name = std::string("qwen_layer0_split_") + diag_tag +
                               "_hc_inject";
      g.layer0_split_snapshots.emplace_back(
          std::string(diag_tag) + "_hc_inject",
          layer0_diag_snapshot(g, ctx, *inject_out, name.c_str()));
    }
  }
  return mixed;
}

ggml_tensor *hc_combine(FullGraph &g, ggml_context *ctx, ggml_tensor *residual,
                        ggml_tensor *block, ggml_tensor *inject) {
  ggml_tensor *w = ggml_scale(
      ctx, ggml_sigmoid(ctx, ggml_scale(ctx, inject, 1.0f / g.HC)), 2.0f);
  w = ggml_reshape_3d(ctx, w, 1, g.HC, g.S);
  ggml_tensor *b = ggml_reshape_3d(ctx, block, g.D, 1, g.S);
  b = ggml_repeat_4d(ctx, b, g.D, g.HC, g.S, 1);
  return ggml_add(ctx, residual, ggml_mul(ctx, b, w));
}

void *alloc_registered_host(FullGraph &g, size_t bytes) {
  void *ptr = alloc_host(g, bytes);
  if (!ptr || bytes == 0)
    return nullptr;
  const cudaError_t err =
      cudaHostRegister(ptr, bytes, cudaHostRegisterPortable);
  if (err != cudaSuccess) {
    cudaGetLastError();
    LOG_WARN("[QWEN4EXP-HOST-NODE] pinned host registration declined for "
             "%zu bytes: %s\n",
             bytes, cudaGetErrorString(err));
    return nullptr;
  }
  g.owned.back().pinned = true;
  g.host_pinned_bytes += bytes;
  return ptr;
}

// Replace one complete routed MoE FFN with a CUDA-captured host marker. The
// marker carries route tensors as hidden GGML dependencies, while the bridge
// uses fixed pinned buffers and an immutable DDR arena. Both the legacy
// interleaved arena and the newer projection-compact arena are supported;
// the latter is passed to GGML through three independent expert-axis strides.
// This is intentionally decode-only: the CPU runner's strided interface is a
// single-token contract and the scheduler-split prefill path remains unchanged.
ggml_tensor *build_qwen4exp_host_node(
    FullGraph &g, ggml_context *ctx, ggml_tensor *cur, ggml_tensor *ids,
    ggml_tensor *weights, int layer) {
  if (!g.decode_mode || g.S != 1 || !g.engine ||
      !g.engine->qwen4exp_cpu_offload_layer_enabled(layer))
    return nullptr;

  const auto &c = g.engine->model_file.architecture_config;
  const auto *arena = g.engine->graph_ram_arena(layer);
  const bool arena_valid =
      arena && g.engine->graph_ram_layer_ready(layer) && arena->base &&
      arena->gate_bytes != 0 && arena->up_bytes != 0 &&
      arena->down_bytes != 0 &&
      (arena->projection_compact
           ? (arena->gate_stride >= arena->gate_bytes &&
              arena->up_stride >= arena->up_bytes &&
              arena->down_stride >= arena->down_bytes)
           : (arena->stride != 0 &&
              arena->stride >= arena->gate_bytes + arena->up_bytes +
                                    arena->down_bytes));
  if (!arena_valid ||
      layer < 0 || layer >= static_cast<int>(g.engine->layers_experts.size()) ||
      g.engine->layers_experts[static_cast<size_t>(layer)].empty()) {
    LOG_WARN(
        "[QWEN4EXP-HOST-NODE] layer=%d declined: complete compatible "
        "DDR arena is required\n",
        layer);
    return nullptr;
  }

  const auto &first = g.engine->layers_experts[static_cast<size_t>(layer)][0];
  if (!first.gate_proj.host_ptr || !first.up_proj.host_ptr ||
      !first.down_proj.host_ptr)
    return nullptr;

  auto node_ctx = std::make_unique<Qwen4ExpCudaHostNodeContext>();
  node_ctx->layer_index = layer;
  const auto split_resident_it = g.split_gpu_expert_ids.find(layer);
  if (split_resident_it != g.split_gpu_expert_ids.end())
    node_ctx->gpu_resident_experts = split_resident_it->second;
  // Reuse the persistent CPU pool already owned by this FullGraph.  The
  // callback runner borrows it; FullGraph remains responsible for teardown.
  node_ctx->shared_cpu_threadpool = g.cpu_threadpool;
  node_ctx->shared_cpu_threads = g.cpu_threads;
  node_ctx->transport_diag =
      env_host_node_transport_diag_enabled(layer) ||
      (layer == 13 && env_layer13_moe_dump_path() != nullptr);
  node_ctx->projection_compact = arena->projection_compact;
  LOG_INFO("[QWEN4EXP-HOST-NODE] layer=%d allocating callback context "
           "arena=%p bytes=%zu layout=%s gate_stride=%zu up_stride=%zu "
           "down_stride=%zu\n",
           layer, static_cast<void *>(arena->base), arena->bytes,
           arena->projection_compact ? "projection-compact" : "interleaved",
           arena->gate_stride, arena->up_stride, arena->down_stride);
  node_ctx->gate_plane = arena->base + arena->gate_offset;
  node_ctx->up_plane = arena->base + arena->up_offset;
  node_ctx->down_plane = arena->base + arena->down_offset;
  node_ctx->layer = {node_ctx->gate_plane,
                     node_ctx->up_plane,
                     node_ctx->down_plane,
                     arena->gate_bytes,
                     arena->up_bytes,
                     arena->down_bytes,
                     static_cast<uint32_t>(first.gate_proj.meta.type),
                     static_cast<uint32_t>(first.up_proj.meta.type),
                     static_cast<uint32_t>(first.down_proj.meta.type)};
  node_ctx->n_experts = c.n_experts;
  node_ctx->d_model = g.D;
  node_ctx->d_ff = c.moe_inter;
  node_ctx->topk = c.topk;
  node_ctx->expert_stride = arena->stride;
  node_ctx->gate_stride = arena->gate_stride;
  node_ctx->up_stride = arena->up_stride;
  node_ctx->down_stride = arena->down_stride;
  node_ctx->input_bytes = static_cast<size_t>(g.D) * sizeof(float);
  node_ctx->output_bytes = node_ctx->input_bytes;
  node_ctx->ids_bytes = static_cast<size_t>(c.topk) * sizeof(int32_t);
  node_ctx->weights_bytes = static_cast<size_t>(c.topk) * sizeof(float);
  node_ctx->host_input = static_cast<float *>(
      alloc_registered_host(g, node_ctx->input_bytes));
  node_ctx->host_output = static_cast<float *>(
      alloc_registered_host(g, node_ctx->output_bytes));
  if (node_ctx->transport_diag) {
    node_ctx->host_output_gpu_check = static_cast<float *>(
        alloc_registered_host(g, node_ctx->output_bytes));
  }
  node_ctx->host_ids = static_cast<int32_t *>(
      alloc_registered_host(g, node_ctx->ids_bytes));
  node_ctx->host_weights = static_cast<float *>(
      alloc_registered_host(g, node_ctx->weights_bytes));
  LOG_INFO("[QWEN4EXP-HOST-NODE] layer=%d callback buffers input=%p output=%p "
           "ids=%p weights=%p\n",
           layer, static_cast<void *>(node_ctx->host_input),
           static_cast<void *>(node_ctx->host_output),
           static_cast<void *>(node_ctx->host_ids),
           static_cast<void *>(node_ctx->host_weights));
  if (!node_ctx->host_input || !node_ctx->host_output ||
      !node_ctx->host_ids || !node_ctx->host_weights ||
      (node_ctx->transport_diag && !node_ctx->host_output_gpu_check))
    return nullptr;

  cudaSetDevice(g.device);
  LOG_INFO("[QWEN4EXP-HOST-NODE] layer=%d creating timing events\n", layer);
  if (cudaEventCreate(&node_ctx->d2h_start) != cudaSuccess ||
      cudaEventCreate(&node_ctx->d2h_end) != cudaSuccess ||
      cudaEventCreate(&node_ctx->h2d_start) != cudaSuccess ||
      cudaEventCreate(&node_ctx->h2d_end) != cudaSuccess ||
      (node_ctx->transport_diag &&
       cudaEventCreate(&node_ctx->post_h2d_check_end) != cudaSuccess))
    return nullptr;

  // Warm the persistent CPU GGML plan before capture. The graph callback must
  // not be the first call that allocates/builds the CPU plan.
  std::fill_n(node_ctx->host_input, g.D, 0.0f);
  std::fill_n(node_ctx->host_output, g.D, 0.0f);
  std::fill_n(node_ctx->host_ids, c.topk, 0);
  std::fill_n(node_ctx->host_weights, c.topk,
              1.0f / static_cast<float>(std::max(1, c.topk)));
  LOG_INFO("[QWEN4EXP-HOST-NODE] layer=%d warming CPU plan type=(%u,%u,%u) "
           "dims=(%d,%d,%d) layout=%s planes=(%p,%p,%p) strides=(%zu,%zu,%zu)\n",
           layer, node_ctx->layer.gate_type, node_ctx->layer.up_type,
           node_ctx->layer.down_type, node_ctx->d_model, node_ctx->d_ff,
           node_ctx->topk,
           node_ctx->projection_compact ? "projection-compact" : "interleaved",
           node_ctx->gate_plane, node_ctx->up_plane, node_ctx->down_plane,
           node_ctx->gate_stride, node_ctx->up_stride,
           node_ctx->down_stride);
  bool warm_ok = false;
  GgmlQwenCpuMoeRunner *warm_runner = nullptr;
  if (node_ctx->projection_compact) {
    warm_runner = &GgmlQwenCpuMoeRunner::instance_for_strided_planes(
            node_ctx->layer, node_ctx->n_experts, node_ctx->d_model,
            node_ctx->d_ff, node_ctx->topk, node_ctx->gate_stride,
            node_ctx->up_stride, node_ctx->down_stride, false);
    warm_ok = warm_runner->run_strided_planes(
        node_ctx->host_input, node_ctx->host_output, node_ctx->layer,
        node_ctx->gate_plane, node_ctx->up_plane, node_ctx->down_plane,
        node_ctx->host_ids, node_ctx->n_experts, node_ctx->host_weights,
        node_ctx->d_model, node_ctx->d_ff, node_ctx->topk,
        node_ctx->gate_stride, node_ctx->up_stride, node_ctx->down_stride);
  } else {
    warm_runner = &GgmlQwenCpuMoeRunner::instance_for_strided(
            node_ctx->layer, node_ctx->n_experts, node_ctx->d_model,
            node_ctx->d_ff, node_ctx->topk, node_ctx->expert_stride, false);
    warm_ok = warm_runner->run_strided(
        node_ctx->host_input, node_ctx->host_output, node_ctx->layer,
        node_ctx->gate_plane, node_ctx->up_plane, node_ctx->down_plane,
        node_ctx->host_ids, node_ctx->n_experts, node_ctx->host_weights,
        node_ctx->d_model, node_ctx->d_ff, node_ctx->topk,
        node_ctx->expert_stride);
  }
  if (!warm_ok) {
    LOG_WARN("[QWEN4EXP-HOST-NODE] layer=%d CPU plan warm-up failed\n", layer);
    return nullptr;
  }
  const char *direct_pool_env = std::getenv("MOESTREAM_QWEN_CPU_DIRECT_POOL");
  const char *direct_fused_env =
      std::getenv("MOESTREAM_QWEN_CPU_DIRECT_FUSED");
  const bool direct_fused_requested =
      direct_fused_env && direct_fused_env[0] == '1';
  const bool direct_pool_requested =
      direct_pool_env && direct_pool_env[0] == '1' &&
      !direct_fused_requested;
  if (warm_runner && direct_pool_requested) {
    // The raw direct executor owns its pre-sized worker team.  Borrowing the
    // FullGraph GGML pool here would tear down that direct signature during
    // warm-up and recreate it from inside the first replay callback, exactly
    // where we need deterministic wake-up behavior.  Prepare the team now,
    // before capture; replay only arms/disarms it.
    warm_runner->prepare_direct_workers();
    node_ctx->cpu_runner = warm_runner;
    LOG_INFO("[QWEN4EXP-HOST-NODE] layer=%d prepared direct CPU team "
             "threads=%d before graph capture\n",
             layer, warm_runner->thread_count());
  } else if (warm_runner && !direct_fused_requested &&
             node_ctx->shared_cpu_threadpool &&
             node_ctx->shared_cpu_threads > 0) {
    warm_runner->bind_external_threadpool(node_ctx->shared_cpu_threadpool,
                                          node_ctx->shared_cpu_threads);
    node_ctx->cpu_runner = warm_runner;
    LOG_INFO("[QWEN4EXP-HOST-NODE] layer=%d borrowing FullGraph CPU pool "
             "%p threads=%d\n",
             layer, static_cast<void *>(node_ctx->shared_cpu_threadpool),
             node_ctx->shared_cpu_threads);
  } else if (warm_runner) {
    node_ctx->cpu_runner = warm_runner;
  }
  // An owned runner may suspend its private native ggml pool between
  // callbacks.  A borrowed FullGraph pool must remain available to ordinary
  // scheduler CPU splits, so it is deliberately never paused here.
  if (warm_runner && !warm_runner->uses_external_threadpool())
    warm_runner->pause_workers();
  LOG_INFO("[QWEN4EXP-HOST-NODE] layer=%d CPU plan warm-up complete\n", layer);

  ggml_tensor *input3 = ggml_reshape_3d(ctx, cur, g.D, 1, g.S);
  if (!input3)
    return nullptr;
  if (input3->type != GGML_TYPE_F32)
    input3 = ggml_cast(ctx, input3, GGML_TYPE_F32);
  input3 = ggml_cont(ctx, input3);
  ggml_tensor *route_ids = ggml_cont(ctx, ids);
  ggml_tensor *route_weights = ggml_cont(ctx, weights);
  if (!input3 || !route_ids || !route_weights)
    return nullptr;

  ggml_tensor *destination = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, g.D, g.S);
  if (!destination)
    return nullptr;
  ggml_set_name(destination,
                ("qwen4exp_host_moe_output_l" + std::to_string(layer))
                    .c_str());
  ggml_tensor *marker = ggml_cpy(ctx, input3, destination);
  if (!marker)
    return nullptr;
  marker->extra = node_ctx.get();
  // These are dependency-only sources. The CUDA bridge copies them to the
  // pinned route buffers before the callback; the ordinary CPY backend never
  // sees this marker because the bridge consumes it first.
  marker->src[2] = route_ids;
  marker->src[3] = route_weights;
  ggml_set_name(marker,
                ("qwen4exp_host_moe_marker_l" + std::to_string(layer))
                    .c_str());
  if (g.sched) {
    ggml_backend_sched_set_tensor_backend(g.sched, destination, g.cuda);
    ggml_backend_sched_set_tensor_backend(g.sched, marker, g.cuda);
  }
  g.host_node_contexts.push_back(std::move(node_ctx));
  LOG_INFO("[QWEN4EXP-HOST-NODE] captured CPU MoE marker for layer=%d "
           "(D=%d topk=%d stride=%zu)\n",
           layer, g.D, c.topk, arena->stride);
  return marker;
}

// Generalizes Option-B's layer-13 staging proof
// (qwen4exp_stage_layer13_before_graph/StageBinding) into a mechanism
// covering every host-resident layer listed in
// engine.qwen4exp_gpu_stage_layers, for real per-token routing instead of
// a forced test set. Where the layer-13 proof issues its
// upload_expert_active_projections_async() call as ordinary host code
// immediately before graph replay -- which only works because it forces a
// fixed route (build_moe's g.stage_route_ids branch) -- this uses the SAME
// CUDA-graph host-node-callback technique qwen4exp_cuda_host_node_callback/
// moestream_qwen4exp_cuda_host_node_enqueue already prove works mid-replay
// with REAL per-token data: a captured D2H(ids)->host-callback(gather)->
// H2D(compact planes) sequence that re-executes on every cudaGraphLaunch
// replay with that token's real routed ids.
//
// Correctness invariant (see also Qwen4ExpGpuStageContext's own comment):
// the callback writes routed expert ids[k]'s bytes into destination slot
// k, for k in [0,topk), in the SAME order `ids` itself lists them. Because
// build_moe's combine weights (`weights = ggml_get_rows(ctx, probs3,
// ids)`) are ALSO indexed by that same order, a MUL_MAT_ID call against
// these compact destinations can use a CONSTANT [0..topk-1] ids tensor --
// slot k IS real expert ids[k] by construction -- with no per-token id
// remap needed. Do NOT reassign `ids` itself here the way the layer-13
// proof does (build_moe's `ids = g.stage_route_ids` line): that would
// desync the real combine weights from the real routing, which the proof
// gets away with only because it forces both consistently to the same
// fake set.
//
// Decode (S==1) only -- see engine.qwen4exp_gpu_stage_enabled's own
// comment for why prefill needs a different, batch-proportional design.
// Returns the index into g.gpu_stage_bindings on success, -1 on any
// failure (caller falls through to the next dispatch strategy for this
// layer -- CPU-offload if enabled, otherwise today's plain CPU path;
// never a hard error for the whole graph build).
int build_gpu_stage_node(FullGraph &g, ggml_context *ctx, ggml_tensor *ids,
                         int layer) {
  if (!g.decode_mode || g.S != 1 || !g.engine ||
      !g.engine->qwen4exp_gpu_stage_layer_enabled(layer))
    return -1;

  const auto &c = g.engine->model_file.architecture_config;
  const auto *arena = g.engine->graph_ram_arena(layer);
  const bool arena_valid =
      arena && g.engine->graph_ram_layer_ready(layer) && arena->base &&
      arena->gate_bytes != 0 && arena->up_bytes != 0 &&
      arena->down_bytes != 0 &&
      (arena->projection_compact
           ? (arena->gate_stride >= arena->gate_bytes &&
              arena->up_stride >= arena->up_bytes &&
              arena->down_stride >= arena->down_bytes)
           : (arena->stride != 0 &&
              arena->stride >= arena->gate_bytes + arena->up_bytes +
                                    arena->down_bytes));
  if (!arena_valid || layer < 0 ||
      layer >= static_cast<int>(g.engine->layers_experts.size()) ||
      g.engine->layers_experts[static_cast<size_t>(layer)].empty()) {
    LOG_WARN("[QWEN4EXP-GPU-STAGE] layer=%d declined: complete compatible "
             "DDR arena is required\n",
             layer);
    return -1;
  }
  const auto &first = g.engine->layers_experts[static_cast<size_t>(layer)][0];
  if (!first.gate_proj.host_ptr || !first.up_proj.host_ptr ||
      !first.down_proj.host_ptr)
    return -1;

  // Compact destination tensors: [cols, rows, topk] -- same shape
  // convention build_split_gpu_tensor already uses for its own compact GPU
  // tensors, sized for `topk` (constant per token) instead of a static
  // hot-set count. Bump-allocated from the SAME device_expert_buffer arena
  // device-resident layers use (see device_expert_bytes()'s staging-bytes
  // block for the matching capacity accounting) -- never copied into here
  // directly; only the per-token host-node callback below ever writes
  // their bytes.
  auto make_dst = [&](const char *stem,
                      ggml_type type) -> ggml_tensor * {
    const GGUFTensor *m = model_tensor(
        *g.engine, "blk." + std::to_string(layer) + "." + stem);
    if (!m || m->shape.size() < 3)
      return nullptr;
    std::vector<int64_t> shape = {static_cast<int64_t>(m->shape[0]),
                                  static_cast<int64_t>(m->shape[1]),
                                  static_cast<int64_t>(c.topk)};
    ggml_tensor *t = new_tensor(ctx, type, shape);
    if (!t)
      return nullptr;
    void *addr = nullptr;
    bool new_allocation = false;
    const std::string key =
        "gpu_stage:" + std::to_string(layer) + ":" + stem;
    if (!phase2_bind_or_allocate_expert(g, t, key, &addr, &new_allocation))
      return nullptr;
    ggml_set_name(t, (std::string("blk.") + std::to_string(layer) + "." +
                      stem + ".gpu_stage")
                         .c_str());
    return t;
  };

  FullGraph::GpuStageBinding binding;
  binding.gate_dst = make_dst(
      "ffn_gate_exps.weight", static_cast<ggml_type>(first.gate_proj.meta.type));
  binding.up_dst = make_dst(
      "ffn_up_exps.weight", static_cast<ggml_type>(first.up_proj.meta.type));
  binding.down_dst = make_dst(
      "ffn_down_exps.weight", static_cast<ggml_type>(first.down_proj.meta.type));
  if (!binding.gate_dst || !binding.up_dst || !binding.down_dst) {
    LOG_WARN("[QWEN4EXP-GPU-STAGE] layer=%d declined: compact device "
             "destination allocation failed (VRAM budget?)\n",
             layer);
    return -1;
  }

  binding.ids_dispatch_const = host_input(
      g, ctx, GGML_TYPE_I32, {c.topk, g.S},
      static_cast<size_t>(c.topk) * static_cast<size_t>(g.S) * sizeof(int32_t),
      (std::string("qwen_gpu_stage_ids_l") + std::to_string(layer)).c_str());
  if (!binding.ids_dispatch_const)
    return -1;
  {
    auto *ids_data = static_cast<int32_t *>(binding.ids_dispatch_const->data);
    for (int s = 0; s < g.S; ++s)
      for (int k = 0; k < c.topk; ++k)
        ids_data[s * c.topk + k] = k;
  }

  auto node_ctx = std::make_unique<Qwen4ExpGpuStageContext>();
  node_ctx->layer_index = layer;
  node_ctx->n_experts = c.n_experts;
  node_ctx->topk = c.topk;
  node_ctx->gate_plane = arena->base + arena->gate_offset;
  node_ctx->up_plane = arena->base + arena->up_offset;
  node_ctx->down_plane = arena->base + arena->down_offset;
  node_ctx->gate_stride = arena->gate_stride;
  node_ctx->up_stride = arena->up_stride;
  node_ctx->down_stride = arena->down_stride;
  node_ctx->gate_bytes = arena->gate_bytes;
  node_ctx->up_bytes = arena->up_bytes;
  node_ctx->down_bytes = arena->down_bytes;
  node_ctx->ids_bytes = static_cast<size_t>(c.topk) * sizeof(int32_t);
  node_ctx->host_ids = static_cast<int32_t *>(
      alloc_registered_host(g, node_ctx->ids_bytes));
  node_ctx->host_gate_bounce = static_cast<uint8_t *>(alloc_registered_host(
      g, static_cast<size_t>(c.topk) * node_ctx->gate_bytes));
  node_ctx->host_up_bounce = static_cast<uint8_t *>(alloc_registered_host(
      g, static_cast<size_t>(c.topk) * node_ctx->up_bytes));
  node_ctx->host_down_bounce = static_cast<uint8_t *>(alloc_registered_host(
      g, static_cast<size_t>(c.topk) * node_ctx->down_bytes));
  if (!node_ctx->host_ids || !node_ctx->host_gate_bounce ||
      !node_ctx->host_up_bounce || !node_ctx->host_down_bounce)
    return -1;
  node_ctx->device_gate_dst = binding.gate_dst->data;
  node_ctx->device_up_dst = binding.up_dst->data;
  node_ctx->device_down_dst = binding.down_dst->data;

  cudaSetDevice(g.device);
  if (cudaEventCreate(&node_ctx->d2h_start) != cudaSuccess ||
      cudaEventCreate(&node_ctx->d2h_end) != cudaSuccess ||
      cudaEventCreate(&node_ctx->h2d_start) != cudaSuccess ||
      cudaEventCreate(&node_ctx->h2d_end) != cudaSuccess)
    return -1;
  // Warm the pinned buffers before capture -- same discipline
  // build_qwen4exp_host_node follows, so capture never observes
  // first-touch page faults.
  std::fill_n(node_ctx->host_ids, c.topk, 0);
  std::memset(node_ctx->host_gate_bounce, 0,
             static_cast<size_t>(c.topk) * node_ctx->gate_bytes);
  std::memset(node_ctx->host_up_bounce, 0,
             static_cast<size_t>(c.topk) * node_ctx->up_bytes);
  std::memset(node_ctx->host_down_bounce, 0,
             static_cast<size_t>(c.topk) * node_ctx->down_bytes);

  // A tiny dummy leaf (same type/shape as gate_dst) satisfies ggml_cpy's
  // own shape/type validation for the marker's formal "source" -- the real
  // work happens entirely inside the enqueue hijack
  // (qwen4exp_cuda_gpu_stage_node_enqueue), never via ordinary CPY
  // execution. `ids` (real routing, ggml_cont'd) is attached as an EXTRA
  // dependency source purely for D2H/ordering purposes, matching the
  // existing host-node marker's own src[2]/src[3] convention.
  ggml_tensor *dummy_src = ggml_dup_tensor(ctx, binding.gate_dst);
  if (!dummy_src)
    return -1;
  ggml_tensor *route_ids = ggml_cont(ctx, ids);
  if (!route_ids)
    return -1;
  ggml_tensor *marker = ggml_cpy(ctx, dummy_src, binding.gate_dst);
  if (!marker)
    return -1;
  marker->extra = node_ctx.get();
  marker->src[2] = route_ids;
  ggml_set_name(marker, (std::string("qwen4exp_gpu_stage_marker_l") +
                        std::to_string(layer))
                           .c_str());
  if (g.sched)
    ggml_backend_sched_set_tensor_backend(g.sched, marker, g.cuda);

  // Force gate/up/down's own MUL_MAT_ID consumers to depend on the marker,
  // so ggml's topological sort schedules the D2H/callback/H2D sequence
  // strictly before anything reads gate_dst/up_dst/down_dst's contents --
  // dst/up/down_dst are plain bump-allocated leaves with no other src[]
  // edge to the marker, so this dependency must be explicit (ggml only
  // tracks producer/consumer ordering through node src[] chains, never
  // through incidental construction order or shared backing memory).
  binding.gpu_stage_marker_dep = marker;

  binding.layer = layer;
  g.gpu_stage_contexts.push_back(std::move(node_ctx));
  g.gpu_stage_bindings.push_back(binding);
  LOG_INFO("[QWEN4EXP-GPU-STAGE] captured GPU staging marker for layer=%d "
           "(D=%d topk=%d)\n",
           layer, g.D, c.topk);
  return static_cast<int>(g.gpu_stage_bindings.size()) - 1;
}

// Builds the two small I32/F32 constant lookup tables a genuinely split
// layer's mul_mat_id pair needs, once, at graph-build time (never
// refreshed per decode step -- these encode a STATIC expert->device
// assignment, exactly like device_expert_layers already is). Shapes are
// [1, n_experts, 1] to satisfy ggml_get_rows' a->ne[2]==b->ne[1] contract
// against `ids`' own [topk, S] shape with S==1 (decode-only, enforced by
// this function's only caller). remap_gpu[e] = local index in [0,M) for a
// hot expert and -1 for a cold one; remap_cpu[e] = e for a cold expert and
// -1 for a hot one. GGML MUL_MAT_ID treats negative ids as inactive route
// slots, so each backend skips the other backend's routes before the matmul.
// mask_gpu/mask_cpu remain as a defensive numerical guard at the merge.
struct ExpertSplitTables {
  ggml_tensor *remap_gpu = nullptr;
  ggml_tensor *remap_cpu = nullptr;
  ggml_tensor *mask_gpu = nullptr;
  ggml_tensor *mask_cpu = nullptr;
};

ExpertSplitTables build_expert_split_tables(FullGraph &g, ggml_context *ctx,
                                            int n_experts,
                                            const std::vector<int> &gpu_ids) {
  ExpertSplitTables r;
  if (n_experts <= 0)
    return r;
  std::vector<int32_t> remap_gpu(static_cast<size_t>(n_experts), -1);
  std::vector<int32_t> remap_cpu(static_cast<size_t>(n_experts), 0);
  std::vector<float> mgpu(static_cast<size_t>(n_experts), 0.0f);
  std::vector<float> mcpu(static_cast<size_t>(n_experts), 1.0f);
  for (int e = 0; e < n_experts; ++e)
    remap_cpu[static_cast<size_t>(e)] = e;
  for (size_t k = 0; k < gpu_ids.size(); ++k) {
    const int e = gpu_ids[k];
    if (e < 0 || e >= n_experts)
      continue;
    remap_gpu[static_cast<size_t>(e)] = static_cast<int32_t>(k);
    remap_cpu[static_cast<size_t>(e)] = -1;
    mgpu[static_cast<size_t>(e)] = 1.0f;
    mcpu[static_cast<size_t>(e)] = 0.0f;
  }
  const std::vector<int64_t> shape = {1, n_experts, 1};
  r.remap_gpu = host_input(g, ctx, GGML_TYPE_I32, shape,
                           remap_gpu.size() * sizeof(int32_t),
                           "split_remap_gpu");
  r.remap_cpu = host_input(g, ctx, GGML_TYPE_I32, shape,
                           remap_cpu.size() * sizeof(int32_t),
                           "split_remap_cpu");
  r.mask_gpu = host_input(g, ctx, GGML_TYPE_F32, shape,
                          mgpu.size() * sizeof(float), "split_mask_gpu");
  r.mask_cpu = host_input(g, ctx, GGML_TYPE_F32, shape,
                          mcpu.size() * sizeof(float), "split_mask_cpu");
  if (!r.remap_gpu || !r.remap_cpu || !r.mask_gpu || !r.mask_cpu)
    return {};
  // host_input() binds these to already-allocated host memory (->data is
  // live immediately, same idiom this file already uses for g.positions);
  // written once here since a split layer's expert->device assignment
  // never changes for this graph's lifetime.
  std::memcpy(r.remap_gpu->data, remap_gpu.data(),
              remap_gpu.size() * sizeof(int32_t));
  std::memcpy(r.remap_cpu->data, remap_cpu.data(),
              remap_cpu.size() * sizeof(int32_t));
  std::memcpy(r.mask_gpu->data, mgpu.data(), mgpu.size() * sizeof(float));
  std::memcpy(r.mask_cpu->data, mcpu.data(), mcpu.size() * sizeof(float));
  return r;
}

ggml_tensor *build_moe(FullGraph &g, ggml_context *ctx, ggml_tensor *cur,
                       int layer, ggml_tensor *router,
                       ggml_tensor *selection_bias, ggml_tensor *gate_exps,
                       ggml_tensor *up_exps, ggml_tensor *down_exps,
                       ggml_tensor *gate_exps_gpu = nullptr,
                       ggml_tensor *up_exps_gpu = nullptr,
                       ggml_tensor *down_exps_gpu = nullptr,
                       const std::vector<int> *split_gpu_ids = nullptr,
                       ggml_tensor *gate_up_exps = nullptr,
                       // Phase 1 partial-layer instrumentation only: when
                       // non-null, receives this call's routing `ids`
                       // tensor so the caller can mark it as a prefill
                       // graph output for a real per-prompt routing
                       // readback. Never used for anything decode does.
                       ggml_tensor **ids_out = nullptr,
                       // Same idea, but captures the raw post-softmax
                       // router distribution (BEFORE selection_bias) so a
                       // diagnostic can check how peaked/confident real
                       // routing actually is -- see the "near-uniform =>
                       // noise" precedent already used for the LM head in
                       // generation_loop.hpp.
                       ggml_tensor **probs_out = nullptr,
                       // Phase 2 prefix builds only: compute the router
                       // (logits/probs/ids/weights, already added to the
                       // graph via the ggml_build_forward_expand below)
                       // and return nullptr immediately, WITHOUT
                       // constructing any dispatch/FFN nodes at all. Lets
                       // the prefix graph capture ids_out for the host
                       // decision without paying for (or even
                       // constructing) the big expert matmuls it will
                       // never use -- the real dispatch happens once,
                       // for real, in whichever suffix graph gets chosen.
                       bool routing_only = false) {
  const auto &c = g.engine->model_file.architecture_config;
  const bool layer13_moe_diag =
      layer == 13 && env_layer13_moe_dump_path() != nullptr;
  if (layer13_moe_diag)
    g.layer13_moe_diag_enabled = true;
  const bool layer0_diag =
      layer == 0 && (env_layer0_iq2_diag_enabled() ||
                     env_layer0_split_dump_path() != nullptr);
  if (layer0_diag) {
    g.layer0_diag_enabled = true;
    LOG_INFO("[QWEN4EXP-LAYER0-DIAG] graph=%p decode=%d S=%d "
             "route_path_candidate=%s\n",
             static_cast<void *>(&g), g.decode_mode ? 1 : 0, g.S,
             (g.decode_mode && env_qwen4exp_host_node_enabled() &&
              g.engine->qwen4exp_cpu_offload_layer_enabled(layer))
                 ? "host-node-CPU"
                 : "ordinary-ggml-dispatch");
    log_layer0_diag_tensor("gate_weight", gate_exps);
    log_layer0_diag_tensor("up_weight", up_exps);
    log_layer0_diag_tensor("down_weight", down_exps);
    log_layer0_diag_tensor("gate_up_weight", gate_up_exps);
  }
  ggml_tensor *logits = ggml_mul_mat(ctx, router, cur);
  ggml_tensor *probs = ggml_soft_max(ctx, logits);
  if (probs_out)
    *probs_out = probs;
  ggml_tensor *selection =
      selection_bias ? ggml_add(ctx, probs, selection_bias) : probs;
  ggml_tensor *ids = ggml_argsort_top_k(ctx, selection, c.topk);
  if (g.decode_mode && g.stage_enabled && layer == g.stage_layer) {
    if (!g.stage_route_ids) {
      g.stage_route_ids = host_input(
          g, ctx, GGML_TYPE_I32, {c.topk, g.S},
          static_cast<size_t>(c.topk) * g.S * sizeof(int32_t),
          "qwen_layer13_stage_route_ids");
      if (g.stage_route_ids) {
        auto *forced = static_cast<int32_t *>(g.stage_route_ids->data);
        for (int s = 0; s < g.S; ++s) {
          for (int k = 0; k < c.topk; ++k) {
            const int index = s * c.topk + k;
            forced[index] = g.stage_experts.empty()
                                ? g.stage_expert
                                : g.stage_experts[k % g.stage_experts.size()];
          }
        }
      }
    }
    if (g.stage_route_ids) {
      ids = g.stage_route_ids;
      if (!g.stage_route_logged) {
        std::ostringstream experts;
        for (size_t i = 0; i < g.stage_experts.size(); ++i)
          experts << (i == 0 ? "" : ",") << g.stage_experts[i];
        LOG_INFO("[OPTION-B-STAGE] forcing Layer %d route IDs to cold "
                 "experts [%s] for the %zu-slot proof\n",
                 g.stage_layer, experts.str().c_str(),
                 g.stage_experts.size());
        g.stage_route_logged = true;
      }
    }
  }
  if (ids_out) {
    // ggml_argsort_top_k is argsort (full n_experts-wide) + a VIEW that
    // only shrinks ne[0] to topk -- the view's row stride (nb[1]) stays
    // at the ORIGINAL n_experts-wide stride, not topk. A single flat
    // ggml_backend_tensor_get over S>1 rows therefore does not read S
    // independent per-token top-k slices; it reads a contiguous run of
    // (topk*S) values out of THIS VIEW's ne[0]=topk logical columns
    // reinterpreted at the wrong stride, which for S*topk <= n_experts
    // (true for every case this repo exercises, e.g. topk=10, S<=51)
    // silently degenerates into reading token 0's OWN top-(topk*S)
    // ranking sliced into S fake "per-token" chunks -- found via real
    // hardware testing, 2026-09-18 (every reported PREFILL coverage/
    // routing-divergence number computed before this fix is invalid;
    // see [[qwen4exp-legacy-vs-fullgraph-routing-divergence]] and
    // [[qwen4exp-partial-layer-phase2-implemented]]). DECODE (S==1) was
    // never affected -- there is only one row, so the flat read already
    // lands on real data with no other row to conflate it with.
    // ggml_cont() forces a real, densely-strided [topk, S] copy so any
    // host-side flat readback of *ids_out is safe.
    *ids_out = ggml_cont(ctx, ids);
  }
  ggml_tensor *probs3 = ggml_reshape_3d(ctx, probs, 1, c.n_experts, g.S);
  ggml_tensor *weights = ggml_get_rows(ctx, probs3, ids);
  ggml_tensor *weights2 = ggml_reshape_2d(ctx, weights, c.topk, g.S);
  if (c.expert_weights_norm) {
    ggml_tensor *sum = ggml_sum_rows(ctx, weights2);
    sum = ggml_clamp(ctx, sum, 6.103515625e-5f, INFINITY);
    weights2 = ggml_div(ctx, weights2, sum);
  }
  if (c.expert_weights_scale != 0.0f && c.expert_weights_scale != 1.0f)
    weights2 = ggml_scale(ctx, weights2, c.expert_weights_scale);
  weights = ggml_reshape_3d(ctx, weights2, 1, c.topk, g.S);
  if (layer13_moe_diag) {
    // These are diagnostic snapshots only. The argsort result is a strided
    // view and ordinary intermediate buffers may be allocator-reused after
    // the graph retires, so retain the logical values in stable host inputs.
    ggml_tensor *diag_input = ggml_reshape_3d(ctx, cur, g.D, 1, g.S);
    if (diag_input && diag_input->type != GGML_TYPE_F32)
      diag_input = ggml_cast(ctx, diag_input, GGML_TYPE_F32);
    if (diag_input)
      diag_input = ggml_cont(ctx, diag_input);
    g.layer13_moe_input_diag = layer13_moe_diag_snapshot(
        g, ctx, diag_input, GGML_TYPE_F32,
        {g.D, g.S}, "qwen_layer13_moe_input_diag");
    g.layer13_moe_ids_diag = layer13_moe_diag_snapshot(
        g, ctx, ids, GGML_TYPE_I32,
        {c.topk, g.S}, "qwen_layer13_moe_ids_diag");
    g.layer13_moe_weights_diag = layer13_moe_diag_snapshot(
        g, ctx, weights, GGML_TYPE_F32,
        {1, c.topk, g.S}, "qwen_layer13_moe_weights_diag");
    if (!g.layer13_moe_input_diag || !g.layer13_moe_ids_diag ||
        !g.layer13_moe_weights_diag)
      return nullptr;
  }
  ggml_build_forward_expand(g.graph, weights);
  if (routing_only) {
    // Phase 2 prefix graph: caller only wanted ids_out/probs_out for the
    // per-token host dispatch decision. Stop here -- no FFN/dispatch
    // nodes constructed, so this layer's real expert compute happens
    // exactly once, in whichever suffix graph actually gets chosen.
    return nullptr;
  }

  // Generalized GPU staging for host-resident decode layers (see
  // engine.qwen4exp_gpu_stage_enabled/_layers, build_gpu_stage_node).
  // Highest precedence among host-resident dispatch strategies: takes over
  // before the CPU-offload host-node marker and the static split_gpu_ids
  // branch get a chance, since staging fully replaces what either would do
  // for this layer's compute. `weights` above was already computed from
  // the real, unmodified `ids` -- staging must NOT reassign `ids` itself
  // (see build_gpu_stage_node's own comment on why that would desync the
  // real combine weights from real routing).
  if (g.decode_mode && g.S == 1 && g.engine &&
      g.engine->qwen4exp_gpu_stage_layer_enabled(layer)) {
    const int stage_idx = build_gpu_stage_node(g, ctx, ids, layer);
    if (stage_idx >= 0) {
      const FullGraph::GpuStageBinding &sb = g.gpu_stage_bindings[
          static_cast<size_t>(stage_idx)];
      ggml_tensor *input3_stage = ggml_reshape_3d(ctx, cur, g.D, 1, g.S);
      ggml_tensor *up_s = ggml_mul_mat_id(ctx, sb.up_dst, input3_stage,
                                          sb.ids_dispatch_const);
      ggml_tensor *gate_s = ggml_mul_mat_id(ctx, sb.gate_dst, input3_stage,
                                            sb.ids_dispatch_const);
      if (up_s && gate_s && sb.gpu_stage_marker_dep) {
        // Force ordering: the marker's D2H/callback/H2D sequence must
        // complete before either matmul reads gate_dst/up_dst -- see
        // GpuStageBinding::gpu_stage_marker_dep's own comment.
        up_s->src[3] = sb.gpu_stage_marker_dep;
        gate_s->src[3] = sb.gpu_stage_marker_dep;
      }
      ggml_tensor *act_s = (up_s && gate_s)
                              ? ggml_swiglu_split(ctx, gate_s, up_s)
                              : nullptr;
      ggml_tensor *down_s =
          act_s ? ggml_mul_mat_id(ctx, sb.down_dst, act_s,
                                  sb.ids_dispatch_const)
                : nullptr;
      if (down_s && sb.gpu_stage_marker_dep)
        down_s->src[3] = sb.gpu_stage_marker_dep;
      if (down_s) {
        // Same weighted-sum-over-topk reduction as the shared tail below
        // (out = view+add loop over `experts`) -- duplicated here rather
        // than falling through to it, matching build_qwen4exp_host_node's
        // own self-contained-early-return convention for a dispatch
        // strategy that fully replaces the ordinary mul_mat_id chain.
        ggml_tensor *weighted_s = ggml_mul(ctx, down_s, weights);
        ggml_tensor *out_s =
            ggml_view_2d(ctx, weighted_s, g.D, g.S, weighted_s->nb[2], 0);
        out_s = ggml_cont(ctx, out_s);
        for (int k = 1; k < c.topk; ++k) {
          ggml_tensor *e = ggml_view_2d(
              ctx, weighted_s, g.D, g.S, weighted_s->nb[2],
              static_cast<size_t>(k) * weighted_s->nb[1]);
          out_s = ggml_add(ctx, out_s, e);
        }
        (void)layer;
        return out_s;
      }
    }
    LOG_WARN("[QWEN4EXP-GPU-STAGE] layer=%d falling through to the next "
             "dispatch strategy (CPU-offload if enabled, else plain CPU)\n",
             layer);
  }

  // The host callback is a single-token strided CPU contract.  Prefill must
  // keep the ordinary batched CPU/GGML MoE path so it can seed recurrent/KV
  // state; entering this branch for S>1 returns a null marker and leaves the
  // graph builder with no valid routed result.
  // A static per-expert split has a GPU branch and a CPU branch constructed
  // below.  Do not let the whole-layer host marker intercept it: doing so
  // silently discards the GPU hot set and runs every routed expert in the
  // callback, which is both slower and contrary to Option B's contract.
  const bool split_planned_for_layer =
      split_gpu_ids && !split_gpu_ids->empty() && gate_exps_gpu &&
      up_exps_gpu && down_exps_gpu;
  if (g.decode_mode && env_qwen4exp_host_node_enabled() &&
      g.engine->qwen4exp_cpu_offload_layer_enabled(layer) &&
      !split_planned_for_layer) {
    // The marker returns the already weighted/reduced routed result, so it
    // intentionally bypasses the ordinary three MUL_MAT_ID + SwiGLU chain.
    // A failure is fatal to this opt-in graph build; falling through to the
    // scheduler CPU path here would make a supposedly unified capture
    // silently contain the old scpu work again.
    ggml_tensor *host_node =
        build_qwen4exp_host_node(g, ctx, cur, ids, weights, layer);
    if (!host_node)
      return nullptr;
    if (layer0_diag)
      LOG_INFO("[QWEN4EXP-LAYER0-DIAG] path=host-node-CPU "
               "compute=CPU-GGML-MUL_MAT_ID (no CUDA IQ2 dot)\n");
    return host_node;
  }
  if (g.decode_mode && env_qwen4exp_host_node_enabled() &&
      g.engine->qwen4exp_cpu_offload_layer_enabled(layer) &&
      split_planned_for_layer && qwen4exp_host_node_timing_enabled()) {
    LOG_INFO("[QWEN4EXP-HOST-NODE] layer=%d bypassed whole-layer marker "
             "because a static GPU hot/cold split is planned\n",
             layer);
  }

  ggml_tensor *input3 = ggml_reshape_3d(ctx, cur, g.D, 1, g.S);
  if (layer13_moe_diag && input3) {
    input3 = ggml_cont(ctx, input3);
  }
  ggml_tensor *experts = nullptr;

  // Genuine per-expert split path: route IDs are remapped independently for
  // each backend. Negative IDs are inactive slots understood by the GGML
  // MUL_MAT_ID kernels, so each branch computes only its assigned routes.
  // The masks remain as a defensive numerical guard at the merge. Every
  // other layer/config takes the unchanged single-mul_mat_id path below.
  if (split_gpu_ids && !split_gpu_ids->empty() && gate_exps_gpu &&
      up_exps_gpu && down_exps_gpu) {
    ExpertSplitTables tab =
        build_expert_split_tables(g, ctx, c.n_experts, *split_gpu_ids);
    if (tab.remap_gpu && tab.remap_cpu && tab.mask_gpu && tab.mask_cpu) {
      ggml_tensor *ids_gpu_raw = ggml_get_rows(ctx, tab.remap_gpu, ids);
      ggml_tensor *ids_gpu = ggml_reshape_2d(ctx, ids_gpu_raw, c.topk, g.S);
      ggml_tensor *ids_cpu_raw = ggml_get_rows(ctx, tab.remap_cpu, ids);
      ggml_tensor *ids_cpu = ggml_reshape_2d(ctx, ids_cpu_raw, c.topk, g.S);
      ggml_tensor *weights_gpu =
          ggml_mul(ctx, weights, ggml_get_rows(ctx, tab.mask_gpu, ids));
      ggml_tensor *weights_cpu =
          ggml_mul(ctx, weights, ggml_get_rows(ctx, tab.mask_cpu, ids));

      ggml_tensor *up_g = ggml_mul_mat_id(ctx, up_exps_gpu, input3, ids_gpu);
      ggml_tensor *gate_g = ggml_mul_mat_id(ctx, gate_exps_gpu, input3, ids_gpu);
      ggml_tensor *act_g = ggml_swiglu_split(ctx, gate_g, up_g);
      ggml_tensor *experts_gpu =
          ggml_mul_mat_id(ctx, down_exps_gpu, act_g, ids_gpu);

      ggml_tensor *up_c = ggml_mul_mat_id(ctx, up_exps, input3, ids_cpu);
      ggml_tensor *gate_c = ggml_mul_mat_id(ctx, gate_exps, input3, ids_cpu);
      ggml_tensor *act_c = ggml_swiglu_split(ctx, gate_c, up_c);
      ggml_tensor *experts_cpu =
          ggml_mul_mat_id(ctx, down_exps, act_c, ids_cpu);

      experts = ggml_add(ctx, ggml_mul(ctx, experts_gpu, weights_gpu),
                         ggml_mul(ctx, experts_cpu, weights_cpu));
    }
  }

  if (!experts) {
    ggml_tensor *up = nullptr;
    ggml_tensor *gate = nullptr;
    ggml_tensor *gate_up = nullptr; // only set on the fused path
    if (gate_up_exps) {
      // Bee's Qwen graph and the CPU runner use one routed matmul for the
      // adjacent gate+up planes, then split the result into two views.  The
      // tensor is a zero-copy view over the shared DDR arena for CPU layers;
      // GPU layers may use the same representation when available.
      gate_up = ggml_mul_mat_id(ctx, gate_up_exps, input3, ids);
      if (gate_up) {
        gate = ggml_view_3d(ctx, gate_up, gate_up->ne[0] / 2,
                            gate_up->ne[1], gate_up->ne[2], gate_up->nb[1],
                            gate_up->nb[2], 0);
        up = ggml_view_3d(ctx, gate_up, gate_up->ne[0] / 2,
                          gate_up->ne[1], gate_up->ne[2], gate_up->nb[1],
                          gate_up->nb[2],
                          static_cast<size_t>(gate_up->ne[0] / 2) *
                              gate_up->nb[0]);
      }
    } else {
      // Separate-tensor fallback for layers whose source arena cannot be
      // presented as adjacent, type-matched gate+up planes.
      up = ggml_mul_mat_id(ctx, up_exps, input3, ids);
      gate = ggml_mul_mat_id(ctx, gate_exps, input3, ids);
    }
    ggml_tensor *act = ggml_swiglu_split(ctx, gate, up);
    ggml_tensor *down = ggml_mul_mat_id(ctx, down_exps, act, ids);
    experts = ggml_mul(ctx, down, weights);
    if (layer0_diag) {
      g.layer0_diag_gate = gate;
      g.layer0_diag_up = up;
      g.layer0_diag_act = act;
      g.layer0_diag_down = down;
      g.layer0_diag_gate_up = gate_up;
      LOG_INFO("[QWEN4EXP-LAYER0-DIAG] path=ordinary-ggml-dispatch "
               "representation=%s\n",
               gate_up ? "merged-gate-up" : "separate-gate-up");
      log_layer0_diag_tensor("gate_output", gate);
      log_layer0_diag_tensor("up_output", up);
      log_layer0_diag_tensor("down_output", down);
    }
    if (g.layer_timing_enabled) {
      // See FullGraph::layer_timing_layer_of's own comment -- these are
      // the actual CPU-dispatched op result tensors when this layer is
      // CPU-resident. On the fused path, `gate`/`up` are VIEWS of
      // `gate_up` (near-zero real compute, but the eval callback still
      // fires an ask=true/ask=false pair around them since they inherit
      // gate_up's CPU backend assignment) -- tag the real compute node
      // (`gate_up`) instead of double-counting the views, so the gap
      // count reflects genuine dispatch handoffs, not view bookkeeping.
      if (gate_up)
        g.layer_timing_layer_of[gate_up] = {layer, "moe_gate_up_fused"};
      else {
        if (up)
          g.layer_timing_layer_of[up] = {layer, "moe_up"};
        if (gate)
          g.layer_timing_layer_of[gate] = {layer, "moe_gate"};
      }
      g.layer_timing_layer_of[act] = {layer, "moe_swiglu"};
      g.layer_timing_layer_of[down] = {layer, "moe_down"};
    }
  }

  ggml_tensor *out = ggml_view_2d(ctx, experts, g.D, g.S, experts->nb[2], 0);
  out = ggml_cont(ctx, out);
  for (int k = 1; k < c.topk; ++k) {
    ggml_tensor *e = ggml_view_2d(ctx, experts, g.D, g.S, experts->nb[2],
                                  static_cast<size_t>(k) * experts->nb[1]);
    out = ggml_add(ctx, out, e);
  }
  if (layer13_moe_diag)
    g.layer13_moe_diag = layer13_moe_diag_snapshot(
        g, ctx, out, GGML_TYPE_F32, {g.D, g.S},
        "qwen_layer13_moe_output_diag");
  (void)layer;
  return out;
}

// Phase 1 "all-or-nothing" partial-layer GPU dispatch. Deliberately NOT a
// generalization of build_moe's existing split_gpu_ids branch above: that
// path computes gate/up/down mul_mat_id TWICE per token (once per device,
// full topk width each time, masked and merged) -- a real, measured,
// ~20% decode regression at every budget tested regardless of split size,
// because the redundant branch's FLOPs are not actually skipped by a
// negative-id route, just zeroed after the fact (see
// [[qwen4exp-expert-vram-split-regression]]). This function instead does
// the single, un-doubled dispatch the name promises: ONE mul_mat_id triple
// over a compact top-N-expert tensor, only ever called for the one layer
// (engine->qwen4exp_partial_layer) that a whole prefill pass's real
// routing already confirmed stays inside that top-N set (see
// qwen4exp_full_graph_run's coverage-gated commit). An id that falls
// outside the compact set anyway (decode routing something prefill never
// saw) maps to -1 via `remap_gpu` and is silently dropped by
// GGML_MUL_MAT_ID's own negative-id contract, i.e. that expert's
// contribution to the weighted sum goes missing for that token -- a real,
// bounded, MEASURED (not swept under the rug) accuracy risk, not a
// design flaw to fix here; see env_partial_layer_min_coverage's own
// comment for the knob that trades commit-rate against this risk.
ggml_tensor *build_moe_partial_gpu(FullGraph &g, ggml_context *ctx,
                                   ggml_tensor *cur, int layer,
                                   ggml_tensor *router,
                                   ggml_tensor *selection_bias,
                                   ggml_tensor *gate_gpu, ggml_tensor *up_gpu,
                                   ggml_tensor *down_gpu,
                                   ggml_tensor *remap_gpu) {
  const auto &c = g.engine->model_file.architecture_config;
  ggml_tensor *logits = ggml_mul_mat(ctx, router, cur);
  ggml_tensor *probs = ggml_soft_max(ctx, logits);
  ggml_tensor *selection =
      selection_bias ? ggml_add(ctx, probs, selection_bias) : probs;
  ggml_tensor *ids = ggml_argsort_top_k(ctx, selection, c.topk);
  ggml_tensor *probs3 = ggml_reshape_3d(ctx, probs, 1, c.n_experts, g.S);
  ggml_tensor *weights = ggml_get_rows(ctx, probs3, ids);
  ggml_tensor *weights2 = ggml_reshape_2d(ctx, weights, c.topk, g.S);
  if (c.expert_weights_norm) {
    ggml_tensor *sum = ggml_sum_rows(ctx, weights2);
    sum = ggml_clamp(ctx, sum, 6.103515625e-5f, INFINITY);
    weights2 = ggml_div(ctx, weights2, sum);
  }
  if (c.expert_weights_scale != 0.0f && c.expert_weights_scale != 1.0f)
    weights2 = ggml_scale(ctx, weights2, c.expert_weights_scale);
  weights = ggml_reshape_3d(ctx, weights2, 1, c.topk, g.S);
  ggml_build_forward_expand(g.graph, weights);

  ggml_tensor *input3 = ggml_reshape_3d(ctx, cur, g.D, 1, g.S);
  ggml_tensor *ids_gpu_raw = ggml_get_rows(ctx, remap_gpu, ids);
  ggml_tensor *ids_gpu = ggml_reshape_2d(ctx, ids_gpu_raw, c.topk, g.S);

  ggml_tensor *up = ggml_mul_mat_id(ctx, up_gpu, input3, ids_gpu);
  ggml_tensor *gate = ggml_mul_mat_id(ctx, gate_gpu, input3, ids_gpu);
  ggml_tensor *act = ggml_swiglu_split(ctx, gate, up);
  ggml_tensor *down = ggml_mul_mat_id(ctx, down_gpu, act, ids_gpu);
  ggml_tensor *experts = ggml_mul(ctx, down, weights);

  if (g.layer_timing_enabled) {
    g.layer_timing_layer_of[up] = {layer, "moe_up_partial_gpu"};
    g.layer_timing_layer_of[gate] = {layer, "moe_gate_partial_gpu"};
    g.layer_timing_layer_of[act] = {layer, "moe_swiglu_partial_gpu"};
    g.layer_timing_layer_of[down] = {layer, "moe_down_partial_gpu"};
  }

  ggml_tensor *out = ggml_view_2d(ctx, experts, g.D, g.S, experts->nb[2], 0);
  out = ggml_cont(ctx, out);
  for (int k = 1; k < c.topk; ++k) {
    ggml_tensor *e = ggml_view_2d(ctx, experts, g.D, g.S, experts->nb[2],
                                  static_cast<size_t>(k) * experts->nb[1]);
    out = ggml_add(ctx, out, e);
  }
  (void)layer;
  return out;
}

bool qwen_moe_layer_cpu_only(const FullGraph &g, int layer) {
  const auto split_it = g.split_gpu_expert_ids.find(layer);
  if (split_it != g.split_gpu_expert_ids.end() && !split_it->second.empty())
    return false;
  return !graph_expert_projection_resident(g, layer, "ffn_gate_exps.weight") &&
         !graph_expert_projection_resident(g, layer, "ffn_up_exps.weight") &&
         !graph_expert_projection_resident(g, layer, "ffn_down_exps.weight");
}

void log_qwen_layout_tensor(const FullGraph &g, int layer, bool cpu_only,
                            const char *role, const ggml_tensor *tensor) {
  if (!tensor)
    return;
  ggml_backend_buffer_t buffer = tensor->buffer;
  if (!buffer && tensor->view_src)
    buffer = tensor->view_src->buffer;
  const char *buft = buffer
                         ? ggml_backend_buft_name(
                               ggml_backend_buffer_get_type(buffer))
                         : "unallocated";
  const char *backend_name = "unassigned";
  if (g.sched) {
    ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(
        g.sched, const_cast<ggml_tensor *>(tensor));
    if (backend)
      backend_name = ggml_backend_name(backend);
  }
  const char *pinned = "unknown";
  if (buffer && ggml_backend_buffer_is_host(buffer)) {
    void *base = ggml_backend_buffer_get_base(buffer);
    unsigned int flags = 0;
    if (base && cudaHostGetFlags(&flags, base) == cudaSuccess) {
      pinned = "yes";
    } else {
      cudaGetLastError();
      pinned = "no";
    }
  }
  LOG_INFO(
      "[GGML-QWEN-LAYOUT] layer=%d cpu_only=%d role=%s op=%s name=%s "
      "type=%s nbytes=%zu ne=[%lld,%lld,%lld,%lld] "
      "nb=[%zu,%zu,%zu,%zu] buft=%s backend=%s pinned=%s\n",
      layer, cpu_only ? 1 : 0, role, ggml_op_name(tensor->op),
      ggml_get_name(tensor), ggml_type_name(tensor->type),
      ggml_nbytes(tensor), (long long) tensor->ne[0],
      (long long) tensor->ne[1], (long long) tensor->ne[2],
      (long long) tensor->ne[3], tensor->nb[0], tensor->nb[1], tensor->nb[2],
      tensor->nb[3], buft, backend_name, pinned);
}

void dump_qwen_moe_layout(const FullGraph &g) {
  if (!graph_layout_debug() || !g.graph || !g.sched)
    return;
  std::unordered_set<const ggml_tensor *> logged;
  for (int i = 0; i < ggml_graph_n_nodes(g.graph); ++i) {
    ggml_tensor *node = ggml_graph_node(g.graph, i);
    if (!node || node->op != GGML_OP_MUL_MAT_ID || !node->src[0])
      continue;
    ggml_backend_t backend =
        ggml_backend_sched_get_tensor_backend(g.sched, node);
    if (backend != g.cpu)
      continue;
    const int layer = layer_timing_parse_layer(ggml_get_name(node->src[0]));
    if (layer < 0)
      continue;
    const bool cpu_only = qwen_moe_layer_cpu_only(g, layer);
    // The existing dump covered only MUL_MAT_ID sources.  Host-node mode
    // changes CUDA ownership at the Layer-13 boundary, so also record the
    // ordinary CPU op's destination: an output buffer changing from the
    // shared CUDA-host arena to a backend-private buffer would explain a
    // split-time fence even when weights and inputs are identical.
    log_qwen_layout_tensor(g, layer, cpu_only, "output", node);
    const char *roles[] = {"weight", "input", "ids"};
    for (int src = 0; src < 3; ++src) {
      ggml_tensor *tensor = node->src[src];
      if (!tensor || !logged.insert(tensor).second)
        continue;
      log_qwen_layout_tensor(g, layer, cpu_only, roles[src], tensor);
    }
  }
}

ggml_tensor *build_shared(FullGraph &g, ggml_context *ctx, ggml_tensor *cur,
                          int layer, ggml_tensor *gate, ggml_tensor *up,
                          ggml_tensor *down, ggml_tensor *scalar_gate) {
  if (!gate || !up || !down)
    return nullptr;
  ggml_tensor *gate_out = ggml_mul_mat(ctx, gate, cur);
  ggml_tensor *up_out = ggml_mul_mat(ctx, up, cur);
  ggml_tensor *act = ggml_swiglu_split(ctx, gate_out, up_out);
  ggml_tensor *out = ggml_mul_mat(ctx, down, act);
  if (scalar_gate) {
    ggml_tensor *sg = ggml_sigmoid(ctx, ggml_mul_mat(ctx, scalar_gate, cur));
    out = ggml_mul(ctx, out, sg);
  }
  (void)layer;
  return out;
}

// ggml-cuda's gated_delta_net kernel reads AND writes its per-head recurrent
// state in one fixed physical layout: M[col][i] = S[i][col] (col=v slower,
// i=k faster; see ggml-cuda/gated_delta_net.cu's own comment). This is the
// only layout the op ever accepts as input, so any device-resident state
// that round-trips straight back into the op (the full-graph decode
// self-loop) must stay in this "native" layout untouched.
//
// This codebase's own legacy CPU step (qwen4exp_gdn_delta_step) uses the
// opposite convention, st[k*hd+v] (k slower, v faster) -- i.e. exactly the
// per-head TRANSPOSE of the native layout. Converting between the two is a
// per-head square-matrix transpose, which is its own inverse, so the same
// helper handles both directions: legacy-in-for-native-consumption and
// native-out-for-legacy-consumption.
void gdn_state_transpose_host(float *dst, const float *src, int hd, int Hv) {
  for (int h = 0; h < Hv; ++h) {
    const float *s = src + static_cast<size_t>(h) * hd * hd;
    float *d = dst + static_cast<size_t>(h) * hd * hd;
    for (int a = 0; a < hd; ++a)
      for (int b = 0; b < hd; ++b)
        d[a * hd + b] = s[b * hd + a];
  }
}

ggml_tensor *build_gdn(FullGraph &g, ggml_context *ctx, ggml_tensor *cur,
                       int layer, ggml_tensor *w_qkv, ggml_tensor *w_gate,
                       ggml_tensor *w_alpha, ggml_tensor *w_beta,
                       ggml_tensor *w_dt, ggml_tensor *w_a, ggml_tensor *w_conv,
                       ggml_tensor *w_norm, ggml_tensor *w_out,
                       ggml_tensor *hist_in, ggml_tensor *state_in,
                       ggml_tensor **state_out, ggml_tensor **hist_out) {
  ggml_tensor *qkv = ggml_mul_mat(ctx, w_qkv, cur);
  auto diag = [&](const char *label, ggml_tensor *value) {
    if (layer == 0 && g.layer0_diag_enabled &&
        env_layer0_split_dump_path()) {
      g.layer0_split_snapshots.emplace_back(
          label, layer0_diag_snapshot(g, ctx, value, label));
    }
  };
  // These snapshots are deliberately taken before and inside the GDN node.
  // The existing attention-output dump only tells us that the discrepancy is
  // present after the recurrent block; these boundaries separate qkv GEMM,
  // convolution, GDN/state update, and the final output projection.
  diag("gdn_qkv", qkv);
  if (layer == 0 && g.layer0_diag_enabled && env_layer0_split_dump_path())
    log_layer0_diag_tensor("gdn_qkv_weight", w_qkv);
  ggml_tensor *z = ggml_mul_mat(ctx, w_gate, cur);
  ggml_tensor *alpha = ggml_mul_mat(ctx, w_alpha, cur);
  ggml_tensor *beta_raw = ggml_mul_mat(ctx, w_beta, cur);
  ggml_tensor *glog =
      ggml_mul(ctx, ggml_softplus(ctx, ggml_add(ctx, alpha, w_dt)), w_a);
  ggml_tensor *beta = ggml_sigmoid(ctx, beta_raw);

  ggml_tensor *conv_in = ggml_concat(ctx, hist_in, ggml_transpose(ctx, qkv), 0);
  diag("gdn_conv_in", conv_in);
  ggml_tensor *conv = ggml_silu(ctx, ggml_ssm_conv(ctx, conv_in, w_conv));
  diag("gdn_conv", conv);
  const size_t row = ggml_row_size(GGML_TYPE_F32, g.conv_dim);
  ggml_tensor *q =
      ggml_view_4d(ctx, conv, g.gdn_hd, g.Hk_gdn, g.S, 1,
                   ggml_row_size(GGML_TYPE_F32, g.gdn_hd), row, row * g.S, 0);
  ggml_tensor *k =
      ggml_view_4d(ctx, conv, g.gdn_hd, g.Hk_gdn, g.S, 1,
                   ggml_row_size(GGML_TYPE_F32, g.gdn_hd), row, row * g.S,
                   ggml_row_size(GGML_TYPE_F32, g.key_dim));
  ggml_tensor *v =
      ggml_view_4d(ctx, conv, g.gdn_hd, g.Hv_gdn, g.S, 1,
                   ggml_row_size(GGML_TYPE_F32, g.gdn_hd), row, row * g.S,
                   ggml_row_size(GGML_TYPE_F32, 2 * g.key_dim));
  q = ggml_l2_norm(ctx, q, g.engine->model_file.architecture_config.eps);
  k = ggml_l2_norm(ctx, k, g.engine->model_file.architecture_config.eps);
  glog = ggml_reshape_4d(ctx, glog, 1, g.Hv_gdn, g.S, 1);
  beta = ggml_reshape_4d(ctx, beta, 1, g.Hv_gdn, g.S, 1);
  ggml_tensor *gdn =
      ggml_gated_delta_net(ctx, q, k, v, glog, beta, state_in, 1);
  diag("gdn_output", gdn);

  const size_t hd_row = ggml_row_size(GGML_TYPE_F32, g.gdn_hd);
  const size_t value_row = ggml_row_size(GGML_TYPE_F32, g.value_dim);
  const size_t state_row = ggml_row_size(GGML_TYPE_F32, g.gdn_hd * g.gdn_hd);
  ggml_tensor *core = ggml_view_4d(ctx, gdn, g.gdn_hd, g.Hv_gdn, g.S, 1, hd_row,
                                   value_row, value_row * g.S, 0);
  diag("gdn_core", core);
  // *state_out is kept in the op's own NATIVE layout (ne[0]=k, ne[1]=v; see
  // the gdn_state_transpose_host comment above) because the full-graph
  // decode self-loop feeds this same tensor straight back into
  // ggml_gated_delta_net as next-token's state_in with no host round trip.
  // Any host-facing consumer (legacy CPU decode, cross-request state
  // persistence) must convert with gdn_state_transpose_host at the host
  // boundary instead of receiving a transposed tensor from here -- a
  // transpose baked in here only benefits the host path and silently
  // breaks the device-resident self-loop instead (this was the actual bug:
  // wrong values from the second decode token onward).
  ggml_tensor *state_raw =
      ggml_view_4d(ctx, gdn, g.gdn_hd, g.gdn_hd, g.Hv_gdn, 1, hd_row, state_row,
                   state_row * g.Hv_gdn,
                   static_cast<size_t>(g.value_dim) * g.S * sizeof(float));
  *state_out = ggml_cont(ctx, state_raw);
  ggml_tensor *norm = ggml_mul(
      ctx,
      ggml_rms_norm(ctx, core, g.engine->model_file.architecture_config.eps),
      w_norm);
  diag("gdn_norm", norm);
  ggml_tensor *z4 = ggml_reshape_4d(ctx, z, g.gdn_hd, g.Hv_gdn, g.S, 1);
  ggml_tensor *gated = ggml_mul(ctx, norm, ggml_sigmoid(ctx, z4));
  diag("gdn_gated", gated);
  ggml_tensor *out =
      ggml_mul_mat(ctx, w_out, ggml_reshape_2d(ctx, gated, g.value_dim, g.S));
  diag("gdn_output_projection", out);
  *hist_out =
      ggml_cont(ctx, ggml_view_3d(ctx, conv_in, g.d_conv - 1, g.conv_dim, 1,
                                  conv_in->nb[1], conv_in->nb[2],
                                  ggml_row_size(conv_in->type, g.S)));
  (void)layer;
  return out;
}

ggml_tensor *build_full_attention(FullGraph &g, ggml_context *ctx,
                                  ggml_tensor *cur, int layer, ggml_tensor *w_q,
                                  ggml_tensor *w_k, ggml_tensor *w_v,
                                  ggml_tensor *w_qn, ggml_tensor *w_kn,
                                  ggml_tensor *w_o, ggml_tensor **k_cache,
                                  ggml_tensor **v_cache) {
  ggml_tensor *qfull = ggml_mul_mat(ctx, w_q, cur);
  ggml_tensor *q =
      ggml_view_3d(ctx, qfull, g.hd, g.H, g.S, sizeof(float) * g.hd * 2,
                   sizeof(float) * g.hd * 2 * g.H, 0);
  q = ggml_mul(
      ctx, ggml_rms_norm(ctx, q, g.engine->model_file.architecture_config.eps),
      w_qn);
  ggml_tensor *k =
      ggml_reshape_3d(ctx, ggml_mul_mat(ctx, w_k, cur), g.hd, g.Hkv, g.S);
  k = ggml_mul(
      ctx, ggml_rms_norm(ctx, k, g.engine->model_file.architecture_config.eps),
      w_kn);
  ggml_tensor *v =
      ggml_reshape_3d(ctx, ggml_mul_mat(ctx, w_v, cur), g.hd, g.Hkv, g.S);
  const int n_rot = g.engine->model_file.architecture_config.rope_n_rot > 0
                        ? g.engine->model_file.architecture_config.rope_n_rot
                        : g.hd;
  q = ggml_rope(ctx, q, g.positions, n_rot, GGML_ROPE_TYPE_NORMAL);
  k = ggml_rope(ctx, k, g.positions, n_rot, GGML_ROPE_TYPE_NORMAL);
  // The host dense KV cache stores the same normalized and rotated K/V values
  // that Bee's graph writes to its cache.  Keep these F32 tensors before the
  // layout-only permute/cast used by flash attention.
  if (k_cache)
    *k_cache = k;
  if (v_cache)
    *v_cache = v;
  q = ggml_permute(ctx, q, 0, 2, 1, 3);
  k = ggml_permute(ctx, k, 0, 2, 1, 3);
  v = ggml_permute(ctx, v, 0, 2, 1, 3);
  // The CUDA MMA flash-attention implementation intentionally consumes Q as
  // F32 and converts/loads K and V as F16.  Casting Q to F16 selects the MMA
  // path and then trips its Q-type assertion during graph execution.
  if (k->type == GGML_TYPE_F32)
    k = ggml_cast(ctx, k, GGML_TYPE_F16);
  if (v->type == GGML_TYPE_F32)
    v = ggml_cast(ctx, v, GGML_TYPE_F16);
  ggml_tensor *attn = ggml_flash_attn_ext(
      ctx, q, k, v, g.attn_mask, 1.0f / std::sqrt(static_cast<float>(g.hd)),
      0.0f, 0.0f);
  ggml_tensor *gate =
      ggml_view_3d(ctx, qfull, g.hd, g.H, g.S, sizeof(float) * g.hd * 2,
                   sizeof(float) * g.hd * 2 * g.H, sizeof(float) * g.hd);
  gate = ggml_reshape_3d(ctx, ggml_cont_2d(ctx, gate, g.H * g.hd, g.S), g.hd,
                         g.H, g.S);
  attn = ggml_mul(ctx, attn, ggml_sigmoid(ctx, gate));
  attn = ggml_reshape_2d(ctx, attn, g.H * g.hd, g.S);
  (void)layer;
  return ggml_mul_mat(ctx, w_o, attn);
}

// Decode variant of build_full_attention().  K/V are appended directly to a
// device-resident cache with one set_rows node.  The graph therefore carries
// the cache update, attention, and output projection in the same scheduler
// submission; no host K/V readback is needed between tokens.
ggml_tensor *build_full_attention_decode(
    FullGraph &g, ggml_context *ctx, ggml_tensor *cur, int layer,
    ggml_tensor *w_q, ggml_tensor *w_k, ggml_tensor *w_v, ggml_tensor *w_qn,
    ggml_tensor *w_kn, ggml_tensor *w_o, ggml_tensor *k_storage,
    ggml_tensor *v_storage) {
  ggml_tensor *qfull = ggml_mul_mat(ctx, w_q, cur);
  ggml_tensor *q = ggml_view_3d(
      ctx, qfull, g.hd, g.H, 1, sizeof(float) * g.hd * 2,
      sizeof(float) * g.hd * 2 * g.H, 0);
  q = ggml_mul(
      ctx, ggml_rms_norm(ctx, q, g.engine->model_file.architecture_config.eps),
      w_qn);
  ggml_tensor *k = ggml_reshape_3d(
      ctx, ggml_mul_mat(ctx, w_k, cur), g.hd, g.Hkv, 1);
  k = ggml_mul(
      ctx, ggml_rms_norm(ctx, k, g.engine->model_file.architecture_config.eps),
      w_kn);
  ggml_tensor *v = ggml_reshape_3d(
      ctx, ggml_mul_mat(ctx, w_v, cur), g.hd, g.Hkv, 1);
  const int n_rot = g.engine->model_file.architecture_config.rope_n_rot > 0
                        ? g.engine->model_file.architecture_config.rope_n_rot
                        : g.hd;
  q = ggml_rope(ctx, q, g.positions, n_rot, GGML_ROPE_TYPE_NORMAL);
  k = ggml_rope(ctx, k, g.positions, n_rot, GGML_ROPE_TYPE_NORMAL);

  ggml_tensor *k2 = ggml_reshape_2d(ctx, k, g.hd, g.Hkv);
  ggml_tensor *v2 = ggml_reshape_2d(ctx, v, g.hd, g.Hkv);
  ggml_tensor *kset = ggml_set_rows(ctx, k_storage, k2, g.decode_indices);
  ggml_tensor *vset = ggml_set_rows(ctx, v_storage, v2, g.decode_indices);
  ggml_tensor *kcache = ggml_reshape_4d(
      ctx, kset, g.hd, g.kv_capacity, g.Hkv, 1);
  ggml_tensor *vcache = ggml_reshape_4d(
      ctx, vset, g.hd, g.kv_capacity, g.Hkv, 1);

  q = ggml_permute(ctx, q, 0, 2, 1, 3);
  if (kcache->type == GGML_TYPE_F32)
    kcache = ggml_cast(ctx, kcache, GGML_TYPE_F16);
  if (vcache->type == GGML_TYPE_F32)
    vcache = ggml_cast(ctx, vcache, GGML_TYPE_F16);
  ggml_tensor *attn = ggml_flash_attn_ext(
      ctx, q, kcache, vcache, g.attn_mask,
      1.0f / std::sqrt(static_cast<float>(g.hd)), 0.0f, 0.0f);
  ggml_tensor *gate = ggml_view_3d(
      ctx, qfull, g.hd, g.H, 1, sizeof(float) * g.hd * 2,
      sizeof(float) * g.hd * 2 * g.H, sizeof(float) * g.hd);
  gate = ggml_reshape_3d(ctx, ggml_cont_2d(ctx, gate, g.H * g.hd, 1), g.hd,
                         g.H, 1);
  attn = ggml_mul(ctx, attn, ggml_sigmoid(ctx, gate));
  attn = ggml_reshape_2d(ctx, attn, g.H * g.hd, 1);
  (void)layer;
  return ggml_mul_mat(ctx, w_o, attn);
}

bool prepare_ple_embedding(FullGraph &g, std::vector<float> &embedding, int S,
                           int pos_base, const int *token_ids) {
  const auto &c = g.engine->model_file.architecture_config;
  if (!has_ple_layer(c) || !token_ids ||
      (pos_base != 0 && !(g.decode_mode && S == 1)))
    return !has_ple_layer(c);
  const GGUFTensor *table =
      model_tensor(*g.engine, "per_layer_token_embd.weight");
  if (!table || table->shape.size() != 2)
    return false;
  const int row_dim = static_cast<int>(table->shape[0]);
  const int rb = row_bytes(table->type, row_dim);
  if (rb <= 0 || row_dim != c.ple_n_embd_per_layer)
    return false;

  std::vector<int32_t> rows(static_cast<size_t>(S) * c.ple_n_heads);
  const int64_t eos = c.ple_eos_token_id;
  for (int i = 0; i < S; ++i) {
    int64_t ctx[8] = {};
    if (c.ple_ngram_size > 8)
      return false;
    ctx[0] = token_ids[i];
    bool cut = false;
    for (int s = 1; s < c.ple_ngram_size; ++s) {
      if (g.decode_mode) {
        const int64_t pos = pos_base + i;
        const int64_t back = s - i;
        const auto &history = g.engine->qwen4exp_ple_tok_hist;
        if (g.engine->qwen4exp_ple_next_pos == pos && back > 0 &&
            back <= static_cast<int64_t>(history.size())) {
          const size_t index = history.size() - static_cast<size_t>(back);
          ctx[s] = history[index];
        } else {
          ctx[s] = eos;
        }
      } else {
        const int j = i - s;
        ctx[s] = j >= 0 ? token_ids[j] : eos;
      }
      if (cut || ctx[s] == eos) {
        ctx[s] = eos;
        cut = true;
      }
    }
    for (int n = 2; n <= c.ple_ngram_size; ++n) {
      int base = (n - 2) * c.ple_heads_per_ngram;
      for (int h = 0; h < c.ple_heads_per_ngram; ++h) {
        int hi = base + h;
        rows[static_cast<size_t>(i) * c.ple_n_heads + hi] =
            static_cast<int32_t>(
                ple_ngram_row(ctx, n, c.ple_layer_multipliers.data(),
                              c.ple_head_vocab_sizes[static_cast<size_t>(hi)],
                              c.ple_head_offsets[static_cast<size_t>(hi)]));
      }
    }
  }

  std::unordered_map<int32_t, std::vector<float>> cache;
  void *raw_mem = nullptr;
  if (posix_memalign(&raw_mem, 4096, static_cast<size_t>(rb)) != 0 ||
      !raw_mem)
    return false;
  std::unique_ptr<uint8_t, decltype(&std::free)> raw_guard(
      static_cast<uint8_t *>(raw_mem), &std::free);
  auto *raw = raw_guard.get();
  const int fd = g.engine->model_file.fd_for_tensor(*table);
  for (int32_t r : rows) {
    if (cache.count(r))
      continue;
    const ssize_t got = direct_io::pread_direct_aware(
        fd, g.engine->model_file.direct_io_active_for(*table),
        g.engine->model_file.direct_io_alignment, raw, static_cast<size_t>(rb),
        table->offset + static_cast<uint64_t>(r) * static_cast<size_t>(rb));
    if (got != static_cast<ssize_t>(rb))
      return false;
    std::vector<float> row(static_cast<size_t>(row_dim));
    if (!dequant_row(table->type, raw, row.data(), row_dim))
      return false;
    cache.emplace(r, std::move(row));
  }
  embedding.resize(static_cast<size_t>(S) * c.ple_n_heads * row_dim);
  for (int i = 0; i < S; ++i) {
    for (int h = 0; h < c.ple_n_heads; ++h) {
      const auto &row =
          cache.at(rows[static_cast<size_t>(i) * c.ple_n_heads + h]);
      std::memcpy(embedding.data() +
                      (static_cast<size_t>(i) * c.ple_n_heads + h) * row_dim,
                  row.data(), static_cast<size_t>(row_dim) * sizeof(float));
    }
  }
  return true;
}

bool build_graph(FullGraph &g, const std::vector<float> *ple_embedding) {
  // Timing split (2026-09-17): full-graph decode's FIRST call pays a real
  // ~5.3s one-time cost, invisible in any --ngen 400 average (6000/400=
  // 15ms/token) but a genuine multi-second latency cliff for interactive
  // use (every new conversation/context reset pays it). See memory:
  // full-graph-decode-graph-construction-5s-latency.md. This split isolates
  // OUR host-side node-construction C++ (this function's own body) from
  // ggml's own ggml_backend_sched_alloc_graph() call below, since the fix
  // differs completely depending on which one dominates.
  const auto t_build_graph_start = std::chrono::steady_clock::now();
  g.dense_upload_ms_total = 0.0;
  g.expert_upload_ms_total = 0.0;
  g.expert_upload_calls = 0;
  g.attn_construction_ms_total = 0.0;
  g.moe_construction_ms_total = 0.0;
  g.moe_weight_lookup_ms_total = 0.0;
  const auto &c = g.engine->model_file.architecture_config;
  if (env_layer0_iq2_diag_enabled() || env_layer0_split_dump_path())
    g.layer0_diag_enabled = true;
  ggml_init_params ip = {};
  ip.mem_size = kTensorContextBytes;
  ip.no_alloc = true;
  if (!MemoryGuard::try_reserve_host_bytes(kTensorContextBytes)) {
    LOG_WARN("[GGML-QWEN-FULL-MEM] host graph context refused: %.2f MiB "
             "(reserved=%.2f GiB, budget=%.2f GB)\n",
             (double)kTensorContextBytes / (1024.0 * 1024.0),
             (double)MemoryGuard::reserved_host_bytes() / 1.0e9,
             (double)MemoryGuard::host_budget_bytes() / 1.0e9);
    return false;
  }
  g.context_reservation_bytes = kTensorContextBytes;
  g.ctx = ggml_init(ip);
  if (!g.ctx) {
    MemoryGuard::release_host_bytes(g.context_reservation_bytes);
    g.context_reservation_bytes = 0;
    return false;
  }
  g.graph = ggml_new_graph_custom(g.ctx, kGraphSize, false);
  if (!g.graph)
    return false;

  g.input = host_input(g, g.ctx, GGML_TYPE_F32, {g.D, g.HC, g.S},
                       static_cast<size_t>(g.D) * g.HC * g.S * sizeof(float),
                       "qwen_full_input");
  g.positions = host_input(g, g.ctx, GGML_TYPE_I32, {g.S},
                           static_cast<size_t>(g.S) * sizeof(int32_t),
                           "qwen_full_positions");
  if (g.decode_mode) {
    g.attn_mask = host_input(
        g, g.ctx, GGML_TYPE_F16, {g.kv_capacity, 1, 1, 1},
        static_cast<size_t>(g.kv_capacity) * sizeof(ggml_fp16_t),
        "qwen_full_decode_mask");
    g.decode_indices = host_input(
        g, g.ctx, GGML_TYPE_I32, {g.Hkv},
        static_cast<size_t>(g.Hkv) * sizeof(int32_t),
        "qwen_full_decode_kv_indices");
  } else {
    g.attn_mask = host_input(
        g, g.ctx, GGML_TYPE_F16, {g.S, g.S},
        static_cast<size_t>(g.S) * g.S * sizeof(ggml_fp16_t),
        "qwen_full_causal_mask");
  }
  if (!g.input || !g.positions || !g.attn_mask ||
      (g.decode_mode && !g.decode_indices))
    return false;
  auto *pos = static_cast<int32_t *>(g.positions->data);
  auto *mask = static_cast<ggml_fp16_t *>(g.attn_mask->data);
  for (int i = 0; i < g.S; ++i)
    pos[i] = i;
  if (g.decode_mode) {
    for (int k = 0; k < g.kv_capacity; ++k)
      mask[k] = ggml_fp32_to_fp16(0.0f);
    auto *indices = static_cast<int32_t *>(g.decode_indices->data);
    for (int h = 0; h < g.Hkv; ++h)
      indices[h] = h;
  } else {
    for (int q = 0; q < g.S; ++q)
      for (int k = 0; k < g.S; ++k)
        mask[static_cast<size_t>(q) * g.S + k] =
            ggml_fp32_to_fp16(k <= q ? 0.0f : -INFINITY);
  }

  ggml_tensor *res = g.input;
  // Set true the moment the Phase 2 prefix instance hits its split-layer
  // early stop, so the post-loop output-head/logits construction below
  // (which assumes every layer got built) is skipped for this instance --
  // its real "output" is the handoff tensor, already marked further up.
  bool phase2_prefix_stopped = false;
  if (g.phase2_mode == FullGraph::Phase2Mode::SuffixGpu ||
      g.phase2_mode == FullGraph::Phase2Mode::SuffixCpu) {
    // Suffix graphs never read g.input at all -- they resume from the
    // prefix instance's own persistent device-resident handoff tensor
    // (the residual right after phase2_split_layer's attention,
    // hc_combine'd but pre-FFN). Binds a fresh leaf, in THIS graph's own
    // ctx, to the SAME device address the prefix instance wrote into --
    // exactly the mechanism split_graph_spike.cpp validated (~1 ULP
    // cross-backend noise, no structural divergence). The prefix instance
    // is just g_graph itself with phase2_mode==Prefix (see
    // qwen4exp_full_graph_decode_prepare's build_one lambda) -- it must
    // already be assigned to g_graph, and must already have run this
    // decode step, before either suffix graph is computed.
    if (!g_graph || !g_graph->phase2_handoff_buf || !g_graph->phase2_handoff_addr)
      return false;
    res = ggml_new_tensor_3d(g.ctx, GGML_TYPE_F32, g.D, g.HC, g.S);
    ggml_backend_tensor_alloc(g_graph->phase2_handoff_buf, res,
                              g_graph->phase2_handoff_addr);
    ggml_set_name(res, "qwen_full_phase2_handoff_in");
  }
  if (g.phase2_mode != FullGraph::Phase2Mode::SuffixGpu &&
      g.phase2_mode != FullGraph::Phase2Mode::SuffixCpu &&
      graph_match_bee_topology()) {
    // The scheduler normally starts directly on the first GPU-weighted
    // operation because g.input is already a CPU buffer.  An explicit CPU
    // CONT makes the initial CPU->GPU handoff visible as the same leading
    // CPU/GPU split that Bee has around model.input_embed/hc_init.
    res = ggml_cont(g.ctx, res);
    if (!res)
      return false;
    ggml_set_name(res, "qwen_full_bee_cpu_prologue");
    g.topology_cpu_prologue = res;
    if (g.sched) {
      // In host-node mode every CPU MoE boundary is represented by a
      // CUDA-owned CPY marker.  Keep the initial contiguous handoff on the
      // same CUDA split as those markers; otherwise this otherwise harmless
      // topology node leaves a second scheduler split in the graph.  The
      // default path remains Bee's original CPU prologue assignment.
      ggml_backend_sched_set_tensor_backend(
          g.sched, res,
          env_qwen4exp_host_node_enabled() &&
                  !env_host_node_keep_cpu_prologue()
              ? g.cuda
              : g.cpu);
    }
  }
  // Decode-only router-usage tap (see Qwen4ExpUsageTrackContext's own
  // comment). Every layer that goes through build_moe() below appends its
  // real routed ids into g.usage_track_ids via an ordinary ggml_concat node;
  // a single flush marker after the loop reads the whole buffer once.
  const bool usage_track_enabled =
      g.decode_mode && g.engine->qwen4exp_track_router_usage;
  for (int l = 0; l < g.n_layers; ++l) {
    // Suffix instances own layers [phase2_split_layer, n_layers) -- layers
    // strictly BEFORE the split live entirely in the prefix instance
    // (which already ran this decode step). The split layer itself is
    // special: its ATTENTION (and any PLE for that same index) already
    // happened in the prefix graph, which handed off `res` right after
    // that layer's post-attention hc_combine -- only ITS FFN still needs
    // building here. Skipping via `continue` (not starting the loop at
    // split_layer) keeps every downstream index (`p`, GDN/KV push_back
    // ordering, layer_timing) identical to the Off/single-graph path for
    // the layers each instance DOES fully build -- g.gdn_layers/
    // g.kv_layers end up holding only this instance's own indices, which
    // the existing state-seed/extract code already iterates
    // instance-locally (no special-casing needed there).
    if ((g.phase2_mode == FullGraph::Phase2Mode::SuffixGpu ||
         g.phase2_mode == FullGraph::Phase2Mode::SuffixCpu) &&
        l < g.phase2_split_layer)
      continue;
    // True only for the split layer, in a suffix instance: skip PLE/
    // attention entirely (the prefix graph already built and applied
    // them into the `res` this instance resumed from via the handoff
    // tensor) and fall straight through to the FFN section below.
    const bool phase2_suffix_skip_attn =
        (g.phase2_mode == FullGraph::Phase2Mode::SuffixGpu ||
         g.phase2_mode == FullGraph::Phase2Mode::SuffixCpu) &&
        l == g.phase2_split_layer;
    const std::string p = "blk." + std::to_string(l) + ".";
    if (l == 1 && g.layer0_diag_enabled &&
        env_layer0_split_dump_path()) {
      g.layer0_split_snapshots.emplace_back(
          "layer1_before_ple",
          layer0_diag_snapshot(g, g.ctx, res,
                               "qwen_layer0_split_layer1_before_ple"));
    }
    if (!phase2_suffix_skip_attn && qwen4exp_is_ple_layer(c, l)) {
      if (!ple_embedding)
        return false;
      g.ple_embedding =
          host_input(g, g.ctx, GGML_TYPE_F32,
                     {c.ple_n_embd_per_layer * c.ple_n_heads, g.S},
                     static_cast<size_t>(c.ple_n_embd_per_layer) *
                         c.ple_n_heads * g.S * sizeof(float),
                     "qwen_full_ple_embedding");
      if (!g.ple_embedding)
        return false;
      std::memcpy(g.ple_embedding->data, ple_embedding->data(),
                  ple_embedding->size() * sizeof(float));
      if (l == 1 && g.layer0_diag_enabled &&
          env_layer0_split_dump_path()) {
        g.layer0_split_snapshots.emplace_back(
            "ple_embd",
            layer0_diag_snapshot(g, g.ctx, g.ple_embedding,
                                 "qwen_layer0_split_ple_embd"));
      }
      g.ple_layer = l;
      // PLE graph is constructed below with the same equations as Bee's
      // build_ple.  The input embedding is host gathered, while its learned
      // projections and the rest of the operation stay in this graph.
      ggml_tensor *ple_key = resident_tensor(g, g.ctx, p + "ple_key.weight");
      ggml_tensor *ple_value =
          resident_tensor(g, g.ctx, p + "ple_value.weight");
      ggml_tensor *ple_nk =
          resident_tensor(g, g.ctx, p + "ple_norm_key.weight");
      ggml_tensor *ple_nq =
          resident_tensor(g, g.ctx, p + "ple_norm_query.weight");
      ggml_tensor *ple_nc =
          resident_tensor(g, g.ctx, p + "ple_norm_conv.weight");
      ggml_tensor *ple_conv =
          resident_tensor(g, g.ctx, p + "ple_conv1d.weight");
      if (!ple_key || !ple_value || !ple_nk || !ple_nq || !ple_nc || !ple_conv)
        return false;
      ggml_tensor *pk = ggml_mul_mat(g.ctx, ple_key, g.ple_embedding);
      ggml_tensor *pv = ggml_mul_mat(g.ctx, ple_value, g.ple_embedding);
      auto group_norm = [&](ggml_tensor *x, ggml_tensor *w) {
        x = ggml_reshape_3d(g.ctx, x, g.D, g.HC, g.S);
        x = ggml_rms_norm(g.ctx, x, c.eps);
        x = ggml_reshape_2d(g.ctx, x, g.D * g.HC, g.S);
        x = ggml_mul(g.ctx, x, w);
        return ggml_reshape_3d(g.ctx, x, g.D, g.HC, g.S);
      };
      pk = group_norm(pk, ple_nk);
      ggml_tensor *pq = group_norm(res, ple_nq);
      ggml_tensor *score =
          ggml_scale(g.ctx, ggml_sum_rows(g.ctx, ggml_mul(g.ctx, pk, pq)),
                     1.0f / std::sqrt(static_cast<float>(g.D)));
      ggml_tensor *gate = ggml_sigmoid(
          g.ctx,
          ggml_mul(g.ctx, ggml_sgn(g.ctx, score),
                   ggml_sqrt(g.ctx, ggml_clamp(g.ctx, ggml_abs(g.ctx, score),
                                               1e-6f, INFINITY))));
      ggml_tensor *v3 = ggml_repeat_4d(
          g.ctx, ggml_reshape_3d(g.ctx, pv, g.D, 1, g.S), g.D, g.HC, g.S, 1);
      ggml_tensor *gated = ggml_mul(g.ctx, v3, gate);
      if (l == 1 && g.layer0_diag_enabled &&
          env_layer0_split_dump_path()) {
        g.layer0_split_snapshots.emplace_back(
            "ple_gate",
            layer0_diag_snapshot(g, g.ctx, gate,
                                 "qwen_layer0_split_ple_gate"));
        g.layer0_split_snapshots.emplace_back(
            "ple_gated_value",
            layer0_diag_snapshot(g, g.ctx, gated,
                                 "qwen_layer0_split_ple_gated_value"));
      }
      ggml_tensor *ple_conv_grouped =
          ggml_reshape_3d(g.ctx, gated, g.D, g.HC, g.S);
      ggml_tensor *ple_conv_rms =
          ggml_rms_norm(g.ctx, ple_conv_grouped, c.eps);
      ple_conv_rms =
          ggml_reshape_2d(g.ctx, ple_conv_rms, g.D * g.HC, g.S);
      ggml_tensor *normed = ggml_mul(g.ctx, ple_conv_rms, ple_nc);
      if (l == 1 && g.layer0_diag_enabled &&
          env_layer0_split_dump_path()) {
        g.layer0_split_snapshots.emplace_back(
            "ple_conv_input",
            layer0_diag_snapshot(g, g.ctx, normed,
                                 "qwen_layer0_split_ple_conv_input"));
        g.layer0_split_snapshots.emplace_back(
            "ple_conv_rms",
            layer0_diag_snapshot(g, g.ctx, ple_conv_rms,
                                 "qwen_layer0_split_ple_conv_rms"));
        g.layer0_split_snapshots.emplace_back(
            "ple_norm_conv_weight",
            layer0_diag_snapshot(g, g.ctx, ple_nc,
                                 "qwen_layer0_split_ple_norm_conv_weight"));
      }

      const int hist = (c.ple_conv_kernel - 1) * c.ple_ngram_size;
      ggml_tensor *ph = nullptr;
      if (g.decode_mode) {
        ph = new_device_state_tensor(
            g, GGML_TYPE_F32, {hist, g.D * g.HC, 1},
            "qwen_full_decode_ple_history");
        g.ple_hist_storage = ph;
      } else {
        ph = host_input(
            g, g.ctx, GGML_TYPE_F32, {hist, g.D * g.HC, 1},
            static_cast<size_t>(hist) * g.D * g.HC * sizeof(float),
            "qwen_full_ple_history");
      }
      if (!ph)
        return false;
      if (!g.decode_mode)
        g.ple_hist_input = static_cast<float *>(ph->data);
      if (!g.decode_mode &&
          l < static_cast<int>(g.engine->qwen4exp_ple_conv_history.size())) {
        auto &h = g.engine->qwen4exp_ple_conv_history[l];
        if (h.size() == static_cast<size_t>(hist) * g.D * g.HC)
          std::memcpy(ph->data, h.data(), h.size() * sizeof(float));
      }
      ggml_tensor *padded =
          ggml_concat(g.ctx, ph, ggml_transpose(g.ctx, normed), 0);
      ggml_tensor *conv_out = nullptr;
      for (int k = 0; k < c.ple_conv_kernel; ++k) {
        const int start = hist - (c.ple_conv_kernel - 1 - k) * c.ple_ngram_size;
        ggml_tensor *shifted = ggml_cont(
            g.ctx,
            ggml_transpose(g.ctx,
                           ggml_view_3d(g.ctx, padded, g.S, g.D * g.HC, 1,
                                        padded->nb[1], padded->nb[2],
                                        ggml_row_size(padded->type, start))));
        ggml_tensor *wk = ggml_cont(
            g.ctx, ggml_view_2d(g.ctx, ple_conv, 1, g.D * g.HC, ple_conv->nb[1],
                                static_cast<size_t>(k) * ple_conv->nb[0]));
        wk = ggml_reshape_1d(g.ctx, wk, g.D * g.HC);
        if (wk->type != GGML_TYPE_F32)
          wk = ggml_cast(g.ctx, wk, GGML_TYPE_F32);
        ggml_tensor *term = ggml_mul(g.ctx, shifted, wk);
        conv_out = conv_out ? ggml_add(g.ctx, conv_out, term) : term;
      }
      ggml_tensor *ple_conv_raw = conv_out;
      if (l == 1 && g.layer0_diag_enabled &&
          env_layer0_split_dump_path()) {
        g.layer0_split_snapshots.emplace_back(
            "ple_conv_raw",
            layer0_diag_snapshot(g, g.ctx, ple_conv_raw,
                                 "qwen_layer0_split_ple_conv_raw"));
      }
      conv_out =
          ggml_reshape_3d(g.ctx, ggml_silu(g.ctx, conv_out), g.D, g.HC, g.S);
      if (l == 1 && g.layer0_diag_enabled &&
          env_layer0_split_dump_path()) {
        g.layer0_split_snapshots.emplace_back(
            "ple_conv_out",
            layer0_diag_snapshot(g, g.ctx, conv_out,
                                 "qwen_layer0_split_ple_conv_out"));
      }
      res = ggml_add(g.ctx, res, ggml_add(g.ctx, gated, conv_out));
      if (l == 1 && g.layer0_diag_enabled &&
          env_layer0_split_dump_path()) {
        g.layer0_split_snapshots.emplace_back(
            "layer1_after_ple",
            layer0_diag_snapshot(g, g.ctx, res,
                                 "qwen_layer0_split_layer1_after_ple"));
      }
      g.ple_hist_out = ggml_cont(
          g.ctx, ggml_view_3d(g.ctx, padded, hist, g.D * g.HC, 1, padded->nb[1],
                              padded->nb[2], ggml_row_size(padded->type, g.S)));
      if (g.decode_mode) {
        g.ple_hist_update =
            ggml_cpy(g.ctx, g.ple_hist_out, g.ple_hist_storage);
      }
    }

    const auto t_attn_start = std::chrono::steady_clock::now();
    ggml_tensor *inject = nullptr;
    // Declared here (not inside the guarded block below) because the FFN
    // section further down reassigns these same four names to its own
    // hc_ffn_* weights -- must stay in scope across the
    // phase2_suffix_skip_attn guard.
    ggml_tensor *hn = nullptr, *hdn = nullptr, *hup = nullptr, *hinj = nullptr;
    if (!phase2_suffix_skip_attn) {
    hn = resident_tensor(g, g.ctx, p + "hc_attn_norm.weight");
    hdn = resident_tensor(g, g.ctx, p + "hc_attn_down.weight");
    hup = resident_tensor(g, g.ctx, p + "hc_attn_up.weight");
    hinj = resident_tensor(g, g.ctx, p + "hc_attn_inject.weight");
    if (!hn || !hdn || !hup || !hinj)
      return false;
    ggml_tensor *attn_in =
        hc_mix(g, g.ctx, res, hn, hdn, hup, hinj, &inject, l, "attn");
    if (l == 0) {
      const char *common_path = nullptr;
      if (g.two_layer_diag && g.S == 1)
        common_path = env_two_layer_s1_common_attn_input_path();
      else if (!g.decode_mode)
        common_path = env_layer0_common_attn_input_path();
      if (common_path) {
        const size_t common_n = static_cast<size_t>(g.D) * g.S;
        ggml_tensor *common = host_input(
            g, g.ctx, GGML_TYPE_F32, {g.D, g.S},
            common_n * sizeof(float),
            g.two_layer_diag ? "qwen_two_layer_s1_common_attn_input"
                             : "qwen_layer0_common_attn_input");
        if (!common || !load_exact_f32_file(
                           common_path, static_cast<float *>(common->data),
                           common_n)) {
          LOG_WARN(
              "[QWEN4EXP-LAYER0-DIAG] common attention input rejected: "
              "path=%s expected_bytes=%zu S=%d\n",
              common_path, common_n * sizeof(float), g.S);
          return false;
        }
        const uint64_t hash = qwen4exp_fnv64(
            common->data, common_n * sizeof(float));
        LOG_INFO("[QWEN4EXP-LAYER0-DIAG] forcing Layer-0 QKV input from %s "
                 "elements=%zu S=%d fnv64=0x%016llx\n",
                 common_path, common_n, g.S,
                 static_cast<unsigned long long>(hash));
        // The remainder of this graph, including Layer 1, consumes the
        // forced result.  This is intentionally a graph input replacement,
        // not a post-GEMM copy, so the QKV backend sees exactly the supplied
        // bytes and its normal quantization path remains under test.
        attn_in = common;
      }
    }
    if (g.two_layer_diag && l == 1)
      g.two_layer_l1_pre_attn = g.layer0_diag_enabled
                                  ? layer0_diag_snapshot(
                                        g, g.ctx, attn_in,
                                        "qwen_two_layer_s1_l1_pre_attn")
                                  : attn_in;
    if (l == 0 && g.layer0_diag_enabled &&
        env_layer0_split_dump_path()) {
      g.layer0_split_snapshots.emplace_back(
          "attention_input",
          layer0_diag_snapshot(g, g.ctx, attn_in,
                               "qwen_layer0_split_attention_input"));
    }
    if (l == 1 && g.layer0_diag_enabled) {
      g.layer0_diag_next_consumer = attn_in;
      g.layer0_diag_next_consumer_snapshot = layer0_diag_snapshot(
          g, g.ctx, attn_in, "qwen_layer0_diag_layer1_consumer");
      if (env_layer0_split_dump_path()) {
        g.layer0_split_snapshots.emplace_back(
            "layer1_attention_input",
            layer0_diag_snapshot(g, g.ctx, attn_in,
                                 "qwen_layer0_split_layer1_attention_input"));
      }
    }
    ggml_tensor *attn_out = nullptr;
    if (qwen4exp_is_gdn_layer(c, l)) {
      ggml_tensor *qkv = resident_tensor(g, g.ctx, p + "attn_qkv.weight");
      ggml_tensor *zg = resident_tensor(g, g.ctx, p + "attn_gate.weight");
      ggml_tensor *al = resident_tensor(g, g.ctx, p + "ssm_alpha.weight");
      ggml_tensor *be = resident_tensor(g, g.ctx, p + "ssm_beta.weight");
      ggml_tensor *dt = resident_tensor(g, g.ctx, p + "ssm_dt.bias");
      ggml_tensor *aa = resident_tensor(g, g.ctx, p + "ssm_a");
      ggml_tensor *cv = resident_tensor(g, g.ctx, p + "ssm_conv1d.weight");
      ggml_tensor *nm = resident_tensor(g, g.ctx, p + "ssm_norm.weight");
      ggml_tensor *oo = resident_tensor(g, g.ctx, p + "ssm_out.weight");
      if (!qkv || !zg || !al || !be || !dt || !aa || !cv || !nm || !oo)
        return false;
      const int hist = g.d_conv - 1;
      ggml_tensor *hi = nullptr;
      ggml_tensor *si = nullptr;
      if (g.decode_mode) {
        hi = new_device_state_tensor(g, GGML_TYPE_F32, {hist, g.conv_dim, 1},
                                      "qwen_full_decode_gdn_history");
        si = new_device_state_tensor(
            g, GGML_TYPE_F32, {g.gdn_hd, g.gdn_hd, g.Hv_gdn, 1},
            "qwen_full_decode_gdn_state");
        g.gdn_hist_storage.push_back(hi);
        g.gdn_state_storage.push_back(si);
      } else {
        hi = host_input(
            g, g.ctx, GGML_TYPE_F32, {hist, g.conv_dim, 1},
            static_cast<size_t>(hist) * g.conv_dim * sizeof(float),
            "qwen_full_gdn_history");
        si = host_input(
            g, g.ctx, GGML_TYPE_F32, {g.gdn_hd, g.gdn_hd, g.Hv_gdn, 1},
            static_cast<size_t>(g.gdn_hd) * g.gdn_hd * g.Hv_gdn *
                sizeof(float),
            "qwen_full_gdn_state");
      }
      if (!hi || !si)
        return false;
      if (g.decode_mode) {
        g.gdn_hist_updates.push_back(nullptr);
        g.gdn_state_updates.push_back(nullptr);
      } else {
        g.gdn_hist_inputs.push_back(static_cast<float *>(hi->data));
        g.gdn_state_inputs.push_back(static_cast<float *>(si->data));
        if (l < static_cast<int>(g.engine->qwen4exp_conv_history.size()) &&
            g.engine->qwen4exp_conv_history[l].size() ==
                static_cast<size_t>(hist) * g.conv_dim)
          std::memcpy(hi->data, g.engine->qwen4exp_conv_history[l].data(),
                      static_cast<size_t>(hist) * g.conv_dim * sizeof(float));
        if (l < static_cast<int>(g.engine->qwen4exp_recur_state.size()) &&
            g.engine->qwen4exp_recur_state[l].size() ==
                static_cast<size_t>(g.gdn_hd) * g.gdn_hd * g.Hv_gdn)
          // qwen4exp_recur_state is host-legacy layout; si->data feeds the
          // ggml op directly and needs native layout. See
          // gdn_state_transpose_host.
          gdn_state_transpose_host(static_cast<float *>(si->data),
                                   g.engine->qwen4exp_recur_state[l].data(),
                                   g.gdn_hd, g.Hv_gdn);
      }
      ggml_tensor *so = nullptr, *ho = nullptr;
      attn_out = build_gdn(g, g.ctx, attn_in, l, qkv, zg, al, be, dt, aa, cv,
                           nm, oo, hi, si, &so, &ho);
      if (g.decode_mode) {
        g.gdn_state_updates.back() =
            ggml_cpy(g.ctx, so, g.gdn_state_storage.back());
        g.gdn_hist_updates.back() =
            ggml_cpy(g.ctx, ho, g.gdn_hist_storage.back());
      } else {
        g.gdn_state_out.push_back(so);
        g.gdn_hist_out.push_back(ho);
      }
      g.gdn_layers.push_back(l);
    } else {
      ggml_tensor *wq = dense_weight(g, g.ctx, l, "attn_q.weight");
      ggml_tensor *wk = dense_weight(g, g.ctx, l, "attn_k.weight");
      ggml_tensor *wv = dense_weight(g, g.ctx, l, "attn_v.weight");
      ggml_tensor *wqn = dense_weight(g, g.ctx, l, "attn_q_norm.weight");
      ggml_tensor *wkn = dense_weight(g, g.ctx, l, "attn_k_norm.weight");
      ggml_tensor *wo = dense_weight(g, g.ctx, l, "attn_output.weight");
      if (!wq || !wk || !wv || !wqn || !wkn || !wo)
        return false;
      if (g.decode_mode) {
        ggml_tensor *k_cache = new_device_state_tensor(
            g, GGML_TYPE_F32, {g.hd, g.kv_capacity * g.Hkv},
            "qwen_full_decode_kv_k");
        ggml_tensor *v_cache = new_device_state_tensor(
            g, GGML_TYPE_F32, {g.hd, g.kv_capacity * g.Hkv},
            "qwen_full_decode_kv_v");
        if (!k_cache || !v_cache)
          return false;
        g.kv_state_k.push_back(k_cache);
        g.kv_state_v.push_back(v_cache);
        attn_out = build_full_attention_decode(
            g, g.ctx, attn_in, l, wq, wk, wv, wqn, wkn, wo, k_cache, v_cache);
      } else {
        ggml_tensor *k_cache = nullptr;
        ggml_tensor *v_cache = nullptr;
        attn_out = build_full_attention(g, g.ctx, attn_in, l, wq, wk, wv, wqn,
                                        wkn, wo, &k_cache, &v_cache);
        if (!k_cache || !v_cache)
          return false;
        g.kv_k.push_back(k_cache);
        g.kv_v.push_back(v_cache);
      }
      g.kv_layers.push_back(l);
    }
    if (l == 0 && g.layer0_diag_enabled &&
        env_layer0_split_dump_path()) {
      g.layer0_split_snapshots.emplace_back(
          "attention_output",
          layer0_diag_snapshot(g, g.ctx, attn_out,
                               "qwen_layer0_split_attention_output"));
    }
    res = hc_combine(g, g.ctx, res, attn_out, inject);
    if (g.two_layer_diag && l == 1)
      g.two_layer_l1_post_attn = g.layer0_diag_enabled
                                    ? layer0_diag_snapshot(
                                          g, g.ctx, res,
                                          "qwen_two_layer_s1_l1_post_attn")
                                    : res;
    if (l == 1 && g.layer0_diag_enabled &&
        env_layer0_split_dump_path()) {
      g.layer0_split_snapshots.emplace_back(
          "layer1_after_attention",
          layer0_diag_snapshot(g, g.ctx, res,
                               "qwen_layer0_split_layer1_after_attention"));
    }
    if (l == 0 && g.layer0_diag_enabled &&
        env_layer0_split_dump_path()) {
      g.layer0_split_snapshots.emplace_back(
          "after_attention",
          layer0_diag_snapshot(g, g.ctx, res,
                               "qwen_layer0_split_after_attention"));
    }
    g.attn_construction_ms_total += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t_attn_start).count();
    } // !phase2_suffix_skip_attn

    const auto t_moe_start = std::chrono::steady_clock::now();
    inject = nullptr;
    hn = resident_tensor(g, g.ctx, p + "hc_ffn_norm.weight");
    hdn = resident_tensor(g, g.ctx, p + "hc_ffn_down.weight");
    hup = resident_tensor(g, g.ctx, p + "hc_ffn_up.weight");
    hinj = resident_tensor(g, g.ctx, p + "hc_ffn_inject.weight");
    if (!hn || !hdn || !hup || !hinj)
      return false;
    ggml_tensor *ffn_in =
        hc_mix(g, g.ctx, res, hn, hdn, hup, hinj, &inject, l, "ffn");
    if (l == 0 && g.layer0_diag_enabled &&
        env_layer0_split_dump_path()) {
      g.layer0_split_snapshots.emplace_back(
          "ffn_input",
          layer0_diag_snapshot(g, g.ctx, ffn_in,
                               "qwen_layer0_split_ffn_input"));
    }
    if (l == 0 && g.layer0_diag_enabled) {
      g.layer0_diag_residual_before = res;
      g.layer0_diag_residual_before_snapshot = layer0_diag_snapshot(
          g, g.ctx, res, "qwen_layer0_diag_residual_before");
    }
    ggml_tensor *router = resident_tensor(g, g.ctx, p + "ffn_gate_inp.weight");
    ggml_tensor *bias = resident_tensor(g, g.ctx, p + "exp_probs_b.bias");
    if (!router)
      return false;
    if (!g.decode_mode && l == g.engine->qwen4exp_partial_layer && bias)
      g.partial_layer_bias_out = bias;

    if (g.phase2_mode == FullGraph::Phase2Mode::Prefix &&
        l == g.phase2_split_layer) {
      // This IS the split layer, and this graph is the Phase 2 prefix:
      // compute routing only (ids/probs, for the host's per-token
      // GPU-vs-CPU decision) and stop building entirely -- the split
      // layer's real FFN, and everything after it, is built exactly
      // once, in whichever suffix graph the host picks. `res` here is
      // the post-attention, pre-FFN residual for this layer; hand it off
      // through a persistent device buffer the chosen suffix graph binds
      // a fresh leaf to at the same address (see build_graph's own
      // res-seeding branch for SuffixGpu/SuffixCpu, and
      // split_graph_spike.cpp for the validated mechanism).
      build_moe(g, g.ctx, ffn_in, l, router, bias, nullptr, nullptr, nullptr,
                nullptr, nullptr, nullptr, nullptr, nullptr,
                &g.partial_layer_ids_out, &g.partial_layer_probs_out,
                /*routing_only=*/true);
      if (!g.partial_layer_ids_out || !g.partial_layer_probs_out)
        return false;
      ggml_set_output(g.partial_layer_ids_out);
      ggml_build_forward_expand(g.graph, g.partial_layer_ids_out);
      ggml_set_output(g.partial_layer_probs_out);
      ggml_build_forward_expand(g.graph, g.partial_layer_probs_out);

      ggml_backend_buffer_type_t cuda_buft =
          ggml_backend_get_default_buffer_type(g.cuda);
      g.phase2_handoff_buf = ggml_backend_buft_alloc_buffer(
          cuda_buft,
          static_cast<size_t>(g.D) * g.HC * g.S * sizeof(float));
      if (!g.phase2_handoff_buf)
        return false;
      g.phase2_handoff_addr =
          ggml_backend_buffer_get_base(g.phase2_handoff_buf);
      ggml_tensor *handoff_out =
          ggml_new_tensor_3d(g.ctx, GGML_TYPE_F32, g.D, g.HC, g.S);
      ggml_backend_tensor_alloc(g.phase2_handoff_buf, handoff_out,
                                g.phase2_handoff_addr);
      ggml_set_name(handoff_out, "qwen_full_phase2_handoff_out");
      ggml_tensor *handoff_cpy = ggml_cpy(g.ctx, res, handoff_out);
      ggml_set_output(handoff_cpy);
      ggml_build_forward_expand(g.graph, handoff_cpy);

      g.final_streams = res;
      g.output = nullptr;
      g.logits = nullptr;
      g.moe_construction_ms_total += std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - t_moe_start).count();
      phase2_prefix_stopped = true;
      break; // Prefix's job stops here -- do not build layer l+1 onward.
    }

    // The per-expert split path still needs separate gate/up tensors because
    // its two branches have different expert-axis remaps.  For ordinary
    // whole-layer CPU/GPU placement, prefer the adjacent gate+up view: on
    // graph-RAM layers this is zero-copy and reduces three routed matmuls to
    // two, matching Bee's reference graph.
    const auto split_it = g.split_gpu_expert_ids.find(l);
    const bool split_planned =
        g.S == 1 && split_it != g.split_gpu_expert_ids.end() &&
        !split_it->second.empty();
    ggml_tensor *moe = nullptr;

    // Phase 1 "all-or-nothing" partial-layer dispatch: only ever takes over
    // for the ONE configured layer, and only once a prior prefill pass's
    // real routing confirmed (via qwen4exp_full_graph_run's coverage check)
    // that this session may commit to the GPU-only compact path. See
    // build_moe_partial_gpu's own comment for why this is a separate
    // function rather than a generalization of the split_gpu_expert_ids
    // branch build_moe already has.
    //
    // Phase 2's SuffixGpu instance reuses this exact same compact-tensor
    // path for the identical layer/expert-set, but its trigger is NOT the
    // Phase 1 session-level commit flag -- the host already made a real
    // PER-TOKEN decision (this token's own routing, read back from the
    // prefix graph, fit entirely inside qwen4exp_partial_layer_gpu_ids)
    // before ever choosing to submit this graph at all. Being asked to
    // build at all IS the decision for this instance.
    const bool phase2_suffix_gpu_here =
        g.phase2_mode == FullGraph::Phase2Mode::SuffixGpu &&
        l == g.phase2_split_layer;
    if ((g.decode_mode && l == g.engine->qwen4exp_partial_layer &&
         g.engine->qwen4exp_partial_layer_committed &&
         !g.engine->qwen4exp_partial_layer_gpu_ids.empty()) ||
        (phase2_suffix_gpu_here &&
         !g.engine->qwen4exp_partial_layer_gpu_ids.empty())) {
      const std::string gname = p + "ffn_gate_exps.weight";
      const std::string uname = p + "ffn_up_exps.weight";
      const std::string dname = p + "ffn_down_exps.weight";
      const GGUFTensor *gm = model_tensor(*g.engine, gname);
      const GGUFTensor *um = model_tensor(*g.engine, uname);
      const GGUFTensor *dm = model_tensor(*g.engine, dname);
      ggml_tensor *pg =
          gm ? build_split_gpu_tensor(g, g.ctx, gname, *gm,
                                      g.engine->qwen4exp_partial_layer_gpu_ids)
             : nullptr;
      ggml_tensor *pu =
          um ? build_split_gpu_tensor(g, g.ctx, uname, *um,
                                      g.engine->qwen4exp_partial_layer_gpu_ids)
             : nullptr;
      ggml_tensor *pd =
          dm ? build_split_gpu_tensor(g, g.ctx, dname, *dm,
                                      g.engine->qwen4exp_partial_layer_gpu_ids)
             : nullptr;
      if (pg && pu && pd) {
        ExpertSplitTables tab = build_expert_split_tables(
            g, g.ctx, c.n_experts, g.engine->qwen4exp_partial_layer_gpu_ids);
        if (tab.remap_gpu)
          moe = build_moe_partial_gpu(g, g.ctx, ffn_in, l, router, bias, pg,
                                      pu, pd, tab.remap_gpu);
      }
      if (!moe)
        LOG_WARN("[PARTIAL-LAYER] layer=%d: compact GPU tensor construction "
                 "failed this run (RAM arena not ready?) -- falling back to "
                 "normal dispatch for this layer\n",
                 l);
    }

    // Prefill-only: capture this layer's routing `ids` for a real per-prompt
    // readback in qwen4exp_full_graph_run, so decode can decide whether to
    // commit to the path above. Never set outside prefill.
    ggml_tensor **partial_ids_out =
        (!g.decode_mode && l == g.engine->qwen4exp_partial_layer)
            ? &g.partial_layer_ids_out
            : nullptr;
    ggml_tensor **partial_probs_out =
        (!g.decode_mode && l == g.engine->qwen4exp_partial_layer)
            ? &g.partial_layer_probs_out
            : nullptr;

    // Decode-only router-usage tap: captures this layer's real routed ids
    // via build_moe's own ids_out param. Safe to fold into partial_ids_out's
    // slot -- usage_track_enabled implies g.decode_mode, and partial_ids_out
    // above is only ever non-null when !g.decode_mode, so the two never
    // collide.
    ggml_tensor *layer_ids_for_usage = nullptr;
    ggml_tensor **ids_out_param =
        usage_track_enabled ? &layer_ids_for_usage : partial_ids_out;

    if (!moe && !split_planned && env_merged_gate_up_enabled()) {
      const auto t_wlookup_start = std::chrono::steady_clock::now();
      const bool gate_up_device =
          graph_expert_projection_resident(g, l, "ffn_gate_exps.weight");
      ggml_tensor *gate_up =
          expert_gate_up_weight(g, g.ctx, l, gate_up_device);
      ggml_tensor *down = expert_weight(g, g.ctx, l,
                                        "ffn_down_exps.weight");
      g.moe_weight_lookup_ms_total +=
          std::chrono::duration<double, std::milli>(
              std::chrono::steady_clock::now() - t_wlookup_start)
              .count();
      if (gate_up && down) {
        moe = build_moe(g, g.ctx, ffn_in, l, router, bias, nullptr, nullptr,
                        down, nullptr, nullptr, nullptr, nullptr, gate_up,
                        ids_out_param, partial_probs_out);
      }
    }

    if (!moe) {
      const auto t_wlookup_start = std::chrono::steady_clock::now();
      SplitExpertWeight ge =
          expert_weight_split(g, g.ctx, l, "ffn_gate_exps.weight");
      SplitExpertWeight ue =
          expert_weight_split(g, g.ctx, l, "ffn_up_exps.weight");
      SplitExpertWeight de =
          expert_weight_split(g, g.ctx, l, "ffn_down_exps.weight");
      g.moe_weight_lookup_ms_total +=
          std::chrono::duration<double, std::milli>(
              std::chrono::steady_clock::now() - t_wlookup_start)
              .count();
      if (!ge.cpu || !ue.cpu || !de.cpu)
        return false;
      // v1 scope: uniform M across gate/up/down for a layer -- only treat
      // this as a real split if ALL THREE projections' gpu tensor succeeded
      // (expert_weight_split() already falls back to null on any failure).
      const bool split_ok = ge.gpu && ue.gpu && de.gpu;
      const std::vector<int> *split_ids =
          split_ok ? &g.split_gpu_expert_ids.at(l) : nullptr;
      moe = build_moe(g, g.ctx, ffn_in, l, router, bias, ge.cpu, ue.cpu,
                      de.cpu, split_ok ? ge.gpu : nullptr,
                      split_ok ? ue.gpu : nullptr,
                      split_ok ? de.gpu : nullptr, split_ids, nullptr,
                      ids_out_param, partial_probs_out);
    }
    if (usage_track_enabled && layer_ids_for_usage) {
      // Ordinary device-side accumulation -- ggml_concat inherits CUDA
      // placement from layer_ids_for_usage's argsort_top_k ancestor, no
      // manual backend pin needed (unlike the host-node CPY markers, this
      // is a totally normal graph op). ggml_cont matches the same
      // defensive-robustness convention build_qwen4exp_host_node's
      // route_ids already uses, even though decode S==1 rows never hit the
      // argsort-view-stride pitfall that convention guards against.
      ggml_tensor *ids_cont = ggml_cont(g.ctx, layer_ids_for_usage);
      g.usage_track_ids = g.usage_track_ids
                              ? ggml_concat(g.ctx, g.usage_track_ids, ids_cont, 0)
                              : ids_cont;
    }
    if (l == 0 && g.layer0_diag_enabled) {
      g.layer0_diag_moe_block = moe;
      if (env_layer0_split_dump_path()) {
        g.layer0_split_snapshots.emplace_back(
            "moe_routed",
            layer0_diag_snapshot(g, g.ctx, moe,
                                 "qwen_layer0_split_moe_routed"));
      }
    }
    ggml_tensor *sg =
        resident_tensor(g, g.ctx, p + "ffn_gate_inp_shexp.weight");
    ggml_tensor *sge = resident_tensor(g, g.ctx, p + "ffn_gate_shexp.weight");
    ggml_tensor *sup = resident_tensor(g, g.ctx, p + "ffn_up_shexp.weight");
    ggml_tensor *sdo = resident_tensor(g, g.ctx, p + "ffn_down_shexp.weight");
    if (sge && sup && sdo) {
      ggml_tensor *shared =
          build_shared(g, g.ctx, ffn_in, l, sge, sup, sdo, sg);
      if (l == 0 && g.layer0_diag_enabled)
        g.layer0_diag_shared = shared;
      if (l == 0 && g.layer0_diag_enabled &&
          env_layer0_split_dump_path()) {
        g.layer0_split_snapshots.emplace_back(
            "shared_output",
            layer0_diag_snapshot(g, g.ctx, shared,
                                 "qwen_layer0_split_shared_output"));
      }
      moe = ggml_add(g.ctx, moe, shared);
      if (l == 0 && g.layer0_diag_enabled)
        g.layer0_diag_moe_plus_shared = moe;
      if (l == 0 && g.layer0_diag_enabled)
        g.layer0_diag_moe_plus_shared_snapshot = layer0_diag_snapshot(
            g, g.ctx, moe, "qwen_layer0_diag_moe_plus_shared");
    }
    if (l == 0 && g.layer0_diag_enabled &&
        env_layer0_split_dump_path()) {
      g.layer0_split_snapshots.emplace_back(
          "ffn_output",
          layer0_diag_snapshot(g, g.ctx, moe,
                               "qwen_layer0_split_ffn_output"));
    }
    res = hc_combine(g, g.ctx, res, moe, inject);
    if (g.two_layer_diag && l == 0)
      g.two_layer_l0_post_ffn = g.layer0_diag_enabled
                                    ? layer0_diag_snapshot(
                                          g, g.ctx, res,
                                          "qwen_two_layer_s1_l0_post_ffn")
                                    : res;
    if (g.two_layer_diag && l == 1)
      g.two_layer_l1_post_ffn = g.layer0_diag_enabled
                                    ? layer0_diag_snapshot(
                                          g, g.ctx, res,
                                          "qwen_two_layer_s1_l1_post_ffn")
                                    : res;
    if (l == 0 && g.layer0_diag_enabled &&
        env_layer0_split_dump_path()) {
      g.layer0_split_snapshots.emplace_back(
          "after_ffn",
          layer0_diag_snapshot(g, g.ctx, res,
                               "qwen_layer0_split_after_ffn"));
    }
    if (env_hidden01_dump_path() && l >= 0 && l < 2 &&
        g.layer0_diag_enabled) {
      const std::string name =
          "qwen_hidden01_layer" + std::to_string(l);
      g.hidden01_snapshot[l] =
          layer0_diag_snapshot(g, g.ctx, res, name.c_str());
    }
    if (l == 0 && g.layer0_diag_enabled) {
      g.layer0_diag_residual_after = res;
      g.layer0_diag_residual_after_snapshot = layer0_diag_snapshot(
          g, g.ctx, res, "qwen_layer0_diag_residual_after");
    }
    g.moe_construction_ms_total += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t_moe_start).count();
  }

  // Decode-only router-usage tap: one flush marker for the whole graph,
  // built once here after every layer has appended into g.usage_track_ids
  // (see the per-layer ggml_concat block above and
  // Qwen4ExpUsageTrackContext's own comment). A construction failure here
  // is non-fatal -- it just means this session's usage histogram stays
  // incomplete, never that decode itself should fail.
  if (usage_track_enabled && g.usage_track_ids) {
    const int total_ids = c.topk * g.n_layers;
    auto usage_ctx = std::make_unique<Qwen4ExpUsageTrackContext>();
    usage_ctx->engine = g.engine;
    usage_ctx->n_layers = g.n_layers;
    usage_ctx->n_experts = c.n_experts;
    usage_ctx->topk = c.topk;
    usage_ctx->ids_bytes = static_cast<size_t>(total_ids) * sizeof(int32_t);
    usage_ctx->host_ids = static_cast<int32_t *>(
        alloc_registered_host(g, usage_ctx->ids_bytes));
    if (usage_ctx->host_ids) {
      std::fill_n(usage_ctx->host_ids, total_ids, 0);
      ggml_tensor *usage_dummy_dst =
          ggml_new_tensor_1d(g.ctx, GGML_TYPE_I32, total_ids);
      ggml_tensor *usage_marker =
          usage_dummy_dst ? ggml_cpy(g.ctx, g.usage_track_ids, usage_dummy_dst)
                          : nullptr;
      if (usage_marker) {
        usage_marker->extra = usage_ctx.get();
        ggml_set_name(usage_marker, "qwen_usage_track_flush_marker");
        if (g.sched) {
          ggml_backend_sched_set_tensor_backend(g.sched, usage_dummy_dst,
                                                g.cuda);
          ggml_backend_sched_set_tensor_backend(g.sched, usage_marker, g.cuda);
        }
        ggml_set_output(usage_marker);
        ggml_build_forward_expand(g.graph, usage_marker);
        g.usage_track_context = std::move(usage_ctx);
        LOG_INFO("[QWEN4EXP-USAGE-TRACK] flush marker captured: n_layers=%d "
                 "topk=%d total_ids=%d\n",
                 g.n_layers, c.topk, total_ids);
      } else {
        LOG_WARN("[QWEN4EXP-USAGE-TRACK] marker construction failed; usage "
                 "tracking disabled for this graph\n");
      }
    } else {
      LOG_WARN("[QWEN4EXP-USAGE-TRACK] pinned host buffer allocation "
               "failed; usage tracking disabled for this graph\n");
    }
  }

  if (!phase2_prefix_stopped) {
  ggml_tensor *on = resident_tensor(g, g.ctx, "output_hc_norm.weight");
  ggml_tensor *od = resident_tensor(g, g.ctx, "output_hc_down.weight");
  ggml_tensor *ou = resident_tensor(g, g.ctx, "output_hc_up.weight");
  if (!on || !od || !ou)
    return false;
  g.final_streams = res;
  g.output = hc_mix(g, g.ctx, res, on, od, ou, nullptr, nullptr);
  if (!g.decode_mode) {
    ggml_set_output(g.final_streams);
    ggml_build_forward_expand(g.graph, g.final_streams);
  }
  ggml_set_output(g.output);
  ggml_build_forward_expand(g.graph, g.output);
  if (g.decode_mode && !g.two_layer_diag) {
    ggml_tensor *lm_in = g.output;
    ggml_tensor *out_norm =
        resident_tensor(g, g.ctx, "output_norm.weight");
    if (out_norm) {
      lm_in = ggml_mul(
          g.ctx,
          ggml_rms_norm(g.ctx, lm_in,
                        g.engine->model_file.architecture_config.eps),
          out_norm);
    }
    ggml_tensor *out_weight =
        resident_tensor(g, g.ctx, "output.weight");
    if (!out_weight)
      return false;
    g.logits = ggml_mul_mat(g.ctx, out_weight, lm_in);
    ggml_set_output(g.logits);
    ggml_build_forward_expand(g.graph, g.logits);
  }
  } // !phase2_prefix_stopped
  if (g.two_layer_diag) {
    // Preserve the actual intermediate tensors as graph terminals. This is
    // the important distinction from the older diagnostic cpy snapshots:
    // GGML must keep these exact values live while the dependent Layer-1
    // subgraph executes, so allocator reuse cannot turn a snapshot into a
    // different tensor's bytes.
    for (ggml_tensor *t : {g.two_layer_l0_post_ffn,
                           g.two_layer_l1_pre_attn,
                           g.two_layer_l1_post_attn,
                           g.two_layer_l1_post_ffn}) {
      if (t) {
        ggml_set_output(t);
        ggml_build_forward_expand(g.graph, t);
      }
    }
  }
  if (!g.decode_mode) {
    for (ggml_tensor *t : g.kv_k) {
      ggml_set_output(t);
      ggml_build_forward_expand(g.graph, t);
    }
    for (ggml_tensor *t : g.kv_v) {
      ggml_set_output(t);
      ggml_build_forward_expand(g.graph, t);
    }
    for (ggml_tensor *t : g.gdn_state_out) {
      ggml_set_output(t);
      ggml_build_forward_expand(g.graph, t);
    }
    for (ggml_tensor *t : g.gdn_hist_out) {
      ggml_set_output(t);
      ggml_build_forward_expand(g.graph, t);
    }
    if (g.ple_hist_out) {
      ggml_set_output(g.ple_hist_out);
      ggml_build_forward_expand(g.graph, g.ple_hist_out);
    }
    if (g.partial_layer_ids_out) {
      ggml_set_output(g.partial_layer_ids_out);
      ggml_build_forward_expand(g.graph, g.partial_layer_ids_out);
    }
    if (g.partial_layer_probs_out) {
      ggml_set_output(g.partial_layer_probs_out);
      ggml_build_forward_expand(g.graph, g.partial_layer_probs_out);
    }
    if (g.partial_layer_bias_out) {
      ggml_set_output(g.partial_layer_bias_out);
      ggml_build_forward_expand(g.graph, g.partial_layer_bias_out);
    }
  } else {
    // State copies are graph nodes, but are intentionally not graph outputs:
    // decode keeps GDN/KV/PLE state on the CUDA backend between submissions.
    for (ggml_tensor *t : g.gdn_state_updates)
      if (t)
        ggml_build_forward_expand(g.graph, t);
    for (ggml_tensor *t : g.gdn_hist_updates)
      if (t)
        ggml_build_forward_expand(g.graph, t);
    if (g.ple_hist_update)
      ggml_build_forward_expand(g.graph, g.ple_hist_update);
  }
  // Real, measured incident (2026-09-09): ggml_backend_sched_alloc_graph's
  // internal ggml_gallocr_reserve_n_impl did not fail gracefully on a real
  // cudaMalloc OOM on beellama's older ggml fork -- it logged "cudaMalloc
  // failed: out of memory" and then segfaulted the whole process (exit 139)
  // instead of returning a checkable failure, at the exact point where
  // main.cpp's own fallback VRAM cache reservation (a fixed 4GB, unaware of
  // device_expert_layers' own VRAM use) left the scheduler's own reserve()
  // short by ~1.8GB. main.cpp now sizes that fallback cache against real
  // free VRAM and this family's own device_expert_layers estimate -- THAT
  // was the actual root cause and is already fixed there. This check stays
  // as a second, independent line of defense: refuse to even attempt the
  // allocator call when real free VRAM is below a floor, and decline to the
  // legacy path instead of risking the same crash again.
  //
  // 2026-09-15: tried twice to replace this flat constant with something
  // smarter -- both reverted, see below -- then settled on this smaller,
  // still-flat, but now EVIDENCE-BACKED value instead of an arbitrary one.
  // (1) Lowering 2 GiB -> 512 MiB: crashed at device_expert_layers=24
  // (needed >=650.24 MiB there -- the exact size of the single buffer
  // ggml_gallocr_reserve_n_impl tried and failed to cudaMalloc). (2)
  // Querying the REAL requirement via ggml_backend_sched_reserve_size() (a
  // safe, no_alloc=true measure pass that cannot itself OOM) reported
  // 0.640 GiB at N=22 -- consistent with N=24's 0.635 GiB failure point
  // above, i.e. the real requirement is roughly STABLE around ~0.64 GiB,
  // not scaling dramatically with layer count as first assumed from just
  // those two data points. But reusing that measure call's split_graph
  // pass on the SAME g.graph object that a real
  // ggml_backend_sched_alloc_graph() call re-splits moments later leaves
  // something behind that a plain ggml_backend_sched_reset() in between
  // does NOT fully clear: real-hardware repro, twice, with and without that
  // reset -- a hard `GGML_ASSERT(src0->type == src1->type)` abort inside
  // ggml_cuda_op_concat on the FIRST real graph compute afterward (prefill,
  // S=7). Do not retry live-measuring this way without either a genuinely
  // separate measure-graph object (llama.cpp's own convention -- never
  // reuse/re-split the exact cgraph that will actually execute) or a
  // from-scratch ggml_backend_sched for the real run after measuring on a
  // throwaway one.
  //
  // Given the two real measurements above both land at ~0.63-0.64 GiB,
  // 1 GiB keeps a real ~1.5-1.6x margin over observed need (vs 2 GiB's
  // ~3.1x, which cost every device_expert_layers config ~1 GiB of pure
  // waste) while recovering enough headroom for device_expert_layers=22
  // to fit (was declining with only 1.83 GiB free). Re-widen if a real
  // OOM/crash resurfaces at some untested N; don't lower below ~0.8 GiB
  // without a fresh measurement -- these two points aren't enough to prove
  // it never goes higher for a different N or model.
  {
    // The scheduler allocator is not a safe place to discover that a static
    // model weight lost its external binding: on older ggml-cuda revisions it
    // will reinterpret the whole unbound expert subtree as a CUDA workspace
    // and attempt an allocation roughly equal to the complete MoE payload
    // (~86 GiB for this model), then may crash instead of returning failure.
    // This can surface only after repeated prompt graph rebuilds, when a
    // stale backend assignment or buffer view is carried across teardown.
    int unbound_static_weights = 0;
    int misplaced_host_weights = 0;
    size_t unbound_static_bytes = 0;
    // GGML may wrap a resident weight in more than one view (for example
    // view -> contiguous -> reshaped for PLE convolution).  The outer view
    // does not necessarily carry the backend buffer itself; walk the full
    // chain before classifying a static tensor as unbound.  A one-hop check
    // falsely rejected these valid weights and sent every build to fallback.
    auto view_buffer = [](ggml_tensor *tensor) -> ggml_backend_buffer_t {
      for (int depth = 0; tensor && depth < 32; ++depth) {
        if (tensor->buffer)
          return tensor->buffer;
        tensor = tensor->view_src;
      }
      return nullptr;
    };
    for (int i = 0; i < ggml_graph_n_nodes(g.graph); ++i) {
      ggml_tensor *node = ggml_graph_node(g.graph, i);
      if (!node)
        continue;
      for (int j = 0; j < GGML_MAX_SRC; ++j) {
        ggml_tensor *src = node->src[j];
        if (!src)
          continue;
        const char *name = ggml_get_name(src);
        const std::string tensor_name = name ? name : "";
        // Operation results can inherit a source weight's display name
        // (the PLE path produces names such as "...weight (cont)" and
        // "...weight (reshaped)").  They are activation/intermediate
        // tensors, not static model-storage leaves, and are expected to be
        // allocated by the graph allocator.  Only GGML_OP_NONE tensors can
        // represent a missing pre-bound model weight here.
        if (src->op != GGML_OP_NONE)
          continue;
        const bool named_static_weight =
            tensor_name.find(".weight") != std::string::npos ||
            tensor_name.find(".bias") != std::string::npos ||
            tensor_name.find("_exps") != std::string::npos;
        if (!named_static_weight)
          continue;
        ggml_backend_buffer_t buffer = view_buffer(src);
        if (!buffer) {
          ++unbound_static_weights;
          if (unbound_static_bytes <= SIZE_MAX - ggml_nbytes(src))
            unbound_static_bytes += ggml_nbytes(src);
          if (unbound_static_weights <= 8)
            LOG_ERROR(
                "[GGML-QWEN-FULL] refusing graph allocation: unbound static "
                "weight name=%s bytes=%zu\n",
                tensor_name.c_str(), ggml_nbytes(src));
          continue;
        }
        if (ggml_backend_buffer_is_host(buffer) &&
            ggml_backend_sched_get_tensor_backend(g.sched, src) == g.cuda &&
            tensor_name.find("_exps") != std::string::npos) {
          ++misplaced_host_weights;
          if (misplaced_host_weights <= 8)
            LOG_ERROR(
                "[GGML-QWEN-FULL] refusing graph allocation: host expert "
                "weight assigned to CUDA name=%s\n",
                tensor_name.c_str());
        }
      }
    }
    if (unbound_static_weights > 0 || misplaced_host_weights > 0) {
      LOG_ERROR(
          "[GGML-QWEN-FULL] static binding validation failed: unbound=%d "
          "(%.2f GiB), misplaced_host_experts=%d; skipping unsafe allocator "
          "reserve\n",
          unbound_static_weights,
          static_cast<double>(unbound_static_bytes) /
              (1024.0 * 1024.0 * 1024.0),
          misplaced_host_weights);
      return false;
    }

    size_t free_b = 0, total_b = 0;
    int prev_device = -1;
    cudaGetDevice(&prev_device);
    cudaSetDevice(g.device);
    const cudaError_t err = cudaMemGetInfo(&free_b, &total_b);
    if (prev_device >= 0)
      cudaSetDevice(prev_device);
    // A Phase-2 suffix borrows the already committed dense/expert arenas,
    // so its only new CUDA allocation is the small scheduler activation
    // workspace.  The measured ordinary 22-layer graph uses 0.13 GiB; keep
    // 512 MiB of headroom for each borrowed suffix while retaining the
    // established 1-GiB admission floor for ordinary/full graph instances.
    const bool borrowed_phase2_suffix =
        g.phase2_mode == FullGraph::Phase2Mode::SuffixGpu ||
        g.phase2_mode == FullGraph::Phase2Mode::SuffixCpu;
    const size_t kMinGraphAllocHeadroomBytes =
        borrowed_phase2_suffix ? 512ull * 1024ull * 1024ull
                               : 1ull * 1024ull * 1024ull * 1024ull;
    if (err != cudaSuccess || free_b < kMinGraphAllocHeadroomBytes) {
      LOG_ERROR(
          "[GGML-QWEN-FULL] refusing graph allocator reserve: device %d "
          "free=%.2f GiB "
          "< %.2f GiB safety floor (device expert arena + dense arena already "
          "committed %.2f GiB) -- falling back to the legacy path\n",
          g.device,
          err == cudaSuccess
              ? static_cast<double>(free_b) / (1024.0 * 1024.0 * 1024.0)
              : -1.0,
          static_cast<double>(kMinGraphAllocHeadroomBytes) /
              (1024.0 * 1024.0 * 1024.0),
          static_cast<double>(g.device_expert_capacity +
                              g.device_dense_capacity) /
              (1024.0 * 1024.0 * 1024.0));
      return false;
    }
  }
  // Safe allocator-size probe for diagnosing unexpectedly large graph
  // workspaces. reserve_size() performs GGML's no_alloc sizing pass, but it
  // also rebuilds the scheduler's internal split graph. Do not reuse this
  // FullGraph for a real allocation/compute after probing; decline it and let
  // the owning prepare path destroy it. This is intentionally opt-in because
  // it disables the full-graph instance for the diagnostic run.
  const char *reserve_diag =
      std::getenv("MOESTREAM_QWEN4EXP_GRAPH_RESERVE_DIAG");
  const bool reserve_diag_enabled =
      reserve_diag &&
      ((std::strcmp(reserve_diag, "1") == 0 ||
        std::strcmp(reserve_diag, "all") == 0) ||
       (std::strcmp(reserve_diag, "decode") == 0 && g.S == 1));
  if (reserve_diag_enabled) {
    const int n_backends = ggml_backend_sched_get_n_backends(g.sched);
    std::vector<size_t> backend_sizes(static_cast<size_t>(n_backends), 0);
    ggml_backend_sched_reserve_size(g.sched, g.graph, backend_sizes.data());
    LOG_WARN("[GGML-QWEN-FULL-RESERVE-DIAG] no-allocation sizing only; "
             "declining this graph instance (splits=%d copies=%d)\n",
             ggml_backend_sched_get_n_splits(g.sched),
             ggml_backend_sched_get_n_copies(g.sched));
    for (int i = 0; i < n_backends; ++i) {
      ggml_backend_t backend = ggml_backend_sched_get_backend(g.sched, i);
      ggml_backend_buffer_type_t buft =
          ggml_backend_sched_get_buffer_type(g.sched, backend);
      const double gib = static_cast<double>(backend_sizes[i]) /
                         (1024.0 * 1024.0 * 1024.0);
      LOG_WARN("[GGML-QWEN-FULL-RESERVE-DIAG] backend=%s buffer=%s "
               "required=%.3f GiB (%zu bytes)\n",
               ggml_backend_name(backend), ggml_backend_buft_name(buft), gib,
               backend_sizes[i]);
    }
    return false;
  }
  const double t_construction_ms = std::chrono::duration<double, std::milli>(
                                       std::chrono::steady_clock::now() -
                                       t_build_graph_start)
                                       .count();
  const auto t_alloc_start = std::chrono::steady_clock::now();
  if (!ggml_backend_sched_alloc_graph(g.sched, g.graph))
    return false;
  const double t_alloc_ms = std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() -
                                t_alloc_start)
                                .count();
  LOG_INFO("[GGML-QWEN-FULL-BUILD-TIME] node_construction=%.1fms "
           "(of which dense_cudaMemcpy=%.1fms/%d tensors, "
           "expert_cudaMemcpy=%.1fms/%ld calls, attn_block=%.1fms, "
           "moe_block=%.1fms [weight_lookup=%.1fms]) sched_alloc_graph=%.1fms\n",
           t_construction_ms, g.dense_upload_ms_total,
           g.device_dense_tensors, g.expert_upload_ms_total,
           g.expert_upload_calls, g.attn_construction_ms_total,
           g.moe_construction_ms_total, g.moe_weight_lookup_ms_total,
           t_alloc_ms);
  g.graph_bytes = ggml_backend_sched_get_buffer_size(g.sched, g.cuda);
  dump_qwen_moe_layout(g);

  // The graph is only useful for performance if the scheduler keeps most of
  // the dependency chain on CUDA.  A high split count can otherwise hide a
  // CPU fallback for an op or a view/copy boundary, which recreates the
  // per-layer host orchestration that this graph is meant to remove.
  int cuda_nodes = 0;
  int cpu_nodes = 0;
  int other_nodes = 0;
  int backend_transitions = 0;
  ggml_backend_t previous_backend = nullptr;
  std::unordered_map<std::string, int> non_cuda_ops;
  for (int i = 0; i < ggml_graph_n_nodes(g.graph); ++i) {
    ggml_tensor *node = ggml_graph_node(g.graph, i);
    ggml_backend_t backend =
        ggml_backend_sched_get_tensor_backend(g.sched, node);
    if (backend == g.cuda) {
      ++cuda_nodes;
    } else if (backend == g.cpu) {
      ++cpu_nodes;
      ++non_cuda_ops[ggml_op_name(node->op)];
    } else {
      ++other_nodes;
      ++non_cuda_ops[backend ? ggml_backend_name(backend) : "NULL"];
    }
    if (previous_backend && backend != previous_backend)
      ++backend_transitions;
    previous_backend = backend;
  }
  LOG_INFO("[GGML-QWEN-FULL] scheduler nodes cuda=%d cpu=%d other=%d "
           "transitions=%d\n",
           cuda_nodes, cpu_nodes, other_nodes, backend_transitions);
  if (!non_cuda_ops.empty()) {
    std::string detail;
    for (const auto &entry : non_cuda_ops) {
      if (!detail.empty())
        detail += ", ";
      detail += entry.first + "=" + std::to_string(entry.second);
    }
    LOG_INFO("[GGML-QWEN-FULL] non-CUDA node ops: %s\n", detail.c_str());
  }
  LOG_INFO("[GGML-QWEN-FULL] graph built layers=%d nodes=%d splits=%d "
           "workspace=%.2f GiB\n",
           g.n_layers, ggml_graph_n_nodes(g.graph),
           ggml_backend_sched_get_n_splits(g.sched),
           static_cast<double>(g.graph_bytes) / (1024.0 * 1024.0 * 1024.0));
  LOG_INFO(
      "[GGML-QWEN-FULL] dense device arena used=%.2f/%.2f GiB; %d tensors\n",
      static_cast<double>(g.device_dense_used) / (1024.0 * 1024.0 * 1024.0),
      static_cast<double>(g.device_dense_capacity) / (1024.0 * 1024.0 * 1024.0),
      g.device_dense_tensors);
  if (g.device_expert_layers > 0) {
    const int complete_records = g.device_expert_layers * c.n_experts;
    LOG_INFO(
        "[GGML-QWEN-FULL] device expert arena used=%.2f/%.2f GiB; "
        "%d complete layer(s), %d expert records, all 3 projections resident\n",
        static_cast<double>(g.device_expert_used) / (1024.0 * 1024.0 * 1024.0),
        static_cast<double>(g.device_expert_capacity) /
            (1024.0 * 1024.0 * 1024.0),
        g.device_expert_layers, complete_records);
  }
  LOG_INFO(
      "[GGML-QWEN-FULL] host expert source=%.2f GiB graph-owned DDR copy; "
      "%.2f GiB aliased from preloaded DDR arena (%zu projection views)\n",
      static_cast<double>(g.host_weight_bytes) / (1024.0 * 1024.0 * 1024.0),
      static_cast<double>(g.host_ram_alias_bytes) / (1024.0 * 1024.0 * 1024.0),
      g.host_ram_alias_tensors);
  if (g.host_weight_bytes > 0) {
    LOG_INFO("[GGML-QWEN-FULL] host expert source pinned=%.2f GiB; "
             "pin failures=%zu\n",
             static_cast<double>(g.host_pinned_bytes) /
                 (1024.0 * 1024.0 * 1024.0),
             g.host_pin_failures);
  }
  LOG_INFO("[GGML-QWEN-FULL-MEM] host ledger=%.2f/%.2f GB; graph source="
           "%.2f GiB + context=%.2f MiB + owned-extra=%.2f MiB; device "
           "workspace=%.2f GiB (GPU, outside host_ram_gb)\n",
           (double)MemoryGuard::reserved_host_bytes() / 1.0e9,
           (double)MemoryGuard::host_budget_bytes() / 1.0e9,
           (double)g.host_weight_bytes / (1024.0 * 1024.0 * 1024.0),
           (double)g.context_reservation_bytes / (1024.0 * 1024.0),
           (double)(g.host_weight_bytes > g.host_reservation_bytes
                        ? g.host_weight_bytes - g.host_reservation_bytes
                        : 0) /
               (1024.0 * 1024.0),
           (double)g.graph_bytes / (1024.0 * 1024.0 * 1024.0));
  g.ready = true;
  return true;
}

bool sync_decode_state_to_host(FullGraph &g) {
  if (!g.decode_mode || !g.ready || !g.engine)
    return true;
  StreamingEngine &engine = *g.engine;
  const size_t state_n = static_cast<size_t>(g.gdn_hd) * g.gdn_hd * g.Hv_gdn;
  const size_t hist_n = static_cast<size_t>(g.d_conv - 1) * g.conv_dim;
  for (size_t i = 0; i < g.gdn_layers.size(); ++i) {
    const int l = g.gdn_layers[i];
    if (i >= g.gdn_state_storage.size() || i >= g.gdn_hist_storage.size() ||
        l < 0 || l >= static_cast<int>(engine.qwen4exp_recur_state.size()) ||
        l >= static_cast<int>(engine.qwen4exp_conv_history.size()) ||
        engine.qwen4exp_recur_state[l].size() != state_n ||
        engine.qwen4exp_conv_history[l].size() != hist_n)
      return false;
    // gdn_state_storage is native layout (it feeds the decode self-loop
    // directly); qwen4exp_recur_state is host-legacy layout for legacy CPU
    // consumption. See gdn_state_transpose_host.
    std::vector<float> native_state(state_n);
    ggml_backend_tensor_get(g.gdn_state_storage[i], native_state.data(), 0,
                            state_n * sizeof(float));
    gdn_state_transpose_host(engine.qwen4exp_recur_state[l].data(),
                             native_state.data(), g.gdn_hd, g.Hv_gdn);
    ggml_backend_tensor_get(g.gdn_hist_storage[i],
                            engine.qwen4exp_conv_history[l].data(), 0,
                            hist_n * sizeof(float));
  }
  const size_t row = static_cast<size_t>(g.Hkv) * g.hd;
  const int rows = std::min(std::max(0, g.decode_last_pos + 1), g.kv_capacity);
  for (size_t i = 0; i < g.kv_layers.size(); ++i) {
    const int l = g.kv_layers[i];
    if (i >= g.kv_state_k.size() || i >= g.kv_state_v.size() || l < 0 ||
        l >= static_cast<int>(engine.Kc.size()) ||
        l >= static_cast<int>(engine.Vc.size()) || !engine.Kc[l] ||
        !engine.Vc[l])
      return false;
    const size_t bytes = static_cast<size_t>(rows) * row * sizeof(float);
    if (bytes > 0) {
      ggml_backend_tensor_get(g.kv_state_k[i], engine.Kc[l], 0, bytes);
      ggml_backend_tensor_get(g.kv_state_v[i], engine.Vc[l], 0, bytes);
    }
  }
  if (g.ple_layer >= 0 && g.ple_hist_storage) {
    const auto &c = engine.model_file.architecture_config;
    const size_t ple_n = static_cast<size_t>(c.ple_conv_kernel - 1) *
                         c.ple_ngram_size * g.HC * g.D;
    if (g.ple_layer >= static_cast<int>(engine.qwen4exp_ple_conv_history.size()) ||
        engine.qwen4exp_ple_conv_history[g.ple_layer].size() != ple_n)
      return false;
    ggml_backend_tensor_get(g.ple_hist_storage,
                            engine.qwen4exp_ple_conv_history[g.ple_layer].data(),
                            0, ple_n * sizeof(float));
  }
  return true;
}

} // namespace

bool qwen4exp_full_graph_enabled() { return env_enabled(); }

int qwen4exp_full_graph_device_expert_layers() {
  return env_enabled() ? env_device_expert_layers() : 0;
}

bool qwen4exp_full_graph_device_expert_trailing() {
  return env_enabled() && env_device_expert_trailing();
}

bool qwen4exp_full_graph_decode_enabled() {
  return env_enabled() && env_decode_enabled();
}

void qwen4exp_full_graph_configure(bool enabled, bool decode_enabled) {
  g_config_full_graph = enabled;
  g_config_full_graph_decode = decode_enabled;
}

void qwen4exp_full_graph_shutdown() {
  // Explicit, EARLY teardown (before the CUDA driver itself starts
  // shutting down) is why this function exists at all -- see g_graph's
  // own destroy()'s comment. The two Phase 2 suffix globals are exactly
  // as CUDA-backed as g_graph and must be reset here too, in this order
  // (suffix instances first): SuffixCpu's GDN/KV/PLE state tensors for
  // layers > split_layer are aliases into SuffixGpu's own
  // device_state_buffer (see FullGraph::phase2_state_source_graph), so
  // SuffixGpu must outlive SuffixCpu's teardown.
  g_graph_p2_suffix_cpu.reset();
  g_graph_p2_suffix_gpu.reset();
  g_graph.reset();
  g_two_layer_diag_graph.reset();
  g_two_layer_diag_done = false;
}

bool qwen4exp_full_graph_prepare(StreamingEngine &engine, int S, int pos_base,
                                 const int *token_ids) {
  if (!env_enabled() || S <= 1 || pos_base != 0 ||
      engine.family_ops.family != ModelFamily::Qwen4Exp)
    return false;
  // CPUOffload's prefill graph still uses the established scheduler CPU
  // branch. The host-node bridge is decode-only because its callback is a
  // single-token strided CPU contract; prefill must retain the ordinary
  // batched MUL_MAT_ID path to seed recurrent/KV state correctly. Decode is
  // allowed below only when every configured offload layer is replaced by a
  // CUDA-visible host marker.
  if (engine.qwen4exp_cpu_offload &&
      !engine.qwen4exp_cpu_offload_layers.empty() &&
      !env_qwen4exp_host_node_enabled()) {
    LOG_WARN(
        "[GGML-QWEN-FULL] disabled: CPUOffload layers are configured, but "
        "the opt-in cudaHostNode bridge is disabled; use "
        "the tested layered CPUOffload path instead\n");
    return false;
  }
  const auto &c = engine.model_file.architecture_config;
  if (engine.qwen_ram_preload_all_experts) {
    const bool narrow_layer0_diag = env_layer0_narrow_graph_diag_enabled();
    for (int layer = 0; layer < c.n_layers; ++layer) {
      if (narrow_layer0_diag && layer != 0)
        continue;
      if (!engine.graph_ram_layer_ready(layer)) {
        LOG_WARN("[GGML-QWEN-FULL] disabled: ram_preload_all_experts "
                 "requires a ready DDR arena for every layer (missing "
                 "layer=%d); no GGUF mmap fallback\n",
                 layer);
        return false;
      }
    }
  }
  const int dev = engine.gpu_device_id >= 0 ? engine.gpu_device_id : 0;
  if (g_graph && g_graph->ready && g_graph->engine == &engine &&
      g_graph->S == S) {
    // PLE rows are request data, so refresh that input when the graph shape
    // is reused for a different prompt.  The model graph itself remains
    // persistent, exactly like Bee's shape cache.
    if (has_ple_layer(c)) {
      std::vector<float> ple;
      if (!prepare_ple_embedding(*g_graph, ple, S, pos_base, token_ids) ||
          !g_graph->ple_embedding ||
          ple.size() * sizeof(float) > ggml_nbytes(g_graph->ple_embedding))
        return false;
      std::memcpy(g_graph->ple_embedding->data, ple.data(),
                  ple.size() * sizeof(float));
    }
    return true;
  }
  // Phase-2 suffix graphs borrow state, dense, and expert storage from the
  // previous prefix graph.  Destroy them before replacing that owner;
  // otherwise a persistent multi-prompt run can leave suffix instances with
  // dangling ggml/CUDA buffer views during the next rebuild.
  g_graph_p2_suffix_cpu.reset();
  g_graph_p2_suffix_gpu.reset();
  g_graph.reset();
  auto g = std::make_unique<FullGraph>();
  g->engine = &engine;
  g->device = dev;
  g->S = S;
  g->D = c.hidden;
  g->HC = c.hc_mult;
  g->n_layers = c.n_layers;
  g->H = c.n_heads;
  g->Hkv = std::max(1, c.n_head_kv);
  g->hd = c.head_dim;
  g->Hk_gdn = c.ssm_group_count;
  g->Hv_gdn = c.ssm_time_step_rank;
  g->gdn_hd = c.ssm_state_size;
  g->key_dim = g->Hk_gdn * g->gdn_hd;
  g->value_dim = c.ssm_inner_size;
  g->conv_dim = 2 * g->key_dim + g->value_dim;
  g->d_conv = c.ssm_conv_kernel;
  g->device_expert_layers =
      std::min(qwen4exp_full_graph_device_expert_layers(), c.n_layers);
  g->device_expert_trailing = env_device_expert_trailing();
  const char *partial_gate_up =
      std::getenv("MOESTREAM_QWEN4EXP_GRAPH_PRELOAD_PARTIAL_GATE_UP");
  // partial_gate_up's boundary math (layer==device_expert_layers, the
  // first CPU layer just past a LEADING block) has no trailing-mode
  // equivalent yet -- force it off rather than apply leading-only math
  // to a trailing block. See device_expert_trailing's own comment.
  g->device_expert_partial_gate_up = g->device_expert_layers > 0 &&
                                     !g->device_expert_trailing &&
                                     partial_gate_up &&
                                     partial_gate_up[0] == '1';
  plan_expert_vram_split(*g); // must run before init_device_expert_arena
  plan_partial_layer(engine); // Phase 1 all-or-nothing partial-layer dispatch
  g->cuda = static_cast<ggml_backend_t>(ggml_shared_backend(dev));
  g->cpu = ggml_backend_cpu_init();
  if (!g->cuda || !g->cpu)
    return false;
  verify_cuda_graphs_compiled_in(g->cuda);
  g->cpu_threads = graph_cpu_threads();
  ggml_backend_cpu_set_n_threads(g->cpu, g->cpu_threads);
  ggml_threadpool_params cpu_tp_params =
      ggml_threadpool_params_default(g->cpu_threads);
  g->cpu_threadpool = ggml_threadpool_new(&cpu_tp_params);
  if (g->cpu_threadpool)
    ggml_backend_cpu_set_threadpool(g->cpu, g->cpu_threadpool);
  LOG_INFO("[GGML-QWEN-FULL] CPU backend configured with %d threads%s\n",
           g->cpu_threads,
           g->cpu_threadpool ? " (persistent threadpool)" : "");

  // Admit the graph-owned expert source set once, before any of its
  // individual host buffers are allocated.  Layers with a complete
  // preloaded graph-RAM arena are excluded: their expert planes are aliased
  // directly from the existing DDR cache below, so this graph does not make
  // a second copy.  Any remaining layers use one graph-owned DDR copy rather
  // than a decode-time GGUF mmap source.  The extra watermark covers the
  // GGML host context, prompt/state inputs, and allocator/I/O fluctuation;
  // no /proc/meminfo read is performed in the per-buffer allocation loop.
  const size_t expert_bytes = full_expert_source_bytes(engine);
  // 10 GiB, per explicit user instruction after the 2026-09-09 crash (was
  // 4 GiB -- the smallest, least-conservative watermark of any admission
  // check in either graph file; aligned to the same 10 GiB floor as the
  // rest).
  constexpr size_t kGraphHostWatermark = 10ull * 1024ull * 1024ull * 1024ull;
  if (expert_bytes > 0 &&
      !MemoryGuard::try_reserve_host_bytes(expert_bytes, kGraphHostWatermark)) {
    LOG_WARN("[GGML-QWEN-FULL] host watermark refused graph source=%.2f GiB + "
             "%.2f GiB reserve; "
             "MemAvailable=%ld MiB\n",
             static_cast<double>(expert_bytes) / (1024.0 * 1024.0 * 1024.0),
             static_cast<double>(kGraphHostWatermark) /
                 (1024.0 * 1024.0 * 1024.0),
             MemoryGuard::last_available_mb());
    return false;
  }
  g->host_reservation_bytes = expert_bytes;
  g->host_precharged_bytes = expert_bytes;
  LOG_INFO("[GGML-QWEN-FULL] host source admission: graph-owned=%.2f GiB + "
           "%.2f GiB reserve; "
           "complete preloaded DDR arenas are aliased directly\n",
           static_cast<double>(expert_bytes) / (1024.0 * 1024.0 * 1024.0),
           static_cast<double>(kGraphHostWatermark) /
               (1024.0 * 1024.0 * 1024.0));
  if (!init_device_dense_arena(*g))
    return false;
  // Reserve fixed dense weights before choosing the variable-size expert
  // arena. Otherwise an oversized all-GPU request can consume the remaining
  // VRAM with experts, appear to fit after clamping, and then fail here when
  // the dense arena is allocated.
  if (!init_device_expert_arena(*g))
    return false;
  ggml_backend_t backends[] = {g->cuda, g->cpu};
  ggml_backend_buffer_type_t sched_bufts[2] = {};
  ggml_backend_buffer_type_t *sched_bufts_ptr = nullptr;
  if (graph_cuda_host_buffers()) {
    sched_bufts[0] = ggml_backend_get_default_buffer_type(g->cuda);
    sched_bufts[1] = ggml_backend_dev_host_buffer_type(
        ggml_backend_get_device(g->cuda));
    if (sched_bufts[1]) {
      sched_bufts_ptr = sched_bufts;
      LOG_INFO("[GGML-QWEN-FULL] scheduler CPU buffer=%s (CUDA host buffer)\n",
               ggml_backend_buft_name(sched_bufts[1]));
    }
  }
  // Match Bee's pipeline_parallel=false scheduler mode by default.  The
  // graph's CPU/GPU transitions are dependency-serial; the event/four-copy
  // mode is retained as an explicit opt-in for comparison.
  //
  // op_offload MUST stay false here. When true,
  // ggml_backend_sched_backend_id_from_cur's host-weight-offload heuristic
  // (ggml-backend.cpp, ~line 1178) can override a host-resident MoE expert
  // weight's otherwise-correct CPU placement and push its consuming
  // MUL_MAT_ID op onto CUDA instead -- but ONLY for S>=32 (ggml's own
  // ggml_backend_offload_op prefill-scale gate), so it is invisible in any
  // S=1 decode test. Confirmed on real hardware: a 26-host-layer /
  // 22-device-layer full-graph prefill crashed with a genuine CUDA OOM
  // trying to cudaMalloc ~84-90 GiB (approximately the complete host-layer
  // expert footprint) the moment a prompt's prefill token count reached 32
  // (38-token prompt crashed; 22/24/30-token prompts on the same config did
  // not) -- exactly the failure mode bind_host_expert_scheduler_backend's
  // own comment already warns about, and exactly what op_offload=false was
  // chosen to prevent (see memory qwen4exp-mixed-gpu-decode-gap.md: "risks
  // the exact host-weight/strided-DDR corruption... for any future S>=32
  // codepath reusing this scheduler"). Device-resident expert layers are
  // unaffected either way -- their tensors are already explicitly bound to
  // the CUDA arena, independent of this flag.
  g->sched =
      ggml_backend_sched_new(backends, sched_bufts_ptr, 2, kGraphSize,
                             graph_scheduler_parallel(), false);
  if (!g->sched)
    return false;
  LOG_INFO("[GGML-QWEN-FULL] scheduler pipeline_parallel=%s copies=%d\n",
           graph_scheduler_parallel() ? "true" : "false",
           ggml_backend_sched_get_n_copies(g->sched));
  // Replay timing must not install ggml's eval callback.  The callback path
  // decomposes CUDA graph work into per-node submissions and therefore
  // changes the execution being measured.  Keep the invasive per-node
  // timing behind its explicit switch; replay timing itself is collected
  // around the scheduler submit/synchronize boundary below.
  g->layer_timing_enabled = env_layer_timing_enabled();
  g->layer_timing_verbose_enabled = env_layer_timing_verbose();
  if (g->layer_timing_enabled || g->layer_timing_verbose_enabled) {
    g->layer_timing_cpu_ms.assign(static_cast<size_t>(g->n_layers), 0.0);
    ggml_backend_sched_set_eval_callback(g->sched, layer_timing_eval_callback,
                                         g.get());
  }

  // The model graph starts from the already expanded HC embedding.  PLE's
  // hash gather is prepared once here, outside the measured prefill window;
  // learned PLE projections and its causal convolution remain graph nodes.
  std::vector<float> ple;
  if (!prepare_ple_embedding(*g, ple, S, pos_base, token_ids) &&
      has_ple_layer(c)) {
    LOG_WARN("[GGML-QWEN-FULL] PLE embedding preparation declined\n");
    return false;
  }
  if (!build_graph(*g, has_ple_layer(c) ? &ple : nullptr)) {
    LOG_WARN("[GGML-QWEN-FULL] graph construction declined; using fallback\n");
    return false;
  }
  g_graph = std::move(g);
  return true;
}

bool qwen4exp_full_graph_decode_prepare(StreamingEngine &engine, int pos_base,
                                        const int *token_ids) {
  if (!qwen4exp_full_graph_decode_enabled() || pos_base < 0 ||
      engine.family_ops.family != ModelFamily::Qwen4Exp)
    return false;
  // Keep the same fail-closed rule as prefill.  A full-graph decode call
  // takes precedence over the layered token dispatcher, so allowing this
  // combination would make the configured CPUOffload layers disappear from
  // the actual execution path rather than run through Milestone B.
  if (engine.qwen4exp_cpu_offload &&
      !engine.qwen4exp_cpu_offload_layers.empty() &&
      !env_qwen4exp_host_node_enabled()) {
    LOG_WARN(
        "[GGML-QWEN-FULL-DECODE] disabled: CPUOffload layers are configured, "
        "but the opt-in cudaHostNode bridge is disabled; falling back "
        "to layered CPUOffload decode\n");
    return false;
  }
  const auto &c = engine.model_file.architecture_config;
  if (engine.max_t <= 0 || pos_base >= engine.max_t)
    return false;

  // The decode graph seeds recurrent/KV state from the graph's validated
  // prefill writeback.  Do not silently combine it with the existing layered
  // GPU attention/GDN state owners; that would create two divergent state
  // stores.  A later capacity growth is allowed because it first downloads
  // the current decode graph state below.
  const bool continuing_decode =
      g_graph && g_graph->ready && g_graph->engine == &engine &&
      g_graph->decode_mode;
  const bool has_graph_prefill =
      g_graph && g_graph->ready && g_graph->engine == &engine &&
      !g_graph->decode_mode;
  if (!continuing_decode && !has_graph_prefill) {
    LOG_WARN("[GGML-QWEN-FULL-DECODE] disabled: decode state has not been "
             "seeded by the opt-in full-graph prefill\n");
    return false;
  }

  // Decode full-graph mode is intentionally stricter than the ordinary graph:
  // all expert planes must already be in the complete DDR graph arena.  This
  // keeps a graph submission from silently introducing an SSD/GGUF source or
  // a per-token lazy-load boundary.  Models with lazy expert ownership stay on
  // the established layered path.
  const bool narrow_layer0_diag = env_layer0_narrow_graph_diag_enabled();
  for (int layer = 0; layer < c.n_layers; ++layer) {
    if (narrow_layer0_diag && layer != 0)
      continue;
    if (!engine.graph_ram_layer_ready(layer)) {
      LOG_WARN("[GGML-QWEN-FULL-DECODE] disabled: layer=%d has no complete "
               "preloaded DDR expert arena; layered decode remains active\n",
               layer);
      return false;
    }
  }

  const int required = pos_base + 1;
  const int dev = engine.gpu_device_id >= 0 ? engine.gpu_device_id : 0;

  // Whether this session should attempt Phase 2 (true per-token dynamic
  // GPU/CPU dispatch for the partial layer) instead of Phase 1's static,
  // session-level commit. Both share the same usage-ranked expert set
  // (plan_partial_layer/qwen4exp_partial_layer_gpu_ids) -- Phase 2 just
  // decides per-token instead of once at prefill.
  plan_partial_layer(engine); // Phase 1/2 shared: usage-ranked expert set
  const bool phase2_wanted = env_partial_layer_phase2_enabled() &&
                             !engine.qwen4exp_partial_layer_phase2_disabled &&
                             engine.qwen4exp_partial_layer >= 0 &&
                             !engine.qwen4exp_partial_layer_gpu_ids.empty();
  const bool phase2_live = g_graph && g_graph->ready &&
                           g_graph->phase2_mode ==
                               FullGraph::Phase2Mode::Prefix &&
                           g_graph_p2_suffix_gpu &&
                           g_graph_p2_suffix_gpu->ready &&
                           g_graph_p2_suffix_cpu && g_graph_p2_suffix_cpu->ready;

  if (g_graph && g_graph->ready && g_graph->engine == &engine &&
      g_graph->decode_mode && g_graph->kv_capacity >= required &&
      g_graph->decode_last_pos + 1 == pos_base &&
      (phase2_wanted == phase2_live)) {
    if (has_ple_layer(c)) {
      // A PLE layer can land on either side of the split (prefix owns
      // layers [0, split_layer], the live suffix owns everything after)
      // -- refresh whichever instance's own ple_layer is actually >= 0,
      // not just the prefix's. Both suffix candidates are refreshed
      // (only one will actually run this token) since it's cheap and
      // keeps whichever gets chosen already current.
      FullGraph *targets[3] = {g_graph.get(),
                               phase2_live ? g_graph_p2_suffix_gpu.get()
                                           : nullptr,
                               phase2_live ? g_graph_p2_suffix_cpu.get()
                                           : nullptr};
      for (FullGraph *t : targets) {
        if (!t || t->ple_layer < 0)
          continue;
        std::vector<float> ple;
        if (!prepare_ple_embedding(*t, ple, 1, pos_base, token_ids) ||
            !t->ple_embedding ||
            ple.size() * sizeof(float) > ggml_nbytes(t->ple_embedding))
          return false;
        std::memcpy(t->ple_embedding->data, ple.data(),
                    ple.size() * sizeof(float));
      }
    }
    return true;
  }

  if (g_graph && g_graph->ready && g_graph->decode_mode &&
      g_graph->engine == &engine) {
    if (!sync_decode_state_to_host(*g_graph)) {
      LOG_WARN("[GGML-QWEN-FULL-DECODE] state download failed while growing "
               "KV capacity; using layered decode\n");
      return false;
    }
    // Phase 2's SuffixGpu instance holds its own GDN/KV/PLE state for
    // layers this prefix doesn't own (SuffixCpu shares the SAME physical
    // memory via phase2_state_source_graph -- downloading it a second
    // time would just re-read identical bytes through a different tensor
    // handle, so it's skipped).
    if (g_graph_p2_suffix_gpu && g_graph_p2_suffix_gpu->ready &&
        !sync_decode_state_to_host(*g_graph_p2_suffix_gpu)) {
      LOG_WARN("[GGML-QWEN-FULL-DECODE-P2] suffix_gpu state download failed "
               "while growing KV capacity; using layered decode\n");
      return false;
    }
  }
  // VRAM diagnostic (2026-09-15): two cheap cudaMemGetInfo reads bracketing
  // prefill's own g_graph teardown (before = prefill's buffers still
  // resident, after = immediately post-free, before decode allocates
  // anything of its own). Confirmed the ~0.33 GiB gap between PREFLIGHT's
  // startup free and this post-teardown free is generic CUDA context/
  // driver overhead (module/PTX loading, cuBLAS handle init) that persists
  // for the process lifetime -- NOT a leak in this file's own arena
  // teardown, which correctly frees essentially 100% of what it allocated
  // (confirmed: freed amount here matched the prefill arena's own tracked
  // size almost exactly). Kept as a permanent low-cost sanity check.
  {
    size_t free_before = 0, total_before = 0;
    cudaMemGetInfo(&free_before, &total_before);
    // SuffixCpu's GDN/KV/PLE state tensors alias SuffixGpu's own
    // device_state_buffer (see FullGraph::phase2_state_source_graph) --
    // free it first, then the instance that actually owns that memory.
    g_graph_p2_suffix_cpu.reset();
    g_graph_p2_suffix_gpu.reset();
    g_graph.reset();
    size_t free_after = 0, total_after = 0;
    cudaMemGetInfo(&free_after, &total_after);
    LOG_INFO("[GGML-QWEN-FULL-VRAM-TRACE] pre-reset free=%.3f GiB, "
             "post-reset free=%.3f GiB (freed %.3f GiB)\n",
             free_before / (1024.0 * 1024.0 * 1024.0),
             free_after / (1024.0 * 1024.0 * 1024.0),
             (free_after - free_before) / (1024.0 * 1024.0 * 1024.0));
  }

  // Builds ONE decode FullGraph instance -- the ordinary single-instance
  // decode graph when mode==Off, or one leg of Phase 2's 3-instance split
  // otherwise. Identical to the pre-Phase-2 body of this function in every
  // other respect (same backends/arenas/scheduler setup, same PLE/state
  // seeding), factored out only so Phase 2 can call it up to three times.
  auto build_one = [&](FullGraph::Phase2Mode mode, int split_layer,
                       FullGraph *state_source,
                       FullGraph *dense_source,
                       FullGraph *expert_source) -> std::unique_ptr<FullGraph> {
    auto g = std::make_unique<FullGraph>();
    g->engine = &engine;
    g->device = dev;
    g->S = 1;
    g->D = c.hidden;
    g->HC = c.hc_mult;
    g->n_layers = c.n_layers;
    g->H = c.n_heads;
    g->Hkv = std::max(1, c.n_head_kv);
    g->hd = c.head_dim;
    g->Hk_gdn = c.ssm_group_count;
    g->Hv_gdn = c.ssm_time_step_rank;
    g->gdn_hd = c.ssm_state_size;
    g->key_dim = g->Hk_gdn * g->gdn_hd;
    g->value_dim = c.ssm_inner_size;
    g->conv_dim = 2 * g->key_dim + g->value_dim;
    g->d_conv = c.ssm_conv_kernel;
    g->decode_mode = true;
    g->kv_capacity = env_decode_kv_capacity(engine.max_t, required);
    g->decode_last_pos = pos_base - 1;
    g->device_expert_layers =
        std::min(qwen4exp_full_graph_device_expert_layers(), c.n_layers);
    g->device_expert_trailing = env_device_expert_trailing();
    const char *partial_gate_up =
        std::getenv("MOESTREAM_QWEN4EXP_GRAPH_PRELOAD_PARTIAL_GATE_UP");
    // partial_gate_up's boundary math (layer==device_expert_layers, the
    // first CPU layer just past a LEADING block) has no trailing-mode
    // equivalent yet -- force it off rather than apply leading-only math
    // to a trailing block. See device_expert_trailing's own comment.
    g->device_expert_partial_gate_up = g->device_expert_layers > 0 &&
                                       !g->device_expert_trailing &&
                                       partial_gate_up &&
                                       partial_gate_up[0] == '1';
    g->phase2_mode = mode;
    g->phase2_split_layer = split_layer;
    g->phase2_state_source_graph = state_source;
    g->phase2_dense_source_graph = dense_source;
    g->phase2_expert_source_graph = expert_source;
    plan_expert_vram_split(*g); // must run before init_device_expert_arena
    g->cuda = static_cast<ggml_backend_t>(ggml_shared_backend(dev));
    g->cpu = ggml_backend_cpu_init();
    if (!g->cuda || !g->cpu)
      return nullptr;
    verify_cuda_graphs_compiled_in(g->cuda);
    g->cpu_threads = graph_cpu_threads();
    ggml_backend_cpu_set_n_threads(g->cpu, g->cpu_threads);
    ggml_threadpool_params cpu_tp_params =
        ggml_threadpool_params_default(g->cpu_threads);
    // 2026-09-15: tested ggml's own threadpool .poll (default 50 -- NOT
    // controlled by OMP_WAIT_POLICY/GOMP_SPINCOUNT, which only affect
    // libgomp's separate pool) as a candidate for the mixed CPU/GPU decode
    // regression via a temporary env override swept across 0/50/400/1000.
    // Ruled out via THROUGHPUT: all four values landed within noise of each
    // other (11.76-12.37 tok/s at device_expert_layers=20).
    // TEMPORARY re-test (2026-09-15, same day): now that the ELEVENTH ROUND
    // per-op gap instrumentation exists (MOESTREAM_QWEN4EXP_GRAPH_LAYER_
    // TIMING=1), re-checking whether .poll moves the GAP time specifically
    // (a real but small effect could have been lost in the earlier
    // throughput-only measurement's noise). Remove this override again if
    // inconclusive, per the original decision record's own discipline.
    if (const char *poll_env =
            std::getenv("MOESTREAM_QWEN4EXP_CPU_POLL_RETEST")) {
      char *end = nullptr;
      const long v = std::strtol(poll_env, &end, 10);
      if (end != poll_env && v >= 0 && v <= 100)
        cpu_tp_params.poll = static_cast<uint32_t>(v);
    }
    g->cpu_threadpool = ggml_threadpool_new(&cpu_tp_params);
    if (g->cpu_threadpool)
      ggml_backend_cpu_set_threadpool(g->cpu, g->cpu_threadpool);
    LOG_INFO("[GGML-QWEN-FULL-DECODE] CPU backend configured with %d "
             "threads%s (phase2_mode=%d)\n",
             g->cpu_threads,
             g->cpu_threadpool ? " (persistent threadpool)" : "",
             static_cast<int>(mode));

    // Every expert view aliases the already populated DDR graph arena, so
    // this mode has no graph-owned expert source admission or GGUF expert
    // read.
    if (!init_device_dense_arena(*g) || !init_device_state_arena(*g) ||
        !init_device_expert_arena(*g))
      return nullptr;
    ggml_backend_t backends[] = {g->cuda, g->cpu};
    ggml_backend_buffer_type_t sched_bufts[2] = {};
    ggml_backend_buffer_type_t *sched_bufts_ptr = nullptr;
    if (graph_cuda_host_buffers()) {
      sched_bufts[0] = ggml_backend_get_default_buffer_type(g->cuda);
      sched_bufts[1] = ggml_backend_dev_host_buffer_type(
          ggml_backend_get_device(g->cuda));
      if (sched_bufts[1]) {
        sched_bufts_ptr = sched_bufts;
        LOG_INFO("[GGML-QWEN-FULL-DECODE] scheduler CPU buffer=%s (CUDA "
                 "host buffer)\n",
                 ggml_backend_buft_name(sched_bufts[1]));
      }
    }
    // Match the prefill scheduler mode above; event/four-copy mode is opt-in.
    // op_offload MUST stay false -- see the prefill call site's own comment
    // (qwen4exp_full_graph_prepare, above) for the real S>=32 OOM/crash this
    // prevents. Decode is always S==1 so this flag is a no-op for it either
    // way; kept false here purely for consistency with the shared scheduler
    // contract, not because decode itself is at risk.
    g->sched =
        ggml_backend_sched_new(backends, sched_bufts_ptr, 2, kGraphSize,
                               graph_scheduler_parallel(), false);
    if (!g->sched)
      return nullptr;
    LOG_INFO(
        "[GGML-QWEN-FULL-DECODE] scheduler pipeline_parallel=%s copies=%d\n",
        graph_scheduler_parallel() ? "true" : "false",
        ggml_backend_sched_get_n_copies(g->sched));
    // See the prefill graph above: GRAPH_REPLAY_TIMING is deliberately
    // non-invasive and must not enable the scheduler eval callback.
    g->layer_timing_enabled = env_layer_timing_enabled();
    g->layer_timing_verbose_enabled = env_layer_timing_verbose();
    if (g->layer_timing_enabled || g->layer_timing_verbose_enabled) {
      g->layer_timing_cpu_ms.assign(static_cast<size_t>(g->n_layers), 0.0);
      ggml_backend_sched_set_eval_callback(g->sched, layer_timing_eval_callback,
                                           g.get());
    }

    std::vector<float> ple;
    if (!prepare_ple_embedding(*g, ple, 1, pos_base, token_ids) &&
        has_ple_layer(c)) {
      LOG_WARN("[GGML-QWEN-FULL-DECODE] PLE embedding preparation declined\n");
      return nullptr;
    }
    if (!build_graph(*g, has_ple_layer(c) ? &ple : nullptr)) {
      LOG_WARN("[GGML-QWEN-FULL-DECODE] graph construction declined "
               "(phase2_mode=%d); layered decode remains active\n",
               static_cast<int>(mode));
      return nullptr;
    }

    // Seed the device state once from the state/cache that the existing
    // prefill or layered path produced. This is a one-time boundary
    // transfer; later decode submissions update these tensors in the graph
    // itself. Skipped when state_source is set: that physical memory was
    // already seeded when the source instance was built (see
    // new_device_state_tensor's replay path -- writing it again here would
    // just re-write the identical bytes through an aliased tensor handle).
    if (!state_source) {
      const size_t state_n =
          static_cast<size_t>(g->gdn_hd) * g->gdn_hd * g->Hv_gdn;
      const size_t hist_n = static_cast<size_t>(g->d_conv - 1) * g->conv_dim;
      for (size_t i = 0; i < g->gdn_layers.size(); ++i) {
        const int l = g->gdn_layers[i];
        if (l < 0 ||
            l >= static_cast<int>(engine.qwen4exp_recur_state.size()) ||
            l >= static_cast<int>(engine.qwen4exp_conv_history.size()) ||
            engine.qwen4exp_recur_state[l].size() != state_n ||
            engine.qwen4exp_conv_history[l].size() != hist_n)
          return nullptr;
        // qwen4exp_recur_state is host-legacy layout; gdn_state_storage
        // feeds the decode self-loop's ggml op directly and needs native
        // layout. See gdn_state_transpose_host.
        std::vector<float> native_state(state_n);
        gdn_state_transpose_host(native_state.data(),
                                 engine.qwen4exp_recur_state[l].data(),
                                 g->gdn_hd, g->Hv_gdn);
        ggml_backend_tensor_set(g->gdn_state_storage[i], native_state.data(),
                                0, state_n * sizeof(float));
        ggml_backend_tensor_set(g->gdn_hist_storage[i],
                                engine.qwen4exp_conv_history[l].data(), 0,
                                hist_n * sizeof(float));
      }
      const size_t kv_row = static_cast<size_t>(g->Hkv) * g->hd;
      const size_t kv_bytes =
          static_cast<size_t>(pos_base) * kv_row * sizeof(float);
      for (size_t i = 0; i < g->kv_layers.size(); ++i) {
        const int l = g->kv_layers[i];
        if (i >= g->kv_state_k.size() || i >= g->kv_state_v.size() ||
            l < 0 || l >= static_cast<int>(engine.Kc.size()) ||
            l >= static_cast<int>(engine.Vc.size()) || !engine.Kc[l] ||
            !engine.Vc[l])
          return nullptr;
        if (kv_bytes > 0) {
          ggml_backend_tensor_set(g->kv_state_k[i], engine.Kc[l], 0,
                                  kv_bytes);
          ggml_backend_tensor_set(g->kv_state_v[i], engine.Vc[l], 0,
                                  kv_bytes);
        }
      }
      if (g->ple_layer >= 0 && g->ple_hist_storage) {
        const size_t ple_n = static_cast<size_t>(c.ple_conv_kernel - 1) *
                             c.ple_ngram_size * g->HC * g->D;
        if (g->ple_layer >=
                static_cast<int>(engine.qwen4exp_ple_conv_history.size()) ||
            engine.qwen4exp_ple_conv_history[g->ple_layer].size() != ple_n)
          return nullptr;
        ggml_backend_tensor_set(
            g->ple_hist_storage,
            engine.qwen4exp_ple_conv_history[g->ple_layer].data(), 0,
            ple_n * sizeof(float));
      }
    }
    g->ready = true;
    return g;
  };

  if (phase2_wanted) {
    auto prefix = build_one(FullGraph::Phase2Mode::Prefix,
                            engine.qwen4exp_partial_layer, nullptr, nullptr,
                            nullptr);
    if (prefix) {
      // Must be visible via the g_graph global BEFORE building either
      // suffix instance: their own build_graph() reads g_graph to find
      // the prefix's phase2_handoff_buf/addr for the res-seeding branch
      // (see build_graph's own comment on that lookup).
      g_graph = std::move(prefix);
      auto suffix_gpu = build_one(FullGraph::Phase2Mode::SuffixGpu,
                                  engine.qwen4exp_partial_layer, nullptr,
                                  g_graph.get(), g_graph.get());
      if (suffix_gpu) {
        auto suffix_cpu = build_one(FullGraph::Phase2Mode::SuffixCpu,
                                    engine.qwen4exp_partial_layer,
                                    suffix_gpu.get(), g_graph.get(),
                                    g_graph.get());
        if (suffix_cpu) {
          LOG_INFO("[GGML-QWEN-FULL-DECODE-P2] enabled: split_layer=%d, 3 "
                   "graph instances (prefix/suffix_gpu/suffix_cpu), KV "
                   "capacity=%d\n",
                   engine.qwen4exp_partial_layer, g_graph->kv_capacity);
          g_graph_p2_suffix_gpu = std::move(suffix_gpu);
          g_graph_p2_suffix_cpu = std::move(suffix_cpu);
          return true;
        }
      }
      g_graph_p2_suffix_gpu.reset();
      g_graph_p2_suffix_cpu.reset();
    }
    g_graph.reset();
    engine.qwen4exp_partial_layer_phase2_disabled = true;
    LOG_WARN("[GGML-QWEN-FULL-DECODE-P2] phase2 graph construction failed; "
             "disabling Phase 2 for this session and falling back to the "
             "ordinary single-graph decode path\n");
  }

  auto g = build_one(FullGraph::Phase2Mode::Off, -1, nullptr, nullptr,
                     nullptr);
  if (!g)
    return false;
  LOG_INFO("[GGML-QWEN-FULL-DECODE] enabled: one scheduler submission per "
           "token, KV capacity=%d, device GDN/KV/PLE state, DDR expert "
           "arenas=%d\n",
           g->kv_capacity, c.n_layers);
  g_graph = std::move(g);
  return true;
}

// Writes the standard positions/mask/decode_indices tensors every decode
// graph instance builds regardless of phase2_mode (see build_graph's own
// unconditional construction of these). Prefix's g.input carries this
// token's real hidden state (fed separately); suffix instances never read
// theirs at all (they resume from the handoff tensor instead), but still
// need positions/mask/indices for their OWN layers' attention.
bool phase2_feed_common_inputs(FullGraph &g, int pos_base) {
  if (!g.positions || !g.attn_mask || !g.decode_indices)
    return false;
  static_cast<int32_t *>(g.positions->data)[0] = pos_base;
  auto *mask = static_cast<ggml_fp16_t *>(g.attn_mask->data);
  for (int k = 0; k < g.kv_capacity; ++k)
    mask[k] = ggml_fp32_to_fp16(k <= pos_base ? 0.0f : -INFINITY);
  auto *indices = static_cast<int32_t *>(g.decode_indices->data);
  for (int h = 0; h < g.Hkv; ++h)
    indices[h] = pos_base * g.Hkv + h;
  return true;
}

// Unifies the prefix + sgpu submission into one captured CUDA graph so the
// common case (this token's routing lands entirely inside
// qwen4exp_partial_layer_gpu_ids) needs a single cudaGraphLaunch instead of
// two scheduler submissions with a host sync/branch between them. The
// "is every routed id already GPU-resident" decision moves to a device-side
// predicate (cuda/qwen4exp_master_graph.cu) driving a real
// cudaGraphCondTypeIf node whose true body is sgpu's own already-captured
// graph, spliced in as a child-graph node.
//
// scpu is deliberately NOT unified here: its ggml_backend_sched graph is a
// genuine multi-backend {cuda, cpu} split (it requires CPU-buffer-backed
// expert weight tensors), so it produces multiple independently-cached
// cudaGraph_t objects rather than one -- harvesting a single handle for it
// would silently drop whatever ggml placed on the CPU backend. When this
// token needs the scpu fallback, this only unifies prefix+predicate; the
// caller still submits scpu the existing way.
//
// Safety-gated: only engages once ggml_backend_sched_get_n_splits() confirms
// BOTH prefix and sgpu are genuinely single-backend (CUDA-only) graphs.
// Falls back permanently for the session if that is ever not true, rather
// than guessing. Opt-in via MOESTREAM_QWEN4EXP_MASTER_GRAPH so the default
// decode path is completely unchanged until this has been validated against
// the existing coherence/output-parity checks.
namespace qwen4exp_master_graph {

struct State {
  bool safety_verified = false;
  bool safety_ok = false;
  cudaGraph_t master_graph = nullptr;
  cudaGraphExec_t master_exec = nullptr;
  cudaGraph_t last_prefix_handle = nullptr;
  cudaGraph_t last_sgpu_handle = nullptr;
  // The predicate kernel writes the device copy.  The host copy is only a
  // pinned destination for the same-stream D2H readback after the graph
  // completes; cudaMallocHost() memory is not necessarily device-mapped and
  // must not be dereferenced by a kernel directly.
  int32_t *device_fallback_flag = nullptr;
  int32_t *host_fallback_flag = nullptr;
  uint8_t *device_hot_bitmap = nullptr;
  int hot_bitmap_capacity = 0;
  size_t last_gpu_ids_hash = 0;
  cudaEvent_t phase_start_event = nullptr;
  cudaEvent_t phase_end_event = nullptr;
};

State &state() {
  static State s;
  return s;
}

bool env_enabled() {
  static const bool enabled =
      std::getenv("MOESTREAM_QWEN4EXP_MASTER_GRAPH") != nullptr;
  return enabled;
}

size_t hash_gpu_ids(const std::vector<int> &ids) {
  size_t h = 1469598103934665603ull; // FNV-1a offset basis
  for (int id : ids) {
    h ^= static_cast<size_t>(static_cast<uint32_t>(id));
    h *= 1099511628211ull; // FNV-1a prime
  }
  return h;
}

// Uploads a fresh device bitmap only when qwen4exp_partial_layer_gpu_ids
// actually changed (normally a rare, session-level event, not per-token).
bool ensure_hot_bitmap(StreamingEngine &engine, int n_experts,
                       cudaStream_t stream) {
  State &st = state();
  if (n_experts <= 0)
    return false;
  const size_t h = hash_gpu_ids(engine.qwen4exp_partial_layer_gpu_ids);
  if (st.device_hot_bitmap && st.hot_bitmap_capacity == n_experts &&
      st.last_gpu_ids_hash == h)
    return true;

  if (!st.device_hot_bitmap || st.hot_bitmap_capacity != n_experts) {
    if (st.device_hot_bitmap) {
      cudaFree(st.device_hot_bitmap);
      st.device_hot_bitmap = nullptr;
    }
    if (cudaMalloc(&st.device_hot_bitmap, static_cast<size_t>(n_experts)) !=
        cudaSuccess) {
      st.device_hot_bitmap = nullptr;
      st.hot_bitmap_capacity = 0;
      return false;
    }
    st.hot_bitmap_capacity = n_experts;
  }

  std::vector<uint8_t> host_bitmap(static_cast<size_t>(n_experts), 0);
  for (int id : engine.qwen4exp_partial_layer_gpu_ids)
    if (id >= 0 && id < n_experts)
      host_bitmap[static_cast<size_t>(id)] = 1;
  if (cudaMemcpyAsync(st.device_hot_bitmap, host_bitmap.data(),
                      host_bitmap.size(), cudaMemcpyHostToDevice,
                      stream) != cudaSuccess)
    return false;
  // Rare path (only on a hot-set change, not per token): safe to sync here.
  if (cudaStreamSynchronize(stream) != cudaSuccess)
    return false;
  st.last_gpu_ids_hash = h;
  return true;
}

// Returns false if the fast path cannot be used this token (falls back to
// the existing prefix/chosen dispatch unchanged); true if it successfully
// launched prefix+predicate(+sgpu) as one graph, in which case
// *needs_scpu_fallback tells the caller whether sgpu already produced the
// result or scpu still needs to be submitted the existing way.
bool try_launch(StreamingEngine &engine, FullGraph &prefix, FullGraph &sgpu,
                bool *needs_scpu_fallback) {
  *needs_scpu_fallback = true;
  if (!env_enabled())
    return false;

  State &st = state();
  const bool phase_timing = env_phase_timing_enabled();
  const auto call_start = std::chrono::steady_clock::now();
  if (!st.safety_verified) {
    st.safety_verified = true;
    const int prefix_splits = ggml_backend_sched_get_n_splits(prefix.sched);
    const int sgpu_splits = ggml_backend_sched_get_n_splits(sgpu.sched);
    st.safety_ok = prefix_splits == 1 && sgpu_splits == 1;
    if (!st.safety_ok)
      LOG_WARN(
          "[QWEN4EXP-MASTER-GRAPH] disabled for this session: prefix/sgpu "
          "graphs are not both single-backend (n_splits=%d/%d) -- a "
          "multi-backend split would silently drop CPU-backend work if "
          "harvested as one cudaGraph_t, so falling back to the existing "
          "two-graph host-branch dispatch permanently.\n",
          prefix_splits, sgpu_splits);
    else
      LOG_INFO("[QWEN4EXP-MASTER-GRAPH] enabled: prefix/sgpu both "
              "single-backend, unifying the sgpu fast path into one "
              "captured graph.\n");
  }
  if (!st.safety_ok)
    return false;

  cudaStream_t stream = moestream_ggml_backend_cuda_get_stream(prefix.cuda);
  if (!stream)
    return false;

  const auto &c = engine.model_file.architecture_config;
  if (!ensure_hot_bitmap(engine, c.n_experts, stream))
    return false;

  cudaGraph_t prefix_handle = moestream_ggml_cuda_get_captured_graph(
      prefix.cuda, ggml_graph_node(prefix.graph, 0));
  cudaGraph_t sgpu_handle = moestream_ggml_cuda_get_captured_graph(
      sgpu.cuda, ggml_graph_node(sgpu.graph, 0));
  if (!prefix_handle || !sgpu_handle)
    return false; // not captured yet (warmup incomplete) -- fall back

  if (!st.device_fallback_flag &&
      cudaMalloc(&st.device_fallback_flag, sizeof(int32_t)) != cudaSuccess)
    return false;
  if (!st.host_fallback_flag &&
      cudaMallocHost(&st.host_fallback_flag, sizeof(int32_t)) !=
          cudaSuccess)
    return false;

  const bool graphs_changed = prefix_handle != st.last_prefix_handle ||
                              sgpu_handle != st.last_sgpu_handle;
  if (graphs_changed || !st.master_exec) {
    if (st.master_exec) {
      cudaGraphExecDestroy(st.master_exec);
      st.master_exec = nullptr;
    }
    if (st.master_graph) {
      cudaGraphDestroy(st.master_graph);
      st.master_graph = nullptr;
    }
    if (cudaGraphCreate(&st.master_graph, 0) != cudaSuccess)
      return false;

    cudaGraphNode_t prefix_node = nullptr;
    if (cudaGraphAddChildGraphNode(&prefix_node, st.master_graph, nullptr, 0,
                                   prefix_handle) != cudaSuccess)
      return false;

    cudaGraphConditionalHandle cond_handle{};
    if (cudaGraphConditionalHandleCreate(&cond_handle, st.master_graph, 0,
                                         cudaGraphCondAssignDefault) !=
        cudaSuccess)
      return false;

    ggml_tensor *routed_ids_tensor = prefix.partial_layer_ids_out;
    auto *routed_ids = static_cast<int32_t *>(routed_ids_tensor->data);
    int topk = c.topk;
    uint8_t *hot_bitmap = st.device_hot_bitmap;
    int32_t *fallback_flag = st.device_fallback_flag;

    cudaKernelNodeParams predicate_kernel{};
    predicate_kernel.func = moestream_qwen4exp_master_predicate_kernel_ptr();
    void *predicate_args[] = {&cond_handle, &routed_ids, &hot_bitmap, &topk,
                              &fallback_flag};
    predicate_kernel.gridDim = dim3(1, 1, 1);
    predicate_kernel.blockDim = dim3(1, 1, 1);
    predicate_kernel.kernelParams = predicate_args;
    cudaGraphNode_t predicate_node = nullptr;
    if (cudaGraphAddKernelNode(&predicate_node, st.master_graph,
                              &prefix_node, 1, &predicate_kernel) !=
        cudaSuccess)
      return false;

    // size=1: cudaGraphCondTypeIf with only a true-body. When the predicate
    // is 0 (scpu needed), the body is skipped entirely -- sgpu's compute
    // genuinely never runs, not "run and ignore".
    cudaGraphNodeParams cond_params{};
    cond_params.type = cudaGraphNodeTypeConditional;
    cond_params.conditional.handle = cond_handle;
    cond_params.conditional.type = cudaGraphCondTypeIf;
    cond_params.conditional.size = 1;
    cudaGraphNode_t cond_node = nullptr;
    if (cudaGraphAddNode(&cond_node, st.master_graph, &predicate_node,
                        nullptr, 1, &cond_params) != cudaSuccess)
      return false;

    cudaGraphNode_t sgpu_child = nullptr;
    if (cudaGraphAddChildGraphNode(
            &sgpu_child, cond_params.conditional.phGraph_out[0], nullptr, 0,
            sgpu_handle) != cudaSuccess)
      return false;

    if (cudaGraphInstantiate(&st.master_exec, st.master_graph, 0) !=
        cudaSuccess)
      return false;

    st.last_prefix_handle = prefix_handle;
    st.last_sgpu_handle = sgpu_handle;
    LOG_INFO("[QWEN4EXP-MASTER-GRAPH] (re)built master graph (prefix=%p "
            "sgpu=%p)\n",
            (void *) prefix_handle, (void *) sgpu_handle);
  }

  bool phase_events_ready = false;
  if (phase_timing) {
    if (!st.phase_start_event &&
        cudaEventCreate(&st.phase_start_event) != cudaSuccess)
      return false;
    if (!st.phase_end_event &&
        cudaEventCreate(&st.phase_end_event) != cudaSuccess)
      return false;
    // The start event is queued immediately before the master launch by the
    // same stream, so the elapsed span covers prefix + predicate + conditional
    // sgpu work without measuring host submission time.
    if (cudaEventRecord(st.phase_start_event, stream) != cudaSuccess)
      return false;
    phase_events_ready = true;
  }
  const auto launch_call_start = std::chrono::steady_clock::now();
  if (cudaGraphLaunch(st.master_exec, stream) != cudaSuccess)
    return false;
  const auto launch_return = std::chrono::steady_clock::now();
  if (phase_events_ready &&
      cudaEventRecord(st.phase_end_event, stream) != cudaSuccess)
    return false;
  // Keep the predicate decision device-native through graph execution.  Only
  // after the graph has completed do we copy the one-word result back so the
  // existing host-side caller can decide whether to submit the CPU suffix.
  if (cudaMemcpyAsync(st.host_fallback_flag, st.device_fallback_flag,
                      sizeof(int32_t), cudaMemcpyDeviceToHost,
                      stream) != cudaSuccess)
    return false;
  if (cudaStreamSynchronize(stream) != cudaSuccess)
    return false;

  *needs_scpu_fallback = *st.host_fallback_flag != 0;
  if (phase_timing) {
    float device_ms = 0.0f;
    const cudaError_t elapsed =
        phase_events_ready
            ? cudaEventElapsedTime(&device_ms, st.phase_start_event,
                                   st.phase_end_event)
            : cudaErrorNotReady;
    const double host_total_ms =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - call_start)
            .count();
    const double launch_us =
        std::chrono::duration<double, std::micro>(launch_return -
                                                  launch_call_start)
            .count();
    LOG_INFO(
        "[QWEN4EXP-MASTER-GRAPH-TIME] host_total=%.3fms launch_return=%.3fus "
        "device_graph=%.3fms fallback=%d event_status=%s\n",
        host_total_ms, launch_us, elapsed == cudaSuccess ? device_ms : -1.0,
        *needs_scpu_fallback ? 1 : 0,
        elapsed == cudaSuccess ? "ok" : cudaGetErrorString(elapsed));
  }
  return true;
}

} // namespace qwen4exp_master_graph

// Phase 2 per-token orchestration: submits the prefix graph (attention
// through the split layer), reads back this token's REAL routing at that
// layer, decides GPU-vs-CPU purely from whether every routed id already
// lives in qwen4exp_partial_layer_gpu_ids, then submits exactly ONE of the
// two suffix graphs -- never both, per the "all-or-nothing" design (see
// FullGraph::Phase2Mode's own comment). Two scheduler submissions per
// token instead of one; that overhead is the real, measured cost this
// feature trades against skipping the split layer's full-width FFN when
// the GPU-compact path applies. When MOESTREAM_QWEN4EXP_MASTER_GRAPH is set
// and both prefix and sgpu are verified single-backend graphs, the common
// (all-GPU-resident routing) case instead runs as one unified captured
// graph -- see qwen4exp_master_graph::try_launch above.
bool qwen4exp_full_graph_decode_run_phase2(StreamingEngine &engine,
                                           float *hidden_states, int pos_base,
                                           const int *token_ids, float *logits,
                                           int vocab) {
  FullGraph &prefix = *g_graph;
  FullGraph &sgpu = *g_graph_p2_suffix_gpu;
  FullGraph &scpu = *g_graph_p2_suffix_cpu;
  if (pos_base >= sgpu.kv_capacity || pos_base >= scpu.kv_capacity ||
      !prefix.input || !prefix.partial_layer_ids_out)
    return false;
  if (engine.hc_stream_buf.size() <
      static_cast<size_t>(prefix.D) * prefix.HC)
    return false;
  if (has_ple_layer(engine.model_file.architecture_config) && !token_ids)
    return false;
  if (logits && (!sgpu.logits || !scpu.logits ||
                ggml_nelements(sgpu.logits) < vocab ||
                ggml_nelements(scpu.logits) < vocab))
    return false;

  std::memcpy(prefix.input->data, engine.hc_stream_buf.data(),
              static_cast<size_t>(prefix.D) * prefix.HC * sizeof(float));
  if (!phase2_feed_common_inputs(prefix, pos_base) ||
      !phase2_feed_common_inputs(sgpu, pos_base) ||
      !phase2_feed_common_inputs(scpu, pos_base))
    return false;

  // Per-layer/per-op CPU timing (env: MOESTREAM_QWEN4EXP_GRAPH_LAYER_TIMING=1),
  // mirroring qwen4exp_full_graph_decode_run's own reset -- each of the
  // (up to) two graphs submitted this token needs its own reset, since
  // each has its own independent eval-callback state.
  for (FullGraph *g : {&prefix, &sgpu, &scpu}) {
    if (g->layer_timing_enabled) {
      std::fill(g->layer_timing_cpu_ms.begin(), g->layer_timing_cpu_ms.end(),
                0.0);
      g->layer_timing_op_ms.clear();
      g->layer_timing_t0.clear();
      g->layer_timing_gap_ms_total = 0.0;
      g->layer_timing_gap_count = 0;
      g->layer_timing_have_last_ask_false = false;
    }
  }

  const auto start = std::chrono::steady_clock::now();
  double prefix_submit_ms = 0.0;
  double route_decision_ms = 0.0;
  double suffix_submit_ms = 0.0;
  double output_readback_ms = 0.0;
  double prefix_replay_submit_ms = 0.0;
  double prefix_replay_sync_ms = 0.0;
  double suffix_replay_submit_ms = 0.0;
  double suffix_replay_sync_ms = 0.0;

  // Keep the host-side replay boundary distinct from the existing phase
  // total.  The latter includes route selection and output readback; these
  // two values isolate scheduler submission from the wait that contains GPU
  // execution, ordinary CPU-backend work, and any host-node callback.
  auto run_timed_replay = [](FullGraph &graph, double &submit_ms,
                             double &sync_ms, double &total_ms) {
    const auto replay_start = std::chrono::steady_clock::now();
    if (ggml_backend_sched_graph_compute_async(graph.sched, graph.graph) !=
        GGML_STATUS_SUCCESS)
      return false;
    const auto replay_submit_end = std::chrono::steady_clock::now();
    ggml_backend_sched_synchronize(graph.sched);
    const auto replay_end = std::chrono::steady_clock::now();
    submit_ms = std::chrono::duration<double, std::milli>(
                    replay_submit_end - replay_start)
                    .count();
    sync_ms = std::chrono::duration<double, std::milli>(
                  replay_end - replay_submit_end)
                  .count();
    total_ms = std::chrono::duration<double, std::milli>(
                   replay_end - replay_start)
                   .count();
    return true;
  };

  // Fast path: prefix + predicate + sgpu unified into one captured graph
  // (see qwen4exp_master_graph::try_launch). Only engages once verified
  // single-backend-safe; otherwise behaves exactly like the pre-existing
  // path below (prefix run the normal way, ids read back, host branch).
  bool used_master_graph = false;
  bool needs_scpu_fallback = true;
  if (qwen4exp_master_graph::try_launch(engine, prefix, sgpu,
                                        &needs_scpu_fallback))
    used_master_graph = true;

  FullGraph *chosen = nullptr;
  if (used_master_graph && !needs_scpu_fallback) {
    // sgpu already computed inside the master graph -- nothing left to
    // submit this token.
    chosen = &sgpu;
  } else {
    if (!used_master_graph) {
      // Step 1: prefix -- real attention through the split layer, plus this
      // token's own routing at that layer (a graph output).
      const auto phase_start = std::chrono::steady_clock::now();
      if (!run_timed_replay(prefix, prefix_replay_submit_ms,
                            prefix_replay_sync_ms, prefix_submit_ms))
        return false;
      prefix_submit_ms = std::chrono::duration<double, std::milli>(
                             std::chrono::steady_clock::now() - phase_start)
                             .count();
    }

    if (used_master_graph) {
      // The master graph's on-device predicate already determined this
      // token needs the scpu fallback -- no host readback required.
      chosen = &scpu;
    } else {
      const auto phase_start = std::chrono::steady_clock::now();
      const auto &c = engine.model_file.architecture_config;
      std::vector<int32_t> ids_host(static_cast<size_t>(c.topk));
      ggml_backend_tensor_get(prefix.partial_layer_ids_out, ids_host.data(),
                              0, ids_host.size() * sizeof(int32_t));
      const std::unordered_set<int> gpu_set(
          engine.qwen4exp_partial_layer_gpu_ids.begin(),
          engine.qwen4exp_partial_layer_gpu_ids.end());
      bool all_in_gpu_set = true;
      for (int32_t id : ids_host) {
        if (id >= 0 && !gpu_set.count(id)) {
          all_in_gpu_set = false;
          break;
        }
      }
      chosen = all_in_gpu_set ? &sgpu : &scpu;
      route_decision_ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - phase_start)
                              .count();
    }

    // Step 2: exactly the one suffix this token's real routing needs -- the
    // split layer's real FFN, plus every layer after it, on ONE device.
    const auto phase_start = std::chrono::steady_clock::now();
    if (!run_timed_replay(*chosen, suffix_replay_submit_ms,
                          suffix_replay_sync_ms, suffix_submit_ms))
      return false;
    suffix_submit_ms = std::chrono::duration<double, std::milli>(
                           std::chrono::steady_clock::now() - phase_start)
                           .count();
  }
  const auto output_start = std::chrono::steady_clock::now();
  ggml_backend_tensor_get(chosen->output, hidden_states, 0,
                          static_cast<size_t>(chosen->D) * sizeof(float));
  if (logits)
    ggml_backend_tensor_get(chosen->logits, logits, 0,
                            static_cast<size_t>(vocab) * sizeof(float));
  output_readback_ms = std::chrono::duration<double, std::milli>(
                           std::chrono::steady_clock::now() - output_start)
                           .count();

  prefix.decode_last_pos = pos_base;
  sgpu.decode_last_pos = pos_base;
  scpu.decode_last_pos = pos_base;

  if (has_ple_layer(engine.model_file.architecture_config)) {
    auto &hist = engine.qwen4exp_ple_tok_hist;
    hist.push_back(token_ids[0]);
    const int keep =
        std::max(0, engine.model_file.architecture_config.ple_ngram_size - 1);
    if (static_cast<int>(hist.size()) > keep)
      hist.erase(hist.begin(), hist.end() - keep);
    engine.qwen4exp_ple_next_pos = pos_base + 1;
  }
  engine.qwen4exp_full_graph_output_ready = true;
  engine.qwen4exp_full_graph_logits_ready = logits != nullptr;
  const double ms = std::chrono::duration<double, std::milli>(
                         std::chrono::steady_clock::now() - start)
                         .count();
  LOG_INFO("[GGML-QWEN-FULL-DECODE-P2-TIME] pos=%d dispatch=%s total=%.3f "
           "ms\n",
           pos_base, chosen == &sgpu ? "gpu" : "cpu", ms);
  if (env_phase_timing_enabled()) {
    LOG_INFO(
        "[GGML-QWEN-FULL-DECODE-P2-BREAKDOWN] pos=%d dispatch=%s "
        "prefix_submit_sync=%.3fms route_decision=%.3fms "
        "suffix_submit_sync=%.3fms output_readback=%.3fms total=%.3fms\n",
        pos_base, chosen == &sgpu ? "gpu" : "cpu", prefix_submit_ms,
        route_decision_ms, suffix_submit_ms, output_readback_ms, ms);
  }
  if (env_graph_replay_timing_enabled()) {
    auto emit_replay_ledger = [&](const char *instance, FullGraph &graph,
                                  double submit_ms, double sync_ms,
                                  double total_ms) {
      double ordinary_cpu_compute_ms = 0.0;
      for (const double layer_ms : graph.layer_timing_cpu_ms)
        ordinary_cpu_compute_ms += layer_ms;
    double host_node_d2h_ms = 0.0;
    double host_node_h2d_ms = 0.0;
    double host_node_callback_span_ms = 0.0;
    double host_node_cpu_ms = 0.0;
      for (const auto &host_node : graph.host_node_contexts) {
        if (!host_node)
          continue;
        float d2h = 0.0f;
        float h2d = 0.0f;
        if (cudaEventElapsedTime(&d2h, host_node->d2h_start,
                                 host_node->d2h_end) == cudaSuccess)
          host_node_d2h_ms += d2h;
        if (cudaEventElapsedTime(&h2d, host_node->h2d_start,
                                 host_node->h2d_end) == cudaSuccess)
          host_node_h2d_ms += h2d;
        // These events bracket the host callback on the captured CUDA
        // stream.  Unlike the host-side callback clock, this includes the
        // stream's observed callback handoff/return latency and lets us
        // separate CPU work from CUDA host-node serialization.
        float callback_span = 0.0f;
        if (cudaEventElapsedTime(&callback_span, host_node->d2h_end,
                                 host_node->h2d_start) == cudaSuccess)
          host_node_callback_span_ms += callback_span;
        host_node_cpu_ms +=
            host_node->last_cpu_ms.load(std::memory_order_acquire);
      }
      LOG_INFO(
          "[QWEN4EXP-GRAPH-REPLAY-LEDGER] pos=%d instance=%s splits=%d "
          "host_nodes=%zu replay_submit=%.3fms replay_sync=%.3fms "
          "replay_total=%.3fms host_node_d2h=%.3fms "
          "host_node_cpu=%.3fms host_node_callback_span=%.3fms "
          "host_node_h2d=%.3fms "
          "ordinary_cpu_compute=%.3fms "
          "ordinary_cpu_between_node_gap=%.3fms "
          "ordinary_cpu_ledger=%.3fms\n",
          pos_base, instance, ggml_backend_sched_get_n_splits(graph.sched),
          graph.host_node_contexts.size(), submit_ms, sync_ms, total_ms,
          host_node_d2h_ms, host_node_cpu_ms, host_node_callback_span_ms,
          host_node_h2d_ms,
          ordinary_cpu_compute_ms, graph.layer_timing_gap_ms_total,
          ordinary_cpu_compute_ms + graph.layer_timing_gap_ms_total);
    };
    if (!used_master_graph)
      emit_replay_ledger("prefix", prefix, prefix_replay_submit_ms,
                         prefix_replay_sync_ms, prefix_submit_ms);
    emit_replay_ledger(chosen == &sgpu ? "suffix_gpu" : "suffix_cpu", *chosen,
                       suffix_replay_submit_ms, suffix_replay_sync_ms,
                       suffix_submit_ms);
  }
  // Same per-layer CPU-timing report as the single-graph decode path, for
  // BOTH graphs actually submitted this token -- lets a specific layer
  // (e.g. the split layer) be compared directly against its own
  // single-graph-baseline cost. Only meaningful for CPU-dispatched nodes
  // (the split layer's compact GPU path has no CPU-side op to time here;
  // its cost is inside the GPU-submission wall time captured above).
  if ((prefix.layer_timing_enabled || chosen->layer_timing_enabled) &&
      ++prefix.layer_timing_tokens_since_emit >= 10) {
    prefix.layer_timing_tokens_since_emit = 0;
    for (auto *tag_g : {&prefix, chosen}) {
      if (!tag_g->layer_timing_enabled)
        continue;
      double total_cpu_ms = 0.0;
      std::string per_layer;
      for (size_t l = 0; l < tag_g->layer_timing_cpu_ms.size(); ++l) {
        const double lms = tag_g->layer_timing_cpu_ms[l];
        total_cpu_ms += lms;
        if (lms > 0.01)
          per_layer += " L" + std::to_string(l) + "=" +
                      std::to_string(static_cast<int>(lms * 100) / 100.0);
      }
      LOG_INFO("[GGML-QWEN-FULL-LAYER-TIMING-P2] pos=%d inst=%s "
               "dispatch=%s cpu_total=%.3fms gap_total=%.3fms (n=%ld "
               "gaps)%s\n",
               pos_base,
               tag_g == &prefix ? "prefix"
               : tag_g == &sgpu ? "suffix_gpu"
                                : "suffix_cpu",
               chosen == &sgpu ? "gpu" : "cpu", total_cpu_ms,
               tag_g->layer_timing_gap_ms_total, tag_g->layer_timing_gap_count,
               per_layer.c_str());
      for (const auto &[stem, oms] : tag_g->layer_timing_op_ms)
        LOG_INFO("[GGML-QWEN-FULL-LAYER-TIMING-P2]   inst=%s op=%s "
                 "cpu=%.3fms\n",
                 tag_g == &prefix ? "prefix"
                 : tag_g == &sgpu ? "suffix_gpu"
                                  : "suffix_cpu",
                 stem.c_str(), oms);
    }
  }
  return true;
}

bool qwen4exp_stage_layer13_before_graph(FullGraph &g, int pos_base) {
  if (!g.stage_enabled)
    return true;
  if (!g.engine || !g.engine->gpu || g.stage_bindings.empty())
    return false;
  if (g.stage_layer < 0 ||
      static_cast<size_t>(g.stage_layer) >= g.engine->graph_ram_arenas.size())
    return false;

  // The proof deliberately requires a pinned DDR source.  Pin only the
  // target layer if the normal graph-arena registration A/B was not enabled;
  // this keeps the proof self-contained without pinning the other 47 arenas.
  auto &arena = g.engine->graph_ram_arenas[static_cast<size_t>(g.stage_layer)];
  if (!arena.pinned) {
    if (g.engine->ram_pinned_budget_bytes == 0 ||
        arena.allocation_base == nullptr || arena.allocation_bytes == 0)
      return false;
    const size_t previous = g.engine->ram_pinned_bytes_used.fetch_add(
        arena.bytes, std::memory_order_relaxed);
    if (previous > g.engine->ram_pinned_budget_bytes ||
        arena.bytes > g.engine->ram_pinned_budget_bytes - previous) {
      g.engine->ram_pinned_bytes_used.fetch_sub(arena.bytes,
                                                std::memory_order_relaxed);
      LOG_WARN("[OPTION-B-STAGE] Layer %d source pin budget exhausted\n",
               g.stage_layer);
      return false;
    }
    cudaSetDevice(g.device);
    const cudaError_t pin_err = cudaHostRegister(
        arena.allocation_base, arena.allocation_bytes, cudaHostRegisterPortable);
    if (pin_err != cudaSuccess) {
      g.engine->ram_pinned_bytes_used.fetch_sub(arena.bytes,
                                                std::memory_order_relaxed);
      cudaGetLastError();
      LOG_WARN("[OPTION-B-STAGE] Layer %d source pin failed: %s\n",
               g.stage_layer, cudaGetErrorString(pin_err));
      return false;
    }
    arena.pinned = true;
    LOG_INFO("[OPTION-B-STAGE] pinned Layer %d DDR arena: %.3f GiB\n",
             g.stage_layer,
             static_cast<double>(arena.bytes) /
                 (1024.0 * 1024.0 * 1024.0));
  }

  cudaSetDevice(g.device);
  if (!g.stage_wait_done_event) {
    if (cudaEventCreate(&g.stage_wait_start_event) != cudaSuccess ||
        cudaEventCreate(&g.stage_wait_done_event) != cudaSuccess ||
        cudaEventCreate(&g.stage_d2d_start_event) != cudaSuccess ||
        cudaEventCreate(&g.stage_d2d_end_event) != cudaSuccess ||
        cudaEventCreate(&g.stage_compute_start_event) != cudaSuccess ||
        cudaEventCreate(&g.stage_compute_end_event) != cudaSuccess)
      return false;
  }

  GpuTopology &topology = g.engine->gpu->topology;
  const auto host_issue_start = std::chrono::steady_clock::now();
  g.stage_slots.clear();
  for (auto &binding : g.stage_bindings) {
    ExpertLocation *slot = topology.upload_expert_active_projections_async(
        g.stage_layer, binding.expert, binding.src_gate, binding.gate_bytes,
        binding.src_up, binding.up_bytes, binding.src_down,
        binding.down_bytes, g.device);
    if (!slot || !slot->upload_event) {
      LOG_WARN("[OPTION-B-STAGE] failed to queue Layer %d expert %d upload "
               "(%zu requested)\n",
               g.stage_layer, binding.expert, g.stage_bindings.size());
      return false;
    }
    binding.slot = slot;
    g.stage_slots.push_back(slot);
  }
  g.stage_slot = g.stage_slots.front();

  cudaStream_t graph_stream =
      moestream_ggml_backend_cuda_get_stream(g.cuda);
  if (!graph_stream) {
    LOG_WARN("[OPTION-B-STAGE] ggml CUDA stream bridge unavailable\n");
    return false;
  }
  if (cudaEventRecord(g.stage_wait_start_event, graph_stream) != cudaSuccess)
    return false;
  for (const auto *slot : g.stage_slots) {
    if (cudaStreamWaitEvent(graph_stream, slot->upload_event, 0) !=
        cudaSuccess)
      return false;
  }
  if (cudaEventRecord(g.stage_wait_done_event, graph_stream) != cudaSuccess ||
      cudaEventRecord(g.stage_d2d_start_event, graph_stream) != cudaSuccess)
    return false;
  for (const auto &binding : g.stage_bindings) {
    const auto *slot = binding.slot;
    if (cudaMemcpyAsync(binding.dst_gate, slot->device_ptr,
                        binding.gate_bytes, cudaMemcpyDeviceToDevice,
                        graph_stream) != cudaSuccess ||
        cudaMemcpyAsync(binding.dst_up,
                        slot->device_ptr + binding.gate_bytes,
                        binding.up_bytes, cudaMemcpyDeviceToDevice,
                        graph_stream) != cudaSuccess ||
        cudaMemcpyAsync(binding.dst_down,
                        slot->device_ptr + binding.gate_bytes + binding.up_bytes,
                        binding.down_bytes, cudaMemcpyDeviceToDevice,
                        graph_stream) != cudaSuccess)
      return false;
  }
  if (cudaEventRecord(g.stage_d2d_end_event, graph_stream) != cudaSuccess ||
      cudaEventRecord(g.stage_compute_start_event, graph_stream) !=
          cudaSuccess)
    return false;
  const double host_issue_us = std::chrono::duration<double, std::micro>(
                                   std::chrono::steady_clock::now() -
                                   host_issue_start)
                                   .count();
  LOG_INFO("[OPTION-B-STAGE] pos=%d queued layer=%d experts=%zu "
           "host_issue=%.3fus source=%s\n",
           pos_base, g.stage_layer, g.stage_slots.size(), host_issue_us,
           g.stage_slots.front()->upload_source_pinned ? "pinned" : "pageable");
  return true;
}

bool qwen4exp_full_graph_decode_run(StreamingEngine &engine,
                                    float *hidden_states, int pos_base,
                                    const int *token_ids, float *logits,
                                    int vocab) {
  if (!qwen4exp_full_graph_decode_enabled() || !hidden_states || pos_base < 0 ||
      !g_graph || !g_graph->ready || !g_graph->decode_mode ||
      g_graph->engine != &engine || g_graph->S != 1 ||
      pos_base >= g_graph->kv_capacity ||
      ((logits == nullptr) != (vocab <= 0)))
    return false;
  if (g_graph->phase2_mode == FullGraph::Phase2Mode::Prefix &&
      g_graph_p2_suffix_gpu && g_graph_p2_suffix_gpu->ready &&
      g_graph_p2_suffix_cpu && g_graph_p2_suffix_cpu->ready) {
    return qwen4exp_full_graph_decode_run_phase2(engine, hidden_states,
                                                 pos_base, token_ids, logits,
                                                 vocab);
  }
  FullGraph &g = *g_graph;
  if (engine.hc_stream_buf.size() < static_cast<size_t>(g.D) * g.HC)
    return false;
  if (has_ple_layer(engine.model_file.architecture_config) &&
      (!g.ple_embedding || !token_ids))
    return false;
  // Validate the optional output before submitting the graph.  If this
  // check were left until after compute, a malformed logits destination
  // could advance device KV/GDN state and then fall back to the layered path
  // with stale host state.
  if (logits && (!g.logits || ggml_nelements(g.logits) < vocab))
    return false;

  std::memcpy(g.input->data, engine.hc_stream_buf.data(),
              static_cast<size_t>(g.D) * g.HC * sizeof(float));
  static_cast<int32_t *>(g.positions->data)[0] = pos_base;
  auto *mask = static_cast<ggml_fp16_t *>(g.attn_mask->data);
  for (int k = 0; k < g.kv_capacity; ++k)
    mask[k] = ggml_fp32_to_fp16(k <= pos_base ? 0.0f : -INFINITY);
  auto *indices = static_cast<int32_t *>(g.decode_indices->data);
  for (int h = 0; h < g.Hkv; ++h)
    indices[h] = pos_base * g.Hkv + h;

  if (g.layer_timing_enabled) {
    std::fill(g.layer_timing_cpu_ms.begin(), g.layer_timing_cpu_ms.end(), 0.0);
    g.layer_timing_op_ms.clear();
    g.layer_timing_t0.clear();
    g.layer_timing_gap_ms_total = 0.0;
    g.layer_timing_gap_count = 0;
    g.layer_timing_have_last_ask_false = false;
  }
  if (!qwen4exp_stage_layer13_before_graph(g, pos_base)) {
    LOG_WARN("[OPTION-B-STAGE] declining graph replay because the Layer-13 "
             "staging proof could not be queued\n");
    return false;
  }
  const auto start = std::chrono::steady_clock::now();
  const auto replay_start = std::chrono::steady_clock::now();
  if (ggml_backend_sched_graph_compute_async(g.sched, g.graph) !=
      GGML_STATUS_SUCCESS)
    return false;
  const auto replay_submit_end = std::chrono::steady_clock::now();
  if (g.stage_enabled && g.stage_compute_end_event) {
    cudaStream_t graph_stream =
        moestream_ggml_backend_cuda_get_stream(g.cuda);
    if (!graph_stream ||
        cudaEventRecord(g.stage_compute_end_event, graph_stream) !=
            cudaSuccess)
      return false;
  }
  ggml_backend_sched_synchronize(g.sched);
  const auto replay_sync_end = std::chrono::steady_clock::now();
  dump_layer0_diag_values(g);
  if (g.layer13_moe_diag_enabled && g.layer13_moe_input_diag &&
      g.layer13_moe_ids_diag && g.layer13_moe_weights_diag) {
    const size_t input_n = ggml_nelements(g.layer13_moe_input_diag);
    const size_t ids_n = ggml_nelements(g.layer13_moe_ids_diag);
    const size_t weights_n = ggml_nelements(g.layer13_moe_weights_diag);
    std::vector<float> input(input_n), weights(weights_n);
    std::vector<int32_t> ids(ids_n);
    ggml_backend_tensor_get(g.layer13_moe_input_diag, input.data(), 0,
                            input_n * sizeof(float));
    ggml_backend_tensor_get(g.layer13_moe_ids_diag, ids.data(), 0,
                            ids_n * sizeof(int32_t));
    ggml_backend_tensor_get(g.layer13_moe_weights_diag, weights.data(), 0,
                            weights_n * sizeof(float));
    LOG_INFO("[QWEN4EXP-LAYER13-ROUTE-DIAG] ordinary input_fnv64=0x%016llx "
             "ids_fnv64=0x%016llx weights_fnv64=0x%016llx "
             "ids=[%d,%d,%d,%d,%d,%d,%d,%d,%d,%d]\n",
             static_cast<unsigned long long>(qwen4exp_fnv64(
                 input.data(), input_n * sizeof(float))),
             static_cast<unsigned long long>(qwen4exp_fnv64(
                 ids.data(), ids_n * sizeof(int32_t))),
             static_cast<unsigned long long>(qwen4exp_fnv64(
                 weights.data(), weights_n * sizeof(float))),
             ids_n > 0 ? ids[0] : -1, ids_n > 1 ? ids[1] : -1,
             ids_n > 2 ? ids[2] : -1, ids_n > 3 ? ids[3] : -1,
             ids_n > 4 ? ids[4] : -1, ids_n > 5 ? ids[5] : -1,
             ids_n > 6 ? ids[6] : -1, ids_n > 7 ? ids[7] : -1,
             ids_n > 8 ? ids[8] : -1, ids_n > 9 ? ids[9] : -1);
    const char *input_dump = env_layer13_moe_ordinary_input_dump_path();
    if (input_dump) {
      const bool dumped = qwen4exp_dump_f32(input_dump, input.data(), input_n);
      LOG_INFO("[QWEN4EXP-LAYER13-INPUT-DIAG] path=ordinary "
               "dump=%s file=%s elements=%zu\n",
               dumped ? "ok" : "failed", input_dump, input_n);
    }
  }
  if (g.layer13_moe_diag_enabled && g.layer13_moe_diag &&
      env_layer13_moe_dump_path() && !g.layer13_moe_diag_dumped) {
    const size_t n = ggml_nelements(g.layer13_moe_diag);
    std::vector<float> values(n);
    ggml_backend_tensor_get(g.layer13_moe_diag, values.data(), 0,
                            n * sizeof(float));
    const uint64_t hash = qwen4exp_fnv64(values.data(), n * sizeof(float));
    const bool dumped = qwen4exp_dump_f32(env_layer13_moe_dump_path(),
                                          values.data(), n);
    g.layer13_moe_diag_dumped = dumped;
    LOG_INFO("[QWEN4EXP-LAYER13-MOE-DIAG] path=ordinary-ggml-dispatch "
             "dump=%s file=%s fnv64=0x%016llx elements=%zu\n",
             dumped ? "ok" : "failed", env_layer13_moe_dump_path(),
             static_cast<unsigned long long>(hash), n);
  }
  double host_node_d2h_ms = 0.0;
  double host_node_h2d_ms = 0.0;
  double host_node_callback_span_ms = 0.0;
  double host_node_cpu_ms = 0.0;
  double host_node_callback_total_ms = 0.0;
  double host_node_callback_guard_wait_ms = 0.0;
  double host_node_callback_body_ms = 0.0;
  bool host_node_all_ok = true;
  if (!g.host_node_contexts.empty()) {
    float d2h_ms = 0.0f;
    float h2d_ms = 0.0f;
    double cpu_ms = 0.0;
    bool all_ok = true;
    for (const auto &host_node : g.host_node_contexts) {
      if (!host_node)
        continue;
      float node_d2h = 0.0f;
      float node_h2d = 0.0f;
      float node_callback_span = 0.0f;
      if (cudaEventElapsedTime(&node_d2h, host_node->d2h_start,
                               host_node->d2h_end) != cudaSuccess ||
          cudaEventElapsedTime(&node_h2d, host_node->h2d_start,
                               host_node->h2d_end) != cudaSuccess) {
        LOG_WARN("[QWEN4EXP-HOST-NODE] event timing unavailable at pos=%d\n",
                 pos_base);
      } else {
        d2h_ms += node_d2h;
        h2d_ms += node_h2d;
      }
      if (cudaEventElapsedTime(&node_callback_span, host_node->d2h_end,
                               host_node->h2d_start) == cudaSuccess)
        host_node_callback_span_ms += node_callback_span;
      cpu_ms += host_node->last_cpu_ms.load(std::memory_order_acquire);
      host_node_callback_total_ms +=
          host_node->last_callback_total_ms.load(std::memory_order_acquire);
      host_node_callback_guard_wait_ms +=
          host_node->last_callback_guard_wait_ms.load(
              std::memory_order_acquire);
      host_node_callback_body_ms +=
          host_node->last_callback_body_ms.load(std::memory_order_acquire);
      all_ok = all_ok && host_node->last_ok.load(std::memory_order_acquire);
      if (host_node->transport_diag && host_node->host_output_gpu_check) {
        const size_t n = host_node->output_bytes / sizeof(float);
        const uint64_t cpu_hash =
            host_node->last_cpu_hash.load(std::memory_order_acquire);
        const uint64_t gpu_hash =
            qwen4exp_fnv64(host_node->host_output_gpu_check,
                           host_node->output_bytes);
        host_node->last_gpu_hash.store(gpu_hash, std::memory_order_release);
        log_layer0_diag_host_values("cpu_output_pre_h2d",
                                   host_node->host_output, n);
        log_layer0_diag_host_values("gpu_output_post_h2d",
                                   host_node->host_output_gpu_check, n);
        float max_abs_delta = 0.0f;
        double sum_delta = 0.0;
        for (size_t i = 0; i < n; ++i) {
          max_abs_delta = std::max(
              max_abs_delta,
              std::abs(host_node->host_output[i] -
                        host_node->host_output_gpu_check[i]));
          sum_delta += static_cast<double>(host_node->host_output_gpu_check[i]) -
                       static_cast<double>(host_node->host_output[i]);
        }
        LOG_INFO("[QWEN4EXP-HOST-NODE-DIAG] layer=%d "
                 "cpu_pre_h2d_fnv64=0x%016llx "
                 "gpu_post_h2d_fnv64=0x%016llx hash_match=%d "
                 "max_abs_delta=%.9g sum_delta=%.9g bytes=%zu\n",
                 host_node->layer_index,
                 static_cast<unsigned long long>(cpu_hash),
                 static_cast<unsigned long long>(gpu_hash),
                 host_node->last_cpu_hash_valid.load(
                         std::memory_order_acquire) &&
                         cpu_hash == gpu_hash
                     ? 1
                     : 0,
                 max_abs_delta, sum_delta, host_node->output_bytes);
        if (host_node->layer_index == 13 &&
            env_layer13_moe_dump_path() && !g.layer13_moe_diag_dumped) {
          const char *path = env_layer13_moe_dump_path();
          const bool dumped = qwen4exp_dump_f32(
              path, host_node->host_output,
              host_node->output_bytes / sizeof(float));
          g.layer13_moe_diag_dumped = dumped;
          LOG_INFO("[QWEN4EXP-LAYER13-MOE-DIAG] path=host-node-CPU "
                   "dump=%s file=%s fnv64=0x%016llx\n",
                   dumped ? "ok" : "failed", path,
                   static_cast<unsigned long long>(cpu_hash));
        }
      }
    }
    const double scheduler_gate_wait_ms =
        qwen4exp_host_node_gate_wait_ms_since_last();
    const double ordinary_cpu_wait_ms =
        qwen4exp_host_node_ordinary_cpu_wait_ms_since_last();
    const double cpu_split_enter_wait_ms =
        qwen4exp_cpu_split_enter_wait_ms_since_last();
    LOG_INFO("[QWEN4EXP-HOST-NODE-TIME] pos=%d nodes=%zu D2H=%.3fms "
             "CPU=%.3fms H2D=%.3fms callback_total=%.3fms "
             "callback_span=%.3fms callback_guard_wait=%.3fms "
             "callback_body=%.3fms "
             "scheduler_gate_wait=%.3fms ordinary_cpu_wait=%.3fms "
             "cpu_split_enter_wait=%.3fms "
             "callback_ok=%d\n",
             pos_base, g.host_node_contexts.size(), d2h_ms, cpu_ms, h2d_ms,
             host_node_callback_total_ms, host_node_callback_span_ms,
             host_node_callback_guard_wait_ms, host_node_callback_body_ms,
             scheduler_gate_wait_ms,
             ordinary_cpu_wait_ms, cpu_split_enter_wait_ms, all_ok ? 1 : 0);
    host_node_d2h_ms = d2h_ms;
    host_node_h2d_ms = h2d_ms;
    host_node_cpu_ms = cpu_ms;
    host_node_all_ok = all_ok;
    if (!all_ok)
      return false;
  }
  if (g.stage_enabled && !g.stage_slots.empty()) {
    float h2d_ms = 0.0f, wait_ms = 0.0f, d2d_ms = 0.0f, compute_ms = 0.0f;
    cudaSetDevice(g.device);
    const ExpertLocation *first_slot = g.stage_slots.front();
    const ExpertLocation *last_slot = g.stage_slots.back();
    cudaEventElapsedTime(&h2d_ms, first_slot->upload_start_event,
                         last_slot->upload_event);
    // This brackets the actual device-side stream fence.  The previous
    // single-slot probe measured upload completion -> wait completion, which
    // excluded the stall when the compute stream arrived before the DMA.
    cudaEventElapsedTime(&wait_ms, g.stage_wait_start_event,
                         g.stage_wait_done_event);
    cudaEventElapsedTime(&d2d_ms, g.stage_d2d_start_event,
                         g.stage_d2d_end_event);
    cudaEventElapsedTime(&compute_ms, g.stage_compute_start_event,
                         g.stage_compute_end_event);
    LOG_INFO("[OPTION-B-STAGE-TIME] pos=%d layer=%d staged=%zu H2D=%.3fms "
             "wait=%.3fms D2D=%.3fms graph_cuda=%.3fms\n",
             pos_base, g.stage_layer, g.stage_slots.size(), h2d_ms, wait_ms,
             d2d_ms, compute_ms);
    for (ExpertLocation *slot : g.stage_slots)
      g.engine->gpu->topology.complete_active_layer_upload_after_compute(slot);
    g.stage_slots.clear();
    g.stage_slot = nullptr;
  }
  ggml_backend_tensor_get(g.output, hidden_states, 0,
                          static_cast<size_t>(g.D) * sizeof(float));
  if (g.stage_enabled) {
    double l1 = 0.0;
    bool finite = true;
    for (int i = 0; i < g.D; ++i) {
      finite = finite && std::isfinite(hidden_states[i]);
      l1 += std::abs(static_cast<double>(hidden_states[i]));
    }
    LOG_INFO("[OPTION-B-STAGE-VERIFY] pos=%d output_finite=%d output_l1=%.6e\n",
             pos_base, finite ? 1 : 0, l1);
    if (!finite)
      return false;
  }
  if (logits) {
    ggml_backend_tensor_get(g.logits, logits, 0,
                            static_cast<size_t>(vocab) * sizeof(float));
  }
  g.decode_last_pos = pos_base;

  // The PLE hash window is host metadata, not numerical recurrent state. Keep
  // it current so a later graph disable/capacity growth can seed the layered
  // path without changing the token-context semantics.
  if (has_ple_layer(engine.model_file.architecture_config)) {
    auto &hist = engine.qwen4exp_ple_tok_hist;
    hist.push_back(token_ids[0]);
    const int keep = std::max(
        0, engine.model_file.architecture_config.ple_ngram_size - 1);
    if (static_cast<int>(hist.size()) > keep)
      hist.erase(hist.begin(), hist.end() - keep);
    engine.qwen4exp_ple_next_pos = pos_base + 1;
  }
  engine.qwen4exp_full_graph_output_ready = true;
  engine.qwen4exp_full_graph_logits_ready = logits != nullptr;
  const double ms = std::chrono::duration<double, std::milli>(
                         std::chrono::steady_clock::now() - start)
                         .count();
  LOG_INFO("[GGML-QWEN-FULL-DECODE-TIME] pos=%d total=%.3f ms\n", pos_base,
           ms);
  if (env_graph_replay_timing_enabled()) {
    double ordinary_cpu_compute_ms = 0.0;
    for (const double layer_ms : g.layer_timing_cpu_ms)
      ordinary_cpu_compute_ms += layer_ms;
    const double ordinary_cpu_gap_ms = g.layer_timing_gap_ms_total;
    const double replay_submit_ms = std::chrono::duration<double, std::milli>(
        replay_submit_end - replay_start).count();
    const double replay_sync_ms = std::chrono::duration<double, std::milli>(
        replay_sync_end - replay_submit_end).count();
    const double replay_total_ms = std::chrono::duration<double, std::milli>(
        replay_sync_end - replay_start).count();
    LOG_INFO(
        "[QWEN4EXP-GRAPH-REPLAY-LEDGER] pos=%d splits=%d host_nodes=%zu "
        "replay_submit=%.3fms replay_sync=%.3fms replay_total=%.3fms "
        "host_node_d2h=%.3fms host_node_cpu=%.3fms host_node_h2d=%.3fms "
        "host_node_ok=%d ordinary_cpu_compute=%.3fms "
        "ordinary_cpu_between_node_gap=%.3fms ordinary_cpu_ledger=%.3fms\n",
        pos_base, ggml_backend_sched_get_n_splits(g.sched),
        g.host_node_contexts.size(), replay_submit_ms, replay_sync_ms,
        replay_total_ms, host_node_d2h_ms, host_node_cpu_ms,
        host_node_h2d_ms, host_node_all_ok ? 1 : 0, ordinary_cpu_compute_ms,
        ordinary_cpu_gap_ms, ordinary_cpu_compute_ms + ordinary_cpu_gap_ms);
  }
  // Per-layer/per-op CPU timing (env: MOESTREAM_QWEN4EXP_GRAPH_LAYER_TIMING=1)
  // -- emitted every 10 tokens (not every token) to avoid log spam; see
  // memory: full-graph-per-layer-timing-instrumentation-plan.md for why
  // this exists and what it answers that nsys/DEBUG counters cannot.
  if (g.layer_timing_enabled &&
      ++g.layer_timing_tokens_since_emit >= 10) {
    g.layer_timing_tokens_since_emit = 0;
    double total_cpu_ms = 0.0;
    std::string per_layer;
    for (size_t l = 0; l < g.layer_timing_cpu_ms.size(); ++l) {
      const double lms = g.layer_timing_cpu_ms[l];
      total_cpu_ms += lms;
      if (lms > 0.01) {
        per_layer += " L" + std::to_string(l) + "=" +
                    std::to_string(static_cast<int>(lms * 100) / 100.0);
      }
    }
    LOG_INFO("[GGML-QWEN-FULL-LAYER-TIMING] pos=%d cpu_total=%.3fms "
             "gap_total=%.3fms (n=%ld gaps, dispatch/barrier/scheduling "
             "overhead BETWEEN CPU nodes, invisible to any op-bracket "
             "timing)%s\n",
             pos_base, total_cpu_ms, g.layer_timing_gap_ms_total,
             g.layer_timing_gap_count, per_layer.c_str());
    for (const auto &[stem, oms] : g.layer_timing_op_ms) {
      LOG_INFO("[GGML-QWEN-FULL-LAYER-TIMING]   op=%s cpu=%.3fms\n",
               stem.c_str(), oms);
    }
  }
  return true;
}

bool qwen4exp_full_graph_run(StreamingEngine &engine, float *hidden_states,
                             int S, int pos_base, const int *token_ids) {
  // Prefill-only while recurrent and dense KV state writeback is validated.
  if (!env_enabled() || !hidden_states || S <= 1 || pos_base != 0 || !g_graph ||
      !g_graph->ready || g_graph->engine != &engine || g_graph->S != S)
    return false;
  FullGraph &g = *g_graph;
  auto fail = [&](const char *why) {
    LOG_WARN("[GGML-QWEN-FULL] execution declined: %s\n", why);
    return false;
  };
  LOG_INFO(
      "[GGML-QWEN-FULL] execute candidate S=%d gdn=%zu kv=%zu ple_layer=%d\n",
      S, g.gdn_layers.size(), g.kv_layers.size(), g.ple_layer);

  // A fresh position-zero graph starts a new recurrent/KV stream.  This is
  // the same reset boundary used by the established Qwen attention path.
  for (size_t i = 0; i < engine.qwen4exp_conv_history.size(); ++i)
    std::fill(engine.qwen4exp_conv_history[i].begin(),
              engine.qwen4exp_conv_history[i].end(), 0.0f);
  for (size_t i = 0; i < engine.qwen4exp_recur_state.size(); ++i)
    std::fill(engine.qwen4exp_recur_state[i].begin(),
              engine.qwen4exp_recur_state[i].end(), 0.0f);
  std::fill(engine.qwen4exp_ple_tok_hist.begin(),
            engine.qwen4exp_ple_tok_hist.end(),
            static_cast<int32_t>(
                engine.model_file.architecture_config.ple_eos_token_id));
  engine.qwen4exp_ple_next_pos = 0;

  for (size_t i = 0; i < g.gdn_layers.size(); ++i) {
    const int l = g.gdn_layers[i];
    if (i >= g.gdn_state_inputs.size() || i >= g.gdn_hist_inputs.size())
      return fail("GDN input bookkeeping mismatch");
    const size_t state_n = static_cast<size_t>(g.gdn_hd) * g.gdn_hd * g.Hv_gdn;
    const size_t hist_n = static_cast<size_t>(g.d_conv - 1) * g.conv_dim;
    if (l < 0 || l >= static_cast<int>(engine.qwen4exp_recur_state.size()) ||
        l >= static_cast<int>(engine.qwen4exp_conv_history.size()) ||
        engine.qwen4exp_recur_state[l].size() != state_n ||
        engine.qwen4exp_conv_history[l].size() != hist_n)
      return fail("GDN state/history size mismatch");
    std::memcpy(g.gdn_state_inputs[i], engine.qwen4exp_recur_state[l].data(),
                state_n * sizeof(float));
    std::memcpy(g.gdn_hist_inputs[i], engine.qwen4exp_conv_history[l].data(),
                hist_n * sizeof(float));
  }
  if (g.ple_layer >= 0 && g.ple_hist_input) {
    const auto &c = engine.model_file.architecture_config;
    // The engine keeps PLE history in the conv layout [HC*D, hist].  The
    // PLE embedding uses [n_heads, embd_per_layer], but those are not the
    // storage dimensions of the causal-convolution state.
    const size_t hist_n = static_cast<size_t>(c.ple_conv_kernel - 1) *
                          c.ple_ngram_size * g.HC * g.D;
    if (g.ple_layer >=
            static_cast<int>(engine.qwen4exp_ple_conv_history.size()) ||
        engine.qwen4exp_ple_conv_history[g.ple_layer].size() != hist_n)
      return fail("PLE history size mismatch");
    std::memcpy(g.ple_hist_input,
                engine.qwen4exp_ple_conv_history[g.ple_layer].data(),
                hist_n * sizeof(float));
  } else if (has_ple_layer(engine.model_file.architecture_config)) {
    return fail("PLE graph input is missing");
  }
  if (g.host_inputs.empty() ||
      engine.hc_stream_buf.size() < static_cast<size_t>(g.D) * g.HC * S)
    return fail("expanded HC input is missing or too small");
  std::memcpy(g.host_inputs[0], engine.hc_stream_buf.data(),
              static_cast<size_t>(g.D) * g.HC * S * sizeof(float));

  // Real-hardware incident (2026-09-09, GLM5NEXT side): a 480-token prompt
  // crashed the whole machine (hard reboot required, no surviving kernel/
  // OOM log -- this box has no persistent journald storage) with the last
  // surviving application-log line printed just before this exact kind of
  // compute call. Every OTHER host allocation on this path (arena prep,
  // expert source, graph context) goes through MemoryGuard::try_reserve_
  // host_bytes first; this compute call did not. Re-check REAL, fresh
  // MemAvailable (not the background watchdog's own possibly-500ms-stale
  // allocation_blocked() flag) immediately before committing to it -- a
  // large gap can open between prepare()'s own admission checks and this
  // moment (preload alone can take minutes). Not a confirmed fix for
  // whatever the real mechanism was (no kernel evidence survived to
  // confirm it), just a cheap, additional, defense-in-depth backstop,
  // applied identically to qwen4exp's own graph for the same reason.
  {
    const long avail_mb = MemoryGuard::read_available_mb();
    constexpr long kMinComputeHeadroomMb =
        10240; // 10 GiB, per explicit user instruction after the 2026-09-09
               // crash
    if (avail_mb >= 0 && avail_mb < kMinComputeHeadroomMb) {
      LOG_ERROR("[GGML-QWEN-FULL] refusing graph compute: MemAvailable=%ld MiB "
                "< %ld MiB safety floor -- falling back to the legacy path\n",
                avail_mb, kMinComputeHeadroomMb);
      return false;
    }
  }

  const auto t_compute_start = std::chrono::steady_clock::now();
  ggml_status st = ggml_backend_sched_graph_compute_async(g.sched, g.graph);
  const auto t_submit_end = std::chrono::steady_clock::now();
  if (st != GGML_STATUS_SUCCESS) {
    LOG_ERROR("[GGML-QWEN-FULL] graph compute failed: status=%d\n", (int)st);
    return false;
  }
  ggml_backend_sched_synchronize(g.sched);
  const auto t_sync_end = std::chrono::steady_clock::now();
  ggml_backend_tensor_get(g.output, hidden_states, 0,
                          static_cast<size_t>(g.D) * S * sizeof(float));
  if (const char *split_dump_path = env_layer0_split_dump_path()) {
    std::unordered_map<std::string, size_t> snapshot_occurrences;
    for (const auto &entry : g.layer0_split_snapshots) {
      const size_t occurrence = snapshot_occurrences[entry.first]++;
      ggml_tensor *t = entry.second;
      if (!t || !t->data)
        continue;
      const size_t n = static_cast<size_t>(ggml_nelements(t));
      std::vector<float> values(n);
      ggml_backend_tensor_get(t, values.data(), 0, n * sizeof(float));
      std::string path = std::string(split_dump_path) + ".prefill." +
                         entry.first;
      if (occurrence != 0)
        path += "." + std::to_string(occurrence);
      path += ".f32";
      if (qwen4exp_dump_f32(path.c_str(), values.data(), n))
        LOG_INFO("[QWEN4EXP-LAYER0-SPLIT] prefill snapshot=%s elements=%zu "
                 "fnv64=0x%016llx path=%s\n",
                 entry.first.c_str(), n,
                 static_cast<unsigned long long>(qwen4exp_fnv64(
                     values.data(), n * sizeof(float))),
                 path.c_str());
    }
  }
  const auto t_output_end = std::chrono::steady_clock::now();
  dump_layer0_diag_values(g);

  for (size_t i = 0; i < g.gdn_layers.size(); ++i) {
    const int l = g.gdn_layers[i];
    const size_t state_n = static_cast<size_t>(g.gdn_hd) * g.gdn_hd * g.Hv_gdn;
    const size_t hist_n = static_cast<size_t>(g.d_conv - 1) * g.conv_dim;
    // gdn_state_out (build_gdn's *state_out) is native layout;
    // qwen4exp_recur_state is host-legacy layout for legacy CPU consumption.
    // See gdn_state_transpose_host.
    std::vector<float> native_state(state_n);
    ggml_backend_tensor_get(g.gdn_state_out[i], native_state.data(), 0,
                            state_n * sizeof(float));
    gdn_state_transpose_host(engine.qwen4exp_recur_state[l].data(),
                             native_state.data(), g.gdn_hd, g.Hv_gdn);
    ggml_backend_tensor_get(g.gdn_hist_out[i],
                            engine.qwen4exp_conv_history[l].data(), 0,
                            hist_n * sizeof(float));
  }
  for (size_t i = 0; i < g.kv_layers.size(); ++i) {
    const int l = g.kv_layers[i];
    if (l < 0 || l >= static_cast<int>(engine.Kc.size()) ||
        l >= static_cast<int>(engine.Vc.size()) || !engine.Kc[l] ||
        !engine.Vc[l])
      return fail("dense KV cache is not allocated");
    const size_t kv_n = static_cast<size_t>(g.S) * g.Hkv * g.hd;
    std::vector<float> k(kv_n), v(kv_n);
    ggml_backend_tensor_get(g.kv_k[i], k.data(), 0, kv_n * sizeof(float));
    ggml_backend_tensor_get(g.kv_v[i], v.data(), 0, kv_n * sizeof(float));
    const size_t row = static_cast<size_t>(g.Hkv) * g.hd;
    for (int s = 0; s < g.S && s + pos_base < engine.max_t; ++s) {
      std::memcpy(engine.Kc[l] + static_cast<size_t>(s + pos_base) * row,
                  k.data() + static_cast<size_t>(s) * row, row * sizeof(float));
      std::memcpy(engine.Vc[l] + static_cast<size_t>(s + pos_base) * row,
                  v.data() + static_cast<size_t>(s) * row, row * sizeof(float));
    }
  }
  if (g.ple_layer >= 0 && g.ple_hist_out) {
    const auto &c = engine.model_file.architecture_config;
    const size_t hist_n = static_cast<size_t>(c.ple_conv_kernel - 1) *
                          c.ple_ngram_size * g.HC * g.D;
    auto &dst = engine.qwen4exp_ple_conv_history[g.ple_layer];
    ggml_backend_tensor_get(g.ple_hist_out, dst.data(), 0,
                            hist_n * sizeof(float));
    // Prefill is the only place where the full S>1 PLE graph writes the
    // recurrent convolution history.  Keep a diagnostic copy of that exact
    // post-compute state so it can be compared with Bee's prefill cache
    // without disturbing the state owner or adding a graph node.
    if (!g.decode_mode) {
      if (const char *split_dump_path = env_layer0_split_dump_path()) {
        const std::string path = std::string(split_dump_path) +
                                 ".prefill.ple_history_after.f32";
        qwen4exp_dump_f32(path.c_str(), dst.data(), hist_n);
        LOG_INFO("[QWEN4EXP-LAYER0-SPLIT] prefill PLE history dump path=%s "
                 "elements=%zu fnv64=0x%016llx\n",
                 path.c_str(), hist_n,
                 static_cast<unsigned long long>(qwen4exp_fnv64(
                     dst.data(), hist_n * sizeof(float))));
      }
    }
    engine.qwen4exp_ple_tok_hist.clear();
    const int keep = std::max(0, c.ple_ngram_size - 1);
    const int start = std::max(0, S - keep);
    for (int i = start; i < S; ++i)
      engine.qwen4exp_ple_tok_hist.push_back(token_ids[i]);
    engine.qwen4exp_ple_next_pos = S;
  }
  if (g.partial_layer_ids_out) {
    // Phase 1 all-or-nothing partial-layer dispatch: this prompt's real
    // layer-N routing, read back once per prefill. Coverage = fraction of
    // this prompt's DISTINCT selected experts that fall inside the
    // usage-ranked top-N pinned set (plan_partial_layer). Committing
    // decode to the GPU-only path only when coverage clears the
    // (default: 100%) threshold is what keeps build_moe_partial_gpu's
    // silent-miss risk rare in practice, not just in theory -- see that
    // function's own comment for what happens on the rare miss anyway.
    const auto &c = engine.model_file.architecture_config;
    const size_t n = static_cast<size_t>(c.topk) * static_cast<size_t>(S);
    std::vector<int32_t> ids_host(n);
    ggml_backend_tensor_get(g.partial_layer_ids_out, ids_host.data(), 0,
                            n * sizeof(int32_t));
    std::unordered_set<int32_t> used(ids_host.begin(), ids_host.end());
    std::unordered_set<int> gpu_set(
        engine.qwen4exp_partial_layer_gpu_ids.begin(),
        engine.qwen4exp_partial_layer_gpu_ids.end());
    size_t covered = 0;
    for (int32_t id : used)
      if (id >= 0 && gpu_set.count(id))
        ++covered;
    const double coverage =
        used.empty() ? 1.0
                     : static_cast<double>(covered) / static_cast<double>(used.size());
    engine.qwen4exp_partial_layer_committed =
        coverage >= engine.qwen4exp_partial_layer_min_coverage;
    // Root-cause diagnostic (2026-09-18): last-prefill-token router picks,
    // in the SAME format as legacy dispatch's own TRACE-ROUTER line
    // (moe_router_service.cpp, "[TRACE l=<layer>] router picks: e<id> ...")
    // -- for a fixed prompt, the last prefill token's input is IDENTICAL
    // regardless of dispatch path (no generated-token drift can have
    // contaminated it yet), so this is the one directly comparable,
    // uncontaminated data point between the two paths. Diff this line
    // against a legacy run's TRACE-ROUTER output (log_level=VERBOSE,
    // trace_filter=TRACE-ROUTER) for the same prompt/layer to test
    // [[qwen4exp-legacy-vs-fullgraph-routing-divergence]]'s BF16-router-
    // precision hypothesis.
    if (S > 0) {
      // Cross-check: what raw probability does THIS SAME build_moe
      // invocation's own probs_out assign to each id its own ids_out
      // selected? If this doesn't match a plausible top-k-by-probability
      // ranking, the divergence is internal to full-graph's own
      // ids/probs pairing, not (only) a legacy-vs-full-graph difference.
      std::vector<float> probs_host2;
      if (g.partial_layer_probs_out) {
        const size_t np2 =
            static_cast<size_t>(c.n_experts) * static_cast<size_t>(S);
        probs_host2.assign(np2, 0.0f);
        ggml_backend_tensor_get(g.partial_layer_probs_out, probs_host2.data(),
                                0, np2 * sizeof(float));
      }
      std::string picks;
      for (int k = 0; k < c.topk; ++k) {
        int32_t eid = ids_host[(S - 1) * c.topk + k];
        picks += " e" + std::to_string(eid);
        if (!probs_host2.empty()) {
          const float *col2 =
              probs_host2.data() + static_cast<size_t>(S - 1) * c.n_experts;
          const float p = (eid >= 0 && eid < c.n_experts) ? col2[eid] : -1.0f;
          picks += "(p=" + std::to_string(p) + ")";
        }
      }
      LOG_INFO("[TRACE-ROUTER-FULLGRAPH l=%d] router picks (last prefill "
               "token):%s\n",
               engine.qwen4exp_partial_layer, picks.c_str());
      // Root-cause diagnostic (2026-09-18), continued: dump full-graph's
      // own top-20 by raw post-softmax probability for the same token, so
      // a legacy pick NOT in full-graph's top-10 can be checked for
      // "just outside, nearly tied" (rank ~11-20, close probability) vs
      // "nowhere close" (rank in the hundreds) -- distinguishes a
      // precision-flips-a-near-tie explanation from a genuinely different
      // computation on one side.
      if (g.partial_layer_probs_out) {
        const size_t np = static_cast<size_t>(c.n_experts) * static_cast<size_t>(S);
        std::vector<float> probs_host(np);
        ggml_backend_tensor_get(g.partial_layer_probs_out, probs_host.data(), 0,
                                np * sizeof(float));
        const float *col = probs_host.data() +
                           static_cast<size_t>(S - 1) * c.n_experts;
        std::vector<int> order(c.n_experts);
        for (int e = 0; e < c.n_experts; ++e)
          order[e] = e;
        std::partial_sort(order.begin(), order.begin() + std::min(20, c.n_experts),
                          order.end(),
                          [&](int a, int b) { return col[a] > col[b]; });
        std::string top20;
        for (int k = 0; k < std::min(20, c.n_experts); ++k)
          top20 += " e" + std::to_string(order[k]) + "(p=" +
                   std::to_string(col[order[k]]) + ")";
        LOG_INFO("[TRACE-ROUTER-FULLGRAPH-TOP20 l=%d] last prefill token:%s\n",
                 engine.qwen4exp_partial_layer, top20.c_str());
      }
      if (g.partial_layer_bias_out) {
        std::vector<float> bias_host(static_cast<size_t>(c.n_experts));
        ggml_backend_tensor_get(g.partial_layer_bias_out, bias_host.data(), 0,
                                bias_host.size() * sizeof(float));
        float bmin = bias_host[0], bmax = bias_host[0], bsum = 0.0f;
        for (float v : bias_host) {
          bmin = std::min(bmin, v);
          bmax = std::max(bmax, v);
          bsum += v;
        }
        LOG_INFO("[TRACE-ROUTER-FULLGRAPH-BIAS l=%d] exp_probs_b.bias: "
                 "min=%.6f max=%.6f mean=%.6f (n=%d)\n",
                 engine.qwen4exp_partial_layer, bmin, bmax,
                 bsum / static_cast<float>(c.n_experts), c.n_experts);
      }
    }
    // Router-confidence diagnostic: is this layer's routing genuinely
    // peaked (real signal) or effectively near-uniform noise? A uniform
    // distribution over n_experts would put EVERY token's top-1
    // probability at 1/n_experts; a working router should sit well above
    // that. Directly tests the "near-zero-collision selection" anomaly
    // this whole check exists because of -- see
    // qwen4exp-full-graph-decode-benchmark-baseline memory's follow-on
    // note for the statistics that motivated it.
    if (g.partial_layer_probs_out) {
      const size_t np = static_cast<size_t>(c.n_experts) * static_cast<size_t>(S);
      std::vector<float> probs_host(np);
      ggml_backend_tensor_get(g.partial_layer_probs_out, probs_host.data(), 0,
                              np * sizeof(float));
      double top1_min = 1.0, top1_max = 0.0, top1_sum = 0.0;
      for (int s = 0; s < S; ++s) {
        float m = 0.0f;
        const float *col = probs_host.data() + static_cast<size_t>(s) * c.n_experts;
        for (int e = 0; e < c.n_experts; ++e)
          m = std::max(m, col[e]);
        top1_min = std::min(top1_min, static_cast<double>(m));
        top1_max = std::max(top1_max, static_cast<double>(m));
        top1_sum += m;
      }
      const double top1_mean = S > 0 ? top1_sum / S : 0.0;
      const double uniform_baseline = 1.0 / static_cast<double>(c.n_experts);
      LOG_INFO(
          "[PARTIAL-LAYER-DIAG] layer=%d router top-1 confidence over %d "
          "token(s): min=%.4f mean=%.4f max=%.4f (uniform/no-signal "
          "baseline would be %.4f) -- %s\n",
          engine.qwen4exp_partial_layer, S, top1_min, top1_mean, top1_max,
          uniform_baseline,
          top1_mean > uniform_baseline * 3.0
              ? "routing looks peaked/real"
              : "routing looks close to uniform -- possible router issue");
    }
    LOG_INFO(
        "[PARTIAL-LAYER] layer=%d prefill routing: %zu distinct expert(s) "
        "selected this prompt, %zu covered by the top-%d pinned set "
        "(%.1f%% coverage, threshold %.1f%%) -- decode will run %s for "
        "this layer\n",
        engine.qwen4exp_partial_layer, used.size(), covered,
        engine.qwen4exp_partial_layer_top_n, coverage * 100.0,
        engine.qwen4exp_partial_layer_min_coverage * 100.0,
        engine.qwen4exp_partial_layer_committed
            ? "GPU-ONLY (committed)"
            : "the normal CPU/GPU path (insufficient coverage)");
  }
  const auto t_state_end = std::chrono::steady_clock::now();
  const auto ms = [](const auto &a, const auto &b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
  };
  LOG_INFO(
      "[GGML-QWEN-FULL-TIME] submit=%.3f ms sync=%.3f ms output_get=%.3f ms "
      "state_cache_get=%.3f ms total=%.3f ms\n",
      ms(t_compute_start, t_submit_end), ms(t_submit_end, t_sync_end),
      ms(t_sync_end, t_output_end), ms(t_output_end, t_state_end),
      ms(t_compute_start, t_state_end));
  engine.qwen4exp_full_graph_output_ready = true;
  return true;
}

bool qwen4exp_two_layer_s1_diag_run(StreamingEngine &engine, int pos_base,
                                    const int *token_ids) {
  if (!env_two_layer_s1_diag_enabled() ||
      engine.family_ops.family != ModelFamily::Qwen4Exp || pos_base < 0)
    return false;
  if (g_two_layer_diag_done)
    return true;

  const auto &c = engine.model_file.architecture_config;
  if (c.n_layers < 2 || engine.max_t <= 0 || pos_base >= engine.max_t ||
      engine.hc_stream_buf.size() <
          static_cast<size_t>(c.hidden) * c.hc_mult)
    return false;
  const int dev = engine.gpu_device_id >= 0 ? engine.gpu_device_id : 0;
  const auto build_start = std::chrono::steady_clock::now();

  auto g = std::make_unique<FullGraph>();
  g->engine = &engine;
  g->device = dev;
  g->S = 1;
  g->D = c.hidden;
  g->HC = c.hc_mult;
  g->n_layers = 2;
  g->H = c.n_heads;
  g->Hkv = std::max(1, c.n_head_kv);
  g->hd = c.head_dim;
  g->Hk_gdn = c.ssm_group_count;
  g->Hv_gdn = c.ssm_time_step_rank;
  g->gdn_hd = c.ssm_state_size;
  g->key_dim = g->Hk_gdn * g->gdn_hd;
  g->value_dim = c.ssm_inner_size;
  g->conv_dim = 2 * g->key_dim + g->value_dim;
  g->d_conv = c.ssm_conv_kernel;
  g->decode_mode = true;
  g->kv_capacity = env_decode_kv_capacity(engine.max_t, pos_base + 1);
  g->decode_last_pos = pos_base - 1;
  g->two_layer_diag = true;
  // Reuse the existing HC boundary instrumentation when the caller also
  // requests a split dump. This keeps the side-graph internals observable
  // without changing the normal two-layer boundary API or graph topology.
  g->layer0_diag_enabled = env_layer0_split_dump_path() != nullptr;
  g->layer_timing_verbose_enabled = env_layer_timing_verbose();

  g->cuda = static_cast<ggml_backend_t>(ggml_shared_backend(dev));
  g->cpu = ggml_backend_cpu_init();
  if (!g->cuda || !g->cpu)
    return false;
  verify_cuda_graphs_compiled_in(g->cuda);
  g->cpu_threads = graph_cpu_threads();
  ggml_backend_cpu_set_n_threads(g->cpu, g->cpu_threads);
  ggml_threadpool_params tp = ggml_threadpool_params_default(g->cpu_threads);
  g->cpu_threadpool = ggml_threadpool_new(&tp);
  if (g->cpu_threadpool)
    ggml_backend_cpu_set_threadpool(g->cpu, g->cpu_threadpool);

  // Only the first two layers can be referenced by this graph. This avoids
  // charging/reserving the remaining 46 layers' expert source bytes when a
  // caller has not enabled the full preloaded DDR arena.
  const size_t expert_bytes = full_expert_source_bytes(engine, g->n_layers);
  if (expert_bytes > 0 &&
      !MemoryGuard::try_reserve_host_bytes(expert_bytes))
    return false;
  g->host_reservation_bytes = expert_bytes;
  g->host_precharged_bytes = expert_bytes;
  if (!init_device_dense_arena(*g) || !init_device_state_arena(*g) ||
      !init_device_expert_arena(*g))
    return false;

  ggml_backend_t backends[] = {g->cuda, g->cpu};
  ggml_backend_buffer_type_t sched_bufts[2] = {};
  ggml_backend_buffer_type_t *sched_bufts_ptr = nullptr;
  if (graph_cuda_host_buffers()) {
    sched_bufts[0] = ggml_backend_get_default_buffer_type(g->cuda);
    sched_bufts[1] = ggml_backend_dev_host_buffer_type(
        ggml_backend_get_device(g->cuda));
    if (sched_bufts[1])
      sched_bufts_ptr = sched_bufts;
  }
  // op_offload MUST stay false -- see qwen4exp_full_graph_prepare's own
  // comment for the real S>=32 host-weight-offload OOM/crash this avoids.
  g->sched = ggml_backend_sched_new(backends, sched_bufts_ptr, 2,
                                    kGraphSize, graph_scheduler_parallel(),
                                    false);
  if (!g->sched)
    return false;
  if (g->layer_timing_verbose_enabled) {
    g->layer_timing_cpu_ms.assign(static_cast<size_t>(g->n_layers), 0.0);
    ggml_backend_sched_set_eval_callback(g->sched, layer_timing_eval_callback,
                                         g.get());
  }

  std::vector<float> ple;
  if (!prepare_ple_embedding(*g, ple, 1, pos_base, token_ids) &&
      has_ple_layer(c))
    return false;
  if (!build_graph(*g, has_ple_layer(c) ? &ple : nullptr))
    return false;

  // Seed only this diagnostic graph's private state/cache copies. The live
  // engine state is never touched by the replay or its state-update nodes.
  const size_t state_n = static_cast<size_t>(g->gdn_hd) * g->gdn_hd * g->Hv_gdn;
  const size_t hist_n = static_cast<size_t>(g->d_conv - 1) * g->conv_dim;
  for (size_t i = 0; i < g->gdn_layers.size(); ++i) {
    const int l = g->gdn_layers[i];
    if (l < 0 || l >= static_cast<int>(engine.qwen4exp_recur_state.size()) ||
        l >= static_cast<int>(engine.qwen4exp_conv_history.size()) ||
        engine.qwen4exp_recur_state[l].size() != state_n ||
        engine.qwen4exp_conv_history[l].size() != hist_n)
      return false;
    std::vector<float> native(state_n);
    gdn_state_transpose_host(native.data(), engine.qwen4exp_recur_state[l].data(),
                              g->gdn_hd, g->Hv_gdn);
    ggml_backend_tensor_set(g->gdn_state_storage[i], native.data(), 0,
                            state_n * sizeof(float));
    ggml_backend_tensor_set(g->gdn_hist_storage[i],
                            engine.qwen4exp_conv_history[l].data(), 0,
                            hist_n * sizeof(float));
  }
  const size_t kv_row = static_cast<size_t>(g->Hkv) * g->hd;
  const size_t kv_bytes = static_cast<size_t>(pos_base) * kv_row * sizeof(float);
  for (size_t i = 0; i < g->kv_layers.size(); ++i) {
    const int l = g->kv_layers[i];
    if (l < 0 || l >= static_cast<int>(engine.Kc.size()) ||
        l >= static_cast<int>(engine.Vc.size()) || !engine.Kc[l] ||
        !engine.Vc[l])
      return false;
    if (kv_bytes > 0) {
      ggml_backend_tensor_set(g->kv_state_k[i], engine.Kc[l], 0, kv_bytes);
      ggml_backend_tensor_set(g->kv_state_v[i], engine.Vc[l], 0, kv_bytes);
    }
  }
  // Layer 1 on this model owns the PLE convolution state.  The diagnostic
  // graph has private state tensors, so it must receive the same history as
  // production before replay; otherwise Layer 0 can match while Layer 1's
  // attention input diverges.
  if (g->ple_layer >= 0 && g->ple_hist_storage) {
    const size_t ple_n = static_cast<size_t>(c.ple_conv_kernel - 1) *
                         c.ple_ngram_size * g->HC * g->D;
    if (g->ple_layer >=
            static_cast<int>(engine.qwen4exp_ple_conv_history.size()) ||
        engine.qwen4exp_ple_conv_history[g->ple_layer].size() != ple_n)
      return false;
    ggml_backend_tensor_set(
        g->ple_hist_storage,
        engine.qwen4exp_ple_conv_history[g->ple_layer].data(), 0,
        ple_n * sizeof(float));
    if (env_two_layer_s1_diag_dump_path()) {
      std::vector<float> history(ple_n);
      ggml_backend_tensor_get(g->ple_hist_storage, history.data(), 0,
                              ple_n * sizeof(float));
      const std::string path =
          std::string(env_two_layer_s1_diag_dump_path()) +
          ".ple_history_before.f32";
      qwen4exp_dump_f32(path.c_str(), history.data(), ple_n);
    }
  }
  if (!phase2_feed_common_inputs(*g, pos_base))
    return false;
  std::memcpy(g->input->data, engine.hc_stream_buf.data(),
              static_cast<size_t>(g->D) * g->HC * sizeof(float));

  const auto build_done = std::chrono::steady_clock::now();
  const auto replay_start = std::chrono::steady_clock::now();
  if (ggml_backend_sched_graph_compute_async(g->sched, g->graph) !=
      GGML_STATUS_SUCCESS)
    return false;
  ggml_backend_sched_synchronize(g->sched);
  if (g->layer0_diag_enabled)
    dump_layer0_diag_values(*g);
  const double replay_ms = std::chrono::duration<double, std::milli>(
                               std::chrono::steady_clock::now() - replay_start)
                               .count();

  const char *dump_base = env_two_layer_s1_diag_dump_path();
  auto dump_boundary = [&](const char *label, ggml_tensor *tensor) {
    if (!tensor)
      return;
    const size_t n = static_cast<size_t>(ggml_nelements(tensor));
    std::vector<float> values(n);
    ggml_backend_tensor_get(tensor, values.data(), 0, n * sizeof(float));
    const uint64_t hash = qwen4exp_fnv64(values.data(), n * sizeof(float));
    LOG_INFO("[QWEN4EXP-TWO-LAYER-S1] boundary=%s elements=%zu "
             "fnv64=0x%016llx\n",
             label, n, static_cast<unsigned long long>(hash));
    if (dump_base) {
      std::string path = std::string(dump_base) + "." + label + ".f32";
      const bool ok = qwen4exp_dump_f32(path.c_str(), values.data(), n);
      LOG_INFO("[QWEN4EXP-TWO-LAYER-S1] dump=%s path=%s\n",
               ok ? "ok" : "failed", path.c_str());
    }
  };
  dump_boundary("layer0_post_ffn", g->two_layer_l0_post_ffn);
  dump_boundary("layer1_pre_attn", g->two_layer_l1_pre_attn);
  dump_boundary("layer1_post_attn", g->two_layer_l1_post_attn);
  dump_boundary("layer1_post_ffn", g->two_layer_l1_post_ffn);
  LOG_INFO("[QWEN4EXP-TWO-LAYER-S1] graph built_layers=%d nodes=%d "
           "splits=%d build_to_replay=%.3f ms replay=%.3f ms\n",
           g->n_layers, ggml_graph_n_nodes(g->graph),
           ggml_backend_sched_get_n_splits(g->sched),
           std::chrono::duration<double, std::milli>(build_done - build_start)
               .count(),
           replay_ms);
  g->ready = true;
  g_two_layer_diag_graph = std::move(g);
  g_two_layer_diag_done = true;
  return true;
}

#else

bool qwen4exp_full_graph_enabled() { return false; }
bool qwen4exp_full_graph_decode_enabled() { return false; }
void qwen4exp_full_graph_configure(bool, bool) {}
int qwen4exp_full_graph_device_expert_layers() { return 0; }
bool qwen4exp_full_graph_device_expert_trailing() { return false; }
bool qwen4exp_full_graph_prepare(StreamingEngine &, int, int, const int *) {
  return false;
}
bool qwen4exp_full_graph_decode_prepare(StreamingEngine &, int, const int *) {
  return false;
}
bool qwen4exp_full_graph_run(StreamingEngine &, float *, int, int,
                             const int *) {
  return false;
}
bool qwen4exp_full_graph_decode_run(StreamingEngine &, float *, int,
                                    const int *, float *, int) {
  return false;
}
bool qwen4exp_two_layer_s1_diag_run(StreamingEngine &, int, const int *) {
  return false;
}
void qwen4exp_full_graph_shutdown() {}

#endif
