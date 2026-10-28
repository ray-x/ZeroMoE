// Qwen4Exp full graph: CUDA host-node / GPU-stage / usage-track callbacks, CPU-execution
// coordination between ordinary CPU nodes and host nodes, memory telemetry.
//
// Split out of the former single-file ggml_qwen4exp_graph.cpp; shared
// declarations live in ggml_qwen4exp_graph_internal.hpp.

#include <cstdio>
#include "ggml_qwen4exp_graph_internal.hpp"

#if defined(ZEROMOE_USE_GGML_CUDA)

using namespace qwen4exp_graph;

namespace qwen4exp_graph {

// Verbose memory telemetry for the partial/staged graph. This
// intentionally samples both CUDA and Linux memory at the same boundary so a
// failure can distinguish VRAM exhaustion from host/cgroup pressure.  It is
// disabled below VERBOSE logging, so normal execution pays no /proc or
// cudaMemGetInfo overhead.
bool qwen4exp_memory_trace_enabled() {
  return Logger::instance().would_log(LogLevel::VERBOSE);
}

long qwen4exp_proc_kb(const char *path, const char *key) {
  std::ifstream input(path);
  std::string line;
  while (std::getline(input, line)) {
    if (line.rfind(key, 0) != 0)
      continue;
    std::istringstream fields(line.substr(std::strlen(key)));
    long value = -1;
    if (fields >> value)
      return value;
  }
  return -1;
}

long long qwen4exp_proc_bytes(const char *path) {
  std::ifstream input(path);
  long long value = -1;
  if (input >> value)
    return value;
  return -1;
}

void qwen4exp_memory_snapshot(const char *tag, int device,
                                     const char *detail) {
  if (!qwen4exp_memory_trace_enabled())
    return;
  if (device < 0)
    cudaGetDevice(&device);
  size_t free_bytes = 0, total_bytes = 0;
  cudaError_t cuda_status = cudaMemGetInfo(&free_bytes, &total_bytes);
  const long available_kb = qwen4exp_proc_kb("/proc/meminfo", "MemAvailable:");
  const long total_kb = qwen4exp_proc_kb("/proc/meminfo", "MemTotal:");
  const long rss_kb = qwen4exp_proc_kb("/proc/self/status", "VmRSS:");
  const long peak_kb = qwen4exp_proc_kb("/proc/self/status", "VmPeak:");
  const long long cgroup_current =
      qwen4exp_proc_bytes("/sys/fs/cgroup/memory.current");
  const long long cgroup_max = qwen4exp_proc_bytes("/sys/fs/cgroup/memory.max");
  const auto gib = [](unsigned long long bytes) {
    return static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0);
  };
  if (cuda_status == cudaSuccess) {
    Logger::instance().log(LogLevel::VERBOSE, "[QWEN4EXP-MEM] tag=%s pid=%d dev=%d vram_free=%.3fGiB "
             "vram_total=%.3fGiB ram_available=%ldMiB ram_total=%ldMiB "
             "rss=%ldMiB peak=%ldMiB cgroup_current=%lld cgroup_max=%lld%s%s\n",
             tag ? tag : "unknown", static_cast<int>(getpid()), device,
             gib(free_bytes), gib(total_bytes), available_kb / 1024,
             total_kb / 1024, rss_kb / 1024, peak_kb / 1024,
             cgroup_current, cgroup_max, detail ? " detail=" : "",
             detail ? detail : "");
  } else {
    Logger::instance().log(LogLevel::VERBOSE, "[QWEN4EXP-MEM-OOM?] tag=%s pid=%d dev=%d cuda_meminfo=%s "
              "ram_available=%ldMiB rss=%ldMiB cgroup_current=%lld "
              "cgroup_max=%lld%s%s\n",
              tag ? tag : "unknown", static_cast<int>(getpid()), device,
              cudaGetErrorString(cuda_status), available_kb / 1024,
              rss_kb / 1024, cgroup_current, cgroup_max,
              detail ? " detail=" : "", detail ? detail : "");
  }
}

void qwen4exp_cuda_failure_snapshot(const char *tag, int device,
                                           cudaError_t status) {
  if (status == cudaErrorMemoryAllocation ||
      status == cudaErrorLaunchOutOfResources ||
      status == cudaErrorInsufficientDriver) {
    qwen4exp_memory_snapshot(tag, device, cudaGetErrorString(status));
  }
}

void CUDART_CB
qwen4exp_cuda_gpu_stage_callback(void *user_data) {
  auto *ctx = static_cast<Qwen4ExpGpuStageContext *>(user_data);
  if (!ctx || ctx->magic != Qwen4ExpGpuStageContext::kMagic)
    return;
  const auto start = std::chrono::steady_clock::now();
  bool ok = true;
  uint8_t *gate_base = ctx->host_combined_bounce;
  uint8_t *up_base = ctx->host_combined_bounce + ctx->combined_up_offset;
  uint8_t *down_base = ctx->host_combined_bounce + ctx->combined_down_offset;
  // Split into three per-projection passes (was one interleaved loop) so
  // gate/up/down memcpy cost can be measured independently -- diagnostic
  // only, same total work and byte-for-byte identical output either way.
  const auto t_gate_start = std::chrono::steady_clock::now();
  for (int k = 0; k < ctx->topk; ++k) {
    const int32_t expert = ctx->host_ids[k];
    if (expert < 0 || expert >= ctx->n_experts) {
      std::memset(gate_base + static_cast<size_t>(k) * ctx->gate_bytes, 0,
                 ctx->gate_bytes);
      continue;
    }
    std::memcpy(gate_base + static_cast<size_t>(k) * ctx->gate_bytes,
               ctx->gate_plane + static_cast<size_t>(expert) * ctx->gate_stride,
               ctx->gate_bytes);
  }
  const auto t_up_start = std::chrono::steady_clock::now();
  for (int k = 0; k < ctx->topk; ++k) {
    const int32_t expert = ctx->host_ids[k];
    if (expert < 0 || expert >= ctx->n_experts) {
      std::memset(up_base + static_cast<size_t>(k) * ctx->up_bytes, 0,
                 ctx->up_bytes);
      continue;
    }
    std::memcpy(up_base + static_cast<size_t>(k) * ctx->up_bytes,
               ctx->up_plane + static_cast<size_t>(expert) * ctx->up_stride,
               ctx->up_bytes);
  }
  const auto t_down_start = std::chrono::steady_clock::now();
  for (int k = 0; k < ctx->topk; ++k) {
    const int32_t expert = ctx->host_ids[k];
    if (expert < 0 || expert >= ctx->n_experts) {
      std::memset(down_base + static_cast<size_t>(k) * ctx->down_bytes, 0,
                 ctx->down_bytes);
      ok = false;
      continue;
    }
    std::memcpy(down_base + static_cast<size_t>(k) * ctx->down_bytes,
               ctx->down_plane + static_cast<size_t>(expert) * ctx->down_stride,
               ctx->down_bytes);
  }
  const auto t_end = std::chrono::steady_clock::now();
  ctx->last_ok.store(ok, std::memory_order_release);
  const double gate_ms =
      std::chrono::duration<double, std::milli>(t_up_start - t_gate_start)
          .count();
  const double up_ms =
      std::chrono::duration<double, std::milli>(t_down_start - t_up_start)
          .count();
  const double down_ms =
      std::chrono::duration<double, std::milli>(t_end - t_down_start).count();
  const double callback_ms = std::chrono::duration<double, std::milli>(
                                 t_end - start)
                                 .count();
  ctx->last_callback_ms.store(callback_ms, std::memory_order_release);
  if (Logger::instance().would_log(LogLevel::DEBUG))
    LOG_DEBUG("[QWEN4EXP-GPU-STAGE-TIMING] layer=%d gate_ms=%.4f up_ms=%.4f "
             "down_ms=%.4f callback_body_ms=%.4f\n",
             ctx->layer_index, gate_ms, up_ms, down_ms, callback_ms);
  if (!ok)
    LOG_ERROR("[QWEN4EXP-GPU-STAGE] layer=%d one or more routed ids were "
              "out of range [0,%d); affected slot(s) zero-filled\n",
              ctx->layer_index, ctx->n_experts);
}

// Enqueues the D2H/callback/H2D sequence for a GPU-stage marker while a CUDA
// stream is being captured -- same technique as
// zeromoe_qwen4exp_cuda_host_node_enqueue below, called from the SAME weak
// hook (dispatch is by ctx->magic, see that function's own top).
bool qwen4exp_cuda_gpu_stage_node_enqueue(const ggml_tensor *node,
                                                 cudaStream_t stream) {
  if (!node || node->op != GGML_OP_CPY || !node->extra || !stream)
    return false;
  auto *ctx = static_cast<Qwen4ExpGpuStageContext *>(node->extra);
  if (ctx->magic != Qwen4ExpGpuStageContext::kMagic || !node->src[2] ||
      !node->src[2]->data || !ctx->host_ids || !ctx->host_combined_bounce ||
      !ctx->device_combined_dst ||
      !ctx->d2h_start || !ctx->d2h_end || !ctx->h2d_start || !ctx->h2d_end)
    return false;

  // Bounded overhead fix (2026-09-25): gate/up/down used to be three
  // independent cudaMemcpyAsync H2D calls. build_gpu_stage_node now
  // allocates them back-to-back from the same bump arena and the callback
  // above writes into one combined pinned buffer at the same relative
  // offsets, so the refresh is ONE H2D covering the whole span -- real,
  // measured CUDA host-node dispatch overhead was found to scale with call
  // COUNT more than payload size (see the usage-router-tap's own finding),
  // so this cuts per-layer API calls from ~9 to ~7 without changing what
  // bytes move. Does NOT fix the layer-to-layer sequential dependency: each
  // staged layer's `ids` still isn't known until that layer's own router
  // runs, which itself depends on the previous layer's output -- so unlike
  // the usage tracker's end-of-graph batch (nothing downstream depends on
  // its output, so every layer's tap could be deferred to one flush), each
  // staged layer's bridge still has to run at its own point in the
  // sequence, one at a time. This is a bounded per-layer fix, not a
  // structural one.
  if (cudaEventRecord(ctx->d2h_start, stream) != cudaSuccess ||
      cudaMemcpyAsync(ctx->host_ids, node->src[2]->data, ctx->ids_bytes,
                      cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
      cudaEventRecord(ctx->d2h_end, stream) != cudaSuccess ||
      cudaLaunchHostFunc(stream, qwen4exp_cuda_gpu_stage_callback, ctx) !=
          cudaSuccess ||
      cudaEventRecord(ctx->h2d_start, stream) != cudaSuccess ||
      cudaMemcpyAsync(ctx->device_combined_dst, ctx->host_combined_bounce,
                      ctx->combined_bytes, cudaMemcpyHostToDevice, stream) !=
          cudaSuccess ||
      cudaEventRecord(ctx->h2d_end, stream) != cudaSuccess)
    return false;
  return true;
}

void CUDART_CB qwen4exp_cuda_usage_track_callback(void *user_data) {
  auto *ctx = static_cast<Qwen4ExpUsageTrackContext *>(user_data);
  if (!ctx || ctx->magic != Qwen4ExpUsageTrackContext::kMagic || !ctx->engine ||
      !ctx->host_ids)
    return;
  const auto start = std::chrono::steady_clock::now();
  const bool hit_log = Logger::instance().would_log(LogLevel::DEBUG);
  const uint64_t token_idx = ctx->token_count++;
  std::string hit_line;
  static FILE *ids_dump = [] {
    const char *path = zeromoe::graphcfg::env("ZEROMOE_QWEN4EXP_ROUTE_IDS_DUMP");
    return path && path[0] ? std::fopen(path, "w") : nullptr;
  }();
  for (int l = 0; l < ctx->n_layers; ++l) {
    int hits = 0;
    const bool has_mask =
        l < static_cast<int>(ctx->hot_mask.size()) && !ctx->hot_mask[l].empty();
    for (int k = 0; k < ctx->topk; ++k) {
      const int32_t expert = ctx->host_ids[static_cast<size_t>(l) * ctx->topk + k];
      if (expert < 0 || expert >= ctx->n_experts)
        continue; // defensive only -- real routing always produces valid ids
      ctx->engine->bump_route_freq(l, expert);
      if (has_mask && ctx->hot_mask[l][expert])
        ++hits;
    }
    if (ids_dump) {
      std::fprintf(ids_dump, "%llu %d", static_cast<unsigned long long>(token_idx), l);
      for (int k = 0; k < ctx->topk; ++k) {
        const int32_t expert = ctx->host_ids[static_cast<size_t>(l) * ctx->topk + k];
        const bool hot = has_mask && expert >= 0 && expert < ctx->n_experts && ctx->hot_mask[l][expert];
        std::fprintf(ids_dump, " %d:%d", expert, hot ? 1 : 0);
      }
      std::fputc('\n', ids_dump);
    }
    if (hit_log)
      hit_line += (has_mask ? std::to_string(hits) : std::string("-")) + ",";
  }
  if (ids_dump)
    std::fflush(ids_dump);
  if (hit_log)
    LOG_DEBUG("[ROUTE-HIT] tok=%llu hits=%s\n",
             static_cast<unsigned long long>(token_idx),
             hit_line.c_str());
  const double ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - start)
                        .count();
  ctx->last_callback_ms.store(ms, std::memory_order_release);
  if (Logger::instance().would_log(LogLevel::DEBUG))
    LOG_DEBUG("[QWEN4EXP-USAGE-TRACK-TIMING] callback_body_ms=%.4f\n", ms);
}

// Enqueues the single D2H + host-callback sequence for the usage-track flush
// marker while a CUDA stream is being captured -- same technique as
// zeromoe_qwen4exp_cuda_host_node_enqueue below, dispatched by magic. No
// H2D write-back: this marker is a pure read-only observer of already-real
// routing, it never feeds back into the compute graph.
bool qwen4exp_cuda_usage_track_node_enqueue(const ggml_tensor *node,
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



// Experimental milestone-2 graph: one static split layer keeps its resident
// GPU branch in the graph and sends the complement through a bounded DDR host
// callback. It is deliberately separate from the validated early-ready path.
bool qwen4exp_host_node_timing_enabled() {
  return Logger::instance().would_log(LogLevel::DEBUG);
}

} // namespace qwen4exp_graph

extern "C" bool zeromoe_qwen4exp_cpu_offload_coordination_enabled() {
  return zeromoe::graphcfg::flag_on("ZEROMOE_QWEN4EXP_HOST_NODE");
}

extern "C" void zeromoe_qwen4exp_host_node_cpu_wait() {
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
    LOG_DEBUG("[DIAG-SCHEDULER] CPU split blocked for %.3f ms "
             "waiting on host-node\n",
             static_cast<double>(wait_ns) / 1000000.0);
  }
}

// Admission protocol for the two independent CPU execution contexts.  The
// retry after increment closes the race where a scheduler split observes an
// idle host callback immediately before the callback publishes itself active.
// This keeps the callback worker pool from competing with an already-running
// ordinary GGML CPU split without taking a mutex on the normal path.
extern "C" void zeromoe_qwen4exp_cpu_split_enter() {
  const auto wait_start = std::chrono::steady_clock::now();
  g_qwen4exp_cpu_execution_mutex.lock();
  g_qwen4exp_cpu_split_enter_wait_ns.fetch_add(
      static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now() - wait_start)
                                .count()),
      std::memory_order_relaxed);
  g_qwen4exp_ordinary_cpu_active.fetch_add(1, std::memory_order_acq_rel);
}

extern "C" void zeromoe_qwen4exp_cpu_split_exit() {
  g_qwen4exp_ordinary_cpu_active.fetch_sub(1, std::memory_order_acq_rel);
  g_qwen4exp_cpu_execution_mutex.unlock();
}

namespace qwen4exp_graph {

double qwen4exp_host_node_ordinary_cpu_wait_ms_since_last() {
  const uint64_t wait_ns = g_qwen4exp_host_node_ordinary_cpu_wait_ns.exchange(
      0, std::memory_order_acq_rel);
  return static_cast<double>(wait_ns) / 1000000.0;
}

double qwen4exp_host_node_gate_wait_ms_since_last() {
  const uint64_t wait_ns = g_qwen4exp_host_node_cpu_gate_wait_ns.exchange(
      0, std::memory_order_acq_rel);
  return static_cast<double>(wait_ns) / 1000000.0;
}

double qwen4exp_cpu_split_enter_wait_ms_since_last() {
  const uint64_t wait_ns = g_qwen4exp_cpu_split_enter_wait_ns.exchange(
      0, std::memory_order_acq_rel);
  return static_cast<double>(wait_ns) / 1000000.0;
}

uint64_t qwen4exp_fnv64(const void *data, size_t bytes) {
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

bool qwen4exp_dump_f32(const char *path, const float *data,
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

void CUDART_CB qwen4exp_cuda_host_node_callback(void *user_data) {
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
  GgmlQwenCpuMoeRunner *runner = ctx->cpu_runner;
  bool ok = false;
  if (ctx->projection_compact) {
    if (!runner)
      runner = &GgmlQwenCpuMoeRunner::instance_for_strided_planes(
          ctx->layer, ctx->n_experts, ctx->d_model, ctx->d_ff, ctx->topk,
          ctx->gate_stride, ctx->up_stride, ctx->down_stride, ctx->gpu_reduce);
    if (!runner->uses_external_threadpool())
      runner->resume_workers();
    if (ctx->gpu_reduce)
      ok = runner->run_strided_planes_down(
          ctx->host_input, ctx->host_output, ctx->layer, ctx->gate_plane,
          ctx->up_plane, ctx->down_plane, ctx->host_ids, ctx->n_experts,
          ctx->d_model, ctx->d_ff, ctx->topk, ctx->gate_stride,
          ctx->up_stride, ctx->down_stride);
    else
      ok = runner->run_strided_planes(
          ctx->host_input, ctx->host_output, ctx->layer, ctx->gate_plane,
          ctx->up_plane, ctx->down_plane, ctx->host_ids, ctx->n_experts,
          ctx->host_weights, ctx->d_model, ctx->d_ff, ctx->topk,
          ctx->gate_stride, ctx->up_stride, ctx->down_stride);
  } else {
    if (!runner)
      runner = &GgmlQwenCpuMoeRunner::instance_for_strided(
          ctx->layer, ctx->n_experts, ctx->d_model, ctx->d_ff, ctx->topk,
          ctx->expert_stride, ctx->gpu_reduce);
    if (!runner->uses_external_threadpool())
      runner->resume_workers();
    if (ctx->gpu_reduce)
      ok = runner->run_strided_down(
          ctx->host_input, ctx->host_output, ctx->layer, ctx->gate_plane,
          ctx->up_plane, ctx->down_plane, ctx->host_ids, ctx->n_experts,
          ctx->d_model, ctx->d_ff, ctx->topk, ctx->expert_stride);
    else
      ok = runner->run_strided(
          ctx->host_input, ctx->host_output, ctx->layer, ctx->gate_plane,
          ctx->up_plane, ctx->down_plane, ctx->host_ids, ctx->n_experts,
          ctx->host_weights, ctx->d_model, ctx->d_ff, ctx->topk,
          ctx->expert_stride);
  }
  if (runner && !runner->uses_external_threadpool())
    runner->pause_workers();
  qwen_cpu_moe_diag_set_layer(-1);
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
void CUDART_CB qwen4exp_cuda_host_node_arm_direct_workers(
    void *user_data) {
  auto *ctx = static_cast<Qwen4ExpCudaHostNodeContext *>(user_data);
  if (!ctx || ctx->magic != Qwen4ExpCudaHostNodeContext::kMagic ||
      !ctx->cpu_runner)
    return;
  ctx->cpu_runner->arm_direct_workers();
}

} // namespace qwen4exp_graph

// Called by the opt-in CUDA backend bridge while a CUDA stream is being
// captured.  All CUDA work is enqueued here, outside the callback itself;
// the callback only runs the synchronous CPU GGML MoE and returns.
extern "C" int zeromoe_ggml_sched_early_cpu(
    ggml_backend_t backend, ggml_cgraph *graph,
    int64_t *wait_us, int64_t *compute_us) {
  EarlyReadyLane *context = nullptr;
  for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
    auto *node = ggml_graph_node(graph, i);
    if (node->op == GGML_OP_VIEW || node->op == GGML_OP_RESHAPE ||
        node->op == GGML_OP_PERMUTE || node->op == GGML_OP_TRANSPOSE)
      continue;
    if (node->op != GGML_OP_MAP_CUSTOM1 || !node->extra || context) return 0;
    auto *candidate = static_cast<EarlyReadyLane *>(node->extra);
    if (candidate->magic != EarlyReadyLane::kMagic) return 0;
    context = candidate;
  }
  if (!context || context->cpu != backend) return 0;
  const int64_t w = context->wait_us, c = context->compute_us;
  if (context->strategy) {
    // One complete invocation through the lifecycle (§3.8): begin() records the (layer,
    // generation) identity, run() consults state() before compute, finish() checks the
    // completed residency barrier and release() drops the identity. The lane strategy is
    // observation-only — run() owns residency — so the driving-strategy path is rejected.
    GGML_ASSERT(!context->strategy->drives_residency());
    // Captured before run(), because run() advances `calls`; Qwen lanes have no strategy yet
    // and take the plain path unchanged.
    zeromoe::graph::ColdMoEInvocation inv;
    inv.layer = context->trace_layer;
    inv.generation = context->calls + 1;
    GGML_ASSERT(context->strategy->begin(inv));
    context->run(); ++context->early_calls;
    GGML_ASSERT(context->strategy->finish(inv));
    context->strategy->release(inv);
  } else {
    context->run(); ++context->early_calls;
  }
  *wait_us = context->wait_us - w; *compute_us = context->compute_us - c;
  return 1;
}

extern "C" bool zeromoe_qwen4exp_cuda_host_node_enqueue(
    const ggml_tensor *node, cudaStream_t stream) {
  if (!node || node->op != GGML_OP_CPY || !node->extra || !stream)
    return false;
  auto *early = static_cast<EarlyReadyLane *>(node->extra);
  if (early->magic == EarlyReadyLane::kMagic) {
    GGML_ASSERT(early->enqueue(node, stream));
    return true;
  }
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
  const bool needs_host_weights = !ctx->gpu_reduce;
  ggml_tensor *activation = node->src[0];
  if (ctx->magic != Qwen4ExpCudaHostNodeContext::kMagic ||
      !node->data || !activation || !node->src[2] ||
      (needs_host_weights && !node->src[3]) ||
      !activation->data || !node->src[2]->data ||
      (needs_host_weights && !node->src[3]->data) ||
      !ctx->host_input || !ctx->host_output || !ctx->host_ids ||
      !ctx->host_weights || !ctx->d2h_start || !ctx->d2h_end ||
      !ctx->h2d_start || !ctx->h2d_end)
    return false;

  ctx->device_output = node->data;
  const char *direct_pool_env =
      zeromoe::graphcfg::env("ZEROMOE_QWEN_CPU_DIRECT_POOL");
  const char *direct_fused_env =
      zeromoe::graphcfg::env("ZEROMOE_QWEN_CPU_DIRECT_FUSED");
  const bool direct_pool_requested =
      direct_pool_env && direct_pool_env[0] == '1' &&
      !(direct_fused_env && direct_fused_env[0] == '1');
  if (direct_pool_requested &&
      cudaLaunchHostFunc(stream, qwen4exp_cuda_host_node_arm_direct_workers,
                         ctx) != cudaSuccess)
    return false;
  if (cudaEventRecord(ctx->d2h_start, stream) != cudaSuccess ||
      cudaMemcpyAsync(ctx->host_input, activation->data, ctx->input_bytes,
                      cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
      cudaMemcpyAsync(ctx->host_ids, node->src[2]->data, ctx->ids_bytes,
                      cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
      (needs_host_weights &&
       cudaMemcpyAsync(ctx->host_weights, node->src[3]->data,
                       ctx->weights_bytes, cudaMemcpyDeviceToHost, stream) !=
           cudaSuccess) ||
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

#endif // ZEROMOE_USE_GGML_CUDA
