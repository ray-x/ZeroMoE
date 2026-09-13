// Persistent CPU GGML routed-expert graph for Qwen4Exp.
//
// This file is intentionally independent of the CUDA MoE runner. It is used
// only by execution_layer.cpp's Qwen CPU-affinity path. It supports both the
// original compact fallback (selected experts copied into a temporary expert
// axis) and a Llama.cpp strided mode that points a persistent
// MUL_MAT_ID graph at complete host-RAM expert arenas.

#include "ggml_moe_backend.hpp"

#include "logger.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <new>
#include <vector>

#if defined(MOESTREAM_USE_GGML)

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace {

constexpr size_t kWeightContextBytes = 64ull * 1024ull;
constexpr size_t kGraphContextBytes = 2ull * 1024ull * 1024ull;

int default_cpu_threads() {
#if defined(_OPENMP)
  const int available = std::max(1, omp_get_max_threads());
#else
  const int available = 1;
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
  const double compute = state.compute_s.load(std::memory_order_relaxed);
  const double output = state.output_s.load(std::memory_order_relaxed);
  const double total = state.total_s.load(std::memory_order_relaxed);
  LOG_STREAM_INFO()
      << "[CPU-MOE-GRAPH-TIMING] calls=" << calls
      << " input_copy=" << input << "s (" << input * 1000.0 / calls
      << "ms/call) compute=" << compute << "s ("
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

} // namespace

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
    if (!runner.graph_)
      return runner;
  }
  // More signatures than the reserved cache is unexpected.  Reusing a slot
  // retains the old safe behavior (ensure_graph() will rebuild it), while all
  // normal Qwen signatures take one of the persistent slots above.
  return runners[0];
}

bool GgmlQwenCpuMoeRunner::compiled_in() { return true; }

bool GgmlQwenCpuMoeRunner::has_strided_signature(
    const GgmlQwenCpuExpertRef &layer, int n_experts, int d_model, int d_ff,
    int topk, size_t expert_stride, bool output_per_expert) const {
  return graph_ && strided_mode_ && n_experts_ == n_experts &&
         d_model_ == d_model && d_ff_ == d_ff && topk_ == topk &&
         gate_type_ == layer.gate_type && up_type_ == layer.up_type &&
         down_type_ == layer.down_type &&
         output_per_expert_ == output_per_expert &&
         expert_stride_ == expert_stride;
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
  cached_sources_.clear();
}

void GgmlQwenCpuMoeRunner::shutdown() {
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
  if (cpu_backend_ && cpu_threadpool_) {
    // Detach before freeing the pool so the backend never retains a dangling
    // threadpool pointer during teardown.
    ggml_backend_cpu_set_threadpool(
        static_cast<ggml_backend_t>(cpu_backend_), nullptr);
  }
  if (cpu_threadpool_) {
    ggml_threadpool_free(static_cast<ggml_threadpool_t>(cpu_threadpool_));
    cpu_threadpool_ = nullptr;
  }
  if (cpu_backend_) {
    ggml_backend_free(static_cast<ggml_backend_t>(cpu_backend_));
    cpu_backend_ = nullptr;
  }
  n_threads_ = 0;
}

bool GgmlQwenCpuMoeRunner::ensure_graph(
    const GgmlQwenCpuExpertRef &ref, int n_experts, int d_model, int d_ff,
    int topk, bool strided, const void *gate_plane, const void *up_plane,
    const void *down_plane, size_t expert_stride, bool output_per_expert,
    bool merged_gate_up) {
  if (!type_is_valid(ref.gate_type) || !type_is_valid(ref.up_type) ||
      !type_is_valid(ref.down_type) || n_experts <= 0 || d_model <= 0 ||
      d_ff <= 0 || topk <= 0 || topk > n_experts ||
      (strided &&
       (!gate_plane || !up_plane || !down_plane || expert_stride == 0)) ||
      (!strided && topk != n_experts) ||
      (merged_gate_up &&
       (!strided || ref.gate_type != ref.up_type ||
        ref.gate_bytes != ref.up_bytes)))
    return false;

  const bool shape_matches =
      graph_ && n_experts_ == n_experts && d_model_ == d_model &&
      d_ff_ == d_ff && topk_ == topk && gate_type_ == ref.gate_type &&
      up_type_ == ref.up_type && down_type_ == ref.down_type &&
      strided_mode_ == strided && output_per_expert_ == output_per_expert &&
      merged_gate_up_ == merged_gate_up &&
      (!strided || expert_stride_ == expert_stride);
  if (shape_matches)
    return true;

  free_graph();
  if (!cpu_backend_) {
    cpu_backend_ = ggml_backend_cpu_init();
    if (!cpu_backend_)
      return false;
    n_threads_ = default_cpu_threads();
    ggml_backend_cpu_set_n_threads(static_cast<ggml_backend_t>(cpu_backend_),
                                   n_threads_);
  }
  if (!cpu_threadpool_) {
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
    // BeeLlama's merged gate_up MUL_MAT_ID without copying or changing the
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
       (expert_stride < gate_bytes_per_expert + up_bytes_per_expert ||
        expert_stride < up_bytes_per_expert ||
        expert_stride < down_bytes_per_expert ||
        (n_experts > 1 &&
         expert_stride > SIZE_MAX / static_cast<size_t>(n_experts - 1))))) {
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
    const size_t storage_bytes = gate_bytes + up_bytes + down_bytes;
    if (posix_memalign(&storage, 64, storage_bytes) != 0 || !storage) {
      ggml_free(wctx);
      return false;
    }
    weight_storage_ = storage;
    weight_storage_bytes_ = storage_bytes;
    wbuf = ggml_backend_cpu_buffer_from_ptr(weight_storage_,
                                            weight_storage_bytes_);
    if (!wbuf ||
        ggml_backend_tensor_alloc(wbuf, gate, weight_storage_) !=
            GGML_STATUS_SUCCESS ||
        ggml_backend_tensor_alloc(
            wbuf, up, static_cast<uint8_t *>(weight_storage_) + gate_bytes) !=
            GGML_STATUS_SUCCESS ||
        ggml_backend_tensor_alloc(wbuf, down,
                                  static_cast<uint8_t *>(weight_storage_) +
                                      gate_bytes + up_bytes) !=
            GGML_STATUS_SUCCESS) {
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
      up->nb[2] = expert_stride;
      gate->nb[3] = expert_stride * static_cast<size_t>(n_experts);
      up->nb[3] = expert_stride * static_cast<size_t>(n_experts);
    }
    down->nb[2] = expert_stride;
    down->nb[3] = expert_stride * static_cast<size_t>(n_experts);

    const void *planes[3] = {gate_plane, up_plane, down_plane};
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
          expert_stride * static_cast<size_t>(n_experts - 1) + plane_bytes[i];
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
  }
  ggml_tensor *act = ggml_swiglu_split(gctx, gate_out, up_out);
  ggml_tensor *down_out = ggml_mul_mat_id(gctx, down, act, ids);
  ggml_tensor *weighted = output_per_expert
                              ? nullptr
                              : ggml_mul(gctx, down_out, weights);
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
    ggml_set_name(result, "qwen4exp_cpu_moe_out");
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
  if (!ensure_graph(first, n_experts, d_model, d_ff, topk, false, nullptr,
                    nullptr, nullptr, 0, false, false))
    return false;

  const size_t gate_plane =
      weight_storage_ == nullptr
          ? 0
          : ggml_nbytes(static_cast<ggml_tensor *>(gate_)) /
                static_cast<size_t>(n_experts_);
  const size_t up_plane = ggml_nbytes(static_cast<ggml_tensor *>(up_)) /
                          static_cast<size_t>(n_experts_);
  const size_t down_plane = ggml_nbytes(static_cast<ggml_tensor *>(down_)) /
                            static_cast<size_t>(n_experts_);
  if (gate_plane != first.gate_bytes || up_plane != first.up_bytes ||
      down_plane != first.down_bytes)
    return false;

  uint8_t *storage = static_cast<uint8_t *>(weight_storage_);
  const size_t gate_total = gate_plane * static_cast<size_t>(n_experts_);
  const size_t up_total = up_plane * static_cast<size_t>(n_experts_);
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
      std::memcpy(storage + static_cast<size_t>(i) * gate_plane, expert.gate,
                  gate_plane);
      std::memcpy(storage + gate_total + static_cast<size_t>(i) * up_plane,
                  expert.up, up_plane);
      std::memcpy(storage + gate_total + up_total +
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
  auto input_start = total_start;
  std::memcpy(static_cast<ggml_tensor *>(x_)->data, input,
              static_cast<size_t>(d_model_) * sizeof(float));
  auto *ids = static_cast<int32_t *>(static_cast<ggml_tensor *>(ids_)->data);
  for (int k = 0; k < topk_; ++k)
    ids[k] = static_cast<int32_t>(expert_ids[k]);
  std::memcpy(static_cast<ggml_tensor *>(weights_)->data, router_weights,
              static_cast<size_t>(topk_) * sizeof(float));

  if (timing.enabled)
    qwen_cpu_graph_timing_add(
        timing.input_s,
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      input_start)
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
  }
}

bool GgmlQwenCpuMoeRunner::start_strided_async(
    const float *input, float *output, const GgmlQwenCpuExpertRef &layer,
    const void *gate_plane, const void *up_plane, const void *down_plane,
    const int *expert_ids, int n_experts, const float *router_weights,
    int d_model, int d_ff, int topk, size_t expert_stride) {
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
                    expert_stride};
  async_result_ = false;
  async_request_done_ = false;
  async_request_pending_ = true;
  async_cv_.notify_one();
  return true;
}

bool GgmlQwenCpuMoeRunner::wait_strided_async() {
  std::unique_lock<std::mutex> lock(async_mutex_);
  if (!async_worker_.joinable() || async_request_done_)
    return async_worker_.joinable() ? async_result_ : false;
  async_cv_.wait(lock, [this] { return async_request_done_; });
  return async_result_;
}

#else

GgmlQwenCpuMoeRunner &GgmlQwenCpuMoeRunner::instance() {
  static GgmlQwenCpuMoeRunner runner;
  return runner;
}

GgmlQwenCpuMoeRunner &GgmlQwenCpuMoeRunner::instance_for_strided(
    const GgmlQwenCpuExpertRef &, int, int, int, int, size_t, bool) {
  return instance();
}

bool GgmlQwenCpuMoeRunner::compiled_in() { return false; }

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
bool GgmlQwenCpuMoeRunner::wait_strided_async() { return false; }

#endif
