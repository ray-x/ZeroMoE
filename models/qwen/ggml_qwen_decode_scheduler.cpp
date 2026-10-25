#include "ggml_qwen_decode_scheduler.hpp"
#include "ggml_moe_internal.hpp"

#include "logger.hpp"
#include "memory_guard.hpp"
#include "perf_timer.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#if defined(ZEROMOE_USE_GGML_CUDA)
#include <cuda_runtime.h>
#include "gpu_topology.hpp"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-cuda.h"
#include "ggml-cpu.h"
#if defined(_OPENMP)
#include <omp.h>
#endif
using namespace moe::detail;

// Qwen decode scheduler graph (full-graph CPU-affinity topology)
// ---------------------------------------------------------------------------

GgmlQwenDecodeGraphRunner &GgmlQwenDecodeGraphRunner::instance() {
  static GgmlQwenDecodeGraphRunner runner;
  return runner;
}

GgmlQwenDecodeGraphRunner &
GgmlQwenDecodeGraphRunner::instance_for_signature(
    uint32_t gate_type, uint32_t up_type, uint32_t down_type,
    size_t expert_stride) {
  // The current Qwen checkpoint uses two signatures. Keep a few spare slots
  // so a future mixed-quant export does not reintroduce the old unsafe
  // destroy/rebuild transition. These runners are used serially by the
  // decode loop, while each owns its own scheduler, graph, and host buffers.
  static GgmlQwenDecodeGraphRunner runners[8];
  for (auto &runner : runners) {
    if (runner.has_signature(gate_type, up_type, down_type, expert_stride))
      return runner;
  }
  for (auto &runner : runners) {
    if (!runner.ready_)
      return runner;
  }
  // More signatures than the bounded cache is unexpected. Reuse the first
  // slot; run() will rebuild it safely after its scheduler has synchronized.
  return runners[0];
}

bool GgmlQwenDecodeGraphRunner::has_signature(
    uint32_t gate_type, uint32_t up_type, uint32_t down_type,
    size_t expert_stride) const {
  return ready_ && gate_type_ == gate_type && up_type_ == up_type &&
         down_type_ == down_type && expert_stride_ == expert_stride;
}

bool GgmlQwenDecodeGraphRunner::compiled_in() { return true; }

GgmlQwenDecodeGraphRunner::~GgmlQwenDecodeGraphRunner() { shutdown(); }

void GgmlQwenDecodeGraphRunner::free_graph() {
  if (sched_) {
    ggml_backend_sched_synchronize((ggml_backend_sched_t)sched_);
    ggml_backend_sched_free((ggml_backend_sched_t)sched_);
    sched_ = nullptr;
  }
  graph_ = nullptr;
  if (graph_ctx_) {
    ggml_free((ggml_context *)graph_ctx_);
    graph_ctx_ = nullptr;
  }
  output_ = x_ = ids_ = weights_ = nullptr;
  shared_gate_ = shared_up_ = shared_down_ = shared_gate_inp_ = nullptr;
  shared_has_gate_ = false;
  shared_clamp_ = 0.0f;
  shared_gate_type_ = shared_up_type_ = shared_down_type_ = 0;
  shared_gate_inp_type_ = 0;
  shared_graph_ = false;
  router_external_ = false;

  for (void *buf : expert_bufs_) {
    if (buf)
      ggml_backend_buffer_free((ggml_backend_buffer_t)buf);
  }
  expert_bufs_.clear();
  gate_ = up_ = down_ = nullptr;
  if (expert_ctx_) {
    ggml_free((ggml_context *)expert_ctx_);
    expert_ctx_ = nullptr;
  }
  if (router_buf_) {
    ggml_backend_buffer_free((ggml_backend_buffer_t)router_buf_);
    router_buf_ = nullptr;
  }
  router_ = bias_ = nullptr;
  if (router_ctx_) {
    ggml_free((ggml_context *)router_ctx_);
    router_ctx_ = nullptr;
  }
  ready_ = false;
  gate_type_ = up_type_ = down_type_ = 0;
  expert_stride_ = 0;
}

void GgmlQwenDecodeGraphRunner::shutdown() {
  free_graph();
  // The external buffer is a small CUDA allocation used only as the GGML
  // ownership marker for dense-pool/shared-weight pointers.  It must outlive
  // graph rebuilds: free_graph() destroys the old graph, but a subsequent
  // rebuild still needs the marker to bind the new graph's external tensors.
  if (external_cuda_buf_) {
    ggml_backend_buffer_free((ggml_backend_buffer_t)external_cuda_buf_);
    external_cuda_buf_ = nullptr;
  }
  if (cpu_backend_ && cpu_threadpool_) {
    // The CPU backend retains the pool pointer; detach it before destroying
    // either object.  This mirrors GgmlQwenCpuMoeRunner and prevents a stale
    // pool handle during scheduler teardown.
    ggml_backend_cpu_set_threadpool(
        (ggml_backend_t)cpu_backend_, nullptr);
  }
  if (cpu_threadpool_) {
    ggml_threadpool_free((ggml_threadpool_t)cpu_threadpool_);
    cpu_threadpool_ = nullptr;
  }
  cpu_threads_ = 0;
  if (cpu_backend_) {
    ggml_backend_free((ggml_backend_t)cpu_backend_);
    cpu_backend_ = nullptr;
  }
  cuda_backend_ = nullptr;
  device_ = -1;
  d_model_ = n_experts_ = topk_ = d_ff_ = 0;
  normalize_ = false;
  weight_scale_ = 0.0f;
}

bool GgmlQwenDecodeGraphRunner::ensure(int device, int d_model,
                                       int n_experts, int topk, int d_ff,
                                       bool normalize, float weight_scale) {
  if (device < 0 || d_model <= 0 || n_experts <= 0 || topk <= 0 ||
      topk > n_experts || d_ff <= 0)
    return false;
  if (cuda_backend_ && device_ == device && d_model_ == d_model &&
      n_experts_ == n_experts && topk_ == topk && d_ff_ == d_ff &&
      normalize_ == normalize && weight_scale_ == weight_scale)
    return true;

  shutdown();
  cudaSetDevice(device);
  cuda_backend_ = ggml_shared_backend(device);
  if (!cuda_backend_)
    return false;
  cpu_backend_ = ggml_backend_cpu_init();
  if (!cpu_backend_) {
    shutdown();
    return false;
  }
  // llama.cpp's whole-model scheduler explicitly gives its CPU backend the same
  // decode threadpool as the rest of the inference graph.  Without this,
  // ZeroMoE's opt-in Qwen topology silently executes CPU MUL_MAT_ID with
  // the backend default (effectively one worker), which is why the first
  // scheduler A/B measured only 1.7 tok/s despite placing the nodes on CPU.
  // Keep this entirely inside the Qwen-only, config-gated runner; the normal
  // per-layer path and every other model family are unchanged.
#if defined(_OPENMP)
  cpu_threads_ = std::max(1, omp_get_max_threads());
#else
  cpu_threads_ = 1;
#endif
  ggml_backend_cpu_set_n_threads((ggml_backend_t)cpu_backend_,
                                 cpu_threads_);
  ggml_threadpool_params tp_params =
      ggml_threadpool_params_default(cpu_threads_);
  cpu_threadpool_ = ggml_threadpool_new(&tp_params);
  if (cpu_threadpool_) {
    ggml_backend_cpu_set_threadpool((ggml_backend_t)cpu_backend_,
                                     (ggml_threadpool_t)cpu_threadpool_);
  } else {
    LOG_WARN("[GGML-QWEN-DECODE-SCHED] CPU threadpool allocation failed; "
             "using backend worker configuration (%d threads)\n",
             cpu_threads_);
  }
  LOG_INFO("[GGML-QWEN-DECODE-SCHED] CPU backend configured with %d threads%s\n",
           cpu_threads_, cpu_threadpool_ ? " (persistent threadpool)" : "");
  external_cuda_buf_ = ggml_backend_alloc_buffer(
      (ggml_backend_t)cuda_backend_, 1);
  if (!external_cuda_buf_) {
    shutdown();
    return false;
  }
  ggml_backend_buffer_set_usage(
      (ggml_backend_buffer_t)external_cuda_buf_,
      GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
  device_ = device;
  d_model_ = d_model;
  n_experts_ = n_experts;
  topk_ = topk;
  d_ff_ = d_ff;
  normalize_ = normalize;
  weight_scale_ = weight_scale;
  return true;
}

bool GgmlQwenDecodeGraphRunner::rebuild(
    const GgmlQwenDecodeGraphRef &ref) {
  auto log_rebuild_failure = [&](const char *reason) {
    if (qwen_decode_sched_debug())
      LOG_WARN("[GGML-QWEN-DECODE-SCHED] rebuild failed before allocation: %s\n",
               reason);
  };
  const bool basic_ok =
      cuda_backend_ && cpu_backend_ &&
      (ref.router_device || ref.router_host) && ref.gate_plane &&
      ref.up_plane && ref.down_plane && ref.router_ne0 == d_model_ &&
      ref.router_ne1 == n_experts_ &&
      ref.router_type < (uint32_t)GGML_TYPE_COUNT &&
      ref.gate_type < (uint32_t)GGML_TYPE_COUNT &&
      ref.up_type < (uint32_t)GGML_TYPE_COUNT &&
      ref.down_type < (uint32_t)GGML_TYPE_COUNT && ref.expert_stride != 0;
  if (!basic_ok) {
    if (qwen_decode_sched_debug())
      LOG_WARN("[GGML-QWEN-DECODE-SCHED] rebuild rejected basic contract "
               "router=%p/%p shape=%lldx%lld expected=%dx%d planes=%p/%p/%p "
               "stride=%zu\n",
               ref.router_device, ref.router_host,
               (long long)ref.router_ne0, (long long)ref.router_ne1, d_model_,
               n_experts_, ref.gate_plane, ref.up_plane, ref.down_plane,
               ref.expert_stride);
    return false;
  }

  const ggml_type router_type = (ggml_type)ref.router_type;
  const ggml_type gate_type = (ggml_type)ref.gate_type;
  const ggml_type up_type = (ggml_type)ref.up_type;
  const ggml_type down_type = (ggml_type)ref.down_type;
  if ((router_type != GGML_TYPE_F32 && router_type != GGML_TYPE_F16 &&
       router_type != GGML_TYPE_BF16 && !ggml_is_quantized(router_type)) ||
      !ggml_is_quantized(gate_type) || !ggml_is_quantized(up_type) ||
      !ggml_is_quantized(down_type)) {
    if (qwen_decode_sched_debug())
      LOG_WARN("[GGML-QWEN-DECODE-SCHED] rebuild rejected unsupported "
               "types router=%u gate=%u up=%u down=%u\n",
               ref.router_type, ref.gate_type, ref.up_type, ref.down_type);
    return false;
  }

  const int64_t shared_inter =
      ref.shared_gate.ne0 > 0 ? ref.shared_gate.ne1 : 0;
  const bool shared_requested =
      ref.shared_gate.device_data && ref.shared_up.device_data &&
      ref.shared_down.device_data && ref.shared_gate.ne0 == d_model_ &&
      ref.shared_up.ne0 == d_model_ && ref.shared_down.ne0 == shared_inter &&
      ref.shared_gate.ne1 == ref.shared_up.ne1 &&
      ref.shared_down.ne1 == d_model_ && ref.shared_gate.ne1 > 0 &&
      ref.shared_gate.device_bytes > 0 && ref.shared_up.device_bytes > 0 &&
      ref.shared_down.device_bytes > 0 &&
      (!ref.shared_has_gate ||
       (ref.shared_gate_inp.device_data &&
        ref.shared_gate_inp.ne0 == d_model_ &&
        ref.shared_gate_inp.ne1 == 1 &&
        ref.shared_gate_inp.device_bytes > 0));
  const bool shared_types_supported =
      !shared_requested ||
      (ref.shared_gate.type < (uint32_t)GGML_TYPE_COUNT &&
       ref.shared_up.type < (uint32_t)GGML_TYPE_COUNT &&
       ref.shared_down.type < (uint32_t)GGML_TYPE_COUNT &&
       type_supported(ref.shared_gate.type) &&
       type_supported(ref.shared_up.type) &&
       type_supported(ref.shared_down.type) &&
       (!ref.shared_has_gate ||
        (ref.shared_gate_inp.type < (uint32_t)GGML_TYPE_COUNT &&
         type_supported(ref.shared_gate_inp.type))));
  const bool use_shared_graph = shared_requested && shared_types_supported;

  const bool same_signature =
      ready_ && gate_type_ == ref.gate_type && up_type_ == ref.up_type &&
      down_type_ == ref.down_type && expert_stride_ == ref.expert_stride &&
      ((bias_ != nullptr) == (ref.selection_bias != nullptr)) &&
      router_external_ == (ref.router_device != nullptr) &&
      shared_graph_ == use_shared_graph &&
      (!use_shared_graph ||
       (shared_gate_type_ == ref.shared_gate.type &&
        shared_up_type_ == ref.shared_up.type &&
        shared_down_type_ == ref.shared_down.type &&
        shared_gate_inp_type_ ==
            (ref.shared_has_gate ? ref.shared_gate_inp.type : 0) &&
        shared_has_gate_ == ref.shared_has_gate &&
        shared_clamp_ == ref.shared_clamp));
  if (same_signature)
    return true;

  free_graph();
  cudaSetDevice(device_);

  ggml_init_params rip = {};
  rip.mem_size = ggml_tensor_overhead() * 16;
  rip.no_alloc = true;
  ggml_context *rctx = ggml_init(rip);
  if (!rctx) {
    log_rebuild_failure("router context");
    return false;
  }
  ggml_tensor *router =
      ggml_new_tensor_2d(rctx, router_type, d_model_, n_experts_);
  ggml_tensor *bias = ref.selection_bias
                          ? ggml_new_tensor_1d(rctx, GGML_TYPE_F32,
                                               n_experts_)
                          : nullptr;
  if (!router || (ref.selection_bias && !bias)) {
    log_rebuild_failure("router or bias tensor");
    ggml_free(rctx);
    return false;
  }
  ggml_backend_buffer_t rbuf = ggml_backend_alloc_ctx_tensors(
      rctx, (ggml_backend_t)cuda_backend_);
  if (!rbuf) {
    log_rebuild_failure("router buffer");
    ggml_free(rctx);
    return false;
  }
  ggml_backend_buffer_set_usage(rbuf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
  if ((ref.router_device && ref.router_device_bytes < ggml_nbytes(router)) ||
      (!ref.router_device &&
       ref.router_host_bytes < ggml_nbytes(router))) {
    log_rebuild_failure("router bytes");
    ggml_backend_buffer_free(rbuf);
    ggml_free(rctx);
    return false;
  }
  // Prefer the dense-pool pointer when one exists.  Otherwise retain the
  // CUDA allocation from rbuf and upload the host-resident router weight
  // asynchronously at each layer submission, matching the existing Qwen
  // GPU-router ownership model without forcing router weights into the
  // dense pool.
  if (ref.router_device) {
    router->data = const_cast<void *>(ref.router_device);
    router->buffer = (ggml_backend_buffer_t)external_cuda_buf_;
    ggml_backend_buffer_init_tensor(router->buffer, router);
  }
  if (bias)
    ggml_backend_tensor_set((ggml_tensor *)bias, ref.selection_bias, 0,
                            (size_t)n_experts_ * sizeof(float));
  ggml_set_name(router, "qwen4exp_decode_router_weight");
  if (bias)
    ggml_set_name(bias, "qwen4exp_decode_router_bias");

  ggml_init_params eip = {};
  eip.mem_size = ggml_tensor_overhead() * 32;
  eip.no_alloc = true;
  ggml_context *ectx = ggml_init(eip);
  if (!ectx) {
    log_rebuild_failure("expert context");
    ggml_backend_buffer_free(rbuf);
    ggml_free(rctx);
    return false;
  }
  ggml_tensor *gate =
      ggml_new_tensor_3d(ectx, gate_type, d_model_, d_ff_, n_experts_);
  ggml_tensor *up =
      ggml_new_tensor_3d(ectx, up_type, d_model_, d_ff_, n_experts_);
  ggml_tensor *down =
      ggml_new_tensor_3d(ectx, down_type, d_ff_, d_model_, n_experts_);
  ggml_tensor *shared_gate = nullptr;
  ggml_tensor *shared_up = nullptr;
  ggml_tensor *shared_down = nullptr;
  ggml_tensor *shared_gate_inp = nullptr;
  if (use_shared_graph) {
    shared_gate = ggml_new_tensor_2d(
        ectx, (ggml_type)ref.shared_gate.type, ref.shared_gate.ne0,
        ref.shared_gate.ne1);
    shared_up = ggml_new_tensor_2d(
        ectx, (ggml_type)ref.shared_up.type, ref.shared_up.ne0,
        ref.shared_up.ne1);
    shared_down = ggml_new_tensor_2d(
        ectx, (ggml_type)ref.shared_down.type, ref.shared_down.ne0,
        ref.shared_down.ne1);
    if (ref.shared_has_gate)
      shared_gate_inp = ggml_new_tensor_2d(
          ectx, (ggml_type)ref.shared_gate_inp.type,
          ref.shared_gate_inp.ne0, ref.shared_gate_inp.ne1);
  }
  if (!gate || !up || !down ||
      (use_shared_graph &&
       (!shared_gate || !shared_up || !shared_down ||
        (ref.shared_has_gate && !shared_gate_inp)))) {
    log_rebuild_failure("expert or shared tensor");
    ggml_free(ectx);
    ggml_backend_buffer_free(rbuf);
    ggml_free(rctx);
    return false;
  }
  const size_t gate_bytes = ggml_nbytes(gate) / (size_t)n_experts_;
  const size_t up_bytes = ggml_nbytes(up) / (size_t)n_experts_;
  const size_t down_bytes = ggml_nbytes(down) / (size_t)n_experts_;
  if (ref.gate_bytes < gate_bytes || ref.up_bytes < up_bytes ||
      ref.down_bytes < down_bytes || ref.expert_stride < gate_bytes ||
      ref.expert_stride < up_bytes || ref.expert_stride < down_bytes) {
    log_rebuild_failure("expert plane bytes or stride");
    ggml_free(ectx);
    ggml_backend_buffer_free(rbuf);
    ggml_free(rctx);
    return false;
  }
  gate->nb[2] = ref.expert_stride;
  up->nb[2] = ref.expert_stride;
  down->nb[2] = ref.expert_stride;
  gate->nb[3] = ref.expert_stride * (size_t)n_experts_;
  up->nb[3] = ref.expert_stride * (size_t)n_experts_;
  down->nb[3] = ref.expert_stride * (size_t)n_experts_;
  const void *planes[] = {ref.gate_plane, ref.up_plane, ref.down_plane};
  const size_t plane_bytes[] = {gate_bytes, up_bytes, down_bytes};
  ggml_tensor *tensors[] = {gate, up, down};
  for (int i = 0; i < 3; ++i) {
    const size_t view_bytes = ref.expert_stride * (size_t)(n_experts_ - 1) +
                              plane_bytes[i];
    ggml_backend_buffer_t b = ggml_backend_cpu_buffer_from_ptr(
        const_cast<void *>(planes[i]), view_bytes);
    if (!b || ggml_backend_tensor_alloc(b, tensors[i],
                                        const_cast<void *>(planes[i])) !=
                 GGML_STATUS_SUCCESS) {
      log_rebuild_failure(i == 0 ? "gate plane buffer" :
                          (i == 1 ? "up plane buffer" : "down plane buffer"));
      if (b)
        ggml_backend_buffer_free(b);
      ggml_free(ectx);
      ggml_backend_buffer_free(rbuf);
      ggml_free(rctx);
      for (void *old : expert_bufs_)
        if (old)
          ggml_backend_buffer_free((ggml_backend_buffer_t)old);
      expert_bufs_.clear();
      return false;
    }
    ggml_backend_buffer_set_usage(b, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    expert_bufs_.push_back(b);
  }
  ggml_set_name(gate, "qwen4exp_decode_gate_experts");
  ggml_set_name(up, "qwen4exp_decode_up_experts");
  ggml_set_name(down, "qwen4exp_decode_down_experts");
  if (use_shared_graph) {
    auto bind_shared = [&](ggml_tensor *tensor,
                           const GgmlSharedWeightRef &weight) {
      const size_t required_bytes = tensor ? ggml_nbytes(tensor) : 0;
      if (!tensor || !external_cuda_buf_ || !weight.device_data ||
          weight.device_bytes < required_bytes) {
        if (qwen_decode_sched_debug())
          LOG_WARN("[GGML-QWEN-DECODE-SCHED] shared bind rejected name=%s "
                   "ptr=%p external=%p bytes=%zu required=%zu type=%u "
                   "shape=%lldx%lld meta_bytes=%zu\n",
                   tensor ? tensor->name : "<null>", weight.device_data,
                   (void *)external_cuda_buf_, weight.device_bytes,
                   required_bytes, weight.type,
                   tensor ? (long long)tensor->ne[0] : 0,
                   tensor ? (long long)tensor->ne[1] : 0, weight.nbytes);
        return false;
      }
      tensor->data = const_cast<void *>(weight.device_data);
      tensor->buffer = (ggml_backend_buffer_t)external_cuda_buf_;
      ggml_backend_buffer_init_tensor(tensor->buffer, tensor);
      return true;
    };
    if (!bind_shared(shared_gate, ref.shared_gate) ||
        !bind_shared(shared_up, ref.shared_up) ||
        !bind_shared(shared_down, ref.shared_down) ||
        (ref.shared_has_gate &&
         !bind_shared(shared_gate_inp, ref.shared_gate_inp))) {
      log_rebuild_failure("shared weight binding");
      ggml_free(ectx);
      ggml_backend_buffer_free(rbuf);
      ggml_free(rctx);
      for (void *old : expert_bufs_)
        if (old)
          ggml_backend_buffer_free((ggml_backend_buffer_t)old);
      expert_bufs_.clear();
      return false;
    }
    ggml_set_name(shared_gate, "qwen4exp_decode_shared_gate");
    ggml_set_name(shared_up, "qwen4exp_decode_shared_up");
    ggml_set_name(shared_down, "qwen4exp_decode_shared_down");
    if (shared_gate_inp)
      ggml_set_name(shared_gate_inp, "qwen4exp_decode_shared_gate_inp");
  }

  ggml_init_params gip = {};
  gip.mem_size = ggml_tensor_overhead() * 384 + ggml_graph_overhead();
  gip.no_alloc = true;
  ggml_context *gctx = ggml_init(gip);
  if (!gctx) {
    log_rebuild_failure("graph context");
    ggml_free(ectx);
    ggml_backend_buffer_free(rbuf);
    ggml_free(rctx);
    return false;
  }
  ggml_cgraph *graph = ggml_new_graph(gctx);
  ggml_tensor *x = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, d_model_, 1);
  if (!graph || !x) {
    log_rebuild_failure("graph or input tensor");
    ggml_free(gctx);
    ggml_free(ectx);
    ggml_backend_buffer_free(rbuf);
    ggml_free(rctx);
    return false;
  }
  ggml_set_input(x);
  ggml_set_name(x, "qwen4exp_decode_router_input");
  ggml_tensor *logits = ggml_mul_mat(gctx, router, x);
  ggml_tensor *probs = logits ? ggml_soft_max(gctx, logits) : nullptr;
  ggml_tensor *selection =
      probs && bias ? ggml_add(gctx, probs, bias) : probs;
  ggml_tensor *ids = selection
                        ? ggml_cont(gctx,
                                    ggml_argsort_top_k(gctx, selection, topk_))
                        : nullptr;
  ggml_tensor *probs3 = probs
                            ? ggml_reshape_3d(gctx, probs, 1, n_experts_, 1)
                            : nullptr;
  ggml_tensor *weights = probs3 && ids
                             ? ggml_get_rows(gctx, probs3, ids)
                             : nullptr;
  if (normalize_ && weights) {
    ggml_tensor *w2 = ggml_reshape_2d(gctx, weights, topk_, 1);
    ggml_tensor *sum = ggml_sum_rows(gctx, w2);
    sum = ggml_clamp(gctx, sum, 6.103515625e-5f, INFINITY);
    w2 = ggml_div(gctx, w2, sum);
    weights = ggml_reshape_3d(gctx, w2, 1, topk_, 1);
  }
  if (weight_scale_ != 0.0f && weight_scale_ != 1.0f && weights)
    weights = ggml_scale(gctx, weights, weight_scale_);
  ggml_tensor *input3 = ggml_reshape_3d(gctx, x, d_model_, 1, 1);
  ggml_tensor *gate_out = input3 && ids
                              ? ggml_mul_mat_id(gctx, gate, input3, ids)
                              : nullptr;
  ggml_tensor *up_out = input3 && ids
                            ? ggml_mul_mat_id(gctx, up, input3, ids)
                            : nullptr;
  ggml_tensor *act = gate_out && up_out
                         ? ggml_swiglu_split(gctx, gate_out, up_out)
                         : nullptr;
  ggml_tensor *down_out = act && ids
                              ? ggml_mul_mat_id(gctx, down, act, ids)
                              : nullptr;
  ggml_tensor *weighted = down_out && weights
                              ? ggml_mul(gctx, down_out, weights)
                              : nullptr;
  ggml_tensor *result = nullptr;
  if (weighted) {
    for (int k = 0; k < topk_; ++k) {
      ggml_tensor *e = ggml_view_2d(
          gctx, weighted, d_model_, 1, weighted->nb[2],
          (size_t)k * weighted->nb[1]);
      result = k == 0 ? e : ggml_add(gctx, result, e);
    }
    if (topk_ > 1 && result)
      result = ggml_cont(gctx, result);
  }
  ggml_tensor *graph_output = result;
  if (use_shared_graph && result) {
    ggml_tensor *shared_gate_out = ggml_mul_mat(gctx, shared_gate, x);
    ggml_tensor *shared_up_out = ggml_mul_mat(gctx, shared_up, x);
    ggml_tensor *shared_act =
        (shared_gate_out && shared_up_out)
            ? ggml_mul(gctx, ggml_silu(gctx, shared_gate_out), shared_up_out)
            : nullptr;
    if (ref.shared_clamp > 0.0f && shared_act)
      shared_act = ggml_clamp(gctx, shared_act, -ref.shared_clamp,
                              ref.shared_clamp);
    ggml_tensor *shared =
        shared_act ? ggml_mul_mat(gctx, shared_down, shared_act) : nullptr;
    if (ref.shared_has_gate && shared) {
      ggml_tensor *scalar = ggml_sigmoid(
          gctx, ggml_mul_mat(gctx, shared_gate_inp, x));
      shared = scalar ? ggml_mul(gctx, shared, scalar) : nullptr;
    }
    graph_output = shared ? ggml_add(gctx, result, shared) : nullptr;
    if (!shared_gate_out || !shared_up_out || !shared_act || !shared ||
        !graph_output) {
      log_rebuild_failure("shared graph operation");
      ggml_free(gctx);
      ggml_free(ectx);
      ggml_backend_buffer_free(rbuf);
      ggml_free(rctx);
      return false;
    }
    ggml_set_name(graph_output, "qwen4exp_decode_routed_plus_shared");
  }
  if (!logits || !probs || !selection || !ids || !probs3 || !weights ||
      !input3 || !gate_out || !up_out || !act || !down_out || !weighted ||
      !result || !graph_output) {
    log_rebuild_failure("router or expert graph operation");
    ggml_free(gctx);
    ggml_free(ectx);
    ggml_backend_buffer_free(rbuf);
    ggml_free(rctx);
    return false;
  }
  ggml_set_output(graph_output);
  if (!use_shared_graph)
    ggml_set_name(graph_output, "qwen4exp_decode_cpu_moe_output");
  ggml_build_forward_expand(graph, graph_output);

  ggml_backend_t backends[] = {(ggml_backend_t)cuda_backend_,
                               (ggml_backend_t)cpu_backend_};
  ggml_backend_sched_t sched =
      ggml_backend_sched_new(backends, nullptr, 2, 4096, false, true);
  // Force only the final routed+shared add to CUDA before allocation.  Its
  // routed input is the CPU graph result, so the scheduler inserts one
  // CPU->CUDA handoff there; the shared FFN stays on CUDA and no D2H routed
  // result copy is needed.  The ordinary CPU-only graph keeps normal
  // placement.
  if (sched) {
    // x is supplied by the device-resident HC path on every decode call.
    // Explicitly classify the input as CUDA before allocation; otherwise the
    // scheduler may place a harmless-looking reshape on CPU and the CPU
    // executor will dereference the CUDA address as a host pointer.
    ggml_backend_sched_set_tensor_backend(
        sched, x, (ggml_backend_t)cuda_backend_);
  }
  if (sched && use_shared_graph)
    ggml_backend_sched_set_tensor_backend(
        sched, graph_output, (ggml_backend_t)cuda_backend_);
  if (sched && qwen_decode_sched_debug()) {
    // The scheduler's allocator can reject a mixed CUDA/CPU graph without
    // exposing the assignment that caused the rejection.  Dump the resolved
    // placement once, before alloc_graph(), so a Qwen-only experimental path
    // can be diagnosed without changing the allocator or other model paths.
    ggml_backend_sched_split_graph(sched, graph);
    LOG_WARN("[GGML-QWEN-DECODE-SCHED] placement splits=%d nodes=%d output=%s\n",
             ggml_backend_sched_get_n_splits(sched), ggml_graph_n_nodes(graph),
             ggml_backend_name(ggml_backend_sched_get_tensor_backend(
                 sched, graph_output)));
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
      ggml_tensor *node = ggml_graph_node(graph, i);
      ggml_backend_t b = ggml_backend_sched_get_tensor_backend(sched, node);
      LOG_WARN("[GGML-QWEN-DECODE-SCHED] node=%d op=%s name=%s backend=%s "
               "bytes=%zu support_cuda=%d support_cpu=%d buffer=%s\n",
               i, ggml_op_name(node->op), node->name,
               b ? ggml_backend_name(b) : "NULL", ggml_nbytes(node),
               ggml_backend_supports_op((ggml_backend_t)cuda_backend_, node)
                   ? 1
                   : 0,
               ggml_backend_supports_op((ggml_backend_t)cpu_backend_, node)
                   ? 1
                   : 0,
               node->buffer ? ggml_backend_buffer_name(node->buffer) : "none");
    }
  }
  if (!sched || !ggml_backend_sched_alloc_graph(sched, graph)) {
    if (qwen_decode_sched_debug())
      LOG_WARN("[GGML-QWEN-DECODE-SCHED] rebuild scheduler allocation failed "
               "(sched=%p graph_nodes=%d shared=%d)\n",
               (void *)sched, graph ? ggml_graph_n_nodes(graph) : -1,
               use_shared_graph ? 1 : 0);
    if (sched)
      ggml_backend_sched_free(sched);
    ggml_free(gctx);
    ggml_free(ectx);
    ggml_backend_buffer_free(rbuf);
    ggml_free(rctx);
    return false;
  }
  router_ctx_ = rctx;
  router_buf_ = rbuf;
  router_ = router;
  bias_ = bias;
  expert_ctx_ = ectx;
  gate_ = gate;
  up_ = up;
  down_ = down;
  shared_gate_ = shared_gate;
  shared_up_ = shared_up;
  shared_down_ = shared_down;
  shared_gate_inp_ = shared_gate_inp;
  shared_has_gate_ = use_shared_graph && ref.shared_has_gate;
  shared_clamp_ = use_shared_graph ? ref.shared_clamp : 0.0f;
  shared_gate_type_ = use_shared_graph ? ref.shared_gate.type : 0;
  shared_up_type_ = use_shared_graph ? ref.shared_up.type : 0;
  shared_down_type_ = use_shared_graph ? ref.shared_down.type : 0;
  shared_gate_inp_type_ =
      use_shared_graph && ref.shared_has_gate ? ref.shared_gate_inp.type : 0;
  shared_graph_ = use_shared_graph;
  router_external_ = ref.router_device != nullptr;
  graph_ctx_ = gctx;
  graph_ = graph;
  sched_ = sched;
  x_ = x;
  output_ = graph_output;
  ids_ = ids;
  weights_ = weights;
  gate_type_ = ref.gate_type;
  up_type_ = ref.up_type;
  down_type_ = ref.down_type;
  expert_stride_ = ref.expert_stride;
  ready_ = true;
  LOG_INFO("[GGML-QWEN-DECODE-SCHED] persistent CUDA+CPU router/MUL_MAT_ID "
           "graph active (D=%d experts=%d topk=%d router=%s shared=%s)\n",
           d_model_, n_experts_, topk_,
           ref.router_device ? "device" : "host-upload",
           use_shared_graph ? "fused" : "fallback");
  return true;
}

bool GgmlQwenDecodeGraphRunner::run(
    const void *device_input, const GgmlQwenDecodeGraphRef &ref,
    float *host_output, const void **device_output, int32_t *host_ids,
    float *host_weights) {
  // The caller owns the shape/device contract and calls ensure() before the
  // first run.  Do not try to infer it here: on the first call all shape
  // fields are zero, and doing so used to make an otherwise valid runner
  // reject its first graph submission.  Keeping ensure() separate also lets
  // the caller decline this path cleanly for lazy/non-arena layers.
  if (device_output)
    *device_output = nullptr;
  if (!device_input || !cuda_backend_)
    return false;
  const bool wants_shared =
      ref.shared_gate.device_data && ref.shared_up.device_data &&
      ref.shared_down.device_data && ref.shared_gate.ne0 == d_model_ &&
      ref.shared_up.ne0 == d_model_ &&
      ref.shared_down.ne0 == ref.shared_gate.ne1 &&
      ref.shared_gate.ne1 == ref.shared_up.ne1 &&
      ref.shared_down.ne1 == d_model_ &&
      ref.shared_gate.ne1 > 0 &&
      (!ref.shared_has_gate ||
       (ref.shared_gate_inp.device_data &&
        ref.shared_gate_inp.ne0 == d_model_ &&
        ref.shared_gate_inp.ne1 == 1)) &&
      ref.shared_gate.type < (uint32_t)GGML_TYPE_COUNT &&
      ref.shared_up.type < (uint32_t)GGML_TYPE_COUNT &&
      ref.shared_down.type < (uint32_t)GGML_TYPE_COUNT &&
      type_supported(ref.shared_gate.type) &&
      type_supported(ref.shared_up.type) &&
      type_supported(ref.shared_down.type) &&
      (!ref.shared_has_gate ||
       (ref.shared_gate_inp.type < (uint32_t)GGML_TYPE_COUNT &&
        type_supported(ref.shared_gate_inp.type)));
  if (!ready_ || gate_type_ != ref.gate_type || up_type_ != ref.up_type ||
      down_type_ != ref.down_type || expert_stride_ != ref.expert_stride ||
      ((bias_ != nullptr) != (ref.selection_bias != nullptr)) ||
      shared_graph_ != wants_shared ||
      router_external_ != (ref.router_device != nullptr)) {
    if (!rebuild(ref)) {
      if (qwen_decode_sched_debug())
        LOG_WARN("[GGML-QWEN-DECODE-SCHED] run declined: graph rebuild "
                 "failed (router=%s shared_request=%d)\n",
                 ref.router_device ? "device" : "host", wants_shared ? 1 : 0);
      return false;
    }
  }
  if (!sched_)
    return false;
  const ggml_tensor *src = (const ggml_tensor *)device_input;
  if (!src || src->type != GGML_TYPE_F32 || src->ne[0] != d_model_ ||
      src->ne[1] < 1 || src->ne[2] != 1 || src->ne[3] != 1 ||
      src->nb[0] != sizeof(float) ||
      src->nb[1] != (size_t)d_model_ * sizeof(float) || !src->data ||
      !src->buffer || (!ref.router_device && !ref.router_host) ||
      !ref.gate_plane ||
      !ref.up_plane || !ref.down_plane)
    return false;

  auto *router = (ggml_tensor *)router_;
  if (ref.router_device) {
    router->data = const_cast<void *>(ref.router_device);
    router->buffer = (ggml_backend_buffer_t)external_cuda_buf_;
    ggml_backend_buffer_init_tensor(router->buffer, router);
  } else {
    if (!router->buffer || ref.router_host_bytes < ggml_nbytes(router))
      return false;
    ggml_backend_tensor_set_async(
        (ggml_backend_t)cuda_backend_, router, ref.router_host, 0,
        ggml_nbytes(router));
  }
  if (bias_) {
    if (!ref.selection_bias)
      return false;
    ggml_backend_tensor_set_async(
        (ggml_backend_t)cuda_backend_, (ggml_tensor *)bias_,
        ref.selection_bias, 0, (size_t)n_experts_ * sizeof(float));
  }
  ((ggml_tensor *)gate_)->data = const_cast<void *>(ref.gate_plane);
  ((ggml_tensor *)up_)->data = const_cast<void *>(ref.up_plane);
  ((ggml_tensor *)down_)->data = const_cast<void *>(ref.down_plane);
  if (shared_graph_) {
    auto rebind_shared = [&](void *opaque,
                             const GgmlSharedWeightRef &weight) {
      auto *tensor = (ggml_tensor *)opaque;
      if (!tensor || !weight.device_data ||
          weight.device_bytes < ggml_nbytes(tensor))
        return false;
      tensor->data = const_cast<void *>(weight.device_data);
      tensor->buffer = (ggml_backend_buffer_t)external_cuda_buf_;
      ggml_backend_buffer_init_tensor(tensor->buffer, tensor);
      return true;
    };
    if (!rebind_shared(shared_gate_, ref.shared_gate) ||
        !rebind_shared(shared_up_, ref.shared_up) ||
        !rebind_shared(shared_down_, ref.shared_down) ||
        (shared_has_gate_ &&
         !rebind_shared(shared_gate_inp_, ref.shared_gate_inp)))
      return false;
  }
  // x_ was allocated and classified as a CUDA scheduler input.  Retargeting
  // only its data pointer is safe because the HC producer and this scheduler
  // share the same CUDA backend; the scheduler still owns the tensor's
  // backend/buffer metadata and will create the CPU copy at the split.
  ((ggml_tensor *)x_)->data = src->data;
  if (ggml_backend_sched_graph_compute_async(
          (ggml_backend_sched_t)sched_, (ggml_cgraph *)graph_) !=
      GGML_STATUS_SUCCESS) {
    if (qwen_decode_sched_debug())
      LOG_WARN("[GGML-QWEN-DECODE-SCHED] graph compute submission failed "
               "(shared=%d)\n",
               shared_graph_ ? 1 : 0);
    return false;
  }
  ggml_backend_sched_synchronize((ggml_backend_sched_t)sched_);
  if (host_output)
    ggml_backend_tensor_get((ggml_tensor *)output_, host_output, 0,
                            (size_t)d_model_ * sizeof(float));
  if (host_ids)
    ggml_backend_tensor_get((ggml_tensor *)ids_, host_ids, 0,
                            (size_t)topk_ * sizeof(int32_t));
  if (host_weights)
    ggml_backend_tensor_get((ggml_tensor *)weights_, host_weights, 0,
                            (size_t)topk_ * sizeof(float));
  if (device_output && shared_graph_)
    *device_output = output_;
  return true;
}

// ---------------------------------------------------------------------------
#else  // !ZEROMOE_USE_GGML_CUDA
bool GgmlQwenDecodeGraphRunner::compiled_in() { return false; }
GgmlQwenDecodeGraphRunner &GgmlQwenDecodeGraphRunner::instance() {
  static GgmlQwenDecodeGraphRunner r;
  return r;
}
GgmlQwenDecodeGraphRunner &
GgmlQwenDecodeGraphRunner::instance_for_signature(uint32_t, uint32_t,
                                                  uint32_t, size_t) {
  return instance();
}
GgmlQwenDecodeGraphRunner::~GgmlQwenDecodeGraphRunner() {}
bool GgmlQwenDecodeGraphRunner::ensure(int, int, int, int, int, bool,
                                       float) {
  return false;
}
bool GgmlQwenDecodeGraphRunner::run(const void *, const GgmlQwenDecodeGraphRef &,
                                    float *, const void **, int32_t *, float *) {
  return false;
}
void GgmlQwenDecodeGraphRunner::shutdown() {}
#endif  // ZEROMOE_USE_GGML_CUDA
