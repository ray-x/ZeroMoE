#include "ggml_router_runner.hpp"
#include "ggml_moe_graph_builder.hpp"
#include "ggml_moe_internal.hpp"
#include "ggml_resource_guards.hpp"

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

// Qwen4Exp device router
// ---------------------------------------------------------------------------

GgmlRouterRunner &GgmlRouterRunner::instance() {
  static GgmlRouterRunner r;
  return r;
}

bool GgmlRouterRunner::compiled_in() { return true; }

GgmlRouterRunner::~GgmlRouterRunner() { shutdown(); }

void GgmlRouterRunner::free_layer(Layer &l) {
  if (l.route_done_event) {
    ggml_backend_event_free((ggml_backend_event_t)l.route_done_event);
    l.route_done_event = nullptr;
  }
  if (l.route_ids_host) {
    cudaFreeHost(l.route_ids_host);
    l.route_ids_host = nullptr;
  }
  if (l.route_weights_host) {
    cudaFreeHost(l.route_weights_host);
    l.route_weights_host = nullptr;
  }
  l.route_readback_pending = false;
  if (l.graph_ctx) {
    ggml_free((ggml_context *)l.graph_ctx);
    l.graph_ctx = nullptr;
  }
  if (l.galloc) {
    ggml_gallocr_free((ggml_gallocr_t)l.galloc);
    l.galloc = nullptr;
  }
  l.graph = l.x = l.ids = l.weights = l.logits = nullptr;
  l.graph_in_flight = false;
  if (l.weight_buf) {
    ggml_backend_buffer_free((ggml_backend_buffer_t)l.weight_buf);
    l.weight_buf = nullptr;
  }
  if (l.weight_ctx) {
    ggml_free((ggml_context *)l.weight_ctx);
    l.weight_ctx = nullptr;
  }
  l.weight = l.bias = nullptr;
  l.ready = false;
}

void GgmlRouterRunner::shutdown() {
  if (device_ >= 0) cudaSetDevice(device_);
  if (backend_) ggml_backend_synchronize((ggml_backend_t)backend_);
  for (Layer &l : layers_) free_layer(l);
  backend_ = nullptr; // borrowed from ggml_shared_backend()
  device_ = -1;
  d_model_ = n_experts_ = topk_ = n_tokens_ = 0;
  normalize_ = false;
  weight_scale_ = 0.0f;
  graph_workspace_bytes_ = 0;
}

bool GgmlRouterRunner::ensure(int device, int d_model, int n_experts,
                              int topk, int n_tokens, bool normalize,
                              float weight_scale) {
  if (device < 0 || d_model <= 0 || n_experts <= 0 || topk <= 0 ||
      topk > n_experts || n_tokens <= 0)
    return false;
  if (backend_ && device_ == device && d_model_ == d_model &&
      n_experts_ == n_experts && topk_ == topk && n_tokens_ == n_tokens &&
      normalize_ == normalize && weight_scale_ == weight_scale)
    return true;
  if (backend_) shutdown();
  cudaSetDevice(device);
  backend_ = ggml_shared_backend(device);
  if (!backend_) return false;
  device_ = device;
  d_model_ = d_model;
  n_experts_ = n_experts;
  topk_ = topk;
  n_tokens_ = n_tokens;
  normalize_ = normalize;
  weight_scale_ = weight_scale;
  LOG_INFO("[GGML-ROUTER] enabled on GPU %d (D=%d experts=%d topk=%d S=%d)\n",
           device, d_model, n_experts, topk, n_tokens);
  return true;
}

bool GgmlRouterRunner::ensure_layer(int layer,
                                    const GgmlRouterWeightRef &router,
                                    const float *selection_bias) {
  if (!backend_ || layer < 0 || layer >= 256 || !router.host_data ||
      router.ne0 != d_model_ || router.ne1 != n_experts_ ||
      router.type >= (uint32_t)GGML_TYPE_COUNT)
    return false;
  Layer &l = layers_[layer];
  if (l.ready) return true;
  const ggml_type wt = (ggml_type)router.type;
  if (wt != GGML_TYPE_F32 && wt != GGML_TYPE_F16 && wt != GGML_TYPE_BF16 &&
      !ggml_is_quantized(wt))
    return false;

  cudaSetDevice(device_);
  ggml_init_params ip = {};
  ip.mem_size = ggml_tensor_overhead() * 8;
  ip.no_alloc = true;
  moe::GgmlContextGuard wctx(ggml_init(ip));
  if (!wctx) return false;
  ggml_tensor *w = ggml_new_tensor_2d(wctx.get(), wt, d_model_, n_experts_);
  ggml_tensor *b = selection_bias
                      ? ggml_new_tensor_1d(wctx.get(), GGML_TYPE_F32,
                                           n_experts_)
                      : nullptr;
  if (!w || (selection_bias && !b)) return false;
  moe::GgmlBackendBufferGuard wbuf(
      ggml_backend_alloc_ctx_tensors(wctx.get(),
                                     (ggml_backend_t)backend_));
  if (!wbuf) return false;
  const size_t wbytes = std::min(router.nbytes, ggml_nbytes(w));
  if (wbytes != ggml_nbytes(w)) {
    LOG_ERROR("[GGML-ROUTER] layer=%d router bytes short (%zu vs %zu)\n",
              layer, router.nbytes, ggml_nbytes(w));
    return false;
  }
  ggml_backend_tensor_set(w, router.host_data, 0, wbytes);
  if (b)
    ggml_backend_tensor_set(b, selection_bias, 0,
                            (size_t)n_experts_ * sizeof(float));
  ggml_set_name(w, "qwen4exp_router_weight");
  if (b) ggml_set_name(b, "qwen4exp_router_bias");

  l.weight_ctx = wctx.release();
  l.weight_buf = wbuf.release();
  l.weight = w;
  l.bias = b;
  if (!build_graph(layer)) {
    free_layer(l);
    return false;
  }
  l.ready = true;
  LOG_INFO("[GGML-ROUTER] layer=%d weight=%zu bytes%s\n", layer, wbytes,
           b ? " + selection bias" : "");
  return true;
}

bool GgmlRouterRunner::build_graph(int layer) {
  Layer &l = layers_[layer];
  if (!l.weight || !backend_) return false;
  ggml_init_params ip = {};
  ip.mem_size = ggml_tensor_overhead() * 128 + ggml_graph_overhead();
  ip.no_alloc = true;
  moe::GgmlContextGuard ctx(ggml_init(ip));
  if (!ctx) return false;
  ggml_cgraph *gf = ggml_new_graph(ctx.get());
  ggml_tensor *x =
      ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, d_model_, n_tokens_);
  if (!gf || !x) return false;
  ggml_set_input(x);
  ggml_set_name(x, "qwen4exp_router_input");
  const moe::RouterGraphNodes route = moe::build_router_topology(
      ctx.get(), (ggml_tensor *)l.weight, x, (ggml_tensor *)l.bias,
      n_experts_, topk_, n_tokens_, normalize_, weight_scale_);
  if (!route.logits || !route.ids || !route.weights) return false;
  ggml_set_name(route.logits, "qwen4exp_router_logits");
  ggml_set_output(route.ids);
  ggml_set_output(route.weights);
  static const bool compare_logits = [] {
    const char *e = std::getenv("ZEROMOE_QWEN_ROUTER_COMPARE");
    return e && e[0] == '1';
  }();
  if (compare_logits) ggml_set_output(route.logits);
  ggml_set_name(route.ids, "qwen4exp_router_ids");
  ggml_set_name(route.weights, "qwen4exp_router_weights");
  ggml_build_forward_expand(gf, route.ids);
  ggml_build_forward_expand(gf, route.weights);

  moe::GgmlGraphAllocation allocation =
      moe::allocate_graph(std::move(ctx), gf, (ggml_backend_t)backend_);
  if (!allocation) return false;
  ggml_gallocr_t ga = allocation.release_allocator();
  const size_t graph_bytes = ggml_gallocr_get_buffer_size(ga, 0);
  if (graph_bytes > graph_workspace_bytes_) {
    graph_workspace_bytes_ = graph_bytes;
    LOG_INFO("[GGML-ROUTER-MEM] graph workspace: %.2f MiB\n",
             (double)graph_bytes / (1024.0 * 1024.0));
  }
  l.graph_ctx = allocation.release_context();
  l.galloc = ga;
  l.graph = gf;
  l.x = x;
  l.ids = route.ids;
  l.weights = route.weights;
  l.logits = route.logits;
  return true;
}

bool GgmlRouterRunner::run(int layer, const void *device_x, int32_t *ids_ks,
                           float *weights_ks) {
  if (!backend_ || layer < 0 || layer >= 256 || !device_x || !ids_ks ||
      !weights_ks || !layers_[layer].ready)
    return false;
  Layer &l = layers_[layer];
  cudaSetDevice(device_);
  const size_t xbytes = (size_t)d_model_ * n_tokens_ * sizeof(float);
  if (cudaMemcpy(((ggml_tensor *)l.x)->data, device_x, xbytes,
                 cudaMemcpyDeviceToDevice) != cudaSuccess)
    return false;
  if (ggml_backend_graph_compute_async((ggml_backend_t)backend_,
                                       (ggml_cgraph *)l.graph) !=
      GGML_STATUS_SUCCESS)
    return false;
  // The graph runs on GGML's private CUDA stream, while the synchronous
  // tensor-get callback uses cudaStreamPerThread.  Synchronize the backend
  // before reading either output so the small route table cannot race the
  // still-running TOP_K/GET_ROWS graph.
  ggml_backend_synchronize((ggml_backend_t)backend_);
  // TOP_K and the final weight tensor are both compact outputs.  A flat
  // transfer is therefore sufficient and keeps the host layout identical to
  // GGML's row-major [tokens, topk] storage used by the MoE runner.
  const size_t route_count = (size_t)topk_ * n_tokens_;
  return copy_router_tensor_to_host(
             device_, (const ggml_tensor *)l.ids, ids_ks,
             route_count * sizeof(int32_t), "ids") &&
         copy_router_tensor_to_host(
             device_, (const ggml_tensor *)l.weights, weights_ks,
             route_count * sizeof(float), "weights");
}

bool GgmlRouterRunner::run_tensor_view(int layer, const void *src_view_ptr,
                                        const void **device_ids,
                                        const void **device_weights) {
  if (device_ids) *device_ids = nullptr;
  if (device_weights) *device_weights = nullptr;
  if (!backend_ || layer < 0 || layer >= 256 || !src_view_ptr ||
      !device_ids || !device_weights || !layers_[layer].ready) {
    return false;
  }

  Layer &l = layers_[layer];
  // A layer's persistent router input/output tensors are reused on the next
  // request.  The normal host-returning path already synchronizes before it
  // returns; the device-output path needs the same request-boundary guard,
  // while still leaving the current request entirely asynchronous.
  if (l.graph_in_flight) {
    ggml_backend_synchronize((ggml_backend_t)backend_);
    l.graph_in_flight = false;
  }

  const ggml_tensor *src_view = (const ggml_tensor *)src_view_ptr;
  if (src_view->type != GGML_TYPE_F32 || src_view->ne[0] != d_model_ ||
      src_view->ne[1] < n_tokens_ || src_view->ne[2] != 1 ||
      src_view->ne[3] != 1 || src_view->nb[0] != sizeof(float) ||
      src_view->nb[1] != (size_t)d_model_ * sizeof(float) ||
      !src_view->buffer || !src_view->data) {
    return false;
  }

  if (use_direct_router_input()) {
    // l.x is a graph input.  Its backing allocation remains owned by the
    // router graph allocator, but the CUDA kernels read the tensor's current
    // data pointer. Retargeting it to HC's persistent device tensor removes
    // one [D,S] D2D copy per layer and preserves the graph's existing output
    // and workspace allocations. Each layer has its own router input tensor;
    // the shared HC state is reused only after the queued graph dependency has
    // advanced to the next layer on the same backend stream.
    ((ggml_tensor *)l.x)->data = src_view->data;
    static bool logged_direct = false;
    if (!logged_direct) {
      logged_direct = true;
      LOG_INFO("[GGML-ROUTER] direct HC input active: route graph consumes "
               "the producer tensor without a D2D input copy\n");
    }
  } else {
    ggml_backend_tensor_copy_async((ggml_backend_t)backend_,
                                   (ggml_backend_t)backend_,
                                   const_cast<ggml_tensor *>(src_view),
                                   (ggml_tensor *)l.x);
  }
  const ggml_status st =
      ggml_backend_graph_compute_async((ggml_backend_t)backend_,
                                        (ggml_cgraph *)l.graph);
  if (st != GGML_STATUS_SUCCESS) {
    return false;
  }

  perf().n_ggml_router_device_calls.fetch_add(1, std::memory_order_relaxed);
  l.graph_in_flight = true;
  *device_ids = l.ids;
  *device_weights = l.weights;
  return true;
}

bool GgmlRouterRunner::run_tensor_device(int layer, const void *device_tensor,
                                         const void **device_ids,
                                         const void **device_weights) {
  if (!backend_ || !device_tensor || !device_ids || !device_weights)
    return false;
  cudaSetDevice(device_);
  ggml_init_params ip = {};
  ip.mem_size = ggml_tensor_overhead() * 8;
  ip.no_alloc = true;
  ggml_context *ctx = ggml_init(ip);
  if (!ctx) return false;

  // HC's persistent tensor may have a larger token capacity than this
  // request. The view keeps the producer-to-router copy at [D,n_tokens].
  const ggml_tensor *src = (const ggml_tensor *)device_tensor;
  const size_t view_bytes =
      ggml_row_size(GGML_TYPE_F32, d_model_) +
      (size_t)(n_tokens_ - 1) * (size_t)src->nb[1];
  if (src->type != GGML_TYPE_F32 || src->ne[0] != d_model_ ||
      src->ne[1] < n_tokens_ || src->ne[2] != 1 || src->ne[3] != 1 ||
      src->nb[0] != sizeof(float) ||
      src->nb[1] != (size_t)d_model_ * sizeof(float) || !src->buffer ||
      !src->data || view_bytes > ggml_nbytes(src)) {
    LOG_ERROR("[GGML-ROUTER] invalid device input view: D=%d S=%d "
              "type=%d ne=[%lld,%lld,%lld,%lld] nb=[%zu,%zu,%zu,%zu] "
              "view=%zu nbytes=%zu\n",
              d_model_, n_tokens_, (int)src->type,
              (long long)src->ne[0], (long long)src->ne[1],
              (long long)src->ne[2], (long long)src->ne[3], src->nb[0],
              src->nb[1], src->nb[2], src->nb[3], view_bytes,
              ggml_nbytes(src));
    ggml_free(ctx);
    return false;
  }
  ggml_tensor *src_view =
      ggml_view_2d(ctx, (ggml_tensor *)src, d_model_, n_tokens_,
                   src->nb[1], 0);
  // ggml_view_2d() leaves the borrowed backend-buffer marker unset when the
  // view is created in this no-allocation metadata context.  The data pointer
  // and strides are still valid, but run_tensor_view() and the backend copy
  // path need the parent CUDA buffer to recognize the view as device-resident.
  if (src_view) src_view->buffer = src->buffer;
  const bool ok = src_view && run_tensor_view(layer, src_view, device_ids,
                                              device_weights);
  ggml_free(ctx);
  return ok;
}

bool GgmlRouterRunner::run_tensor_stream0(int layer, const void *device_tensor,
                                          int token_index, int32_t *ids_ks,
                                          float *weights_ks) {
  if (!backend_ || !device_tensor || token_index < 0 || !ids_ks ||
      !weights_ks || n_tokens_ != 1)
    return false;
  const void *device_ids = nullptr;
  const void *device_weights = nullptr;
  if (!run_tensor_stream0_device(layer, device_tensor, token_index,
                                 &device_ids, &device_weights))
    return false;
  // run_tensor_view queues the graph; export its compact outputs through the
  // same synchronized host path used by run_tensor().
  Layer &l = layers_[layer];
  ggml_backend_synchronize((ggml_backend_t)backend_);
  l.graph_in_flight = false;
  const size_t route_count = (size_t)topk_;
  const bool ids_ok = copy_router_tensor_to_host(
      device_, (const ggml_tensor *)l.ids, ids_ks,
      route_count * sizeof(int32_t), "pilot ids");
  const bool weights_ok = ids_ok && copy_router_tensor_to_host(
      device_, (const ggml_tensor *)l.weights, weights_ks,
      route_count * sizeof(float), "pilot weights");
  return ids_ok && weights_ok;
}

bool GgmlRouterRunner::run_tensor_stream0_device(
    int layer, const void *device_tensor, int token_index,
    const void **device_ids, const void **device_weights) {
  if (!backend_ || !device_tensor || token_index < 0 || !device_ids ||
      !device_weights || n_tokens_ != 1)
    return false;
  cudaSetDevice(device_);
  ggml_init_params ip = {};
  ip.mem_size = ggml_tensor_overhead() * 8;
  ip.no_alloc = true;
  ggml_context *ctx = ggml_init(ip);
  if (!ctx) return false;
  const ggml_tensor *streams = (const ggml_tensor *)device_tensor;
  if (streams->type != GGML_TYPE_F32 || streams->ne[0] != d_model_ ||
      streams->ne[1] <= 0 || streams->ne[2] <= token_index ||
      streams->ne[3] != 1 || streams->nb[0] != sizeof(float) ||
      streams->nb[1] != (size_t)d_model_ * sizeof(float) ||
      streams->nb[2] == 0 || !streams->buffer || !streams->data) {
    ggml_free(ctx);
    return false;
  }
  // t_streams_ is [D,HC,S]. Stream 0 of token `token_index` starts at the
  // token stride; a 2-D view presents exactly one contiguous [D] row to the
  // existing router graph. Copy and graph execution remain on the shared
  // backend stream, so the preceding HC post-combine is ordered without a
  // host device-wide synchronization.
  ggml_tensor *src_view = ggml_view_2d(
      ctx, (ggml_tensor *)streams, d_model_, 1, streams->nb[1],
      (size_t)token_index * streams->nb[2]);
  if (src_view) src_view->buffer = streams->buffer;
  const bool queued = src_view && run_tensor_view(
                                      layer, src_view, device_ids,
                                      device_weights);
  ggml_free(ctx);
  return queued;
}

bool GgmlRouterRunner::run_tensor(int layer, const void *device_tensor,
                                  int32_t *ids_ks, float *weights_ks) {
  if (!backend_ || layer < 0 || layer >= 256 || !device_tensor || !ids_ks ||
      !weights_ks || !layers_[layer].ready)
    return false;

  // IDs and weights are consumed by the host-side expert-set bookkeeping and
  // therefore must be available before the MoE graph is submitted. This is a
  // small synchronization: the large HC activation never crosses PCIe.
  const auto total_start = std::chrono::steady_clock::now();
  const void *device_ids = nullptr;
  const void *device_weights = nullptr;
  if (!run_tensor_device(layer, device_tensor, &device_ids,
                         &device_weights))
    return false;
  const auto sync_start = std::chrono::steady_clock::now();
  ggml_backend_synchronize((ggml_backend_t)backend_);
  const double sync_s = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - sync_start).count();
  perf().t_ggml_router_host_sync.fetch_add(sync_s, std::memory_order_relaxed);
  perf().n_ggml_router_host_calls.fetch_add(1, std::memory_order_relaxed);
  layers_[layer].graph_in_flight = false;
  const auto get_start = std::chrono::steady_clock::now();
  const size_t route_count = (size_t)topk_ * n_tokens_;
  const bool ids_ok = copy_router_tensor_to_host(
      device_, (const ggml_tensor *)device_ids, ids_ks,
      route_count * sizeof(int32_t), "ids");
  const bool weights_ok = ids_ok && copy_router_tensor_to_host(
      device_, (const ggml_tensor *)device_weights, weights_ks,
      route_count * sizeof(float), "weights");
  const double get_s = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - get_start).count();
  perf().t_ggml_router_host_get.fetch_add(get_s, std::memory_order_relaxed);
  perf().t_ggml_router_host_call.fetch_add(
      std::chrono::duration<double>(std::chrono::steady_clock::now() -
          total_start)
          .count(),
      std::memory_order_relaxed);
  return ids_ok && weights_ok;
}

bool GgmlRouterRunner::copy_route_to_host(int layer, int32_t *ids_ks,
                                          float *weights_ks) {
  if (!backend_ || layer < 0 || layer >= 256 || !ids_ks || !weights_ks ||
      !layers_[layer].ready)
    return false;
  Layer &l = layers_[layer];
  if (l.route_readback_pending) {
    if (!l.route_done_event || !l.route_ids_host || !l.route_weights_host)
      return false;
    // The event is recorded after both asynchronous D2H copies on GGML's
    // CUDA backend stream.  Poll first so normal decode can consume the
    // compact route table without a host-side synchronization; only the rare
    // not-ready case waits for the router work that attention failed to hide.
    cudaError_t ready = cudaEventQuery(
        (cudaEvent_t)((ggml_backend_event_t)l.route_done_event)->context);
    if (ready == cudaErrorNotReady) {
      ggml_backend_event_synchronize(
          (ggml_backend_event_t)l.route_done_event);
      ready = cudaSuccess;
    }
    if (ready != cudaSuccess) {
      cudaGetLastError();
      return false;
    }
    const size_t route_count = (size_t)topk_ * n_tokens_;
    std::memcpy(ids_ks, l.route_ids_host,
                route_count * sizeof(int32_t));
    std::memcpy(weights_ks, l.route_weights_host,
                route_count * sizeof(float));
    l.route_readback_pending = false;
    l.graph_in_flight = false;
    return true;
  }
  if (l.graph_in_flight) {
    ggml_backend_synchronize((ggml_backend_t)backend_);
    l.graph_in_flight = false;
  }
  const size_t route_count = (size_t)topk_ * n_tokens_;
  return copy_router_tensor_to_host(
             device_, (const ggml_tensor *)l.ids, ids_ks,
             route_count * sizeof(int32_t), "ids") &&
         copy_router_tensor_to_host(
             device_, (const ggml_tensor *)l.weights, weights_ks,
             route_count * sizeof(float), "weights");
}

bool GgmlRouterRunner::queue_route_to_host(int layer) {
  if (!backend_ || layer < 0 || layer >= 256 || !layers_[layer].ready)
    return false;
  Layer &l = layers_[layer];
  if (l.route_readback_pending || !l.graph_in_flight)
    return false;

  const size_t route_count = (size_t)topk_ * n_tokens_;
  if (!l.route_ids_host &&
      cudaHostAlloc((void **)&l.route_ids_host,
                    route_count * sizeof(int32_t),
                    cudaHostAllocPortable) != cudaSuccess) {
    l.route_ids_host = nullptr;
    return false;
  }
  if (!l.route_weights_host &&
      cudaHostAlloc((void **)&l.route_weights_host,
                    route_count * sizeof(float),
                    cudaHostAllocPortable) != cudaSuccess) {
    l.route_weights_host = nullptr;
    return false;
  }
  if (!l.route_done_event) {
    l.route_done_event = ggml_backend_event_new(
        ggml_backend_get_device((ggml_backend_t)backend_));
    if (!l.route_done_event) return false;
  }

  // run_tensor_stream0_device() has queued the router graph on this same
  // backend stream.  These two gets therefore execute after TOP_K/GET_ROWS,
  // and the event below fences the copies without forcing the caller to wait.
  ggml_backend_tensor_get_async(
      (ggml_backend_t)backend_, (const ggml_tensor *)l.ids,
      l.route_ids_host, 0, route_count * sizeof(int32_t));
  ggml_backend_tensor_get_async(
      (ggml_backend_t)backend_, (const ggml_tensor *)l.weights,
      l.route_weights_host, 0, route_count * sizeof(float));
  ggml_backend_event_record((ggml_backend_event_t)l.route_done_event,
                            (ggml_backend_t)backend_);
  l.route_readback_pending = true;
  return true;
}

bool GgmlRouterRunner::copy_logits_to_host(int layer, float *logits) {
  if (!backend_ || layer < 0 || layer >= 256 || !logits ||
      !layers_[layer].ready)
    return false;
  Layer &l = layers_[layer];
  if (l.graph_in_flight) {
    ggml_backend_synchronize((ggml_backend_t)backend_);
    l.graph_in_flight = false;
  }
  return copy_router_tensor_to_host(
      device_, (const ggml_tensor *)l.logits, logits,
      (size_t)n_experts_ * (size_t)n_tokens_ * sizeof(float), "logits");
}

// ---------------------------------------------------------------------------
#else  // !ZEROMOE_USE_GGML_CUDA
bool GgmlRouterRunner::compiled_in() { return false; }
GgmlRouterRunner &GgmlRouterRunner::instance() {
  static GgmlRouterRunner r;
  return r;
}
GgmlRouterRunner::~GgmlRouterRunner() {}
void GgmlRouterRunner::shutdown() {}
bool GgmlRouterRunner::ensure(int, int, int, int, int, bool, float) {
  return false;
}
bool GgmlRouterRunner::ensure_layer(int, const GgmlRouterWeightRef &,
                                    const float *) {
  return false;
}
bool GgmlRouterRunner::run(int, const void *, int32_t *, float *) {
  return false;
}
bool GgmlRouterRunner::run_tensor(int, const void *, int32_t *, float *) {
  return false;
}
bool GgmlRouterRunner::run_tensor_device(int, const void *, const void **,
                                         const void **) {
  return false;
}
bool GgmlRouterRunner::run_tensor_stream0(int, const void *, int, int32_t *,
                                          float *) {
  return false;
}
bool GgmlRouterRunner::run_tensor_stream0_device(
    int, const void *, int, const void **, const void **) {
  return false;
}
bool GgmlRouterRunner::copy_route_to_host(int, int32_t *, float *) {
  return false;
}
bool GgmlRouterRunner::queue_route_to_host(int) { return false; }
bool GgmlRouterRunner::copy_logits_to_host(int, float *) { return false; }
#endif  // ZEROMOE_USE_GGML_CUDA
