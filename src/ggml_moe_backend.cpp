// ---------------------------------------------------------------------------
// ggml MUL_MAT_ID prefill backend -- implementation.
// See include/ggml_moe_backend.hpp for the design rationale and the shape
// contract this file implements.
// ---------------------------------------------------------------------------
#include "ggml_moe_backend.hpp"

#include "logger.hpp"
#include "memory_guard.hpp"
#include "perf_timer.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <vector>

#if defined(MOESTREAM_USE_GGML_CUDA)

#include <cuda_runtime.h>

#include "gpu_topology.hpp"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-cuda.h"

extern "C" cudaError_t launch_ggml_moe_unpack(
    const void *src, void *gate, void *up, void *down, int count,
    size_t src_stride, size_t gate_bytes, size_t up_bytes, size_t down_bytes,
    size_t gate_stride, size_t up_stride, size_t down_stride,
    cudaStream_t stream);

extern "C" cudaError_t launch_ggml_moe_unpack_scatter(
    const void *src, void *gate, void *up, void *down,
    const int32_t *dst_slots, int count, size_t src_stride,
    size_t gate_bytes, size_t up_bytes, size_t down_bytes,
    size_t gate_stride, size_t up_stride, size_t down_stride,
    cudaStream_t stream);

extern "C" void launch_expert_scatter_add(
    const float *src, const int *tok, float *dst, int n, int D, int S,
    cudaStream_t stream);

namespace {

// ggml-cuda pads the LAST row of a quantised matrix out to a multiple of this
// so MMQ's k-tiling can read past the real end without going out of bounds
// (src/ggml-cuda/common.cuh's MATRIX_ROW_PADDING). It is not exported, so it
// is mirrored here; ggml_backend_buffer_get_alloc_size() applies the same rule
// internally and ggml_backend_tensor_alloc() asserts the buffer has room for
// it, so getting this wrong fails loudly at allocation rather than silently.
constexpr int64_t kMatrixRowPadding = 512;

// Extra bytes ggml-cuda will consider part of a [ne0, ne1, ne2] tensor of
// `type` beyond ggml_nbytes(). Zero unless ne0 is not already a multiple of
// kMatrixRowPadding.
size_t cuda_row_pad_bytes(ggml_type type, int64_t ne0) {
  if (!ggml_is_quantized(type)) return 0;
  const int64_t rem = ne0 % kMatrixRowPadding;
  if (rem == 0) return 0;
  return ggml_row_size(type, kMatrixRowPadding - rem);
}

size_t align_up(size_t v, size_t a) { return (v + a - 1) / a * a; }

size_t lcm_size(size_t a, size_t b) {
  if (a == 0 || b == 0) return 0;
  return (a / std::gcd(a, b)) * b;
}

// A quant type is usable here only if ggml can actually multiply with it AND
// its row size divides evenly, which every real GGUF weight type does. This
// deliberately does NOT try to predict WHICH ggml-cuda path (MMQ / MMVQ / the
// dequantise+cuBLAS fallback) will be chosen -- all three are correct; only
// their speed differs, and predicting that here would just be a second copy of
// ggml_cuda_mul_mat_id's own dispatch logic that could drift out of sync.
bool type_supported(uint32_t t) {
  if (t >= (uint32_t)GGML_TYPE_COUNT) return false;
  const ggml_type gt = (ggml_type)t;
  if (gt == GGML_TYPE_F32 || gt == GGML_TYPE_F16 || gt == GGML_TYPE_BF16)
    return true;
  return ggml_is_quantized(gt);
}

// Route IDs and weights are compact host-visible outputs.  Do not use
// ggml_backend_tensor_get() here: the CUDA backend implements that callback
// with cudaMemcpyAsync() on cudaStreamPerThread and CUDA_CHECK(), so a
// transient invalid pointer/stream or a previously surfaced asynchronous
// error aborts the whole prefill before the caller can use its established
// CPU-router fallback.  All callers below have already synchronized the
// shared GGML backend.  A synchronous copy is both safe for these few bytes
// and, unlike the backend callback, gives us a recoverable error path.
bool copy_router_tensor_to_host(int device, const ggml_tensor *tensor,
                                void *host, size_t bytes, const char *name) {
  if (device < 0 || !tensor || !tensor->data || !host || bytes == 0)
    return false;
  const size_t available = ggml_nbytes(tensor);
  if (bytes > available) {
    LOG_ERROR("[GGML-ROUTER] %s readback bounds error: requested=%zu "
              "available=%zu\n",
              name, bytes, available);
    return false;
  }
  cudaSetDevice(device);
  const cudaError_t err =
      cudaMemcpy(host, tensor->data, bytes, cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    LOG_ERROR("[GGML-ROUTER] %s readback failed: %s (bytes=%zu "
              "tensor_bytes=%zu device=%d data=%p)\n",
              name, cudaGetErrorString(err), bytes, available, device,
              tensor->data);
    return false;
  }
  return true;
}

bool backend_stream_copy_override = false;

// "Bee-style" H2D-queued-on-shared-GGML-stream mode (MOESTREAM_GGML_BACKEND_
// STREAM_COPY=1 or the override flag above). TRIED earlier this session:
// gave no measurable improvement over the plain copy_stream_ path (61.70 vs
// 62.72 tok/s baseline) -- left OFF by default. RE-VERIFIED 2026-09-10
// against the current, materially faster post-fix baseline (RAM/VRAM tier
// independence fix + vram_preload_prefix_layers + the GgmlQwenSharedRunner
// device_x fix): same 574-token real-text prompt, prefill= from the
// shutdown report's Rates line -- ON: 86.01 tok/s, vs the same-config OFF
// baseline's 79.84/86.37/85.66 tok/s (mean 83.96) -- squarely inside the
// OFF spread, not a separate cluster. Conclusion unchanged: still no real
// effect, confirmed against the current baseline too.
bool use_backend_stream_copies() {
  static const bool env_enabled = [] {
    const char *env = std::getenv("MOESTREAM_GGML_BACKEND_STREAM_COPY");
    return env && env[0] == '1';
  }();
  return env_enabled || backend_stream_copy_override;
}

// The HC runner and the router share the same CUDA backend.  In the normal
// path the router copies HC's [D,S] result into one persistent input tensor
// before every route graph.  Bee's full graph passes the preceding activation
// directly to the router node, so the copy is unnecessary when the producer
// tensor is already on this device.  Keep this as an A/B switch until the
// direct-input lifetime and numerical result have been verified on the real
// Qwen prefill.
bool use_direct_router_input() {
  static const bool enabled = [] {
    const char *env = std::getenv("MOESTREAM_GGML_ROUTER_DIRECT_INPUT");
    return env && env[0] == '1';
  }();
  return enabled;
}

bool moe_ggml_planar_copies_override = false;

// The packed transfer path is the experimental optimization under test. Keep
// a runtime escape hatch to the old three-planar-copy path so correctness can
// be bisected without rebuilding the engine or changing the GGML graph.
bool use_packed_transfer() {
  static const bool enabled = [] {
    const char *env = std::getenv("MOESTREAM_GGML_PLANAR_COPIES");
    return !(env && env[0] == '1');
  }();
  return enabled && !moe_ggml_planar_copies_override;
}

bool trace_pure_h2d() {
  static const bool enabled = [] {
    const char *env = std::getenv("MOESTREAM_TRACE_PURE_H2D");
    return env && env[0] == '1';
  }();
  return enabled;
}

bool async_moe_graph_reuse_override = false;

// The backend-stream copy experiment originally forced a full backend
// synchronize after every MoE graph because this runner reused one graph
// allocator and one output tensor. GGML's allocator reset only rewinds its
// host-side virtual allocator; it does not clear device bytes, and all graph
// work is ordered on the shared backend stream. Once HC has queued its
// producer-output copy on that same stream, the next graph can reuse those
// addresses without a CPU wait. Keep this opt-in until it is validated on the
// real Qwen prompt; the default remains the established conservative path.
bool use_async_moe_graph_reuse() {
  static const bool enabled = [] {
    const char *env = std::getenv("MOESTREAM_GGML_MOE_ASYNC_GRAPH_REUSE");
    return env && env[0] == '1';
  }();
  return enabled || async_moe_graph_reuse_override;
}

bool debug_sync_route_inputs() {
  static const bool enabled = [] {
    const char *env = std::getenv("MOESTREAM_GGML_DEBUG_SYNC_ROUTE");
    return env && env[0] == '1';
  }();
  return enabled;
}

bool debug_device_sync() {
  static const bool enabled = [] {
    const char *env = std::getenv("MOESTREAM_GGML_DEBUG_DEVICE_SYNC");
    return env && env[0] == '1';
  }();
  return enabled;
}

bool debug_copy_sync() {
  static const bool enabled = [] {
    const char *env = std::getenv("MOESTREAM_GGML_DEBUG_COPY_SYNC");
    return env && env[0] == '1';
  }();
  return enabled;
}

bool trace_backend_calls() {
  static const bool enabled = [] {
    const char *env = std::getenv("MOESTREAM_GGML_TRACE_BACKEND_CALLS");
    return env && env[0] == '1';
  }();
  return enabled;
}

bool debug_check_device_output() {
  static const bool enabled = [] {
    const char *env = std::getenv("MOESTREAM_GGML_DEBUG_CHECK_DEVICE_OUTPUT");
    return env && env[0] == '1';
  }();
  return enabled;
}

}  // namespace

void set_moe_ggml_planar_copies(bool enabled) {
  moe_ggml_planar_copies_override = enabled;
}

void set_ggml_moe_async_graph_reuse(bool enabled) {
  async_moe_graph_reuse_override = enabled;
  // The async graph mode is only safe and useful when expert weights are
  // double-buffered and copied through the backend's ordered copy path. The
  // legacy raw-copy mode has one compact weight buffer and must synchronize
  // before every layer reuses it, which silently defeats this configuration.
  // Keep the environment variable as an explicit opt-in for callers that
  // want backend-stream copies without enabling graph reuse.
  backend_stream_copy_override = enabled;
}

void *ggml_shared_backend(int device) {
  // One backend per device, kept for the process lifetime. Never freed: the
  // runners that borrow it have process lifetime too, and tearing it down
  // while another runner still holds tensors in its buffers would be a
  // use-after-free with no upside.
  static ggml_backend_t cached[16] = {nullptr};
  static int cached_dev[16] = {-1};
  if (device < 0 || device >= 16) return nullptr;
  if (cached[device] && cached_dev[device] == device) return cached[device];
  ggml_backend_t b = ggml_backend_cuda_init(device);
  if (!b) {
    LOG_ERROR("[GGML] ggml_backend_cuda_init(%d) failed\n", device);
    return nullptr;
  }
  cached[device] = b;
  cached_dev[device] = device;
  return b;
}

// ---------------------------------------------------------------------------
// Test-only device tensor helper (2026-09-10) -- see the header's own
// comment. Deliberately minimal: one F32 2-D tensor, no_alloc context plus a
// real backend buffer, populated via a synchronous tensor_set so the data is
// guaranteed resident before the caller uses the pointer.
// ---------------------------------------------------------------------------

bool ggml_moe_test_make_device_tensor(int device, int64_t ne0, int64_t ne1,
                                      const float *host_data,
                                      void **out_tensor, void **out_ctx,
                                      void **out_buf) {
  if (!out_tensor || !out_ctx || !out_buf || ne0 <= 0 || ne1 <= 0 ||
      !host_data)
    return false;
  *out_tensor = nullptr;
  *out_ctx = nullptr;
  *out_buf = nullptr;
  ggml_backend_t backend = (ggml_backend_t)ggml_shared_backend(device);
  if (!backend) return false;
  cudaSetDevice(device);
  ggml_init_params ip = {};
  ip.mem_size = ggml_tensor_overhead() * 4;
  ip.no_alloc = true;
  ggml_context *ctx = ggml_init(ip);
  if (!ctx) return false;
  ggml_tensor *t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0, ne1);
  if (!t) {
    ggml_free(ctx);
    return false;
  }
  ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
  if (!buf) {
    ggml_free(ctx);
    return false;
  }
  ggml_backend_tensor_set(t, host_data, 0,
                          (size_t)ne0 * (size_t)ne1 * sizeof(float));
  *out_tensor = t;
  *out_ctx = ctx;
  *out_buf = buf;
  return true;
}

void ggml_moe_test_free_device_tensor(void *ctx, void *buf) {
  if (buf) ggml_backend_buffer_free((ggml_backend_buffer_t)buf);
  if (ctx) ggml_free((ggml_context *)ctx);
}

bool ggml_moe_test_download_tensor(int device, const void *tensor,
                                   float *host_out, size_t n_floats) {
  if (!tensor || !host_out) return false;
  cudaSetDevice(device);
  cudaDeviceSynchronize();
  ggml_backend_tensor_get((const ggml_tensor *)tensor, host_out, 0,
                          n_floats * sizeof(float));
  return true;
}

// ---------------------------------------------------------------------------
// Qwen4Exp device router
// ---------------------------------------------------------------------------

GgmlRouterRunner &GgmlRouterRunner::instance() {
  static GgmlRouterRunner r;
  return r;
}

bool GgmlRouterRunner::compiled_in() { return true; }

GgmlRouterRunner::~GgmlRouterRunner() { shutdown(); }

void GgmlRouterRunner::free_layer(Layer &l) {
  if (l.graph_ctx) {
    ggml_free((ggml_context *)l.graph_ctx);
    l.graph_ctx = nullptr;
  }
  if (l.galloc) {
    ggml_gallocr_free((ggml_gallocr_t)l.galloc);
    l.galloc = nullptr;
  }
  l.graph = l.x = l.ids = l.weights = nullptr;
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
  ggml_context *wctx = ggml_init(ip);
  if (!wctx) return false;
  ggml_tensor *w = ggml_new_tensor_2d(wctx, wt, d_model_, n_experts_);
  ggml_tensor *b = selection_bias
                      ? ggml_new_tensor_1d(wctx, GGML_TYPE_F32, n_experts_)
                      : nullptr;
  if (!w || (selection_bias && !b)) {
    ggml_free(wctx);
    return false;
  }
  ggml_backend_buffer_t wbuf =
      ggml_backend_alloc_ctx_tensors(wctx, (ggml_backend_t)backend_);
  if (!wbuf) {
    ggml_free(wctx);
    return false;
  }
  const size_t wbytes = std::min(router.nbytes, ggml_nbytes(w));
  if (wbytes != ggml_nbytes(w)) {
    LOG_ERROR("[GGML-ROUTER] layer=%d router bytes short (%zu vs %zu)\n",
              layer, router.nbytes, ggml_nbytes(w));
    ggml_backend_buffer_free(wbuf);
    ggml_free(wctx);
    return false;
  }
  ggml_backend_tensor_set(w, router.host_data, 0, wbytes);
  if (b)
    ggml_backend_tensor_set(b, selection_bias, 0,
                            (size_t)n_experts_ * sizeof(float));
  ggml_set_name(w, "qwen4exp_router_weight");
  if (b) ggml_set_name(b, "qwen4exp_router_bias");

  l.weight_ctx = wctx;
  l.weight_buf = wbuf;
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
  ggml_context *ctx = ggml_init(ip);
  if (!ctx) return false;
  ggml_cgraph *gf = ggml_new_graph(ctx);
  ggml_tensor *x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_model_, n_tokens_);
  if (!gf || !x) {
    ggml_free(ctx);
    return false;
  }
  ggml_set_input(x);
  ggml_set_name(x, "qwen4exp_router_input");
  ggml_tensor *logits = ggml_mul_mat(ctx, (ggml_tensor *)l.weight, x);
  if (!logits) {
    ggml_free(ctx);
    return false;
  }
  ggml_set_name(logits, "qwen4exp_router_logits");
  ggml_tensor *probs = ggml_soft_max(ctx, logits);
  ggml_tensor *selection = probs;
  if (l.bias) selection = ggml_add(ctx, selection, (ggml_tensor *)l.bias);
  // TOP_K writes a compact [topk, tokens] I32 result directly.  The older
  // argsort_top_k helper returns a strided view into a [n_experts, tokens]
  // argsort result; that view is useful to the fused Bee routing path but is
  // unnecessarily fragile when this small graph exports IDs to host code.
  ggml_tensor *ids = ggml_top_k(ctx, selection, topk_);
  ggml_tensor *probs3 = ggml_reshape_3d(ctx, probs, 1, n_experts_, n_tokens_);
  ggml_tensor *weights = ggml_get_rows(ctx, probs3, ids);
  if (!ids || !probs3 || !weights) {
    ggml_free(ctx);
    return false;
  }
  if (normalize_) {
    ggml_tensor *w2 = ggml_reshape_2d(ctx, weights, topk_, n_tokens_);
    ggml_tensor *sum = ggml_sum_rows(ctx, w2);
    sum = ggml_clamp(ctx, sum, 6.103515625e-5f, INFINITY);
    w2 = ggml_div(ctx, w2, sum);
    weights = ggml_reshape_3d(ctx, w2, 1, topk_, n_tokens_);
  }
  if (weight_scale_ != 0.0f && weight_scale_ != 1.0f)
    weights = ggml_scale(ctx, weights, weight_scale_);
  if (!weights) {
    ggml_free(ctx);
    return false;
  }
  ggml_set_output(ids);
  ggml_set_output(weights);
  ggml_set_name(ids, "qwen4exp_router_ids");
  ggml_set_name(weights, "qwen4exp_router_weights");
  ggml_build_forward_expand(gf, ids);
  ggml_build_forward_expand(gf, weights);

  ggml_backend_buffer_type_t bt =
      ggml_backend_get_default_buffer_type((ggml_backend_t)backend_);
  ggml_gallocr_t ga = ggml_gallocr_new(bt);
  if (!ga || !ggml_gallocr_alloc_graph(ga, gf)) {
    if (ga) ggml_gallocr_free(ga);
    ggml_free(ctx);
    return false;
  }
  const size_t graph_bytes = ggml_gallocr_get_buffer_size(ga, 0);
  if (graph_bytes > graph_workspace_bytes_) {
    graph_workspace_bytes_ = graph_bytes;
    LOG_INFO("[GGML-ROUTER-MEM] graph workspace: %.2f MiB\n",
             (double)graph_bytes / (1024.0 * 1024.0));
  }
  l.graph_ctx = ctx;
  l.galloc = ga;
  l.graph = gf;
  l.x = x;
  l.ids = ids;
  l.weights = weights;
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

bool GgmlRouterRunner::run_tensor_device(int layer, const void *device_tensor,
                                         const void **device_ids,
                                         const void **device_weights) {
  if (device_ids) *device_ids = nullptr;
  if (device_weights) *device_weights = nullptr;
  if (!backend_ || layer < 0 || layer >= 256 || !device_tensor ||
      !device_ids || !device_weights || !layers_[layer].ready)
    return false;

  Layer &l = layers_[layer];
  // A layer's persistent router input/output tensors are reused on the next
  // request.  The normal host-returning path already synchronizes before it
  // returns; the device-output path needs the same request-boundary guard,
  // while still leaving the current request entirely asynchronous.
  if (l.graph_in_flight) {
    ggml_backend_synchronize((ggml_backend_t)backend_);
    l.graph_in_flight = false;
  }

  ggml_tensor *src = (ggml_tensor *)device_tensor;
  if (src->type != GGML_TYPE_F32 || src->ne[0] != d_model_ ||
      src->ne[1] < n_tokens_ || src->ne[2] != 1 || src->ne[3] != 1 ||
      src->nb[0] != sizeof(float) ||
      src->nb[1] != (size_t)d_model_ * sizeof(float) || !src->buffer ||
      !src->data)
    return false;

  cudaSetDevice(device_);
  ggml_init_params ip = {};
  ip.mem_size = ggml_tensor_overhead() * 8;
  ip.no_alloc = true;
  ggml_context *ctx = ggml_init(ip);
  if (!ctx) return false;

  // HC's persistent tensor may have a larger token capacity than this
  // request.  The view keeps the producer-to-router copy at [D,n_tokens].
  ggml_tensor *src_view =
      ggml_view_2d(ctx, src, d_model_, n_tokens_, src->nb[1], 0);
  if (!src_view) {
    ggml_free(ctx);
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
                                   (ggml_backend_t)backend_, src_view,
                                   (ggml_tensor *)l.x);
  }
  const ggml_status st =
      ggml_backend_graph_compute_async((ggml_backend_t)backend_,
                                        (ggml_cgraph *)l.graph);
  ggml_free(ctx);
  if (st != GGML_STATUS_SUCCESS)
    return false;

  perf().n_ggml_router_device_calls.fetch_add(1, std::memory_order_relaxed);
  l.graph_in_flight = true;
  *device_ids = l.ids;
  *device_weights = l.weights;
  return true;
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

// ---------------------------------------------------------------------------
// Qwen4Exp shared expert
// ---------------------------------------------------------------------------

GgmlQwenSharedRunner &GgmlQwenSharedRunner::instance() {
  static GgmlQwenSharedRunner r;
  return r;
}

bool GgmlQwenSharedRunner::compiled_in() { return true; }

GgmlQwenSharedRunner::~GgmlQwenSharedRunner() { shutdown(); }

void GgmlQwenSharedRunner::free_graph() {
  if (graph_ctx_) {
    ggml_free((ggml_context *)graph_ctx_);
    graph_ctx_ = nullptr;
  }
  if (galloc_) {
    ggml_gallocr_free((ggml_gallocr_t)galloc_);
    galloc_ = nullptr;
  }
  graph_ = x_ = routed_ = output_ = nullptr;
  graph_ready_ = false;
}

void GgmlQwenSharedRunner::shutdown() {
  if (device_ >= 0) cudaSetDevice(device_);
  if (backend_) ggml_backend_synchronize((ggml_backend_t)backend_);
  free_graph();
  if (weight_buf_) {
    ggml_backend_buffer_free((ggml_backend_buffer_t)weight_buf_);
    weight_buf_ = nullptr;
  }
  if (weight_ctx_) {
    ggml_free((ggml_context *)weight_ctx_);
    weight_ctx_ = nullptr;
  }
  gate_ = up_ = down_ = gate_inp_ = nullptr;
  backend_ = nullptr; // borrowed from ggml_shared_backend()
  device_ = -1;
  d_model_ = shared_inter_ = n_tokens_ = 0;
  has_gate_ = false;
  clamp_ = 0.0f;
  gate_type_ = up_type_ = down_type_ = gate_inp_type_ = 0;
}

bool GgmlQwenSharedRunner::ensure(int device, int d_model, int shared_inter,
                                  int n_tokens, bool has_gate, float clamp) {
  if (device < 0 || d_model <= 0 || shared_inter <= 0 || n_tokens <= 0)
    return false;
  if (backend_ && device_ == device && d_model_ == d_model &&
      shared_inter_ == shared_inter && n_tokens_ == n_tokens &&
      has_gate_ == has_gate && clamp_ == clamp)
    return true;
  if (backend_) shutdown();
  cudaSetDevice(device);
  backend_ = ggml_shared_backend(device);
  if (!backend_) return false;
  device_ = device;
  d_model_ = d_model;
  shared_inter_ = shared_inter;
  n_tokens_ = n_tokens;
  has_gate_ = has_gate;
  clamp_ = clamp;
  LOG_INFO("[GGML-SHARED] enabled on GPU %d (D=%d inter=%d S=%d gate=%s)\n",
           device, d_model, shared_inter, n_tokens, has_gate ? "yes" : "no");
  return true;
}

bool GgmlQwenSharedRunner::build_graph() {
  if (!backend_ || !gate_ || !up_ || !down_ ||
      (has_gate_ && !gate_inp_))
    return false;
  ggml_init_params ip = {};
  ip.mem_size = ggml_tensor_overhead() * 96 + ggml_graph_overhead();
  ip.no_alloc = true;
  ggml_context *ctx = ggml_init(ip);
  if (!ctx) return false;
  ggml_cgraph *gf = ggml_new_graph(ctx);
  ggml_tensor *x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_model_, n_tokens_);
  ggml_tensor *routed =
      ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_model_, n_tokens_);
  if (!gf || !x || !routed) {
    ggml_free(ctx);
    return false;
  }
  ggml_set_input(x);
  ggml_set_input(routed);
  ggml_set_name(x, "qwen4exp_shared_input");
  ggml_set_name(routed, "qwen4exp_routed_output");

  ggml_tensor *gate = ggml_mul_mat(ctx, (ggml_tensor *)gate_, x);
  ggml_tensor *up = ggml_mul_mat(ctx, (ggml_tensor *)up_, x);
  ggml_tensor *act = ggml_mul(ctx, ggml_silu(ctx, gate), up);
  if (clamp_ > 0.0f)
    act = ggml_clamp(ctx, act, -clamp_, clamp_);
  ggml_tensor *shared = ggml_mul_mat(ctx, (ggml_tensor *)down_, act);
  if (has_gate_) {
    ggml_tensor *gate_scalar =
        ggml_sigmoid(ctx, ggml_mul_mat(ctx, (ggml_tensor *)gate_inp_, x));
    shared = ggml_mul(ctx, shared, gate_scalar);
  }
  ggml_tensor *combined = ggml_add(ctx, routed, shared);
  if (!gate || !up || !act || !shared || !combined) {
    ggml_free(ctx);
    return false;
  }
  ggml_set_output(combined);
  ggml_set_name(combined, "qwen4exp_routed_plus_shared");
  ggml_build_forward_expand(gf, combined);

  ggml_backend_buffer_type_t bt =
      ggml_backend_get_default_buffer_type((ggml_backend_t)backend_);
  ggml_gallocr_t ga = ggml_gallocr_new(bt);
  if (!ga || !ggml_gallocr_alloc_graph(ga, gf)) {
    if (ga) ggml_gallocr_free(ga);
    ggml_free(ctx);
    return false;
  }
  LOG_INFO("[GGML-SHARED-MEM] graph workspace: %.2f MiB\n",
           (double)ggml_gallocr_get_buffer_size(ga, 0) /
               (1024.0 * 1024.0));
  graph_ctx_ = ctx;
  galloc_ = ga;
  graph_ = gf;
  x_ = x;
  routed_ = routed;
  output_ = combined;
  graph_ready_ = true;
  return true;
}

bool GgmlQwenSharedRunner::ensure_layer(
    int layer, const GgmlSharedWeightRef &gate,
    const GgmlSharedWeightRef &up, const GgmlSharedWeightRef &down,
    const GgmlSharedWeightRef *gate_inp) {
  if (!backend_ || layer < 0 || !gate.host_data || !up.host_data ||
      !down.host_data || gate.ne0 != d_model_ || gate.ne1 != shared_inter_ ||
      up.ne0 != d_model_ || up.ne1 != shared_inter_ ||
      down.ne0 != shared_inter_ || down.ne1 != d_model_ ||
      (has_gate_ && (!gate_inp || !gate_inp->host_data || gate_inp->ne0 != d_model_ ||
                     gate_inp->ne1 != 1)))
    return false;
  if (gate.type >= (uint32_t)GGML_TYPE_COUNT ||
      up.type >= (uint32_t)GGML_TYPE_COUNT ||
      down.type >= (uint32_t)GGML_TYPE_COUNT ||
      (gate_inp && gate_inp->type >= (uint32_t)GGML_TYPE_COUNT))
    return false;
  if (!type_supported(gate.type) || !type_supported(up.type) ||
      !type_supported(down.type) ||
      (has_gate_ && !type_supported(gate_inp->type)))
    return false;

  const uint32_t new_gate_inp_type = has_gate_ ? gate_inp->type : 0;
  const bool type_changed =
      weight_ctx_ &&
      (gate_type_ != gate.type || up_type_ != up.type ||
       down_type_ != down.type || gate_inp_type_ != new_gate_inp_type);
  if (type_changed) {
    ggml_backend_synchronize((ggml_backend_t)backend_);
    free_graph();
    if (weight_buf_) {
      ggml_backend_buffer_free((ggml_backend_buffer_t)weight_buf_);
      weight_buf_ = nullptr;
    }
    if (weight_ctx_) {
      ggml_free((ggml_context *)weight_ctx_);
      weight_ctx_ = nullptr;
    }
    gate_ = up_ = down_ = gate_inp_ = nullptr;
  }

  cudaSetDevice(device_);
  if (!weight_ctx_) {
    ggml_init_params ip = {};
    ip.mem_size = ggml_tensor_overhead() * 12;
    ip.no_alloc = true;
    ggml_context *ctx = ggml_init(ip);
    if (!ctx) return false;
    ggml_tensor *g = ggml_new_tensor_2d(ctx, (ggml_type)gate.type,
                                        d_model_, shared_inter_);
    ggml_tensor *u = ggml_new_tensor_2d(ctx, (ggml_type)up.type,
                                        d_model_, shared_inter_);
    ggml_tensor *d = ggml_new_tensor_2d(ctx, (ggml_type)down.type,
                                        shared_inter_, d_model_);
    ggml_tensor *gi = has_gate_
                         ? ggml_new_tensor_2d(ctx, (ggml_type)gate_inp->type,
                                              d_model_, 1)
                         : nullptr;
    if (!g || !u || !d || (has_gate_ && !gi)) {
      ggml_free(ctx);
      return false;
    }
    ggml_backend_buffer_t buf =
        ggml_backend_alloc_ctx_tensors(ctx, (ggml_backend_t)backend_);
    if (!buf) {
      ggml_free(ctx);
      return false;
    }
    weight_ctx_ = ctx;
    weight_buf_ = buf;
    gate_ = g;
    up_ = u;
    down_ = d;
    gate_inp_ = gi;
    gate_type_ = gate.type;
    up_type_ = up.type;
    down_type_ = down.type;
    gate_inp_type_ = new_gate_inp_type;
  }

  const size_t gb = ggml_nbytes((ggml_tensor *)gate_);
  const size_t ub = ggml_nbytes((ggml_tensor *)up_);
  const size_t db = ggml_nbytes((ggml_tensor *)down_);
  if (gate.nbytes < gb || up.nbytes < ub || down.nbytes < db ||
      (has_gate_ && gate_inp->nbytes <
                         ggml_nbytes((ggml_tensor *)gate_inp_)))
    return false;
  // All weight updates and the following graph run share one GGML CUDA
  // stream.  This queues the next layer's weights behind the previous layer's
  // HC post-combine instead of synchronizing the host for each projection.
  ggml_backend_tensor_set_async((ggml_backend_t)backend_,
                                (ggml_tensor *)gate_, gate.host_data, 0, gb);
  ggml_backend_tensor_set_async((ggml_backend_t)backend_,
                                (ggml_tensor *)up_, up.host_data, 0, ub);
  ggml_backend_tensor_set_async((ggml_backend_t)backend_,
                                (ggml_tensor *)down_, down.host_data, 0, db);
  if (has_gate_)
    ggml_backend_tensor_set_async(
        (ggml_backend_t)backend_, (ggml_tensor *)gate_inp_,
        gate_inp->host_data, 0, ggml_nbytes((ggml_tensor *)gate_inp_));
  if (!graph_ready_ && !build_graph()) {
    free_graph();
    return false;
  }
  if (layer == 0)
    LOG_INFO("[GGML-SHARED] layer weights ready (gate=%u up=%u down=%u%s)\n",
             (unsigned)gate.type, (unsigned)up.type, (unsigned)down.type,
             has_gate_ ? " + scalar gate" : "");
  return true;
}

bool GgmlQwenSharedRunner::run(int layer, const void *device_x,
                               const void *routed_output,
                               const void **combined_output) {
  if (!backend_ || !graph_ready_ || layer < 0 || !device_x ||
      !routed_output || !combined_output)
    return false;
  ggml_tensor *routed = (ggml_tensor *)routed_output;
  ggml_tensor *x = (ggml_tensor *)x_;
  ggml_tensor *r = (ggml_tensor *)routed_;
  if (r->type != GGML_TYPE_F32 || r->ne[0] != d_model_ ||
      r->ne[1] != n_tokens_ || r->ne[2] != 1 || r->ne[3] != 1 ||
      r->nb[0] != sizeof(float) ||
      r->nb[1] != (size_t)d_model_ * sizeof(float) || !r->data || !r->buffer)
    return false;
  // 2026-09-10: real, confirmed root cause of the shared-expert NaN
  // corruption bug (see Test 35, which reproduces it in isolation with a
  // CPU oracle). Every production caller of run() -- run_qwen4exp_shared_device()
  // in execution_layer.cpp, fed by GgmlHcRunner::pre_mix()'s
  // `*device_output = t_mixed_` -- passes device_x as a `ggml_tensor *`
  // (t_mixed_), exactly like routed_output below. This function instead
  // fed device_x straight into cudaMemcpy() as if it were already a raw
  // CUDA device pointer: copying from a host ggml_tensor struct's address
  // via cudaMemcpyDeviceToDevice is undefined behavior, and in practice
  // silently produced an all-zero (or garbage) x_ input -- silu(0)*0=0
  // down the whole gate/up/act/shared chain, so `combined` collapsed to
  // just `routed` (see the "combined == routed" trace). Dereference it the
  // same way `src` is handled just below, instead of memcpy'ing the wrong
  // bytes.
  ggml_tensor *hc_in = (ggml_tensor *)device_x;
  if (hc_in->type != GGML_TYPE_F32 || hc_in->ne[0] != d_model_ ||
      hc_in->ne[1] != n_tokens_ || hc_in->ne[2] != 1 || hc_in->ne[3] != 1 ||
      hc_in->nb[0] != sizeof(float) ||
      hc_in->nb[1] != (size_t)d_model_ * sizeof(float) || !hc_in->data ||
      !hc_in->buffer)
    return false;
  cudaSetDevice(device_);
  const size_t bytes = (size_t)d_model_ * n_tokens_ * sizeof(float);
  if (cudaMemcpy(x->data, hc_in->data, bytes, cudaMemcpyDeviceToDevice) !=
      cudaSuccess)
    return false;
  ggml_tensor *src = (ggml_tensor *)routed_output;
  if (src->type != GGML_TYPE_F32 || src->ne[0] != d_model_ ||
      src->ne[1] != n_tokens_ || src->ne[2] != 1 || src->ne[3] != 1 ||
      src->nb[0] != sizeof(float) ||
      src->nb[1] != (size_t)d_model_ * sizeof(float) || !src->data ||
      !src->buffer)
    return false;
  // The routed result is produced by GgmlMoeRunner's graph.  Copy that
  // producer tensor into this graph's independent input; copying the
  // destination tensor to itself would leave stale routed data after the
  // first layer and change all subsequent routing decisions.
  ggml_backend_tensor_copy_async((ggml_backend_t)backend_,
                                 (ggml_backend_t)backend_, src, r);
  if (ggml_backend_graph_compute_async((ggml_backend_t)backend_,
                                       (ggml_cgraph *)graph_) !=
      GGML_STATUS_SUCCESS)
    return false;
  // 2026-09-10: added -- unlike GgmlMoeRunner (which exposes an explicit
  // wait_device_output() specifically because its graph_out_ is consumed by
  // ANOTHER runner on a potentially different stream, see that method's own
  // comment), this class had no equivalent: run() returned right after
  // queuing the async compute, with no guarantee output_'s data was actually
  // ready before the caller's combined_output pointer got used. Real,
  // confirmed correctness bug when this path was tried: 100% NaN corruption
  // end to end (hidden_states 1249280/1249280, logits 248320/248320), while
  // throughput looked great (140.59 tok/s) -- exactly the shape of a race
  // that reads uninitialized/stale VRAM. Synchronizing here is the same
  // safety margin ggml_backend_sched_graph_compute (the ordinary synchronous
  // wrapper) already gives every other graph in this codebase.
  ggml_backend_synchronize((ggml_backend_t)backend_);
  *combined_output = output_;
  return true;
}

bool GgmlMoeRunner::compiled_in() { return true; }

namespace {
constexpr int kMoeRunnerPoolSize = 4;
// Slot 0 is the legacy/default runner used by non-Qwen callers. Slots 1..3
// are keyed by the projection quantization signature. The model currently
// needs two, but leaving one spare keeps this dispatch safe for a future
// mixed-quant model without returning to a graph rebuild on every transition.
uint32_t g_moe_runner_keys[kMoeRunnerPoolSize][3] = {};
bool g_moe_runner_key_valid[kMoeRunnerPoolSize] = {};
int g_active_moe_runner = 0;
}

GgmlMoeRunner *GgmlMoeRunner::runner_pool() {
  static GgmlMoeRunner pool[kMoeRunnerPoolSize];
  return pool;
}

GgmlMoeRunner &GgmlMoeRunner::instance() {
  return runner_pool()[0];
}

GgmlMoeRunner &GgmlMoeRunner::instance_for(uint32_t gate_type,
                                            uint32_t up_type,
                                            uint32_t down_type) {
  for (int i = 1; i < kMoeRunnerPoolSize; ++i) {
    if (g_moe_runner_key_valid[i] &&
        g_moe_runner_keys[i][0] == gate_type &&
        g_moe_runner_keys[i][1] == up_type &&
        g_moe_runner_keys[i][2] == down_type) {
      g_active_moe_runner = i;
      return runner_pool()[i];
    }
  }
  for (int i = 1; i < kMoeRunnerPoolSize; ++i) {
    if (!g_moe_runner_key_valid[i]) {
      g_moe_runner_keys[i][0] = gate_type;
      g_moe_runner_keys[i][1] = up_type;
      g_moe_runner_keys[i][2] = down_type;
      g_moe_runner_key_valid[i] = true;
      g_active_moe_runner = i;
      LOG_INFO("[GGML-MOE] persistent runner slot=%d signature=%u/%u/%u\n",
               i, (unsigned)gate_type, (unsigned)up_type,
               (unsigned)down_type);
      return runner_pool()[i];
    }
  }
  // More signatures than the bounded pool can retain: use the legacy runner
  // and let begin_layer's existing shape validation handle it safely.
  LOG_WARN("[GGML-MOE] persistent runner pool exhausted for signature=%u/%u/%u; "
           "falling back to the default runner\n",
           (unsigned)gate_type, (unsigned)up_type, (unsigned)down_type);
  g_active_moe_runner = 0;
  return runner_pool()[0];
}

GgmlMoeRunner &GgmlMoeRunner::active_instance() {
  if (g_active_moe_runner < 0 || g_active_moe_runner >= kMoeRunnerPoolSize)
    g_active_moe_runner = 0;
  return runner_pool()[g_active_moe_runner];
}

void GgmlMoeRunner::finish_all_h2d() {
  GgmlMoeRunner *pool = runner_pool();
  for (int i = 0; i < kMoeRunnerPoolSize; ++i)
    if (pool[i].backend_) pool[i].finish_h2d();
}

void GgmlMoeRunner::collect_all_h2d(double &seconds, size_t &bytes) {
  GgmlMoeTransferStats stats;
  collect_all_transfer_stats(stats);
  seconds = stats.h2d_seconds;
  bytes = stats.h2d_bytes;
}

void GgmlMoeRunner::collect_all_transfer_stats(GgmlMoeTransferStats &stats) {
  stats = GgmlMoeTransferStats{};
  GgmlMoeRunner *pool = runner_pool();
  for (int i = 0; i < kMoeRunnerPoolSize; ++i) {
    if (!pool[i].backend_) continue;
    GgmlMoeTransferStats one;
    pool[i].collect_transfer_stats(one);
    stats.d2d_seconds += one.d2d_seconds;
    stats.h2d_seconds += one.h2d_seconds;
    stats.h2d_dma_seconds += one.h2d_dma_seconds;
    stats.d2d_bytes += one.d2d_bytes;
    stats.h2d_bytes += one.h2d_bytes;
  }
}

GgmlMoeRunner::~GgmlMoeRunner() { shutdown(); }

bool GgmlMoeRunner::ensure_h2d_timing_events() {
  if (!trace_pure_h2d()) return true;
  for (int buf = 0; buf < NBUF; ++buf) {
    for (int i = 0; i < NH2D_TIMING_EVENTS; ++i) {
      if (!h2d_t0_event_[buf][i] &&
          cudaEventCreate((cudaEvent_t *)&h2d_t0_event_[buf][i]) != cudaSuccess)
        return false;
      if (!h2d_t1_event_[buf][i] &&
          cudaEventCreate((cudaEvent_t *)&h2d_t1_event_[buf][i]) != cudaSuccess)
        return false;
    }
  }
  return true;
}

bool GgmlMoeRunner::begin_h2d_timing(int buf, int &slot) {
  slot = -1;
  if (!trace_pure_h2d()) return true;
  if (buf < 0 || buf >= NBUF) return false;
  const int next = h2d_timing_count_[buf];
  if (next >= NH2D_TIMING_EVENTS) return true; // tracer capacity only
  if (cudaEventRecord((cudaEvent_t)h2d_t0_event_[buf][next],
                      (cudaStream_t)copy_stream_) != cudaSuccess)
    return false;
  h2d_timing_count_[buf] = next + 1;
  slot = next;
  return true;
}

bool GgmlMoeRunner::end_h2d_timing(int buf, int slot) {
  if (!trace_pure_h2d() || slot < 0) return true;
  return cudaEventRecord((cudaEvent_t)h2d_t1_event_[buf][slot],
                         (cudaStream_t)copy_stream_) == cudaSuccess;
}

void GgmlMoeRunner::shutdown() {
  if (device_ >= 0) cudaSetDevice(device_);  // see wait_stage_free()
  if (copy_stream_) {
    cudaStreamSynchronize((cudaStream_t)copy_stream_);
  }
  if (copy_backend_) {
    ggml_backend_synchronize((ggml_backend_t)copy_backend_);
  }
  if (backend_) {
    ggml_backend_synchronize((ggml_backend_t)backend_);
  }
  // All deferred sources are safe after the stream synchronizations above.
  // Release them before destroying the runner's events and buffers, including
  // when a process exits without reaching the normal HC collapse hook.
  release_completed_vram_holds();
  free_graph();
  free_weight_tensors();
  if (external_buf_) {
    ggml_backend_buffer_free((ggml_backend_buffer_t)external_buf_);
    external_buf_ = nullptr;
  }
  for (int i = 0; i < NWEIGHT_BUFS; ++i) {
    if (wbuf_[i]) {
      ggml_backend_buffer_free((ggml_backend_buffer_t)wbuf_[i]);
      wbuf_[i] = nullptr;
    }
    if (packed_wbuf_[i]) {
      cudaFree(packed_wbuf_[i]);
      packed_wbuf_[i] = nullptr;
    }
  }
  wbuf_bytes_ = 0;
  packed_wbuf_bytes_ = 0;
  for (int i = 0; i < NBUF; ++i) {
    if (h_stage_[i]) {
      cudaFreeHost(h_stage_[i]);
      h_stage_[i] = nullptr;
    }
    if (d_stage_slot_map_[i]) {
      cudaFree(d_stage_slot_map_[i]);
      d_stage_slot_map_[i] = nullptr;
    }
    stage_slot_map_[i].clear();
    if (ev_free_[i]) {
      cudaEventDestroy((cudaEvent_t)ev_free_[i]);
      ev_free_[i] = nullptr;
    }
    if (ev_t0_[i]) {
      cudaEventDestroy((cudaEvent_t)ev_t0_[i]);
      ev_t0_[i] = nullptr;
    }
    if (backend_stage_event_[i]) {
      ggml_backend_event_free(
          (ggml_backend_event_t)backend_stage_event_[i]);
      backend_stage_event_[i] = nullptr;
    }
    if (backend_copy_t0_event_[i]) {
      cudaEventDestroy((cudaEvent_t)backend_copy_t0_event_[i]);
      backend_copy_t0_event_[i] = nullptr;
    }
    if (backend_copy_t1_event_[i]) {
      cudaEventDestroy((cudaEvent_t)backend_copy_t1_event_[i]);
      backend_copy_t1_event_[i] = nullptr;
    }
    if (backend_copy_split_event_[i]) {
      cudaEventDestroy((cudaEvent_t)backend_copy_split_event_[i]);
      backend_copy_split_event_[i] = nullptr;
    }
    for (int j = 0; j < NH2D_TIMING_EVENTS; ++j) {
      if (h2d_t0_event_[i][j]) {
        cudaEventDestroy((cudaEvent_t)h2d_t0_event_[i][j]);
        h2d_t0_event_[i][j] = nullptr;
      }
      if (h2d_t1_event_[i][j]) {
        cudaEventDestroy((cudaEvent_t)h2d_t1_event_[i][j]);
        h2d_t1_event_[i][j] = nullptr;
      }
    }
    h2d_timing_count_[i] = 0;
    if (chunk_scatter_done_events_[i]) {
      cudaEventDestroy((cudaEvent_t)chunk_scatter_done_events_[i]);
      chunk_scatter_done_events_[i] = nullptr;
    }
    pending_[i] = false;
    copy_batch_started_[i] = false;
    direct_h2d_bytes_[i] = 0;
    pending_d2d_bytes_[i] = 0;
    copy_split_recorded_[i] = false;
  }
  if (route_ids_pinned_) {
    cudaFreeHost(route_ids_pinned_);
    route_ids_pinned_ = nullptr;
  }
  if (route_weights_pinned_) {
    cudaFreeHost(route_weights_pinned_);
    route_weights_pinned_ = nullptr;
  }
  route_stage_capacity_ = 0;
  route_stage_index_ = 0;
  MemoryGuard::release_host_bytes(stage_host_reservation_bytes_);
  stage_host_reservation_bytes_ = 0;
  for (int i = 0; i < NWEIGHT_EVENTS; ++i) {
    if (backend_weight_done_event_[i]) {
      ggml_backend_event_free(
          (ggml_backend_event_t)backend_weight_done_event_[i]);
      backend_weight_done_event_[i] = nullptr;
    }
  }
  for (int i = 0; i < NWEIGHT_BUFS; ++i) {
    backend_weight_last_event_[i] = nullptr;
    backend_weight_buffer_in_use_[i] = false;
  }
  backend_weight_event_next_ = 0;
  weight_buffer_index_ = 0;
  stage_bytes_ = 0;
  stage_slot_capacity_ = 0;
  if (copy_stream_) {
    cudaStreamDestroy((cudaStream_t)copy_stream_);
    copy_stream_ = nullptr;
  }
  if (copy_backend_) {
    ggml_backend_free((ggml_backend_t)copy_backend_);
    copy_backend_ = nullptr;
  }
  free_chunk_state();
  if (galloc_) {
    ggml_gallocr_free((ggml_gallocr_t)galloc_);
    galloc_ = nullptr;
  }
  if (graph_done_event_) {
    ggml_backend_event_free((ggml_backend_event_t)graph_done_event_);
    graph_done_event_ = nullptr;
  }
  if (chunk_ready_event_) {
    ggml_backend_event_free((ggml_backend_event_t)chunk_ready_event_);
    chunk_ready_event_ = nullptr;
  }
  if (chunk_graph_event_) {
    ggml_backend_event_free((ggml_backend_event_t)chunk_graph_event_);
    chunk_graph_event_ = nullptr;
  }
  for (int i = 0; i < NBUF; ++i) {
    if (chunk_graph_events_[i]) {
      ggml_backend_event_free((ggml_backend_event_t)chunk_graph_events_[i]);
      chunk_graph_events_[i] = nullptr;
    }
  }
  if (chunk_accum_event_) {
    ggml_backend_event_free((ggml_backend_event_t)chunk_accum_event_);
    chunk_accum_event_ = nullptr;
  }
  for (int i = 0; i < NBUF; ++i) {
    if (chunk_weight_done_events_[i]) {
      ggml_backend_event_free(
          (ggml_backend_event_t)chunk_weight_done_events_[i]);
      chunk_weight_done_events_[i] = nullptr;
    }
    if (chunk_token_device_[i]) {
      cudaFree(chunk_token_device_[i]);
      chunk_token_device_[i] = nullptr;
    }
  }
  graph_in_flight_ = false;
  // backend_ is shared (ggml_shared_backend) and deliberately not freed here.
  backend_ = nullptr;
  device_ = -1;
  backend_stream_copies_ = false;
  raw_d2d_pending_ = false;
  layer_ready_ = false;
  weight_slots_ = 0;
  slot_capacity_ = 0;
  weight_expert_stride_ = 0;
  weight_raw_expert_stride_ = 0;
  weight_tail_padding_ = 0;
  weight_gate_plane_offset_ = 0;
  weight_up_plane_offset_ = 0;
  weight_down_plane_offset_ = 0;
  packed_base_ = nullptr;
  weight_d_model_ = weight_d_ff_ = 0;
  weight_gate_type_ = weight_up_type_ = weight_down_type_ = 0;
  weight_external_ = false;
}

bool GgmlMoeRunner::ensure_route_stage(size_t route_count) {
  if (route_count == 0)
    return true;
  if (route_ids_pinned_ && route_weights_pinned_ &&
      route_stage_capacity_ >= route_count)
    return true;

  // A larger request may require reallocating the pinned ring.  The previous
  // ring can still be the source of an asynchronous H2D queued on the shared
  // backend, so drain it before releasing or reusing that storage.  This is a
  // request-shape change, not a per-layer synchronization point.
  if (backend_)
    ggml_backend_synchronize((ggml_backend_t)backend_);
  if (route_ids_pinned_) {
    cudaFreeHost(route_ids_pinned_);
    route_ids_pinned_ = nullptr;
  }
  if (route_weights_pinned_) {
    cudaFreeHost(route_weights_pinned_);
    route_weights_pinned_ = nullptr;
  }
  route_stage_capacity_ = 0;
  route_stage_index_ = 0;

  if (route_count > std::numeric_limits<size_t>::max() /
                          (size_t)NROUTE_STAGE ||
      route_count * (size_t)NROUTE_STAGE >
          std::numeric_limits<size_t>::max() / sizeof(int32_t))
    return false;
  const size_t entries = route_count * (size_t)NROUTE_STAGE;
  if (cudaHostAlloc((void **)&route_ids_pinned_,
                    entries * sizeof(int32_t), cudaHostAllocPortable) !=
          cudaSuccess ||
      cudaHostAlloc((void **)&route_weights_pinned_,
                    entries * sizeof(float), cudaHostAllocPortable) !=
          cudaSuccess) {
    if (route_ids_pinned_) {
      cudaFreeHost(route_ids_pinned_);
      route_ids_pinned_ = nullptr;
    }
    if (route_weights_pinned_) {
      cudaFreeHost(route_weights_pinned_);
      route_weights_pinned_ = nullptr;
    }
    return false;
  }
  route_stage_capacity_ = route_count;
  return true;
}

void GgmlMoeRunner::invalidate_graph() {
  // Same drain-then-free sequence ensure_graph() uses on its own shape-
  // mismatch rebuild path -- see this method's own header comment (ggml_
  // moe_backend.hpp) for the real crash this was extracted to fix. Unlike
  // ensure_graph()'s internal call, this is reachable with the runner never
  // having been set up at all (backend_ == nullptr) -- guard it explicitly
  // since this method is public.
  if (graph_in_flight_ && backend_) {
    ggml_backend_synchronize((ggml_backend_t)backend_);
    graph_in_flight_ = false;
  }
  free_graph();
}

void GgmlMoeRunner::free_graph() {
  if (graph_ctx_) {
    ggml_free((ggml_context *)graph_ctx_);
    graph_ctx_ = nullptr;
  }
  graph_ = nullptr;
  graph_x_ = graph_ids_ = graph_w_ = nullptr;
  graph_token_ids_ = nullptr;
  graph_gate_out_ = graph_up_out_ = graph_act_out_ = graph_down_out_ = nullptr;
  graph_out_ = nullptr;
  graph_tokens_ = graph_source_tokens_ = graph_topk_ = graph_d_model_ =
      graph_d_ff_ = 0;
  graph_slots_ = 0;
  graph_gate_type_ = graph_up_type_ = graph_down_type_ = 0;
  graph_chunked_ = false;
}

void GgmlMoeRunner::free_chunk_state() {
  if (device_ >= 0) cudaSetDevice(device_);
  if (chunk_accum_) {
    cudaFree(chunk_accum_);
    chunk_accum_ = nullptr;
  }
  chunk_accum_bytes_ = 0;
  chunk_token_bytes_ = 0;
  if (chunk_final_buf_) {
    ggml_backend_buffer_free((ggml_backend_buffer_t)chunk_final_buf_);
    chunk_final_buf_ = nullptr;
  }
  if (chunk_final_ctx_) {
    ggml_free((ggml_context *)chunk_final_ctx_);
    chunk_final_ctx_ = nullptr;
  }
  chunk_final_ = nullptr;
  chunk_final_tokens_ = 0;
  chunk_ready_recorded_ = false;
  chunk_accum_recorded_ = false;
  chunk_input_published_ = false;
  chunk_has_submission_ = false;
  chunk_stage_index_ = 0;
  for (int i = 0; i < NBUF; ++i) {
    chunk_weight_recorded_[i] = false;
    chunk_scatter_recorded_[i] = false;
  }
  for (int i = 0; i < NBUF; ++i) {
    chunk_token_stage_[i].clear();
    chunk_id_stage_[i].clear();
    chunk_weight_stage_[i].clear();
  }
  chunk_token_keepalive_.clear();
  chunk_id_keepalive_.clear();
  chunk_weight_keepalive_.clear();
}

bool GgmlMoeRunner::ensure_chunk_state() {
  if (!backend_ || !plan_.chunked || plan_.source_tokens <= 0 ||
      plan_.n_tokens <= 0 || plan_.d_model <= 0)
    return false;
  const size_t token_bytes = (size_t)plan_.n_tokens * sizeof(int32_t);
  if (chunk_token_bytes_ != token_bytes) {
    for (int i = 0; i < NBUF; ++i) {
      if (chunk_token_device_[i]) {
        cudaFree(chunk_token_device_[i]);
        chunk_token_device_[i] = nullptr;
      }
      chunk_scatter_recorded_[i] = false;
    }
    chunk_token_bytes_ = 0;
  }
  if (chunk_token_bytes_ == 0) {
    for (int i = 0; i < NBUF; ++i) {
      if (cudaMalloc(&chunk_token_device_[i], token_bytes) != cudaSuccess ||
          !chunk_token_device_[i]) {
        LOG_ERROR("[GGML-MOE] chunk token snapshot allocation failed (%zu bytes)\n",
                  token_bytes);
        for (int j = 0; j <= i; ++j) {
          if (chunk_token_device_[j]) {
            cudaFree(chunk_token_device_[j]);
            chunk_token_device_[j] = nullptr;
          }
        }
        return false;
      }
    }
    chunk_token_bytes_ = token_bytes;
  }
  const size_t accum_bytes = (size_t)plan_.source_tokens * plan_.d_model *
                             sizeof(float);
  if (!chunk_accum_ || chunk_accum_bytes_ != accum_bytes) {
    // begin_layer() has already fenced an old graph if the shape changed.
    if (chunk_accum_) cudaFree(chunk_accum_);
    chunk_accum_ = nullptr;
    if (cudaMalloc(&chunk_accum_, accum_bytes) != cudaSuccess || !chunk_accum_) {
      LOG_ERROR("[GGML-MOE] chunk accumulator allocation failed (%.2f MiB)\n",
                (double)accum_bytes / (1024.0 * 1024.0));
      chunk_accum_bytes_ = 0;
      return false;
    }
    chunk_accum_bytes_ = accum_bytes;
  }

  if (!chunk_final_ || chunk_final_buf_ == nullptr || chunk_final_ctx_ == nullptr ||
      chunk_final_tokens_ != plan_.source_tokens) {
    if (chunk_final_buf_) {
      ggml_backend_buffer_free((ggml_backend_buffer_t)chunk_final_buf_);
      chunk_final_buf_ = nullptr;
    }
    if (chunk_final_ctx_) {
      ggml_free((ggml_context *)chunk_final_ctx_);
      chunk_final_ctx_ = nullptr;
    }
    ggml_init_params ip = {};
    ip.mem_size = ggml_tensor_overhead();
    ip.no_alloc = true;
    ggml_context *ctx = ggml_init(ip);
    if (!ctx) return false;
    ggml_tensor *out = ggml_new_tensor_2d(
        ctx, GGML_TYPE_F32, plan_.d_model, plan_.source_tokens);
    if (!out) {
      ggml_free(ctx);
      return false;
    }
    ggml_backend_buffer_t buf =
        ggml_backend_alloc_ctx_tensors(ctx, (ggml_backend_t)backend_);
    if (!buf) {
      ggml_free(ctx);
      return false;
    }
    chunk_final_ctx_ = ctx;
    chunk_final_buf_ = buf;
    chunk_final_ = out;
    chunk_final_tokens_ = plan_.source_tokens;
  }

  cudaSetDevice(device_);
  if (chunk_accum_recorded_)
    cudaStreamWaitEvent(0, (cudaEvent_t)
                              ((ggml_backend_event_t)chunk_accum_event_)->context,
                        0);
  if (cudaMemsetAsync(chunk_accum_, 0, accum_bytes, 0) != cudaSuccess)
    return false;
  if (cudaEventRecord(
          (cudaEvent_t)((ggml_backend_event_t)chunk_ready_event_)->context, 0) !=
      cudaSuccess)
    return false;
  chunk_ready_recorded_ = true;
  chunk_accum_recorded_ = false;
  chunk_input_published_ = false;
  chunk_has_submission_ = false;
  chunk_stage_index_ = 0;
  for (int i = 0; i < NBUF; ++i) chunk_scatter_recorded_[i] = false;
  return true;
}

void GgmlMoeRunner::free_weight_tensors() {
  free_graph();
  if (wctx_) {
    // Frees the tensor structs only; the bytes they point at belong to wbuf_.
    ggml_free((ggml_context *)wctx_);
    wctx_ = nullptr;
  }
  t_gate_ = t_up_ = t_down_ = nullptr;
  weight_slots_ = 0;
  weight_expert_stride_ = 0;
  weight_raw_expert_stride_ = 0;
  weight_tail_padding_ = 0;
  weight_d_model_ = weight_d_ff_ = 0;
  weight_gate_type_ = weight_up_type_ = weight_down_type_ = 0;
}

bool GgmlMoeRunner::ensure_backend(int device) {
  if (backend_ && device_ == device) return true;
  if (backend_) {
    // Switching devices mid-run is not something this path is built for; tear
    // the old one down completely rather than leaving two live backends whose
    // buffers point at different devices.
    shutdown();
  }
  backend_ = ggml_shared_backend(device);
  if (!backend_) return false;
  owns_backend_ = false;  // shared; see ggml_shared_backend()
  galloc_ = ggml_gallocr_new(
      ggml_backend_get_default_buffer_type((ggml_backend_t)backend_));
  if (!galloc_) {
    LOG_ERROR("[GGML-MOE] ggml_gallocr_new failed\n");
    backend_ = nullptr;
    return false;
  }
  graph_done_event_ = ggml_backend_event_new(
      ggml_backend_get_device((ggml_backend_t)backend_));
  if (!graph_done_event_) {
    LOG_ERROR("[GGML-MOE] graph completion event creation failed\n");
    ggml_gallocr_free((ggml_gallocr_t)galloc_);
    galloc_ = nullptr;
    backend_ = nullptr;
    return false;
  }
  external_buf_ = ggml_backend_alloc_buffer((ggml_backend_t)backend_, 1);
  if (!external_buf_) {
    LOG_ERROR("[GGML-MOE] external marker buffer allocation failed\n");
    ggml_backend_event_free((ggml_backend_event_t)graph_done_event_);
    graph_done_event_ = nullptr;
    ggml_gallocr_free((ggml_gallocr_t)galloc_);
    galloc_ = nullptr;
    backend_ = nullptr;
    return false;
  }
  // The marker has no model bytes. COMPUTE usage prevents GGML-CUDA's
  // quantized-buffer initializer from trying to clear padding beyond it.
  ggml_backend_buffer_set_usage(
      (ggml_backend_buffer_t)external_buf_, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
  chunk_ready_event_ = ggml_backend_event_new(
      ggml_backend_get_device((ggml_backend_t)backend_));
  chunk_graph_event_ = ggml_backend_event_new(
      ggml_backend_get_device((ggml_backend_t)backend_));
  chunk_accum_event_ = ggml_backend_event_new(
      ggml_backend_get_device((ggml_backend_t)backend_));
  for (int i = 0; i < NBUF; ++i) {
    chunk_graph_events_[i] = ggml_backend_event_new(
        ggml_backend_get_device((ggml_backend_t)backend_));
  }
  for (int i = 0; i < NBUF; ++i) {
    chunk_weight_done_events_[i] = ggml_backend_event_new(
        ggml_backend_get_device((ggml_backend_t)backend_));
    if (cudaEventCreateWithFlags(
            (cudaEvent_t *)&chunk_scatter_done_events_[i],
            cudaEventDisableTiming) != cudaSuccess) {
      LOG_ERROR("[GGML-MOE] chunk scatter event creation failed\n");
      return false;
    }
  }
  if (!chunk_ready_event_ || !chunk_graph_event_ || !chunk_accum_event_ ||
      !chunk_graph_events_[0] || !chunk_graph_events_[1] ||
      !chunk_weight_done_events_[0] || !chunk_weight_done_events_[1] ||
      !chunk_scatter_done_events_[0] || !chunk_scatter_done_events_[1]) {
    LOG_ERROR("[GGML-MOE] chunk pipeline event creation failed\n");
    return false;
  }
  device_ = device;
  // Planar copies use GGML's CUDA tensor-copy backend. Packed copies still
  // use the raw copy stream because their destination is an external packed
  // allocation followed by a CUDA unpack kernel; there is no GGML tensor for
  // that intermediate buffer. The backend-stream flag nevertheless remains
  // enabled for packed transfers: the raw stream records the same GGML event
  // bridge below, so the compute stream and HC consumer cannot observe the
  // unpacked planes before the copy/unpack work is complete. Disabling this
  // bridge made the packed path appear faster under diagnostic readback while
  // occasionally exposing stale/partial down-projection bytes to HC.
  backend_stream_copies_ = use_backend_stream_copies();
  if (backend_stream_copies_) {
    // H2D setters need their own GGML CUDA stream. The compute backend waits
    // on one event from that stream before submitting the MoE graph. Using a
    // second backend is important: recording and waiting on the same backend
    // stream cannot create a copy->compute dependency when the copy itself is
    // queued on that stream.
    copy_backend_ = ggml_backend_cuda_init(device);
    if (!copy_backend_) {
      LOG_ERROR("[GGML-MOE] GGML copy backend init failed on GPU %d\n", device);
      if (external_buf_) {
        ggml_backend_buffer_free((ggml_backend_buffer_t)external_buf_);
        external_buf_ = nullptr;
      }
      ggml_gallocr_free((ggml_gallocr_t)galloc_);
      galloc_ = nullptr;
      backend_ = nullptr;
      backend_stream_copies_ = false;
      return false;
    }
    LOG_INFO("[GGML-MOE] backend-stream expert copies enabled; staged "
             "pinned sources will use a dedicated copy backend\n");
  }
  return true;
}

bool GgmlMoeRunner::ensure_weight_buffer(size_t bytes, size_t packed_bytes) {
  bool have_all = wbuf_bytes_ >= bytes && packed_wbuf_bytes_ >= packed_bytes &&
                  wbuf_[0] && packed_wbuf_[0];
  if (have_all) return true;
  const size_t old_capacity = wbuf_bytes_;
  // Anything still reading the old buffer must be done before it is freed.
  // Nothing here is fire-and-forget across layers, but a failed run_layer()
  // can leave an H2D in flight, and cudaFree does NOT wait for one.
  if (copy_stream_) cudaStreamSynchronize((cudaStream_t)copy_stream_);
  if (copy_backend_) ggml_backend_synchronize((ggml_backend_t)copy_backend_);
  if (backend_) ggml_backend_synchronize((ggml_backend_t)backend_);
  free_weight_tensors();
  for (int i = 0; i < NWEIGHT_BUFS; ++i) {
    if (wbuf_[i]) {
      ggml_backend_buffer_free((ggml_backend_buffer_t)wbuf_[i]);
      wbuf_[i] = nullptr;
    }
    if (packed_wbuf_[i]) {
      cudaFree(packed_wbuf_[i]);
      packed_wbuf_[i] = nullptr;
    }
  }
  wbuf_bytes_ = 0;
  packed_wbuf_bytes_ = 0;
  ggml_backend_buffer_type_t buft = ggml_backend_cuda_buffer_type(device_);
  // This is a single per-layer compact-weight buffer.  Geometric growth used
  // to turn the first 700 MB layer followed by a 754.69 MB layer into a 1.4 GB
  // allocation, stranding about 645 MB on a 24 GB card.  slot_capacity_ already
  // keeps the tensor/graph shape at its high-water mark, so the extra geometric
  // slack does not buy useful reuse here.
  (void)old_capacity;
  const size_t alloc_bytes = bytes;
  const int n_buffers = NWEIGHT_BUFS;
  for (int i = 0; i < n_buffers; ++i) {
    ggml_backend_buffer_t b =
        ggml_backend_buft_alloc_buffer(buft, alloc_bytes);
    if (!b) {
      LOG_ERROR("[GGML-MOE] weight buffer alloc failed (%.2f MB on GPU %d)\n",
                (double)alloc_bytes / (1024.0 * 1024.0), device_);
      for (int j = 0; j < i; ++j) {
        ggml_backend_buffer_free((ggml_backend_buffer_t)wbuf_[j]);
        wbuf_[j] = nullptr;
      }
      return false;
    }
    // WEIGHTS usage keeps ggml-cuda from treating the tail padding as a
    // compute scratch region it has to memset on every matmul.
    ggml_backend_buffer_set_usage(b, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    wbuf_[i] = b;
    void *packed = nullptr;
    if (cudaMalloc(&packed, packed_bytes) != cudaSuccess || !packed) {
      LOG_ERROR("[GGML-MOE] packed transfer buffer alloc failed (%.2f MB on "
                "GPU %d)\n",
                (double)packed_bytes / (1024.0 * 1024.0), device_);
      ggml_backend_buffer_free(b);
      wbuf_[i] = nullptr;
      for (int j = 0; j < i; ++j) {
        ggml_backend_buffer_free((ggml_backend_buffer_t)wbuf_[j]);
        wbuf_[j] = nullptr;
        cudaFree(packed_wbuf_[j]);
        packed_wbuf_[j] = nullptr;
      }
      return false;
    }
    packed_wbuf_[i] = packed;
  }
  wbuf_bytes_ = alloc_bytes;
  packed_wbuf_bytes_ = packed_bytes;
  weight_buffer_index_ = 0;
  for (int i = 0; i < NWEIGHT_BUFS; ++i) {
    backend_weight_last_event_[i] = nullptr;
    backend_weight_buffer_in_use_[i] = false;
  }
  LOG_INFO("[GGML-MOE] weight buffer capacity: %.2f MB (needed %.2f MB), "
           "packed transfer capacity: %.2f MB (needed %.2f MB) on GPU %d "
           "(%d buffers)\n",
           (double)alloc_bytes / (1024.0 * 1024.0),
           (double)bytes / (1024.0 * 1024.0),
           (double)packed_bytes / (1024.0 * 1024.0),
           (double)packed_bytes / (1024.0 * 1024.0), device_, n_buffers);
  return true;
}

bool GgmlMoeRunner::ensure_stage_buffers(size_t bytes_per_buf) {
  if (stage_bytes_ >= bytes_per_buf && h_stage_[0] && h_stage_[1] &&
      stage_slot_capacity_ >= plan_.chunk_experts &&
      d_stage_slot_map_[0] && d_stage_slot_map_[1])
    return true;
  if (copy_stream_) cudaStreamSynchronize((cudaStream_t)copy_stream_);
  if (copy_backend_) ggml_backend_synchronize((ggml_backend_t)copy_backend_);

  if (bytes_per_buf == 0 ||
      bytes_per_buf > std::numeric_limits<size_t>::max() / NBUF) {
    LOG_WARN("[GGML-MOE] pinned staging size overflow -- disabled\n");
    return false;
  }
  const size_t host_bytes = bytes_per_buf * NBUF;

  // Drop the old buffers only after all streams have drained, then charge the
  // replacement before asking CUDA for pages.  The old implementation had no
  // host admission at this generic GGML/Qwen path, so a graph run could pass
  // the RAM-tier check and still allocate another multi-GB pinned staging
  // pair outside host_ram_gb.
  for (int i = 0; i < NBUF; ++i) {
    if (h_stage_[i]) {
      cudaFreeHost(h_stage_[i]);
      h_stage_[i] = nullptr;
    }
    if (d_stage_slot_map_[i]) {
      cudaFree(d_stage_slot_map_[i]);
      d_stage_slot_map_[i] = nullptr;
    }
    stage_slot_map_[i].clear();
    pending_[i] = false;
    copy_batch_started_[i] = false;
    direct_h2d_bytes_[i] = 0;
    pending_d2d_bytes_[i] = 0;
    copy_split_recorded_[i] = false;
  }
  MemoryGuard::release_host_bytes(stage_host_reservation_bytes_);
  stage_host_reservation_bytes_ = 0;
  stage_bytes_ = 0;
  stage_slot_capacity_ = 0;

  if (!MemoryGuard::try_reserve_host_bytes(host_bytes)) {
    LOG_WARN("[MEM-GUARD] GGML-MOE pinned staging refused: %.2f MiB "
             "(reserved=%.2f GiB, budget=%.2f GB)\n",
             (double)host_bytes / (1024.0 * 1024.0),
             (double)MemoryGuard::reserved_host_bytes() / 1.0e9,
             (double)MemoryGuard::host_budget_bytes() / 1.0e9);
    return false;
  }
  stage_host_reservation_bytes_ = host_bytes;

  auto cleanup_failed_stage = [&]() {
    for (int j = 0; j < NBUF; ++j) {
      if (h_stage_[j]) {
        cudaFreeHost(h_stage_[j]);
        h_stage_[j] = nullptr;
      }
      if (d_stage_slot_map_[j]) {
        cudaFree(d_stage_slot_map_[j]);
        d_stage_slot_map_[j] = nullptr;
      }
      stage_slot_map_[j].clear();
    }
    MemoryGuard::release_host_bytes(stage_host_reservation_bytes_);
    stage_host_reservation_bytes_ = 0;
    stage_bytes_ = 0;
    stage_slot_capacity_ = 0;
  };

  for (int i = 0; i < NBUF; ++i) {
    void *p = nullptr;
    if (cudaMallocHost(&p, bytes_per_buf) != cudaSuccess || !p) {
      LOG_ERROR("[GGML-MOE] pinned staging alloc failed (%.2f MB)\n",
                (double)bytes_per_buf / (1024.0 * 1024.0));
      cleanup_failed_stage();
      return false;
    }
    h_stage_[i] = (uint8_t *)p;
  }
  for (int i = 0; i < NBUF; ++i) {
    if (cudaMalloc(&d_stage_slot_map_[i],
                   (size_t)plan_.chunk_experts * sizeof(int32_t)) !=
            cudaSuccess ||
        !d_stage_slot_map_[i]) {
      LOG_ERROR("[GGML-MOE] packed stage slot-map allocation failed\n");
      cleanup_failed_stage();
      return false;
    }
    stage_slot_map_[i].reserve((size_t)plan_.chunk_experts);
  }
  stage_bytes_ = bytes_per_buf;
  stage_slot_capacity_ = plan_.chunk_experts;
  if (!copy_stream_) {
    cudaStream_t s = nullptr;
    if (cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking) != cudaSuccess) {
      LOG_ERROR("[GGML-MOE] copy stream creation failed\n");
      cleanup_failed_stage();
      return false;
    }
    copy_stream_ = s;
  }
  for (int i = 0; i < NBUF; ++i) {
    if (!backend_stream_copies_ && !ev_free_[i]) {
      cudaEvent_t e = nullptr;
      if (cudaEventCreate(&e) != cudaSuccess) {
        cleanup_failed_stage();
        return false;
      }
      ev_free_[i] = e;
    }
    if (!backend_stream_copies_ && !ev_t0_[i]) {
      cudaEvent_t e = nullptr;
      if (cudaEventCreate(&e) != cudaSuccess) {
        cleanup_failed_stage();
        return false;
      }
      ev_t0_[i] = e;
    }
    if (!backend_copy_split_event_[i]) {
      cudaEvent_t e = nullptr;
      if (cudaEventCreate(&e) != cudaSuccess) {
        cleanup_failed_stage();
        return false;
      }
      backend_copy_split_event_[i] = e;
    }
    if (backend_stream_copies_ && !backend_stage_event_[i]) {
      ggml_backend_event_t e = ggml_backend_event_new(
          ggml_backend_get_device((ggml_backend_t)backend_));
      if (!e) {
        LOG_ERROR("[GGML-MOE] GGML staging event creation failed\n");
        cleanup_failed_stage();
        return false;
      }
      backend_stage_event_[i] = e;
    }
    if (backend_stream_copies_ && !backend_copy_t0_event_[i]) {
      cudaEvent_t e0 = nullptr;
      cudaEvent_t e1 = nullptr;
      if (cudaEventCreate(&e0) != cudaSuccess ||
          cudaEventCreate(&e1) != cudaSuccess) {
        if (e0) cudaEventDestroy(e0);
        if (e1) cudaEventDestroy(e1);
        LOG_ERROR("[GGML-MOE] backend copy timing event creation failed\n");
        cleanup_failed_stage();
        return false;
      }
      backend_copy_t0_event_[i] = e0;
      backend_copy_t1_event_[i] = e1;
    }
  }
  if (backend_stream_copies_) {
    for (int i = 0; i < NWEIGHT_EVENTS; ++i) {
      if (backend_weight_done_event_[i]) continue;
      ggml_backend_event_t e = ggml_backend_event_new(
          ggml_backend_get_device((ggml_backend_t)backend_));
      if (!e) {
        LOG_ERROR("[GGML-MOE] GGML weight completion event creation failed\n");
        cleanup_failed_stage();
        return false;
      }
      backend_weight_done_event_[i] = e;
    }
  }
  return true;
}

bool GgmlMoeRunner::begin_copy_batch(int buf) {
  if (!layer_ready_ || buf < 0 || buf >= NBUF)
    return false;
  if (backend_stream_copies_) {
    if (!backend_stage_event_[buf]) return false;
    if (copy_batch_started_[buf]) return true;
    cudaSetDevice(device_);
    if (plan_.chunked && chunk_weight_recorded_[buf] &&
        chunk_weight_done_events_[buf]) {
      // The alternating chunk slot is reused only after the previous graph
      // has finished reading it. This dependency stays on the copy stream;
      // the host does not round-trip at the chunk boundary.
      if (cudaStreamWaitEvent(
              (cudaStream_t)copy_stream_,
              (cudaEvent_t)((ggml_backend_event_t)
                                chunk_weight_done_events_[buf])
                  ->context,
              0) != cudaSuccess)
        return false;
    }
    if (plan_.chunked && chunk_scatter_recorded_[buf] &&
        chunk_scatter_done_events_[buf]) {
      // The default-stream scatter reads the per-slot token snapshot.  The
      // copy stream must not reuse that slot until the previous reduction has
      // finished, even though the GGML graph itself finished earlier.
      if (cudaStreamWaitEvent(
              (cudaStream_t)copy_stream_,
              (cudaEvent_t)chunk_scatter_done_events_[buf], 0) != cudaSuccess)
        return false;
    }
    direct_h2d_bytes_[buf] = 0;
    h2d_timing_count_[buf] = 0;
    if (cudaEventRecord((cudaEvent_t)backend_copy_t0_event_[buf],
                        (cudaStream_t)copy_stream_) != cudaSuccess)
      return false;
    backend_copy_t0_[buf] = std::chrono::steady_clock::now();
    copy_batch_started_[buf] = true;
    pending_d2d_bytes_[buf] = 0;
    copy_split_recorded_[buf] = false;
    return true;
  }
  if (!copy_stream_ || !ev_t0_[buf]) return false;
  cudaSetDevice(device_);
  if (copy_batch_started_[buf]) return true;
  if (cudaEventRecord((cudaEvent_t)ev_t0_[buf],
                      (cudaStream_t)copy_stream_) != cudaSuccess)
    return false;
  backend_copy_t0_[buf] = std::chrono::steady_clock::now();
  direct_h2d_bytes_[buf] = 0;
  h2d_timing_count_[buf] = 0;
  pending_d2d_bytes_[buf] = 0;
  copy_split_recorded_[buf] = false;
  copy_batch_started_[buf] = true;
  return true;
}

bool GgmlMoeRunner::mark_copy_d2d_done(int buf) {
  if (!layer_ready_ || buf < 0 || buf >= NBUF || !copy_batch_started_[buf])
    return false;
  if (copy_split_recorded_[buf]) return true;
  cudaSetDevice(device_);
  if (!backend_copy_split_event_[buf] ||
      cudaEventRecord((cudaEvent_t)backend_copy_split_event_[buf],
                      (cudaStream_t)copy_stream_) != cudaSuccess)
    return false;
  copy_split_recorded_[buf] = true;
  return true;
}

bool GgmlMoeRunner::begin_layer(const GgmlMoeLayerPlan &plan) {
  layer_ready_ = false;
  if (plan.device_id < 0 || plan.n_experts <= 0 || plan.n_tokens <= 0 ||
      plan.topk <= 0 || plan.d_model <= 0 || plan.d_ff <= 0 ||
      plan.chunk_experts <= 0)
    return false;
  const int slots =
      plan.n_expert_slots > 0 ? plan.n_expert_slots : plan.n_experts;
  if (slots < plan.n_experts) return false;
  GgmlMoeLayerPlan normalized_plan = plan;
  if (normalized_plan.source_tokens <= 0)
    normalized_plan.source_tokens = normalized_plan.n_tokens;
  if (normalized_plan.source_tokens <= 0) return false;
  plan_ = normalized_plan;
  if (!type_supported(plan.gate_type) || !type_supported(plan.up_type) ||
      !type_supported(plan.down_type))
    return false;

  // Setting the device here (not just inside ggml) matters: every raw CUDA
  // call below -- cudaMallocHost, the stream, the events, the H2D copies --
  // targets the CURRENT device, and MoEStream legitimately leaves GPU 1
  // current between layers (attention/dense work runs there).
  cudaSetDevice(plan.device_id);
  if (!ensure_backend(plan.device_id)) return false;
  if (!ensure_h2d_timing_events()) return false;

  // Raw-copy mode has one compact weight buffer.  The device-output path
  // submits the previous MoE graph asynchronously so HC can consume its
  // result on the shared GGML stream; before this layer reuses the same
  // weight bytes, wait for that complete backend chain.  Without this handoff
  // the copy stream can overwrite an expert while the preceding layer is
  // still reading it, which first appears as corrupted routing several
  // layers later.  Backend-stream mode has independent weight buffers/events
  // and keeps its existing queued dependency path.
  if (!backend_stream_copies_ && graph_in_flight_) {
    ggml_backend_synchronize((ggml_backend_t)backend_);
    graph_in_flight_ = false;
  }

  const ggml_type gt = (ggml_type)plan.gate_type;
  const ggml_type ut = (ggml_type)plan.up_type;
  const ggml_type dt = (ggml_type)plan.down_type;

  // Row length must be a whole number of blocks or the tensor cannot be
  // expressed at all. Every real GGUF weight satisfies this; a model that
  // did not would otherwise produce a wrong-but-plausible result.
  if (plan.d_model % ggml_blck_size(gt) != 0 ||
      plan.d_model % ggml_blck_size(ut) != 0 ||
      plan.d_ff % ggml_blck_size(dt) != 0)
    return false;

  gate_exp_bytes_ = ggml_row_size(gt, plan.d_model) * (size_t)plan.d_ff;
  up_exp_bytes_ = ggml_row_size(ut, plan.d_model) * (size_t)plan.d_ff;
  down_exp_bytes_ = ggml_row_size(dt, plan.d_ff) * (size_t)plan.d_model;

  const size_t raw_expert_stride =
      gate_exp_bytes_ + up_exp_bytes_ + down_exp_bytes_;
  if (plan.external_weights &&
      (!external_buf_ || plan.chunked || !plan.external_gate ||
       !plan.external_up || !plan.external_down ||
       plan.external_expert_stride < raw_expert_stride ||
       plan.external_expert_stride % ggml_row_size(gt, plan.d_model) != 0 ||
       plan.external_expert_stride % ggml_row_size(ut, plan.d_model) != 0 ||
       plan.external_expert_stride % ggml_row_size(dt, plan.d_ff) != 0)) {
    LOG_WARN("[GGML-MOE] external resident layer rejected: missing marker, "
             "invalid shape, pointers, or row-incompatible stride (raw=%zu "
             "stride=%zu row=%zu/%zu/%zu)\n",
             raw_expert_stride, plan.external_expert_stride,
             ggml_row_size(gt, plan.d_model),
             ggml_row_size(ut, plan.d_model),
             ggml_row_size(dt, plan.d_ff));
    return false;
  }

  // Keep the compact expert axis at a grow-only high-water mark. The ids
  // passed to MUL_MAT_ID are still restricted to [0, n_experts), so slots
  // beyond this layer's routed set are never observed by the graph. This
  // avoids rebuilding the weight tensors and graph merely because a layer
  // routed a different number of unique experts.
  // Round the high-water mark in modest chunks. Exact n_used would still
  // rebuild the graph whenever one more expert is routed; rounding bounds the
  // unused compact slots to <64 while making the persistent plan useful
  // across the normal layer-to-layer variation.
  constexpr int kSlotGranularity = 64;
  const int rounded_slots =
      ((slots + kSlotGranularity - 1) / kSlotGranularity) * kSlotGranularity;
  // A direct VRAM view must preserve the layer's real expert-ID axis.  Do not
  // feed that full model axis into the compact allocator's high-water mark:
  // an external 512-expert layer is common, while the next scattered layer
  // may route only ~300 experts.  Letting the external shape raise
  // slot_capacity_ strands hundreds of MiB in the compact planar weight
  // buffer and is exactly the VRAM pressure that prevents Bee-like resident
  // layers.  Compact layers still retain their own rounded grow-only shape.
  if (!plan.external_weights)
    slot_capacity_ = std::max(slot_capacity_, rounded_slots);
  const int tensor_slots = plan.external_weights ? slots : slot_capacity_;
  bool same_weight_shape =
      t_gate_ && weight_slots_ == tensor_slots &&
      weight_d_model_ == plan.d_model && weight_d_ff_ == plan.d_ff &&
      weight_gate_type_ == plan.gate_type && weight_up_type_ == plan.up_type &&
      weight_down_type_ == plan.down_type &&
      weight_external_ == plan.external_weights;

  if (plan.external_weights) {
    // This is the Bee-style path: the graph's weight tensors borrow the
    // already resident VRAM slabs.  No compact weight buffer, staging pool,
    // D2D repack, or H2D is allocated for this layer.
    if (!same_weight_shape) {
      if (copy_stream_) cudaStreamSynchronize((cudaStream_t)copy_stream_);
      if (copy_backend_)
        ggml_backend_synchronize((ggml_backend_t)copy_backend_);
      if (backend_) ggml_backend_synchronize((ggml_backend_t)backend_);
      free_weight_tensors();

      ggml_init_params ip = {};
      ip.mem_size = ggml_tensor_overhead() * 8;
      ip.no_alloc = true;
      ggml_context * wc = ggml_init(ip);
      if (!wc) return false;
      wctx_ = wc;
      ggml_tensor * tg = ggml_new_tensor_3d(
          wc, gt, plan.d_model, plan.d_ff, tensor_slots);
      ggml_tensor * tu = ggml_new_tensor_3d(
          wc, ut, plan.d_model, plan.d_ff, tensor_slots);
      ggml_tensor * td = ggml_new_tensor_3d(
          wc, dt, plan.d_ff, plan.d_model, tensor_slots);
      if (!tg || !tu || !td) {
        free_weight_tensors();
        return false;
      }
      auto bind_external = [&](ggml_tensor * t, const void * data) {
        t->nb[2] = plan.external_expert_stride;
        t->nb[3] = t->nb[2] * (size_t)t->ne[2];
        t->data = const_cast<void *>(data);
        t->buffer = (ggml_backend_buffer_t)external_buf_;
        return ggml_backend_buffer_init_tensor(t->buffer, t) ==
               GGML_STATUS_SUCCESS;
      };
      if (!bind_external(tg, plan.external_gate) ||
          !bind_external(tu, plan.external_up) ||
          !bind_external(td, plan.external_down)) {
        free_weight_tensors();
        return false;
      }
      ggml_set_name(tg, "moe_external_gate_exps");
      ggml_set_name(tu, "moe_external_up_exps");
      ggml_set_name(td, "moe_external_down_exps");
      t_gate_ = tg;
      t_up_ = tu;
      t_down_ = td;
      weight_slots_ = tensor_slots;
      weight_d_model_ = plan.d_model;
      weight_d_ff_ = plan.d_ff;
      weight_gate_type_ = plan.gate_type;
      weight_up_type_ = plan.up_type;
      weight_down_type_ = plan.down_type;
    } else {
      // The GGML graph stores the tensor descriptors, while the cache may
      // have placed this layer at a different pool offset after a request.
      // Refresh the borrowed pointers and stride before submitting the graph.
      auto rebind_external = [&](void * opaque, const void * data) {
        ggml_tensor * t = (ggml_tensor *)opaque;
        t->nb[2] = plan.external_expert_stride;
        t->nb[3] = t->nb[2] * (size_t)t->ne[2];
        t->data = const_cast<void *>(data);
        t->buffer = (ggml_backend_buffer_t)external_buf_;
        return ggml_backend_buffer_init_tensor(t->buffer, t) ==
               GGML_STATUS_SUCCESS;
      };
      if (!rebind_external(t_gate_, plan.external_gate) ||
          !rebind_external(t_up_, plan.external_up) ||
          !rebind_external(t_down_, plan.external_down))
        return false;
    }
    weight_external_ = true;
    weight_expert_stride_ = plan.external_expert_stride;
    weight_raw_expert_stride_ = raw_expert_stride;
    weight_tail_padding_ = 0;
    weight_gate_plane_offset_ = 0;
    weight_up_plane_offset_ = gate_exp_bytes_;
    weight_down_plane_offset_ = gate_exp_bytes_ + up_exp_bytes_;
    packed_base_ = nullptr;
    w_gate_base_ = (uint8_t *)plan.external_gate;
    w_up_base_ = (uint8_t *)plan.external_up;
    w_down_base_ = (uint8_t *)plan.external_down;
    plan_.n_expert_slots = tensor_slots;
    if (!ensure_graph()) return false;
    layer_ready_ = true;
    static bool logged_external = false;
    if (!logged_external) {
      LOG_INFO("[GGML-MOE] external resident expert slabs active: "
               "zero-repack MUL_MAT_ID weights\n");
      logged_external = true;
    }
    return true;
  }

  const size_t align = std::max<size_t>(
      256, ggml_backend_buft_get_alignment(ggml_backend_cuda_buffer_type(device_)));
  // Store all three projections for one transfer record next to each other.
  // The GGML tensors themselves remain planar; this record is only the
  // H2D/D2D landing format used before the device-local unpack kernel.
  // Keep each transfer record page-aligned. This makes the staged source
  // records and the packed device landing buffer directly suitable for DMA,
  // while the RAM cache may still expose the original tight slab for direct
  // one-expert copies. The extra <=4095 bytes per expert is tiny compared
  // with these projection records and avoids the invalid overlapping GGML
  // tensor layout that the first experiment used.
  const size_t expert_stride = align_up(raw_expert_stride, 4096);
  const size_t gate_tail = cuda_row_pad_bytes(gt, plan.d_model);
  const size_t up_tail = cuda_row_pad_bytes(ut, plan.d_model);
  const size_t down_tail = cuda_row_pad_bytes(dt, plan.d_ff);
  const size_t gate_plane_bytes =
      gate_exp_bytes_ * (size_t)tensor_slots + gate_tail;
  const size_t up_plane_bytes = up_exp_bytes_ * (size_t)tensor_slots + up_tail;
  const size_t down_plane_bytes =
      down_exp_bytes_ * (size_t)tensor_slots + down_tail;
  const size_t up_plane_offset = align_up(gate_plane_bytes, align);
  const size_t down_plane_offset =
      align_up(up_plane_offset + up_plane_bytes, align);
  const size_t need =
      align_up(down_plane_offset + down_plane_bytes, align);
  // The packed buffer is only a transfer staging area.  The unpack kernel
  // consumes one chunk at a time, so sizing it for every routed slot wastes
  // hundreds of MiB and competes directly with the expert cache for VRAM.
  // The planar GGML weight buffer still covers the complete routed set.
  const size_t packed_need = expert_stride * (size_t)plan.chunk_experts;

  if (!same_weight_shape) {
    // A shape change invalidates both the graph and the ggml weight tensor
    // structs. Synchronize only the streams that can reference them; unlike
    // the old path this is not a device-wide barrier on every layer.
    if (copy_stream_) cudaStreamSynchronize((cudaStream_t)copy_stream_);
    if (copy_backend_)
      ggml_backend_synchronize((ggml_backend_t)copy_backend_);
    if (backend_) ggml_backend_synchronize((ggml_backend_t)backend_);
    // 2026-09-10: ANSWERED, keeping the counter (cheap, fires ~2x/session).
    // Hypothesis was that alternating gate/up/down quant TYPES between
    // layers (UD's mixed quantization -- real per-layer graph= spikes of
    // 11-145ms looked correlated with type changes) forces repeated,
    // avoidable weight-tensor rebuilds despite instance_for()'s per-
    // signature runner pool. REFUTED by direct measurement on a real
    // 488-token prefill: only 2 rebuilds fired total across all 48 layers,
    // both had_tensors=0 (first-ever build for that signature, not churn).
    // instance_for() caches correctly; the graph= spikes are one-time JIT/
    // cudaMalloc cost (2 occurrences), not a per-layer recurring tax -- NOT
    // part of the still-open ~115ms/layer H2D+wait overhead floor.
    LOG_INFO("[GGML-MOE-REBUILD] gate_type=%u up_type=%u down_type=%u "
             "old_slots=%d new_slots=%d had_tensors=%d\n",
             weight_gate_type_, weight_up_type_, weight_down_type_,
             weight_slots_, tensor_slots, t_gate_ ? 1 : 0);
    free_weight_tensors();
  }

  // Grow-only capacity prevents reallocations as successive layers expose a
  // larger routed set. The compact buffer is protected by a device completion
  // event before the next layer overwrites it.
  if (!wbuf_[0] || wbuf_bytes_ < need || packed_wbuf_bytes_ < packed_need) {
    if (!ensure_weight_buffer(need, packed_need)) return false;
  }

  const int next_weight_buffer =
      backend_stream_copies_ ? (weight_buffer_index_ + 1) % NWEIGHT_BUFS : 0;
  if (backend_stream_copies_ &&
      backend_weight_buffer_in_use_[next_weight_buffer] &&
      backend_weight_last_event_[next_weight_buffer]) {
    // The compact weight buffer is safe to overwrite only after the preceding
    // MoE graph has finished using it. The actual H2D path below is a raw CUDA
    // stream, so put the dependency on that stream directly. Waiting on the
    // separate GGML copy backend would leave the raw stream unordered.
    if (copy_stream_) {
      auto *event = (ggml_backend_event_t)
          backend_weight_last_event_[next_weight_buffer];
      cudaSetDevice(device_);
      cudaStreamWaitEvent((cudaStream_t)copy_stream_,
                          (cudaEvent_t)event->context, 0);
    }
  }
  weight_buffer_index_ = next_weight_buffer;
  const bool rebuild_weights = !same_weight_shape || !t_gate_;
  uint8_t *const base = (uint8_t *)ggml_backend_buffer_get_base(
      (ggml_backend_buffer_t)wbuf_[weight_buffer_index_]);
  // These pointers must follow the active buffer even when the tensor geometry
  // is unchanged and only the active transfer buffer changes.
  w_gate_base_ = base;
  w_up_base_ = base + up_plane_offset;
  w_down_base_ = base + down_plane_offset;
  packed_base_ = (uint8_t *)packed_wbuf_[weight_buffer_index_];
  weight_expert_stride_ = expert_stride;
  weight_raw_expert_stride_ = raw_expert_stride;
  weight_tail_padding_ = std::max({gate_tail, up_tail, down_tail});
  weight_gate_plane_offset_ = 0;
  weight_up_plane_offset_ = up_plane_offset;
  weight_down_plane_offset_ = down_plane_offset;
  if (rebuild_weights) {
    // Zero the tail padding AND every slot past this layer's routed set. The
    // padding is read by MMQ's k-tiling; unused slots are not referenced by
    // `ids`, but initializing both keeps allocator contents deterministic.
    const size_t tail = 0;
    // In backend-stream mode the previous graph may still be reading this
    // buffer. The selected IDs never reference slots beyond n_experts, so
    // the tail is dead data; clearing it with the default stream would race
    // the queued graph. The copy stream is fenced below before new weights
    // are written. The synchronous raw path retains deterministic tail
    // clearing.
    if (!backend_stream_copies_) {
      if (gate_tail) cudaMemset(base + gate_exp_bytes_ * tensor_slots, 0, gate_tail);
      if (up_tail) cudaMemset(w_up_base_ + up_exp_bytes_ * tensor_slots, 0, up_tail);
      if (down_tail)
        cudaMemset(w_down_base_ + down_exp_bytes_ * tensor_slots, 0, down_tail);
    }

    ggml_init_params ip = {};
    ip.mem_size = ggml_tensor_overhead() * 8;
    ip.mem_buffer = nullptr;
    ip.no_alloc = true;
    ggml_context *wc = ggml_init(ip);
    if (!wc) return false;
    wctx_ = wc;

    ggml_tensor *tg =
        ggml_new_tensor_3d(wc, gt, plan.d_model, plan.d_ff, tensor_slots);
    ggml_tensor *tu =
        ggml_new_tensor_3d(wc, ut, plan.d_model, plan.d_ff, tensor_slots);
    ggml_tensor *td =
        ggml_new_tensor_3d(wc, dt, plan.d_ff, plan.d_model, tensor_slots);
    if (!tg || !tu || !td) return false;
    // These are ordinary contiguous planar tensors. Do not override nb[2]:
    // GGML CUDA's quantized MUL_MAT_ID kernels and its allocator assumptions
    // have been validated for this geometry.
    auto bind_weight = [&](ggml_tensor *t, void *data) {
      t->buffer = (ggml_backend_buffer_t)wbuf_[weight_buffer_index_];
      t->data = data;
    };
    bind_weight(tg, w_gate_base_);
    bind_weight(tu, w_up_base_);
    bind_weight(td, w_down_base_);
    ggml_set_name(tg, "moe_gate_exps");
    ggml_set_name(tu, "moe_up_exps");
    ggml_set_name(td, "moe_down_exps");
    t_gate_ = tg;
    t_up_ = tu;
    t_down_ = td;
    weight_slots_ = tensor_slots;
    weight_d_model_ = plan.d_model;
    weight_d_ff_ = plan.d_ff;
    weight_gate_type_ = plan.gate_type;
    weight_up_type_ = plan.up_type;
    weight_down_type_ = plan.down_type;
  } else {
    // Keep the persistent graph nodes, but move their storage to the newly
    // selected compact buffer. The CUDA buffer init hook is unnecessary here:
    // both buffers have identical geometry and were initialized when first
    // allocated, while calling ggml_backend_tensor_alloc again would assert
    // because these tensor structs are already allocated.
    ggml_tensor *tg = (ggml_tensor *)t_gate_;
    ggml_tensor *tu = (ggml_tensor *)t_up_;
    ggml_tensor *td = (ggml_tensor *)t_down_;
    tg->buffer = (ggml_backend_buffer_t)wbuf_[weight_buffer_index_];
    tu->buffer = (ggml_backend_buffer_t)wbuf_[weight_buffer_index_];
    td->buffer = (ggml_backend_buffer_t)wbuf_[weight_buffer_index_];
    tg->data = w_gate_base_;
    tu->data = w_up_base_;
    td->data = w_down_base_;

  }
  weight_external_ = false;

  const size_t per_buf = weight_expert_stride_ * (size_t)plan.chunk_experts;
  if (!ensure_stage_buffers(per_buf)) return false;

  // GGML-CUDA's quantized MMQ kernels may read the MATRIX_ROW_PADDING bytes
  // after the final row of a tensor.  Those bytes are outside ggml_nbytes()
  // and are therefore not filled by the expert copies themselves, but they
  // must still be deterministic.  In backend-stream mode this clear belongs
  // on the same copy stream as the H2D/unpack work: the selected alternating
  // weight buffer has already been fenced above, and the graph submission
  // waits for this stream's staging event below.  Clearing on the default
  // stream here would race a previous graph using the same buffer.
  auto clear_quant_tail = [&](uint8_t *ptr, size_t offset, size_t bytes) {
    if (!bytes) return true;
    if (backend_stream_copies_ && copy_stream_) {
      return cudaMemsetAsync(ptr + offset, 0, bytes,
                             (cudaStream_t)copy_stream_) == cudaSuccess;
    }
    return cudaMemset(ptr + offset, 0, bytes) == cudaSuccess;
  };
  if (!clear_quant_tail(w_gate_base_, gate_exp_bytes_ * (size_t)tensor_slots,
                        gate_tail) ||
      !clear_quant_tail(w_up_base_, up_exp_bytes_ * (size_t)tensor_slots,
                        up_tail) ||
      !clear_quant_tail(w_down_base_, down_exp_bytes_ * (size_t)tensor_slots,
                        down_tail)) {
    LOG_ERROR("[GGML-MOE] quantized weight tail clear failed\n");
    return false;
  }

  plan_.n_expert_slots = tensor_slots;
  if (!ensure_graph()) return false;
  if (plan_.chunked) {
    if (!ensure_chunk_state()) return false;
  } else if (chunk_has_submission_) {
    // A mode switch can only happen at a shape/topology boundary. Drain the
    // shared backend before releasing the raw chunk accumulator and its GGML
    // publication tensor.
    ggml_backend_synchronize((ggml_backend_t)backend_);
    free_chunk_state();
  }
  layer_ready_ = true;
  return true;
}

bool GgmlMoeRunner::ensure_graph() {
  if (!t_gate_ || !t_up_ || !t_down_ || !backend_) return false;

  const bool matches =
      graph_ctx_ && graph_ && graph_tokens_ == plan_.n_tokens &&
      graph_source_tokens_ == plan_.source_tokens &&
      graph_topk_ == plan_.topk && graph_d_model_ == plan_.d_model &&
      graph_d_ff_ == plan_.d_ff && graph_slots_ == plan_.n_expert_slots &&
      graph_gate_type_ == plan_.gate_type &&
      graph_up_type_ == plan_.up_type &&
      graph_down_type_ == plan_.down_type &&
      graph_chunked_ == plan_.chunked;
  if (matches) return true;

  // A layer may change the compact-slot count or quantization type, forcing
  // this persistent graph to be rebuilt. Its allocator workspace cannot be
  // reset or freed while the previous asynchronous graph is still running.
  // The MoE result is also consumed by HC post-combine on this shared
  // backend, so the MoE completion event alone is insufficient here: it can
  // fire before the downstream D2D handoff has finished. Drain the shared
  // backend at this topology reuse point; normal same-shape layers retain
  // the existing queued graph and do not pay this boundary.
  invalidate_graph();  // drains in-flight work and frees graph_ctx_

  // Real crash, found on real hardware (2026-09-09, Qwen4Exp MTP bring-up):
  // "ggml_new_object: not enough space in the context's memory pool
  // (needed 132144, available 131760)" -- the SAME two numbers on every
  // occurrence, regardless of how many prior calls had already run
  // (confirmed by testing invalidate_graph() between every caller-level
  // forward pass, which did NOT fix it) -- meaning this is a deterministic
  // per-shape shortfall in a FRESHLY REBUILT context, not cross-call
  // accumulation: ggml_graph_overhead()'s own fixed cost (hash table +
  // node/leaf arrays, sized for ggml_new_graph()'s default max-node count)
  // plus this function's own ~20-30 tensor objects (more with a large
  // topk -- the accumulation loop below allocates one view+add pair PER
  // selected expert) leaves the 128-tensor-slot term with essentially no
  // slack, and run_layer_device_impl's OWN extra ggml_view_2d call (the
  // one that actually aborts) needs just one more slot than remains.
  // 256 gives a real, generous margin (the added ~49 KB of host-side
  // metadata is irrelevant against this process's GB-scale budgets)
  // instead of chasing the exact minimum.
  ggml_init_params ip = {};
  ip.mem_size = ggml_tensor_overhead() * 256 + ggml_graph_overhead();
  ip.mem_buffer = nullptr;
  ip.no_alloc = true;
  ggml_context *ctx = ggml_init(ip);
  if (!ctx) return false;

  ggml_cgraph *gf = ggml_new_graph(ctx);
  if (!gf) {
    ggml_free(ctx);
    return false;
  }

  const int S = plan_.n_tokens;
  const int K = plan_.topk;
  const int D = plan_.d_model;
  // Keep the ordinary graph's input storage two-dimensional. In chunk mode
  // the same flat storage is reshaped to [D,source_tokens,1], gathered by
  // token_ids, and then presented to MUL_MAT_ID as [D,1,pairs].
  ggml_tensor *t_x_flat = ggml_new_tensor_2d(
      ctx, GGML_TYPE_F32, D, plan_.source_tokens);
  // MUL_MAT_ID's b tensor is [D, n_expert_used, n_tokens].  The ordinary
  // top-k graph therefore needs [D,1,S], while the chunk graph gathers rows
  // from [D,S,1] and then presents the gathered result as [D,1,pairs].
  ggml_tensor *t_x = nullptr;
  if (t_x_flat) {
    t_x = plan_.chunked
              ? ggml_reshape_3d(ctx, t_x_flat, D, plan_.source_tokens, 1)
              : ggml_reshape_3d(ctx, t_x_flat, D, 1, plan_.n_tokens);
  }
  ggml_tensor *t_ids = nullptr;
  ggml_tensor *t_w = nullptr;
  ggml_tensor *t_token_ids = nullptr;
  ggml_tensor *expert_x = nullptr;
  if (plan_.chunked) {
    t_ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 1, S);
    t_w = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, 1, S);
    // GET_ROWS uses [pair_count,1] indices for a [D,source_tokens,1]
    // activation. Reshape its [D,pairs,1] result to the [D,1,pairs] layout
    // expected by MUL_MAT_ID.
    t_token_ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, S, 1);
    ggml_tensor *rows = t_token_ids ? ggml_get_rows(ctx, t_x, t_token_ids)
                                    : nullptr;
    expert_x = rows ? ggml_reshape_3d(ctx, rows, D, 1, S) : nullptr;
  } else {
    t_ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, K, S);
    t_w = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, K, S);
    expert_x = t_x;
  }
  if (!t_x || !t_ids || !t_w || !expert_x ||
      (plan_.chunked && !t_token_ids)) {
    ggml_free(ctx);
    return false;
  }
  ggml_set_input(t_x_flat);
  ggml_set_input(t_ids);
  ggml_set_input(t_w);
  if (t_token_ids) ggml_set_input(t_token_ids);
  ggml_set_name(t_x_flat, plan_.chunked ? "moe_chunk_x" : "moe_x");
  ggml_set_name(t_ids, "moe_ids");
  ggml_set_name(t_w, "moe_w");
  if (t_token_ids) ggml_set_name(t_token_ids, "moe_token_ids");

  ggml_tensor *g =
      ggml_mul_mat_id(ctx, (ggml_tensor *)t_gate_, expert_x, t_ids);
  ggml_tensor *u =
      ggml_mul_mat_id(ctx, (ggml_tensor *)t_up_, expert_x, t_ids);
  ggml_tensor *a = ggml_swiglu_split(ctx, g, u);
  ggml_tensor *e =
      ggml_mul_mat_id(ctx, (ggml_tensor *)t_down_, a, t_ids);
  e = ggml_mul(ctx, e, t_w);

  ggml_tensor *moe = nullptr;
  if (plan_.chunked) {
    // The chunk output is [D,1,pairs]. A 2-D view gives the CUDA scatter
    // kernel the same contiguous [pairs,D] row layout used by the legacy
    // expert accumulator.
    moe = ggml_view_2d(ctx, e, D, S, e->nb[2], 0);
  } else {
    for (int k = 0; k < K; ++k) {
      ggml_tensor *v =
          ggml_view_2d(ctx, e, D, S, e->nb[2], (size_t)k * e->nb[1]);
      moe = (k == 0) ? v : ggml_add(ctx, moe, v);
    }
    if (K == 1) moe = ggml_cont(ctx, moe);
  }
  if (!moe) {
    ggml_free(ctx);
    return false;
  }
  ggml_set_output(moe);
  ggml_set_name(moe, "moe_out");
  ggml_build_forward_expand(gf, moe);

  if (!ggml_gallocr_alloc_graph((ggml_gallocr_t)galloc_, gf)) {
    LOG_ERROR("[GGML-MOE] graph allocation failed (S=%d topk=%d experts=%d "
              "slots=%d)\n",
              S, K, plan_.n_experts, plan_.n_expert_slots);
    ggml_free(ctx);
    return false;
  }

  const size_t graph_bytes =
      ggml_gallocr_get_buffer_size((ggml_gallocr_t)galloc_, 0);
  if (graph_bytes != graph_workspace_bytes_) {
    graph_workspace_bytes_ = graph_bytes;
    LOG_INFO("[GGML-MOE-MEM] graph workspace: %.2f MiB (S=%d slots=%d)\n",
             (double)graph_bytes / (1024.0 * 1024.0), S,
             plan_.n_expert_slots);
  }

  graph_ctx_ = ctx;
  graph_ = gf;
  graph_x_ = t_x_flat;
  graph_ids_ = t_ids;
  graph_w_ = t_w;
  graph_token_ids_ = t_token_ids;
  graph_gate_out_ = g;
  graph_up_out_ = u;
  graph_act_out_ = a;
  graph_down_out_ = e;
  graph_out_ = moe;
  graph_tokens_ = S;
  graph_source_tokens_ = plan_.source_tokens;
  graph_topk_ = K;
  graph_d_model_ = D;
  graph_d_ff_ = plan_.d_ff;
  graph_slots_ = plan_.n_expert_slots;
  graph_gate_type_ = plan_.gate_type;
  graph_up_type_ = plan_.up_type;
  graph_down_type_ = plan_.down_type;
  graph_chunked_ = plan_.chunked;
  LOG_INFO("[GGML-MOE] persistent graph: S=%d topk=%d D=%d d_ff=%d "
           "experts=%d slots=%d types=%u/%u/%u\n",
           S, K, D, plan_.d_ff, plan_.n_experts, plan_.n_expert_slots,
           (unsigned)plan_.gate_type, (unsigned)plan_.up_type,
           (unsigned)plan_.down_type);
  return true;
}

GgmlMoeStageSlot GgmlMoeRunner::stage_slot(int buf, int slot) const {
  GgmlMoeStageSlot s;
  if (!layer_ready_ || buf < 0 || buf >= NBUF || slot < 0 ||
      slot >= plan_.chunk_experts || !h_stage_[buf])
    return s;
  uint8_t *base = h_stage_[buf];
  const size_t stride = weight_expert_stride_;
  uint8_t *expert = base + (size_t)slot * stride;
  s.gate = expert;
  s.up = expert + gate_exp_bytes_;
  s.down = s.up + up_exp_bytes_;
  return s;
}

void GgmlMoeRunner::account_copy_buffer(int buf) {
  if (buf < 0 || buf >= NBUF || !pending_[buf]) return;

  cudaSetDevice(device_);
  float d2d_ms = 0.0f;
  float h2d_ms = 0.0f;
  bool timed = false;
  if (copy_split_recorded_[buf]) {
    if (backend_stream_copies_) {
      timed = cudaEventElapsedTime(
                  &d2d_ms, (cudaEvent_t)backend_copy_t0_event_[buf],
                  (cudaEvent_t)backend_copy_split_event_[buf]) == cudaSuccess &&
              cudaEventElapsedTime(
                  &h2d_ms, (cudaEvent_t)backend_copy_split_event_[buf],
                  (cudaEvent_t)backend_copy_t1_event_[buf]) == cudaSuccess;
    } else {
      timed = cudaEventElapsedTime(
                  &d2d_ms, (cudaEvent_t)ev_t0_[buf],
                  (cudaEvent_t)backend_copy_split_event_[buf]) == cudaSuccess &&
              cudaEventElapsedTime(
                  &h2d_ms, (cudaEvent_t)backend_copy_split_event_[buf],
                  (cudaEvent_t)ev_free_[buf]) == cudaSuccess;
    }
  }
  if (!timed) {
    float total_ms = 0.0f;
    if (backend_stream_copies_) {
      timed = cudaEventElapsedTime(
                  &total_ms, (cudaEvent_t)backend_copy_t0_event_[buf],
                  (cudaEvent_t)backend_copy_t1_event_[buf]) == cudaSuccess;
    } else {
      timed = cudaEventElapsedTime(&total_ms, (cudaEvent_t)ev_t0_[buf],
                                   (cudaEvent_t)ev_free_[buf]) == cudaSuccess;
    }
    // Without a valid split event, retain a conservative total in the H2D
    // bucket. This is a diagnostics fallback only; normal runs record the
    // split before the first RAM upload.
    h2d_ms = total_ms;
  }
  if (timed) {
    d2d_seconds_ += (double)d2d_ms / 1000.0;
    h2d_seconds_ += (double)h2d_ms / 1000.0;
  } else {
    h2d_seconds_ += std::chrono::duration<double>(
                         std::chrono::steady_clock::now() -
                         backend_copy_t0_[buf])
                         .count();
  }
  if (trace_pure_h2d()) {
    for (int i = 0; i < h2d_timing_count_[buf]; ++i) {
      float dma_ms = 0.0f;
      if (cudaEventElapsedTime(
              &dma_ms, (cudaEvent_t)h2d_t0_event_[buf][i],
              (cudaEvent_t)h2d_t1_event_[buf][i]) == cudaSuccess)
        h2d_dma_seconds_ += (double)dma_ms / 1000.0;
    }
  }
  d2d_bytes_ += pending_d2d_bytes_[buf];
  h2d_bytes_ += pending_bytes_[buf];
  pending_[buf] = false;
  pending_bytes_[buf] = 0;
  pending_d2d_bytes_[buf] = 0;
  copy_batch_started_[buf] = false;
  copy_split_recorded_[buf] = false;
  h2d_timing_count_[buf] = 0;
  direct_h2d_bytes_[buf] = 0;
}

void GgmlMoeRunner::wait_stage_free(int buf) {
  if (buf < 0 || buf >= NBUF || !pending_[buf]) return;
  if (backend_stream_copies_) {
    cudaSetDevice(device_);
    ggml_backend_event_synchronize(
        (ggml_backend_event_t)backend_stage_event_[buf]);
    // This is the copy-stream service interval. It excludes the host's wait
    // for a reusable double-buffer slot and any time spent after the copy was
    // queued while the compute stream was running.
    account_copy_buffer(buf);
    return;
  }
  // MUST pin the device. A CUDA event belongs to the device that was current
  // when it was recorded, and waiting on it from a DIFFERENT current device
  // does not reliably complete -- it hangs. This is reachable in practice:
  // GpuTopology::finish_expert_hit() does cudaSetDevice(loc.gpu_id) and never
  // restores it (see its own comment -- the set is deliberate, the lack of a
  // restore is just how it is), so a single find_expert() hit on an expert
  // resident on the OTHER vault GPU silently leaves that GPU current, and the
  // next wait here blocks forever. MEASURED 2026-09-05: adding the VRAM-reuse
  // residency check introduced exactly that call and hung layer 0 at the
  // first chunk with a pending event, spinning in userspace at 0% SM
  // utilisation. Same class as PrefillExpertStream::pin_device().
  cudaSetDevice(device_);
  cudaEventSynchronize((cudaEvent_t)ev_free_[buf]);
  account_copy_buffer(buf);
}

bool GgmlMoeRunner::push_chunk(int buf, int first_expert, int count,
                               const bool *staged) {
  if (!layer_ready_ || buf < 0 || buf >= NBUF || count <= 0) return false;
  if (first_expert < 0 || first_expert + count > plan_.n_experts) return false;
  if (count > plan_.chunk_experts) return false;
  if (!begin_copy_batch(buf)) return false;
  if (!mark_copy_d2d_done(buf)) return false;
  cudaSetDevice(device_);

  size_t total = 0;
  cudaError_t e = cudaSuccess;
  if (use_packed_transfer()) {
    if (!packed_base_ || !d_stage_slot_map_[buf]) return false;

    // Cache hits leave holes in the chunk.  Compact the staged records in
    // place before the H2D: one chunk then becomes one contiguous DMA even
    // when its VRAM-resident experts are interleaved with RAM misses.  The
    // device map restores the original expert slot after the transfer, so
    // the GGML planes keep their normal global expert-major layout.
    auto &slot_map = stage_slot_map_[buf];
    slot_map.clear();
    for (int i = 0; i < count; ++i) {
      if (staged && !staged[i]) continue;
      const GgmlMoeStageSlot src_slot = stage_slot(buf, i);
      if (!src_slot.gate) return false;
      const size_t compact = slot_map.size();
      const GgmlMoeStageSlot dst_slot = stage_slot(buf, (int)compact);
      if (!dst_slot.gate) return false;
      if (compact != (size_t)i) {
        std::memmove(dst_slot.gate, src_slot.gate,
                     weight_expert_stride_);
      }
      slot_map.push_back(first_expert + i);
    }

    const int compact_count = (int)slot_map.size();
    if (debug_check_device_output()) {
      int slot31_compact = -1;
      int slot63_compact = -1;
      for (int compact = 0; compact < compact_count; ++compact) {
        if (slot_map[(size_t)compact] == 31) {
          slot31_compact = compact;
        }
        if (slot_map[(size_t)compact] == 63) {
          slot63_compact = compact;
        }
      }
      LOG_INFO("[GGML-MOE-DEBUG] compact transfer buf=%d first=%d count=%d "
               "staged=%d slot31_compact=%d slot63_compact=%d\n",
               buf, first_expert, count, compact_count, slot31_compact,
               slot63_compact);
    }
    if (compact_count > 0) {
      const size_t bytes = weight_expert_stride_ * (size_t)compact_count;
      int h2d_timing_slot = -1;
      if (!begin_h2d_timing(buf, h2d_timing_slot)) return false;
      e = cudaMemcpyAsync(
          packed_base_, h_stage_[buf], bytes, cudaMemcpyHostToDevice,
          (cudaStream_t)copy_stream_);
      if (!end_h2d_timing(buf, h2d_timing_slot)) return false;
      if (e == cudaSuccess) {
        e = cudaMemcpyAsync(
            d_stage_slot_map_[buf], slot_map.data(),
            (size_t)compact_count * sizeof(int32_t), cudaMemcpyHostToDevice,
            (cudaStream_t)copy_stream_);
      }
      if (e == cudaSuccess) {
        e = launch_ggml_moe_unpack_scatter(
            packed_base_,
            w_gate_base_, w_up_base_, w_down_base_,
            (const int32_t *)d_stage_slot_map_[buf], compact_count,
            weight_expert_stride_, gate_exp_bytes_, up_exp_bytes_,
            down_exp_bytes_, gate_exp_bytes_, up_exp_bytes_, down_exp_bytes_,
            (cudaStream_t)copy_stream_);
      }
      if (e == cudaSuccess && debug_check_device_output()) {
        // Diagnostic only: compare the edge rows immediately after the
        // device-local unpack, before MUL_MAT_ID consumes them.
        e = cudaStreamSynchronize((cudaStream_t)copy_stream_);
        if (e == cudaSuccess) {
          const size_t gate_row = ggml_row_size(
              (ggml_type)plan_.gate_type, plan_.d_model);
          const size_t up_row = ggml_row_size(
              (ggml_type)plan_.up_type, plan_.d_model);
          const size_t down_row = ggml_row_size(
              (ggml_type)plan_.down_type, plan_.d_ff);
          std::vector<uint8_t> device_row;
          auto verify_row = [&](const char *name, const uint8_t *host,
                                const uint8_t *device, size_t row_bytes,
                                int slot, int compact) {
            if (row_bytes == 0) return;
            device_row.resize(row_bytes);
            const cudaError_t ce = cudaMemcpy(
                device_row.data(), device, row_bytes,
                cudaMemcpyDeviceToHost);
            const bool same = ce == cudaSuccess &&
                              std::memcmp(host, device_row.data(), row_bytes) ==
                                  0;
            if (!same) {
              LOG_INFO("[GGML-MOE-DEBUG] unpack mismatch chunk_buf=%d "
                       "compact=%d slot=%d plane=%s bytes=%zu copy=%s\n",
                       buf, compact, slot, name, row_bytes,
                       ce == cudaSuccess ? "ok" : cudaGetErrorString(ce));
            }
          };
          for (int compact = 0; compact < compact_count; ++compact) {
            const int slot = slot_map[(size_t)compact];
            const uint8_t *src_record =
                h_stage_[buf] + (size_t)compact * weight_expert_stride_;
            verify_row("gate-first", src_record,
                       w_gate_base_ + gate_exp_bytes_ * (size_t)slot,
                       gate_row, slot, compact);
            verify_row("gate-last",
                       src_record + gate_exp_bytes_ - gate_row,
                       w_gate_base_ + gate_exp_bytes_ * (size_t)slot +
                           gate_exp_bytes_ - gate_row,
                       gate_row, slot, compact);
            verify_row("up-first", src_record + gate_exp_bytes_,
                       w_up_base_ + up_exp_bytes_ * (size_t)slot, up_row, slot,
                       compact);
            verify_row("up-last",
                       src_record + gate_exp_bytes_ + up_exp_bytes_ - up_row,
                       w_up_base_ + up_exp_bytes_ * (size_t)slot +
                           up_exp_bytes_ - up_row,
                       up_row, slot, compact);
            verify_row("down-first", src_record + gate_exp_bytes_ +
                           up_exp_bytes_,
                       w_down_base_ + down_exp_bytes_ * (size_t)slot,
                       down_row, slot, compact);
            verify_row("down-last",
                       src_record + gate_exp_bytes_ + up_exp_bytes_ +
                           down_exp_bytes_ - down_row,
                       w_down_base_ + down_exp_bytes_ * (size_t)slot +
                           down_exp_bytes_ - down_row,
                       down_row, slot, compact);
          }
        }
      }
      // The route-map upload is small but is still a real H2D operation; keep
      // it in the source-specific accounting instead of silently dropping it.
      total += bytes + (size_t)compact_count * sizeof(int32_t);
    }
  } else {
    // Walk maximal runs of consecutive staged experts. The reference path
    // keeps the established planar layout and deliberately does not use the
    // compact packed DMA.
    int i = 0;
    while (i < count && e == cudaSuccess) {
      if (staged && !staged[i]) {
        ++i;
        continue;
      }
      int j = i;
      while (j < count && (!staged || staged[j])) ++j;
      const GgmlMoeStageSlot s0 = stage_slot(buf, i);
      if (!s0.gate) return false;
      // Correctness reference: preserve the established planar GGML layout
      // and copy each projection through its own pitched host record. This
      // is intentionally slower, but removes the new device unpack kernel
      // from the experiment while keeping the graph and all routing identical.
      const size_t rows = (size_t)(j - i);
      e = cudaMemcpy2DAsync(
          w_gate_base_ + gate_exp_bytes_ * (size_t)(first_expert + i),
          gate_exp_bytes_, s0.gate, weight_expert_stride_, gate_exp_bytes_,
          rows, cudaMemcpyHostToDevice, (cudaStream_t)copy_stream_);
      if (e == cudaSuccess) {
        e = cudaMemcpy2DAsync(
            w_up_base_ + up_exp_bytes_ * (size_t)(first_expert + i),
            up_exp_bytes_, s0.up, weight_expert_stride_, up_exp_bytes_, rows,
            cudaMemcpyHostToDevice, (cudaStream_t)copy_stream_);
      }
      if (e == cudaSuccess) {
        e = cudaMemcpy2DAsync(
            w_down_base_ + down_exp_bytes_ * (size_t)(first_expert + i),
            down_exp_bytes_, s0.down, weight_expert_stride_, down_exp_bytes_,
            rows, cudaMemcpyHostToDevice, (cudaStream_t)copy_stream_);
      }
      total += (gate_exp_bytes_ + up_exp_bytes_ + down_exp_bytes_) * rows;
      i = j;
    }
  }
  if (e != cudaSuccess) {
    LOG_ERROR("[GGML-MOE] weight H2D failed: %s\n", cudaGetErrorString(e));
    return false;
  }
  // Recorded even for an all-resident chunk (total == 0): the event is what
  // wait_stage_free() polls before letting the caller refill this buffer, and
  // skipping it would leave a stale event from two chunks ago as the gate.
  if (backend_stream_copies_) {
    // backend_stage_event_ is a GGML event wrapper, but its CUDA event can be
    // recorded on the raw copy stream. The compute GGML stream waits on this
    // event immediately before graph submission.
    if (cudaEventRecord((cudaEvent_t)backend_copy_t1_event_[buf],
                        (cudaStream_t)copy_stream_) != cudaSuccess)
      return false;
    auto *event = (ggml_backend_event_t)backend_stage_event_[buf];
    if (cudaEventRecord((cudaEvent_t)event->context,
                        (cudaStream_t)copy_stream_) != cudaSuccess)
      return false;
  } else if (cudaEventRecord((cudaEvent_t)ev_free_[buf],
                             (cudaStream_t)copy_stream_) != cudaSuccess) {
    return false;
  }
  pending_[buf] = true;
  pending_bytes_[buf] = total + direct_h2d_bytes_[buf];
  direct_h2d_bytes_[buf] = 0;
  copy_batch_started_[buf] = false;
  return true;
}

bool GgmlMoeRunner::copy_expert_h2d(int buf, int expert_index,
                                    const void *src_slab) {
  if (!layer_ready_ || buf < 0 || buf >= NBUF || !src_slab ||
      expert_index < 0 || expert_index >= plan_.n_experts)
    return false;
  if (!begin_copy_batch(buf)) return false;

  cudaSetDevice(device_);
  const uint8_t *src = (const uint8_t *)src_slab;
  cudaStream_t st = (cudaStream_t)copy_stream_;
  if (!st) return false;
  cudaError_t e = cudaSuccess;
  int h2d_timing_slot = -1;
  if (!begin_h2d_timing(buf, h2d_timing_slot)) return false;
  if (!use_packed_transfer()) {
    uint8_t *gate_dst = w_gate_base_ + gate_exp_bytes_ * (size_t)expert_index;
    uint8_t *up_dst = w_up_base_ + up_exp_bytes_ * (size_t)expert_index;
    uint8_t *down_dst = w_down_base_ + down_exp_bytes_ * (size_t)expert_index;
    e = cudaMemcpyAsync(gate_dst, src, gate_exp_bytes_, cudaMemcpyHostToDevice,
                        st);
    if (e == cudaSuccess)
      e = cudaMemcpyAsync(up_dst, src + gate_exp_bytes_, up_exp_bytes_,
                          cudaMemcpyHostToDevice, st);
    if (e == cudaSuccess)
      e = cudaMemcpyAsync(down_dst, src + gate_exp_bytes_ + up_exp_bytes_,
                          down_exp_bytes_, cudaMemcpyHostToDevice, st);
  } else {
    if (!packed_base_) return false;
    // Direct calls are serialized on the same copy stream. Reuse slot zero
    // in the chunk-sized packed buffer; the destination planar slot remains
    // the global expert_index below.
    e = cudaMemcpyAsync(packed_base_, src, weight_raw_expert_stride_,
                        cudaMemcpyHostToDevice, st);
    if (e == cudaSuccess) {
      e = launch_ggml_moe_unpack(
          packed_base_, w_gate_base_ + gate_exp_bytes_ * (size_t)expert_index,
          w_up_base_ + up_exp_bytes_ * (size_t)expert_index,
          w_down_base_ + down_exp_bytes_ * (size_t)expert_index, 1,
          weight_expert_stride_, gate_exp_bytes_, up_exp_bytes_,
          down_exp_bytes_, gate_exp_bytes_, up_exp_bytes_, down_exp_bytes_,
          (cudaStream_t)copy_stream_);
    }
  }
  if (!end_h2d_timing(buf, h2d_timing_slot)) return false;
  if (e != cudaSuccess) {
    LOG_ERROR("[GGML-MOE] direct RAM H2D failed for expert=%d: %s\n",
              expert_index, cudaGetErrorString(e));
    return false;
  }
  direct_h2d_bytes_[buf] += weight_raw_expert_stride_;
  return true;
}

bool GgmlMoeRunner::copy_expert_d2d(int buf, int expert_index,
                                    const void *src_slab) {
  if (!layer_ready_ || !src_slab || buf < 0 || buf >= NBUF ||
      !copy_batch_started_[buf])
    return false;
  if (expert_index < 0 || expert_index >= plan_.n_experts) return false;
  cudaSetDevice(device_);
  cudaStream_t st = (cudaStream_t)copy_stream_;
  if (!st) return false;
  const uint8_t *src = (const uint8_t *)src_slab;
  if (!use_packed_transfer()) {
    uint8_t *gate_dst = w_gate_base_ + gate_exp_bytes_ * (size_t)expert_index;
    uint8_t *up_dst = w_up_base_ + up_exp_bytes_ * (size_t)expert_index;
    uint8_t *down_dst = w_down_base_ + down_exp_bytes_ * (size_t)expert_index;
    cudaError_t e = cudaMemcpyAsync(gate_dst, src, gate_exp_bytes_,
                                    cudaMemcpyDeviceToDevice, st);
    if (e == cudaSuccess)
      e = cudaMemcpyAsync(up_dst, src + gate_exp_bytes_, up_exp_bytes_,
                          cudaMemcpyDeviceToDevice, st);
    if (e == cudaSuccess)
      e = cudaMemcpyAsync(down_dst, src + gate_exp_bytes_ + up_exp_bytes_,
                          down_exp_bytes_, cudaMemcpyDeviceToDevice, st);
    if (e != cudaSuccess) {
      LOG_ERROR("[GGML-MOE] planar weight D2D failed: %s\n",
                cudaGetErrorString(e));
      return false;
    }
    pending_d2d_bytes_[buf] += gate_exp_bytes_ + up_exp_bytes_ + down_exp_bytes_;
    raw_d2d_pending_ = true;
    return true;
  }
  // The VRAM cache stores an expert exactly as the RAM tier's slab does --
  // [gate|up|down] contiguous, no padding. Land it once in the packed device
  // buffer, then unpack into the three ordinary GGML planes.
  // As with direct H2D, all D2D calls are ordered on the copy stream, so a
  // single chunk-sized packed record is sufficient.
  cudaError_t e = cudaMemcpyAsync(
      packed_base_, src,
      weight_raw_expert_stride_, cudaMemcpyDeviceToDevice, st);
  if (e == cudaSuccess) {
    e = launch_ggml_moe_unpack(
        packed_base_,
        w_gate_base_ + gate_exp_bytes_ * (size_t)expert_index,
        w_up_base_ + up_exp_bytes_ * (size_t)expert_index,
        w_down_base_ + down_exp_bytes_ * (size_t)expert_index, 1,
        weight_expert_stride_, gate_exp_bytes_, up_exp_bytes_, down_exp_bytes_,
        gate_exp_bytes_, up_exp_bytes_, down_exp_bytes_, st);
  }
  if (e != cudaSuccess) {
    LOG_ERROR("[GGML-MOE] weight D2D failed: %s\n", cudaGetErrorString(e));
    return false;
  }
  pending_d2d_bytes_[buf] += weight_raw_expert_stride_;
  if (debug_check_device_output()) {
    e = cudaStreamSynchronize(st);
    if (e != cudaSuccess) return false;
    const size_t gate_row =
        ggml_row_size((ggml_type)plan_.gate_type, plan_.d_model);
    const size_t up_row =
        ggml_row_size((ggml_type)plan_.up_type, plan_.d_model);
    const size_t down_row =
        ggml_row_size((ggml_type)plan_.down_type, plan_.d_ff);
    std::vector<uint8_t> source_row;
    std::vector<uint8_t> device_row;
    auto verify_d2d_row = [&](const char *name, const uint8_t *host,
                              const uint8_t *device, size_t row_bytes) {
      source_row.resize(row_bytes);
      device_row.resize(row_bytes);
      const cudaError_t src_e = cudaMemcpy(source_row.data(), host, row_bytes,
                                           cudaMemcpyDeviceToHost);
      const cudaError_t dst_e = cudaMemcpy(device_row.data(), device, row_bytes,
                                           cudaMemcpyDeviceToHost);
      if (src_e != cudaSuccess || dst_e != cudaSuccess ||
          std::memcmp(source_row.data(), device_row.data(), row_bytes) != 0) {
        LOG_INFO("[GGML-MOE-DEBUG] D2D unpack mismatch slot=%d plane=%s "
                 "bytes=%zu copy=%s\n",
                 expert_index, name, row_bytes,
                 src_e != cudaSuccess
                     ? cudaGetErrorString(src_e)
                     : (dst_e == cudaSuccess ? "ok" : cudaGetErrorString(dst_e)));
      }
    };
    const uint8_t *src_record = src;
    verify_d2d_row("gate-last", src_record + gate_exp_bytes_ - gate_row,
                   w_gate_base_ + gate_exp_bytes_ * (size_t)expert_index +
                       gate_exp_bytes_ - gate_row,
                   gate_row);
    verify_d2d_row("up-last", src_record + gate_exp_bytes_ + up_exp_bytes_ -
                                      up_row,
                   w_up_base_ + up_exp_bytes_ * (size_t)expert_index +
                       up_exp_bytes_ - up_row,
                   up_row);
    verify_d2d_row("down-last",
                   src_record + gate_exp_bytes_ + up_exp_bytes_ +
                       down_exp_bytes_ - down_row,
                   w_down_base_ + down_exp_bytes_ * (size_t)expert_index +
                       down_exp_bytes_ - down_row,
                   down_row);
  }
  raw_d2d_pending_ = true;
  return true;
}

void GgmlMoeRunner::sync_copies() {
  // Backend-stream copies are already ordered before the graph. Waiting here
  // would restore the old per-layer barrier. Completion events are drained
  // by collect_h2d() after the graph has been submitted.
  if (backend_stream_copies_) {
    if (raw_d2d_pending_ && copy_stream_ && !use_async_moe_graph_reuse()) {
      cudaSetDevice(device_);
      cudaStreamSynchronize((cudaStream_t)copy_stream_);
    }
    raw_d2d_pending_ = false;
    return;
  }
  if (!copy_stream_) return;
  cudaSetDevice(device_);
  cudaStreamSynchronize((cudaStream_t)copy_stream_);
  for (int i = 0; i < NBUF; ++i) {
    if (!pending_[i]) continue;
    account_copy_buffer(i);
  }
}

void GgmlMoeRunner::collect_h2d(double &seconds, size_t &bytes) {
  GgmlMoeTransferStats stats;
  collect_transfer_stats(stats);
  seconds = stats.h2d_seconds;
  bytes = stats.h2d_bytes;
}

void GgmlMoeRunner::collect_transfer_stats(GgmlMoeTransferStats &stats) {
  if (backend_stream_copies_ && !use_async_moe_graph_reuse()) {
    // Preserve the conservative backend-stream mode when queued graph reuse
    // is disabled.  The asynchronous mode leaves these events pending across
    // layers; finish_h2d() performs the single request-level drain.
    for (int i = 0; i < NBUF; ++i) wait_stage_free(i);
  }
  stats.d2d_seconds = d2d_seconds_;
  stats.h2d_seconds = h2d_seconds_;
  stats.h2d_dma_seconds = h2d_dma_seconds_;
  stats.d2d_bytes = d2d_bytes_;
  stats.h2d_bytes = h2d_bytes_;
  d2d_seconds_ = 0.0;
  d2d_bytes_ = 0;
  h2d_seconds_ = 0.0;
  h2d_dma_seconds_ = 0.0;
  h2d_bytes_ = 0;
}

void GgmlMoeRunner::finish_h2d() {
  if (backend_stream_copies_) {
    // wait_stage_free() waits for the backend event and accounts the
    // transfer. The deferred source holds are released only after this
    // drain, so an eviction cannot overwrite a VRAM source still read by the
    // copy stream.
    for (int i = 0; i < NBUF; ++i) wait_stage_free(i);
  }
  // External resident-layer weights are read directly by the shared GGML
  // compute stream rather than by copy_stream_.  Drain that stream before
  // releasing their temporary cache holds at the request boundary.
  if (backend_ && !deferred_vram_holds_.empty())
    ggml_backend_synchronize((ggml_backend_t)backend_);
  release_completed_vram_holds();
}

void GgmlMoeRunner::defer_vram_hold(void *topology, int layer, int expert,
                                     const uint8_t *device_ptr) {
  if (!topology || !device_ptr) return;
  deferred_vram_holds_.push_back({topology, layer, expert, device_ptr});
}

void GgmlMoeRunner::release_completed_vram_holds() {
  // This method is called at a runner reuse point or request end. Waiting here
  // is the safe lifetime boundary for source VRAM; callers no longer need to
  // impose that wait at the end of every layer.
  for (int i = 0; i < NBUF; ++i) wait_stage_free(i);
  for (const DeferredVramHold &hold : deferred_vram_holds_) {
    auto *topology = static_cast<GpuTopology *>(hold.topology);
    topology->release_hold(hold.layer, hold.expert, hold.device_ptr, false);
  }
  deferred_vram_holds_.clear();
  for (auto &release : deferred_host_releases_) {
    if (release)
      release();
  }
  deferred_host_releases_.clear();
}

void GgmlMoeRunner::defer_host_release(std::function<void()> release) {
  if (release)
    deferred_host_releases_.push_back(std::move(release));
}

bool GgmlMoeRunner::run_layer_device_impl(const float *host_x,
                                           const void *device_x,
                                           const int32_t *ids,
                                           const float *w,
                                           const void **device_output,
                                           bool device_tensor,
                                           const void *device_ids,
                                           const void *device_weights) {
  if (device_output) *device_output = nullptr;
  if (!layer_ready_ || (!host_x && !device_x) || (host_x && device_x) ||
      !device_output || ((device_ids == nullptr) != (device_weights == nullptr)) ||
      (!device_ids && (!ids || !w)))
    return false;
  cudaSetDevice(device_);
  if (trace_backend_calls())
    LOG_INFO("[GGML-MOE-TRACE] enter tokens=%d external=%d backend_stream=%d "
             "device_tensor=%d device_route=%d pending=%d,%d\n",
             plan_.n_tokens, weight_external_ ? 1 : 0,
             backend_stream_copies_ ? 1 : 0, device_tensor ? 1 : 0,
             device_ids ? 1 : 0, pending_[0] ? 1 : 0, pending_[1] ? 1 : 0);

  // The compact graph owns reusable input/output/workspace tensors.  The
  // shared GGML backend may have additional HC graphs queued after the
  // previous MoE graph, so the MoE completion event alone is not sufficient
  // to prove that those tensors are no longer being consumed.  Until the
  // runner has per-layer graph storage, drain the shared backend before
  // overwriting the tensors for the next layer.  H2D copies remain on the
  // separate copy backend and can overlap the host-side routing/staging work
  // that leads up to this boundary.
  if (backend_stream_copies_ && graph_in_flight_ &&
      !use_async_moe_graph_reuse()) {
    ggml_backend_synchronize((ggml_backend_t)backend_);
    graph_in_flight_ = false;
  }

  // Raw-copy mode needs an explicit cross-stream handoff. Backend-stream
  // mode queues H2D and graph work on the same GGML CUDA stream, so waiting
  // here would defeat the queued dependency.
  if (!backend_stream_copies_) {
    for (int i = 0; i < NBUF; ++i) wait_stage_free(i);
  }

  if (!ensure_graph()) return false;

  const int S = plan_.n_tokens;
  const int K = plan_.topk;
  const int D = plan_.d_model;

  const size_t x_bytes = (size_t)S * D * sizeof(float);
  if (device_x && device_tensor) {
    ggml_tensor *src = (ggml_tensor *)device_x;
    if (src->type != GGML_TYPE_F32 || src->ne[0] != D || src->ne[1] < S ||
        src->ne[2] != 1 || src->ne[3] != 1 || src->nb[0] != sizeof(float) ||
        src->nb[1] != (size_t)D * sizeof(float) || !src->buffer ||
        !src->data) {
      LOG_ERROR("[GGML-MOE] device tensor input has unexpected layout\n");
      return false;
    }
    ggml_tensor *src_view = ggml_view_2d(
        (ggml_context *)graph_ctx_, src, D, S, src->nb[1], 0);
    if (!src_view) return false;
    // Both tensors belong to the same shared CUDA backend.  This queues the
    // HC -> MoE dependency on GGML's stream, where it is ordered after the
    // producer graph and before the MUL_MAT_ID graph.
    ggml_backend_tensor_copy_async((ggml_backend_t)backend_,
                                   (ggml_backend_t)backend_, src_view,
                                   (ggml_tensor *)graph_x_);
    if (trace_backend_calls())
      LOG_INFO("[GGML-MOE-TRACE] input tensor copy queued\n");
    // Diagnostic only: verify the exact bytes handed to MUL_MAT_ID before
    // trying to explain a bad expert result.  A finite HC source can still be
    // published incorrectly if the view/copy path mishandles the [D,S]
    // prefix of HC's persistent [D,S_cap] tensor.  Use the same opt-in flag
    // as the output checker so normal runs pay nothing here.
    if (debug_check_device_output()) {
      cudaDeviceSynchronize();
      const size_t n = (size_t)S * D;
      static thread_local std::vector<float> src_check;
      static thread_local std::vector<float> dst_check;
      src_check.resize(n);
      dst_check.resize(n);
      const cudaError_t src_e = cudaMemcpy(
          src_check.data(), src_view->data, n * sizeof(float),
          cudaMemcpyDeviceToHost);
      const cudaError_t dst_e = cudaMemcpy(
          dst_check.data(), ((ggml_tensor *)graph_x_)->data,
          n * sizeof(float), cudaMemcpyDeviceToHost);
      size_t src_bad = 0;
      size_t dst_bad = 0;
      size_t first_diff = n;
      double max_diff = 0.0;
      double src_max = 0.0;
      double dst_max = 0.0;
      if (src_e == cudaSuccess && dst_e == cudaSuccess) {
        for (size_t i = 0; i < n; ++i) {
          const float sv = src_check[i];
          const float dv = dst_check[i];
          if (!std::isfinite(sv)) ++src_bad;
          if (!std::isfinite(dv)) ++dst_bad;
          if (std::isfinite(sv))
            src_max = std::max(src_max, std::abs((double)sv));
          if (std::isfinite(dv))
            dst_max = std::max(dst_max, std::abs((double)dv));
          const double diff = std::abs((double)sv - (double)dv);
          if (diff > max_diff) {
            max_diff = diff;
            first_diff = i;
          }
        }
      }
      LOG_INFO("[GGML-MOE-DEBUG] input S=%d src_finite=%zu/%zu "
               "dst_finite=%zu/%zu src_max=%g dst_max=%g max_diff=%g "
               "first_diff=%s\n",
               plan_.n_tokens, src_e == cudaSuccess ? n - src_bad : 0, n,
               dst_e == cudaSuccess ? n - dst_bad : 0, n, src_max, dst_max,
               max_diff,
               first_diff == n ? "none" : std::to_string(first_diff).c_str());
      if (src_e != cudaSuccess || dst_e != cudaSuccess) return false;
    }
  } else if (device_x) {
    // The producer graph has completed before this function is called. A
    // synchronous device-to-device copy is intentional here: unlike a host
    // upload it does not cross PCIe, and the GGML backend stream is opaque to
    // this project, so this is the safe way to publish the input tensor.
    if (cudaMemcpy(((ggml_tensor *)graph_x_)->data, device_x, x_bytes,
                   cudaMemcpyDeviceToDevice) != cudaSuccess)
      return false;
  } else {
    // Queue all inputs on GGML's own CUDA stream. The old synchronous tensor
    // setter performed a stream synchronization after each input, which
    // needlessly added three CPU round trips to every MoE layer.
    ggml_backend_tensor_set_async((ggml_backend_t)backend_,
                                  (ggml_tensor *)graph_x_, host_x, 0, x_bytes);
  }
  const size_t route_count = (size_t)S * K;
  if (device_ids) {
    // The router and this runner share the same GGML CUDA backend.  Queue the
    // tiny route-table copies after the router graph, avoiding a host
    // round-trip while keeping the runner's persistent input tensors stable.
    ggml_tensor *src_ids = (ggml_tensor *)device_ids;
    ggml_tensor *src_weights = (ggml_tensor *)device_weights;
    if (!src_ids || !src_weights || src_ids->type != GGML_TYPE_I32 ||
        src_ids->ne[0] != K || src_ids->ne[1] < S || src_ids->ne[2] != 1 ||
        src_ids->ne[3] != 1 || src_ids->nb[0] != sizeof(int32_t) ||
        src_ids->nb[1] != (size_t)K * sizeof(int32_t) || !src_ids->buffer ||
        !src_ids->data || src_weights->type != GGML_TYPE_F32 ||
        src_weights->ne[0] != 1 || src_weights->ne[1] != K ||
        src_weights->ne[2] < S || src_weights->ne[3] != 1 ||
        src_weights->nb[0] != sizeof(float) ||
        src_weights->nb[1] != sizeof(float) ||
        src_weights->nb[2] != (size_t)K * sizeof(float) ||
        !src_weights->buffer || !src_weights->data)
      return false;
    ggml_backend_tensor_copy_async((ggml_backend_t)backend_,
                                   (ggml_backend_t)backend_, src_ids,
                                   (ggml_tensor *)graph_ids_);
    ggml_backend_tensor_copy_async((ggml_backend_t)backend_,
                                   (ggml_backend_t)backend_, src_weights,
                                   (ggml_tensor *)graph_w_);
    if (trace_backend_calls())
      LOG_INFO("[GGML-MOE-TRACE] device route copies queued\n");
  } else if (use_async_moe_graph_reuse() && !debug_sync_route_inputs()) {
    // Route tables are tiny, but this path is exercised once per MoE layer.
    // The old synchronous setter made each pair a blocking cudaMemcpy.  Use
    // a persistent pinned ring so GGML can queue both writes on its shared
    // CUDA stream; that stream already orders them before this layer's graph.
    if (!ensure_route_stage(route_count)) {
      LOG_WARN("[GGML-MOE] pinned route-table staging unavailable; "
               "falling back to synchronous route copies\n");
      ggml_backend_tensor_set((ggml_tensor *)graph_ids_, ids, 0,
                              route_count * sizeof(int32_t));
      ggml_backend_tensor_set((ggml_tensor *)graph_w_, w, 0,
                              route_count * sizeof(float));
    } else {
      static std::atomic<bool> logged_async_route_stage{false};
      bool expected = false;
      if (logged_async_route_stage.compare_exchange_strong(
              expected, true, std::memory_order_relaxed)) {
        LOG_INFO("[GGML-MOE] async pinned route-table staging active\n");
      }
      if (route_stage_index_ >= NROUTE_STAGE) {
        // The normal request path consumes fewer than NROUTE_STAGE slots.
        // Keep wrap-around safe for long-lived interactive sessions without
        // adding a synchronization to ordinary layer dispatch.
        ggml_backend_synchronize((ggml_backend_t)backend_);
        route_stage_index_ = 0;
      }
      const int slot = route_stage_index_++;
      int32_t *route_ids = route_ids_pinned_ +
                           (size_t)slot * route_stage_capacity_;
      float *route_weights = route_weights_pinned_ +
                             (size_t)slot * route_stage_capacity_;
      std::memcpy(route_ids, ids, route_count * sizeof(int32_t));
      std::memcpy(route_weights, w, route_count * sizeof(float));
      ggml_backend_tensor_set_async((ggml_backend_t)backend_,
                                     (ggml_tensor *)graph_ids_, route_ids, 0,
                                     route_count * sizeof(int32_t));
      ggml_backend_tensor_set_async((ggml_backend_t)backend_,
                                    (ggml_tensor *)graph_w_, route_weights, 0,
                                    route_count * sizeof(float));
    }
  } else if (backend_stream_copies_ && use_async_moe_graph_reuse() &&
             debug_sync_route_inputs()) {
    // Diagnostic isolation: a synchronous device write removes both the
    // host-source lifetime question and the backend copy-stream ordering
    // question for the tiny route tables. If this arm is correct while the
    // persistent asynchronous arm is not, the remaining fault is in route
    // table publication rather than expert weights or graph execution.
    ggml_backend_tensor_set((ggml_tensor *)graph_ids_, ids, 0,
                            route_count * sizeof(int32_t));
    ggml_backend_tensor_set((ggml_tensor *)graph_w_, w, 0,
                            route_count * sizeof(float));
  } else {
    // The non-backend-stream path has the same pageable-source lifetime
    // problem as the branch above. The route tables are only S*K integers and
    // floats, so use the synchronous setter here as well rather than handing
    // caller-owned vectors to GGML's raw cudaMemcpyAsync entry point.
    ggml_backend_tensor_set((ggml_tensor *)graph_ids_, ids, 0,
                            route_count * sizeof(int32_t));
    ggml_backend_tensor_set((ggml_tensor *)graph_w_, w, 0,
                            route_count * sizeof(float));
  }

  // H2D writes are queued on the dedicated copy backend. Insert the device
  // side dependency on the compute backend immediately before the graph so
  // the host does not wait for the full transfer batch.
  if (backend_stream_copies_) {
    if (debug_copy_sync() && copy_stream_) {
      const cudaError_t e =
          cudaStreamSynchronize((cudaStream_t)copy_stream_);
      if (e != cudaSuccess) {
        LOG_ERROR("[GGML-MOE] diagnostic copy-stream synchronize failed: %s\n",
                  cudaGetErrorString(e));
        return false;
      }
    }
    for (int i = 0; i < NBUF; ++i) {
      if (!pending_[i]) continue;
      if (trace_backend_calls())
        LOG_INFO("[GGML-MOE-TRACE] waiting for stage event buf=%d\n", i);
      ggml_backend_event_wait(
          (ggml_backend_t)backend_,
          (ggml_backend_event_t)backend_stage_event_[i]);
      if (trace_backend_calls())
        LOG_INFO("[GGML-MOE-TRACE] stage event queued buf=%d\n", i);
    }
  }

  if (trace_backend_calls())
    LOG_INFO("[GGML-MOE-TRACE] graph compute begin\n");

  // Keep the result on the shared GGML CUDA stream.  The device consumer
  // (Qwen4Exp HC post-combine) queues its D2D handoff on this same backend,
  // so making this call synchronous would add a CPU round trip before every
  // layer's post-combine graph.  The caller still has an explicit
  // synchronization point when it needs host data (the next HC pre-mix
  // router download, or the final collapse).
  const ggml_status st =
      ggml_backend_graph_compute_async((ggml_backend_t)backend_,
                                       (ggml_cgraph *)graph_);
  if (st != GGML_STATUS_SUCCESS) {
    LOG_ERROR("[GGML-MOE] graph compute failed (status=%d)\n", (int)st);
    return false;
  }
  if (trace_backend_calls())
    LOG_INFO("[GGML-MOE-TRACE] graph compute returned\n");

  // Protect this graph's temporary allocator workspace until the next shape
  // change forces ensure_graph() to rebuild it.
  ggml_backend_event_record((ggml_backend_event_t)graph_done_event_,
                            (ggml_backend_t)backend_);
  graph_in_flight_ = true;
  if (trace_backend_calls())
    LOG_INFO("[GGML-MOE-TRACE] graph completion event recorded\n");

  if (backend_stream_copies_ && !weight_external_) {
    // Protect the compact weight buffer from the next layer's H2D overwrite.
    // Use a fresh event for every layer in this request: the copy backend may
    // still have an earlier wait queued when the host reaches the next layer.
    void *event = backend_weight_done_event_[backend_weight_event_next_];
    ggml_backend_event_record((ggml_backend_event_t)event,
                              (ggml_backend_t)backend_);
    backend_weight_last_event_[weight_buffer_index_] = event;
    backend_weight_buffer_in_use_[weight_buffer_index_] = true;
    backend_weight_event_next_ =
        (backend_weight_event_next_ + 1) % NWEIGHT_EVENTS;
    if (trace_backend_calls())
      LOG_INFO("[GGML-MOE-TRACE] weight completion event recorded\n");
  }

  // Diagnostic only. The asynchronous path normally lets the next layer
  // discover CUDA failures at a later API call; this opt-in sync pins the
  // error to the first MoE graph that produced it.
  static const bool debug_sync = [] {
    const char *env = std::getenv("MOESTREAM_GGML_DEBUG_SYNC");
    return env && env[0] == '1';
  }();
  if (debug_sync || debug_device_sync() || (backend_stream_copies_ &&
                     !use_async_moe_graph_reuse())) {
    // The default backend-stream mode keeps the old finite/correct behavior.
    // The opt-in asynchronous mode relies on same-stream ordering: HC queues
    // its copy of graph_out_ before the next graph reuses the allocator.
    ggml_backend_synchronize((ggml_backend_t)backend_);
    if (debug_device_sync()) {
      const cudaError_t e = cudaDeviceSynchronize();
      if (e != cudaSuccess) {
        LOG_ERROR("[GGML-MOE] diagnostic cudaDeviceSynchronize failed: %s\n",
                  cudaGetErrorString(e));
        return false;
      }
    }
    graph_in_flight_ = false;
  }

  if (debug_check_device_output()) {
    const size_t n = (size_t)plan_.n_tokens * plan_.d_model;
    static thread_local std::vector<float> check;
    check.resize(n);
    const cudaError_t e = cudaMemcpy(check.data(),
                                     ((ggml_tensor *)graph_out_)->data,
                                     n * sizeof(float),
                                     cudaMemcpyDeviceToHost);
    size_t bad = 0;
    size_t first_bad = n;
    if (e == cudaSuccess) {
      for (size_t i = 0; i < n; ++i) {
        if (!std::isfinite(check[i])) {
          if (first_bad == n) first_bad = i;
          ++bad;
        }
      }
    }
    static int call = 0;
    LOG_INFO("[GGML-MOE-DEBUG] output call=%d finite=%zu/%zu bad=%zu copy=%s\n",
             call++, e == cudaSuccess ? n - bad : 0, n, bad,
             e == cudaSuccess ? "ok" : cudaGetErrorString(e));
    if (first_bad != n) {
      LOG_INFO("[GGML-MOE-DEBUG] first_bad linear=%zu token=%zu d=%zu value=%g\n",
               first_bad, first_bad / (size_t)plan_.d_model,
               first_bad % (size_t)plan_.d_model, (double)check[first_bad]);
    }
    if (e != cudaSuccess) return false;

    // If the final reduction is bad, inspect the first internal tensor that
    // went bad.  This separates an IQ1_M/IQ4_NL projection failure from a
    // later SwiGLU, routed-weight reduction, or top-k weighting failure.  The
    // tensors are retained only as debug handles; normal executions do not
    // copy or scan these intermediates.
    if (bad > 0) {
      auto check_intermediate = [&](const char *name, void *opaque) -> size_t {
        ggml_tensor *t = (ggml_tensor *)opaque;
        if (!t || t->type != GGML_TYPE_F32 || !t->data) return (size_t)-1;
        const size_t count = (size_t)ggml_nelements(t);
        static thread_local std::vector<float> values;
        values.resize(count);
        const cudaError_t ce = cudaMemcpy(values.data(), t->data,
                                           count * sizeof(float),
                                           cudaMemcpyDeviceToHost);
        size_t nonfinite = 0;
        size_t first = count;
        double max_abs = 0.0;
        if (ce == cudaSuccess) {
          for (size_t i = 0; i < count; ++i) {
            if (!std::isfinite(values[i])) {
              if (first == count) first = i;
              ++nonfinite;
            } else {
              max_abs = std::max(max_abs, std::abs((double)values[i]));
            }
          }
        }
        LOG_INFO("[GGML-MOE-DEBUG] %s finite=%zu/%zu bad=%zu max_abs=%g "
                 "first=%s\n",
                 name, ce == cudaSuccess ? count - nonfinite : 0, count,
                 nonfinite, max_abs,
                 first == count ? "none" : std::to_string(first).c_str());
        return first;
      };
      check_intermediate("gate", graph_gate_out_);
      check_intermediate("up", graph_up_out_);
      check_intermediate("act", graph_act_out_);
      const size_t down_first = check_intermediate("down", graph_down_out_);

      // The first bad output is laid out as [d, topk-slot, token].  For a
      // down projection this identifies one exact compact expert and one
      // exact quantized weight row, allowing us to tell an invalid transferred
      // IQ4_NL scale from a numerical reduction problem.
      if (down_first != (size_t)-1 && graph_ids_ && t_down_ && w_down_base_) {
        const size_t down_plane = (size_t)plan_.d_model * plan_.topk;
        const size_t token = down_first / down_plane;
        const size_t rem = down_first % down_plane;
        const size_t k = rem / (size_t)plan_.d_model;
        const size_t d = rem % (size_t)plan_.d_model;
        std::vector<int32_t> route_ids((size_t)plan_.topk * plan_.n_tokens);
        const cudaError_t id_e = cudaMemcpy(
            route_ids.data(), ((ggml_tensor *)graph_ids_)->data,
            route_ids.size() * sizeof(int32_t), cudaMemcpyDeviceToHost);
        const int32_t slot = id_e == cudaSuccess
                                 ? route_ids[token * (size_t)plan_.topk + k]
                                 : -1;
        const size_t row_bytes =
            ggml_row_size((ggml_type)plan_.down_type, plan_.d_ff);
        std::vector<uint8_t> weight_row(row_bytes);
        const size_t weight_offset =
            slot >= 0 ? (size_t)slot * down_exp_bytes_ + d * row_bytes : 0;
        const cudaError_t weight_e =
            slot >= 0 && slot < plan_.n_expert_slots
                ? cudaMemcpy(weight_row.data(), w_down_base_ + weight_offset,
                             row_bytes, cudaMemcpyDeviceToHost)
                : cudaErrorInvalidValue;
        const uint16_t scale_bits =
            weight_row.size() >= 2
                ? (uint16_t)weight_row[0] | ((uint16_t)weight_row[1] << 8)
                : 0;
        LOG_INFO("[GGML-MOE-DEBUG] bad down token=%zu d=%zu k=%zu "
                 "compact_slot=%d id_copy=%s row_bytes=%zu scale_bits=0x%04x "
                 "weight_copy=%s\n",
                 token, d, k, slot,
                 id_e == cudaSuccess ? "ok" : cudaGetErrorString(id_e),
                 row_bytes, (unsigned)scale_bits,
                 weight_e == cudaSuccess ? "ok" : cudaGetErrorString(weight_e));
      }
    }
  }

  *device_output = graph_out_;
  return true;
}

bool GgmlMoeRunner::run_layer_device(const float *x, const int32_t *ids,
                                      const float *w,
                                      const void **device_output) {
  return run_layer_device_impl(x, nullptr, ids, w, device_output, false);
}

bool GgmlMoeRunner::run_layer_device_input(const void *device_x,
                                            const int32_t *ids,
                                            const float *w,
                                            const void **device_output) {
  return run_layer_device_impl(nullptr, device_x, ids, w, device_output,
                               false);
}

bool GgmlMoeRunner::run_layer_device_input_tensor(
    const void *device_tensor, const int32_t *ids, const float *w,
    const void **device_output) {
  return run_layer_device_impl(nullptr, device_tensor, ids, w, device_output,
                               true);
}

bool GgmlMoeRunner::run_layer_device_input_tensor_device_route(
    const void *device_tensor, const void *device_ids,
    const void *device_weights, const void **device_output) {
  return run_layer_device_impl(nullptr, device_tensor, nullptr, nullptr,
                               device_output, true, device_ids,
                               device_weights);
}

bool GgmlMoeRunner::run_chunk_device_input_tensor(
    const void *device_tensor, const int32_t *token_ids, const int32_t *ids,
    const float *weights, int copy_buf, const void **device_output) {
  if (device_output) *device_output = nullptr;
  if (!layer_ready_ || !plan_.chunked || !backend_ || !device_tensor ||
      !token_ids || !ids || !weights || !device_output ||
      copy_buf < 0 || copy_buf >= NBUF || !graph_token_ids_ || !chunk_accum_ ||
      !chunk_final_ || !chunk_ready_recorded_)
    return false;

  ggml_tensor *src = (ggml_tensor *)device_tensor;
  if (src->type != GGML_TYPE_F32 || src->ne[0] != plan_.d_model ||
      src->ne[1] < plan_.source_tokens || src->ne[2] != 1 || src->ne[3] != 1 ||
      src->nb[0] != sizeof(float) ||
      src->nb[1] != (size_t)plan_.d_model * sizeof(float) || !src->buffer ||
      !src->data)
    return false;

  cudaSetDevice(device_);
  if (!chunk_input_published_) {
    ggml_tensor *src_view = ggml_view_2d(
        (ggml_context *)graph_ctx_, src, plan_.d_model, plan_.source_tokens,
        src->nb[1], 0);
    if (!src_view) return false;
    ggml_backend_tensor_copy_async((ggml_backend_t)backend_,
                                   (ggml_backend_t)backend_, src_view,
                                   (ggml_tensor *)graph_x_);
    chunk_input_published_ = true;
  }

  const size_t pairs = (size_t)plan_.n_tokens;
  // Keep every queued source alive until the final reduction.  The backend
  // setter is asynchronous and the next chunk may be submitted before the
  // previous graph has consumed its route table; recycling a two-entry host
  // ring here corrupts IDs/weights while the GPU is still reading them.
  const size_t keep = chunk_token_keepalive_.size();
  chunk_token_keepalive_.emplace_back(token_ids, token_ids + pairs);
  chunk_id_keepalive_.emplace_back(ids, ids + pairs);
  chunk_weight_keepalive_.emplace_back(weights, weights + pairs);
  const int route_slot = chunk_stage_index_++ % NBUF;
  chunk_token_stage_[route_slot] = chunk_token_keepalive_[keep];
  chunk_id_stage_[route_slot] = chunk_id_keepalive_[keep];
  chunk_weight_stage_[route_slot] = chunk_weight_keepalive_[keep];
  ggml_backend_tensor_set_async(
      (ggml_backend_t)backend_, (ggml_tensor *)graph_token_ids_,
      chunk_token_keepalive_[keep].data(), 0,
      pairs * sizeof(int32_t));
  ggml_backend_tensor_set_async(
      (ggml_backend_t)backend_, (ggml_tensor *)graph_ids_,
      chunk_id_keepalive_[keep].data(), 0, pairs * sizeof(int32_t));
  ggml_backend_tensor_set_async(
      (ggml_backend_t)backend_, (ggml_tensor *)graph_w_,
      chunk_weight_keepalive_[keep].data(), 0, pairs * sizeof(float));

  // First chunk waits for the zeroing memset; subsequent chunks wait for the
  // preceding default-stream scatter-add. The dependency is queued on the
  // shared backend and never blocks the CPU.
  if (chunk_accum_recorded_) {
    ggml_backend_event_wait((ggml_backend_t)backend_,
                            (ggml_backend_event_t)chunk_accum_event_);
  } else {
    ggml_backend_event_wait((ggml_backend_t)backend_,
                            (ggml_backend_event_t)chunk_ready_event_);
  }
  if (pending_[copy_buf]) {
    ggml_backend_event_wait(
        (ggml_backend_t)backend_,
        (ggml_backend_event_t)backend_stage_event_[copy_buf]);
  }

  if (ggml_backend_graph_compute_async((ggml_backend_t)backend_,
                                       (ggml_cgraph *)graph_) !=
      GGML_STATUS_SUCCESS)
    return false;

  // Chunk mode has two asynchronous consumers of the graph's reusable
  // tensors: the next GGML submission and the default-stream scatter.  Keep
  // a bounded diagnostic here so a bad value can be attributed to the graph
  // itself before either consumer is involved.  This is intentionally opt-in
  // because it synchronizes and copies every inspected intermediate.
  static int debug_chunks = 0;
  const bool debug_this_chunk = debug_check_device_output() && debug_chunks < 512;
  if (debug_this_chunk) {
      ++debug_chunks;
      ggml_backend_synchronize((ggml_backend_t)backend_);
      auto check_f32 = [&](const char *name, void *opaque) -> size_t {
        ggml_tensor *t = (ggml_tensor *)opaque;
        if (!t || t->type != GGML_TYPE_F32 || !t->data) return (size_t)-1;
        const size_t n = (size_t)ggml_nelements(t);
        static thread_local std::vector<float> values;
        values.resize(n);
        const cudaError_t ce = cudaMemcpy(values.data(), t->data,
                                          n * sizeof(float),
                                          cudaMemcpyDeviceToHost);
        size_t bad = 0;
        size_t first = n;
        double max_abs = 0.0;
        if (ce == cudaSuccess) {
          for (size_t i = 0; i < n; ++i) {
            if (!std::isfinite(values[i])) {
              if (first == n) first = i;
              ++bad;
            } else {
              max_abs = std::max(max_abs, std::abs((double)values[i]));
            }
          }
        }
        LOG_INFO("[GGML-MOE-CHUNK-DEBUG] chunk=%d %s finite=%zu/%zu "
                 "bad=%zu max_abs=%g copy=%s first=%s\n",
                 debug_chunks - 1, name,
                 ce == cudaSuccess ? n - bad : 0, n, bad, max_abs,
                 ce == cudaSuccess ? "ok" : cudaGetErrorString(ce),
                 first == n ? "none" : std::to_string(first).c_str());
        return first;
      };
      auto check_i32 = [&](const char *name, void *opaque) {
        ggml_tensor *t = (ggml_tensor *)opaque;
        if (!t || t->type != GGML_TYPE_I32 || !t->data) return;
        const size_t n = (size_t)ggml_nelements(t);
        static thread_local std::vector<int32_t> values;
        values.resize(n);
        const cudaError_t ce = cudaMemcpy(values.data(), t->data,
                                          n * sizeof(int32_t),
                                          cudaMemcpyDeviceToHost);
        int32_t lo = 0, hi = 0;
        if (ce == cudaSuccess && n > 0) {
          lo = hi = values[0];
          for (size_t i = 1; i < n; ++i) {
            lo = std::min(lo, values[i]);
            hi = std::max(hi, values[i]);
          }
        }
        LOG_INFO("[GGML-MOE-CHUNK-DEBUG] chunk=%d %s n=%zu range=%d..%d "
                 "copy=%s\n",
                 debug_chunks - 1, name, n, (int)lo, (int)hi,
                 ce == cudaSuccess ? "ok" : cudaGetErrorString(ce));
      };
      check_f32("input", graph_x_);
      check_i32("token_ids", graph_token_ids_);
      check_i32("expert_ids", graph_ids_);
      check_f32("weights", graph_w_);
      check_f32("gate", graph_gate_out_);
      check_f32("up", graph_up_out_);
      check_f32("act", graph_act_out_);
      const size_t down_first = check_f32("down", graph_down_out_);
      check_f32("output", graph_out_);

      if (down_first != (size_t)-1 &&
          down_first < (size_t)ggml_nelements((ggml_tensor *)graph_down_out_)) {
        const size_t down_plane = (size_t)plan_.d_model * plan_.topk;
        const size_t token = down_first / down_plane;
        const size_t rem = down_first % down_plane;
        const size_t k = rem / (size_t)plan_.d_model;
        const size_t d = rem % (size_t)plan_.d_model;
        std::vector<int32_t> route_ids((size_t)plan_.n_tokens * plan_.topk);
        const cudaError_t id_e = cudaMemcpy(
            route_ids.data(), ((ggml_tensor *)graph_ids_)->data,
            route_ids.size() * sizeof(int32_t), cudaMemcpyDeviceToHost);
        const int32_t slot =
            id_e == cudaSuccess
                ? route_ids[token * (size_t)plan_.topk + k]
                : -1;
        const size_t row_bytes =
            ggml_row_size((ggml_type)plan_.down_type, plan_.d_ff);
        std::vector<uint8_t> row(row_bytes);
        const size_t weight_offset =
            slot >= 0 ? (size_t)slot * down_exp_bytes_ + d * row_bytes : 0;
        const cudaError_t weight_e =
            slot >= 0 && slot < plan_.n_expert_slots
                ? cudaMemcpy(row.data(), w_down_base_ + weight_offset,
                             row_bytes, cudaMemcpyDeviceToHost)
                : cudaErrorInvalidValue;
        const uint16_t first_bits =
            row.size() >= 2
                ? (uint16_t)row[0] | ((uint16_t)row[1] << 8)
                : 0;
        LOG_INFO("[GGML-MOE-CHUNK-DEBUG] bad down chunk=%d token=%zu "
                 "d=%zu k=%zu slot=%d id_copy=%s row_bytes=%zu "
                 "first_bits=0x%04x weight_copy=%s\n",
                 debug_chunks - 1, token, d, k, slot,
                 id_e == cudaSuccess ? "ok" : cudaGetErrorString(id_e),
                 row_bytes, (unsigned)first_bits,
                 weight_e == cudaSuccess ? "ok" : cudaGetErrorString(weight_e));
      }
  }

  // The graph has produced a weighted [D,pairs] chunk. Bridge to the default
  // CUDA stream for the existing scatter kernel, then publish that completion
  // back to the GGML stream before the next chunk graph is queued.
  ggml_backend_event_record(
      (ggml_backend_event_t)chunk_graph_events_[copy_buf],
      (ggml_backend_t)backend_);
  if (cudaStreamWaitEvent(
          0,
          (cudaEvent_t)((ggml_backend_event_t)
                            chunk_graph_events_[copy_buf])->context,
          0) != cudaSuccess)
    return false;
  if (cudaMemcpyAsync(chunk_token_device_[copy_buf],
                      ((ggml_tensor *)graph_token_ids_)->data,
                      pairs * sizeof(int32_t), cudaMemcpyDeviceToDevice, 0) !=
      cudaSuccess)
    return false;
  launch_expert_scatter_add(
      (const float *)((ggml_tensor *)graph_out_)->data,
      (const int *)chunk_token_device_[copy_buf],
      (float *)chunk_accum_, plan_.n_tokens, plan_.d_model,
      plan_.source_tokens, 0);
  if (cudaEventRecord(
          (cudaEvent_t)((ggml_backend_event_t)chunk_accum_event_)->context,
          0) != cudaSuccess)
    return false;
  if (cudaEventRecord((cudaEvent_t)chunk_scatter_done_events_[copy_buf], 0) !=
      cudaSuccess)
    return false;
  if (debug_this_chunk) {
    cudaStreamSynchronize(0);
    const size_t n = (size_t)plan_.source_tokens * plan_.d_model;
    static thread_local std::vector<float> accum_values;
    accum_values.resize(n);
    const cudaError_t ce = cudaMemcpy(accum_values.data(), chunk_accum_,
                                      n * sizeof(float),
                                      cudaMemcpyDeviceToHost);
    size_t bad = 0;
    size_t first = n;
    double max_abs = 0.0;
    if (ce == cudaSuccess) {
      for (size_t i = 0; i < n; ++i) {
        if (!std::isfinite(accum_values[i])) {
          if (first == n) first = i;
          ++bad;
        } else {
          max_abs = std::max(max_abs, std::abs((double)accum_values[i]));
        }
      }
    }
    LOG_INFO("[GGML-MOE-CHUNK-DEBUG] chunk=%d accumulator finite=%zu/%zu "
             "bad=%zu max_abs=%g copy=%s first=%s\n",
             debug_chunks - 1, ce == cudaSuccess ? n - bad : 0, n, bad,
             max_abs, ce == cudaSuccess ? "ok" : cudaGetErrorString(ce),
             first == n ? "none" : std::to_string(first).c_str());
  }

  ggml_backend_event_record(
      (ggml_backend_event_t)chunk_weight_done_events_[copy_buf],
      (ggml_backend_t)backend_);
  chunk_weight_recorded_[copy_buf] = true;
  chunk_scatter_recorded_[copy_buf] = true;
  chunk_accum_recorded_ = true;
  chunk_has_submission_ = true;
  graph_in_flight_ = true;
  return true;
}

bool GgmlMoeRunner::finish_chunked_output(const void **device_output) {
  if (device_output) *device_output = nullptr;
  if (!layer_ready_ || !plan_.chunked || !device_output ||
      !chunk_has_submission_ || !chunk_accum_recorded_ || !chunk_final_)
    return false;
  cudaSetDevice(device_);
  if (cudaStreamWaitEvent(
          0,
          (cudaEvent_t)((ggml_backend_event_t)chunk_accum_event_)->context,
          0) != cudaSuccess)
    return false;
  const size_t bytes = (size_t)plan_.source_tokens * plan_.d_model *
                       sizeof(float);
  if (cudaMemcpyAsync(((ggml_tensor *)chunk_final_)->data, chunk_accum_, bytes,
                      cudaMemcpyDeviceToDevice, 0) != cudaSuccess)
    return false;
  // Re-record after the final copy. The backend has no later wait on this
  // event from the last chunk, and the next layer's memset is ordered after
  // this publication on the same default stream.
  if (cudaEventRecord(
          (cudaEvent_t)((ggml_backend_event_t)chunk_accum_event_)->context,
          0) != cudaSuccess)
    return false;
  ggml_backend_event_wait((ggml_backend_t)backend_,
                          (ggml_backend_event_t)chunk_accum_event_);
  ggml_backend_event_record((ggml_backend_event_t)graph_done_event_,
                            (ggml_backend_t)backend_);
  if (backend_stream_copies_) {
    // Chunked execution uses the same alternating VRAM weight buffers as the
    // ordinary device-output path, but it does not pass through
    // run_layer_device_impl(), where the buffer completion event is normally
    // recorded.  Without recording it here, begin_layer() can reuse this
    // buffer for a later layer while this layer's queued IQ1_M GEMM is still
    // reading it.  The resulting corruption first appears as sporadic NaNs
    // inside the quantized up/down projection.
    void *event = backend_weight_done_event_[backend_weight_event_next_];
    if (!event) return false;
    ggml_backend_event_record((ggml_backend_event_t)event,
                              (ggml_backend_t)backend_);
    backend_weight_last_event_[weight_buffer_index_] = event;
    backend_weight_buffer_in_use_[weight_buffer_index_] = true;
    backend_weight_event_next_ =
        (backend_weight_event_next_ + 1) % NWEIGHT_EVENTS;
  }
  *device_output = chunk_final_;
  return true;
}

void GgmlMoeRunner::wait_device_output() {
  if (!backend_stream_copies_ || !backend_ || !graph_in_flight_ ||
      !graph_done_event_)
    return;
  // Keep this asynchronous: the event is consumed by the next operation on
  // the shared backend, which is the HC post-combine graph.
  ggml_backend_event_wait((ggml_backend_t)backend_,
                          (ggml_backend_event_t)graph_done_event_);
}

bool GgmlMoeRunner::copy_device_output_to_host(const void *device_output,
                                                float *host_out) {
  if (!layer_ready_ || !device_output || !host_out ||
      (device_output != graph_out_ && device_output != chunk_final_))
    return false;
  cudaSetDevice(device_);
  const int out_tokens = plan_.chunked ? plan_.source_tokens : plan_.n_tokens;
  const size_t bytes = (size_t)out_tokens * plan_.d_model * sizeof(float);
  if (plan_.chunked)
    ggml_backend_event_synchronize((ggml_backend_event_t)graph_done_event_);
  ggml_backend_tensor_get((ggml_tensor *)device_output, host_out, 0, bytes);
  return true;
}

bool GgmlMoeRunner::run_layer(const float *x, const int32_t *ids,
                              const float *w, float *out_add) {
  if (!layer_ready_ || !x || !ids || !w || !out_add) return false;

  const void *device_output = nullptr;
  if (!run_layer_device(x, ids, w, &device_output)) return false;

  // moe is [D, S] F32 contiguous, i.e. element (s,d) at s*D+d -- the same
  // row-major [S, D] layout the caller's `output` uses, so this is a straight
  // element-wise add with no transpose.
  static thread_local std::vector<float> host_out;
  host_out.resize((size_t)plan_.n_tokens * plan_.d_model);
  if (!copy_device_output_to_host(device_output, host_out.data())) return false;
  const float *src = host_out.data();
  for (size_t i = 0, n = (size_t)plan_.n_tokens * plan_.d_model; i < n; ++i)
    out_add[i] += src[i];

  return true;
}

#else  // !MOESTREAM_USE_GGML_CUDA

void set_moe_ggml_planar_copies(bool) {}
void set_ggml_moe_async_graph_reuse(bool) {}

bool ggml_moe_test_make_device_tensor(int, int64_t, int64_t, const float *,
                                      void **, void **, void **) {
  return false;
}
void ggml_moe_test_free_device_tensor(void *, void *) {}
bool ggml_moe_test_download_tensor(int, const void *, float *, size_t) {
  return false;
}

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

bool GgmlQwenSharedRunner::compiled_in() { return false; }
GgmlQwenSharedRunner &GgmlQwenSharedRunner::instance() {
  static GgmlQwenSharedRunner r;
  return r;
}
GgmlQwenSharedRunner::~GgmlQwenSharedRunner() {}
void GgmlQwenSharedRunner::shutdown() {}
bool GgmlQwenSharedRunner::ensure(int, int, int, int, bool, float) {
  return false;
}
bool GgmlQwenSharedRunner::ensure_layer(
    int, const GgmlSharedWeightRef &, const GgmlSharedWeightRef &,
    const GgmlSharedWeightRef &, const GgmlSharedWeightRef *) {
  return false;
}
bool GgmlQwenSharedRunner::run(int, const void *, const void *, const void **) {
  return false;
}

bool GgmlMoeRunner::compiled_in() { return false; }
GgmlMoeRunner &GgmlMoeRunner::instance() {
  static GgmlMoeRunner r;
  return r;
}
GgmlMoeRunner &GgmlMoeRunner::instance_for(uint32_t, uint32_t, uint32_t) {
  return instance();
}
GgmlMoeRunner &GgmlMoeRunner::active_instance() { return instance(); }
void GgmlMoeRunner::finish_all_h2d() {}
void GgmlMoeRunner::collect_all_h2d(double &s, size_t &b) {
  s = 0.0;
  b = 0;
}
void GgmlMoeRunner::collect_all_transfer_stats(GgmlMoeTransferStats &stats) {
  stats = GgmlMoeTransferStats{};
}
GgmlMoeRunner::~GgmlMoeRunner() {}
void GgmlMoeRunner::shutdown() {}
void GgmlMoeRunner::free_weight_tensors() {}
bool GgmlMoeRunner::ensure_backend(int) { return false; }
bool GgmlMoeRunner::ensure_weight_buffer(size_t, size_t) { return false; }
bool GgmlMoeRunner::ensure_stage_buffers(size_t) { return false; }
bool GgmlMoeRunner::ensure_chunk_state() { return false; }
void GgmlMoeRunner::free_chunk_state() {}
bool GgmlMoeRunner::begin_layer(const GgmlMoeLayerPlan &) { return false; }
GgmlMoeStageSlot GgmlMoeRunner::stage_slot(int, int) const {
  return GgmlMoeStageSlot{};
}
void GgmlMoeRunner::wait_stage_free(int) {}
bool GgmlMoeRunner::push_chunk(int, int, int, const bool *) { return false; }
bool GgmlMoeRunner::begin_copy_batch(int) { return false; }
bool GgmlMoeRunner::copy_expert_h2d(int, int, const void *) { return false; }
bool GgmlMoeRunner::copy_expert_d2d(int, int, const void *) { return false; }
bool GgmlMoeRunner::mark_copy_d2d_done(int) { return false; }
void GgmlMoeRunner::sync_copies() {}
bool GgmlMoeRunner::run_layer(const float *, const int32_t *, const float *,
                              float *) {
  return false;
}
bool GgmlMoeRunner::run_layer_device(const float *, const int32_t *,
                                      const float *, const void **) {
  return false;
}
bool GgmlMoeRunner::run_layer_device_input(const void *, const int32_t *,
                                            const float *, const void **) {
  return false;
}
bool GgmlMoeRunner::run_layer_device_input_tensor(
    const void *, const int32_t *, const float *, const void **) {
  return false;
}
bool GgmlMoeRunner::run_layer_device_input_tensor_device_route(
    const void *, const void *, const void *, const void **) {
  return false;
}
bool GgmlMoeRunner::run_chunk_device_input_tensor(
    const void *, const int32_t *, const int32_t *, const float *, int,
    const void **) {
  return false;
}
bool GgmlMoeRunner::finish_chunked_output(const void **) { return false; }
void GgmlMoeRunner::wait_device_output() {}
bool GgmlMoeRunner::copy_device_output_to_host(const void *, float *) {
  return false;
}
void GgmlMoeRunner::collect_h2d(double &s, size_t &b) { s = 0.0; b = 0; }
void GgmlMoeRunner::collect_transfer_stats(GgmlMoeTransferStats &stats) {
  stats = GgmlMoeTransferStats{};
}
void GgmlMoeRunner::finish_h2d() {}
void GgmlMoeRunner::defer_vram_hold(void *, int, int, const uint8_t *) {}
void GgmlMoeRunner::release_completed_vram_holds() {}

#endif  // MOESTREAM_USE_GGML_CUDA
