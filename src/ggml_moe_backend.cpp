// ---------------------------------------------------------------------------
// ggml MUL_MAT_ID prefill backend -- implementation.
// See include/ggml_moe_backend.hpp for the design rationale and the shape
// contract this file implements.
// ---------------------------------------------------------------------------
#include "ggml_moe_backend.hpp"
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
#include <vector>

#if defined(MOESTREAM_USE_GGML_CUDA)

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

namespace moe::detail {

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

bool qwen_decode_sched_debug() {
  static const bool enabled = [] {
    const char *e = std::getenv("MOESTREAM_QWEN_DECODE_SCHED_DEBUG");
    return e && e[0] == '1';
  }();
  return enabled;
}

// Route IDs and weights are compact host-visible outputs.  GGML's CUDA VMM
// allocations are valid device pointers, but this machine's CUDA runtime
// rejects the D2H route-table copy when its destination is pageable memory.
// Keep one persistent pinned scratch buffer per inference thread and copy
// from that buffer into the caller's ordinary vector.  This is only 10 IDs +
// 10 weights for decode, and avoids both a per-call cudaHostRegister and the
// invalid-value failure that otherwise turns a valid GPU route into a stale
// CPU fallback.
struct RouterPinnedHostScratch {
  void *data = nullptr;
  size_t capacity = 0;

  ~RouterPinnedHostScratch() {
    if (data) cudaFreeHost(data);
  }

  bool ensure(size_t bytes) {
    if (data && capacity >= bytes) return true;
    if (data) {
      cudaFreeHost(data);
      data = nullptr;
      capacity = 0;
    }
    if (cudaMallocHost(&data, bytes) != cudaSuccess) {
      data = nullptr;
      return false;
    }
    capacity = bytes;
    return true;
  }
};

RouterPinnedHostScratch &router_pinned_host_scratch() {
  static thread_local RouterPinnedHostScratch scratch;
  return scratch;
}

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
  if (tensor->buffer && ggml_backend_buffer_is_host(tensor->buffer)) {
    std::memcpy(host, tensor->data, bytes);
    return true;
  }
  cudaSetDevice(device);
  auto &scratch = router_pinned_host_scratch();
  if (!scratch.ensure(bytes)) {
    LOG_ERROR("[GGML-ROUTER] %s could not allocate pinned readback buffer "
              "(%zu bytes)\n",
              name, bytes);
    return false;
  }
  // All current callers either synchronize the GGML backend immediately
  // before exporting the route table or synchronize here when a graph is
  // still in flight.  The old Async+cudaStreamPerThread pair introduced a
  // second stream-specific fence after that backend fence.  In addition to
  // adding one CUDA API call per route export, that fence could not overlap
  // anything useful because the host needs the IDs before continuing the
  // expert-cache decision.  Copy directly after the request-boundary fence;
  // this preserves ordering while removing the redundant stream sync.
  cudaError_t err = cudaMemcpy(scratch.data, tensor->data, bytes,
                               cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    LOG_ERROR("[GGML-ROUTER] %s readback failed: %s (bytes=%zu "
              "tensor_bytes=%zu device=%d data=%p)\n",
              name, cudaGetErrorString(err), bytes, available, device,
              tensor->data);
    return false;
  }
  std::memcpy(host, scratch.data, bytes);
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

// Qwen4Exp's shared-expert graph consumes two tensors produced immediately
// before it on this same shared GGML CUDA backend: the HC mixed input and the
// routed-MoE result.  In async graph-reuse mode, retargeting the graph-input
// tensor data pointers is safe because all three graphs are ordered on the
// same backend stream.  It removes two device-to-device copies per layer and
// lets the caller queue HC post-combine behind this graph without a host
// synchronization.  Keep an explicit escape hatch for bisecting correctness;
// the default follows the already opt-in async graph-reuse setting.
bool use_direct_shared_inputs() {
  static const bool enabled = [] {
    const char *env = std::getenv("MOESTREAM_GGML_SHARED_DIRECT_INPUT");
    if (env) return env[0] == '1';
    return use_async_moe_graph_reuse();
  }();
  return enabled && use_async_moe_graph_reuse();
}

// Qwen4Exp's device-HC handoff calls the MoE runner with an F32 ggml_tensor
// produced by the immediately preceding HC graph.  In async graph-reuse mode
// both graphs use this same shared CUDA backend stream, so the reusable MoE
// input tensor can consume the producer's data pointer directly.  Keep an
// explicit override because the copy path is useful for bisecting stream or
// lifetime problems.  The public tensor-input entry point is only used by the
// Qwen device-HC handoff; raw-pointer callers retain their existing path.
bool use_direct_moe_input() {
  static const bool enabled = [] {
    const char *env = std::getenv("MOESTREAM_GGML_MOE_DIRECT_INPUT");
    if (env) return env[0] == '1';
    return use_async_moe_graph_reuse();
  }();
  return enabled && use_async_moe_graph_reuse();
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
  moe::detail::moe_ggml_planar_copies_override = enabled;
}

void set_ggml_moe_async_graph_reuse(bool enabled) {
  moe::detail::async_moe_graph_reuse_override = enabled;
  // The async graph mode is only safe and useful when expert weights are
  // double-buffered and copied through the backend's ordered copy path. The
  // legacy raw-copy mode has one compact weight buffer and must synchronize
  // before every layer reuses it, which silently defeats this configuration.
  // Keep the environment variable as an explicit opt-in for callers that
  // want backend-stream copies without enabling graph reuse.
  moe::detail::backend_stream_copy_override = enabled;
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
  moe::GgmlContextGuard ctx(ggml_init(ip));
  if (!ctx) return false;
  ggml_tensor *t = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, ne0, ne1);
  if (!t) return false;
  moe::GgmlBackendBufferGuard buf(
      ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
  if (!buf) return false;
  ggml_backend_tensor_set(t, host_data, 0,
                          (size_t)ne0 * (size_t)ne1 * sizeof(float));
  *out_tensor = t;
  *out_ctx = ctx.release();
  *out_buf = buf.release();
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

void *ggml_shared_backend(int) { return nullptr; }
#endif  // MOESTREAM_USE_GGML_CUDA
