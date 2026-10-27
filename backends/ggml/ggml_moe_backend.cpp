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
#include "graph_config.hpp"

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

bool moe_ggml_planar_copies_override = false;

// Keep the supported planar-copy choice in the typed EngineConfig. There is
// no environment override for this transfer-layout setting.
bool use_packed_transfer() {
  return !moe_ggml_planar_copies_override;
}

bool trace_backend_calls() {
  return Logger::instance().would_log(LogLevel::DEBUG);
}

}  // namespace

void set_moe_ggml_planar_copies(bool enabled) {
  moe::detail::moe_ggml_planar_copies_override = enabled;
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
#else  // !ZEROMOE_USE_GGML_CUDA
void set_moe_ggml_planar_copies(bool) {}

bool ggml_moe_test_make_device_tensor(int, int64_t, int64_t, const float *,
                                      void **, void **, void **) {
  return false;
}
void ggml_moe_test_free_device_tensor(void *, void *) {}
bool ggml_moe_test_download_tensor(int, const void *, float *, size_t) {
  return false;
}

void *ggml_shared_backend(int) { return nullptr; }
#endif  // ZEROMOE_USE_GGML_CUDA
