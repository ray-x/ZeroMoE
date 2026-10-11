// EXL3 matmul entry points backed by upstream ExLlamaV3's exl3_gemm (compiled
// torch-free through third_party/exl3_torch_stub). Implements the same
// pointer-level contract as the launch_exl3_mul1_* functions in
// cuda/exl3_matmul.cu so the ggml bridge and Exl3GpuExecutor are unchanged.
//
// float activations are converted to fp16 for upstream's kernels (its input
// type). `partials` arguments of the legacy API are unused here.
#include "exl3_cuda.hpp"
#include "exl3_stub.hpp"
#include "quant/exl3_gemm.cuh"
#include "quant/exl3_moe_coop.cuh"

#include <cuda_fp16.h>

#include <cstdlib>
#include <cstring>
#include <map>
#include <vector>
#include <mutex>
#include <utility>

namespace {

__global__ void f32_to_f16(const float *in, __half *out, size_t n) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) out[i] = __float2half(in[i]);
}

__global__ void add_bias(float *out, const uint16_t *bias, int width, size_t n) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) out[i] += __half2float(__ushort_as_half(bias[i % width]));
}

__global__ void accumulate_rows(float *acc, const float *src, float weight,
                                const float *device_weight, int width,
                                size_t n) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float w = device_weight ? device_weight[i / width] : weight;
  acc[i] += w * src[i];
}

__global__ void swiglu_inplace(float *gate, const float *up, size_t n) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) gate[i] = gate[i] / (1.0f + expf(-gate[i])) * up[i];
}

struct Workspace {
  void *half_in = nullptr;
  void *had = nullptr;
  float *tmp = nullptr;
  size_t half_bytes = 0;
  size_t tmp_bytes = 0;
};

std::mutex g_mutex;
std::map<std::pair<int, cudaStream_t>, Workspace> g_workspaces;

Workspace &workspace(int device, cudaStream_t stream) {
  return g_workspaces[{device, stream}];
}

cudaError_t ensure(Workspace &w, size_t half_bytes, size_t tmp_bytes) {
  if (half_bytes > w.half_bytes) {
    cudaFree(w.half_in);
    cudaFree(w.had);
    w.half_in = w.had = nullptr;
    w.half_bytes = 0;
    if (cudaError_t e = cudaMalloc(&w.half_in, half_bytes); e != cudaSuccess) return e;
    if (cudaError_t e = cudaMalloc(&w.had, half_bytes); e != cudaSuccess) return e;
    w.half_bytes = half_bytes;
  }
  if (tmp_bytes > w.tmp_bytes) {
    cudaFree(w.tmp);
    w.tmp = nullptr;
    w.tmp_bytes = 0;
    if (cudaError_t e = cudaMalloc(reinterpret_cast<void **>(&w.tmp), tmp_bytes);
        e != cudaSuccess)
      return e;
    w.tmp_bytes = tmp_bytes;
  }
  return cudaSuccess;
}

bool usable(const Exl3CudaMatrixView &m, int rows) {
  return m.trellis && m.suh && m.svh && m.bits >= 1 && m.bits <= 8 &&
         m.input_features % 16 == 0 && m.output_features % 128 == 0 &&
         rows > 0;
}

// result[rows, n] (float) = EXL3 matmul of float input[rows, k].
cudaError_t gemm(Workspace &w, const Exl3CudaMatrixView &m, const float *input,
                 float *result, int rows, cudaStream_t stream, int device) {
  const size_t in_elems = static_cast<size_t>(rows) * m.input_features;
  f32_to_f16<<<(in_elems + 255) / 256, 256, 0, stream>>>(
      input, static_cast<__half *>(w.half_in), in_elems);
  using at::Tensor;
  auto *trellis = const_cast<uint16_t *>(m.trellis);
  Tensor A(w.half_in, {rows, m.input_features}, at::kHalf, device);
  Tensor B(trellis, {m.input_features / 16, m.output_features / 16, 16 * m.bits},
           at::kShort, device);
  Tensor C(result, {rows, m.output_features}, at::kFloat, device);
  c10::optional<Tensor> suh = Tensor(const_cast<uint16_t *>(m.suh),
                                     {m.input_features / 16}, at::kHalf, device);
  c10::optional<Tensor> had =
      Tensor(w.had, {rows, m.input_features}, at::kHalf, device);
  c10::optional<Tensor> svh = Tensor(const_cast<uint16_t *>(m.svh),
                                     {m.output_features / 16}, at::kHalf, device);
  moestream_exl3::set_current_stream(stream);
  try {
    exl3_gemm(A, B, C, suh, had, svh, 0, false, true, 0);
  } catch (const c10::Error &) {
    return cudaErrorInvalidValue;
  }
  if (m.bias) {
    const size_t n = static_cast<size_t>(rows) * m.output_features;
    add_bias<<<(n + 255) / 256, 256, 0, stream>>>(result, m.bias,
                                                  m.output_features, n);
  }
  return cudaGetLastError();
}

} // namespace

// Fused exl3_moe_coop routed FFN (two launches per layer). Opt-out for A/B:
// MOESTREAM_EXL3_MOE_COOP=0 uses one upstream exl3_gemm sequence per expert.
bool exl3_upstream_moe_coop_enabled() {
  static const bool active = [] {
    const char *v = std::getenv("MOESTREAM_EXL3_MOE_COOP");
    return !v || v[0] != '0';
  }();
  return active;
}

cudaError_t exl3_upstream_matmul(const Exl3CudaMatrixView &m, const float *input,
                                 float *output, float *accumulator, float weight,
                                 const float *device_weight, int rows,
                                 cudaStream_t stream) {
  if (!usable(m, rows) || !input || (!output && !accumulator))
    return cudaErrorNotSupported;
  int device = 0;
  cudaGetDevice(&device);
  std::lock_guard<std::mutex> lock(g_mutex);
  Workspace &w = workspace(device, stream);
  const size_t out_elems = static_cast<size_t>(rows) * m.output_features;
  const size_t half_bytes =
      static_cast<size_t>(rows) * m.input_features * sizeof(__half);
  if (cudaError_t e = ensure(w, half_bytes, accumulator ? out_elems * 4 : 0);
      e != cudaSuccess)
    return e;
  if (!accumulator)
    return gemm(w, m, input, output, rows, stream, device);
  if (cudaError_t e = gemm(w, m, input, w.tmp, rows, stream, device);
      e != cudaSuccess)
    return e;
  accumulate_rows<<<(out_elems + 255) / 256, 256, 0, stream>>>(
      accumulator, w.tmp, weight, device_weight, m.output_features, out_elems);
  return cudaGetLastError();
}

cudaError_t exl3_upstream_gate_up(const Exl3CudaMatrixView &gate,
                                  const Exl3CudaMatrixView &up,
                                  const float *input, float *activation,
                                  int rows, float *up_scratch,
                                  cudaStream_t stream) {
  if (!usable(gate, rows) || !usable(up, rows) || !input || !activation ||
      !up_scratch || gate.input_features != up.input_features ||
      gate.output_features != up.output_features)
    return cudaErrorNotSupported;
  int device = 0;
  cudaGetDevice(&device);
  std::lock_guard<std::mutex> lock(g_mutex);
  Workspace &w = workspace(device, stream);
  const size_t half_bytes =
      static_cast<size_t>(rows) * gate.input_features * sizeof(__half);
  if (cudaError_t e = ensure(w, half_bytes, 0); e != cudaSuccess) return e;
  if (cudaError_t e = gemm(w, gate, input, activation, rows, stream, device);
      e != cudaSuccess)
    return e;
  if (cudaError_t e = gemm(w, up, input, up_scratch, rows, stream, device);
      e != cudaSuccess)
    return e;
  const size_t n = static_cast<size_t>(rows) * gate.output_features;
  swiglu_inplace<<<(n + 255) / 256, 256, 0, stream>>>(activation, up_scratch, n);
  return cudaGetLastError();
}

// ---------------------------------------------------------------------------
// Routed FFN through upstream's fused exl3_moe_coop (two launches per call).
//
// Every (row, slot) route becomes one "local expert" entry of the pointer
// tables: sel[r] = r, tables[r] = that route's resolved gate/up/down device
// pointers. The bridge's slot ownership/residency model is unchanged; only the
// math launch differs.
namespace {

__global__ void f32_to_f16_weights(const float *in, __half *out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = __float2half(in[i]);
}

struct CoopScratch {
  size_t slots = 0, rows = 0, hi = 0, inter = 0, ho = 0;
  __half *x = nullptr, *rw = nullptr, *had_g = nullptr, *had_u = nullptr,
         *gu_g = nullptr, *gu_u = nullptr, *act = nullptr;
  float *d_out = nullptr;
  int *ctr = nullptr;
  int64_t *table = nullptr; // sel + 9 pointer tables
  size_t ctr_len = 0;

  void release() {
    cudaFree(x); cudaFree(rw); cudaFree(had_g); cudaFree(had_u);
    cudaFree(gu_g); cudaFree(gu_u); cudaFree(act); cudaFree(d_out);
    cudaFree(ctr); cudaFree(table);
    *this = CoopScratch{};
  }
};

std::map<std::pair<int, cudaStream_t>, CoopScratch> g_coop;

cudaError_t coop_prepare(CoopScratch &c, size_t rows, size_t slots, size_t hi,
                         size_t inter, size_t ho) {
  if (c.slots == slots && c.rows == rows && c.hi == hi && c.inter == inter &&
      c.ho == ho)
    return cudaSuccess;
  c.release();
  const size_t ctr_len = static_cast<size_t>(
      exl3_moe_coop_ctr_len(slots, rows, inter, ho));
  auto alloc = [](void **p, size_t bytes) { return cudaMalloc(p, bytes); };
  cudaError_t e = cudaSuccess;
#define COOP_ALLOC(ptr, bytes) \
  if ((e = alloc(reinterpret_cast<void **>(&(ptr)), (bytes))) != cudaSuccess) { c.release(); return e; }
  COOP_ALLOC(c.x, rows * hi * 2);
  COOP_ALLOC(c.rw, slots * 2);
  COOP_ALLOC(c.had_g, slots * hi * 2);
  COOP_ALLOC(c.had_u, slots * hi * 2);
  COOP_ALLOC(c.gu_g, slots * inter * 2);
  COOP_ALLOC(c.gu_u, slots * inter * 2);
  COOP_ALLOC(c.act, slots * inter * 2);
  COOP_ALLOC(c.d_out, slots * ho * 4);
  COOP_ALLOC(c.ctr, ctr_len * 4);
  COOP_ALLOC(c.table, slots * 10 * 8);
#undef COOP_ALLOC
  cudaMemset(c.ctr, 0, ctr_len * 4);
  c.slots = slots; c.rows = rows; c.hi = hi; c.inter = inter; c.ho = ho;
  c.ctr_len = ctr_len;
  return cudaSuccess;
}

} // namespace

cudaError_t exl3_upstream_routed_ffn(
    const Exl3CudaMatrixView *const *gate, const Exl3CudaMatrixView *const *up,
    const Exl3CudaMatrixView *const *down, const float *host_weights,
    const float *device_weights, const float *input, float *output, int rows,
    int topk, int hidden, int intermediate, cudaStream_t stream) {
  const int slots = rows * topk;
  if (rows <= 0 || rows > 8 || topk <= 0 || slots > 256 || hidden % 128 ||
      intermediate % 128 || (!host_weights && !device_weights))
    return cudaErrorNotSupported;
  const int bits_gu = gate[0]->bits, bits_d = down[0]->bits;
  for (int r = 0; r < slots; ++r) {
    const auto &g = *gate[r], &u = *up[r], &d = *down[r];
    if (g.bits != bits_gu || u.bits != bits_gu || d.bits != bits_d ||
        g.bias || u.bias || d.bias || !g.trellis || !u.trellis || !d.trellis)
      return cudaErrorNotSupported; // mixed bit widths / biased: legacy path
  }
  int device = 0;
  cudaGetDevice(&device);
  std::lock_guard<std::mutex> lock(g_mutex);
  CoopScratch &c = g_coop[{device, stream}];
  if (cudaError_t e = coop_prepare(c, rows, slots, hidden, intermediate, hidden);
      e != cudaSuccess)
    return e;

  // Host side: sel + 9 pointer tables (+ half routing weights when host-side).
  std::vector<int64_t> host(static_cast<size_t>(slots) * 10);
  auto ptr = [](const void *p) { return static_cast<int64_t>(reinterpret_cast<intptr_t>(p)); };
  for (int r = 0; r < slots; ++r) {
    int64_t *row = host.data();
    row[r] = r;
    row[(1 + 0) * slots + r] = ptr(gate[r]->trellis);
    row[(1 + 1) * slots + r] = ptr(gate[r]->suh);
    row[(1 + 2) * slots + r] = ptr(gate[r]->svh);
    row[(1 + 3) * slots + r] = ptr(up[r]->trellis);
    row[(1 + 4) * slots + r] = ptr(up[r]->suh);
    row[(1 + 5) * slots + r] = ptr(up[r]->svh);
    row[(1 + 6) * slots + r] = ptr(down[r]->trellis);
    row[(1 + 7) * slots + r] = ptr(down[r]->suh);
    row[(1 + 8) * slots + r] = ptr(down[r]->svh);
  }
  if (cudaError_t e = cudaMemcpyAsync(c.table, host.data(), host.size() * 8,
                                      cudaMemcpyHostToDevice, stream);
      e != cudaSuccess)
    return e;
  // Pageable-source copy has staged the data by the time it returns, so
  // `host` may go out of scope.

  const size_t x_elems = static_cast<size_t>(rows) * hidden;
  f32_to_f16<<<(x_elems + 255) / 256, 256, 0, stream>>>(input, c.x, x_elems);
  if (device_weights) {
    f32_to_f16_weights<<<(slots + 255) / 256, 256, 0, stream>>>(device_weights, c.rw, slots);
  } else {
    std::vector<__half> rw(static_cast<size_t>(slots));
    for (int r = 0; r < slots; ++r) rw[r] = __float2half(host_weights[r]);
    if (cudaError_t e = cudaMemcpyAsync(c.rw, rw.data(), rw.size() * 2,
                                        cudaMemcpyHostToDevice, stream);
        e != cudaSuccess)
      return e;
  }

  MoeCoopParams p = {};
  p.x = c.x; p.x_stride = hidden;
  p.sel = c.table; p.rw = c.rw;
  p.bsz = rows; p.topk = topk;
  p.H = hidden; p.Hi = hidden; p.I = intermediate; p.Ho = hidden; p.H_out = hidden;
  p.min_expert = -1; p.max_expert = -1;
  p.g_trellis = c.table + 1 * slots; p.g_suh = c.table + 2 * slots; p.g_svh = c.table + 3 * slots;
  p.u_trellis = c.table + 4 * slots; p.u_suh = c.table + 5 * slots; p.u_svh = c.table + 6 * slots;
  p.d_trellis = c.table + 7 * slots; p.d_suh = c.table + 8 * slots; p.d_svh = c.table + 9 * slots;
  p.act = MOE_COOP_ACT_SILU; p.act_limit = 0.0f; p.gated = true;
  p.had_g = c.had_g; p.had_u = c.had_u;
  p.gu_g = c.gu_g; p.gu_u = c.gu_u; p.gu_f32 = false;
  p.act_out = c.act; p.d_out = c.d_out;
  p.slots_max = slots; p.rows_max = rows; p.n_local = slots;
  p.ctr_a_len = slots * (intermediate / 128);
  p.ctr_b_len = rows * (hidden / 128);
  p.ctr_a = c.ctr;
  p.ctr_b = p.ctr_a + p.ctr_a_len;
  p.runs = p.ctr_b + p.ctr_b_len;
  p.out = output; p.out_stride = hidden;
  try {
    moestream_exl3::set_current_stream(stream);
    exl3_moe_coop_launch(p, static_cast<float>(bits_gu), static_cast<float>(bits_d),
                         /*cb=mul1*/ 2, device, stream);
  } catch (const c10::Error &) {
    return cudaErrorInvalidValue;
  }
  return cudaGetLastError();
}

// ---------------------------------------------------------------------------
// Public EXL3 matmul API (declared in include/exl3_cuda.hpp). `partials`
// arguments are kept for source compatibility with the previous in-tree
// kernels and are unused: upstream's launchers manage their own workspace.
namespace {

bool matrix_ok(const Exl3CudaMatrixView &m) {
  return m.trellis && m.suh && m.svh && m.input_features > 0 &&
         m.output_features > 0 && m.input_features % 128 == 0 &&
         m.output_features % 128 == 0 && m.bits >= 1 && m.bits <= 8;
}

} // namespace

cudaError_t launch_exl3_mul1_matmul(const Exl3CudaMatrixView &matrix,
                                    const float *input, float *output, int rows,
                                    float * /*partials*/, cudaStream_t stream) {
  if (!matrix_ok(matrix) || !input || !output || rows <= 0 || rows > 65535)
    return cudaErrorInvalidValue;
  return exl3_upstream_matmul(matrix, input, output, nullptr, 0.0f, nullptr,
                              rows, stream);
}

cudaError_t launch_exl3_mul1_matmul_accumulate(
    const Exl3CudaMatrixView &matrix, const float *input, float *accumulator,
    float weight, int rows, float * /*partials*/, cudaStream_t stream) {
  if (!matrix_ok(matrix) || !input || !accumulator || rows <= 0 || rows > 65535)
    return cudaErrorInvalidValue;
  return exl3_upstream_matmul(matrix, input, nullptr, accumulator, weight,
                              nullptr, rows, stream);
}

cudaError_t launch_exl3_mul1_matmul_accumulate_device_weight(
    const Exl3CudaMatrixView &matrix, const float *input, float *accumulator,
    const float *weight, int rows, float * /*partials*/, cudaStream_t stream) {
  if (!matrix_ok(matrix) || !input || !accumulator || !weight || rows <= 0 ||
      rows > 65535)
    return cudaErrorInvalidValue;
  return exl3_upstream_matmul(matrix, input, nullptr, accumulator, 0.0f, weight,
                              rows, stream);
}

cudaError_t launch_exl3_mul1_gate_up(
    const Exl3CudaMatrixView &gate, const Exl3CudaMatrixView &up,
    const float *input, float *activation, int rows, float * /*gate_partials*/,
    float *up_partials, cudaStream_t stream) {
  if (!matrix_ok(gate) || !matrix_ok(up) || !input || !activation ||
      !up_partials || gate.input_features != up.input_features ||
      gate.output_features != up.output_features || rows <= 0 || rows > 65535)
    return cudaErrorInvalidValue;
  return exl3_upstream_gate_up(gate, up, input, activation, rows, up_partials,
                               stream);
}
