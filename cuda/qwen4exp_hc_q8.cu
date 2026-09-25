// Optional exact-Q8 Qwen4Exp HC decode kernels.
//
// The ordinary GGML CUDA Q8 path first quantizes the F32 activation to Q8_1.
// That is a good general GEMM trade-off, but HC's low-rank gate is also the
// input to the next router.  This path keeps the Q8_0 weights compressed and
// performs the exact F32-activation dot product directly, avoiding the ~25 MiB
// per-bracket F32 expansion used by the earlier exact device-HC experiment.

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>

extern "C" void launch_q8_0_matmul_stream(float *y, const float *x,
                                           const void *weights, int S, int I,
                                           int O, cudaStream_t stream);

namespace {

__global__ void hc_rmsnorm_scale_kernel(const float *__restrict__ streams,
                                        const float *__restrict__ norm,
                                        float *__restrict__ xn, int rows,
                                        int D, float eps) {
  const int row = blockIdx.x;
  if (row >= rows) return;
  const float *src = streams + (size_t)row * D;
  float sum = 0.0f;
  for (int d = threadIdx.x; d < D; d += blockDim.x) {
    const float v = src[d];
    sum += v * v;
  }
  __shared__ float partial[32];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  for (int off = 16; off; off >>= 1)
    sum += __shfl_down_sync(0xffffffff, sum, off);
  if (lane == 0) partial[warp] = sum;
  __syncthreads();
  if (warp == 0) {
    const int warps = (blockDim.x + 31) >> 5;
    sum = lane < warps ? partial[lane] : 0.0f;
    for (int off = 16; off; off >>= 1)
      sum += __shfl_down_sync(0xffffffff, sum, off);
    if (lane == 0) {
      const float inv = rsqrtf(sum / (float)D + eps);
      float *dst = xn + (size_t)row * D;
      for (int d = 0; d < D; ++d)
        dst[d] = src[d] * inv * norm[row * D + d];
    }
  }
}

__global__ void hc_silu_scale_kernel(float *__restrict__ lo, size_t n,
                                      float scale) {
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += (size_t)gridDim.x * blockDim.x) {
    const float x = lo[i] * scale;
    lo[i] = x / (1.0f + expf(-x));
  }
}

__global__ void hc_gate_collapse_kernel(const float *__restrict__ xn,
                                        const float *__restrict__ gate,
                                        float *__restrict__ mixed, int S,
                                        int HC, int D) {
  const size_t n = (size_t)S * D;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += (size_t)gridDim.x * blockDim.x) {
    const int s = (int)(i / D);
    const int d = (int)(i % D);
    float sum = 0.0f;
    const size_t stream_stride = (size_t)HC * D;
    const size_t row = (size_t)s * stream_stride;
    for (int h = 0; h < HC; ++h) {
      const float g = gate[row + (size_t)h * D + d];
      sum += xn[row + (size_t)h * D + d] /
             (1.0f + expf(-g));
    }
    mixed[i] = sum / (float)HC;
  }
}

__global__ void hc_f32_matmul_kernel(float *__restrict__ y,
                                     const float *__restrict__ x,
                                     const float *__restrict__ w, int S, int I,
                                     int O) {
  const int o = blockIdx.x;
  const int s = blockIdx.y;
  if (o >= O || s >= S) return;
  float sum = 0.0f;
  const float *xr = x + (size_t)s * I;
  const float *wr = w + (size_t)o * I;
  for (int i = threadIdx.x; i < I; i += blockDim.x)
    sum += xr[i] * wr[i];
  __shared__ float partial[32];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  for (int off = 16; off; off >>= 1)
    sum += __shfl_down_sync(0xffffffff, sum, off);
  if (lane == 0) partial[warp] = sum;
  __syncthreads();
  if (warp == 0) {
    const int warps = (blockDim.x + 31) >> 5;
    sum = lane < warps ? partial[lane] : 0.0f;
    for (int off = 16; off; off >>= 1)
      sum += __shfl_down_sync(0xffffffff, sum, off);
    if (lane == 0) y[(size_t)s * O + o] = sum;
  }
}

}  // namespace

extern "C" void launch_qwen4exp_hc_q8_pre(float *streams, const float *norm,
                                           float *xn, float *lo, float *gate,
                                           float *mixed, float *inject,
                                           const void *down, const void *up,
                                           const void *inject_w, int S, int HC,
                                           int D, int LR, float eps,
                                           int inject_q8, cudaStream_t stream) {
  const int rows = S * HC;
  hc_rmsnorm_scale_kernel<<<rows, 256, 0, stream>>>(
      streams, norm, xn, rows, D, eps);

  launch_q8_0_matmul_stream(lo, xn, down, S, HC * D, LR, stream);
  const size_t lo_count = (size_t)S * LR;
  const int lo_blocks = (int)((lo_count + 255) / 256);
  hc_silu_scale_kernel<<<lo_blocks > 1 ? lo_blocks : 1, 256, 0, stream>>>(
      lo, lo_count, 1.0f / (float)HC);

  launch_q8_0_matmul_stream(gate, lo, up, S, LR, HC * D, stream);
  if (inject_q8) {
    launch_q8_0_matmul_stream(inject, xn, inject_w, S, HC * D, HC, stream);
  } else {
    dim3 grid(HC, S);
    hc_f32_matmul_kernel<<<grid, 256, 0, stream>>>(
        inject, xn, static_cast<const float *>(inject_w), S, HC * D, HC);
  }

  const size_t mixed_count = (size_t)S * D;
  const int mixed_blocks = (int)((mixed_count + 255) / 256);
  hc_gate_collapse_kernel<<<mixed_blocks > 1 ? mixed_blocks : 1, 256, 0,
                            stream>>>(xn, gate, mixed, S, HC, D);
}

namespace {
__global__ void hc_post_kernel(float *__restrict__ streams,
                               const float *__restrict__ sub_out,
                               const float *__restrict__ inject, int S, int HC,
                               int D) {
  const size_t n = (size_t)S * HC * D;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += (size_t)gridDim.x * blockDim.x) {
    const int d = (int)(i % D);
    const size_t per_token = (size_t)HC * D;
    const int s = (int)(i / per_token);
    const int h = (int)((i / D) % HC);
    const float z = inject[(size_t)s * HC + h] / (float)HC;
    streams[i] += sub_out[(size_t)s * D + d] *
                  (2.0f / (1.0f + expf(-z)));
  }
}
}  // namespace

extern "C" void launch_qwen4exp_hc_q8_post(float *streams,
                                            const float *sub_out,
                                            const float *inject, int S, int HC,
                                            int D, cudaStream_t stream) {
  const size_t n = (size_t)S * HC * D;
  const int blocks = (int)((n + 255) / 256);
  hc_post_kernel<<<blocks > 1 ? blocks : 1, 256, 0, stream>>>(
      streams, sub_out, inject, S, HC, D);
}
