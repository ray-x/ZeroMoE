// rmsnorm.cu — F32 RMSNorm on GPU: y[S,D] = rmsnorm(x[S,D], w[D]).
//   per row: ss = mean(x^2); r = rsqrt(ss + eps); y = x * r * w
// One block per sequence row; 256 threads cover D in strides, with a two-level
// warp reduction for the sum-of-squares.
#include <cuda_runtime.h>
#include <cstdint>

__global__ void rmsnorm_kernel(
    float* __restrict__       y,
    const float* __restrict__ x,
    const float* __restrict__ w,
    int D, float eps)
{
    int s = blockIdx.x;
    int tid = threadIdx.x;
    const float* xr = x + (size_t)s * D;

    float partial = 0.0f;
    for (int i = tid; i < D; i += blockDim.x) partial += xr[i] * xr[i];
    for (int off = 16; off > 0; off >>= 1) partial += __shfl_down_sync(0xffffffff, partial, off);

    __shared__ float s_ss[32];
    int lane = tid % 32, warp = tid / 32;
    if (lane == 0) s_ss[warp] = partial;
    __syncthreads();

    int nwarps = blockDim.x / 32;
    if (warp == 0) {
        float ss = (lane < nwarps) ? s_ss[lane] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) ss += __shfl_down_sync(0xffffffff, ss, off);
        if (lane == 0) s_ss[0] = ss;
    }
    __syncthreads();

    float inv = rsqrtf(s_ss[0] / (float)D + eps);
    float* yr = y + (size_t)s * D;
    for (int i = tid; i < D; i += blockDim.x) yr[i] = xr[i] * inv * w[i];
}

extern "C" void launch_rmsnorm(float* y, const float* x, const float* w, int S, int D, float eps) {
    rmsnorm_kernel<<<S, 256>>>(y, x, w, D, eps);
}
