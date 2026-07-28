// rmsnorm.cu — F32 RMSNorm on GPU: y[S,D] = rmsnorm(x[S,D], w[D]).
// Optimized with float4 vectorized loads/stores for maximum memory bandwidth.
#include <cuda_runtime.h>
#include <cstdint>

__global__ void rmsnorm_kernel_float4(
    float* __restrict__       y,
    const float* __restrict__ x,
    const float* __restrict__ w,
    int D, float eps)
{
    int s = blockIdx.x;
    int tid = threadIdx.x;
    
    // Cast to float4 for 128-bit vectorized loads. 
    // Assumes D is a multiple of 4 (standard for LLM hidden dimensions).
    int D_vec = D / 4;
    const float4* x_vec = (const float4*)(x + (size_t)s * D);
    const float4* w_vec = (const float4*)w;
    float4* y_vec = (float4*)(y + (size_t)s * D);

    // Phase 1: Sum of Squares
    float partial = 0.0f;
    for (int i = tid; i < D_vec; i += blockDim.x) {
        float4 xv = x_vec[i];
        partial += (xv.x * xv.x) + (xv.y * xv.y) + (xv.z * xv.z) + (xv.w * xv.w);
    }
    
    // Level 1: Warp reduction
    for (int off = 16; off > 0; off >>= 1) {
        partial += __shfl_down_sync(0xffffffff, partial, off);
    }

    // Shared memory for block reduction
    __shared__ float s_ss[32];
    int lane = tid % 32;
    int warp = tid / 32;
    
    if (lane == 0) {
        s_ss[warp] = partial;
    }
    __syncthreads();

    // Level 2: Block reduction (first warp reduces the partials)
    int nwarps = blockDim.x / 32;
    if (warp == 0) {
        float ss = (lane < nwarps) ? s_ss[lane] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) {
            ss += __shfl_down_sync(0xffffffff, ss, off);
        }
        if (lane == 0) {
            s_ss[0] = ss;
        }
    }
    __syncthreads();

    // Calculate inverse RMS
    float inv = rsqrtf(s_ss[0] / (float)D + eps);
    
    // Phase 2: Apply Normalization & Write Output
    for (int i = tid; i < D_vec; i += blockDim.x) {
        float4 xv = x_vec[i];
        float4 wv = w_vec[i];
        float4 yv;
        
        yv.x = xv.x * inv * wv.x;
        yv.y = xv.y * inv * wv.y;
        yv.z = xv.z * inv * wv.z;
        yv.w = xv.w * inv * wv.w;
        
        y_vec[i] = yv;
    }
}

extern "C" void launch_rmsnorm(float* y, const float* x, const float* w, int S, int D, float eps) {
    // Note: This optimization requires D to be a multiple of 4.
    rmsnorm_kernel_float4<<<S, 256>>>(y, x, w, D, eps);
}