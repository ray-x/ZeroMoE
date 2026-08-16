// attn_score_softmax.cu — combine (nope+rope)*scale + row-softmax kernel for
// EngineConfig::attn_score_gpu (GLM-5.2's MLA score phase). One block per
// attention head; the actual QK^T-equivalent and weighted-V-sum are plain
// GEMMs (cuBLAS, dispatched from CudaManager::score_gpu_batched_host,
// cuda_manager.hpp) since MLA's latent K/V is shared across every head --
// this kernel is the one piece that isn't a GEMM. See
// tests/bench_score_cpu_vs_gpu.cu for the real-hardware feasibility numbers
// that motivated this (5x-51x speedup over CPU, growing with context
// length) and CLAUDE.md for the full writeup.
//
// NOT the same file as cuda/mla_attention.cu, which CLAUDE.md documents as
// dead code (never called anywhere) -- deliberately a fresh file to avoid
// any confusion with that one.

#include <cuda_runtime.h>

__global__ void combine_scale_softmax_kernel(
    const float* __restrict__ scores_nope,
    const float* __restrict__ scores_rope,
    float* __restrict__ scores,
    int T, float scale)
{
    int h = blockIdx.x;
    extern __shared__ float sh[];
    const float* nope_row = scores_nope + (size_t)h * T;
    const float* rope_row = scores_rope + (size_t)h * T;
    float* row = scores + (size_t)h * T;

    // combine + local max
    float local_max = -1e30f;
    for (int t = threadIdx.x; t < T; t += blockDim.x) {
        float v = (nope_row[t] + rope_row[t]) * scale;
        row[t] = v;
        local_max = fmaxf(local_max, v);
    }
    sh[threadIdx.x] = local_max;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sh[threadIdx.x] = fmaxf(sh[threadIdx.x], sh[threadIdx.x + s]);
        __syncthreads();
    }
    float row_max = sh[0];
    __syncthreads();

    // exp + local sum
    float local_sum = 0.0f;
    for (int t = threadIdx.x; t < T; t += blockDim.x) {
        float e = __expf(row[t] - row_max);
        row[t] = e;
        local_sum += e;
    }
    sh[threadIdx.x] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    float row_sum = sh[0];
    __syncthreads();

    // normalize
    for (int t = threadIdx.x; t < T; t += blockDim.x) row[t] /= row_sum;
}

extern "C" void launch_combine_scale_softmax(
    const float* scores_nope, const float* scores_rope, float* scores,
    int H, int T, float scale, cudaStream_t stream)
{
    int threads = 256;
    combine_scale_softmax_kernel<<<H, threads, threads * sizeof(float), stream>>>(
        scores_nope, scores_rope, scores, T, scale);
}
