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

// 2026-09-01: extended from one block per HEAD to one block per (head, query)
// pair, so prefill (S>1) can go through this path instead of falling back to
// the CPU causal loop in attention_layer.cpp. Row index r = h*S + s, matching
// the [H,S,*] layout Q_absorbed already uses (see the qabsorb fallback loop in
// attention_layer.cpp, which indexes h*S*kvl + s*kvl).
//
// `valid` is the ONLY new logic: a query at absolute position pos_base+s may
// attend to keys [0, pos_base+s] and no further. Rather than write -inf into
// masked slots and let them underflow through expf, the three loops simply
// stop at `valid` and the tail is zeroed once -- fewer flops, and no
// inf-arithmetic to reason about. Threads with threadIdx.x >= valid never
// enter a loop, leaving local_max/local_sum at their identity values
// (-1e30f / 0.0f), which both reductions already handle.
//
// Decode is unchanged by construction: at S=1, s is always 0 and the caller
// passes T = pos_base+1, so valid == T and the mask is inert.
__global__ void combine_scale_softmax_kernel(
    const float* __restrict__ scores_nope,
    const float* __restrict__ scores_rope,
    float* __restrict__ scores,
    int T, float scale, int S, int pos_base)
{
    int r = blockIdx.x;                 // r = h*S + s
    int s = (S > 1) ? (r % S) : 0;
    int valid = pos_base + s + 1;
    if (valid > T) valid = T;
    if (valid < 1) valid = 1;           // defensive: a row must attend to >=1 key

    extern __shared__ float sh[];
    const float* nope_row = scores_nope + (size_t)r * T;
    const float* rope_row = scores_rope + (size_t)r * T;
    float* row = scores + (size_t)r * T;

    // combine + local max
    float local_max = -1e30f;
    for (int t = threadIdx.x; t < valid; t += blockDim.x) {
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
    for (int t = threadIdx.x; t < valid; t += blockDim.x) {
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
    for (int t = threadIdx.x; t < valid; t += blockDim.x) row[t] /= row_sum;
    // Masked tail: GEMM3 multiplies these weights against Lc[t], so they must
    // be a hard 0, not left as whatever the previous call wrote (the score
    // buffers are grow-only and reused across layers/tokens).
    for (int t = valid + threadIdx.x; t < T; t += blockDim.x) row[t] = 0.0f;
}

extern "C" void launch_combine_scale_softmax(
    const float* scores_nope, const float* scores_rope, float* scores,
    int H, int T, float scale, int S, int pos_base, cudaStream_t stream)
{
    int threads = 256;
    combine_scale_softmax_kernel<<<H * S, threads, threads * sizeof(float), stream>>>(
        scores_nope, scores_rope, scores, T, scale, S, pos_base);
}
