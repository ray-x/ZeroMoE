// dsv4_score_softmax.cu — scale + optional per-head attention-sink logit +
// row-softmax kernel for EngineConfig::dsv4_score_gpu (DeepSeek-V4 raw-SWA
// and HCA attention score phase). Unlike attn_score_softmax.cu (GLM-5.2's
// MLA score, which needs a nope+rope COMBINE step because Lc/Rc are two
// separate latent tensors), DeepSeek-V4's raw KVc cache is a single shared
// K==V buffer per layer (no absorption, no nope/rope split at the cache
// level -- see attention_layer.cpp's dsv4_attention_gguf), so the QK^T-
// equivalent GEMM already produces one combined [H,T] score buffer; this
// kernel only needs to scale it and fold in an optional attention-sink
// logit before softmax. The sink logit (one scalar per head, no matching V
// row) is included in the running max/sum but never written into row[] --
// it only shrinks the real weights' softmax mass, exactly matching the CPU
// reference (attention_layer.cpp's `attn_scores[...] = sinks[h]` column,
// added unscaled -- see dsv4_score_gpu_batched_host's own comment in
// cuda_manager.hpp).
//
// GPU-accelerates only ratio==0 (raw-SWA) and ratio==128 (HCA) layers --
// CSA (ratio==4) layers keep their whole score phase on CPU because their
// Lightning-Indexer-selected blocks are a small, data-dependent, per-query
// gather that would need a second, separately-computed softmax merged back
// into this one (an online-softmax/log-sum-exp merge across the CPU/GPU
// boundary) -- real correctness-risk surface area for a workload this
// already small. See EngineConfig::dsv4_score_gpu's own comment.
//
// UNVERIFIED ON REAL HARDWARE (2026-08-31) -- ported by direct analogy to
// attn_score_softmax.cu's already-bench-verified shape (see that file's own
// header and tests/bench_score_cpu_vs_gpu.cu), but this kernel's sink-
// folding logic is new and has no GPU-side test of its own yet (this Mac
// has no CUDA toolchain -- see tests/bench_dsv4_score_cpu_vs_gpu.cu for the
// bench to run on real hardware before enabling dsv4_score_gpu by default).

#include <cuda_runtime.h>

__global__ void dsv4_score_softmax_kernel(
    float* __restrict__ scores,
    const float* __restrict__ sinks,
    int T, float scale)
{
    int h = blockIdx.x;
    extern __shared__ float sh[];
    float* row = scores + (size_t)h * T;
    // Unscaled, matching the CPU reference's `attn_scores[...] = sinks[h]`
    // (sink logits never go through score_key's attn_scale multiply).
    const float sink_logit = sinks ? sinks[h] : -1e30f;

    // scale + local max. The sink logit competes for the row max but is
    // never written into row[] -- it has no matching value row to blend,
    // it only ever contributes to the softmax denominator below.
    float local_max = sink_logit;
    for (int t = threadIdx.x; t < T; t += blockDim.x) {
        float v = row[t] * scale;
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

    // exp + local sum. Thread 0 alone folds in the sink's own exp term so
    // it's counted exactly once toward the shared row_sum reduced below.
    float local_sum = (threadIdx.x == 0 && sinks) ? __expf(sink_logit - row_max) : 0.0f;
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

    // normalize -- row_sum already includes the sink's mass (if any), so
    // the real T weights correctly sum to < 1 when a sink is present,
    // matching the CPU reference's "sink shrinks the real weights, it has
    // no value contribution of its own" comment.
    for (int t = threadIdx.x; t < T; t += blockDim.x) row[t] /= row_sum;
}

extern "C" void launch_dsv4_score_softmax(
    float* scores, const float* sinks, int H, int T, float scale, cudaStream_t stream)
{
    int threads = 256;
    dsv4_score_softmax_kernel<<<H, threads, threads * sizeof(float), stream>>>(
        scores, sinks, T, scale);
}
