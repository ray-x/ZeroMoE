// expert_accum.cu -- GPU-side scatter-accumulation of routed-expert outputs.
//
// The batched prefill path used to download EVERY expert's [n,D] output to the
// host and do the weighted add in a CPU loop:
//     for (i < n) for (d < D) dst[d] += w * src[d];
// At qwen4exp's real shape that is 3511 tokens x 48 layers x 10 experts =
// 1.685M expert-token rows x D=2560 floats = ~16.9 GB of device->host traffic
// per prefill, measured at dn=16.81s -- against cmp=0.27s for the expert
// matmuls themselves. llama.cpp keeps this accumulation inside the graph, so
// its expert outputs never cross PCIe at all.
//
// Doing it here instead leaves one [S,D] download per LAYER (36 MB) in place of
// per-expert downloads: ~1.7 GB total, a ~10x cut.
//
// NO atomics, deliberately: within one expert's bucket every token row is
// distinct (top-k picks distinct experts, so a token cannot appear twice for
// the same expert), and experts are dispatched sequentially on one stream, so
// cross-expert writes to the same row are already ordered by the stream.

#include <cuda_runtime.h>

__global__ void expert_scatter_accum_kernel(
    const float* __restrict__ src,   // [n, D] this expert's output rows
    const int*   __restrict__ tok,   // [n] destination token row for each
    const float* __restrict__ w,     // [n] router weight for each
    float*       __restrict__ dst,   // [S, D] layer accumulator
    int n, int D, int S)
{
    const int i = blockIdx.x;
    if (i >= n) return;
    const int s = tok[i];
    if (s < 0 || s >= S) return;     // defensive: a bad index must not corrupt
    const float wt = w ? w[i] : 1.0f;
    const float* sr = src + (size_t)i * D;
    float* ds = dst + (size_t)s * D;
    for (int d = threadIdx.x; d < D; d += blockDim.x) ds[d] += wt * sr[d];
}

extern "C" void launch_expert_scatter_accum(
    const float* src, const int* tok, const float* w, float* dst,
    int n, int D, int S, cudaStream_t stream)
{
    if (n <= 0 || D <= 0) return;
    const int threads = (D < 256) ? 64 : 256;
    expert_scatter_accum_kernel<<<n, threads, 0, stream>>>(src, tok, w, dst, n, D, S);
}

// The chunked GGML MoE path applies router weights inside its graph, then
// needs only an unweighted scatter-add of the resulting [D,pairs] rows. Keep
// that operation on the same tiny kernel so the output never returns to the
// host between expert chunks.
extern "C" void launch_expert_scatter_add(
    const float *src, const int *tok, float *dst, int n, int D, int S,
    cudaStream_t stream)
{
    if (n <= 0 || D <= 0) return;
    const int threads = (D < 256) ? 64 : 256;
    expert_scatter_accum_kernel<<<n, threads, 0, stream>>>(
        src, tok, nullptr, dst, n, D, S);
}
