// dsv41_indexer_score.cu -- DeepSeek-V4.1 Lightning-indexer block scores for
// single-token decode. For every completed compressed block b:
//   score[b] = sum_h  max(0, dot(q[h], key[b])) * w[h] * scale
// with the exact operation order of the host loop in dsv41_attention_impl:
// the dot is a serial multiply-then-add chain over d (the engine builds with
// -std=c++23, which disables FMA contraction), the head terms are added
// serially over h as score + ((max(0,dot) * w[h]) * scale). Each
// thread owns one block, so every score is bit-identical to the CPU loop
// (checked by tests/test_dsv41_indexer_score.cpp). Top-k stays on the host.

#include <cuda_runtime.h>

__global__ void dsv41_indexer_score_kernel(
    const float* __restrict__ iq, const float* __restrict__ iw,
    const float* __restrict__ keys, int visible, int heads, int klen,
    float scale, float* __restrict__ out)
{
    extern __shared__ float sh[];
    float* qs = sh;                    // [heads * klen]
    float* ws = qs + (size_t)heads * klen;  // [heads]
    for (int i = threadIdx.x; i < heads * klen; i += blockDim.x) qs[i] = iq[i];
    for (int i = threadIdx.x; i < heads; i += blockDim.x) ws[i] = iw[i];
    __syncthreads();
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= visible) return;
    const float* key = keys + (size_t)b * klen;
    float score = 0.0f;
    for (int h = 0; h < heads; ++h) {
        const float* q = qs + (size_t)h * klen;
        float dot = 0.0f;
        for (int d = 0; d < klen; ++d) dot = __fadd_rn(dot, __fmul_rn(q[d], key[d]));
        score = __fadd_rn(score, __fmul_rn(__fmul_rn(fmaxf(0.0f, dot), ws[h]), scale));
    }
    out[b] = score;
}

extern "C" bool launch_dsv41_indexer_score(
    const float* iq, const float* iw, const float* keys, int visible,
    int heads, int klen, float scale, float* out, cudaStream_t stream)
{
    const size_t shared = ((size_t)heads * klen + (size_t)heads) * sizeof(float);
    if (visible <= 0 || heads <= 0 || klen <= 0 || shared > 48 * 1024) return false;
    const int threads = 128;
    dsv41_indexer_score_kernel<<<(visible + threads - 1) / threads, threads, shared, stream>>>(
        iq, iw, keys, visible, heads, klen, scale, out);
    return true;
}
