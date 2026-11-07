// dsv41_head_attention.cu -- single-token DeepSeek-V4.1 per-head attention for
// the compressed (CSA/HCA) layers. One block per head: scores over the sliding
// window keys plus the indexer-selected compressed keys, a softmax that also
// contains the per-head attention-sink logit (no value row), and the weighted
// sum of the key rows. K == V, matching the CPU loop in attention_layer.cpp
// (dsv41_attention_impl). The inverse RoPE of the output's rope slice stays
// on the host. Window and compressed keys are addressed on the device, so only
// the live window and the compressed rows up to the largest selected block
// are uploaded per call.

#include <cuda_runtime.h>

__global__ void dsv41_head_attn_kernel(
    const float* __restrict__ q, const float* __restrict__ win,
    const float* __restrict__ comp, const int* __restrict__ sel, int nsel,
    const float* __restrict__ sinks, int pos, int past, int window, int qh,
    float scale, float* __restrict__ out)
{
    const int h = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int nwarps = blockDim.x >> 5;
    const int n = past + nsel;
    extern __shared__ float sh[];
    float* qs = sh;                        // [qh]
    float* scores = qs + qh;               // [n + 1]
    int* koff = reinterpret_cast<int*>(scores + n + 1);  // [n] row index
    float* red = reinterpret_cast<float*>(koff + n);     // [32]

    for (int d = tid; d < qh; d += blockDim.x) qs[d] = q[(size_t)h * qh + d];
    for (int i = tid; i < n; i += blockDim.x) {
        // Row offsets are element rows into either the window or the
        // compressed buffer; the high bit marks a compressed row.
        if (i < past) {
            int abs_pos = pos - past + 1 + i;
            koff[i] = abs_pos % window;
        } else {
            koff[i] = sel[i - past] | 0x40000000;
        }
    }
    __syncthreads();

    for (int i = warp; i < n; i += nwarps) {
        const int r = koff[i];
        const float* key = (r & 0x40000000)
            ? comp + (size_t)(r & 0x3fffffff) * qh
            : win + (size_t)r * qh;
        float acc = 0.0f;
        for (int j = lane * 4; j < qh; j += 128) {
            const float4 kv = *reinterpret_cast<const float4*>(key + j);
            acc = fmaf(qs[j], kv.x, acc);
            acc = fmaf(qs[j + 1], kv.y, acc);
            acc = fmaf(qs[j + 2], kv.z, acc);
            acc = fmaf(qs[j + 3], kv.w, acc);
        }
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
        if (lane == 0) scores[i] = acc * scale;
    }
    if (tid == 0) scores[n] = sinks ? sinks[h] : -1e30f;  // sink is unscaled
    __syncthreads();

    // softmax over scores[0..n] (sink included in max and denominator)
    float m = -1e30f;
    for (int i = tid; i <= n; i += blockDim.x) m = fmaxf(m, scores[i]);
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_down_sync(0xffffffffu, m, o));
    if (lane == 0) red[warp] = m;
    __syncthreads();
    m = red[0];
    for (int w = 1; w < nwarps; ++w) m = fmaxf(m, red[w]);
    __syncthreads();
    float s = 0.0f;
    for (int i = tid; i <= n; i += blockDim.x) {
        const float e = expf(scores[i] - m);
        scores[i] = e;
        s += e;
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xffffffffu, s, o);
    if (lane == 0) red[warp] = s;
    __syncthreads();
    s = 0.0f;
    for (int w = 0; w < nwarps; ++w) s += red[w];
    const float inv = 1.0f / s;
    __syncthreads();
    for (int i = tid; i < n; i += blockDim.x) scores[i] *= inv;
    __syncthreads();

    if (qh == 2 * (int)blockDim.x) {
        // Each thread owns output d0 = tid and d1 = tid + blockDim.x. The two
        // accumulator chains are independent and each still adds the keys in
        // order i = 0..n-1 with the same fused multiply-add, so every output is
        // bit-identical to the generic loop below; unrolling by 4 issues the
        // key loads before the dependent FMAs to hide memory latency.
        const int d0 = tid, d1 = tid + (int)blockDim.x;
        float a0 = 0.0f, a1 = 0.0f;
        int i = 0;
        for (; i + 4 <= n; i += 4) {
            float k0[4], k1[4], sc[4];
            #pragma unroll
            for (int u = 0; u < 4; ++u) {
                const int r = koff[i + u];
                const float* key = (r & 0x40000000)
                    ? comp + (size_t)(r & 0x3fffffff) * qh
                    : win + (size_t)r * qh;
                k0[u] = key[d0];
                k1[u] = key[d1];
                sc[u] = scores[i + u];
            }
            #pragma unroll
            for (int u = 0; u < 4; ++u) {
                a0 = fmaf(sc[u], k0[u], a0);
                a1 = fmaf(sc[u], k1[u], a1);
            }
        }
        for (; i < n; ++i) {
            const int r = koff[i];
            const float* key = (r & 0x40000000)
                ? comp + (size_t)(r & 0x3fffffff) * qh
                : win + (size_t)r * qh;
            a0 = fmaf(scores[i], key[d0], a0);
            a1 = fmaf(scores[i], key[d1], a1);
        }
        out[(size_t)h * qh + d0] = a0;
        out[(size_t)h * qh + d1] = a1;
        return;
    }
    for (int d = tid; d < qh; d += blockDim.x) {
        float acc = 0.0f;
        for (int i = 0; i < n; ++i) {
            const int r = koff[i];
            const float* key = (r & 0x40000000)
                ? comp + (size_t)(r & 0x3fffffff) * qh
                : win + (size_t)r * qh;
            acc = fmaf(scores[i], key[d], acc);
        }
        out[(size_t)h * qh + d] = acc;
    }
}

extern "C" void launch_dsv41_head_attention(
    const float* q, const float* win, const float* comp, const int* sel,
    int nsel, const float* sinks, int pos, int past, int window, int qh,
    int H, float scale, float* out, cudaStream_t stream)
{
    const int n = past + nsel;
    const size_t shared = ((size_t)qh + (size_t)n + 1 + (size_t)n + 32) * sizeof(float);
    dsv41_head_attn_kernel<<<H, 256, shared, stream>>>(
        q, win, comp, sel, nsel, sinks, pos, past, window, qh, scale, out);
}
