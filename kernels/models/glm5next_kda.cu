// GLM5NEXT KDA recurrent delta rule.
//
// GLM5NEXT's KDA state is [head, key, value] and its decay is per key
// channel. That is not the same recurrence as Qwen4Exp's scalar-decay GDN,
// so it cannot use qwen4exp_gdn.cu's chunkwise closed form. Tokens remain
// sequential inside each head, but all heads and all state elements within a
// token run in parallel on CUDA.

#include <cuda_runtime.h>
#include <cmath>

extern "C" {

__global__ void glm5next_kda_delta_rule_kernel(
    const float* __restrict__ q,       // [S,H,hd]
    const float* __restrict__ k,       // [S,H,hd]
    const float* __restrict__ v,       // [S,H,hd]
    const float* __restrict__ beta,    // [S,H]
    const float* __restrict__ g_log,   // [S,H,hd]
    float* __restrict__ state,         // [H,hd,hd], [key,value]
    float* __restrict__ out,           // [S,H,hd]
    int S, int H, int hd)
{
    const int h = blockIdx.x;
    if (h >= H) return;

    extern __shared__ float sk[];
    float* st = state + (size_t)h * hd * hd;
    const float q_scale = 1.0f / sqrtf((float)hd);

    // The recurrent dependency is only across tokens. Every block owns one
    // head, and every phase has a barrier so the same state matrix is used by
    // all participating threads before the next phase starts.
    for (int s = 0; s < S; ++s) {
        const float* q_h = q + ((size_t)s * H + h) * hd;
        const float* k_h = k + ((size_t)s * H + h) * hd;
        const float* v_h = v + ((size_t)s * H + h) * hd;
        const float* g_h = g_log + ((size_t)s * H + h) * hd;
        float* o_h = out + ((size_t)s * H + h) * hd;
        const float b = beta[(size_t)s * H + h];

        // state[key,value] *= exp(g_log[key])
        for (int idx = threadIdx.x; idx < hd * hd; idx += blockDim.x) {
            const int key = idx / hd;
            st[idx] *= expf(g_h[key]);
        }
        __syncthreads();

        // sk[value] = sum_key state[key,value] * k[key]
        for (int value = threadIdx.x; value < hd; value += blockDim.x) {
            float sum = 0.0f;
            for (int key = 0; key < hd; ++key)
                sum += st[key * hd + value] * k_h[key];
            sk[value] = sum;
        }
        __syncthreads();

        // state[key,value] += k[key] * (v[value] - sk[value]) * beta
        for (int idx = threadIdx.x; idx < hd * hd; idx += blockDim.x) {
            const int key = idx / hd;
            const int value = idx - key * hd;
            st[idx] += k_h[key] * (v_h[value] - sk[value]) * b;
        }
        __syncthreads();

        // out[value] = sum_key state[key,value] * q[key] / sqrt(hd)
        for (int value = threadIdx.x; value < hd; value += blockDim.x) {
            float sum = 0.0f;
            for (int key = 0; key < hd; ++key)
                sum += st[key * hd + value] * q_h[key] * q_scale;
            o_h[value] = sum;
        }
        __syncthreads();
    }
}

void launch_glm5next_kda_delta_rule(
    const float* q, const float* k, const float* v, const float* beta,
    const float* g_log, float* state, float* out,
    int S, int H, int hd, cudaStream_t stream)
{
    if (S <= 0 || H <= 0 || hd <= 0) return;
    constexpr int block = 128;
    glm5next_kda_delta_rule_kernel<<<H, block, (size_t)hd * sizeof(float), stream>>>(
        q, k, v, beta, g_log, state, out, S, H, hd);
}

} // extern "C"
