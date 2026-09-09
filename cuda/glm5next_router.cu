// Native GLM5NEXT router post-processing.
//
// The router matmul already has a CUDA implementation through
// CudaManager::dense_matmul_device. This kernel keeps the rest of the normal
// learned-router path on CUDA too: gating activation, selection-bias ranking,
// unbiased weight gather, top-k normalization, and expert-weight scaling.
// One block handles one token. The expert count is small enough that a
// deterministic thread-0 top-k scan is preferable to a larger reduction
// implementation, while different prompt tokens still run concurrently.

#include <cuda_runtime.h>
#include <cmath>

namespace {

__device__ float router_gate(float x, int gating_func) {
    // GatingFunc values from gguf_loader.hpp:
    // Softmax=0, Sigmoid=1, SqrtSoftplus=2.
    if (gating_func == 1) return 1.0f / (1.0f + expf(-x));
    if (gating_func == 2) {
        const float softplus = (x > 0.0f)
            ? (x + log1pf(expf(-x)))
            : log1pf(expf(x));
        return sqrtf(softplus);
    }
    return x; // softmax is applied row-wise below
}

} // namespace

extern "C" {

__global__ void glm5next_router_topk_kernel(
    float* __restrict__ logits,       // [S,n_experts], overwritten with probs
    const float* __restrict__ bias,    // [n_experts], optional zero row
    int32_t* __restrict__ ids,         // [S,topk]
    float* __restrict__ weights,       // [S,topk]
    int S, int n_experts, int topk,
    int gating_func, int normalize, float weight_scale)
{
    const int s = blockIdx.x;
    if (s >= S || threadIdx.x != 0) return;

    float* row = logits + (size_t)s * n_experts;
    if (gating_func == 0) {
        float max_v = row[0];
        for (int e = 1; e < n_experts; ++e) max_v = fmaxf(max_v, row[e]);
        float sum = 0.0f;
        for (int e = 0; e < n_experts; ++e) {
            row[e] = expf(row[e] - max_v);
            sum += row[e];
        }
        const float inv = 1.0f / sum;
        for (int e = 0; e < n_experts; ++e) row[e] *= inv;
    } else {
        for (int e = 0; e < n_experts; ++e)
            row[e] = router_gate(row[e], gating_func);
    }

    float sum_w = 0.0f;
    for (int k = 0; k < topk; ++k) {
        int best = -1;
        constexpr float neg_inf = -3.402823466e+38F;
        float best_score = neg_inf;
        for (int e = 0; e < n_experts; ++e) {
            const float score = row[e] + (bias ? bias[e] : 0.0f);
            if (best < 0 || score > best_score ||
                (score == best_score && e < best)) {
                best = e;
                best_score = score;
            }
        }
        if (best < 0) {
            ids[(size_t)s * topk + k] = 0;
            weights[(size_t)s * topk + k] = 0.0f;
            continue;
        }
        ids[(size_t)s * topk + k] = best;
        weights[(size_t)s * topk + k] = row[best];
        row[best] = neg_inf;
        sum_w += weights[(size_t)s * topk + k];
    }

    if (normalize) {
        const float inv = 1.0f / fmaxf(sum_w, 6.103515625e-5f);
        for (int k = 0; k < topk; ++k)
            weights[(size_t)s * topk + k] *= inv;
    }
    if (weight_scale != 0.0f) {
        for (int k = 0; k < topk; ++k)
            weights[(size_t)s * topk + k] *= weight_scale;
    }
}

void launch_glm5next_router_topk(float* logits, const float* bias,
                                 int32_t* ids, float* weights,
                                 int S, int n_experts, int topk,
                                 int gating_func, int normalize,
                                 float weight_scale, cudaStream_t stream)
{
    if (S <= 0 || n_experts <= 0 || topk <= 0 || topk > n_experts) return;
    glm5next_router_topk_kernel<<<S, 256, 0, stream>>>(
        logits, bias, ids, weights, S, n_experts, topk,
        gating_func, normalize, weight_scale);
}

} // extern "C"
