// mla_attention.cu — GPU MLA attention kernel with on-the-fly Q4_0 KV dequantization
// Implements the absorbed Multi-head Latent Attention for GLM-5.2 / DeepSeek-V3.
// KV cache stored as FP32 latent vectors (Lc) and rope vectors (Rc) on device.
// Query absorption is done via the kv_b weight (device_ptr).
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>

// Q4_0 block for weight dequantization (matches kernels_cpu.hpp)
struct block_q4_0_cuda {
    uint16_t d;       // FP16 scale
    uint8_t  qs[16];  // 32 x 4-bit values packed
};

// ============================================================
// Kernel: Compute attention scores for one head/token pair
// Each block handles one (sequence_pos, head) pair.
// Threads cooperatively compute: score = q_absorbed · L_t + q_rope · R_t
// ============================================================
__global__ void mla_score_kernel(
    float* __restrict__ attn_scores,  // [S, H, max_t]
    const float* __restrict__ q_absorbed,  // [S, H, kv_lora] — pre-absorbed queries
    const float* __restrict__ q_rope,      // [S, H, qk_rope]
    const float* __restrict__ Lc,          // [max_t, kv_lora] — latent cache for this layer
    const float* __restrict__ Rc,          // [max_t, qk_rope] — rope cache for this layer
    int S, int H, int kv_lora, int qk_rope, int context_len,
    float attn_scale
) {
    int s = blockIdx.y;
    int h = blockIdx.x;
    int tid = threadIdx.x;
    if (s >= S || h >= H) return;

    int current_pos = s;  // caller adjusts for pos_base externally
    int past_len = current_pos + 1;
    if (past_len > context_len) past_len = context_len;

    const float* q_abs = q_absorbed + ((int64_t)s * H + h) * kv_lora;
    const float* q_rp = q_rope + ((int64_t)s * H + h) * qk_rope;

    // Each thread handles a subset of time steps
    for (int t = tid; t < past_len; t += blockDim.x) {
        const float* L_t = Lc + (int64_t)t * kv_lora;
        const float* R_t = Rc + (int64_t)t * qk_rope;

        // Dot product: q_absorbed · L_t
        float score_nope = 0.0f;
        for (int i = 0; i < kv_lora; ++i) {
            score_nope += q_abs[i] * L_t[i];
        }

        // Dot product: q_rope · R_t
        float score_rope = 0.0f;
        for (int i = 0; i < qk_rope; ++i) {
            score_rope += q_rp[i] * R_t[i];
        }

        attn_scores[((int64_t)s * H + h) * context_len + t] = (score_nope + score_rope) * attn_scale;
    }
}

// ============================================================
// Kernel: Softmax over attention scores (per head)
// ============================================================
__global__ void mla_softmax_kernel(
    float* __restrict__ scores,  // [S, H, max_t]
    int S, int H, int context_len,
    const int* __restrict__ seq_lens  // [S] actual length per sequence position
) {
    int s = blockIdx.y;
    int h = blockIdx.x;
    if (s >= S || h >= H) return;

    int len = seq_lens[s];
    float* row = scores + ((int64_t)s * H + h) * context_len;

    // Find max
    float max_val = -1e30f;
    for (int t = threadIdx.x; t < len; t += blockDim.x) {
        if (row[t] > max_val) max_val = row[t];
    }
    // Warp reduction for max
    for (int offset = 16; offset > 0; offset /= 2)
        max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, offset));
    __shared__ float s_max;
    if (threadIdx.x == 0) s_max = max_val;
    __syncthreads();
    max_val = s_max;

    // Exp and sum
    float sum = 0.0f;
    for (int t = threadIdx.x; t < len; t += blockDim.x) {
        row[t] = expf(row[t] - max_val);
        sum += row[t];
    }
    // Warp reduction for sum
    for (int offset = 16; offset > 0; offset /= 2)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    __shared__ float s_sum;
    if (threadIdx.x == 0) s_sum = sum;
    __syncthreads();
    sum = s_sum;

    // Normalize
    float inv_sum = 1.0f / sum;
    for (int t = threadIdx.x; t < len; t += blockDim.x) {
        row[t] *= inv_sum;
    }
}

// ============================================================
// Kernel: Weighted sum of latent vectors using attention weights
// output[s,h] = sum_t(alpha_t * L_t) — this gives the aggregated latent
// ============================================================
__global__ void mla_value_aggregate_kernel(
    float* __restrict__ latent_out,  // [S, H, kv_lora]
    const float* __restrict__ scores, // [S, H, max_t]
    const float* __restrict__ Lc,     // [max_t, kv_lora]
    int S, int H, int kv_lora, int context_len,
    const int* __restrict__ seq_lens
) {
    int s = blockIdx.y;
    int h = blockIdx.x;
    int tid = threadIdx.x;
    if (s >= S || h >= H) return;

    int len = seq_lens[s];
    const float* alpha = scores + ((int64_t)s * H + h) * context_len;
    float* out = latent_out + ((int64_t)s * H + h) * kv_lora;

    // Each thread accumulates a range of kv_lora dimensions
    for (int i = tid; i < kv_lora; i += blockDim.x) {
        float acc = 0.0f;
        for (int t = 0; t < len; ++t) {
            acc += alpha[t] * Lc[(int64_t)t * kv_lora + i];
        }
        out[i] = acc;
    }
}

// ============================================================
// Host-callable launcher for the full MLA attention forward pass
// ============================================================
extern "C" void launch_mla_attention(
    float* d_out_context,      // [S, H*vh] output
    const float* d_q_absorbed, // [S, H, kv_lora] pre-absorbed queries
    const float* d_q_rope,     // [S, H, qk_rope] rope queries
    const float* d_Lc,         // [max_t, kv_lora] device latent cache
    const float* d_Rc,         // [max_t, qk_rope] device rope cache
    float* d_scores,           // [S, H, max_t] scratch for attention scores
    float* d_latent_out,       // [S, H, kv_lora] scratch for aggregated latent
    int* d_seq_lens,           // [S] sequence lengths
    int S, int H, int kv_lora, int qk_rope, int vh,
    int context_len, float attn_scale
) {
    // Phase 1: Compute attention scores
    {
        dim3 grid(H, S);
        dim3 block(32);  // one warp per (h,s) pair, iterates over time
        mla_score_kernel<<<grid, block>>>(
            d_scores, d_q_absorbed, d_q_rope, d_Lc, d_Rc,
            S, H, kv_lora, qk_rope, context_len, attn_scale
        );
    }

    // Phase 2: Softmax
    {
        dim3 grid(H, S);
        dim3 block(32);
        mla_softmax_kernel<<<grid, block>>>(d_scores, S, H, context_len, d_seq_lens);
    }

    // Phase 3: Weighted aggregation of latent vectors
    {
        dim3 grid(H, S);
        dim3 block(128);  // 128 threads per head to cover kv_lora dims
        mla_value_aggregate_kernel<<<grid, block>>>(
            d_latent_out, d_scores, d_Lc,
            S, H, kv_lora, context_len, d_seq_lens
        );
    }

    // Phase 4: The kv_b projection (latent → value) and output projection
    // are done by the caller using the quantized matmul kernels.
    // d_latent_out [S, H, kv_lora] → project via kv_b → [S, H, vh] → concat → output matmul
}
