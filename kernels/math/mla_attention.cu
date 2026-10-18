// mla_attention.cu — GPU MLA attention kernel with on-the-fly Q4_0 KV dequantization
// Implements the absorbed Multi-head Latent Attention for GLM-5.2 / DeepSeek-V3.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>

struct block_q4_0_cuda {
    uint16_t d;       
    uint8_t  qs[16];  
};

// ============================================================
// Kernel 1: Score Kernel (Warp-Coalesced)
// ============================================================
__global__ void mla_score_kernel(
    float* __restrict__ attn_scores,  
    const float* __restrict__ q_absorbed,  
    const float* __restrict__ q_rope,      
    const float* __restrict__ Lc,          
    const float* __restrict__ Rc,          
    int S, int H, int kv_lora, int qk_rope, int context_len,
    float attn_scale
) {
    int s = blockIdx.y;
    int h = blockIdx.x;
    if (s >= S || h >= H) return;

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int num_warps = blockDim.x / 32;

    int past_len = s + 1;
    if (past_len > context_len) past_len = context_len;

    // Cache the query vectors in shared memory to avoid redundant global reads
    extern __shared__ float s_q[];
    float* s_q_abs = s_q;
    float* s_q_rp = s_q + kv_lora;

    const float* global_q_abs = q_absorbed + ((int64_t)s * H + h) * kv_lora;
    const float* global_q_rp = q_rope + ((int64_t)s * H + h) * qk_rope;

    for (int i = threadIdx.x; i < kv_lora; i += blockDim.x) s_q_abs[i] = global_q_abs[i];
    for (int i = threadIdx.x; i < qk_rope; i += blockDim.x) s_q_rp[i] = global_q_rp[i];
    __syncthreads();

    // Each WARP handles a distinct time step 't' to ensure perfectly coalesced memory reads
    for (int t = warp_id; t < past_len; t += num_warps) {
        const float* L_t = Lc + (int64_t)t * kv_lora;
        const float* R_t = Rc + (int64_t)t * qk_rope;

        float dot = 0.0f;
        
        // 32 threads stride across the continuous kv_lora/qk_rope memory chunks
        for (int i = lane_id; i < kv_lora; i += 32) {
            dot += s_q_abs[i] * L_t[i];
        }
        for (int i = lane_id; i < qk_rope; i += 32) {
            dot += s_q_rp[i] * R_t[i];
        }

        // Fast warp reduction for the final dot product sum
        for (int offset = 16; offset > 0; offset /= 2) {
            dot += __shfl_down_sync(0xffffffff, dot, offset);
        }

        if (lane_id == 0) {
            attn_scores[((int64_t)s * H + h) * context_len + t] = dot * attn_scale;
        }
    }
}

// ============================================================
// Kernel 2: Softmax Kernel (Block-Level Reduction)
// ============================================================
__global__ void mla_softmax_kernel(
    float* __restrict__ scores,  
    int S, int H, int context_len,
    const int* __restrict__ seq_lens  
) {
    int s = blockIdx.y;
    int h = blockIdx.x;
    if (s >= S || h >= H) return;

    int len = seq_lens[s];
    float* row = scores + ((int64_t)s * H + h) * context_len;

    int lane = threadIdx.x % 32;
    int warp = threadIdx.x / 32;
    __shared__ float s_reduce[32];

    // --- Phase 1: Max Reduction ---
    float max_val = -1e30f;
    for (int t = threadIdx.x; t < len; t += blockDim.x) {
        if (row[t] > max_val) max_val = row[t];
    }
    
    for (int offset = 16; offset > 0; offset /= 2) 
        max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, offset));
    
    if (lane == 0) s_reduce[warp] = max_val;
    __syncthreads();
    
    max_val = (threadIdx.x < blockDim.x / 32) ? s_reduce[lane] : -1e30f;
    if (warp == 0) {
        for (int offset = 16; offset > 0; offset /= 2) 
            max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, offset));
        s_reduce[0] = max_val; // Broadcast via smem
    }
    __syncthreads();
    max_val = s_reduce[0];

    // --- Phase 2: Exp and Sum Reduction ---
    float sum = 0.0f;
    for (int t = threadIdx.x; t < len; t += blockDim.x) {
        float e = expf(row[t] - max_val);
        row[t] = e;
        sum += e;
    }
    
    for (int offset = 16; offset > 0; offset /= 2) 
        sum += __shfl_down_sync(0xffffffff, sum, offset);
        
    if (lane == 0) s_reduce[warp] = sum;
    __syncthreads();
    
    sum = (threadIdx.x < blockDim.x / 32) ? s_reduce[lane] : 0.0f;
    if (warp == 0) {
        for (int offset = 16; offset > 0; offset /= 2) 
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        s_reduce[0] = sum;
    }
    __syncthreads();
    sum = s_reduce[0];

    // --- Phase 3: Normalize ---
    float inv_sum = 1.0f / sum;
    for (int t = threadIdx.x; t < len; t += blockDim.x) {
        row[t] *= inv_sum;
    }
}

// ============================================================
// Kernel 3: Value Aggregate (Higher Occupancy)
// ============================================================
__global__ void mla_value_aggregate_kernel(
    float* __restrict__ latent_out,  
    const float* __restrict__ scores, 
    const float* __restrict__ Lc,     
    int S, int H, int kv_lora, int context_len,
    const int* __restrict__ seq_lens
) {
    int s = blockIdx.y;
    int h = blockIdx.x;
    if (s >= S || h >= H) return;

    int len = seq_lens[s];
    const float* alpha = scores + ((int64_t)s * H + h) * context_len;
    float* out = latent_out + ((int64_t)s * H + h) * kv_lora;

    // Already coalesced. Thread iterates sequence length for a specific kv_lora slice.
    for (int i = threadIdx.x; i < kv_lora; i += blockDim.x) {
        float acc = 0.0f;
        for (int t = 0; t < len; ++t) {
            acc += alpha[t] * Lc[(int64_t)t * kv_lora + i];
        }
        out[i] = acc;
    }
}

extern "C" void launch_mla_attention(
    float* d_out_context,      
    const float* d_q_absorbed, 
    const float* d_q_rope,     
    const float* d_Lc,         
    const float* d_Rc,         
    float* d_scores,           
    float* d_latent_out,       
    int* d_seq_lens,           
    int S, int H, int kv_lora, int qk_rope, int vh,
    int context_len, float attn_scale
) {
    // Increased block sizes for vastly improved Streaming Multiprocessor (SM) occupancy
    int threads_per_block = 256; 

    // Phase 1: Score
    {
        dim3 grid(H, S);
        dim3 block(threads_per_block); 
        size_t shared_mem_size = (kv_lora + qk_rope) * sizeof(float);
        mla_score_kernel<<<grid, block, shared_mem_size>>>(
            d_scores, d_q_absorbed, d_q_rope, d_Lc, d_Rc,
            S, H, kv_lora, qk_rope, context_len, attn_scale
        );
    }

    // Phase 2: Softmax
    {
        dim3 grid(H, S);
        dim3 block(threads_per_block);
        mla_softmax_kernel<<<grid, block>>>(d_scores, S, H, context_len, d_seq_lens);
    }

    // Phase 3: Aggregation
    {
        dim3 grid(H, S);
        dim3 block(threads_per_block);  
        mla_value_aggregate_kernel<<<grid, block>>>(
            d_latent_out, d_scores, d_Lc,
            S, H, kv_lora, context_len, d_seq_lens
        );
    }
}