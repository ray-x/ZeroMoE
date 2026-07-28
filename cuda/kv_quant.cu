// kv_quant.cu — 4-bit KV cache quantization and dequantization kernels
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

// Q4_0 block: 32 values packed into 18 bytes
struct BlockQ4_KV {
    half d;           
    uint8_t qs[16];   
};

// ============================================================
// Quantize FP32 → Q4_0 (Optimized: Single Global Read)
// ============================================================
__global__ void kv_quantize_q4_kernel(
    BlockQ4_KV* __restrict__ output,
    const float* __restrict__ input,
    int n_elements  
) {
    int block_idx = blockIdx.x;
    int tid = threadIdx.x;  // 0..31
    int base = block_idx * 32;

    if (base + 31 >= n_elements) return;

    // 1. Load exactly once from global memory and cache
    float val = input[base + tid];
    __shared__ float s_vals[32];
    s_vals[tid] = val;
    __syncthreads(); 

    float abs_val = fabsf(val);

    // 2. Warp reduction for max
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        abs_val = fmaxf(abs_val, __shfl_down_sync(0xffffffff, abs_val, offset));
    }
    
    // Broadcast amax to all threads in the warp directly
    float amax = __shfl_sync(0xffffffff, abs_val, 0);

    float scale = amax / 7.0f;  
    if (scale < 1e-12f) scale = 1e-12f;
    float inv_scale = 1.0f / scale;

    // 3. Threads 0-15 pack the values (coalesced write)
    if (tid < 16) {
        int q_lo = __float2int_rn(s_vals[tid * 2] * inv_scale) + 8;
        int q_hi = __float2int_rn(s_vals[tid * 2 + 1] * inv_scale) + 8;
        q_lo = max(0, min(15, q_lo));
        q_hi = max(0, min(15, q_hi));
        
        output[block_idx].qs[tid] = (uint8_t)((q_hi << 4) | q_lo);
        
        if (tid == 0) {
            output[block_idx].d = __float2half(scale);
        }
    }
}

// ============================================================
// Dequantize Q4_0 → FP32 (Optimized: Cooperative Load)
// ============================================================
__global__ void kv_dequantize_q4_kernel(
    float* __restrict__ output,
    const BlockQ4_KV* __restrict__ input,
    int n_blocks
) {
    int block_idx = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    int lane = threadIdx.x % 32;
    if (block_idx >= n_blocks) return;

    float scale = __half2float(input[block_idx].d);
    
    // Threads 0-15 cleanly read the 16 bytes (no duplicated access)
    uint8_t my_byte = 0;
    if (lane < 16) {
        my_byte = input[block_idx].qs[lane];
    }
    
    // Broadcast byte `k` to threads `k*2` and `k*2+1`
    uint8_t packed = __shfl_sync(0xffffffff, my_byte, lane / 2);
    
    int q_val = (lane % 2 == 0) ? (packed & 0x0F) : (packed >> 4);
    output[block_idx * 32 + lane] = ((float)q_val - 8.0f) * scale;
}

// ============================================================
// Fused Dot Product (Optimized: 1 Warp = 1 Block)
// ============================================================
__global__ void kv_q4_dot_kernel(
    float* __restrict__ scores,       
    const float* __restrict__ queries, 
    const BlockQ4_KV* __restrict__ kv_cache, 
    int n_queries, int dim, int context_len,
    float scale_factor
) {
    int q_idx = blockIdx.y;
    int t_idx = blockIdx.x;
    int tid = threadIdx.x;
    int lane = tid % 32;
    int warp = tid / 32;
    int num_warps = blockDim.x / 32;

    if (q_idx >= n_queries || t_idx >= context_len) return;

    int blocks_per_entry = dim / 32;
    const float* q = queries + (int64_t)q_idx * dim;
    const BlockQ4_KV* kv_row = kv_cache + (int64_t)t_idx * blocks_per_entry;

    float partial_sum = 0.0f;

    // Warp cooperatively processes Q4_0 blocks
    for (int b = warp; b < blocks_per_entry; b += num_warps) {
        float kv_scale = __half2float(kv_row[b].d);
        
        // Cooperative coalesced read of 16-byte quantized data
        uint8_t my_byte = 0;
        if (lane < 16) {
            my_byte = kv_row[b].qs[lane];
        }
        uint8_t packed = __shfl_sync(0xffffffff, my_byte, lane / 2);
        
        int q_val = (lane % 2 == 0) ? (packed & 0x0F) : (packed >> 4);
        float kv_val = ((float)q_val - 8.0f) * kv_scale;

        // Perfectly coalesced 32-float read for queries
        float q_val_f = q[b * 32 + lane];
        
        partial_sum += q_val_f * kv_val;
    }

    // Fast Warp Reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        partial_sum += __shfl_down_sync(0xffffffff, partial_sum, offset);
    }

    // Collect warp sums and reduce across the block
    __shared__ float s_sum[32];
    if (lane == 0) s_sum[warp] = partial_sum;
    __syncthreads();

    if (warp == 0) {
        float block_sum = (lane < num_warps) ? s_sum[lane] : 0.0f;
        
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            block_sum += __shfl_down_sync(0xffffffff, block_sum, offset);
        }
        
        if (lane == 0) {
            scores[(int64_t)q_idx * context_len + t_idx] = block_sum * scale_factor;
        }
    }
}

// ============================================================
// Host launchers
// ============================================================
extern "C" void launch_kv_quantize_q4(void* output, const float* input, int n_elements) {
    int n_blocks = n_elements / 32;
    kv_quantize_q4_kernel<<<n_blocks, 32>>>((BlockQ4_KV*)output, input, n_elements);
}

extern "C" void launch_kv_dequantize_q4(float* output, const void* input, int n_elements) {
    int n_blocks = n_elements / 32;
    int threads = 256;
    int grid = (n_blocks * 32 + threads - 1) / threads;
    kv_dequantize_q4_kernel<<<grid, threads>>>(output, (const BlockQ4_KV*)input, n_blocks);
}

extern "C" void launch_kv_q4_dot(float* scores, const float* queries, const void* kv_cache,
                                  int n_queries, int dim, int context_len, float scale) {
    dim3 grid(context_len, n_queries);
    dim3 block(128); // Safely sets 4 warps per block
    kv_q4_dot_kernel<<<grid, block>>>(scores, queries, (const BlockQ4_KV*)kv_cache,
                                       n_queries, dim, context_len, scale);
}