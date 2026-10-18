#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>

struct block_q4_0 {
    half d;           
    uint8_t qs[16];   
};

__global__ void gguf_q4_0_dp4a_matmul(float* __restrict__ y,
                                       const float* __restrict__ x,
                                       const block_q4_0* __restrict__ weights,
                                       int S, int I, int O) 
{
    // A thread block handles 1 Sequence position and a tile of 4 Output rows.
    int s = blockIdx.y;
    int o_base = blockIdx.x * 4; 
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int o = o_base + warp_id;
    
    int blocks_per_row = I / 32;
    const float* x_row = x + (s * I);
    
    // Shared memory cache for quantizing a 1024-element chunk of X at a time
    __shared__ int8_t s_x_q8[1024]; 
    __shared__ float s_x_scale[32];
    
    float row_sum = 0.0f;
    
    // Process I in chunks of 32 blocks (1024 elements) to keep shared memory tiny and fast
    for (int chunk_start = 0; chunk_start < blocks_per_row; chunk_start += 32) {
        int chunk_blocks = min(32, blocks_per_row - chunk_start);
        
        // --- PHASE 1: Cooperative Shared Memory Quantization ---
        // All 128 threads work together to quantize the activation chunk exactly once
        for (int b_in_chunk = warp_id; b_in_chunk < chunk_blocks; b_in_chunk += 4) {
            int b_global = chunk_start + b_in_chunk;
            float x_val = x_row[b_global * 32 + lane_id];
            float abs_val = fabsf(x_val);
            
            // Fast warp reduction for amax
            float amax = abs_val;
            for (int offset = 16; offset > 0; offset /= 2) {
                amax = fmaxf(amax, __shfl_down_sync(0xffffffff, amax, offset));
            }
            
            float x_scale = amax / 127.0f;
            if (x_scale < 1e-12f) x_scale = 1e-12f;
            float inv_scale = 1.0f / x_scale;
            
            s_x_q8[b_in_chunk * 32 + lane_id] = (int8_t)rintf(x_val * inv_scale);
            if (lane_id == 0) s_x_scale[b_in_chunk] = x_scale;
        }
        __syncthreads(); // Wait for the chunk to be fully cached
        
        // --- PHASE 2: Fused Warp-Level DP4A Matmul ---
        // Each warp is assigned to one of the 4 output rows
        if (o < O) {
            const block_q4_0* w_row = weights + (o * blocks_per_row);
            
            // Warp processes 8 Q4_0 blocks concurrently (4 threads per block)
            for (int b_base = 0; b_base < chunk_blocks; b_base += 8) {
                int b_in_chunk = b_base + (lane_id / 4);
                float scaled_dot = 0.0f;
                
                if (b_in_chunk < chunk_blocks) {
                    int b_global = chunk_start + b_in_chunk;
                    
                    // Safely load 4 contiguous weight bytes (avoids misaligned struct crashes)
                    const uint8_t* q_ptr = &w_row[b_global].qs[(lane_id % 4) * 4];
                    uint32_t raw_w = q_ptr[0] | (q_ptr[1] << 8) | (q_ptr[2] << 16) | (q_ptr[3] << 24);
                    
                    // Aligned vectorized read of 4 packed int8 activations from shared memory
                    int x_low = *(const int*)&s_x_q8[b_in_chunk * 32 + (lane_id % 4) * 4];
                    int x_high = *(const int*)&s_x_q8[b_in_chunk * 32 + 16 + (lane_id % 4) * 4];
                    
                    int w_low = raw_w & 0x0F0F0F0F;
                    int w_high = (raw_w >> 4) & 0x0F0F0F0F;
                    
                    // Mathematical shift trick: 
                    // dot(w - 8, x) == dot(w, x) - 8 * sum(x)
                    // We sum 'x' insanely fast using dp4a with an all-ones vector
                    int ones = 0x01010101;
                    int x_sum = __dp4a(ones, x_low, 0) + __dp4a(ones, x_high, 0);
                    
                    int dot = 0;
                    dot = __dp4a(w_low, x_low, dot);
                    dot = __dp4a(w_high, x_high, dot);
                    dot -= 8 * x_sum; // Applies the -8 scale shift without touching the weight bits
                    
                    float scale = __half2float(w_row[b_global].d) * s_x_scale[b_in_chunk];
                    scaled_dot = dot * scale;
                }
                
                // Reduce the 4 threads down to the block total
                scaled_dot += __shfl_down_sync(0xffffffff, scaled_dot, 2);
                scaled_dot += __shfl_down_sync(0xffffffff, scaled_dot, 1);
                
                if (lane_id % 4 == 0) row_sum += scaled_dot;
            }
        }
        __syncthreads(); // Sync before overwriting shared memory with the next chunk
    }
    
    // --- PHASE 3: Final Output Reduction ---
    if (o < O) {
        // Zero out threads that didn't accumulate a chunk sum
        if (lane_id % 4 != 0) row_sum = 0.0f;
        
        // Full warp reduction to get the final row value
        for (int offset = 16; offset > 0; offset /= 2) {
            row_sum += __shfl_down_sync(0xffffffff, row_sum, offset);
        }
        
        if (lane_id == 0) {
            y[s * O + o] = row_sum;
        }
    }
}

extern "C" void launch_gguf_q4_0_dp4a(float* y, const float* x, const void* weights, int S, int I, int O) {
    // 4 rows per block, 128 threads (4 warps) per block
    dim3 grid((O + 3) / 4, S);
    dim3 block(128);
    gguf_q4_0_dp4a_matmul<<<grid, block>>>(y, x, (const block_q4_0*)weights, S, I, O);
}