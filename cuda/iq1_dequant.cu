// iq1_dequant.cu — Optimized 1-bit and Ternary quantization CUDA kernels
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

// ============================================================
// TQ1_0 Fused Matmul Kernel (Optimized)
// ============================================================
__global__ void tq1_matmul_kernel_optimized(
    float* __restrict__ y,
    const float* __restrict__ x,
    const uint8_t* __restrict__ weights,
    const half* __restrict__ scales,
    int S, int I, int O
) {
    int o = blockIdx.x;
    int s = blockIdx.y;
    int tid = threadIdx.x;
    if (o >= O || s >= S) return;

    int blocks_per_row = I / 256;
    const int bytes_per_block = 54;
    const uint8_t* w_row = weights + (size_t)o * blocks_per_row * bytes_per_block;
    const half* s_row = scales + (size_t)o * blocks_per_row;
    const float* x_row = x + (size_t)s * I;

    float partial_sum = 0.0f;

    for (int blk = tid; blk < blocks_per_row; blk += blockDim.x) {
        float scale = __half2float(s_row[blk]);
        const uint8_t* blk_data = w_row + (size_t)blk * bytes_per_block;
        const float* x_blk = x_row + blk * 256;

        float block_sum = 0.0f;
        int val_idx = 0;

        // 1. Branchless Trit Decoding
        #pragma unroll 4
        for (int b = 0; b < 32; ++b) {
            uint32_t packed = blk_data[b];
            #pragma unroll
            for (int t = 0; t < 5; ++t) {
                // NVCC compiler will optimize /3 and %3 into fast multiply-adds via magic constants
                int trit = packed % 3; 
                packed /= 3;
                
                // Map (0, 1, 2) directly to (-1.0f, 0.0f, 1.0f) using math, avoiding branches entirely
                float w = (float)trit - 1.0f; 
                block_sum += w * x_blk[val_idx];
                val_idx++;
            }
        }
        
        // Handle overflow bytes
        #pragma unroll
        for (int b = 32; b < 36; ++b) {
            uint32_t packed = blk_data[b];
            #pragma unroll
            for (int t = 0; t < 5 && val_idx < 256; ++t) {
                int trit = packed % 3;
                packed /= 3;
                float w = (float)trit - 1.0f;
                block_sum += w * x_blk[val_idx];
                val_idx++;
            }
        }
        partial_sum += block_sum * scale;
    }

    // Warp + Block Reduction
    unsigned mask = 0xffffffff;
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        partial_sum += __shfl_down_sync(mask, partial_sum, offset);
    }

    static __shared__ float s_sum[32];
    int lane = tid % 32, warp = tid / 32;
    if (lane == 0) s_sum[warp] = partial_sum;
    __syncthreads();

    if (warp == 0) {
        partial_sum = (lane < blockDim.x / 32) ? s_sum[lane] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            partial_sum += __shfl_down_sync(mask, partial_sum, offset);
        }
        if (lane == 0) y[(size_t)s * O + o] = partial_sum;
    }
}

// ============================================================
// IQ1_S/IQ1_M Fused Matmul Kernel (Optimized)
// ============================================================
__global__ void iq1_matmul_kernel_optimized(
    float* __restrict__ y,
    const float* __restrict__ x,
    const uint8_t* __restrict__ weights,
    int S, int I, int O,
    int bytes_per_block
) {
    int o = blockIdx.x;
    int s = blockIdx.y;
    int tid = threadIdx.x;
    if (o >= O || s >= S) return;

    int blocks_per_row = I / 256;
    const uint8_t* w_row = weights + (size_t)o * blocks_per_row * bytes_per_block;
    const float* x_row = x + (size_t)s * I;

    float partial_sum = 0.0f;

    for (int blk = tid; blk < blocks_per_row; blk += blockDim.x) {
        const uint8_t* blk_data = w_row + (size_t)blk * bytes_per_block;
        float scale = __half2float(*(const half*)blk_data);
        
        // FIX (same class of bug as q8_0_matmul.cu -- see that file's own
        // comment for the full explanation, confirmed by a real Xid 13
        // "Misaligned Address" crash there): `bytes_per_block` is a
        // caller-supplied per-block byte stride with no guarantee it's a
        // multiple of 4, so `blk_data + 2` (blk_data itself already
        // `blk*bytes_per_block` bytes into the row) can land at a
        // non-4-byte-aligned address depending on `blk`'s value -- casting
        // that straight to `const uint32_t*` and dereferencing it risks the
        // identical hardware-illegal unaligned load. Read the 4 bytes
        // individually instead (uint8_t alignment is always 1, safe
        // unconditionally) and reassemble the same little-endian uint32_t a
        // wide read would have produced -- NVIDIA GPUs are little-endian,
        // matching this composition to the original bit-layout exactly.
        const uint8_t* signs_bytes = blk_data + 2;
        const float4* x_blk_f4 = reinterpret_cast<const float4*>(x_row + blk * 256);

        float block_sum = 0.0f;

        // 2. Vectorized Math and Branchless Accumulation
        #pragma unroll 4
        for (int i = 0; i < 8; ++i) { // 8 iterations * 32 bits = 256 weights
            const uint8_t* sb = signs_bytes + i * 4;
            uint32_t bitpack = (uint32_t)sb[0] | ((uint32_t)sb[1] << 8)
                              | ((uint32_t)sb[2] << 16) | ((uint32_t)sb[3] << 24);

            // Process 8 float4s (32 floats) per iteration
            #pragma unroll
            for (int v = 0; v < 8; ++v) {
                float4 x4 = x_blk_f4[i * 8 + v];
                
                // Extract bits and map 0 -> -1.0f, 1 -> 1.0f using: 2*bit - 1
                float w0 = (float)((bitpack >> 0) & 1) * 2.0f - 1.0f;
                float w1 = (float)((bitpack >> 1) & 1) * 2.0f - 1.0f;
                float w2 = (float)((bitpack >> 2) & 1) * 2.0f - 1.0f;
                float w3 = (float)((bitpack >> 3) & 1) * 2.0f - 1.0f;
                
                block_sum += w0 * x4.x + w1 * x4.y + w2 * x4.z + w3 * x4.w;
                bitpack >>= 4; // Shift down for the next float4
            }
        }
        partial_sum += block_sum * scale;
    }

    // Warp + block reduction
    unsigned mask = 0xffffffff;
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        partial_sum += __shfl_down_sync(mask, partial_sum, offset);
    }

    static __shared__ float s_sum[32];
    int lane = tid % 32, warp = tid / 32;
    if (lane == 0) s_sum[warp] = partial_sum;
    __syncthreads();

    if (warp == 0) {
        partial_sum = (lane < blockDim.x / 32) ? s_sum[lane] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            partial_sum += __shfl_down_sync(mask, partial_sum, offset);
        }
        if (lane == 0) y[(size_t)s * O + o] = partial_sum;
    }
}