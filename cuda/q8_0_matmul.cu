// q8_0_matmul.cu — fused Q8_0 dequant + matmul: y[S,O] = x[S,I] @ W^T
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

struct block_q8_0_k {
    half    d;        // FP16 per-block scale
    int8_t  qs[32];   // 32 int8 values
};  // sizeof == 34

__global__ void q8_0_matmul_kernel_optimized(
    float* __restrict__       y,
    const float* __restrict__ x,
    const block_q8_0_k* __restrict__ W,
    int S, int I, int O)
{
    int o = blockIdx.x;
    int s = blockIdx.y;
    int tid = threadIdx.x;
    
    if (o >= O || s >= S) return;

    int blocks_per_row = I / 32;
    const block_q8_0_k* w_row = W + (size_t)o * blocks_per_row;
    const float* x_row = x + (size_t)s * I;

    float row_sum = 0.0f;

    // 1. Vectorized Memory Access & Unrolled Accumulation
    //
    // FIX (real bug -- caused an Xid 13 "Misaligned Address" GPU exception
    // on real hardware, confirmed by dmesg output across dozens of SMs at
    // once): block_q8_0_k is 34 bytes (2-byte half + 32-byte int8 array),
    // which is the REAL, on-disk GGUF Q8_0 block layout -- it can't be
    // padded to a multiple of 4 without misinterpreting actual model
    // weight bytes. Because 34 is not a multiple of 4, w_row[b].qs lands
    // at a 4-byte-ALIGNED offset only when (o*blocks_per_row + b) is odd,
    // and at a MISALIGNED offset (2 mod 4) whenever it's even -- including
    // the very first block (o=0, b=0). The previous version cast that
    // address straight to `const int32_t*` and dereferenced it, which is a
    // hardware-illegal unaligned 32-bit load on roughly half of all blocks
    // processed. `qs` is read byte-by-byte here instead (int8_t has
    // alignment 1, always safe regardless of block parity) -- the x4
    // float4 read is untouched since x_row's own alignment (16-byte
    // periodic stride from a cudaMalloc'd, heavily-aligned base) was never
    // the problem.
    for (int b = tid; b < blocks_per_row; b += blockDim.x) {
        float d = __half2float(w_row[b].d);

        const float4* xb_f4 = reinterpret_cast<const float4*>(x_row + b * 32);
        const int8_t* qs = w_row[b].qs;

        float dot = 0.0f;

        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            float4 x4 = xb_f4[i];   // Load 4x float activations in one 128-bit transaction

            // qs is already signed int8_t -- no manual sign-extension needed
            dot += (float)qs[i * 4 + 0] * x4.x;
            dot += (float)qs[i * 4 + 1] * x4.y;
            dot += (float)qs[i * 4 + 2] * x4.z;
            dot += (float)qs[i * 4 + 3] * x4.w;
        }
        row_sum += dot * d;
    }

    // 2. Warp-Level Reduction using Shuffle Primitives
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        row_sum += __shfl_down_sync(0xffffffff, row_sum, offset);
    }

    // 3. Block-Level Reduction across Warps
    static __shared__ float s_warp_sums[32]; // Max 32 warps (1024 threads)
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    if (lane_id == 0) {
        s_warp_sums[warp_id] = row_sum;
    }
    __syncthreads();

    // Final reduction by the first warp
    if (warp_id == 0) {
        int num_warps = blockDim.x / 32;
        row_sum = (lane_id < num_warps) ? s_warp_sums[lane_id] : 0.0f;

        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            row_sum += __shfl_down_sync(0xffffffff, row_sum, offset);
        }

        if (lane_id == 0) {
            y[(size_t)s * O + o] = row_sum;
        }
    }
}

extern "C" void launch_q8_0_matmul(float* y, const float* x, const void* weights,
                                    int S, int I, int O) {
    dim3 grid(O, S);
    dim3 block(256);
    q8_0_matmul_kernel_optimized<<<grid, block>>>(y, x, (const block_q8_0_k*)weights, S, I, O);
}