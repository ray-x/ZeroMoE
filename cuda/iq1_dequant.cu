// iq1_dequant.cu — 1-bit quantization CUDA kernels (IQ1_S, IQ1_M, TQ1_0)
// Ternary weights {-1, 0, +1} with per-block scaling.
// The key insight: ternary matmul reduces to conditional add/subtract,
// which is extremely fast on GPU (no actual multiplication needed).
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

// ============================================================
// TQ1_0 Fused Matmul Kernel
// y[S,O] = x[S,I] @ W^T where W is ternary {-1, 0, +1} with scale
// For ternary: dot(x, w) = scale * (sum_of_x_where_w=+1 - sum_of_x_where_w=-1)
// This is MUCH faster than general multiply — just conditional accumulate.
// ============================================================

// TQ1_0: base-3 packed (5 trits per byte, 32 bytes = 160 trits, + 4 overflow = 180 trits)
// But actual ggml packs 256 values in 54 bytes total (with scale)
// Simplified: each byte holds 8 sign bits (1-bit per weight), scale converts ternary

__global__ void tq1_matmul_kernel(
    float* __restrict__ y,
    const float* __restrict__ x,
    const uint8_t* __restrict__ weights,
    const half* __restrict__ scales,  // one FP16 scale per 256-element block
    int S, int I, int O
) {
    int o = blockIdx.x;
    int s = blockIdx.y;
    int tid = threadIdx.x;
    if (o >= O || s >= S) return;

    int blocks_per_row = I / 256;
    // Offset to this output row's weight data
    // TQ1_0: 54 bytes per block (simplified: 32 bytes ternary + 2 byte scale = 34)
    const int bytes_per_block = 54;
    const uint8_t* w_row = weights + (size_t)o * blocks_per_row * bytes_per_block;
    const half* s_row = scales + (size_t)o * blocks_per_row;
    const float* x_row = x + (size_t)s * I;

    float partial_sum = 0.0f;

    // Each thread handles a subset of blocks
    for (int blk = tid; blk < blocks_per_row; blk += blockDim.x) {
        float scale = __half2float(s_row[blk]);
        const uint8_t* blk_data = w_row + (size_t)blk * bytes_per_block;
        const float* x_blk = x_row + blk * 256;

        // Decode ternary values and accumulate
        // Base-3 decoding: each byte = 5 trits
        float block_sum = 0.0f;
        int val_idx = 0;

        for (int b = 0; b < 32 && val_idx < 256; ++b) {
            uint8_t packed = blk_data[b];
            for (int t = 0; t < 5 && val_idx < 256; ++t) {
                int trit = packed % 3;  // 0=-1, 1=0, 2=+1
                packed /= 3;
                // Ternary multiply: just add or subtract
                if (trit == 2) block_sum += x_blk[val_idx];       // w = +1
                else if (trit == 0) block_sum -= x_blk[val_idx];  // w = -1
                // trit == 1: w = 0, skip (no multiply needed!)
                val_idx++;
            }
        }
        // Handle overflow bytes (qh)
        for (int b = 32; b < 36 && val_idx < 256; ++b) {
            uint8_t packed = blk_data[b];
            for (int t = 0; t < 5 && val_idx < 256; ++t) {
                int trit = packed % 3;
                packed /= 3;
                if (trit == 2) block_sum += x_blk[val_idx];
                else if (trit == 0) block_sum -= x_blk[val_idx];
                val_idx++;
            }
        }

        partial_sum += block_sum * scale;
    }

    // Warp reduction
    unsigned mask = 0xffffffff;
    for (int offset = 16; offset > 0; offset /= 2) {
        partial_sum += __shfl_down_sync(mask, partial_sum, offset);
    }

    __shared__ float s_sum[32];
    int lane = tid % 32, warp = tid / 32;
    if (lane == 0) s_sum[warp] = partial_sum;
    __syncthreads();

    if (warp == 0) {
        partial_sum = (lane < blockDim.x / 32) ? s_sum[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset /= 2)
            partial_sum += __shfl_down_sync(mask, partial_sum, offset);
        if (lane == 0) {
            y[(size_t)s * O + o] = partial_sum;
        }
    }
}

// ============================================================
// IQ1_S/IQ1_M Fused Matmul Kernel
// Similar to ternary but uses a 1-bit sign + grid codebook.
// Simplified: treat as binary {-1, +1} with scale (no zeros).
// ============================================================
__global__ void iq1_matmul_kernel(
    float* __restrict__ y,
    const float* __restrict__ x,
    const uint8_t* __restrict__ weights,
    int S, int I, int O,
    int bytes_per_block  // 50 for IQ1_S, 56 for IQ1_M
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
        // Scale is first 2 bytes (FP16)
        float scale = __half2float(*(const half*)blk_data);
        const uint8_t* signs = blk_data + 2;  // 32 bytes of sign bits
        const float* x_blk = x_row + blk * 256;

        // Binary matmul: each bit is a sign. w_i = sign_bit ? +1 : -1
        // dot = scale * sum_i(x_i * sign_i)  where sign_i ∈ {-1, +1}
        //     = scale * (sum_positive - sum_negative)
        float block_sum = 0.0f;
        for (int i = 0; i < 256; i += 8) {
            uint8_t byte = signs[i / 8];
            for (int j = 0; j < 8 && (i + j) < 256; ++j) {
                float xi = x_blk[i + j];
                // bit=1 → +1, bit=0 → -1
                block_sum += ((byte >> j) & 1) ? xi : -xi;
            }
        }
        partial_sum += block_sum * scale;
    }

    // Warp + block reduction
    unsigned mask = 0xffffffff;
    for (int offset = 16; offset > 0; offset /= 2)
        partial_sum += __shfl_down_sync(mask, partial_sum, offset);

    __shared__ float s_sum[32];
    int lane = tid % 32, warp = tid / 32;
    if (lane == 0) s_sum[warp] = partial_sum;
    __syncthreads();

    if (warp == 0) {
        partial_sum = (lane < blockDim.x / 32) ? s_sum[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset /= 2)
            partial_sum += __shfl_down_sync(mask, partial_sum, offset);
        if (lane == 0) {
            y[(size_t)s * O + o] = partial_sum;
        }
    }
}

// ============================================================
// Host-callable launchers
// ============================================================
extern "C" void launch_tq1_matmul(float* y, const float* x, const void* weights,
                                   const void* scales, int S, int I, int O) {
    dim3 grid(O, S);
    dim3 block(128);
    tq1_matmul_kernel<<<grid, block>>>(y, x, (const uint8_t*)weights,
                                        (const half*)scales, S, I, O);
}

extern "C" void launch_iq1s_matmul(float* y, const float* x, const void* weights,
                                    int S, int I, int O) {
    dim3 grid(O, S);
    dim3 block(128);
    iq1_matmul_kernel<<<grid, block>>>(y, x, (const uint8_t*)weights, S, I, O, 50);
}

extern "C" void launch_iq1m_matmul(float* y, const float* x, const void* weights,
                                    int S, int I, int O) {
    dim3 grid(O, S);
    dim3 block(128);
    iq1_matmul_kernel<<<grid, block>>>(y, x, (const uint8_t*)weights, S, I, O, 56);
}
