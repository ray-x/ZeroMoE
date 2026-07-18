// kv_quant.cu — 4-bit KV cache quantization and dequantization kernels
// Compresses KV cache entries from FP32 to Q4_0 on write (saves 8x VRAM),
// dequantizes on-the-fly during attention score computation.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

// Q4_0 block: 32 values packed into 18 bytes (2-byte scale + 16-byte data)
struct BlockQ4_KV {
    half d;           // Block scale factor
    uint8_t qs[16];   // 32 x 4-bit values, 2 per byte
};

// ============================================================
// Quantize FP32 → Q4_0: called when writing to KV cache
// Each thread block quantizes 32 elements (one Q4_0 block).
// Grid: (num_blocks), Block: (32) threads
// ============================================================
__global__ void kv_quantize_q4_kernel(
    BlockQ4_KV* __restrict__ output,
    const float* __restrict__ input,
    int n_elements  // must be multiple of 32
) {
    int block_idx = blockIdx.x;
    int tid = threadIdx.x;  // 0..31
    int base = block_idx * 32;

    if (base + 31 >= n_elements) return;

    // Load and find absmax (warp-level reduction)
    float val = input[base + tid];
    float abs_val = fabsf(val);

    // Warp reduction for max
    for (int offset = 16; offset > 0; offset /= 2) {
        abs_val = fmaxf(abs_val, __shfl_down_sync(0xffffffff, abs_val, offset));
    }

    __shared__ float s_amax;
    if (tid == 0) s_amax = abs_val;
    __syncthreads();
    float amax = s_amax;

    // Compute scale
    float scale = amax / 7.0f;  // Q4_0 range: [-8, 7], but asymmetric offset
    if (scale < 1e-12f) scale = 1e-12f;
    float inv_scale = 1.0f / scale;

    // Quantize: map to [0, 15] unsigned with bias 8
    int q_val = __float2int_rn(val * inv_scale) + 8;
    q_val = max(0, min(15, q_val));

    // Pack pairs of 4-bit values into bytes
    // Thread tid and tid+16 share a byte (tid in low nibble, tid+16 in high)
    __shared__ uint8_t s_packed[16];
    if (tid < 16) {
        // Low nibble from thread tid, high nibble from thread tid+16
        int q_lo = __float2int_rn(input[base + tid * 2] * inv_scale) + 8;
        int q_hi = __float2int_rn(input[base + tid * 2 + 1] * inv_scale) + 8;
        q_lo = max(0, min(15, q_lo));
        q_hi = max(0, min(15, q_hi));
        s_packed[tid] = (uint8_t)((q_hi << 4) | q_lo);
    }
    __syncthreads();

    // Thread 0 writes the block
    if (tid == 0) {
        output[block_idx].d = __float2half(scale);
        for (int i = 0; i < 16; ++i) {
            output[block_idx].qs[i] = s_packed[i];
        }
    }
}

// ============================================================
// Dequantize Q4_0 → FP32: called during attention to unpack KV
// Each thread block handles one Q4_0 block (32 output floats).
// ============================================================
__global__ void kv_dequantize_q4_kernel(
    float* __restrict__ output,
    const BlockQ4_KV* __restrict__ input,
    int n_blocks
) {
    int block_idx = blockIdx.x * blockDim.x / 32 + threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    if (block_idx >= n_blocks) return;

    float scale = __half2float(input[block_idx].d);
    int byte_idx = lane / 2;
    uint8_t packed = input[block_idx].qs[byte_idx];
    int q_val = (lane % 2 == 0) ? (packed & 0x0F) : (packed >> 4);
    float value = ((float)q_val - 8.0f) * scale;

    output[block_idx * 32 + lane] = value;
}

// ============================================================
// Fused: Dot product of query vector against Q4_0-quantized KV cache row
// Avoids materializing the full dequantized row.
// One block per (head, time_step) pair.
// ============================================================
__global__ void kv_q4_dot_kernel(
    float* __restrict__ scores,       // [n_queries, context_len]
    const float* __restrict__ queries, // [n_queries, dim]
    const BlockQ4_KV* __restrict__ kv_cache, // [context_len, dim/32] blocks
    int n_queries, int dim, int context_len,
    float scale_factor
) {
    int q_idx = blockIdx.y;
    int t_idx = blockIdx.x;
    int tid = threadIdx.x;
    if (q_idx >= n_queries || t_idx >= context_len) return;

    int blocks_per_entry = dim / 32;
    const float* q = queries + (int64_t)q_idx * dim;
    const BlockQ4_KV* kv_row = kv_cache + (int64_t)t_idx * blocks_per_entry;

    float partial_sum = 0.0f;
    for (int b = tid; b < blocks_per_entry; b += blockDim.x) {
        float kv_scale = __half2float(kv_row[b].d);
        const float* q_block = q + b * 32;

        float block_dot = 0.0f;
        for (int i = 0; i < 16; ++i) {
            uint8_t packed = kv_row[b].qs[i];
            float v0 = ((float)(packed & 0x0F) - 8.0f) * kv_scale;
            float v1 = ((float)(packed >> 4) - 8.0f) * kv_scale;
            block_dot += q_block[i * 2] * v0 + q_block[i * 2 + 1] * v1;
        }
        partial_sum += block_dot;
    }

    // Warp reduction
    for (int offset = 16; offset > 0; offset /= 2)
        partial_sum += __shfl_down_sync(0xffffffff, partial_sum, offset);

    __shared__ float s_sum[32];
    int lane = tid % 32, warp = tid / 32;
    if (lane == 0) s_sum[warp] = partial_sum;
    __syncthreads();

    if (warp == 0) {
        partial_sum = (lane < blockDim.x / 32) ? s_sum[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset /= 2)
            partial_sum += __shfl_down_sync(0xffffffff, partial_sum, offset);
        if (lane == 0) {
            scores[(int64_t)q_idx * context_len + t_idx] = partial_sum * scale_factor;
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
    dim3 block(128);
    kv_q4_dot_kernel<<<grid, block>>>(scores, queries, (const BlockQ4_KV*)kv_cache,
                                       n_queries, dim, context_len, scale);
}
