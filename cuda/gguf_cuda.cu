#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

struct block_q4_0 {
    half d;                
    uint8_t qs[16];    
};

__global__ static void quant_matmul_dp4a_optimized(
    float *y, const float *x, const void *weights,
    const float *scales, int fmt, int S, int I, int O, size_t rb) 
{
    int o = blockIdx.x;
    int s = blockIdx.y;
    int tid = threadIdx.x;
         
    if (fmt != 2 || o >= O || s >= S) return;

    const uint8_t *w_row = static_cast<const uint8_t*>(weights) + (size_t)o * rb;
    const float *xs = x + (size_t)s * I;

    // --- 1. Full Multi-Warp Block Reduction for amax ---
    float amax = 0.0f;
    for (int i = tid; i < I; i += blockDim.x) {
        float abs_val = fabsf(xs[i]);
        if (abs_val > amax) amax = abs_val;
    }
         
    unsigned int mask = 0xffffffff;
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        amax = fmaxf(amax, __shfl_down_sync(mask, amax, offset));
    }
         
    static __shared__ float s_warp_max[32];
    int lane = tid % 32;
    int warp = tid / 32;

    if (lane == 0) s_warp_max[warp] = amax;
    __syncthreads();

    // Final reduction by Warp 0
    if (warp == 0) {
        amax = (lane < (blockDim.x / 32)) ? s_warp_max[lane] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            amax = fmaxf(amax, __shfl_down_sync(mask, amax, offset));
        }
    }
    
    __shared__ float s_max;
    if (tid == 0) s_max = amax;
    __syncthreads();
         
    float x_scale = s_max / 127.0f;
    if (x_scale < 1e-12f) x_scale = 1e-12f;
    float inv_x_scale = 1.0f / x_scale;

    // --- 2. Vectorized DP4A Execution Loop ---
    int32_t sum_dp4a = 0;

    // Process 8 weights per thread per iteration (2x float4 reads)
    for (int i = tid * 8; i < I; i += blockDim.x * 8) {
        // FIX (same class of bug as q8_0_matmul.cu -- see that file's own
        // comment for the full explanation, confirmed by a real Xid 13
        // "Misaligned Address" crash there): the offset WITHIN a row
        // (i/2) is always a multiple of 4 here (i is always a multiple of
        // 8), but `w_row` itself is `weights + o*rb` with `rb=(I+1)/2` --
        // nothing guarantees `o*rb` is a multiple of 4 for every `o`
        // (only holds if `rb` itself is divisible by 4, i.e. `I`
        // divisible by 8, which isn't true for every possible I this
        // kernel could be called with). Read the 4 bytes individually
        // instead (uint8_t alignment is always 1, safe unconditionally)
        // and reassemble the same little-endian uint32_t a wide read
        // would have produced -- NVIDIA GPUs are little-endian, matching
        // this composition to the original bit-layout exactly.
        const uint8_t* wb = w_row + (i / 2);
        uint32_t w_bytes = (uint32_t)wb[0] | ((uint32_t)wb[1] << 8)
                          | ((uint32_t)wb[2] << 16) | ((uint32_t)wb[3] << 24);

        // Extract low nibbles
        uint32_t w0 = ((w_bytes >> 0)  & 0x0F) - 8;
        uint32_t w1 = ((w_bytes >> 8)  & 0x0F) - 8;
        uint32_t w2 = ((w_bytes >> 16) & 0x0F) - 8;
        uint32_t w3 = ((w_bytes >> 24) & 0x0F) - 8;
        uint32_t packed_w_low = (w0 & 0xFF) | ((w1 & 0xFF) << 8) | ((w2 & 0xFF) << 16) | ((w3 & 0xFF) << 24);

        // Extract high nibbles
        uint32_t w4 = ((w_bytes >> 4)  & 0x0F) - 8;
        uint32_t w5 = ((w_bytes >> 12) & 0x0F) - 8;
        uint32_t w6 = ((w_bytes >> 20) & 0x0F) - 8;
        uint32_t w7 = ((w_bytes >> 28) & 0x0F) - 8;
        uint32_t packed_w_high = (w4 & 0xFF) | ((w5 & 0xFF) << 8) | ((w6 & 0xFF) << 16) | ((w7 & 0xFF) << 24);

        // Vectorized 128-bit float4 loads
        float4 x_vec0 = *reinterpret_cast<const float4*>(xs + i);
        float4 x_vec1 = *reinterpret_cast<const float4*>(xs + i + 4);

        // Fast float -> int8 rounding via PTX intrinsic
        int32_t x0 = __float2int_rn(x_vec0.x * inv_x_scale);
        int32_t x1 = __float2int_rn(x_vec0.y * inv_x_scale);
        int32_t x2 = __float2int_rn(x_vec0.z * inv_x_scale);
        int32_t x3 = __float2int_rn(x_vec0.w * inv_x_scale);

        int32_t x4 = __float2int_rn(x_vec1.x * inv_x_scale);
        int32_t x5 = __float2int_rn(x_vec1.y * inv_x_scale);
        int32_t x6 = __float2int_rn(x_vec1.z * inv_x_scale);
        int32_t x7 = __float2int_rn(x_vec1.w * inv_x_scale);

        uint32_t packed_x_low  = (x0 & 0xFF) | ((x1 & 0xFF) << 8) | ((x2 & 0xFF) << 16) | ((x3 & 0xFF) << 24);
        uint32_t packed_x_high = (x4 & 0xFF) | ((x5 & 0xFF) << 8) | ((x6 & 0xFF) << 16) | ((x7 & 0xFF) << 24);

        // Execute DP4A dot products
        sum_dp4a = __dp4a(static_cast<int>(packed_w_low), static_cast<int>(packed_x_low), sum_dp4a);
        sum_dp4a = __dp4a(static_cast<int>(packed_w_high), static_cast<int>(packed_x_high), sum_dp4a);
    }

    // --- 3. Block Output Reduction ---
    float sum = (float)sum_dp4a * x_scale;
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(mask, sum, offset);
    }

    static __shared__ float s_sum[32]; 
    if (lane == 0) s_sum[warp] = sum;
    __syncthreads();

    if (warp == 0) {
        sum = (lane < (blockDim.x / 32)) ? s_sum[lane] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(mask, sum, offset);
        }
        if (lane == 0) {
            y[(size_t)s * O + o] = sum * (scales ? scales[o] : 1.0f);
        }
    }
}

extern "C" void launch_raw_q4_0_dp4a(float* y, const float* x, const void* weights,
                                      const float* scales, int S, int I, int O) {
    dim3 grid(O, S);
    size_t rb = (I + 1) / 2;
    quant_matmul_dp4a_optimized<<<grid, 256>>>(y, x, weights, scales, 2, S, I, O, rb);
}