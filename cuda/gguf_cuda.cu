#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

struct block_q4_0 {
    half d;                
    uint8_t qs[16];    
};

__global__ static void quant_matmul_dp4a(float *y, const float *x, const void *weights,
                                         const float *scales, int fmt, int S, int I, int O,
                                         size_t rb) {
    int o = blockIdx.x;
    int s = blockIdx.y;
    int tid = threadIdx.x;
         
    if (fmt != 2) return;

    const uint8_t *w_row = static_cast<const uint8_t*>(weights) + (size_t)o * rb;
    const float *xs = x + (size_t)s * I;

    float amax = 0.0f;
    for (int i = tid; i < I; i += blockDim.x) {
        float abs_val = fabsf(xs[i]);
        if (abs_val > amax) amax = abs_val;
    }
         
    unsigned int mask = 0xffffffff;
    for (int offset = 16; offset > 0; offset /= 2) {
        amax = fmaxf(amax, __shfl_down_sync(mask, amax, offset));
    }
         
    __shared__ float s_max;
    if (tid == 0) s_max = amax;
    __syncthreads();
         
    float x_scale = s_max / 127.0f;
    if (x_scale < 1e-12f) x_scale = 1e-12f;
    float inv_x_scale = 1.0f / x_scale;

    int32_t sum_dp4a = 0;
    for (int i = tid * 8; i < I; i += blockDim.x * 8) {
        uint32_t packed_w_low = 0;
        uint32_t packed_w_high = 0;
                 
        for (int j = 0; j < 4; ++j) {
            if (i + j * 2 >= I) break;
            uint8_t byte = w_row[(i >> 1) + j];
            int8_t w0 = (int8_t)(byte & 0x0F) - 8;
            int8_t w1 = (int8_t)(byte >> 4) - 8;
            packed_w_low  |= ((uint32_t)(uint8_t)w0) << (j * 8);
            packed_w_high |= ((uint32_t)(uint8_t)w1) << (j * 8);
        }
                 
        uint32_t packed_x_low = 0;
        uint32_t packed_x_high = 0;
        for (int j = 0; j < 4; ++j) {
            if (i + j * 2 >= I) break;
            int8_t x0 = (int8_t)rintf(xs[i + j * 2] * inv_x_scale);
            int8_t x1 = (int8_t)rintf(xs[i + j * 2 + 1] * inv_x_scale);
            packed_x_low  |= ((uint32_t)(uint8_t)x0) << (j * 8);
            packed_x_high |= ((uint32_t)(uint8_t)x1) << (j * 8);
        }
                 
        // Explicitly cast to 'int' to resolve NVCC overload ambiguity
        sum_dp4a = __dp4a(static_cast<int>(packed_w_low), static_cast<int>(packed_x_low), sum_dp4a);
        sum_dp4a = __dp4a(static_cast<int>(packed_w_high), static_cast<int>(packed_x_high), sum_dp4a);
    }

    float sum = (float)sum_dp4a * x_scale;
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(mask, sum, offset);
    }

    __shared__ float s_sum[32]; 
    int lane = tid % 32;
    int warp = tid / 32;
         
    if (lane == 0) s_sum[warp] = sum;
    __syncthreads();

    if (warp == 0) {
        sum = (lane < (blockDim.x / 32)) ? s_sum[lane] : 0.0f;
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
    quant_matmul_dp4a<<<grid, 256>>>(y, x, weights, scales, 2, S, I, O, rb);
}