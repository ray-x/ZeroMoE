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
    int o = blockIdx.x; 
    int s = blockIdx.y;
    int tid = threadIdx.x;
         
    int blocks_per_row = I / 32;
    const block_q4_0* w_row = weights + (o * blocks_per_row);
    const float* x_row = x + (s * I);
    float row_sum = 0.0f;

    for (int b = tid; b < blocks_per_row; b += blockDim.x) {
        const float* x_block = x_row + (b * 32);
        float amax = 0.0f;
        for(int i=0; i<32; ++i) {
            float abs_val = fabsf(x_block[i]);
            if (abs_val > amax) amax = abs_val;
        }
        float x_scale = amax / 127.0f;
        if (x_scale < 1e-12f) x_scale = 1e-12f;
        float inv_x_scale = 1.0f / x_scale;

        int8_t x_q8[32];
        for(int i=0; i<32; ++i) {
            x_q8[i] = (int8_t)rintf(x_block[i] * inv_x_scale);
        }

        float w_scale = __half2float(w_row[b].d);
        const uint8_t* w_qs = w_row[b].qs;
        int32_t block_dot = 0;

        for (int i = 0; i < 16; i += 4) {
            uint32_t packed_w_low = 0;
            uint32_t packed_w_high = 0;
            for(int j=0; j<4; ++j) {
                uint8_t byte = w_qs[i + j];
                int8_t w0 = (int8_t)(byte & 0x0F) - 8;
                int8_t w1 = (int8_t)(byte >> 4) - 8;
                                 
                packed_w_low  |= ((uint32_t)(uint8_t)w0) << (j * 8);
                packed_w_high |= ((uint32_t)(uint8_t)w1) << (j * 8);
            }

            uint32_t packed_x_low = 0;
            uint32_t packed_x_high = 0;
            for(int j=0; j<4; ++j) {
                packed_x_low  |= ((uint32_t)(uint8_t)x_q8[(i + j) * 2])     << (j * 8);
                packed_x_high |= ((uint32_t)(uint8_t)x_q8[(i + j) * 2 + 1]) << (j * 8);
            }

            block_dot = __dp4a((int)packed_w_low, (int)packed_x_low, block_dot);
            block_dot = __dp4a((int)packed_w_high, (int)packed_x_high, block_dot);
        }
        row_sum += ((float)block_dot) * w_scale * x_scale;
    }

    __shared__ float shared_sums[256];
    shared_sums[tid] = row_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sums[tid] += shared_sums[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        y[(size_t)s * O + o] = shared_sums[0];
    }
}

extern "C" void launch_gguf_q4_0_dp4a(float* y, const float* x, const void* weights, int S, int I, int O) {
    dim3 grid(O, S);
    dim3 block(256);
    gguf_q4_0_dp4a_matmul<<<grid, block>>>(y, x, (const block_q4_0*)weights, S, I, O);
}