#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

struct block_q2_0_cuda {
    half d;
    uint8_t qs[16];
};

static_assert(sizeof(block_q2_0_cuda) == 18, "unexpected Q2_0 block layout");

__global__ void q2_0_matmul_kernel(float* __restrict__ output,
                                   const float* __restrict__ input,
                                   const block_q2_0_cuda* __restrict__ weights,
                                   int token_count, int input_size, int output_size) {
    const int output_index = blockIdx.x;
    const int token_index = blockIdx.y;
    const int thread_index = threadIdx.x;
    if (output_index >= output_size || token_index >= token_count) return;

    const int blocks_per_row = input_size / 64;
    const block_q2_0_cuda* weight_row = weights +
        static_cast<size_t>(output_index) * blocks_per_row;
    const float* input_row = input + static_cast<size_t>(token_index) * input_size;

    float partial_sum = 0.0f;
    for (int block_index = thread_index; block_index < blocks_per_row;
         block_index += blockDim.x) {
        const block_q2_0_cuda& block = weight_row[block_index];
        const float scale = __half2float(block.d);
        const float* input_block = input_row + block_index * 64;
        float block_sum = 0.0f;

        #pragma unroll
        for (int element_index = 0; element_index < 64; ++element_index) {
            const int packed_byte = element_index >> 2;
            const int bit_shift = (element_index & 3) << 1;
            const int code = (block.qs[packed_byte] >> bit_shift) & 0x03;
            block_sum += static_cast<float>(code - 1) * input_block[element_index];
        }
        partial_sum += scale * block_sum;
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        partial_sum += __shfl_down_sync(0xffffffff, partial_sum, offset);

    if (blockDim.x <= 32) {
        if (thread_index == 0)
            output[static_cast<size_t>(token_index) * output_size + output_index] = partial_sum;
        return;
    }

    __shared__ float warp_sums[8];
    const int lane_index = thread_index & 31;
    const int warp_index = thread_index >> 5;
    if (lane_index == 0) warp_sums[warp_index] = partial_sum;
    __syncthreads();

    if (warp_index == 0) {
        const int warp_count = blockDim.x >> 5;
        float total = lane_index < warp_count ? warp_sums[lane_index] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            total += __shfl_down_sync(0xffffffff, total, offset);
        if (lane_index == 0)
            output[static_cast<size_t>(token_index) * output_size + output_index] = total;
    }
}

extern "C" void launch_q2_0_matmul(float* output, const float* input,
                                    const void* weights, int token_count,
                                    int input_size, int output_size) {
    if (token_count <= 0 || input_size <= 0 || output_size <= 0 ||
        input_size % 64 != 0)
        return;

    const int blocks_per_row = input_size / 64;
    int threads = ((blocks_per_row + 31) / 32) * 32;
    if (threads < 32) threads = 32;
    if (threads > 256) threads = 256;

    const dim3 grid(static_cast<unsigned>(output_size),
                    static_cast<unsigned>(token_count));
    q2_0_matmul_kernel<<<grid, threads>>>(
        output, input, reinterpret_cast<const block_q2_0_cuda*>(weights),
        token_count, input_size, output_size);
}
