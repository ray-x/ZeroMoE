// q8_0_matmul.cu — fused Q8_0 dequant + matmul: y[S,O] = x[S,I] @ W^T
//
// Q8_0 block (ggml block_q8_0): { half d; int8_t qs[32]; } = 34 bytes per 32
// elements. Dequant value_i = qs[i] * d. This is the simplest block quant (no
// packing tricks, no codebook), so it is the reference GPU dequant-matmul: the
// K-quant / IQ2 kernels that follow should mirror this launch/reduce structure.
//
// Grid: (O, S), one thread block per (output_row o, sequence s). Each thread
// strides over the I/32 blocks of weight row o, accumulating dot(qs, x)*d.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

struct block_q8_0_k {
    half    d;        // FP16 per-block scale
    int8_t  qs[32];   // 32 int8 values
};  // sizeof == 34, alignof == 2 → no padding (matches ggml block_q8_0)

__global__ void q8_0_matmul_kernel(
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
    for (int b = tid; b < blocks_per_row; b += blockDim.x) {
        float d = __half2float(w_row[b].d);
        const int8_t* qs = w_row[b].qs;
        const float* xb = x_row + b * 32;
        float dot = 0.0f;
        #pragma unroll
        for (int i = 0; i < 32; ++i) dot += (float)qs[i] * xb[i];
        row_sum += dot * d;
    }

    // Tree reduction across the block
    __shared__ float s_sum[256];
    s_sum[tid] = row_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) s_sum[tid] += s_sum[tid + stride];
        __syncthreads();
    }
    if (tid == 0) y[(size_t)s * O + o] = s_sum[0];
}

extern "C" void launch_q8_0_matmul(float* y, const float* x, const void* weights,
                                    int S, int I, int O) {
    dim3 grid(O, S);
    dim3 block(256);
    q8_0_matmul_kernel<<<grid, block>>>(y, x, (const block_q8_0_k*)weights, S, I, O);
}
