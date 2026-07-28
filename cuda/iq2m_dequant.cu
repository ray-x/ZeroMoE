// iq2m_dequant.cu — IQ2_M dequantization kernel using shared memory codebook
// Each warp cooperatively unpacks 2-bit nibbles and resolves f32 values via
// the codebook stored in fast __shared__ memory.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

// IQ2_M block structure (256 elements per block, 84 bytes total)
// Layout: scales[2] + qs[64] + signs[16] + unused[2]
// The 2-bit values index into a 4-entry codebook per sub-block.
struct block_iq2_m_cuda {
    uint16_t scales;       // packed FP16 super-block scale
    uint8_t  qs[64];      // 2-bit quantized values (4 per byte)
    uint8_t  signs[16];   // sign bits (1 per value, packed 8 per byte)
    uint8_t  pad[2];      // padding to 84 bytes
};

// The IQ2_M codebook (256 entries) — loaded into shared memory per block
// This is a simplified 4-value codebook for the sub-grid; real implementation
// uses the full ggml IQ2_M grid values.
__constant__ float d_iq2m_grid[4] = {-1.0f, -0.333f, 0.333f, 1.0f};

// Dequantize one IQ2_M super-block (256 elements) into fp32
// Grid: one thread block per super-block row
// Threads: 256 per block (one thread per output element)
__global__ void kernel_dequant_iq2m(
    float* __restrict__ output,
    const uint8_t* __restrict__ weights,
    int n_blocks 
) {
    __shared__ float s_grid[4];
    if (threadIdx.x < 4) {
        s_grid[threadIdx.x] = d_iq2m_grid[threadIdx.x];
    }
    __syncthreads();

    int block_idx = blockIdx.x;
    if (block_idx >= n_blocks) return;

    int tid = threadIdx.x;  
    if (tid >= 256) return;

    const uint8_t* blk = weights + (size_t)block_idx * 84;
    
    uint16_t scale_u16 = *(const uint16_t*)blk;
    float scale = __half2float(*(const half*)&scale_u16);

    int byte_idx = tid / 4;
    int shift = (tid % 4) * 2;
    uint8_t q_byte = blk[2 + byte_idx];  
    int q_val = (q_byte >> shift) & 0x03;

    int sign_byte_idx = tid / 8;
    int sign_bit_idx = tid % 8;
    uint8_t sign_byte = blk[66 + sign_byte_idx];  
    float sign = (sign_byte >> sign_bit_idx) & 1 ? -1.0f : 1.0f;

    float value = s_grid[q_val] * sign * scale;
    output[(size_t)block_idx * 256 + tid] = value;
}

// Fused IQ2_M dequant + matmul: y[S,O] = x[S,I] @ W^T where W is IQ2_M quantized
// Each thread block handles one output row.
// This avoids materializing the full dequantized weight matrix.
__global__ void kernel_iq2m_matmul(
    float* __restrict__ y,
    const float* __restrict__ x,
    const uint8_t* __restrict__ weights,
    int S, int I, int O
) {
    __shared__ float s_grid[4];
    if (threadIdx.x < 4) s_grid[threadIdx.x] = d_iq2m_grid[threadIdx.x];
    __syncthreads();

    int o = blockIdx.x;  
    int s = blockIdx.y;  
    int tid = threadIdx.x; // 0..255 (One thread per super-block element)

    if (o >= O || s >= S) return;

    int blocks_per_row = I / 256;
    const uint8_t* w_row = weights + (size_t)o * blocks_per_row * 84;
    const float* x_row = x + (size_t)s * I;

    float partial_sum = 0.0f;

    for (int b = 0; b < blocks_per_row; ++b) {
        const uint8_t* blk = w_row + (size_t)b * 84;
        
        // Coalesced Scale Fetch
        uint16_t scale_u16 = *(const uint16_t*)blk;
        float scale = __half2float(*(const half*)&scale_u16);
        
        // Coalesced Grid/Sign Fetch
        int byte_idx = tid / 4;
        int shift = (tid % 4) * 2;
        uint8_t q_byte = blk[2 + byte_idx];
        int q_val = (q_byte >> shift) & 0x03;

        int sign_byte_idx = tid / 8;
        int sign_bit_idx = tid % 8;
        uint8_t sign_byte = blk[66 + sign_byte_idx];
        float sign = ((sign_byte >> sign_bit_idx) & 1) ? -1.0f : 1.0f;
        
        float w_val = s_grid[q_val] * sign * scale;
        partial_sum += x_row[b * 256 + tid] * w_val;
    }

    // Standard block reduction
    __shared__ float sm[256];
    sm[tid] = partial_sum;
    __syncthreads();

    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) sm[tid] += sm[tid + stride];
        __syncthreads();
    }

    if (tid == 0) y[(size_t)s * O + o] = sm[0];
}

// Host-callable launchers
extern "C" void launch_iq2m_dequant(float* output, const uint8_t* weights, int numel) {
    int n_blocks = numel / 256;
    kernel_dequant_iq2m<<<n_blocks, 256>>>(output, weights, n_blocks);
}

extern "C" void launch_iq2m_matmul(float* y, const float* x, const void* weights,
                                    int S, int I, int O) {
    dim3 grid(O, S);
    dim3 block(128);  // 128 threads per output element
    kernel_iq2m_matmul<<<grid, block>>>(y, x, (const uint8_t*)weights, S, I, O);
}
