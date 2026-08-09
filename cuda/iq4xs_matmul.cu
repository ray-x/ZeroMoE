// iq4xs_matmul.cu — fused dequant + matmul for IQ4_XS weight tensors.
//   y[S,O] = x[S,I] @ W^T,  W packed row-major [O,I].
//
// The per-super-block dequant below is a verbatim port of ggml's
// dequantize_row_iq4_xs (ggml/src/ggml-quants.c). Unlike the IQ2/IQ3 family
// (cuda/iq2_matmul.cu, cuda/iq3_matmul.cu -- large 256-1024 entry variable
// codebook grids + sign tables), IQ4_XS uses a FIXED 16-entry non-linear
// lookup table (kvalues_iq4nl) + a simple per-32-element linear scale --
// structurally much closer to the K-quant family (cuda/kquant_matmul.cu)
// than to its IQ2/IQ3 siblings, despite the similar naming. Block struct
// layout matches ggml's block_iq4_xs byte-for-byte (ggml-common.h):
// d(2) + scales_h(2) + scales_l[4] + qs[128] = 136 bytes/256 elems
// (4.25 bits/weight, matches this project's own row_bytes() table).
//

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

#define QK_K 256

// Field order matches ggml's real block_iq4_xs exactly
struct block_iq4_xs_cu { __half d; uint16_t scales_h; uint8_t scales_l[QK_K/64]; uint8_t qs[QK_K/2]; }; // 136 bytes

__constant__ int8_t c_kvalues_iq4nl[16];

extern "C" {
const void* ms_grid_kvalues_iq4nl();
}

// __constant__ memory is per-device: cudaMemcpyToSymbol only ever populates
// the CURRENTLY-ACTIVE device's copy. A single process-wide bool here would
// let the first GPU touched in a run latch this true globally, silently
// skipping initialization (all-zero lookup table, all-zero dequantized
// weights) for every OTHER GPU in a multi-GPU config -- this was a real,
// confirmed bug (see CLAUDE.md's [SILU-TRACE-GPU] investigation, same
// pattern as cuda/iq2_matmul.cu and cuda/iq3_matmul.cu). Tracked per-device
// instead; MAX_CUDA_DEVICES is a generous static bound (no dynamic
// allocation needed for any realistic consumer multi-GPU setup).
#define MAX_CUDA_DEVICES 16
static bool iq4xs_table_ready[MAX_CUDA_DEVICES] = {false};
static void init_iq4xs_table() {
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev < 0 || dev >= MAX_CUDA_DEVICES) dev = 0;
    if (iq4xs_table_ready[dev]) return;
    cudaMemcpyToSymbol(c_kvalues_iq4nl, ms_grid_kvalues_iq4nl(), 16 * sizeof(int8_t));
    iq4xs_table_ready[dev] = true;
}

// ---- fused matmul (cooperative parallel dequant + warp-level dot reduction) ----
__global__ void iq4_xs_matmul_kernel(
    float* __restrict__ y, const float* __restrict__ x, const block_iq4_xs_cu* __restrict__ W,
    int S, int I, int O)
{
    int o = blockIdx.x;
    int s = blockIdx.y;
    int tid = threadIdx.x;
    
    if (o >= O || s >= S) return;
    
    int bpr = I / QK_K;
    const block_iq4_xs_cu* w_row = W + (size_t)o * bpr;
    const float* x_row = x + (size_t)s * I;

    // Pre-compute mathematical thread-mappings for cooperative dequantization
    // tid ranges from 0 to 255. 
    // There are 8 sub-blocks of 32 elements.
    const int ib = tid >> 5;                  // 0..7 (sub-block index)
    const int local_tid = tid & 31;           // 0..31 (index within sub-block)
    const int j = local_tid & 15;             // 0..15 (byte index within qs)
    const int is_high = local_tid >> 4;       // 0 or 1 (low nibble or high nibble)
    
    // Pre-compute scale shifts
    const int scales_l_idx = ib >> 1;
    const int scales_l_shift = (ib & 1) << 2; // 0 or 4
    const int scales_h_shift = ib << 1;       // 0, 2, 4, ... 14

    float acc = 0.0f;

    for (int b = 0; b < bpr; ++b) {
        // Cooperative Dequantization: each thread computes exactly 1 weight directly in-register
        const float d = __half2float(w_row[b].d);
        
        // Extract 6-bit scale index `ls`
        const uint8_t scales_l_val = w_row[b].scales_l[scales_l_idx];
        const uint16_t scales_h_val = w_row[b].scales_h;
        
        const int ls = ((scales_l_val >> scales_l_shift) & 0xf) |
                       (((scales_h_val >> scales_h_shift) & 3) << 4);
                       
        const float dl = d * (float)(ls - 32);
        
        // Extract the correct nibble for this specific thread
        const uint8_t qs_byte = w_row[b].qs[(ib << 4) + j];
        const int nibble = (qs_byte >> (is_high << 2)) & 0xf;
        
        // Lookup non-linear value and apply scale
        const float w_val = dl * (float)c_kvalues_iq4nl[nibble];
        
        // Immediate FMA without shared memory staging
        acc += x_row[(b * QK_K) + tid] * w_val;
    }

    // Warp-level shuffle reduction (far faster than block-wide shared memory tree)
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        acc += __shfl_down_sync(0xffffffff, acc, offset);
    }

    // Collect warp sums into shared memory
    __shared__ float warp_sums[8];
    const int lane_id = tid % 32;
    const int warp_id = tid / 32;

    if (lane_id == 0) {
        warp_sums[warp_id] = acc;
    }
    __syncthreads();

    // Final reduction of the 8 warp sums by the first warp
    if (warp_id == 0) {
        float sum = (lane_id < 8) ? warp_sums[lane_id] : 0.0f;
        
        #pragma unroll
        for (int offset = 4; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0x000000ff, sum, offset);
        }
        
        // Thread 0 writes the final dot product for this (s, o) pair
        if (tid == 0) {
            y[(size_t)s * O + o] = sum;
        }
    }
}

extern "C" void launch_iq4_xs_matmul(float* y, const float* x, const void* w, int S, int I, int O) {
    init_iq4xs_table();
    iq4_xs_matmul_kernel<<<dim3(O, S), 256>>>(y, x, (const block_iq4_xs_cu*)w, S, I, O);
}
