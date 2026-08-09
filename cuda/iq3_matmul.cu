// iq3_matmul.cu — fused dequant + matmul for IQ3_XXS weight tensors.
//   y[S,O] = x[S,I] @ W^T, W packed [O,I].
//
// IQ3_XXS uses a 256-entry codebook grid (4 packed bytes per entry) plus the
// SAME sign/mask tables as IQ2 (confirmed against ggml's own reference:
// dequantize_row_iq3_xxs in ggml/src/ggml-quants.c calls ksigns_iq2xs/
// kmask_iq2xs directly — there is no separate IQ3 sign table in the format).
// Grid/sign/mask DATA all come from libggml (via src/ggml_grids.cpp) and are
// copied once into this file's own __constant__ memory. No relocatable
// device code (-rdc=true) is configured in this project's CMakeLists.txt,
// so __constant__ symbols are NOT shared across .cu translation units —
// these are this file's own copies, independent of iq2_matmul.cu's.
//
// The dequant below is a verbatim port of ggml's dequantize_row_iq3_xxs.
// Block struct matches ggml's block_iq3_xxs byte size (98 bytes/256 elems:
// ggml_half d + uint8_t qs[3*QK_K/8]).
//
// Kernel: grid(O, S), 256 threads per (o,s). Thread 0 dequants one 256-elem
// super-block into shared mem, all 256 threads dot one element each, then
// tree-reduce — same structure as iq2_matmul.cu / kquant_matmul.cu.


#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

#define QK_K 256

struct block_iq3_xxs_cu { uint16_t d; uint8_t qs[3*QK_K/8]; }; // 98 bytes

__constant__ uint32_t c_iq3xxs_grid[256];
__constant__ uint8_t  c_ksigns_iq2xs[128];
__constant__ uint8_t  c_kmask_iq2xs[8];

extern "C" {
const void* ms_grid_iq3xxs();
const void* ms_grid_ksigns_iq2xs();
const void* ms_grid_kmask_iq2xs();
}

// __constant__ memory is per-device: cudaMemcpyToSymbol only ever populates
// the CURRENTLY-ACTIVE device's copy. A single process-wide bool here would
// let the first GPU touched in a run latch this true globally, silently
// skipping initialization (all-zero grid/sign/mask tables, all-zero
// dequantized weights) for every OTHER GPU in a multi-GPU config -- this
// was a real, confirmed bug (see CLAUDE.md's [SILU-TRACE-GPU] investigation,
// same pattern as cuda/iq2_matmul.cu). Tracked per-device instead;
// MAX_CUDA_DEVICES is a generous static bound (no dynamic allocation needed
// for any realistic consumer multi-GPU setup).
#define MAX_CUDA_DEVICES 16
static bool iq3_grids_ready[MAX_CUDA_DEVICES] = {false};
static void init_iq3_grids() {
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev < 0 || dev >= MAX_CUDA_DEVICES) dev = 0;
    if (iq3_grids_ready[dev]) return;
    cudaMemcpyToSymbol(c_iq3xxs_grid,  ms_grid_iq3xxs(),       256 * sizeof(uint32_t));
    cudaMemcpyToSymbol(c_ksigns_iq2xs, ms_grid_ksigns_iq2xs(), 128);
    cudaMemcpyToSymbol(c_kmask_iq2xs,  ms_grid_kmask_iq2xs(),    8);
    iq3_grids_ready[dev] = true;
}

// ---- fused matmul (cooperative parallel dequant + warp-level dot reduction) ----
__global__ void iq3_xxs_matmul_kernel(
    float* __restrict__ y, const float* __restrict__ x, const block_iq3_xxs_cu* __restrict__ W,
    int S, int I, int O)
{
    int o = blockIdx.x;
    int s = blockIdx.y;
    int tid = threadIdx.x;
    
    if (o >= O || s >= S) return;
    
    int bpr = I / QK_K;
    const block_iq3_xxs_cu* w_row = W + (size_t)o * bpr;
    const float* x_row = x + (size_t)s * I;
    
    // Pre-compute mathematical thread-mappings for cooperative dequantization
    // tid ranges from 0 to 255.
    const int ib32 = tid >> 5;                  // 0..7
    const int l = (tid >> 3) & 3;               // 0..3
    const int j_full = tid & 7;                 // 0..7
    const int is_second_half = j_full >> 2;     // 0 or 1
    const int j = j_full & 3;                   // 0..3

    float acc = 0.0f;

    for (int b = 0; b < bpr; ++b) {
        // Cooperative Dequantization: each thread computes exactly 1 weight directly in-register
        const float d = __half2float(*reinterpret_cast<const __half*>(&w_row[b].d));
        
        const uint8_t* qs_base = w_row[b].qs + (ib32 * 8);
        const uint8_t* sas = w_row[b].qs + 64 + (4 * ib32);

        uint32_t aux32 = (uint32_t)sas[0] | ((uint32_t)sas[1] << 8) |
                         ((uint32_t)sas[2] << 16) | ((uint32_t)sas[3] << 24);
                         
        const float db = d * (0.5f + (aux32 >> 28)) * 0.5f;
        const uint8_t signs = c_ksigns_iq2xs[(aux32 >> (7 * l)) & 127];
        
        uint8_t q_idx = qs_base[2 * l + is_second_half];
        const uint8_t* grid = (const uint8_t*)&c_iq3xxs_grid[q_idx];
        
        float w_val = db * (float)grid[j] * ((signs & c_kmask_iq2xs[j_full]) ? -1.f : 1.f);
        
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

extern "C" void launch_iq3_xxs_matmul(float* y, const float* x, const void* w, int S, int I, int O) {
    init_iq3_grids();
    iq3_xxs_matmul_kernel<<<dim3(O, S), 256>>>(y, x, (const block_iq3_xxs_cu*)w, S, I, O);
}


