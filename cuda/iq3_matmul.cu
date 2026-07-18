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

struct block_iq3_xxs_cu { uint16_t d; uint8_t qs[3*QK_K/8]; }; // 98

__constant__ uint32_t c_iq3xxs_grid[256];
__constant__ uint8_t  c_ksigns_iq2xs[128];
__constant__ uint8_t  c_kmask_iq2xs[8];

// Host grid pointers (src/ggml_grids.cpp, backed by libggml tables).
extern "C" {
const void* ms_grid_iq3xxs();
const void* ms_grid_ksigns_iq2xs();
const void* ms_grid_kmask_iq2xs();
}

static bool iq3_grids_ready = false;
static void init_iq3_grids() {
    if (iq3_grids_ready) return;
    cudaMemcpyToSymbol(c_iq3xxs_grid,  ms_grid_iq3xxs(),       256 * sizeof(uint32_t));
    cudaMemcpyToSymbol(c_ksigns_iq2xs, ms_grid_ksigns_iq2xs(), 128);
    cudaMemcpyToSymbol(c_kmask_iq2xs,  ms_grid_kmask_iq2xs(),    8);
    iq3_grids_ready = true;
}

// Verbatim port of ggml's dequantize_row_iq3_xxs (ggml/src/ggml-quants.c).
// qs[0..63] holds 2 grid-byte-indices per (ib32,l) pair (64 = 8 ib32 * 8
// bytes); qs[64..95] ("scales_and_signs") holds one packed uint32 per ib32:
// top 4 bits = block scale index, low 28 bits = four 7-bit sign selectors
// (one per l). Assembled manually byte-by-byte (not via memcpy/reinterpret)
// to sidestep any device-side alignment/aliasing subtlety.
__device__ void dequant_iq3_xxs_block(const block_iq3_xxs_cu* xb, float* y) {
    const float d = __half2float(*reinterpret_cast<const __half*>(&xb->d));
    const uint8_t* qs = xb->qs;
    const uint8_t* scales_and_signs = qs + QK_K/4;  // qs + 64

    for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
        const uint8_t* sas = scales_and_signs + 4*ib32;
        uint32_t aux32 = (uint32_t)sas[0] | ((uint32_t)sas[1] << 8) |
                         ((uint32_t)sas[2] << 16) | ((uint32_t)sas[3] << 24);
        const float db = d * (0.5f + (aux32 >> 28)) * 0.5f;
        for (int l = 0; l < 4; ++l) {
            const uint8_t signs = c_ksigns_iq2xs[(aux32 >> 7*l) & 127];
            const uint8_t* grid1 = (const uint8_t*)&c_iq3xxs_grid[qs[2*l+0]];
            const uint8_t* grid2 = (const uint8_t*)&c_iq3xxs_grid[qs[2*l+1]];
            for (int j = 0; j < 4; ++j) {
                y[ib32*32 + l*8 + j+0] = db * (float)grid1[j] * ((signs & c_kmask_iq2xs[j+0]) ? -1.f : 1.f);
                y[ib32*32 + l*8 + j+4] = db * (float)grid2[j] * ((signs & c_kmask_iq2xs[j+4]) ? -1.f : 1.f);
            }
        }
        qs += 8;
    }
}

// ---- fused matmul (thread-0 verbatim dequant of a super-block + 256-way dot) ----
__global__ void iq3_xxs_matmul_kernel(
    float* __restrict__ y, const float* __restrict__ x, const block_iq3_xxs_cu* __restrict__ W,
    int S, int I, int O)
{
    int o = blockIdx.x, s = blockIdx.y, tid = threadIdx.x;
    if (o >= O || s >= S) return;
    int bpr = I / 256;
    const block_iq3_xxs_cu* w_row = W + (size_t)o * bpr;
    const float* x_row = x + (size_t)s * I;
    __shared__ float yq[256];
    float acc = 0.0f;
    for (int b = 0; b < bpr; ++b) {
        if (tid == 0) dequant_iq3_xxs_block(&w_row[b], yq);
        __syncthreads();
        if (tid < 256) acc += x_row[b * 256 + tid] * yq[tid];
        __syncthreads();
    }
    __shared__ float sm[256];
    sm[tid] = acc; __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) sm[tid] += sm[tid + stride];
        __syncthreads();
    }
    if (tid == 0) y[(size_t)s * O + o] = sm[0];
}

extern "C" void launch_iq3_xxs_matmul(float* y, const float* x, const void* w, int S, int I, int O) {
    init_iq3_grids();
    iq3_xxs_matmul_kernel<<<dim3(O, S), 256>>>(y, x, (const block_iq3_xxs_cu*)w, S, I, O);
}
