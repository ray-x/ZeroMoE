// iq2_matmul.cu — fused dequant + matmul for IQ2 weight tensors (XXS/XS/S).
//   y[S,O] = x[S,I] @ W^T, W packed [O,I].
//
// IQ2 uses 8-byte codebook grids + sign tables. The grid DATA comes from
// libggml (via src/ggml_grids.cpp) and is copied once into __constant__ memory.
// The per-super-block dequant below is a verbatim port of ggml's
// dequantize_row_iq2_xxs / xs / s (ggml/src/ggml-quants.c). Block struct layouts
// match ggml byte sizes (XXS=66, XS=74, S=82 per 256 elems).
//
// Kernel: grid(O, S), 256 threads per (o,s). Thread 0 dequants one 256-elem
// super-block into shared mem, all 256 threads dot one element each, then
// tree-reduce — same structure as kquant_matmul.cu.
//
// IQ3_XXS lives in its own file (cuda/iq3_matmul.cu) rather than here, even
// though it reuses this file's sign/mask table VALUES conceptually (ggml's
// dequantize_row_iq3_xxs calls the same ksigns_iq2xs/kmask_iq2xs tables) —
// no relocatable device code (-rdc=true) is configured in this project, so
// __constant__ memory can't be shared across .cu translation units anyway;
// iq3_matmul.cu has its own copies.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

#define QK_K 256

// Block structs (match ggml block_iq2_xxs/xs/s field order & sizes).
struct block_iq2_xxs_cu { uint16_t d; uint16_t qs[QK_K/8]; };                         // 66
struct block_iq2_xs_cu  { uint16_t d; uint16_t qs[QK_K/8]; uint8_t scales[QK_K/32]; }; // 74
struct block_iq2_s_cu   { uint16_t d; uint8_t qs[QK_K/4]; uint8_t qh[QK_K/32]; uint8_t scales[QK_K/32]; }; // 82

// Grids in constant memory (populated once by init_iq2_grids).
__constant__ uint64_t c_iq2xxs_grid[256];
__constant__ uint64_t c_iq2xs_grid[512];
__constant__ uint64_t c_iq2s_grid[1024];
__constant__ uint8_t  c_ksigns_iq2xs[128];
__constant__ uint8_t  c_kmask_iq2xs[8];

// Host grid pointers (src/ggml_grids.cpp, backed by libggml tables).
extern "C" {
const void* ms_grid_iq2xxs();
const void* ms_grid_iq2xs();
const void* ms_grid_iq2s();
const void* ms_grid_ksigns_iq2xs();
const void* ms_grid_kmask_iq2xs();
}

static bool iq2_grids_ready = false;
static void init_iq2_grids() {
    if (iq2_grids_ready) return;
    cudaMemcpyToSymbol(c_iq2xxs_grid,    ms_grid_iq2xxs(),       256 * sizeof(uint64_t));
    cudaMemcpyToSymbol(c_iq2xs_grid,     ms_grid_iq2xs(),        512 * sizeof(uint64_t));
    cudaMemcpyToSymbol(c_iq2s_grid,      ms_grid_iq2s(),        1024 * sizeof(uint64_t));
    cudaMemcpyToSymbol(c_ksigns_iq2xs,   ms_grid_ksigns_iq2xs(), 128);
    cudaMemcpyToSymbol(c_kmask_iq2xs,    ms_grid_kmask_iq2xs(),    8);
    iq2_grids_ready = true;
}

__device__ __forceinline__ float hf_to_f(const void* p) {
    return __half2float(*reinterpret_cast<const __half*>(p));
}

// ---- verbatim ports of ggml dequantize_row_iq2_* (grids read from __constant__) ----
__device__ void dequant_iq2_xxs_block(const block_iq2_xxs_cu* xb, float* y) {
    const float d = hf_to_f(&xb->d);
    uint32_t aux32[2];
    const uint8_t* aux8 = (const uint8_t*)aux32;
    for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
        const uint16_t* q16 = xb->qs + 4*ib32;
        aux32[0] = (uint32_t)q16[0] | ((uint32_t)q16[1] << 16);
        aux32[1] = (uint32_t)q16[2] | ((uint32_t)q16[3] << 16);
        const float db = d * (0.5f + (aux32[1] >> 28)) * 0.25f;
        for (int l = 0; l < 4; ++l) {
            uint64_t g = c_iq2xxs_grid[aux8[l]];
            const uint8_t* grid = (const uint8_t*)&g;
            const uint8_t signs = c_ksigns_iq2xs[(aux32[1] >> 7*l) & 127];
            for (int j = 0; j < 8; ++j)
                y[ib32*32 + l*8 + j] = db * (float)grid[j] * ((signs & c_kmask_iq2xs[j]) ? -1.f : 1.f);
        }
    }
}

__device__ void dequant_iq2_xs_block(const block_iq2_xs_cu* xb, float* y) {
    const float d = hf_to_f(&xb->d);
    for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
        const float db0 = d * (0.5f + (xb->scales[ib32] & 0xf)) * 0.25f;
        const float db1 = d * (0.5f + (xb->scales[ib32] >>  4)) * 0.25f;
        for (int l = 0; l < 4; ++l) {
            const uint16_t q = xb->qs[4*ib32 + l];
            uint64_t g = c_iq2xs_grid[q & 511];
            const uint8_t* grid = (const uint8_t*)&g;
            const uint8_t signs = c_ksigns_iq2xs[q >> 9];
            const float dl = (l/2 == 0) ? db0 : db1;
            for (int j = 0; j < 8; ++j)
                y[ib32*32 + l*8 + j] = dl * (float)grid[j] * ((signs & c_kmask_iq2xs[j]) ? -1.f : 1.f);
        }
    }
}

__device__ void dequant_iq2_s_block(const block_iq2_s_cu* xb, float* y) {
    const float d = hf_to_f(&xb->d);
    const uint8_t* qs    = xb->qs;
    const uint8_t* qh    = xb->qh;
    const uint8_t* signs = xb->qs + QK_K/8;  // signs overlap the back half of qs
    for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
        const float db0 = d * (0.5f + (xb->scales[ib32] & 0xf)) * 0.25f;
        const float db1 = d * (0.5f + (xb->scales[ib32] >>  4)) * 0.25f;
        for (int l = 0; l < 4; ++l) {
            const float dl = (l/2 == 0) ? db0 : db1;
            const uint32_t idx = qs[l] | ((qh[ib32] << (8 - 2*l)) & 0x300);
            uint64_t g = c_iq2s_grid[idx];
            const uint8_t* grid = (const uint8_t*)&g;
            for (int j = 0; j < 8; ++j)
                y[ib32*32 + l*8 + j] = dl * (float)grid[j] * ((signs[l] & c_kmask_iq2xs[j]) ? -1.f : 1.f);
        }
        qs += 4; signs += 4;
    }
}

// ---- fused matmul (thread-0 verbatim dequant of a super-block + 256-way dot) ----
#define IQ2_MATMUL_KERNEL(NAME, BLOCKT, DEQUANT) \
__global__ void NAME##_iq2_matmul_kernel( \
    float* __restrict__ y, const float* __restrict__ x, const BLOCKT* __restrict__ W, \
    int S, int I, int O) \
{ \
    int o = blockIdx.x, s = blockIdx.y, tid = threadIdx.x; \
    if (o >= O || s >= S) return; \
    int bpr = I / 256; \
    const BLOCKT* w_row = W + (size_t)o * bpr; \
    const float* x_row = x + (size_t)s * I; \
    __shared__ float yq[256]; \
    float acc = 0.0f; \
    for (int b = 0; b < bpr; ++b) { \
        if (tid == 0) DEQUANT(&w_row[b], yq); \
        __syncthreads(); \
        if (tid < 256) acc += x_row[b * 256 + tid] * yq[tid]; \
        __syncthreads(); \
    } \
    __shared__ float sm[256]; \
    sm[tid] = acc; __syncthreads(); \
    for (int stride = 128; stride > 0; stride >>= 1) { \
        if (tid < stride) sm[tid] += sm[tid + stride]; \
        __syncthreads(); \
    } \
    if (tid == 0) y[(size_t)s * O + o] = sm[0]; \
}

IQ2_MATMUL_KERNEL(iq2_xxs, block_iq2_xxs_cu, dequant_iq2_xxs_block)
IQ2_MATMUL_KERNEL(iq2_xs,  block_iq2_xs_cu,  dequant_iq2_xs_block)
IQ2_MATMUL_KERNEL(iq2_s,   block_iq2_s_cu,   dequant_iq2_s_block)

extern "C" void launch_iq2_xxs_matmul(float* y, const float* x, const void* w, int S, int I, int O) {
    init_iq2_grids();
    iq2_xxs_iq2_matmul_kernel<<<dim3(O, S), 256>>>(y, x, (const block_iq2_xxs_cu*)w, S, I, O);
}
extern "C" void launch_iq2_xs_matmul(float* y, const float* x, const void* w, int S, int I, int O) {
    init_iq2_grids();
    iq2_xs_iq2_matmul_kernel<<<dim3(O, S), 256>>>(y, x, (const block_iq2_xs_cu*)w, S, I, O);
}
extern "C" void launch_iq2_s_matmul(float* y, const float* x, const void* w, int S, int I, int O) {
    init_iq2_grids();
    iq2_s_iq2_matmul_kernel<<<dim3(O, S), 256>>>(y, x, (const block_iq2_s_cu*)w, S, I, O);
}
