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

struct block_iq2_xxs_cu { uint16_t d; uint16_t qs[QK_K/8]; };
struct block_iq2_xs_cu  { uint16_t d; uint16_t qs[QK_K/8]; uint8_t scales[QK_K/32]; };
struct block_iq2_s_cu   { uint16_t d; uint8_t qs[QK_K/4]; uint8_t qh[QK_K/32]; uint8_t scales[QK_K/32]; };

__constant__ uint64_t c_iq2xxs_grid[256];
__constant__ uint64_t c_iq2xs_grid[512];
__constant__ uint64_t c_iq2s_grid[1024];
__constant__ uint8_t  c_ksigns_iq2xs[128];
__constant__ uint8_t  c_kmask_iq2xs[8];

extern "C" {
    const void* ms_grid_iq2xxs();
    const void* ms_grid_iq2xs();
    const void* ms_grid_iq2s();
    const void* ms_grid_ksigns_iq2xs();
    const void* ms_grid_kmask_iq2xs();
}

// __constant__ memory is per-device: cudaMemcpyToSymbol only ever populates
// the CURRENTLY-ACTIVE device's copy. A single process-wide bool here would
// let the first GPU touched in a run latch this true globally, silently
// skipping initialization (all-zero grid/sign/mask tables, all-zero
// dequantized weights) for every OTHER GPU in a multi-GPU config -- this
// was a real, confirmed bug (see CLAUDE.md's [SILU-TRACE-GPU] investigation).
// Tracked per-device instead; MAX_CUDA_DEVICES is a generous static bound
// (no dynamic allocation needed for any realistic consumer multi-GPU setup).
#define MAX_CUDA_DEVICES 16
static bool iq2_grids_ready[MAX_CUDA_DEVICES] = {false};
static void init_iq2_grids() {
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev < 0 || dev >= MAX_CUDA_DEVICES) dev = 0;
    if (iq2_grids_ready[dev]) return;
    cudaMemcpyToSymbol(c_iq2xxs_grid,    ms_grid_iq2xxs(),       256 * sizeof(uint64_t));
    cudaMemcpyToSymbol(c_iq2xs_grid,     ms_grid_iq2xs(),        512 * sizeof(uint64_t));
    cudaMemcpyToSymbol(c_iq2s_grid,      ms_grid_iq2s(),        1024 * sizeof(uint64_t));
    cudaMemcpyToSymbol(c_ksigns_iq2xs,   ms_grid_ksigns_iq2xs(), 128);
    cudaMemcpyToSymbol(c_kmask_iq2xs,    ms_grid_kmask_iq2xs(),    8);
    iq2_grids_ready[dev] = true;
}

__device__ __forceinline__ float hf_to_f(const void* p) {
    return __half2float(*reinterpret_cast<const __half*>(p));
}

// Cooperative Dequantization: Thread `tid` unpacks exactly 1 weight value.
__device__ __forceinline__ float dequant_iq2_xxs_coop(const block_iq2_xxs_cu* xb, int tid) {
    const float d = hf_to_f(&xb->d);
    int ib32 = tid / 32;
    int l = (tid % 32) / 8;
    int j = tid % 8;

    const uint16_t* q16 = xb->qs + 4 * ib32;
    uint32_t aux32_1 = (uint32_t)q16[2] | ((uint32_t)q16[3] << 16);
    const float db = d * (0.5f + (aux32_1 >> 28)) * 0.25f;

    uint32_t aux32_0 = (uint32_t)q16[0] | ((uint32_t)q16[1] << 16);
    uint8_t aux8_l = ((const uint8_t*)&aux32_0)[l];

    uint64_t g = c_iq2xxs_grid[aux8_l];
    uint8_t grid_j = ((const uint8_t*)&g)[j];
    uint8_t signs = c_ksigns_iq2xs[(aux32_1 >> 7*l) & 127];

    return db * (float)grid_j * ((signs & c_kmask_iq2xs[j]) ? -1.f : 1.f);
}

__device__ __forceinline__ float dequant_iq2_xs_coop(const block_iq2_xs_cu* xb, int tid) {
    const float d = hf_to_f(&xb->d);
    int ib32 = tid / 32;
    int l = (tid % 32) / 8;
    int j = tid % 8;

    const float db0 = d * (0.5f + (xb->scales[ib32] & 0xf)) * 0.25f;
    const float db1 = d * (0.5f + (xb->scales[ib32] >>  4)) * 0.25f;
    const float dl = (l < 2) ? db0 : db1;

    const uint16_t q = xb->qs[4 * ib32 + l];
    uint64_t g = c_iq2xs_grid[q & 511];
    uint8_t grid_j = ((const uint8_t*)&g)[j];
    uint8_t signs = c_ksigns_iq2xs[q >> 9];

    return dl * (float)grid_j * ((signs & c_kmask_iq2xs[j]) ? -1.f : 1.f);
}

__device__ __forceinline__ float dequant_iq2_s_coop(const block_iq2_s_cu* xb, int tid) {
    const float d = hf_to_f(&xb->d);
    int ib32 = tid / 32;
    int l = (tid % 32) / 8;
    int j = tid % 8;

    const float db0 = d * (0.5f + (xb->scales[ib32] & 0xf)) * 0.25f;
    const float db1 = d * (0.5f + (xb->scales[ib32] >>  4)) * 0.25f;
    const float dl = (l < 2) ? db0 : db1;

    const uint8_t* qs = xb->qs + ib32 * 4;
    const uint8_t* signs = xb->qs + QK_K/8 + ib32 * 4;

    const uint32_t idx = qs[l] | ((xb->qh[ib32] << (8 - 2*l)) & 0x300);
    uint64_t g = c_iq2s_grid[idx];
    uint8_t grid_j = ((const uint8_t*)&g)[j];

    return dl * (float)grid_j * ((signs[l] & c_kmask_iq2xs[j]) ? -1.f : 1.f);
}

// Reduction: 256-thread shared-memory tree (8 rounds of shared-mem add +
// __syncthreads()). An external suggestion proposed warp-shuffle reduction
// here instead, matching the pattern already used in cuda/q8_0_matmul.cu,
// cuda/kquant_matmul.cu, cuda/iq3_matmul.cu, cuda/iq4xs_matmul.cu, and
// cuda/mxfp4_matmul.cu -- REAL-HARDWARE TESTED (RTX 3090, real gate/up
// expert shape I=6144 O=2048) and found to be NO improvement here: a
// first, FLAWED A/B (comparing dense_matmul_host's full H2D+kernel+D2H
// pipeline for tree against a raw kernel-only launch for shfl) wrongly
// suggested 1.45-1.64x; a corrected kernel-only-both-sides re-measurement
// showed 0.9999x (statistically identical), confirmed independently by a
// real [PREFILL] run's own dense_cmp counter barely moving (1.57s->1.55s,
// ~1.3%, within noise) after the shfl variant briefly shipped. Root cause:
// unlike q8_0/K-quant's simpler dequant loops (where the tree reduction's
// fixed 8-sync cost is a meaningful fraction of a cheap kernel), IQ2_XXS's
// per-element dequant (grid/sign table lookups from __constant__ memory)
// keeps all 256 threads doing real, comparatively expensive work every
// iteration -- the reduction stage is a much smaller slice of total kernel
// time here, so switching its algorithm doesn't move the needle. Reverted
// to the tree kernel (this comment kept as a marker so a future session
// doesn't re-propose and re-test the identical idea from scratch).
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
    float acc = 0.0f; \
    for (int b = 0; b < bpr; ++b) { \
        float w_val = DEQUANT(&w_row[b], tid); \
        acc += x_row[b * 256 + tid] * w_val; \
    } \
    __shared__ float sm[256]; \
    sm[tid] = acc; \
    __syncthreads(); \
    for (int stride = 128; stride > 0; stride >>= 1) { \
        if (tid < stride) sm[tid] += sm[tid + stride]; \
        __syncthreads(); \
    } \
    if (tid == 0) y[(size_t)s * O + o] = sm[0]; \
}

IQ2_MATMUL_KERNEL(iq2_xxs, block_iq2_xxs_cu, dequant_iq2_xxs_coop)
IQ2_MATMUL_KERNEL(iq2_xs,  block_iq2_xs_cu,  dequant_iq2_xs_coop)
IQ2_MATMUL_KERNEL(iq2_s,   block_iq2_s_cu,   dequant_iq2_s_coop)

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