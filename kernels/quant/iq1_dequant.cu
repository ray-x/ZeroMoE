// iq1_dequant.cu — Optimized 1-bit and Ternary quantization CUDA kernels
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

// ============================================================
// TQ1_0 Fused Matmul Kernel (Optimized)
// ============================================================
__global__ void tq1_matmul_kernel_optimized(
    float* __restrict__ y,
    const float* __restrict__ x,
    const uint8_t* __restrict__ weights,
    const half* __restrict__ scales,
    int S, int I, int O
) {
    int o = blockIdx.x;
    int s = blockIdx.y;
    int tid = threadIdx.x;
    if (o >= O || s >= S) return;

    int blocks_per_row = I / 256;
    const int bytes_per_block = 54;
    const uint8_t* w_row = weights + (size_t)o * blocks_per_row * bytes_per_block;
    const half* s_row = scales + (size_t)o * blocks_per_row;
    const float* x_row = x + (size_t)s * I;

    float partial_sum = 0.0f;

    for (int blk = tid; blk < blocks_per_row; blk += blockDim.x) {
        float scale = __half2float(s_row[blk]);
        const uint8_t* blk_data = w_row + (size_t)blk * bytes_per_block;
        const float* x_blk = x_row + blk * 256;

        float block_sum = 0.0f;
        int val_idx = 0;

        // 1. Branchless Trit Decoding
        #pragma unroll 4
        for (int b = 0; b < 32; ++b) {
            uint32_t packed = blk_data[b];
            #pragma unroll
            for (int t = 0; t < 5; ++t) {
                // NVCC compiler will optimize /3 and %3 into fast multiply-adds via magic constants
                int trit = packed % 3; 
                packed /= 3;
                
                // Map (0, 1, 2) directly to (-1.0f, 0.0f, 1.0f) using math, avoiding branches entirely
                float w = (float)trit - 1.0f; 
                block_sum += w * x_blk[val_idx];
                val_idx++;
            }
        }
        
        // Handle overflow bytes
        #pragma unroll
        for (int b = 32; b < 36; ++b) {
            uint32_t packed = blk_data[b];
            #pragma unroll
            for (int t = 0; t < 5 && val_idx < 256; ++t) {
                int trit = packed % 3;
                packed /= 3;
                float w = (float)trit - 1.0f;
                block_sum += w * x_blk[val_idx];
                val_idx++;
            }
        }
        partial_sum += block_sum * scale;
    }

    // Warp + Block Reduction
    unsigned mask = 0xffffffff;
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        partial_sum += __shfl_down_sync(mask, partial_sum, offset);
    }

    static __shared__ float s_sum[32];
    int lane = tid % 32, warp = tid / 32;
    if (lane == 0) s_sum[warp] = partial_sum;
    __syncthreads();

    if (warp == 0) {
        partial_sum = (lane < blockDim.x / 32) ? s_sum[lane] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            partial_sum += __shfl_down_sync(mask, partial_sum, offset);
        }
        if (lane == 0) y[(size_t)s * O + o] = partial_sum;
    }
}

// ============================================================
// IQ1_S/IQ1_M "Fused Matmul Kernel (Optimized)" -- NOT A REAL IQ1_S/IQ1_M
// IMPLEMENTATION. Verified wrong this session (2026-09) while wiring up a
// real launch_iq1s_matmul for the first time: this kernel decodes each
// weight as a flat, independent sign bit (0/1 -> -1.0f/+1.0f) straight out
// of `weights + 2`, with no per-block scale group, no grid-codebook lookup,
// and no delta term. Real IQ1_S (ggml's block_iq1_s: `d` (half) + `qs[32]`
// + `qh[8]` (uint16), 50 bytes/256-block, confirmed against
// build/_deps/ggml-src/src/ggml-quants.c's own dequantize_row_iq1_s) is a
// genuine 2048-entry TERNARY grid-codebook format (like IQ2_XXS/IQ3_XXS,
// not like TQ1_0's simple per-weight trit): each 8-weight sub-group's grid
// INDEX is `qs[l] | (((qh[ib]>>3*l)&7)<<8)`, the grid row itself holds real
// {-1,0,+1} values (not just {-1,+1}), and every 32-weight group ALSO
// applies a per-group scale (`2*((qh[ib]>>12)&7)+1`) and a per-group signed
// delta offset added to every grid value before scaling
// (`IQ1S_DELTA=0.125`, sign from qh's top bit) -- none of which this
// function computes. Left in place, UNUSED by anything (no launch_iq1m_
// matmul exists to call it, and gpu_type_supported()/the dispatch table in
// cuda_manager.hpp never route IQ1_S to it), as a documented negative
// finding rather than silently deleted -- a correct launch_iq1s_matmul
// (verbatim port of dequantize_row_iq1_s, real ggml grid table via
// ms_grid_iq1s()/c_iq1s_grid) is defined further down this file instead.
// IQ1_M/TQ1_0 remain genuinely out of scope -- IQ1_M's real block layout
// (block_iq1_m: qs[32]+qh[16]+scales[16], a DIFFERENT 3-bit-scale/4-group-
// delta encoding, no per-block `d` at all) is close in spirit but not
// identical to IQ1_S's, and this kernel doesn't implement that either; not
// verified or fixed this session. TQ1_0's own kernel above (branchless trit
// decode, no grid table) is structurally closer to what a true ternary
// format needs and was NOT found wrong by this investigation -- but it also
// was not independently re-verified against dequantize_row_tq1_0 this
// session, so treat it as "not yet re-confirmed", not "confirmed correct".
// ============================================================
__global__ void iq1_matmul_kernel_optimized(
    float* __restrict__ y,
    const float* __restrict__ x,
    const uint8_t* __restrict__ weights,
    int S, int I, int O,
    int bytes_per_block
) {
    int o = blockIdx.x;
    int s = blockIdx.y;
    int tid = threadIdx.x;
    if (o >= O || s >= S) return;

    int blocks_per_row = I / 256;
    const uint8_t* w_row = weights + (size_t)o * blocks_per_row * bytes_per_block;
    const float* x_row = x + (size_t)s * I;

    float partial_sum = 0.0f;

    for (int blk = tid; blk < blocks_per_row; blk += blockDim.x) {
        const uint8_t* blk_data = w_row + (size_t)blk * bytes_per_block;
        float scale = __half2float(*(const half*)blk_data);
        
        // FIX (same class of bug as q8_0_matmul.cu -- see that file's own
        // comment for the full explanation, confirmed by a real Xid 13
        // "Misaligned Address" crash there): `bytes_per_block` is a
        // caller-supplied per-block byte stride with no guarantee it's a
        // multiple of 4, so `blk_data + 2` (blk_data itself already
        // `blk*bytes_per_block` bytes into the row) can land at a
        // non-4-byte-aligned address depending on `blk`'s value -- casting
        // that straight to `const uint32_t*` and dereferencing it risks the
        // identical hardware-illegal unaligned load. Read the 4 bytes
        // individually instead (uint8_t alignment is always 1, safe
        // unconditionally) and reassemble the same little-endian uint32_t a
        // wide read would have produced -- NVIDIA GPUs are little-endian,
        // matching this composition to the original bit-layout exactly.
        const uint8_t* signs_bytes = blk_data + 2;
        const float4* x_blk_f4 = reinterpret_cast<const float4*>(x_row + blk * 256);

        float block_sum = 0.0f;

        // 2. Vectorized Math and Branchless Accumulation
        #pragma unroll 4
        for (int i = 0; i < 8; ++i) { // 8 iterations * 32 bits = 256 weights
            const uint8_t* sb = signs_bytes + i * 4;
            uint32_t bitpack = (uint32_t)sb[0] | ((uint32_t)sb[1] << 8)
                              | ((uint32_t)sb[2] << 16) | ((uint32_t)sb[3] << 24);

            // Process 8 float4s (32 floats) per iteration
            #pragma unroll
            for (int v = 0; v < 8; ++v) {
                float4 x4 = x_blk_f4[i * 8 + v];
                
                // Extract bits and map 0 -> -1.0f, 1 -> 1.0f using: 2*bit - 1
                float w0 = (float)((bitpack >> 0) & 1) * 2.0f - 1.0f;
                float w1 = (float)((bitpack >> 1) & 1) * 2.0f - 1.0f;
                float w2 = (float)((bitpack >> 2) & 1) * 2.0f - 1.0f;
                float w3 = (float)((bitpack >> 3) & 1) * 2.0f - 1.0f;
                
                block_sum += w0 * x4.x + w1 * x4.y + w2 * x4.z + w3 * x4.w;
                bitpack >>= 4; // Shift down for the next float4
            }
        }
        partial_sum += block_sum * scale;
    }

    // Warp + block reduction
    unsigned mask = 0xffffffff;
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        partial_sum += __shfl_down_sync(mask, partial_sum, offset);
    }

    static __shared__ float s_sum[32];
    int lane = tid % 32, warp = tid / 32;
    if (lane == 0) s_sum[warp] = partial_sum;
    __syncthreads();

    if (warp == 0) {
        partial_sum = (lane < blockDim.x / 32) ? s_sum[lane] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            partial_sum += __shfl_down_sync(mask, partial_sum, offset);
        }
        if (lane == 0) y[(size_t)s * O + o] = partial_sum;
    }
}
// ============================================================
// REAL IQ1_S Fused Matmul Kernel -- verbatim port of ggml's own
// dequantize_row_iq1_s (build/_deps/ggml-src/src/ggml-quants.c), the same
// "port the CPU reference exactly, don't re-derive from bit-level first
// principles" discipline already used for iq2_matmul.cu/iq3_matmul.cu.
// block_iq1_s: `d` (half, 2 bytes) + `qs[QK_K/8]` (32 bytes, low 8 bits of
// each 8-weight sub-group's grid index) + `qh[QK_K/32]` (8 x uint16, high 3
// grid-index bits + 3-bit group scale + 1-bit delta sign, per 32-weight
// group) = 50 bytes/256-block, matching row_bytes()'s existing
// (I/256)*50.
//
// Cooperative layout (one thread per weight lane within a 256-block,
// matching iq2_matmul.cu's IQ2_MATMUL_KERNEL macro exactly): tid in
// [0,256) maps to ib=tid/32 (which of 8 32-weight groups), l=(tid%32)/8
// (which of 4 8-weight sub-groups within that group), j=tid%8 (which
// weight within the sub-group) -- the same (ib, l, j) triple
// dequantize_row_iq1_s's own nested loop walks.
// ============================================================
extern "C" {
    const void* ms_grid_iq1s(); // uint64_t[2048], see src/ggml_grids.cpp
}

#define IQ1S_DELTA_GPU 0.125f

__constant__ uint64_t c_iq1s_grid[2048];

// Per-device init, same MAX_CUDA_DEVICES/ready-array pattern as
// iq2_matmul.cu's init_iq2_grids() (this file's own separate translation
// unit needs its OWN guard array -- `static` gives internal linkage, so it
// cannot share iq2_matmul.cu's -- see that file's comment for the real,
// confirmed multi-GPU bug this pattern exists to avoid re-introducing).
#ifndef MAX_CUDA_DEVICES
#define MAX_CUDA_DEVICES 16
#endif
static bool iq1s_grid_ready[MAX_CUDA_DEVICES] = {false};
static void init_iq1s_grid() {
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev < 0 || dev >= MAX_CUDA_DEVICES) dev = 0;
    if (iq1s_grid_ready[dev]) return;
    cudaMemcpyToSymbol(c_iq1s_grid, ms_grid_iq1s(), 2048 * sizeof(uint64_t));
    iq1s_grid_ready[dev] = true;
}

struct block_iq1_s_cu {
    __half   d;
    uint8_t  qs[32];
    uint16_t qh[8];
};
static_assert(sizeof(block_iq1_s_cu) == 50, "block_iq1_s_cu must match row_bytes()'s (I/256)*50 and the real GGUF block_iq1_s layout");

__device__ __forceinline__ float dequant_iq1_s_coop(const block_iq1_s_cu* xb, int tid) {
    const float d = __half2float(xb->d);
    int ib  = tid / 32;        // which 32-weight group (0..7) -- indexes qh[]
    int rem = tid % 32;
    int l   = rem / 8;         // which 8-weight sub-group within the group (0..3)
    int j   = rem % 8;         // which weight within the sub-group (0..7)

    uint16_t qh_ib = xb->qh[ib];
    float dl = d * (float)(2 * ((qh_ib >> 12) & 7) + 1);
    float delta = (qh_ib & 0x8000) ? -IQ1S_DELTA_GPU : IQ1S_DELTA_GPU;

    uint16_t grid_idx = (uint16_t)xb->qs[ib * 4 + l] |
                         (uint16_t)(((qh_ib >> (3 * l)) & 7) << 8);
    uint64_t g = c_iq1s_grid[grid_idx];
    int8_t grid_j = ((const int8_t*)&g)[j];

    return dl * ((float)grid_j + delta);
}

__global__ void iq1_s_matmul_kernel(
    float* __restrict__ y, const float* __restrict__ x,
    const block_iq1_s_cu* __restrict__ W, int S, int I, int O)
{
    int o = blockIdx.x, s = blockIdx.y, tid = threadIdx.x;
    if (o >= O || s >= S) return;

    int bpr = I / 256;
    const block_iq1_s_cu* w_row = W + (size_t)o * bpr;
    const float* x_row = x + (size_t)s * I;

    float acc = 0.0f;
    for (int b = 0; b < bpr; ++b) {
        float w_val = dequant_iq1_s_coop(&w_row[b], tid);
        acc += x_row[b * 256 + tid] * w_val;
    }

    __shared__ float sm[256];
    sm[tid] = acc;
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) sm[tid] += sm[tid + stride];
        __syncthreads();
    }
    if (tid == 0) y[(size_t)s * O + o] = sm[0];
}

extern "C" void launch_iq1s_matmul(float* y, const float* x, const void* w, int S, int I, int O) {
    init_iq1s_grid();
    iq1_s_matmul_kernel<<<dim3(O, S), 256>>>(y, x, (const block_iq1_s_cu*)w, S, I, O);
}

// ============================================================
// IQ1_M -- verbatim port of ggml's dequantize_row_iq1_m
// (build/_deps/ggml-src/src/ggml-quants.c). Reuses the SAME 2048-entry
// c_iq1s_grid table as IQ1_S above (IQ1_M's own reference indexes the
// identical iq1s_grid, confirmed by reading the real ggml source) -- only
// the per-block scale/index packing differs. block_iq1_m has NO standalone
// `d` half; the super-block scale is packed into the top 4 bits of each of
// scales[]'s 4 uint16 words (a real, deliberate space-saving trick ggml
// uses here, not a bug). Same (ib, l, j) thread-lane decomposition as
// dequant_iq1_s_coop above: tid in [0,256), ib=tid/32 (0..7, which 32-
// weight group), l=(tid%32)/8 (0..3, which 8-weight sub-group), j=tid%8.
// ============================================================
struct block_iq1_m_cu {
    uint8_t qs[32];
    uint8_t qh[16];
    uint8_t scales[8];
};
static_assert(sizeof(block_iq1_m_cu) == 56, "block_iq1_m_cu must match row_bytes()'s (I/256)*56 and the real GGUF block_iq1_m layout");

union iq1m_scale_cu { __half f16; uint16_t u16; };

__device__ __forceinline__ float dequant_iq1_m_coop(const block_iq1_m_cu* xb, int tid) {
    const uint16_t* sc = (const uint16_t*)xb->scales;  // 4 x uint16, from the 8-byte scales[]
    iq1m_scale_cu scale;
    scale.u16 = (uint16_t)((sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) |
                            ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000));
    float d = __half2float(scale.f16);

    int ib  = tid / 32;   // 0..7 -- which 32-weight group
    int rem = tid % 32;
    int l   = rem / 8;    // 0..3 -- which 8-weight sub-group within the group
    int j   = rem % 8;    // 0..7 -- which weight within the sub-group

    uint16_t sc_pair = sc[ib / 2];
    int shift_base = 6 * (ib % 2);
    float dl1 = d * (float)(2 * ((sc_pair >> (shift_base + 0)) & 0x7) + 1);
    float dl2 = d * (float)(2 * ((sc_pair >> (shift_base + 3)) & 0x7) + 1);

    const uint8_t* qs = xb->qs + ib * 4;
    const uint8_t* qh = xb->qh + ib * 2;

    // Verbatim translation of the CPU reference's idx[]/delta[] arrays --
    // every thread computes all 4 (cheap bitwise ops) and picks its own l.
    uint16_t idx[4] = {
        (uint16_t)(qs[0] | (((uint16_t)qh[0] << 8) & 0x700)),
        (uint16_t)(qs[1] | (((uint16_t)qh[0] << 4) & 0x700)),
        (uint16_t)(qs[2] | (((uint16_t)qh[1] << 8) & 0x700)),
        (uint16_t)(qs[3] | (((uint16_t)qh[1] << 4) & 0x700)),
    };
    float delta[4] = {
        (qh[0] & 0x08) ? -IQ1S_DELTA_GPU : IQ1S_DELTA_GPU,
        (qh[0] & 0x80) ? -IQ1S_DELTA_GPU : IQ1S_DELTA_GPU,
        (qh[1] & 0x08) ? -IQ1S_DELTA_GPU : IQ1S_DELTA_GPU,
        (qh[1] & 0x80) ? -IQ1S_DELTA_GPU : IQ1S_DELTA_GPU,
    };
    float dl = (l < 2) ? dl1 : dl2;

    uint64_t g = c_iq1s_grid[idx[l]];
    int8_t grid_j = ((const int8_t*)&g)[j];
    return dl * ((float)grid_j + delta[l]);
}

__global__ void iq1_m_matmul_kernel(
    float* __restrict__ y, const float* __restrict__ x,
    const block_iq1_m_cu* __restrict__ W, int S, int I, int O)
{
    int o = blockIdx.x, s = blockIdx.y, tid = threadIdx.x;
    if (o >= O || s >= S) return;

    int bpr = I / 256;
    const block_iq1_m_cu* w_row = W + (size_t)o * bpr;
    const float* x_row = x + (size_t)s * I;

    float acc = 0.0f;
    for (int b = 0; b < bpr; ++b) {
        float w_val = dequant_iq1_m_coop(&w_row[b], tid);
        acc += x_row[b * 256 + tid] * w_val;
    }

    __shared__ float sm[256];
    sm[tid] = acc;
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) sm[tid] += sm[tid + stride];
        __syncthreads();
    }
    if (tid == 0) y[(size_t)s * O + o] = sm[0];
}

extern "C" void launch_iq1m_matmul(float* y, const float* x, const void* w, int S, int I, int O) {
    init_iq1s_grid();  // same shared grid table as IQ1_S, idempotent per-device
    iq1_m_matmul_kernel<<<dim3(O, S), 256>>>(y, x, (const block_iq1_m_cu*)w, S, I, O);
}
