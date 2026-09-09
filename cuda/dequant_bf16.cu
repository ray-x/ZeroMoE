// dequant_bf16.cu -- standalone quant -> BF16 dequantisation, so a prefill
// GEMM can be handed to cuBLAS instead of running through this project's own
// fused matmul kernels.
//
// WHY (all measured 2026-09-04 at attn_qkv's real prefill shape,
// S=3189 I=2560 O=10240, 167.2 GFLOP, on the RTX 3090):
//
//     our fused q8_0 kernel (GEMV-shaped)   194.41 ms     778 GFLOPS (2.2% peak)
//     dequant->f32  2.53 + cuBLAS SGEMM 6.84 =  9.37 ms              (20.7x)
//     dequant->bf16 0.70 + cuBLAS GemmEx 2.35 = 3.11 ms              (62.5x)
//
// The BF16 GEMM alone runs at 71.1 TFLOPS -- essentially 100% of this card's
// BF16 tensor-core peak. The dequantisation is only 0.70 ms of that 3.11 ms
// path, i.e. it is nearly free; keeping the weights permanently resident in
// BF16 instead is NOT an option (dense 5.5 -> 9.9 GB, experts 77 -> 242 GB,
// against 40 GB of VRAM), so dequantising per call into a reused grow-only
// scratch buffer is the right shape.
//
// Every fused kernel in cuda/ decodes these formats already; the decode is
// simply not callable on its own. This file exposes it.
//
// Q8_0 first because EVERY dense projection in qwen4exp is Q8_0 (attn_qkv,
// attn_gate, ssm_out, attn_q/k/v/output, hc_*, shexp) -- one kernel covers
// the whole 57.1s dense-projection bucket. The expert types (Q4_K/Q5_K/Q5_1)
// are the separate, larger 111.9s bucket and come next.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>

// Real on-disk GGUF layout (ggml-common.h block_q8_0): 2-byte half scale +
// 32 int8 quants = 34 bytes. NOT 4-byte aligned, so qs is read byte-wise --
// same discipline as q8_0_matmul.cu, which took a real Xid 13 "Misaligned
// Address" fault from assuming otherwise.
struct block_q8_0_deq { __half d; int8_t qs[32]; };

__global__ void dequant_q8_0_to_bf16_kernel(__nv_bfloat16* __restrict__ dst,
                                            const block_q8_0_deq* __restrict__ src,
                                            int64_t nblocks) {
    const int64_t b = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nblocks) return;
    const float d = __half2float(src[b].d);
    const int8_t* qs = src[b].qs;
    __nv_bfloat16* o = dst + b * 32;
    #pragma unroll
    for (int i = 0; i < 32; ++i) o[i] = __float2bfloat16(d * (float)qs[i]);
}

// n_elements must be a whole number of 32-element blocks (always true for a
// real GGUF Q8_0 tensor); the launcher refuses otherwise rather than write a
// partial trailing block.
extern "C" void launch_dequant_q8_0_to_bf16(void* dst_bf16, const void* src_q8_0,
                                            int64_t n_elements) {
    if (!dst_bf16 || !src_q8_0 || n_elements <= 0 || (n_elements % 32) != 0) return;
    const int64_t nblocks = n_elements / 32;
    const int threads = 256;
    const int64_t grid = (nblocks + threads - 1) / threads;
    dequant_q8_0_to_bf16_kernel<<<(unsigned)grid, threads>>>(
        (__nv_bfloat16*)dst_bf16, (const block_q8_0_deq*)src_q8_0, nblocks);
}

__global__ void f32_to_bf16_kernel(__nv_bfloat16* __restrict__ dst,
                                   const float* __restrict__ src, int64_t n) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2bfloat16(src[i]);
}

// Activations change every call (unlike weights), so this runs per GEMM.
// Measured at 0.06 ms for [3189 x 2560] -- negligible.
extern "C" void launch_f32_to_bf16(void* dst_bf16, const float* src_f32, int64_t n) {
    if (!dst_bf16 || !src_f32 || n <= 0) return;
    const int threads = 256;
    const int64_t grid = (n + threads - 1) / threads;
    f32_to_bf16_kernel<<<(unsigned)grid, threads>>>((__nv_bfloat16*)dst_bf16, src_f32, n);
}


// ---------------------------------------------------------------------------
// Expert weight types: Q4_K / Q5_K / Q5_1.
//
// These are the routed-expert bucket (112.9s of a 291s prefill on qwen4exp --
// gate/up are Q4_K on 47 layers and Q5_K on 1, down is Q5_1 on 43 and Q8_0 on
// 5). Measured headroom: a single [62x2560]x[2560x640] BF16 GEMM is 0.0109 ms,
// so all 73,728 expert GEMMs in a 3,189-token prefill would total ~0.8s.
//
// The per-element decode below is copied VERBATIM from the already-verified
// device helpers in cuda/kquant_matmul.cu (dequant_q4_K_block_coop,
// dequant_q5_K_block_coop) and cuda/q5_1_matmul.cu. It is duplicated rather
// than shared because this project builds without -rdc=true, so a __device__
// function cannot cross translation units -- the same reason iq3_matmul.cu
// keeps its own copy of the sign/grid tables rather than sharing
// iq2_matmul.cu's. If the originals are ever corrected, correct these too.
struct block_q4_K_deq { __half d; __half dmin; uint8_t scales[12]; uint8_t qs[128]; };
struct block_q5_K_deq { __half d; __half dmin; uint8_t scales[12]; uint8_t qh[32]; uint8_t qs[128]; };
struct block_q6_K_deq { uint8_t ql[128]; uint8_t qh[64]; int8_t scales[16]; __half d; };
struct block_q5_1_deq { __half d; __half m; uint8_t qh[4]; uint8_t qs[16]; };

__device__ inline void get_scale_min_k4_deq(int j, const uint8_t* q,
                                            uint8_t* d, uint8_t* m) {
    if (j < 4) { *d = q[j] & 63; *m = q[j + 4] & 63; }
    else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >>  4) | ((q[j - 0] >> 6) << 4);
    }
}

// One CUDA block per 256-element super-block; thread tid emits element tid.
__global__ void dequant_q4_K_to_bf16_kernel(__nv_bfloat16* __restrict__ dst,
                                            const block_q4_K_deq* __restrict__ src,
                                            int64_t nblocks) {
    const int64_t sb = blockIdx.x;
    if (sb >= nblocks) return;
    const int tid = threadIdx.x;                 // 0..255
    const block_q4_K_deq* xb = src + sb;
    const float d = __half2float(xb->d), mn = __half2float(xb->dmin);
    const int chunk = tid / 64, l = tid % 64, is = chunk * 2;
    const uint8_t* qc = xb->qs + chunk * 32;
    uint8_t sc0, m0, sc1, m1;
    get_scale_min_k4_deq(is + 0, xb->scales, &sc0, &m0);
    get_scale_min_k4_deq(is + 1, xb->scales, &sc1, &m1);
    const float v = (l < 32) ? ((d * sc0) * (qc[l]      & 0xF) - (mn * m0))
                             : ((d * sc1) * (qc[l - 32] >>  4) - (mn * m1));
    dst[sb * 256 + tid] = __float2bfloat16(v);
}

__global__ void dequant_q5_K_to_bf16_kernel(__nv_bfloat16* __restrict__ dst,
                                            const block_q5_K_deq* __restrict__ src,
                                            int64_t nblocks) {
    const int64_t sb = blockIdx.x;
    if (sb >= nblocks) return;
    const int tid = threadIdx.x;
    const block_q5_K_deq* xb = src + sb;
    const float d = __half2float(xb->d), mn = __half2float(xb->dmin);
    const int chunk = tid / 64, l = tid % 64, is = chunk * 2;
    const uint8_t* qc = xb->qs + chunk * 32;
    const uint8_t* qh = xb->qh;
    const uint8_t u1 = 1 << (chunk * 2), u2 = 1 << (chunk * 2 + 1);
    uint8_t sc0, m0, sc1, m1;
    get_scale_min_k4_deq(is + 0, xb->scales, &sc0, &m0);
    get_scale_min_k4_deq(is + 1, xb->scales, &sc1, &m1);
    const float v = (l < 32)
        ? ((d * sc0) * ((qc[l]      & 0xF) + ((qh[l]      & u1) ? 16 : 0)) - (mn * m0))
        : ((d * sc1) * ((qc[l - 32] >>  4) + ((qh[l - 32] & u2) ? 16 : 0)) - (mn * m1));
    dst[sb * 256 + tid] = __float2bfloat16(v);
}

// Q5_1: element k of the 32-element block uses bit k of qh, and the low
// nibble of qs[k] for k<16 / the high nibble of qs[k-16] for k>=16 -- the
// uniformity derived in q5_1_matmul.cu's header from ggml's own shifts.
__global__ void dequant_q5_1_to_bf16_kernel(__nv_bfloat16* __restrict__ dst,
                                            const block_q5_1_deq* __restrict__ src,
                                            int64_t nblocks) {
    const int64_t b = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nblocks) return;
    const block_q5_1_deq& blk = src[b];
    const float d = __half2float(blk.d), m = __half2float(blk.m);
    const uint32_t qh = (uint32_t)blk.qh[0] | ((uint32_t)blk.qh[1] << 8) |
                        ((uint32_t)blk.qh[2] << 16) | ((uint32_t)blk.qh[3] << 24);
    __nv_bfloat16* o = dst + b * 32;
    #pragma unroll
    for (int k = 0; k < 32; ++k) {
        const uint8_t qsj = blk.qs[k & 15];
        const int nib = (k < 16) ? (qsj & 0x0F) : (qsj >> 4);
        const int q = nib | (int)(((qh >> k) & 1u) << 4);
        o[k] = __float2bfloat16((float)q * d + m);
    }
}

// type ids match GGUFType / ggml: 7=Q5_1, 8=Q8_0, 12=Q4_K, 13=Q5_K.
// Returns false for anything without a kernel here, so the caller keeps its
// existing fused-kernel path instead of computing garbage.
// Q6_K -> BF16. Verbatim port of ggml's dequantize_row_q6_K: each 256-element
// super-block is two 128-element halves, and within a half each of 32 lanes
// produces FOUR outputs (l, l+32, l+64, l+96) from two ql nibbles and two
// 2-bit qh fields, each with its own int8 scale:
//   q1 = (ql[l]    & 0xF) | ((qh[l] >> 0 & 3) << 4) - 32   -> sc[is+0]
//   q2 = (ql[l+32] & 0xF) | ((qh[l] >> 2 & 3) << 4) - 32   -> sc[is+2]
//   q3 = (ql[l]    >>  4) | ((qh[l] >> 4 & 3) << 4) - 32   -> sc[is+4]
//   q4 = (ql[l+32] >>  4) | ((qh[l] >> 6 & 3) << 4) - 32   -> sc[is+6]
// One thread per OUTPUT element (256 threads) picking the single (half,l,k) it
// owns, matching the Q4_K/Q5_K kernels above rather than looping like ggml.
__global__ void dequant_q6_K_to_bf16_kernel(__nv_bfloat16* __restrict__ dst,
                                            const block_q6_K_deq* __restrict__ src,
                                            int64_t nblocks) {
    const int64_t sb = blockIdx.x;
    if (sb >= nblocks) return;
    const int tid = threadIdx.x;                 // 0..255
    const block_q6_K_deq* xb = src + sb;
    const float d = __half2float(xb->d);

    const int half_i = tid >> 7;                 // 0 or 1 (which 128-element half)
    const int j      = tid & 127;                // 0..127 within that half
    const int l      = j & 31;                   // 0..31
    const int k      = j >> 5;                   // 0..3 -> q1..q4
    const int is     = l >> 4;                   // 0 or 1

    const uint8_t* ql = xb->ql + half_i * 64;
    const uint8_t* qh = xb->qh + half_i * 32;
    const int8_t*  sc = xb->scales + half_i * 8;

    const int  ql_idx   = (k & 1) ? (l + 32) : l;   // q2,q4 read ql[l+32]
    const bool hi_nib   = (k >= 2);                 // q3,q4 take the high nibble
    const int  qh_shift = k * 2;
    const int  sc_idx   = is + k * 2;

    const int lo = hi_nib ? (ql[ql_idx] >> 4) : (ql[ql_idx] & 0xF);
    const int q  = (lo | (((qh[l] >> qh_shift) & 3) << 4)) - 32;
    dst[sb * 256 + tid] = __float2bfloat16(d * (float)sc[sc_idx] * (float)q);
}

extern "C" bool launch_dequant_to_bf16(int ggml_type, void* dst_bf16,
                                       const void* src, int64_t n_elements) {
    if (!dst_bf16 || !src || n_elements <= 0) return false;
    const int threads = 256;
    switch (ggml_type) {
        case 8: {                                    // Q8_0
            if (n_elements % 32) return false;
            launch_dequant_q8_0_to_bf16(dst_bf16, src, n_elements);
            return true;
        }
        case 7: {                                    // Q5_1
            if (n_elements % 32) return false;
            const int64_t nb = n_elements / 32;
            dequant_q5_1_to_bf16_kernel<<<(unsigned)((nb + threads - 1) / threads), threads>>>(
                (__nv_bfloat16*)dst_bf16, (const block_q5_1_deq*)src, nb);
            return true;
        }
        case 12: {                                   // Q4_K
            if (n_elements % 256) return false;
            const int64_t nb = n_elements / 256;
            dequant_q4_K_to_bf16_kernel<<<(unsigned)nb, 256>>>(
                (__nv_bfloat16*)dst_bf16, (const block_q4_K_deq*)src, nb);
            return true;
        }
        case 13: {                                   // Q5_K
            if (n_elements % 256) return false;
            const int64_t nb = n_elements / 256;
            dequant_q5_K_to_bf16_kernel<<<(unsigned)nb, 256>>>(
                (__nv_bfloat16*)dst_bf16, (const block_q5_K_deq*)src, nb);
            return true;
        }
        case 14: {                                   // Q6_K
            if (n_elements % 256) return false;
            const int64_t nb = n_elements / 256;
            dequant_q6_K_to_bf16_kernel<<<(unsigned)nb, 256>>>(
                (__nv_bfloat16*)dst_bf16, (const block_q6_K_deq*)src, nb);
            return true;
        }
        default: return false;
    }
}
