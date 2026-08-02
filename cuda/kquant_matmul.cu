// kquant_matmul.cu — fused dequant + matmul for K-quant weight tensors.
//   y[S,O] = x[S,I] @ W^T,  W packed row-major [O,I].
//
// The per-super-block dequant functions below are ported VERBATIM from ggml's
// dequantize_row_q2_K / q3_K / q4_K / q5_K / q6_K (ggml/src/ggml-quants.c) —
// this is the legitimate "deep change" case (CPU dequant -> GPU fused kernel),
// not a copy of the CPU path (which is linked via libggml). Block struct
// layouts match ggml's byte sizes (Q2_K=84, Q3_K=110, Q4_K=144, Q5_K=176,
// Q6_K=210 per 256 elems).
//
// Kernel: grid(O, S), 256 threads per (o,s). Each thread dequantizes exactly
// its own element (indexed purely by `tid`, no cross-thread dependency) and
// accumulates directly into a register, then a final cross-warp reduction
// combines all 256 threads' partial sums. FIX (real perf issue, found during
// a review sweep): this used to route each dequantized value through a
// `__shared__ yq[256]` array (DEQUANT wrote `yq[tid]`, the caller then read
// it straight back) bracketed by two `__syncthreads()` per super-block
// iteration -- entirely unnecessary, since every dequant function below only
// ever touches `y[tid]`/index `tid`, never another thread's slot, so nothing
// was actually being shared. That cost 2*bpr full block barriers per (o,s)
// dispatch for no reason (e.g. bpr=24 for I=6144 -> 48 syncs). The IQ2/IQ3/
// IQ4_XS sibling kernels (cuda/iq2_matmul.cu etc.) already accumulate
// straight into a register with no shared-memory round-trip; this file now
// matches that pattern -- DEQUANT returns a float by value instead of
// writing through a pointer.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

#define QK_K 256

struct block_q2_K_cu { uint8_t scales[16]; uint8_t qs[64]; __half d; __half dmin; };          
struct block_q4_K_cu { __half d; __half dmin; uint8_t scales[12]; uint8_t qs[128]; };          
struct block_q5_K_cu { __half d; __half dmin; uint8_t scales[12]; uint8_t qh[32]; uint8_t qs[128]; }; 
struct block_q6_K_cu { uint8_t ql[128]; uint8_t qh[64]; int8_t scales[16]; __half d; };        
struct block_q3_K_cu { uint8_t hmask[QK_K/8]; uint8_t qs[QK_K/4]; uint8_t scales[12]; __half d; }; 

__device__ inline void get_scale_min_k4_d(int j, const uint8_t* q, uint8_t* d, uint8_t* m) {
    if (j < 4) {
        *d = q[j] & 63; *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >>  4) | ((q[j - 0] >> 6) << 4);
    }
}

// ----------------------------------------------------------------------------
// COOPERATIVE DEQUANTIZATION
// Each thread computes exactly one float output based on its `tid` (0 to 255).
// ----------------------------------------------------------------------------

__device__ float dequant_q2_K_block_coop(const block_q2_K_cu* xb, int tid) {
    const float d   = __half2float(xb->d);
    const float min = __half2float(xb->dmin);

    int n_idx = tid / 128;            // 0 or 1
    int j = (tid % 128) / 32;         // 0..3
    int l = tid % 32;                 // 0..31

    int is = n_idx * 8 + j * 2;
    int shift = j * 2;
    const uint8_t* q_chunk = xb->qs + n_idx * 32;

    uint8_t sc = (l < 16) ? xb->scales[is] : xb->scales[is + 1];
    float dl = d * (sc & 0xF);
    float ml = min * (sc >> 4);

    return dl * ((int8_t)((q_chunk[l] >> shift) & 3)) - ml;
}

__device__ float dequant_q4_K_block_coop(const block_q4_K_cu* xb, int tid) {
    const float d   = __half2float(xb->d);
    const float min = __half2float(xb->dmin);

    int chunk = tid / 64;             // 0..3
    int l = tid % 64;                 // 0..63
    int is = chunk * 2;
    const uint8_t* q_chunk = xb->qs + chunk * 32;

    uint8_t sc0, m0, sc1, m1;
    get_scale_min_k4_d(is + 0, xb->scales, &sc0, &m0);
    get_scale_min_k4_d(is + 1, xb->scales, &sc1, &m1);

    if (l < 32) {
        return (d * sc0) * (q_chunk[l] & 0xF) - (min * m0);
    } else {
        return (d * sc1) * (q_chunk[l - 32] >> 4) - (min * m1);
    }
}

__device__ float dequant_q5_K_block_coop(const block_q5_K_cu* xb, int tid) {
    const float d   = __half2float(xb->d);
    const float min = __half2float(xb->dmin);

    int chunk = tid / 64;             // 0..3
    int l = tid % 64;                 // 0..63
    int is = chunk * 2;

    const uint8_t* ql_chunk = xb->qs + chunk * 32;
    const uint8_t* qh = xb->qh;

    uint8_t u1 = 1 << (chunk * 2);
    uint8_t u2 = 1 << (chunk * 2 + 1);

    uint8_t sc0, m0, sc1, m1;
    get_scale_min_k4_d(is + 0, xb->scales, &sc0, &m0);
    get_scale_min_k4_d(is + 1, xb->scales, &sc1, &m1);

    if (l < 32) {
        return (d * sc0) * ((ql_chunk[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - (min * m0);
    } else {
        return (d * sc1) * ((ql_chunk[l - 32] >> 4) + (qh[l - 32] & u2 ? 16 : 0)) - (min * m1);
    }
}

__device__ float dequant_q6_K_block_coop(const block_q6_K_cu* xb, int tid) {
    const float d = __half2float(xb->d);
    int n_idx = tid / 128;            // 0 or 1
    int group = (tid % 128) / 32;     // 0, 1, 2, or 3
    int l = tid % 32;                 // 0..31

    const uint8_t* ql = xb->ql + n_idx * 64;
    const uint8_t* qh = xb->qh + n_idx * 32;
    const int8_t*  sc = xb->scales + n_idx * 8;

    int is = l / 16;
    int8_t q_val;
    float sc_val;

    if (group == 0) {
        q_val = (int8_t)((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
        sc_val = sc[is + 0];
    } else if (group == 1) {
        q_val = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
        sc_val = sc[is + 2];
    } else if (group == 2) {
        q_val = (int8_t)((ql[l + 0] >> 4)  | (((qh[l] >> 4) & 3) << 4)) - 32;
        sc_val = sc[is + 4];
    } else {
        q_val = (int8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
        sc_val = sc[is + 6];
    }

    return d * sc_val * q_val;
}

__device__ float dequant_q3_K_block_coop(const block_q3_K_cu* xb, int tid) {
    const float d_all = __half2float(*reinterpret_cast<const __half*>(&xb->d));
    const uint32_t kmask1 = 0x03030303;
    const uint32_t kmask2 = 0x0f0f0f0f;

    // Fast redundant register evaluation instead of synced shared memory
    uint32_t aux[4];
    aux[0] = (uint32_t)xb->scales[0] | ((uint32_t)xb->scales[1] << 8) | ((uint32_t)xb->scales[2] << 16) | ((uint32_t)xb->scales[3] << 24);
    aux[1] = (uint32_t)xb->scales[4] | ((uint32_t)xb->scales[5] << 8) | ((uint32_t)xb->scales[6] << 16) | ((uint32_t)xb->scales[7] << 24);
    uint32_t tmp = (uint32_t)xb->scales[8] | ((uint32_t)xb->scales[9] << 8) | ((uint32_t)xb->scales[10] << 16) | ((uint32_t)xb->scales[11] << 24);
    aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
    aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
    const int8_t* scales = (const int8_t*)aux;

    int n_idx = tid / 128;            // 0 or 1
    int j = (tid % 128) / 32;         // 0..3
    int l = tid % 32;                 // 0..31
    
    int is = n_idx * 8 + j * 2;
    int shift = j * 2;
    uint8_t m = 1 << (n_idx * 4 + j); 
    
    const uint8_t* q_chunk = xb->qs + n_idx * 32;
    const uint8_t* hm_chunk = xb->hmask + n_idx * 16;

    int scale_idx = (l < 16) ? is : is + 1;
    float dl = d_all * (scales[scale_idx] - 32);

    return dl * ((int8_t)((q_chunk[l] >> shift) & 3) - ((hm_chunk[l] & m) ? 0 : 4));
}

// ---- fused matmul: cooperative dequant of a super-block + 256-way dot ----
#define KQUANT_MATMUL_KERNEL(NAME, BLOCKT, DEQUANT) \
__global__ void NAME##_kquant_matmul_kernel( \
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
        float wv = DEQUANT(&w_row[b], tid); \
        acc += x_row[b * 256 + tid] * wv; \
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

KQUANT_MATMUL_KERNEL(q2_K, block_q2_K_cu, dequant_q2_K_block_coop)
KQUANT_MATMUL_KERNEL(q3_K, block_q3_K_cu, dequant_q3_K_block_coop)
KQUANT_MATMUL_KERNEL(q4_K, block_q4_K_cu, dequant_q4_K_block_coop)
KQUANT_MATMUL_KERNEL(q5_K, block_q5_K_cu, dequant_q5_K_block_coop)
KQUANT_MATMUL_KERNEL(q6_K, block_q6_K_cu, dequant_q6_K_block_coop)

extern "C" void launch_q2_K_matmul(float* y, const float* x, const void* w, int S, int I, int O) {
    q2_K_kquant_matmul_kernel<<<dim3(O, S), 256>>>(y, x, (const block_q2_K_cu*)w, S, I, O);
}
extern "C" void launch_q3_K_matmul(float* y, const float* x, const void* w, int S, int I, int O) {
    q3_K_kquant_matmul_kernel<<<dim3(O, S), 256>>>(y, x, (const block_q3_K_cu*)w, S, I, O);
}
extern "C" void launch_q4_K_matmul(float* y, const float* x, const void* w, int S, int I, int O) {
    q4_K_kquant_matmul_kernel<<<dim3(O, S), 256>>>(y, x, (const block_q4_K_cu*)w, S, I, O);
}
extern "C" void launch_q5_K_matmul(float* y, const float* x, const void* w, int S, int I, int O) {
    q5_K_kquant_matmul_kernel<<<dim3(O, S), 256>>>(y, x, (const block_q5_K_cu*)w, S, I, O);
}
extern "C" void launch_q6_K_matmul(float* y, const float* x, const void* w, int S, int I, int O) {
    q6_K_kquant_matmul_kernel<<<dim3(O, S), 256>>>(y, x, (const block_q6_K_cu*)w, S, I, O);
}
