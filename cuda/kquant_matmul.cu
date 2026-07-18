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
// Kernel: grid(O, S), 256 threads per (o,s). Thread 0 dequants one 256-elem
// super-block into shared mem (verbatim port), all 256 threads dot one element
// each, accumulate per-thread, then tree-reduce. Correctness-first; the
// serial dequant by thread 0 is an optimization target (parallelize elementwise).
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

#define QK_K 256

struct block_q2_K_cu { uint8_t scales[16]; uint8_t qs[64]; __half d; __half dmin; };          // 84
struct block_q4_K_cu { __half d; __half dmin; uint8_t scales[12]; uint8_t qs[128]; };          // 144
struct block_q5_K_cu { __half d; __half dmin; uint8_t scales[12]; uint8_t qh[32]; uint8_t qs[128]; }; // 176
struct block_q6_K_cu { uint8_t ql[128]; uint8_t qh[64]; int8_t scales[16]; __half d; };        // 210
// Field order matches ggml's real block_q3_K (ggml-common.h) EXACTLY --
// hmask, qs, scales, d, in that order. An earlier version of this struct had
// the fields in a different order (d first), which compiles fine (same
// total size, 110 bytes) but silently reinterprets every field at the wrong
// byte offset -- e.g. xb->d would read bytes that are actually the start of
// hmask, producing a garbage fp16 "scale" that can itself be NaN/Inf, or a
// huge finite value that overflows downstream. Caught by src/main_test.cpp's
// Test 5 (CUDA fused kernel vs CPU reference on real ggml-quantized bytes),
// which is exactly the class of bug that test exists to catch: this
// compiled and ran without crashing, it just silently produced wrong (NaN)
// numbers -- structurally identical to the row_bytes() stride bug that
// caused an actual SIGSEGV earlier in Q3_K's sibling IQ3_XXS kernel.
struct block_q3_K_cu { uint8_t hmask[QK_K/8]; uint8_t qs[QK_K/4]; uint8_t scales[12]; __half d; }; // 110

// ---- dequant ports (from ggml-quants.c) ----
__device__ inline void get_scale_min_k4_d(int j, const uint8_t* q, uint8_t* d, uint8_t* m) {
    if (j < 4) {
        *d = q[j] & 63; *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >>  4) | ((q[j - 0] >> 6) << 4);
    }
}

__device__ void dequant_q2_K_block(const block_q2_K_cu* xb, float* y) {
    const float d   = __half2float(xb->d);
    const float min = __half2float(xb->dmin);
    const uint8_t* q = xb->qs;
    int is = 0;
    float dl, ml;
    for (int n = 0; n < QK_K; n += 128) {
        int shift = 0;
        for (int j = 0; j < 4; ++j) {
            uint8_t sc = xb->scales[is++];
            dl = d * (sc & 0xF); ml = min * (sc >> 4);
            for (int l = 0; l < 16; ++l) y[n + j*32 + l] = dl * ((int8_t)((q[l] >> shift) & 3)) - ml;
            sc = xb->scales[is++];
            dl = d * (sc & 0xF); ml = min * (sc >> 4);
            for (int l = 0; l < 16; ++l) y[n + j*32 + 16 + l] = dl * ((int8_t)((q[l+16] >> shift) & 3)) - ml;
            shift += 2;
        }
        q += 32;
    }
}

// Verbatim port of ggml's dequantize_row_q4_K (ggml/src/ggml-quants.c) —
// structurally identical to dequant_q5_K_block below minus the high-bit qh
// array: 4-bit values read directly from qs's low/high nibbles, same
// get_scale_min_k4_d scale/min unpacking scheme shared with Q5_K.
__device__ void dequant_q4_K_block(const block_q4_K_cu* xb, float* y) {
    const uint8_t* q = xb->qs;
    const float d   = __half2float(xb->d);
    const float min = __half2float(xb->dmin);
    int is = 0; uint8_t sc, m;
    for (int j = 0; j < QK_K; j += 64) {
        get_scale_min_k4_d(is + 0, xb->scales, &sc, &m); float d1 = d * sc, m1 = min * m;
        get_scale_min_k4_d(is + 1, xb->scales, &sc, &m); float d2 = d * sc, m2 = min * m;
        for (int l = 0; l < 32; ++l) y[j + l]      = d1 * (q[l] & 0xF) - m1;
        for (int l = 0; l < 32; ++l) y[j + 32 + l] = d2 * (q[l] >>  4) - m2;
        q += 32; is += 2;
    }
}

__device__ void dequant_q5_K_block(const block_q5_K_cu* xb, float* y) {
    const uint8_t* ql = xb->qs;
    const uint8_t* qh = xb->qh;
    const float d   = __half2float(xb->d);
    const float min = __half2float(xb->dmin);
    int is = 0; uint8_t sc, m; uint8_t u1 = 1, u2 = 2;
    for (int j = 0; j < QK_K; j += 64) {
        get_scale_min_k4_d(is + 0, xb->scales, &sc, &m); float d1 = d * sc, m1 = min * m;
        get_scale_min_k4_d(is + 1, xb->scales, &sc, &m); float d2 = d * sc, m2 = min * m;
        for (int l = 0; l < 32; ++l) y[j + l]        = d1 * ((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1;
        for (int l = 0; l < 32; ++l) y[j + 32 + l]   = d2 * ((ql[l] >> 4)  + (qh[l] & u2 ? 16 : 0)) - m2;
        ql += 32; is += 2; u1 <<= 2; u2 <<= 2;
    }
}

__device__ void dequant_q6_K_block(const block_q6_K_cu* xb, float* y) {
    const float d = __half2float(xb->d);
    const uint8_t* ql = xb->ql;
    const uint8_t* qh = xb->qh;
    const int8_t*  sc = xb->scales;
    for (int n = 0; n < QK_K; n += 128) {
        for (int l = 0; l < 32; ++l) {
            int is = l / 16;
            int8_t q1 = (int8_t)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
            int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
            int8_t q3 = (int8_t)((ql[l +  0] >> 4)  | (((qh[l] >> 4) & 3) << 4)) - 32;
            int8_t q4 = (int8_t)((ql[l + 32] >> 4)  | (((qh[l] >> 6) & 3) << 4)) - 32;
            y[n + l +  0] = d * sc[is + 0] * q1;
            y[n + l + 32] = d * sc[is + 2] * q2;
            y[n + l + 64] = d * sc[is + 4] * q3;
            y[n + l + 96] = d * sc[is + 6] * q4;
        }
        ql += 64; qh += 32; sc += 8;
    }
}

// Q3_K (110 B / 256). Verbatim port of ggml dequantize_row_q3_K. 3-bit values
// with a per-bit sign mask (hmask) and 16 6-bit scales unpacked from 12 bytes.
__device__ void dequant_q3_K_block(const block_q3_K_cu* xb, float* y) {
    const float d_all = __half2float(*reinterpret_cast<const __half*>(&xb->d));
    const uint8_t* q  = xb->qs;
    const uint8_t* hm = xb->hmask;
    const uint32_t kmask1 = 0x03030303;
    const uint32_t kmask2 = 0x0f0f0f0f;

    // Unpack the 16 6-bit scales from scales[12] into aux[4] (16 int8).
    uint32_t aux[4];
    aux[0] = (uint32_t)xb->scales[0]  | ((uint32_t)xb->scales[1]  << 8) |
             ((uint32_t)xb->scales[2]  << 16) | ((uint32_t)xb->scales[3]  << 24);
    aux[1] = (uint32_t)xb->scales[4]  | ((uint32_t)xb->scales[5]  << 8) |
             ((uint32_t)xb->scales[6]  << 16) | ((uint32_t)xb->scales[7]  << 24);
    uint32_t tmp = (uint32_t)xb->scales[8]  | ((uint32_t)xb->scales[9]  << 8) |
                   ((uint32_t)xb->scales[10] << 16) | ((uint32_t)xb->scales[11] << 24);
    aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
    aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
    const int8_t* scales = (const int8_t*)aux;

    uint8_t m = 1;   // persists across n iterations — 8 distinct masks total
    int is = 0;
    for (int n = 0; n < QK_K; n += 128) {
        int shift = 0;
        for (int j = 0; j < 4; ++j) {
            float dl = d_all * (scales[is++] - 32);
            for (int l = 0; l < 16; ++l)
                y[n + j*32 + l] = dl * ((int8_t)((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4));
            dl = d_all * (scales[is++] - 32);
            for (int l = 0; l < 16; ++l)
                y[n + j*32 + 16 + l] = dl * ((int8_t)((q[l+16] >> shift) & 3) - ((hm[l+16] & m) ? 0 : 4));
            shift += 2;
            m <<= 1;
        }
        q += 32;
    }
}

// ---- fused matmul: thread-0 verbatim dequant of a super-block + 256-way dot ----
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

KQUANT_MATMUL_KERNEL(q2_K, block_q2_K_cu, dequant_q2_K_block)
KQUANT_MATMUL_KERNEL(q3_K, block_q3_K_cu, dequant_q3_K_block)
KQUANT_MATMUL_KERNEL(q4_K, block_q4_K_cu, dequant_q4_K_block)
KQUANT_MATMUL_KERNEL(q5_K, block_q5_K_cu, dequant_q5_K_block)
KQUANT_MATMUL_KERNEL(q6_K, block_q6_K_cu, dequant_q6_K_block)

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
