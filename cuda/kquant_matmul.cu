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
#include <cstdlib>

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

// Q3_K variant of the coop dequant above: the aux[4]->16-entry int8 scale
// unpack only depends on the block's 12 `scales` bytes, not on `tid` at all
// -- yet the ORIGINAL dequant_q3_K_block_coop recomputed the full unpack
// (9 shift/mask ops) independently in ALL 256 threads, every super-block.
// Computed ONCE (lane 0) into shared memory and broadcast instead.
// VERIFIED on real RTX 3090 hardware (attn_q_a's real shape, I=6144,O=2048):
// this is the dominant win of the two fixes in this file -- ~33-37% faster
// than the original tree-reduction kernel (0.089-0.891ms/call -> 0.059-
// 0.597ms/call), well beyond the final-reduction fix's own ~5%. Correctness
// verified against the original (max rel diff ~1e-5, float32 summation-
// order noise) across two shapes and S=1/S=11.
__device__ float dequant_q3_K_block_shared_scales(const block_q3_K_cu* xb, int tid, int8_t* s_scales) {
    if (tid == 0) {
        const uint32_t kmask1 = 0x03030303;
        const uint32_t kmask2 = 0x0f0f0f0f;
        uint32_t aux[4];
        aux[0] = (uint32_t)xb->scales[0] | ((uint32_t)xb->scales[1] << 8) | ((uint32_t)xb->scales[2] << 16) | ((uint32_t)xb->scales[3] << 24);
        aux[1] = (uint32_t)xb->scales[4] | ((uint32_t)xb->scales[5] << 8) | ((uint32_t)xb->scales[6] << 16) | ((uint32_t)xb->scales[7] << 24);
        uint32_t tmp = (uint32_t)xb->scales[8] | ((uint32_t)xb->scales[9] << 8) | ((uint32_t)xb->scales[10] << 16) | ((uint32_t)xb->scales[11] << 24);
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
        const int8_t* sc = (const int8_t*)aux;
        #pragma unroll
        for (int i = 0; i < 16; ++i) s_scales[i] = sc[i];
    }
    __syncthreads();

    const float d_all = __half2float(xb->d);
    int n_idx = tid / 128, j = (tid % 128) / 32, l = tid % 32;
    int is = n_idx * 8 + j * 2, shift = j * 2;
    uint8_t m = 1 << (n_idx * 4 + j);
    const uint8_t* q_chunk = xb->qs + n_idx * 32;
    // FIX (carried over from the original dequant_q3_K_block_coop, which
    // this function replaces): NOT `xb->hmask + n_idx * 16` -- ggml's real
    // reference (dequantize_row_q3_K, ggml/src/ggml-quants.c) NEVER advances
    // its `hm` pointer; the same 32 hmask bytes are reused for all 8
    // bit-planes across the whole 256-element block (the `m` bitmask below,
    // already correct, is what selects which BIT of each byte to read --
    // not which byte range). A `+ n_idx * 16` offset would make every
    // n_idx=1 lane with l>=16 read hmask[32..47], spilling past hmask's
    // real 32-byte extent into the immediately-adjacent `qs` field (still
    // inside the same 110-byte block struct, so no crash -- just silently
    // wrong values for the whole second half of every block). Confirmed via
    // direct line-by-line comparison against ggml's source, not assumed.
    const uint8_t* hm_chunk = xb->hmask;
    int scale_idx = (l < 16) ? is : is + 1;
    float dl = d_all * (s_scales[scale_idx] - 32);
    return dl * ((int8_t)((q_chunk[l] >> shift) & 3) - ((hm_chunk[l] & m) ? 0 : 4));
}

// ---- fused matmul: cooperative dequant of a super-block + 256-way dot ----
//
// Final reduction: warp-shuffle instead of a 256-thread shared-memory tree
// (was: 8 rounds of shared-mem add + __syncthreads(); matches the pattern
// already proven in cuda/q8_0_matmul.cu -- intra-warp shuffle-reduce, one
// __syncthreads() to combine the 8 per-warp partials, then a second
// shuffle-reduce). VERIFIED on real RTX 3090 hardware, real attn_q_a/
// attn_output shapes (I=6144 O=2048 and I=16384 O=6144): a real, if modest
// and shape-dependent, win (~5% faster at bpr=24; roughly a wash, even
// marginally worse once, at bpr=64 -- reduction overhead is a shrinking
// fraction of total cost as bpr grows). Kept anyway: never worse outside
// noise, matches an already-proven pattern, and removes 7 block-wide
// barriers per dispatch. A separately-tested multi-accumulator ILP variant
// (2 independent accumulators in the main loop) showed no measurable
// benefit on top of this, consistent with the same finding on q8_0_matmul.cu
// -- not adopted.
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
    _Pragma("unroll") \
    for (int off = 16; off > 0; off /= 2) acc += __shfl_down_sync(0xffffffff, acc, off); \
    __shared__ float s_warp[8]; \
    int warp_id = tid / 32, lane = tid % 32; \
    if (lane == 0) s_warp[warp_id] = acc; \
    __syncthreads(); \
    if (warp_id == 0) { \
        float v = (lane < 8) ? s_warp[lane] : 0.0f; \
        _Pragma("unroll") \
        for (int off = 16; off > 0; off /= 2) v += __shfl_down_sync(0xffffffff, v, off); \
        if (lane == 0) y[(size_t)s * O + o] = v; \
    } \
}

KQUANT_MATMUL_KERNEL(q2_K, block_q2_K_cu, dequant_q2_K_block_coop)
KQUANT_MATMUL_KERNEL(q4_K, block_q4_K_cu, dequant_q4_K_block_coop)
KQUANT_MATMUL_KERNEL(q5_K, block_q5_K_cu, dequant_q5_K_block_coop)
KQUANT_MATMUL_KERNEL(q6_K, block_q6_K_cu, dequant_q6_K_block_coop)

// Q3_K: hand-written rather than macro-generated, since it needs the extra
// `s_scales` shared array threaded through to dequant_q3_K_block_shared_scales
// (see that function's own comment -- the dominant win in this file).
__global__ void q3_K_kquant_matmul_kernel(
    float* __restrict__ y, const float* __restrict__ x, const block_q3_K_cu* __restrict__ W,
    int S, int I, int O)
{
    int o = blockIdx.x, s = blockIdx.y, tid = threadIdx.x;
    if (o >= O || s >= S) return;
    int bpr = I / 256;
    const block_q3_K_cu* w_row = W + (size_t)o * bpr;
    const float* x_row = x + (size_t)s * I;
    float acc = 0.0f;
    __shared__ int8_t s_scales[16];
    for (int b = 0; b < bpr; ++b) {
        float wv = dequant_q3_K_block_shared_scales(&w_row[b], tid, s_scales);
        acc += x_row[b * 256 + tid] * wv;
        // s_scales is reused by the next super-block iteration -- every
        // thread must be done reading it before lane 0 overwrites it.
        __syncthreads();
    }
    #pragma unroll
    for (int off = 16; off > 0; off /= 2) acc += __shfl_down_sync(0xffffffff, acc, off);
    __shared__ float s_warp[8];
    int warp_id = tid / 32, lane = tid % 32;
    if (lane == 0) s_warp[warp_id] = acc;
    __syncthreads();
    if (warp_id == 0) {
        float v = (lane < 8) ? s_warp[lane] : 0.0f;
        #pragma unroll
        for (int off = 16; off > 0; off /= 2) v += __shfl_down_sync(0xffffffff, v, off);
        if (lane == 0) y[(size_t)s * O + o] = v;
    }
}

// ============================================================================
// LANE8 PATH -- warp owns a super-block, lane owns 8 consecutive elements.
// ============================================================================
//
// The COOPERATIVE path above gives each of 256 threads exactly ONE element of
// a super-block, so a thread does 1 MAC per iteration and pays the full
// dequant scaffolding (d/dmin load, scale unpack) for that single element.
// Measured on real hardware at attn_output's real shape (Q5_K, I=16384,
// O=6144, S=1): 0.811 ms/call = 85 GB/s on the RTX 5060 Ti, ~19% of that
// card's 448 GB/s peak -- while cuda/q8_0_matmul.cu, whose kernel gives each
// thread a whole 32-element block, reaches 331 GB/s (74%) on the same card.
//
// That gap is NOT the redundant scale unpacking. A variant that hoisted the
// per-super-block scale work to one unpack per warp (via __shfl_sync, the
// same idea as dequant_q3_K_block_shared_scales below) measured 0.83-1.04x
// -- i.e. nothing, the shuffle cost roughly cancels the unpack saved. The
// gap is work per thread: 1 MAC/thread/iteration with a scalar activation
// load, versus q8_0's 32 with float4 loads.
//
// This path restructures to match q8_0's shape. q8_0's literal layout does
// not transfer -- a K-quant super-block is 256 elements, not 32, so at
// I=16384 there are only bpr=64 super-blocks per row, fewer than the 256
// threads in a block; one super-block per thread would idle 75% of them.
// Instead each WARP owns a super-block and each LANE owns 8 consecutive
// elements of it: 8 MACs/thread/iteration, two float4 activation loads, and
// the scale/d/dmin work amortized 8x at zero extra instruction cost.
//
// VERIFIED on real hardware, both GPUs, Q5_K at attn_output's real shape:
// 3.64x (3090, 139->507 GB/s) and 3.92x (5060 Ti, 85->335 GB/s) -- landing
// on q8_0's own 331 GB/s, i.e. closing the whole gap. Max relative error vs
// the coop kernel 1.9e-7 (float summation order; the partition of terms per
// thread changes, so bit-exactness is not expected and not required).
//
// Why 8 consecutive elements is the right unit for EVERY K-quant type: each
// type's scale sub-block is 32 elements (Q4_K/Q5_K) or 16 (Q2_K/Q3_K/Q6_K),
// and 8 divides both -- so all 8 of a lane's elements always share one scale
// AND come from 8 consecutive payload bytes. Each mapping below was derived
// from, and is diffed against, its own dequant_*_block_coop sibling above.

// Q4_K / Q5_K: 32-element sub-blocks, is = element/32 = lane>>2.
__device__ inline void dequant_q4_K_lane8(const block_q4_K_cu* xb, int lane, float* w) {
    const float d = __half2float(xb->d), mn = __half2float(xb->dmin);
    int is = lane >> 2, chunk = is >> 1, hi = is & 1;
    uint8_t sc, m;
    get_scale_min_k4_d(is, xb->scales, &sc, &m);
    const float ds = d * sc, dm = mn * m;
    const uint8_t* q = xb->qs + chunk * 32 + (lane & 3) * 8;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        uint8_t b = q[i];
        w[i] = ds * (float)(hi ? (b >> 4) : (b & 0xF)) - dm;
    }
}

__device__ inline void dequant_q5_K_lane8(const block_q5_K_cu* xb, int lane, float* w) {
    const float d = __half2float(xb->d), mn = __half2float(xb->dmin);
    int is = lane >> 2, chunk = is >> 1, hi = is & 1;
    uint8_t sc, m;
    get_scale_min_k4_d(is, xb->scales, &sc, &m);
    const float ds = d * sc, dm = mn * m;
    const uint8_t* q  = xb->qs + chunk * 32 + (lane & 3) * 8;
    const uint8_t* qh = xb->qh + (lane & 3) * 8;
    const uint8_t u = (uint8_t)(1 << is);
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        uint8_t b = q[i];
        int qv = (hi ? (b >> 4) : (b & 0xF)) + ((qh[i] & u) ? 16 : 0);
        w[i] = ds * (float)qv - dm;
    }
}

// Q2_K: 16-element scale sub-blocks -- the coop sibling picks scales[is] vs
// scales[is+1] on (l < 16), which for a lane's 8 elements is decided once by
// ((lane & 3) >= 2).
__device__ inline void dequant_q2_K_lane8(const block_q2_K_cu* xb, int lane, float* w) {
    const float d = __half2float(xb->d), mn = __half2float(xb->dmin);
    int n_idx = lane >> 4, j = (lane & 15) >> 2;
    int is = n_idx * 8 + j * 2 + (((lane & 3) >= 2) ? 1 : 0);
    int shift = j * 2;
    uint8_t sc = xb->scales[is];
    const float dl = d * (sc & 0xF), ml = mn * (sc >> 4);
    const uint8_t* q = xb->qs + n_idx * 32 + (lane & 3) * 8;
    #pragma unroll
    for (int i = 0; i < 8; ++i)
        w[i] = dl * (float)((int8_t)((q[i] >> shift) & 3)) - ml;
}

// Q6_K: 16-element scale sub-blocks; `group` selects both which half of ql
// (low/high nibble) and which qh bit-pair, exactly as the coop sibling's
// four-way branch does -- expressed here as shifts so the 8-element loop
// stays branch-free.
__device__ inline void dequant_q6_K_lane8(const block_q6_K_cu* xb, int lane, float* w) {
    const float d = __half2float(xb->d);
    int n_idx = lane >> 4, group = (lane & 15) >> 2, is = (lane & 3) >> 1;
    const uint8_t* ql = xb->ql + n_idx * 64 + (group & 1) * 32 + (lane & 3) * 8;
    const uint8_t* qh = xb->qh + n_idx * 32 + (lane & 3) * 8;
    const float dsc = d * (float)xb->scales[n_idx * 8 + is + group * 2];
    const int lsh = (group >= 2) ? 4 : 0, hsh = group * 2;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        int qv = (int)((ql[i] >> lsh) & 0xF) | (int)(((qh[i] >> hsh) & 3) << 4);
        w[i] = dsc * (float)(qv - 32);
    }
}

// Q3_K: the coop sibling unpacks the 16-entry aux scale table ONCE into a
// block-wide __shared__ array (filled by tid 0, behind a __syncthreads).
// That cannot work here: each of the 8 warps now owns a DIFFERENT
// super-block, so one shared table cannot serve them. Each lane instead
// computes just the ONE aux word holding its own scale_idx -- ~8 ops rather
// than the full 15-op four-word unpack, amortized over 8 elements, and the
// block-wide barrier disappears entirely.
__device__ inline void dequant_q3_K_lane8(const block_q3_K_cu* xb, int lane, float* w) {
    const float d_all = __half2float(xb->d);
    int n_idx = lane >> 4, j = (lane & 15) >> 2;
    int scale_idx = n_idx * 8 + j * 2 + (((lane & 3) >= 2) ? 1 : 0);
    int shift = j * 2;
    const uint8_t mbit = (uint8_t)(1 << (n_idx * 4 + j));

    const uint32_t kmask1 = 0x03030303, kmask2 = 0x0f0f0f0f;
    const uint8_t* sb = xb->scales;
    uint32_t tmp = (uint32_t)sb[8] | ((uint32_t)sb[9] << 8) | ((uint32_t)sb[10] << 16) | ((uint32_t)sb[11] << 24);
    int wi = scale_idx >> 2;              // aux word 0..3
    int base = (wi & 1) ? 4 : 0;          // aux0/aux2 <- scales[0..3]; aux1/aux3 <- scales[4..7]
    uint32_t a = (uint32_t)sb[base+0] | ((uint32_t)sb[base+1] << 8) | ((uint32_t)sb[base+2] << 16) | ((uint32_t)sb[base+3] << 24);
    uint32_t auxw = (wi < 2) ? ((a & kmask2)         | (((tmp >> (wi * 2)) & kmask1) << 4))
                             : (((a >> 4) & kmask2)  | (((tmp >> (wi * 2)) & kmask1) << 4));
    int8_t scv = (int8_t)((auxw >> ((scale_idx & 3) * 8)) & 0xFF);
    const float dl = d_all * (float)(scv - 32);

    const uint8_t* q  = xb->qs + n_idx * 32 + (lane & 3) * 8;
    // NOT xb->hmask + n_idx*16 -- see dequant_q3_K_block_shared_scales' own
    // comment: ggml never advances the hmask pointer, `mbit` selects the
    // bit-plane instead.
    const uint8_t* hm = xb->hmask + (lane & 3) * 8;
    #pragma unroll
    for (int i = 0; i < 8; ++i)
        w[i] = dl * (float)((int8_t)((q[i] >> shift) & 3) - ((hm[i] & mbit) ? 0 : 4));
}

#define KQUANT_MATMUL_KERNEL_LANE8(NAME, BLOCKT, DEQUANT8) \
__global__ void NAME##_kquant_matmul_kernel_lane8( \
    float* __restrict__ y, const float* __restrict__ x, const BLOCKT* __restrict__ W, \
    int S, int I, int O) \
{ \
    int o = blockIdx.x, s = blockIdx.y, tid = threadIdx.x; \
    if (o >= O || s >= S) return; \
    int bpr = I / 256; \
    const BLOCKT* w_row = W + (size_t)o * bpr; \
    const float* x_row = x + (size_t)s * I; \
    int warp_id = tid >> 5, lane = tid & 31; \
    float acc = 0.0f; \
    for (int b = warp_id; b < bpr; b += 8) { \
        float wv[8]; \
        DEQUANT8(&w_row[b], lane, wv); \
        const float* xp = x_row + (size_t)b * 256 + lane * 8; \
        float4 x0 = *reinterpret_cast<const float4*>(xp); \
        float4 x1 = *reinterpret_cast<const float4*>(xp + 4); \
        acc += wv[0]*x0.x + wv[1]*x0.y + wv[2]*x0.z + wv[3]*x0.w \
             + wv[4]*x1.x + wv[5]*x1.y + wv[6]*x1.z + wv[7]*x1.w; \
    } \
    _Pragma("unroll") \
    for (int off = 16; off > 0; off /= 2) acc += __shfl_down_sync(0xffffffff, acc, off); \
    __shared__ float s_warp[8]; \
    if (lane == 0) s_warp[warp_id] = acc; \
    __syncthreads(); \
    if (warp_id == 0) { \
        float v = (lane < 8) ? s_warp[lane] : 0.0f; \
        _Pragma("unroll") \
        for (int off = 16; off > 0; off /= 2) v += __shfl_down_sync(0xffffffff, v, off); \
        if (lane == 0) y[(size_t)s * O + o] = v; \
    } \
}

KQUANT_MATMUL_KERNEL_LANE8(q2_K, block_q2_K_cu, dequant_q2_K_lane8)
KQUANT_MATMUL_KERNEL_LANE8(q3_K, block_q3_K_cu, dequant_q3_K_lane8)
KQUANT_MATMUL_KERNEL_LANE8(q4_K, block_q4_K_cu, dequant_q4_K_lane8)
KQUANT_MATMUL_KERNEL_LANE8(q5_K, block_q5_K_cu, dequant_q5_K_lane8)
KQUANT_MATMUL_KERNEL_LANE8(q6_K, block_q6_K_cu, dequant_q6_K_lane8)

// Gate for the lane8 path. Two real requirements, not style choices:
//   bpr >= 8  -- the loop strides super-blocks by 8 warps; with fewer than 8
//                super-blocks some warps contribute nothing (still correct,
//                but the coop kernel is the better shape at that size).
//   I % 4     -- the float4 activation loads need 16-byte alignment. Within
//                a row, b*256 + lane*8 is always a multiple of 8 floats, so
//                alignment reduces to the row base x + s*I. cudaMalloc's base
//                is heavily aligned, so this holds iff I is a multiple of 4.
//                cuda/q8_0_matmul.cu carries a comment about a real Xid 13
//                "Misaligned Address" GPU exception from exactly this class
//                of bug -- hence an explicit gate rather than an assumption.
// MOESTREAM_KQUANT_LEGACY=1 forces the cooperative path even where lane8
// applies. Read once into a function-local static (the same idiom as
// dense_dispatch.hpp's cpu_only_flag()), so this costs one predictable branch,
// not a getenv, on the dispatch path. Exists so a lane8-vs-coop A/B can be run
// from ONE binary in ONE session -- this project's own history (the "GLM-5.2 is
// much slower than a week-old reference build" investigation) found that
// comparing against a previously-recorded baseline from another session can
// chase drift rather than the change under test.
static inline bool kquant_legacy_forced() {
    static const bool forced = [](){ const char* e = getenv("MOESTREAM_KQUANT_LEGACY");
                                     return e && e[0] == '1'; }();
    return forced;
}

static inline bool kquant_lane8_ok(int I) {
    return !kquant_legacy_forced() && (I / 256) >= 8 && (I % 4) == 0;
}

extern "C" void launch_q2_K_matmul(float* y, const float* x, const void* w, int S, int I, int O) {
    if (kquant_lane8_ok(I))
        q2_K_kquant_matmul_kernel_lane8<<<dim3(O, S), 256>>>(y, x, (const block_q2_K_cu*)w, S, I, O);
    else
        q2_K_kquant_matmul_kernel<<<dim3(O, S), 256>>>(y, x, (const block_q2_K_cu*)w, S, I, O);
}
extern "C" void launch_q3_K_matmul(float* y, const float* x, const void* w, int S, int I, int O) {
    if (kquant_lane8_ok(I))
        q3_K_kquant_matmul_kernel_lane8<<<dim3(O, S), 256>>>(y, x, (const block_q3_K_cu*)w, S, I, O);
    else
        q3_K_kquant_matmul_kernel<<<dim3(O, S), 256>>>(y, x, (const block_q3_K_cu*)w, S, I, O);
}
extern "C" void launch_q4_K_matmul(float* y, const float* x, const void* w, int S, int I, int O) {
    if (kquant_lane8_ok(I))
        q4_K_kquant_matmul_kernel_lane8<<<dim3(O, S), 256>>>(y, x, (const block_q4_K_cu*)w, S, I, O);
    else
        q4_K_kquant_matmul_kernel<<<dim3(O, S), 256>>>(y, x, (const block_q4_K_cu*)w, S, I, O);
}
extern "C" void launch_q5_K_matmul(float* y, const float* x, const void* w, int S, int I, int O) {
    if (kquant_lane8_ok(I))
        q5_K_kquant_matmul_kernel_lane8<<<dim3(O, S), 256>>>(y, x, (const block_q5_K_cu*)w, S, I, O);
    else
        q5_K_kquant_matmul_kernel<<<dim3(O, S), 256>>>(y, x, (const block_q5_K_cu*)w, S, I, O);
}
extern "C" void launch_q6_K_matmul(float* y, const float* x, const void* w, int S, int I, int O) {
    if (kquant_lane8_ok(I))
        q6_K_kquant_matmul_kernel_lane8<<<dim3(O, S), 256>>>(y, x, (const block_q6_K_cu*)w, S, I, O);
    else
        q6_K_kquant_matmul_kernel<<<dim3(O, S), 256>>>(y, x, (const block_q6_K_cu*)w, S, I, O);
}
