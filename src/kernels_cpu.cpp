#include "kernels_cpu.hpp"
#include "logger.hpp"
#include <immintrin.h>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <algorithm>
#include <cstring>
#include <atomic>

// Thread-safe "log this exactly once" -- cpu_quant_matmul's OMP parallel
// region can call dequant_row from many threads concurrently, so a plain
// `static bool` guard races (multiple threads read `false` before any of
// them writes `true`, each prints). exchange(true) is atomic: only the one
// thread that flips it from false->true gets true back from this call.
static bool should_log_once(std::atomic<bool>& flag) {
    return !flag.exchange(true, std::memory_order_relaxed);
}

float fp16_to_fp32(uint16_t h) {
    return _cvtsh_ss(h);
} 

uint16_t fp32_to_fp16(float f) {
    __m128 v = _mm_set_ss(f);
    __m128i h = _mm_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    return _mm_extract_epi16(h, 0);
}

// ----------------------------------------------------
// Core matmul: y[S,O] = x[S,I] @ W^T, W is [O,I] row-major
// Optimized loop ordering: outer O enables write-once per output element.
// ----------------------------------------------------
void matmul(float* y, const float* x, const float* W, int S, int I, int O) {
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < O; ++o) {
        const float* w = W + (int64_t)o * I;
        for (int s = 0; s < S; ++s) {
            const float* xs = x + (int64_t)s * I;
            __m512 acc = _mm512_setzero_ps();
            int i = 0;
            for (; i + 16 <= I; i += 16) {
                acc = _mm512_fmadd_ps(_mm512_loadu_ps(xs + i), _mm512_loadu_ps(w + i), acc);
            }
            float sum = _mm512_reduce_add_ps(acc);
            for (; i < I; ++i) sum += xs[i] * w[i];
            y[(int64_t)s * O + o] = sum;
        }
    }
}

float dot_f32_avx512(const float* a, const float* b, int n) {
    __m512 acc = _mm512_setzero_ps();
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        acc = _mm512_fmadd_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i), acc);
    }
    float sum = _mm512_reduce_add_ps(acc);
    for (; i < n; ++i) sum += a[i] * b[i];
    return sum;
}

// Use double accumulation to avoid catastrophic cancellation on large D
void rmsnorm_row(float* out, const float* x, const float* w, int D, float eps) {
    double ss = 0.0;
    for (int i = 0; i < D; ++i) {
        ss += (double)x[i] * x[i];
    }
    float r = 1.0f / std::sqrt((float)(ss / D) + eps);
    int i = 0;
    __m512 v_r = _mm512_set1_ps(r);
    for (; i + 16 <= D; i += 16) {
        __m512 vx = _mm512_loadu_ps(x + i);
        __m512 vw = _mm512_loadu_ps(w + i);
        _mm512_storeu_ps(out + i, _mm512_mul_ps(_mm512_mul_ps(vx, v_r), vw));
    }
    for (; i < D; ++i) {
        out[i] = x[i] * r * w[i];
    }
}

void softmax_row(float* x, int size) {
    float max_val = x[0];
    for (int i = 1; i < size; ++i) {
        max_val = std::max(max_val, x[i]);
    }
    float sum = 0.0f;
    for (int i = 0; i < size; ++i) {
        x[i] = std::exp(x[i] - max_val);
        sum += x[i];
    }
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < size; ++i) {
        x[i] *= inv_sum;
    }
}

// SiLU: x * sigmoid(x) = x / (1 + exp(-x))
void silu_inplace(float* x, int n) {
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 v = _mm512_loadu_ps(x + i);
        __m512 neg_v = _mm512_sub_ps(_mm512_setzero_ps(), v);
        // exp(-x) approximated via _mm512_exp2a23_ps for speed, or use the stdlib path
        // Use precise path: sigmoid = 1/(1+exp(-x))
        // Store intermediate to avoid calling exp in SIMD (no native _mm512_exp_ps in all compilers)
        float tmp[16];
        _mm512_storeu_ps(tmp, neg_v);
        for (int j = 0; j < 16; ++j) tmp[j] = 1.0f / (1.0f + std::exp(tmp[j]));
        __m512 sig = _mm512_loadu_ps(tmp);
        _mm512_storeu_ps(x + i, _mm512_mul_ps(v, sig));
    }
    for (; i < n; ++i) {
        x[i] = x[i] / (1.0f + std::exp(-x[i]));
    }
}

// Element-wise multiply: out[i] = a[i] * b[i]
void elem_mul(float* out, const float* a, const float* b, int n) {
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        _mm512_storeu_ps(out + i, _mm512_mul_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i)));
    }
    for (; i < n; ++i) out[i] = a[i] * b[i];
}

// ============================================================
// AVX-512 K-quant dequant: Q2_K / Q3_K
//
// Ported from ggml's dequantize_row_q2_K / dequantize_row_q3_K
// (ggml/src/ggml-quants.c), which this project otherwise links verbatim as
// its CPU reference/fallback dequantizer (src/ggml_dequant.cpp) -- that
// reference implementation is a plain scalar loop, never vectorized
// upstream either, and was found (via the dequant/dot perf split added
// this session) to be the dominant cost inside the CPU expert-FFN fallback
// path (~40x the AVX-512 dot product it's paired with). Q2_K/Q3_K
// specifically because that's what this project's target GLM-5.2 UD
// checkpoints ship for routed experts (gate/up=Q2_K, down=Q3_K).
//
// Extracts 16 packed 2-bit codes from 16 consecutive bytes at a fixed
// shift. Widens each byte to its own 16-bit lane BEFORE shifting
// (_mm256_cvtepu8_epi16) so the shift can never bleed bits in from a
// neighboring byte -- a raw _mm256_srli_epi16 on packed bytes would corrupt
// exactly that way. Returns 16 x int32 in [0,3].
static inline __m512i unpack_2bit_x16(const uint8_t* q, int shift) {
    __m128i raw = _mm_loadu_si128((const __m128i*)q);
    __m256i wide16 = _mm256_cvtepu8_epi16(raw);
    __m256i shifted = _mm256_srli_epi16(wide16, shift);
    __m256i masked = _mm256_and_si256(shifted, _mm256_set1_epi16(3));
    return _mm512_cvtepi16_epi32(masked);
}

// One Q2_K 16-element sub-group: y[l] = dl*qval[l] - ml, qval unsigned [0,3].
// Separate mul+sub (not FMA) to bit-exactly match the scalar reference's
// two-rounding-step float arithmetic -- this is a pure per-element affine
// transform (no summation), so exact reproduction is achievable and worth
// insisting on given this project's history of subtle numeric-layout bugs.
static inline void dequant_q2k_group16(const uint8_t* q, int shift, float dl, float ml, float* out) {
    __m512i vi = unpack_2bit_x16(q, shift);
    __m512 vf = _mm512_cvtepi32_ps(vi);
    __m512 res = _mm512_sub_ps(_mm512_mul_ps(_mm512_set1_ps(dl), vf), _mm512_set1_ps(ml));
    _mm512_storeu_ps(out, res);
}

void dequantize_row_q2_K_avx512(const uint8_t* block, float* y, int64_t k) {
    // Unmistakable, unconditional (LOG_ERROR -- always visible regardless of
    // log_level) one-time marker -- added specifically because multiple
    // rebuild cycles reported unchanged dequant timing after this function
    // landed, and inferring "is this even being called" from aggregate
    // timing numbers alone kept being ambiguous (measurement overhead,
    // confounds, stale-build guesses). Goes through Logger (not a raw
    // fprintf(stderr,...)) specifically so it lands in the SAME destination
    // as every other log line the user is already capturing -- log_path, if
    // set, redirects Logger's output to a file, and a raw stderr write
    // would silently miss that file entirely, which is itself indistinguishable
    // from "this code isn't running" without checking log_path first.
    static std::atomic<bool> logged_q2k{false};
    if (should_log_once(logged_q2k)) LOG_ERROR("[AVX512-DEQUANT] dequantize_row_q2_K_avx512 is live\n");
    constexpr int QK_K = 256;
    constexpr size_t BLOCK_BYTES = 84; // scales[16] qs[64] d(2) dmin(2)
    const int64_t nb = k / QK_K;
    for (int64_t i = 0; i < nb; ++i) {
        const uint8_t* base = block + (size_t)i * BLOCK_BYTES;
        const uint8_t* scales = base;        // [16]
        const uint8_t* qs = base + 16;        // [64]
        uint16_t d_half, dmin_half;
        std::memcpy(&d_half, base + 16 + 64, 2);
        std::memcpy(&dmin_half, base + 16 + 64 + 2, 2);
        const float d = fp16_to_fp32(d_half);
        const float dmin = fp16_to_fp32(dmin_half);

        const uint8_t* q = qs;
        int is = 0;
        float* yy = y + (size_t)i * QK_K;
        for (int n = 0; n < QK_K; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                uint8_t sc0 = scales[is++];
                dequant_q2k_group16(q, shift, d * (sc0 & 0xF), dmin * (sc0 >> 4), yy);
                yy += 16;

                uint8_t sc1 = scales[is++];
                dequant_q2k_group16(q + 16, shift, d * (sc1 & 0xF), dmin * (sc1 >> 4), yy);
                yy += 16;

                shift += 2;
            }
            q += 32;
        }
    }
}

// One Q3_K 16-element sub-group: y[l] = dl * (qval[l] - (hmbit_clear[l] ? 4 : 0)),
// where qval is the unsigned 2-bit low code and hmbit_clear tests bit `m` of
// hm[l]. The (qval - 4-or-0) subtraction happens in INTEGER domain (matching
// the scalar reference's int8_t arithmetic before its single multiply by
// dl), so there's exactly one float rounding step, matching the reference.
static inline void dequant_q3k_group16(const uint8_t* q, const uint8_t* hm, uint8_t m, int shift,
                                        float dl, float* out) {
    __m512i lowbits = unpack_2bit_x16(q, shift);

    __m128i hmraw = _mm_loadu_si128((const __m128i*)hm);
    __m128i mvec = _mm_set1_epi8((char)m);
    __m128i andres = _mm_and_si128(hmraw, mvec);
    __m128i iszero = _mm_cmpeq_epi8(andres, _mm_setzero_si128());  // 0xFF (=-1) where hm bit CLEAR
    __m256i iszero16 = _mm256_cvtepi8_epi16(iszero);               // sign-extend
    __m512i iszero32 = _mm512_cvtepi16_epi32(iszero16);            // sign-extend -> -1 or 0
    __m512i sub4 = _mm512_and_si512(iszero32, _mm512_set1_epi32(4)); // 4 where clear, 0 where set

    __m512i val3 = _mm512_sub_epi32(lowbits, sub4);
    __m512 vf = _mm512_cvtepi32_ps(val3);
    __m512 res = _mm512_mul_ps(_mm512_set1_ps(dl), vf);
    _mm512_storeu_ps(out, res);
}

void dequantize_row_q3_K_avx512(const uint8_t* block, float* y, int64_t k) {
    static std::atomic<bool> logged_q3k{false};
    if (should_log_once(logged_q3k)) LOG_ERROR("[AVX512-DEQUANT] dequantize_row_q3_K_avx512 is live\n");
    constexpr int QK_K = 256;
    constexpr size_t BLOCK_BYTES = 110; // hmask[32] qs[64] scales[12] d(2)
    const int64_t nb = k / QK_K;
    const uint32_t kmask1 = 0x03030303, kmask2 = 0x0f0f0f0f;
    for (int64_t i = 0; i < nb; ++i) {
        const uint8_t* base = block + (size_t)i * BLOCK_BYTES;
        const uint8_t* hmask = base;                 // [32]
        const uint8_t* qs = base + 32;                // [64]
        const uint8_t* scale_bytes = base + 32 + 64;  // [12]
        uint16_t d_half;
        std::memcpy(&d_half, base + 32 + 64 + 12, 2);
        const float d_all = fp16_to_fp32(d_half);

        // 16 x 6-bit scales unpacked from 12 bytes into aux[4] (16 int8) --
        // verbatim scalar port of ggml's bit trick; cheap (12 input bytes)
        // relative to the 256-element unpack below, not worth vectorizing.
        uint32_t aux[4];
        std::memcpy(aux, scale_bytes, 12);
        uint32_t tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
        const int8_t* scales = (const int8_t*)aux;

        const uint8_t* q = qs;
        const uint8_t* hm = hmask;  // NOT advanced across n -- matches reference (same 32 bytes reused for all 8 bit-planes)
        uint8_t m = 1;
        int is = 0;
        float* yy = y + (size_t)i * QK_K;
        for (int n = 0; n < QK_K; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                float dl0 = d_all * (scales[is++] - 32);
                dequant_q3k_group16(q, hm, m, shift, dl0, yy);
                yy += 16;

                float dl1 = d_all * (scales[is++] - 32);
                dequant_q3k_group16(q + 16, hm + 16, m, shift, dl1, yy);
                yy += 16;

                shift += 2;
                m = (uint8_t)(m << 1);
            }
            q += 32;
        }
    }
}

// ============================================================
// AVX-512/AVX2 IQ2_XXS / IQ3_XXS dequant (codebook/grid-based)
//
// The actual routed-expert quant types for this project's target GLM-5.2
// UD-IQ2_M checkpoints (see kernels_cpu.hpp's header comment on these
// functions for the full story of why Q2_K/Q3_K above turned out to be the
// wrong target). Unlike K-quants' linear scale+shift, each 8-element output
// group here comes from a table lookup (one "grid" entry) plus a sign flip
// per element from a shared 128-entry sign table -- ported verbatim from
// this project's own already Test-5-verified CUDA kernels
// (cuda/iq2_matmul.cu's dequant_iq2_xxs_block, cuda/iq3_matmul.cu's
// dequant_iq3_xxs_block), which are themselves verbatim ports of ggml's
// dequantize_row_iq2_xxs/iq3_xxs. Table DATA (not duplicated here) comes
// from src/ggml_grids.cpp's ms_grid_* accessors -- the same host pointers
// the CUDA kernels copy into __constant__ memory.
// ============================================================
extern "C" {
    const void* ms_grid_iq2xxs();
    const void* ms_grid_iq3xxs();
    const void* ms_grid_ksigns_iq2xs();
    const void* ms_grid_kmask_iq2xs();
}

static const uint64_t* iq2xxs_grid_ptr() {
    static const uint64_t* p = (const uint64_t*)ms_grid_iq2xxs(); // [256]
    return p;
}
static const uint32_t* iq3xxs_grid_ptr() {
    static const uint32_t* p = (const uint32_t*)ms_grid_iq3xxs(); // [256]
    return p;
}
static const uint8_t* ksigns_iq2xs_ptr() {
    static const uint8_t* p = (const uint8_t*)ms_grid_ksigns_iq2xs(); // [128]
    return p;
}
// kmask_iq2xs[8] widened to 8 x int32 lanes, cached once -- this table never
// changes across calls, so building the widened vector every call would be
// pure waste.
static __m256i kmask_iq2xs_vec32() {
    static __m256i v = [] {
        const uint8_t* kmask8 = (const uint8_t*)ms_grid_kmask_iq2xs();
        int32_t tmp[8];
        for (int j = 0; j < 8; ++j) tmp[j] = (int32_t)kmask8[j];
        return _mm256_loadu_si256((const __m256i*)tmp);
    }();
    return v;
}

// Applies a per-lane sign flip + scale to one 8-byte grid entry, producing 8
// output floats: out[j] = db * grid8[j] * ((signs & kmask[j]) ? -1 : 1).
// Fully vectorized (no scalar loop over j) -- kmask_vec32 is the same 8
// lanes every call, so the AND/compare/blend below replaces what would
// otherwise be 8 scalar branches.
static inline void apply_grid_group8(const uint8_t* grid8, uint8_t signs, __m256i kmask_vec32,
                                      float db, float* out) {
    __m128i g = _mm_loadl_epi64((const __m128i*)grid8);      // low 8 bytes
    __m256i gi32 = _mm256_cvtepu8_epi32(g);                  // 8 x i32, zero-extended
    __m256 gf = _mm256_cvtepi32_ps(gi32);                    // 8 x float

    __m256i signs_bcast = _mm256_set1_epi32((int32_t)signs);
    __m256i andres = _mm256_and_si256(signs_bcast, kmask_vec32);
    __m256i bit_clear = _mm256_cmpeq_epi32(andres, _mm256_setzero_si256()); // all-1s where bit CLEAR (positive case)

    __m256 pos = _mm256_mul_ps(_mm256_set1_ps(db), gf);
    __m256 neg = _mm256_sub_ps(_mm256_setzero_ps(), pos);
    __m256 sel = _mm256_castsi256_ps(bit_clear);
    __m256 res = _mm256_blendv_ps(neg, pos, sel); // bit_clear (MSB set) -> pos, else -> neg
    _mm256_storeu_ps(out, res);
}

void dequantize_row_iq2_xxs_avx512(const uint8_t* block, float* y, int64_t k) {
    static std::atomic<bool> logged{false};
    if (should_log_once(logged)) LOG_ERROR("[AVX512-DEQUANT] dequantize_row_iq2_xxs_avx512 is live\n");

    constexpr int QK_K = 256;
    constexpr size_t BLOCK_BYTES = 66; // d(2) + qs[32] uint16 (64 bytes)
    const int64_t nb = k / QK_K;

    const uint64_t* grid_tbl = iq2xxs_grid_ptr();
    const uint8_t* ksigns = ksigns_iq2xs_ptr();
    __m256i kmask_vec = kmask_iq2xs_vec32();

    for (int64_t i = 0; i < nb; ++i) {
        const uint8_t* base = block + (size_t)i * BLOCK_BYTES;
        uint16_t d_half; std::memcpy(&d_half, base, 2);
        const float d = fp16_to_fp32(d_half);
        const uint8_t* qs_bytes = base + 2; // 32 x uint16, little-endian

        float* yy = y + (size_t)i * QK_K;
        for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
            uint16_t q0, q1, q2, q3;
            std::memcpy(&q0, qs_bytes + (size_t)(4 * ib32 + 0) * 2, 2);
            std::memcpy(&q1, qs_bytes + (size_t)(4 * ib32 + 1) * 2, 2);
            std::memcpy(&q2, qs_bytes + (size_t)(4 * ib32 + 2) * 2, 2);
            std::memcpy(&q3, qs_bytes + (size_t)(4 * ib32 + 3) * 2, 2);

            // aux32[0] = q0|q1<<16, aux32[1] = q2|q3<<16 -- matches ggml's
            // `aux32[2]` layout exactly; aux8[l] for l=0..3 (used as the
            // grid index below) are aux32[0]'s 4 individual bytes in
            // native (little-endian, x86) order, matching the reference's
            // `(const uint8_t*)aux32` reinterpretation on this platform.
            uint32_t aux32_0 = (uint32_t)q0 | ((uint32_t)q1 << 16);
            uint32_t aux32_1 = (uint32_t)q2 | ((uint32_t)q3 << 16);
            uint8_t aux8[4];
            std::memcpy(aux8, &aux32_0, 4);

            const float db = d * (0.5f + (float)(aux32_1 >> 28)) * 0.25f;

            for (int l = 0; l < 4; ++l) {
                uint64_t g = grid_tbl[aux8[l]];
                uint8_t grid_bytes[8];
                std::memcpy(grid_bytes, &g, 8);
                uint8_t signs = ksigns[(aux32_1 >> (7 * l)) & 127];
                apply_grid_group8(grid_bytes, signs, kmask_vec, db, yy + ib32 * 32 + l * 8);
            }
        }
    }
}

void dequantize_row_iq3_xxs_avx512(const uint8_t* block, float* y, int64_t k) {
    static std::atomic<bool> logged{false};
    if (should_log_once(logged)) LOG_ERROR("[AVX512-DEQUANT] dequantize_row_iq3_xxs_avx512 is live\n");

    constexpr int QK_K = 256;
    constexpr size_t BLOCK_BYTES = 98; // d(2) + qs[96]
    const int64_t nb = k / QK_K;

    const uint32_t* grid_tbl = iq3xxs_grid_ptr(); // 4-byte entries
    const uint8_t* ksigns = ksigns_iq2xs_ptr();
    __m256i kmask_vec = kmask_iq2xs_vec32();

    for (int64_t i = 0; i < nb; ++i) {
        const uint8_t* base = block + (size_t)i * BLOCK_BYTES;
        uint16_t d_half; std::memcpy(&d_half, base, 2);
        const float d = fp16_to_fp32(d_half);
        const uint8_t* qs = base + 2;                // [96]
        const uint8_t* scales_and_signs = qs + 64;    // [32] = 4 bytes x 8 ib32

        float* yy = y + (size_t)i * QK_K;
        const uint8_t* qptr = qs;
        for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
            const uint8_t* sas = scales_and_signs + 4 * ib32;
            uint32_t aux32 = (uint32_t)sas[0] | ((uint32_t)sas[1] << 8) |
                             ((uint32_t)sas[2] << 16) | ((uint32_t)sas[3] << 24);
            const float db = d * (0.5f + (float)(aux32 >> 28)) * 0.5f;

            for (int l = 0; l < 4; ++l) {
                uint8_t signs = ksigns[(aux32 >> (7 * l)) & 127];
                // grid1 supplies output[0..3], grid2 supplies output[4..7]
                // of this 8-element group -- assembled into one combined
                // 8-byte array so apply_grid_group8's single vectorized
                // call covers both, matching the reference's two four-wide
                // loops (j+0 from grid1, j+4 from grid2) as one 8-wide op.
                uint32_t g1 = grid_tbl[qptr[2 * l + 0]];
                uint32_t g2 = grid_tbl[qptr[2 * l + 1]];
                uint8_t combined[8];
                std::memcpy(combined, &g1, 4);
                std::memcpy(combined + 4, &g2, 4);
                apply_grid_group8(combined, signs, kmask_vec, db, yy + ib32 * 32 + l * 8);
            }
            qptr += 8;
        }
    }
}

// ============================================================
// IDOT (Integer Dot Product) Kernels — the biggest single win
// ============================================================
// Quantize one row of activations to int8 symmetric, return the scale.
float qrow_i8(const float* x, int8_t* q, int I) {
    // Find absmax using AVX-512
    __m512 v_amax = _mm512_setzero_ps();
    int i = 0;
    for (; i + 16 <= I; i += 16) {
        v_amax = _mm512_max_ps(v_amax, _mm512_abs_ps(_mm512_loadu_ps(x + i)));
    }
    float amax = _mm512_reduce_max_ps(v_amax);
    for (; i < I; ++i) { float a = std::abs(x[i]); if (a > amax) amax = a; }
    
    float s = amax / 127.0f;
    if (s < 1e-12f) s = 1e-12f;
    float inv = 1.0f / s;
    
    // Quantize using AVX-512 conversion
    i = 0;
    __m512 v_inv = _mm512_set1_ps(inv);
    for (; i + 16 <= I; i += 16) {
        __m512 vx = _mm512_mul_ps(_mm512_loadu_ps(x + i), v_inv);
        __m512i vi32 = _mm512_cvtps_epi32(vx);
        // Pack 32-bit -> 16-bit -> 8-bit
        __m256i vi16 = _mm512_cvtepi32_epi16(vi32);
        __m128i vi8 = _mm256_cvtepi16_epi8(vi16);
        _mm_storeu_si128((__m128i*)(q + i), vi8);
    }
    for (; i < I; ++i) {
        q[i] = (int8_t)lrintf(x[i] * inv);
    }
    return s;
}

// int8×int8 dot product using AVX-512 VNNI (vpdpbusd: unsigned×signed→int32)
// Uses the sign trick: |w| as unsigned, fold sign of w into x.
// Safe: each vpdpbusd lane accumulates 4 products ≤ 128*127 = 16256, fits s32 up to I=16384.
int32_t dot_i8i8_avx512(const int8_t* w, const int8_t* x, int I) {
    int32_t sum = 0;
    int i = 0;

#if defined(__AVX512VNNI__) && defined(__AVX512BW__)
    __m512i acc = _mm512_setzero_si512();
    for (; i + 64 <= I; i += 64) {
        __m512i wv = _mm512_loadu_si512((const void*)(w + i));
        __m512i xv = _mm512_loadu_si512((const void*)(x + i));
        // Sign trick: where w is negative, negate x; use |w| as unsigned operand
        __mmask64 neg = _mm512_movepi8_mask(wv);  // mask of negative w bytes
        __m512i xs = _mm512_mask_sub_epi8(xv, neg, _mm512_setzero_si512(), xv);
        acc = _mm512_dpbusd_epi32(acc, _mm512_abs_epi8(wv), xs);
    }
    sum = _mm512_reduce_add_epi32(acc);
#endif
    // Scalar tail
    for (; i < I; ++i) sum += (int32_t)w[i] * x[i];
    return sum;
}

// int4(packed)×int8 dot product using AVX-512 VNNI
// Unpacks 2 nibbles/byte to int8 [-8,7], then does the sign-trick VNNI dot.
int32_t dot_i4i8_avx512(const uint8_t* w4, const int8_t* x, int I) {
    int32_t sum = 0;
    int i = 0;

#if defined(__AVX512VNNI__) && defined(__AVX512BW__)
    const __m256i m4v = _mm256_set1_epi8(0x0F);
    const __m512i b8v = _mm512_set1_epi8(8);
    // unpacklo/hi produces per-128-lane ordering; permute x to match
    const __m512i xidx = _mm512_setr_epi64(0, 1, 4, 5, 2, 3, 6, 7);
    __m512i acc = _mm512_setzero_si512();
    for (; i + 64 <= I; i += 64) {
        // 32 bytes = 64 nibbles
        __m256i by = _mm256_loadu_si256((const __m256i*)(w4 + (i >> 1)));
        __m256i lo = _mm256_and_si256(by, m4v);
        __m256i hi = _mm256_and_si256(_mm256_srli_epi16(by, 4), m4v);
        __m256i z0 = _mm256_unpacklo_epi8(lo, hi);
        __m256i z1 = _mm256_unpackhi_epi8(lo, hi);
        __m512i wv = _mm512_sub_epi8(
            _mm512_inserti64x4(_mm512_castsi256_si512(z0), z1, 1), b8v);
        __m512i xv = _mm512_permutexvar_epi64(xidx, _mm512_loadu_si512((const void*)(x + i)));
        __mmask64 neg = _mm512_movepi8_mask(wv);
        __m512i xs = _mm512_mask_sub_epi8(xv, neg, _mm512_setzero_si512(), xv);
        acc = _mm512_dpbusd_epi32(acc, _mm512_abs_epi8(wv), xs);
    }
    sum = _mm512_reduce_add_epi32(acc);
#endif
    // Scalar tail
    for (; i + 1 < I; i += 2) {
        uint8_t b = w4[i >> 1];
        sum += ((int)(b & 0xF) - 8) * x[i] + ((int)(b >> 4) - 8) * x[i + 1];
    }
    if (i < I) { uint8_t b = w4[i >> 1]; sum += ((int)(b & 0xF) - 8) * x[i]; }
    return sum;
}

// IDOT matmul: quantize activations to int8, then integer dot product
// y[S,O] = x[S,I] @ W^T, W = int8 per-row quantized with scale[O]
void matmul_q8_idot(float* y, const float* x, const int8_t* q, const float* scale, int S, int I, int O) {
    // Quantize all activation rows first
    int8_t* xq = (int8_t*)malloc((size_t)S * I);
    float* sx = (float*)malloc((size_t)S * sizeof(float));
    for (int s = 0; s < S; ++s) {
        sx[s] = qrow_i8(x + (int64_t)s * I, xq + (int64_t)s * I, I);
    }
    
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < O; ++o) {
        const int8_t* w = q + (int64_t)o * I;
        float sc = scale[o];
        for (int s = 0; s < S; ++s) {
            y[(int64_t)s * O + o] = (float)dot_i8i8_avx512(w, xq + (int64_t)s * I, I) * sc * sx[s];
        }
    }
    
    free(xq);
    free(sx);
}

// IDOT matmul for int4 packed: y[S,O] = x[S,I] @ W^T, W = int4-packed with scale[O]
void matmul_i4_idot(float* y, const float* x, const uint8_t* q4, const float* scale, int S, int I, int O) {
    int rb = (I + 1) / 2;  // row bytes for int4 packed
    int8_t* xq = (int8_t*)malloc((size_t)S * I);
    float* sx = (float*)malloc((size_t)S * sizeof(float));
    for (int s = 0; s < S; ++s) {
        sx[s] = qrow_i8(x + (int64_t)s * I, xq + (int64_t)s * I, I);
    }
    
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < O; ++o) {
        const uint8_t* w = q4 + (int64_t)o * rb;
        float sc = scale[o];
        for (int s = 0; s < S; ++s) {
            y[(int64_t)s * O + o] = (float)dot_i4i8_avx512(w, xq + (int64_t)s * I, I) * sc * sx[s];
        }
    }
    
    free(xq);
    free(sx);
}

// ----------------------------------------------------
// AVX-512 Block Math Operations (Q4_0 × Q8_0)
// ----------------------------------------------------
float dot_q4_0_q8_0_avx512(const block_q4_0* w, const block_q8_0* x, int num_blocks) {
    float sum_f32 = 0.0f;
    const __m256i m4 = _mm256_set1_epi8(0x0F);
              
    for (int i = 0; i < num_blocks; i += 2) {
        if (i + 1 >= num_blocks) {
            // Handle odd last block
            float scale = fp16_to_fp32(w[i].d) * fp16_to_fp32(x[i].d);
            for (int j = 0; j < 16; ++j) {
                int8_t w0 = (int8_t)(w[i].qs[j] & 0xF) - 8;
                int8_t w1 = (int8_t)(w[i].qs[j] >> 4) - 8;
                sum_f32 += scale * ((float)w0 * x[i].qs[j*2] + (float)w1 * x[i].qs[j*2+1]);
            }
            break;
        }
        __m256i raw_w = _mm256_loadu_si256((const __m256i*)w[i].qs);
        __m256i lo_nibbles = _mm256_and_si256(raw_w, m4);
        __m256i hi_nibbles = _mm256_and_si256(_mm256_srli_epi16(raw_w, 4), m4);
                          
        __m512i v_w = _mm512_inserti64x4(_mm512_castsi256_si512(lo_nibbles), hi_nibbles, 1);
        __m512i v_x = _mm512_loadu_si512((const void*)x[i].qs);
        __m512i acc_dp = _mm512_setzero_si512();
        acc_dp = _mm512_dpbusd_epi32(acc_dp, v_w, v_x);
        __m512i ones_8  = _mm512_set1_epi8(1);
        __m512i ones_16 = _mm512_set1_epi16(1);
                          
        __m512i sum_x = _mm512_maddubs_epi16(ones_8, v_x);
        sum_x = _mm512_madd_epi16(sum_x, ones_16);
                          
        __m512i correction = _mm512_slli_epi32(sum_x, 3);
        __m512i final_int32 = _mm512_sub_epi32(acc_dp, correction);
        __m256i int_sum_0 = _mm512_castsi512_si256(final_int32);
        int_sum_0 = _mm256_add_epi32(int_sum_0, _mm256_srli_si256(int_sum_0, 8));
        int_sum_0 = _mm256_add_epi32(int_sum_0, _mm256_srli_si256(int_sum_0, 4));
        int val_0 = _mm256_extract_epi32(int_sum_0, 0) + _mm256_extract_epi32(int_sum_0, 4);
        __m256i int_sum_1 = _mm512_extracti64x4_epi64(final_int32, 1);
        int_sum_1 = _mm256_add_epi32(int_sum_1, _mm256_srli_si256(int_sum_1, 8));
        int_sum_1 = _mm256_add_epi32(int_sum_1, _mm256_srli_si256(int_sum_1, 4));
        int val_1 = _mm256_extract_epi32(int_sum_1, 0) + _mm256_extract_epi32(int_sum_1, 4);
        float s0 = fp16_to_fp32(w[i].d) * fp16_to_fp32(x[i].d);
        float s1 = fp16_to_fp32(w[i+1].d) * fp16_to_fp32(x[i+1].d);
        sum_f32 += (val_0 * s0) + (val_1 * s1);
    }
    return sum_f32;
}

void quantize_activations_q8_0(const float* __restrict__ x, block_q8_0* __restrict__ y, int num_elements) {
    int num_blocks = num_elements / 32;
    #pragma omp parallel for schedule(static)
    for (int b = 0; b < num_blocks; ++b) {
        const float* block_src = x + (b * 32);
                          
        __m512 v0 = _mm512_loadu_ps(block_src);
        __m512 v1 = _mm512_loadu_ps(block_src + 16);
        __m512 abs0 = _mm512_abs_ps(v0);
        __m512 abs1 = _mm512_abs_ps(v1);
        __m512 max_v = _mm512_max_ps(abs0, abs1);
        float amax = _mm512_reduce_max_ps(max_v);
        float scale = amax / 127.0f;
        if (scale < 1e-12f) scale = 1e-12f;
        float inv_scale = 1.0f / scale;
        y[b].d = fp32_to_fp16(scale);
        
        __m512i r0 = _mm512_cvtps_epi32(_mm512_mul_ps(v0, _mm512_set1_ps(inv_scale)));
        __m512i r1 = _mm512_cvtps_epi32(_mm512_mul_ps(v1, _mm512_set1_ps(inv_scale)));
        __m256i packed16_0 = _mm512_cvtepi32_epi16(r0);
        __m256i packed16_1 = _mm512_cvtepi32_epi16(r1);
        
        __m512i combined = _mm512_inserti64x4(_mm512_castsi256_si512(packed16_0), packed16_1, 1);
        _mm256_storeu_si256((__m256i*)(y[b].qs), _mm512_cvtepi16_epi8(combined));
    }
}