// standalone: stock AVX2 iq3_s x q8_K dot (verbatim copy of ggml-cpu/arch/x86/quants.c) vs AVX-512BW/VL variants; checks
// bit-exactness row by row and times each. gcc -O3 -march=native -I<ggml>/src iq3s_dot_bench.c
#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#define GGML_COMMON_DECL_C
#define GGML_COMMON_IMPL_C
#include "ggml-common.h"
#define GGML_CPU_FP16_TO_FP32(x) _cvtsh_ss(x)
static inline float hsum_float_8(const __m256 x) {
    __m128 r = _mm_add_ps(_mm256_extractf128_ps(x,1), _mm256_castps256_ps128(x));
    r = _mm_add_ps(r, _mm_movehl_ps(r,r)); r = _mm_add_ss(r, _mm_movehdup_ps(r)); return _mm_cvtss_f32(r);
}
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}

// ---- reference: the stock kernel ----
static void ref(int n, float *s, const block_iq3_s *x, const block_q8_K *y) {
    static const uint8_t k_mask1[32] = {0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3};
    static const uint8_t k_mask2[32] = {1,2,4,8,16,32,64,128,1,2,4,8,16,32,64,128,1,2,4,8,16,32,64,128,1,2,4,8,16,32,64,128};
    const __m256i mask1 = _mm256_loadu_si256((const __m256i*)k_mask1), mask2 = _mm256_loadu_si256((const __m256i*)k_mask2);
    const __m256i idx_shift = _mm256_set_epi32(1,2,3,4,5,6,7,8), idx_mask = _mm256_set1_epi32(256);
    typedef union { __m256i vec[2]; uint32_t index[16]; } index_t; index_t idx;
    const int nb = n / QK_K; __m256 accumf = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
        const uint8_t *qs = x[i].qs, *qh = x[i].qh; const uint16_t *signs = (const uint16_t *)x[i].signs; const int8_t *q8 = y[i].qs;
        __m256i sumi1 = _mm256_setzero_si256(), sumi2 = _mm256_setzero_si256();
        for (int ib32 = 0; ib32 < QK_K/32; ib32 += 2) {
            const __m256i q8_1 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
            const __m256i q8_2 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
            const __m256i idx_l = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)qs)); qs += 16;
            idx.vec[0] = _mm256_set1_epi32(qh[ib32+0]); idx.vec[1] = _mm256_set1_epi32(qh[ib32+1]);
            idx.vec[0] = _mm256_and_si256(_mm256_sllv_epi32(idx.vec[0], idx_shift), idx_mask);
            idx.vec[1] = _mm256_and_si256(_mm256_sllv_epi32(idx.vec[1], idx_shift), idx_mask);
            idx.vec[0] = _mm256_or_si256(idx.vec[0], _mm256_cvtepi16_epi32(_mm256_castsi256_si128(idx_l)));
            idx.vec[1] = _mm256_or_si256(idx.vec[1], _mm256_cvtepi16_epi32(_mm256_extractf128_si256(idx_l, 1)));
            const __m256i q2_1 = _mm256_set_epi32(iq3s_grid[idx.index[7]], iq3s_grid[idx.index[6]], iq3s_grid[idx.index[5]], iq3s_grid[idx.index[4]],
                                                  iq3s_grid[idx.index[3]], iq3s_grid[idx.index[2]], iq3s_grid[idx.index[1]], iq3s_grid[idx.index[0]]);
            const __m256i q2_2 = _mm256_set_epi32(iq3s_grid[idx.index[15]], iq3s_grid[idx.index[14]], iq3s_grid[idx.index[13]], iq3s_grid[idx.index[12]],
                                                  iq3s_grid[idx.index[11]], iq3s_grid[idx.index[10]], iq3s_grid[idx.index[9]], iq3s_grid[idx.index[8]]);
            __m256i aux256 = _mm256_set1_epi32(signs[0] | (signs[1] << 16));
            aux256 = _mm256_and_si256(_mm256_shuffle_epi8(aux256, mask1), mask2);
            const __m256i s2_1 = _mm256_cmpeq_epi8(aux256, mask2);
            const __m256i q8s_1 = _mm256_sub_epi8(_mm256_xor_si256(s2_1, q8_1), s2_1);
            aux256 = _mm256_set1_epi32(signs[2] | (signs[3] << 16));
            aux256 = _mm256_and_si256(_mm256_shuffle_epi8(aux256, mask1), mask2);
            const __m256i s2_2 = _mm256_cmpeq_epi8(aux256, mask2);
            const __m256i q8s_2 = _mm256_sub_epi8(_mm256_xor_si256(s2_2, q8_2), s2_2);
            signs += 4;
            const __m256i dot1 = _mm256_maddubs_epi16(q2_1, q8s_1), dot2 = _mm256_maddubs_epi16(q2_2, q8s_2);
            const uint16_t ls1 = x[i].scales[ib32/2] & 0xf, ls2 = x[i].scales[ib32/2] >> 4;
            sumi1 = _mm256_add_epi32(sumi1, _mm256_madd_epi16(dot1, _mm256_set1_epi16(2*ls1+1)));
            sumi2 = _mm256_add_epi32(sumi2, _mm256_madd_epi16(dot2, _mm256_set1_epi16(2*ls2+1)));
        }
        accumf = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1, sumi2)), accumf);
    }
    *s = hsum_float_8(accumf);
}

// ---- V1: the 32 sign bits of a group are a byte mask: one masked subtract replaces shuffle/and/cmpeq/xor/sub ----
static void v1(int n, float *s, const block_iq3_s *x, const block_q8_K *y) {
    const __m256i idx_shift = _mm256_set_epi32(1,2,3,4,5,6,7,8), idx_mask = _mm256_set1_epi32(256), zero = _mm256_setzero_si256();
    typedef union { __m256i vec[2]; uint32_t index[16]; } index_t; index_t idx;
    const int nb = n / QK_K; __m256 accumf = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
        const uint8_t *qs = x[i].qs, *qh = x[i].qh, *signs = x[i].signs; const int8_t *q8 = y[i].qs;
        __m256i sumi1 = _mm256_setzero_si256(), sumi2 = _mm256_setzero_si256();
        for (int ib32 = 0; ib32 < QK_K/32; ib32 += 2) {
            const __m256i q8_1 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
            const __m256i q8_2 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
            const __m256i idx_l = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)qs)); qs += 16;
            idx.vec[0] = _mm256_set1_epi32(qh[ib32+0]); idx.vec[1] = _mm256_set1_epi32(qh[ib32+1]);
            idx.vec[0] = _mm256_and_si256(_mm256_sllv_epi32(idx.vec[0], idx_shift), idx_mask);
            idx.vec[1] = _mm256_and_si256(_mm256_sllv_epi32(idx.vec[1], idx_shift), idx_mask);
            idx.vec[0] = _mm256_or_si256(idx.vec[0], _mm256_cvtepi16_epi32(_mm256_castsi256_si128(idx_l)));
            idx.vec[1] = _mm256_or_si256(idx.vec[1], _mm256_cvtepi16_epi32(_mm256_extractf128_si256(idx_l, 1)));
            const __m256i q2_1 = _mm256_set_epi32(iq3s_grid[idx.index[7]], iq3s_grid[idx.index[6]], iq3s_grid[idx.index[5]], iq3s_grid[idx.index[4]],
                                                  iq3s_grid[idx.index[3]], iq3s_grid[idx.index[2]], iq3s_grid[idx.index[1]], iq3s_grid[idx.index[0]]);
            const __m256i q2_2 = _mm256_set_epi32(iq3s_grid[idx.index[15]], iq3s_grid[idx.index[14]], iq3s_grid[idx.index[13]], iq3s_grid[idx.index[12]],
                                                  iq3s_grid[idx.index[11]], iq3s_grid[idx.index[10]], iq3s_grid[idx.index[9]], iq3s_grid[idx.index[8]]);
            uint32_t sg1, sg2; memcpy(&sg1, signs, 4); memcpy(&sg2, signs + 4, 4); signs += 8;
            const __m256i q8s_1 = _mm256_mask_sub_epi8(q8_1, (__mmask32)sg1, zero, q8_1);
            const __m256i q8s_2 = _mm256_mask_sub_epi8(q8_2, (__mmask32)sg2, zero, q8_2);
            const __m256i dot1 = _mm256_maddubs_epi16(q2_1, q8s_1), dot2 = _mm256_maddubs_epi16(q2_2, q8s_2);
            const uint16_t ls1 = x[i].scales[ib32/2] & 0xf, ls2 = x[i].scales[ib32/2] >> 4;
            sumi1 = _mm256_add_epi32(sumi1, _mm256_madd_epi16(dot1, _mm256_set1_epi16(2*ls1+1)));
            sumi2 = _mm256_add_epi32(sumi2, _mm256_madd_epi16(dot2, _mm256_set1_epi16(2*ls2+1)));
        }
        accumf = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1, sumi2)), accumf);
    }
    *s = hsum_float_8(accumf);
}

// ---- V2: V1 with the grid lookup done by a hardware gather ----
static void v2(int n, float *s, const block_iq3_s *x, const block_q8_K *y) {
    const __m256i idx_shift = _mm256_set_epi32(1,2,3,4,5,6,7,8), idx_mask = _mm256_set1_epi32(256), zero = _mm256_setzero_si256();
    const int nb = n / QK_K; __m256 accumf = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
        const uint8_t *qs = x[i].qs, *qh = x[i].qh, *signs = x[i].signs; const int8_t *q8 = y[i].qs;
        __m256i sumi1 = _mm256_setzero_si256(), sumi2 = _mm256_setzero_si256();
        for (int ib32 = 0; ib32 < QK_K/32; ib32 += 2) {
            const __m256i q8_1 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
            const __m256i q8_2 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
            const __m256i idx_l = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)qs)); qs += 16;
            __m256i i0 = _mm256_and_si256(_mm256_sllv_epi32(_mm256_set1_epi32(qh[ib32+0]), idx_shift), idx_mask);
            __m256i i1 = _mm256_and_si256(_mm256_sllv_epi32(_mm256_set1_epi32(qh[ib32+1]), idx_shift), idx_mask);
            i0 = _mm256_or_si256(i0, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(idx_l)));
            i1 = _mm256_or_si256(i1, _mm256_cvtepi16_epi32(_mm256_extractf128_si256(idx_l, 1)));
            const __m256i q2_1 = _mm256_i32gather_epi32((const int *)iq3s_grid, i0, 4);
            const __m256i q2_2 = _mm256_i32gather_epi32((const int *)iq3s_grid, i1, 4);
            uint32_t sg1, sg2; memcpy(&sg1, signs, 4); memcpy(&sg2, signs + 4, 4); signs += 8;
            const __m256i q8s_1 = _mm256_mask_sub_epi8(q8_1, (__mmask32)sg1, zero, q8_1);
            const __m256i q8s_2 = _mm256_mask_sub_epi8(q8_2, (__mmask32)sg2, zero, q8_2);
            const __m256i dot1 = _mm256_maddubs_epi16(q2_1, q8s_1), dot2 = _mm256_maddubs_epi16(q2_2, q8s_2);
            const uint16_t ls1 = x[i].scales[ib32/2] & 0xf, ls2 = x[i].scales[ib32/2] >> 4;
            sumi1 = _mm256_add_epi32(sumi1, _mm256_madd_epi16(dot1, _mm256_set1_epi16(2*ls1+1)));
            sumi2 = _mm256_add_epi32(sumi2, _mm256_madd_epi16(dot2, _mm256_set1_epi16(2*ls2+1)));
        }
        accumf = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1, sumi2)), accumf);
    }
    *s = hsum_float_8(accumf);
}

// ---- V3: 512-bit, 64 weights per step; lanes 0-7 are stock sumi1, lanes 8-15 stock sumi2, so the integer sums match ----
static void v3(int n, float *s, const block_iq3_s *x, const block_q8_K *y) {
    const __m512i sh = _mm512_set_epi32(1,2,3,4,5,6,7,8, 1,2,3,4,5,6,7,8), m256 = _mm512_set1_epi32(256), zero = _mm512_setzero_si512();
    const int nb = n / QK_K; __m256 accumf = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
        const uint8_t *qs = x[i].qs, *qh = x[i].qh, *signs = x[i].signs; const int8_t *q8 = y[i].qs;
        __m512i sumi = _mm512_setzero_si512();
        for (int ib32 = 0; ib32 < QK_K/32; ib32 += 2) {
            const __m512i q8v = _mm512_loadu_si512((const void *)q8); q8 += 64;
            const __m512i idx_l = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *)qs)); qs += 16;
            const __m512i qhv = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_set1_epi32(qh[ib32+0])), _mm256_set1_epi32(qh[ib32+1]), 1);
            const __m512i idx = _mm512_or_si512(_mm512_and_si512(_mm512_sllv_epi32(qhv, sh), m256), idx_l);
            const __m512i q2 = _mm512_i32gather_epi32(idx, (const int *)iq3s_grid, 4);
            uint64_t sg; memcpy(&sg, signs, 8); signs += 8;
            const __m512i q8s = _mm512_mask_sub_epi8(q8v, (__mmask64)sg, zero, q8v);
            const __m512i dot = _mm512_maddubs_epi16(q2, q8s);
            const uint16_t ls1 = x[i].scales[ib32/2] & 0xf, ls2 = x[i].scales[ib32/2] >> 4;
            const __m512i sc = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_set1_epi16(2*ls1+1)), _mm256_set1_epi16(2*ls2+1), 1);
            sumi = _mm512_add_epi32(sumi, _mm512_madd_epi16(dot, sc));
        }
        const __m256i tot = _mm256_add_epi32(_mm512_castsi512_si256(sumi), _mm512_extracti64x4_epi64(sumi, 1));
        accumf = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(tot), accumf);
    }
    *s = hsum_float_8(accumf);
}


// ---- V4: scalar index math + grid loads stored to a buffer, one vector load back (no vector assembly from scalars) ----
static void v4(int n, float *s, const block_iq3_s *x, const block_q8_K *y) {
    const __m256i zero = _mm256_setzero_si256();
    _Alignas(64) uint32_t buf[16];
    const int nb = n / QK_K; __m256 accumf = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
        const uint8_t *qs = x[i].qs, *qh = x[i].qh, *signs = x[i].signs; const int8_t *q8 = y[i].qs;
        __m256i sumi1 = _mm256_setzero_si256(), sumi2 = _mm256_setzero_si256();
        for (int ib32 = 0; ib32 < QK_K/32; ib32 += 2) {
            const __m256i q8_1 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
            const __m256i q8_2 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
            const uint32_t h0 = qh[ib32+0], h1 = qh[ib32+1];
            for (int j = 0; j < 8; ++j) {
                buf[j]     = iq3s_grid[qs[j]     | ((h0 << (8 - j)) & 256)];
                buf[8 + j] = iq3s_grid[qs[8 + j] | ((h1 << (8 - j)) & 256)];
            }
            qs += 16;
            const __m256i q2_1 = _mm256_load_si256((const __m256i *)buf), q2_2 = _mm256_load_si256((const __m256i *)(buf + 8));
            uint32_t sg1, sg2; memcpy(&sg1, signs, 4); memcpy(&sg2, signs + 4, 4); signs += 8;
            const __m256i q8s_1 = _mm256_mask_sub_epi8(q8_1, (__mmask32)sg1, zero, q8_1);
            const __m256i q8s_2 = _mm256_mask_sub_epi8(q8_2, (__mmask32)sg2, zero, q8_2);
            const __m256i dot1 = _mm256_maddubs_epi16(q2_1, q8s_1), dot2 = _mm256_maddubs_epi16(q2_2, q8s_2);
            const uint16_t ls1 = x[i].scales[ib32/2] & 0xf, ls2 = x[i].scales[ib32/2] >> 4;
            sumi1 = _mm256_add_epi32(sumi1, _mm256_madd_epi16(dot1, _mm256_set1_epi16(2*ls1+1)));
            sumi2 = _mm256_add_epi32(sumi2, _mm256_madd_epi16(dot2, _mm256_set1_epi16(2*ls2+1)));
        }
        accumf = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1, sumi2)), accumf);
    }
    *s = hsum_float_8(accumf);
}

typedef void (*fn_t)(int, float *, const block_iq3_s *, const block_q8_K *);
int main(void) {
    const char *names[] = {"stock", "v1 masked-sub", "v2 +gather", "v3 zmm", "v4 scalar+store"}; fn_t fns[] = {ref, v1, v2, v3, v4};
    const int sizes[2] = {2048, 8192}; const size_t rows_for[2] = {4096, 16384};   // 3.6 MB (cache-warm) and 57 MB (cold)
    for (int sz = 0; sz < 2; ++sz) {
        const int K = sizes[sz], nb = K / QK_K; const size_t rows = rows_for[sz];
        block_iq3_s *x = aligned_alloc(64, ((sizeof(block_iq3_s) * nb * rows + 63) / 64) * 64); block_q8_K *y = aligned_alloc(64, sizeof(block_q8_K) * nb);
        srand(7); for (size_t i = 0; i < sizeof(block_iq3_s) * nb * rows; i++) ((uint8_t *)x)[i] = rand();
        for (size_t r = 0; r < rows * nb; r++) x[r].d = 0x3400 + (rand() & 0xff);
        for (int b = 0; b < nb; b++) { y[b].d = 0.01f; for (int j = 0; j < QK_K; j++) y[b].qs[j] = (int8_t)(rand() % 256 - 128); }  // includes -128
        float *o[5]; for (int v = 0; v < 5; ++v) o[v] = malloc(4 * rows);
        for (size_t r = 0; r < rows; r++) for (int v = 0; v < 5; ++v) fns[v](K, &o[v][r], x + r * nb, y);
        printf("K=%d rows=%zu (%.1f MB weights)\n", K, rows, sizeof(block_iq3_s) * nb * rows / 1e6);
        for (int v = 1; v < 5; ++v) { size_t bad = 0; for (size_t r = 0; r < rows; r++) if (memcmp(&o[0][r], &o[v][r], 4)) bad++; printf("  %-16s mismatching rows: %zu / %zu\n", names[v], bad, rows); }
        double best[5] = {1e9, 1e9, 1e9, 1e9, 1e9};
        for (int rep = 0; rep < 5; ++rep) for (int v = 0; v < 5; ++v) {
            double t0 = now(); for (int it = 0; it < 10; ++it) for (size_t r = 0; r < rows; r++) fns[v](K, &o[v][r], x + r * nb, y);
            double t = (now() - t0) / 10; if (t < best[v]) best[v] = t; }
        for (int v = 0; v < 5; ++v) printf("  %-16s %8.1f us/matrix  %6.2fx  %5.2f GB/s of weights  %6.0f M weights/s\n", names[v], best[v] * 1e6, best[0] / best[v],
            sizeof(block_iq3_s) * nb * rows / best[v] / 1e9, (double)K * rows / best[v] / 1e6);
    }
    return 0;
}
