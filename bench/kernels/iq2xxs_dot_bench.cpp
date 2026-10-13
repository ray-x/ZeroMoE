// iq2xxs_dot_bench.cpp -- IQ2_XXS x Q8_K row dot: reference AVX2 vs candidate kernels.
//   g++ -O3 -march=cascadelake -std=c++20 -I<ggml>/src iq2xxs_dot_bench.cpp -o iq2xxs_dot_bench
// Checks every candidate bit-exact against the reference, then times it with the weights
// resident in L2 (compute-bound) and streamed from DRAM (as a decode expert is).
#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"
#include <immintrin.h>
#include "keven_signs_q2xs.inc"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

static inline float hsum8(__m256 x) {
  __m128 r = _mm_add_ps(_mm256_castps256_ps128(x), _mm256_extractf128_ps(x, 1));
  r = _mm_add_ps(r, _mm_movehl_ps(r, r));
  r = _mm_add_ss(r, _mm_movehdup_ps(r));
  return _mm_cvtss_f32(r);
}
static inline float fp16(ggml_half h) { return _cvtsh_ss(h); }

// ---- reference: the shipped AVX2 kernel ----------------------------------
static float dot_ref(int n, const block_iq2_xxs *x, const block_q8_K *y) {
  const uint64_t *signs64 = (const uint64_t *)keven_signs_q2xs;
  uint32_t aux32[4];
  const uint8_t *aux8 = (const uint8_t *)aux32;
  __m256 accumf = _mm256_setzero_ps();
  for (int i = 0; i < n / QK_K; ++i) {
    const float d = fp16(x[i].d) * y[i].d;
    const uint16_t *q2 = x[i].qs;
    const int8_t *q8 = y[i].qs;
    __m256i sumi1 = _mm256_setzero_si256(), sumi2 = _mm256_setzero_si256();
    for (int ib32 = 0; ib32 < QK_K / 32; ib32 += 2) {
      const __m256i q8_1 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
      const __m256i q8_2 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
      memcpy(aux32, q2, 16); q2 += 8;
      const __m256i q2_1 = _mm256_set_epi64x(iq2xxs_grid[aux8[3]], iq2xxs_grid[aux8[2]], iq2xxs_grid[aux8[1]], iq2xxs_grid[aux8[0]]);
      const __m256i q2_2 = _mm256_set_epi64x(iq2xxs_grid[aux8[11]], iq2xxs_grid[aux8[10]], iq2xxs_grid[aux8[9]], iq2xxs_grid[aux8[8]]);
      const __m256i s2_1 = _mm256_set_epi64x(signs64[(aux32[1] >> 21) & 127], signs64[(aux32[1] >> 14) & 127], signs64[(aux32[1] >> 7) & 127], signs64[aux32[1] & 127]);
      const __m256i s2_2 = _mm256_set_epi64x(signs64[(aux32[3] >> 21) & 127], signs64[(aux32[3] >> 14) & 127], signs64[(aux32[3] >> 7) & 127], signs64[aux32[3] & 127]);
      const __m256i dot1 = _mm256_maddubs_epi16(q2_1, _mm256_sign_epi8(q8_1, s2_1));
      const __m256i dot2 = _mm256_maddubs_epi16(q2_2, _mm256_sign_epi8(q8_2, s2_2));
      const uint16_t ls1 = aux32[1] >> 28, ls2 = aux32[3] >> 28;
      sumi1 = _mm256_add_epi32(sumi1, _mm256_madd_epi16(dot1, _mm256_set1_epi16(2 * ls1 + 1)));
      sumi2 = _mm256_add_epi32(sumi2, _mm256_madd_epi16(dot2, _mm256_set1_epi16(2 * ls2 + 1)));
    }
    accumf = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1, sumi2)), accumf);
  }
  return 0.125f * hsum8(accumf);
}

// ---- candidate A: AVX2 gathers + VNNI-free, 4 sub-blocks per step ------------
static float dot_gather256(int n, const block_iq2_xxs *x, const block_q8_K *y) {
  const long long *grid = (const long long *)iq2xxs_grid;
  const long long *signs = (const long long *)keven_signs_q2xs;
  const __m128i sh = _mm_setr_epi32(0, 7, 14, 21);
  const __m128i m7 = _mm_set1_epi32(127);
  __m256 accumf = _mm256_setzero_ps();
  for (int i = 0; i < n / QK_K; ++i) {
    const float d = fp16(x[i].d) * y[i].d;
    const uint32_t *q2 = (const uint32_t *)x[i].qs;
    const int8_t *q8 = y[i].qs;
    __m256i sumi1 = _mm256_setzero_si256(), sumi2 = _mm256_setzero_si256();
    for (int ib32 = 0; ib32 < QK_K / 32; ib32 += 2) {
      const __m256i q8_1 = _mm256_loadu_si256((const __m256i *)q8);
      const __m256i q8_2 = _mm256_loadu_si256((const __m256i *)(q8 + 32)); q8 += 64;
      const __m128i w = _mm_loadu_si128((const __m128i *)q2); q2 += 4;       // a0 s0 a1 s1
      const __m128i gi1 = _mm_cvtepu8_epi32(w);                                // grid idx, sub-block 0
      const __m128i gi2 = _mm_cvtepu8_epi32(_mm_srli_si128(w, 8));             // grid idx, sub-block 1
      const __m128i si1 = _mm_and_si128(_mm_srlv_epi32(_mm_shuffle_epi32(w, 0x55), sh), m7);
      const __m128i si2 = _mm_and_si128(_mm_srlv_epi32(_mm_shuffle_epi32(w, 0xFF), sh), m7);
      const __m256i q2_1 = _mm256_i32gather_epi64(grid, gi1, 8);
      const __m256i q2_2 = _mm256_i32gather_epi64(grid, gi2, 8);
      const __m256i s2_1 = _mm256_i32gather_epi64(signs, si1, 8);
      const __m256i s2_2 = _mm256_i32gather_epi64(signs, si2, 8);
      const __m256i dot1 = _mm256_maddubs_epi16(q2_1, _mm256_sign_epi8(q8_1, s2_1));
      const __m256i dot2 = _mm256_maddubs_epi16(q2_2, _mm256_sign_epi8(q8_2, s2_2));
      const uint32_t ls1 = (uint32_t)_mm_extract_epi32(w, 1) >> 28, ls2 = (uint32_t)_mm_extract_epi32(w, 3) >> 28;
      sumi1 = _mm256_add_epi32(sumi1, _mm256_madd_epi16(dot1, _mm256_set1_epi16((short)(2 * ls1 + 1))));
      sumi2 = _mm256_add_epi32(sumi2, _mm256_madd_epi16(dot2, _mm256_set1_epi16((short)(2 * ls2 + 1))));
    }
    accumf = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1, sumi2)), accumf);
  }
  return 0.125f * hsum8(accumf);
}

// ---- candidate B: AVX-512 BW+VNNI, 64 elements per step ---------------------
// zmm lanes 0-3 qwords = sub-block ib32, 4-7 = ib32+1. Sign: 0xFF byte -> negate via mask.
static float dot_zmm(int n, const block_iq2_xxs *x, const block_q8_K *y) {
  const __m128i sh = _mm_setr_epi32(0, 7, 14, 21);
  const __m256i sh2 = _mm256_setr_epi32(0, 7, 14, 21, 0, 7, 14, 21);
  const __m256i m7 = _mm256_set1_epi32(127);
  const __m512i zero = _mm512_setzero_si512();
  (void)sh;
  __m256 accumf = _mm256_setzero_ps();
  for (int i = 0; i < n / QK_K; ++i) {
    const float d = fp16(x[i].d) * y[i].d;
    const uint32_t *q2 = (const uint32_t *)x[i].qs;
    const int8_t *q8 = y[i].qs;
    __m512i acc = zero;
    for (int ib = 0; ib < QK_K / 64; ++ib) {
      const __m128i w = _mm_loadu_si128((const __m128i *)q2); q2 += 4;       // a0 s0 a1 s1
      const __m256i gi = _mm256_cvtepu8_epi32(_mm_shuffle_epi32(w, 0x08)); // 8 grid idx
      const __m256i sw = _mm256_set_m128i(_mm_shuffle_epi32(w, 0xFF), _mm_shuffle_epi32(w, 0x55));
      const __m256i si = _mm256_and_si256(_mm256_srlv_epi32(sw, sh2), m7);
      const __m512i g = _mm512_i32gather_epi64(gi, (const long long *)iq2xxs_grid, 8);
      const __m512i s = _mm512_i32gather_epi64(si, (const long long *)keven_signs_q2xs, 8);
      const __m512i a = _mm512_loadu_si512((const void *)q8); q8 += 64;
      const __mmask64 neg = _mm512_movepi8_mask(s);
      const __m512i as = _mm512_mask_sub_epi8(a, neg, zero, a);
      const __m512i dot = _mm512_dpbusd_epi32(zero, g, as);
      const uint32_t ls1 = (uint32_t)_mm_extract_epi32(w, 1) >> 28, ls2 = (uint32_t)_mm_extract_epi32(w, 3) >> 28;
      const __m512i sc = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_set1_epi32(2 * ls1 + 1)), _mm256_set1_epi32(2 * ls2 + 1), 1);
      acc = _mm512_add_epi32(acc, _mm512_mullo_epi32(dot, sc));
    }
    const __m256i s8 = _mm256_add_epi32(_mm512_castsi512_si256(acc), _mm512_extracti64x4_epi64(acc, 1));
    accumf = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(s8), accumf);
  }
  return 0.125f * hsum8(accumf);
}

// ---- candidate C: like B but signs from a 128-byte table, no sign gather ---
static uint8_t g_sbyte[128];
static float dot_zmm_tab(int n, const block_iq2_xxs *x, const block_q8_K *y) {
  const __m512i zero = _mm512_setzero_si512();
  __m256 accumf = _mm256_setzero_ps();
  for (int i = 0; i < n / QK_K; ++i) {
    const float d = fp16(x[i].d) * y[i].d;
    const uint32_t *q2 = (const uint32_t *)x[i].qs;
    const int8_t *q8 = y[i].qs;
    __m512i acc = zero;
    for (int ib = 0; ib < QK_K / 64; ++ib) {
      const __m128i w = _mm_loadu_si128((const __m128i *)q2); q2 += 4;
      const uint32_t s1 = (uint32_t)_mm_extract_epi32(w, 1), s2 = (uint32_t)_mm_extract_epi32(w, 3);
      const __m256i gi = _mm256_cvtepu8_epi32(_mm_shuffle_epi32(w, 0x08));
      const __m512i g = _mm512_i32gather_epi64(gi, (const long long *)iq2xxs_grid, 8);
      const uint64_t m = (uint64_t)g_sbyte[s1 & 127] | (uint64_t)g_sbyte[(s1 >> 7) & 127] << 8 |
                         (uint64_t)g_sbyte[(s1 >> 14) & 127] << 16 | (uint64_t)g_sbyte[(s1 >> 21) & 127] << 24 |
                         (uint64_t)g_sbyte[s2 & 127] << 32 | (uint64_t)g_sbyte[(s2 >> 7) & 127] << 40 |
                         (uint64_t)g_sbyte[(s2 >> 14) & 127] << 48 | (uint64_t)g_sbyte[(s2 >> 21) & 127] << 56;
      const __m512i a = _mm512_loadu_si512((const void *)q8); q8 += 64;
      const __m512i as = _mm512_mask_sub_epi8(a, (__mmask64)m, zero, a);
      const __m512i dot = _mm512_dpbusd_epi32(zero, g, as);
      const __m512i sc = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_set1_epi32(2 * (s1 >> 28) + 1)),
                                            _mm256_set1_epi32(2 * (s2 >> 28) + 1), 1);
      acc = _mm512_add_epi32(acc, _mm512_mullo_epi32(dot, sc));
    }
    const __m256i s8 = _mm256_add_epi32(_mm512_castsi512_si256(acc), _mm512_extracti64x4_epi64(acc, 1));
    accumf = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(s8), accumf);
  }
  return 0.125f * hsum8(accumf);
}

// ---- candidate D: ref's ymm structure, signs by mask (no sign-vector build), no gathers ----
static float dot_mask256(int n, const block_iq2_xxs *x, const block_q8_K *y) {
  const __m256i zero = _mm256_setzero_si256();
  uint32_t a[4];
  const uint8_t *a8 = (const uint8_t *)a;
  __m256 accumf = _mm256_setzero_ps();
  for (int i = 0; i < n / QK_K; ++i) {
    const float d = fp16(x[i].d) * y[i].d;
    const uint8_t *q2 = (const uint8_t *)x[i].qs;
    const int8_t *q8 = y[i].qs;
    __m256i sumi1 = zero, sumi2 = zero;
    for (int ib = 0; ib < QK_K / 64; ++ib) {
      memcpy(a, q2, 16); q2 += 16;
      const __m256i q8_1 = _mm256_loadu_si256((const __m256i *)q8);
      const __m256i q8_2 = _mm256_loadu_si256((const __m256i *)(q8 + 32)); q8 += 64;
      const __m256i g1 = _mm256_set_epi64x(iq2xxs_grid[a8[3]], iq2xxs_grid[a8[2]], iq2xxs_grid[a8[1]], iq2xxs_grid[a8[0]]);
      const __m256i g2 = _mm256_set_epi64x(iq2xxs_grid[a8[11]], iq2xxs_grid[a8[10]], iq2xxs_grid[a8[9]], iq2xxs_grid[a8[8]]);
      const uint32_t s1 = a[1], s2 = a[3];
      const uint32_t m1 = g_sbyte[s1 & 127] | (uint32_t)g_sbyte[(s1 >> 7) & 127] << 8 |
                          (uint32_t)g_sbyte[(s1 >> 14) & 127] << 16 | (uint32_t)g_sbyte[(s1 >> 21) & 127] << 24;
      const uint32_t m2 = g_sbyte[s2 & 127] | (uint32_t)g_sbyte[(s2 >> 7) & 127] << 8 |
                          (uint32_t)g_sbyte[(s2 >> 14) & 127] << 16 | (uint32_t)g_sbyte[(s2 >> 21) & 127] << 24;
      const __m256i n1 = _mm256_mask_sub_epi8(q8_1, (__mmask32)m1, zero, q8_1);
      const __m256i n2 = _mm256_mask_sub_epi8(q8_2, (__mmask32)m2, zero, q8_2);
      const __m256i dot1 = _mm256_maddubs_epi16(g1, n1);
      const __m256i dot2 = _mm256_maddubs_epi16(g2, n2);
      sumi1 = _mm256_add_epi32(sumi1, _mm256_madd_epi16(dot1, _mm256_set1_epi16((short)(2 * (s1 >> 28) + 1))));
      sumi2 = _mm256_add_epi32(sumi2, _mm256_madd_epi16(dot2, _mm256_set1_epi16((short)(2 * (s2 >> 28) + 1))));
    }
    accumf = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1, sumi2)), accumf);
  }
  return 0.125f * hsum8(accumf);
}

// ---- candidate E: per-block vectorised sign-mask + scale precompute, grid idx from memory ----
static float dot_blockpre(int n, const block_iq2_xxs *x, const block_q8_K *y) {
  const __m256i zero = _mm256_setzero_si256();
  const __m256i m7 = _mm256_set1_epi32(127), m0f = _mm256_set1_epi32(0x0F0F0F0F), m01 = _mm256_set1_epi32(0x01010101);
  const __m256i k1 = _mm256_set1_epi32(1), k10001 = _mm256_set1_epi32(0x00010001);
  alignas(32) uint32_t mk[8], sc[8];
  static const int pos[8] = {0, 1, 4, 5, 2, 3, 6, 7};  // sub-block -> slot after shuffle_ps
  __m256 accumf = _mm256_setzero_ps();
  for (int i = 0; i < n / QK_K; ++i) {
    const float d = fp16(x[i].d) * y[i].d;
    const uint8_t *q2 = (const uint8_t *)x[i].qs;
    const int8_t *q8 = y[i].qs;
    // 8 scale/sign words (odd dwords of the block)
    const __m256i Q0 = _mm256_loadu_si256((const __m256i *)q2);
    const __m256i Q1 = _mm256_loadu_si256((const __m256i *)(q2 + 32));
    const __m256i S = _mm256_castps_si256(_mm256_shuffle_ps(_mm256_castsi256_ps(Q0), _mm256_castsi256_ps(Q1), 0xDD));
    // scale (2*ls+1) duplicated into both 16-bit halves so a dword broadcast feeds madd_epi16
    const __m256i ls = _mm256_srli_epi32(S, 28);
    const __m256i sc32 = _mm256_add_epi32(_mm256_slli_epi32(ls, 1), k1);
    _mm256_store_si256((__m256i *)sc, _mm256_mullo_epi32(sc32, k10001));
    // 4 x 7-bit sign indices per word -> 4 bytes, then parity bit into bit 7 of each byte
    __m256i v = _mm256_and_si256(S, m7);
    v = _mm256_or_si256(v, _mm256_slli_epi32(_mm256_and_si256(_mm256_srli_epi32(S, 7), m7), 8));
    v = _mm256_or_si256(v, _mm256_slli_epi32(_mm256_and_si256(_mm256_srli_epi32(S, 14), m7), 16));
    v = _mm256_or_si256(v, _mm256_slli_epi32(_mm256_and_si256(_mm256_srli_epi32(S, 21), m7), 24));
    __m256i t = _mm256_xor_si256(v, _mm256_and_si256(_mm256_srli_epi32(v, 4), m0f));
    t = _mm256_xor_si256(t, _mm256_srli_epi32(t, 2));
    t = _mm256_xor_si256(t, _mm256_srli_epi32(t, 1));
    v = _mm256_or_si256(v, _mm256_slli_epi32(_mm256_and_si256(t, m01), 7));
    _mm256_store_si256((__m256i *)mk, v);
    __m256i sumi1 = zero, sumi2 = zero;
    for (int ib = 0; ib < 4; ++ib) {
      const uint8_t *a = q2 + 16 * ib;
      const int p1 = pos[2 * ib], p2 = pos[2 * ib + 1];
      const __m256i q8_1 = _mm256_loadu_si256((const __m256i *)q8);
      const __m256i q8_2 = _mm256_loadu_si256((const __m256i *)(q8 + 32)); q8 += 64;
      const __m256i g1 = _mm256_set_epi64x(iq2xxs_grid[a[3]], iq2xxs_grid[a[2]], iq2xxs_grid[a[1]], iq2xxs_grid[a[0]]);
      const __m256i g2 = _mm256_set_epi64x(iq2xxs_grid[a[11]], iq2xxs_grid[a[10]], iq2xxs_grid[a[9]], iq2xxs_grid[a[8]]);
      const __m256i n1 = _mm256_mask_sub_epi8(q8_1, (__mmask32)mk[p1], zero, q8_1);
      const __m256i n2 = _mm256_mask_sub_epi8(q8_2, (__mmask32)mk[p2], zero, q8_2);
      sumi1 = _mm256_add_epi32(sumi1, _mm256_madd_epi16(_mm256_maddubs_epi16(g1, n1), _mm256_set1_epi32((int)sc[p1])));
      sumi2 = _mm256_add_epi32(sumi2, _mm256_madd_epi16(_mm256_maddubs_epi16(g2, n2), _mm256_set1_epi32((int)sc[p2])));
    }
    accumf = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1, sumi2)), accumf);
  }
  return 0.125f * hsum8(accumf);
}


// ---- candidates F/G/H: restructurings of "block-precompute" ------------------------------------------------------------------
static inline __m256i grid4(const uint8_t *a) {  // 4 grid rows without vpinsrq: movq loads + punpcklqdq + one vinserti128
  const __m128i lo = _mm_unpacklo_epi64(_mm_loadl_epi64((const __m128i *)&iq2xxs_grid[a[0]]), _mm_loadl_epi64((const __m128i *)&iq2xxs_grid[a[1]]));
  const __m128i hi = _mm_unpacklo_epi64(_mm_loadl_epi64((const __m128i *)&iq2xxs_grid[a[2]]), _mm_loadl_epi64((const __m128i *)&iq2xxs_grid[a[3]]));
  return _mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1);
}
struct BlockPre { alignas(32) uint32_t mk[8], sc[8]; };
static inline void precompute(const block_iq2_xxs &x, BlockPre &b) {
  const __m256i m7 = _mm256_set1_epi32(127), m0f = _mm256_set1_epi32(0x0F0F0F0F), m01 = _mm256_set1_epi32(0x01010101);
  const __m256i k1 = _mm256_set1_epi32(1), k10001 = _mm256_set1_epi32(0x00010001);
  const uint8_t *q2 = (const uint8_t *)x.qs;
  const __m256i Q0 = _mm256_loadu_si256((const __m256i *)q2), Q1 = _mm256_loadu_si256((const __m256i *)(q2 + 32));
  const __m256i S = _mm256_castps_si256(_mm256_shuffle_ps(_mm256_castsi256_ps(Q0), _mm256_castsi256_ps(Q1), 0xDD));
  const __m256i ls = _mm256_srli_epi32(S, 28);
  _mm256_store_si256((__m256i *)b.sc, _mm256_mullo_epi32(_mm256_add_epi32(_mm256_slli_epi32(ls, 1), k1), k10001));
  __m256i v = _mm256_and_si256(S, m7);
  v = _mm256_or_si256(v, _mm256_slli_epi32(_mm256_and_si256(_mm256_srli_epi32(S, 7), m7), 8));
  v = _mm256_or_si256(v, _mm256_slli_epi32(_mm256_and_si256(_mm256_srli_epi32(S, 14), m7), 16));
  v = _mm256_or_si256(v, _mm256_slli_epi32(_mm256_and_si256(_mm256_srli_epi32(S, 21), m7), 24));
  __m256i t = _mm256_xor_si256(v, _mm256_and_si256(_mm256_srli_epi32(v, 4), m0f));
  t = _mm256_xor_si256(t, _mm256_srli_epi32(t, 2));
  t = _mm256_xor_si256(t, _mm256_srli_epi32(t, 1));
  v = _mm256_or_si256(v, _mm256_slli_epi32(_mm256_and_si256(t, m01), 7));
  _mm256_store_si256((__m256i *)b.mk, v);
}
static const int kpos[8] = {0, 1, 4, 5, 2, 3, 6, 7};
// F: grid4 (no vpinsrq)
static float dot_F(int n, const block_iq2_xxs *x, const block_q8_K *y) {
  const __m256i zero = _mm256_setzero_si256();
  __m256 accumf = _mm256_setzero_ps();
  BlockPre b;
  for (int i = 0; i < n / QK_K; ++i) {
    precompute(x[i], b);
    const uint8_t *q2 = (const uint8_t *)x[i].qs; const int8_t *q8 = y[i].qs;
    __m256i s1 = zero, s2 = zero;
    for (int ib = 0; ib < 4; ++ib) {
      const uint8_t *a = q2 + 16 * ib; const int p1 = kpos[2 * ib], p2 = kpos[2 * ib + 1];
      const __m256i q8a = _mm256_loadu_si256((const __m256i *)q8), q8b = _mm256_loadu_si256((const __m256i *)(q8 + 32)); q8 += 64;
      const __m256i n1 = _mm256_mask_sub_epi8(q8a, (__mmask32)b.mk[p1], zero, q8a), n2 = _mm256_mask_sub_epi8(q8b, (__mmask32)b.mk[p2], zero, q8b);
      s1 = _mm256_add_epi32(s1, _mm256_madd_epi16(_mm256_maddubs_epi16(grid4(a), n1), _mm256_set1_epi32((int)b.sc[p1])));
      s2 = _mm256_add_epi32(s2, _mm256_madd_epi16(_mm256_maddubs_epi16(grid4(a + 8), n2), _mm256_set1_epi32((int)b.sc[p2])));
    }
    accumf = _mm256_fmadd_ps(_mm256_set1_ps(y[i].d * fp16(x[i].d)), _mm256_cvtepi32_ps(_mm256_add_epi32(s1, s2)), accumf);
  }
  return 0.125f * hsum8(accumf);
}
// G: F + vpdpwssd (VNNI) fuses the scale multiply-accumulate
static float dot_G(int n, const block_iq2_xxs *x, const block_q8_K *y) {
  const __m256i zero = _mm256_setzero_si256();
  __m256 accumf = _mm256_setzero_ps();
  BlockPre b;
  for (int i = 0; i < n / QK_K; ++i) {
    precompute(x[i], b);
    const uint8_t *q2 = (const uint8_t *)x[i].qs; const int8_t *q8 = y[i].qs;
    __m256i s1 = zero, s2 = zero;
    for (int ib = 0; ib < 4; ++ib) {
      const uint8_t *a = q2 + 16 * ib; const int p1 = kpos[2 * ib], p2 = kpos[2 * ib + 1];
      const __m256i q8a = _mm256_loadu_si256((const __m256i *)q8), q8b = _mm256_loadu_si256((const __m256i *)(q8 + 32)); q8 += 64;
      const __m256i n1 = _mm256_mask_sub_epi8(q8a, (__mmask32)b.mk[p1], zero, q8a), n2 = _mm256_mask_sub_epi8(q8b, (__mmask32)b.mk[p2], zero, q8b);
      s1 = _mm256_dpwssd_epi32(s1, _mm256_maddubs_epi16(grid4(a), n1), _mm256_set1_epi32((int)b.sc[p1]));
      s2 = _mm256_dpwssd_epi32(s2, _mm256_maddubs_epi16(grid4(a + 8), n2), _mm256_set1_epi32((int)b.sc[p2]));
    }
    accumf = _mm256_fmadd_ps(_mm256_set1_ps(y[i].d * fp16(x[i].d)), _mm256_cvtepi32_ps(_mm256_add_epi32(s1, s2)), accumf);
  }
  return 0.125f * hsum8(accumf);
}
// H: G with two blocks per iteration (independent chains, precompute of block i+1 overlaps block i)
static float dot_H(int n, const block_iq2_xxs *x, const block_q8_K *y) {
  const __m256i zero = _mm256_setzero_si256();
  __m256 accumf = _mm256_setzero_ps();
  BlockPre b0, b1;
  const int nb = n / QK_K;
  int i = 0;
  for (; i + 1 < nb; i += 2) {
    precompute(x[i], b0); precompute(x[i + 1], b1);
    const uint8_t *qa = (const uint8_t *)x[i].qs, *qb = (const uint8_t *)x[i + 1].qs;
    const int8_t *ya = y[i].qs, *yb = y[i + 1].qs;
    __m256i a1 = zero, a2 = zero, c1 = zero, c2 = zero;
    for (int ib = 0; ib < 4; ++ib) {
      const int p1 = kpos[2 * ib], p2 = kpos[2 * ib + 1];
      const __m256i xa = _mm256_loadu_si256((const __m256i *)ya), xb = _mm256_loadu_si256((const __m256i *)(ya + 32)); ya += 64;
      const __m256i za = _mm256_loadu_si256((const __m256i *)yb), zb = _mm256_loadu_si256((const __m256i *)(yb + 32)); yb += 64;
      const uint8_t *a = qa + 16 * ib, *c = qb + 16 * ib;
      a1 = _mm256_dpwssd_epi32(a1, _mm256_maddubs_epi16(grid4(a), _mm256_mask_sub_epi8(xa, (__mmask32)b0.mk[p1], zero, xa)), _mm256_set1_epi32((int)b0.sc[p1]));
      c1 = _mm256_dpwssd_epi32(c1, _mm256_maddubs_epi16(grid4(c), _mm256_mask_sub_epi8(za, (__mmask32)b1.mk[p1], zero, za)), _mm256_set1_epi32((int)b1.sc[p1]));
      a2 = _mm256_dpwssd_epi32(a2, _mm256_maddubs_epi16(grid4(a + 8), _mm256_mask_sub_epi8(xb, (__mmask32)b0.mk[p2], zero, xb)), _mm256_set1_epi32((int)b0.sc[p2]));
      c2 = _mm256_dpwssd_epi32(c2, _mm256_maddubs_epi16(grid4(c + 8), _mm256_mask_sub_epi8(zb, (__mmask32)b1.mk[p2], zero, zb)), _mm256_set1_epi32((int)b1.sc[p2]));
    }
    accumf = _mm256_fmadd_ps(_mm256_set1_ps(y[i].d * fp16(x[i].d)), _mm256_cvtepi32_ps(_mm256_add_epi32(a1, a2)), accumf);
    accumf = _mm256_fmadd_ps(_mm256_set1_ps(y[i + 1].d * fp16(x[i + 1].d)), _mm256_cvtepi32_ps(_mm256_add_epi32(c1, c2)), accumf);
  }
  for (; i < nb; ++i) accumf = _mm256_add_ps(accumf, _mm256_setzero_ps());  // n is a multiple of 512 here; tail handled by F below
  if (i < nb) return dot_F(n, x, y);
  return 0.125f * hsum8(accumf);
}

using Fn = float (*)(int, const block_iq2_xxs *, const block_q8_K *);
static double bench(Fn f, int n, const block_iq2_xxs *w, size_t rows, const block_q8_K *y, int reps, float *sink) {
  const int nb = n / QK_K;
  const auto t0 = std::chrono::steady_clock::now();
  for (int r = 0; r < reps; ++r)
    for (size_t row = 0; row < rows; ++row) *sink += f(n, w + row * nb, y);
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

int main() {
  // 7-bit index -> sign byte with the parity bit in bit 7 (what keven_signs encodes as 0x01/0xFF per bit)
  for (int i = 0; i < 128; ++i) {
    uint8_t b = (uint8_t)i;
    if (__builtin_popcount(i) & 1) b |= 0x80;
    g_sbyte[i] = b;
  }
  const int n = 4096;  // GLM-5.3-Flash expert gate/up K (16 blocks)
  const int nb = n / QK_K;
  std::mt19937_64 rng(7);
  auto make = [&](size_t rows) {
    std::vector<block_iq2_xxs> w(rows * nb);
    for (auto &b : w) {
      b.d = _cvtss_sh(0.01f + (rng() % 100) * 1e-4f, 0);
      for (auto &q : b.qs) q = (uint16_t)rng();
    }
    return w;
  };
  std::vector<block_q8_K> y(nb);
  for (auto &b : y) {
    b.d = 0.02f;
    for (auto &q : b.qs) q = (int8_t)(rng() % 255 - 127);
  }
  struct { const char *name; Fn f; } fns[] = {
      {"ref avx2", dot_ref}, {"gather avx2", dot_gather256}, {"zmm vnni gather-signs", dot_zmm},
      {"zmm vnni table-signs", dot_zmm_tab}, {"mask256 no gather", dot_mask256}, {"block-precompute", dot_blockpre},
      {"F grid4 no pinsrq", dot_F}, {"G F+vpdpwssd", dot_G}, {"H G x2 blocks", dot_H}};
  { // exactness
    auto w = make(512);
    for (auto &fn : fns) {
      size_t bad = 0;
      for (size_t r = 0; r < 512; ++r) {
        float a = dot_ref(n, &w[r * nb], y.data()), b = fn.f(n, &w[r * nb], y.data());
        if (memcmp(&a, &b, 4)) ++bad;
      }
      printf("%-24s exact vs ref: %s (%zu/512 differ)\n", fn.name, bad ? "NO" : "yes", bad);
    }
  }
  float sink = 0;
  struct Case { const char *name; size_t rows; int reps; } cases[] = {{"L2-resident (512 rows)", 512, 4000},
                                                                      {"DRAM stream (160k rows)", 160000, 6}};
  for (auto &c : cases) {
    auto w = make(c.rows);
    const double bytes = (double)c.rows * nb * sizeof(block_iq2_xxs) * c.reps;
    printf("\n%s\n", c.name);
    double base = 0;
    for (auto &fn : fns) {
      bench(fn.f, n, w.data(), c.rows, y.data(), 1, &sink);  // warm
      const double t = bench(fn.f, n, w.data(), c.rows, y.data(), c.reps, &sink);
      if (!base) base = t;
      printf("  %-24s %7.2f GB/s  %5.2fx\n", fn.name, bytes / t / 1e9, base / t);
    }
  }
  return sink == 12345.f;
}
