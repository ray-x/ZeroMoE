// Q2_K x Q8_K dot product: ggml's stock AVX2 kernel vs a 512-bit variant (AVX512-VNNI).
// Checks bit-exactness against the stock kernel on random blocks, then times both at 1/6/10 threads.
//   g++ -O3 -march=native -fopenmp bench/q2k_dot_bench.cpp -o /tmp/q2k_dot_bench && /tmp/q2k_dot_bench [rows]
// The weight set (rows x 16 blocks x 84 B) must exceed the caches to measure DRAM streaming: the default 1,200,000 rows is 1.6 GB.
#include <immintrin.h>
#include <omp.h>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#define QK_K 256
struct block_q2_K { uint8_t scales[16]; uint8_t qs[64]; uint16_t d, dmin; };
struct block_q8_K { float d; int8_t qs[256]; int16_t bsums[16]; };
static inline float h2f(uint16_t h) { return _cvtsh_ss(h); }
#define MM256_SET_M128I(a, b) _mm256_insertf128_si256(_mm256_castsi128_si256(b), (a), 1)
static inline float hsum_float_8(const __m256 x) {
  __m128 r = _mm256_extractf128_ps(x, 1);
  r = _mm_add_ps(r, _mm256_castps256_ps128(x));
  r = _mm_add_ps(r, _mm_movehl_ps(r, r));
  r = _mm_add_ss(r, _mm_movehdup_ps(r));
  return _mm_cvtss_f32(r);
}
static const uint8_t k_shuffle[128] = {
 0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1, 2,3,2,3,2,3,2,3,2,3,2,3,2,3,2,3, 4,5,4,5,4,5,4,5,4,5,4,5,4,5,4,5, 6,7,6,7,6,7,6,7,6,7,6,7,6,7,6,7,
 8,9,8,9,8,9,8,9,8,9,8,9,8,9,8,9, 10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11, 12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,
 14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15 };
static inline __m256i shuf(int i) { return _mm256_loadu_si256((const __m256i *)k_shuffle + i); }

// ggml/src/ggml-cpu/arch/x86/quants.c, ggml_vec_dot_q2_K_q8_K (__AVX2__ branch), unchanged.
float dot_stock(int n, const block_q2_K *x, const block_q8_K *y) {
  const int nb = n / QK_K;
  const __m256i m3 = _mm256_set1_epi8(3);
  const __m128i m4 = _mm_set1_epi8(0xF);
  __m256 acc = _mm256_setzero_ps();
  for (int i = 0; i < nb; ++i) {
    const float d = y[i].d * h2f(x[i].d), dmin = -y[i].d * h2f(x[i].dmin);
    const uint8_t *q2 = x[i].qs; const int8_t *q8 = y[i].qs;
    const __m128i ms = _mm_loadu_si128((const __m128i *)x[i].scales);
    const __m128i scales8 = _mm_and_si128(ms, m4), mins8 = _mm_and_si128(_mm_srli_epi16(ms, 4), m4);
    const __m256i prod = _mm256_madd_epi16(_mm256_cvtepi8_epi16(mins8), _mm256_loadu_si256((const __m256i *)y[i].bsums));
    acc = _mm256_fmadd_ps(_mm256_broadcast_ss(&dmin), _mm256_cvtepi32_ps(prod), acc);
    const __m256i all_scales = _mm256_cvtepi8_epi16(scales8);
    const __m128i l = _mm256_extracti128_si256(all_scales, 0), h = _mm256_extracti128_si256(all_scales, 1);
    const __m256i scales[2] = {MM256_SET_M128I(l, l), MM256_SET_M128I(h, h)};
    __m256i sumi = _mm256_setzero_si256();
    for (int j = 0; j < QK_K / 128; ++j) {
      const __m256i q2bits = _mm256_loadu_si256((const __m256i *)q2); q2 += 32;
      const __m256i a0 = _mm256_loadu_si256((const __m256i *)q8), a1 = _mm256_loadu_si256((const __m256i *)(q8 + 32)),
                    a2 = _mm256_loadu_si256((const __m256i *)(q8 + 64)), a3 = _mm256_loadu_si256((const __m256i *)(q8 + 96));
      q8 += 128;
      __m256i p0 = _mm256_maddubs_epi16(_mm256_and_si256(q2bits, m3), a0);
      __m256i p1 = _mm256_maddubs_epi16(_mm256_and_si256(_mm256_srli_epi16(q2bits, 2), m3), a1);
      __m256i p2 = _mm256_maddubs_epi16(_mm256_and_si256(_mm256_srli_epi16(q2bits, 4), m3), a2);
      __m256i p3 = _mm256_maddubs_epi16(_mm256_and_si256(_mm256_srli_epi16(q2bits, 6), m3), a3);
      p0 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j], shuf(0)), p0);
      p1 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j], shuf(1)), p1);
      p2 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j], shuf(2)), p2);
      p3 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j], shuf(3)), p3);
      sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(_mm256_add_epi32(p0, p1), _mm256_add_epi32(p2, p3)));
    }
    acc = _mm256_fmadd_ps(_mm256_broadcast_ss(&d), _mm256_cvtepi32_ps(sumi), acc);
  }
  return hsum_float_8(acc);
}

static inline __m512i dup256(__m256i v) { return _mm512_inserti64x4(_mm512_castsi256_si512(v), v, 1); }
// Both 128-value halves of a block in one 512-bit register (low 256 bits = j0, high = j1); integer sums are identical.
float dot_512(int n, const block_q2_K *x, const block_q8_K *y) {
  const int nb = n / QK_K;
  const __m512i m3 = _mm512_set1_epi8(3);
  const __m128i m4 = _mm_set1_epi8(0xF);
  const __m512i sh0 = dup256(shuf(0)), sh1 = dup256(shuf(1)), sh2 = dup256(shuf(2)), sh3 = dup256(shuf(3));
  __m256 acc = _mm256_setzero_ps();
  for (int i = 0; i < nb; ++i) {
    const float d = y[i].d * h2f(x[i].d), dmin = -y[i].d * h2f(x[i].dmin);
    const uint8_t *q2 = x[i].qs; const int8_t *q8 = y[i].qs;
    const __m128i ms = _mm_loadu_si128((const __m128i *)x[i].scales);
    const __m128i scales8 = _mm_and_si128(ms, m4), mins8 = _mm_and_si128(_mm_srli_epi16(ms, 4), m4);
    const __m256i prod = _mm256_madd_epi16(_mm256_cvtepi8_epi16(mins8), _mm256_loadu_si256((const __m256i *)y[i].bsums));
    acc = _mm256_fmadd_ps(_mm256_broadcast_ss(&dmin), _mm256_cvtepi32_ps(prod), acc);
    const __m256i all_scales = _mm256_cvtepi8_epi16(scales8);
    const __m128i l = _mm256_extracti128_si256(all_scales, 0), h = _mm256_extracti128_si256(all_scales, 1);
    const __m512i scales = _mm512_inserti64x4(_mm512_castsi256_si512(MM256_SET_M128I(l, l)), MM256_SET_M128I(h, h), 1);
    const __m512i q2bits = _mm512_loadu_si512((const void *)q2);
#define Q8(k) _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_loadu_si256((const __m256i *)(q8 + 32 * (k)))), \
                                 _mm256_loadu_si256((const __m256i *)(q8 + 128 + 32 * (k))), 1)
    __m512i s = _mm512_setzero_si512();
    s = _mm512_dpwssd_epi32(s, _mm512_shuffle_epi8(scales, sh0), _mm512_maddubs_epi16(_mm512_and_si512(q2bits, m3), Q8(0)));
    s = _mm512_dpwssd_epi32(s, _mm512_shuffle_epi8(scales, sh1), _mm512_maddubs_epi16(_mm512_and_si512(_mm512_srli_epi16(q2bits, 2), m3), Q8(1)));
    s = _mm512_dpwssd_epi32(s, _mm512_shuffle_epi8(scales, sh2), _mm512_maddubs_epi16(_mm512_and_si512(_mm512_srli_epi16(q2bits, 4), m3), Q8(2)));
    s = _mm512_dpwssd_epi32(s, _mm512_shuffle_epi8(scales, sh3), _mm512_maddubs_epi16(_mm512_and_si512(_mm512_srli_epi16(q2bits, 6), m3), Q8(3)));
#undef Q8
    const __m256i sumi = _mm256_add_epi32(_mm512_castsi512_si256(s), _mm512_extracti64x4_epi64(s, 1));
    acc = _mm256_fmadd_ps(_mm256_broadcast_ss(&d), _mm256_cvtepi32_ps(sumi), acc);
  }
  return hsum_float_8(acc);
}

int main(int argc, char **argv) {
  const int K = 4096, nbk = K / QK_K;
  const long rows = argc > 1 ? atol(argv[1]) : 1200000;
  std::vector<block_q2_K> W((size_t)rows * nbk);
  std::vector<block_q8_K> Y(nbk);
  srand(1);
  for (auto &b : W) {
    for (auto &s : b.scales) s = (uint8_t)rand();
    for (auto &q : b.qs) q = (uint8_t)rand();
    b.d = _cvtss_sh((rand() & 1023) / 8192.0f + 0.01f, 0); b.dmin = _cvtss_sh((rand() & 1023) / 8192.0f, 0);
  }
  for (auto &b : Y) {
    b.d = 0.01f + (rand() & 255) / 25600.0f;
    for (auto &q : b.qs) q = (int8_t)rand();
    for (int g = 0; g < 16; ++g) { int s = 0; for (int t = 0; t < 16; ++t) s += b.qs[g * 16 + t]; b.bsums[g] = (int16_t)s; }
  }
  long bad = 0;
  for (long r = 0; r < std::min(rows, 200000L); ++r) {
    float a = dot_stock(K, &W[(size_t)r * nbk], Y.data()), b = dot_512(K, &W[(size_t)r * nbk], Y.data());
    if (memcmp(&a, &b, 4)) ++bad;
  }
  printf("bitwise mismatches in %ld rows: %ld\n", std::min(rows, 200000L), bad);
  const double gb = (double)rows * nbk * sizeof(block_q2_K) / 1e9;
  for (int threads : {1, 6, 10}) {
    omp_set_num_threads(threads);
    for (int which = 0; which < 2; ++which) {
      double best = 1e9; volatile float sink = 0;
      for (int rep = 0; rep < 5; ++rep) {
        auto t0 = std::chrono::steady_clock::now();
#pragma omp parallel for schedule(static)
        for (long r = 0; r < rows; ++r) {
          float v = which ? dot_512(K, &W[(size_t)r * nbk], Y.data()) : dot_stock(K, &W[(size_t)r * nbk], Y.data());
          if (v == 12345.678f) sink = v;
        }
        best = std::min(best, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
      }
      printf("threads=%2d %-6s %.4f s  %.1f GB/s of weights\n", threads, which ? "avx512" : "stock", best, gb / best);
    }
  }
}
