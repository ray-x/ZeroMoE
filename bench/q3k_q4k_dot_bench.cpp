// Q3_K / Q4_K x Q8_K CPU dot: ggml's stock kernel (AVX2 in the pinned ggml)
// vs a 512-bit AVX-512/VNNI Q4_K variant. Built to answer whether a VNNI
// kernel can help the early GLM layers, whose experts are Q4_K/Q3_K.
//
//   g++ -O3 -march=native -fopenmp -std=c++20 bench/q3k_q4k_dot_bench.cpp \
//       -Lbuild/ggml-build/src -lggml-cpu -lggml-base -lggml \
//       -Wl,-rpath,$PWD/build/ggml-build/src -o /tmp/q3k_q4k_dot_bench
//   /tmp/q3k_q4k_dot_bench [rows_q4k] [rows_q3k]
//
// The stock functions are the exported ggml_vec_dot_{q3,q4}_K_q8_K symbols.
// The Q4_K 512 variant uses _mm512_maddubs_epi16 + _mm512_dpwssd_epi32
// (AVX512-VNNI) with the same integer grouping and float per-block order.
#include <immintrin.h>
#include <omp.h>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define QK_K 256
#define K_SCALE_SIZE 12

struct block_q3_K { uint8_t hmask[32]; uint8_t qs[64]; uint8_t scales[12]; uint16_t d; };
struct block_q4_K { uint16_t d, dmin; uint8_t scales[12]; uint8_t qs[128]; };
struct block_q8_K { float d; int8_t qs[256]; int16_t bsums[16]; };

extern "C" void ggml_vec_dot_q3_K_q8_K(int n, float *s, size_t bs, const void *vx,
                                       size_t bx, const void *vy, size_t by, int nrc);
extern "C" void ggml_vec_dot_q4_K_q8_K(int n, float *s, size_t bs, const void *vx,
                                       size_t bx, const void *vy, size_t by, int nrc);

static inline float h2f(uint16_t h) { return _cvtsh_ss(h); }
static inline float hsum_float_8(const __m256 x) {
  __m128 r = _mm_add_ps(_mm256_extractf128_ps(x, 1), _mm256_castps256_ps128(x));
  r = _mm_add_ps(r, _mm_movehl_ps(r, r));
  r = _mm_add_ss(r, _mm_movehdup_ps(r));
  return _mm_cvtss_f32(r);
}
#define MM256_SET_M128I(a, b) _mm256_insertf128_si256(_mm256_castsi128_si256(b), (a), 1)

// ggml/src/ggml-cpu/arch/x86/quants.c, get_scale_shuffle_k4.
static inline __m256i get_scale_shuffle_k4(int i) {
  static const uint8_t k_shuffle[256] = {
     0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,
     2,3,2,3,2,3,2,3,2,3,2,3,2,3,2,3,2,3,2,3,2,3,2,3,2,3,2,3,2,3,2,3,
     4,5,4,5,4,5,4,5,4,5,4,5,4,5,4,5,4,5,4,5,4,5,4,5,4,5,4,5,4,5,4,5,
     6,7,6,7,6,7,6,7,6,7,6,7,6,7,6,7,6,7,6,7,6,7,6,7,6,7,6,7,6,7,6,7,
     8,9,8,9,8,9,8,9,8,9,8,9,8,9,8,9,8,9,8,9,8,9,8,9,8,9,8,9,8,9,8,9,
    10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,
    12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,
    14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15};
  return _mm256_loadu_si256((const __m256i *)k_shuffle + i);
}

static float dot_q3k_stock(int n, const block_q3_K *x, const block_q8_K *y) {
  float s = 0;
  ggml_vec_dot_q3_K_q8_K(n, &s, 0, x, 0, y, 0, 1);
  return s;
}
static float dot_q4k_stock(int n, const block_q4_K *x, const block_q8_K *y) {
  float s = 0;
  ggml_vec_dot_q4_K_q8_K(n, &s, 0, x, 0, y, 0, 1);
  return s;
}

static inline __m512i cat256(__m256i a, __m256i b) {
  return _mm512_inserti64x4(_mm512_castsi256_si512(a), b, 1);
}

// 512-bit AVX512-VNNI Q4_K: two 64-weight j iterations at a time.
static float dot_q4k_512(int n, const block_q4_K *x, const block_q8_K *y) {
  const int nb = n / QK_K;
  const uint32_t kmask1 = 0x3f3f3f3f, kmask2 = 0x0f0f0f0f, kmask3 = 0x03030303;
  const __m256i m4 = _mm256_set1_epi8(0xF);
  __m256 acc = _mm256_setzero_ps();
  __m128 acc_m = _mm_setzero_ps();
  for (int i = 0; i < nb; ++i) {
    const float d = y[i].d * h2f(x[i].d);
    const float dmin = -y[i].d * h2f(x[i].dmin);
    uint32_t utmp[4];
    std::memcpy(utmp, x[i].scales, 12);
    utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
    const uint32_t uaux = utmp[1] & kmask1;
    utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
    utmp[2] = uaux;
    utmp[0] &= kmask1;
    const __m256i mins_and_scales =
        _mm256_cvtepu8_epi16(_mm_set_epi32(utmp[3], utmp[2], utmp[1], utmp[0]));
    const __m256i q8sums = _mm256_loadu_si256((const __m256i *)y[i].bsums);
    const __m128i q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0),
                                       _mm256_extracti128_si256(q8sums, 1));
    const __m128i prod = _mm_madd_epi16(_mm256_extracti128_si256(mins_and_scales, 1), q8s);
    acc_m = _mm_fmadd_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(prod), acc_m);
    const __m128i sc128 = _mm256_extracti128_si256(mins_and_scales, 0);
    const __m256i scales = MM256_SET_M128I(sc128, sc128);
    const uint8_t *q4 = x[i].qs;
    const int8_t *q8 = y[i].qs;
    __m512i sumi = _mm512_setzero_si512();
    for (int j = 0; j < 4; j += 2) {
      const __m256i qb0 = _mm256_loadu_si256((const __m256i *)(q4 + 32 * j));
      const __m256i qb1 = _mm256_loadu_si256((const __m256i *)(q4 + 32 * (j + 1)));
      const __m512i q4l = cat256(_mm256_and_si256(qb0, m4), _mm256_and_si256(qb1, m4));
      const __m512i q4h = cat256(_mm256_and_si256(_mm256_srli_epi16(qb0, 4), m4),
                                 _mm256_and_si256(_mm256_srli_epi16(qb1, 4), m4));
      const __m512i q8l = cat256(_mm256_loadu_si256((const __m256i *)(q8 + 64 * j)),
                                 _mm256_loadu_si256((const __m256i *)(q8 + 64 * (j + 1))));
      const __m512i q8h = cat256(_mm256_loadu_si256((const __m256i *)(q8 + 64 * j + 32)),
                                 _mm256_loadu_si256((const __m256i *)(q8 + 64 * (j + 1) + 32)));
      const __m512i sl = cat256(_mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * j + 0)),
                                _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * (j + 1) + 0)));
      const __m512i sh = cat256(_mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * j + 1)),
                                _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * (j + 1) + 1)));
      sumi = _mm512_dpwssd_epi32(sumi, sl, _mm512_maddubs_epi16(q4l, q8l));
      sumi = _mm512_dpwssd_epi32(sumi, sh, _mm512_maddubs_epi16(q4h, q8h));
    }
    const __m256i s256 = _mm256_add_epi32(_mm512_castsi512_si256(sumi),
                                          _mm512_extracti64x4_epi64(sumi, 1));
    acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(s256), acc);
  }
  acc_m = _mm_add_ps(acc_m, _mm_movehl_ps(acc_m, acc_m));
  acc_m = _mm_add_ss(acc_m, _mm_movehdup_ps(acc_m));
  return hsum_float_8(acc) + _mm_cvtss_f32(acc_m);
}

static void fill_q8(std::vector<block_q8_K> &Y) {
  for (auto &b : Y) {
    b.d = 0.01f + (rand() & 255) / 25600.0f;
    for (auto &q : b.qs) q = (int8_t)rand();
    for (int g = 0; g < 16; ++g) {
      int s = 0;
      for (int t = 0; t < 16; ++t) s += b.qs[g * 16 + t];
      b.bsums[g] = (int16_t)s;
    }
  }
}
static void fill_q4(std::vector<block_q4_K> &W) {
  for (auto &b : W) {
    b.d = _cvtss_sh((rand() & 1023) / 8192.0f + 0.01f, 0);
    b.dmin = _cvtss_sh((rand() & 1023) / 8192.0f, 0);
    for (auto &s : b.scales) s = (uint8_t)rand();
    for (auto &q : b.qs) q = (uint8_t)rand();
  }
}
static void fill_q3(std::vector<block_q3_K> &W) {
  for (auto &b : W) {
    b.d = _cvtss_sh((rand() & 1023) / 8192.0f + 0.01f, 0);
    for (auto &s : b.scales) s = (uint8_t)rand();
    for (auto &q : b.qs) q = (uint8_t)rand();
    for (auto &h : b.hmask) h = (uint8_t)rand();
  }
}

template <typename B, typename F>
static double time_rows(int threads, const std::vector<B> &W, const std::vector<block_q8_K> &Y,
                        int nbk, long rows, F fn) {
  omp_set_num_threads(threads);
  double best = 1e9;
  volatile float sink = 0;
  for (int rep = 0; rep < 5; ++rep) {
    auto t0 = std::chrono::steady_clock::now();
#pragma omp parallel for schedule(static)
    for (long r = 0; r < rows; ++r) sink = sink + fn(QK_K * nbk, &W[(size_t)r * nbk], Y.data());
    auto t1 = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    if (ms < best) best = ms;
  }
  (void)sink;
  return best;
}

int main(int argc, char **argv) {
  const int K = 4096, nbk = K / QK_K;
  const long rows4 = argc > 1 ? atol(argv[1]) : 700000;  // ~1.6 GB Q4_K
  const long rows3 = argc > 2 ? atol(argv[2]) : 900000;  // ~1.7 GB Q3_K
  std::vector<block_q4_K> W4((size_t)rows4 * nbk);
  std::vector<block_q3_K> W3((size_t)rows3 * nbk);
  std::vector<block_q8_K> Y(nbk);
  srand(1);
  fill_q4(W4);
  fill_q3(W3);
  fill_q8(Y);

  printf("Q4_K: %ld rows, %.2f GB; Q3_K: %ld rows, %.2f GB; K=%d\n", rows4,
         (double)rows4 * nbk * sizeof(block_q4_K) / 1e9, rows3,
         (double)rows3 * nbk * sizeof(block_q3_K) / 1e9, K);

  {
    long bad = 0;
    double maxdiff = 0;
    const long ncheck = std::min(rows4, 100000L);
    for (long r = 0; r < ncheck; ++r) {
      float a = dot_q4k_stock(K, &W4[(size_t)r * nbk], Y.data());
      float b = dot_q4k_512(K, &W4[(size_t)r * nbk], Y.data());
      if (memcmp(&a, &b, 4)) ++bad;
      const double da = (double)a, db = (double)b;
      const double diff = da > db ? da - db : db - da;
      if (diff > maxdiff) maxdiff = diff;
    }
    printf("Q4_K 512-vnni vs stock: %ld/%ld bitwise mismatches, max abs diff %.6g\n",
           bad, ncheck, maxdiff);
  }

  for (int threads : {1, 8, 10}) {
    const double t4s = time_rows(threads, W4, Y, nbk, rows4, dot_q4k_stock);
    const double t4v = time_rows(threads, W4, Y, nbk, rows4, dot_q4k_512);
    const double t3s = time_rows(threads, W3, Y, nbk, rows3, dot_q3k_stock);
    const double gb4 = (double)rows4 * nbk * sizeof(block_q4_K) / 1e9;
    const double gb3 = (double)rows3 * nbk * sizeof(block_q3_K) / 1e9;
    printf("threads=%2d  Q4_K stock %7.1f ms (%5.1f GB/s)  Q4_K 512 %7.1f ms (%5.1f GB/s, x%.3f)  Q3_K stock %7.1f ms (%5.1f GB/s)\n",
           threads, t4s, gb4 / (t4s / 1e3), t4v, gb4 / (t4v / 1e3), t4s / t4v, t3s, gb3 / (t3s / 1e3));
  }
  return 0;
}
