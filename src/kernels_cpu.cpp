#include "kernels_cpu.hpp"
#include "logger.hpp"
#include "perf_timer.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <immintrin.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <cstdint>
#include <mutex>
#include <vector>
#if defined(MOESTREAM_USE_GGML)
// Keep these two ggml-internal declarations local instead of including
// ggml-common.h: that header defines QK_K/QK8_0 macros and block_q*_0 types
// which intentionally have different compatibility structs in kernels_cpu.hpp.
struct moestream_ggml_block_q8_K {
  float d;
  int8_t qs[256];
  int16_t bsums[16];
};
struct moestream_ggml_block_iq1_m {
  uint8_t qs[32];
  uint8_t qh[16];
  uint8_t scales[8];
};
static_assert(sizeof(moestream_ggml_block_q8_K) == 292);
static_assert(sizeof(moestream_ggml_block_iq1_m) == 56);
extern "C" void quantize_row_q8_K(const float *, void *, int64_t);
extern "C" void ggml_vec_dot_iq1_m_q8_K(int, float *, size_t, const void *,
                                        size_t, const void *, size_t, int);
extern "C" void ggml_vec_dot_iq2_xxs_q8_K(int, float *, size_t, const void *,
                                          size_t, const void *, size_t, int);
extern "C" void ggml_vec_dot_iq3_xxs_q8_K(int, float *, size_t, const void *,
                                          size_t, const void *, size_t, int);
extern "C" void quantize_row_q8_0(const float *, void *, int64_t);
extern "C" void ggml_vec_dot_iq4_nl_q8_0(int, float *, size_t, const void *,
                                         size_t, const void *, size_t, int);
extern "C" void ggml_cpu_init(void);
// The IQ1_M codebook is already linked into this binary for the CUDA IQ1_M
// kernels.  Reuse the same data for the CPU AVX-512 probe instead of copying
// another 2048-entry table into the CPU implementation.
extern "C" const void *ms_grid_iq1s();

// GGML's quantized CPU dot kernels use ggml_table_f32_f16, which is filled by
// ggml_cpu_init().  The normal GGML backend initializes it as a side effect of
// ggml_backend_cpu_init(), but these direct Q8_K adapters can be called before
// any persistent GGML graph exists (for example, from the Qwen CPU-affinity
// fallback).  Without this guard the table is still zero-initialized and the
// dot kernels silently return zero for otherwise valid IQ1/IQ2/IQ3 rows.
static void ensure_ggml_cpu_initialized() {
  static std::once_flag once;
  std::call_once(once, [] { ggml_cpu_init(); });
}
#endif

// Qwen4Exp's direct grouped CPU path is intentionally model-local. The
// process-wide omp_max_threads setting remains the default, but a decode
// benchmark on a machine with SMT may need to compare physical-core-only
// execution without changing OpenMP behavior for attention or other model
// families. Keep the override at Qwen-only call sites rather than calling
// omp_set_num_threads() and perturbing concurrent work globally.
int qwen_cpu_grouped_threads() {
#if defined(_OPENMP)
  static const int requested = [] {
    const char *env = std::getenv("MOESTREAM_QWEN_CPU_THREADS");
    if (!env || !env[0])
      return 0;
    char *end = nullptr;
    const long value = std::strtol(env, &end, 10);
    return end != env && *end == '\0' && value > 0
               ? static_cast<int>(value)
               : 0;
  }();
  const int available = std::max(1, omp_get_max_threads());
  return requested > 0 ? std::min(requested, available) : available;
#else
  return 1;
#endif
}

// Keep the fused-path stage clock behind the same Qwen-only diagnostic switch
// as the non-fused grouped path.  This file cannot use execution_layer.cpp's
// private helper, so mirror the read-only environment check locally rather
// than adding a process-wide timing cost to every CPU kernel.
static bool qwen_cpu_stage_timing_enabled_local() {
  static const bool enabled = [] {
    const char *env = std::getenv("MOESTREAM_QWEN_CPU_STAGE_TIMING");
    return env && env[0] == '1';
  }();
  return enabled;
}

static void qwen_cpu_stage_add(std::atomic<double> &counter,
                               std::chrono::steady_clock::time_point start) {
  PerfCounters::atomic_add(
      counter,
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
          .count());
}

// Thread-safe "log this exactly once" -- cpu_quant_matmul's OMP parallel
// region can call dequant_row from many threads concurrently, so a plain
// `static bool` guard races (multiple threads read `false` before any of
// them writes `true`, each prints). exchange(true) is atomic: only the one
// thread that flips it from false->true gets true back from this call.
static bool should_log_once(std::atomic<bool> &flag) {
  return !flag.exchange(true, std::memory_order_relaxed);
}

float fp16_to_fp32(uint16_t h) { return _cvtsh_ss(h); }

uint16_t fp32_to_fp16(float f) {
  __m128 v = _mm_set_ss(f);
  __m128i h = _mm_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  return _mm_extract_epi16(h, 0);
}

// ----------------------------------------------------
// Core matmul: y[S,O] = x[S,I] @ W^T, W is [O,I] row-major
// Optimized loop ordering: outer O enables write-once per output element.
// ----------------------------------------------------
void matmul(float *y, const float *x, const float *W, int S, int I, int O) {
#pragma omp parallel for schedule(static)
  for (int o = 0; o < O; ++o) {
    const float *w = W + (int64_t)o * I;

    for (int s = 0; s < S; ++s) {
      const float *xs = x + (int64_t)s * I;

      __m512 acc0 = _mm512_setzero_ps();
      __m512 acc1 = _mm512_setzero_ps();
      __m512 acc2 = _mm512_setzero_ps();
      __m512 acc3 = _mm512_setzero_ps();

      int i = 0;
      // 4-way unrolled inner loop (64 floats / 256 bytes per step)
      for (; i + 64 <= I; i += 64) {
        // Prefetch memory for the next unrolled step
        _mm_prefetch((const char *)(xs + i + 128), _MM_HINT_T0);
        _mm_prefetch((const char *)(w + i + 128), _MM_HINT_T0);

        acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(xs + i), _mm512_loadu_ps(w + i),
                               acc0);
        acc1 = _mm512_fmadd_ps(_mm512_loadu_ps(xs + i + 16),
                               _mm512_loadu_ps(w + i + 16), acc1);
        acc2 = _mm512_fmadd_ps(_mm512_loadu_ps(xs + i + 32),
                               _mm512_loadu_ps(w + i + 32), acc2);
        acc3 = _mm512_fmadd_ps(_mm512_loadu_ps(xs + i + 48),
                               _mm512_loadu_ps(w + i + 48), acc3);
      }

      // Combine tree-style
      __m512 acc01 = _mm512_add_ps(acc0, acc1);
      __m512 acc23 = _mm512_add_ps(acc2, acc3);
      __m512 acc_total = _mm512_add_ps(acc01, acc23);

      // Vector tail (16 floats per step)
      for (; i + 16 <= I; i += 16) {
        acc_total = _mm512_fmadd_ps(_mm512_loadu_ps(xs + i),
                                    _mm512_loadu_ps(w + i), acc_total);
      }

      float sum = _mm512_reduce_add_ps(acc_total);

      // Scalar cleanup tail
      for (; i < I; ++i) {
        sum += xs[i] * w[i];
      }

      y[(int64_t)s * O + o] = sum;
    }
  }
}

float dot_f32_avx512(const float *a, const float *b, int n) {
  __m512 acc0 = _mm512_setzero_ps();
  __m512 acc1 = _mm512_setzero_ps();
  __m512 acc2 = _mm512_setzero_ps();
  __m512 acc3 = _mm512_setzero_ps();

  int i = 0;
  // Main loop: 4 vectors = 64 floats per iteration
  for (; i + 64 <= n; i += 64) {
    acc0 =
        _mm512_fmadd_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i), acc0);
    acc1 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i + 16),
                           _mm512_loadu_ps(b + i + 16), acc1);
    acc2 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i + 32),
                           _mm512_loadu_ps(b + i + 32), acc2);
    acc3 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i + 48),
                           _mm512_loadu_ps(b + i + 48), acc3);
  }

  // Combine accumulators tree-style
  __m512 acc01 = _mm512_add_ps(acc0, acc1);
  __m512 acc23 = _mm512_add_ps(acc2, acc3);
  __m512 acc_total = _mm512_add_ps(acc01, acc23);

  // Single vector tail loop (16 floats per step)
  for (; i + 16 <= n; i += 16) {
    acc_total = _mm512_fmadd_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i),
                                acc_total);
  }

  float sum = _mm512_reduce_add_ps(acc_total);

  // Scalar cleanup tail
  for (; i < n; ++i) {
    sum += a[i] * b[i];
  }

  return sum;
}

// Use double accumulation to avoid catastrophic cancellation on large D
void rmsnorm_row(float *out, const float *x, const float *w, int D, float eps) {
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

void softmax_row(float *x, int size) {
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

// Helper: Vectorized exp(-x) approximation for x >= 0 using 512-bit AVX-512.
// Computes e^(-x) using Base-2 range reduction: e^(-x) = 2^(-x * log2(e))
static inline __m512 _mm512_exp_neg_ps(__m512 vx) {
  vx = _mm512_max_ps(vx, _mm512_set1_ps(-88.0f));
  vx = _mm512_min_ps(vx, _mm512_set1_ps(88.0f));

  // Conversion factor to Base-2
  const __m512 log2e = _mm512_set1_ps(1.4426950408889634f);

  // True Base-2 Polynomial coefficients for 2^(-f) on [-0.5, 0.5]
  // Derived from the Taylor series of 2^(-f) = e^(-f * ln(2))
  const __m512 p0 = _mm512_set1_ps(1.0f);
  const __m512 p1 = _mm512_set1_ps(-0.69314718f); // -ln(2)
  const __m512 p2 = _mm512_set1_ps(0.24022650f);  // ln(2)^2 / 2
  const __m512 p3 = _mm512_set1_ps(-0.05550410f); // -ln(2)^3 / 6
  const __m512 p4 = _mm512_set1_ps(0.00961812f);  // ln(2)^4 / 24
  const __m512 p5 = _mm512_set1_ps(-0.00133251f); // -ln(2)^5 / 120

  // 1. Calculate z = x * log2(e)
  __m512 z = _mm512_mul_ps(vx, log2e);

  // 2. n = floor(z + 0.5) (Extract integer exponent)
  __m512 n = _mm512_floor_ps(_mm512_add_ps(z, _mm512_set1_ps(0.5f)));

  // 3. f = z - n (Extract fractional remainder, strictly between -0.5 and 0.5)
  __m512 f = _mm512_sub_ps(z, n);

  // 4. Evaluate polynomial approximation: P(f) approx 2^(-f)
  __m512 p = _mm512_fmadd_ps(p5, f, p4);
  p = _mm512_fmadd_ps(p, f, p3);
  p = _mm512_fmadd_ps(p, f, p2);
  p = _mm512_fmadd_ps(p, f, p1);
  p = _mm512_fmadd_ps(p, f, p0);

  // 5. Scale by 2^(-n) via exponent bias adjustment in integer domain
  __m512i i_n = _mm512_cvtps_epi32(n);
  __m512i bias_n = _mm512_sub_epi32(_mm512_set1_epi32(127), i_n);
  __m512 pow2_neg_n = _mm512_castsi512_ps(_mm512_slli_epi32(bias_n, 23));

  // Final result: 2^(-f) * 2^(-n) = 2^(-z) = e^(-x)
  return _mm512_mul_ps(p, pow2_neg_n);
}

// Vectorized SiLU: x * sigmoid(x) = x / (1 + exp(-x))
// Unrolled 2-way (32 elements per iteration) to hide FMA and division
// latencies.
void silu_inplace(float *x, int n) {
  const __m512 v_one = _mm512_set1_ps(1.0f);
  int i = 0;

  // Process 32 elements per iteration (2x AVX-512 registers)
  for (; i + 32 <= n; i += 32) {
    __m512 v0 = _mm512_loadu_ps(x + i);
    __m512 v1 = _mm512_loadu_ps(x + i + 16);

    // Pass raw v0/v1. The helper natively computes exp(-x)
    __m512 exp_neg0 = _mm512_exp_neg_ps(v0);
    __m512 exp_neg1 = _mm512_exp_neg_ps(v1);

    // denom = 1.0 + exp(-x)
    __m512 denom0 = _mm512_add_ps(v_one, exp_neg0);
    __m512 denom1 = _mm512_add_ps(v_one, exp_neg1);

    // Fast reciprocal: approx_inv = 1 / denom
    __m512 inv0 = _mm512_rcp14_ps(denom0);
    __m512 inv1 = _mm512_rcp14_ps(denom1);

    // One step of Newton-Raphson refinement for single-precision accuracy:
    inv0 = _mm512_mul_ps(inv0,
                         _mm512_fnmadd_ps(denom0, inv0, _mm512_set1_ps(2.0f)));
    inv1 = _mm512_mul_ps(inv1,
                         _mm512_fnmadd_ps(denom1, inv1, _mm512_set1_ps(2.0f)));

    // silu = v * sigmoid(v)
    _mm512_storeu_ps(x + i, _mm512_mul_ps(v0, inv0));
    _mm512_storeu_ps(x + i + 16, _mm512_mul_ps(v1, inv1));
  }

  // Single 16-element vector tail
  for (; i + 16 <= n; i += 16) {
    __m512 v = _mm512_loadu_ps(x + i);
    __m512 exp_neg = _mm512_exp_neg_ps(v);
    __m512 denom = _mm512_add_ps(v_one, exp_neg);

    __m512 inv = _mm512_rcp14_ps(denom);
    inv =
        _mm512_mul_ps(inv, _mm512_fnmadd_ps(denom, inv, _mm512_set1_ps(2.0f)));

    _mm512_storeu_ps(x + i, _mm512_mul_ps(v, inv));
  }

  // Scalar fallback tail
  for (; i < n; ++i) {
    x[i] = x[i] / (1.0f + std::exp(-x[i]));
  }
}

// Element-wise multiply: out[i] = a[i] * b[i]
void elem_mul(float *out, const float *a, const float *b, int n) {
  int i = 0;
  for (; i + 16 <= n; i += 16) {
    _mm512_storeu_ps(
        out + i, _mm512_mul_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i)));
  }
  for (; i < n; ++i)
    out[i] = a[i] * b[i];
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
static inline __m512i unpack_2bit_x16(const uint8_t *q, int shift) {
  __m128i raw = _mm_loadu_si128((const __m128i *)q);
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
static inline void dequant_q2k_group16(const uint8_t *q, int shift, float dl,
                                       float ml, float *out) {
  __m512i vi = unpack_2bit_x16(q, shift);
  __m512 vf = _mm512_cvtepi32_ps(vi);
  __m512 res =
      _mm512_sub_ps(_mm512_mul_ps(_mm512_set1_ps(dl), vf), _mm512_set1_ps(ml));
  _mm512_storeu_ps(out, res);
}

void dequantize_row_q2_K_avx512(const uint8_t *block, float *y, int64_t k) {
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
  if (should_log_once(logged_q2k))
    LOG_INFO("[AVX512-DEQUANT] dequantize_row_q2_K_avx512 is live\n");
  constexpr int QK_K = 256;
  constexpr size_t BLOCK_BYTES = 84; // scales[16] qs[64] d(2) dmin(2)
  const int64_t nb = k / QK_K;
  for (int64_t i = 0; i < nb; ++i) {
    const uint8_t *base = block + (size_t)i * BLOCK_BYTES;
    const uint8_t *scales = base;  // [16]
    const uint8_t *qs = base + 16; // [64]
    uint16_t d_half, dmin_half;
    std::memcpy(&d_half, base + 16 + 64, 2);
    std::memcpy(&dmin_half, base + 16 + 64 + 2, 2);
    const float d = fp16_to_fp32(d_half);
    const float dmin = fp16_to_fp32(dmin_half);

    const uint8_t *q = qs;
    int is = 0;
    float *yy = y + (size_t)i * QK_K;
    for (int n = 0; n < QK_K; n += 128) {
      int shift = 0;
      for (int j = 0; j < 4; ++j) {
        uint8_t sc0 = scales[is++];
        dequant_q2k_group16(q, shift, d * (sc0 & 0xF), dmin * (sc0 >> 4), yy);
        yy += 16;

        uint8_t sc1 = scales[is++];
        dequant_q2k_group16(q + 16, shift, d * (sc1 & 0xF), dmin * (sc1 >> 4),
                            yy);
        yy += 16;

        shift += 2;
      }
      q += 32;
    }
  }
}

// One Q3_K 16-element sub-group: y[l] = dl * (qval[l] - (hmbit_clear[l] ? 4 :
// 0)), where qval is the unsigned 2-bit low code and hmbit_clear tests bit `m`
// of hm[l]. The (qval - 4-or-0) subtraction happens in INTEGER domain (matching
// the scalar reference's int8_t arithmetic before its single multiply by
// dl), so there's exactly one float rounding step, matching the reference.
static inline void dequant_q3k_group16(const uint8_t *q, const uint8_t *hm,
                                       uint8_t m, int shift, float dl,
                                       float *out) {
  __m512i lowbits = unpack_2bit_x16(q, shift);

  __m128i hmraw = _mm_loadu_si128((const __m128i *)hm);
  __m128i mvec = _mm_set1_epi8((char)m);
  __m128i andres = _mm_and_si128(hmraw, mvec);
  __m128i iszero = _mm_cmpeq_epi8(
      andres, _mm_setzero_si128()); // 0xFF (=-1) where hm bit CLEAR
  __m256i iszero16 = _mm256_cvtepi8_epi16(iszero);    // sign-extend
  __m512i iszero32 = _mm512_cvtepi16_epi32(iszero16); // sign-extend -> -1 or 0
  __m512i sub4 = _mm512_and_si512(
      iszero32, _mm512_set1_epi32(4)); // 4 where clear, 0 where set

  __m512i val3 = _mm512_sub_epi32(lowbits, sub4);
  __m512 vf = _mm512_cvtepi32_ps(val3);
  __m512 res = _mm512_mul_ps(_mm512_set1_ps(dl), vf);
  _mm512_storeu_ps(out, res);
}

void dequantize_row_q3_K_avx512(const uint8_t *block, float *y, int64_t k) {
  static std::atomic<bool> logged_q3k{false};
  if (should_log_once(logged_q3k))
    LOG_INFO("[AVX512-DEQUANT] dequantize_row_q3_K_avx512 is live\n");
  constexpr int QK_K = 256;
  constexpr size_t BLOCK_BYTES = 110; // hmask[32] qs[64] scales[12] d(2)
  const int64_t nb = k / QK_K;
  const uint32_t kmask1 = 0x03030303, kmask2 = 0x0f0f0f0f;
  for (int64_t i = 0; i < nb; ++i) {
    const uint8_t *base = block + (size_t)i * BLOCK_BYTES;
    const uint8_t *hmask = base;                 // [32]
    const uint8_t *qs = base + 32;               // [64]
    const uint8_t *scale_bytes = base + 32 + 64; // [12]
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
    const int8_t *scales = (const int8_t *)aux;

    const uint8_t *q = qs;
    const uint8_t *hm = hmask; // NOT advanced across n -- matches reference
                               // (same 32 bytes reused for all 8 bit-planes)
    uint8_t m = 1;
    int is = 0;
    float *yy = y + (size_t)i * QK_K;
    for (int n = 0; n < QK_K; n += 128) {
      int shift = 0;
      for (int j = 0; j < 4; ++j) {
        // FIX: this used to read scales[is+j] / scales[is+j+4] with a single
        // `is += 8` once per n ("Q3_K scales are transposed" per the old
        // comment here) -- that claim was WRONG, found by direct comparison
        // against ggml's real dequantize_row_q3_K (ggml/src/ggml-quants.c),
        // which uses a plain SEQUENTIAL post-increment `scales[is++]` for
        // BOTH groups of every j (order 0,1,2,3,4,5,6,7 per n-block), not a
        // "transposed" 0,4,1,5,2,6,3,7 pattern. Confirmed as the root cause
        // of a real Test 5 failure (max_rel_diff=33.5, values far outside
        // the [-1,1] input range) -- wrong scale applied to the wrong
        // 16-element group, not an out-of-bounds access (both old and new
        // indexing stay within scales[0..15], so this was a pure VALUE bug,
        // not a memory-safety one).
        float dl0 = d_all * (scales[is] - 32);
        is++;
        dequant_q3k_group16(q, hm, m, shift, dl0, yy);
        yy += 16;

        float dl1 = d_all * (scales[is] - 32);
        is++;
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
const void *ms_grid_iq2xxs();
const void *ms_grid_iq2xs();
const void *ms_grid_iq3xxs();
const void *ms_grid_iq2s();
const void *ms_grid_ksigns_iq2xs();
const void *ms_grid_kmask_iq2xs();
}

static inline float fp16_to_fp32_fused(uint16_t h) { return _cvtsh_ss(h); }

// Widened kmask (8 x int32 duplicated across 16 lanes)
static inline __m512i kmask_iq2xs_vec32_16_fused() {
  static __m512i v = [] {
    const uint8_t *kmask8 = (const uint8_t *)ms_grid_kmask_iq2xs();
    int32_t tmp[16];
    for (int j = 0; j < 8; ++j) {
      tmp[j] = (int32_t)kmask8[j];
      tmp[j + 8] = (int32_t)kmask8[j];
    }
    return _mm512_loadu_si512((const void *)tmp);
  }();
  return v;
}
static const uint64_t *iq2xxs_grid_ptr() {
  static const uint64_t *p = (const uint64_t *)ms_grid_iq2xxs(); // [256]
  return p;
}
static const uint64_t *iq2xs_grid_ptr() {
  static const uint64_t *p = (const uint64_t *)ms_grid_iq2xs(); // [512]
  return p;
}
static const uint32_t *iq3xxs_grid_ptr() {
  static const uint32_t *p = (const uint32_t *)ms_grid_iq3xxs(); // [256]
  return p;
}
static const uint64_t *iq2s_grid_ptr() {
  static const uint64_t *p = (const uint64_t *)ms_grid_iq2s(); // [1024]
  return p;
}
static const uint8_t *ksigns_iq2xs_ptr() {
  static const uint8_t *p = (const uint8_t *)ms_grid_ksigns_iq2xs(); // [128]
  return p;
}
// kmask_iq2xs[8] widened to 8 x int32 lanes, cached once -- this table never
// changes across calls, so building the widened vector every call would be
// pure waste.
static __m256i kmask_iq2xs_vec32() {
  static __m256i v = [] {
    const uint8_t *kmask8 = (const uint8_t *)ms_grid_kmask_iq2xs();
    int32_t tmp[8];
    for (int j = 0; j < 8; ++j)
      tmp[j] = (int32_t)kmask8[j];
    return _mm256_loadu_si256((const __m256i *)tmp);
  }();
  return v;
}
// In-register Dequantization + Accumulation into 512-bit register
static inline void dequant_and_fmadd_16(uint64_t grid64_lo, uint64_t grid64_hi,
                                        uint8_t signs_lo, uint8_t signs_hi,
                                        __m512i kmask_vec16, float db,
                                        const float *x_ptr, __m512 &acc) {
  // 1. Direct register load of grid indices into 16 x int32
  __m128i g = _mm_set_epi64x((int64_t)grid64_hi, (int64_t)grid64_lo);
  __m512i gi32 = _mm512_cvtepu8_epi32(g);
  __m512 gf = _mm512_cvtepi32_ps(gi32);

  // 2. AVX-512 Native Masking for sign bits
  __m256i signs_lo_bcast = _mm256_set1_epi32((int32_t)signs_lo);
  __m256i signs_hi_bcast = _mm256_set1_epi32((int32_t)signs_hi);
  __m512i signs_bcast = _mm512_inserti32x8(
      _mm512_castsi256_si512(signs_lo_bcast), signs_hi_bcast, 1);

  __m512i andres = _mm512_and_si512(signs_bcast, kmask_vec16);
  __mmask16 bit_set_mask =
      _mm512_cmpneq_epi32_mask(andres, _mm512_setzero_si512());

  // 3. Construct 0x80000000 sign-bit register and XOR
  __m512 sign_bit = _mm512_castsi512_ps(_mm512_mask_set1_epi32(
      _mm512_setzero_si512(), bit_set_mask, (int32_t)0x80000000));

  __m512 pos = _mm512_mul_ps(_mm512_set1_ps(db), gf);
  __m512 weights_v =
      _mm512_xor_ps(pos, sign_bit); // Final 16 unpacked weights in register

  // 4. Immediate FMA with Activation Vector x (NO intermediate store to RAM)
  __m512 x_v = _mm512_loadu_ps(x_ptr);
  acc = _mm512_fmadd_ps(weights_v, x_v, acc);
}

// Direct fused kernel: computes dot product of IQ2_XXS block weights with float
// activation x
float dot_iq2_xxs_f32_avx512(const uint8_t *block, const float *x, int64_t k) {
  constexpr int QK_K = 256;
  constexpr size_t BLOCK_BYTES = 66; // d(2) + qs[32] uint16 (64 bytes)
  const int64_t nb = k / QK_K;

  const uint64_t *grid_tbl = (const uint64_t *)ms_grid_iq2xxs();
  const uint8_t *ksigns = (const uint8_t *)ms_grid_ksigns_iq2xs();
  __m512i kmask_vec16 = kmask_iq2xs_vec32_16_fused();

  // 4 parallel accumulators to maximize ILP across CPU FMA pipelines
  __m512 acc0 = _mm512_setzero_ps();
  __m512 acc1 = _mm512_setzero_ps();
  __m512 acc2 = _mm512_setzero_ps();
  __m512 acc3 = _mm512_setzero_ps();

  for (int64_t i = 0; i < nb; ++i) {
    const uint8_t *base = block + (size_t)i * BLOCK_BYTES;
    uint16_t d_half;
    std::memcpy(&d_half, base, 2);
    const float d = fp16_to_fp32_fused(d_half);
    const uint8_t *qs_bytes = base + 2;

    const float *x_block = x + (size_t)i * QK_K;

    for (int ib32 = 0; ib32 < QK_K / 32; ib32 += 2) {
      // --- First 32 weights (feeds acc0, acc1) ---
      uint32_t aux32_0, aux32_1;
      std::memcpy(&aux32_0, qs_bytes + ib32 * 8 + 0, 4);
      std::memcpy(&aux32_1, qs_bytes + ib32 * 8 + 4, 4);

      const uint8_t *aux8 = (const uint8_t *)&aux32_0;
      const float db = d * (0.5f + (float)(aux32_1 >> 28)) * 0.25f;

      dequant_and_fmadd_16(grid_tbl[aux8[0]], grid_tbl[aux8[1]],
                           ksigns[(aux32_1 >> 0) & 127],
                           ksigns[(aux32_1 >> 7) & 127], kmask_vec16, db,
                           x_block + ib32 * 32 + 0, acc0);

      dequant_and_fmadd_16(grid_tbl[aux8[2]], grid_tbl[aux8[3]],
                           ksigns[(aux32_1 >> 14) & 127],
                           ksigns[(aux32_1 >> 21) & 127], kmask_vec16, db,
                           x_block + ib32 * 32 + 16, acc1);

      // --- Next 32 weights (feeds acc2, acc3) ---
      int ib32_next = ib32 + 1;
      uint32_t aux32_0_next, aux32_1_next;
      std::memcpy(&aux32_0_next, qs_bytes + ib32_next * 8 + 0, 4);
      std::memcpy(&aux32_1_next, qs_bytes + ib32_next * 8 + 4, 4);

      const uint8_t *aux8_next = (const uint8_t *)&aux32_0_next;
      const float db_next = d * (0.5f + (float)(aux32_1_next >> 28)) * 0.25f;

      dequant_and_fmadd_16(grid_tbl[aux8_next[0]], grid_tbl[aux8_next[1]],
                           ksigns[(aux32_1_next >> 0) & 127],
                           ksigns[(aux32_1_next >> 7) & 127], kmask_vec16,
                           db_next, x_block + ib32_next * 32 + 0, acc2);

      dequant_and_fmadd_16(grid_tbl[aux8_next[2]], grid_tbl[aux8_next[3]],
                           ksigns[(aux32_1_next >> 14) & 127],
                           ksigns[(aux32_1_next >> 21) & 127], kmask_vec16,
                           db_next, x_block + ib32_next * 32 + 16, acc3);
    }
  }

  // Combine accumulators tree-style
  __m512 acc01 = _mm512_add_ps(acc0, acc1);
  __m512 acc23 = _mm512_add_ps(acc2, acc3);
  __m512 acc_total = _mm512_add_ps(acc01, acc23);

  return _mm512_reduce_add_ps(acc_total);
}

// Applies a per-lane sign flip + scale to one 8-byte grid entry, producing 8
// output floats: out[j] = db * grid8[j] * ((signs & kmask[j]) ? -1 : 1).
// Fully vectorized (no scalar loop over j) -- kmask_vec32 is the same 8
// lanes every call, so the AND/compare/blend below replaces what would
// otherwise be 8 scalar branches.
// Applies per-lane sign flip + scale to one 8-byte grid entry (8 output
// floats). Uses AVX-512 bitmask intrinsics (__mmask8) for native sign handling.
static inline void apply_grid_group8(uint64_t grid64, uint8_t signs,
                                     __m256i kmask_vec32, float db,
                                     float *out) {
  // 1. Direct register load: uint64_t -> low 64 bits of XMM register ->
  // zero-extend to 8 x int32
  __m128i g = _mm_cvtsi64_si128(grid64);
  __m256i gi32 = _mm256_cvtepu8_epi32(g);
  __m256 gf = _mm256_cvtepi32_ps(gi32);

  // 2. AVX-512 Native Bitmask Testing:
  // Test (signs & kmask) > 0 directly into an 8-bit AVX-512 k-register mask
  // (__mmask8)
  __m256i signs_bcast = _mm256_set1_epi32((int32_t)signs);
  __m256i andres = _mm256_and_si256(signs_bcast, kmask_vec32);
  __mmask8 bit_set_mask =
      _mm256_cmpneq_epi32_mask(andres, _mm256_setzero_si256());

  // 3. Construct 256-bit sign vector (0x80000000 where mask bit is 1,
  // 0x00000000 where 0)
  __m256 sign_bit = _mm256_castsi256_ps(_mm256_mask_set1_epi32(
      _mm256_setzero_si256(), bit_set_mask, (int32_t)0x80000000));

  // 4. Calculate scaled magnitudes
  __m256 pos = _mm256_mul_ps(_mm256_set1_ps(db), gf);

  // 5. Apply sign flips via 1-cycle float bitwise XOR
  __m256 res = _mm256_xor_ps(pos, sign_bit);

  _mm256_storeu_ps(out, res);
}
void dequantize_row_iq2_xxs_avx512(const uint8_t *block, float *y, int64_t k) {
  static std::atomic<bool> logged{false};
  if (should_log_once(logged))
    LOG_INFO("[AVX512-DEQUANT] dequantize_row_iq2_xxs_avx512 is live\n");

  constexpr int QK_K = 256;
  constexpr size_t BLOCK_BYTES = 66; // d(2) + qs[32] uint16 (64 bytes)
  const int64_t nb = k / QK_K;

  const uint64_t *grid_tbl = iq2xxs_grid_ptr();
  const uint8_t *ksigns = ksigns_iq2xs_ptr();
  __m256i kmask_vec = kmask_iq2xs_vec32();

  for (int64_t i = 0; i < nb; ++i) {
    const uint8_t *base = block + (size_t)i * BLOCK_BYTES;
    uint16_t d_half;
    std::memcpy(&d_half, base, 2);
    const float d = fp16_to_fp32(d_half);
    const uint8_t *qs_bytes = base + 2; // 32 x uint16, little-endian

    float *yy = y + (size_t)i * QK_K;
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
        uint8_t signs = ksigns[(aux32_1 >> (7 * l)) & 127];
        apply_grid_group8(g, signs, kmask_vec, db, yy + ib32 * 32 + l * 8);
      }
    }
  }
}

// Verbatim port of cuda/iq2_matmul.cu's dequant_iq2_xs_coop (itself a
// verbatim port of ggml's dequantize_row_iq2_xs, cross-checked directly
// against build-metal/_deps/ggml-src/src/ggml-quants.c), following the same
// "port from the already-verified, already-running CUDA kernel" methodology
// as the IQ2_XXS/IQ3_XXS/IQ2_S ports above. Reuses apply_grid_group8
// unchanged, and the SAME ksigns_iq2xs/kmask_iq2xs tables IQ2_XXS/IQ2_S use
// (ggml shares these sign/mask tables across all three formats). Structurally
// simpler than IQ2_S: the grid index and sign-table index both come directly
// out of one qs[] uint16 per l (& 511 and >> 9 respectively) -- no separate
// qh high-bit array, no signs-overlapping-the-back-of-qs layout. Block is
// d(2) + qs[32] uint16 (64 bytes) + scales[8] = 74 bytes, matching
// row_bytes()'s existing (I/256)*74 formula.
void dequantize_row_iq2_xs_avx512(const uint8_t *block, float *y, int64_t k) {
  static std::atomic<bool> logged{false};
  if (should_log_once(logged))
    LOG_INFO("[AVX512-DEQUANT] dequantize_row_iq2_xs_avx512 is live\n");

  constexpr int QK_K = 256;
  constexpr size_t BLOCK_BYTES = 74; // d(2) + qs[32] uint16 (64) + scales[8]
  const int64_t nb = k / QK_K;

  const uint64_t *grid_tbl = iq2xs_grid_ptr(); // [512]
  const uint8_t *ksigns = ksigns_iq2xs_ptr();  // [128], shared w/ IQ2_XXS/IQ2_S
  __m256i kmask_vec = kmask_iq2xs_vec32();

  for (int64_t i = 0; i < nb; ++i) {
    const uint8_t *base = block + (size_t)i * BLOCK_BYTES;
    uint16_t d_half;
    std::memcpy(&d_half, base, 2);
    const float d = fp16_to_fp32(d_half);
    const uint8_t *qs_bytes = base + 2;    // 32 x uint16_t, little-endian
    const uint8_t *scales = base + 2 + 64; // [8]

    float *yy = y + (size_t)i * QK_K;
    for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
      const float db0 = d * (0.5f + (float)(scales[ib32] & 0xf)) * 0.25f;
      const float db1 = d * (0.5f + (float)(scales[ib32] >> 4)) * 0.25f;

      for (int l = 0; l < 4; ++l) {
        uint16_t q;
        std::memcpy(&q, qs_bytes + (size_t)(4 * ib32 + l) * 2, 2);
        const float dl = (l < 2) ? db0 : db1;
        uint64_t g = grid_tbl[q & 511];
        uint8_t signs = ksigns[q >> 9];
        apply_grid_group8(g, signs, kmask_vec, dl, yy + ib32 * 32 + l * 8);
      }
    }
  }
}

void dequantize_row_iq3_xxs_avx512(const uint8_t *block, float *y, int64_t k) {
  static std::atomic<bool> logged{false};
  if (should_log_once(logged))
    LOG_INFO("[AVX512-DEQUANT] dequantize_row_iq3_xxs_avx512 is live\n");

  constexpr int QK_K = 256;
  constexpr size_t BLOCK_BYTES = 98; // d(2) + qs[96]
  const int64_t nb = k / QK_K;

  const uint32_t *grid_tbl = iq3xxs_grid_ptr(); // 4-byte entries
  const uint8_t *ksigns = ksigns_iq2xs_ptr();
  __m256i kmask_vec = kmask_iq2xs_vec32();

  for (int64_t i = 0; i < nb; ++i) {
    const uint8_t *base = block + (size_t)i * BLOCK_BYTES;
    uint16_t d_half;
    std::memcpy(&d_half, base, 2);
    const float d = fp16_to_fp32(d_half);
    const uint8_t *qs = base + 2;              // [96]
    const uint8_t *scales_and_signs = qs + 64; // [32] = 4 bytes x 8 ib32

    float *yy = y + (size_t)i * QK_K;
    const uint8_t *qptr = qs;
    for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
      const uint8_t *sas = scales_and_signs + 4 * ib32;
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
        uint64_t g_comb = (uint64_t)g1 | ((uint64_t)g2 << 32);
        apply_grid_group8(g_comb, signs, kmask_vec, db, yy + ib32 * 32 + l * 8);
      }
      qptr += 8;
    }
  }
}

// Verbatim port of cuda/iq2_matmul.cu's dequant_iq2_s_block (itself a
// verbatim port of ggml's dequantize_row_iq2_s), cross-checked against that
// already-verified, already-running CUDA kernel rather than derived fresh
// from ggml's source -- same methodology as the IQ2_XXS/IQ3_XXS ports
// above. Reuses apply_grid_group8 unchanged for the actual vectorized
// grid-lookup + sign-flip application; only the per-l grid-index and
// sign-byte computation differs from IQ2_XXS (IQ2_S packs its qh "high
// bits" and per-ib32 4-bit-pair scales in separate arrays, and its sign
// bytes overlap the BACK half of its own qs array rather than being folded
// into an aux32 word like IQ2_XXS -- see block_iq2_s_cu's field layout in
// the CUDA file for the exact byte offsets this mirrors: d(2) + qs[64] +
// qh[8] + scales[8] = 82 bytes/block, matching IQ2_S's 2.5625 bits/weight).
// Added specifically because a direct throughput benchmark (main_test.cpp's
// bench_cpu_dequant_matmul) measured this type's PREVIOUS scalar fallback
// (ggml_dequant_row) at roughly 11x slower per byte than the already-
// vectorized IQ2_XXS/IQ3_XXS types on the same hardware.
void dequantize_row_iq2_s_avx512(const uint8_t *block, float *y, int64_t k) {
  static std::atomic<bool> logged{false};
  if (should_log_once(logged))
    LOG_INFO("[AVX512-DEQUANT] dequantize_row_iq2_s_avx512 is live\n");

  constexpr int QK_K = 256;
  constexpr size_t BLOCK_BYTES = 82; // d(2) + qs[64] + qh[8] + scales[8]
  const int64_t nb = k / QK_K;

  const uint64_t *grid_tbl = iq2s_grid_ptr(); // [1024]
  __m256i kmask_vec = kmask_iq2xs_vec32();

  for (int64_t i = 0; i < nb; ++i) {
    const uint8_t *base = block + (size_t)i * BLOCK_BYTES;
    uint16_t d_half;
    std::memcpy(&d_half, base, 2);
    const float d = fp16_to_fp32(d_half);
    const uint8_t *qs = base + 2;              // [64]
    const uint8_t *qh = base + 2 + 64;         // [8]
    const uint8_t *scales = base + 2 + 64 + 8; // [8]
    const uint8_t *signs = qs + 32; // back half of qs (ggml's overlap layout)

    float *yy = y + (size_t)i * QK_K;
    for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
      const float db0 = d * (0.5f + (float)(scales[ib32] & 0xf)) * 0.25f;
      const float db1 = d * (0.5f + (float)(scales[ib32] >> 4)) * 0.25f;
      const uint8_t *qs_ib = qs + 4 * ib32;
      const uint8_t *signs_ib = signs + 4 * ib32;

      for (int l = 0; l < 4; ++l) {
        const float dl = (l / 2 == 0) ? db0 : db1;
        const uint32_t idx =
            qs_ib[l] | (((uint32_t)qh[ib32] << (8 - 2 * l)) & 0x300);
        uint64_t g = grid_tbl[idx];
        // Pass `g` directly, no memcpy needed
        apply_grid_group8(g, signs_ib[l], kmask_vec, dl,
                          yy + ib32 * 32 + l * 8);
      }
    }
  }
}

// ============================================================
// EXPERIMENTAL: real 512-bit (16-lane) widened siblings of
// apply_grid_group8/dequantize_row_iq{2_xxs,3_xxs,2_s}_avx512.
//
// apply_grid_group8 processes one 8-element grid group using __m256 (256-
// bit AVX2 width) -- CLAUDE.md's original justification was "the dominant
// cost is the data-dependent grid/sign table loads, not raw ALU
// throughput," so 8-wide was assumed sufficient. Revisited after a direct
// throughput benchmark (main_test.cpp Test 8) found the ALREADY-vectorized
// IQ2_XXS/IQ3_XXS/IQ2_S types only reach ~1-3 GB/s on the target machine,
// vs DMA 6-23 GB/s for its (structurally simpler, plain
// int4) format -- a gap too large to keep assuming ALU width is a non-
// factor without actually testing it.
//
// Deliberately NOT wired into dequant_row()'s dispatch and does NOT modify
// apply_grid_group8 or the three existing dequantize_row_iq*_avx512
// functions above in any way -- those stay exactly as they are, already
// verified on real hardware. These are NEW, ADDITIVE functions specifically
// so they can be independently throughput-benchmarked (added to Test 8)
// and correctness-verified (a new local logic test, same methodology as
// every other dequant addition this session) BEFORE ever being considered
// for the real dispatch path. Do not wire these into dequant_row() until
// both (a) the local logic test passes and (b) Test 8 shows this actually
// beats the 8-wide version on real hardware -- if it doesn't, the original
// "ALU width isn't the bottleneck" assumption was right after all, and
// these should simply not be used.
// ============================================================

// Widened kmask: the same 8-entry table, duplicated across both 8-lane
// halves of one 16-lane register -- built once, cached, exactly mirroring
// kmask_iq2xs_vec32()'s own caching pattern.
static __m512i kmask_iq2xs_vec32_16() {
  static __m512i v = [] {
    const uint8_t *kmask8 = (const uint8_t *)ms_grid_kmask_iq2xs();
    int32_t tmp[16];
    for (int j = 0; j < 8; ++j) {
      tmp[j] = (int32_t)kmask8[j];
      tmp[j + 8] = (int32_t)kmask8[j];
    }
    return _mm512_loadu_si512((const void *)tmp);
  }();
  return v;
}

// Applies a per-lane sign flip + scale to TWO 8-byte grid entries (16
// output floats) in ONE 512-bit operation, instead of two separate 8-wide
// apply_grid_group8 calls. db_lo/db_hi let the two halves use DIFFERENT
// scale factors (needed for IQ2_S/IQ2_XS's db0/db1 alternation every 2
// groups within a super-block; pass the same value twice for IQ2_XXS/
// IQ3_XXS, which use one scale across all 4 groups). Same algorithm as
// apply_grid_group8, just doubled in width -- zero-extend bytes to i32
// lanes BEFORE any shift/compare (same reasoning as the existing Q2_K/Q3_K
// AVX-512 unpack: never operate on tightly-packed bytes directly, or a
// shift can bleed bits in from a neighboring byte).
// Applies per-lane sign flip + scale to TWO 8-byte grid entries (16 output
// floats) in ONE 512-bit op.
static inline void apply_grid_group16(uint64_t grid64_lo, uint64_t grid64_hi,
                                      uint8_t signs_lo, uint8_t signs_hi,
                                      __m512i kmask_vec32_16, float db_lo,
                                      float db_hi, float *out) {
  // 1. Pack grid64_lo and grid64_hi directly into a single 128-bit register (16
  // bytes)
  __m128i g = _mm_set_epi64x((int64_t)grid64_hi,
                             (int64_t)grid64_lo); // [lo: 8 bytes, hi: 8 bytes]
  __m512i gi32 = _mm512_cvtepu8_epi32(g);         // 16 x i32, zero-extended
  __m512 gf = _mm512_cvtepi32_ps(gi32);           // 16 x float

  // 2. Broadcast signs for lo and hi halves
  __m256i signs_lo_bcast = _mm256_set1_epi32((int32_t)signs_lo);
  __m256i signs_hi_bcast = _mm256_set1_epi32((int32_t)signs_hi);
  __m512i signs_bcast = _mm512_inserti32x8(
      _mm512_castsi256_si512(signs_lo_bcast), signs_hi_bcast, 1);

  // 3. Test bit masks: produce 16-bit AVX-512 mask where bits are SET
  __m512i andres = _mm512_and_si512(signs_bcast, kmask_vec32_16);
  __mmask16 bit_set_mask =
      _mm512_cmpneq_epi32_mask(andres, _mm512_setzero_si512());

  // 4. Construct sign bit mask register (0x80000000 where bit is set,
  // 0x00000000 where clear)
  __m512 sign_bit = _mm512_castsi512_ps(_mm512_mask_set1_epi32(
      _mm512_setzero_si512(), bit_set_mask, (int32_t)0x80000000));

  // 5. Build scale vector
  __m256 db_lo_bcast = _mm256_set1_ps(db_lo);
  __m256 db_hi_bcast = _mm256_set1_ps(db_hi);
  __m512 db_vec =
      _mm512_insertf32x8(_mm512_castps256_ps512(db_lo_bcast), db_hi_bcast, 1);

  // 6. Multiply magnitude and perform 1-cycle sign flip
  __m512 pos = _mm512_mul_ps(db_vec, gf);
  __m512 res = _mm512_xor_ps(pos, sign_bit);

  _mm512_storeu_ps(out, res);
}

void dequantize_row_iq2_xxs_avx512_wide(const uint8_t *block, float *y,
                                        int64_t k) {
  static std::atomic<bool> logged{false};
  if (should_log_once(logged))
    LOG_INFO("[AVX512-DEQUANT] dequantize_row_iq2_xxs_avx512_wide is live\n");

  constexpr int QK_K = 256;
  constexpr size_t BLOCK_BYTES = 66;
  const int64_t nb = k / QK_K;

  const uint64_t *grid_tbl = iq2xxs_grid_ptr();
  const uint8_t *ksigns = ksigns_iq2xs_ptr();
  __m512i kmask_vec16 = kmask_iq2xs_vec32_16();

  for (int64_t i = 0; i < nb; ++i) {
    const uint8_t *base = block + (size_t)i * BLOCK_BYTES;
    uint16_t d_half;
    std::memcpy(&d_half, base, 2);
    const float d = fp16_to_fp32(d_half);
    const uint8_t *qs_bytes = base + 2;

    float *yy = y + (size_t)i * QK_K;
    for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
      uint16_t q0, q1, q2, q3;
      std::memcpy(&q0, qs_bytes + (size_t)(4 * ib32 + 0) * 2, 2);
      std::memcpy(&q1, qs_bytes + (size_t)(4 * ib32 + 1) * 2, 2);
      std::memcpy(&q2, qs_bytes + (size_t)(4 * ib32 + 2) * 2, 2);
      std::memcpy(&q3, qs_bytes + (size_t)(4 * ib32 + 3) * 2, 2);
      uint32_t aux32_0 = (uint32_t)q0 | ((uint32_t)q1 << 16);
      uint32_t aux32_1 = (uint32_t)q2 | ((uint32_t)q3 << 16);
      uint8_t aux8[4];
      std::memcpy(aux8, &aux32_0, 4);
      const float db = d * (0.5f + (float)(aux32_1 >> 28)) * 0.25f;

      // All 4 l-groups in this ib32 share the SAME db -- pair
      // l=(0,1) and l=(2,3) into two 16-wide ops instead of four
      // 8-wide ones.
      for (int pair = 0; pair < 2; ++pair) {
        int l0 = pair * 2, l1 = pair * 2 + 1;
        uint64_t g0 = grid_tbl[aux8[l0]];
        uint64_t g1 = grid_tbl[aux8[l1]];
        uint8_t signs0 = ksigns[(aux32_1 >> (7 * l0)) & 127];
        uint8_t signs1 = ksigns[(aux32_1 >> (7 * l1)) & 127];

        apply_grid_group16(g0, g1, signs0, signs1, kmask_vec16, db, db,
                           yy + ib32 * 32 + l0 * 8);
      }
    }
  }
}

void dequantize_row_iq3_xxs_avx512_wide(const uint8_t *block, float *y,
                                        int64_t k) {
  static std::atomic<bool> logged{false};
  if (should_log_once(logged))
    LOG_INFO("[AVX512-DEQUANT] dequantize_row_iq3_xxs_avx512_wide is live\n");

  constexpr int QK_K = 256;
  constexpr size_t BLOCK_BYTES = 98;
  const int64_t nb = k / QK_K;

  const uint32_t *grid_tbl = iq3xxs_grid_ptr();
  const uint8_t *ksigns = ksigns_iq2xs_ptr();
  __m512i kmask_vec16 = kmask_iq2xs_vec32_16();

  for (int64_t i = 0; i < nb; ++i) {
    const uint8_t *base = block + (size_t)i * BLOCK_BYTES;
    uint16_t d_half;
    std::memcpy(&d_half, base, 2);
    const float d = fp16_to_fp32(d_half);
    const uint8_t *qs = base + 2;
    const uint8_t *scales_and_signs = qs + 64;

    float *yy = y + (size_t)i * QK_K;
    const uint8_t *qptr = qs;
    for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
      const uint8_t *sas = scales_and_signs + 4 * ib32;
      uint32_t aux32 = (uint32_t)sas[0] | ((uint32_t)sas[1] << 8) |
                       ((uint32_t)sas[2] << 16) | ((uint32_t)sas[3] << 24);
      const float db = d * (0.5f + (float)(aux32 >> 28)) * 0.5f;

      // Each l-group is itself two 4-byte grid entries (grid1/grid2,
      // output[0..3]/[4..7]) -- combine BOTH l-groups of a pair into
      // one 16-wide op: l0's grid1+grid2 (8 bytes) + l1's grid1+grid2
      // (8 bytes) = 16 bytes, same db for both (all 4 l's share db).
      for (int pair = 0; pair < 2; ++pair) {
        int l0 = pair * 2, l1 = pair * 2 + 1;
        uint32_t g0a = grid_tbl[qptr[2 * l0 + 0]],
                 g0b = grid_tbl[qptr[2 * l0 + 1]];
        uint32_t g1a = grid_tbl[qptr[2 * l1 + 0]],
                 g1b = grid_tbl[qptr[2 * l1 + 1]];
        uint8_t signs0 = ksigns[(aux32 >> (7 * l0)) & 127];
        uint8_t signs1 = ksigns[(aux32 >> (7 * l1)) & 127];
        uint64_t g0 = (uint64_t)g0a | ((uint64_t)g0b << 32);
        uint64_t g1 = (uint64_t)g1a | ((uint64_t)g1b << 32);
        apply_grid_group16(g0, g1, signs0, signs1, kmask_vec16, db, db,
                           yy + ib32 * 32 + l0 * 8);
      }
      qptr += 8;
    }
  }
}

void dequantize_row_iq2_s_avx512_wide(const uint8_t *block, float *y,
                                      int64_t k) {
  static std::atomic<bool> logged{false};
  if (should_log_once(logged))
    LOG_INFO("[AVX512-DEQUANT] dequantize_row_iq2_s_avx512_wide is live\n");

  constexpr int QK_K = 256;
  constexpr size_t BLOCK_BYTES = 82;
  const int64_t nb = k / QK_K;

  const uint64_t *grid_tbl = iq2s_grid_ptr();
  __m512i kmask_vec16 = kmask_iq2xs_vec32_16();

  for (int64_t i = 0; i < nb; ++i) {
    const uint8_t *base = block + (size_t)i * BLOCK_BYTES;
    uint16_t d_half;
    std::memcpy(&d_half, base, 2);
    const float d = fp16_to_fp32(d_half);
    const uint8_t *qs = base + 2;
    const uint8_t *qh = base + 2 + 64;
    const uint8_t *scales = base + 2 + 64 + 8;
    const uint8_t *signs = qs + 32;

    float *yy = y + (size_t)i * QK_K;
    for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
      const float db0 = d * (0.5f + (float)(scales[ib32] & 0xf)) * 0.25f;
      const float db1 = d * (0.5f + (float)(scales[ib32] >> 4)) * 0.25f;
      const uint8_t *qs_ib = qs + 4 * ib32;
      const uint8_t *signs_ib = signs + 4 * ib32;

      // l=(0,1) both use db0, l=(2,3) both use db1 -- this pairing
      // is exactly the natural db0/db1 split, so each 16-wide op
      // needs only ONE scale value (db_lo==db_hi), same as the other
      // two types above, just for a different structural reason.
      for (int pair = 0; pair < 2; ++pair) {
        int l0 = pair * 2, l1 = pair * 2 + 1;
        const float dbp = (pair == 0) ? db0 : db1;
        const uint32_t idx0 =
            qs_ib[l0] | (((uint32_t)qh[ib32] << (8 - 2 * l0)) & 0x300);
        const uint32_t idx1 =
            qs_ib[l1] | (((uint32_t)qh[ib32] << (8 - 2 * l1)) & 0x300);
        uint64_t g0 = grid_tbl[idx0];
        uint64_t g1 = grid_tbl[idx1];

        // Pass `g0` and `g1` directly as the first two parameters
        apply_grid_group16(g0, g1, signs_ib[l0], signs_ib[l1], kmask_vec16, dbp,
                           dbp, yy + ib32 * 32 + l0 * 8);
      }
    }
  }
}

// Offline weight quantization: Quantizes to int8, then offsets by +128 to map
// to uint8
float qrow_w_u8_offset(const float *w_in, uint8_t *w_out, int I) {
  __m512 v_amax = _mm512_setzero_ps();
  int i = 0;

  for (; i + 16 <= I; i += 16) {
    v_amax = _mm512_max_ps(v_amax, _mm512_abs_ps(_mm512_loadu_ps(w_in + i)));
  }
  float amax = _mm512_reduce_max_ps(v_amax);
  for (; i < I; ++i) {
    float a = std::abs(w_in[i]);
    if (a > amax)
      amax = a;
  }

  float s = amax / 127.0f;
  if (s < 1e-12f)
    s = 1e-12f;
  float inv = 1.0f / s;

  i = 0;
  __m512 v_inv = _mm512_set1_ps(inv);
  const __m512i offset_128 = _mm512_set1_epi8((char)128);

  // Unrolled 64-element block
  for (; i + 64 <= I; i += 64) {
    __m512i i0 =
        _mm512_cvtps_epi32(_mm512_mul_ps(_mm512_loadu_ps(w_in + i), v_inv));
    __m512i i1 = _mm512_cvtps_epi32(
        _mm512_mul_ps(_mm512_loadu_ps(w_in + i + 16), v_inv));
    __m512i i2 = _mm512_cvtps_epi32(
        _mm512_mul_ps(_mm512_loadu_ps(w_in + i + 32), v_inv));
    __m512i i3 = _mm512_cvtps_epi32(
        _mm512_mul_ps(_mm512_loadu_ps(w_in + i + 48), v_inv));

    __m256i s16_0 = _mm512_cvtepi32_epi16(i0);
    __m256i s16_1 = _mm512_cvtepi32_epi16(i1);
    __m256i s16_2 = _mm512_cvtepi32_epi16(i2);
    __m256i s16_3 = _mm512_cvtepi32_epi16(i3);

    __m512i comb_16_01 =
        _mm512_inserti64x4(_mm512_castsi256_si512(s16_0), s16_1, 1);
    __m512i comb_16_23 =
        _mm512_inserti64x4(_mm512_castsi256_si512(s16_2), s16_3, 1);

    __m256i s8_01 = _mm512_cvtepi16_epi8(comb_16_01);
    __m256i s8_23 = _mm512_cvtepi16_epi8(comb_16_23);

    __m512i final_s8 =
        _mm512_inserti64x4(_mm512_castsi256_si512(s8_01), s8_23, 1);

    // Add 128 to shift from [-127, 127] to [1, 255]
    __m512i final_u8 = _mm512_add_epi8(final_s8, offset_128);
    _mm512_storeu_si512((void *)(w_out + i), final_u8);
  }

  // Scalar tail
  for (; i < I; ++i) {
    w_out[i] =
        (uint8_t)(std::max(-127, std::min(127, (int)lrintf(w_in[i] * inv))) +
                  128);
  }
  return s;
}

// ============================================================
// IDOT (Integer Dot Product) Kernels — the biggest single win
// ============================================================
// Quantize one row of activations to int8 symmetric, 64-way unrolled return the
// scale.
float qrow_i8(const float *x, int8_t *q, int I) {
  __m512 v_amax = _mm512_setzero_ps();
  int i = 0;

  for (; i + 16 <= I; i += 16) {
    v_amax = _mm512_max_ps(v_amax, _mm512_abs_ps(_mm512_loadu_ps(x + i)));
  }
  float amax = _mm512_reduce_max_ps(v_amax);
  for (; i < I; ++i) {
    float a = std::abs(x[i]);
    if (a > amax)
      amax = a;
  }

  float s = amax / 127.0f;
  if (s < 1e-12f)
    s = 1e-12f;
  float inv = 1.0f / s;

  i = 0;
  __m512 v_inv = _mm512_set1_ps(inv);

  for (; i + 64 <= I; i += 64) {
    __m512i i0 =
        _mm512_cvtps_epi32(_mm512_mul_ps(_mm512_loadu_ps(x + i), v_inv));
    __m512i i1 =
        _mm512_cvtps_epi32(_mm512_mul_ps(_mm512_loadu_ps(x + i + 16), v_inv));
    __m512i i2 =
        _mm512_cvtps_epi32(_mm512_mul_ps(_mm512_loadu_ps(x + i + 32), v_inv));
    __m512i i3 =
        _mm512_cvtps_epi32(_mm512_mul_ps(_mm512_loadu_ps(x + i + 48), v_inv));

    __m256i s16_0 = _mm512_cvtepi32_epi16(i0);
    __m256i s16_1 = _mm512_cvtepi32_epi16(i1);
    __m256i s16_2 = _mm512_cvtepi32_epi16(i2);
    __m256i s16_3 = _mm512_cvtepi32_epi16(i3);

    __m512i comb_16_01 =
        _mm512_inserti64x4(_mm512_castsi256_si512(s16_0), s16_1, 1);
    __m512i comb_16_23 =
        _mm512_inserti64x4(_mm512_castsi256_si512(s16_2), s16_3, 1);

    __m256i s8_01 = _mm512_cvtepi16_epi8(comb_16_01);
    __m256i s8_23 = _mm512_cvtepi16_epi8(comb_16_23);

    __m512i final_s8 =
        _mm512_inserti64x4(_mm512_castsi256_si512(s8_01), s8_23, 1);
    _mm512_storeu_si512((void *)(q + i), final_s8);
  }

  for (; i < I; ++i) {
    q[i] = (int8_t)lrintf(x[i] * inv);
  }
  return s;
}

// uint8 x int8 VNNI dot product
int32_t dot_u8s8_avx512(const uint8_t *w, const int8_t *x, int I) {
  int32_t sum = 0;
  int i = 0;

#if defined(__AVX512VNNI__) && defined(__AVX512BW__)
  __m512i acc_dp = _mm512_setzero_si512();
  __m512i acc_x = _mm512_setzero_si512();

  const __m512i ones_8 = _mm512_set1_epi8(1);
  const __m512i ones_16 = _mm512_set1_epi16(1);

  for (; i + 64 <= I; i += 64) {
    __m512i wv = _mm512_loadu_si512((const void *)(w + i)); // natively u8
    __m512i xv = _mm512_loadu_si512((const void *)(x + i)); // natively s8

    // 1. Direct hardware VNNI
    acc_dp = _mm512_dpbusd_epi32(acc_dp, wv, xv);

    // 2. Accumulate sum(x) using maddubs (u8 x s8 -> s16)
    __m512i sum_x_s16 = _mm512_maddubs_epi16(ones_8, xv);
    acc_x = _mm512_add_epi32(acc_x, _mm512_madd_epi16(sum_x_s16, ones_16));
  }

  int32_t raw_dot = _mm512_reduce_add_epi32(acc_dp);
  int32_t sum_x_total = _mm512_reduce_add_epi32(acc_x);

  // Reverse the +128 offset mathematically:
  // sum((w + 128) * x) = sum(w * x) + 128 * sum(x)
  sum = raw_dot - (sum_x_total * 128);
#endif

  // Scalar tail
  for (; i < I; ++i) {
    sum += ((int)w[i] - 128) * x[i];
  }
  return sum;
}

// int4(packed)×int8 dot product using AVX-512 VNNI
// Unpacks 2 nibbles/byte to int8 [-8,7], then does the sign-trick VNNI dot.
int32_t dot_i4i8_avx512(const uint8_t *w4, const int8_t *x, int I) {
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
    __m256i by = _mm256_loadu_si256((const __m256i *)(w4 + (i >> 1)));
    __m256i lo = _mm256_and_si256(by, m4v);
    __m256i hi = _mm256_and_si256(_mm256_srli_epi16(by, 4), m4v);
    __m256i z0 = _mm256_unpacklo_epi8(lo, hi);
    __m256i z1 = _mm256_unpackhi_epi8(lo, hi);
    __m512i wv = _mm512_sub_epi8(
        _mm512_inserti64x4(_mm512_castsi256_si512(z0), z1, 1), b8v);
    __m512i xv = _mm512_permutexvar_epi64(
        xidx, _mm512_loadu_si512((const void *)(x + i)));
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
  if (i < I) {
    uint8_t b = w4[i >> 1];
    sum += ((int)(b & 0xF) - 8) * x[i];
  }
  return sum;
}

// Thread-local scratch storage to completely avoid malloc/free overhead on
// every call.
struct ThreadScratch {
  std::vector<int8_t> xq;
  std::vector<float> sx;

  void ensure_size(size_t S, size_t I) {
    if (xq.size() < S * I)
      xq.resize(S * I);
    if (sx.size() < S)
      sx.resize(S);
  }
};

static thread_local ThreadScratch g_scratch;

// IDOT matmul: y[S,O] = x[S,I] @ W^T
// W is now unsigned uint8_t (pre-offset by +128)
void matmul_u8_idot(float *y, const float *x, const uint8_t *W_u8,
                    const float *scale, int S, int I, int O) {
  g_scratch.ensure_size(S, I);
  int8_t *xq = g_scratch.xq.data();
  float *sx = g_scratch.sx.data();

#pragma omp parallel for schedule(static)
  for (int s = 0; s < S; ++s) {
    sx[s] = qrow_i8(x + (int64_t)s * I, xq + (int64_t)s * I, I);
  }

#pragma omp parallel for schedule(static)
  for (int o = 0; o < O; ++o) {
    const uint8_t *w = W_u8 + (int64_t)o * I;
    float sc = scale[o];

    if (o + 1 < O) {
      _mm_prefetch((const char *)(W_u8 + (int64_t)(o + 1) * I), _MM_HINT_T0);
    }

    for (int s = 0; s < S; ++s) {
      const int8_t *xs = xq + (int64_t)s * I;
      int32_t dot = dot_u8s8_avx512(w, xs, I);
      y[(int64_t)s * O + o] = (float)dot * sc * sx[s];
    }
  }
}

// IDOT matmul: quantize activations to int8, then integer dot product
// y[S,O] = x[S,I] @ W^T, W = int8 per-row quantized with scale[O]
// deprecated, use matmul_u8_idot
void matmul_q8_idot(float *y, const float *x, const int8_t *q,
                    const float *scale, int S, int I, int O) {
  // 1. Thread-safe, non-allocating memory reuse
  matmul_u8_idot(y, x, (const uint8_t *)q, scale, S, I, O);
}

// IDOT matmul for int4 packed: y[S,O] = x[S,I] @ W^T, W = int4-packed with
// scale[O]
void matmul_i4_idot(float *y, const float *x, const uint8_t *q4,
                    const float *scale, int S, int I, int O) {
  int rb = (I + 1) / 2; // row bytes for int4 packed

  // 1. Thread-safe, non-allocating memory reuse
  g_scratch.ensure_size(S, I);
  int8_t *xq = g_scratch.xq.data();
  float *sx = g_scratch.sx.data();

// 2. Parallelize activation quantization across rows
#pragma omp parallel for schedule(static)
  for (int s = 0; s < S; ++s) {
    sx[s] = qrow_i8(x + (int64_t)s * I, xq + (int64_t)s * I, I);
  }

// 3. Optimized parallel matrix multiplication loop
#pragma omp parallel for schedule(static)
  for (int o = 0; o < O; ++o) {
    const uint8_t *w = q4 + (int64_t)o * rb;
    float sc = scale[o];

    // Prefetch next row's weight vector into L1/L2 cache
    if (o + 1 < O) {
      _mm_prefetch((const char *)(q4 + (int64_t)(o + 1) * rb), _MM_HINT_T0);
    }

    for (int s = 0; s < S; ++s) {
      const int8_t *xs = xq + (int64_t)s * I;
      int32_t dot = dot_i4i8_avx512(w, xs, I);
      y[(int64_t)s * O + o] = (float)dot * sc * sx[s];
    }
  }
}

// ----------------------------------------------------
// AVX-512 Block Math Operations (Q4_0 × Q8_0)
// ----------------------------------------------------
float dot_q4_0_q8_0_avx512(const block_q4_0 *w, const block_q8_0 *x,
                           int num_blocks) {
  float sum_f32 = 0.0f;
  const __m256i m4 = _mm256_set1_epi8(0x0F);

  for (int i = 0; i < num_blocks; i += 2) {
    if (i + 1 >= num_blocks) {
      // Handle odd last block
      float scale = fp16_to_fp32(w[i].d) * fp16_to_fp32(x[i].d);
      for (int j = 0; j < 16; ++j) {
        int8_t w0 = (int8_t)(w[i].qs[j] & 0xF) - 8;
        int8_t w1 = (int8_t)(w[i].qs[j] >> 4) - 8;
        sum_f32 += scale * ((float)w0 * x[i].qs[j * 2] +
                            (float)w1 * x[i].qs[j * 2 + 1]);
      }
      break;
    }
    __m256i raw_w = _mm256_loadu_si256((const __m256i *)w[i].qs);
    __m256i lo_nibbles = _mm256_and_si256(raw_w, m4);
    __m256i hi_nibbles = _mm256_and_si256(_mm256_srli_epi16(raw_w, 4), m4);

    __m512i v_w =
        _mm512_inserti64x4(_mm512_castsi256_si512(lo_nibbles), hi_nibbles, 1);
    __m512i v_x = _mm512_loadu_si512((const void *)x[i].qs);
    __m512i acc_dp = _mm512_setzero_si512();
    __m512i ones_8 = _mm512_set1_epi8(1);
    __m512i ones_16 = _mm512_set1_epi16(1);
#if defined(__AVX512VNNI__) && defined(__AVX512BW__)
    acc_dp = _mm512_dpbusd_epi32(acc_dp, v_w, v_x);
#else
    // Portable fallback for AVX-512 hosts without VNNI (e.g. early
    // Skylake-X/Cascade Lake, which have the rest of the AVX-512 baseline
    // this file needs but not VNNI specifically): madd(maddubs(u8,s8),
    // ones16) computes the same per-int32-lane sum of 4 byte-products
    // _mm512_dpbusd_epi32 would, just via two AVX512BW instructions
    // instead of one VNNI instruction -- the same equivalence already
    // used for the sum_x correction term just below, and the same
    // guard+fallback pattern dot_u8s8_avx512/dot_i4i8_avx512 already use
    // elsewhere in this file (this call site was the one place that
    // equivalent guard was missing, surfaced by a real build on a host
    // whose GCC treats an unguarded VNNI intrinsic as a hard error rather
    // than silently miscompiling).
    acc_dp = _mm512_madd_epi16(_mm512_maddubs_epi16(v_w, v_x), ones_16);
#endif

    __m512i sum_x = _mm512_maddubs_epi16(ones_8, v_x);
    sum_x = _mm512_madd_epi16(sum_x, ones_16);

    __m512i correction = _mm512_slli_epi32(sum_x, 3);
    __m512i final_int32 = _mm512_sub_epi32(acc_dp, correction);
    __m256i int_sum_0 = _mm512_castsi512_si256(final_int32);
    int_sum_0 = _mm256_add_epi32(int_sum_0, _mm256_srli_si256(int_sum_0, 8));
    int_sum_0 = _mm256_add_epi32(int_sum_0, _mm256_srli_si256(int_sum_0, 4));
    int val_0 =
        _mm256_extract_epi32(int_sum_0, 0) + _mm256_extract_epi32(int_sum_0, 4);
    __m256i int_sum_1 = _mm512_extracti64x4_epi64(final_int32, 1);
    int_sum_1 = _mm256_add_epi32(int_sum_1, _mm256_srli_si256(int_sum_1, 8));
    int_sum_1 = _mm256_add_epi32(int_sum_1, _mm256_srli_si256(int_sum_1, 4));
    int val_1 =
        _mm256_extract_epi32(int_sum_1, 0) + _mm256_extract_epi32(int_sum_1, 4);
    float s0 = fp16_to_fp32(w[i].d) * fp16_to_fp32(x[i].d);
    float s1 = fp16_to_fp32(w[i + 1].d) * fp16_to_fp32(x[i + 1].d);
    sum_f32 += (val_0 * s0) + (val_1 * s1);
  }
  return sum_f32;
}

void quantize_activations_q8_0(const float *__restrict__ x,
                               block_q8_0 *__restrict__ y, int num_elements) {
  int num_blocks = num_elements / 32;
#pragma omp parallel for schedule(static)
  for (int b = 0; b < num_blocks; ++b) {
    const float *block_src = x + (b * 32);

    __m512 v0 = _mm512_loadu_ps(block_src);
    __m512 v1 = _mm512_loadu_ps(block_src + 16);
    __m512 abs0 = _mm512_abs_ps(v0);
    __m512 abs1 = _mm512_abs_ps(v1);
    __m512 max_v = _mm512_max_ps(abs0, abs1);
    float amax = _mm512_reduce_max_ps(max_v);
    float scale = amax / 127.0f;
    if (scale < 1e-12f)
      scale = 1e-12f;
    float inv_scale = 1.0f / scale;
    y[b].d = fp32_to_fp16(scale);

    __m512i r0 =
        _mm512_cvtps_epi32(_mm512_mul_ps(v0, _mm512_set1_ps(inv_scale)));
    __m512i r1 =
        _mm512_cvtps_epi32(_mm512_mul_ps(v1, _mm512_set1_ps(inv_scale)));
    __m256i packed16_0 = _mm512_cvtepi32_epi16(r0);
    __m256i packed16_1 = _mm512_cvtepi32_epi16(r1);

    __m512i combined =
        _mm512_inserti64x4(_mm512_castsi256_si512(packed16_0), packed16_1, 1);
    _mm256_storeu_si256((__m256i *)(y[b].qs), _mm512_cvtepi16_epi8(combined));
  }
}

// Fused Matrix Multiplication Wrapper for IQ2_XXS:
// Calculates y[S, O] = x[S, I] @ W^T where W is IQ2_XXS quantized.
// Bypasses temporary dequantization memory buffers completely!
void matmul_iq2_xxs_fused(float *y, const float *x, const uint8_t *W_quantized,
                          int S, int I, int O) {
  constexpr int QK_K = 256;
  constexpr size_t BLOCK_BYTES = 66; // 66 bytes per 256 weights in IQ2_XXS
  const size_t row_bytes = (size_t)(I / QK_K) * BLOCK_BYTES;

  // Outer loop over sequence length S
  for (int s = 0; s < S; ++s) {
    const float *xs = x + (int64_t)s * I;
    float *ys = y + (int64_t)s * O;

    // 2026-08-29: TRIED an S==1 (decode) serial-dispatch gate here, matching
    // the S>1-gating pattern this project used successfully elsewhere
    // (q_nope_gather_all, this same session's attention_layer.cpp WUV
    // fixes) -- REVERTED, a real, measured regression, not a win. Unlike
    // those earlier fixes (an already-parallel OUTER loop meant the inner
    // OMP was purely redundant -- a genuinely free removal), this call has
    // no outer parallelism to fall back on: GLM-5.2's real expert dims are
    // gate/up I=6144,O=2048 and down I=2048,O=6144 (confirmed by reading
    // the live GGUF's own tensor shapes, not assumed) -- large enough that
    // going single-threaded over O rows is a real, substantial loss of
    // parallelism, not just avoided team-spawn overhead. Real hardware
    // confirmed this: per-CPU-dispatched-expert cost went from ~2.7ms
    // (parallel, baseline) to ~7.5ms (serial, after the gate) -- consistent
    // with the ~10x serial-vs-10-thread-parallel slowdown these dims would
    // predict. The real perf-profiling finding that motivated trying this
    // (this call site is one of the few resolvable callers into libgomp's
    // barrier-wait hotspot, CLAUDE.md's 2026-08-29 entry) is still real --
    // but per that same entry's own point 4 (OMP_WAIT_POLICY=PASSIVE making
    // things WORSE), the background spin time isn't actually costing
    // wall-clock time, so there was no real problem here to fix in the
    // first place. Left as the original, always-parallel form.
#pragma omp parallel for schedule(static)
    for (int o = 0; o < O; ++o) {
      const uint8_t *row_w = W_quantized + (size_t)o * row_bytes;

      // Software Prefetch next row's weight vector into L1/L2 cache
      if (o + 1 < O) {
        _mm_prefetch((const char *)(W_quantized + (size_t)(o + 1) * row_bytes),
                     _MM_HINT_T0);
      }

      // Execute in-register fused dequantization + dot product
      ys[o] = dot_iq2_xxs_f32_avx512(row_w, xs, I);
    }
  }
}

// ============================================================
// EXPERIMENTAL VNNI prototype for IQ2_XXS -- feasibility check only, NOT
// wired into any production dispatch path. Exists to answer "would a VNNI
// int8 dot beat the existing dot_iq2_xxs_f32_avx512 FMA path above" before
// committing to a real production kernel (see CLAUDE.md's "CPU expert
// dispatch is unexpectedly slow" / cpu_offload_max_experts investigation).
//
// Reuses the EXACT SAME grid/sign bit-unpacking as dot_iq2_xxs_f32_avx512
// above (same grid_tbl/ksigns/kmask_vec16 lookups, byte-for-byte -- only
// the final combine step differs) specifically so this prototype inherits
// that function's already-proven-correct byte interpretation instead of
// re-deriving it and risking a NEW parsing bug. Sign is applied to the
// ALREADY-int8-quantized ACTIVATION lane instead of the weight
// (mathematically equivalent: sign*mag*x == mag*(sign*x)) -- the grid
// magnitude byte is fed directly, unsigned, to a VNNI uint8xint8 dot with
// no +128 debias needed (unlike a Q8_0-style signed-weight VNNI dot),
// since the grid magnitude byte is already non-negative by construction,
// exactly as dot_iq2_xxs_f32_avx512's own _mm512_cvtepu8_epi32 already
// treats it.
static inline void dequant_pack16_u8s8(uint64_t grid64_lo, uint64_t grid64_hi,
                                       uint8_t signs_lo, uint8_t signs_hi,
                                       __m512i kmask_vec16,
                                       const int8_t *xq_ptr, uint8_t *mag_out,
                                       int8_t *xsigned_out) {
  // Raw grid bytes ARE the unsigned magnitude (0-255) -- identical read to
  // dequant_and_fmadd_16's own `g`/`gi32` above.
  __m128i g = _mm_set_epi64x((int64_t)grid64_hi, (int64_t)grid64_lo);
  _mm_storeu_si128((__m128i *)mag_out, g);

  // Identical sign-bit derivation to dequant_and_fmadd_16 above (same
  // signs_bcast/andres/bit_set_mask construction) -- only the consumer of
  // bit_set_mask differs (blend an int8 activation lane instead of XOR'ing
  // a float sign bit).
  __m256i signs_lo_bcast = _mm256_set1_epi32((int32_t)signs_lo);
  __m256i signs_hi_bcast = _mm256_set1_epi32((int32_t)signs_hi);
  __m512i signs_bcast = _mm512_inserti32x8(
      _mm512_castsi256_si512(signs_lo_bcast), signs_hi_bcast, 1);
  __m512i andres = _mm512_and_si512(signs_bcast, kmask_vec16);
  __mmask16 bit_set_mask =
      _mm512_cmpneq_epi32_mask(andres, _mm512_setzero_si512());

  __m128i xq = _mm_loadu_si128((const __m128i *)xq_ptr);
  __m128i xq_neg = _mm_sub_epi8(_mm_setzero_si128(), xq);
  __m128i xsigned = _mm_mask_blend_epi8(bit_set_mask, xq, xq_neg);
  _mm_storeu_si128((__m128i *)xsigned_out, xsigned);
}

// Combined 64-lane VNNI dot: processes TWO 32-element groups (each with its
// own db scale) in ONE _mm512_dpbusd_epi32 call, splitting the resulting
// 16x int32 accumulator into two 8-lane halves (lanes 0-7 <- bytes 0-31,
// lanes 8-15 <- bytes 32-63) so each group's own dot product can still be
// scaled independently by its caller.
//
// REAL BUG this replaces, caught by this bench's own honest numbers before
// any conclusion was drawn (not assumed): an earlier version called a
// 32-lane-only `dot_u8s8_direct_avx512(w, x, 32)` per group -- but its VNNI
// branch was gated `for (; i+64<=I; i+=64)`, and 32 < 64 means that loop
// condition is false on the FIRST check, so the VNNI branch never executed
// even once; every call silently fell through to the scalar tail loop (32
// plain int32 multiply-adds). The first bench run this produced (VNNI
// 1.18us/call vs FMA 0.34us/call, i.e. VNNI looking ~3.5x SLOWER) was
// therefore comparing scalar-vs-SIMD, not VNNI-vs-FMA -- a real, if
// unintentional, instance of this project's own documented "the two
// benchmark sides must go through the same dispatch path" lesson (see
// CLAUDE.md's iq2_matmul.cu shfl-reduction entry). Fixed by actually
// filling the full 64-lane register with two real 32-element groups.
static void dot_u8s8_direct_avx512_x2(const uint8_t *w64, const int8_t *x64,
                                      int32_t &sum_lo, int32_t &sum_hi) {
#if defined(__AVX512VNNI__) && defined(__AVX512BW__) && defined(__AVX512VL__)
  __m512i wv = _mm512_loadu_si512((const void *)w64);
  __m512i xv = _mm512_loadu_si512((const void *)x64);
  __m512i acc = _mm512_dpbusd_epi32(_mm512_setzero_si512(), wv, xv);
  __m256i acc_lo = _mm512_extracti64x4_epi64(acc, 0);
  __m256i acc_hi = _mm512_extracti64x4_epi64(acc, 1);
  // GCC does not provide an _mm256_reduce_add_epi32 intrinsic (the
  // corresponding reduction helper exists for __m512i only).  Keep this
  // helper usable with both GCC and Clang by reducing the two extracted
  // eight-lane vectors explicitly.
  alignas(32) int32_t lanes[8];
  _mm256_store_si256((__m256i *)lanes, acc_lo);
  sum_lo = lanes[0] + lanes[1] + lanes[2] + lanes[3] +
           lanes[4] + lanes[5] + lanes[6] + lanes[7];
  _mm256_store_si256((__m256i *)lanes, acc_hi);
  sum_hi = lanes[0] + lanes[1] + lanes[2] + lanes[3] +
           lanes[4] + lanes[5] + lanes[6] + lanes[7];
#else
  sum_lo = 0;
  for (int i = 0; i < 32; ++i)
    sum_lo += (int32_t)w64[i] * (int32_t)x64[i];
  sum_hi = 0;
  for (int i = 32; i < 64; ++i)
    sum_hi += (int32_t)w64[i] * (int32_t)x64[i];
#endif
}

// Full IQ2_XXS row dot via the VNNI prototype above. `xq` must be
// pre-quantized (e.g. via quantize_activations_q8_0) over the SAME k
// elements, 32-wide blocks -- num_blocks = k/32 entries, xq[block_idx]
// aligns 1:1 with this function's own ib32 sub-block index.
float dot_iq2_xxs_vnni_prototype(const uint8_t *block, const block_q8_0 *xq,
                                 int64_t k) {
  constexpr int QK_K = 256;
  constexpr size_t BLOCK_BYTES = 66;
  const int64_t nb = k / QK_K;

  const uint64_t *grid_tbl = (const uint64_t *)ms_grid_iq2xxs();
  const uint8_t *ksigns = (const uint8_t *)ms_grid_ksigns_iq2xs();
  __m512i kmask_vec16 = kmask_iq2xs_vec32_16_fused();

  double total = 0.0; // double accumulation -- this is a reference-quality
                      // implementation for a feasibility bench, not yet
                      // optimized for the accumulator width production
                      // code would use.
  alignas(64) uint8_t mag64[64];
  alignas(64) int8_t xsigned64[64];

  for (int64_t i = 0; i < nb; ++i) {
    const uint8_t *base = block + (size_t)i * BLOCK_BYTES;
    uint16_t d_half;
    std::memcpy(&d_half, base, 2);
    const float d = fp16_to_fp32_fused(d_half);
    const uint8_t *qs_bytes = base + 2;

    // Two 32-element groups per iteration (ib32, ib32+1) -- fills one full
    // 64-lane VNNI call instead of a 32-lane-only one, see
    // dot_u8s8_direct_avx512_x2's own comment for why that matters.
    for (int ib32 = 0; ib32 < QK_K / 32; ib32 += 2) {
      uint32_t aux32_0, aux32_1;
      std::memcpy(&aux32_0, qs_bytes + ib32 * 8 + 0, 4);
      std::memcpy(&aux32_1, qs_bytes + ib32 * 8 + 4, 4);
      const uint8_t *aux8 = (const uint8_t *)&aux32_0;
      const float db = d * (0.5f + (float)(aux32_1 >> 28)) * 0.25f;

      int block_idx = (int)(i * (QK_K / 32) + ib32);
      const int8_t *xq_row = xq[block_idx].qs;

      dequant_pack16_u8s8(grid_tbl[aux8[0]], grid_tbl[aux8[1]],
                          ksigns[(aux32_1 >> 0) & 127],
                          ksigns[(aux32_1 >> 7) & 127], kmask_vec16, xq_row + 0,
                          mag64 + 0, xsigned64 + 0);
      dequant_pack16_u8s8(grid_tbl[aux8[2]], grid_tbl[aux8[3]],
                          ksigns[(aux32_1 >> 14) & 127],
                          ksigns[(aux32_1 >> 21) & 127], kmask_vec16,
                          xq_row + 16, mag64 + 16, xsigned64 + 16);

      int ib32_next = ib32 + 1;
      uint32_t aux32_0_next, aux32_1_next;
      std::memcpy(&aux32_0_next, qs_bytes + ib32_next * 8 + 0, 4);
      std::memcpy(&aux32_1_next, qs_bytes + ib32_next * 8 + 4, 4);
      const uint8_t *aux8_next = (const uint8_t *)&aux32_0_next;
      const float db_next = d * (0.5f + (float)(aux32_1_next >> 28)) * 0.25f;

      int block_idx_next = (int)(i * (QK_K / 32) + ib32_next);
      const int8_t *xq_row_next = xq[block_idx_next].qs;

      dequant_pack16_u8s8(grid_tbl[aux8_next[0]], grid_tbl[aux8_next[1]],
                          ksigns[(aux32_1_next >> 0) & 127],
                          ksigns[(aux32_1_next >> 7) & 127], kmask_vec16,
                          xq_row_next + 0, mag64 + 32, xsigned64 + 32);
      dequant_pack16_u8s8(grid_tbl[aux8_next[2]], grid_tbl[aux8_next[3]],
                          ksigns[(aux32_1_next >> 14) & 127],
                          ksigns[(aux32_1_next >> 21) & 127], kmask_vec16,
                          xq_row_next + 16, mag64 + 48, xsigned64 + 48);

      int32_t idot_lo, idot_hi;
      dot_u8s8_direct_avx512_x2(mag64, xsigned64, idot_lo, idot_hi);
      float xq_scale = fp16_to_fp32(xq[block_idx].d);
      float xq_scale_next = fp16_to_fp32(xq[block_idx_next].d);
      total += (double)idot_lo * (double)db * (double)xq_scale;
      total += (double)idot_hi * (double)db_next * (double)xq_scale_next;
    }
  }
  return (float)total;
}

// ============================================================
// EXPERIMENTAL AVX2 prototype #2 for IQ2_XXS -- a faithful port of
// upstream llama.cpp/ggml's own ggml_vec_dot_iq2_xxs_q8_K (vendored at
// build/_deps/ggml-src/src/ggml-cpu/arch/x86/quants.c), NOT this project's
// invention. Built after the first VNNI prototype above came out ~2.1x
// SLOWER than the existing FMA path -- checking what upstream's own
// production CPU backend does for this exact type turned up a genuinely
// different structural choice, worth measuring for real rather than
// assuming either "VNNI must be better" or "upstream must have already
// solved this."
//
// Two structural differences from the VNNI prototype above, both aimed at
// the exact weakness that sank it (packing overhead / memory round-trips):
//  1. NO VNNI. Uses the classic pre-VNNI two-instruction int8 dot:
//     _mm256_maddubs_epi16 (u8 x s8 -> s16 pairs) + _mm256_madd_epi16
//     (s16 x s16 -> s32 pairs) -- confirmed by reading upstream's own
//     source that this function has an __AVX2__ branch and an __AVX__
//     fallback but NO __AVX512VNNI__ branch at all for this specific type.
//  2. Weight packing is REGISTER-RESIDENT: _mm256_set_epi64x assembles the
//     4 grid-table lookups directly into a register in one instruction --
//     no store-to-memory-then-reload, unlike dequant_pack16_u8s8 above.
//  Per-32-subblock scale is folded as an INTEGER multiplier (2*ls+1) via
//  _mm256_madd_epi16 and accumulated in int32 across the WHOLE 256-element
//  superblock -- only ONE float scale multiply per superblock (not one per
//  32-group like both earlier prototypes in this file), matching upstream
//  exactly. The trailing 0.125f is the same constant baked differently
//  into dot_iq2_xxs_f32_avx512's own "(0.5+ls)*0.25" -- algebraically
//  identical: (2*ls+1)*0.125 == (0.5+ls)*0.25.
//
// Reuses THIS project's own grid_tbl/ksigns/kmask (already Test-7-verified
// byte-identical to ggml's reference dequant) rather than duplicating
// ggml's literal keven_signs_q2xs table (which is `static` in ggml's own
// x86 quants.c, not linkable) -- iq2xxs_sign_pattern_table() below derives
// the equivalent pre-expanded 8-byte-per-index sign pattern ONCE (lazy
// static init, same idiom as kmask_iq2xs_vec32_16_fused() above) from that
// already-correct source data.

static inline float hsum_f32_avx2(__m256 v) {
  __m128 lo = _mm256_castps256_ps128(v);
  __m128 hi = _mm256_extractf128_ps(v, 1);
  lo = _mm_add_ps(lo, hi);
  __m128 shuf = _mm_movehdup_ps(lo);
  __m128 sums = _mm_add_ps(lo, shuf);
  shuf = _mm_movehl_ps(shuf, sums);
  sums = _mm_add_ss(sums, shuf);
  return _mm_cvtss_f32(sums);
}

// Precomputed 128-entry x 8-byte sign-pattern table (0x01=keep, 0xFF=negate
// per lane), directly usable as _mm256_sign_epi8's second operand --
// equivalent to ggml's own `keven_signs_q2xs`, derived from this project's
// existing ksigns/kmask bytes instead of duplicating that literal table.
static const uint64_t *iq2xxs_sign_pattern_table() {
  static uint64_t tbl[128];
  static bool ready = [] {
    const uint8_t *kmask8 = (const uint8_t *)ms_grid_kmask_iq2xs();  // [8]
    const uint8_t *ksigns = (const uint8_t *)ms_grid_ksigns_iq2xs(); // [128]
    for (int idx = 0; idx < 128; ++idx) {
      uint8_t bytes[8];
      for (int j = 0; j < 8; ++j)
        bytes[j] = (ksigns[idx] & kmask8[j]) ? 0xFF : 0x01;
      std::memcpy(&tbl[idx], bytes, 8);
    }
    return true;
  }();
  (void)ready;
  return tbl;
}

static_assert(sizeof(block_q8_K_local) == 292,
              "block_q8_K_local layout must match ggml's block_q8_K exactly");

float dot_iq2_xxs_avx2_prototype(const uint8_t *block,
                                 const block_q8_K_local *y, int64_t k) {
  constexpr int QK_K = 256;
  constexpr size_t BLOCK_BYTES = 66;
  const int64_t nb = k / QK_K;

  const uint64_t *grid_tbl = (const uint64_t *)ms_grid_iq2xxs();
  const uint64_t *signs64 = iq2xxs_sign_pattern_table();

  __m256 accumf = _mm256_setzero_ps();
  for (int64_t i = 0; i < nb; ++i) {
    const uint8_t *base = block + (size_t)i * BLOCK_BYTES;
    uint16_t d_half;
    std::memcpy(&d_half, base, 2);
    const float d = fp16_to_fp32_fused(d_half) * y[i].d;
    const uint8_t *qs_bytes = base + 2;
    const int8_t *q8 = y[i].qs;

    __m256i sumi1 = _mm256_setzero_si256();
    __m256i sumi2 = _mm256_setzero_si256();
    for (int ib32 = 0; ib32 < QK_K / 32; ib32 += 2) {
      const __m256i q8_1 = _mm256_loadu_si256((const __m256i *)q8);
      q8 += 32;
      const __m256i q8_2 = _mm256_loadu_si256((const __m256i *)q8);
      q8 += 32;

      uint32_t aux32_0, aux32_1;
      std::memcpy(&aux32_0, qs_bytes + ib32 * 8 + 0, 4);
      std::memcpy(&aux32_1, qs_bytes + ib32 * 8 + 4, 4);
      const uint8_t *aux8 = (const uint8_t *)&aux32_0;
      int ib32_next = ib32 + 1;
      uint32_t aux32_0_next, aux32_1_next;
      std::memcpy(&aux32_0_next, qs_bytes + ib32_next * 8 + 0, 4);
      std::memcpy(&aux32_1_next, qs_bytes + ib32_next * 8 + 4, 4);
      const uint8_t *aux8_next = (const uint8_t *)&aux32_0_next;

      const __m256i q2_1 = _mm256_set_epi64x(
          (int64_t)grid_tbl[aux8[3]], (int64_t)grid_tbl[aux8[2]],
          (int64_t)grid_tbl[aux8[1]], (int64_t)grid_tbl[aux8[0]]);
      const __m256i q2_2 = _mm256_set_epi64x(
          (int64_t)grid_tbl[aux8_next[3]], (int64_t)grid_tbl[aux8_next[2]],
          (int64_t)grid_tbl[aux8_next[1]], (int64_t)grid_tbl[aux8_next[0]]);
      const __m256i s2_1 =
          _mm256_set_epi64x((int64_t)signs64[(aux32_1 >> 21) & 127],
                            (int64_t)signs64[(aux32_1 >> 14) & 127],
                            (int64_t)signs64[(aux32_1 >> 7) & 127],
                            (int64_t)signs64[(aux32_1 >> 0) & 127]);
      const __m256i s2_2 =
          _mm256_set_epi64x((int64_t)signs64[(aux32_1_next >> 21) & 127],
                            (int64_t)signs64[(aux32_1_next >> 14) & 127],
                            (int64_t)signs64[(aux32_1_next >> 7) & 127],
                            (int64_t)signs64[(aux32_1_next >> 0) & 127]);
      const __m256i q8s_1 = _mm256_sign_epi8(q8_1, s2_1);
      const __m256i q8s_2 = _mm256_sign_epi8(q8_2, s2_2);
      const __m256i dot1 = _mm256_maddubs_epi16(q2_1, q8s_1);
      const __m256i dot2 = _mm256_maddubs_epi16(q2_2, q8s_2);
      const uint16_t ls1 = (uint16_t)(aux32_1 >> 28);
      const uint16_t ls2 = (uint16_t)(aux32_1_next >> 28);
      const __m256i p1 =
          _mm256_madd_epi16(dot1, _mm256_set1_epi16((short)(2 * ls1 + 1)));
      const __m256i p2 =
          _mm256_madd_epi16(dot2, _mm256_set1_epi16((short)(2 * ls2 + 1)));
      sumi1 = _mm256_add_epi32(sumi1, p1);
      sumi2 = _mm256_add_epi32(sumi2, p2);
    }
    // Plain mul+add, not _mm256_fmadd_ps -- this file's build flags enable
    // AVX-512F (which covers 512-bit FMA, already used elsewhere in this
    // file) but not the separate legacy FMA3 flag the 256-bit FMA
    // intrinsic needs ("target specific option mismatch" at compile time).
    accumf = _mm256_add_ps(
        _mm256_mul_ps(_mm256_set1_ps(d),
                      _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1, sumi2))),
        accumf);
  }
  return 0.125f * hsum_f32_avx2(accumf);
}

// OpenMP wrapper for dot_iq2_xxs_avx2_prototype -- EngineConfig::
// cpu_dot_ggml_avx2's own matmul entry point (kquant_dequant.hpp's
// cpu_quant_matmul, gated behind cpu_dot_ggml_avx2_enabled()), mirroring
// matmul_iq2_xxs_fused's own structure exactly but quantizing activations
// to Q8_K ONCE per token (not per output row) before the parallel O-loop --
// same "plain static, not thread_local" reasoning as matmul_q8_0_fused's
// own g_xq_q8 buffer above (single decode-thread-only caller, fully
// written before the parallel region reads it).
void matmul_iq2_xxs_ggml_avx2(float *y, const float *x,
                              const uint8_t *W_quantized, int S, int I, int O) {
  constexpr int QK_K = 256;
  constexpr size_t BLOCK_BYTES = 66;
  const size_t row_bytes = (size_t)(I / QK_K) * BLOCK_BYTES;
  const int num_superblocks = I / QK_K;

  static std::vector<block_q8_K_local> g_xq_k;
  if (g_xq_k.size() < (size_t)(S * num_superblocks))
    g_xq_k.resize((size_t)S * num_superblocks);

  // Gated on S>1 -- an OMP team-spawn for a single S=1 (decode, the
  // dominant call volume for this dispatch path) serial iteration is pure
  // overhead with zero parallelism gained, the exact same "OMP-spawn cost
  // exceeds a tiny workload" regression this project has hit and fixed
  // multiple times before (see CLAUDE.md's qabsorb q_nope_gather_all entry
  // for the canonical writeup) -- copied uncritically from matmul_q8_0_
  // fused's own identical pattern above, which never got the same fix
  // because Q8_0 isn't a real routed-expert type for this model (never
  // exercised at S=1 in production, so the inefficiency was never
  // surfaced). Real hardware run caught this: wiring this function in
  // showed ZERO decode improvement over the existing FMA path despite a
  // genuine ~1.8x isolated per-call win -- this is the fix for that gap.
  if (S > 1) {
#pragma omp parallel for schedule(static)
    for (int s = 0; s < S; ++s) {
      quantize_row_q8_K_ref(x + (int64_t)s * I,
                            g_xq_k.data() + (int64_t)s * num_superblocks, I);
    }
  } else if (S == 1) {
    quantize_row_q8_K_ref(x, g_xq_k.data(), I);
  }

  for (int s = 0; s < S; ++s) {
    const block_q8_K_local *xq = g_xq_k.data() + (int64_t)s * num_superblocks;
    float *ys = y + (int64_t)s * O;
#pragma omp parallel for schedule(static)
    for (int o = 0; o < O; ++o) {
      const uint8_t *row_w = W_quantized + (size_t)o * row_bytes;
      if (o + 1 < O) {
        _mm_prefetch((const char *)(W_quantized + (size_t)(o + 1) * row_bytes),
                     _MM_HINT_T0);
      }
      ys[o] = dot_iq2_xxs_avx2_prototype(row_w, xq, I);
    }
  }
}
// ============================================================

// Llama.cpp/ggml's Qwen CPU route uses IQ1_M weights against one Q8_K-
// quantized activation row. MoEStream's old fallback instead dequantized
// every IQ1_M row to FP32 and then ran an AVX-512 FP32 dot, leaving the
// integer unpack/lookup work in ggml_dequant_row. Reuse ggml's production
// entry point here. The current GGML IQ1_M implementation is AVX2-width even
// in this build, which has AVX-512 and AVX-512 VNNI enabled globally: its
// IQ1_M branch is guarded by __AVX2__, not by an AVX-512/VNNI specialization.
bool matmul_iq1_m_ggml_q8k(float *y, const float *x, const uint8_t *W_quantized,
                           int S, int I, int O) {
#if defined(MOESTREAM_USE_GGML)
  constexpr int QK_K = 256;
  if (!y || !x || !W_quantized || S <= 0 || I <= 0 || O <= 0 || I % QK_K != 0) {
    return false;
  }
  ensure_ggml_cpu_initialized();
  const int nblocks = I / QK_K;
  const size_t row_bytes =
      static_cast<size_t>(nblocks) * sizeof(moestream_ggml_block_iq1_m);
  // Decode invokes this wrapper repeatedly for adjacent Qwen layers. Reuse
  // the per-caller Q8 scratch instead of allocating a fresh vector for every
  // projection; the buffer is thread-local because callers may use separate
  // decode/request threads, while each call itself remains synchronous.
  static thread_local std::vector<moestream_ggml_block_q8_K> xq;
  xq.resize(static_cast<size_t>(S) * nblocks);
  for (int s = 0; s < S; ++s) {
    quantize_row_q8_K(x + static_cast<int64_t>(s) * I,
                      xq.data() + static_cast<size_t>(s) * nblocks, I);
  }
  const moestream_ggml_block_q8_K *xq_data = xq.data();

#pragma omp parallel for num_threads(qwen_cpu_grouped_threads()) schedule(static)
  for (int o = 0; o < O; ++o) {
    const uint8_t *wrow = W_quantized + static_cast<size_t>(o) * row_bytes;
    for (int s = 0; s < S; ++s) {
      ggml_vec_dot_iq1_m_q8_K(I, y + static_cast<int64_t>(s) * O + o, 0, wrow,
                              0, xq_data + static_cast<size_t>(s) * nblocks,
                              sizeof(moestream_ggml_block_q8_K), 1);
    }
  }
  return true;
#else
  (void)y;
  (void)x;
  (void)W_quantized;
  (void)S;
  (void)I;
  (void)O;
  return false;
#endif
}

#if defined(MOESTREAM_USE_GGML) && defined(__AVX512F__) && \
    defined(__AVX512BW__)
namespace {

static inline __m512i concat_i256(__m256i lo, __m256i hi) {
  return _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1);
}

// Exact AVX-512-width port of ggml_vec_dot_iq1_m_q8_K.  The current GGML
// implementation is AVX2-width even when GGML_AVX512 and GGML_AVX512_VNNI
// are enabled, because its IQ1_M branch is guarded only by __AVX2__.  Keep
// this separate from the production GGML adapter until the numerical and
// hardware A/B checks below are complete.
static float dot_iq1_m_q8k_avx512(const uint8_t *weight,
                                  const moestream_ggml_block_q8_K *input,
                                  int n) {
  constexpr int QK_K = 256;
  constexpr float IQ1M_DELTA = 0.125f;
  if (!weight || !input || n <= 0 || n % QK_K != 0)
    return 0.0f;

  const auto *x = reinterpret_cast<const moestream_ggml_block_iq1_m *>(weight);
  const auto *y = input;
  const int nb = n / QK_K;
  const auto *grid = static_cast<const uint64_t *>(ms_grid_iq1s());
  if (!grid)
    return 0.0f;

  float sum = 0.0f;
  for (int block = 0; block < nb; ++block) {
    const uint8_t *qs = x[block].qs;
    const uint8_t *qh = x[block].qh;
    const uint16_t *sc = reinterpret_cast<const uint16_t *>(x[block].scales);
    const uint16_t scale_u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) |
                               ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);

    __m512 accum = _mm512_setzero_ps();
    __m512 accum_delta = _mm512_setzero_ps();
    const int8_t *q8 = y[block].qs;
    const __m512i one8 = _mm512_set1_epi8(1);
    const __m256i two8 = _mm256_set1_epi8(2);

    // One iteration consumes two 32-byte IQ1 grid groups, exactly matching
    // GGML's AVX2 loop.  Widening the same arithmetic to 512 bits preserves
    // its integer rounding and sign behavior.
    __m256i scales = _mm256_set1_epi64x(*reinterpret_cast<const int64_t *>(sc));
    const __m256i scale_shifts = _mm256_set_epi64x(9, 3, 6, 0);
    scales = _mm256_srlv_epi64(scales, scale_shifts);
    scales = _mm256_add_epi16(_mm256_slli_epi16(_mm256_and_si256(scales, _mm256_set1_epi16(0x7)), 1),
                              _mm256_set1_epi16(1));
    __m256i scales_idx1 = _mm256_set1_epi16(0x0100);
    __m256i scales_idx2 = _mm256_add_epi8(scales_idx1, _mm256_set1_epi8(8));

    for (int ib = 0; ib < QK_K / 32; ib += 2) {
      __m256i q1b_1;
      __m256i q1b_2;
#ifdef __BMI2__
      const uint64_t packed_idx1 =
          _pdep_u64(*reinterpret_cast<const uint32_t *>(qs),
                    0x00ff00ff00ff00ffULL) |
          _pdep_u64(*reinterpret_cast<const uint16_t *>(qh) & 0x7777,
                    0xf000f000f000f00ULL);
      const uint64_t packed_idx2 =
          _pdep_u64(*reinterpret_cast<const uint32_t *>(qs + 4),
                    0x00ff00ff00ff00ffULL) |
          _pdep_u64(*reinterpret_cast<const uint16_t *>(qh + 2) & 0x7777,
                    0xf000f000f000f00ULL);
      const uint16_t *idx1 = reinterpret_cast<const uint16_t *>(&packed_idx1);
      const uint16_t *idx2 = reinterpret_cast<const uint16_t *>(&packed_idx2);
      q1b_1 = _mm256_set_epi64x(grid[idx1[3]], grid[idx1[2]], grid[idx1[1]],
                                grid[idx1[0]]);
      q1b_2 = _mm256_set_epi64x(grid[idx2[3]], grid[idx2[2]], grid[idx2[1]],
                                grid[idx2[0]]);
#else
      q1b_1 = _mm256_set_epi64x(
          grid[qs[3] | (((uint16_t)qh[1] << 4) & 0x700)],
          grid[qs[2] | (((uint16_t)qh[1] << 8) & 0x700)],
          grid[qs[1] | (((uint16_t)qh[0] << 4) & 0x700)],
          grid[qs[0] | (((uint16_t)qh[0] << 8) & 0x700)]);
      q1b_2 = _mm256_set_epi64x(
          grid[qs[7] | (((uint16_t)qh[3] << 4) & 0x700)],
          grid[qs[6] | (((uint16_t)qh[3] << 8) & 0x700)],
          grid[qs[5] | (((uint16_t)qh[2] << 4) & 0x700)],
          grid[qs[4] | (((uint16_t)qh[2] << 8) & 0x700)]);
#endif
      const __m256i delta1 = _mm256_set_epi64x(
          qh[1] & 0x80 ? -1LL : 0x0101010101010101LL,
          qh[1] & 0x08 ? -1LL : 0x0101010101010101LL,
          qh[0] & 0x80 ? -1LL : 0x0101010101010101LL,
          qh[0] & 0x08 ? -1LL : 0x0101010101010101LL);
      const __m256i delta2 = _mm256_set_epi64x(
          qh[3] & 0x80 ? -1LL : 0x0101010101010101LL,
          qh[3] & 0x08 ? -1LL : 0x0101010101010101LL,
          qh[2] & 0x80 ? -1LL : 0x0101010101010101LL,
          qh[2] & 0x08 ? -1LL : 0x0101010101010101LL);
      const __m256i q8b_1 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(q8));
      const __m256i q8b_2 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(q8 + 32));

      const __m512i q1 = concat_i256(q1b_1, q1b_2);
      const __m512i q8v = concat_i256(q8b_1, q8b_2);
      const __m512i dsign = concat_i256(delta1, delta2);
      const __m512i zero = _mm512_setzero_si512();
      const __mmask64 q1_neg = _mm512_cmp_epi8_mask(q1, zero, _MM_CMPINT_LT);
      const __mmask64 q1_zero = _mm512_cmpeq_epi8_mask(q1, zero);
      const __mmask64 delta_neg =
          _mm512_cmp_epi8_mask(dsign, zero, _MM_CMPINT_LT);
      const __m512i q1_abs = _mm512_abs_epi8(q1);
      __m512i q1_q8 = _mm512_mask_sub_epi8(q8v, q1_neg, zero, q8v);
      q1_q8 = _mm512_mask_mov_epi8(q1_q8, q1_zero, zero);
      const __m512i d_q8 = _mm512_mask_sub_epi8(q8v, delta_neg, zero, q8v);
      const __m512i dot = _mm512_maddubs_epi16(q1_abs, q1_q8);
      const __m512i dot_delta = _mm512_maddubs_epi16(one8, d_q8);

      const __m256i scale1 = _mm256_shuffle_epi8(scales, scales_idx1);
      const __m256i scale2 = _mm256_shuffle_epi8(scales, scales_idx2);
      const __m512i scale_vec = concat_i256(scale1, scale2);
      const __m512i p = _mm512_madd_epi16(dot, scale_vec);
      const __m512i p_delta = _mm512_madd_epi16(dot_delta, scale_vec);

      const __m512 d = _mm512_set1_ps(y[block].d * fp16_to_fp32(scale_u16));
      accum = _mm512_fmadd_ps(d, _mm512_cvtepi32_ps(p), accum);
      accum_delta = _mm512_fmadd_ps(d, _mm512_cvtepi32_ps(p_delta), accum_delta);

      qs += 8;
      qh += 4;
      q8 += 64;
      scales_idx1 = _mm256_add_epi8(scales_idx1, two8);
      scales_idx2 = _mm256_add_epi8(scales_idx2, two8);
    }
    sum += _mm512_reduce_add_ps(accum) +
           IQ1M_DELTA * _mm512_reduce_add_ps(accum_delta);
  }
  return sum;
}

} // namespace
#endif

bool matmul_iq1_m_avx512_q8k_grouped(
    float *const *y, const float *x, const uint8_t *const *W_quantized,
    int n_matrices, int S, int I, int O) {
#if defined(MOESTREAM_USE_GGML) && defined(__AVX512F__) && defined(__AVX512BW__)
  constexpr int QK_K = 256;
  if (!y || !x || !W_quantized || n_matrices <= 0 || S <= 0 || I <= 0 ||
      O <= 0 || I % QK_K != 0)
    return false;
  const int nblocks = I / QK_K;
  const size_t row_bytes = static_cast<size_t>(nblocks) * sizeof(moestream_ggml_block_iq1_m);
  static thread_local std::vector<moestream_ggml_block_q8_K> xq;
  xq.resize(static_cast<size_t>(S) * nblocks);
  for (int s = 0; s < S; ++s)
    quantize_row_q8_K(x + static_cast<int64_t>(s) * I,
                      xq.data() + static_cast<size_t>(s) * nblocks, I);
  const moestream_ggml_block_q8_K *xq_data = xq.data();

#pragma omp parallel for num_threads(qwen_cpu_grouped_threads()) schedule(static)
  for (int64_t flat = 0; flat < static_cast<int64_t>(n_matrices) * S * O; ++flat) {
    const int o = static_cast<int>(flat % O);
    const int64_t matrix_row = flat / O;
    const int s = static_cast<int>(matrix_row % S);
    const int m = static_cast<int>(matrix_row / S);
    const uint8_t *row = W_quantized[m] + static_cast<size_t>(o) * row_bytes;
    y[m][static_cast<int64_t>(s) * O + o] = dot_iq1_m_q8k_avx512(
        row, xq_data + static_cast<size_t>(s) * nblocks, I);
  }
  return true;
#else
  (void)y; (void)x; (void)W_quantized; (void)n_matrices; (void)S; (void)I; (void)O;
  return false;
#endif
}

namespace {

bool matmul_iq1_m_ggml_q8k_grouped_impl(float *const *y, const float *const *x,
                                        const uint8_t *const *W_quantized,
                                        int n_matrices, int S, int I, int O) {
#if defined(MOESTREAM_USE_GGML)
  constexpr int QK_K = 256;
  if (!y || !x || !W_quantized || n_matrices <= 0 || S <= 0 || I <= 0 ||
      O <= 0 || I % QK_K != 0)
    return false;
  ensure_ggml_cpu_initialized();
  const int nblocks = I / QK_K;
  const size_t row_bytes =
      static_cast<size_t>(nblocks) * sizeof(moestream_ggml_block_iq1_m);
  // Llama.cpp's MUL_MAT_ID path quantizes a shared activation once for the
  // whole expert group.  The ordinary grouped-input API also supports a
  // different activation per matrix, so retain that form only when callers
  // actually provide distinct pointers.
  bool shared_input = true;
  for (int m = 1; m < n_matrices; ++m)
    shared_input = shared_input && (x[m] == x[0]);
  const size_t xq_rows = static_cast<size_t>(shared_input ? S : n_matrices * S);
  static thread_local std::vector<moestream_ggml_block_q8_K> xq;
  xq.resize(xq_rows * nblocks);
  for (int m = 0; m < (shared_input ? 1 : n_matrices); ++m) {
    for (int s = 0; s < S; ++s) {
      quantize_row_q8_K(
          x[m] + static_cast<int64_t>(s) * I,
          xq.data() +
              (static_cast<size_t>(shared_input ? 0 : m) * S + s) * nblocks,
          I);
    }
  }
  const moestream_ggml_block_q8_K *xq_data = xq.data();

  const auto t0 = std::chrono::high_resolution_clock::now();
  const int64_t total = static_cast<int64_t>(n_matrices) * S * O;
#pragma omp parallel for num_threads(qwen_cpu_grouped_threads()) schedule(static)
  for (int64_t flat = 0; flat < total; ++flat) {
    const int o = static_cast<int>(flat % O);
    const int64_t matrix_row = flat / O;
    const int s = static_cast<int>(matrix_row % S);
    const int m = static_cast<int>(matrix_row / S);
    const uint8_t *wrow = W_quantized[m] + static_cast<size_t>(o) * row_bytes;
    y[m][static_cast<int64_t>(s) * O + o] = 0.0f;
    ggml_vec_dot_iq1_m_q8_K(
        I, y[m] + static_cast<int64_t>(s) * O + o, 0, wrow, 0,
        xq_data +
            (static_cast<size_t>(shared_input ? 0 : m) * S + s) * nblocks,
        sizeof(moestream_ggml_block_q8_K), 1);
  }
  const auto t1 = std::chrono::high_resolution_clock::now();
  PerfCounters::atomic_add(perf().t_cpu_dot,
                           std::chrono::duration<double>(t1 - t0).count());
  return true;
#else
  (void)y;
  (void)x;
  (void)W_quantized;
  (void)n_matrices;
  (void)S;
  (void)I;
  (void)O;
  return false;
#endif
}

} // namespace

bool matmul_iq1_m_ggml_q8k_grouped(float *const *y, const float *x,
                                   const uint8_t *const *W_quantized,
                                   int n_matrices, int S, int I, int O) {
  if (!x)
    return false;
  std::vector<const float *> inputs(static_cast<size_t>(n_matrices), x);
  return matmul_iq1_m_ggml_q8k_grouped_impl(y, inputs.data(), W_quantized,
                                            n_matrices, S, I, O);
}

bool matmul_iq1_m_ggml_q8k_grouped_pair(
    float *const *gate_y, float *const *up_y, const float *x,
    const uint8_t *const *gate_weights, const uint8_t *const *up_weights,
    int n_matrices, int S, int I, int O) {
#if defined(MOESTREAM_USE_GGML)
  constexpr int QK_K = 256;
  if (!gate_y || !up_y || !x || !gate_weights || !up_weights ||
      n_matrices <= 0 || S <= 0 || I <= 0 || O <= 0 || I % QK_K != 0)
    return false;
  ensure_ggml_cpu_initialized();
  const int nblocks = I / QK_K;
  const size_t row_bytes =
      static_cast<size_t>(nblocks) * sizeof(moestream_ggml_block_iq1_m);
  static thread_local std::vector<moestream_ggml_block_q8_K> xq;
  xq.resize(static_cast<size_t>(S) * nblocks);
  for (int s = 0; s < S; ++s)
    quantize_row_q8_K(x + static_cast<int64_t>(s) * I,
                      xq.data() + static_cast<size_t>(s) * nblocks, I);
  // Capture the caller thread's TLS buffer before entering OpenMP. Referring
  // to xq.data() inside the loop would resolve xq separately on each worker.
  const moestream_ggml_block_q8_K *xq_data = xq.data();

  const auto t0 = std::chrono::high_resolution_clock::now();
  const int64_t total = static_cast<int64_t>(n_matrices) * S * O;
#pragma omp parallel for num_threads(qwen_cpu_grouped_threads()) schedule(static)
  for (int64_t flat = 0; flat < total; ++flat) {
    const int o = static_cast<int>(flat % O);
    const int64_t matrix_row = flat / O;
    const int s = static_cast<int>(matrix_row % S);
    const int m = static_cast<int>(matrix_row / S);
    const auto *xrow = xq_data + static_cast<size_t>(s) * nblocks;
    float *gate_out = gate_y[m] + static_cast<int64_t>(s) * O + o;
    float *up_out = up_y[m] + static_cast<int64_t>(s) * O + o;
    const uint8_t *gate_row =
        gate_weights[m] + static_cast<size_t>(o) * row_bytes;
    const uint8_t *up_row =
        up_weights[m] + static_cast<size_t>(o) * row_bytes;
    *gate_out = 0.0f;
    *up_out = 0.0f;
    ggml_vec_dot_iq1_m_q8_K(I, gate_out, 0, gate_row, 0, xrow,
                            sizeof(moestream_ggml_block_q8_K), 1);
    ggml_vec_dot_iq1_m_q8_K(I, up_out, 0, up_row, 0, xrow,
                            sizeof(moestream_ggml_block_q8_K), 1);
  }
  const auto t1 = std::chrono::high_resolution_clock::now();
  PerfCounters::atomic_add(perf().t_cpu_dot,
                           std::chrono::duration<double>(t1 - t0).count());
  return true;
#else
  (void)gate_y;
  (void)up_y;
  (void)x;
  (void)gate_weights;
  (void)up_weights;
  (void)n_matrices;
  (void)S;
  (void)I;
  (void)O;
  return false;
#endif
}

bool matmul_iq1_m_ggml_q8k_grouped_inputs(float *const *y,
                                          const float *const *x,
                                          const uint8_t *const *W_quantized,
                                          int n_matrices, int S, int I, int O) {
  return matmul_iq1_m_ggml_q8k_grouped_impl(y, x, W_quantized, n_matrices, S, I,
                                            O);
}

// Direct fused kernel: computes dot product of IQ3_XXS block weights with FP32
// activation vector x
float dot_iq3_xxs_f32_avx512(const uint8_t *block, const float *x, int64_t k) {
  constexpr int QK_K = 256;
  constexpr size_t BLOCK_BYTES = 98; // d(2) + qs[96]
  const int64_t nb = k / QK_K;

  const uint32_t *grid_tbl =
      (const uint32_t *)ms_grid_iq3xxs(); // 4-byte entries
  const uint8_t *ksigns = (const uint8_t *)ms_grid_ksigns_iq2xs();
  __m512i kmask_vec16 = kmask_iq2xs_vec32_16_fused();

  __m512 acc0 = _mm512_setzero_ps();
  __m512 acc1 = _mm512_setzero_ps();

  for (int64_t i = 0; i < nb; ++i) {
    const uint8_t *base = block + (size_t)i * BLOCK_BYTES;
    uint16_t d_half;
    std::memcpy(&d_half, base, 2);
    const float d = fp16_to_fp32_fused(d_half);

    const uint8_t *qs = base + 2;              // [96]
    const uint8_t *scales_and_signs = qs + 64; // [32]
    const float *x_block = x + (size_t)i * QK_K;

    const uint8_t *qptr = qs;
    for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
      const uint8_t *sas = scales_and_signs + 4 * ib32;
      uint32_t aux32 = (uint32_t)sas[0] | ((uint32_t)sas[1] << 8) |
                       ((uint32_t)sas[2] << 16) | ((uint32_t)sas[3] << 24);
      const float db = d * (0.5f + (float)(aux32 >> 28)) * 0.5f;

      // Process two 8-element sub-groups at a time (16 weights per iteration)
      for (int pair = 0; pair < 2; ++pair) {
        int l0 = pair * 2, l1 = pair * 2 + 1;

        uint32_t g0a = grid_tbl[qptr[2 * l0 + 0]],
                 g0b = grid_tbl[qptr[2 * l0 + 1]];
        uint32_t g1a = grid_tbl[qptr[2 * l1 + 0]],
                 g1b = grid_tbl[qptr[2 * l1 + 1]];

        uint64_t g0 = (uint64_t)g0a | ((uint64_t)g0b << 32);
        uint64_t g1 = (uint64_t)g1a | ((uint64_t)g1b << 32);

        uint8_t signs0 = ksigns[(aux32 >> (7 * l0)) & 127];
        uint8_t signs1 = ksigns[(aux32 >> (7 * l1)) & 127];

        dequant_and_fmadd_16(g0, g1, signs0, signs1, kmask_vec16, db,
                             x_block + ib32 * 32 + l0 * 8,
                             (pair == 0 ? acc0 : acc1));
      }
      qptr += 8;
    }
  }

  return _mm512_reduce_add_ps(_mm512_add_ps(acc0, acc1));
}

// OpenMP wrapper bypassing intermediate allocation
void matmul_iq3_xxs_fused(float *y, const float *x, const uint8_t *W_quantized,
                          int S, int I, int O) {
  constexpr int QK_K = 256;
  constexpr size_t BLOCK_BYTES = 98;
  const size_t row_bytes = (size_t)(I / QK_K) * BLOCK_BYTES;

  for (int s = 0; s < S; ++s) {
    const float *xs = x + (int64_t)s * I;
    float *ys = y + (int64_t)s * O;

    // 2026-08-29: same S>1 gate TRIED and REVERTED here as matmul_iq2_xxs_
    // fused's own sibling above -- see that function's comment for the
    // full rationale and real-hardware measurement. Left as the original,
    // always-parallel form.
#pragma omp parallel for schedule(static)
    for (int o = 0; o < O; ++o) {
      const uint8_t *row_w = W_quantized + (size_t)o * row_bytes;
      if (o + 1 < O) {
        _mm_prefetch((const char *)(W_quantized + (size_t)(o + 1) * row_bytes),
                     _MM_HINT_T0);
      }
      ys[o] = dot_iq3_xxs_f32_avx512(row_w, xs, I);
    }
  }
}

// ============================================================
// EXPERIMENTAL AVX2 prototype for IQ3_XXS -- faithful port of upstream
// llama.cpp/ggml's own ggml_vec_dot_iq3_xxs_q8_K (vendored at build/_deps/
// ggml-src/src/ggml-cpu/arch/x86/quants.c), same rationale and design as
// dot_iq2_xxs_avx2_prototype above (see that function's own header comment
// for the full story -- this is its IQ3_XXS sibling, GLM-5.2's down-proj
// type, 71-76/78 layers). Reuses iq2xxs_sign_pattern_table() unchanged --
// confirmed via source that dot_iq3_xxs_f32_avx512 above already shares the
// SAME ksigns/kmask table with IQ2_XXS (ms_grid_ksigns_iq2xs()), and ggml's
// own kernel likewise shares its single keven_signs_q2xs table across both
// types -- no new sign table needed.
//
// Structural difference from IQ2_XXS's port: IQ3_XXS's grid indices are
// flat bytes in their own 64-byte region (x[i].qs, 4-weight/32-bit grid
// entries, 8 bytes -> 8 entries -> 32 weights per iteration) rather than
// packed into the same uint32 as the sign/scale data -- the sign/scale
// aux32 words live in a SEPARATE 32-byte region right after (`gas`/
// `scales_and_signs`). Confirmed byte-for-byte against dot_iq3_xxs_f32_
// avx512 above (same BLOCK_BYTES=98, same qs/scales_and_signs split, same
// aux32 bit layout) before writing this, not re-derived from scratch.
float dot_iq3_xxs_avx2_prototype(const uint8_t *block,
                                 const block_q8_K_local *y, int64_t k) {
  constexpr int QK_K = 256;
  constexpr size_t BLOCK_BYTES = 98;
  const int64_t nb = k / QK_K;

  const uint32_t *grid_tbl = (const uint32_t *)ms_grid_iq3xxs();
  const uint64_t *signs64 = iq2xxs_sign_pattern_table();

  __m256 accumf = _mm256_setzero_ps();
  for (int64_t i = 0; i < nb; ++i) {
    const uint8_t *base = block + (size_t)i * BLOCK_BYTES;
    uint16_t d_half;
    std::memcpy(&d_half, base, 2);
    const float d = fp16_to_fp32_fused(d_half) * y[i].d;
    const uint8_t *q3 = base + 2;       // [64] grid indices
    const uint8_t *gas = base + 2 + 64; // [32] sign/scale aux32 words
    const int8_t *q8 = y[i].qs;

    __m256i sumi1 = _mm256_setzero_si256();
    __m256i sumi2 = _mm256_setzero_si256();
    for (int ib32 = 0; ib32 < QK_K / 32; ib32 += 2) {
      const __m256i q8_1 = _mm256_loadu_si256((const __m256i *)q8);
      q8 += 32;
      const __m256i q8_2 = _mm256_loadu_si256((const __m256i *)q8);
      q8 += 32;

      const __m256i q2_1 = _mm256_set_epi32(
          (int)grid_tbl[q3[7]], (int)grid_tbl[q3[6]], (int)grid_tbl[q3[5]],
          (int)grid_tbl[q3[4]], (int)grid_tbl[q3[3]], (int)grid_tbl[q3[2]],
          (int)grid_tbl[q3[1]], (int)grid_tbl[q3[0]]);
      q3 += 8;
      const __m256i q2_2 = _mm256_set_epi32(
          (int)grid_tbl[q3[7]], (int)grid_tbl[q3[6]], (int)grid_tbl[q3[5]],
          (int)grid_tbl[q3[4]], (int)grid_tbl[q3[3]], (int)grid_tbl[q3[2]],
          (int)grid_tbl[q3[1]], (int)grid_tbl[q3[0]]);
      q3 += 8;

      uint32_t aux32_0, aux32_1;
      std::memcpy(&aux32_0, gas, 4);
      std::memcpy(&aux32_1, gas + 4, 4);
      gas += 8;

      const __m256i s2_1 =
          _mm256_set_epi64x((int64_t)signs64[(aux32_0 >> 21) & 127],
                            (int64_t)signs64[(aux32_0 >> 14) & 127],
                            (int64_t)signs64[(aux32_0 >> 7) & 127],
                            (int64_t)signs64[(aux32_0 >> 0) & 127]);
      const __m256i s2_2 =
          _mm256_set_epi64x((int64_t)signs64[(aux32_1 >> 21) & 127],
                            (int64_t)signs64[(aux32_1 >> 14) & 127],
                            (int64_t)signs64[(aux32_1 >> 7) & 127],
                            (int64_t)signs64[(aux32_1 >> 0) & 127]);
      const __m256i q8s_1 = _mm256_sign_epi8(q8_1, s2_1);
      const __m256i q8s_2 = _mm256_sign_epi8(q8_2, s2_2);
      const __m256i dot1 = _mm256_maddubs_epi16(q2_1, q8s_1);
      const __m256i dot2 = _mm256_maddubs_epi16(q2_2, q8s_2);
      const uint16_t ls1 = (uint16_t)(aux32_0 >> 28);
      const uint16_t ls2 = (uint16_t)(aux32_1 >> 28);
      const __m256i p1 =
          _mm256_madd_epi16(dot1, _mm256_set1_epi16((short)(2 * ls1 + 1)));
      const __m256i p2 =
          _mm256_madd_epi16(dot2, _mm256_set1_epi16((short)(2 * ls2 + 1)));
      sumi1 = _mm256_add_epi32(sumi1, p1);
      sumi2 = _mm256_add_epi32(sumi2, p2);
    }
    accumf = _mm256_add_ps(
        _mm256_mul_ps(_mm256_set1_ps(d),
                      _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1, sumi2))),
        accumf);
  }
  return 0.25f * hsum_f32_avx2(accumf);
}

// OpenMP wrapper for dot_iq3_xxs_avx2_prototype -- IQ3_XXS sibling of
// matmul_iq2_xxs_ggml_avx2 above, same design (see that function's own
// comment).
void matmul_iq3_xxs_ggml_avx2(float *y, const float *x,
                              const uint8_t *W_quantized, int S, int I, int O) {
  constexpr int QK_K = 256;
  constexpr size_t BLOCK_BYTES = 98;
  const size_t row_bytes = (size_t)(I / QK_K) * BLOCK_BYTES;
  const int num_superblocks = I / QK_K;

  static std::vector<block_q8_K_local> g_xq_k;
  if (g_xq_k.size() < (size_t)(S * num_superblocks))
    g_xq_k.resize((size_t)S * num_superblocks);

  // Gated on S>1 -- an OMP team-spawn for a single S=1 (decode, the
  // dominant call volume for this dispatch path) serial iteration is pure
  // overhead with zero parallelism gained, the exact same "OMP-spawn cost
  // exceeds a tiny workload" regression this project has hit and fixed
  // multiple times before (see CLAUDE.md's qabsorb q_nope_gather_all entry
  // for the canonical writeup) -- copied uncritically from matmul_q8_0_
  // fused's own identical pattern above, which never got the same fix
  // because Q8_0 isn't a real routed-expert type for this model (never
  // exercised at S=1 in production, so the inefficiency was never
  // surfaced). Real hardware run caught this: wiring this function in
  // showed ZERO decode improvement over the existing FMA path despite a
  // genuine ~1.8x isolated per-call win -- this is the fix for that gap.
  if (S > 1) {
#pragma omp parallel for schedule(static)
    for (int s = 0; s < S; ++s) {
      quantize_row_q8_K_ref(x + (int64_t)s * I,
                            g_xq_k.data() + (int64_t)s * num_superblocks, I);
    }
  } else if (S == 1) {
    quantize_row_q8_K_ref(x, g_xq_k.data(), I);
  }

  for (int s = 0; s < S; ++s) {
    const block_q8_K_local *xq = g_xq_k.data() + (int64_t)s * num_superblocks;
    float *ys = y + (int64_t)s * O;
#pragma omp parallel for schedule(static)
    for (int o = 0; o < O; ++o) {
      const uint8_t *row_w = W_quantized + (size_t)o * row_bytes;
      if (o + 1 < O) {
        _mm_prefetch((const char *)(W_quantized + (size_t)(o + 1) * row_bytes),
                     _MM_HINT_T0);
      }
      ys[o] = dot_iq3_xxs_avx2_prototype(row_w, xq, I);
    }
  }
}

namespace {

using q8k_dot_fn = float (*)(const uint8_t *, const block_q8_K_local *,
                             int64_t);

#if defined(MOESTREAM_USE_GGML)
// The linked GGML CPU backend exports the same production dot functions that
// its MUL_MAT_ID implementation calls.  Keep the small adapter here so the
// grouped Qwen path can share the existing matrix loop without changing the
// older, experimental AVX2 prototype entry points used by other callers.
static float dot_iq2_xxs_ggml_q8k_exact(const uint8_t *w,
                                        const block_q8_K_local *xq, int64_t n) {
  float out = 0.0f;
  ggml_vec_dot_iq2_xxs_q8_K((int)n, &out, 0, w, 0, xq, 0, 1);
  return out;
}

static float dot_iq3_xxs_ggml_q8k_exact(const uint8_t *w,
                                        const block_q8_K_local *xq, int64_t n) {
  float out = 0.0f;
  ggml_vec_dot_iq3_xxs_q8_K((int)n, &out, 0, w, 0, xq, 0, 1);
  return out;
}
#endif

bool matmul_iqxxs_ggml_avx2_grouped_impl(float *const *y, const float *x,
                                         const uint8_t *const *W_quantized,
                                         int n_matrices, int S, int I, int O,
                                         size_t block_bytes, q8k_dot_fn dot) {
#if defined(MOESTREAM_USE_GGML)
  constexpr int QK_K = 256;
  if (!y || !x || !W_quantized || !dot || n_matrices <= 0 || S <= 0 || I <= 0 ||
      O <= 0 || I % QK_K != 0)
    return false;
  ensure_ggml_cpu_initialized();
  const int nblocks = I / QK_K;
  const size_t row_bytes = static_cast<size_t>(nblocks) * block_bytes;
  std::vector<block_q8_K_local> xq(static_cast<size_t>(S) * nblocks);
  const auto t0 = std::chrono::high_resolution_clock::now();
  for (int s = 0; s < S; ++s)
    // The local mirror has the same ABI/layout as GGML's block_q8_K.  Use
    // GGML's production quantizer here, as the IQ1_M Q8_K path already does;
    // retain the ordinary vector lifetime because the grouped IQ tests also
    // exercise this helper from nested callers.
    quantize_row_q8_K(x + static_cast<int64_t>(s) * I,
                      xq.data() + static_cast<size_t>(s) * nblocks, I);

  const int64_t total = static_cast<int64_t>(n_matrices) * S * O;
#pragma omp parallel for num_threads(qwen_cpu_grouped_threads()) schedule(static)
  for (int64_t flat = 0; flat < total; ++flat) {
    const int o = static_cast<int>(flat % O);
    const int64_t matrix_row = flat / O;
    const int s = static_cast<int>(matrix_row % S);
    const int m = static_cast<int>(matrix_row / S);
    const uint8_t *row = W_quantized[m] + static_cast<size_t>(o) * row_bytes;
    y[m][static_cast<int64_t>(s) * O + o] =
        dot(row, xq.data() + static_cast<size_t>(s) * nblocks, I);
  }
  const auto t1 = std::chrono::high_resolution_clock::now();
  PerfCounters::atomic_add(perf().t_cpu_dot,
                           std::chrono::duration<double>(t1 - t0).count());
  return true;
#else
  (void)y;
  (void)x;
  (void)W_quantized;
  (void)n_matrices;
  (void)S;
  (void)I;
  (void)O;
  (void)block_bytes;
  (void)dot;
  return false;
#endif
}

bool matmul_iq4_nl_ggml_q8_0_grouped_impl(float *const *y,
                                          const float *const *x,
                                          const uint8_t *const *W_quantized,
                                          int n_matrices, int S, int I, int O) {
#if defined(MOESTREAM_USE_GGML)
  constexpr int QK4_NL = 32;
  if (!y || !x || !W_quantized || n_matrices <= 0 || S <= 0 || I <= 0 ||
      O <= 0 || I % QK4_NL != 0)
    return false;
  ensure_ggml_cpu_initialized();
  const int nblocks = I / QK4_NL;
  const size_t row_bytes = static_cast<size_t>(nblocks) * 18;
  static thread_local std::vector<block_q8_0> xq;
  xq.resize(static_cast<size_t>(n_matrices) * S * nblocks);
  const auto t0 = std::chrono::high_resolution_clock::now();
  for (int m = 0; m < n_matrices; ++m) {
    if (!x[m])
      return false;
    for (int s = 0; s < S; ++s)
      quantize_row_q8_0(x[m] + static_cast<int64_t>(s) * I,
                        xq.data() + (static_cast<size_t>(m) * S + s) * nblocks,
                        I);
  }
  const block_q8_0 *xq_data = xq.data();

  const int64_t total = static_cast<int64_t>(n_matrices) * S * O;
#pragma omp parallel for num_threads(qwen_cpu_grouped_threads()) schedule(static)
  for (int64_t flat = 0; flat < total; ++flat) {
    const int o = static_cast<int>(flat % O);
    const int64_t matrix_row = flat / O;
    const int s = static_cast<int>(matrix_row % S);
    const int m = static_cast<int>(matrix_row / S);
    const uint8_t *row = W_quantized[m] + static_cast<size_t>(o) * row_bytes;
    ggml_vec_dot_iq4_nl_q8_0(
        I, y[m] + static_cast<int64_t>(s) * O + o, 0, row, 0,
        xq_data + (static_cast<size_t>(m) * S + s) * nblocks,
        sizeof(block_q8_0), 1);
  }
  const auto t1 = std::chrono::high_resolution_clock::now();
  PerfCounters::atomic_add(perf().t_cpu_dot,
                           std::chrono::duration<double>(t1 - t0).count());
  return true;
#else
  (void)y;
  (void)x;
  (void)W_quantized;
  (void)n_matrices;
  (void)S;
  (void)I;
  (void)O;
  return false;
#endif
}

} // namespace

bool matmul_qwen_iq1_m_iq4_nl_fused(
    float *const *gate_y, float *const *up_y, float *const *down_y,
    const float *x, const uint8_t *const *gate_weights,
    const uint8_t *const *up_weights, const uint8_t *const *down_weights,
    int n_matrices, int I, int gate_O, int down_O, int n_threads) {
#if defined(MOESTREAM_USE_GGML)
  constexpr int QK_K = 256;
  constexpr int QK4_NL = 32;
  if (!gate_y || !up_y || !down_y || !x || !gate_weights || !up_weights ||
      !down_weights || n_matrices <= 0 || I <= 0 || gate_O <= 0 ||
      down_O <= 0 || I % QK_K != 0 || gate_O % QK4_NL != 0)
    return false;

  ensure_ggml_cpu_initialized();
  const int input_blocks = I / QK_K;
  const int down_blocks = gate_O / QK4_NL;
  const size_t gate_row_bytes =
      static_cast<size_t>(input_blocks) * sizeof(moestream_ggml_block_iq1_m);
  const size_t down_row_bytes = static_cast<size_t>(down_blocks) * 18;

  // Keep these buffers local.  This probe must remain safe if a test invokes
  // it from a caller that already owns another grouped-kernel scratch area;
  // the previous IQ2/3 TLS experiment demonstrated that aliasing nested
  // callers is not worth the small allocation saving.
  // The scratch must be shared by the OpenMP team: worker code accesses the
  // down-input vector after the gate/up phase.  A thread_local vector here is
  // incorrect because each worker would see a different, unresized vector.
  // Keep input_q thread-local only because workers consume its captured data
  // pointer; make down_input_q invocation-local and therefore shared and safe
  // for independent request threads.
  static thread_local std::vector<moestream_ggml_block_q8_K> input_q;
  static thread_local std::vector<block_q8_0> down_input_q;
  input_q.resize(static_cast<size_t>(input_blocks));
  down_input_q.resize(static_cast<size_t>(n_matrices) * down_blocks);
  const int team_threads = n_threads > 0 ? n_threads : qwen_cpu_grouped_threads();
  const bool stage_timing = qwen_cpu_stage_timing_enabled_local();
  std::chrono::steady_clock::time_point stage_start;
  if (stage_timing)
    stage_start = std::chrono::steady_clock::now();
  quantize_row_q8_K(x, input_q.data(), I);
  const auto *input_q_data = input_q.data();
  // The vector objects above are caller-thread TLS, but the byte buffer must
  // be shared by the OpenMP team.  Capture the data pointer before entering
  // the parallel region; referring to down_input_q itself inside the region
  // would resolve each worker's separate TLS instance, leaving its down
  // projection input empty and causing an out-of-bounds dot-product read.
  auto *down_input_q_data = down_input_q.data();

  // One persistent team performs all three projection phases.  The barriers
  // are required by the data dependencies, but the team itself is created
  // only once per routed layer/token instead of once for gate/up and once
  // again for down.
#pragma omp parallel num_threads(team_threads)
  {
#pragma omp for schedule(static) nowait
    for (int64_t flat = 0;
         flat < static_cast<int64_t>(n_matrices) * gate_O; ++flat) {
      const int m = static_cast<int>(flat / gate_O);
      const int o = static_cast<int>(flat % gate_O);
      float *gate_out = gate_y[m] + o;
      float *up_out = up_y[m] + o;
      const uint8_t *gate_row =
          gate_weights[m] + static_cast<size_t>(o) * gate_row_bytes;
      const uint8_t *up_row =
          up_weights[m] + static_cast<size_t>(o) * gate_row_bytes;
      ggml_vec_dot_iq1_m_q8_K(I, gate_out, 0, gate_row, 0, input_q_data, 0,
                              1);
      ggml_vec_dot_iq1_m_q8_K(I, up_out, 0, up_row, 0, input_q_data, 0, 1);
    }

    if (stage_timing) {
#pragma omp barrier
#pragma omp master
      {
        qwen_cpu_stage_add(perf().t_qwen_cpu_grouped_gate_up, stage_start);
        stage_start = std::chrono::steady_clock::now();
      }
#pragma omp barrier
    } else {
#pragma omp barrier
    }
#pragma omp for schedule(static) nowait
    for (int m = 0; m < n_matrices; ++m) {
      silu_inplace(gate_y[m], gate_O);
      elem_mul(gate_y[m], gate_y[m], up_y[m], gate_O);
    }

    // GGML's optimized Q8_0 quantizer is not safe to enter concurrently from
    // this fused OpenMP team on all CPU builds.  The rows are tiny at decode
    // time, so keep the elementwise SwiGLU parallel and make only the
    // post-SwiGLU quantization single-owner.  This removes the latent worker
    // crash without serializing either projection GEMV or the down GEMV.
#pragma omp barrier
#pragma omp single
    {
      for (int m = 0; m < n_matrices; ++m)
        quantize_row_q8_0(
            gate_y[m],
            down_input_q_data + static_cast<size_t>(m) * down_blocks,
            gate_O);
    }

    if (stage_timing) {
#pragma omp barrier
#pragma omp master
      {
        qwen_cpu_stage_add(perf().t_qwen_cpu_grouped_swiglu, stage_start);
        stage_start = std::chrono::steady_clock::now();
      }
#pragma omp barrier
    } else {
#pragma omp barrier
    }
#pragma omp for schedule(static) nowait
    for (int64_t flat = 0;
         flat < static_cast<int64_t>(n_matrices) * down_O; ++flat) {
      const int m = static_cast<int>(flat / down_O);
      const int d = static_cast<int>(flat % down_O);
      const uint8_t *down_row =
          down_weights[m] + static_cast<size_t>(d) * down_row_bytes;
      ggml_vec_dot_iq4_nl_q8_0(
          gate_O, down_y[m] + d, 0, down_row, 0,
          down_input_q_data + static_cast<size_t>(m) * down_blocks,
          sizeof(block_q8_0), 1);
    }
  }
  if (stage_timing)
    qwen_cpu_stage_add(perf().t_qwen_cpu_grouped_down, stage_start);
  return true;
#else
  (void)gate_y;
  (void)up_y;
  (void)down_y;
  (void)x;
  (void)gate_weights;
  (void)up_weights;
  (void)down_weights;
  (void)n_matrices;
  (void)I;
  (void)gate_O;
  (void)down_O;
  return false;
#endif
}

bool matmul_qwen_iq2_xxs_iq4_nl_fused(
    float *const *gate_y, float *const *up_y, float *const *down_y,
    const float *x, const uint8_t *const *gate_weights,
    const uint8_t *const *up_weights, const uint8_t *const *down_weights,
    int n_matrices, int I, int gate_O, int down_O, int n_threads) {
#if defined(MOESTREAM_USE_GGML)
  constexpr int QK_K = 256;
  constexpr int QK4_NL = 32;
  constexpr size_t IQ2_XXS_BLOCK_BYTES = 66;
  if (!gate_y || !up_y || !down_y || !x || !gate_weights || !up_weights ||
      !down_weights || n_matrices <= 0 || I <= 0 || gate_O <= 0 ||
      down_O <= 0 || I % QK_K != 0 || gate_O % QK4_NL != 0)
    return false;

  ensure_ggml_cpu_initialized();
  const int input_blocks = I / QK_K;
  const int down_blocks = gate_O / QK4_NL;
  const size_t gate_row_bytes =
      static_cast<size_t>(input_blocks) * IQ2_XXS_BLOCK_BYTES;
  const size_t down_row_bytes = static_cast<size_t>(down_blocks) * 18;

  // GGML's IQ2_XXS dot kernel consumes the production Q8_K activation
  // layout.  Quantize the shared input once for both projections, matching
  // the IQ1_M fused path and Bee/ggml MUL_MAT_ID's grouped route.
  // Reuse the same per-thread scratch across decode layer calls. This avoids
  // the two small heap allocations that would otherwise occur for every
  // IQ2_XXS fused layer invocation.
  static thread_local std::vector<moestream_ggml_block_q8_K> input_q;
  static thread_local std::vector<block_q8_0> down_input_q;
  input_q.resize(static_cast<size_t>(input_blocks));
  down_input_q.resize(static_cast<size_t>(n_matrices) * down_blocks);
  const int team_threads = n_threads > 0 ? n_threads : qwen_cpu_grouped_threads();
  const bool stage_timing = qwen_cpu_stage_timing_enabled_local();
  std::chrono::steady_clock::time_point stage_start;
  if (stage_timing)
    stage_start = std::chrono::steady_clock::now();
  quantize_row_q8_K(x, input_q.data(), I);
  auto *down_input_q_data = down_input_q.data();

#pragma omp parallel num_threads(team_threads)
  {
#pragma omp for schedule(static) nowait
    for (int64_t flat = 0;
         flat < static_cast<int64_t>(n_matrices) * gate_O; ++flat) {
      const int m = static_cast<int>(flat / gate_O);
      const int o = static_cast<int>(flat % gate_O);
      float *gate_out = gate_y[m] + o;
      float *up_out = up_y[m] + o;
      const uint8_t *gate_row =
          gate_weights[m] + static_cast<size_t>(o) * gate_row_bytes;
      const uint8_t *up_row =
          up_weights[m] + static_cast<size_t>(o) * gate_row_bytes;
      ggml_vec_dot_iq2_xxs_q8_K(I, gate_out, 0, gate_row, 0,
                                input_q.data(), 0, 1);
      ggml_vec_dot_iq2_xxs_q8_K(I, up_out, 0, up_row, 0, input_q.data(), 0,
                                1);
    }

    if (stage_timing) {
#pragma omp barrier
#pragma omp master
      {
        qwen_cpu_stage_add(perf().t_qwen_cpu_grouped_gate_up, stage_start);
        stage_start = std::chrono::steady_clock::now();
      }
#pragma omp barrier
    } else {
#pragma omp barrier
    }
#pragma omp for schedule(static) nowait
    for (int m = 0; m < n_matrices; ++m) {
      silu_inplace(gate_y[m], gate_O);
      elem_mul(gate_y[m], gate_y[m], up_y[m], gate_O);
    }

#pragma omp barrier
#pragma omp single
    {
      for (int m = 0; m < n_matrices; ++m)
      quantize_row_q8_0(
          gate_y[m],
          down_input_q_data + static_cast<size_t>(m) * down_blocks,
          gate_O);
    }

    if (stage_timing) {
#pragma omp barrier
#pragma omp master
      {
        qwen_cpu_stage_add(perf().t_qwen_cpu_grouped_swiglu, stage_start);
        stage_start = std::chrono::steady_clock::now();
      }
#pragma omp barrier
    } else {
#pragma omp barrier
    }
#pragma omp for schedule(static) nowait
    for (int64_t flat = 0;
         flat < static_cast<int64_t>(n_matrices) * down_O; ++flat) {
      const int m = static_cast<int>(flat / down_O);
      const int d = static_cast<int>(flat % down_O);
      const uint8_t *down_row =
          down_weights[m] + static_cast<size_t>(d) * down_row_bytes;
      ggml_vec_dot_iq4_nl_q8_0(
          gate_O, down_y[m] + d, 0, down_row, 0,
          down_input_q_data + static_cast<size_t>(m) * down_blocks,
          sizeof(block_q8_0), 1);
    }
  }
  if (stage_timing)
    qwen_cpu_stage_add(perf().t_qwen_cpu_grouped_down, stage_start);
  return true;
#else
  (void)gate_y;
  (void)up_y;
  (void)down_y;
  (void)x;
  (void)gate_weights;
  (void)up_weights;
  (void)down_weights;
  (void)n_matrices;
  (void)I;
  (void)gate_O;
  (void)down_O;
  return false;
#endif
}

bool matmul_iq2_xxs_ggml_avx2_grouped(float *const *y, const float *x,
                                      const uint8_t *const *W_quantized,
                                      int n_matrices, int S, int I, int O) {
  return matmul_iqxxs_ggml_avx2_grouped_impl(
      y, x, W_quantized, n_matrices, S, I, O, 66, dot_iq2_xxs_avx2_prototype);
}

bool matmul_iq3_xxs_ggml_avx2_grouped(float *const *y, const float *x,
                                      const uint8_t *const *W_quantized,
                                      int n_matrices, int S, int I, int O) {
  return matmul_iqxxs_ggml_avx2_grouped_impl(
      y, x, W_quantized, n_matrices, S, I, O, 98, dot_iq3_xxs_avx2_prototype);
}

bool matmul_iq2_xxs_ggml_q8k_grouped(float *const *y, const float *x,
                                     const uint8_t *const *W_quantized,
                                     int n_matrices, int S, int I, int O) {
#if defined(MOESTREAM_USE_GGML)
  return matmul_iqxxs_ggml_avx2_grouped_impl(
      y, x, W_quantized, n_matrices, S, I, O, 66, dot_iq2_xxs_ggml_q8k_exact);
#else
  (void)y;
  (void)x;
  (void)W_quantized;
  (void)n_matrices;
  (void)S;
  (void)I;
  (void)O;
  return false;
#endif
}

bool matmul_iq3_xxs_ggml_q8k_grouped(float *const *y, const float *x,
                                     const uint8_t *const *W_quantized,
                                     int n_matrices, int S, int I, int O) {
#if defined(MOESTREAM_USE_GGML)
  return matmul_iqxxs_ggml_avx2_grouped_impl(
      y, x, W_quantized, n_matrices, S, I, O, 98, dot_iq3_xxs_ggml_q8k_exact);
#else
  (void)y;
  (void)x;
  (void)W_quantized;
  (void)n_matrices;
  (void)S;
  (void)I;
  (void)O;
  return false;
#endif
}

bool matmul_iq4_nl_ggml_q8_0_grouped_inputs(float *const *y,
                                            const float *const *x,
                                            const uint8_t *const *W_quantized,
                                            int n_matrices, int S, int I,
                                            int O) {
  return matmul_iq4_nl_ggml_q8_0_grouped_impl(y, x, W_quantized, n_matrices, S,
                                              I, O);
}

// ============================================================

// Fast AVX-512 dequantization helper for Q5_K 16-element sub-groups
// Corrected AVX-512 dequantization helper for Q5_K 16-element sub-groups
static inline void dequant_q5k_group16(const uint8_t *q, const uint8_t *qh,
                                       int shift_qh, int shift_qs, float dl,
                                       float ml, float *out) {
  // 1. Extract 4-bit low nibbles
  __m128i raw_q = _mm_loadu_si128((const __m128i *)q);
  __m256i wide_q = _mm256_cvtepu8_epi16(raw_q);
  __m256i lo_nibbles = _mm256_and_si256(_mm256_srli_epi16(wide_q, shift_qs),
                                        _mm256_set1_epi16(0x0F));
  __m512i v_lo = _mm512_cvtepi16_epi32(lo_nibbles);

  // 2. Extract 5th high bit from qh mask plane
  __m128i v_qh = _mm_loadu_si128((const __m128i *)qh);

  // Isolate the specific bit for this sub-block across all 16 bytes
  __m128i bit_mask = _mm_set1_epi8(1 << shift_qh);
  __m128i masked = _mm_and_si128(v_qh, bit_mask);

  // Create a 0xFF byte mask where the bit was set, 0x00 where clear
  __m128i cmp = _mm_cmpeq_epi8(masked, bit_mask);

  // Extract the top bit of each byte into a native AVX-512 16-bit k-mask
  __mmask16 hbit_mask = _mm_movemask_epi8(cmp);

  // 3. Add 16 to elements where 5th bit is set
  __m512i v_q5 =
      _mm512_mask_add_epi32(v_lo, hbit_mask, v_lo, _mm512_set1_epi32(16));

  // 4. Transform to FP32 float: y = dl * q5 - ml
  __m512 vf = _mm512_cvtepi32_ps(v_q5);
  __m512 res =
      _mm512_sub_ps(_mm512_mul_ps(_mm512_set1_ps(dl), vf), _mm512_set1_ps(ml));
  _mm512_storeu_ps(out, res);
}

// Helper to extract scale and min for Q5_K (ported directly from CUDA kernels)
static inline void get_scale_min_k4_d(int j, const uint8_t *scales, uint8_t *d,
                                      uint8_t *m) {
  if (j < 4) {
    *d = scales[j] & 63;
    *m = scales[j + 4] & 63;
  } else {
    *d = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4);
    *m = (scales[j + 4] >> 4) | ((scales[j - 0] >> 6) << 4);
  }
}

// ----------------------------------------------------
// Fused Q5_K Matmul
// ----------------------------------------------------
void matmul_q5_k_fused(float *y, const float *x, const uint8_t *W_quantized,
                       int S, int I, int O) {
  constexpr int QK_K = 256;
  constexpr size_t BLOCK_BYTES = 176;
  const size_t row_bytes = (size_t)(I / QK_K) * BLOCK_BYTES;

  for (int s = 0; s < S; ++s) {
    const float *xs = x + (int64_t)s * I;
    float *ys = y + (int64_t)s * O;

#pragma omp parallel for schedule(static)
    for (int o = 0; o < O; ++o) {
      const uint8_t *row_w = W_quantized + (size_t)o * row_bytes;
      if (o + 1 < O) {
        _mm_prefetch((const char *)(W_quantized + (size_t)(o + 1) * row_bytes),
                     _MM_HINT_T0);
      }

      float sum = 0.0f;
      float yq[256];

      for (int b = 0; b < I / QK_K; ++b) {
        const uint8_t *base = row_w + b * BLOCK_BYTES;
        uint16_t d_half, dmin_half;
        std::memcpy(&d_half, base, 2);
        std::memcpy(&dmin_half, base + 2, 2);
        const float d = fp16_to_fp32(d_half);
        const float min = fp16_to_fp32(dmin_half);

        const uint8_t *scales = base + 4;
        const uint8_t *qh = base + 16;
        const uint8_t *qs = base + 48;

        int is = 0;
        int shift_qh1 = 0;
        int shift_qh2 = 1;

        for (int j = 0; j < QK_K; j += 64) {
          uint8_t sc1, m1, sc2, m2;
          get_scale_min_k4_d(is + 0, scales, &sc1, &m1);
          get_scale_min_k4_d(is + 1, scales, &sc2, &m2);

          float dl1 = d * sc1, ml1 = min * m1;
          float dl2 = d * sc2, ml2 = min * m2;

          // Lower 32 elements of the 64-element block
          dequant_q5k_group16(qs, qh, shift_qh1, 0, dl1, ml1, yq + j);
          dequant_q5k_group16(qs + 16, qh + 16, shift_qh1, 0, dl1, ml1,
                              yq + j + 16);

          // Upper 32 elements of the 64-element block
          dequant_q5k_group16(qs, qh, shift_qh2, 4, dl2, ml2, yq + j + 32);
          dequant_q5k_group16(qs + 16, qh + 16, shift_qh2, 4, dl2, ml2,
                              yq + j + 48);

          qs += 32;
          is += 2;
          shift_qh1 += 2;
          shift_qh2 += 2;
        }

        // Dot product for this 256-element block
        sum += dot_f32_avx512(yq, xs + b * 256, 256);
      }
      ys[o] = sum;
    }
  }
}

inline void quantize_activations_q6_k(const float *x, block_xq_q6 *xq, int I) {
  int num_blocks = I / 256;
  for (int b = 0; b < num_blocks; ++b) {
    for (int i = 0; i < 16; ++i) { // 16 sub-blocks of 16 elements
      float max_val = 0.0f;
      for (int j = 0; j < 16; ++j) {
        max_val = std::max(max_val, std::abs(x[b * 256 + i * 16 + j]));
      }
      float scale = max_val / 127.0f;
      float inv_scale = scale > 0 ? 1.0f / scale : 0.0f;

      float sum = 0.0f;
      for (int j = 0; j < 16; ++j) {
        float v = x[b * 256 + i * 16 + j];
        int8_t q = (int8_t)std::round(v * inv_scale);
        xq[b].qs[i * 16 + j] = q;
        sum += q; // Cache the sum of x for the VNNI offset trick
      }
      xq[b].scales[i] = scale;
      xq[b].sums[i] = sum;
    }
  }
}

// ----------------------------------------------------
// Fused Q6_K Matmul (VNNI Integer Math)
// ----------------------------------------------------
void matmul_q6_k_fused(float *y, const float *x, const uint8_t *W_quantized,
                       int S, int I, int O) {
  constexpr int QK6_K = 256;
  constexpr size_t BLOCK_BYTES =
      210; // 128 (ql) + 64 (qh) + 16 (scales) + 2 (d)
  const int num_blocks = I / QK6_K;

  // FIX: was `static thread_local` -- resized here on the CALLING thread,
  // then written to from inside the `#pragma omp parallel for` region
  // below. OpenMP worker threads are a SEPARATE thread pool from the
  // calling thread; a `thread_local` variable does not travel with the
  // parallel region -- each worker thread that actually executes an
  // iteration sees its OWN, never-resized (size-0) copy of this vector.
  // `quantize_activations_q6_k` then wrote through `g_xq_q6.data()` on that
  // empty vector -- an invalid-pointer write, the real cause of a real
  // SIGSEGV crash inside this function's outlined omp-for body (confirmed
  // via a real gdb backtrace: `matmul_q6_k_fused(...) [clone ._omp_fn.0]`).
  // Plain `static` (shared across every thread) is correct instead: the
  // buffer is fully written by one call before this function returns, this
  // function is only ever invoked serially from the single decode thread
  // (never concurrently from two callers), and each omp iteration writes
  // only to its own disjoint `s`-indexed sub-range -- no data race.
  static std::vector<block_xq_q6> g_xq_q6;
  if (g_xq_q6.size() < (size_t)(S * num_blocks)) {
    g_xq_q6.resize(S * num_blocks);
  }

#pragma omp parallel for schedule(static)
  for (int s = 0; s < S; ++s) {
    quantize_activations_q6_k(x + (int64_t)s * I,
                              g_xq_q6.data() + (int64_t)s * num_blocks, I);
  }

#pragma omp parallel for schedule(static)
  for (int o = 0; o < O; ++o) {
    const uint8_t *row_w = W_quantized + (size_t)o * num_blocks * BLOCK_BYTES;

    for (int s = 0; s < S; ++s) {
      const block_xq_q6 *xq = g_xq_q6.data() + (int64_t)s * num_blocks;
      float sum_f32 = 0.0f;

      for (int b = 0; b < num_blocks; ++b) {
        const block_q6_K *bw = (const block_q6_K *)(row_w + b * BLOCK_BYTES);
        float block_d = fp16_to_fp32(bw->d);

#if defined(__AVX512VNNI__) && defined(__AVX512BW__)
        // Process in chunks of 64 elements (4 sub-blocks of 16)
        for (int chunk = 0; chunk < 4; ++chunk) {
          // 1. Extract 64 Q6_K weights into uint8 (0-63 range)
          // FIX: the old "idx % 128" / "idx % 64" formula here was a real,
          // confirmed bug (found via a real Test 5 FAILED result + direct
          // hand-tracing against ggml's actual dequantize_row_q6_K,
          // ggml/src/ggml-quants.c) -- it only happened to be correct for
          // the FIRST 32-element group of each 128-element half; every
          // other group read the wrong ql byte/nibble and the wrong qh
          // byte/shift entirely. ggml's real layout: for output position
          // idx, n_half=idx/128 selects which half (ql/qh/scales each
          // advance by 64/32/8 bytes between halves); within a half,
          // group=(idx%128)/32 selects which of the 4 q1..q4 sub-ranges
          // (LOW nibble of ql[..] for group 0/1, HIGH nibble for group
          // 2/3; ql byte offset +32 for group 1/3 vs 0/2); lpos=idx%32 is
          // the position shared by ALL FOUR groups' qh[lpos] byte, with
          // the 2-bit field selected by shift=group*2 (0,2,4,6) -- ql and
          // qh are indexed completely differently (byte offset vs bit
          // shift), which the old "idx % 128" / "idx % 64" pair conflated
          // into one wrong formula.
          uint8_t w_expanded[64];
          for (int i = 0; i < 64; ++i) {
            int idx = chunk * 64 + i;
            int n_half = idx / 128;
            int grp = (idx % 128) / 32;
            int lpos = idx % 32;
            uint8_t ql_byte = bw->ql[n_half * 64 + lpos + 32 * (grp & 1)];
            uint8_t l = (grp < 2) ? (ql_byte & 0x0F) : (ql_byte >> 4);
            uint8_t h = (bw->qh[n_half * 32 + lpos] >> (grp * 2)) & 0x03;
            w_expanded[i] = l | (h << 4); // natively unsigned 0..63
          }

          __m512i vw = _mm512_loadu_si512((const void *)w_expanded);
          __m512i vx =
              _mm512_loadu_si512((const void *)(xq[b].qs + chunk * 64));

          // 2. Hardware VNNI dot product (uint8 * int8)
          // Yields sixteen 32-bit integers. Each int32 is the sum of 4
          // elements.
          __m512i acc_dp = _mm512_dpbusd_epi32(_mm512_setzero_si512(), vw, vx);

          // 3. Extract four 16-element sub-block sums from the AVX register
          // We horizontally add the 32-bit lanes to group them by 16 elements.
          alignas(64) int32_t dp_lanes[16];
          _mm512_store_si512(dp_lanes, acc_dp);

          for (int sub = 0; sub < 4; ++sub) {
            int sb_idx = chunk * 4 + sub;

            // Horizontally sum 4 lanes of dp_lanes to get a 16-element dot
            // product
            int32_t true_dot_16 = dp_lanes[sub * 4 + 0] +
                                  dp_lanes[sub * 4 + 1] +
                                  dp_lanes[sub * 4 + 2] + dp_lanes[sub * 4 + 3];

            // 4. Apply the -32 Offset Trick mathematically
            // sum(w_true * x) = sum(w_u8 * x) - 32 * sum(x)
            float dot_adjusted =
                (float)true_dot_16 - (32.0f * xq[b].sums[sb_idx]);

            // 5. Multiply by scales and accumulate
            sum_f32 += dot_adjusted * (float)bw->scales[sb_idx] *
                       xq[b].scales[sb_idx] * block_d;
          }
        }
#else
        // Scalar Fallback
        // FIX: same real bug as the VNNI branch's w_expanded extraction
        // above (see that comment for the full derivation against ggml's
        // real dequantize_row_q6_K) -- the old "idx % 128" / "idx % 64"
        // formula only happened to be correct for the first 32-element
        // group of each 128-element half.
        for (int i = 0; i < 16; ++i) { // 16 sub-blocks
          int32_t sub_dot = 0;
          for (int j = 0; j < 16; ++j) {
            int idx = i * 16 + j;
            int n_half = idx / 128;
            int grp = (idx % 128) / 32;
            int lpos = idx % 32;
            uint8_t ql_byte = bw->ql[n_half * 64 + lpos + 32 * (grp & 1)];
            uint8_t l = (grp < 2) ? (ql_byte & 0x0F) : (ql_byte >> 4);
            uint8_t h = (bw->qh[n_half * 32 + lpos] >> (grp * 2)) & 0x03;
            int8_t w_val = (int8_t)(l | (h << 4)) - 32; // signed mapping

            sub_dot += w_val * xq[b].qs[idx];
          }
          sum_f32 += sub_dot * (float)bw->scales[i] * xq[b].scales[i] * block_d;
        }
#endif
      }
      y[(int64_t)s * O + o] = sum_f32;
    }
  }
}

// ----------------------------------------------------
// Fused Q8_0 Matmul (VNNI Integer Math)
// ----------------------------------------------------
void matmul_q8_0_fused(float *y, const float *x, const uint8_t *W_quantized,
                       int S, int I, int O) {
  constexpr int QK8_0 = 32;
  constexpr size_t BLOCK_BYTES = 34; // 2 bytes scale + 32 bytes qs
  const int num_blocks = I / QK8_0;
  const size_t row_bytes = (size_t)num_blocks * BLOCK_BYTES;

  // FIX: same real bug as matmul_q6_k_fused's g_xq_q6 above (see that
  // comment for the full explanation) -- `thread_local` here meant every
  // OpenMP worker thread saw its own never-resized, empty copy of this
  // vector inside the `#pragma omp parallel for` regions below, an
  // invalid-pointer write waiting to happen the first time this path
  // actually ran (Q6_K crashed first since it's tested first; this
  // function was never reached in that same run, but has the identical
  // defect). Plain `static` is correct: single decode-thread-only caller,
  // fully written before return, disjoint per-`s` sub-ranges across
  // worker threads.
  static std::vector<block_q8_0> g_xq_q8;
  if (g_xq_q8.size() < (size_t)(S * num_blocks)) {
    g_xq_q8.resize(S * num_blocks);
  }

// 2. Parallelize activation quantization across sequence length S
#pragma omp parallel for schedule(static)
  for (int s = 0; s < S; ++s) {
    quantize_activations_q8_0(x + (int64_t)s * I,
                              g_xq_q8.data() + (int64_t)s * num_blocks, I);
  }

// 3. Optimized parallel matrix multiplication loop using VNNI
#pragma omp parallel for schedule(static)
  for (int o = 0; o < O; ++o) {
    const uint8_t *row_w = W_quantized + (size_t)o * row_bytes;
    if (o + 1 < O) {
      _mm_prefetch((const char *)(W_quantized + (size_t)(o + 1) * row_bytes),
                   _MM_HINT_T0);
    }

    for (int s = 0; s < S; ++s) {
      const block_q8_0 *xq = g_xq_q8.data() + (int64_t)s * num_blocks;

#if defined(__AVX512VNNI__) && defined(__AVX512BW__)
      __m512 f_acc = _mm512_setzero_ps();
      const __m512i offset_128 = _mm512_set1_epi8((char)128);
      const __m512i ones_8 = _mm512_set1_epi8(1);
      const __m512i ones_16 = _mm512_set1_epi16(1);

      int b = 0;
      // Pack and process 2 blocks (64 bytes/elements) per AVX-512 iteration
      for (; b + 1 < num_blocks; b += 2) {
        const uint8_t *base_w0 = row_w + b * BLOCK_BYTES;
        const uint8_t *base_w1 = row_w + (b + 1) * BLOCK_BYTES;

        uint16_t d_w0;
        std::memcpy(&d_w0, base_w0, 2);
        uint16_t d_w1;
        std::memcpy(&d_w1, base_w1, 2);

        // Precalculate merged scale for both blocks
        float scale0 = fp16_to_fp32(d_w0) * fp16_to_fp32(xq[b].d);
        float scale1 = fp16_to_fp32(d_w1) * fp16_to_fp32(xq[b + 1].d);
        __m256 vscale0 = _mm256_set1_ps(scale0);
        __m256 vscale1 = _mm256_set1_ps(scale1);
        __m512 vscale =
            _mm512_insertf32x8(_mm512_castps256_ps512(vscale0), vscale1, 1);

        // Load weights into 512-bit register
        __m256i w0 = _mm256_loadu_si256((const __m256i *)(base_w0 + 2));
        __m256i w1 = _mm256_loadu_si256((const __m256i *)(base_w1 + 2));
        __m512i vw = _mm512_inserti64x4(_mm512_castsi256_si512(w0), w1, 1);

        // Load quantized activations into 512-bit register
        __m256i x0 = _mm256_loadu_si256((const __m256i *)xq[b].qs);
        __m256i x1 = _mm256_loadu_si256((const __m256i *)xq[b + 1].qs);
        __m512i vx = _mm512_inserti64x4(_mm512_castsi256_si512(x0), x1, 1);

        // Shift weights to unsigned domain for VNNI compatibility
        __m512i vw_u8 = _mm512_add_epi8(vw, offset_128);

        // Execute hardware dot product
        __m512i acc_dp = _mm512_dpbusd_epi32(_mm512_setzero_si512(), vw_u8, vx);

        // Reverse the +128 offset bias mathematically
        __m512i sum_x_s16 = _mm512_maddubs_epi16(ones_8, vx);
        __m512i acc_x = _mm512_madd_epi16(sum_x_s16, ones_16);
        __m512i true_dot =
            _mm512_sub_epi32(acc_dp, _mm512_slli_epi32(acc_x, 7));

        // Convert finalized int32 to float, multiply by scales, and accumulate
        f_acc = _mm512_fmadd_ps(_mm512_cvtepi32_ps(true_dot), vscale, f_acc);
      }

      float sum_f32 = _mm512_reduce_add_ps(f_acc);

      // Clean up any remaining odd block (Scalar fallback)
      for (; b < num_blocks; ++b) {
        const uint8_t *base_w = row_w + b * BLOCK_BYTES;
        uint16_t d_w;
        std::memcpy(&d_w, base_w, 2);
        float scale = fp16_to_fp32(d_w) * fp16_to_fp32(xq[b].d);

        const int8_t *qs_w = (const int8_t *)(base_w + 2);
        const int8_t *qs_x = xq[b].qs;
        int32_t dot = 0;

        for (int j = 0; j < 32; ++j)
          dot += qs_w[j] * qs_x[j];
        sum_f32 += dot * scale;
      }
#else
      // Safe fallback if compiled without VNNI architecture targeting
      float sum_f32 = 0.0f;
      for (int b = 0; b < num_blocks; ++b) {
        const uint8_t *base_w = row_w + b * BLOCK_BYTES;
        uint16_t d_w;
        std::memcpy(&d_w, base_w, 2);
        float scale = fp16_to_fp32(d_w) * fp16_to_fp32(xq[b].d);
        const int8_t *qs_w = (const int8_t *)(base_w + 2);
        const int8_t *qs_x = xq[b].qs;

        int32_t dot = 0;
        for (int j = 0; j < 32; ++j)
          dot += qs_w[j] * qs_x[j];
        sum_f32 += dot * scale;
      }
#endif

      y[(int64_t)s * O + o] = sum_f32;
    }
  }
}

// ----------------------------------------------------
// Standard F32 Matmul (AVX-512 FMA)
// ----------------------------------------------------
void matmul_f32_avx512(float *y, const float *x, const float *w, int S, int I,
                       int O) {
#pragma omp parallel for schedule(static)
  for (int o = 0; o < O; ++o) {
    const float *row_w = w + (size_t)o * I;

    for (int s = 0; s < S; ++s) {
      const float *row_x = x + (int64_t)s * I;
      float sum = 0.0f;
      int i = 0;

#if defined(__AVX512F__)
      __m512 acc = _mm512_setzero_ps();

      // Process 16 floats (512 bits) per iteration
      for (; i + 16 <= I; i += 16) {
        __m512 vx = _mm512_loadu_ps(row_x + i);
        __m512 vw = _mm512_loadu_ps(row_w + i);
        // acc = (vx * vw) + acc
        acc = _mm512_fmadd_ps(vx, vw, acc);
      }

      // Horizontally sum the 16 lanes of the AVX register
      sum = _mm512_reduce_add_ps(acc);
#endif

      // Scalar tail for remaining elements
      for (; i < I; ++i) {
        sum += row_x[i] * row_w[i];
      }

      y[(int64_t)s * O + o] = sum;
    }
  }
}
