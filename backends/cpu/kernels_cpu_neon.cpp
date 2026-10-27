// src/kernels_cpu_neon.cpp — Apple Silicon (M3/M4/M5) optimized CPU kernels.
//
// This file is compiled INSTEAD OF src/kernels_cpu.cpp on arm64 macOS.  It
// provides the same public symbols so that src/execution_layer.cpp and the
// tests can link without #ifdef soup at every call site.
//
// Current coverage:
//   * f32 matmul / dot / rmsnorm / silu / elem-mul — NEON-4/NEON-8
//   * Q4_0 / Q8_0 / IDOT paths — NEON SDOT when available, scalar tail
//   * K-quants (Q2_K/Q3_K) and IQ types — ggml ARM dequantizers.
//
// The function names retain their historical "_avx512" suffix in a few
// places; this is an intentional ABI compatibility shim while the wider
// modernization (renaming everything to platform-neutral names) is in
// progress.

#include "kernels_cpu.hpp"
#include "logger.hpp"

#if defined(__APPLE__) && defined(ZEROMOE_METAL)
#include "kernels_metal.hpp"
#endif

#include <algorithm>
#include <arm_neon.h>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <vector>

namespace {

// Thread-safe "log this exactly once" helper, mirrored from the x86 file.
bool should_log_once(std::atomic<bool> &flag) {
  return !flag.exchange(true, std::memory_order_relaxed);
}

// Portable IEEE-754 binary16 <-> binary32 conversions.
// Apple Silicon has fp16 hardware, but using integer bit-twiddling keeps this
// file compilable on any ARM NEON toolchain and avoids ABI/signature quirks.
float fp16_bits_to_f32(uint16_t h) {
  const uint32_t sign = (h >> 15) & 0x1u;
  const uint32_t exp = (h >> 10) & 0x1fu;
  const uint32_t mant = h & 0x3ffu;

  if (exp == 0) {
    if (mant == 0)
      return sign ? -0.0f : 0.0f;
    // Denormal
    const float f = static_cast<float>(mant) * (1.0f / 1024.0f);
    return std::ldexp(sign ? -f : f, -14);
  }
  if (exp == 0x1f) {
    const uint32_t inf = 0x7f800000u | (mant ? 0x00400000u : 0u);
    float f;
    std::memcpy(&f, &inf, sizeof(f));
    return sign ? -f : f;
  }
  uint32_t f32 = (sign << 31) | ((exp + 112u) << 23) | (mant << 13);
  float f;
  std::memcpy(&f, &f32, sizeof(f));
  return f;
}

uint16_t f32_to_fp16_bits(float f_in) {
  uint32_t f;
  std::memcpy(&f, &f_in, sizeof(f));
  const uint32_t sign = (f >> 31) & 0x1u;
  const int32_t exp = static_cast<int32_t>((f >> 23) & 0xffu) - 127;
  const uint32_t mant = f & 0x007fffffu;

  if (exp <= -15) {
    if (exp <= -25 || (exp == -24 && mant == 0))
      return sign ? 0x8000u : 0u;
    // Denormal
    const uint32_t m = (mant | 0x00800000u) >> (-exp - 14);
    return static_cast<uint16_t>((sign << 15) | (m >> 13));
  }
  if (exp >= 16) {
    return static_cast<uint16_t>((sign << 15) | 0x7c00u); // inf
  }
  return static_cast<uint16_t>((sign << 15) | ((exp + 15) << 10) |
                               (mant >> 13));
}

// Scalar SiLU fallback.  NEON has no vector exp; a polynomial exp is possible
// but the f32 path dominates dense attention/FFN on Apple Silicon, so we keep
// this simple and correct.
inline float scalar_silu(float x) { return x / (1.0f + std::exp(-x)); }

// Scalar softmax (rmsnorm already uses the same algorithm as x86).
void scalar_softmax(float *x, int size) {
  float max_val = x[0];
  for (int i = 1; i < size; ++i)
    max_val = std::max(max_val, x[i]);
  float sum = 0.0f;
  for (int i = 0; i < size; ++i) {
    x[i] = std::exp(x[i] - max_val);
    sum += x[i];
  }
  const float inv_sum = 1.0f / sum;
  for (int i = 0; i < size; ++i)
    x[i] *= inv_sum;
}

#if defined(__ARM_FEATURE_DOTPROD)
// 16-element int8 dot product using SDOT (mandatory on Apple Silicon M1+).
inline int32_t neon_dot_i8i8_16(const int8_t *a, const int8_t *b) {
  int8x16_t va = vld1q_s8(a);
  int8x16_t vb = vld1q_s8(b);
  int32x4_t acc = vdupq_n_s32(0);
  acc = vdotq_s32(acc, va, vb);
  return vaddvq_s32(acc);
}
#endif

} // namespace

float fp16_to_fp32(uint16_t h) { return fp16_bits_to_f32(h); }
uint16_t fp32_to_fp16(float f) { return f32_to_fp16_bits(f); }

// ----------------------------------------------------------------------------
// f32 x f32 matmul: y[S,O] = x[S,I] @ W^T, W is [O,I] row-major.
// ----------------------------------------------------------------------------
void matmul(float *y, const float *x, const float *W, int S, int I, int O) {
#if defined(__APPLE__) && defined(ZEROMOE_METAL)
  if (zeromoe_metal::available() &&
      zeromoe_metal::matmul_f32(y, x, W, S, I, O))
    return;
#endif

#pragma omp parallel for schedule(static)
  for (int o = 0; o < O; ++o) {
    const float *w = W + static_cast<int64_t>(o) * I;
    for (int s = 0; s < S; ++s) {
      const float *xs = x + static_cast<int64_t>(s) * I;

      float32x4_t acc0 = vdupq_n_f32(0.0f);
      float32x4_t acc1 = vdupq_n_f32(0.0f);
      float32x4_t acc2 = vdupq_n_f32(0.0f);
      float32x4_t acc3 = vdupq_n_f32(0.0f);

      int i = 0;
      // 32 floats per iteration (4x 128-bit registers).
      for (; i + 32 <= I; i += 32) {
        acc0 = vfmaq_f32(acc0, vld1q_f32(xs + i), vld1q_f32(w + i));
        acc1 = vfmaq_f32(acc1, vld1q_f32(xs + i + 4), vld1q_f32(w + i + 4));
        acc2 = vfmaq_f32(acc2, vld1q_f32(xs + i + 8), vld1q_f32(w + i + 8));
        acc3 = vfmaq_f32(acc3, vld1q_f32(xs + i + 12), vld1q_f32(w + i + 12));

        acc0 = vfmaq_f32(acc0, vld1q_f32(xs + i + 16), vld1q_f32(w + i + 16));
        acc1 = vfmaq_f32(acc1, vld1q_f32(xs + i + 20), vld1q_f32(w + i + 20));
        acc2 = vfmaq_f32(acc2, vld1q_f32(xs + i + 24), vld1q_f32(w + i + 24));
        acc3 = vfmaq_f32(acc3, vld1q_f32(xs + i + 28), vld1q_f32(w + i + 28));
      }

      float32x4_t acc_total =
          vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));

      for (; i + 4 <= I; i += 4) {
        acc_total = vfmaq_f32(acc_total, vld1q_f32(xs + i), vld1q_f32(w + i));
      }

      float sum = vaddvq_f32(acc_total);
      for (; i < I; ++i)
        sum += xs[i] * w[i];

      y[static_cast<int64_t>(s) * O + o] = sum;
    }
  }
}

// ----------------------------------------------------------------------------
// f32 dot product — name kept for ABI compatibility with execution_layer.cpp.
// ----------------------------------------------------------------------------
float dot_f32_avx512(const float *a, const float *b, int n) {
  float32x4_t acc0 = vdupq_n_f32(0.0f);
  float32x4_t acc1 = vdupq_n_f32(0.0f);
  float32x4_t acc2 = vdupq_n_f32(0.0f);
  float32x4_t acc3 = vdupq_n_f32(0.0f);

  int i = 0;
  for (; i + 16 <= n; i += 16) {
    acc0 = vfmaq_f32(acc0, vld1q_f32(a + i), vld1q_f32(b + i));
    acc1 = vfmaq_f32(acc1, vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
    acc2 = vfmaq_f32(acc2, vld1q_f32(a + i + 8), vld1q_f32(b + i + 8));
    acc3 = vfmaq_f32(acc3, vld1q_f32(a + i + 12), vld1q_f32(b + i + 12));
  }

  float32x4_t acc_total =
      vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
  for (; i + 4 <= n; i += 4) {
    acc_total = vfmaq_f32(acc_total, vld1q_f32(a + i), vld1q_f32(b + i));
  }

  float sum = vaddvq_f32(acc_total);
  for (; i < n; ++i)
    sum += a[i] * b[i];
  return sum;
}

// ----------------------------------------------------------------------------
// RMSNorm: out[i] = x[i] / sqrt(mean(x^2)+eps) * w[i]
// ----------------------------------------------------------------------------
void rmsnorm_row(float *out, const float *x, const float *w, int D, float eps) {
  double ss = 0.0;
  for (int i = 0; i < D; ++i)
    ss += static_cast<double>(x[i]) * x[i];

  const float r = 1.0f / std::sqrt(static_cast<float>(ss / D) + eps);
  const float32x4_t v_r = vdupq_n_f32(r);

  int i = 0;
  for (; i + 4 <= D; i += 4) {
    float32x4_t vx = vld1q_f32(x + i);
    float32x4_t vw = vld1q_f32(w + i);
    vst1q_f32(out + i, vmulq_f32(vmulq_f32(vx, v_r), vw));
  }
  for (; i < D; ++i)
    out[i] = x[i] * r * w[i];
}

void softmax_row(float *x, int size) { scalar_softmax(x, size); }

// ----------------------------------------------------------------------------
// SiLU: x * sigmoid(x)
// ----------------------------------------------------------------------------
void silu_inplace(float *x, int n) {
  int i = 0;
  for (; i + 4 <= n; i += 4) {
    float32x4_t v = vld1q_f32(x + i);
    float vals[4];
    vst1q_f32(vals, v);
    for (int j = 0; j < 4; ++j)
      vals[j] = scalar_silu(vals[j]);
    vst1q_f32(x + i, vld1q_f32(vals));
  }
  for (; i < n; ++i)
    x[i] = scalar_silu(x[i]);
}

// ----------------------------------------------------------------------------
// Element-wise multiply
// ----------------------------------------------------------------------------
void elem_mul(float *out, const float *a, const float *b, int n) {
  int i = 0;
  for (; i + 4 <= n; i += 4) {
    vst1q_f32(out + i, vmulq_f32(vld1q_f32(a + i), vld1q_f32(b + i)));
  }
  for (; i < n; ++i)
    out[i] = a[i] * b[i];
}

// ----------------------------------------------------------------------------
// Q8_0 activation quantization
// ----------------------------------------------------------------------------
void quantize_activations_q8_0(const float *__restrict__ x,
                               block_q8_0 *__restrict__ y, int num_elements) {
  constexpr int QK8_0 = 32;
  const int nb = num_elements / QK8_0;

#pragma omp parallel for schedule(static)
  for (int b = 0; b < nb; ++b) {
    const float *src = x + static_cast<size_t>(b) * QK8_0;
    block_q8_0 *dst = y + b;

    float max_abs = 0.0f;
    for (int i = 0; i < QK8_0; ++i) {
      max_abs = std::max(max_abs, std::fabs(src[i]));
    }

    const float d = max_abs / 127.0f;
    dst->d = fp32_to_fp16(d);
    const float id = (d != 0.0f) ? 1.0f / d : 0.0f;

    for (int i = 0; i < QK8_0; ++i) {
      int v = static_cast<int>(std::round(src[i] * id));
      dst->qs[i] = static_cast<int8_t>(std::clamp(v, -127, 127));
    }
  }
}

// ----------------------------------------------------------------------------
// Q4_0 * Q8_0 dot product (used by some CPU expert paths)
// ----------------------------------------------------------------------------
float dot_q4_0_q8_0_neon(const block_q4_0 *w, const block_q8_0 *x,
                         int num_blocks) {
  float sum_f32 = 0.0f;

#if defined(__ARM_FEATURE_DOTPROD)
  const int8x16_t v_zero_point = vdupq_n_s8(8);
  for (int i = 0; i < num_blocks; ++i) {
    const uint8_t *q4 = w[i].qs;
    const int8_t *q8 = x[i].qs;

    uint8x16_t q4_u8 = vld1q_u8(q4);
    uint8x8_t q4_lo = vget_low_u8(q4_u8);
    uint8x8_t q4_hi = vget_high_u8(q4_u8);

    // Unpack nibbles to signed bytes: q - 8 (Q4_0 zero point).
    uint8x8_t lo_n = vand_u8(q4_lo, vdup_n_u8(0x0f));
    uint8x8_t hi_n = vshr_n_u8(q4_lo, 4);
    int8x16_t q4_l =
        vsubq_s8(vreinterpretq_s8_u8(vcombine_u8(lo_n, hi_n)), v_zero_point);

    lo_n = vand_u8(q4_hi, vdup_n_u8(0x0f));
    hi_n = vshr_n_u8(q4_hi, 4);
    int8x16_t q4_h =
        vsubq_s8(vreinterpretq_s8_u8(vcombine_u8(lo_n, hi_n)), v_zero_point);

    int8x16_t x_l = vld1q_s8(q8);
    int8x16_t x_h = vld1q_s8(q8 + 16);

    int32x4_t acc = vdupq_n_s32(0);
    acc = vdotq_s32(acc, q4_l, x_l);
    acc = vdotq_s32(acc, q4_h, x_h);
    int32_t idot = vaddvq_s32(acc);

    sum_f32 += idot * fp16_to_fp32(w[i].d) * fp16_to_fp32(x[i].d);
  }
#else
  // Scalar reference path for pre-SDOT ARM cores.
  for (int i = 0; i < num_blocks; ++i) {
    int bsum = 0;
    for (int j = 0; j < 16; ++j) {
      uint8_t q = w[i].qs[j];
      int w0 = static_cast<int>(q & 0x0f) - 8;
      int w1 = static_cast<int>(q >> 4) - 8;
      bsum += w0 * x[i].qs[2 * j] + w1 * x[i].qs[2 * j + 1];
    }
    sum_f32 += bsum * fp16_to_fp32(w[i].d) * fp16_to_fp32(x[i].d);
  }
#endif

  return sum_f32;
}

float dot_q4_0_q8_0_avx512(const block_q4_0 *w, const block_q8_0 *x,
                           int num_blocks) {
  return dot_q4_0_q8_0_neon(w, x, num_blocks);
}

// ----------------------------------------------------------------------------
// IDOT helpers — quantize a row to int8 and dot products.
// ----------------------------------------------------------------------------
float qrow_i8(const float *x, int8_t *q, int I) {
  float max_abs = 0.0f;
  for (int i = 0; i < I; ++i)
    max_abs = std::max(max_abs, std::fabs(x[i]));

  const float scale = max_abs / 127.0f;
  const float inv_scale = (scale != 0.0f) ? 1.0f / scale : 0.0f;

  for (int i = 0; i < I; ++i) {
    int v = static_cast<int>(std::round(x[i] * inv_scale));
    q[i] = static_cast<int8_t>(std::clamp(v, -127, 127));
  }
  return scale;
}

int32_t dot_i8i8_avx512(const int8_t *w, const int8_t *x, int I) {
  int32_t sum = 0;
#if defined(__ARM_FEATURE_DOTPROD)
  int32x4_t acc = vdupq_n_s32(0);
  int i = 0;
  for (; i + 16 <= I; i += 16) {
    acc = vdotq_s32(acc, vld1q_s8(w + i), vld1q_s8(x + i));
  }
  sum = vaddvq_s32(acc);
  for (; i < I; ++i)
    sum += w[i] * x[i];
#else
  for (int i = 0; i < I; ++i)
    sum += w[i] * x[i];
#endif
  return sum;
}

int32_t dot_i4i8_avx512(const uint8_t *w4, const int8_t *x, int I) {
  // I is the number of weights; half the byte count.
  int32_t sum = 0;
  for (int i = 0; i < I; ++i) {
    uint8_t b = w4[i / 2];
    int w = (i & 1) ? (b >> 4) : (b & 0x0f);
    w -= 8;
    sum += w * x[i];
  }
  return sum;
}

// ----------------------------------------------------------------------------
// IDOT matmuls
// ----------------------------------------------------------------------------
void matmul_u8_idot(float *y, const float *x, const int8_t *q,
                    const float *scale, int S, int I, int O) {
  std::vector<int8_t> xq(static_cast<size_t>(S) * I);
  std::vector<float> xscale(S);

#pragma omp parallel for schedule(static)
  for (int s = 0; s < S; ++s) {
    xscale[s] = qrow_i8(x + static_cast<int64_t>(s) * I,
                        xq.data() + static_cast<int64_t>(s) * I, I);
  }

#pragma omp parallel for schedule(static)
  for (int o = 0; o < O; ++o) {
    const int8_t *wrow = q + static_cast<int64_t>(o) * I;
    for (int s = 0; s < S; ++s) {
      const int8_t *xrow = xq.data() + static_cast<int64_t>(s) * I;
      int32_t idot = dot_i8i8_avx512(wrow, xrow, I);
      y[static_cast<int64_t>(s) * O + o] =
          scale[o] * xscale[s] * static_cast<float>(idot);
    }
  }
}

void matmul_i4_idot(float *y, const float *x, const uint8_t *q4,
                    const float *scale, int S, int I, int O) {
  std::vector<int8_t> xq(static_cast<size_t>(S) * I);
  std::vector<float> xscale(S);

#pragma omp parallel for schedule(static)
  for (int s = 0; s < S; ++s) {
    xscale[s] = qrow_i8(x + static_cast<int64_t>(s) * I,
                        xq.data() + static_cast<int64_t>(s) * I, I);
  }

#pragma omp parallel for schedule(static)
  for (int o = 0; o < O; ++o) {
    const uint8_t *wrow = q4 + static_cast<int64_t>(o) * I / 2;
    for (int s = 0; s < S; ++s) {
      const int8_t *xrow = xq.data() + static_cast<int64_t>(s) * I;
      int32_t idot = dot_i4i8_avx512(wrow, xrow, I);
      y[static_cast<int64_t>(s) * O + o] =
          scale[o] * xscale[s] * static_cast<float>(idot);
    }
  }
}

// Deprecated alias; maps to the u8 variant.
void matmul_q8_idot(float *y, const float *x, const int8_t *q,
                    const float *scale, int S, int I, int O) {
  matmul_u8_idot(y, x, q, scale, S, I, O);
}

// ----------------------------------------------------------------------------
// K-quant / IQ dequant — routed through ggml's optimized ARM dequantizers.
// ----------------------------------------------------------------------------
extern "C" int ggml_dequant_row(uint32_t type, const void *src, float *out,
                                int n);

namespace {

void dequant_with_ggml(const char *name, const uint8_t *block, float *y,
                       int64_t k, uint32_t type) {
  if (ggml_dequant_row(type, block, y, static_cast<int>(k)))
    return;
  static std::atomic<bool> logged{false};
  if (should_log_once(logged)) {
    LOG_INFO("[ARM-NEON] %s dequant failed; zero-filling output\n", name);
  }
  std::fill(y, y + k, 0.0f);
}

} // namespace

void dequantize_row_q2_K_avx512(const uint8_t *block, float *y, int64_t k) {
  dequant_with_ggml("Q2_K", block, y, k,
                   static_cast<uint32_t>(10)); // GGML_TYPE_Q2_K
}

void dequantize_row_q3_K_avx512(const uint8_t *block, float *y, int64_t k) {
  dequant_with_ggml("Q3_K", block, y, k,
                   static_cast<uint32_t>(11)); // GGML_TYPE_Q3_K
}

void dequantize_row_iq2_xxs_avx512(const uint8_t *block, float *y, int64_t k) {
  dequant_with_ggml("IQ2_XXS", block, y, k, static_cast<uint32_t>(16));
}

void dequantize_row_iq3_xxs_avx512(const uint8_t *block, float *y, int64_t k) {
  dequant_with_ggml("IQ3_XXS", block, y, k, static_cast<uint32_t>(18));
}

void dequantize_row_iq2_s_avx512(const uint8_t *block, float *y, int64_t k) {
  dequant_with_ggml("IQ2_S", block, y, k, static_cast<uint32_t>(22));
}

void dequantize_row_iq2_xxs_avx512_wide(const uint8_t *block, float *y,
                                        int64_t k) {
  dequantize_row_iq2_xxs_avx512(block, y, k);
}

void dequantize_row_iq3_xxs_avx512_wide(const uint8_t *block, float *y,
                                        int64_t k) {
  dequantize_row_iq3_xxs_avx512(block, y, k);
}

void dequantize_row_iq2_s_avx512_wide(const uint8_t *block, float *y,
                                      int64_t k) {
  dequantize_row_iq2_s_avx512(block, y, k);
}
