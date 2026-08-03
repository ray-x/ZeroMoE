#pragma once
#include <cstdint>

// Platform backend selection:
//   * On x86-64 with AVX-512F, src/kernels_cpu.cpp provides these symbols.
//   * On Apple Silicon / ARM64, src/kernels_cpu_neon.cpp provides the same
//     symbols so that callers do not need platform #ifdefs.  The historical
//     "_avx512" suffixes are therefore a misnomer on ARM; they are kept as a
//     compatibility shim while the codebase is being modernized.
//
// CMake controls which implementation file is compiled:
//   - MOESTREAM_CPU_ONLY=ON forces the CPU-only path (auto-set on macOS).
//   - On arm64 Apple, kernels_cpu_neon.cpp is selected.
//   - Otherwise kernels_cpu.cpp (AVX-512) is selected.

struct block_q4_0 {
  uint16_t d;
  uint8_t qs[16];
};

struct block_q8_0 {
  uint16_t d;
  int8_t qs[32];
};

// Standard Q6_K block (256 elements)
struct block_q6_K {
  uint8_t ql[128];   // Lower 4 bits
  uint8_t qh[64];    // Upper 2 bits
  int8_t scales[16]; // Scales for each 16-element sub-block
  uint16_t d;        // FP16 block scale
};

// Activation block matched to Q6_K granularity
struct block_xq_q6 {
  int8_t qs[256]; // Quantized activations
  float sums[16]; // sum(x) for each 16-element sub-block (for the VNNI offset)
  float scales[16]; // FP32 scales for each 16-element sub-block
};

float fp16_to_fp32(uint16_t h);
uint16_t fp32_to_fp16(float f);
float dot_q4_0_q8_0_avx512(const block_q4_0 *w, const block_q8_0 *x,
                           int num_blocks);
void quantize_activations_q8_0(const float *__restrict__ x,
                               block_q8_0 *__restrict__ y, int num_elements);

// --- High-performance matmul variants ---
void matmul(float *y, const float *x, const float *W, int S, int I, int O);
void rmsnorm_row(float *out, const float *x, const float *w, int D, float eps);
void softmax_row(float *x, int size);

// f32 x f32 dot product, AVX-512 FMA (16 lanes/iter) with a scalar tail.
// Shared by matmul() and cpu_quant_matmul()'s post-dequant dot — the CPU
// expert-FFN fallback path previously did a naive scalar `for i: acc += ...`
// per output row per token, which is the actual hot loop when an expert's
// quant type has no GPU kernel (e.g. IQ2/IQ3 grid types before their fused
// kernels landed) or the GPU path is otherwise unavailable.
float dot_f32_avx512(const float *a, const float *b, int n);

// IDOT: integer dot product kernels (1.4-2.5x speedup over f32 path)
// Quantizes activations to int8 per-row, then uses VNNI vpdpbusd for
// int8×int8→int32. No f32 conversion in the hot loop.
float qrow_i8(const float *x, int8_t *q, int I);
int32_t dot_i8i8_avx512(const int8_t *w, const int8_t *x, int I);
int32_t dot_i4i8_avx512(const uint8_t *w4, const int8_t *x, int I);

// IDOT matmul: y[S,O] = x[S,I] @ W^T, W quantized int8 per-row + scale[O]
// Deprecated:
void matmul_q8_idot(float *y, const float *x, const int8_t *q,
                    const float *scale, int S, int I, int O);
// Use this:
void matmul_u8_idot(float *y, const float *x, const int8_t *q,
                    const float *scale, int S, int I, int O);

// IDOT matmul for int4 packed weights (2 values/byte) + scale[O]
void matmul_i4_idot(float *y, const float *x, const uint8_t *q4,
                    const float *scale, int S, int I, int O);

// SiLU activation: x * sigmoid(x)
void silu_inplace(float *x, int n);
// Element-wise multiply: out[i] = a[i] * b[i]
void elem_mul(float *out, const float *a, const float *b, int n);

// AVX-512 ports of ggml's dequantize_row_q2_K / dequantize_row_q3_K
// (ggml/src/ggml-quants.c -- the plain scalar reference this project links
// against, never vectorized upstream either). Verbatim algorithmic port:
// same nesting, same scale/min unpacking, same element order -- only the
// innermost 16-element unpack+affine-transform loop is vectorized, using
// separate multiply+subtract (not FMA) to bit-exactly match the scalar
// reference's two-rounding-step arithmetic. `block` points to raw
// contiguous quantized bytes (row_bytes(Q2_K/Q3_K, I) per row); struct
// layout confirmed against the already-verified block_q2_K_cu/block_q3_K_cu
// in cuda/kquant_matmul.cu (Q2_K: scales[16] qs[64] d dmin, 84B/block;
// Q3_K: hmask[32] qs[64] scales[12] d, 110B/block -- the Q3_K field order
// was the exact bug fixed earlier this session, see CLAUDE.md).
void dequantize_row_q2_K_avx512(const uint8_t *block, float *y, int64_t k);
void dequantize_row_q3_K_avx512(const uint8_t *block, float *y, int64_t k);

// AVX-512/AVX2 ports of ggml's dequantize_row_iq2_xxs / dequantize_row_iq3_xxs
// -- the ACTUAL routed-expert quant types for this project's target GLM-5.2
// UD-IQ2_M checkpoints (71/76 MoE layers: gate=up=IQ2_XXS, down=IQ3_XXS; see
// the real [QUANT] startup report, CLAUDE.md). The Q2_K/Q3_K functions above
// were vectorized first based on a config-file comment that turned out to be
// wrong for this specific checkpoint -- IQ2_XXS/IQ3_XXS is the format that
// actually dominates the CPU expert-FFN fallback cost. Codebook/grid-based
// (NOT linear scale+shift like K-quants): each 8-element group looks up one
// entry in a shared table (`src/ggml_grids.cpp`'s ms_grid_* host accessors --
// the SAME table data the already-verified CUDA kernels use, cuda/iq2_matmul.cu
// / iq3_matmul.cu, no table content duplicated/hand-transcribed here) and
// applies a per-element sign flip from a shared 128-entry sign table. Ported
// verbatim from those CUDA kernels (structurally identical to ggml's own
// dequantize_row_iq2_xxs/iq3_xxs). `block` layout: IQ2_XXS = d(2) + qs[32]
// uint16 (66B/block); IQ3_XXS = d(2) + qs[96] (98B/block).
void dequantize_row_iq2_xxs_avx512(const uint8_t *block, float *y, int64_t k);
void dequantize_row_iq3_xxs_avx512(const uint8_t *block, float *y, int64_t k);
// IQ2_S: verbatim port of cuda/iq2_matmul.cu's dequant_iq2_s_block. `block`
// layout: d(2) + qs[64] + qh[8] + scales[8] = 82B/block (2.5625 bits/weight).
void dequantize_row_iq2_s_avx512(const uint8_t *block, float *y, int64_t k);

// EXPERIMENTAL widened (512-bit/16-lane, vs the above's 256-bit/8-lane)
// siblings -- NOT wired into dequant_row()'s dispatch, see kernels_cpu.cpp's
// header comment above their definitions for why. Exposed here only so
// main_test.cpp's Test 8 benchmark can call them directly for a throughput
// comparison against the 8-wide versions.
void dequantize_row_iq2_xxs_avx512_wide(const uint8_t *block, float *y,
                                        int64_t k);
void dequantize_row_iq3_xxs_avx512_wide(const uint8_t *block, float *y,
                                        int64_t k);
void dequantize_row_iq2_s_avx512_wide(const uint8_t *block, float *y,
                                      int64_t k);

float dot_iq2_xxs_f32_avx512(const uint8_t *block, const float *x, int64_t k);
void matmul_iq2_xxs_fused(float *y, const float *x, const uint8_t *W_quantized,
                          int S, int I, int O);
void matmul_iq3_xxs_fused(float *y, const float *x, const uint8_t *W_quantized,
                          int S, int I, int O);
void matmul_q5_k_fused(float *y, const float *x, const uint8_t *W_quantized,
                       int S, int I, int O);
void matmul_q6_k_fused(float *y, const float *x, const uint8_t *W_quantized,
                       int S, int I, int O);
void matmul_q8_0_fused(float *y, const float *x, const uint8_t *W_quantized,
                       int S, int I, int O);

void matmul_f32_avx512(float *y, const float *x, const float *w, int S, int I,
                       int O);

// ----------------------------------------------------------------------------
// Platform-neutral aliases for new / modernized code.
// Older call sites still use the historical names above; these aliases let new
// code avoid the misleading "avx512" suffix.
// ----------------------------------------------------------------------------
inline float dot_f32(const float *a, const float *b, int n) {
  return dot_f32_avx512(a, b, n);
}
inline void matmul_f32(float *y, const float *x, const float *w, int S, int I,
                       int O) {
  matmul(y, x, w, S, I, O);
}