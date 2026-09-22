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
// IQ2_XS: verbatim port of cuda/iq2_matmul.cu's dequant_iq2_xs_coop. `block`
// layout: d(2) + qs[32] uint16 (64B) + scales[8] = 74B/block (2.3125
// bits/weight). Grid/sign index both come from one qs[] entry per l (& 511,
// >> 9) -- simpler than IQ2_S below, no separate qh array.
void dequantize_row_iq2_xs_avx512(const uint8_t *block, float *y, int64_t k);
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
// EXPERIMENTAL VNNI prototype -- feasibility check only, see its own header
// comment (kernels_cpu.cpp). NOT wired into any production dispatch path.
float dot_iq2_xxs_vnni_prototype(const uint8_t *block, const block_q8_0 *xq, int64_t k);
// EXPERIMENTAL AVX2 prototype #2 (ggml upstream port, no VNNI) -- see its
// own header comment (kernels_cpu.cpp). NOT wired into any production path.
// Layout-identical local mirror of ggml's own block_q8_K (ggml-common.h),
// declared here (not just in kernels_cpu.cpp) so callers like main_test.cpp
// can allocate/index it directly -- quantize_row_q8_K_ref only needs a
// layout-compatible pointer, not the exact type name (GGML_API extern "C").
struct block_q8_K_local {
  float d;
  int8_t qs[256];
  int16_t bsums[16];
};
extern "C" void quantize_row_q8_K_ref(const float *x, block_q8_K_local *y, int64_t k);
float dot_iq2_xxs_avx2_prototype(const uint8_t *block, const block_q8_K_local *y, int64_t k);
float dot_iq3_xxs_avx2_prototype(const uint8_t *block, const block_q8_K_local *y, int64_t k);
// Production-callable OMP-parallel matmul wrappers around the two dot
// functions above -- called from kquant_dequant.hpp's cpu_quant_matmul,
// gated behind EngineConfig::cpu_dot_ggml_avx2 (dense_dispatch.hpp's
// cpu_dot_ggml_avx2_enabled()). See their own header comments (kernels_cpu.cpp).
void matmul_iq2_xxs_ggml_avx2(float *y, const float *x, const uint8_t *W_quantized,
                              int S, int I, int O);
void matmul_iq3_xxs_ggml_avx2(float *y, const float *x, const uint8_t *W_quantized,
                              int S, int I, int O);
// Qwen4Exp/UD-IQ1_M CPU fallback using ggml's production IQ1_M x Q8_K
// quantized-activation path.  Kept as a separate opt-in symbol so GLM and
// other model CPU paths remain unchanged.
// Returns the effective team size used by Qwen's direct grouped kernels;
// MOESTREAM_QWEN_CPU_THREADS is a Qwen-only diagnostic override.
int qwen_cpu_grouped_threads();
bool matmul_iq1_m_ggml_q8k(float *y, const float *x, const uint8_t *W_quantized,
                           int S, int I, int O);
// Qwen-only AVX-512-width IQ1_M x Q8_K probe.  This intentionally remains a
// separate entry point from the GGML adapter until its numerical and
// end-to-end performance checks justify making it the default.
bool matmul_iq1_m_avx512_q8k_grouped(
    float *const *y, const float *x, const uint8_t *const *W_quantized,
    int n_matrices, int S, int I, int O);
// Grouped Qwen CPU route: one Q8_K activation quantization feeds several
// compact expert matrices. `W_quantized[m]` points to one [O,I] matrix and
// `y[m]` points to its [S,O] result. This removes the per-expert OpenMP
// fork/join and activation quantization that made the old CPU fallback much
// slower than BeeLlama's ggml MUL_MAT_ID path.
bool matmul_iq1_m_ggml_q8k_grouped(float *const *y, const float *x,
                                   const uint8_t *const *W_quantized,
                                   int n_matrices, int S, int I, int O);
// Qwen decode gate+up pair with the same activation: quantize the input once
// and execute both IQ1_M projections in one OpenMP region. This is restricted
// to the Qwen CPU-affinity path; it does not alter generic model kernels.
bool matmul_iq1_m_ggml_q8k_grouped_pair(
    float *const *gate_y, float *const *up_y, const float *x,
    const uint8_t *const *gate_weights, const uint8_t *const *up_weights,
    int n_matrices, int S, int I, int O);
// Same grouped kernel when each matrix has its own activation row (the down
// projection after per-expert SwiGLU).
bool matmul_iq1_m_ggml_q8k_grouped_inputs(
    float *const *y, const float *const *x,
    const uint8_t *const *W_quantized, int n_matrices, int S, int I, int O);
// Grouped Qwen CPU routes for the other quantized projections present in the
// Qwen4Exp checkpoint.  IQ2/ IQ3 use the same Q8_K activation contract as the
// existing AVX2 per-matrix wrappers; IQ4_NL uses ggml's Q8_0 dot kernel for
// the down projection, whose activation differs after each expert's SwiGLU.
bool matmul_iq2_xxs_ggml_avx2_grouped(
    float *const *y, const float *x, const uint8_t *const *W_quantized,
    int n_matrices, int S, int I, int O);
bool matmul_iq3_xxs_ggml_avx2_grouped(
    float *const *y, const float *x, const uint8_t *const *W_quantized,
    int n_matrices, int S, int I, int O);
// Qwen-only grouped wrappers around the production GGML CPU dot kernels.
// Unlike the older AVX2 prototypes above, these use the same dispatched
// ggml_vec_dot_iq*_q8_K implementation as BeeLlama's MUL_MAT_ID path.
bool matmul_iq2_xxs_ggml_q8k_grouped(
    float *const *y, const float *x, const uint8_t *const *W_quantized,
    int n_matrices, int S, int I, int O);
bool matmul_iq3_xxs_ggml_q8k_grouped(
    float *const *y, const float *x, const uint8_t *const *W_quantized,
    int n_matrices, int S, int I, int O);
bool matmul_iq4_nl_ggml_q8_0_grouped_inputs(
    float *const *y, const float *const *x,
    const uint8_t *const *W_quantized, int n_matrices, int S, int I, int O);
// Qwen-only decode probe: keep IQ1_M gate/up, SwiGLU, IQ4_NL activation
// quantization, and down projection inside one OpenMP team.  This is an
// optional A/B entry point for the direct DDR path; callers must keep the
// existing grouped implementation as the fallback until numerical and
// end-to-end measurements prove it beneficial.
bool matmul_qwen_iq1_m_iq4_nl_fused(
    float *const *gate_y, float *const *up_y, float *const *down_y,
    const float *x, const uint8_t *const *gate_weights,
    const uint8_t *const *up_weights, const uint8_t *const *down_weights,
    int n_matrices, int I, int gate_O, int down_O, int n_threads = 0);
// Qwen-only sibling for the IQ2_XXS gate/up layers in the same checkpoint.
// Uses GGML's production IQ2_XXS x Q8_K dot kernel, but keeps the two
// projections, activation, down-input quantization, and IQ4_NL down projection
// inside one OpenMP team just like the IQ1_M fused path above.
bool matmul_qwen_iq2_xxs_iq4_nl_fused(
    float *const *gate_y, float *const *up_y, float *const *down_y,
    const float *x, const uint8_t *const *gate_weights,
    const uint8_t *const *up_weights, const uint8_t *const *down_weights,
    int n_matrices, int I, int gate_O, int down_O, int n_threads = 0);
float dot_iq3_xxs_f32_avx512(const uint8_t *block, const float *x, int64_t k);
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
