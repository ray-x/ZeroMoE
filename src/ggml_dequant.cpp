// src/ggml_dequant.cpp — bridge to libggml's reference dequantize_row_*.
//
// This is the ONLY translation unit that includes ggml-quants.h (which pulls
// the block struct definitions and the IQ lookup grids from ggml-common.h).
// That isolation is deliberate: the rest of MoEStream defines its own
// block_q4_0 in kernels_cpu.hpp, which would collide with ggml's. We LINK
// libggml for these functions — no dequant code is copied.
//
// Type ids are ggml's GGML_TYPE_* values (identical to our GGUFType enum).
#include "ggml-quants.h" // GGML_COMMON_DECL_C -> ggml-common.h + ggml.h

#include <cstdint>
#include <cstdio>
#include <vector>

// Dequantize one contiguous run of n elements of the given ggml type id into
// f32. Returns 1 if ggml has a dequant path for the type, 0 otherwise (caller
// zeroes).
extern "C" int ggml_dequant_row(uint32_t type, const void *src, float *out,
                                int n) {
  if (!src || !out || n <= 0) {
    std::fprintf(
        stderr,
        "[GGML-DEQUANT] Invalid arguments: type=%u src=%p out=%p n=%d\n", type,
        static_cast<const void *>(src), static_cast<void *>(out), n);
    return 0;
  }
  switch (type) {
  case 2:
    dequantize_row_q4_0((const block_q4_0 *)src, out, n);
    return 1;
  case 6:
    dequantize_row_q5_0((const block_q5_0 *)src, out, n);
    return 1;
  case 7:
    dequantize_row_q5_1((const block_q5_1 *)src, out, n);
    return 1;
  case 8:
    dequantize_row_q8_0((const block_q8_0 *)src, out, n);
    return 1;
  case 10:
    dequantize_row_q2_K((const block_q2_K *)src, out, n);
    return 1;
  case 11:
    dequantize_row_q3_K((const block_q3_K *)src, out, n);
    return 1;
  case 12:
    dequantize_row_q4_K((const block_q4_K *)src, out, n);
    return 1;
  case 13:
    dequantize_row_q5_K((const block_q5_K *)src, out, n);
    return 1;
  case 14:
    dequantize_row_q6_K((const block_q6_K *)src, out, n);
    return 1;
  case 16:
    dequantize_row_iq2_xxs((const block_iq2_xxs *)src, out, n);
    return 1;
  case 17:
    dequantize_row_iq2_xs((const block_iq2_xs *)src, out, n);
    return 1;
  case 18:
    dequantize_row_iq3_xxs((const block_iq3_xxs *)src, out, n);
    return 1;
  case 19:
    dequantize_row_iq1_s((const block_iq1_s *)src, out, n);
    return 1;
  case 20:
    dequantize_row_iq4_nl((const block_iq4_nl *)src, out, n);
    return 1;
  case 21:
    dequantize_row_iq3_s((const block_iq3_s *)src, out, n);
    return 1;
  case 22:
    dequantize_row_iq2_s((const block_iq2_s *)src, out, n);
    return 1;
  case 23:
    dequantize_row_iq4_xs((const block_iq4_xs *)src, out, n);
    return 1;
  // FIX: these 3 case labels used to be 24/29/30 -- the STALE numbering
  // gguf_loader.hpp's GGUFType enum itself just got corrected for (see
  // that enum's own comment: found investigating a real DeepSeek-V4-Flash
  // dequant failure). Real, current ggml type ids (confirmed directly
  // against build-metal/_deps/ggml-src/include/ggml.h): IQ1_M=29,
  // TQ1_0=34, TQ2_0=35 -- NOT 24/29/30. Under the old labels, a real
  // BF16 tensor (raw type 30) would have silently hit `case 30:` here and
  // been dequantized as if it were TQ2_0-quantized bytes -- garbage, no
  // error printed. type=%u below now genuinely means what it says.
  case 29:
    dequantize_row_iq1_m((const block_iq1_m *)src, out, n);
    return 1;
  case 34:
    dequantize_row_tq1_0((const block_tq1_0 *)src, out, n);
    return 1;
  case 35:
    dequantize_row_tq2_0((const block_tq2_0 *)src, out, n);
    return 1;
  case 39:
    dequantize_row_mxfp4((const block_mxfp4 *)src, out, n);
    return 1;
  case 40:
    dequantize_row_nvfp4((const block_nvfp4 *)src, out, n);
    return 1;
  // F32/F16/BF16 are handled inline by dequant_row() (no ggml call).
  // IQ2_M: no real GGML_TYPE_* slot in the currently-linked ggml at all
  // (see gguf_loader.hpp's GGUFType enum comment) -- nothing to case on.
  default:
    std::fprintf(stderr, "[GGML-DEQUANT] Unsupported ggml type %u for n=%d\n",
                 type, n);
    return 0;
  }
}

// Encode n f32 elements (n a multiple of the type's block size) into REAL
// ggml-valid quantized bytes at dst, using ggml's own reference quantizer —
// not synthetic/hand-rolled bit patterns. This exists so tests can build
// weight rows that are bit-for-bit what a real GGUF file would contain, then
// feed the SAME bytes through both cpu_quant_matmul (ggml_dequant_row) and a
// CUDA fused kernel and expect matching output. Only covers the types that
// currently have a fused CUDA kernel worth testing this way (see
// src/main_test.cpp); extend as more kernels are added.
// IQ3_XXS needs ggml_quantize_init() first to build its codebook search
// table (kmap_q3xs) -- quantize_row_iq3_xxs_ref asserts/aborts without it.
// Safe to call on every invocation: ggml only builds the table once and a
// no-op on repeat calls for the same type (see ggml.h's documented contract).
//
// IQ2_XXS/XS/S have no single-row "_ref" export -- only the multi-row
// quantize_iq2_xxs/xs/s(src, dst, nrow, n_per_row, imatrix) entry points,
// which additionally REQUIRE a non-null importance-weight array per element
// (ggml asserts "missing quantization weights" otherwise) -- real GGUF
// quantization uses a calibration imatrix here, but a synthetic all-ones
// array is fine for this purpose: only whether the CUDA kernel decodes the
// resulting bytes identically to ggml's own dequantizer is under test, not
// quantization quality. Called with nrow=1 so this still fits the
// one-row-at-a-time shape the rest of this function and its caller use.
extern "C" int ggml_quantize_row_ref(uint32_t type, const float *src, void *dst,
                                     int64_t n) {
  switch (type) {
  case 10:
    quantize_row_q2_K_ref(src, (block_q2_K *)dst, n);
    return 1;
  case 11:
    quantize_row_q3_K_ref(src, (block_q3_K *)dst, n);
    return 1;
  case 12:
    quantize_row_q4_K_ref(src, (block_q4_K *)dst, n);
    return 1;
  case 13:
    quantize_row_q5_K_ref(src, (block_q5_K *)dst, n);
    return 1;
  case 14:
    quantize_row_q6_K_ref(src, (block_q6_K *)dst, n);
    return 1;
  case 16: {
    ggml_quantize_init(GGML_TYPE_IQ2_XXS);
    std::vector<float> imatrix(n, 1.0f);
    quantize_iq2_xxs(src, dst, 1, n, imatrix.data());
    return 1;
  }
  case 17: {
    ggml_quantize_init(GGML_TYPE_IQ2_XS);
    std::vector<float> imatrix(n, 1.0f);
    quantize_iq2_xs(src, dst, 1, n, imatrix.data());
    return 1;
  }
  case 22: {
    ggml_quantize_init(GGML_TYPE_IQ2_S);
    std::vector<float> imatrix(n, 1.0f);
    quantize_iq2_s(src, dst, 1, n, imatrix.data());
    return 1;
  }
  case 18:
    ggml_quantize_init(GGML_TYPE_IQ3_XXS);
    quantize_row_iq3_xxs_ref(src, (block_iq3_xxs *)dst, n);
    return 1;
  // IQ1_S: same shape as the IQ2_XXS/XS/S cases above -- a real grid-search
  // quantizer (quantize_iq1_s, ggml-quants.c), no single-row "_ref" export,
  // requires ggml_quantize_init(GGML_TYPE_IQ1_S) first (builds the 2048-
  // entry kmap_1bit search table) and a non-null quant_weights array (a
  // synthetic all-ones array is fine here -- only whether the CUDA kernel
  // decodes the resulting real bytes identically to ggml's own dequantizer
  // is under test, not quantization quality, same rationale as IQ2_XXS/XS/S
  // above). Added 2026-09 while wiring up cuda/iq1_dequant.cu's real
  // launch_iq1s_matmul so Test 5 can drive it with real ggml-valid bytes
  // instead of silently SKIPPING (this function previously had no case 19
  // at all, so any earlier IQ1_S test attempt would have returned 0 here).
  case 19: {
    ggml_quantize_init(GGML_TYPE_IQ1_S);
    std::vector<float> imatrix(n, 1.0f);
    quantize_iq1_s(src, dst, 1, n, imatrix.data());
    return 1;
  }
  // IQ1_M: same shape as IQ1_S immediately above -- a real grid-search
  // quantizer (quantize_iq1_m, ggml-quants.c) sharing the SAME
  // ggml_quantize_init(GGML_TYPE_IQ1_S) table build (IQ1_M indexes the
  // identical iq1s_grid, confirmed by reading dequantize_row_iq1_m
  // directly -- no separate GGML_TYPE_IQ1_M init call needed or
  // available). Added 2026-09 alongside IQ1_S's own case so Test 5 can
  // drive the new launch_iq1m_matmul with real ggml-valid bytes instead
  // of silently SKIPPING -- this function previously had no case 29 at
  // all, so any Test 5 attempt at IQ1_M would have returned 0 here.
  case 29: {
    ggml_quantize_init(GGML_TYPE_IQ1_S);
    std::vector<float> imatrix(n, 1.0f);
    quantize_iq1_m(src, dst, 1, n, imatrix.data());
    return 1;
  }
  // IQ3_S: same shape as IQ1_S/IQ2_XXS/XS/S above -- a real grid-search
  // quantizer (quantize_iq3_s, ggml-quants.c), needs ggml_quantize_init()
  // first (builds the IQ3_S codebook search table) and a non-null
  // quant_weights array (synthetic all-ones is fine, same rationale as
  // every other case in this block: only whether the CUDA kernel decodes
  // the resulting real bytes identically to ggml's own dequantizer is under
  // test here, not quantization quality). Added while wiring up a real
  // launch_iq3s_matmul (cuda/iq3_matmul.cu) -- this function previously had
  // no case 21 at all, so any Test 5 attempt at IQ3_S would have silently
  // returned 0 (SKIPPED) here.
  case 21: {
    ggml_quantize_init(GGML_TYPE_IQ3_S);
    std::vector<float> imatrix(n, 1.0f);
    quantize_iq3_s(src, dst, 1, n, imatrix.data());
    return 1;
  }
  // IQ4_XS: a fixed 16-entry lookup table (kvalues_iq4nl) + per-block
  // linear scale, NOT the harder IQ2/IQ3 variable-codebook family --
  // ggml exposes a real single-row _ref quantizer (no imatrix
  // needed), same shape as the Q2_K..Q6_K cases above. Added
  // specifically to unblock benchmarking this type's CPU dequant
  // throughput (it was previously the only routed-expert-relevant
  // type this bridge couldn't encode test data for).
  case 23:
    quantize_row_iq4_xs_ref(src, (block_iq4_xs *)dst, n);
    return 1;
  // IQ4_NL: same 16-entry kvalues_iq4nl codebook as IQ4_XS above, but a
  // 32-element block with a single fp16 scale (18 bytes) instead of 256 with
  // hierarchical sub-scales. Real single-row _ref quantizer, no imatrix.
  // Added 2026-09-04 to give Test 5 real bytes for the new
  // launch_iq4_nl_matmul -- IQ4_NL is the down projection on all 48 MoE
  // layers of Qwen3.8-Flash-Next UD-IQ1_M.
  case 20:
    quantize_row_iq4_nl_ref(src, (block_iq4_nl *)dst, n);
    return 1;
  // MXFP4: fixed 16-entry table + one E8M0 scale byte per 32-element block
  // (see cuda/mxfp4_matmul.cu's header comment for the full algorithm) --
  // same shape as the K-quant/IQ4_XS cases above, a real single-row _ref
  // quantizer, no imatrix needed. Added to let src/main_test.cpp's Test 5
  // drive the new (currently dormant) GPU kernel with real ggml-valid bytes.
  case 39:
    quantize_row_mxfp4_ref(src, (block_mxfp4 *)dst, n);
    return 1;
  // Q8_0: a real single-row _ref quantizer, same simple shape as the cases
  // above (no imatrix, no init call needed). Was missing entirely --
  // src/main_test.cpp's Test 5 had tolerance handling already prepared for
  // this type (see its comment on Q6_K/Q8_0's lossy-CPU-reference caveat)
  // but no case ever actually ran, since this function silently returned 0
  // and the test just SKIPPED. Added while touching cuda/q8_0_matmul.cu's
  // reuse-across-S kernel, so that change has real GPU-vs-CPU-reference
  // coverage instead of relying on lint + a live model run alone.
  case 7:
    // Q5_1 -- added 2026-09-04 alongside cuda/q5_1_matmul.cu so Test 5 can
    // build a real quantized weight buffer for it. Same simple single-row
    // shape as the Q8_0 case below; no imatrix, no ggml_quantize_init.
    quantize_row_q5_1_ref(src, (block_q5_1 *)dst, n);
    return 1;
  case 8:
    quantize_row_q8_0_ref(src, (block_q8_0 *)dst, n);
    return 1;
  default:
    return 0;
  }
}
