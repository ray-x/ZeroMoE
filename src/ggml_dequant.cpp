// src/ggml_dequant.cpp — bridge to libggml's reference dequantize_row_*.
//
// This is the ONLY translation unit that includes ggml-quants.h (which pulls the
// block struct definitions and the IQ lookup grids from ggml-common.h). That
// isolation is deliberate: the rest of MoEStream defines its own block_q4_0 in
// kernels_cpu.hpp, which would collide with ggml's. We LINK libggml for these
// functions — no dequant code is copied.
//
// Type ids are ggml's GGML_TYPE_* values (identical to our GGUFType enum).
#include "ggml-quants.h"   // GGML_COMMON_DECL_C -> ggml-common.h + ggml.h

#include <cstdint>
#include <vector>

// Dequantize one contiguous run of n elements of the given ggml type id into f32.
// Returns 1 if ggml has a dequant path for the type, 0 otherwise (caller zeroes).
extern "C" int ggml_dequant_row(uint32_t type, const void* src, float* out, int n) {
    switch (type) {
        case 2:  dequantize_row_q4_0   ((const block_q4_0   *)src, out, n); return 1;
        case 6:  dequantize_row_q5_0   ((const block_q5_0   *)src, out, n); return 1;
        case 8:  dequantize_row_q8_0   ((const block_q8_0   *)src, out, n); return 1;
        case 10: dequantize_row_q2_K   ((const block_q2_K   *)src, out, n); return 1;
        case 11: dequantize_row_q3_K   ((const block_q3_K   *)src, out, n); return 1;
        case 12: dequantize_row_q4_K   ((const block_q4_K   *)src, out, n); return 1;
        case 13: dequantize_row_q5_K   ((const block_q5_K   *)src, out, n); return 1;
        case 14: dequantize_row_q6_K   ((const block_q6_K   *)src, out, n); return 1;
        case 16: dequantize_row_iq2_xxs((const block_iq2_xxs*)src, out, n); return 1;
        case 17: dequantize_row_iq2_xs ((const block_iq2_xs *)src, out, n); return 1;
        case 18: dequantize_row_iq3_xxs((const block_iq3_xxs*)src, out, n); return 1;
        case 19: dequantize_row_iq1_s  ((const block_iq1_s  *)src, out, n); return 1;
        case 20: dequantize_row_iq4_nl ((const block_iq4_nl *)src, out, n); return 1;
        case 21: dequantize_row_iq3_s  ((const block_iq3_s  *)src, out, n); return 1;
        case 22: dequantize_row_iq2_s  ((const block_iq2_s  *)src, out, n); return 1;
        case 23: dequantize_row_iq4_xs ((const block_iq4_xs *)src, out, n); return 1;
        case 24: dequantize_row_iq1_m  ((const block_iq1_m  *)src, out, n); return 1;
        case 29: dequantize_row_tq1_0  ((const block_tq1_0  *)src, out, n); return 1;
        case 30: dequantize_row_tq2_0  ((const block_tq2_0  *)src, out, n); return 1;
        // F32/F16/BF16 are handled inline by dequant_row() (no ggml call).
        // 31 (IQ2_M): ggml has no standalone dequantize_row export in this build.
        default: return 0;
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
extern "C" int ggml_quantize_row_ref(uint32_t type, const float* src, void* dst, int64_t n) {
    switch (type) {
        case 10: quantize_row_q2_K_ref(src, (block_q2_K*)dst, n); return 1;
        case 11: quantize_row_q3_K_ref(src, (block_q3_K*)dst, n); return 1;
        case 12: quantize_row_q4_K_ref(src, (block_q4_K*)dst, n); return 1;
        case 13: quantize_row_q5_K_ref(src, (block_q5_K*)dst, n); return 1;
        case 14: quantize_row_q6_K_ref(src, (block_q6_K*)dst, n); return 1;
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
            quantize_row_iq3_xxs_ref(src, (block_iq3_xxs*)dst, n);
            return 1;
        default: return 0;
    }
}
