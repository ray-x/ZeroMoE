// kquant_dequant.hpp — CPU reference dequant + dequant-then-matmul.
//
// All quantized types are dequantized by libggml's reference dequantize_row_*
// (linked via the ggml_dequant_row bridge in src/ggml_dequant.cpp — NOT copied).
// F32/F16/BF16 are handled inline. This is the correctness reference and the CPU
// fallback; the GPU fused kernels in cuda/kquant_matmul.cu are the hot path for
// dense/expert K-quants and must produce identical output.
#pragma once
#include "kernels_cpu.hpp"   // fp16_to_fp32, matmul
#include "engine.hpp"        // QuantizedTensor, GGUFType
#include "logger.hpp"
#include "perf_timer.hpp"    // t_cpu_dequant/t_cpu_dot breakdown
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include <algorithm>
#include <chrono>

// Bridge into libggml (src/ggml_dequant.cpp). Returns 1 on success, 0 if ggml
// has no dequant for this type id.
extern "C" int ggml_dequant_row(uint32_t type, const void* src, float* out, int n);

// True if execution_layer.cpp's gpu_matmul_dispatch has a real fused kernel
// for this type — kept in sync with that switch by hand (single source of
// truth would require the cuda/*.cu launcher externs here too, which this
// header intentionally doesn't pull in). Shared by every VRAM-promotion
// decision point (ram_manager.hpp's hit-driven promote, execution_layer.cpp's
// cold-start warm-up, async_prefetcher.hpp's speculative promotion) so none
// of them waste PCIe bandwidth + VRAM cache slots promoting an expert whose
// quant type can never actually run on GPU regardless of residency.
inline bool gpu_type_supported(GGUFType type) {
    switch (type) {
        case GGUFType::Q4_0: case GGUFType::Q2_K: case GGUFType::Q3_K:
        case GGUFType::Q4_K: case GGUFType::Q5_K: case GGUFType::Q6_K:
        case GGUFType::IQ2_XXS: case GGUFType::IQ2_XS: case GGUFType::IQ2_S:
        // IQ3_XXS: cuda/iq3_matmul.cu's launch_iq3_xxs_matmul, a verbatim
        // port of ggml's dequantize_row_iq3_xxs (ggml/src/ggml-quants.c) in
        // its own dedicated file (this project has no relocatable device
        // code / -rdc=true, so it keeps its own __constant__ copies of the
        // sign/mask tables rather than sharing iq2_matmul.cu's — see that
        // file's own header comment). Added because this was, by
        // far, the single biggest CPU-bound bottleneck observed in practice:
        // one GLM-5.2 UD-IQ2_M checkpoint had 71/76 MoE layers using
        // gate/up=IQ2_XXS + down=IQ3_XXS, meaning virtually the entire
        // routed-expert compute was CPU-only until this landed.
        case GGUFType::IQ3_XXS:
            return true;
        // IQ4_XS: RE-ENABLED. Two prior incidents (layer=40, then layer=7)
        // both produced NaN with this enabled despite Test 5's isolated
        // kernel test passing -- but the real root cause, found by
        // re-reading expert_ffn_resolve_gpu_loc (src/execution_layer.cpp)
        // after the NaN was reported as non-deterministic/code-version-
        // independent, was a genuine RACE CONDITION shared by EVERY
        // GPU-dispatched expert of ANY quant type, not anything specific
        // to IQ4_XS: hold_for_compute() was only ever called for the
        // decode-time-cold-miss path, leaving the (far more common)
        // already-VRAM-resident and S>=4 cold-batch-upload paths
        // completely unprotected against concurrent eviction mid-compute.
        // IQ4_XS was never uniquely at fault -- just an expert type like
        // any other, sitting in the same unprotected state as everything
        // else. See CLAUDE.md for the full incident writeup and the fix
        // (hold_for_compute() now called uniformly for every path that
        // resolves a GPU location). Re-enabled now that this is fixed;
        // still verify with a real run and watch "Temp-pin skips" in
        // dump_report() before trusting further.
        case GGUFType::IQ4_XS:
            return true;
        // No GPU kernel implemented anywhere in cuda/*.cu for these: IQ3_S,
        // IQ4_NL, TQ1_0, TQ2_0, IQ2_M. See gpu_unsupported_reason() for why
        // each specific one isn't supported yet.
        default: return false;
    }
}

// Human-readable name for diagnostics — logs otherwise only show the raw
// numeric GGUFType id (e.g. "gate=16"), which means nothing without cross-
// referencing gguf_loader.hpp's enum by hand.
inline const char* gpu_type_name(GGUFType type) {
    switch (type) {
        case GGUFType::F32:     return "F32";
        case GGUFType::F16:     return "F16";
        case GGUFType::Q4_0:    return "Q4_0";
        case GGUFType::Q4_1:    return "Q4_1";
        case GGUFType::Q5_0:    return "Q5_0";
        case GGUFType::Q5_1:    return "Q5_1";
        case GGUFType::Q8_0:    return "Q8_0";
        case GGUFType::Q8_1:    return "Q8_1";
        case GGUFType::Q2_K:    return "Q2_K";
        case GGUFType::Q3_K:    return "Q3_K";
        case GGUFType::Q4_K:    return "Q4_K";
        case GGUFType::Q5_K:    return "Q5_K";
        case GGUFType::Q6_K:    return "Q6_K";
        case GGUFType::Q8_K:    return "Q8_K";
        case GGUFType::IQ2_XXS: return "IQ2_XXS";
        case GGUFType::IQ2_XS:  return "IQ2_XS";
        case GGUFType::IQ3_XXS: return "IQ3_XXS";
        case GGUFType::IQ1_S:   return "IQ1_S";
        case GGUFType::IQ4_NL:  return "IQ4_NL";
        case GGUFType::IQ3_S:   return "IQ3_S";
        case GGUFType::IQ2_S:   return "IQ2_S";
        case GGUFType::IQ4_XS:  return "IQ4_XS";
        case GGUFType::IQ1_M:   return "IQ1_M";
        case GGUFType::BF16:    return "BF16";
        case GGUFType::TQ1_0:   return "TQ1_0";
        case GGUFType::TQ2_0:   return "TQ2_0";
        case GGUFType::IQ2_M:   return "IQ2_M";
        default:                return "UNKNOWN";
    }
}

// Effective bits-per-weight, computed from the same block sizes as row_bytes()
// below (bytes_per_block * 8 / elements_per_block) — for answering "is this
// about storage efficiency". Rounded to 2 decimals worth of precision isn't
// needed here; these are exact ratios from the GGUF quant spec.
inline double gpu_type_bits_per_weight(GGUFType type) {
    switch (type) {
        case GGUFType::F32:     return 32.0;
        case GGUFType::F16: case GGUFType::BF16: return 16.0;
        case GGUFType::Q8_0:    return 8.5;
        case GGUFType::Q6_K:    return 6.5625;
        case GGUFType::Q5_K: case GGUFType::Q5_0: return 5.5;
        case GGUFType::Q4_K: case GGUFType::Q4_0: return 4.5;
        case GGUFType::IQ4_NL:  return 4.5;
        case GGUFType::IQ4_XS:  return 4.25;
        case GGUFType::Q3_K: case GGUFType::IQ3_S: return 3.4375;
        case GGUFType::IQ3_XXS: return 3.0625;
        case GGUFType::Q2_K: case GGUFType::IQ2_M: return 2.625;
        case GGUFType::IQ2_S:   return 2.5625;
        case GGUFType::IQ2_XS:  return 2.3125;
        case GGUFType::IQ2_XXS: return 2.0625;
        case GGUFType::TQ2_0:   return 2.0625;
        case GGUFType::IQ1_M:   return 1.75;
        case GGUFType::TQ1_0:   return 1.6875;
        case GGUFType::IQ1_S:   return 1.5625;
        default:                return 0.0;  // unknown — don't claim a number
    }
}

// WHY this type has no fused GPU matmul kernel — answers "is it because of
// the bit-packing" directly. Short answer: no, not fundamentally — IQ2_XXS
// at 2.06 bits/weight (even smaller than IQ3_S's 3.44) DOES have a kernel
// (cuda/iq2_matmul.cu), so sub-byte packing alone isn't the blocker. Two
// genuinely different reasons hide behind "IQ-series", verified against
// ggml's actual dequant source (ggml/src/ggml-quants.c) rather than assumed:
//   - IQ3_S (IQ3_XXS was in this category too, until cuda/iq2_matmul.cu's
//     launch_iq3_xxs_matmul was added): each block picks a combination from
//     a LARGE variable grid table (256-1024 entries, ggml-common.h's
//     iq3xxs_grid/iq3s_grid) plus separate sign bits to reconstruct values —
//     genuinely the same family/difficulty as the already-supported
//     IQ2_XXS/XS/S grids (and now IQ3_XXS).
//   - IQ4_NL/IQ4_XS: dequantize_row_iq4_nl/_xs BOTH just index a single
//     FIXED 16-entry table (kvalues_iq4nl, identical for every block) times
//     a per-block linear scale — structurally close to the ALREADY-SUPPORTED
//     K-quants (Q4_K/Q5_K/Q6_K), not the IQ2/IQ3 grid family. IQ4_XS has a
//     kernel again (cuda/iq4xs_matmul.cu's launch_iq4_xs_matmul,
//     re-enabled after the real bug — a shared eviction race, not this
//     kernel — was found and fixed; see gpu_type_supported()'s own comment
//     and CLAUDE.md); IQ4_NL is the one still genuinely missing.
inline const char* gpu_unsupported_reason(GGUFType type) {
    if (gpu_type_supported(type)) return "supported";
    switch (type) {
        case GGUFType::IQ3_S:
            return "non-linear codebook/grid quantization (same family as the "
                   "already-supported IQ2_XXS/XS/S and IQ3_XXS: a large "
                   "256-1024 entry variable grid + sign bits per block) -- no "
                   "fused CUDA kernel written for THIS specific format yet. "
                   "Not a fundamental bit-width limit; it's unimplemented, "
                   "not unimplementable.";
        // IQ4_XS moved to gpu_type_supported()'s TRUE case above (re-
        // enabled) -- this reason string only applies to IQ4_NL now, since
        // gpu_unsupported_reason() already returns early with "supported"
        // for anything gpu_type_supported() accepts.
        case GGUFType::IQ4_NL:
            return "fixed 16-entry lookup table (kvalues_iq4nl) + per-block "
                   "linear scale -- structurally close to the ALREADY-SUPPORTED "
                   "Q4_K/Q5_K/Q6_K/IQ4_XS kernels. No fused CUDA kernel written "
                   "yet for THIS specific block layout (IQ4_NL's is smaller, "
                   "32 elements/block vs IQ4_XS's 256).";
        case GGUFType::IQ1_S: case GGUFType::IQ1_M: case GGUFType::TQ1_0:
            return "1-2 bit ternary/codebook format -- CUDA kernels for these "
                   "actually EXIST (cuda/iq1_dequant.cu: launch_iq1s_matmul / "
                   "launch_iq1m_matmul / launch_tq1_matmul) but are not wired "
                   "into the dispatch table yet -- pure plumbing gap.";
        case GGUFType::TQ2_0:
            return "ternary quantization -- no fused CUDA kernel implemented.";
        case GGUFType::IQ2_M:
            return "codebook quantization -- no ggml CPU dequant export exists "
                   "either (see kquant_dequant.hpp's dequant_row), so this type "
                   "is unsupported on BOTH paths, not just GPU.";
        default:
            return "no fused CUDA kernel implemented for this type.";
    }
}

static inline float bf16_to_fp32(uint16_t b) {
    uint32_t u = ((uint32_t)b) << 16;  // bf16 = top 16 bits of fp32
    float f; std::memcpy(&f, &u, sizeof(f));
    return f;
}

// Round-to-nearest-even truncation of an f32 to bf16 (top 16 bits, with the
// standard "round half to even" bias added before truncating rather than a
// flat chop). NOT currently wired into the live fast-path: was used by a
// bf16 variant of CudaManager::upload_dequantized_f32 to halve the
// attn_output fast-path's VRAM cost, but that variant produced real NaN
// corruption on real hardware and was reverted (see CLAUDE.md and
// upload_dequantized_f32's own comment). The rounding math itself was NOT
// identified as the cause on static review -- now exercised directly by
// src/main_test.cpp's Test 9 (test_bf16_cublas_dense_matmul), which tests
// cublasGemmEx's BF16-weight/F32-activation path in isolation. Do not wire
// this back into the live fast-path until Test 9 has actually been run and
// confirmed passing on real Linux/CUDA hardware.
static inline uint16_t fp32_to_bf16(float f) {
    uint32_t u; std::memcpy(&u, &f, sizeof(u));
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x0040u);  // NaN: keep quiet
    uint32_t rounding_bias = 0x00007fffu + ((u >> 16) & 1u);  // round-to-nearest-even
    return (uint16_t)((u + rounding_bias) >> 16);
}

// Bytes occupied by one row of I elements for a packed weight [O,I].
inline int row_bytes(GGUFType type, int I) {
    switch (type) {
        case GGUFType::F32:      return I * 4;
        case GGUFType::F16:      return I * 2;
        case GGUFType::BF16:     return I * 2;
        case GGUFType::Q4_0:     return (I / 32)  * 18;
        case GGUFType::Q5_0:     return (I / 32)  * 22;
        case GGUFType::Q8_0:     return (I / 32)  * 34;
        case GGUFType::Q2_K:     return (I / 256) * 84;
        case GGUFType::Q3_K:     return (I / 256) * 110;
        case GGUFType::Q4_K:     return (I / 256) * 144;
        case GGUFType::Q5_K:     return (I / 256) * 176;
        case GGUFType::Q6_K:     return (I / 256) * 210;
        case GGUFType::IQ2_XXS:  return (I / 256) * 66;
        case GGUFType::IQ2_XS:   return (I / 256) * 74;
        case GGUFType::IQ2_S:    return (I / 256) * 82;
        case GGUFType::IQ1_S:    return (I / 256) * 50;
        case GGUFType::IQ1_M:    return (I / 256) * 56;
        case GGUFType::IQ2_M:    return (I / 256) * 84;
        // These four were missing entirely (fell through to the `default: I`
        // fallback below) even though ggml_dequant_row DOES dispatch them —
        // that mismatch is what caused a real crash: cpu_quant_matmul walked
        // `W + o*rb` with a bogus stride, reading further past the actual
        // buffer as `o` grew until dequantize_row_iq3_xxs dereferenced
        // unmapped memory. Sizes verified against ggml's block struct
        // static_asserts (ggml/src/ggml-common.h), not re-derived by guess.
        case GGUFType::IQ3_XXS:  return (I / 256) * 98;
        case GGUFType::IQ3_S:    return (I / 256) * 110;
        case GGUFType::IQ4_XS:   return (I / 256) * 136;
        case GGUFType::TQ1_0:    return (I / 256) * 54;
        case GGUFType::TQ2_0:    return (I / 256) * 66;
        case GGUFType::IQ4_NL:   return (I / 32)  * 18;  // block size 32, not 256 — like Q4_0
        default:
            LOG_ERROR("[ROW_BYTES] type=%u has no known row size — "
                      "this WILL misalign cpu_quant_matmul and can crash. "
                      "Add its block size here (see ggml-common.h).\n", (unsigned)type);
            return I;  // unknown — sizes won't be correct
    }
}

// Dequantize one row of n elements at ptr into f32. True if the type is handled.
inline bool dequant_row(GGUFType type, const uint8_t* ptr, float* out, int n) {
    switch (type) {
        case GGUFType::F32:
            std::memcpy(out, ptr, (size_t)n * sizeof(float));
            return true;
        case GGUFType::F16:
            for (int i = 0; i < n; ++i) out[i] = fp16_to_fp32(((const uint16_t*)ptr)[i]);
            return true;
        case GGUFType::BF16:
            for (int i = 0; i < n; ++i) out[i] = bf16_to_fp32(((const uint16_t*)ptr)[i]);
            return true;
        // AVX-512 ports of ggml's plain scalar reference dequant -- see
        // kernels_cpu.hpp's header comment. Added after profiling (this
        // session's t_cpu_dequant/t_cpu_dot split) showed dequant, not the
        // dot product, dominates the CPU expert-FFN fallback cost by ~40x.
        // Q2_K/Q3_K were vectorized FIRST based on a config-file comment
        // that turned out to be wrong for this project's actual target
        // GLM-5.2 UD-IQ2_M checkpoints -- the real startup [QUANT] report
        // (graph_initializer.cpp) shows 71/76 routed-expert layers are
        // gate=up=IQ2_XXS, down=IQ3_XXS, not Q2_K/Q3_K at all (which this
        // checkpoint family doesn't use for routed experts anywhere). Both
        // are kept: Q2_K/Q3_K for other checkpoints that do use them,
        // IQ2_XXS/IQ3_XXS for the checkpoint that was actually profiled.
        // n must be a multiple of 256 (QK_K) -- the same precondition
        // ggml's own reference asserts (stripped in Release, same as
        // upstream).
        case GGUFType::Q2_K:
            dequantize_row_q2_K_avx512(ptr, out, n);
            return true;
        case GGUFType::Q3_K:
            dequantize_row_q3_K_avx512(ptr, out, n);
            return true;
        case GGUFType::IQ2_XXS:
            dequantize_row_iq2_xxs_avx512(ptr, out, n);
            return true;
        case GGUFType::IQ3_XXS:
            dequantize_row_iq3_xxs_avx512(ptr, out, n);
            return true;
        case GGUFType::IQ2_S:
            dequantize_row_iq2_s_avx512(ptr, out, n);
            return true;
        default: {
            int ok = ggml_dequant_row((uint32_t)type, ptr, out, n);
            if (!ok) std::fill(out, out + n, 0.0f);
            return ok != 0;
        }
    }
}

// Dequantize an entire [O,I] row-major tensor to F32, parallel over rows.
// One-time cost (called at load, not per-token) — backs the dense-tensor
// "fast path" upgrade (CudaManager::upload_dequantized_f32) for static
// per-layer weights whose fused GPU kernel has no GEMV specialization and is
// disproportionately slow at S=1 decode (e.g. attn_output, see CLAUDE.md).
// Returns false if the type isn't supported by dequant_row (out is left
// zero-filled for any unsupported row, same convention as dequant_row).
inline bool dequant_tensor_f32(GGUFType type, const uint8_t* W, int O, int I, float* out) {
    int rb = row_bytes(type, I);
    int unsup = 0;
    #pragma omp parallel for schedule(static) reduction(|:unsup)
    for (int o = 0; o < O; ++o) {
        if (!dequant_row(type, W + (int64_t)o * rb, out + (int64_t)o * I, I)) unsup = 1;
    }
    return unsup == 0;
}

// Dequantize one row of a QuantizedTensor (used by MLA v_b absorption).
inline bool dequant_v_b_row(const QuantizedTensor& t, int row, float* out) {
    int I = (int)t.meta.shape[1];
    int rb = row_bytes(t.meta.type, I);
    return dequant_row(t.meta.type, t.host_ptr + (int64_t)row * rb, out, I);
}

// CPU reference quantized matmul: y[S,O] = x[S,I] @ W^T, W packed [O,I].
// Dequantizes each weight row on the fly and dots. Correct for every type
// dequant_row handles; rows of unsupported types are zeroed (returns false).
inline bool cpu_quant_matmul(GGUFType type, const uint8_t* W,
                             const float* x, float* y, int S, int I, int O) {
    int rb = row_bytes(type, I);
    int unsup = 0;
    // Per-thread accumulation via OMP reduction (summed once when the
    // parallel region ends), not a per-row atomic add -- this answers
    // whether the ~27ms/call CPU expert cost is dominated by dequant_row
    // (K-quant/IQ bit-unpacking -- ggml's plain scalar reference
    // implementation, never vectorized) or by the dot product (already
    // AVX-512 vectorized via dot_f32_avx512 earlier this session). If
    // dequant dominates, the dot-product vectorization had limited effect
    // and dequant itself is the real next target.
    double dequant_total = 0.0, dot_total = 0.0;
    #pragma omp parallel for schedule(static) reduction(|:unsup) reduction(+:dequant_total,dot_total)
    for (int o = 0; o < O; ++o) {
        thread_local static std::vector<float> wrow;
        if ((int)wrow.size() < I) wrow.resize(I);

        auto t0 = std::chrono::high_resolution_clock::now();
        if (!dequant_row(type, W + (int64_t)o * rb, wrow.data(), I)) unsup = 1;
        auto t1 = std::chrono::high_resolution_clock::now();
        dequant_total += std::chrono::duration<double>(t1 - t0).count();

        for (int s = 0; s < S; ++s) {
            const float* xs = x + (int64_t)s * I;
            auto t2 = std::chrono::high_resolution_clock::now();
            y[(int64_t)s * O + o] = dot_f32_avx512(wrow.data(), xs, I);
            auto t3 = std::chrono::high_resolution_clock::now();
            dot_total += std::chrono::duration<double>(t3 - t2).count();
        }
    }
    PerfCounters::atomic_add(perf().t_cpu_dequant, dequant_total);
    PerfCounters::atomic_add(perf().t_cpu_dot, dot_total);
    if (unsup) {
        LOG_ERROR("[DEQUANT] type=%u unsupported by ggml — rows zeroed\n",
                  (unsigned)type);
    }
    return unsup == 0;
}
