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
// --- dense_bf16_prefill_gemm (EngineConfig, default OFF) -----------------
// Route S>1 dense projections through "dequantise the weight to BF16, then
// cuBLAS" instead of this project's own fused matmul kernels.
//
// WHY (measured 2026-09-04, attn_qkv's real prefill shape S=3189 I=2560
// O=10240 on an RTX 3090): every fused kernel in cuda/ is GEMV-shaped -- one
// output row per block, streaming the activation -- so each of the O blocks
// re-reads the whole [S, I] activation. Correct at S=1, badly wrong at
// prefill batch sizes:
//     fused q8_0 kernel                  194.41 ms   ( 2.2% of peak)
//     dequant->BF16 + cublasGemmEx         3.11 ms   (71.1 TFLOPS, ~100% of
//                                                     the card's BF16 peak)
// Only Q8_0 is routed for now -- which happens to be EVERY dense projection
// in qwen4exp (attn_qkv/attn_gate/ssm_out/attn_q/k/v/output/hc_*/shexp), so
// it covers that whole bucket. The expert types (Q4_K/Q5_K/Q5_1) are the
// separate, larger bucket and come next.
//
// !! SHARED PATH !! dense_matmul_host is family-agnostic, so GLM-5.3 and
// DeepSeek-V4 reach this too and both have Q8_0 dense tensors. DEFAULT ON as
// of 2026-09-04, after verifying GLM-5.3 as well as qwen4exp; set
// dense_bf16_prefill_gemm=false to restore the fused kernels unchanged.
inline bool& dense_bf16_prefill_gemm_flag() {
    static bool v = false;
    return v;
}
inline bool dense_bf16_prefill_gemm() { return dense_bf16_prefill_gemm_flag(); }

// VRAM watermark fraction (EngineConfig::vram_max_fraction, llama.cpp-style
// 0.9). Lives HERE rather than dense_dispatch.hpp for the same reason
// dense_bf16_prefill_gemm() does: dense_dispatch.hpp includes cuda_manager.hpp,
// so cuda_manager.hpp cannot include it back.
// Minimum FREE VRAM to keep on each device when growing a dynamic buffer.
// A free-memory floor, NOT a fraction of total -- see vram_has_headroom()'s
// comment for why the fraction form is wrong for this engine.
inline size_t &vram_min_free_bytes_flag() { static size_t b = 256ull*1024*1024; return b; }
inline void set_vram_min_free_mb(long mb) { vram_min_free_bytes_flag() = (size_t)mb*1024*1024; }
inline size_t vram_min_free_bytes() { return vram_min_free_bytes_flag(); }

// dense_bf16_kquant: gates the 2026-09-05 EXTENSION of the dense BF16 GEMM to
// Q5_K/Q6_K, separately from dense_bf16_prefill_gemm's original Q8_0 path.
// Two reasons they are separate knobs rather than one:
//   * The Q8_0 path was verified COHERENT on GLM-5.3; the K-quant extension
//     has only been verified on qwen4exp. A model should be able to take the
//     proven half without the unproven half.
//   * They have very different VRAM cost profiles. GLM's dense pool has 309
//     Q5_K + 81 Q6_K tensors, some 201 MB once expanded to BF16, on GPUs that
//     already sit at 23.7/24 GB -- so on that model the extension mostly gets
//     refused by the watermark anyway, at the cost of trying.
inline bool &dense_bf16_kquant_flag() { static bool f = false; return f; }
inline void set_dense_bf16_kquant(bool v) { dense_bf16_kquant_flag() = v; }
inline bool dense_bf16_kquant() { return dense_bf16_kquant_flag(); }
inline void set_dense_bf16_prefill_gemm(bool v) { dense_bf16_prefill_gemm_flag() = v; }

inline bool gpu_type_supported(GGUFType type) {
    switch (type) {
        case GGUFType::Q4_0: case GGUFType::Q2_0: case GGUFType::Q2_K: case GGUFType::Q3_K:
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
        // MXFP4: cuda/mxfp4_matmul.cu's launch_mxfp4_matmul (verbatim port
        // of ggml's dequantize_row_mxfp4, warp-per-32-element-block design)
        // — RE-ENABLED after passing src/main_test.cpp's Test 5 case on
        // real Linux/CUDA hardware (RTX 3090 + RTX 5060 Ti): all 3 shapes
        // (I=512/O=8/S=3, I=4096/O=4/S=2, and the real production shape
        // I=2048/O=6144/S=1 matching DeepSeek-V4-Flash's actual
        // blk.42.ffn_down_exps.weight) PASSED with float-noise-level
        // max_rel_diff (2.7e-6, 9.9e-7, 1.8e-4). Per the IQ4_XS incident's
        // own lesson, this was deliberately left disabled until that real
        // pass — see CLAUDE.md and this change's plan file for the writeup.
        case GGUFType::MXFP4:
            return true;
        // IQ1_S: cuda/iq1_dequant.cu's launch_iq1s_matmul, a verbatim port
        // of ggml's dequantize_row_iq1_s (real 2048-entry ternary grid
        // codebook + per-32-weight-group scale/delta, NOT the flat
        // sign-bit decode the file's pre-existing "iq1_matmul_kernel_
        // optimized" turned out to implement -- found wrong this session
        // while wiring this up for real, see that kernel's own updated
        // comment). Correctness verified via src/main_test.cpp's Test 5
        // against ggml's own exact dequant+dot reference (same path Q6_K/
        // Q8_0 use, not the lossy cpu_quant_matmul fast path -- IQ1_S has
        // no CPU AVX-512 fast-path kernel of its own, so there's no
        // "production fast path vs test reference" ambiguity to resolve
        // here the way that comment block had to for Q6_K/Q8_0).
        // Motivation: GLM-5.3's real Unsloth "UD-IQ1_S" release keeps
        // gate/up at IQ1_S for a real subset of MoE layers (confirmed
        // 2026-09 against a real downloaded shard: 12 layers on this
        // checkpoint) while leaving down at IQ3_XXS (already supported
        // above) and dense/output-adjacent layers at IQ2_XXS -- without
        // this case, EVERY expert in an IQ1_S layer would fall back to the
        // CPU AVX-512 path entirely (the gpu_type_supported() check below
        // is a per-expert AND across gate/up/down, so one unsupported
        // tensor forces the whole expert off GPU), at roughly the ~9-10x
        // per-call CPU-vs-GPU tax this project has already measured
        // elsewhere (see CLAUDE.md's cpu_offload_max_experts investigation)
        // -- which would swamp any disk/PCIe savings the smaller IQ1_S
        // tensors themselves provide. IQ1_M/TQ1_0 remain unsupported (see
        // gpu_unsupported_reason() below) -- not verified or fixed this
        // session, do not assume this case's correctness work covers them.
        case GGUFType::IQ1_S:
            return true;
        // IQ3_S: cuda/iq3_matmul.cu's launch_iq3s_matmul, a verbatim port of
        // ggml's dequantize_row_iq3_s (real 512-entry grid codebook + qh
        // 9th-bit extension + per-32-weight nibble scale -- same grid family
        // as IQ2_XXS/XS/S and IQ3_XXS above, just with the extra qh byte and
        // a different scale encoding). Correctness verified via
        // src/main_test.cpp's Test 5, comparing against cpu_quant_matmul's
        // own exact dequant+dot path (IQ3_S has no lossy AVX-512 fast-path
        // kernel of its own, so unlike Q6_K/Q8_0 there's no "production fast
        // path vs test reference" ambiguity to resolve here). Motivation:
        // real Unsloth Dynamic-quant "UD-IQ3_XXS"-labeled GLM-5.3 files keep
        // gate/up at IQ2_S and down at IQ3_S for the large majority of MoE
        // layers (confirmed 2026-09 against a real downloaded 7-shard file:
        // uniform IQ2_S/IQ3_S except the same handful of special layers this
        // checkpoint's own IQ2_M/IQ1_S files also elevate) -- without this
        // case, every expert using that source's down_proj would fall back
        // to CPU AVX-512 entirely.
        case GGUFType::IQ3_S:
            return true;
        // IQ1_M: cuda/iq1_dequant.cu's launch_iq1m_matmul, a verbatim port
        // of ggml's dequantize_row_iq1_m -- reuses IQ1_S's own 2048-entry
        // c_iq1s_grid table (same ternary grid, different per-block
        // scale/index packing: block_iq1_m is qs[32]+qh[16]+scales[8], no
        // standalone `d` half -- the super-block scale is packed into
        // scales[]'s own top nibbles). Added 2026-09 alongside the IQ1_S
        // fix, once IQ1_S's own investigation found the OLD "iq1_matmul_
        // kernel_optimized" this project already had was ALSO meant to
        // cover IQ1_M and was ALSO wrong (see that kernel's own comment) --
        // that dead kernel was never wired to gpu_type_supported() for
        // either type, so this was never a live correctness bug, just an
        // unimplemented one. Correctness verified via src/main_test.cpp's
        // Test 5 against ggml's own exact dequant+dot reference (same
        // ggml_dequant_row path IQ1_S/Q6_K/Q8_0 use -- IQ1_M has no lossy
        // CPU AVX-512 fast-path of its own either). NOT present in the
        // real GLM-5.3 checkpoint this project currently tests against
        // (confirmed via a direct header parse of every downloaded
        // shard: 0 IQ1_M tensors) -- unlike IQ1_S, this has NOT been
        // confirmed via a real production run with real IQ1_M weights,
        // only Test 5's synthetic data. Treat with the same caution as
        // any Test-5-only-verified type until real data exists to check
        // against.
        case GGUFType::IQ1_M:
            return true;
        // Q8_0: cuda/q8_0_matmul.cu's launch_q8_0_matmul -- long-standing,
        // real, and already load-bearing for the DENSE path
        // (cuda_manager.hpp's dense_matmul_host switch), plus covered by
        // src/main_test.cpp's Test 5 at attn_q_b's real production shape
        // (I=2048, O=16384, S=1) against an exact f32 reference. It was
        // simply never advertised here, so no ROUTED EXPERT could use it.
        // Added 2026-09-04 during qwen4exp (Qwen3.8-Flash-Next) bring-up:
        // that checkpoint's ffn_down_exps is Q8_0 on 5 of 48 MoE layers, and
        // because this predicate is an AND over gate/up/down, that single
        // tensor was pinning those layers' entire expert compute to CPU.
        // Q5_1 (the other 43 layers) genuinely has no kernel and stays
        // unsupported -- see gpu_unsupported_reason().
        case GGUFType::Q8_0:
            return true;
        // Q5_1: cuda/q5_1_matmul.cu's launch_q5_1_matmul, a direct port of
        // ggml's dequantize_row_q5_1 (per-32 scale d + minimum m over 4-bit
        // nibbles, with each weight's 5th bit packed into a 32-bit qh).
        // Written 2026-09-04 specifically because qwen4exp
        // (Qwen3.8-Flash-Next) stores ffn_down_exps as Q5_1 on 43 of 48 MoE
        // layers, which -- this predicate being an AND over gate/up/down --
        // pinned ~90% of that model's routed-expert compute to CPU.
        // Verified via src/main_test.cpp's Test 5 against an exact f32
        // reference at the real production shape, plus a real generation run.
        case GGUFType::Q5_1:
            return true;
        case GGUFType::IQ4_NL:
            // 2026-09-04: cuda/iq4xs_matmul.cu's launch_iq4_nl_matmul. Shares
            // IQ4_XS's 16-entry codebook; simpler block (one fp16 scale per 32,
            // no sub-scales). Unblocks Qwen3.8-Flash-Next UD-IQ1_M, whose down
            // projection is IQ4_NL on all 48 MoE layers -- that single gap made
            // every layer CPU-bound.
            return true;
        // No GPU kernel implemented anywhere in cuda/*.cu for these:
        // TQ1_0, TQ2_0, IQ2_M. (IQ4_NL gained one 2026-09-04.) See
        // gpu_unsupported_reason()
        // for why each specific one isn't supported yet.
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
        // Naming this independent of gpu_type_supported()'s decision — logs
        // should say "MXFP4", not "UNKNOWN", even while GPU dispatch for it
        // is still dormant (see gpu_type_supported()'s own comment above).
        case GGUFType::MXFP4:   return "MXFP4";
        case GGUFType::NVFP4:   return "NVFP4";
        case GGUFType::Q2_0:    return "Q2_0";
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
        // Q4_1/Q5_1 were missing here too, so the [QUANT] startup census
        // printed "down=Q5_1 (0 bits/weight)" for a real model -- the
        // `default: return 0.0` branch, not a real measurement. 20 and 24
        // bytes per 32-element block (see row_bytes above) -> 5.0 and 6.0.
        case GGUFType::Q4_1:    return 5.0;
        case GGUFType::Q5_1:    return 6.0;
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
        case GGUFType::MXFP4:   return 4.25;   // 17 bytes * 8 / 32 elements
        case GGUFType::NVFP4:   return 4.5;    // 36 bytes * 8 / 64 elements
        case GGUFType::Q2_0:    return 2.25;   // 18 bytes * 8 / 64 elements
        default:                return 0.0;  // unknown — don't claim a number
    }
}

// WHY this type has no fused GPU matmul kernel — answers "is it because of
// the bit-packing" directly. Short answer: no, not fundamentally — IQ2_XXS
// at 2.06 bits/weight (even smaller than IQ3_S's 3.44) DOES have a kernel
// (cuda/iq2_matmul.cu), so sub-byte packing alone isn't the blocker. Two
// genuinely different reasons hide behind "IQ-series", verified against
// ggml's actual dequant source (ggml/src/ggml-quants.c) rather than assumed:
//   - The IQ2_XXS/XS/S, IQ3_XXS, and IQ3_S family: each block picks a
//     combination from a LARGE variable grid table (256-1024 entries,
//     ggml-common.h's iq2xxs_grid/iq2xs_grid/iq2s_grid/iq3xxs_grid/
//     iq3s_grid) plus separate sign bits to reconstruct values — all five
//     now have a fused kernel (cuda/iq2_matmul.cu, cuda/iq3_matmul.cu).
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
        // IQ3_S moved to gpu_type_supported()'s TRUE case above (real
        // grid-codebook kernel added 2026-09, see that case's own comment)
        // -- unreachable here now.
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
        // IQ1_S and IQ1_M both moved to gpu_type_supported()'s TRUE case
        // above (real grid-codebook kernels, 2026-09) -- unreachable here
        // now. This reason string covers ONLY TQ1_0.
        case GGUFType::TQ1_0:
            return "1-bit ternary format -- cuda/iq1_dequant.cu has a "
                   "kernel shaped for this (tq1_matmul_kernel_optimized, "
                   "branchless trit decode, no grid table needed for a "
                   "true per-weight ternary format -- structurally closer "
                   "to correct than the OLD IQ1_S/IQ1_M kernel this file "
                   "also had, which was found genuinely wrong). Still not "
                   "independently re-verified against dequantize_row_"
                   "tq1_0 (ggml-quants.c) -- do not assume correct just "
                   "because it wasn't flagged wrong; it simply hasn't "
                   "been checked. Not wired into the dispatch table; "
                   "needs the same Test-5-style verification IQ1_S/IQ1_M "
                   "got before that's safe.";
        case GGUFType::TQ2_0:
            return "ternary quantization -- no fused CUDA kernel implemented.";
        // Q4_1/Q5_0: the two remaining legacy non-K linear quants. Their
        // sibling Q5_1 IS supported now (cuda/q5_1_matmul.cu, 2026-09-04),
        // and these two are structurally the same shape -- a per-32-element
        // scale, plus a minimum for Q4_1 and a packed 5th bit for Q5_0, over
        // 4-bit nibbles. So this is no longer "hard", it is just unwritten:
        // q5_1_matmul.cu is close to a template for either (Q4_1 = drop the
        // qh handling, Q5_0 = drop m and re-center the nibble at -16). Left
        // undone deliberately rather than speculatively, since no checkpoint
        // this engine has been pointed at actually uses them for experts --
        // and ggml_dequant_row (src/ggml_dequant.cpp) has no Q4_1 case
        // either, so Q4_1 would need a CPU path too, not just a kernel.
        case GGUFType::Q4_1:
        case GGUFType::Q5_0:
            return "legacy linear quant (per-32 scale [+ min] over 4-bit "
                   "nibbles, plus a packed 5th bit for Q5_x) -- structurally "
                   "simpler than the supported Q4_0/K-quant kernels, just no "
                   "fused CUDA kernel written for it yet.";
        case GGUFType::IQ2_M:
            return "codebook quantization -- no ggml CPU dequant export exists "
                   "either (see kquant_dequant.hpp's dequant_row), so this type "
                   "is unsupported on BOTH paths, not just GPU.";
        // MXFP4 moved to gpu_type_supported()'s TRUE case above (re-enabled
        // after a real-hardware Test 5 pass) -- this switch is unreachable
        // for it now, since gpu_unsupported_reason() already returns early
        // with "supported" for anything gpu_type_supported() accepts. Same
        // pattern as IQ4_XS's own case just above.
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
    uint32_t abs_bits = u & 0x7fffffffu;
    if (abs_bits > 0x7f800000u) return (uint16_t)((u >> 16) | 0x0040u);  // NaN: keep quiet
    if (abs_bits == 0x7f800000u) return (uint16_t)(u >> 16);  // +-Inf already: exact, no rounding needed
    uint32_t rounding_bias = 0x00007fffu + ((u >> 16) & 1u);  // round-to-nearest-even
    uint32_t rounded = u + rounding_bias;
    // OVERFLOW-ON-ROUND GUARD: `f` was just verified finite above, but
    // adding the rounding bias can carry out of the kept mantissa bits and
    // into the exponent field -- if the input's exponent was already the
    // maximum finite value (|f| within the top ULP-band just below
    // FLT_MAX, ~3.4e38), that carry pushes BF16's exponent field to all-1s
    // (0xFF), its reserved Inf/NaN pattern, silently turning a genuinely
    // finite number into +-Infinity. This is technically IEEE-754-
    // correct rounding behavior for a narrowing conversion, but this
    // project's standing discipline (see dense_dispatch.hpp's own header
    // comment: never silently let a numeric edge case propagate as
    // "the silent-garbage bug this project is eliminating") is to make
    // this loud and safe rather than let an Inf weight/activation
    // silently poison every downstream cuBLAS output. Clamp to the
    // largest finite BF16 magnitude (same sign) instead, and count it
    // (perf().n_bf16_overflow_clamped) -- should be essentially
    // unreachable for real LLM weights/activations, so a nonzero count
    // is a genuine, actionable signal (a runaway value upstream), not
    // routine rounding noise.
    if (((rounded >> 16) & 0x7f80u) == 0x7f80u) {
        perf().n_bf16_overflow_clamped.fetch_add(1, std::memory_order_relaxed);
        uint16_t sign = (uint16_t)((u >> 16) & 0x8000u);
        return (uint16_t)(sign | 0x7f7fu);  // largest finite BF16 magnitude, same sign
    }
    return (uint16_t)(rounded >> 16);
}

// Bytes occupied by one row of I elements for a packed weight [O,I].
inline int row_bytes(GGUFType type, int I) {
    switch (type) {
        case GGUFType::F32:      return I * 4;
        case GGUFType::F16:      return I * 2;
        case GGUFType::BF16:     return I * 2;
        case GGUFType::Q4_0:     return (I / 32)  * 18;
        case GGUFType::Q5_0:     return (I / 32)  * 22;
        // Q4_1/Q5_1: the SAME table-hole that already bit this project twice
        // (IQ3_XXS/IQ3_S/IQ4_XS, then MXFP4/NVFP4 -- see both comment blocks
        // below), hit a third time. Found 2026-09-04 by a real SIGSEGV during
        // qwen4exp (Qwen3.8-Flash-Next) prefill: that checkpoint stores
        // ffn_down_exps as Q5_1 on 43 of 48 MoE layers, `default:` returned a
        // stride of I instead of (I/32)*24, and cpu_quant_matmul walked
        // `W + o*rb` off the end of the buffer until dequantize_row_q5_1
        // dereferenced unmapped memory (confirmed from the coredump:
        // dequantize_row_q5_1 <- ggml_dequant_row <- cpu_quant_matmul).
        // Sizes taken from ggml's own static_asserts (ggml-common.h), not
        // re-derived: block_q4_1 = 2*ggml_half + QK4_1/2 = 20, block_q5_1 =
        // 2*ggml_half + uint32_t + QK5_1/2 = 24, both over 32 elements. They
        // agree with gguf_loader.hpp's calculate_size(), which is an
        // INDEPENDENT table that already had both (which is exactly why the
        // model loaded with correct tensor sizes and only crashed later, in
        // compute -- the same split-brain signature as the MXFP4 case).
        // NOTE: only Q5_1 is end-to-end usable today -- ggml_dequant_row
        // (src/ggml_dequant.cpp) dispatches Q5_1 (case 7) but has no Q4_1
        // (case 3) case, so Q4_1 here just replaces a guaranteed-wrong
        // stride with the right one; it still needs a dequant case before a
        // real Q4_1 weight could compute.
        case GGUFType::Q4_1:     return (I / 32)  * 20;
        case GGUFType::Q5_1:     return (I / 32)  * 24;
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
        // Missing exactly like the IQ3_XXS/IQ3_S/IQ4_XS gap above (same
        // failure mode: cpu_quant_matmul walks W+o*rb with a bogus stride,
        // NaN/garbage rather than a clean crash for a real DeepSeek-V4-Flash
        // MXFP4 tensor -- see gguf_loader.hpp's calculate_size(), which
        // needed the identical addition; both tables are independent, this
        // one wasn't caught by that earlier fix). Sizes verified against
        // ggml's block struct static_asserts (ggml-common.h): block_mxfp4
        // = 1(e) + QK_MXFP4/2(16) = 17 bytes per 32-element block;
        // block_nvfp4 = QK_NVFP4/QK_NVFP4_SUB(4) + QK_NVFP4/2(32) = 36
        // bytes per 64-element block.
        case GGUFType::MXFP4:    return (I / 32)  * 17;
        case GGUFType::NVFP4:    return (I / 64)  * 36;
        // Q2_0: real legacy-style linear 2-bit format (block_q2_0,
        // ggml-common.h), block=64 elements, 2(d)+16(qs) bytes. Added
        // 2026-09-28 for a real GSQ-RCO Qwen3.8-Flash-Next checkpoint
        // storing ffn_down_exps this way on every MoE layer -- without
        // this case cpu_quant_matmul would walk W+o*rb with the `default:`
        // stride of I bytes/row instead of 18, corrupting every row past
        // the first exactly like the IQ3_XXS/MXFP4 gaps this file's other
        // comments describe.
        case GGUFType::Q2_0:     return (I / 64)  * 18;
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
        case GGUFType::IQ2_XS:
            dequantize_row_iq2_xs_avx512(ptr, out, n);
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

// cpu_dot_ggml_avx2 (config): switches the CPU-fallback dot product for
// IQ2_XXS/IQ3_XXS (GLM-5.2's actual routed-expert types, 71-76/78 layers,
// dispatched from cpu_quant_matmul below) from the existing FMA path
// (dot_iq2/3_xxs_f32_avx512, F32 dequant + FMA against F32 activations) to
// a ported upstream llama.cpp/ggml AVX2 kernel (matmul_iq2/3_xxs_ggml_
// avx2, register-resident int8 dot via _mm256_maddubs_epi16+madd_epi16
// against Q8_K-quantized activations). Real isolated microbenchmark showed
// ~1.7-1.9x faster per-call on this hardware (see CLAUDE.md's "CPU-offload
// feasibility" entries for the full investigation) -- but ALSO a real,
// not-yet-resolved accuracy caveat for IQ3_XXS specifically (20-row
// worst-case 1.73% relative error, above this project's own ~1% tolerance
// convention used elsewhere). This flag exists to A/B the REAL end-to-end
// effect on decode throughput before deciding whether that tradeoff is
// worth it -- default off, NOT yet a recommended production setting.
// Same function-local-static-storage pattern as cpu_only_flag() (dense_
// dispatch.hpp) -- defined HERE (not there) because dense_dispatch.hpp
// already #includes this file, and the reverse would be circular.
inline bool& cpu_dot_ggml_avx2_flag() {
    static bool v = false;
    return v;
}
inline bool cpu_dot_ggml_avx2_enabled() { return cpu_dot_ggml_avx2_flag(); }
inline void enable_cpu_dot_ggml_avx2() { cpu_dot_ggml_avx2_flag() = true; }

// Qwen4Exp-only CPU path.  It mirrors BeeLlama's ggml IQ1_M x Q8_K route
// and is deliberately independent of the GLM IQ2/IQ3 AVX2 experiment above.
// Note that GGML's IQ1_M kernel is AVX2-width in the current dependency even
// when the process was built with AVX-512/VNNI enabled; the Qwen-only
// AVX-512 probe in kernels_cpu.cpp remains opt-in until it wins end-to-end.
inline bool& cpu_dot_ggml_iq1m_flag() {
    static bool v = false;
    return v;
}
inline bool cpu_dot_ggml_iq1m_enabled() { return cpu_dot_ggml_iq1m_flag(); }
inline void enable_cpu_dot_ggml_iq1m() { cpu_dot_ggml_iq1m_flag() = true; }

// CPU reference quantized matmul: y[S,O] = x[S,I] @ W^T, W packed [O,I].
// Dequantizes each weight row on the fly and dots. Correct for every type
// dequant_row handles; rows of unsupported types are zeroed (returns false).
inline bool cpu_quant_matmul(GGUFType type, const uint8_t* W,
                             const float* x, float* y, int S, int I, int O) {
                             
    // ----------------------------------------------------
    // 1. FAST PATH: Fused AVX-512 In-Register Matmuls
    // Bypasses intermediate FP32 allocation and OMP row-loops
    // ----------------------------------------------------



    
    if (type == GGUFType::IQ2_XXS) {
        auto t0 = std::chrono::high_resolution_clock::now();
        // cpu_dot_ggml_avx2: see this flag's own comment above for the
        // real speedup/accuracy tradeoff this switches between.
        if (cpu_dot_ggml_avx2_enabled()) matmul_iq2_xxs_ggml_avx2(y, x, W, S, I, O);
        else matmul_iq2_xxs_fused(y, x, W, S, I, O);
        auto t1 = std::chrono::high_resolution_clock::now();
        PerfCounters::atomic_add(perf().t_cpu_dot, std::chrono::duration<double>(t1 - t0).count());
        return true;
    }

    if (type == GGUFType::IQ3_XXS) {
        auto t0 = std::chrono::high_resolution_clock::now();
        if (cpu_dot_ggml_avx2_enabled()) matmul_iq3_xxs_ggml_avx2(y, x, W, S, I, O);
        else matmul_iq3_xxs_fused(y, x, W, S, I, O);
        auto t1 = std::chrono::high_resolution_clock::now();
        PerfCounters::atomic_add(perf().t_cpu_dot, std::chrono::duration<double>(t1 - t0).count());
        return true;
    }

    if (type == GGUFType::IQ1_M && cpu_dot_ggml_iq1m_enabled()) {
        auto t0 = std::chrono::high_resolution_clock::now();
        if (matmul_iq1_m_ggml_q8k(y, x, W, S, I, O)) {
            auto t1 = std::chrono::high_resolution_clock::now();
            PerfCounters::atomic_add(perf().t_cpu_dot,
                                     std::chrono::duration<double>(t1 - t0).count());
            return true;
        }
        // Invalid shape or a non-ggml build: preserve the existing
        // correctness fallback below rather than returning partial output.
    }

    if (type == GGUFType::Q5_K) {
        auto t0 = std::chrono::high_resolution_clock::now();
        matmul_q5_k_fused(y, x, W, S, I, O);
        auto t1 = std::chrono::high_resolution_clock::now();
        PerfCounters::atomic_add(perf().t_cpu_dot, std::chrono::duration<double>(t1 - t0).count());
        return true;
    }
    if (type == GGUFType::Q6_K) {
        auto t0 = std::chrono::high_resolution_clock::now();
        matmul_q6_k_fused(y, x, W, S, I, O);
        auto t1 = std::chrono::high_resolution_clock::now();
        PerfCounters::atomic_add(perf().t_cpu_dot, std::chrono::duration<double>(t1 - t0).count());
        return true;
    }
    if (type == GGUFType::Q8_0) {
        auto t0 = std::chrono::high_resolution_clock::now();
        matmul_q8_0_fused(y, x, W, S, I, O);
        auto t1 = std::chrono::high_resolution_clock::now();
        PerfCounters::atomic_add(perf().t_cpu_dot, std::chrono::duration<double>(t1 - t0).count());
        return true;
    }
    if (type == GGUFType::F32) {
        auto t0 = std::chrono::high_resolution_clock::now();
        matmul_f32_avx512(y, x, (const float*)W, S, I, O);
        auto t1 = std::chrono::high_resolution_clock::now();
        PerfCounters::atomic_add(perf().t_cpu_dot, std::chrono::duration<double>(t1 - t0).count());
        return true;
    }
    // ----------------------------------------------------
    // 2. STANDARD PATH: Temporary Buffer Dequant + Dot
    // ----------------------------------------------------
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
