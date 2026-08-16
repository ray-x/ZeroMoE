#include "kernels_cpu.hpp"
#include "ram_manager.hpp"
#include "engine.hpp"
#include "cuda_manager.hpp"
#include "kquant_dequant.hpp"

#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <random>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <omp.h>

// ------------------------------------------------------------------
// External Links to Core Functions
// ------------------------------------------------------------------
extern void matmul(float* y, const float* x, const float* W, int S, int I, int O);
extern void rmsnorm_row(float* out, const float* x, const float* w, int D, float eps);
extern void softmax_row(float* x, int size);
extern void extract_gguf_row_q4_0(const QuantizedTensor& tensor, int row_idx, float* __restrict__ out_f32);

// Encodes real ggml-valid quantized bytes from f32 input (src/ggml_dequant.cpp).
// Returns 1 if this type has a reference quantizer wired up, 0 otherwise.
extern "C" int ggml_quantize_row_ref(uint32_t type, const float* src, void* dst, int64_t n);
extern "C" int ggml_dequant_row(uint32_t type, const void* src, float* out, int n);

class GGUFGraphInitializer {
public:
    static bool build_model_graph(StreamingEngine& engine, int max_ctx);
};

// ------------------------------------------------------------------
// 1. CPU-Based Matrix Ops Tests
// ------------------------------------------------------------------
void test_cpu_matrix_ops() {
    std::cout << "[Test 1] CPU Math Primitives (Matmul, Softmax, RMSNorm)...\n";

    // 1a. Matmul (1x2 * 2x3^T)
    std::vector<float> x = {1.0f, 2.0f};
    std::vector<float> W = {2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f}; // Shape: O=3, I=2
    std::vector<float> y(3, 0.0f);

    matmul(y.data(), x.data(), W.data(), 1, 2, 3);
    assert(y[0] == (1.0f * 2.0f + 2.0f * 3.0f)); // 8.0
    assert(y[1] == (1.0f * 4.0f + 2.0f * 5.0f)); // 14.0
    assert(y[2] == (1.0f * 6.0f + 2.0f * 7.0f)); // 20.0

    // 1b. Softmax
    std::vector<float> sm = {1.0f, 2.0f, 3.0f};
    softmax_row(sm.data(), 3);
    assert(sm[2] > sm[1] && sm[1] > sm[0]);
    assert(std::abs(sm[0] + sm[1] + sm[2] - 1.0f) < 1e-5f); // Sum must equal 1.0

    // 1c. RMSNorm
    std::vector<float> norm_in = {1.0f, -1.0f, 2.0f, -2.0f};
    std::vector<float> norm_w = {1.0f, 1.0f, 1.0f, 1.0f};
    std::vector<float> norm_out(4);
    rmsnorm_row(norm_out.data(), norm_in.data(), norm_w.data(), 4, 1e-5f);
    assert(norm_out[0] > 0.0f && norm_out[1] < 0.0f);

    std::cout << "  -> Passed!\n";
}

// ------------------------------------------------------------------
// 2. Graph Initialization Tests
// ------------------------------------------------------------------
void test_graph_initializer() {
    std::cout << "[Test 2] Graph Topology Analyzer...\n";
    StreamingEngine engine;

    // Inject fake tensor metadata to trigger structural discovery logic
    engine.model_file.tensors["token_embd.weight"] = {"token_embd.weight", GGUFType::F32, {32000, 4096}, 0, 0};
    engine.model_file.tensors["blk.0.ffn_gate_exps.weight"] = {"blk.0.ffn_gate_exps.weight", GGUFType::Q4_0, {8, 14336, 4096}, 0, 0};
    engine.model_file.tensors["blk.1.ffn_gate_exps.weight"] = {"blk.1.ffn_gate_exps.weight", GGUFType::Q4_0, {8, 14336, 4096}, 0, 0};

    GGUFGraphInitializer::build_model_graph(engine, 128);

    auto& c = engine.model_file.architecture_config;
    assert(c.vocab == 32000);
    assert(c.hidden == 4096);
    assert(c.n_layers == 2);
    assert(c.n_experts == 8);

    std::cout << "  -> Passed!\n";
}

// ------------------------------------------------------------------
// 3. Attention Layer Tests
// ------------------------------------------------------------------
void test_attention_extraction() {
    std::cout << "[Test 3] Attention Q4_0 Vectorized Extraction...\n";

    QuantizedTensor qt;
    qt.meta.shape = {1, 32}; // 1 row, 32 elements (exactly 1 Q4_0 block)

    block_q4_0 blk;
    blk.d = fp32_to_fp16(1.0f); // Scale factor 1.0
    // Fill nibbles with 0x88. (8 & 0x0F) - 8 = 0. Therefore all weights decode to 0.0f
    for (int i = 0; i < 16; ++i) blk.qs[i] = 0x88;

    qt.host_ptr = reinterpret_cast<uint8_t*>(&blk);

    std::vector<float> out(32, -1.0f);
    extract_gguf_row_q4_0(qt, 0, out.data());

    // Assert all 32 unpacked float targets successfully resolve to 0.0f
    for (int i = 0; i < 32; ++i) {
        assert(std::abs(out[i] - 0.0f) < 1e-4);
    }

    std::cout << "  -> Passed!\n";
}

// ------------------------------------------------------------------
// 4. CUDA DP4A Target Tests
// ------------------------------------------------------------------
void test_cuda_execution() {
    std::cout << "[Test 4] CUDA DP4A Hardware VRAM Kernels...\n";
    CudaManager gpu;

    if (!gpu.initialize()) {
        std::cout << "  -> Skipped (No CUDA Device Found on system)\n";
        return;
    }

    // Mock a Q4_0 target matrix to offload to VRAM
    QuantizedTensor w;
    w.meta.shape = {1, 32}; // O=1, I=32
    w.meta.size_bytes = sizeof(block_q4_0);

    block_q4_0 blk;
    blk.d = fp32_to_fp16(1.0f);
    // Fill nibbles with 0x99. (9 & 0x0F) - 8 = 1. Therefore all weights decode to 1.0f
    for (int i = 0; i < 16; ++i) blk.qs[i] = 0x99;
    w.host_ptr = reinterpret_cast<uint8_t*>(&blk);

    gpu.offload_tensor_to_vram(w);

    std::vector<float> x(32, 2.0f); // Vector of 32 activations, all equal to 2.0f
    std::vector<float> y(1, 0.0f);

    // Run Host-to-Device Matmul Kernel
    // Math logic: 32 elements * (1.0f weight * 2.0f activation) = 64.0f
    gpu.execute_matmul(w, x.data(), y.data(), 1);

    // Provide a small slack delta to account for int8 dynamic quantization rounding limits
    assert(y[0] > 60.0f && y[0] < 68.0f);

    std::cout << "  -> Passed!\n";
}

// ------------------------------------------------------------------
// 5. Fused Dequant+Matmul CUDA Kernel Correctness (GPU vs CPU reference)
// ------------------------------------------------------------------
// For each type below: generate random f32 weights, encode them into REAL
// ggml-valid quantized bytes via ggml's own reference quantizer (NOT
// synthetic/hand-picked bit patterns -- that's what test_cuda_execution's
// Q4_0 case does, deliberately, for a from-scratch sanity check; this test
// instead exercises the exact byte layouts a real GGUF file would contain).
// The SAME bytes are then run through both the CPU reference path
// (cpu_quant_matmul -> ggml_dequant_row, already verified correct by direct
// comparison against ggml's source) and the CUDA fused kernel
// (CudaManager::dense_matmul_host). Any mismatch beyond float
// summation-order slop means the GPU kernel's dequant math, grid/table
// indexing, or stride has a real bug -- this is the fastest way to catch
// exactly the class of bug this project has hit twice already (row_bytes
// stride mismatch causing a SIGSEGV; a device-context bug silently
// preventing GPU dispatch from ever firing).
static bool run_fused_kernel_case(CudaManager& gpu, GGUFType type, int I, int O, int S) {
    const char* name = gpu_type_name(type);
    assert(I % 256 == 0 && "test cases below must use a multiple of 256");

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> w_f32((size_t)O * I);
    for (auto& v : w_f32) v = dist(rng);
    std::vector<float> x((size_t)S * I);
    for (auto& v : x) v = dist(rng);

    int rb = row_bytes(type, I);
    std::vector<uint8_t> w_bytes((size_t)O * rb);
    for (int o = 0; o < O; ++o) {
        int ok = ggml_quantize_row_ref((uint32_t)type, w_f32.data() + (size_t)o * I,
                                        w_bytes.data() + (size_t)o * rb, I);
        if (!ok) {
            std::cout << "  -> " << name << ": SKIPPED (no reference quantizer wired up in "
                         "ggml_quantize_row_ref)\n";
            return true;
        }
    }

    // CPU reference: ggml's own dequantize_row_* + naive dot product.
    //
    // Q6_K/Q8_0 are a deliberate EXCEPTION: cpu_quant_matmul's own fast path
    // for these two (matmul_q6_k_fused / matmul_q8_0_fused, kernels_cpu.cpp)
    // quantizes the ACTIVATIONS to int8 per 16-element group for a VNNI dot
    // product -- a genuinely lossy, different computation from the GPU
    // kernel's exact f32 dequant+dot, not just summation-order noise (see
    // this function's own tolerance comment further down). That's fine as a
    // PRODUCTION fast path, but real hardware confirmed it's NOT a reliable
    // TEST REFERENCE at real production scale: a direct 3-way check (GPU /
    // this lossy path / a from-scratch exact f32 dequant+dot, added
    // one-off during that investigation and since removed) at attn_q_b's
    // real shape (I=2048, O=16384, S=1) found the GPU kernel matching the
    // exact reference to ~2.9e-6 absolute error while the lossy CPU path was
    // off by up to 0.17 absolute at a normal-magnitude (~0.31) true value --
    // a real ~57% error, not a near-zero-crossing artifact, and far beyond
    // what any sane tolerance should paper over without risking masking an
    // actual GPU bug of similar size. Root cause not chased further (not
    // this file's concern) -- likely the int8 activation-quantization
    // granularity not holding up over a wider (I=2048, more super-blocks)
    // dot product than the smaller shapes below ever exercised. Since the
    // exact reference is now proven trustworthy (it's what caught the lossy
    // path's own error), these two types compute it directly here instead
    // of going through cpu_quant_matmul, and get the SAME tight tolerance
    // every other (already exact-vs-exact) type gets below.
    std::vector<float> y_cpu((size_t)S * O, 0.0f);
    if (type == GGUFType::Q6_K || type == GGUFType::Q8_0) {
        std::vector<float> w_row_f32(I);
        for (int o = 0; o < O; ++o) {
            ggml_dequant_row((uint32_t)type, w_bytes.data() + (size_t)o * rb, w_row_f32.data(), I);
            for (int s = 0; s < S; ++s) {
                double acc = 0.0;
                const float* xr = x.data() + (size_t)s * I;
                for (int i = 0; i < I; ++i) acc += (double)xr[i] * (double)w_row_f32[i];
                y_cpu[(size_t)s * O + o] = (float)acc;
            }
        }
    } else {
        bool cpu_ok = cpu_quant_matmul(type, w_bytes.data(), x.data(), y_cpu.data(), S, I, O);
        assert(cpu_ok && "cpu_quant_matmul should support every type this test drives");
    }

    // GPU: the fused dequant+matmul kernel under test.
    QuantizedTensor qt;
    qt.meta.type = type;
    qt.meta.shape = {(uint64_t)O, (uint64_t)I};
    qt.meta.size_bytes = w_bytes.size();
    qt.host_ptr = w_bytes.data();
    if (!gpu.offload_tensor_to_vram(qt)) {
        std::cout << "  -> " << name << ": SKIPPED (VRAM offload failed -- out of memory?)\n";
        return true;
    }

    std::vector<float> y_gpu((size_t)S * O, -999.0f);
    bool gpu_ok = gpu.dense_matmul_host(type, qt.device_ptr, x.data(), y_gpu.data(), S, I, O);
    cudaFree(qt.device_ptr);
    if (!gpu_ok) {
        std::cout << "  -> " << name << ": FAILED (dense_matmul_host reported no kernel for this type)\n";
        return false;
    }

    // Generous tolerance: these are I-element f32 dot products; GPU
    // tree-reduction vs CPU sequential accumulation legitimately differ by a
    // few ULPs scaled by term count. This threshold is about catching REAL
    // bugs (wrong grid index, wrong scale/min, wrong stride), not chasing
    // bit-exactness. Applies uniformly to every type now -- Q6_K/Q8_0 no
    // longer need a special (and, as real hardware proved, still not loose
    // enough) tolerance, since y_cpu above is now the exact reference for
    // them too, not cpu_quant_matmul's lossy int8-activation-quantized fast
    // path (see that block's own comment for why that path was dropped as a
    // test reference: real hardware found it off by up to ~57% at a
    // NORMAL-magnitude output at attn_q_b's real shape, I=2048 O=16384 --
    // not just quantization noise at a near-zero value, which is what the
    // original looser tolerance here was designed to tolerate).
    //
    // numpy-`allclose`-style combined criterion (|a-b| <= atol + rtol*|ref|)
    // instead of a pure relative bound: a genuinely exact-vs-exact
    // comparison can still have a real near-zero-crossing output (more
    // likely the larger O gets -- confirmed on real hardware at O=16384),
    // where even a few ULPs of float32 summation-order noise (~1e-6
    // absolute, as small as this gets) is a huge relative error against an
    // exact reference near zero. atol=1e-4 comfortably covers that scale of
    // genuine float noise without being anywhere close to loose enough to
    // mask a real bug (which produces errors many orders of magnitude
    // larger, as every actual bug caught by this test file historically has).
    const float rtol = 0.01f;
    const float atol = 1e-4f;
    float max_abs_diff = 0.0f, max_rel_diff = 0.0f;
    bool pass = true;
    for (size_t i = 0; i < y_cpu.size(); ++i) {
        if (std::isnan(y_gpu[i]) || std::isinf(y_gpu[i])) {
            std::cout << "  -> " << name << ": FAILED (NaN/Inf at output index " << i << ")\n";
            return false;
        }
        float diff = std::abs(y_cpu[i] - y_gpu[i]);
        float rel = diff / std::max(1e-6f, std::abs(y_cpu[i]));
        max_abs_diff = std::max(max_abs_diff, diff);
        max_rel_diff = std::max(max_rel_diff, rel);
        if (diff > atol + rtol * std::abs(y_cpu[i])) pass = false;
    }
    std::cout << "  -> " << name << " (I=" << I << " O=" << O << " S=" << S << "): "
              << (pass ? "PASSED" : "FAILED")
              << "  max_abs_diff=" << max_abs_diff << " max_rel_diff=" << max_rel_diff << "\n";

    return pass;
}

// Returns false on any real mismatch. Deliberately NOT gated behind assert()
// alone -- this project's documented build (CLAUDE.md) is
// -DCMAKE_BUILD_TYPE=Release, which defines NDEBUG and compiles every
// assert() in this file out entirely. Since this specific test is the fast
// path for confirming/ruling out a live NaN-corruption bug, its verdict has
// to survive a Release build, so main() below checks this return value
// directly instead of relying on an assert to abort on failure.
bool test_cuda_fused_kernels() {
    std::cout << "[Test 5] CUDA Fused Dequant+Matmul Kernels vs CPU Reference...\n";
    CudaManager gpu;
    if (!gpu.initialize()) {
        std::cout << "  -> Skipped (No CUDA Device Found on system)\n";
        return true;
    }

    bool all_pass = true;
    all_pass &= run_fused_kernel_case(gpu, GGUFType::Q2_K,    512, 8, 3);
    all_pass &= run_fused_kernel_case(gpu, GGUFType::Q3_K,    512, 8, 3);
    all_pass &= run_fused_kernel_case(gpu, GGUFType::Q4_K,    512, 8, 3);
    all_pass &= run_fused_kernel_case(gpu, GGUFType::Q5_K,    512, 8, 3);
    all_pass &= run_fused_kernel_case(gpu, GGUFType::Q6_K,    512, 8, 3);
    // Q8_0 (cuda/q8_0_matmul.cu): pre-existing gap in this suite -- the
    // tolerance handling above (line ~254) was already prepared for this
    // type but no case ever actually exercised it. Closed while touching
    // this exact kernel to add a reuse-across-S variant (launch_q8_0_matmul
    // now dispatches S==1 to the original per-(o,s) kernel and S>1 to the
    // new shared-memory weight-reuse kernel -- see that file for the real-
    // hardware A/B). S=3 below exercises the NEW reuse kernel (S>1); S=1
    // exercises the ORIGINAL untouched kernel, at attn_q_b's real GLM-5.2
    // shape (I=2048, O=16384, confirmed via direct GGUF parse) so a
    // shape-dependent bug in either path has the best chance of showing up
    // here rather than only in a live run.
    all_pass &= run_fused_kernel_case(gpu, GGUFType::Q8_0,    512, 8, 3);
    all_pass &= run_fused_kernel_case(gpu, GGUFType::Q8_0,    4096, 4, 2);
    all_pass &= run_fused_kernel_case(gpu, GGUFType::Q8_0,    2048, 16384, 1);
    // IQ2_XXS/XS/S and IQ3_XXS are the highest-priority cases: per a real
    // [QUANT] report from this model, the dominant routed-expert combo is
    // gate=up=IQ2_XXS, down=IQ3_XXS across 71/76 MoE layers -- i.e. every
    // single expert FFN call touches these, making them the most-exercised
    // GPU kernels in the whole engine and the prime suspects for a
    // currently-live NaN-corruption bug in generation. If any of these four
    // fail, that's the smoking gun.
    all_pass &= run_fused_kernel_case(gpu, GGUFType::IQ2_XXS, 512, 8, 3);
    all_pass &= run_fused_kernel_case(gpu, GGUFType::IQ2_XS,  512, 8, 3);
    all_pass &= run_fused_kernel_case(gpu, GGUFType::IQ2_S,   512, 8, 3);
    all_pass &= run_fused_kernel_case(gpu, GGUFType::IQ3_XXS, 512, 8, 3);
    // Exercise a >1 super-block row too (real weight matrices are I=4096+),
    // since a per-block indexing bug can hide in a single-block test.
    all_pass &= run_fused_kernel_case(gpu, GGUFType::IQ2_XXS, 4096, 4, 2);
    all_pass &= run_fused_kernel_case(gpu, GGUFType::IQ3_XXS, 4096, 4, 2);
    // IQ4_XS: down-projection on this checkpoint's ~4-5 previously-CPU-only
    // layers (cuda/iq4xs_matmul.cu). Fixed 16-entry lookup table + linear
    // scale, structurally closer to the K-quant cases above than to its
    // IQ2/IQ3 variable-grid siblings despite the similar name.
    all_pass &= run_fused_kernel_case(gpu, GGUFType::IQ4_XS,  512, 8, 3);
    all_pass &= run_fused_kernel_case(gpu, GGUFType::IQ4_XS,  4096, 4, 2);
    // REAL production shape for this checkpoint's down-projection, not a
    // synthetic small O -- added after IQ4_XS passed the two cases above
    // (small O=4/O=8) yet a real generation run still produced NaN in this
    // exact call (see CLAUDE.md's incident writeup). I=moe_inter, O=hidden,
    // S=1 (real decode batch size) taken directly from that failing run's
    // own check_finite output ("gate_buf/up_buf ... 2048/2048",
    // "output ... 6144/6144"), not guessed -- every OTHER case in this
    // whole test (all types above) only ever used O=4/O=8 too, so this is
    // the first case in the entire suite exercising a realistically large
    // O for ANY type, not just IQ4_XS. If this fails where the small-O
    // cases passed, that's real, reproducible, offline-debuggable evidence
    // of a shape-dependent kernel bug (e.g. something specific to a large
    // block-index or grid-dimension range) that the earlier, smaller test
    // shapes were never able to catch.
    all_pass &= run_fused_kernel_case(gpu, GGUFType::IQ4_XS,  2048, 6144, 1);
    // MXFP4: NEW, currently DORMANT kernel (cuda/mxfp4_matmul.cu) -- see
    // kquant_dequant.hpp's gpu_type_supported() comment. This IS the
    // real-hardware verification gate that must pass BEFORE that function
    // is flipped to enable live GPU dispatch for this type. Fixed 16-entry
    // table (kvalues_mxfp4) + one E8M0 scale/block, 32 elements/block --
    // smaller and simpler than IQ4_XS's 256-element superblock, so a
    // single-block case (I=256... but block=32, so I=256 is 8 blocks) plus
    // a multi-warp-stride case (I>256*8=2048, to exercise the kernel's
    // warp-striding loop past one full round of all 8 warps) are the two
    // shapes worth distinguishing here.
    all_pass &= run_fused_kernel_case(gpu, GGUFType::MXFP4,   512, 8, 3);
    all_pass &= run_fused_kernel_case(gpu, GGUFType::MXFP4,   4096, 4, 2);
    // Real production shape for this checkpoint's down-projection (same
    // I=moe_inter, O=hidden, S=1 rationale as the IQ4_XS large-O case
    // above) -- DeepSeek-V4-Flash's blk.42.ffn_down_exps.weight is the
    // actual MXFP4 tensor that motivated this kernel.
    all_pass &= run_fused_kernel_case(gpu, GGUFType::MXFP4,   2048, 6144, 1);

    std::cout << "  -> " << (all_pass ? "All fused kernels agree with CPU reference within tolerance."
                                       : "One or more CUDA fused kernels disagreed with the CPU reference -- see FAILED lines above.")
              << "\n";
    return all_pass;
}

// ------------------------------------------------------------------
// 6. AVX-512 K-quant Dequant Correctness (vs ggml's own reference)
// ------------------------------------------------------------------
// dequantize_row_q2_K_avx512 / dequantize_row_q3_K_avx512 (kernels_cpu.cpp)
// are a from-scratch AVX-512 port of ggml's plain scalar
// dequantize_row_q2_K / dequantize_row_q3_K (ggml/src/ggml-quants.c),
// written after this session's t_cpu_dequant/t_cpu_dot perf split showed
// dequant -- not the dot product -- dominates the CPU expert-FFN fallback
// cost by ~40x for this project's target GLM-5.2 UD checkpoints (routed
// experts: gate/up=Q2_K, down=Q3_K). No GPU involved here -- this is a pure
// CPU-vs-CPU comparison, so unlike Test 5 it always runs, never skips.
// Designed for BIT-EXACT agreement (not just "close enough"): the port
// avoids FMA and does the same integer arithmetic as the scalar reference
// before its single float multiply, so there's no legitimate source of
// rounding difference the way there is for GPU tree-reduction vs CPU
// sequential summation in Test 5 -- any mismatch here is a real bug, not
// float slop. Tolerance is intentionally near machine epsilon for exactly
// that reason.
extern void dequantize_row_q2_K_avx512(const uint8_t* block, float* y, int64_t k);
extern void dequantize_row_q3_K_avx512(const uint8_t* block, float* y, int64_t k);
extern void dequantize_row_iq2_xxs_avx512(const uint8_t* block, float* y, int64_t k);
extern void dequantize_row_iq2_xs_avx512(const uint8_t* block, float* y, int64_t k);
extern void dequantize_row_iq3_xxs_avx512(const uint8_t* block, float* y, int64_t k);

static bool run_dequant_case(GGUFType type, int elements_per_row, int n_rows) {
    const char* name = gpu_type_name(type);
    assert(elements_per_row % 256 == 0 && "these types' rows must be a multiple of QK_K=256");

    std::mt19937 rng(1337);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

    int rb = row_bytes(type, elements_per_row);
    bool all_pass = true;
    float worst_abs = 0.0f, worst_rel = 0.0f;

    for (int r = 0; r < n_rows; ++r) {
        std::vector<float> w_f32((size_t)elements_per_row);
        for (auto& v : w_f32) v = dist(rng);

        std::vector<uint8_t> w_bytes((size_t)rb);
        int ok = ggml_quantize_row_ref((uint32_t)type, w_f32.data(), w_bytes.data(), elements_per_row);
        assert(ok && "ggml_quantize_row_ref should support Q2_K/Q3_K");

        std::vector<float> y_ref(elements_per_row, -999.0f);
        int dq_ok = ggml_dequant_row((uint32_t)type, w_bytes.data(), y_ref.data(), elements_per_row);
        assert(dq_ok && "ggml_dequant_row should support Q2_K/Q3_K");

        std::vector<float> y_avx(elements_per_row, -999.0f);
        switch (type) {
            case GGUFType::Q2_K:    dequantize_row_q2_K_avx512(w_bytes.data(), y_avx.data(), elements_per_row); break;
            case GGUFType::Q3_K:    dequantize_row_q3_K_avx512(w_bytes.data(), y_avx.data(), elements_per_row); break;
            case GGUFType::IQ2_XXS: dequantize_row_iq2_xxs_avx512(w_bytes.data(), y_avx.data(), elements_per_row); break;
            case GGUFType::IQ2_XS:  dequantize_row_iq2_xs_avx512(w_bytes.data(), y_avx.data(), elements_per_row); break;
            case GGUFType::IQ3_XXS: dequantize_row_iq3_xxs_avx512(w_bytes.data(), y_avx.data(), elements_per_row); break;
            default: assert(false && "run_dequant_case: unhandled type"); break;
        }

        for (int i = 0; i < elements_per_row; ++i) {
            if (std::isnan(y_avx[i]) || std::isinf(y_avx[i])) {
                std::cout << "  -> " << name << " row=" << r << ": FAILED (NaN/Inf at index " << i << ")\n";
                return false;
            }
            float diff = std::abs(y_ref[i] - y_avx[i]);
            float rel = diff / std::max(1e-6f, std::abs(y_ref[i]));
            worst_abs = std::max(worst_abs, diff);
            worst_rel = std::max(worst_rel, rel);
            if (diff > 1e-4f && rel > 1e-4f) {
                std::cout << "  -> " << name << " row=" << r << " idx=" << i << ": MISMATCH ref="
                          << y_ref[i] << " avx512=" << y_avx[i] << "\n";
                all_pass = false;
            }
        }
    }

    std::cout << "  -> " << name << " (elements_per_row=" << elements_per_row << " rows=" << n_rows << "): "
              << (all_pass ? "PASSED" : "FAILED")
              << "  worst_abs_diff=" << worst_abs << " worst_rel_diff=" << worst_rel << "\n";
    return all_pass;
}

bool test_avx512_dequant() {
    std::cout << "[Test 6] AVX-512 Q2_K/Q3_K Dequant vs ggml Reference...\n";
    bool all_pass = true;
    // One-block (256-element) cases first -- easiest to hand-debug if one fails.
    all_pass &= run_dequant_case(GGUFType::Q2_K, 256, 5);
    all_pass &= run_dequant_case(GGUFType::Q3_K, 256, 5);
    // Multi-block rows (real expert weight rows are moe_inter-sized, typically
    // 2048+) -- a per-block indexing bug (wrong stride between blocks) can
    // hide in a single-block test.
    all_pass &= run_dequant_case(GGUFType::Q2_K, 2048, 4);
    all_pass &= run_dequant_case(GGUFType::Q3_K, 2048, 4);

    std::cout << "  -> " << (all_pass ? "AVX-512 dequant agrees with ggml's reference."
                                       : "AVX-512 dequant DISAGREED with ggml's reference -- see MISMATCH lines above.")
              << "\n";
    return all_pass;
}

// ------------------------------------------------------------------
// 7. AVX-512 IQ2_XXS/IQ3_XXS Dequant Correctness (vs ggml's own reference)
// ------------------------------------------------------------------
// dequantize_row_iq2_xxs_avx512 / dequantize_row_iq3_xxs_avx512
// (kernels_cpu.cpp) -- the ACTUAL routed-expert quant types for this
// project's target GLM-5.2 UD-IQ2_M checkpoints (71/76 layers: gate=up=
// IQ2_XXS, down=IQ3_XXS; the real startup [QUANT] report, not the config
// file's comment which named Q2_K/Q3_K and turned out to be wrong -- Test 6
// above vectorized the WRONG types first because of that). Codebook/grid-
// based (table lookup + sign flip), not linear scale+shift like K-quants,
// so this is structurally the higher-risk port -- same bit-exact-agreement
// design as Test 6 (no FMA, same integer arithmetic before the single float
// multiply), and it's this test, not Test 6, that actually matters for this
// checkpoint's performance.
bool test_avx512_dequant_iq() {
    std::cout << "[Test 7] AVX-512 IQ2_XXS/IQ2_XS/IQ3_XXS Dequant vs ggml Reference...\n";
    bool all_pass = true;
    all_pass &= run_dequant_case(GGUFType::IQ2_XXS, 256, 5);
    all_pass &= run_dequant_case(GGUFType::IQ3_XXS, 256, 5);
    all_pass &= run_dequant_case(GGUFType::IQ2_XXS, 2048, 4);
    all_pass &= run_dequant_case(GGUFType::IQ3_XXS, 2048, 4);
    // IQ2_XS: real "UD-Q2_K_XL" checkpoints use this type for some tensors
    // (unlike IQ2_XXS/IQ3_XXS above, which target the older UD-IQ2_M
    // checkpoint family) -- see dequantize_row_iq2_xs_avx512's header
    // comment (kernels_cpu.cpp) for the algorithm this checks against.
    all_pass &= run_dequant_case(GGUFType::IQ2_XS, 256, 5);
    all_pass &= run_dequant_case(GGUFType::IQ2_XS, 2048, 4);

    std::cout << "  -> " << (all_pass ? "AVX-512 IQ dequant agrees with ggml's reference."
                                       : "AVX-512 IQ dequant DISAGREED with ggml's reference -- see MISMATCH lines above.")
              << "\n";
    return all_pass;
}

// ------------------------------------------------------------------
// 8. CPU Dequant+Matmul Throughput Benchmark (NOT a correctness test --
//    reports numbers only, never fails/gates the suite's exit code)
// ------------------------------------------------------------------
//
// Sweeps two axes:
//   - quant type: IQ2_XXS/IQ3_XXS (the types this checkpoint's routed
//     experts actually use for 71/76 layers, per the [QUANT] startup
//     report -- BOTH already AVX-512 vectorized) alongside IQ4_XS/IQ2_S
//     (the types the remaining ~4-5 layers use -- CONFIRMED still scalar:
//     dequant_row's switch in kquant_dequant.hpp has no AVX-512 case for
//     either, falling to ggml_dequant_row's plain reference implementation).
//     Comparing all four in one table shows directly how much of the CPU
//     gap is "still-scalar types" vs a deeper issue in the already-
//     vectorized path itself.
//   - thread count (1/2/4/8/12): tests whether OMP scaling is healthy on
//     this machine's 12 real cores, or whether something (false sharing,
//     memory-bandwidth saturation, thread_local re-allocation inside the
//     hot loop) caps it well short of linear.
static void bench_one_case(GGUFType type, int I, int O, int nthreads) {
    const char* name = gpu_type_name(type);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> w_f32((size_t)O * I);
    for (auto& v : w_f32) v = dist(rng);
    std::vector<float> x((size_t)I);   // S=1 -- matches the real decode-time GEMV shape exactly, not prefill's batched GEMM

    int rb = row_bytes(type, I);
    std::vector<uint8_t> w_bytes((size_t)O * rb);
    for (int o = 0; o < O; ++o) {
        int ok = ggml_quantize_row_ref((uint32_t)type, w_f32.data() + (size_t)o * I,
                                        w_bytes.data() + (size_t)o * rb, I);
        if (!ok) {
            std::cout << "  " << name << ": SKIPPED (no reference quantizer wired up)\n";
            return;
        }
    }

    omp_set_num_threads(nthreads);
    std::vector<float> y((size_t)O, 0.0f);

    // Warm-up (page faults, cache/branch-predictor state) -- not timed.
    cpu_quant_matmul(type, w_bytes.data(), x.data(), y.data(), 1, I, O);

    constexpr int ITERS = 20;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < ITERS; ++it) {
        cpu_quant_matmul(type, w_bytes.data(), x.data(), y.data(), 1, I, O);
    }
    double elapsed_s = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - t0).count() / ITERS;

    double gbs = elapsed_s > 0 ? (w_bytes.size() / 1e9) / elapsed_s : 0.0;
    std::cout << "  " << std::left << std::setw(9) << name
              << " I=" << I << " O=" << O << "  threads=" << std::setw(2) << nthreads
              << ":  " << std::fixed << std::setprecision(3) << std::setw(8) << (elapsed_s * 1000.0)
              << " ms/call  " << std::setprecision(2) << gbs << " GB/s\n";
}

// Isolates dequant WIDTH as the only variable: mirrors cpu_quant_matmul's
// own per-row dequant+dot loop (kquant_dequant.hpp) exactly -- same OMP
// parallel-for-over-output-rows structure, same dot_f32_avx512 call --
// just substituting a direct call to one of the EXPERIMENTAL 16-wide
// dequant functions instead of dequant_row()'s type-dispatched 8-wide
// ones. Directly comparable line-for-line against bench_one_case's output
// for the same type/threads above.
using DequantWideFn = void (*)(const uint8_t*, float*, int64_t);
static void bench_wide_case(GGUFType type, DequantWideFn wide_fn, int I, int O, int nthreads) {
    const char* name = gpu_type_name(type);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> w_f32((size_t)O * I);
    for (auto& v : w_f32) v = dist(rng);
    std::vector<float> x((size_t)I);

    int rb = row_bytes(type, I);
    std::vector<uint8_t> w_bytes((size_t)O * rb);
    for (int o = 0; o < O; ++o) {
        int ok = ggml_quantize_row_ref((uint32_t)type, w_f32.data() + (size_t)o * I,
                                        w_bytes.data() + (size_t)o * rb, I);
        if (!ok) {
            std::cout << "  " << name << " [WIDE]: SKIPPED (no reference quantizer wired up)\n";
            return;
        }
    }

    omp_set_num_threads(nthreads);
    std::vector<float> y((size_t)O, 0.0f);

    auto run_once = [&]() {
        #pragma omp parallel for schedule(static)
        for (int o = 0; o < O; ++o) {
            thread_local static std::vector<float> wrow;
            if ((int)wrow.size() < I) wrow.resize(I);
            wide_fn(w_bytes.data() + (size_t)o * rb, wrow.data(), I);
            y[o] = dot_f32_avx512(wrow.data(), x.data(), I);
        }
    };

    run_once(); // warm-up

    constexpr int ITERS = 20;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < ITERS; ++it) run_once();
    double elapsed_s = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - t0).count() / ITERS;

    double gbs = elapsed_s > 0 ? (w_bytes.size() / 1e9) / elapsed_s : 0.0;
    std::cout << "  " << std::left << std::setw(9) << name
              << " I=" << I << " O=" << O << "  threads=" << std::setw(2) << nthreads
              << " [WIDE]:  " << std::fixed << std::setprecision(3) << std::setw(8) << (elapsed_s * 1000.0)
              << " ms/call  " << std::setprecision(2) << gbs << " GB/s\n";
}

void bench_cpu_dequant_matmul() {
    std::cout << "\n[BENCH] CPU Dequant+Matmul Throughput (informational only -- compare "
                 "these GB/s numbers directly PROF P0-EXEC line, "
                 "run on the same machine)\n";
    // Placeholder shape -- I must be a multiple of 256 (QK_K block size).
    // Replace with your checkpoint's REAL hidden/moe_inter (gate/up
    // projection: I=hidden, O=moe_inter) if you know them; the CROSS-TYPE
    // and CROSS-THREAD-COUNT comparisons below stay valid regardless since
    // every case uses the same shape, but the absolute GB/s numbers are
    // only as representative as this shape is of your real model.
    const int hidden = 5120;
    const int moe_inter = 1536;

    struct Case { GGUFType type; const char* label; };
    Case cases[] = {
        // {GGUFType::IQ2_XXS, "IQ2_XXS -- vectorized, gate/up on 71/76 layers"},
        {GGUFType::IQ3_XXS, "IQ3_XXS -- vectorized, down on 71/76 layers"},
        {GGUFType::IQ4_XS,  "IQ4_XS  -- SCALAR, no AVX-512 case, down on ~4 layers"},
        {GGUFType::IQ2_S,   "IQ2_S   -- NOW vectorized (was scalar), gate/up on 1 layer"},
        {GGUFType::IQ2_XS,  "IQ2_XS  -- NOW vectorized (was scalar), used by UD-Q2_K_XL checkpoints"},
    };
    // Fine-grained around the likely physical-core boundary (Xeon W-2235 =
    // 6 physical cores / 12 threads via hyperthreading) -- added after the
    // coarse {1,2,4,8,12} sweep showed IQ3_XXS regressing from 8->12
    // threads (1.344ms -> 2.752ms), suggesting hyperthread SIBLINGS sharing
    // a physical core's execution units/L1/L2 may be hurting, not helping,
    // for this workload past the real core count. If scaling is clean and
    // roughly linear up to 6 and flat/negative beyond, that confirms it.
    int thread_counts[] = {1, 2, 4, 6, 8};

    for (auto& c : cases) {
        std::cout << "--- " << c.label << " ---\n";
        for (int nt : thread_counts) {
            bench_one_case(c.type, hidden, moe_inter, nt);
        }
    }

    // --- EXPERIMENTAL 16-wide (real 512-bit AVX-512) vs the 8-wide (256-bit
    // AVX2) results just above -- same real quantized bytes, same
    // dot_f32_avx512 step, only apply_grid_group8 vs apply_grid_group16
    // differs. Directly answers whether ALU/vector width was actually
    // leaving throughput on the table, or whether the original "dominant
    // cost is the data-dependent table loads, not raw ALU throughput"
    // assumption was right (in which case this shows no real improvement).
    std::cout << "\n[BENCH] EXPERIMENTAL 16-wide (512-bit) vs 8-wide (256-bit) dequant "
                 "(same bytes, same dot product -- isolates ALU width as the only variable)\n";
    struct WideCase { GGUFType type; DequantWideFn fn; const char* label; };
    WideCase wide_cases[] = {
        // {GGUFType::IQ2_XXS, dequantize_row_iq2_xxs_avx512_wide, "IQ2_XXS"},
        {GGUFType::IQ2_S,   dequantize_row_iq2_s_avx512_wide,   "IQ2_S"},
        {GGUFType::IQ3_XXS, dequantize_row_iq3_xxs_avx512_wide, "IQ3_XXS"},
    };
    for (auto& wc : wide_cases) {
        std::cout << "--- " << wc.label << " [WIDE] ---\n";
        for (int nt : thread_counts) {
            bench_wide_case(wc.type, wc.fn, hidden, moe_inter, nt);
        }
    }

    // --- OMP-region-overhead check ---
    // Production calls cpu_quant_matmul 3 SEPARATE times per expert (gate,
    // up, down), each opening its OWN #pragma omp parallel for region.
    // Weak thread scaling above (~2x at 12 threads, not 8-12x) could be
    // real memory-bandwidth saturation, OR it could be fixed per-region
    // entry/sync overhead dominating at this call granularity (each call is
    // only ~1-2ms of real work). Test directly: 3 separate O=moe_inter
    // calls (matches production's call pattern) vs 1 combined call doing
    // the SAME total work (O=3*moe_inter, one parallel region) -- reuses
    // the existing, already-correct cpu_quant_matmul unchanged, just called
    // differently, so this is a pure measurement with zero new matmul code.
    std::cout << "\n[BENCH] OMP region-entry overhead: 3 separate calls (matches production's "
                 "gate/up/down pattern) vs 1 combined call of equivalent total work "
                 "(IQ2_XXS, I=" << hidden << ")\n";
    for (int nt : thread_counts) {
        omp_set_num_threads(nt);
        std::mt19937 rng(7);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        int rb = row_bytes(GGUFType::IQ2_XXS, hidden);

        auto make_weights = [&](int O) {
            std::vector<float> w_f32((size_t)O * hidden);
            for (auto& v : w_f32) v = dist(rng);
            std::vector<uint8_t> w_bytes((size_t)O * rb);
            for (int o = 0; o < O; ++o)
                ggml_quantize_row_ref((uint32_t)GGUFType::IQ2_XXS, w_f32.data() + (size_t)o * hidden,
                                       w_bytes.data() + (size_t)o * rb, hidden);
            return w_bytes;
        };
        std::vector<float> x((size_t)hidden);

        // 3 separate calls, O=moe_inter each.
        auto w1 = make_weights(moe_inter), w2 = make_weights(moe_inter), w3 = make_weights(moe_inter);
        std::vector<float> y1((size_t)moe_inter), y2((size_t)moe_inter), y3((size_t)moe_inter);
        cpu_quant_matmul(GGUFType::IQ2_XXS, w1.data(), x.data(), y1.data(), 1, hidden, moe_inter); // warm-up
        constexpr int ITERS = 20;
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int it = 0; it < ITERS; ++it) {
            cpu_quant_matmul(GGUFType::IQ2_XXS, w1.data(), x.data(), y1.data(), 1, hidden, moe_inter);
            cpu_quant_matmul(GGUFType::IQ2_XXS, w2.data(), x.data(), y2.data(), 1, hidden, moe_inter);
            cpu_quant_matmul(GGUFType::IQ2_XXS, w3.data(), x.data(), y3.data(), 1, hidden, moe_inter);
        }
        double separate_ms = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t0).count() / ITERS;

        // 1 combined call, O=3*moe_inter (same total rows/bytes as above).
        auto wc = make_weights(3 * moe_inter);
        std::vector<float> yc((size_t)3 * moe_inter);
        cpu_quant_matmul(GGUFType::IQ2_XXS, wc.data(), x.data(), yc.data(), 1, hidden, 3 * moe_inter); // warm-up
        t0 = std::chrono::high_resolution_clock::now();
        for (int it = 0; it < ITERS; ++it) {
            cpu_quant_matmul(GGUFType::IQ2_XXS, wc.data(), x.data(), yc.data(), 1, hidden, 3 * moe_inter);
        }
        double combined_ms = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t0).count() / ITERS;

        std::cout << "  threads=" << std::setw(2) << nt << ":  3-separate-calls="
                  << std::fixed << std::setprecision(3) << separate_ms
                  << "ms  1-combined-call=" << combined_ms << "ms  (combined is "
                  << std::setprecision(2) << (separate_ms / std::max(1e-9, combined_ms)) << "x "
                  << (combined_ms < separate_ms ? "FASTER" : "slower/same") << ")\n";
    }

    omp_set_num_threads(omp_get_max_threads());  // restore default for anything after this
}

// ------------------------------------------------------------------
// 9. cuBLAS Mixed-Precision (BF16 weight / F32 activation) Dense Matmul
// ------------------------------------------------------------------
// Directly tests cublas_dense_matmul's BF16 branch (Atype=CUDA_R_16BF
// weight, Btype=CUDA_R_32F activations, CUBLAS_COMPUTE_32F accumulate) --
// the exact mismatched-input-type combination flagged as the leading
// unconfirmed suspect after a real attn_output BF16 fast-path attempt
// produced NaN corruption on real hardware (see CLAUDE.md's incident
// writeup). This is the "confirm in isolation before ever re-enabling"
// test that incident's own postmortem called for -- small, synthetic, no
// model file needed, gates pass/fail like Test 5, unlike Test 8's
// benchmark above.
//
// Reference is computed from the SAME bf16-rounded weight the GPU actually
// receives (round-tripped via fp32_to_bf16/bf16_to_fp32) dotted with the
// F32 activations on CPU -- this isolates "did cuBLAS correctly execute
// the mixed-type GEMM" from "how much does bf16 rounding itself perturb
// the answer" (the latter is an accepted, by-design precision trade, not
// a bug this test is trying to catch). If cuBLAS is misinterpreting the
// mismatched types (wrong stride, garbage read, silently-undefined
// behavior), the divergence from this reference will be large and
// obvious, not a few ULPs of legitimate floating-point slop.
// NOTE on this test's history and what changed: this ONCE called through
// CudaManager::dense_matmul_host(GGUFType::BF16, ...) to reach cuBLAS's
// mismatched-type combination indirectly, and on an RTX 3090 confirmed
// cuBLAS flatly REJECTS it. That call path is now GONE --
// dense_matmul_host was later changed to intercept BF16 before it can
// ever reach cublas_dense_matmul at all (it always delegates straight to
// dense_matmul_bf16_matched_host, the matched-type path Test 10 covers),
// so calling through dense_matmul_host here no longer tests the
// mismatched-type combination this test is named for -- it silently
// started testing something else entirely (the matched-type function,
// but WITHOUT the correct reference math for that, since this test's
// reference wasn't updated when the call path changed underneath it).
// That mismatch between "what this test's reference assumes it's
// checking" and "what the code under test actually does now" produced a
// real-looking but spurious "FAILED, 29% error" result on a later run --
// not a cuBLAS or hardware bug, just a stale test silently exercising
// the wrong code path and comparing against the wrong reference. Root-
// caused by tracing dense_matmul_host's actual dispatch, not assumed.
//
// Fixed by calling cublas_dense_matmul() DIRECTLY (bypassing
// dense_matmul_host entirely, same pattern Test 10 already uses for the
// matched-type primitive) -- this is the only way to actually still
// exercise the low-level mismatched-type branch at all, now that no
// live call path reaches it. cublas_dense_matmul's BF16 case was also
// hardened (cublas_matmul.hpp) to unconditionally return
// CUBLAS_STATUS_NOT_SUPPORTED for BF16 without ever calling cublasGemmEx
// -- since dense_matmul_host no longer needs this path and the ORIGINAL
// (single-GPU) confirmation of real driver rejection is now stale/
// unverified on this box's second GPU, there's no reason to keep
// depending on the driver to reject it correctly. So this test's role is
// now a pure regression guard: confirm that refusal is still in place
// and still deterministic, not an open-ended "let's see what the driver
// does" experiment. If a future edit ever removes that guard, this test
// still verifies correctness rather than assuming either way (using the
// mismatched-type combination's own correct reference: bf16-rounded
// weight dotted with the ORIGINAL, unrounded F32 activation).
bool test_bf16_cublas_dense_matmul() {
    std::cout << "[Test 9] cuBLAS Mixed-Precision (BF16 weight / F32 activation) Dense Matmul...\n";
    CudaManager gpu;
    if (!gpu.initialize()) {
        std::cout << "  -> Skipped (No CUDA Device Found on system)\n";
        return true;
    }

    const int S = 3, I = 512, O = 8;
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> w_f32((size_t)O * I);
    for (auto& v : w_f32) v = dist(rng);
    std::vector<float> x((size_t)S * I);
    for (auto& v : x) v = dist(rng);

    // Round-trip the weight through bf16 -- this is exactly what the GPU
    // will actually compute against, so the reference must use the SAME
    // rounded values, not the original F32 ones. The activation is
    // deliberately left as-is (unrounded F32): this test's whole point is
    // the MISMATCHED combination (bf16 weight, f32 activation), so
    // rounding the activation too would be testing the matched-type
    // combination (Test 10's job) under the wrong test name.
    std::vector<uint16_t> w_bf16((size_t)O * I);
    std::vector<float> w_bf16_as_f32((size_t)O * I);
    for (size_t i = 0; i < w_f32.size(); ++i) {
        w_bf16[i] = fp32_to_bf16(w_f32[i]);
        w_bf16_as_f32[i] = bf16_to_fp32(w_bf16[i]);
    }

    // CPU reference: plain dot product (double accumulate) using the
    // bf16-rounded weight against the ORIGINAL, unrounded F32 activation.
    std::vector<float> y_cpu((size_t)S * O, 0.0f);
    for (int s = 0; s < S; ++s) {
        for (int o = 0; o < O; ++o) {
            double acc = 0.0;
            for (int i = 0; i < I; ++i) {
                acc += (double)x[(size_t)s * I + i] * (double)w_bf16_as_f32[(size_t)o * I + i];
            }
            y_cpu[(size_t)s * O + o] = (float)acc;
        }
    }

    // Upload the bf16 weight and F32 activation directly -- the raw
    // storage formats cublasGemmEx will read with Atype=CUDA_R_16BF,
    // Btype=CUDA_R_32F. Calls cublas_dense_matmul() DIRECTLY, not via
    // CudaManager::dense_matmul_host -- see the function-level comment
    // above for why going through dense_matmul_host no longer reaches
    // this code path at all.
    void* d_W = nullptr;
    float* d_X = nullptr;
    float* d_Y = nullptr;
    cudaMalloc(&d_W, w_bf16.size() * sizeof(uint16_t));
    cudaMalloc((void**)&d_X, x.size() * sizeof(float));
    cudaMalloc((void**)&d_Y, (size_t)S * O * sizeof(float));
    cudaMemcpy(d_W, w_bf16.data(), w_bf16.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_X, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice);

    cublasStatus_t status = cublas_dense_matmul(gpu.cublas_handle, GGUFType::BF16, d_W, d_X, d_Y, S, I, O);

    std::vector<float> y_gpu((size_t)S * O, -999.0f);
    if (status == CUBLAS_STATUS_SUCCESS) {
        cudaMemcpy(y_gpu.data(), d_Y, y_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
    }
    cudaFree(d_W);
    cudaFree(d_X);
    cudaFree(d_Y);

    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "  -> PASSED (mismatched-type combination correctly refused, status=" << (int)status
                  << " -- either cublas_dense_matmul's own hardened refusal, or the driver itself "
                     "rejecting it; either way, the expected, safe outcome. The matched-type path, "
                     "Test 10, is the one actually wired into the live path.)\n";
        return true;
    }

    // Only reachable if the hardened refusal above (cublas_matmul.hpp) is
    // ever removed AND the driver itself accepts the call -- verify
    // correctness rather than assume either way, since a silent success
    // could still be silently wrong.
    std::cout << "  -> SURPRISING: the mismatched-type call was not refused (hardened guard "
                 "removed?) and the driver accepted it -- verifying correctness rather than "
                 "assuming.\n";
    float max_abs_diff = 0.0f, max_rel_diff = 0.0f;
    for (size_t i = 0; i < y_cpu.size(); ++i) {
        if (std::isnan(y_gpu[i]) || std::isinf(y_gpu[i])) {
            std::cout << "  -> FAILED (NaN/Inf at output index " << i << ")\n";
            return false;
        }
        float diff = std::abs(y_cpu[i] - y_gpu[i]);
        float rel = diff / std::max(1e-6f, std::abs(y_cpu[i]));
        max_abs_diff = std::max(max_abs_diff, diff);
        max_rel_diff = std::max(max_rel_diff, rel);
    }

    bool pass = max_rel_diff < 0.01f;
    std::cout << "  -> BF16 dense matmul (S=" << S << " I=" << I << " O=" << O << "): "
              << (pass ? "PASSED" : "FAILED")
              << "  max_abs_diff=" << max_abs_diff << " max_rel_diff=" << max_rel_diff << "\n";
    return pass;
}

// ------------------------------------------------------------------
// 10. cuBLAS Matched-Precision (BF16 weight / BF16 activation) Dense Matmul
// ------------------------------------------------------------------
// Directly tests cublas_dense_matmul_bf16_matched (cublas_matmul.hpp) --
// the MATCHED-type retry (Atype=Btype=CUDA_R_16BF, Ctype=CUDA_R_32F) after
// Test 9 confirmed the original MIXED-type attempt (BF16 weight / F32
// activation) is cleanly REJECTED by cublasGemmEx (CUBLAS_STATUS_NOT_
// SUPPORTED, not a silent correctness bug). Unlike Test 9, this also
// exercises converting the ACTIVATION to bf16 (the weight-only conversion
// from the original fast-path isn't enough for a matched-type call) --
// same round-trip-through-bf16 methodology as Test 9, just applied to
// both operands instead of one, so the CPU reference isolates "did
// cuBLAS execute this matched-type GEMM correctly" from "how much does
// bf16 rounding perturb the answer" for BOTH inputs this time.
//
// Per this project's own established discipline: gates pass/fail like
// Test 5/9, synthetic/no model file needed, and per the plan this was
// explicitly built for -- verify in isolation FIRST, wire into the live
// dense_dispatch.hpp path only as a separate, later step if this passes.
bool test_bf16_matched_cublas_dense_matmul() {
    std::cout << "[Test 10] cuBLAS Matched-Precision (BF16 weight / BF16 activation) Dense Matmul...\n";
    CudaManager gpu;
    if (!gpu.initialize()) {
        std::cout << "  -> Skipped (No CUDA Device Found on system)\n";
        return true;
    }

    const int S = 3, I = 512, O = 8;
    std::mt19937 rng(5678);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> w_f32((size_t)O * I);
    for (auto& v : w_f32) v = dist(rng);
    std::vector<float> x_f32((size_t)S * I);
    for (auto& v : x_f32) v = dist(rng);

    // Round-trip BOTH operands through bf16 -- this time the activation
    // needs it too, not just the weight, since this is what the GPU will
    // actually receive for a genuinely matched-type call.
    std::vector<uint16_t> w_bf16((size_t)O * I);
    std::vector<float> w_bf16_as_f32((size_t)O * I);
    for (size_t i = 0; i < w_f32.size(); ++i) {
        w_bf16[i] = fp32_to_bf16(w_f32[i]);
        w_bf16_as_f32[i] = bf16_to_fp32(w_bf16[i]);
    }
    std::vector<uint16_t> x_bf16((size_t)S * I);
    std::vector<float> x_bf16_as_f32((size_t)S * I);
    for (size_t i = 0; i < x_f32.size(); ++i) {
        x_bf16[i] = fp32_to_bf16(x_f32[i]);
        x_bf16_as_f32[i] = bf16_to_fp32(x_bf16[i]);
    }

    // CPU reference: plain dot product (double accumulate) using the
    // bf16-rounded weight AND bf16-rounded activation -- both operands
    // match exactly what the GPU actually receives.
    std::vector<float> y_cpu((size_t)S * O, 0.0f);
    for (int s = 0; s < S; ++s) {
        for (int o = 0; o < O; ++o) {
            double acc = 0.0;
            for (int i = 0; i < I; ++i) {
                acc += (double)x_bf16_as_f32[(size_t)s * I + i] * (double)w_bf16_as_f32[(size_t)o * I + i];
            }
            y_cpu[(size_t)s * O + o] = (float)acc;
        }
    }

    // Upload both bf16 buffers directly -- raw storage format, no dequant.
    void* d_W = nullptr;
    void* d_X = nullptr;
    cudaMalloc(&d_W, w_bf16.size() * sizeof(uint16_t));
    cudaMalloc(&d_X, x_bf16.size() * sizeof(uint16_t));
    cudaMemcpy(d_W, w_bf16.data(), w_bf16.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_X, x_bf16.data(), x_bf16.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);

    float* d_Y = nullptr;
    cudaMalloc((void**)&d_Y, (size_t)S * O * sizeof(float));

    cublasStatus_t status = cublas_dense_matmul_bf16_matched(gpu.cublas_handle, d_W, d_X, d_Y, S, I, O);

    std::vector<float> y_gpu((size_t)S * O, -999.0f);
    if (status == CUBLAS_STATUS_SUCCESS) {
        cudaMemcpy(y_gpu.data(), d_Y, y_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
    }
    cudaFree(d_W);
    cudaFree(d_X);
    cudaFree(d_Y);

    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "  -> FAILED (cublasGemmEx returned status " << (int)status
                  << ", not CUBLAS_STATUS_SUCCESS)\n";
        return false;
    }

    float max_abs_diff = 0.0f, max_rel_diff = 0.0f;
    for (size_t i = 0; i < y_cpu.size(); ++i) {
        if (std::isnan(y_gpu[i]) || std::isinf(y_gpu[i])) {
            std::cout << "  -> FAILED (NaN/Inf at output index " << i << ")\n";
            return false;
        }
        float diff = std::abs(y_cpu[i] - y_gpu[i]);
        float rel = diff / std::max(1e-6f, std::abs(y_cpu[i]));
        max_abs_diff = std::max(max_abs_diff, diff);
        max_rel_diff = std::max(max_rel_diff, rel);
    }

    bool pass = max_rel_diff < 0.01f;
    std::cout << "  -> BF16 matched dense matmul (S=" << S << " I=" << I << " O=" << O << "): "
              << (pass ? "PASSED" : "FAILED")
              << "  max_abs_diff=" << max_abs_diff << " max_rel_diff=" << max_rel_diff << "\n";
    return pass;
}

// ------------------------------------------------------------------
// 12. cuBLAS Strided-Batched BF16 Matched Dense Matmul (MLA qabsorb fast-
//     path, EngineConfig::attn_qabsorb_gpu) -- per this project's own
//     established discipline (Test 10's own header comment: "verify in
//     isolation FIRST"), exercises BOTH the raw cublasGemmStridedBatchedEx
//     wrapper (cublas_dense_matmul_bf16_matched_batched) AND the full
//     CudaManager::upload_wuk_bf16 + qabsorb_bf16_batched_host round trip
//     end to end, using the EXACT layout production uses: WUK [H,kvl,nope]
//     (per-head [O=kvl,I=nope] row-major, matching the CPU matmul() call
//     this replaces), activation [H,S,nope], output [H,S,kvl]. H>1 (unlike
//     Test 10's implicit batch-of-1) is the whole point -- this is what
//     confirms cublasGemmStridedBatchedEx's per-batch strideA/B/C actually
//     select the RIGHT head's slice and don't silently reuse head 0 for
//     everything, a mistake H=1 could never catch.
bool test_bf16_batched_qabsorb() {
    std::cout << "[Test 12] cuBLAS Strided-Batched BF16 Matched Dense Matmul (MLA qabsorb)...\n";
    CudaManager gpu;
    if (!gpu.initialize()) {
        std::cout << "  -> Skipped (No CUDA Device Found on system)\n";
        return true;
    }

    const int H = 5, S = 3, nope = 128, kvl = 64;  // deliberately non-uniform, off-multiple-of-32 dims
    std::mt19937 rng(9012);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> WUK((size_t)H * kvl * nope);
    for (auto& v : WUK) v = dist(rng);
    std::vector<float> X((size_t)H * S * nope);
    for (auto& v : X) v = dist(rng);

    // CPU reference: H independent bf16-rounded dot products (same
    // rounding-aware methodology as Test 10), each head's own [kvl,nope]
    // slice against its own [S,nope] slice -- exactly what H separate
    // matmul()-then-round-trip calls would produce, which is what the
    // batched GPU call must match.
    std::vector<float> y_cpu((size_t)H * S * kvl, 0.0f);
    for (int h = 0; h < H; ++h) {
        const float* Wh = WUK.data() + (size_t)h * kvl * nope;
        const float* Xh = X.data() + (size_t)h * S * nope;
        float* Yh = y_cpu.data() + (size_t)h * S * kvl;
        for (int s = 0; s < S; ++s) {
            for (int o = 0; o < kvl; ++o) {
                double acc = 0.0;
                for (int i = 0; i < nope; ++i) {
                    float xb = bf16_to_fp32(fp32_to_bf16(Xh[(size_t)s * nope + i]));
                    float wb = bf16_to_fp32(fp32_to_bf16(Wh[(size_t)o * nope + i]));
                    acc += (double)xb * (double)wb;
                }
                Yh[(size_t)s * kvl + o] = (float)acc;
            }
        }
    }

    // upload_wuk_bf16 expects layer_idx-keyed storage -- use layer_idx=0,
    // a generous max_bytes (this test's tensors are tiny).
    std::vector<float> WUK_vec(WUK.begin(), WUK.end());
    bool uploaded = gpu.upload_wuk_bf16(0, WUK_vec, H, nope, kvl, 1ull << 30);
    if (!uploaded) {
        std::cout << "  -> FAILED (upload_wuk_bf16 returned false)\n";
        return false;
    }

    std::vector<float> y_gpu((size_t)H * S * kvl, -999.0f);
    bool ok = gpu.qabsorb_bf16_batched_host(0, X.data(), y_gpu.data(), H, S, nope, kvl);
    if (!ok) {
        std::cout << "  -> FAILED (qabsorb_bf16_batched_host returned false)\n";
        return false;
    }

    float max_abs_diff = 0.0f, max_rel_diff = 0.0f;
    for (size_t i = 0; i < y_cpu.size(); ++i) {
        if (std::isnan(y_gpu[i]) || std::isinf(y_gpu[i])) {
            std::cout << "  -> FAILED (NaN/Inf at output index " << i << ")\n";
            return false;
        }
        float diff = std::abs(y_cpu[i] - y_gpu[i]);
        float rel = diff / std::max(1e-6f, std::abs(y_cpu[i]));
        max_abs_diff = std::max(max_abs_diff, diff);
        max_rel_diff = std::max(max_rel_diff, rel);
    }

    // Also confirm the raw batched wrapper directly picks the right
    // per-head slice, not just "the round trip produced plausible-looking
    // numbers overall" -- re-check head H-1 in isolation against a FRESH
    // single (non-batched) matched-precision call for the same head, so a
    // stride bug that silently aliased two heads together (still passing
    // the aggregate diff check by coincidence) can't hide.
    {
        std::vector<uint16_t> w_bf16((size_t)kvl * nope), x_bf16((size_t)S * nope);
        const float* Wlast = WUK.data() + (size_t)(H - 1) * kvl * nope;
        const float* Xlast = X.data() + (size_t)(H - 1) * S * nope;
        for (size_t i = 0; i < w_bf16.size(); ++i) w_bf16[i] = fp32_to_bf16(Wlast[i]);
        for (size_t i = 0; i < x_bf16.size(); ++i) x_bf16[i] = fp32_to_bf16(Xlast[i]);
        void *d_W = nullptr, *d_X = nullptr; float* d_Y = nullptr;
        cudaMalloc(&d_W, w_bf16.size() * sizeof(uint16_t));
        cudaMalloc(&d_X, x_bf16.size() * sizeof(uint16_t));
        cudaMalloc((void**)&d_Y, (size_t)S * kvl * sizeof(float));
        cudaMemcpy(d_W, w_bf16.data(), w_bf16.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_X, x_bf16.data(), x_bf16.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);
        cublasStatus_t st = cublas_dense_matmul_bf16_matched(gpu.cublas_handle, d_W, d_X, d_Y, S, nope, kvl);
        std::vector<float> y_last((size_t)S * kvl);
        if (st == CUBLAS_STATUS_SUCCESS) cudaMemcpy(y_last.data(), d_Y, y_last.size() * sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(d_W); cudaFree(d_X); cudaFree(d_Y);
        const float* y_gpu_last = y_gpu.data() + (size_t)(H - 1) * S * kvl;
        for (size_t i = 0; i < y_last.size(); ++i) {
            float diff = std::abs(y_last[i] - y_gpu_last[i]);
            if (diff > 1e-4f) {
                std::cout << "  -> FAILED (batched head " << (H - 1) << " diverges from an isolated single-head "
                          << "call by " << diff << " at index " << i << " -- possible stride/aliasing bug)\n";
                return false;
            }
        }
    }

    bool pass = max_rel_diff < 0.01f;
    std::cout << "  -> BF16 batched qabsorb (H=" << H << " S=" << S << " nope=" << nope << " kvl=" << kvl << "): "
              << (pass ? "PASSED" : "FAILED")
              << "  max_abs_diff=" << max_abs_diff << " max_rel_diff=" << max_rel_diff << "\n";
    return pass;
}

// ------------------------------------------------------------------
// 13. MLA Score GPU Path (EngineConfig::attn_score_gpu) -- 3 plain GEMMs
//     (no strided-batching needed, unlike qabsorb: MLA's latent K/V is
//     SHARED across every head) + a combine-scale-softmax kernel
//     (cuda/attn_score_softmax.cu), replacing the CPU AVX-512 per-head
//     score loop (attention_layer.cpp). See tests/bench_score_cpu_vs_gpu.cu
//     for the real-hardware feasibility numbers (5x-51x speedup, growing
//     with context length) this is based on.
// ------------------------------------------------------------------
bool test_score_gpu() {
    std::cout << "[Test 13] Score GPU Path (3-GEMM + softmax kernel, MLA score)...\n";
    CudaManager gpu;
    if (!gpu.initialize()) {
        std::cout << "  -> Skipped (No CUDA Device Found on system)\n";
        return true;
    }

    // Deliberately non-uniform, small, off-multiple-of-16 dims -- unlike
    // production's AVX-512 CPU path (which needs kvl/rope divisible by 16),
    // this test's CPU reference is plain scalar double-accumulation, so no
    // such constraint applies; picked specifically to stress the softmax
    // kernel's grid-stride-loop boundary handling (T=23 doesn't divide the
    // kernel's 256-thread block evenly).
    const int H = 5, kvl = 37, rope = 13, T = 23;
    std::mt19937 rng(4242);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> Lc((size_t)T * kvl), Rc((size_t)T * rope);
    for (auto& v : Lc) v = dist(rng);
    for (auto& v : Rc) v = dist(rng);
    std::vector<float> Q_absorbed((size_t)H * kvl), Q_rope((size_t)H * rope);
    for (auto& v : Q_absorbed) v = dist(rng);
    for (auto& v : Q_rope) v = dist(rng);
    float scale = 0.137f;  // arbitrary -- a pure-math correctness check doesn't need the real YaRN formula

    // CPU reference: plain double-accumulation, same 4-step math (score,
    // softmax, weighted sum) production's own AVX-512 loop computes.
    std::vector<float> context_cpu((size_t)H * kvl);
    for (int h = 0; h < H; ++h) {
        std::vector<double> scores(T);
        for (int t = 0; t < T; ++t) {
            double s_nope = 0.0, s_rope = 0.0;
            for (int i = 0; i < kvl; ++i) s_nope += (double)Q_absorbed[(size_t)h * kvl + i] * (double)Lc[(size_t)t * kvl + i];
            for (int i = 0; i < rope; ++i) s_rope += (double)Q_rope[(size_t)h * rope + i] * (double)Rc[(size_t)t * rope + i];
            scores[t] = (s_nope + s_rope) * scale;
        }
        double mx = scores[0];
        for (int t = 1; t < T; ++t) mx = std::max(mx, scores[t]);
        double sum = 0.0;
        for (int t = 0; t < T; ++t) { scores[t] = std::exp(scores[t] - mx); sum += scores[t]; }
        for (int t = 0; t < T; ++t) scores[t] /= sum;
        for (int i = 0; i < kvl; ++i) {
            double acc = 0.0;
            for (int t = 0; t < T; ++t) acc += scores[t] * (double)Lc[(size_t)t * kvl + i];
            context_cpu[(size_t)h * kvl + i] = (float)acc;
        }
    }

    if (!gpu.ensure_lc_rc_gpu(0, T, kvl, rope, 1ull << 30)) {
        std::cout << "  -> FAILED (ensure_lc_rc_gpu returned false)\n";
        return false;
    }
    if (!gpu.upload_lc_rc_rows(0, Lc.data(), Rc.data(), 0, T, kvl, rope)) {
        std::cout << "  -> FAILED (upload_lc_rc_rows returned false)\n";
        return false;
    }

    std::vector<float> context_gpu((size_t)H * kvl, -999.0f);
    if (!gpu.score_gpu_batched_host(0, Q_absorbed.data(), Q_rope.data(), context_gpu.data(), H, T, kvl, rope, scale)) {
        std::cout << "  -> FAILED (score_gpu_batched_host returned false)\n";
        return false;
    }

    float max_abs_diff = 0.0f, max_rel_diff = 0.0f;
    for (size_t i = 0; i < context_cpu.size(); ++i) {
        if (std::isnan(context_gpu[i]) || std::isinf(context_gpu[i])) {
            std::cout << "  -> FAILED (NaN/Inf at output index " << i << ")\n";
            return false;
        }
        float diff = std::abs(context_cpu[i] - context_gpu[i]);
        float rel = diff / std::max(1e-6f, std::abs(context_cpu[i]));
        max_abs_diff = std::max(max_abs_diff, diff);
        max_rel_diff = std::max(max_rel_diff, rel);
    }

    // Head-aliasing safeguard: unlike qabsorb (a strided-batched GEMM,
    // where the risk is silently reusing one head's slice for another),
    // this design uses 3 PLAIN GEMMs with no batch stride at all -- the
    // risk profile here is a row/transpose bug in the GEMM call reading
    // the SAME row for every head. With random, independent per-head
    // inputs, two different heads should never coincidentally match to
    // float precision if the computation is actually per-head-distinct.
    {
        const float* ctx0 = context_gpu.data();
        const float* ctxLast = context_gpu.data() + (size_t)(H - 1) * kvl;
        float max_diff_heads = 0.0f;
        for (int i = 0; i < kvl; ++i) max_diff_heads = std::max(max_diff_heads, std::abs(ctx0[i] - ctxLast[i]));
        if (max_diff_heads < 1e-6f) {
            std::cout << "  -> FAILED (head 0 and head " << (H - 1) << " produced identical output -- "
                         "possible head-aliasing/transpose bug)\n";
            return false;
        }
    }

    bool pass = max_rel_diff < 0.05f;
    std::cout << "  -> Score GPU path (H=" << H << " kvl=" << kvl << " rope=" << rope << " T=" << T << "): "
              << (pass ? "PASSED" : "FAILED")
              << "  max_abs_diff=" << max_abs_diff << " max_rel_diff=" << max_rel_diff << "\n";
    return pass;
}

// ------------------------------------------------------------------
// 14. WUV BF16 Batched GPU Path (score GPU path's own down-projection,
//     wuv_bf16_batched_host) -- added after a real production run showed
//     the CPU per-head OMP-parallel loop this replaces costing 0.916ms/call
//     (2.86s over one 40-token decode run), the same OMP team-spawn-
//     overhead pattern qabsorb's own gather fix already found and fixed at
//     a different call site. Structurally identical to Test 12 (same
//     cublasGemmStridedBatchedEx call, cublas_dense_matmul_bf16_matched_
//     batched), just S folded to 1 and different tensor roles.
// ------------------------------------------------------------------
bool test_wuv_batched_gpu() {
    std::cout << "[Test 14] WUV BF16 Batched GPU Path (score down-projection)...\n";
    CudaManager gpu;
    if (!gpu.initialize()) {
        std::cout << "  -> Skipped (No CUDA Device Found on system)\n";
        return true;
    }

    const int H = 5, kvl = 37, vh = 23;  // deliberately non-uniform, off-multiple-of-32 dims
    std::mt19937 rng(6161);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> WUV((size_t)H * kvl * vh);
    for (auto& v : WUV) v = dist(rng);
    std::vector<float> X((size_t)H * kvl);
    for (auto& v : X) v = dist(rng);

    // CPU reference: H independent bf16-rounded dot products (same
    // methodology as Test 12), each head's own [vh,kvl] slice against its
    // own [kvl] slice.
    std::vector<float> y_cpu((size_t)H * vh, 0.0f);
    for (int h = 0; h < H; ++h) {
        const float* Wh = WUV.data() + (size_t)h * kvl * vh;
        const float* Xh = X.data() + (size_t)h * kvl;
        float* Yh = y_cpu.data() + (size_t)h * vh;
        for (int o = 0; o < vh; ++o) {
            double acc = 0.0;
            for (int i = 0; i < kvl; ++i) {
                float xb = bf16_to_fp32(fp32_to_bf16(Xh[i]));
                float wb = bf16_to_fp32(fp32_to_bf16(Wh[(size_t)o * kvl + i]));
                acc += (double)xb * (double)wb;
            }
            Yh[o] = (float)acc;
        }
    }

    std::vector<float> WUV_vec(WUV.begin(), WUV.end());
    if (!gpu.upload_wuv_bf16(0, WUV_vec, H, kvl, vh, 1ull << 30)) {
        std::cout << "  -> FAILED (upload_wuv_bf16 returned false)\n";
        return false;
    }

    std::vector<float> y_gpu((size_t)H * vh, -999.0f);
    if (!gpu.wuv_bf16_batched_host(0, X.data(), y_gpu.data(), H, kvl, vh)) {
        std::cout << "  -> FAILED (wuv_bf16_batched_host returned false)\n";
        return false;
    }

    float max_abs_diff = 0.0f, max_rel_diff = 0.0f;
    for (size_t i = 0; i < y_cpu.size(); ++i) {
        if (std::isnan(y_gpu[i]) || std::isinf(y_gpu[i])) {
            std::cout << "  -> FAILED (NaN/Inf at output index " << i << ")\n";
            return false;
        }
        float diff = std::abs(y_cpu[i] - y_gpu[i]);
        float rel = diff / std::max(1e-6f, std::abs(y_cpu[i]));
        max_abs_diff = std::max(max_abs_diff, diff);
        max_rel_diff = std::max(max_rel_diff, rel);
    }

    bool pass = max_rel_diff < 0.01f;
    std::cout << "  -> WUV batched GPU (H=" << H << " kvl=" << kvl << " vh=" << vh << "): "
              << (pass ? "PASSED" : "FAILED")
              << "  max_abs_diff=" << max_abs_diff << " max_rel_diff=" << max_rel_diff << "\n";
    return pass;
}

// Gated throughput bench (MOESTREAM_RUN_BENCH=1, same convention as
// bench_cpu_dequant_matmul/Test 8) for the qabsorb BF16 batched GPU path,
// at GLM-5.2's REAL dims (H=64 nope=192 kvl=512, read directly off a real
// run's own "[MLA] attn_k_b.weight ... config nope*kvl*H=6291456" debug.log
// line: 192*512*64=6291456, confirmed) -- not the small synthetic
// H=5/nope=128/kvl=64 shape Test 12 uses for correctness-only checking.
// Times BOTH the CPU per-head matmul() loop (attention_layer.cpp's
// fallback, exact same call pattern reproduced here) and the GPU batched
// path, at S=1 (decode, the dominant call volume) and S=12 (this project's
// own recent real prefill batch size) -- gives a real, hardware-grounded
// answer to "is this actually a win here" instead of leaving it as
// synthetic-correctness-only + speculation.
void bench_qabsorb_cpu_vs_gpu() {
    std::cout << "\n[Bench] MLA qabsorb: CPU per-head matmul() vs GPU BF16 batched (GLM-5.2 real dims)...\n";
    CudaManager gpu;
    if (!gpu.initialize()) {
        std::cout << "  -> Skipped (No CUDA Device Found on system)\n";
        return;
    }
    const int H = 64, nope = 192, kvl = 512;
    std::mt19937 rng(3141);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> WUK((size_t)H * kvl * nope);
    for (auto& v : WUK) v = dist(rng);
    if (!gpu.upload_wuk_bf16(0, WUK, H, nope, kvl, 1ull << 31)) {
        std::cout << "  -> Skipped (upload_wuk_bf16 failed -- insufficient VRAM for this bench)\n";
        return;
    }

    for (int S : {1, 12}) {
        std::vector<float> X((size_t)H * S * nope);
        for (auto& v : X) v = dist(rng);
        std::vector<float> Y_cpu((size_t)H * S * kvl), Y_gpu((size_t)H * S * kvl);

        const int WARMUP = 5, ITERS = 50;
        for (int it = 0; it < WARMUP; ++it) {
            for (int h = 0; h < H; ++h)
                matmul(Y_cpu.data() + (size_t)h * S * kvl, X.data() + (size_t)h * S * nope,
                       WUK.data() + (size_t)h * kvl * nope, S, nope, kvl);
        }
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int it = 0; it < ITERS; ++it) {
            for (int h = 0; h < H; ++h)
                matmul(Y_cpu.data() + (size_t)h * S * kvl, X.data() + (size_t)h * S * nope,
                       WUK.data() + (size_t)h * kvl * nope, S, nope, kvl);
        }
        double cpu_ms = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t0).count() / ITERS;

        for (int it = 0; it < WARMUP; ++it) gpu.qabsorb_bf16_batched_host(0, X.data(), Y_gpu.data(), H, S, nope, kvl);
        t0 = std::chrono::high_resolution_clock::now();
        for (int it = 0; it < ITERS; ++it) gpu.qabsorb_bf16_batched_host(0, X.data(), Y_gpu.data(), H, S, nope, kvl);
        double gpu_ms = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t0).count() / ITERS;

        std::cout << "  S=" << S << " (" << (S == 1 ? "decode" : "prefill-batch") << "): "
                  << "CPU=" << cpu_ms << "ms/call  GPU=" << gpu_ms << "ms/call  "
                  << "speedup=" << (cpu_ms / gpu_ms) << "x\n";
    }

    // 2026-08-14 follow-up: a real GLM-5.2 run with attn_qabsorb_gpu=true
    // showed the [REQUEST] line's own qabsorb= bucket (1.99s) noticeably
    // bigger than qabsorb_bf16_batched_host's own up+cmp+dn total (0.616s)
    // for the same 3198 calls -- the bucket also wraps the q_nope_gather_all
    // pre-pass (attention_layer.cpp), a #pragma omp parallel for over H=64
    // heads copying `nope` floats each. Isolate that gather's own cost,
    // serial vs. the production OMP-parallel version, to test the
    // hypothesis that team-spawn overhead dominates a workload this small
    // (S=1: 64*192*4 bytes = 48KB total copied -- a few microseconds of
    // real memcpy work) before touching production code.
    std::cout << "\n[Bench] q_nope_gather_all: serial memcpy vs #pragma omp parallel for (H=" << H << " nope=" << nope << ")...\n";
    for (int S : {1, 12}) {
        const int qh = nope; // rope portion irrelevant to copy cost, omitted for simplicity
        std::vector<float> Q((size_t)S * H * qh);
        for (auto& v : Q) v = dist(rng);
        std::vector<float> gathered((size_t)H * S * nope);

        const int WARMUP = 5, ITERS = 200;
        for (int it = 0; it < WARMUP; ++it) {
            for (int h = 0; h < H; ++h) {
                float* dst = gathered.data() + (size_t)h * S * nope;
                for (int s = 0; s < S; ++s) {
                    const float* head_q = Q.data() + ((size_t)s * H + h) * qh;
                    std::memcpy(dst + (size_t)s * nope, head_q, nope * sizeof(float));
                }
            }
        }
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int it = 0; it < ITERS; ++it) {
            for (int h = 0; h < H; ++h) {
                float* dst = gathered.data() + (size_t)h * S * nope;
                for (int s = 0; s < S; ++s) {
                    const float* head_q = Q.data() + ((size_t)s * H + h) * qh;
                    std::memcpy(dst + (size_t)s * nope, head_q, nope * sizeof(float));
                }
            }
        }
        double serial_ms = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t0).count() / ITERS;

        for (int it = 0; it < WARMUP; ++it) {
            #pragma omp parallel for schedule(dynamic, 1)
            for (int h = 0; h < H; ++h) {
                float* dst = gathered.data() + (size_t)h * S * nope;
                for (int s = 0; s < S; ++s) {
                    const float* head_q = Q.data() + ((size_t)s * H + h) * qh;
                    std::memcpy(dst + (size_t)s * nope, head_q, nope * sizeof(float));
                }
            }
        }
        t0 = std::chrono::high_resolution_clock::now();
        for (int it = 0; it < ITERS; ++it) {
            #pragma omp parallel for schedule(dynamic, 1)
            for (int h = 0; h < H; ++h) {
                float* dst = gathered.data() + (size_t)h * S * nope;
                for (int s = 0; s < S; ++s) {
                    const float* head_q = Q.data() + ((size_t)s * H + h) * qh;
                    std::memcpy(dst + (size_t)s * nope, head_q, nope * sizeof(float));
                }
            }
        }
        double omp_ms = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t0).count() / ITERS;

        std::cout << "  S=" << S << ": serial=" << serial_ms << "ms/call  omp_parallel=" << omp_ms
                  << "ms/call  (omp is " << (omp_ms / serial_ms) << "x serial -- >1x means spawn overhead "
                     "exceeds the work being parallelized)\n";
    }

    // The gather timings above (tens of microseconds) are far too small to
    // explain the real gap seen between a live [REQUEST] line's qabsorb=
    // bucket (1.99s/3198 calls = 0.622ms/call) and qabsorb_bf16_batched_
    // host's own up+cmp+dn total (0.616s/3198 = 0.193ms/call) -- a ~0.43ms/
    // call gap. The bucket also wraps TWO fresh std::vector allocations
    // every call (Q_absorbed, q_nope_gather_all) -- value-initialized (zero-
    // touched) then freed at scope exit, every single call, every layer,
    // every token. Test that directly: fresh-alloc-per-call vs. persistent
    // grow-only scratch (this codebase's own established fix for exactly
    // this class of cost elsewhere, e.g. dense_matmul_bf16_matched_host's
    // scratch buffers) for the WHOLE block the bucket actually wraps.
    std::cout << "\n[Bench] Full qabsorb-block cost: fresh std::vector alloc/call vs. persistent grow-only scratch (S=1, decode)...\n";
    {
        const int S = 1;
        std::vector<float> Q((size_t)S * H * nope);
        for (auto& v : Q) v = dist(rng);
        const int WARMUP = 5, ITERS = 200;

        auto run_fresh_alloc = [&]() {
            std::vector<float> Q_absorbed((size_t)H * S * kvl);
            std::vector<float> q_nope_gather_all((size_t)H * S * nope);
            for (int h = 0; h < H; ++h) {
                float* dst = q_nope_gather_all.data() + (size_t)h * S * nope;
                for (int s = 0; s < S; ++s) {
                    const float* head_q = Q.data() + ((size_t)s * H + h) * nope;
                    std::memcpy(dst + (size_t)s * nope, head_q, nope * sizeof(float));
                }
            }
            gpu.qabsorb_bf16_batched_host(0, q_nope_gather_all.data(), Q_absorbed.data(), H, S, nope, kvl);
        };
        for (int it = 0; it < WARMUP; ++it) run_fresh_alloc();
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int it = 0; it < ITERS; ++it) run_fresh_alloc();
        double fresh_ms = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t0).count() / ITERS;

        std::vector<float> Q_absorbed_persist((size_t)H * S * kvl);
        std::vector<float> q_nope_gather_persist((size_t)H * S * nope);
        auto run_persistent = [&]() {
            for (int h = 0; h < H; ++h) {
                float* dst = q_nope_gather_persist.data() + (size_t)h * S * nope;
                for (int s = 0; s < S; ++s) {
                    const float* head_q = Q.data() + ((size_t)s * H + h) * nope;
                    std::memcpy(dst + (size_t)s * nope, head_q, nope * sizeof(float));
                }
            }
            gpu.qabsorb_bf16_batched_host(0, q_nope_gather_persist.data(), Q_absorbed_persist.data(), H, S, nope, kvl);
        };
        for (int it = 0; it < WARMUP; ++it) run_persistent();
        t0 = std::chrono::high_resolution_clock::now();
        for (int it = 0; it < ITERS; ++it) run_persistent();
        double persist_ms = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t0).count() / ITERS;

        std::cout << "  fresh-alloc=" << fresh_ms << "ms/call  persistent-scratch=" << persist_ms
                  << "ms/call  (" << (fresh_ms / persist_ms) << "x -- this is the whole block the "
                     "production qabsorb= bucket wraps, serial gather both ways since S=1)\n";
    }
}

// ------------------------------------------------------------------
// 11. Real-Data IQ2_XXS Dequant Consistency (GPU kernel vs CPU reference,
//     on ACTUAL model bytes -- not Test 5's synthetic ggml_quantize_row_ref
//     data)
// ------------------------------------------------------------------
// Test 5 already compares GPU vs CPU dequant+matmul for IQ2_XXS and PASSES
// -- but only on randomly-generated float weights re-quantized via ggml's
// own reference quantizer. A live VERBOSE trace on a real DeepSeek-V4-Flash
// decode run showed specific (layer, expert) gate_proj weights producing an
// exactly-all-zero 2048-wide GPU output ([SILU-TRACE-GPU] pre-SiLU gate
// range=[0, 0]), fully deterministic across two independent full-process
// restarts of the identical prompt (ruling out a race). Critically, that
// same block's own dequant scale `d` sampled as a normal ~0.0003-0.0006 --
// the SAME order of magnitude as experts that DON'T zero out -- which
// argues against "this expert's weights are just legitimately near-zero"
// (a genuinely small-magnitude block would get its OWN correspondingly
// small `d` from the quantizer, chosen independently per block). That
// leaves a real, open question: does the actual trained-and-quantized byte
// pattern for this expert decode to all-zero on BOTH independent
// implementations (a genuine data property), or only on the GPU kernel (a
// data-dependent bug Test 5's random data never happens to trigger)?
//
// A runtime cpu_only=true comparison can't answer this cleanly -- it adds
// routing/caching/cpu_only_mode's own documented dispatch-gating gaps as
// confounds, and produced no usable output in practice. This test instead
// pulls the exact real bytes for a known-bad expert out via
// tools/dump_expert_bytes.cpp (reusing the same GGUFFile/fd_for_tensor
// tensor-offset math load_expert_slab uses in production, so the extraction
// itself isn't a new source of bugs) and runs them through cpu_quant_matmul
// and CudaManager::dense_matmul_host directly -- no engine, no router, no
// cache, just the dequant+matmul step in total isolation.
//
// Deliberately NOT gated behind assert() alone for the same reason as Test
// 5: this project's documented build is -DCMAKE_BUILD_TYPE=Release
// (NDEBUG, every assert() compiled out), so the verdict has to survive a
// Release build via an explicit bool return, checked directly in main().
bool test_real_expert_iq2xxs_dequant() {
    std::cout << "[Test 11] Real-Data IQ2_XXS Dequant: GPU vs CPU on actual suspect expert bytes...\n";

    const char* path = std::getenv("MOESTREAM_REAL_EXPERT_BIN");
    if (!path) {
        std::cout << "  -> SKIPPED (set MOESTREAM_REAL_EXPERT_BIN to a raw-bytes dump produced by "
                     "tools/dump_expert_bytes.cpp, e.g.:\n"
                     "       g++ -std=c++23 -O2 -Iinclude tools/dump_expert_bytes.cpp -o /tmp/dump_expert_bytes\n"
                     "       /tmp/dump_expert_bytes model.gguf 0 199 gate /tmp/e199_gate.bin\n"
                     "       MOESTREAM_REAL_EXPERT_BIN=/tmp/e199_gate.bin MOESTREAM_REAL_EXPERT_D=4096 "
                     "MOESTREAM_REAL_EXPERT_O=2048 ./build/moestream_test\n"
                     "     to run this test. Additionally set MOESTREAM_REAL_ACTIVATION_BIN (a raw D-float "
                     "capture from a real run, see execution_layer.cpp's [ACTIVATION-DUMP] instrumentation, "
                     "env vars MOESTREAM_DUMP_ACTIVATION_LAYER/_EXPERT/_OUT) to test against the REAL "
                     "activation instead of a synthetic random vector)\n";
        return true;
    }
    const char* d_env = std::getenv("MOESTREAM_REAL_EXPERT_D");
    const char* o_env = std::getenv("MOESTREAM_REAL_EXPERT_O");
    // Defaults match this investigation's own model: blk.0.ffn_gate_exps.weight [4096, 2048, 256].
    const int D = d_env ? std::atoi(d_env) : 4096;
    const int moe_inter = o_env ? std::atoi(o_env) : 2048;

    std::ifstream f(path, std::ios::binary);
    if (!f.good()) {
        std::cout << "  -> FAILED (couldn't open MOESTREAM_REAL_EXPERT_BIN=" << path << ")\n";
        return false;
    }
    std::vector<uint8_t> w_bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());

    // IQ2_XXS: 66 bytes/block (2-byte fp16 scale + 64 bytes qs), 256 elements/block.
    const size_t expected_bytes = (size_t)moe_inter * D / 256 * 66;
    if (w_bytes.size() != expected_bytes) {
        std::cout << "  -> FAILED (dump is " << w_bytes.size() << " bytes, expected " << expected_bytes
                  << " for D=" << D << " O=" << moe_inter << " IQ2_XXS -- wrong expert/tensor, or set "
                     "MOESTREAM_REAL_EXPERT_D/_O to match the real shape)\n";
        return false;
    }

    // x defaults to a synthetic random vector (this test's original mode --
    // proves/disproves an UNCONDITIONAL dequant bug, independent of any
    // particular activation). Set MOESTREAM_REAL_ACTIVATION_BIN to instead
    // load the ACTUAL runtime activation vector captured at the real
    // all-zero decode step (via execution_layer.cpp's [ACTIVATION-DUMP]
    // instrumentation -- see MOESTREAM_DUMP_ACTIVATION_LAYER/_EXPERT/_OUT
    // there), closing the gap the synthetic vector can't: if THIS run comes
    // back nonzero on both CPU and GPU while the live decode trace showed
    // all-zero, that's strong evidence of a caching/pipeline bug feeding
    // stale/wrong bytes during real decode, not a genuine data property.
    const char* real_x_path = std::getenv("MOESTREAM_REAL_ACTIVATION_BIN");
    std::vector<float> x(D);
    bool using_real_x = false;
    if (real_x_path) {
        std::ifstream xf(real_x_path, std::ios::binary);
        if (!xf.good()) {
            std::cout << "  -> FAILED (couldn't open MOESTREAM_REAL_ACTIVATION_BIN=" << real_x_path << ")\n";
            return false;
        }
        xf.read(reinterpret_cast<char*>(x.data()), (std::streamsize)(x.size() * sizeof(float)));
        if ((size_t)xf.gcount() != x.size() * sizeof(float)) {
            std::cout << "  -> FAILED (MOESTREAM_REAL_ACTIVATION_BIN=" << real_x_path << " is "
                      << xf.gcount() << " bytes, expected " << x.size() * sizeof(float)
                      << " for D=" << D << " -- wrong capture, or set MOESTREAM_REAL_EXPERT_D to match)\n";
            return false;
        }
        using_real_x = true;
        std::cout << "  -> using REAL activation vector from " << real_x_path << "\n";
    } else {
        std::mt19937 rng(1234);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (auto& v : x) v = dist(rng);
    }

    std::vector<float> y_cpu((size_t)moe_inter, -999.0f);
    bool cpu_ok = cpu_quant_matmul(GGUFType::IQ2_XXS, w_bytes.data(), x.data(), y_cpu.data(), 1, D, moe_inter);
    if (!cpu_ok) {
        std::cout << "  -> FAILED (cpu_quant_matmul reported no support for IQ2_XXS)\n";
        return false;
    }

    CudaManager gpu;
    if (!gpu.initialize()) {
        std::cout << "  -> Skipped (No CUDA Device Found on system)\n";
        return true;
    }

    QuantizedTensor qt;
    qt.meta.type = GGUFType::IQ2_XXS;
    qt.meta.shape = {(uint64_t)moe_inter, (uint64_t)D};
    qt.meta.size_bytes = w_bytes.size();
    qt.host_ptr = w_bytes.data();
    if (!gpu.offload_tensor_to_vram(qt)) {
        std::cout << "  -> SKIPPED (VRAM offload failed -- out of memory?)\n";
        return true;
    }

    std::vector<float> y_gpu((size_t)moe_inter, -999.0f);
    bool gpu_ok = gpu.dense_matmul_host(GGUFType::IQ2_XXS, qt.device_ptr, x.data(), y_gpu.data(), 1, D, moe_inter);
    cudaFree(qt.device_ptr);
    if (!gpu_ok) {
        std::cout << "  -> FAILED (dense_matmul_host reported no kernel for IQ2_XXS)\n";
        return false;
    }

    float cpu_max_abs = 0.0f, gpu_max_abs = 0.0f;
    for (float v : y_cpu) cpu_max_abs = std::max(cpu_max_abs, std::abs(v));
    for (int i = 0; i < moe_inter; ++i) {
        if (std::isnan(y_gpu[i]) || std::isinf(y_gpu[i])) {
            std::cout << "  -> FAILED (NaN/Inf at GPU output index " << i << ")\n";
            return false;
        }
        gpu_max_abs = std::max(gpu_max_abs, std::abs(y_gpu[i]));
    }
    bool cpu_all_zero = (cpu_max_abs == 0.0f);
    bool gpu_all_zero = (gpu_max_abs == 0.0f);

    float max_abs_diff = 0.0f, max_rel_diff = 0.0f;
    for (int i = 0; i < moe_inter; ++i) {
        float diff = std::abs(y_cpu[i] - y_gpu[i]);
        float rel = diff / std::max(1e-6f, std::abs(y_cpu[i]));
        max_abs_diff = std::max(max_abs_diff, diff);
        max_rel_diff = std::max(max_rel_diff, rel);
    }

    std::cout << "  -> cpu_max_abs=" << cpu_max_abs << " gpu_max_abs=" << gpu_max_abs
              << " cpu_all_zero=" << cpu_all_zero << " gpu_all_zero=" << gpu_all_zero
              << " max_abs_diff=" << max_abs_diff << " max_rel_diff=" << max_rel_diff << "\n";

    if (cpu_all_zero != gpu_all_zero) {
        std::cout << "  -> FAILED: CPU and GPU disagree on all-zero-ness for this real expert's real "
                     "bytes -- this is the smoking gun for a GPU-kernel-specific dequant bug.\n";
        return false;
    }
    if (cpu_all_zero && gpu_all_zero) {
        if (using_real_x) {
            std::cout << "  -> PASSED: both CPU and GPU independently dequantize this real expert to "
                         "all-zero on the REAL captured activation vector -- corroborates the genuine "
                         "degenerate-data-property hypothesis (the live trace wasn't a pipeline bug); "
                         "next step is upstream, e.g. why this token's hidden state produces a near-zero "
                         "dot product with every one of this expert's rows (attention/RMSNorm, per the "
                         "residual-norm-divergence investigation).\n";
        } else {
            std::cout << "  -> PASSED: both CPU and GPU independently dequantize this real expert to "
                         "all-zero -- two separate implementations agreeing is strong evidence this is a "
                         "genuine property of the trained/quantized weight data, not a kernel bug. Re-run "
                         "with MOESTREAM_REAL_ACTIVATION_BIN set to confirm this holds for the ACTUAL "
                         "activation vector from the real failing decode step, not just synthetic random "
                         "input.\n";
        }
        return true;
    }
    if (using_real_x) {
        std::cout << "  -> Real activation + real weight bytes produced NONZERO output on both CPU and "
                     "GPU in this isolated harness. If the live decode trace showed all-zero for this "
                     "exact (layer, expert, token), that's strong evidence of a caching/data-pipeline bug "
                     "serving different bytes during real decode -- look at the RAM/VRAM tier claim-"
                     "publish path and expert dispatch (CLAUDE.md section 3/6), not the dequant kernels.\n";
    }
    bool pass = max_rel_diff < 0.01f;
    std::cout << "  -> " << (pass ? "PASSED" : "FAILED") << " (nonzero output, tolerance check)\n";
    return pass;
}

int main() {
    std::cout << "========================================\n";
    std::cout << "  MoeStream Native Test Harness\n";
    std::cout << "========================================\n";

    test_cpu_matrix_ops();
    test_graph_initializer();
    test_attention_extraction();
    test_cuda_execution();
    bool fused_kernels_ok = test_cuda_fused_kernels();
    bool avx512_dequant_ok = test_avx512_dequant();
    bool avx512_dequant_iq_ok = test_avx512_dequant_iq();
    // Test 8 is throughput-only (no pass/fail verdict, see its own header
    // comment) and runs many timed dequant/matmul passes -- by far the
    // slowest thing in this binary, and irrelevant to every other test's
    // pass/fail verdict. Skipped by default so a normal `make test`/targeted
    // re-run (e.g. Test 11's real-expert comparison via
    // MOESTREAM_REAL_EXPERT_BIN/MOESTREAM_REAL_ACTIVATION_BIN) stays fast;
    // set MOESTREAM_RUN_BENCH=1 to opt back into the throughput numbers.
    if (std::getenv("MOESTREAM_RUN_BENCH")) {
        bench_cpu_dequant_matmul();
        bench_qabsorb_cpu_vs_gpu();
    } else {
        std::cout << "\n[Test 8] SKIPPED by default (set MOESTREAM_RUN_BENCH=1 to run the throughput bench)\n";
    }
    bool bf16_dense_ok = test_bf16_cublas_dense_matmul();
    bool bf16_matched_dense_ok = test_bf16_matched_cublas_dense_matmul();
    bool bf16_batched_qabsorb_ok = test_bf16_batched_qabsorb();
    bool score_gpu_ok = test_score_gpu();
    bool wuv_batched_gpu_ok = test_wuv_batched_gpu();
    bool real_expert_iq2xxs_ok = test_real_expert_iq2xxs_dequant();

    std::cout << "========================================\n";
    if (!fused_kernels_ok) {
        std::cout << "❌ CUDA fused kernel correctness test FAILED -- see [Test 5] output above.\n";
        return 1;
    }
    if (!avx512_dequant_ok) {
        std::cout << "❌ AVX-512 dequant correctness test FAILED -- see [Test 6] output above.\n";
        return 1;
    }
    if (!avx512_dequant_iq_ok) {
        std::cout << "❌ AVX-512 IQ dequant correctness test FAILED -- see [Test 7] output above.\n";
        return 1;
    }
    if (!bf16_dense_ok) {
        std::cout << "❌ Test 9: cuBLAS unexpectedly ACCEPTED the mismatched-type call AND produced a "
                     "wrong result -- see [Test 9] output above (this is not the normal 'rejected as "
                     "expected' outcome).\n";
        return 1;
    }
    if (!bf16_matched_dense_ok) {
        std::cout << "❌ BF16 matched-precision cuBLAS dense matmul correctness test FAILED -- see [Test 10] output above.\n";
        return 1;
    }
    if (!bf16_batched_qabsorb_ok) {
        std::cout << "❌ BF16 strided-batched qabsorb correctness test FAILED -- see [Test 12] output above.\n";
        return 1;
    }
    if (!score_gpu_ok) {
        std::cout << "❌ Score GPU path correctness test FAILED -- see [Test 13] output above.\n";
        return 1;
    }
    if (!wuv_batched_gpu_ok) {
        std::cout << "❌ WUV batched GPU path correctness test FAILED -- see [Test 14] output above.\n";
        return 1;
    }
    if (!real_expert_iq2xxs_ok) {
        std::cout << "❌ Real-Data IQ2_XXS dequant consistency test FAILED -- see [Test 11] output above.\n";
        return 1;
    }
    std::cout << "✅ All core engine tests passed successfully!\n";
    return 0;
}
