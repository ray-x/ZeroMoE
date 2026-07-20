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
    std::vector<float> y_cpu((size_t)S * O, 0.0f);
    bool cpu_ok = cpu_quant_matmul(type, w_bytes.data(), x.data(), y_cpu.data(), S, I, O);
    assert(cpu_ok && "cpu_quant_matmul should support every type this test drives");

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

    float max_abs_diff = 0.0f, max_rel_diff = 0.0f;
    for (size_t i = 0; i < y_cpu.size(); ++i) {
        if (std::isnan(y_gpu[i]) || std::isinf(y_gpu[i])) {
            std::cout << "  -> " << name << ": FAILED (NaN/Inf at output index " << i << ")\n";
            return false;
        }
        float diff = std::abs(y_cpu[i] - y_gpu[i]);
        float rel = diff / std::max(1e-6f, std::abs(y_cpu[i]));
        max_abs_diff = std::max(max_abs_diff, diff);
        max_rel_diff = std::max(max_rel_diff, rel);
    }

    // Generous tolerance: these are I-element f32 dot products; GPU
    // tree-reduction vs CPU sequential accumulation legitimately differ by a
    // few ULPs scaled by term count. This threshold is about catching REAL
    // bugs (wrong grid index, wrong scale/min, wrong stride), not chasing
    // bit-exactness.
    bool pass = max_rel_diff < 0.01f;
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
    std::cout << "[Test 7] AVX-512 IQ2_XXS/IQ3_XXS Dequant vs ggml Reference...\n";
    bool all_pass = true;
    all_pass &= run_dequant_case(GGUFType::IQ2_XXS, 256, 5);
    all_pass &= run_dequant_case(GGUFType::IQ3_XXS, 256, 5);
    all_pass &= run_dequant_case(GGUFType::IQ2_XXS, 2048, 4);
    all_pass &= run_dequant_case(GGUFType::IQ3_XXS, 2048, 4);

    std::cout << "  -> " << (all_pass ? "AVX-512 IQ dequant agrees with ggml's reference."
                                       : "AVX-512 IQ dequant DISAGREED with ggml's reference -- see MISMATCH lines above.")
              << "\n";
    return all_pass;
}

// ------------------------------------------------------------------
// 8. CPU Dequant+Matmul Throughput Benchmark (NOT a correctness test --
//    reports numbers only, never fails/gates the suite's exit code)
// ------------------------------------------------------------------
// Added to root-cause a real, measured gap: colibri's own PROF report
// (its P0-EXEC line, real steady-state DECODE phase, same machine) shows
// routed-expert CPU dequant+matmul running at 16-23 GB/s; MoEStream's own
// dump_report numbers (avg CPU expert ms/call, back-computed to GB/s) show
// roughly 1.2-1.6 GB/s on the SAME hardware -- a ~4x gap in what turned out
// to be MoEStream's single LARGEST per-token cost bucket (bigger than
// disk-wait). This isolates the CPU dequant-matmul path completely from
// disk I/O, caching, and routing, so the number is attributable to the
// compute path alone. Uses REAL ggml-valid quantized bytes (same
// ggml_quantize_row_ref pattern as Test 5/6/7), not synthetic bit patterns.
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
                 "these GB/s numbers directly against colibri's own PROF P0-EXEC line, "
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
        {GGUFType::IQ2_XXS, "IQ2_XXS -- vectorized, gate/up on 71/76 layers"},
        {GGUFType::IQ3_XXS, "IQ3_XXS -- vectorized, down on 71/76 layers"},
        {GGUFType::IQ4_XS,  "IQ4_XS  -- SCALAR, no AVX-512 case, down on ~4 layers"},
        {GGUFType::IQ2_S,   "IQ2_S   -- NOW vectorized (was scalar), gate/up on 1 layer"},
    };
    // Fine-grained around the likely physical-core boundary (Xeon W-2235 =
    // 6 physical cores / 12 threads via hyperthreading) -- added after the
    // coarse {1,2,4,8,12} sweep showed IQ3_XXS regressing from 8->12
    // threads (1.344ms -> 2.752ms), suggesting hyperthread SIBLINGS sharing
    // a physical core's execution units/L1/L2 may be hurting, not helping,
    // for this workload past the real core count. If scaling is clean and
    // roughly linear up to 6 and flat/negative beyond, that confirms it.
    int thread_counts[] = {1, 2, 3, 4, 5, 6, 7, 8, 10, 12};

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
        {GGUFType::IQ2_XXS, dequantize_row_iq2_xxs_avx512_wide, "IQ2_XXS"},
        {GGUFType::IQ3_XXS, dequantize_row_iq3_xxs_avx512_wide, "IQ3_XXS"},
        {GGUFType::IQ2_S,   dequantize_row_iq2_s_avx512_wide,   "IQ2_S"},
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
    // rounded values, not the original F32 ones.
    std::vector<uint16_t> w_bf16((size_t)O * I);
    std::vector<float> w_bf16_as_f32((size_t)O * I);
    for (size_t i = 0; i < w_f32.size(); ++i) {
        w_bf16[i] = fp32_to_bf16(w_f32[i]);
        w_bf16_as_f32[i] = bf16_to_fp32(w_bf16[i]);
    }

    // CPU reference: plain dot product (double accumulate) using the
    // bf16-rounded weight.
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

    // Upload the bf16 bytes directly (no dequant involved -- this IS the
    // raw storage format cublasGemmEx will read with atype=CUDA_R_16BF).
    void* d_W = nullptr;
    cudaMalloc(&d_W, w_bf16.size() * sizeof(uint16_t));
    cudaMemcpy(d_W, w_bf16.data(), w_bf16.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);

    std::vector<float> y_gpu((size_t)S * O, -999.0f);
    bool gpu_ok = gpu.dense_matmul_host(GGUFType::BF16, d_W, x.data(), y_gpu.data(), S, I, O);
    cudaFree(d_W);
    if (!gpu_ok) {
        std::cout << "  -> FAILED (dense_matmul_host reported no kernel for BF16)\n";
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
    std::cout << "  -> BF16 dense matmul (S=" << S << " I=" << I << " O=" << O << "): "
              << (pass ? "PASSED" : "FAILED")
              << "  max_abs_diff=" << max_abs_diff << " max_rel_diff=" << max_rel_diff << "\n";
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
    bench_cpu_dequant_matmul();
    bool bf16_dense_ok = test_bf16_cublas_dense_matmul();

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
        std::cout << "❌ BF16 cuBLAS dense matmul correctness test FAILED -- see [Test 9] output above.\n";
        return 1;
    }
    std::cout << "✅ All core engine tests passed successfully!\n";
    return 0;
}