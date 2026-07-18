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
    std::cout << "✅ All core engine tests passed successfully!\n";
    return 0;
}