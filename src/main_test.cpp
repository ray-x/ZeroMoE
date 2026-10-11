#include "qwen4exp_gdn.hpp"   // qwen4exp_gdn_delta_chunk (CPU oracle for Test 25)
#include "ggml_gdn_proj_backend.hpp"  // GgmlGdnProjRunner (Test 29)
#include "ggml_glm5next_proj_backend.hpp"  // GLM5NEXT KDA fusion (Test 34)
#include "ggml_moe_backend.hpp"  // GgmlQwenSharedRunner (Test 35)
#include "ggml_hc_backend.hpp"   // GgmlHcRunner (Test 36)
#include "kernels_cpu.hpp"
#include "ram_manager.hpp"
#include "engine.hpp"
#include "cuda_manager.hpp"
#include "kquant_dequant.hpp"
#include "engine_config.hpp"
#include "exl3_cpu.hpp"
#include "exl3_cuda.hpp"
#include "exl3_gpu.hpp"
#include "cuda_kernel_dispatcher.hpp"
#include "exl3_format.hpp"
#include "exl3_model_loader.hpp"
#if defined(MOESTREAM_USE_GGML_CUDA)
#include "exl3_ggml_ffn.hpp"
#endif

#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include <array>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <limits>
#include <numeric>
#include <random>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <unordered_map>
#include <omp.h>
#ifdef __linux__
#include <sys/mman.h>
#include <unistd.h>
#endif

// Defined (non-static) in src/attention_layer.cpp specifically so Test 28
// can compare qwen4exp_conv1d_silu_gpu_host's GPU kernel directly against
// this already-proven CPU reference. No header declares it (it's an
// internal helper, not part of any public API), so it's forward-declared
// here to match.
void attn_causal_conv1d_silu(const float* x, const float* w, float* out,
                             int S, int channels, int kernel, float* history);

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
extern "C" void quantize_row_q8_0(const float *x, void *y, int64_t k);
extern "C" void quantize_row_q8_K(const float *x, void *y, int64_t k);
extern "C" void ggml_vec_dot_q3_K_q8_K(
    int n, float *s, size_t bs, const void *vx, size_t bx, const void *vy,
    size_t by, int nrc);
extern "C" void ggml_vec_dot_q2_0_q8_0(
    int n, float *s, size_t bs, const void *vx, size_t bx, const void *vy,
    size_t by, int nrc);
extern "C" void ggml_vec_dot_iq4_nl_q8_0(
    int n, float *s, size_t bs, const void *vx, size_t bx, const void *vy,
    size_t by, int nrc);
extern "C" void ggml_cpu_init(void);

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
    const int required_block_size = type == GGUFType::Q2_0 ? 64 : 256;
    assert(I % required_block_size == 0 && "test case input size is not block aligned");

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
    // IQ1_S and IQ1_M both join this branch: neither has a cpu_quant_matmul
    // fast-path kernel of its own (never had one, so no "lossy production
    // fast path vs exact reference" ambiguity to worry about the way
    // Q6_K/Q8_0's own comment above documents) -- ggml_dequant_row already
    // dispatches both correctly (dequantize_row_iq1_s/_iq1_m, confirmed via
    // src/ggml_dequant.cpp), so the exact dequant+naive-dot path is both
    // the simplest and the only correct reference available here.
    if (type == GGUFType::Q6_K || type == GGUFType::Q8_0 ||
        type == GGUFType::IQ1_S || type == GGUFType::IQ1_M) {
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
    all_pass &= run_fused_kernel_case(gpu, GGUFType::Q2_0,    512, 8, 3);
    all_pass &= run_fused_kernel_case(gpu, GGUFType::Q2_0,    640, 2560, 1);
    all_pass &= run_fused_kernel_case(gpu, GGUFType::Q2_0,    640, 32, 3);
    all_pass &= run_fused_kernel_case(gpu, GGUFType::Q2_0,   2048, 16, 1);
    all_pass &= run_fused_kernel_case(gpu, GGUFType::Q2_K,    512, 8, 3);
    all_pass &= run_fused_kernel_case(gpu, GGUFType::Q3_K,    512, 8, 3);
    all_pass &= run_fused_kernel_case(gpu, GGUFType::Q4_K,    512, 8, 3);
    all_pass &= run_fused_kernel_case(gpu, GGUFType::Q5_K,    512, 8, 3);
    all_pass &= run_fused_kernel_case(gpu, GGUFType::Q6_K,    512, 8, 3);

    // 2026-09-01: the five cases above all use I=512 -> bpr = I/256 = 2, which
    // sits BELOW kquant_lane8_ok()'s `bpr >= 8` gate (cuda/kquant_matmul.cu) --
    // so before these were added, Test 5 exercised ONLY the cooperative path
    // and had literally zero coverage of the lane8 kernels, despite lane8
    // being what every real GLM-5.3 dense tensor actually dispatches to
    // (attn_output is I=16384 -> bpr=64). Same class of gap as the Q8_0 one
    // this test already documents finding and fixing.
    //
    // I=16384 is attn_output's real production shape (309 Q5_K + 81 Q6_K
    // dense tensors in GLM-5.3); I=6144 is the hidden-size square shape.
    // Both are >= 2048, so both take the lane8 path.
    all_pass &= run_fused_kernel_case(gpu, GGUFType::Q2_K,   16384, 64, 1);
    all_pass &= run_fused_kernel_case(gpu, GGUFType::Q3_K,   16384, 64, 1);
    all_pass &= run_fused_kernel_case(gpu, GGUFType::Q4_K,   16384, 64, 1);
    all_pass &= run_fused_kernel_case(gpu, GGUFType::Q5_K,   16384, 64, 1);
    all_pass &= run_fused_kernel_case(gpu, GGUFType::Q6_K,   16384, 64, 1);
    // S>1 as well: the lane8 kernel keeps grid(O,S), so an S>1 launch must be
    // confirmed to index x_row correctly per row, not just at s=0.
    all_pass &= run_fused_kernel_case(gpu, GGUFType::Q5_K,    6144, 32, 3);
    all_pass &= run_fused_kernel_case(gpu, GGUFType::Q6_K,    6144, 32, 3);
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

    // Q5_1 (cuda/q5_1_matmul.cu, new 2026-09-04). No lossy CPU fast path
    // exists for this type in cpu_quant_matmul, so the default comparison
    // above is already exact-vs-exact (ggml's own dequantize_row_q5_1 +
    // naive dot) -- no exact-reference special case needed the way Q6_K/
    // Q8_0 required one.
    //
    // These shapes are chosen to cover BOTH of the launcher's reduction
    // paths, which the real model alone would not: launch_q5_1_matmul sizes
    // blockDim from blocks_per_row, so bpr<=32 takes the single-warp
    // shuffle-only path (no shared memory, no __syncthreads) and bpr>32
    // takes the shared-memory cross-warp path. qwen4exp's real down_proj is
    // I=640 -> bpr=20, i.e. it only ever exercises the FIRST path -- a bug
    // in the second would have shipped invisibly.
    all_pass &= run_fused_kernel_case(gpu, GGUFType::Q5_1,     512, 8, 3);     // bpr=16,  1 warp,  S>1
    all_pass &= run_fused_kernel_case(gpu, GGUFType::Q5_1,     640, 2560, 1);  // REAL qwen4exp ffn_down_exps shape, decode
    all_pass &= run_fused_kernel_case(gpu, GGUFType::Q5_1,     640, 32, 3);    // real I, prefill S>1
    all_pass &= run_fused_kernel_case(gpu, GGUFType::Q5_1,    2048, 16, 1);    // bpr=64,  multi-warp shared-mem reduction
    all_pass &= run_fused_kernel_case(gpu, GGUFType::Q5_1,    4096, 4, 2);     // bpr=128, threads stride >1 block each
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
    // IQ1_S: added 2026-09 when a real launch_iq1s_matmul (verbatim port of
    // ggml's dequantize_row_iq1_s, real 2048-entry grid codebook) replaced
    // this project's pre-existing but WRONG "iq1_matmul_kernel_optimized"
    // (found this session to decode flat sign bits with no grid lookup, no
    // per-group scale, no delta -- see cuda/iq1_dequant.cu's own comment).
    // Real motivation: GLM-5.3's Unsloth "UD-IQ1_S" release keeps gate/up at
    // this type for a real subset of MoE layers. Small shape first, then
    // this model's real per-expert gate/up dims (I=2048, O=6144, confirmed
    // against a real downloaded shard's own byte count: row_bytes(IQ1_S,
    // 2048)=400, 400*6144=2,457,600 bytes, matching the file exactly) at
    // both S=1 (decode) and a real prefill-batch-sized S.
    all_pass &= run_fused_kernel_case(gpu, GGUFType::IQ1_S,   512, 8, 3);
    all_pass &= run_fused_kernel_case(gpu, GGUFType::IQ1_S,   2048, 6144, 1);
    all_pass &= run_fused_kernel_case(gpu, GGUFType::IQ1_S,   2048, 6144, 12);
    // IQ1_M: added 2026-09 alongside IQ1_S -- launch_iq1m_matmul reuses the
    // same c_iq1s_grid table (see that kernel's own header comment) but a
    // genuinely different per-block scale/index packing (block_iq1_m:
    // qs[32]+qh[16]+scales[8], no standalone `d`), so it needs its own
    // independent correctness check, not just "IQ1_S passed, so this
    // probably does too". No real downloaded GLM-5.3 shard has ever been
    // confirmed to contain IQ1_M tensors (checked: 0 across every shard
    // this project currently has) -- these synthetic-data shapes are the
    // ONLY verification this type has, unlike IQ1_S's later real-model
    // confirmation. Same three shapes as IQ1_S for a direct comparison.
    all_pass &= run_fused_kernel_case(gpu, GGUFType::IQ1_M,   512, 8, 3);
    all_pass &= run_fused_kernel_case(gpu, GGUFType::IQ1_M,   2048, 6144, 1);
    all_pass &= run_fused_kernel_case(gpu, GGUFType::IQ1_M,   2048, 6144, 12);
    // IQ3_S: added 2026-09 when a real launch_iq3s_matmul (verbatim port of
    // ggml's dequantize_row_iq3_s, real 512-entry grid codebook + qh 9th-bit
    // extension) replaced the "no fused CUDA kernel" state this type had
    // (see kquant_dequant.hpp's gpu_type_supported() -- IQ3_S was the last
    // member of the IQ2/IQ3 grid-codebook family without one). Real
    // motivation: a real downloaded Unsloth Dynamic-quant GLM-5.3 shard set
    // (7 shards, "UD-IQ3_XXS"-labeled) keeps gate/up=IQ2_S, down=IQ3_S
    // uniformly across nearly all 75 MoE layers (confirmed via direct probe
    // of every layer, 2026-09) -- production shape I=2048, O=6144 matches
    // this file's own real down_proj dims exactly (row_bytes(IQ3_S,
    // 2048)=880, 880*6144=5,406,720 bytes, matching the file byte-for-byte).
    all_pass &= run_fused_kernel_case(gpu, GGUFType::IQ3_S,   512, 8, 3);
    all_pass &= run_fused_kernel_case(gpu, GGUFType::IQ3_S,   2048, 6144, 1);
    all_pass &= run_fused_kernel_case(gpu, GGUFType::IQ3_S,   2048, 6144, 12);
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
    // IQ4_NL: added 2026-09-04 with launch_iq4_nl_matmul. I=640 is the REAL
    // down-projection input dim for Qwen3.8-Flash-Next (moe_inter), the shape
    // that model's 48 MoE layers actually run; the 2048 case adds a larger
    // bpr, and S=1/S>1 are both covered since the kernel strides blocks by 8.
    all_pass &= run_fused_kernel_case(gpu, GGUFType::IQ4_NL,  640, 8, 1);
    all_pass &= run_fused_kernel_case(gpu, GGUFType::IQ4_NL,  640, 4, 3);
    all_pass &= run_fused_kernel_case(gpu, GGUFType::IQ4_NL,  2048, 5, 2);
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
// 7b. CPU IQ matmul paths and grouped-wrapper consistency
// ------------------------------------------------------------------
// The decode path uses three different CPU contracts here:
//   * IQ2_XXS/IQ3_XXS default to MoEStream's exact dequant+F32 dot path;
//   * IQ1_M optionally uses ggml's production IQ1_M x Q8_K dot path;
//   * the Qwen grouped wrappers must produce the same result as their
//     one-matrix form, regardless of whether the caller has one or many
//     routed experts.
// These tests deliberately use real GGML-quantized bytes, not synthetic block
// contents.  They catch row-stride, block-size, expert-index, and grouped
// activation-quantization mistakes without loading a model.
static bool exact_iq_matmul_reference(GGUFType type, const uint8_t *weights,
                                      const float *x, float *y, int S, int I,
                                      int O) {
    const int rb = row_bytes(type, I);
    std::vector<float> row((size_t)I);
    for (int o = 0; o < O; ++o) {
        if (!ggml_dequant_row((uint32_t)type,
                              weights + (size_t)o * rb, row.data(), I))
            return false;
        for (int s = 0; s < S; ++s) {
            double acc = 0.0;
            for (int i = 0; i < I; ++i)
                acc += (double)row[(size_t)i] *
                       (double)x[(size_t)s * I + i];
            y[(size_t)s * O + o] = (float)acc;
        }
    }
    return true;
}

static bool compare_iq_outputs(const std::string &label, const float *a,
                               const float *b, size_t n, float atol,
                               float rtol) {
    float max_abs = 0.0f;
    float max_rel = 0.0f;
    bool ok = true;
    for (size_t i = 0; i < n; ++i) {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i])) {
            ok = false;
            continue;
        }
        const float diff = std::abs(a[i] - b[i]);
        max_abs = std::max(max_abs, diff);
        max_rel = std::max(max_rel,
                           diff / std::max(1e-6f, std::abs(b[i])));
        if (diff > atol + rtol * std::abs(b[i])) ok = false;
    }
    std::cout << "  -> " << label << ": " << (ok ? "PASSED" : "FAILED")
              << " max_abs=" << max_abs << " max_rel=" << max_rel << "\n";
    return ok;
}

bool test_qwen_q2_0_vnni_kernel() {
    ggml_cpu_init();
    constexpr int D = 2560;
    constexpr int FF = 640;
    constexpr int N = 10;
    constexpr int ITERATIONS = 20;
    const int input_row_bytes = row_bytes(GGUFType::Q2_0, D);
    const int down_row_bytes = row_bytes(GGUFType::Q2_0, FF);
    std::mt19937 rng(20260928);
    std::uniform_real_distribution<float> dist(-0.15f, 0.15f);
    std::vector<uint8_t> gate((size_t)N * FF * input_row_bytes);
    std::vector<uint8_t> up((size_t)N * FF * input_row_bytes);
    std::vector<uint8_t> down((size_t)N * D * down_row_bytes);
    // Quantize each real GGUF Q2_0 row using GGML's reference quantizer.
    auto fill_plane = [&](std::vector<uint8_t> &plane, int rows, int width,
                          int rb) {
        std::vector<float> values((size_t)width);
        for (int e = 0; e < N; ++e) {
            for (int r = 0; r < rows; ++r) {
                for (float &v : values)
                    v = dist(rng);
                if (!ggml_quantize_row_ref(
                        (uint32_t)GGUFType::Q2_0, values.data(),
                        plane.data() + ((size_t)e * rows + r) * (size_t)rb,
                        width))
                    return false;
            }
        }
        return true;
    };
    if (!fill_plane(gate, FF, D, input_row_bytes) ||
        !fill_plane(up, FF, D, input_row_bytes) ||
        !fill_plane(down, D, FF, down_row_bytes)) {
        std::cout << "  -> Q2_0 fused VNNI: FAILED (quantizer unavailable)\n";
        return false;
    }

    std::vector<const uint8_t *> gate_ptrs(N), up_ptrs(N), down_ptrs(N);
    std::vector<std::vector<float>> gate_y(N, std::vector<float>(FF));
    std::vector<std::vector<float>> up_y(N, std::vector<float>(FF));
    std::vector<std::vector<float>> down_y(N, std::vector<float>(D));
    std::vector<float *> gate_out(N), up_out(N), down_out(N);
    for (int e = 0; e < N; ++e) {
        gate_ptrs[e] = gate.data() + (size_t)e * FF * input_row_bytes;
        up_ptrs[e] = up.data() + (size_t)e * FF * input_row_bytes;
        down_ptrs[e] = down.data() + (size_t)e * D * down_row_bytes;
        gate_out[e] = gate_y[e].data();
        up_out[e] = up_y[e].data();
        down_out[e] = down_y[e].data();
    }
    std::vector<float> input((size_t)D);
    for (float &v : input)
        v = dist(rng);
    std::vector<block_q8_0> input_q((size_t)D / 32);
    quantize_row_q8_0(input.data(), input_q.data(), D);
    for (int e = 0; e < N; ++e) {
        for (int o = 0; o < FF; ++o) {
            ggml_vec_dot_q2_0_q8_0(
                D, &gate_y[e][o], 0,
                gate_ptrs[e] + (size_t)o * input_row_bytes, 0,
                input_q.data(), 0, 1);
            ggml_vec_dot_q2_0_q8_0(
                D, &up_y[e][o], 0,
                up_ptrs[e] + (size_t)o * input_row_bytes, 0,
                input_q.data(), 0, 1);
        }
    }
    const auto reference_gate_dot = gate_y;
    const auto reference_up_dot = up_y;
    for (int e = 0; e < N; ++e) {
        silu_inplace(gate_y[e].data(), FF);
        elem_mul(gate_y[e].data(), gate_y[e].data(), up_y[e].data(), FF);
    }
    std::vector<block_q8_0> down_q((size_t)N * FF / 32);
    for (int e = 0; e < N; ++e) {
        quantize_row_q8_0(gate_y[e].data(),
                          down_q.data() + (size_t)e * FF / 32, FF);
        for (int o = 0; o < D; ++o)
            ggml_vec_dot_q2_0_q8_0(
                FF, &down_y[e][o], 0,
                down_ptrs[e] + (size_t)o * down_row_bytes, 0,
                down_q.data() + (size_t)e * FF / 32, 0, 1);
    }
    const auto reference_gate = gate_y;
    const auto reference_up = up_y;
    const auto reference_down = down_y;

    // Isolated dot-only A/B on the same real GGUF Q2_0 rows and already
    // quantized Q8_0 inputs used by the fused-route test below. Quantization,
    // activation, routing, and output reduction are deliberately outside the
    // timed region. Keep the same five-worker team as the direct-route test.
    struct Q2DotCase {
        const uint8_t *weights;
        const block_q8_0 *activations;
        int n;
        float expected;
    };
    constexpr int DOT_THREADS = 5;
    constexpr int DOT_ITERATIONS = 20;
    std::vector<Q2DotCase> dot_cases;
    dot_cases.reserve((size_t)N * (2 * FF + D));
    for (int e = 0; e < N; ++e) {
        for (int o = 0; o < FF; ++o) {
            dot_cases.push_back({gate_ptrs[e] + (size_t)o * input_row_bytes,
                                 input_q.data(), D, reference_gate_dot[e][o]});
            dot_cases.push_back({up_ptrs[e] + (size_t)o * input_row_bytes,
                                 input_q.data(), D, reference_up_dot[e][o]});
        }
        const block_q8_0 *expert_down_q =
            down_q.data() + (size_t)e * FF / 32;
        for (int o = 0; o < D; ++o)
            dot_cases.push_back({down_ptrs[e] + (size_t)o * down_row_bytes,
                                 expert_down_q, FF, reference_down[e][o]});
    }
    std::vector<float> dot_output(dot_cases.size());
    std::vector<float> ggml_dot_output;
    const bool use_vnni = q2_0_vnni_enabled();
    auto run_dot_set = [&](bool use_vnni_backend) {
#pragma omp parallel for schedule(static) num_threads(DOT_THREADS)
        for (int64_t i = 0; i < (int64_t)dot_cases.size(); ++i) {
            const Q2DotCase &test = dot_cases[(size_t)i];
            dot_output[(size_t)i] = dot_q2_0_q8_0_cpu(
                test.weights, test.activations, test.n, use_vnni_backend);
        }
    };
    run_dot_set(false);
    ggml_dot_output = dot_output;
    run_dot_set(use_vnni);
    double ggml_dot_ms = 0.0;
    double selected_dot_ms = 0.0;
    for (int iteration = 0; iteration < DOT_ITERATIONS; ++iteration) {
        auto start = std::chrono::steady_clock::now();
        const bool first_is_vnni = (iteration & 1) != 0;
        run_dot_set(first_is_vnni && use_vnni);
        double elapsed = std::chrono::duration<double, std::milli>(
                             std::chrono::steady_clock::now() - start)
                             .count();
        if (first_is_vnni) selected_dot_ms += elapsed;
        else ggml_dot_ms += elapsed;

        start = std::chrono::steady_clock::now();
        run_dot_set(!first_is_vnni && use_vnni);
        elapsed = std::chrono::duration<double, std::milli>(
                      std::chrono::steady_clock::now() - start)
                      .count();
        if (first_is_vnni) ggml_dot_ms += elapsed;
        else selected_dot_ms += elapsed;
    }
    ggml_dot_ms /= DOT_ITERATIONS;
    selected_dot_ms /= DOT_ITERATIONS;
    run_dot_set(use_vnni);
    std::vector<float> dot_expected;
    dot_expected.reserve(dot_cases.size());
    for (const Q2DotCase &test : dot_cases)
        dot_expected.push_back(test.expected);
    bool ggml_dot_ok = compare_iq_outputs(
        "Q2_0 GGML dot", ggml_dot_output.data(), dot_expected.data(),
        ggml_dot_output.size(), 2e-5f, 2e-5f);
    bool dot_ok = compare_iq_outputs("Q2_0 dot-only selected", dot_output.data(),
                                     dot_expected.data(), dot_output.size(),
                                     2e-5f, 2e-5f) && ggml_dot_ok;
    float grouped_dot[4]{};
    float grouped_reference[4]{};
    for (int row = 0; row < 4; ++row) {
        ggml_vec_dot_q2_0_q8_0(
            D, &grouped_reference[row], 0,
            gate_ptrs[0] + (size_t)row * input_row_bytes, 0,
            input_q.data(), 0, 1);
    }
    ggml_vec_dot_q2_0_q8_0(
        D, grouped_dot, sizeof(float), gate_ptrs[0], input_row_bytes,
        input_q.data(), 0, 4);
    dot_ok = compare_iq_outputs(
                 "Q2_0 four-row VNNI dot", grouped_dot, grouped_reference, 4,
                 2e-5f, 2e-5f) && dot_ok;
    std::cout << "  -> Q2_0 dot-only (" << dot_cases.size() << " real rows, "
              << DOT_THREADS << " threads): GGML " << std::fixed
              << std::setprecision(3) << ggml_dot_ms << " ms/pass | selected "
              << selected_dot_ms << " ms/pass ("
              << (q2_0_vnni_enabled() ? "VNNI" : "GGML fallback") << ")\n";

    std::vector<float> expected((size_t)D, 0.0f);
    std::vector<int> expert_ids((size_t)N);
    std::vector<float> route_weights((size_t)N, 1.0f / N);
    for (int e = 0; e < N; ++e) {
        expert_ids[(size_t)e] = e;
        for (int i = 0; i < D; ++i)
            expected[(size_t)i] += route_weights[(size_t)e] *
                                  reference_down[(size_t)e][(size_t)i];
    }

    auto run_kernel = [&] {
        return matmul_qwen_q2_0_q2_0_fused(
            gate_out.data(), up_out.data(), down_out.data(), input.data(),
            gate_ptrs.data(), up_ptrs.data(), down_ptrs.data(), N, D, FF, D,
            std::max(1, omp_get_max_threads()));
    };
    // Restore reference arrays after the kernel overwrites its outputs.
    bool ok = run_kernel() && dot_ok;
    for (int e = 0; e < N; ++e) {
        ok = compare_iq_outputs("Q2_0 gate", gate_y[e].data(),
                               reference_gate[e].data(), FF, 2e-5f, 2e-5f) && ok;
        ok = compare_iq_outputs("Q2_0 up", up_y[e].data(),
                               reference_up[e].data(), FF, 2e-5f, 2e-5f) && ok;
        ok = compare_iq_outputs("Q2_0 down", down_y[e].data(),
                               reference_down[e].data(), D, 2e-5f, 2e-5f) && ok;
    }

    for (int i = 0; i < 3; ++i)
        run_kernel();

    // Exercise the actual direct Qwen dispatch with this measured all-Q2_0
    // signature. This is the opt-in CPU host route used by the smoke config.
    setenv("MOESTREAM_QWEN_CPU_DIRECT_POOL", "1", 1);
    setenv("MOESTREAM_QWEN_CPU_DIRECT_FUSED", "1", 1);
    GgmlQwenCpuExpertRef ref{
        gate_ptrs[0], up_ptrs[0], down_ptrs[0],
        (size_t)FF * input_row_bytes, (size_t)FF * input_row_bytes,
        (size_t)D * down_row_bytes, (uint32_t)GGUFType::Q2_0,
        (uint32_t)GGUFType::Q2_0, (uint32_t)GGUFType::Q2_0};
    std::vector<float> routed((size_t)D, 0.0f);
    auto run_routed = [&] {
        return GgmlQwenCpuMoeRunner::instance().run_strided(
            input.data(), routed.data(), ref, gate_ptrs[0], up_ptrs[0],
            down_ptrs[0], expert_ids.data(), N, route_weights.data(), D, FF,
            N, (size_t)FF * input_row_bytes);
    };
    ok = run_routed() && ok;
    ok = compare_iq_outputs("Q2_0 direct Qwen route", routed.data(),
                            expected.data(), D, 2e-5f, 2e-5f) && ok;

    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < ITERATIONS; ++i)
        ok = run_routed() && ok;
    const double ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - start)
                          .count() /
                      ITERATIONS;
    const char *fused_backend = q2_0_vbmi_enabled()
                                    ? "AVX-512 VNNI+VBMI"
                                    : q2_0_vnni_enabled() ? "AVX-512 VNNI"
                                                          : "GGML fallback";
    std::cout << "  -> Q2_0 fused " << fused_backend
              << " direct route: " << std::fixed << std::setprecision(3) << ms
              << " ms/call (D=" << D << ", FF=" << FF << ", experts=" << N
              << ") " << (ok ? "PASSED" : "FAILED") << "\n";
    GgmlQwenCpuMoeRunner::instance().shutdown();
    return ok;
}

bool test_cpu_iq_matmul_paths() {
    std::cout << "[Test 7b] CPU IQ1_M/IQ2_XXS/IQ3_XXS matmul paths...\n";
    constexpr int I = 1024;
    constexpr int O = 9;
    constexpr int S = 2;
    std::mt19937 rng(20260911);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> x((size_t)S * I);
    for (float &v : x) v = dist(rng);

    bool ok = true;
    for (GGUFType type : {GGUFType::IQ1_M, GGUFType::IQ2_XXS,
                          GGUFType::IQ3_XXS, GGUFType::Q2_0}) {
        const int rb = row_bytes(type, I);
        std::vector<float> wf((size_t)O * I);
        for (float &v : wf) v = dist(rng);
        std::vector<uint8_t> w((size_t)O * rb);
        bool quantized = true;
        for (int o = 0; o < O; ++o) {
            if (!ggml_quantize_row_ref(
                    (uint32_t)type, wf.data() + (size_t)o * I,
                    w.data() + (size_t)o * rb, I)) {
                quantized = false;
                break;
            }
        }
        if (!quantized) {
            std::cout << "  -> " << gpu_type_name(type)
                      << ": SKIPPED (reference quantizer unavailable)\n";
            continue;
        }
        std::vector<float> reference((size_t)S * O, 0.0f);
        std::vector<float> actual((size_t)S * O, 0.0f);
        ok = exact_iq_matmul_reference(type, w.data(), x.data(),
                                       reference.data(), S, I, O) && ok;
        ok = cpu_quant_matmul(type, w.data(), x.data(), actual.data(), S, I,
                              O) && ok;
        // IQ2/3's default fused path is intended to reproduce GGML's
        // dequantized FP32 dot. IQ1_M's default path is the same exact
        // fallback; its optional Q8_K path is tested separately below.
        ok = compare_iq_outputs(gpu_type_name(type), actual.data(),
                                reference.data(), actual.size(), 2e-3f,
                                2e-3f) && ok;

        if (type == GGUFType::IQ1_M) {
            std::vector<float> q8k((size_t)S * O, 0.0f);
            const bool q8k_ok = matmul_iq1_m_ggml_q8k(
                q8k.data(), x.data(), w.data(), S, I, O);
            bool q8k_matches = q8k_ok;
            if (q8k_ok) {
                // Q8_K activation quantization is intentionally lossy, so
                // use a broad numerical bound here. A row/stride bug is much
                // larger than this and is exactly what this test is meant to
                // catch; the grouped consistency check below is strict.
                q8k_matches = compare_iq_outputs(
                    "IQ1_M ggml Q8_K vs exact", q8k.data(), reference.data(),
                    q8k.size(), 0.15f, 0.08f);
            }
            ok = q8k_matches && ok;

            // The experimental AVX-512-width probe must agree with GGML's
            // production AVX2-width Q8_K result before it is allowed into the
            // Qwen grouped CPU path.  A mismatch here means an IQ1_M packing,
            // sign, or scale-porting error rather than normal Q8_K loss.
            std::vector<float> avx512_q8k((size_t)S * O, 0.0f);
            float *avx512_out = avx512_q8k.data();
            const uint8_t *avx512_w = w.data();
            const bool avx512_ok = matmul_iq1_m_avx512_q8k_grouped(
                &avx512_out, x.data(), &avx512_w, 1, S, I, O);
            if (avx512_ok) {
                ok = compare_iq_outputs(
                         "IQ1_M AVX-512 Q8_K vs GGML Q8_K", avx512_q8k.data(),
                         q8k.data(), q8k.size(), 2e-4f, 2e-4f) &&
                     ok;
            } else {
                std::cout << "  -> IQ1_M AVX-512 Q8_K: SKIPPED (ISA/build unavailable)\n";
            }
        }
    }

    // Grouped production wrappers: compare n=3 against three n=1 calls.
    // This isolates the shared activation quantization and row indexing from
    // the expected IQ quantization error relative to an FP32 reference.
    constexpr int M = 3;
    std::vector<std::vector<float>> wf((size_t)M,
                                       std::vector<float>((size_t)O * I));
    std::vector<std::vector<uint8_t>> wb((size_t)M);
    std::vector<const uint8_t *> wp((size_t)M);
    bool all_iq1m_quantized = true;
    for (int m = 0; m < M; ++m) {
        for (float &v : wf[(size_t)m]) v = dist(rng);
        const int rb = row_bytes(GGUFType::IQ1_M, I);
        wb[(size_t)m].resize((size_t)O * rb);
        bool quantized = true;
        for (int o = 0; o < O; ++o)
            if (!ggml_quantize_row_ref(
                    (uint32_t)GGUFType::IQ1_M,
                    wf[(size_t)m].data() + (size_t)o * I,
                    wb[(size_t)m].data() + (size_t)o * rb, I)) {
                quantized = false;
                break;
            }
        if (!quantized) {
            all_iq1m_quantized = false;
            break;
        }
        wp[(size_t)m] = wb[(size_t)m].data();
    }
    if (!all_iq1m_quantized) {
        std::cout << "  -> IQ1_M grouped checks skipped (reference quantizer "
                     "unavailable)\n";
        std::cout << "  -> CPU IQ path tests " << (ok ? "PASSED" : "FAILED")
                  << "\n";
        return ok;
    }
    std::vector<std::vector<float>> grouped(
        (size_t)M, std::vector<float>((size_t)O, 0.0f));
    std::vector<std::vector<float>> single(
        (size_t)M, std::vector<float>((size_t)O, 0.0f));
    std::vector<float *> grouped_ptr((size_t)M), single_ptr((size_t)M);
    for (int m = 0; m < M; ++m) {
        grouped_ptr[(size_t)m] = grouped[(size_t)m].data();
        single_ptr[(size_t)m] = single[(size_t)m].data();
        ok = matmul_iq1_m_ggml_q8k(single_ptr[(size_t)m], x.data(),
                                   wp[(size_t)m], 1, I, O) &&
             ok;
    }
    ok = matmul_iq1_m_ggml_q8k_grouped(
             grouped_ptr.data(), x.data(), wp.data(), M, 1, I, O) &&
         ok;
    for (int m = 0; m < M; ++m)
        ok = compare_iq_outputs("IQ1_M grouped vs single",
                                grouped[(size_t)m].data(),
                                single[(size_t)m].data(), O, 1e-6f,
                                1e-6f) &&
             ok;

    // The Qwen decode fast path uses the same activation for IQ1_M gate/up.
    // The fused pair must be numerically identical to two independent
    // grouped calls while quantizing Q8_K only once internally.
    std::vector<std::vector<float>> pair_gate(
        (size_t)M, std::vector<float>((size_t)O, 0.0f));
    std::vector<std::vector<float>> pair_up(
        (size_t)M, std::vector<float>((size_t)O, 0.0f));
    std::vector<float *> pair_gate_ptr((size_t)M), pair_up_ptr((size_t)M);
    for (int m = 0; m < M; ++m) {
        pair_gate_ptr[(size_t)m] = pair_gate[(size_t)m].data();
        pair_up_ptr[(size_t)m] = pair_up[(size_t)m].data();
    }
    ok = matmul_iq1_m_ggml_q8k_grouped_pair(
             pair_gate_ptr.data(), pair_up_ptr.data(), x.data(), wp.data(),
             wp.data(), M, 1, I, O) &&
         ok;
    for (int m = 0; m < M; ++m) {
        ok = compare_iq_outputs("IQ1_M fused pair gate vs grouped",
                                pair_gate[(size_t)m].data(),
                                grouped[(size_t)m].data(), O, 1e-6f,
                                1e-6f) &&
             ok;
        ok = compare_iq_outputs("IQ1_M fused pair up vs grouped",
                                pair_up[(size_t)m].data(),
                                grouped[(size_t)m].data(), O, 1e-6f,
                                1e-6f) &&
             ok;
    }

    // The direct Qwen strided path uses IQ4_NL down projections with a
    // separate SwiGLU activation for each selected expert.  Check the
    // grouped-input wrapper against one-matrix calls as well; this catches
    // the per-expert activation-pointer indexing that a shared-input test
    // would miss.
    {
        constexpr int DOWN_I = 640;
        constexpr int DOWN_O = 17;
        std::vector<std::vector<float>> down_x(
            (size_t)M, std::vector<float>((size_t)DOWN_I));
        std::vector<std::vector<float>> down_wf(
            (size_t)M, std::vector<float>((size_t)DOWN_O * DOWN_I));
        std::vector<std::vector<uint8_t>> down_w((size_t)M);
        std::vector<const float *> down_x_ptr((size_t)M);
        std::vector<const uint8_t *> down_w_ptr((size_t)M);
        std::vector<std::vector<float>> down_grouped(
            (size_t)M, std::vector<float>((size_t)DOWN_O, 0.0f));
        std::vector<std::vector<float>> down_single(
            (size_t)M, std::vector<float>((size_t)DOWN_O, 0.0f));
        std::vector<float *> down_grouped_ptr((size_t)M),
            down_single_ptr((size_t)M);
        const int down_rb = row_bytes(GGUFType::IQ4_NL, DOWN_I);
        bool down_quantized = true;
        for (int m = 0; m < M; ++m) {
            for (float &v : down_x[(size_t)m]) v = dist(rng);
            for (float &v : down_wf[(size_t)m]) v = dist(rng);
            down_w[(size_t)m].resize((size_t)DOWN_O * down_rb);
            for (int o = 0; o < DOWN_O; ++o) {
                if (!ggml_quantize_row_ref(
                        (uint32_t)GGUFType::IQ4_NL,
                        down_wf[(size_t)m].data() + (size_t)o * DOWN_I,
                        down_w[(size_t)m].data() + (size_t)o * down_rb,
                        DOWN_I)) {
                    down_quantized = false;
                    break;
                }
            }
            down_x_ptr[(size_t)m] = down_x[(size_t)m].data();
            down_w_ptr[(size_t)m] = down_w[(size_t)m].data();
            down_grouped_ptr[(size_t)m] = down_grouped[(size_t)m].data();
            down_single_ptr[(size_t)m] = down_single[(size_t)m].data();
        }
        if (!down_quantized) {
            std::cout << "  -> IQ4_NL grouped-input checks skipped (reference "
                         "quantizer unavailable)\n";
        } else {
            for (int m = 0; m < M; ++m) {
                const float *one_x = down_x_ptr[(size_t)m];
                const uint8_t *one_w = down_w_ptr[(size_t)m];
                float *one_y = down_single_ptr[(size_t)m];
                ok = matmul_iq4_nl_ggml_q8_0_grouped_inputs(
                         &one_y, &one_x, &one_w, 1, 1, DOWN_I, DOWN_O) &&
                     ok;
            }
            ok = matmul_iq4_nl_ggml_q8_0_grouped_inputs(
                     down_grouped_ptr.data(), down_x_ptr.data(),
                     down_w_ptr.data(), M, 1, DOWN_I, DOWN_O) &&
                 ok;
            for (int m = 0; m < M; ++m)
                ok = compare_iq_outputs(
                         "IQ4_NL grouped inputs vs single",
                         down_grouped[(size_t)m].data(),
                         down_single[(size_t)m].data(), DOWN_O, 1e-6f,
                         1e-6f) &&
                     ok;
        }
    }

    // Optional Qwen decode probe: gate/up IQ1_M, SwiGLU, and down IQ4_NL
    // execute under one OpenMP team.  Compare every intermediate/output with
    // the already-tested grouped composition so this optimization cannot
    // silently change the routed expert result.
    {
        constexpr int FUSED_I = 1024;
        constexpr int FUSED_GATE_O = 64; // also satisfies IQ4_NL's QK=32
        constexpr int FUSED_DOWN_O = 17;
        std::vector<float> fused_x((size_t)FUSED_I);
        for (float &v : fused_x) v = dist(rng);
        const int gate_rb = row_bytes(GGUFType::IQ1_M, FUSED_I);
        const int down_rb = row_bytes(GGUFType::IQ4_NL, FUSED_GATE_O);
        std::vector<std::vector<uint8_t>> fused_gate_w((size_t)M),
            fused_up_w((size_t)M), fused_down_w((size_t)M);
        std::vector<const uint8_t *> fused_gate_wp((size_t)M),
            fused_up_wp((size_t)M), fused_down_wp((size_t)M);
        bool fused_quantized = true;
        for (int m = 0; m < M && fused_quantized; ++m) {
            std::vector<float> gate_f((size_t)FUSED_GATE_O * FUSED_I),
                up_f((size_t)FUSED_GATE_O * FUSED_I),
                down_f((size_t)FUSED_DOWN_O * FUSED_GATE_O);
            for (float &v : gate_f) v = dist(rng);
            for (float &v : up_f) v = dist(rng);
            for (float &v : down_f) v = dist(rng);
            fused_gate_w[(size_t)m].resize((size_t)FUSED_GATE_O * gate_rb);
            fused_up_w[(size_t)m].resize((size_t)FUSED_GATE_O * gate_rb);
            fused_down_w[(size_t)m].resize((size_t)FUSED_DOWN_O * down_rb);
            for (int o = 0; o < FUSED_GATE_O && fused_quantized; ++o) {
                fused_quantized =
                    ggml_quantize_row_ref(
                        (uint32_t)GGUFType::IQ1_M,
                        gate_f.data() + (size_t)o * FUSED_I,
                        fused_gate_w[(size_t)m].data() + (size_t)o * gate_rb,
                        FUSED_I) &&
                    ggml_quantize_row_ref(
                        (uint32_t)GGUFType::IQ1_M,
                        up_f.data() + (size_t)o * FUSED_I,
                        fused_up_w[(size_t)m].data() + (size_t)o * gate_rb,
                        FUSED_I);
            }
            for (int o = 0; o < FUSED_DOWN_O && fused_quantized; ++o)
                fused_quantized = ggml_quantize_row_ref(
                    (uint32_t)GGUFType::IQ4_NL,
                    down_f.data() + (size_t)o * FUSED_GATE_O,
                    fused_down_w[(size_t)m].data() + (size_t)o * down_rb,
                    FUSED_GATE_O);
            fused_gate_wp[(size_t)m] = fused_gate_w[(size_t)m].data();
            fused_up_wp[(size_t)m] = fused_up_w[(size_t)m].data();
            fused_down_wp[(size_t)m] = fused_down_w[(size_t)m].data();
        }
        if (!fused_quantized) {
            std::cout << "  -> Qwen fused IQ1/IQ4 check skipped (reference "
                         "quantizer unavailable)\n";
        } else {
            std::vector<std::vector<float>> ref_gate(
                (size_t)M, std::vector<float>((size_t)FUSED_GATE_O));
            std::vector<std::vector<float>> ref_up(
                (size_t)M, std::vector<float>((size_t)FUSED_GATE_O));
            std::vector<std::vector<float>> ref_down(
                (size_t)M, std::vector<float>((size_t)FUSED_DOWN_O));
            std::vector<std::vector<float>> fused_gate = ref_gate,
                fused_up = ref_up, fused_down = ref_down;
            std::vector<float *> ref_gate_p((size_t)M), ref_up_p((size_t)M),
                ref_down_p((size_t)M), fused_gate_p((size_t)M),
                fused_up_p((size_t)M), fused_down_p((size_t)M);
            std::vector<const float *> ref_down_x((size_t)M);
            for (int m = 0; m < M; ++m) {
                ref_gate_p[(size_t)m] = ref_gate[(size_t)m].data();
                ref_up_p[(size_t)m] = ref_up[(size_t)m].data();
                ref_down_p[(size_t)m] = ref_down[(size_t)m].data();
                fused_gate_p[(size_t)m] = fused_gate[(size_t)m].data();
                fused_up_p[(size_t)m] = fused_up[(size_t)m].data();
                fused_down_p[(size_t)m] = fused_down[(size_t)m].data();
            }
            ok = matmul_iq1_m_ggml_q8k_grouped_pair(
                     ref_gate_p.data(), ref_up_p.data(), fused_x.data(),
                     fused_gate_wp.data(), fused_up_wp.data(), M, 1,
                     FUSED_I, FUSED_GATE_O) &&
                 ok;
            for (int m = 0; m < M; ++m) {
                silu_inplace(ref_gate[(size_t)m].data(), FUSED_GATE_O);
                elem_mul(ref_gate[(size_t)m].data(), ref_gate[(size_t)m].data(),
                         ref_up[(size_t)m].data(), FUSED_GATE_O);
                ref_down_x[(size_t)m] = ref_gate[(size_t)m].data();
            }
            ok = matmul_iq4_nl_ggml_q8_0_grouped_inputs(
                     ref_down_p.data(), ref_down_x.data(),
                     fused_down_wp.data(), M, 1, FUSED_GATE_O, FUSED_DOWN_O) &&
                 ok;
            ok = matmul_qwen_iq1_m_iq4_nl_fused(
                     fused_gate_p.data(), fused_up_p.data(), fused_down_p.data(),
                     fused_x.data(), fused_gate_wp.data(), fused_up_wp.data(),
                     fused_down_wp.data(), M, FUSED_I, FUSED_GATE_O,
                     FUSED_DOWN_O) &&
                 ok;
            for (int m = 0; m < M; ++m) {
                ok = compare_iq_outputs(
                         "Qwen fused IQ1/IQ4 gate", fused_gate[(size_t)m].data(),
                         ref_gate[(size_t)m].data(), FUSED_GATE_O, 1e-6f,
                         1e-6f) &&
                     ok;
                ok = compare_iq_outputs(
                         "Qwen fused IQ1/IQ4 up", fused_up[(size_t)m].data(),
                         ref_up[(size_t)m].data(), FUSED_GATE_O, 1e-6f,
                         1e-6f) &&
                     ok;
                ok = compare_iq_outputs(
                         "Qwen fused IQ1/IQ4 down", fused_down[(size_t)m].data(),
                         ref_down[(size_t)m].data(), FUSED_DOWN_O, 1e-6f,
                         1e-6f) &&
                     ok;
            }
        }
    }

    for (GGUFType type : {GGUFType::IQ2_XXS, GGUFType::IQ3_XXS}) {
        const int rb = row_bytes(type, I);
        wb.assign((size_t)M, std::vector<uint8_t>{});
        wp.assign((size_t)M, nullptr);
        bool all_quantized = true;
        for (int m = 0; m < M; ++m) {
            wb[(size_t)m].resize((size_t)O * rb);
            bool quantized = true;
            for (int o = 0; o < O; ++o)
                if (!ggml_quantize_row_ref(
                        (uint32_t)type,
                        wf[(size_t)m].data() + (size_t)o * I,
                        wb[(size_t)m].data() + (size_t)o * rb, I)) {
                    quantized = false;
                    break;
                }
            if (!quantized) {
                all_quantized = false;
                break;
            }
            wp[(size_t)m] = wb[(size_t)m].data();
        }
        if (!all_quantized) {
            std::cout << "  -> " << gpu_type_name(type)
                      << " grouped checks skipped (reference quantizer "
                         "unavailable)\n";
            continue;
        }
        grouped.assign((size_t)M, std::vector<float>((size_t)O, 0.0f));
        single.assign((size_t)M, std::vector<float>((size_t)O, 0.0f));
        grouped_ptr.resize((size_t)M);
        single_ptr.resize((size_t)M);
        for (int m = 0; m < M; ++m) {
            grouped_ptr[(size_t)m] = grouped[(size_t)m].data();
            single_ptr[(size_t)m] = single[(size_t)m].data();
            float *one_y = single_ptr[(size_t)m];
            const uint8_t *one_w = wp[(size_t)m];
            ok = (type == GGUFType::IQ2_XXS
                      ? matmul_iq2_xxs_ggml_q8k_grouped(
                            &one_y, x.data(), &one_w, 1, 1, I, O)
                      : matmul_iq3_xxs_ggml_q8k_grouped(
                            &one_y, x.data(), &one_w, 1, 1, I, O)) &&
                 ok;
        }
        ok = (type == GGUFType::IQ2_XXS
                  ? matmul_iq2_xxs_ggml_q8k_grouped(
                        grouped_ptr.data(), x.data(), wp.data(), M, 1, I, O)
                  : matmul_iq3_xxs_ggml_q8k_grouped(
                        grouped_ptr.data(), x.data(), wp.data(), M, 1, I, O)) &&
             ok;
        for (int m = 0; m < M; ++m)
            ok = compare_iq_outputs(
                     std::string(gpu_type_name(type)) +
                         " grouped vs single",
                     grouped[(size_t)m].data(), single[(size_t)m].data(), O,
                     1e-6f, 1e-6f) &&
                 ok;

        // Consistency alone can make a systematically-zero kernel look
        // healthy.  Also compare the production Q8_K grouped result with
        // the independent GGML dequantize+F32 oracle.
        for (int m = 0; m < M; ++m) {
            std::vector<float> reference_one((size_t)O, 0.0f);
            const bool reference_ok = exact_iq_matmul_reference(
                type, wb[(size_t)m].data(), x.data(), reference_one.data(), 1,
                I, O);
            bool semantic_ok = reference_ok;
            if (reference_ok)
                semantic_ok = compare_iq_outputs(
                    std::string(gpu_type_name(type)) +
                        " grouped vs dequant oracle",
                    grouped[(size_t)m].data(), reference_one.data(), O, 0.20f,
                    0.25f);
            ok = semantic_ok && ok;
        }
    }
    std::cout << "  -> CPU IQ path tests " << (ok ? "PASSED" : "FAILED")
              << "\n";
    return ok;
}

bool test_qwen_cpu_moe_graph() {
    std::cout << "[Test 7c] Qwen persistent CPU GGML MoE graph...\n";
    if (!GgmlQwenCpuMoeRunner::compiled_in()) {
        std::cout << "  -> Skipped (GGML CPU backend not compiled)\n";
        return true;
    }
    constexpr int D = 8;
    constexpr int FF = 12;
    constexpr int N = 2;
    std::vector<std::vector<float>> gate(
        N, std::vector<float>((size_t)D * FF));
    std::vector<std::vector<float>> up(N, std::vector<float>((size_t)D * FF));
    std::vector<std::vector<float>> down(
        N, std::vector<float>((size_t)FF * D));
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> dist(-0.4f, 0.4f);
    for (int e = 0; e < N; ++e) {
        for (float &v : gate[(size_t)e]) v = dist(rng);
        for (float &v : up[(size_t)e]) v = dist(rng);
        for (float &v : down[(size_t)e]) v = dist(rng);
    }
    std::vector<GgmlQwenCpuExpertRef> refs((size_t)N);
    for (int e = 0; e < N; ++e) {
        refs[(size_t)e] = {
            gate[(size_t)e].data(), up[(size_t)e].data(), down[(size_t)e].data(),
            gate[(size_t)e].size() * sizeof(float),
            up[(size_t)e].size() * sizeof(float),
            down[(size_t)e].size() * sizeof(float),
            static_cast<uint32_t>(GGUFType::F32),
            static_cast<uint32_t>(GGUFType::F32),
            static_cast<uint32_t>(GGUFType::F32)};
    }

    auto oracle = [&](const std::vector<float> &x,
                      const std::array<float, N> &route,
                      std::vector<float> &out) {
        std::fill(out.begin(), out.end(), 0.0f);
        for (int e = 0; e < N; ++e) {
            std::vector<float> act((size_t)FF);
            for (int j = 0; j < FF; ++j) {
                float g = 0.0f, u = 0.0f;
                for (int i = 0; i < D; ++i) {
                    g += gate[(size_t)e][(size_t)j * D + i] * x[(size_t)i];
                    u += up[(size_t)e][(size_t)j * D + i] * x[(size_t)i];
                }
                act[(size_t)j] = g / (1.0f + std::exp(-g)) * u;
            }
            for (int d = 0; d < D; ++d) {
                float v = 0.0f;
                for (int j = 0; j < FF; ++j)
                    v += down[(size_t)e][(size_t)d * FF + j] *
                         act[(size_t)j];
                out[(size_t)d] += route[(size_t)e] * v;
            }
        }
    };

    bool ok = true;
    for (int pass = 0; pass < 2; ++pass) {
        std::vector<float> x((size_t)D);
        for (float &v : x) v = dist(rng);
        const std::array<float, N> route =
            pass == 0 ? std::array<float, N>{0.35f, 0.65f}
                      : std::array<float, N>{0.8f, 0.2f};
        std::vector<float> expected((size_t)D), actual((size_t)D);
        oracle(x, route, expected);
        ok = GgmlQwenCpuMoeRunner::instance().run(
                 x.data(), actual.data(), refs.data(), N, route.data(), D, FF,
                 N) &&
             ok;
        ok = compare_iq_outputs("Qwen CPU graph F32 pass " +
                                    std::to_string(pass),
                                actual.data(), expected.data(), D, 2e-5f,
                                2e-5f) &&
             ok;
    }

    // The production CPU-affinity path does not compact selected experts.
    // It points MUL_MAT_ID at one fixed-stride [gate|up|down] DDR arena and
    // supplies the real global expert IDs.  Exercise that layout explicitly
    // so a future graph change cannot silently fall back to a file-backed or
    // compact-copy source.
    const size_t gate_bytes = gate[0].size() * sizeof(float);
    const size_t up_bytes = up[0].size() * sizeof(float);
    const size_t down_bytes = down[0].size() * sizeof(float);
    const size_t raw_stride = gate_bytes + up_bytes + down_bytes;
    const size_t stride = raw_stride + 64; // include inter-expert padding
    void *arena_raw = nullptr;
    const bool arena_allocated =
        posix_memalign(&arena_raw, 64, stride * static_cast<size_t>(N)) == 0 &&
        arena_raw != nullptr;
    if (!arena_allocated) {
        std::cout << "  -> Qwen CPU graph strided DDR arena: allocation FAILED\n";
        GgmlQwenCpuMoeRunner::instance().shutdown();
        return false;
    }
    auto *arena = static_cast<uint8_t *>(arena_raw);
    std::memset(arena, 0, stride * static_cast<size_t>(N));
    for (int e = 0; e < N; ++e) {
        uint8_t *slot = arena + static_cast<size_t>(e) * stride;
        std::memcpy(slot, gate[(size_t)e].data(), gate_bytes);
        std::memcpy(slot + gate_bytes, up[(size_t)e].data(), up_bytes);
        std::memcpy(slot + gate_bytes + up_bytes, down[(size_t)e].data(),
                    down_bytes);
    }
    const int ids[N] = {1, 0};
    const float ordered_weights[N] = {0.65f, 0.35f};
    const std::array<float, N> by_id = {0.35f, 0.65f};
    std::vector<float> x((size_t)D), expected((size_t)D), actual((size_t)D);
    for (float &v : x) v = dist(rng);
    oracle(x, by_id, expected);
    ok = GgmlQwenCpuMoeRunner::instance().run_strided(
             x.data(), actual.data(), refs[0], arena, arena + gate_bytes,
             arena + gate_bytes + up_bytes, ids, N, ordered_weights,
             D, FF, N, stride) &&
         ok;
    ok = compare_iq_outputs("Qwen CPU graph strided DDR arena", actual.data(),
                            expected.data(), D, 2e-5f, 2e-5f) &&
         ok;

    // Exercise the persistent-worker entry point used by the Qwen CPU/GPU
    // overlap path. The result must match the synchronous strided graph; the
    // test deliberately waits before releasing the arena backing storage.
    std::vector<float> async_actual((size_t)D, 0.0f);
    const bool async_started =
        GgmlQwenCpuMoeRunner::instance().start_strided_async(
            x.data(), async_actual.data(), refs[0], arena, arena + gate_bytes,
            arena + gate_bytes + up_bytes, ids, N, ordered_weights, D, FF, N,
            stride);
    const bool async_ok = async_started &&
                          GgmlQwenCpuMoeRunner::instance().wait_strided_async();
    ok = async_ok &&
         compare_iq_outputs("Qwen CPU graph async worker", async_actual.data(),
                            expected.data(), D, 2e-5f, 2e-5f) &&
         ok;
    std::free(arena_raw);
    GgmlQwenCpuMoeRunner::instance().shutdown();
    std::cout << "  -> Qwen CPU graph test " << (ok ? "PASSED" : "FAILED")
              << "\n";
    return ok;
}

// Exercise the same persistent graph with a real IQ1_M expert.  This is
// intentionally a one-expert case: it isolates the quantized MUL_MAT_ID
// contract from routing/compaction and makes a zero-output CPU dot kernel
// immediately visible against an independent dequantize+F32 oracle.
bool test_qwen_cpu_moe_iq1m_graph() {
    std::cout << "[Test 7d] Qwen IQ1_M CPU GGML MoE graph...\n";
    if (!GgmlQwenCpuMoeRunner::compiled_in()) {
        std::cout << "  -> Skipped (GGML CPU backend not compiled)\n";
        return true;
    }
    constexpr int D = 256;
    constexpr int FF = 256;
    const GGUFType type = GGUFType::IQ1_M;
    const int gate_row_bytes = row_bytes(type, D);
    const int down_row_bytes = row_bytes(type, FF);
    std::mt19937 rng(20260911 + 7);
    std::uniform_real_distribution<float> dist(-0.4f, 0.4f);

    std::vector<float> gate_f((size_t)FF * D);
    std::vector<float> up_f((size_t)FF * D);
    std::vector<float> down_f((size_t)D * FF);
    for (float &v : gate_f) v = dist(rng);
    for (float &v : up_f) v = dist(rng);
    for (float &v : down_f) v = dist(rng);

    std::vector<uint8_t> gate((size_t)FF * gate_row_bytes);
    std::vector<uint8_t> up((size_t)FF * gate_row_bytes);
    std::vector<uint8_t> down((size_t)D * down_row_bytes);
    bool ok = true;
    for (int row = 0; row < FF; ++row) {
        ok = ggml_quantize_row_ref(
                 (uint32_t)type, gate_f.data() + (size_t)row * D,
                 gate.data() + (size_t)row * gate_row_bytes, D) &&
             ok;
        ok = ggml_quantize_row_ref(
                 (uint32_t)type, up_f.data() + (size_t)row * D,
                 up.data() + (size_t)row * gate_row_bytes, D) &&
             ok;
    }
    for (int row = 0; row < D; ++row) {
        ok = ggml_quantize_row_ref(
                 (uint32_t)type, down_f.data() + (size_t)row * FF,
                 down.data() + (size_t)row * down_row_bytes, FF) &&
             ok;
    }
    if (!ok) {
        std::cout << "  -> FAILED (IQ1_M reference quantizer unavailable)\n";
        GgmlQwenCpuMoeRunner::instance().shutdown();
        return false;
    }

    std::vector<float> input((size_t)D);
    for (float &v : input) v = dist(rng);
    std::vector<float> gate_out((size_t)FF), up_out((size_t)FF), act((size_t)FF);
    std::vector<float> expected((size_t)D), actual((size_t)D);
    ok = exact_iq_matmul_reference(type, gate.data(), input.data(),
                                   gate_out.data(), 1, D, FF) &&
         ok;
    ok = exact_iq_matmul_reference(type, up.data(), input.data(),
                                   up_out.data(), 1, D, FF) &&
         ok;
    for (int j = 0; j < FF; ++j)
        act[(size_t)j] = gate_out[(size_t)j] /
                         (1.0f + std::exp(-gate_out[(size_t)j])) *
                         up_out[(size_t)j];
    ok = exact_iq_matmul_reference(type, down.data(), act.data(),
                                   expected.data(), 1, FF, D) &&
         ok;

    GgmlQwenCpuExpertRef ref{
        gate.data(), up.data(), down.data(), gate.size(), up.size(), down.size(),
        static_cast<uint32_t>(type), static_cast<uint32_t>(type),
        static_cast<uint32_t>(type)};
    const float route = 1.0f;
    ok = GgmlQwenCpuMoeRunner::instance().run(
             input.data(), actual.data(), &ref, 1, &route, D, FF, 1) &&
         ok;
    ok = compare_iq_outputs("Qwen IQ1_M graph vs dequant oracle", actual.data(),
                            expected.data(), expected.size(), 0.20f, 0.25f) &&
         ok;
    double expected_norm2 = 0.0;
    double actual_norm2 = 0.0;
    double dot = 0.0;
    for (size_t i = 0; i < expected.size(); ++i) {
        expected_norm2 += (double)expected[i] * expected[i];
        actual_norm2 += (double)actual[i] * actual[i];
        dot += (double)expected[i] * actual[i];
    }
    const double expected_norm = std::sqrt(expected_norm2);
    const double actual_norm = std::sqrt(actual_norm2);
    const double cosine = expected_norm > 0.0 && actual_norm > 0.0
                              ? dot / (expected_norm * actual_norm)
                              : 0.0;
    const bool signal_ok = expected_norm > 1e-4 &&
                           actual_norm >= 0.25 * expected_norm && cosine > 0.25;
    std::cout << "  -> IQ1_M graph signal: " << (signal_ok ? "PASSED" : "FAILED")
              << " expected_norm=" << expected_norm
              << " actual_norm=" << actual_norm << " cosine=" << cosine
              << "\n";
    ok = signal_ok && ok;
    GgmlQwenCpuMoeRunner::instance().shutdown();
    std::cout << "  -> Qwen IQ1_M graph test " << (ok ? "PASSED" : "FAILED")
              << "\n";
    return ok;
}

// Exercise the exact mixed quantization used by the Qwen CPU-affinity path
// through the fixed-stride, global-expert-ID graph.  Test 7d only covers one
// IQ1_M expert and the throughput probe below uses zero-filled blocks, so
// neither catches a projection-type mismatch or an indexing error when
// IQ1_M/IQ1_M gate/up is combined with IQ4_NL down and top-k > 1.
bool test_qwen_cpu_moe_mixed_strided_graph() {
    std::cout << "[Test 7e] Qwen mixed-type strided CPU GGML MoE graph...\n";
    if (!GgmlQwenCpuMoeRunner::compiled_in()) {
        std::cout << "  -> Skipped (GGML CPU backend not compiled)\n";
        return true;
    }

    constexpr int D = 256;
    constexpr int FF = 256;
    constexpr int N = 4;
    constexpr int TOPK = 3;
    const GGUFType gate_type = GGUFType::IQ1_M;
    const GGUFType up_type = GGUFType::IQ1_M;
    const GGUFType down_type = GGUFType::IQ4_NL;
    const size_t gate_row = row_bytes(gate_type, D);
    const size_t up_row = row_bytes(up_type, D);
    const size_t down_row = row_bytes(down_type, FF);
    const size_t gate_used = (size_t)FF * gate_row;
    const size_t up_used = (size_t)FF * up_row;
    const size_t down_used = (size_t)D * down_row;
    const size_t stride = gate_used + up_used + down_used + 64;
    const size_t arena_bytes = stride * (size_t)N;

    void *arena_raw = nullptr;
    if (posix_memalign(&arena_raw, 64, arena_bytes) != 0 || !arena_raw) {
        std::cout << "  -> FAILED (arena allocation)\n";
        return false;
    }
    auto *arena = static_cast<uint8_t *>(arena_raw);
    auto *gate_plane = arena;
    auto *up_plane = arena + gate_used;
    auto *down_plane = arena + gate_used + up_used;
    std::memset(arena, 0, arena_bytes);

    std::mt19937 rng(20260912 + 7);
    std::uniform_real_distribution<float> dist(-0.4f, 0.4f);
    bool ok = true;
    auto quantize_plane = [&](uint8_t *plane, GGUFType type, int rows,
                              int width, size_t row_bytes_value) {
        std::vector<float> row((size_t)width);
        for (int expert = 0; expert < N && ok; ++expert) {
            uint8_t *expert_plane = plane + (size_t)expert * stride;
            for (int r = 0; r < rows; ++r) {
                for (float &v : row)
                    v = dist(rng);
                ok = ggml_quantize_row_ref(
                         (uint32_t)type, row.data(),
                         expert_plane + (size_t)r * row_bytes_value, width) &&
                     ok;
            }
        }
    };
    quantize_plane(gate_plane, gate_type, FF, D, gate_row);
    quantize_plane(up_plane, up_type, FF, D, up_row);
    quantize_plane(down_plane, down_type, D, FF, down_row);
    if (!ok) {
        std::cout << "  -> FAILED (reference quantizer unavailable)\n";
        std::free(arena_raw);
        GgmlQwenCpuMoeRunner::instance().shutdown();
        return false;
    }

    const GgmlQwenCpuExpertRef layer{
        gate_plane, up_plane, down_plane, gate_used, up_used, down_used,
        (uint32_t)gate_type, (uint32_t)up_type, (uint32_t)down_type};
    std::vector<float> input((size_t)D);
    for (float &v : input)
        v = dist(rng);

    auto oracle = [&](const std::array<int, TOPK> &ids,
                      const std::array<float, TOPK> &weights,
                      std::vector<float> &out) {
        std::fill(out.begin(), out.end(), 0.0f);
        std::vector<float> gate_out((size_t)FF);
        std::vector<float> up_out((size_t)FF);
        std::vector<float> act((size_t)FF);
        std::vector<float> down_out((size_t)D);
        bool oracle_ok = true;
        for (int k = 0; k < TOPK; ++k) {
            const size_t expert_offset = (size_t)ids[(size_t)k] * stride;
            oracle_ok = exact_iq_matmul_reference(
                            gate_type, gate_plane + expert_offset,
                            input.data(), gate_out.data(), 1, D, FF) &&
                        oracle_ok;
            oracle_ok = exact_iq_matmul_reference(
                            up_type, up_plane + expert_offset, input.data(),
                            up_out.data(), 1, D, FF) &&
                        oracle_ok;
            for (int j = 0; j < FF; ++j)
                act[(size_t)j] = gate_out[(size_t)j] /
                                 (1.0f + std::exp(-gate_out[(size_t)j])) *
                                 up_out[(size_t)j];
            oracle_ok = exact_iq_matmul_reference(
                            down_type, down_plane + expert_offset,
                            act.data(), down_out.data(), 1, FF, D) &&
                        oracle_ok;
            for (int d = 0; d < D; ++d)
                out[(size_t)d] += weights[(size_t)k] * down_out[(size_t)d];
        }
        return oracle_ok;
    };

    auto &runner = GgmlQwenCpuMoeRunner::instance();
    const std::array<std::array<int, TOPK>, 2> ids = {{{3, 1, 0},
                                                        {2, 0, 3}}};
    const std::array<std::array<float, TOPK>, 2> weights = {{
        std::array<float, TOPK>{0.2f, 0.5f, 0.3f},
        std::array<float, TOPK>{0.6f, 0.1f, 0.3f}}};
    std::vector<float> expected((size_t)D), actual((size_t)D);
    for (int pass = 0; pass < 2; ++pass) {
        ok = oracle(ids[(size_t)pass], weights[(size_t)pass], expected) && ok;
        ok = runner.run_strided(
                 input.data(), actual.data(), layer, gate_plane, up_plane,
                 down_plane, ids[(size_t)pass].data(), N,
                 weights[(size_t)pass].data(), D, FF, TOPK, stride) &&
             ok;
        ok = compare_iq_outputs(
                 "Qwen mixed IQ1_M/IQ4_NL strided pass " +
                     std::to_string(pass),
                 actual.data(), expected.data(), expected.size(), 0.30f,
                 0.35f) &&
             ok;
    }

    runner.shutdown();
    std::free(arena_raw);
    std::cout << "  -> Qwen mixed-type strided graph "
              << (ok ? "PASSED" : "FAILED") << "\n";
    return ok;
}

// The checkpoint has a second routed projection family with IQ2_XXS
// gate/up. Keep an independent test for it instead of assuming the IQ1_M
// graph result generalizes: both types share the same graph shape, but their
// GGML CPU dot kernels and row sizes are different.
bool test_qwen_cpu_moe_iq2_strided_graph() {
    std::cout << "[Test 7f] Qwen IQ2_XXS strided CPU GGML MoE graph...\n";
    if (!GgmlQwenCpuMoeRunner::compiled_in()) {
        std::cout << "  -> Skipped (GGML CPU backend not compiled)\n";
        return true;
    }

    constexpr int D = 256;
    constexpr int FF = 256;
    constexpr int N = 3;
    constexpr int TOPK = 2;
    const GGUFType gate_type = GGUFType::IQ2_XXS;
    const GGUFType down_type = GGUFType::IQ4_NL;
    const size_t gate_row = row_bytes(gate_type, D);
    const size_t down_row = row_bytes(down_type, FF);
    const size_t gate_used = (size_t)FF * gate_row;
    const size_t down_used = (size_t)D * down_row;
    const size_t stride = gate_used + gate_used + down_used + 64;
    const size_t arena_bytes = stride * (size_t)N;
    void *arena_raw = nullptr;
    if (posix_memalign(&arena_raw, 64, arena_bytes) != 0 || !arena_raw) {
        std::cout << "  -> FAILED (arena allocation)\n";
        return false;
    }
    auto *arena = static_cast<uint8_t *>(arena_raw);
    auto *gate_plane = arena;
    auto *up_plane = arena + gate_used;
    auto *down_plane = arena + gate_used + gate_used;
    std::memset(arena, 0, arena_bytes);

    std::mt19937 rng(20260912 + 17);
    std::uniform_real_distribution<float> dist(-0.4f, 0.4f);
    bool ok = true;
    auto quantize_plane = [&](uint8_t *plane, GGUFType type, int rows,
                              int width, size_t row_bytes_value) {
        std::vector<float> row((size_t)width);
        for (int expert = 0; expert < N && ok; ++expert) {
            uint8_t *expert_plane = plane + (size_t)expert * stride;
            for (int r = 0; r < rows; ++r) {
                for (float &v : row)
                    v = dist(rng);
                ok = ggml_quantize_row_ref(
                         (uint32_t)type, row.data(),
                         expert_plane + (size_t)r * row_bytes_value, width) &&
                     ok;
            }
        }
    };
    quantize_plane(gate_plane, gate_type, FF, D, gate_row);
    quantize_plane(up_plane, gate_type, FF, D, gate_row);
    quantize_plane(down_plane, down_type, D, FF, down_row);
    if (!ok) {
        std::cout << "  -> FAILED (reference quantizer unavailable)\n";
        std::free(arena_raw);
        GgmlQwenCpuMoeRunner::instance().shutdown();
        return false;
    }

    const GgmlQwenCpuExpertRef layer{
        gate_plane, up_plane, down_plane, gate_used, gate_used, down_used,
        (uint32_t)gate_type, (uint32_t)gate_type, (uint32_t)down_type};
    std::vector<float> input((size_t)D);
    for (float &v : input)
        v = dist(rng);
    const std::array<int, TOPK> ids = {2, 0};
    const std::array<float, TOPK> weights = {0.35f, 0.65f};
    std::vector<float> expected((size_t)D, 0.0f);
    std::vector<float> actual((size_t)D, 0.0f);
    std::vector<float> gate_out((size_t)FF);
    std::vector<float> up_out((size_t)FF);
    std::vector<float> act((size_t)FF);
    std::vector<float> down_out((size_t)D);
    for (int k = 0; k < TOPK; ++k) {
        const size_t off = (size_t)ids[(size_t)k] * stride;
        ok = exact_iq_matmul_reference(gate_type, gate_plane + off,
                                       input.data(), gate_out.data(), 1, D,
                                       FF) &&
             ok;
        ok = exact_iq_matmul_reference(gate_type, up_plane + off,
                                       input.data(), up_out.data(), 1, D, FF) &&
             ok;
        for (int j = 0; j < FF; ++j)
            act[(size_t)j] = gate_out[(size_t)j] /
                             (1.0f + std::exp(-gate_out[(size_t)j])) *
                             up_out[(size_t)j];
        ok = exact_iq_matmul_reference(down_type, down_plane + off,
                                       act.data(), down_out.data(), 1, FF,
                                       D) &&
             ok;
        for (int d = 0; d < D; ++d)
            expected[(size_t)d] += weights[(size_t)k] * down_out[(size_t)d];
    }

    ok = GgmlQwenCpuMoeRunner::instance().run_strided(
             input.data(), actual.data(), layer, gate_plane, up_plane,
             down_plane, ids.data(), N, weights.data(), D, FF, TOPK, stride) &&
         ok;
    ok = compare_iq_outputs("Qwen IQ2_XXS/IQ4_NL strided", actual.data(),
                            expected.data(), expected.size(), 0.30f, 0.40f) &&
         ok;
    GgmlQwenCpuMoeRunner::instance().shutdown();
    std::free(arena_raw);
    std::cout << "  -> Qwen IQ2_XXS strided graph "
              << (ok ? "PASSED" : "FAILED") << "\n";
    return ok;
}

// Informational benchmark for the production Qwen CPU-affinity graph.  The
// shape and quantization match Qwen3.8-Flash-Next's routed experts:
// IQ1_M or IQ2_XXS gate/up, IQ4_NL down, D=2560, FF=640, top-k=10.  A compact 32-expert
// synthetic arena is enough to exercise MUL_MAT_ID and keeps this diagnostic
// runnable without allocating a full 512-expert model layer; the selected
// expert work is identical.  Run in isolation with
// MOESTREAM_TEST_ONLY_QWEN_CPU_BENCH=1 and vary OMP_NUM_THREADS to measure
// physical-core versus hyperthread scaling.  Set
// MOESTREAM_QWEN_CPU_BENCH_TYPE=iq2_xxs to measure the other production type.
bool bench_qwen_cpu_moe() {
    std::cout << "\n[BENCH] Qwen CPU strided GGML MoE path (informational)\n";
    if (!GgmlQwenCpuMoeRunner::compiled_in()) {
        std::cout << "  -> Skipped (GGML CPU backend not compiled)\n";
        return true;
    }

    constexpr int D = 2560;
    constexpr int FF = 640;
    constexpr int N = 32;
    constexpr int TOPK = 10;
    const char *type_env = std::getenv("MOESTREAM_QWEN_CPU_BENCH_TYPE");
    const bool use_iq2 = type_env &&
                         (std::strcmp(type_env, "iq2") == 0 ||
                          std::strcmp(type_env, "iq2_xxs") == 0 ||
                          std::strcmp(type_env, "IQ2_XXS") == 0);
    const GGUFType gate_type =
        use_iq2 ? GGUFType::IQ2_XXS : GGUFType::IQ1_M;
    const GGUFType up_type = gate_type;
    const GGUFType down_type = GGUFType::IQ4_NL;
    const size_t gate_bytes = (size_t)FF * row_bytes(gate_type, D);
    const size_t up_bytes = (size_t)FF * row_bytes(up_type, D);
    const size_t down_bytes = (size_t)D * row_bytes(down_type, FF);
    // Match the production graph-RAM source contract: each expert is one
    // fixed-stride [gate|up|down] slab.  This is also what enables the
    // llama.cpp-style merged gate_up MUL_MAT_ID path being measured here.
    const size_t raw_stride = gate_bytes + up_bytes + down_bytes;
    const size_t stride = raw_stride + 64;
    const size_t arena_bytes = stride * (size_t)N;

    void *arena_raw = nullptr;
    if (posix_memalign(&arena_raw, 64, arena_bytes) != 0 || !arena_raw) {
        std::free(arena_raw);
        std::cout << "  -> FAILED (arena allocation)\n";
        return false;
    }
    auto *arena = static_cast<uint8_t *>(arena_raw);
    std::memset(arena, 0, arena_bytes);
    auto *gate_raw = arena;
    auto *up_raw = arena + gate_bytes;
    auto *down_raw = arena + gate_bytes + up_bytes;

    // This is a throughput benchmark, not a quantizer-correctness test.  The
    // GGML kernels consume fixed-size block bytes, so deterministic raw block
    // bytes are sufficient and avoid spending most of the run in the slow
    // IQ1_M reference search quantizer.  Test 7d covers real quantizer output
    // and the independent correctness oracle separately.
    auto fill_plane = [&](void *raw, GGUFType type, int rows, int width,
                          size_t bytes_per_expert) {
        auto *base = static_cast<uint8_t *>(raw);
        const int rb = row_bytes(type, width);
        for (int expert = 0; expert < N; ++expert) {
            uint8_t *dst = base + (size_t)expert * stride;
            const size_t used = (size_t)rows * (size_t)rb;
            // Zero-filled blocks are structurally valid and keep this
            // throughput probe finite.  Random bytes are not valid IQ1_M
            // codebook/scales and can legitimately produce NaNs, which makes
            // the benchmark report misleading even though its timing is fine.
            std::memset(dst, 0, used);
            // Make accidental short-row/stride mistakes visible while
            // keeping the padding deterministic.
            if (bytes_per_expert < used)
                std::abort();
        }
    };
    fill_plane(gate_raw, gate_type, FF, D, gate_bytes);
    fill_plane(up_raw, up_type, FF, D, up_bytes);
    fill_plane(down_raw, down_type, D, FF, down_bytes);
    bool ok = true;

    GgmlQwenCpuExpertRef ref{
        gate_raw, up_raw, down_raw, gate_bytes, up_bytes, down_bytes,
        (uint32_t)gate_type, (uint32_t)up_type, (uint32_t)down_type};
    std::vector<float> input((size_t)D, 0.1f);
    std::vector<float> output((size_t)D, 0.0f);
    std::array<int, TOPK> ids{};
    std::array<float, TOPK> weights{};
    for (int k = 0; k < TOPK; ++k) {
        ids[(size_t)k] = (k * 3) % N;
        weights[(size_t)k] = 1.0f / (float)TOPK;
    }

    const char *iter_env = std::getenv("MOESTREAM_QWEN_CPU_BENCH_ITERS");
    const int iterations = std::max(1, iter_env ? std::atoi(iter_env) : 12);
    auto &runner = GgmlQwenCpuMoeRunner::instance();
    if (!runner.run_strided(input.data(), output.data(), ref, gate_raw, up_raw,
                            down_raw, ids.data(), N, weights.data(), D, FF,
                            TOPK, stride)) {
        runner.shutdown();
        std::free(arena_raw);
        std::cout << "  -> FAILED (runner setup/warm-up)\n";
        return false;
    }
    for (int i = 0; i < 2; ++i)
        if (!runner.run_strided(input.data(), output.data(), ref, gate_raw,
                                up_raw, down_raw, ids.data(), N,
                                weights.data(), D, FF, TOPK, stride))
            ok = false;
    const auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        if (!runner.run_strided(input.data(), output.data(), ref, gate_raw,
                                up_raw, down_raw, ids.data(), N,
                                weights.data(), D, FF, TOPK, stride))
            ok = false;
    }
    const double elapsed_ms = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - t0).count();
    const double ms_per_layer = elapsed_ms / (double)iterations;
    double checksum = 0.0;
    for (float v : output)
        checksum += v;
    const int threads = omp_get_max_threads();
    std::cout << "  shape D=" << D << " FF=" << FF << " experts=" << N
              << " topk=" << TOPK << " types="
              << (use_iq2 ? "IQ2_XXS/IQ2_XXS" : "IQ1_M/IQ1_M")
              << "/IQ4_NL"
              << " OMP threads=" << threads << "\n"
              << "  " << iterations << " calls: " << std::fixed
              << std::setprecision(3) << ms_per_layer << " ms/layer"
              << " | 6 CPU layers=" << (1000.0 / (6.0 * ms_per_layer))
              << " tok/s | 16 CPU layers="
              << (1000.0 / (16.0 * ms_per_layer))
              << " tok/s | checksum=" << checksum << "\n";

    // Optional apples-to-apples kernel probe.  The full GGML graph above is
    // still the production reference; this isolates only the IQ1_M gate/up
    // projection so an AVX-512 improvement is not hidden by down/IQ4_NL or
    // graph scheduling time.  It is opt-in because it adds a second timed
    // experiment to this diagnostic benchmark.
    if (!use_iq2 && std::getenv("MOESTREAM_QWEN_CPU_BENCH_AVX512")) {
        std::vector<const uint8_t *> gate_weights((size_t)TOPK);
        std::vector<std::vector<float>> ggml_out(
            (size_t)TOPK, std::vector<float>((size_t)FF));
        std::vector<std::vector<float>> avx512_out(
            (size_t)TOPK, std::vector<float>((size_t)FF));
        std::vector<float *> ggml_ptr((size_t)TOPK);
        std::vector<float *> avx512_ptr((size_t)TOPK);
        for (int k = 0; k < TOPK; ++k) {
            gate_weights[(size_t)k] =
                static_cast<const uint8_t *>(gate_raw) +
                (size_t)ids[(size_t)k] * stride;
            ggml_ptr[(size_t)k] = ggml_out[(size_t)k].data();
            avx512_ptr[(size_t)k] = avx512_out[(size_t)k].data();
        }
        for (int i = 0; i < 2; ++i) {
            matmul_iq1_m_ggml_q8k_grouped(
                ggml_ptr.data(), input.data(), gate_weights.data(), TOPK, 1,
                D, FF);
            matmul_iq1_m_avx512_q8k_grouped(
                avx512_ptr.data(), input.data(), gate_weights.data(), TOPK, 1,
                D, FF);
        }
        const int probe_iters = std::max(1, iterations * 2);
        auto probe = [&](bool avx512) {
            const auto begin = std::chrono::steady_clock::now();
            for (int i = 0; i < probe_iters; ++i) {
                if (avx512)
                    matmul_iq1_m_avx512_q8k_grouped(
                        avx512_ptr.data(), input.data(), gate_weights.data(),
                        TOPK, 1, D, FF);
                else
                    matmul_iq1_m_ggml_q8k_grouped(
                        ggml_ptr.data(), input.data(), gate_weights.data(),
                        TOPK, 1, D, FF);
            }
            return std::chrono::duration<double, std::milli>(
                       std::chrono::steady_clock::now() - begin)
                       .count() /
                   probe_iters;
        };
        const double ggml_ms = probe(false);
        const double avx512_ms = probe(true);
        double max_diff = 0.0;
        for (int k = 0; k < TOPK; ++k)
            for (int o = 0; o < FF; ++o)
                max_diff = std::max(
                    max_diff,
                    static_cast<double>(std::abs(
                        ggml_out[(size_t)k][(size_t)o] -
                        avx512_out[(size_t)k][(size_t)o])));
        std::cout << "  IQ1_M gate/up kernel probe: GGML-AVX2=" << ggml_ms
                  << " ms/grouped-call AVX-512=" << avx512_ms
                  << " ms/grouped-call speedup=" << (ggml_ms / avx512_ms)
                  << "x max_abs_diff=" << max_diff << "\n";
    }

    // Optional direct-kernel A/B for the Qwen strided grouped path. The
    // persistent GGML graph benchmark above does not exercise these wrappers;
    // this isolates the gate/up pair's single quantization/team launch from
    // the rest of the CPU MoE graph.
    if (!use_iq2 &&
        std::getenv("MOESTREAM_QWEN_CPU_BENCH_DIRECT_GROUPED")) {
        constexpr int DIRECT_M = TOPK;
        const int direct_rb = row_bytes(GGUFType::IQ1_M, D);
        std::vector<std::vector<uint8_t>> direct_weights(
            (size_t)DIRECT_M,
            std::vector<uint8_t>((size_t)FF * direct_rb, 0));
        std::vector<const uint8_t *> direct_ptr((size_t)DIRECT_M);
        for (int m = 0; m < DIRECT_M; ++m)
            direct_ptr[(size_t)m] = direct_weights[(size_t)m].data();
        std::vector<std::vector<float>> direct_gate(
            (size_t)DIRECT_M, std::vector<float>((size_t)FF));
        std::vector<std::vector<float>> direct_up(
            (size_t)DIRECT_M, std::vector<float>((size_t)FF));
        std::vector<float *> direct_gate_ptr((size_t)DIRECT_M),
            direct_up_ptr((size_t)DIRECT_M);
        for (int m = 0; m < DIRECT_M; ++m) {
            direct_gate_ptr[(size_t)m] = direct_gate[(size_t)m].data();
            direct_up_ptr[(size_t)m] = direct_up[(size_t)m].data();
        }
        for (int i = 0; i < 2; ++i) {
            matmul_iq1_m_ggml_q8k_grouped(
                direct_gate_ptr.data(), input.data(), direct_ptr.data(),
                DIRECT_M, 1, D, FF);
            matmul_iq1_m_ggml_q8k_grouped(
                direct_up_ptr.data(), input.data(), direct_ptr.data(),
                DIRECT_M, 1, D, FF);
            matmul_iq1_m_ggml_q8k_grouped_pair(
                direct_gate_ptr.data(), direct_up_ptr.data(), input.data(),
                direct_ptr.data(), direct_ptr.data(), DIRECT_M, 1, D, FF);
        }
        const int direct_iters = std::max(1, iterations * 2);
        auto direct_time = [&](bool pair) {
            const auto begin = std::chrono::steady_clock::now();
            for (int i = 0; i < direct_iters; ++i) {
                if (pair) {
                    matmul_iq1_m_ggml_q8k_grouped_pair(
                        direct_gate_ptr.data(), direct_up_ptr.data(),
                        input.data(), direct_ptr.data(), direct_ptr.data(),
                        DIRECT_M, 1, D, FF);
                } else {
                    matmul_iq1_m_ggml_q8k_grouped(
                        direct_gate_ptr.data(), input.data(), direct_ptr.data(),
                        DIRECT_M, 1, D, FF);
                    matmul_iq1_m_ggml_q8k_grouped(
                        direct_up_ptr.data(), input.data(), direct_ptr.data(),
                        DIRECT_M, 1, D, FF);
                }
            }
            return std::chrono::duration<double, std::milli>(
                       std::chrono::steady_clock::now() - begin)
                       .count() /
                   direct_iters;
        };
        const double separate_ms = direct_time(false);
        const double pair_ms = direct_time(true);
        std::cout << "  direct IQ1_M gate/up: separate=" << separate_ms
                  << " ms/pair pair=" << pair_ms
                  << " ms/pair speedup=" << (separate_ms / pair_ms)
                  << "x (OMP threads=" << omp_get_max_threads() << ")\n";
    }
    runner.shutdown();
    std::free(arena_raw);
    std::cout << "  -> Qwen CPU benchmark " << (ok ? "COMPLETED" : "FAILED")
              << "\n";
    return ok;
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

static void bench_q3k_cpu_only() {
    std::cout << "\n[BENCH] Q3_K CPU dequant+matmul (S=1, I=5120, O=1536)\n";
    ggml_cpu_init();
    constexpr int hidden = 5120;
    constexpr int output = 1536;
    constexpr int iterations = 20;
    const size_t rb = (size_t)row_bytes(GGUFType::Q3_K, hidden);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> weights_f32((size_t)output * hidden);
    std::vector<float> x((size_t)hidden);
    for (float &v : weights_f32) v = dist(rng);
    for (float &v : x) v = dist(rng);
    std::vector<uint8_t> weights((size_t)output * rb);
    for (int o = 0; o < output; ++o) {
        const int ok = ggml_quantize_row_ref(
            (uint32_t)GGUFType::Q3_K,
            weights_f32.data() + (size_t)o * hidden,
            weights.data() + (size_t)o * rb, hidden);
        assert(ok && "GGML Q3_K reference quantizer should be available");
    }
    std::vector<float> y_moe((size_t)output), y_ggml((size_t)output);
    std::vector<block_q8_K_local> x_q8((size_t)hidden / 256);

    for (int nthreads : {1, 2, 4, 6, 8}) {
        omp_set_num_threads(nthreads);
        quantize_row_q8_K(x.data(), x_q8.data(), hidden);
        cpu_quant_matmul(GGUFType::Q3_K, weights.data(), x.data(),
                         y_moe.data(), 1, hidden, output);
        #pragma omp parallel for schedule(static)
        for (int o = 0; o < output; ++o) {
            float dot = 0.0f;
            ggml_vec_dot_q3_K_q8_K(hidden, &dot, 0,
                weights.data() + (size_t)o * rb, 0,
                x_q8.data(), 0, 1);
            y_ggml[o] = dot;
        }
        double max_abs_diff = 0.0;
        double max_abs_output = 0.0;
        for (int o = 0; o < output; ++o)
        {
            max_abs_diff = std::max(max_abs_diff,
                std::abs((double)y_moe[o] - (double)y_ggml[o]));
            max_abs_output = std::max(max_abs_output,
                std::abs((double)y_moe[o]));
        }

        auto start = std::chrono::high_resolution_clock::now();
        for (int it = 0; it < iterations; ++it)
            cpu_quant_matmul(GGUFType::Q3_K, weights.data(), x.data(),
                             y_moe.data(), 1, hidden, output);
        const double moe_ms = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - start).count() / iterations;

        start = std::chrono::high_resolution_clock::now();
        for (int it = 0; it < iterations; ++it) {
            #pragma omp parallel for schedule(static)
            for (int o = 0; o < output; ++o) {
                float dot = 0.0f;
                ggml_vec_dot_q3_K_q8_K(hidden, &dot, 0,
                    weights.data() + (size_t)o * rb, 0,
                    x_q8.data(), 0, 1);
                y_ggml[o] = dot;
            }
        }
        const double ggml_ms = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - start).count() / iterations;

        start = std::chrono::high_resolution_clock::now();
        for (int it = 0; it < iterations; ++it)
            quantize_row_q8_K(x.data(), x_q8.data(), hidden);
        const double quant_us = std::chrono::duration<double, std::micro>(
            std::chrono::high_resolution_clock::now() - start).count() / iterations;
        std::cout << "  threads=" << nthreads << " MoEStream-FP32=" << std::fixed
                  << std::setprecision(3) << moe_ms << " ms, GGML-Q8_K-dot="
                  << ggml_ms << " ms (+ activation quant " << quant_us
                  << " us), speedup=" << (moe_ms / ggml_ms)
                  << "x, max_abs_diff=" << max_abs_diff
                  << " (max |MoE output|=" << max_abs_output << ")\n";
    }
}

#ifdef __linux__
static size_t smaps_anon_huge_kb(const void *address) {
    std::ifstream smaps("/proc/self/smaps");
    std::string line;
    const uintptr_t target = reinterpret_cast<uintptr_t>(address);
    bool in_mapping = false;
    while (std::getline(smaps, line)) {
        unsigned long long begin = 0, end = 0;
        if (std::sscanf(line.c_str(), "%llx-%llx", &begin, &end) == 2) {
            if (in_mapping) break;
            in_mapping = target >= begin && target < end;
            continue;
        }
        if (in_mapping && line.rfind("AnonHugePages:", 0) == 0) {
            unsigned long long kb = 0;
            if (std::sscanf(line.c_str(), "AnonHugePages: %llu kB", &kb) == 1)
                return (size_t)kb;
        }
    }
    return 0;
}

static void bench_q3k_thp_cpu_only() {
    std::cout << "\n[BENCH] Q3_K 2 MiB THP A/B (CPU only, 4 threads)\n";
    constexpr int hidden = 5120;
    constexpr int output = 8192; // 18 MiB Q3_K weights; larger than this CPU's LLC
    constexpr int samples = 9;
    constexpr int threads = 4;
    const size_t rb = (size_t)row_bytes(GGUFType::Q3_K, hidden);
    const size_t bytes = (size_t)output * rb;
    const long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0) {
        std::cerr << "  ERROR: cannot determine system page size\n";
        return;
    }
    const size_t map_bytes = (bytes + (size_t)page_size - 1) /
                             (size_t)page_size * (size_t)page_size;
    enum class Advice { NoHuge, Huge };
    struct Mapping { const char *name; Advice advice; uint8_t *ptr; };
    Mapping maps[] = {{"4K forced", Advice::NoHuge, nullptr},
                      {"2M THP", Advice::Huge, nullptr}};

    std::mt19937 rng(901);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> source((size_t)output * hidden);
    std::vector<float> x((size_t)hidden), y((size_t)output);
    for (float &v : source) v = dist(rng);
    for (float &v : x) v = dist(rng);

    for (auto &m : maps) {
        void *p = mmap(nullptr, map_bytes, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (p == MAP_FAILED) {
            std::cerr << "  ERROR: mmap failed for " << m.name << "\n";
            for (auto &old : maps) if (old.ptr) munmap(old.ptr, map_bytes);
            return;
        }
        m.ptr = static_cast<uint8_t *>(p);
        const int advice = m.advice == Advice::Huge ? MADV_HUGEPAGE
                                                    : MADV_NOHUGEPAGE;
        if (madvise(m.ptr, map_bytes, advice) != 0) {
            std::cerr << "  ERROR: madvise failed for " << m.name << "\n";
            for (auto &old : maps) if (old.ptr) munmap(old.ptr, map_bytes);
            return;
        }
        for (int o = 0; o < output; ++o) {
            const int ok = ggml_quantize_row_ref(
                (uint32_t)GGUFType::Q3_K,
                source.data() + (size_t)o * hidden,
                m.ptr + (size_t)o * rb, hidden);
            assert(ok && "GGML Q3_K reference quantizer should be available");
        }
    }

    ggml_cpu_init();
    omp_set_num_threads(threads);
    std::cout << "  global THP policy: ";
    { std::ifstream f("/sys/kernel/mm/transparent_hugepage/enabled");
      std::string s; std::getline(f, s); std::cout << s << "\n"; }
    for (auto &m : maps)
        std::cout << "  " << m.name << " resident AnonHugePages="
                  << (smaps_anon_huge_kb(m.ptr) / 1024.0) << " MiB\n";

    // Warm both mappings, then rotate their measurement order each round
    // so one advice mode does not always benefit from being measured first.
    for (auto &m : maps)
        cpu_quant_matmul(GGUFType::Q3_K, m.ptr, x.data(), y.data(), 1,
                         hidden, output);
    std::vector<double> times[2];
    for (int sample = 0; sample < samples; ++sample) {
        for (int j = 0; j < 2; ++j) {
            const int idx = (j + sample) % 2;
            auto start = std::chrono::steady_clock::now();
            cpu_quant_matmul(GGUFType::Q3_K, maps[idx].ptr, x.data(), y.data(),
                             1, hidden, output);
            times[idx].push_back(std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start).count());
        }
    }
    for (int i = 0; i < 2; ++i) {
        std::sort(times[i].begin(), times[i].end());
        const double median_ms = times[i][times[i].size() / 2];
        const double gbps = (bytes / 1e9) / (median_ms / 1000.0);
        std::cout << "  " << maps[i].name << ": median=" << std::fixed
                  << std::setprecision(3) << median_ms << " ms, "
                  << std::setprecision(2) << gbps << " GB/s\n";
    }
    for (auto &m : maps) munmap(m.ptr, map_bytes);
}
#endif

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

// Regression test for GLM5NEXT/DSA's nope-only score path. A zero-sized RoPE
// component must still run the no-PE GEMM, one joint softmax, and latent
// context GEMM on CUDA instead of being rejected and falling back to the
// O(T) AVX-512 CPU loop.
bool test_score_gpu_nope_only() {
    std::cout << "[Test 33] Score GPU path with rope=0 (DSA/nope-only)...\n";
    CudaManager gpu;
    if (!gpu.initialize()) {
        std::cout << "  -> Skipped (No CUDA Device Found on system)\n";
        return true;
    }
    const int H = 4, kvl = 13, rope = 0, T = 17;
    const float scale = 0.19f;
    std::mt19937 rng(5330);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> Lc((size_t)T * kvl), Q((size_t)H * kvl);
    for (auto& v : Lc) v = dist(rng);
    for (auto& v : Q) v = dist(rng);

    std::vector<float> ref((size_t)H * kvl, 0.0f);
    for (int h = 0; h < H; ++h) {
        std::vector<float> scores(T);
        for (int t = 0; t < T; ++t) {
            double dot = 0.0;
            for (int i = 0; i < kvl; ++i)
                dot += (double)Q[(size_t)h * kvl + i] * Lc[(size_t)t * kvl + i];
            scores[t] = (float)(dot * scale);
        }
        float mx = *std::max_element(scores.begin(), scores.end());
        float sum = 0.0f;
        for (float& v : scores) { v = std::exp(v - mx); sum += v; }
        for (float& v : scores) v /= sum;
        for (int t = 0; t < T; ++t)
            for (int i = 0; i < kvl; ++i)
                ref[(size_t)h * kvl + i] += scores[t] * Lc[(size_t)t * kvl + i];
    }

    if (!gpu.ensure_lc_rc_gpu(0, T, kvl, rope, 1ull << 30) ||
        !gpu.upload_lc_rc_rows(0, Lc.data(), nullptr, 0, T, kvl, rope)) {
        std::cout << "  -> FAILED: nope-only Lc/Rc setup rejected\n";
        return false;
    }
    std::vector<float> got(ref.size(), -999.0f);
    if (!gpu.score_gpu_batched_host(0, Q.data(), nullptr, got.data(),
                                    H, T, kvl, rope, scale, 1, T - 1)) {
        std::cout << "  -> FAILED: nope-only score GPU path returned false\n";
        return false;
    }
    double max_abs = 0.0, ref_abs = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
        max_abs = std::max(max_abs, (double)std::fabs(got[i] - ref[i]));
        ref_abs = std::max(ref_abs, (double)std::fabs(ref[i]));
    }
    const double normalized = max_abs / (ref_abs + 1e-9);
    const bool ok = normalized < 5e-4;
    std::cout << "  -> " << (ok ? "PASSED" : "FAILED")
              << " max_abs=" << max_abs << " normalized=" << normalized << "\n";
    return ok;
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
    // pos_base=T-1: this test's own CPU reference above attends over the
    // FULL T-length context with no causal masking at all -- the GPU call
    // must be told "this query's absolute position is the last one" so its
    // causal-mask logic (combine_scale_softmax_kernel's `valid = pos_base +
    // s + 1`, added 2026-09-01, see that kernel's own comment) computes
    // valid=T and stays inert, matching what this test actually checks.
    // FOUND AS A REAL BUG this session: left at the function's default
    // pos_base=0, `valid` collapsed to 1 (only key t=0 ever attended to),
    // making every head's context equal Lc row 0 regardless of its own
    // Q -- i.e. every head's output collapsed to the SAME value, tripping
    // the head-aliasing safeguard below. Not a production bug: Test 16 (the
    // S>1 sibling, added alongside the masking feature) already threads
    // pos_base through correctly; this S=1 test simply predates the
    // masking feature and was never updated for it.
    if (!gpu.score_gpu_batched_host(0, Q_absorbed.data(), Q_rope.data(), context_gpu.data(), H, T, kvl, rope, scale,
                                    /*S=*/1, /*pos_base=*/T - 1)) {
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

// ------------------------------------------------------------------
// 15. WUV int8 Batched GPU Path (score down-projection, int8-persistent
//     since 2026-08-29) -- see CudaManager::upload_wuv_int8/wuv_int8_
//     batched_host's own comments (cuda_manager.hpp) for the full design.
//     Two checks against two different references, matching this
//     project's own "isolate kernel correctness from inherent quantization
//     error" discipline (see the Q6_K/Q8_0 exact-reference fix elsewhere
//     in this file for the same principle applied to a different path):
//       (a) same-quantization CPU reference -- quantizes/dequantizes on
//           the CPU with the IDENTICAL per-row max-abs/127 scheme the GPU
//           path uses, then computes the reference dot products from
//           those dequantized values. Isolates "did the dequant kernel +
//           GEMM correctly implement the intended math" from int8's own
//           inherent quantization error -- should be tight (same order as
//           Test 14's own BF16 tolerance).
//       (b) exact F32 reference -- the real, honest answer to "how much
//           accuracy does switching from BF16 to int8 actually cost" that
//           a user of this path should expect, reported but not gated on
//           a tight tolerance (int8 quantization error is real and
//           expected here, not a bug).
// ------------------------------------------------------------------
// ---------------------------------------------------------------------------
// 16. MLA Score GPU Path at S>1 (PREFILL) -- causal mask + [H,S,*] layouts.
//
// Test 13 above covers S=1 (decode), where the causal mask is inert and every
// [H,S,*] buffer collapses to [H,*]. That means it exercises none of what the
// 2026-09-01 prefill batching actually added, and would stay green even if the
// mask or the s-major/h-major transposes were completely wrong -- the same
// coverage trap Test 5's K-quant cases fell into (all I=512, i.e. entirely
// below the lane8 dispatch gate).
//
// Two failure modes this is built to catch, both of which produce plausible
// finite numbers rather than a crash:
//   * causal mask off-by-one or absent -- a query attending to keys it must
//     not see. Caught because the CPU reference masks at exactly t <= pos_base+s
//     and the outputs diverge sharply if the GPU disagrees by even one key.
//   * row-major confusion -- Q_absorbed/context are [H,S,*] (h-major) while Q
//     and head_contexts are [S,H,*] (s-major). Swapping them scrambles WHICH
//     query's context lands in which output row. Caught by using S and H that
//     are both > 1 and DIFFERENT (a square S==H would let a transpose bug pass).
// pos_base > 0 so the mask is genuinely per-row (valid lengths 4..6 here),
// not just "row s sees s+1 keys".
bool test_score_gpu_prefill() {
    std::cout << "[Test 16] Score GPU Path at S>1 (prefill, causal mask + [H,S,*] layout)...\n";
    CudaManager gpu;
    if (!gpu.initialize()) {
        std::cout << "  -> Skipped (No CUDA Device Found on system)\n";
        return true;
    }

    // H != S deliberately (see above). T = pos_base + S, matching what
    // attention_layer.cpp now passes.
    const int H = 5, S = 3, kvl = 37, rope = 13, pos_base = 3;
    const int T = pos_base + S;   // 6
    std::mt19937 rng(97531);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> Lc((size_t)T * kvl), Rc((size_t)T * rope);
    for (auto& v : Lc) v = dist(rng);
    for (auto& v : Rc) v = dist(rng);
    // [H,S,*] h-major, exactly as attention_layer.cpp builds them.
    std::vector<float> Q_absorbed((size_t)H * S * kvl), Q_rope((size_t)H * S * rope);
    for (auto& v : Q_absorbed) v = dist(rng);
    for (auto& v : Q_rope) v = dist(rng);
    float scale = 0.211f;

    // CPU reference: same 4-step math as Test 13, but each row (h,s) attends
    // only to keys [0, pos_base+s].
    std::vector<float> context_cpu((size_t)H * S * kvl);
    for (int h = 0; h < H; ++h) {
        for (int sq = 0; sq < S; ++sq) {
            const int valid = pos_base + sq + 1;
            const float* qa = Q_absorbed.data() + ((size_t)h * S + sq) * kvl;
            const float* qr = Q_rope.data()     + ((size_t)h * S + sq) * rope;
            std::vector<double> scores(valid);
            for (int t = 0; t < valid; ++t) {
                double s_nope = 0.0, s_rope = 0.0;
                for (int i = 0; i < kvl; ++i)  s_nope += (double)qa[i] * (double)Lc[(size_t)t * kvl + i];
                for (int i = 0; i < rope; ++i) s_rope += (double)qr[i] * (double)Rc[(size_t)t * rope + i];
                scores[t] = (s_nope + s_rope) * scale;
            }
            double mx = scores[0];
            for (int t = 1; t < valid; ++t) mx = std::max(mx, scores[t]);
            double sum = 0.0;
            for (int t = 0; t < valid; ++t) { scores[t] = std::exp(scores[t] - mx); sum += scores[t]; }
            for (int t = 0; t < valid; ++t) scores[t] /= sum;
            float* out = context_cpu.data() + ((size_t)h * S + sq) * kvl;
            for (int i = 0; i < kvl; ++i) {
                double acc = 0.0;
                for (int t = 0; t < valid; ++t) acc += scores[t] * (double)Lc[(size_t)t * kvl + i];
                out[i] = (float)acc;
            }
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

    std::vector<float> context_gpu((size_t)H * S * kvl, -999.0f);
    if (!gpu.score_gpu_batched_host(0, Q_absorbed.data(), Q_rope.data(), context_gpu.data(),
                                    H, T, kvl, rope, scale, S, pos_base)) {
        std::cout << "  -> FAILED (score_gpu_batched_host returned false at S>1)\n";
        return false;
    }

    float max_abs_diff = 0.0f, max_rel_diff = 0.0f;
    for (size_t i = 0; i < context_cpu.size(); ++i) {
        if (std::isnan(context_gpu[i]) || std::isinf(context_gpu[i])) {
            std::cout << "  -> FAILED (NaN/Inf at output index " << i << ")\n";
            return false;
        }
        float diff = std::abs(context_cpu[i] - context_gpu[i]);
        max_abs_diff = std::max(max_abs_diff, diff);
        max_rel_diff = std::max(max_rel_diff, diff / std::max(1e-6f, std::abs(context_cpu[i])));
    }

    // Explicit anti-aliasing check: row (h,s) must NOT equal row (h,s') for
    // s != s'. Every row shares the same Lc/Rc and differs only in its query
    // and its causal limit, so a bug that broadcasts one row over the batch
    // (a strideB/rows mix-up in the GEMM) would still produce finite, sane
    // numbers -- and would still pass an aggregate diff check if the CPU
    // reference were computed the same wrong way. This compares GPU rows
    // against EACH OTHER, so it is independent of the reference entirely.
    for (int h = 0; h < H; ++h) {
        for (int a = 0; a + 1 < S; ++a) {
            const float* ra = context_gpu.data() + ((size_t)h * S + a) * kvl;
            const float* rb = context_gpu.data() + ((size_t)h * S + a + 1) * kvl;
            float d = 0.0f;
            for (int i = 0; i < kvl; ++i) d = std::max(d, std::abs(ra[i] - rb[i]));
            if (d < 1e-6f) {
                std::cout << "  -> FAILED (rows s=" << a << " and s=" << (a + 1)
                          << " of head " << h << " are identical -- batch dim collapsed)\n";
                return false;
            }
        }
    }

    bool pass = (max_abs_diff < 1e-4f) || (max_rel_diff < 1e-4f);
    std::cout << "  -> Score GPU prefill (H=" << H << " S=" << S << " kvl=" << kvl
              << " rope=" << rope << " pos_base=" << pos_base << " T=" << T << "): "
              << (pass ? "PASSED" : "FAILED")
              << "  max_abs_diff=" << max_abs_diff << " max_rel_diff=" << max_rel_diff << "\n";
    return pass;
}

// ---------------------------------------------------------------------------
// 17. WUV BF16 batched path at S>1 -- the [H,S,kvl] -> [S,H,vh] scatter.
//
// The prefill path relies on cuBLAS writing each batch's columns straight into
// an INTERLEAVED destination (ldc = H*vh, strideC = vh) so no transpose pass is
// needed. That is the single most likely thing to be silently wrong: a packed
// [H,S,vh] write produces the right VALUES in the wrong PLACES, which downstream
// looks like a subtly incoherent model rather than an error. H != S again so a
// transpose cannot alias.
bool test_wuv_batched_gpu_prefill() {
    std::cout << "[Test 17] WUV BF16 Batched GPU Path at S>1 ([H,S,kvl] -> [S,H,vh] scatter)...\n";
    CudaManager gpu;
    if (!gpu.initialize()) {
        std::cout << "  -> Skipped (No CUDA Device Found on system)\n";
        return true;
    }
    const int H = 5, S = 3, kvl = 37, vh = 23;
    std::mt19937 rng(24680);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> WUV_vec((size_t)H * kvl * vh);   // [H][vh][kvl]
    for (auto& v : WUV_vec) v = dist(rng);
    std::vector<float> X((size_t)H * S * kvl);          // [H,S,kvl]
    for (auto& v : X) v = dist(rng);

    // CPU reference, written directly in the [S,H,vh] destination layout.
    // Same-quantization methodology as Test 14 (its S=1 sibling) and Test
    // 15's own "(a)" reference: pre-round x/w through the SAME fp32->bf16->
    // fp32 round-trip the GPU path applies, before computing the reference
    // dot product. FOUND AS A REAL BUG this session: this test originally
    // compared against an EXACT F32 reference (no bf16 rounding at all)
    // with a tolerance far too tight for genuine BF16 GEMM noise (1e-4),
    // which fails on ordinary, expected BF16 rounding error (~0.3-0.6%
    // relative on values in the 1-5 range) -- confirmed via a one-off debug
    // dump showing every per-head value already agreeing to within normal
    // BF16 precision (e.g. cpu=1.08911 gpu=1.09167), not a real correctness
    // bug (a real scatter/layout bug was also ruled out separately via an
    // isolated cublasGemmStridedBatchedEx repro of the exact same ldc/
    // strideC scatter trick, which matched its expected output exactly).
    std::vector<float> y_cpu((size_t)S * H * vh);
    for (int h = 0; h < H; ++h)
        for (int sq = 0; sq < S; ++sq)
            for (int o = 0; o < vh; ++o) {
                double acc = 0.0;
                const float* x = X.data() + ((size_t)h * S + sq) * kvl;
                const float* w = WUV_vec.data() + (size_t)h * kvl * vh + (size_t)o * kvl;
                for (int i = 0; i < kvl; ++i) {
                    float xb = bf16_to_fp32(fp32_to_bf16(x[i]));
                    float wb = bf16_to_fp32(fp32_to_bf16(w[i]));
                    acc += (double)xb * (double)wb;
                }
                y_cpu[((size_t)sq * H + h) * vh + o] = (float)acc;
            }

    if (!gpu.upload_wuv_bf16(0, WUV_vec, H, kvl, vh, 1ull << 30)) {
        std::cout << "  -> FAILED (upload_wuv_bf16 returned false)\n";
        return false;
    }
    std::vector<float> y_gpu((size_t)S * H * vh, -999.0f);
    if (!gpu.wuv_bf16_batched_host(0, X.data(), y_gpu.data(), H, kvl, vh, S)) {
        std::cout << "  -> FAILED (wuv_bf16_batched_host returned false at S>1)\n";
        return false;
    }

    float max_abs_diff = 0.0f, max_rel_diff = 0.0f;
    for (size_t i = 0; i < y_cpu.size(); ++i) {
        if (std::isnan(y_gpu[i]) || std::isinf(y_gpu[i])) {
            std::cout << "  -> FAILED (NaN/Inf at output index " << i << ")\n";
            return false;
        }
        float diff = std::abs(y_cpu[i] - y_gpu[i]);
        max_abs_diff = std::max(max_abs_diff, diff);
        max_rel_diff = std::max(max_rel_diff, diff / std::max(1e-6f, std::abs(y_cpu[i])));
    }
    // Same-quantization reference now (see above) -- tighten to match Test
    // 14's own S=1 tolerance exactly, since both are now isolating pure GEMM
    // correctness from BF16's own inherent rounding noise the same way.
    bool pass = max_rel_diff < 0.01f;
    std::cout << "  -> WUV batched GPU prefill (H=" << H << " S=" << S << " kvl=" << kvl
              << " vh=" << vh << "): " << (pass ? "PASSED" : "FAILED")
              << "  max_abs_diff=" << max_abs_diff << " max_rel_diff=" << max_rel_diff << "\n";
    return pass;
}

// 2026-09-02: S>1 coverage for the int8 WUV path. Test 15 (its S=1 sibling)
// would stay green even if the batched strides and the [H,S,kvl]->[S,H,vh]
// scatter were both completely wrong, because at S=1 every stride collapses
// and ldc is unused -- the exact trap Test 5's K-quant cases fell into. H!=S
// deliberately, so a transposed index cannot alias into a passing answer.
bool test_wuv_int8_batched_gpu_prefill() {
    std::cout << "[Test 22] WUV int8 Batched GPU Path at S>1 ([H,S,kvl] -> [S,H,vh] scatter)...\n";
    CudaManager gpu;
    if (!gpu.initialize()) {
        std::cout << "  -> Skipped (No CUDA Device Found on system)\n";
        return true;
    }
    const int H = 5, S = 3, kvl = 37, vh = 23;
    std::mt19937 rng(13579);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> WUV_vec((size_t)H * kvl * vh);   // [H][vh][kvl]
    for (auto& v : WUV_vec) v = dist(rng);
    std::vector<float> X((size_t)H * S * kvl);          // [H,S,kvl]
    for (auto& v : X) v = dist(rng);

    // Same-quantization reference, written directly in the [S,H,vh]
    // destination layout: per-output-row symmetric int8 quantize/dequantize
    // (byte-for-byte upload_wuv_int8's scheme), then the same fp32->bf16
    // round-trip on both operands that the BF16-matched GEMM applies. This
    // isolates batched-stride/scatter correctness from int8's own accuracy
    // cost, exactly as Test 15 does at S=1.
    std::vector<float> y_cpu((size_t)S * H * vh);
    for (int h = 0; h < H; ++h)
        for (int o = 0; o < vh; ++o) {
            const float* w = WUV_vec.data() + (size_t)h * kvl * vh + (size_t)o * kvl;
            float max_abs = 0.0f;
            for (int i = 0; i < kvl; ++i) max_abs = std::max(max_abs, std::abs(w[i]));
            const float scale = max_abs / 127.0f;
            for (int sq = 0; sq < S; ++sq) {
                const float* x = X.data() + ((size_t)h * S + sq) * kvl;
                double acc = 0.0;
                for (int i = 0; i < kvl; ++i) {
                    float w_dq = w[i];
                    if (scale > 0.0f) {
                        float qf = std::round(w[i] / scale);
                        qf = std::max(-127.0f, std::min(127.0f, qf));
                        w_dq = qf * scale;
                    }
                    acc += (double)bf16_to_fp32(fp32_to_bf16(w_dq))
                         * (double)bf16_to_fp32(fp32_to_bf16(x[i]));
                }
                y_cpu[((size_t)sq * H + h) * vh + o] = (float)acc;
            }
        }

    if (!gpu.upload_wuv_int8(0, WUV_vec, H, kvl, vh, 1ull << 30)) {
        std::cout << "  -> FAILED (upload_wuv_int8 returned false)\n";
        return false;
    }
    std::vector<float> y_gpu((size_t)S * H * vh, -999.0f);
    if (!gpu.wuv_int8_batched_host(0, X.data(), y_gpu.data(), H, kvl, vh, S)) {
        std::cout << "  -> FAILED (wuv_int8_batched_host returned false at S>1)\n";
        return false;
    }

    float max_abs_diff = 0.0f, max_rel_diff = 0.0f;
    for (size_t i = 0; i < y_cpu.size(); ++i) {
        if (std::isnan(y_gpu[i]) || std::isinf(y_gpu[i])) {
            std::cout << "  -> FAILED (NaN/Inf at output index " << i << ")\n";
            return false;
        }
        float diff = std::abs(y_cpu[i] - y_gpu[i]);
        max_abs_diff = std::max(max_abs_diff, diff);
        max_rel_diff = std::max(max_rel_diff, diff / std::max(1e-6f, std::abs(y_cpu[i])));
    }
    // Independent anti-aliasing check, NOT dependent on the CPU reference:
    // rows sq and sq+1 of the same head share the weight matrix and differ
    // only in the query, so a strideB/ldc mix-up that broadcast one query
    // across the batch would still produce finite, plausible numbers AND
    // would still pass an aggregate diff check if the reference happened to
    // be computed the same wrong way. Distinct random queries must give
    // distinct outputs.
    bool distinct = false;
    for (int o = 0; o < vh && !distinct; ++o)
        if (std::abs(y_gpu[(size_t)0 * H * vh + o] - y_gpu[(size_t)1 * H * vh + o]) > 1e-6f)
            distinct = true;
    if (!distinct) {
        std::cout << "  -> FAILED (query rows s=0 and s=1 of head 0 are identical -- "
                     "batched stride/scatter is broadcasting one row)\n";
        return false;
    }

    bool pass = max_rel_diff < 0.01f;
    std::cout << "  -> WUV int8 batched GPU prefill (H=" << H << " S=" << S << " kvl=" << kvl
              << " vh=" << vh << "): " << (pass ? "PASSED" : "FAILED")
              << "  max_abs_diff=" << max_abs_diff << " max_rel_diff=" << max_rel_diff << "\n";
    return pass;
}

bool test_wuv_int8_batched_gpu() {
    std::cout << "[Test 15] WUV int8 Batched GPU Path (score down-projection)...\n";
    CudaManager gpu;
    if (!gpu.initialize()) {
        std::cout << "  -> Skipped (No CUDA Device Found on system)\n";
        return true;
    }

    const int H = 5, kvl = 37, vh = 23;  // same deliberately non-uniform shape as Test 14
    std::mt19937 rng(7171);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> WUV((size_t)H * kvl * vh);
    for (auto& v : WUV) v = dist(rng);
    std::vector<float> X((size_t)H * kvl);
    for (auto& v : X) v = dist(rng);

    // Same-quantization CPU reference: per-row (per h,o) symmetric int8
    // quantize/dequantize, byte-for-byte the same scheme upload_wuv_int8
    // uses, then an exact-F32 dot product from the dequantized values
    // (activation X is NOT quantized in this design -- only the weight
    // is -- matching wuv_int8_batched_host's own contract).
    std::vector<float> y_same_quant((size_t)H * vh, 0.0f);
    // Exact F32 reference (no quantization anywhere) -- the honest
    // real-world accuracy number.
    std::vector<float> y_exact((size_t)H * vh, 0.0f);
    for (int h = 0; h < H; ++h) {
        const float* Wh = WUV.data() + (size_t)h * kvl * vh;
        const float* Xh = X.data() + (size_t)h * kvl;
        float* Yh_q = y_same_quant.data() + (size_t)h * vh;
        float* Yh_e = y_exact.data() + (size_t)h * vh;
        for (int o = 0; o < vh; ++o) {
            const float* Wo = Wh + (size_t)o * kvl;
            float max_abs = 0.0f;
            for (int i = 0; i < kvl; ++i) max_abs = std::max(max_abs, std::abs(Wo[i]));
            float scale = max_abs / 127.0f;
            double acc_q = 0.0, acc_e = 0.0;
            for (int i = 0; i < kvl; ++i) {
                float w_dequant = Wo[i];
                if (scale > 0.0f) {
                    float qf = std::round(Wo[i] / scale);
                    qf = std::max(-127.0f, std::min(127.0f, qf));
                    w_dequant = qf * scale;
                }
                // Weight side also passes through the same fp32->bf16
                // round-trip the dequant kernel's own output feeds into
                // (the GEMM itself is BF16-matched) -- and the activation
                // side is bf16-rounded too, matching wuv_int8_batched_
                // host's own h_wuv_ctx_bf16_scratch conversion.
                float w_bf16 = bf16_to_fp32(fp32_to_bf16(w_dequant));
                float x_bf16 = bf16_to_fp32(fp32_to_bf16(Xh[i]));
                acc_q += (double)w_bf16 * (double)x_bf16;
                acc_e += (double)Wo[i] * (double)Xh[i];
            }
            Yh_q[o] = (float)acc_q;
            Yh_e[o] = (float)acc_e;
        }
    }

    std::vector<float> WUV_vec(WUV.begin(), WUV.end());
    if (!gpu.upload_wuv_int8(0, WUV_vec, H, kvl, vh, 1ull << 30)) {
        std::cout << "  -> FAILED (upload_wuv_int8 returned false)\n";
        return false;
    }

    std::vector<float> y_gpu((size_t)H * vh, -999.0f);
    if (!gpu.wuv_int8_batched_host(0, X.data(), y_gpu.data(), H, kvl, vh)) {
        std::cout << "  -> FAILED (wuv_int8_batched_host returned false)\n";
        return false;
    }

    float max_abs_diff_q = 0.0f, max_rel_diff_q = 0.0f;
    float max_abs_diff_e = 0.0f, max_rel_diff_e = 0.0f;
    for (size_t i = 0; i < y_gpu.size(); ++i) {
        if (std::isnan(y_gpu[i]) || std::isinf(y_gpu[i])) {
            std::cout << "  -> FAILED (NaN/Inf at output index " << i << ")\n";
            return false;
        }
        float diff_q = std::abs(y_same_quant[i] - y_gpu[i]);
        float rel_q = diff_q / std::max(1e-6f, std::abs(y_same_quant[i]));
        max_abs_diff_q = std::max(max_abs_diff_q, diff_q);
        max_rel_diff_q = std::max(max_rel_diff_q, rel_q);

        float diff_e = std::abs(y_exact[i] - y_gpu[i]);
        float rel_e = diff_e / std::max(1e-6f, std::abs(y_exact[i]));
        max_abs_diff_e = std::max(max_abs_diff_e, diff_e);
        max_rel_diff_e = std::max(max_rel_diff_e, rel_e);
    }

    // Gate ONLY on the same-quantization comparison (kernel correctness,
    // should be tight) -- the exact-F32 comparison is reported for
    // visibility into int8's real accuracy cost, not gated, since a large
    // relative error there is expected int8 quantization behavior, not a
    // bug (same principle as this file's own Q6_K/Q8_0 exact-vs-lossy
    // distinction elsewhere). CONFIRMED via a one-off real-dims run
    // (H=64,kvl=512,vh=256, GLM-5.2's actual shape, not kept as the
    // permanent test shape here since the small/non-uniform shape above is
    // better for catching indexing/stride bugs): every worst-relative-
    // error element traced back to a near-zero y_exact (e.g. 0.00106),
    // while absolute error stayed small and consistent (0.005-0.055)
    // across every output magnitude including large ones (0.05% relative
    // error at y_exact=10.9) -- the classic near-zero-crossing artifact,
    // not a real accuracy problem.
    bool pass = max_rel_diff_q < 0.01f;
    std::cout << "  -> WUV int8 batched GPU (H=" << H << " kvl=" << kvl << " vh=" << vh << "): "
              << (pass ? "PASSED" : "FAILED")
              << "  vs-same-quant: max_abs_diff=" << max_abs_diff_q << " max_rel_diff=" << max_rel_diff_q
              << "  |  vs-exact-F32 (informational): max_abs_diff=" << max_abs_diff_e
              << " max_rel_diff=" << max_rel_diff_e << "\n";
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

// Gated feasibility bench (MOESTREAM_RUN_BENCH=1, same convention as the
// others above) for dot_iq2_xxs_vnni_prototype (kernels_cpu.cpp) -- answers
// "does a VNNI int8 dot beat the existing dot_iq2_xxs_f32_avx512 FMA path"
// BEFORE committing to a real production kernel (see CLAUDE.md's
// cpu_offload_max_experts investigation: this codebase's own CPU expert
// path runs ~9x slower per call than the GPU path, and IQ2_XXS/IQ3_XXS --
// GLM-5.2's actual routed-expert types, 71-76/78 layers -- were confirmed
// via source read to have NO VNNI fast path at all, unlike Q4_0/Q6_K/Q8_0
// elsewhere in this file). Times BOTH the existing FMA baseline and the new
// VNNI prototype at a real production row size, AND checks accuracy of
// both against ggml's own independent reference (ggml_dequant_row + an
// exact F64 dot) -- not just against each other, matching this project's
// own established lesson (see CLAUDE.md's Q8_0/K-quant VNNI investigation)
// that a lossy path must never be its own test oracle.
void bench_iq2xxs_vnni_prototype() {
    std::cout << "\n[Bench] IQ2_XXS CPU dot: existing FMA path vs EXPERIMENTAL VNNI prototype...\n";
    const int I = 2048;  // real production row size (moe_inter-class dim)
    assert(I % 256 == 0);
    std::mt19937 rng(2718);
    std::uniform_real_distribution<float> wdist(-2.0f, 2.0f);
    std::uniform_real_distribution<float> xdist(-1.0f, 1.0f);

    std::vector<float> w_f32((size_t)I);
    for (auto& v : w_f32) v = wdist(rng);
    int rb = row_bytes(GGUFType::IQ2_XXS, I);
    std::vector<uint8_t> w_bytes((size_t)rb);
    int ok = ggml_quantize_row_ref((uint32_t)GGUFType::IQ2_XXS, w_f32.data(), w_bytes.data(), I);
    assert(ok && "ggml_quantize_row_ref should support IQ2_XXS");

    std::vector<float> x((size_t)I);
    for (auto& v : x) v = xdist(rng);

    // Exact reference: ggml's OWN independent dequant (not this project's
    // AVX-512 port) + a plain F64-accumulated dot -- the "don't let a lossy
    // path be its own oracle" lesson from the Q8_0/K-quant investigation.
    std::vector<float> w_exact((size_t)I, -999.0f);
    int dq_ok = ggml_dequant_row((uint32_t)GGUFType::IQ2_XXS, w_bytes.data(), w_exact.data(), I);
    assert(dq_ok && "ggml_dequant_row should support IQ2_XXS");
    double exact = 0.0;
    for (int i = 0; i < I; ++i) exact += (double)w_exact[i] * (double)x[i];

    // Existing FMA baseline (production code, unmodified).
    float y_fma = dot_iq2_xxs_f32_avx512(w_bytes.data(), x.data(), I);

    // VNNI prototype needs int8-quantized activations (32-wide blocks, same
    // granularity dot_iq2_xxs_vnni_prototype indexes by).
    int num_blocks = I / 32;
    std::vector<block_q8_0> xq(num_blocks);
    quantize_activations_q8_0(x.data(), xq.data(), I);
    float y_vnni = dot_iq2_xxs_vnni_prototype(w_bytes.data(), xq.data(), I);

    // AVX2 prototype (ggml upstream port) needs Q8_K-quantized activations
    // (one scale per 256-element superblock, not per-32 like block_q8_0).
    int num_superblocks = I / 256;
    std::vector<block_q8_K_local> xk(num_superblocks);
    quantize_row_q8_K_ref(x.data(), xk.data(), I);
    float y_avx2 = dot_iq2_xxs_avx2_prototype(w_bytes.data(), xk.data(), I);

    auto rel_err = [&](double v) {
        return std::abs(v - exact) / std::max(1e-6, std::abs(exact));
    };
    std::cout << std::setprecision(6)
              << "  Accuracy (I=" << I << "): exact_ref=" << exact
              << "  FMA=" << y_fma << " (rel_err=" << rel_err(y_fma) << ")"
              << "  VNNI=" << y_vnni << " (rel_err=" << rel_err(y_vnni) << ")"
              << "  AVX2(ggml-port)=" << y_avx2 << " (rel_err=" << rel_err(y_avx2) << ")\n";

    // A single I=2048 dot is sub-microsecond -- WARMUP/ITERS sized (and
    // reported in us, not ms) specifically so the per-call number clears
    // std::chrono's real resolution floor instead of rounding to noise.
    const int WARMUP = 200, ITERS = 200000;
    for (int it = 0; it < WARMUP; ++it) dot_iq2_xxs_f32_avx512(w_bytes.data(), x.data(), I);
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < ITERS; ++it) dot_iq2_xxs_f32_avx512(w_bytes.data(), x.data(), I);
    double fma_us = std::chrono::duration<double, std::micro>(
        std::chrono::high_resolution_clock::now() - t0).count() / ITERS;

    for (int it = 0; it < WARMUP; ++it) dot_iq2_xxs_vnni_prototype(w_bytes.data(), xq.data(), I);
    t0 = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < ITERS; ++it) dot_iq2_xxs_vnni_prototype(w_bytes.data(), xq.data(), I);
    double vnni_us = std::chrono::duration<double, std::micro>(
        std::chrono::high_resolution_clock::now() - t0).count() / ITERS;

    // quantize_activations_q8_0's own cost -- paid ONCE per token (all O
    // output rows share the same quantized activation), not once per row
    // like the two dot timings above, so report it separately rather than
    // fold it into vnni_us (which would misrepresent the real per-row cost
    // at any realistic O > 1).
    for (int it = 0; it < WARMUP; ++it) quantize_activations_q8_0(x.data(), xq.data(), I);
    t0 = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < ITERS; ++it) quantize_activations_q8_0(x.data(), xq.data(), I);
    double qact_us = std::chrono::duration<double, std::micro>(
        std::chrono::high_resolution_clock::now() - t0).count() / ITERS;

    for (int it = 0; it < WARMUP; ++it) dot_iq2_xxs_avx2_prototype(w_bytes.data(), xk.data(), I);
    t0 = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < ITERS; ++it) dot_iq2_xxs_avx2_prototype(w_bytes.data(), xk.data(), I);
    double avx2_us = std::chrono::duration<double, std::micro>(
        std::chrono::high_resolution_clock::now() - t0).count() / ITERS;

    for (int it = 0; it < WARMUP; ++it) quantize_row_q8_K_ref(x.data(), xk.data(), I);
    t0 = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < ITERS; ++it) quantize_row_q8_K_ref(x.data(), xk.data(), I);
    double qk_us = std::chrono::duration<double, std::micro>(
        std::chrono::high_resolution_clock::now() - t0).count() / ITERS;

    std::cout << std::setprecision(4)
              << "  Throughput (single dot, I=" << I << "): FMA=" << fma_us
              << "us/call  VNNI=" << vnni_us << "us/call (speedup=" << (fma_us / vnni_us) << "x)"
              << "  AVX2(ggml-port)=" << avx2_us << "us/call (speedup=" << (fma_us / avx2_us) << "x)\n"
              << "  Activation quant cost (paid once/token, not once/row): "
                 "Q8_0-style=" << qact_us << "us  Q8_K-style=" << qk_us << "us\n";
}

// IQ3_XXS sibling of the bench above -- GLM-5.2's down-proj type (same
// 71-76/78 layers as IQ2_XXS gate/up). Only the ggml-port AVX2 kernel is
// compared here (the VNNI prototype was never built for IQ3_XXS -- its own
// verdict on IQ2_XXS already ruled that avenue out before IQ3_XXS's own
// port would have been attempted).
void bench_iq3xxs_avx2_prototype() {
    std::cout << "\n[Bench] IQ3_XXS CPU dot: existing FMA path vs EXPERIMENTAL AVX2 (ggml-port) prototype...\n";
    const int I = 2048;
    assert(I % 256 == 0);
    std::uniform_real_distribution<float> wdist(-2.0f, 2.0f);
    std::uniform_real_distribution<float> xdist(-1.0f, 1.0f);

    // Accuracy across N independent rows/seeds -- a single draw (the first
    // real number here came back at 1.6% rel_err, notably higher than
    // IQ2_XXS's 0.19%) isn't enough to tell "this format is just noisier"
    // from "this seed happened to be unlucky" apart. Matches Test 7's own
    // run_dequant_case discipline (multiple rows, not one).
    const int N_ROWS = 20;
    double worst_fma = 0.0, worst_avx2 = 0.0, sum_avx2 = 0.0;
    float y_fma_last = 0.0f, y_avx2_last = 0.0f;
    double exact_last = 0.0;
    std::vector<uint8_t> w_bytes_last;
    std::vector<float> x_last;
    std::vector<block_q8_K_local> xk_last;

    int rb = row_bytes(GGUFType::IQ3_XXS, I);
    for (int row = 0; row < N_ROWS; ++row) {
        std::mt19937 rng(1618 + row);
        std::vector<float> w_f32((size_t)I);
        for (auto& v : w_f32) v = wdist(rng);
        std::vector<uint8_t> w_bytes((size_t)rb);
        int ok = ggml_quantize_row_ref((uint32_t)GGUFType::IQ3_XXS, w_f32.data(), w_bytes.data(), I);
        assert(ok && "ggml_quantize_row_ref should support IQ3_XXS");

        std::vector<float> x((size_t)I);
        for (auto& v : x) v = xdist(rng);

        std::vector<float> w_exact((size_t)I, -999.0f);
        int dq_ok = ggml_dequant_row((uint32_t)GGUFType::IQ3_XXS, w_bytes.data(), w_exact.data(), I);
        assert(dq_ok && "ggml_dequant_row should support IQ3_XXS");
        double exact = 0.0;
        for (int i = 0; i < I; ++i) exact += (double)w_exact[i] * (double)x[i];

        float y_fma = dot_iq3_xxs_f32_avx512(w_bytes.data(), x.data(), I);
        int num_superblocks = I / 256;
        std::vector<block_q8_K_local> xk(num_superblocks);
        quantize_row_q8_K_ref(x.data(), xk.data(), I);
        float y_avx2 = dot_iq3_xxs_avx2_prototype(w_bytes.data(), xk.data(), I);

        auto rel_err = [&](double v) { return std::abs(v - exact) / std::max(1e-6, std::abs(exact)); };
        worst_fma = std::max(worst_fma, rel_err(y_fma));
        double re = rel_err(y_avx2);
        worst_avx2 = std::max(worst_avx2, re);
        sum_avx2 += re;

        if (row == N_ROWS - 1) {
            y_fma_last = y_fma; y_avx2_last = y_avx2; exact_last = exact;
            w_bytes_last = std::move(w_bytes); x_last = std::move(x); xk_last = std::move(xk);
        }
    }
    std::cout << std::setprecision(6)
              << "  Accuracy over " << N_ROWS << " row(s) (I=" << I << "): "
                 "FMA worst_rel_err=" << worst_fma
              << "  AVX2(ggml-port) avg_rel_err=" << (sum_avx2 / N_ROWS)
              << " worst_rel_err=" << worst_avx2
              << "  (last row: exact_ref=" << exact_last << " FMA=" << y_fma_last
              << " AVX2=" << y_avx2_last << ")\n";

    const uint8_t* w_bytes_ptr = w_bytes_last.data();
    const float* x_ptr = x_last.data();
    const block_q8_K_local* xk_ptr = xk_last.data();

    const int WARMUP = 200, ITERS = 200000;
    for (int it = 0; it < WARMUP; ++it) dot_iq3_xxs_f32_avx512(w_bytes_ptr, x_ptr, I);
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < ITERS; ++it) dot_iq3_xxs_f32_avx512(w_bytes_ptr, x_ptr, I);
    double fma_us = std::chrono::duration<double, std::micro>(
        std::chrono::high_resolution_clock::now() - t0).count() / ITERS;

    for (int it = 0; it < WARMUP; ++it) dot_iq3_xxs_avx2_prototype(w_bytes_ptr, xk_ptr, I);
    t0 = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < ITERS; ++it) dot_iq3_xxs_avx2_prototype(w_bytes_ptr, xk_ptr, I);
    double avx2_us = std::chrono::duration<double, std::micro>(
        std::chrono::high_resolution_clock::now() - t0).count() / ITERS;

    std::cout << std::setprecision(4)
              << "  Throughput (single dot, I=" << I << "): FMA=" << fma_us
              << "us/call  AVX2(ggml-port)=" << avx2_us << "us/call (speedup=" << (fma_us / avx2_us) << "x)\n";
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

// ------------------------------------------------------------------
// 18. Mixed-precision-within-one-layer Phase 1 correctness test.
//
// Tests the FILE/loader/GPU-dispatch layer directly, independent of
// StreamingEngine's own sanity_check_unpacked_format() (that function's
// own relaxation is separately verified by Test 19, right below). Given a
// real, hand-built GGUF file whose layer 0 has two experts of GENUINELY
// DIFFERENT real quant types (unpacked per-expert tensor naming,
// blk.0.ffn_{gate,up,down}.<E>.weight), does gguf_loader.hpp correctly
// parse and offset-resolve BOTH experts, and does the GPU fused-kernel
// dispatch path correctly handle two different types loaded side-by-side
// in one session (no cross-type __constant__-memory bleed, no
// offset/stride confusion)?
//
// Both experts' gate/up/down bytes are copied VERBATIM (no re-quantization)
// from two real, already-verified-correct sources this session established:
// expert 0 from production (IQ2_XXS/IQ2_XXS/IQ3_XXS, layer 20 expert 5),
// expert 1 from the real downloaded IQ1_S release (IQ1_S/IQ1_S/IQ3_XXS,
// layer 60 expert 7 -- inside the confirmed-real-IQ1_S range 56-67).
// Gated (SKIPPED, not FAILED) behind both source paths existing, matching
// Test 11's own "real files this exact session has, not every checkout"
// convention -- this is a real-hardware/real-data verification, not a
// portable unit test.
bool test_mixed_precision_layer() {
    std::cout << "[Test 18] Mixed-Precision-Within-One-Layer (Phase 1: file read + VRAM dispatch)...\n";

    const std::string prod_path = "/home/ray/vllm-workspace/GLM-5.3-GGUF-align";
    const std::string iq1s_path = "/home/ray/vllm-workspace/GLM-5.3-GGUF-align/"
        "repacked-GLM-5.3-UD-IQ1_S-00005-of-00006-part00001-of-00001.gguf";

    struct ExpertSpec { std::string src_path; int src_layer; int src_expert; int out_expert; };
    std::vector<ExpertSpec> specs = {
        {prod_path, 20, 5, 0},
        {iq1s_path, 60, 7, 1},
    };

    struct OutTensor { std::string name; GGUFType type; std::vector<uint64_t> shape; std::vector<uint8_t> data; };
    std::vector<OutTensor> out_tensors;

    for (auto& spec : specs) {
        GGUFFile f;
        if (!f.load_header(spec.src_path)) {
            std::cout << "  -> SKIPPED (real source not available: " << spec.src_path << ")\n";
            return true;
        }
        for (const char* kind : {"gate", "up", "down"}) {
            std::string tn = "blk." + std::to_string(spec.src_layer) + ".ffn_" + kind + "_exps.weight";
            auto it = f.tensors.find(tn);
            if (it == f.tensors.end()) {
                std::cout << "  -> FAILED (tensor not found: " << tn << " in " << spec.src_path << ")\n";
                return false;
            }
            const GGUFTensor& t = it->second;
            uint64_t n_experts = t.shape.back();
            uint64_t bpe = t.size_bytes / n_experts;
            uint64_t eoff = t.offset + (uint64_t)spec.src_expert * bpe;
            std::vector<uint8_t> raw(bpe);
            ssize_t got = pread(f.shard_fds[t.shard_idx], raw.data(), bpe, eoff);
            if (got != (ssize_t)bpe) {
                std::cout << "  -> FAILED (short read for " << tn << " expert " << spec.src_expert << ")\n";
                return false;
            }
            OutTensor ot;
            ot.name = "blk.0.ffn_" + std::string(kind) + "." + std::to_string(spec.out_expert) + ".weight";
            ot.type = t.type;
            ot.shape.assign(t.shape.begin(), t.shape.end() - 1);
            ot.data = std::move(raw);
            out_tensors.push_back(std::move(ot));
        }
    }

    // Write a real, minimal, valid GGUF file (0 KV pairs -- this test drives
    // gguf_loader.hpp's tensor-table parsing directly, not the full
    // StreamingEngine/architecture-discovery pipeline, which doesn't need
    // any KV keys to exercise what's under test here).
    std::string out_path = "/tmp/moestream_test_mixed_layer0.gguf";
    {
        FILE* out = fopen(out_path.c_str(), "wb");
        if (!out) { std::cout << "  -> FAILED (cannot write " << out_path << ")\n"; return false; }
        auto w_u32 = [&](uint32_t v) { fwrite(&v, 4, 1, out); };
        auto w_u64 = [&](uint64_t v) { fwrite(&v, 8, 1, out); };
        auto w_str = [&](const std::string& s) { uint64_t n = s.size(); fwrite(&n, 8, 1, out); fwrite(s.data(), 1, n, out); };
        fwrite("GGUF", 1, 4, out);
        w_u32(3);
        w_u64(out_tensors.size());
        w_u64(0);
        const uint64_t ALIGN = 32;
        uint64_t running = 0;
        std::vector<uint64_t> offs;
        for (auto& ot : out_tensors) {
            w_str(ot.name);
            w_u32((uint32_t)ot.shape.size());
            for (auto d : ot.shape) w_u64(d);
            w_u32((uint32_t)ot.type);
            offs.push_back(running);
            w_u64(running);
            running += ot.data.size();
            running = (running + ALIGN - 1) & ~(ALIGN - 1);
        }
        long pos = ftell(out);
        long data_start = (pos + (long)ALIGN - 1) & ~((long)ALIGN - 1);
        while (ftell(out) < data_start) fputc(0, out);
        for (size_t i = 0; i < out_tensors.size(); ++i) {
            fseek(out, data_start + (long)offs[i], SEEK_SET);
            fwrite(out_tensors[i].data.data(), 1, out_tensors[i].data.size(), out);
        }
        fclose(out);
    }

    // Reload through the REAL loader -- this is the actual thing under test.
    GGUFFile mixed;
    if (!mixed.load_header(out_path)) {
        std::cout << "  -> FAILED (gguf_loader.hpp could not parse the hand-built mixed file)\n";
        return false;
    }
    if (mixed.tensors.size() != out_tensors.size()) {
        std::cout << "  -> FAILED (loaded " << mixed.tensors.size() << " tensors, expected " << out_tensors.size() << ")\n";
        return false;
    }

    CudaManager gpu;
    bool have_gpu = gpu.initialize();
    if (!have_gpu) std::cout << "  (no CUDA device -- VRAM dispatch checks skipped, file-read checks still run)\n";

    bool all_ok = true;
    for (auto& [name, t] : mixed.tensors) {
        uint64_t numel = 1;
        for (auto d : t.shape) numel *= d;

        std::vector<uint8_t> raw(t.size_bytes);
        ssize_t got = pread(mixed.shard_fds[t.shard_idx], raw.data(), t.size_bytes, t.offset);
        if (got != (ssize_t)t.size_bytes) {
            std::cout << "  -> FAILED (" << name << ": short read from reloaded file)\n";
            all_ok = false; continue;
        }

        // Round-trip: dequant the reloaded bytes, compare byte-for-byte
        // against the SAME expert's bytes read fresh from its own original
        // source (not just "dequants without crashing" -- proves the
        // write+reload path preserved the data exactly).
        auto& spec = (name.find(".0.weight") != std::string::npos) ? specs[0] : specs[1];
        std::string kind = name.find("gate") != std::string::npos ? "gate" : name.find("up") != std::string::npos ? "up" : "down";
        GGUFFile fsrc;
        bool byte_identical = false;
        if (fsrc.load_header(spec.src_path)) {
            std::string sn = "blk." + std::to_string(spec.src_layer) + ".ffn_" + kind + "_exps.weight";
            auto sit = fsrc.tensors.find(sn);
            if (sit != fsrc.tensors.end()) {
                const GGUFTensor& st = sit->second;
                uint64_t bpe = st.size_bytes / st.shape.back();
                uint64_t soff = st.offset + (uint64_t)spec.src_expert * bpe;
                std::vector<uint8_t> sraw(bpe);
                pread(fsrc.shard_fds[st.shard_idx], sraw.data(), bpe, soff);
                byte_identical = (sraw.size() == raw.size()) && (std::memcmp(sraw.data(), raw.data(), raw.size()) == 0);
            }
        }
        std::vector<float> deq(numel);
        bool dequant_ok = ggml_dequant_row((uint32_t)t.type, raw.data(), deq.data(), (int)numel);
        size_t nan_ct = 0;
        for (auto v : deq) if (std::isnan(v) || std::isinf(v)) nan_ct++;

        std::cout << "  " << name << " type=" << (t.type == GGUFType::IQ1_S ? "IQ1_S" :
                     t.type == GGUFType::IQ2_XXS ? "IQ2_XXS" : t.type == GGUFType::IQ3_XXS ? "IQ3_XXS" : "?")
                   << " round-trip=" << (byte_identical ? "BYTE-IDENTICAL" : "MISMATCH")
                   << " dequant_NaN=" << nan_ct << "/" << numel;
        if (!byte_identical || !dequant_ok || nan_ct > 0) all_ok = false;

        // VRAM push + real fused-kernel dispatch, mirroring Test 5's own
        // offload_tensor_to_vram/dense_matmul_host usage exactly.
        if (have_gpu) {
            QuantizedTensor qt;
            qt.meta.type = t.type;
            qt.meta.shape = t.shape;
            qt.meta.size_bytes = t.size_bytes;
            qt.host_ptr = raw.data();
            int I = (int)t.shape[0], O = (int)t.shape[1];
            if (!gpu.offload_tensor_to_vram(qt)) {
                std::cout << " VRAM=FAIL(offload)\n"; all_ok = false; continue;
            }
            std::mt19937 rng(4321);
            std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
            std::vector<float> x(I);
            for (auto& v : x) v = dist(rng);
            std::vector<float> y_gpu(O, -999.0f);
            bool gpu_ok = gpu.dense_matmul_host(t.type, qt.device_ptr, x.data(), y_gpu.data(), 1, I, O);
            cudaFree(qt.device_ptr);
            if (!gpu_ok) { std::cout << " VRAM=FAIL(matmul)\n"; all_ok = false; continue; }

            int rb = (int)(t.size_bytes / O);
            std::vector<float> wrow(I), y_cpu(O, 0.0f);
            for (int o = 0; o < O; ++o) {
                ggml_dequant_row((uint32_t)t.type, raw.data() + (size_t)o * rb, wrow.data(), I);
                double acc = 0.0;
                for (int i = 0; i < I; ++i) acc += (double)x[i] * (double)wrow[i];
                y_cpu[o] = (float)acc;
            }
            float max_abs = 0, max_rel = 0;
            for (int o = 0; o < O; ++o) {
                float d = std::abs(y_cpu[o] - y_gpu[o]);
                max_abs = std::max(max_abs, d);
                max_rel = std::max(max_rel, d / std::max(1e-6f, std::abs(y_cpu[o])));
            }
            bool vram_ok = (max_abs < 1e-2f) || (max_rel < 0.02f);
            std::cout << " VRAM(I=" << I << ",O=" << O << ")=" << (vram_ok ? "OK" : "SUSPECT")
                       << " max_abs=" << max_abs << " max_rel=" << max_rel;
            if (!vram_ok) all_ok = false;
        }
        std::cout << "\n";
    }

    std::cout << "  -> " << (all_ok ? "PASSED" : "FAILED")
              << " (two different real quant types, loaded side-by-side in one layer, "
                 "one process: file read, dequant round-trip, and GPU dispatch all checked)\n";
    return all_ok;
}

// ------------------------------------------------------------------
// 19. sanity_check_unpacked_format()'s RELAXATION itself, exercised
//     through the real StreamingEngine methods (not a reimplementation of
//     their logic) -- reuses the same hand-built mixed-type file Test 18
//     already wrote, so this test depends on Test 18 having run first in
//     the same process (main() below preserves that order).
//
// Three things checked, all against the real engine.hpp code:
//   (a) sanity_check_unpacked_format() no longer FATAL-refuses a layer
//       whose two experts have different byte sizes (the old behavior,
//       now removed) -- it must return true.
//   (b) it correctly flags that layer in layer_has_mixed_expert_sizes.
//   (c) expert_layer_geometry() -- the "one number represents the whole
//       layer" shortcut every RAM/VRAM-sizing caller still uses, and which
//       has NOT been made per-expert-aware yet -- correctly REFUSES
//       (returns false) for this flagged layer, rather than silently
//       handing back expert 0's size as if it applied to expert 1 too.
//       This is the safety net: Phase 2/3 callers that haven't been
//       updated yet get a clean "not supported for this layer" signal
//       instead of a silent wrong-size bug.
// unpacked_expert_geometry() itself (the real per-expert lookup, which
// Phase 2 work will call directly) is deliberately NOT re-tested here --
// Test 18 already exercises it for both experts via the exact same code
// path, reading back genuinely different, correct byte counts for each.
// ------------------------------------------------------------------
bool test_unpacked_format_relaxed_check() {
    std::cout << "[Test 19] sanity_check_unpacked_format() relaxation (mixed-size layer accepted, "
                 "geometry shortcut refuses)...\n";

    const std::string mixed_path = "/tmp/moestream_test_mixed_layer0.gguf";
    StreamingEngine engine;
    if (!engine.model_file.load_header(mixed_path)) {
        std::cout << "  -> SKIPPED (Test 18's mixed-layer file not found -- did it run first?)\n";
        return true;
    }
    // Test 18's file carries 0 KV pairs by design (see its own comment) --
    // architecture_config is normally populated by graph_initializer.cpp's
    // discovery pass, which this test deliberately does not invoke (out of
    // scope: this test drives sanity_check_unpacked_format()/expert_layer_
    // geometry() directly, not the full model-load pipeline). Set exactly
    // what those two functions read.
    engine.model_file.architecture_config.n_layers = 1;
    engine.model_file.architecture_config.n_experts = 2;
    engine.unpacked_expert_format = true;

    bool all_ok = true;

    bool sanity_ok = engine.sanity_check_unpacked_format();
    std::cout << "  sanity_check_unpacked_format(): " << (sanity_ok ? "true (accepted)" : "false (REFUSED)") << "\n";
    if (!sanity_ok) {
        std::cout << "  -> FAILED (a genuinely mixed-size layer should be ACCEPTED now, not refused -- "
                     "the old hard-fail should be gone)\n";
        all_ok = false;
    }

    bool flagged_mixed = !engine.layer_has_mixed_expert_sizes.empty() && engine.layer_has_mixed_expert_sizes[0];
    std::cout << "  layer_has_mixed_expert_sizes[0]: " << (flagged_mixed ? "true" : "false") << "\n";
    if (!flagged_mixed) {
        std::cout << "  -> FAILED (layer 0 genuinely has two different-sized experts and must be flagged)\n";
        all_ok = false;
    }

    size_t gb, ub, db;
    const GGUFTensor *gM, *uM, *dM;
    bool geometry_refused = !engine.expert_layer_geometry(0, gb, ub, db, gM, uM, dM);
    std::cout << "  expert_layer_geometry(0, ...): " << (geometry_refused ? "false (REFUSED, correct)" : "true (returned a size)") << "\n";
    if (!geometry_refused) {
        std::cout << "  -> FAILED (this shortcut must refuse for a mixed layer -- returning expert 0's "
                     "size here would silently mis-size every RAM/VRAM caller that still assumes one "
                     "size per layer)\n";
        all_ok = false;
    }

    std::cout << "  -> " << (all_ok ? "PASSED" : "FAILED")
              << " (relaxed check accepts real mixed sizes; the layer-level shortcut correctly refuses "
                 "instead of silently mis-sizing)\n";
    return all_ok;
}

// ------------------------------------------------------------------
// 20. Phase 2: compute_expert_slab_bytes() must return the LAYER'S MAX
//     per-expert size for a mixed layer, not just expert 0's.
//
// Deliberately builds a SEPARATE file from Test 18/19's, with the two
// experts' output indices SWAPPED (small IQ1_S expert at out_expert=0,
// larger baseline expert at out_expert=1) -- Test 18/19's own file
// happens to have the larger expert already at index 0, so checking
// compute_expert_slab_bytes against it wouldn't actually distinguish the
// old "expert 0 only" behavior from the new "true max" behavior; both
// would coincidentally return the same number. This ordering is chosen
// specifically so the two behaviors disagree: the old code would return
// expert 0's (SMALLER) IQ1_S total; only the fix returns the true
// (LARGER) max.
// ------------------------------------------------------------------
bool test_compute_expert_slab_bytes_max() {
    std::cout << "[Test 20] compute_expert_slab_bytes() returns the layer's TRUE MAX for a mixed layer...\n";

    const std::string prod_path = "/home/ray/vllm-workspace/GLM-5.3-GGUF-align";
    const std::string iq1s_path = "/home/ray/vllm-workspace/GLM-5.3-GGUF-align/"
        "repacked-GLM-5.3-UD-IQ1_S-00005-of-00006-part00001-of-00001.gguf";

    struct ExpertSpec { std::string src_path; int src_layer; int src_expert; int out_expert; };
    // SWAPPED vs Test 18: small (IQ1_S) expert first, large (baseline) expert second.
    std::vector<ExpertSpec> specs = {
        {iq1s_path, 60, 7, 0},
        {prod_path, 20, 5, 1},
    };

    struct OutTensor { std::string name; GGUFType type; std::vector<uint64_t> shape; std::vector<uint8_t> data; };
    std::vector<OutTensor> out_tensors;
    size_t expert_total[2] = {0, 0};

    for (auto& spec : specs) {
        GGUFFile f;
        if (!f.load_header(spec.src_path)) {
            std::cout << "  -> SKIPPED (real source not available: " << spec.src_path << ")\n";
            return true;
        }
        for (const char* kind : {"gate", "up", "down"}) {
            std::string tn = "blk." + std::to_string(spec.src_layer) + ".ffn_" + kind + "_exps.weight";
            auto it = f.tensors.find(tn);
            if (it == f.tensors.end()) { std::cout << "  -> FAILED (tensor not found: " << tn << ")\n"; return false; }
            const GGUFTensor& t = it->second;
            uint64_t n_experts = t.shape.back();
            uint64_t bpe = t.size_bytes / n_experts;
            uint64_t eoff = t.offset + (uint64_t)spec.src_expert * bpe;
            std::vector<uint8_t> raw(bpe);
            if (pread(f.shard_fds[t.shard_idx], raw.data(), bpe, eoff) != (ssize_t)bpe) {
                std::cout << "  -> FAILED (short read)\n"; return false;
            }
            OutTensor ot;
            ot.name = "blk.0.ffn_" + std::string(kind) + "." + std::to_string(spec.out_expert) + ".weight";
            ot.type = t.type;
            ot.shape.assign(t.shape.begin(), t.shape.end() - 1);
            expert_total[spec.out_expert] += bpe;
            ot.data = std::move(raw);
            out_tensors.push_back(std::move(ot));
        }
    }

    std::string out_path = "/tmp/moestream_test_mixed_layer0_swapped.gguf";
    {
        FILE* out = fopen(out_path.c_str(), "wb");
        if (!out) { std::cout << "  -> FAILED (cannot write " << out_path << ")\n"; return false; }
        auto w_u32 = [&](uint32_t v) { fwrite(&v, 4, 1, out); };
        auto w_u64 = [&](uint64_t v) { fwrite(&v, 8, 1, out); };
        auto w_str = [&](const std::string& s) { uint64_t n = s.size(); fwrite(&n, 8, 1, out); fwrite(s.data(), 1, n, out); };
        fwrite("GGUF", 1, 4, out);
        w_u32(3);
        w_u64(out_tensors.size());
        w_u64(0);
        const uint64_t ALIGN = 32;
        uint64_t running = 0;
        std::vector<uint64_t> offs;
        for (auto& ot : out_tensors) {
            w_str(ot.name);
            w_u32((uint32_t)ot.shape.size());
            for (auto d : ot.shape) w_u64(d);
            w_u32((uint32_t)ot.type);
            offs.push_back(running);
            w_u64(running);
            running += ot.data.size();
            running = (running + ALIGN - 1) & ~(ALIGN - 1);
        }
        long pos = ftell(out);
        long data_start = (pos + (long)ALIGN - 1) & ~((long)ALIGN - 1);
        while (ftell(out) < data_start) fputc(0, out);
        for (size_t i = 0; i < out_tensors.size(); ++i) {
            fseek(out, data_start + (long)offs[i], SEEK_SET);
            fwrite(out_tensors[i].data.data(), 1, out_tensors[i].data.size(), out);
        }
        fclose(out);
    }

    StreamingEngine engine;
    if (!engine.model_file.load_header(out_path)) { std::cout << "  -> FAILED (reload)\n"; return false; }
    engine.model_file.architecture_config.n_layers = 1;
    engine.model_file.architecture_config.n_experts = 2;
    engine.unpacked_expert_format = true;

    size_t expected_max = std::max(expert_total[0], expert_total[1]);
    size_t got = engine.compute_expert_slab_bytes(0);

    std::cout << "  expert 0 (small, IQ1_S) total=" << expert_total[0]
              << "  expert 1 (large, baseline) total=" << expert_total[1]
              << "  expected_max=" << expected_max << "  got=" << got << "\n";

    bool pass = (got == expected_max) && (got != expert_total[0] || expert_total[0] == expert_total[1]);
    if (got == expert_total[0] && expert_total[0] != expected_max) {
        std::cout << "  -> FAILED (returned expert 0's SMALLER size -- the old 'expert 0 only' bug is still present)\n";
        pass = false;
    } else if (got != expected_max) {
        std::cout << "  -> FAILED (returned " << got << ", neither expert's real total matches this)\n";
        pass = false;
    }
    std::cout << "  -> " << (pass ? "PASSED" : "FAILED") << "\n";
    return pass;
}

// ------------------------------------------------------------------
// 20b. Compact VRAM slot planner: variable-sized layer regions must fit the
//     requested byte budget, never exceed the model expert count, and keep
//     every layer with a usable slab represented when the budget permits.
//
// This is deliberately a pure topology test. It does not allocate CUDA
// memory, touch a model file, or start an engine, so it is safe to run in the
// component test target and gives the compact layout a deterministic guard.
// ------------------------------------------------------------------
bool test_compact_vram_slot_planner() {
    std::cout << "[Test 20b] compact VRAM slot planner preserves layer regions within budget...\n";

    const std::vector<size_t> layer_bytes = {10, 20, 10};
    const std::vector<int> counts =
        GpuTopology::compact_layer_slot_counts(100, 4, layer_bytes);
    size_t used = 0;
    bool ok = counts.size() == layer_bytes.size();
    for (size_t l = 0; l < counts.size(); ++l) {
        ok = ok && counts[l] >= 1 && counts[l] <= 4;
        used += (size_t)counts[l] * layer_bytes[l];
    }
    ok = ok && used <= 100 && counts == std::vector<int>({3, 2, 3});

    const std::vector<int> sparse =
        GpuTopology::compact_layer_slot_counts(50, 2, {0, 20, 30});
    ok = ok && sparse == std::vector<int>({0, 1, 1});

    const std::vector<int> full_prefix =
        GpuTopology::compact_layer_slot_counts(100, 4, layer_bytes, 1);
    ok = ok && full_prefix == std::vector<int>({4, 2, 2});

    // Explicit CPU-affinity layers must not strand their physical cache
    // regions. The same 100-byte pool should be redistributed across the
    // two GPU-executed layers rather than leaving layer 1's share unusable.
    const std::vector<int> cpu_affinity_excluded =
        GpuTopology::compact_layer_slot_counts(
            100, 4, layer_bytes, 0, std::vector<int>{1});
    ok = ok && cpu_affinity_excluded == std::vector<int>({4, 0, 4});

    std::cout << "  counts=";
    for (int count : counts) std::cout << count << " ";
    std::cout << "used=" << used << " bytes, full-prefix=";
    for (int count : full_prefix) std::cout << count << " ";
    std::cout << "-> " << (ok ? "PASSED" : "FAILED") << "\n";
    return ok;
}

// ------------------------------------------------------------------
// 20c. RAM source lifetime: an asynchronous H2D caller must receive an
// atomic hold together with its ready-slot lookup. A separate lookup followed
// by ram_pin() leaves a real eviction/regrow window for prefetch workers.
// This stays CPU-only: it exercises the RAM state machine and does not start
// CUDA, allocate a VRAM pool, or read a model file.
// ------------------------------------------------------------------
bool test_ram_async_source_hold() {
    std::cout << "[Test 20c] RAM ready lookup holds an async source slab...\n";

    StreamingEngine engine;
    engine.layers_experts.resize(1);
    engine.layers_experts[0].resize(1);

    RAMSafetyManager ram;
    ram.slots_per_layer = 1;
    // Heap-allocated (not a stack array): StreamingEngine's destructor calls
    // plain free() on any slot.slab_buffer that isn't slab_from_shared_arena
    // (engine.hpp, ~StreamingEngine) -- matching real production ownership,
    // where every slab_buffer is always a malloc/posix_memalign allocation.
    // A stack array here previously made `engine`'s destructor call free()
    // on stack memory at end of scope, corrupting the heap ("double free or
    // corruption (out)") the moment this test function returned -- not a
    // production bug, just this test not giving the engine a real buffer to
    // own. No manual free() needed below; the destructor now does it correctly.
    constexpr size_t kSourceBytes = 16;
    uint8_t *source_bytes = static_cast<uint8_t *>(calloc(1, kSourceBytes));
    MoEExpertBlock &slot = engine.layers_experts[0][0];
    slot.expert_id = 7;
    slot.slab_buffer = source_bytes;
    slot.slab_capacity = kSourceBytes;
    ram.preload_ram_slot(engine, 0, 7, 0);

    const uint8_t *captured = nullptr;
    MoEExpertBlock *held = ram.find_ram_slot_if_ready_held(
        engine.layers_experts[0], 1, 0, 7, &captured, nullptr);
    bool ok = held == &slot && captured == source_bytes &&
              slot.ram_temp_pin_count == 1;
    if (held)
        ram.ram_unpin(*held);

    // Also cover the wait path used when another thread published the slot:
    // pin_on_ready must be taken under the same lock that observed Ready.
    MoEExpertBlock *waited =
        ram.wait_for_ram_slot(engine, 0, 7, /*pin_on_ready=*/true);
    ok = ok && waited == &slot && slot.ram_temp_pin_count == 1;
    if (waited)
        ram.ram_unpin(*waited);

    // The unified Hit path has the same atomic-pinning contract used by
    // get_expert(..., pin_for_caller=true).
    RamClaimResult hit = ram.claim_or_wait(
        engine, 0, 7, RamClaimCaller::MainThread, nullptr,
        /*pin_on_hit=*/true);
    ok = ok && hit.status == RamClaimStatus::Hit && hit.slot == &slot &&
         slot.ram_temp_pin_count == 1;
    if (hit.slot)
        ram.ram_unpin(*hit.slot);

    ok = ok && slot.ram_temp_pin_count == 0;
    std::cout << "  -> " << (ok ? "PASSED" : "FAILED") << "\n";
    return ok;
}

// ------------------------------------------------------------------
// 21. Phase 3 (VRAM): the REAL GpuTopology cache path -- upload_expert +
//     find_expert + a real fused-kernel dispatch -- with two genuinely
//     different-sized real experts sharing ONE fixed-size slot pool.
//
// Unlike Test 18 (which used the simpler, direct offload_tensor_to_vram/
// dense_matmul_host helper Test 5 also uses -- correct, but not the real
// production cache path), this drives the actual GpuTopology::upload_
// expert/find_expert machinery every real decode call goes through.
//
// The real question Phase 3 needed answered: GpuTopology's VRAM pool is
// ONE fixed slot size for the whole pool (reindex_expert_vram_cache,
// offset += expert_slab_bytes per slot) -- does uploading a SMALLER
// expert into a slot sized for the LARGER one actually work correctly,
// or does it read/write the wrong number of bytes? Traced through the
// real source (gpu_topology.cpp) before writing this test: every upload
// path (upload_expert, upload_expert_async, upload_expert_active[_async])
// already computes `copy_bytes = std::min(bytes, slot->size)` -- i.e. it
// was ALREADY built to copy exactly the real caller-supplied byte count
// into a possibly-larger slot, not a fixed slot-sized copy. This predates
// per-expert mixed precision entirely -- it's what already made
// GLM-5.2-style CROSS-layer type heterogeneity (Q2_K vs Q3_K per layer,
// one pool sized to the global max) work. This test confirms the same
// mechanism, unmodified, genuinely extends to WITHIN-layer heterogeneity:
// no VRAM-tier code changes were needed for Phase 3, only real
// verification that the claim holds.
// ------------------------------------------------------------------
bool test_vram_cache_mixed_sizes() {
    std::cout << "[Test 21] Phase 3: real GpuTopology cache (upload_expert/find_expert) with two "
                 "different-sized real experts in one fixed-size slot pool...\n";

    const std::string prod_path = "/home/ray/vllm-workspace/GLM-5.3-GGUF-align";
    const std::string iq1s_path = "/home/ray/vllm-workspace/GLM-5.3-GGUF-align/"
        "repacked-GLM-5.3-UD-IQ1_S-00005-of-00006-part00001-of-00001.gguf";

    struct ExpertSpec { std::string src_path; int src_layer; int src_expert; };
    // expert 0 = large (baseline), expert 1 = small (IQ1_S) -- real sizes,
    // real bytes, same two sources every other test in this file uses.
    std::vector<ExpertSpec> specs = {
        {prod_path, 20, 5},
        {iq1s_path, 60, 7},
    };

    struct RealExpert {
        std::vector<uint8_t> slab;  // gate @ 0, up @ gb, down @ gb+ub -- matches populate_packed_slab's layout
        size_t gb = 0, ub = 0, db = 0;
        GGUFType gtype, utype, dtype;
    };
    std::vector<RealExpert> experts;

    for (auto& spec : specs) {
        GGUFFile f;
        if (!f.load_header(spec.src_path)) {
            std::cout << "  -> SKIPPED (real source not available: " << spec.src_path << ")\n";
            return true;
        }
        RealExpert re;
        GGUFType* types[3] = {&re.gtype, &re.utype, &re.dtype};
        size_t* sizes[3] = {&re.gb, &re.ub, &re.db};
        int ti = 0;
        for (const char* kind : {"gate", "up", "down"}) {
            std::string tn = "blk." + std::to_string(spec.src_layer) + ".ffn_" + kind + "_exps.weight";
            auto it = f.tensors.find(tn);
            if (it == f.tensors.end()) { std::cout << "  -> FAILED (tensor not found: " << tn << ")\n"; return false; }
            const GGUFTensor& t = it->second;
            uint64_t n_experts = t.shape.back();
            uint64_t bpe = t.size_bytes / n_experts;
            uint64_t eoff = t.offset + (uint64_t)spec.src_expert * bpe;
            size_t base = re.slab.size();
            re.slab.resize(base + bpe);
            if (pread(f.shard_fds[t.shard_idx], re.slab.data() + base, bpe, eoff) != (ssize_t)bpe) {
                std::cout << "  -> FAILED (short read)\n"; return false;
            }
            *sizes[ti] = bpe;
            *types[ti] = t.type;
            ++ti;
        }
        experts.push_back(std::move(re));
    }

    size_t slab_bytes[2] = {experts[0].slab.size(), experts[1].slab.size()};
    size_t pool_slot_size = std::max(slab_bytes[0], slab_bytes[1]);
    std::cout << "  expert 0 (baseline) slab=" << slab_bytes[0]
              << "  expert 1 (IQ1_S) slab=" << slab_bytes[1]
              << "  pool_slot_size(=max)=" << pool_slot_size << "\n";

    CudaManager gpu;
    if (!gpu.initialize()) {
        std::cout << "  -> SKIPPED (no CUDA device)\n";
        return true;
    }
    // Force the vault pool onto the SAME GPU dense_matmul_host below
    // targets (CudaManager::device_id, "Core engine GPU" -- can genuinely
    // differ from topology.vault_gpus on a real multi-GPU box, confirmed
    // on this one: upload_expert placed the slot on the real vault GPU
    // while dense_matmul_host's own cudaSetDevice(device_id) targeted the
    // core GPU, an unintended cross-device pointer access -- "illegal
    // memory access", not a bug in upload_expert or dense_matmul_host
    // individually, just this test's simplified single-device helper
    // meeting a genuinely multi-GPU cache placement it wasn't told to
    // match). Production code dispatches multi-GPU-aware; this override
    // keeps the test scoped to Phase 3's real question (does variable-size
    // upload/dispatch work on a fixed-size slot pool) without also having
    // to reimplement cross-GPU dispatch here.
    gpu.topology.vault_gpus = {gpu.device_id};
    // No config passed -- allocate_expert_cache's own documented fallback
    // for a caller without one ("safe default... e.g. tests"): grabs
    // available VRAM, sized to pool_slot_size per slot. n_experts_total=2
    // is just a hint for internal bookkeeping array sizing, not a hard cap.
    gpu.topology.allocate_expert_cache(pool_slot_size, 2, nullptr);

    bool all_ok = true;
    for (int e = 0; e < 2; ++e) {
        ExpertLocation* loc = gpu.topology.upload_expert(0, e, experts[e].slab.data(), experts[e].slab.size());
        if (!loc) { std::cout << "  -> FAILED (upload_expert returned null for expert " << e << ")\n"; return false; }
        if (loc->size != pool_slot_size) {
            std::cout << "  -> FAILED (slot size=" << loc->size << ", expected the pool's fixed " << pool_slot_size << ")\n";
            all_ok = false;
        }

        ExpertLocation* found = gpu.topology.find_expert(0, e);
        if (found != loc) { std::cout << "  -> FAILED (find_expert didn't return the same slot just uploaded)\n"; all_ok = false; }

        // Real dispatch against each projection, at its own real offset
        // WITHIN the slot (gate@0, up@gb, down@gb+ub -- exactly what
        // populate_packed_slab/dispatch_expert_k assume) and its own real
        // type -- the same shape of call the production hot path makes.
        int I_gate = 6144, O_gate = 2048, I_down = 2048, O_down = 6144;
        struct Proj { const char* name; size_t off; size_t bytes; GGUFType type; int I, O; };
        Proj projs[3] = {
            {"gate", 0,                              experts[e].gb, experts[e].gtype, I_gate, O_gate},
            {"up",   experts[e].gb,                  experts[e].ub, experts[e].utype, I_gate, O_gate},
            {"down", experts[e].gb + experts[e].ub,   experts[e].db, experts[e].dtype, I_down, O_down},
        };
        std::mt19937 rng(9999 + e);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (auto& p : projs) {
            std::vector<float> x(p.I);
            for (auto& v : x) v = dist(rng);
            std::vector<float> y_gpu(p.O, -999.0f);
            bool ok = gpu.dense_matmul_host(p.type, loc->device_ptr + p.off, x.data(), y_gpu.data(), 1, p.I, p.O);
            if (!ok) { std::cout << "  -> FAILED (expert " << e << " " << p.name << ": dense_matmul_host failed)\n"; all_ok = false; continue; }

            // uint8_t, not float -- a real bug this test hit on its first
            // real run: declaring this as std::vector<float> still compiles
            // (p.bytes silently becomes an element count, not a byte count)
            // but makes host_bytes.data() + o*rb below stride by o*rb
            // FLOATS (4x too far) instead of o*rb bytes. Row 0 (offset 0
            // either way) came out byte-correct; every later row read from
            // the wrong place -- confirmed via a one-off debug dump showing
            // y_gpu[0]==y_cpu[0] exactly but every later element diverging,
            // which is what pointed at a stride bug specifically rather
            // than a real GPU-side one (a real raw D2H readback, also
            // added and removed during that same debugging pass, had
            // already confirmed the uploaded VRAM bytes were byte-for-byte
            // identical to what was uploaded).
            std::vector<uint8_t> host_bytes(p.bytes);
            std::memcpy(host_bytes.data(), experts[e].slab.data() + p.off, p.bytes);
            int rb = (int)(p.bytes / p.O);
            std::vector<float> wrow(p.I), y_cpu(p.O, 0.0f);
            for (int o = 0; o < p.O; ++o) {
                ggml_dequant_row((uint32_t)p.type, host_bytes.data() + (size_t)o * rb, wrow.data(), p.I);
                double acc = 0.0;
                for (int i = 0; i < p.I; ++i) acc += (double)x[i] * (double)wrow[i];
                y_cpu[o] = (float)acc;
            }
            float max_abs = 0, max_rel = 0;
            for (int o = 0; o < p.O; ++o) {
                float d = std::abs(y_cpu[o] - y_gpu[o]);
                max_abs = std::max(max_abs, d);
                max_rel = std::max(max_rel, d / std::max(1e-6f, std::abs(y_cpu[o])));
            }
            bool proj_ok = (max_abs < 1e-2f) || (max_rel < 0.02f);
            std::cout << "  expert=" << e << " " << p.name << " (in-slot offset=" << p.off << " bytes=" << p.bytes
                      << "): " << (proj_ok ? "OK" : "SUSPECT") << " max_abs=" << max_abs << " max_rel=" << max_rel << "\n";
            if (!proj_ok) all_ok = false;
        }
    }

    std::cout << "  -> " << (all_ok ? "PASSED" : "FAILED")
              << " (real upload_expert/find_expert cache path, two different-sized real experts, "
                 "one fixed-size slot pool, real per-projection dispatch all checked)\n";
    return all_ok;
}

// Test 23: fast, deterministic repro (milliseconds, no model/disk needed) of
// the real compute-sanitizer finding from a live GLM-5.3 generation run --
// 1375 flagged cudaErrorHostMemoryAlreadyRegistered "errors" inside
// upload_expert_async's speculative cudaHostRegister, once
// ram_pinned_budget_gb had persistently pinned most RAM-tier slabs (see
// upload_expert_async's own header comment). Reproduces the EXACT CUDA call
// sequence directly: manually cudaHostRegister a buffer (standing in for
// ensure_slab_capacity's persistent pin), then call the real
// upload_expert_async on it with the fixed, real production call shape
// (src_already_pinned=true) -- the permanent regression guard, always runs,
// expected to be sanitizer-clean.
//
// Set MOESTREAM_REPRO_SANITIZER_ISSUE=1 to ALSO run a second, deliberate
// call with the hint omitted (src_already_pinned=false, the pre-fix shape)
// -- this is what actually reproduces the flagged-but-harmless
// cudaErrorHostMemoryAlreadyRegistered under compute-sanitizer (confirmed:
// `MOESTREAM_REPRO_SANITIZER_ISSUE=1 compute-sanitizer --tool memcheck
// ./moestream_tests` -> exactly 2 errors, both attributed to this call, 0
// anywhere else). Off by default so a normal `compute-sanitizer` pass over
// this whole suite stays a trustworthy "0 errors" signal -- this
// demonstration call is not itself a bug, it's the OLD, still-correct
// fallback shape that any future caller forgetting to pass the hint would
// still safely (if noisily, under sanitizer) fall back to.
bool test_upload_async_already_pinned_src() {
    std::cout << "[Test 23] upload_expert_async + already-cudaHostRegister'd source "
                 "(fast repro of the real compute-sanitizer finding)...\n";

    CudaManager gpu;
    if (!gpu.initialize()) {
        std::cout << "  -> SKIPPED (no CUDA device)\n";
        return true;
    }
    gpu.topology.vault_gpus = {gpu.device_id};
    const size_t buf_bytes = 128 * 1024;  // small and arbitrary -- only the
                                           // cudaHostRegister interaction matters here
    EngineConfig cache_config;
    cache_config.gpus.emplace_back();
    cache_config.gpus.back().device_id = gpu.device_id;
    // This pinning test needs two tiny experts, not the allocator's default
    // whole-card cache. Leave headroom for other native GPU fixtures.
    cache_config.gpus.back().moe_cache_gb = double(4 * buf_bytes) / 1e9;
    gpu.topology.allocate_expert_cache(buf_bytes, 2, &cache_config);

    auto make_registered_buf = [&](uint32_t seed) -> std::vector<uint8_t> {
        std::vector<uint8_t> buf(buf_bytes);
        std::mt19937 rng(seed);
        for (auto& b : buf) b = (uint8_t)(rng() & 0xFF);
        // Simulate ensure_slab_capacity's persistent pin under
        // ram_pinned_budget_gb -- the exact state that makes
        // upload_expert_async's own speculative register call doomed.
        cudaError_t err = cudaHostRegister(buf.data(), buf_bytes, cudaHostRegisterPortable);
        if (err != cudaSuccess) {
            std::cout << "  -> SKIPPED (cudaHostRegister setup failed: "
                      << cudaGetErrorString(err) << ")\n";
            buf.clear();
        }
        return buf;
    };

    auto verify_upload = [&](const char* label, ExpertLocation* loc,
                              const std::vector<uint8_t>& src) -> bool {
        if (!loc) { std::cout << "  -> FAILED (" << label << ": upload_expert_async returned null)\n"; return false; }
        cudaDeviceSynchronize();
        std::vector<uint8_t> readback(buf_bytes);
        cudaError_t err = cudaMemcpy(readback.data(), loc->device_ptr, buf_bytes, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            std::cout << "  -> FAILED (" << label << ": readback cudaMemcpy: " << cudaGetErrorString(err) << ")\n";
            return false;
        }
        bool ok = (readback == src);
        std::cout << "  " << label << ": " << (ok ? "OK (VRAM bytes match)" : "MISMATCH") << "\n";
        return ok;
    };

    bool all_ok = true;
    std::vector<uint8_t> src_a;

    // Deliberate, opt-in-only demonstration of the OLD (still-correct,
    // still-safe) no-hint shape -- this is the one that actually reproduces
    // the flagged sanitizer error, since host_data is already registered by
    // this test above and upload_expert_async will still try
    // cudaHostRegister on it speculatively when not told otherwise.
    if (std::getenv("MOESTREAM_REPRO_SANITIZER_ISSUE")) {
        src_a = make_registered_buf(1001);
        if (src_a.empty()) return true;  // setup skip already logged
        ExpertLocation* loc_a = gpu.topology.upload_expert_async(0, 0, src_a.data(), buf_bytes);
        all_ok &= verify_upload("no-hint call (expect a flagged-but-harmless sanitizer error above)", loc_a, src_a);
    } else {
        std::cout << "  no-hint demonstration: SKIPPED by default (set MOESTREAM_REPRO_SANITIZER_ISSUE=1 "
                     "to reproduce the flagged-but-harmless sanitizer error this test's fix removes)\n";
    }

    // The permanent regression guard: real production call shape after the
    // fix -- src_already_pinned=true. Always runs; expected sanitizer-clean.
    std::vector<uint8_t> src_b = make_registered_buf(2002);
    if (src_b.empty()) return true;
    ExpertLocation* loc_b = gpu.topology.upload_expert_async(0, 1, src_b.data(), buf_bytes, -1, /*src_already_pinned=*/true);
    all_ok &= verify_upload("with-hint call (expect zero sanitizer errors)", loc_b, src_b);

    if (!src_a.empty()) cudaHostUnregister(src_a.data());
    cudaHostUnregister(src_b.data());

    std::cout << "  -> " << (all_ok ? "PASSED" : "FAILED")
              << " (both calls uploaded correct bytes; compare a compute-sanitizer log around "
                 "this test's two calls to confirm the hint actually removes the flagged error)\n";
    return all_ok;
}






// 2026-09-04: GPU-side routed-expert accumulation (cuda/expert_accum.cu).
// An end-to-end output diff CANNOT validate this: the engine is already
// non-deterministic run-to-run (cache-state-dependent dispatch), AND the
// feature legitimately changes summation order (device-accumulated experts are
// summed first, then added onto CPU-dispatched ones). So this compares the
// kernel directly against the exact CPU loop it replaces, on fixed data.
//
// Shapes chosen so a stride/index bug cannot alias into a pass: n != D, tokens
// deliberately REPEATED across experts (so the accumulate really accumulates
// rather than just scattering), one token left untouched by any expert (must
// stay exactly zero), and non-uniform weights.
bool test_expert_gpu_accum() {
    std::cout << "[Test 26] MoE GPU-side expert accumulation (device scatter-add vs CPU loop)...\n";
    CudaManager gpu;
    if (!gpu.initialize()) {
        std::cout << "  -> Skipped (No CUDA Device Found on system)\n";
        return true;
    }
    const int S = 7, D = 20, n_experts = 3;
    // pairs: (expert, token, weight) -- token 5 is touched by NO expert
    const std::vector<std::vector<int>> ex_tok = {{0, 2, 3}, {1, 2}, {4, 6, 0}};
    std::mt19937 rng(777);
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);

    std::vector<int> tok; std::vector<float> w;
    for (auto& e : ex_tok) for (int t : e) { tok.push_back(t); w.push_back(d(rng)); }
    const int total_pairs = (int)tok.size();

    // per-expert output rows
    std::vector<std::vector<float>> eout;
    for (auto& e : ex_tok) { std::vector<float> v((size_t)e.size() * D); for (auto& x : v) x = d(rng); eout.push_back(v); }

    // CPU reference: exactly the loop this replaces
    std::vector<float> ref((size_t)S * D, 0.0f);
    { int off = 0;
      for (size_t e = 0; e < ex_tok.size(); ++e) {
        for (size_t i = 0; i < ex_tok[e].size(); ++i) {
            const int s = tok[off + i]; const float wt = w[off + i];
            for (int c = 0; c < D; ++c) ref[(size_t)s * D + c] += wt * eout[e][i * D + c];
        }
        off += (int)ex_tok[e].size();
      } }

    std::vector<float> got((size_t)S * D, 0.0f);
    if (!gpu.expert_accum_begin(0, S, D, tok.data(), w.data(), total_pairs)) {
        std::cout << "  -> FAILED: expert_accum_begin returned false\n"; return false;
    }
    int off = 0;
    for (size_t e = 0; e < ex_tok.size(); ++e) {
        const int n = (int)ex_tok[e].size();
        float* d_src = nullptr;
        if (cudaMalloc((void**)&d_src, (size_t)n * D * sizeof(float)) != cudaSuccess) {
            std::cout << "  -> FAILED: cudaMalloc\n"; return false;
        }
        cudaMemcpy(d_src, eout[e].data(), (size_t)n * D * sizeof(float), cudaMemcpyHostToDevice);
        if (!gpu.expert_accum_add(0, d_src, off, n, D, S)) {
            std::cout << "  -> FAILED: expert_accum_add\n"; cudaFree(d_src); return false;
        }
        cudaDeviceSynchronize();
        cudaFree(d_src);
        off += n;
    }
    if (!gpu.expert_accum_end(0, got.data(), S, D)) {
        std::cout << "  -> FAILED: expert_accum_end returned false\n"; return false;
    }

    double max_abs = 0.0, refmax = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) { max_abs = std::max(max_abs, (double)std::fabs(got[i]-ref[i]));
                                              refmax = std::max(refmax, (double)std::fabs(ref[i])); }
    // token 5 must be exactly untouched
    double untouched = 0.0;
    for (int c = 0; c < D; ++c) untouched = std::max(untouched, (double)std::fabs(got[(size_t)5*D + c]));
    std::cout << "  -> max_abs=" << max_abs << " normalized=" << (max_abs/(refmax+1e-9))
              << " untouched_row_max=" << untouched << "\n";
    if (untouched != 0.0) { std::cout << "  -> FAILED: a row no expert routed to was written\n"; return false; }
    if (!(max_abs/(refmax+1e-9) < 1e-6)) { std::cout << "  -> FAILED: diverges from the CPU loop\n"; return false; }
    std::cout << "  -> PASSED\n";
    return true;
}


// 2026-09-05: dense BF16 prefill GEMM extended from Q8_0-only to Q5_K/Q6_K.
// Test 5 does NOT cover this path: dense_bf16_prefill_gemm() is a config-set
// global that defaults to false in the test binary, so Test 5's Q5_K/Q6_K
// cases go through the fused kernels instead (visible as ~1e-6 error, i.e.
// F32 precision, where BF16 would show ~1e-2). This calls the BF16 host
// function DIRECTLY so the new dequant kernels are actually exercised.
//
// Tolerance is deliberately BF16-appropriate: 8 mantissa bits gives ~0.4%
// per-element, and the dot product accumulates over I terms, so this is judged
// on max_abs/|ref|max (the normalized GEMM metric) rather than per-element
// relative error -- same reasoning as the lane8 and qwen4exp-attn tests.
bool test_dense_bf16_kquant() {
    std::cout << "[Test 27] Dense BF16 prefill GEMM for Q5_K / Q6_K (new dequant kernels)...\n";
    CudaManager gpu;
    if (!gpu.initialize()) {
        std::cout << "  -> Skipped (No CUDA Device Found on system)\n";
        return true;
    }
    struct Case { GGUFType t; const char* name; int I, O, S; };
    const Case cases[] = {
        {GGUFType::Q5_K, "Q5_K", 512,  8, 3},
        {GGUFType::Q6_K, "Q6_K", 512,  8, 3},
        {GGUFType::Q5_K, "Q5_K", 2560, 16, 4},   // real dense shape order for qwen4exp
        {GGUFType::Q6_K, "Q6_K", 2560, 16, 4},
    };
    bool all = true;
    for (const auto& c : cases) {
        std::mt19937 rng(9001);
        std::uniform_real_distribution<float> d(-1.0f, 1.0f);
        std::vector<float> Wf((size_t)c.O * c.I), x((size_t)c.S * c.I);
        for (auto& v : Wf) v = d(rng);
        for (auto& v : x)  v = d(rng);

        // quantize each row, then dequantize back for an EXACT reference --
        // comparing against the original f32 would fold in the quantizer's own
        // loss and tell us nothing about the BF16 GEMM.
        const size_t rb = row_bytes(c.t, c.I);
        std::vector<uint8_t> Wq((size_t)c.O * rb);
        std::vector<float> Wdq((size_t)c.O * c.I);
        for (int o = 0; o < c.O; ++o) {
            if (!ggml_quantize_row_ref((uint32_t)c.t, Wf.data() + (size_t)o * c.I, Wq.data() + (size_t)o * rb, c.I)) {
                std::cout << "  -> " << c.name << ": SKIPPED (no ref quantizer)\n"; goto next_case;
            }
            ggml_dequant_row((uint32_t)c.t, Wq.data() + (size_t)o * rb, Wdq.data() + (size_t)o * c.I, c.I);
        }
        {
            std::vector<double> ref((size_t)c.S * c.O, 0.0);
            for (int si = 0; si < c.S; ++si)
                for (int o = 0; o < c.O; ++o) {
                    double acc = 0.0;
                    for (int i = 0; i < c.I; ++i) acc += (double)x[(size_t)si*c.I+i] * Wdq[(size_t)o*c.I+i];
                    ref[(size_t)si*c.O+o] = acc;
                }
            void* d_W = nullptr;
            if (cudaMalloc(&d_W, Wq.size()) != cudaSuccess) { std::cout << "  -> cudaMalloc failed\n"; return false; }
            cudaMemcpy(d_W, Wq.data(), Wq.size(), cudaMemcpyHostToDevice);
            std::vector<float> got((size_t)c.S * c.O, -1e30f);
            const bool ok = gpu.dense_matmul_bf16_host(c.t, d_W, x.data(), got.data(), c.S, c.I, c.O);
            cudaFree(d_W);
            if (!ok) { std::cout << "  -> " << c.name << ": FAILED (dense_matmul_bf16_host returned false)\n"; all = false; goto next_case; }
            double mx = 0.0, refmax = 0.0;
            for (size_t i = 0; i < ref.size(); ++i) {
                mx = std::max(mx, std::fabs((double)got[i] - ref[i]));
                refmax = std::max(refmax, std::fabs(ref[i]));
            }
            const double norm = mx / (refmax + 1e-9);
            std::cout << "  -> " << c.name << " (I=" << c.I << " O=" << c.O << " S=" << c.S
                      << "): max_abs=" << mx << " normalized=" << norm
                      << (norm < 2e-2 ? "  PASSED" : "  FAILED") << "\n";
            if (!(norm < 2e-2)) all = false;
        }
        next_case:;
    }
    return all;
}

// 2026-09-05: GDN causal conv1d+SiLU GPU path (cuda/gdn_conv1d.cu,
// EngineConfig::qwen4exp_gdn_conv_gpu). Compares the new GPU kernel
// directly against attn_causal_conv1d_silu (the already-proven CPU
// reference, itself already fixed once for cache-locality -- see CLAUDE.md's
// "conv1d loop order: 29.4x" entry) rather than an independent from-scratch
// implementation, so a shared conceptual bug in both can't produce a false
// pass. Checks BOTH the conv+SiLU output AND the post-call history update
// (the GPU path does the history tail on the host, byte-identical to the
// CPU function's own tail -- this catches a divergence there too, not just
// in the kernel itself).
//
// Two cases: S=37 (deliberately not a multiple of channels/kernel, to catch
// a stride/index bug that a round S could hide) exercises the common
// S>=hist path; S=2 < hist=3 exercises the history-update tail's OTHER
// branch (new history partly comes from OLD history, not just from x) --
// a real edge case (a short window-slid prefill chunk) the S=37 case alone
// cannot reach.
bool test_gdn_conv1d_gpu() {
    std::cout << "[Test 28] GDN causal conv1d+SiLU GPU path (vs CPU reference)...\n";
    CudaManager gpu;
    if (!gpu.initialize()) {
        std::cout << "  -> Skipped (No CUDA Device Found on system)\n";
        return true;
    }
    const int channels = 130, kernel = 4;
    const int hist = kernel - 1;
    std::mt19937 rng(9090);
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);

    auto run_case = [&](int S, const char* label) -> bool {
        std::vector<float> x((size_t)S * channels), w((size_t)channels * kernel);
        std::vector<float> hist0((size_t)channels * hist);
        for (auto& v : x) v = d(rng);
        for (auto& v : w) v = d(rng);
        for (auto& v : hist0) v = d(rng);

        std::vector<float> hist_cpu = hist0, hist_gpu = hist0;
        std::vector<float> out_cpu((size_t)S * channels, -999.0f);
        std::vector<float> out_gpu((size_t)S * channels, -777.0f);

        attn_causal_conv1d_silu(x.data(), w.data(), out_cpu.data(), S, channels, kernel,
                                hist_cpu.data());
        if (!gpu.qwen4exp_conv1d_silu_gpu_host(x.data(), w.data(), out_gpu.data(), S,
                                               channels, kernel, hist_gpu.data())) {
            std::cout << "  -> " << label << ": FAILED (qwen4exp_conv1d_silu_gpu_host returned false)\n";
            return false;
        }

        double o_abs = 0.0, o_refmax = 0.0, h_abs = 0.0, h_refmax = 0.0;
        for (size_t i = 0; i < out_cpu.size(); ++i) {
            o_abs = std::max(o_abs, (double)std::fabs(out_gpu[i] - out_cpu[i]));
            o_refmax = std::max(o_refmax, (double)std::fabs(out_cpu[i]));
        }
        for (size_t i = 0; i < hist_cpu.size(); ++i) {
            h_abs = std::max(h_abs, (double)std::fabs(hist_gpu[i] - hist_cpu[i]));
            h_refmax = std::max(h_refmax, (double)std::fabs(hist_cpu[i]));
        }
        const double o_rel = o_abs / (o_refmax + 1e-9);
        const double h_rel = h_abs / (h_refmax + 1e-9);
        std::cout << "  -> " << label << " (S=" << S << " channels=" << channels
                  << " kernel=" << kernel << "): out max_abs=" << o_abs << " normalized=" << o_rel
                  << " | history max_abs=" << h_abs << " normalized=" << h_rel << "\n";
        if (!(o_rel < 1e-4 && h_rel < 1e-4)) {
            std::cout << "  -> " << label << ": FAILED -- GPU path diverges from the CPU reference\n";
            return false;
        }
        return true;
    };

    bool ok = true;
    ok &= run_case(37, "S >= hist");
    ok &= run_case(2, "S < hist");
    std::cout << (ok ? "  -> PASSED\n" : "  -> at least one case FAILED\n");
    return ok;
}

// 2026-09-05: GgmlGdnProjRunner::run_with_conv (the conv1d round-trip
// elimination -- chains the causal conv1d + SiLU directly onto the qkv
// tensor the 4-projection-fusion graph already produces, instead of
// downloading qkv then re-uploading the identical bytes for a separate
// conv1d call). Compares run_with_conv's FULL output (conv_out, gate,
// g_log, beta, AND the updated history) against the composition of two
// already-proven-correct pieces: run() (the existing, unfused-conv 4-
// projection graph) followed by attn_causal_conv1d_silu (the CPU reference
// Test 28 already validated) -- not an independent from-scratch
// implementation, so a shared conceptual bug in run_with_conv's ggml-graph
// transpose/concat layout can't produce a false pass.
bool test_gdn_conv_fused() {
    std::cout << "[Test 29] GDN proj+conv1d fused GPU path (run_with_conv vs run()+CPU conv)...\n";
    // conv_dim MUST be a multiple of 128 -- ggml-cuda's ssm_conv kernel
    // (ggml-cuda/ssm-conv.cu) hardcodes a 128-thread block and hard-asserts
    // this (GGML_ASSERT, not a graceful failure) if violated. Found by this
    // test itself; ensure_conv_weight() now checks it and declines instead
    // of crashing for a non-conforming model (see that function's comment).
    const int D = 24, conv_dim = 128, value_dim = 16, H_v = 4, d_conv = 4;
    const int hist = d_conv - 1;
    GgmlGdnProjRunner &r = GgmlGdnProjRunner::instance();
    if (!r.ensure(0, D, conv_dim, value_dim, H_v)) {
        std::cout << "  -> Skipped (No CUDA Device Found on system, or ensure() failed)\n";
        return true;
    }
    const int layer = 0;
    std::mt19937 rng(31337);
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);

    std::vector<float> w_qkv((size_t)D * conv_dim), w_gate((size_t)D * value_dim),
        w_alpha((size_t)D * H_v), w_beta((size_t)D * H_v),
        dt_bias(H_v), ssm_a(H_v), w_conv((size_t)conv_dim * d_conv);
    for (auto &v : w_qkv) v = d(rng);
    for (auto &v : w_gate) v = d(rng);
    for (auto &v : w_alpha) v = d(rng);
    for (auto &v : w_beta) v = d(rng);
    for (auto &v : dt_bias) v = d(rng);
    for (auto &v : ssm_a) v = -std::fabs(d(rng));  // real -A_log.exp()-style: negative decay
    for (auto &v : w_conv) v = d(rng);

    auto ref = [&](const std::vector<float> &v, int64_t ne0, int64_t ne1) {
        GgmlGdnProjWeightRef out;
        out.host_data = v.data();
        out.type = 0;  // GGML_TYPE_F32
        out.ne0 = ne0;
        out.ne1 = ne1;
        out.nbytes = v.size() * sizeof(float);
        return out;
    };
    if (!r.ensure_weights(layer, ref(w_qkv, D, conv_dim), ref(w_gate, D, value_dim),
                          ref(w_alpha, D, H_v), ref(w_beta, D, H_v),
                          dt_bias.data(), ssm_a.data())) {
        std::cout << "  -> FAILED: ensure_weights returned false\n";
        return false;
    }
    if (!r.ensure_conv_weight(layer, ref(w_conv, d_conv, conv_dim))) {
        std::cout << "  -> FAILED: ensure_conv_weight returned false\n";
        return false;
    }

    auto run_case = [&](int S, const char *label) -> bool {
        std::vector<float> hidden((size_t)S * D), hist0((size_t)conv_dim * hist);
        for (auto &v : hidden) v = d(rng);
        for (auto &v : hist0) v = d(rng);

        // Reference: unfused proj (run()) + the already-proven CPU conv.
        std::vector<float> qkv_ref((size_t)S * conv_dim), gate_ref((size_t)S * value_dim),
            glog_ref((size_t)S * H_v), beta_ref((size_t)S * H_v);
        if (!r.run(layer, hidden.data(), S, qkv_ref.data(), gate_ref.data(),
                   glog_ref.data(), beta_ref.data())) {
            std::cout << "  -> " << label << ": FAILED (run() returned false)\n";
            return false;
        }
        std::vector<float> hist_ref = hist0;
        std::vector<float> conv_ref((size_t)S * conv_dim, -999.0f);
        attn_causal_conv1d_silu(qkv_ref.data(), w_conv.data(), conv_ref.data(), S,
                                conv_dim, d_conv, hist_ref.data());

        // Test: the fully-fused path.
        std::vector<float> gate_test((size_t)S * value_dim), glog_test((size_t)S * H_v),
            beta_test((size_t)S * H_v), conv_test((size_t)S * conv_dim, -777.0f);
        std::vector<float> hist_test = hist0;
        if (!r.run_with_conv(layer, hidden.data(), S, conv_test.data(), gate_test.data(),
                             glog_test.data(), beta_test.data(), hist_test.data(), d_conv)) {
            std::cout << "  -> " << label << ": FAILED (run_with_conv returned false)\n";
            return false;
        }

        auto cmp = [](const std::vector<float> &a, const std::vector<float> &b) {
            double m = 0.0, ref = 0.0;
            for (size_t i = 0; i < a.size(); ++i) {
                m = std::max(m, (double)std::fabs(a[i] - b[i]));
                ref = std::max(ref, (double)std::fabs(b[i]));
            }
            return m / (ref + 1e-9);
        };
        const double conv_rel = cmp(conv_test, conv_ref);
        const double gate_rel = cmp(gate_test, gate_ref);
        const double glog_rel = cmp(glog_test, glog_ref);
        const double beta_rel = cmp(beta_test, beta_ref);
        const double hist_rel = cmp(hist_test, hist_ref);
        std::cout << "  -> " << label << " (S=" << S << "): conv=" << conv_rel
                  << " gate=" << gate_rel << " g_log=" << glog_rel
                  << " beta=" << beta_rel << " history=" << hist_rel << "\n";
        if (!(conv_rel < 1e-4 && gate_rel < 1e-4 && glog_rel < 1e-4 &&
              beta_rel < 1e-4 && hist_rel < 1e-4)) {
            std::cout << "  -> " << label << ": FAILED -- fused path diverges from run()+CPU-conv\n";
            return false;
        }
        return true;
    };

    bool ok = true;
    ok &= run_case(29, "S >= hist");
    ok &= run_case(2, "S < hist");
    std::cout << (ok ? "  -> PASSED\n" : "  -> at least one case FAILED\n");
    return ok;
}

// Test 35: GgmlQwenSharedRunner (moe_ggml_shared_expert, EXPERIMENTAL, kept
// OFF in every shipped config -- see configs/qwen4exp_3090_only.toml's own
// comment). Written 2026-09-10 to isolate a real, confirmed correctness bug
// found on real hardware: 100% NaN corruption end to end (hidden_states
// 1249280/1249280 NaN, logits 248320/248320 NaN), traced via VERBOSE tracing
// to layer=46 of 48 specifically -- finite input, NaN router weights with a
// suspicious exact stride-4 expert-ID pattern (e292,e288,e284,...), pointing
// at a buffer-aliasing/overwrite bug rather than the missing-sync race first
// suspected (that race was real too -- see run()'s own ggml_backend_synchronize
// fix -- but re-testing after it still aborts under trace_abort_on_nan).
// This test calls ensure_layer()+run() across MANY synthetic "layers" in a
// loop (mimicking the real model's per-layer weight-swap pattern) checking
// EVERY iteration against a hand-computed CPU oracle, specifically to see
// whether a similar "breaks after N calls" pattern reproduces here -- a
// compute-sanitizer run against the real 48-layer model timed out entirely
// inside startup preload before reaching layer 46 (impractical: the model's
// own preload dominates runtime even before compute-sanitizer's overhead is
// added), so this minimal harness is the fast path to isolating it.
bool test_qwen_shared_expert_runner() {
    std::cout << "[Test 35] Qwen4Exp shared-expert GGML runner (moe_ggml_shared_expert, vs CPU oracle)...\n";
    if (!GgmlMoeRunner::compiled_in()) {
        std::cout << "  -> Skipped (GGML CUDA backend not compiled)\n";
        return true;
    }
    CudaManager gpu;
    if (!gpu.initialize()) {
        std::cout << "  -> Skipped (No CUDA Device Found on system)\n";
        return true;
    }
    const int device = 0;
    const int D = 16, inter = 12, S = 5;
    const int n_synthetic_layers = 50;  // mirrors the real model's 48 + margin
    std::mt19937 rng(90210);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    auto sigmoid = [](float v) { return 1.0f / (1.0f + std::exp(-v)); };
    auto silu = [&](float v) { return v * sigmoid(v); };

    auto &runner = GgmlQwenSharedRunner::instance();
    bool all_ok = true;
    int first_bad_layer = -1;
    double worst_rel = 0.0;

    for (int layer = 0; layer < n_synthetic_layers && all_ok; ++layer) {
        std::vector<float> w_gate((size_t)inter * D), w_up((size_t)inter * D),
            w_down((size_t)D * inter), w_gate_inp((size_t)D);
        for (auto &v : w_gate) v = dist(rng);
        for (auto &v : w_up) v = dist(rng);
        for (auto &v : w_down) v = dist(rng);
        for (auto &v : w_gate_inp) v = dist(rng);
        std::vector<float> x((size_t)D * S), routed((size_t)D * S);
        for (auto &v : x) v = dist(rng);
        for (auto &v : routed) v = dist(rng);

        // CPU oracle, matching build_graph()'s exact op sequence:
        //   gate = W_gate @ x ; up = W_up @ x ; act = silu(gate) * up
        //   shared = W_down @ act ; gate_scalar = sigmoid(W_gate_inp @ x)
        //   combined = routed + shared * gate_scalar
        std::vector<float> gate_ref((size_t)inter * S), up_ref((size_t)inter * S),
            act_ref((size_t)inter * S), shared_ref((size_t)D * S),
            gate_scalar_ref((size_t)S), combined_ref((size_t)D * S);
        for (int t = 0; t < S; ++t) {
            for (int o = 0; o < inter; ++o) {
                double g = 0.0, u = 0.0;
                for (int d = 0; d < D; ++d) {
                    g += (double)w_gate[(size_t)o * D + d] * x[(size_t)t * D + d];
                    u += (double)w_up[(size_t)o * D + d] * x[(size_t)t * D + d];
                }
                gate_ref[(size_t)o * S + t] = (float)g;
                up_ref[(size_t)o * S + t] = (float)u;
                act_ref[(size_t)o * S + t] = silu((float)g) * (float)u;
            }
            double gs = 0.0;
            for (int d = 0; d < D; ++d)
                gs += (double)w_gate_inp[d] * x[(size_t)t * D + d];
            gate_scalar_ref[t] = sigmoid((float)gs);
            for (int o = 0; o < D; ++o) {
                double s = 0.0;
                for (int i = 0; i < inter; ++i)
                    s += (double)w_down[(size_t)o * inter + i] * act_ref[(size_t)i * S + t];
                shared_ref[(size_t)o * S + t] = (float)s;
                combined_ref[(size_t)t * D + o] =
                    routed[(size_t)t * D + o] +
                    (float)s * gate_scalar_ref[t];
            }
        }

        void *x_tensor = nullptr, *x_ctx = nullptr, *x_buf = nullptr;
        void *routed_tensor = nullptr, *routed_ctx = nullptr, *routed_buf = nullptr;
        if (!ggml_moe_test_make_device_tensor(device, D, S, x.data(), &x_tensor,
                                              &x_ctx, &x_buf) ||
            !ggml_moe_test_make_device_tensor(device, D, S, routed.data(),
                                              &routed_tensor, &routed_ctx,
                                              &routed_buf)) {
            std::cout << "  -> layer=" << layer << ": FAILED (device tensor setup)\n";
            all_ok = false;
            break;
        }

        auto make_shared_ref = [](const void *data, int64_t ne0, int64_t ne1,
                                  size_t nbytes) {
            GgmlSharedWeightRef ref;
            ref.host_data = data;
            ref.type = 0;  // GGML_TYPE_F32
            ref.ne0 = ne0;
            ref.ne1 = ne1;
            ref.nbytes = nbytes;
            return ref;
        };
        GgmlSharedWeightRef gate_ref_w =
            make_shared_ref(w_gate.data(), D, inter,
                            w_gate.size() * sizeof(float));
        GgmlSharedWeightRef up_ref_w =
            make_shared_ref(w_up.data(), D, inter,
                            w_up.size() * sizeof(float));
        GgmlSharedWeightRef down_ref_w =
            make_shared_ref(w_down.data(), inter, D,
                            w_down.size() * sizeof(float));
        GgmlSharedWeightRef gate_inp_ref_w =
            make_shared_ref(w_gate_inp.data(), D, 1,
                            w_gate_inp.size() * sizeof(float));

        const void *combined_output = nullptr;
        bool step_ok =
            runner.ensure(device, D, inter, S, /*has_gate=*/true, /*clamp=*/0.0f) &&
            runner.ensure_layer(layer, gate_ref_w, up_ref_w, down_ref_w, &gate_inp_ref_w) &&
            runner.run(layer, x_tensor, routed_tensor, &combined_output);

        std::vector<float> combined_test((size_t)D * S, -999.0f);
        if (step_ok)
            step_ok = ggml_moe_test_download_tensor(device, combined_output,
                                                     combined_test.data(),
                                                     combined_test.size());

        ggml_moe_test_free_device_tensor(x_ctx, x_buf);
        ggml_moe_test_free_device_tensor(routed_ctx, routed_buf);

        if (!step_ok) {
            std::cout << "  -> layer=" << layer << ": FAILED (runner call returned false)\n";
            all_ok = false;
            break;
        }

        double m = 0.0, ref_max = 0.0;
        bool has_nan = false;
        for (size_t i = 0; i < combined_test.size(); ++i) {
            if (!std::isfinite(combined_test[i])) has_nan = true;
            m = std::max(m, (double)std::fabs(combined_test[i] - combined_ref[i]));
            ref_max = std::max(ref_max, (double)std::fabs(combined_ref[i]));
        }
        const double rel = m / (ref_max + 1e-9);
        worst_rel = std::max(worst_rel, rel);
        if (has_nan || rel > 1e-3) {
            if (first_bad_layer < 0) first_bad_layer = layer;
            std::cout << "  -> layer=" << layer << ": FAILED -- rel=" << rel
                      << (has_nan ? " (NaN/Inf present)" : "") << "\n";
            if (layer == 0) {
                std::cout << "     first 8 test:  ";
                for (int i = 0; i < 8 && i < (int)combined_test.size(); ++i)
                    std::cout << combined_test[i] << " ";
                std::cout << "\n     first 8 ref:   ";
                for (int i = 0; i < 8 && i < (int)combined_ref.size(); ++i)
                    std::cout << combined_ref[i] << " ";
                std::cout << "\n     first 8 routed:";
                for (int i = 0; i < 8 && i < (int)routed.size(); ++i)
                    std::cout << " " << routed[i];
                std::cout << "\n     first 8 shared_ref:";
                for (int i = 0; i < 8 && i < (int)shared_ref.size(); ++i)
                    std::cout << " " << shared_ref[i];
                std::cout << "\n";
            }
            all_ok = false;
        }
    }

    if (all_ok) {
        std::cout << "  -> " << n_synthetic_layers << " synthetic layers, worst rel="
                  << worst_rel << "\n  -> PASSED\n";
    } else {
        std::cout << "  -> first bad layer=" << first_bad_layer << " of "
                  << n_synthetic_layers << "\n";
    }
    return all_ok;
}

// Test 36: Qwen4Exp HC pre-mix graph against the reference operation order.
// This specifically covers the S==1 decode shape and the flattened
// [D,HC,S] -> [HC*D,S] view used by the production device-HC handoff.
bool test_qwen_hc_runner() {
    std::cout << "[Test 36] Qwen4Exp HC GGML pre-mix (S=1/decode shape, vs CPU oracle)...\n";
    if (!GgmlHcRunner::compiled_in()) {
        std::cout << "  -> Skipped (GGML CUDA backend not compiled)\n";
        return true;
    }
    CudaManager gpu;
    if (!gpu.initialize()) {
        std::cout << "  -> Skipped (No CUDA Device Found on system)\n";
        return true;
    }

    const int device = 0;
    const int D = 8;
    const int HC = 2;
    const int LR = 3;
    const int S = 1;
    const int I = HC * D;
    const float eps = 1.0e-6f;
    std::mt19937 rng(3617);
    std::uniform_real_distribution<float> dist(-0.08f, 0.08f);

    std::vector<float> norm((size_t)I), down((size_t)I * LR),
        up((size_t)LR * I), inject((size_t)I * HC),
        streams((size_t)S * I);
    for (float &v : norm) v = 0.8f + dist(rng);
    for (float &v : down) v = dist(rng);
    for (float &v : up) v = dist(rng);
    for (float &v : inject) v = dist(rng);
    for (float &v : streams) v = dist(rng);

    auto ref = [](const void *data, uint32_t type, int64_t ne0, int64_t ne1,
                  size_t bytes) {
        GgmlHcWeightRef r;
        r.host_data = data;
        r.type = type;
        r.ne0 = ne0;
        r.ne1 = ne1;
        r.nbytes = bytes;
        return r;
    };
    const uint32_t f32 = 0;  // GGML_TYPE_F32
    const GgmlHcWeightRef norm_ref =
        ref(norm.data(), f32, I, 1, norm.size() * sizeof(float));
    const GgmlHcWeightRef down_ref =
        ref(down.data(), f32, I, LR, down.size() * sizeof(float));
    const GgmlHcWeightRef up_ref =
        ref(up.data(), f32, LR, I, up.size() * sizeof(float));
    const GgmlHcWeightRef inject_ref =
        ref(inject.data(), f32, I, HC, inject.size() * sizeof(float));

    auto sigmoid = [](float x) {
        return 1.0f / (1.0f + std::exp(-x));
    };
    auto silu = [&](float x) { return x * sigmoid(x); };
    std::vector<float> xn((size_t)S * I), lo((size_t)S * LR),
        gate((size_t)S * I), expected((size_t)S * D),
        expected_inject((size_t)S * HC);
    for (int s = 0; s < S; ++s) {
        for (int h = 0; h < HC; ++h) {
            const float *src = streams.data() + (size_t)s * I + (size_t)h * D;
            double ss = 0.0;
            for (int d = 0; d < D; ++d) ss += (double)src[d] * src[d];
            const float inv = 1.0f / std::sqrt((float)(ss / D) + eps);
            for (int d = 0; d < D; ++d)
                xn[(size_t)s * I + (size_t)h * D + d] =
                    src[d] * inv * norm[(size_t)h * D + d];
        }
        for (int j = 0; j < LR; ++j) {
            double sum = 0.0;
            for (int i = 0; i < I; ++i)
                sum += (double)down[(size_t)j * I + i] * xn[(size_t)s * I + i];
            lo[(size_t)s * LR + j] = silu((float)(sum / HC));
        }
        for (int k = 0; k < I; ++k) {
            double sum = 0.0;
            for (int j = 0; j < LR; ++j)
                sum += (double)up[(size_t)k * LR + j] * lo[(size_t)s * LR + j];
            gate[(size_t)s * I + k] = sigmoid((float)sum);
        }
        for (int d = 0; d < D; ++d) {
            double sum = 0.0;
            for (int h = 0; h < HC; ++h)
                sum += (double)xn[(size_t)s * I + (size_t)h * D + d] *
                       gate[(size_t)s * I + (size_t)h * D + d];
            expected[(size_t)s * D + d] = (float)(sum / HC);
        }
        for (int h = 0; h < HC; ++h) {
            double sum = 0.0;
            for (int i = 0; i < I; ++i)
                sum += (double)inject[(size_t)h * I + i] * xn[(size_t)s * I + i];
            expected_inject[(size_t)s * HC + h] = (float)sum;
        }
    }

    auto &runner = GgmlHcRunner::instance();
    bool ok = runner.ensure(device, D, HC, LR, eps) &&
              runner.ensure_weights(0, 1, norm_ref, down_ref, up_ref,
                                    inject_ref) &&
              runner.upload_streams(streams.data(), S);
    std::vector<float> actual((size_t)S * D), actual_inject((size_t)S * HC);
    if (ok)
        ok = runner.pre_mix(0, 1, S, actual.data(), nullptr) &&
             runner.download_inject(actual_inject.data(), S);
    runner.shutdown();

    double max_abs = 0.0;
    double max_inject_abs = 0.0;
    double ref_scale = 0.0;
    for (size_t i = 0; i < actual.size(); ++i) {
        max_abs = std::max(max_abs, (double)std::abs(actual[i] - expected[i]));
        ref_scale = std::max(ref_scale, (double)std::abs(expected[i]));
    }
    for (size_t i = 0; i < actual_inject.size(); ++i)
        max_inject_abs = std::max(
            max_inject_abs,
            (double)std::abs(actual_inject[i] - expected_inject[i]));
    const double rel = max_abs / (ref_scale + 1.0e-6);
    ok = ok && std::isfinite(rel) && rel < 2.0e-3 && max_inject_abs < 2.0e-3;
    std::cout << "  -> max_hidden_abs=" << max_abs
              << " max_hidden_rel=" << rel
              << " max_inject_abs=" << max_inject_abs << "\n";
    std::cout << (ok ? "  -> PASSED\n" : "  -> FAILED\n");
    return ok;
}

// Test 37: the Qwen device-HC escape hatch for Q8_0 HC projections.  GGML's
// normal CUDA Q8_0 matmul quantizes an F32 activation to Q8_1; the exact path
// dequantizes the small HC weights once and uses an F32 graph instead.  This
// test keeps the quantized source bytes, computes the oracle from those same
// dequantized bytes, and verifies the opt-in ref flag selects the exact path.
bool test_qwen_hc_runner_q8_exact() {
    std::cout << "[Test 37] Qwen4Exp HC exact Q8_0 down/up projections (vs oracle)...\n";
    if (!GgmlHcRunner::compiled_in()) {
        std::cout << "  -> Skipped (GGML CUDA backend not compiled)\n";
        return true;
    }
    CudaManager gpu;
    if (!gpu.initialize()) {
        std::cout << "  -> Skipped (No CUDA Device Found on system)\n";
        return true;
    }

    const int device = 0;
    const int D = 32;
    const int HC = 2;
    const int LR = 32;
    const int S = 1;
    const int I = HC * D;
    const float eps = 1.0e-6f;
    const uint32_t q8 = (uint32_t)GGUFType::Q8_0;
    std::mt19937 rng(3701);
    std::uniform_real_distribution<float> dist(-0.08f, 0.08f);

    std::vector<float> norm((size_t)I), down((size_t)I * LR),
        up((size_t)LR * I), inject((size_t)I * HC), streams((size_t)S * I);
    for (float &v : norm) v = 0.8f + dist(rng);
    for (float &v : down) v = dist(rng);
    for (float &v : up) v = dist(rng);
    for (float &v : inject) v = dist(rng);
    for (float &v : streams) v = dist(rng);

    auto quantize_rows = [&](const std::vector<float> &src, int ne0, int ne1,
                             std::vector<unsigned char> &qbytes,
                             std::vector<float> &dequant) -> bool {
        const size_t row_bytes = (size_t)(ne0 / 32) * 34;
        qbytes.resize(row_bytes * (size_t)ne1);
        dequant.resize((size_t)ne0 * ne1);
        for (int row = 0; row < ne1; ++row) {
            if (!ggml_quantize_row_ref(
                    q8, src.data() + (size_t)row * ne0,
                    qbytes.data() + (size_t)row * row_bytes, ne0) ||
                !ggml_dequant_row(
                    q8, qbytes.data() + (size_t)row * row_bytes,
                    dequant.data() + (size_t)row * ne0, ne0))
                return false;
        }
        return true;
    };
    std::vector<unsigned char> down_q8, up_q8;
    std::vector<float> down_f32, up_f32;
    bool ok = quantize_rows(down, I, LR, down_q8, down_f32) &&
              quantize_rows(up, LR, I, up_q8, up_f32);
    if (!ok) {
        std::cout << "  -> FAILED (Q8_0 reference quantizer unavailable)\n";
        return false;
    }

    auto make_ref = [](const void *data, uint32_t type, int64_t ne0,
                       int64_t ne1, size_t bytes, bool exact_f32_q8) {
        GgmlHcWeightRef r;
        r.host_data = data;
        r.type = type;
        r.ne0 = ne0;
        r.ne1 = ne1;
        r.nbytes = bytes;
        r.exact_f32_q8 = exact_f32_q8;
        return r;
    };
    const GgmlHcWeightRef norm_ref =
        make_ref(norm.data(), 0, I, 1, norm.size() * sizeof(float), false);
    const GgmlHcWeightRef down_ref =
        make_ref(down_q8.data(), q8, I, LR, down_q8.size(), true);
    const GgmlHcWeightRef up_ref =
        make_ref(up_q8.data(), q8, LR, I, up_q8.size(), true);
    const GgmlHcWeightRef inject_ref =
        make_ref(inject.data(), 0, I, HC, inject.size() * sizeof(float), false);

    auto sigmoid = [](float x) {
        return 1.0f / (1.0f + std::exp(-x));
    };
    auto silu = [&](float x) { return x * sigmoid(x); };
    std::vector<float> xn((size_t)S * I), lo((size_t)S * LR),
        gate((size_t)S * I), expected((size_t)S * D),
        expected_inject((size_t)S * HC);
    for (int s = 0; s < S; ++s) {
        for (int h = 0; h < HC; ++h) {
            const float *src = streams.data() + (size_t)s * I + (size_t)h * D;
            double ss = 0.0;
            for (int d = 0; d < D; ++d) ss += (double)src[d] * src[d];
            const float inv = 1.0f / std::sqrt((float)(ss / D) + eps);
            for (int d = 0; d < D; ++d)
                xn[(size_t)s * I + (size_t)h * D + d] =
                    src[d] * inv * norm[(size_t)h * D + d];
        }
        for (int j = 0; j < LR; ++j) {
            double sum = 0.0;
            for (int i = 0; i < I; ++i)
                sum += (double)down_f32[(size_t)j * I + i] *
                       xn[(size_t)s * I + i];
            lo[(size_t)s * LR + j] = silu((float)(sum / HC));
        }
        for (int k = 0; k < I; ++k) {
            double sum = 0.0;
            for (int j = 0; j < LR; ++j)
                sum += (double)up_f32[(size_t)k * LR + j] *
                       lo[(size_t)s * LR + j];
            gate[(size_t)s * I + k] = sigmoid((float)sum);
        }
        for (int d = 0; d < D; ++d) {
            double sum = 0.0;
            for (int h = 0; h < HC; ++h)
                sum += (double)xn[(size_t)s * I + (size_t)h * D + d] *
                       gate[(size_t)s * I + (size_t)h * D + d];
            expected[(size_t)s * D + d] = (float)(sum / HC);
        }
        for (int h = 0; h < HC; ++h) {
            double sum = 0.0;
            for (int i = 0; i < I; ++i)
                sum += (double)inject[(size_t)h * I + i] *
                       xn[(size_t)s * I + i];
            expected_inject[(size_t)s * HC + h] = (float)sum;
        }
    }

    auto &runner = GgmlHcRunner::instance();
    ok = runner.ensure(device, D, HC, LR, eps) &&
         runner.ensure_weights(0, 1, norm_ref, down_ref, up_ref, inject_ref) &&
         runner.upload_streams(streams.data(), S);
    std::vector<float> actual((size_t)S * D), actual_inject((size_t)S * HC);
    if (ok)
        ok = runner.pre_mix(0, 1, S, actual.data(), nullptr) &&
             runner.download_inject(actual_inject.data(), S);
    runner.shutdown();

    double max_abs = 0.0, max_inject_abs = 0.0, scale = 0.0;
    for (size_t i = 0; i < actual.size(); ++i) {
        max_abs = std::max(max_abs, (double)std::abs(actual[i] - expected[i]));
        scale = std::max(scale, (double)std::abs(expected[i]));
    }
    for (size_t i = 0; i < actual_inject.size(); ++i)
        max_inject_abs = std::max(
            max_inject_abs,
            (double)std::abs(actual_inject[i] - expected_inject[i]));
    const double rel = max_abs / (scale + 1.0e-6);
    ok = ok && std::isfinite(rel) && rel < 2.0e-3 && max_inject_abs < 2.0e-3;
    std::cout << "  -> max_hidden_abs=" << max_abs
              << " max_hidden_rel=" << rel
              << " max_inject_abs=" << max_inject_abs << "\n";
    std::cout << (ok ? "  -> PASSED\n" : "  -> FAILED\n");
    return ok;
}

// Test 34: GLM5NEXT KDA projection graph. This is an opt-in production
// optimization, so the test uses small F32 matrices and compares every graph
// output with the existing CPU matmul oracle. It also exercises the f_a->f_b
// and g_a->g_b dependency branches rather than testing only the three shared
// input projections.
bool test_glm5next_dense_projection_fusion() {
    std::cout << "[Test 34] GLM5NEXT KDA dense projection fusion (vs CPU oracle)...\n";
    if (!GgmlGlm5nextKdaProjRunner::compiled_in()) {
        std::cout << "  -> Skipped (GGML CUDA backend not compiled)\n";
        return true;
    }
    CudaManager gpu;
    if (!gpu.initialize()) {
        std::cout << "  -> Skipped (No CUDA Device Found on system)\n";
        return true;
    }
    const int S = 3, D = 5, inner = 8, hd = 4, H = 2;
    std::mt19937 rng(5340);
    std::uniform_real_distribution<float> dist(-0.6f, 0.6f);
    std::vector<float> x((size_t)S * D);
    std::vector<float> wq((size_t)inner * D), wk(wq.size()), wv(wq.size());
    std::vector<float> wfa((size_t)hd * D), wfb((size_t)inner * hd);
    std::vector<float> wbeta((size_t)H * D), wga((size_t)hd * D),
        wgb((size_t)inner * hd);
    for (auto* values : {&x, &wq, &wk, &wv, &wfa, &wfb, &wbeta, &wga, &wgb})
        for (float& value : *values) value = dist(rng);

    auto ref = [&](const std::vector<float>& input, const std::vector<float>& w,
                   int input_dim, int output_dim) {
        std::vector<float> out((size_t)S * output_dim, 0.0f);
        matmul(out.data(), input.data(), w.data(), S, input_dim, output_dim);
        return out;
    };
    const std::vector<float> q_ref = ref(x, wq, D, inner);
    const std::vector<float> k_ref = ref(x, wk, D, inner);
    const std::vector<float> v_ref = ref(x, wv, D, inner);
    const std::vector<float> fa_ref = ref(x, wfa, D, hd);
    const std::vector<float> fb_ref = ref(fa_ref, wfb, hd, inner);
    const std::vector<float> beta_ref = ref(x, wbeta, D, H);
    const std::vector<float> ga_ref = ref(x, wga, D, hd);
    const std::vector<float> gb_ref = ref(ga_ref, wgb, hd, inner);

    auto make_ref = [](const std::vector<float>& w, int I, int O) {
        GgmlGlm5nextProjWeightRef r;
        r.host_data = w.data();
        r.type = (uint32_t)GGUFType::F32;
        r.ne0 = I;
        r.ne1 = O;
        r.nbytes = w.size() * sizeof(float);
        return r;
    };
    auto& runner = GgmlGlm5nextKdaProjRunner::instance();
    bool ok = runner.ensure(gpu.device_id, D, inner, hd, H, S);
    ok = ok && runner.ensure_layer(0, make_ref(wq, D, inner),
                                   make_ref(wk, D, inner),
                                   make_ref(wv, D, inner),
                                   make_ref(wfa, D, hd),
                                   make_ref(wfb, hd, inner),
                                   make_ref(wbeta, D, H),
                                   make_ref(wga, D, hd),
                                   make_ref(wgb, hd, inner));
    std::vector<float> q(q_ref.size()), k(k_ref.size()), v(v_ref.size()),
        fb(fb_ref.size()), beta(beta_ref.size()), gb(gb_ref.size());
    ok = ok && runner.run(0, x.data(), S, q.data(), k.data(), v.data(),
                          fb.data(), beta.data(), gb.data());
    auto max_err = [](const std::vector<float>& got,
                      const std::vector<float>& expected) {
        double e = 0.0;
        for (size_t i = 0; i < got.size(); ++i)
            e = std::max(e, (double)std::fabs(got[i] - expected[i]));
        return e;
    };
    const double eq = max_err(q, q_ref), ek = max_err(k, k_ref),
                 ev = max_err(v, v_ref), efb = max_err(fb, fb_ref),
                 ebeta = max_err(beta, beta_ref), egb = max_err(gb, gb_ref);
    std::cout << "  -> max_abs q=" << eq << " k=" << ek << " v=" << ev
              << " f_b=" << efb << " beta=" << ebeta << " g_b=" << egb
              << (ok && eq < 1e-4 && ek < 1e-4 && ev < 1e-4 && efb < 1e-4 &&
                          ebeta < 1e-4 && egb < 1e-4
                      ? " PASSED\n"
                      : " FAILED\n");
    runner.shutdown();
    return ok && eq < 1e-4 && ek < 1e-4 && ev < 1e-4 && efb < 1e-4 &&
           ebeta < 1e-4 && egb < 1e-4;
}

// Test 30: GLM5NEXT KDA recurrent delta rule on CUDA, compared with an
// independent scalar CPU oracle. The dimensions intentionally do not match
// a warp size and the initial state is non-zero, so state layout, per-key
// decay, token carry-over, and output scaling are all exercised.
bool test_glm5next_kda_gpu() {
    std::cout << "[Test 30] GLM5NEXT KDA delta-rule GPU path (vs CPU oracle)...\n";
    CudaManager gpu;
    if (!gpu.initialize()) {
        std::cout << "  -> Skipped (No CUDA Device Found on system)\n";
        return true;
    }

    const int S = 5, H = 3, hd = 7;
    std::mt19937 rng(5300);
    std::uniform_real_distribution<float> value(-0.7f, 0.7f);
    std::uniform_real_distribution<float> beta_dist(0.05f, 0.95f);
    std::uniform_real_distribution<float> decay_dist(-0.8f, -0.02f);
    std::vector<float> Q((size_t)S * H * hd), K(Q.size()), V(Q.size());
    std::vector<float> beta((size_t)S * H), g_log(Q.size());
    for (auto& x : Q) x = value(rng);
    for (auto& x : K) x = value(rng);
    for (auto& x : V) x = value(rng);
    for (auto& x : beta) x = beta_dist(rng);
    for (auto& x : g_log) x = decay_dist(rng);

    std::vector<float> state0((size_t)H * hd * hd);
    for (auto& x : state0) x = value(rng) * 0.2f;
    std::vector<float> state_cpu = state0;
    std::vector<float> out_cpu((size_t)S * H * hd, 0.0f);
    const float scale = 1.0f / std::sqrt((float)hd);
    std::vector<float> sk(hd);
    for (int s = 0; s < S; ++s) {
        for (int h = 0; h < H; ++h) {
            float* st = state_cpu.data() + (size_t)h * hd * hd;
            const float* q = Q.data() + ((size_t)s * H + h) * hd;
            const float* k = K.data() + ((size_t)s * H + h) * hd;
            const float* v = V.data() + ((size_t)s * H + h) * hd;
            const float* g = g_log.data() + ((size_t)s * H + h) * hd;
            const float b = beta[(size_t)s * H + h];
            for (int key = 0; key < hd; ++key) {
                float* row = st + (size_t)key * hd;
                const float decay = std::exp(g[key]);
                for (int out = 0; out < hd; ++out) row[out] *= decay;
            }
            std::fill(sk.begin(), sk.end(), 0.0f);
            for (int key = 0; key < hd; ++key) {
                const float* row = st + (size_t)key * hd;
                for (int out = 0; out < hd; ++out) sk[out] += row[out] * k[key];
            }
            for (int key = 0; key < hd; ++key) {
                float* row = st + (size_t)key * hd;
                for (int out = 0; out < hd; ++out)
                    row[out] += k[key] * (v[out] - sk[out]) * b;
            }
            float* out = out_cpu.data() + ((size_t)s * H + h) * hd;
            for (int out_i = 0; out_i < hd; ++out_i) {
                float sum = 0.0f;
                for (int key = 0; key < hd; ++key)
                    sum += st[(size_t)key * hd + out_i] * q[key] * scale;
                out[out_i] = sum;
            }
        }
    }

    std::vector<float> state_gpu = state0;
    std::vector<float> out_gpu(out_cpu.size(), -999.0f);
    if (!gpu.glm5next_kda_delta_rule_gpu_host(
            Q.data(), K.data(), V.data(), beta.data(), g_log.data(),
            state_gpu.data(), out_gpu.data(), S, H, hd)) {
        std::cout << "  -> FAILED: glm5next_kda_delta_rule_gpu_host returned false\n";
        return false;
    }
    auto max_normalized = [](const std::vector<float>& got,
                             const std::vector<float>& ref) {
        double max_abs = 0.0, ref_abs = 0.0;
        for (size_t i = 0; i < got.size(); ++i) {
            max_abs = std::max(max_abs, (double)std::fabs(got[i] - ref[i]));
            ref_abs = std::max(ref_abs, (double)std::fabs(ref[i]));
        }
        return max_abs / (ref_abs + 1e-9);
    };
    const double out_err = max_normalized(out_gpu, out_cpu);
    const double state_err = max_normalized(state_gpu, state_cpu);
    std::cout << "  -> out normalized=" << out_err
              << " state normalized=" << state_err << "\n";
    const bool ok = out_err < 2e-4 && state_err < 2e-4;
    std::cout << (ok ? "  -> PASSED\n" : "  -> FAILED\n");

    // The production path must carry state across TWO separate calls without
    // requiring a state D2H after the first one. Use one additional token as
    // an independent continuation oracle and explicitly synchronize only at
    // the end, matching the prompt-cache boundary contract.
    auto cpu_one = [&](std::vector<float>& st, std::vector<float>& out_one) {
        out_one.assign((size_t)H * hd, 0.0f);
        for (int h = 0; h < H; ++h) {
            float* sh = st.data() + (size_t)h * hd * hd;
            const float* q = Q.data() + (size_t)h * hd;
            const float* k = K.data() + (size_t)h * hd;
            const float* v = V.data() + (size_t)h * hd;
            const float* g = g_log.data() + (size_t)h * hd;
            const float b = beta[(size_t)h];
            for (int key = 0; key < hd; ++key) {
                float* row = sh + (size_t)key * hd;
                const float decay = std::exp(g[key]);
                for (int value = 0; value < hd; ++value) row[value] *= decay;
            }
            std::vector<float> sk_one((size_t)hd, 0.0f);
            for (int key = 0; key < hd; ++key) {
                const float* row = sh + (size_t)key * hd;
                for (int value = 0; value < hd; ++value) sk_one[value] += row[value] * k[key];
            }
            for (int key = 0; key < hd; ++key) {
                float* row = sh + (size_t)key * hd;
                for (int value = 0; value < hd; ++value)
                    row[value] += k[key] * (v[value] - sk_one[value]) * b;
            }
            float* out_h = out_one.data() + (size_t)h * hd;
            for (int value = 0; value < hd; ++value) {
                for (int key = 0; key < hd; ++key)
                    out_h[value] += sh[(size_t)key * hd + value] * q[key] * scale;
            }
        }
    };
    std::vector<float> state_cpu_cont = state_cpu;
    std::vector<float> out_cpu_cont;
    cpu_one(state_cpu_cont, out_cpu_cont);
    std::vector<float> state_persistent = state0;
    std::vector<float> out_persistent(out_cpu.size(), -999.0f);
    std::vector<float> out_persistent_cont;
    const bool persistent_first = gpu.glm5next_kda_delta_rule_gpu_device(
        Q.data(), K.data(), V.data(), beta.data(), g_log.data(),
        state_persistent.data(), out_persistent.data(), 0, S, H, hd, true);
    bool persistent_ok = persistent_first;
    if (persistent_first) {
        // The continuation is deliberately a separate S=1 call. Its output
        // is sized here before passing the pointer to the device bridge.
        out_persistent_cont.assign((size_t)H * hd, -999.0f);
        persistent_ok = gpu.glm5next_kda_delta_rule_gpu_device(
            Q.data(), K.data(), V.data(), beta.data(), g_log.data(),
            state_persistent.data(), out_persistent_cont.data(), 0, 1, H, hd, false);
    }
    std::vector<float> state_synced = state0;
    if (persistent_ok) {
        persistent_ok = gpu.glm5next_kda_sync_state_to_host(
            0, state_synced.data(), state_synced.size());
    }
    const double persistent_first_err = max_normalized(out_persistent, out_cpu);
    const double persistent_cont_err = persistent_ok
        ? max_normalized(out_persistent_cont, out_cpu_cont) : 1.0;
    const double persistent_state_err = persistent_ok
        ? max_normalized(state_synced, state_cpu_cont) : 1.0;
    persistent_ok = persistent_ok && persistent_first_err < 2e-4 &&
                    persistent_cont_err < 2e-4 && persistent_state_err < 2e-4;
    std::cout << "  -> persistent state: first=" << persistent_first_err
              << " continuation=" << persistent_cont_err
              << " synced_state=" << persistent_state_err
              << (persistent_ok ? " PASSED\n" : " FAILED\n");
    return ok && persistent_ok;
}

// Test 31: GLM5NEXT native router end-to-end (device matmul plus CUDA
// gating/top-k) against a scalar reference. This deliberately uses an F32
// router tensor so it tests the host bridge independently of the model's
// quantized expert format.
bool test_glm5next_router_gpu() {
    std::cout << "[Test 31] GLM5NEXT native router GPU path (matmul + top-k)...\n";
    CudaManager gpu;
    if (!gpu.initialize()) {
        std::cout << "  -> Skipped (No CUDA Device Found on system)\n";
        return true;
    }
    const int S = 3, D = 9, E = 11, K = 4;
    std::mt19937 rng(5310);
    std::uniform_real_distribution<float> dist(-0.8f, 0.8f);
    std::vector<float> w((size_t)E * D), x((size_t)S * D), bias(E);
    for (auto& v : w) v = dist(rng);
    for (auto& v : x) v = dist(rng);
    for (auto& v : bias) v = dist(rng) * 0.15f;

    QuantizedTensor router;
    router.meta.type = GGUFType::F32;
    router.meta.shape = {(uint64_t)D, (uint64_t)E};
    router.meta.size_bytes = w.size() * sizeof(float);
    router.host_ptr = reinterpret_cast<uint8_t*>(w.data());
    if (!gpu.offload_tensor_to_vram(router)) {
        std::cout << "  -> FAILED: could not offload test router weight\n";
        return false;
    }

    constexpr float weight_scale = 1.7f;
    std::vector<int32_t> ids((size_t)S * K, -1);
    std::vector<float> weights((size_t)S * K, -1.0f);
    if (!gpu.glm5next_router_gpu_host(
            router, x.data(), bias.data(), S, D, E, K,
            GatingFunc::Softmax, true, weight_scale,
            ids.data(), weights.data())) {
        std::cout << "  -> FAILED: glm5next_router_gpu_host returned false\n";
        cudaFree(router.device_ptr);
        router.device_ptr = nullptr;
        return false;
    }

    std::vector<int32_t> ref_ids((size_t)S * K, -1);
    std::vector<float> ref_weights((size_t)S * K, 0.0f);
    for (int s = 0; s < S; ++s) {
        std::vector<float> probs(E);
        float max_logit = -std::numeric_limits<float>::infinity();
        for (int e = 0; e < E; ++e) {
            float z = 0.0f;
            for (int d = 0; d < D; ++d) z += x[(size_t)s * D + d] * w[(size_t)e * D + d];
            probs[e] = z;
            max_logit = std::max(max_logit, z);
        }
        float sum = 0.0f;
        for (float& p : probs) { p = std::exp(p - max_logit); sum += p; }
        for (float& p : probs) p /= sum;
        std::vector<int> candidates(E);
        std::iota(candidates.begin(), candidates.end(), 0);
        std::partial_sort(candidates.begin(), candidates.begin() + K, candidates.end(),
                          [&](int a, int b) {
                              const float sa = probs[a] + bias[a];
                              const float sb = probs[b] + bias[b];
                              return sa != sb ? sa > sb : a < b;
                          });
        float selected_sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            ref_ids[(size_t)s * K + k] = candidates[k];
            ref_weights[(size_t)s * K + k] = probs[candidates[k]];
            selected_sum += ref_weights[(size_t)s * K + k];
        }
        for (int k = 0; k < K; ++k)
            ref_weights[(size_t)s * K + k] =
                ref_weights[(size_t)s * K + k] / selected_sum * weight_scale;
    }

    double max_diff = 0.0;
    bool ok = true;
    for (size_t i = 0; i < ids.size(); ++i) {
        if (ids[i] != ref_ids[i]) ok = false;
        max_diff = std::max(max_diff, (double)std::fabs(weights[i] - ref_weights[i]));
    }
    std::cout << "  -> IDs " << (ok ? "match" : "MISMATCH")
              << ", max_weight_diff=" << max_diff << "\n";
    ok = ok && max_diff < 3e-4;
    cudaFree(router.device_ptr);
    router.device_ptr = nullptr;
    std::cout << (ok ? "  -> PASSED\n" : "  -> FAILED\n");
    return ok;
}

// Test 32: a config allow-list must prevent an omitted physical adapter from
// becoming an ExpertVault. This is intentionally a topology-only test: it
// creates streams but no model, RAM tier, or expert cache allocation.
bool test_gpu_device_allowlist() {
    std::cout << "[Test 32] GPU config allow-list (omitted adapters stay unused)...\n";
    CudaManager gpu;
    gpu.topology.configured_device_ids = {0};
    if (!gpu.initialize()) {
        std::cout << "  -> Skipped (No CUDA Device Found on system)\n";
        return true;
    }
    bool ok = gpu.topology.core_gpu >= 0 &&
              gpu.topology.devices[(size_t)gpu.topology.core_gpu].id == 0 &&
              gpu.topology.vault_gpus.size() == 1 &&
              gpu.topology.vault_gpus[0] == gpu.topology.core_gpu;
    for (const auto& dev : gpu.topology.devices) {
        if (dev.id == 0) {
            ok = ok && dev.role == GpuRole::CoreEngine;
        } else {
            ok = ok && dev.role == GpuRole::Unused;
        }
    }
    std::cout << "  -> " << (ok ? "PASSED" : "FAILED")
              << " (configured IDs={0}, physical devices="
              << gpu.topology.devices.size() << ")\n";
    return ok;
}

bool test_qwen_prefill_state_handoff_config() {
    std::cout << "[Test] Qwen prefill state handoff config...\n";
    char config_path[] = "/tmp/moestream_state_handoff_config_XXXXXX";
    const int config_fd = mkstemp(config_path);
    if (config_fd < 0)
        return false;
    close(config_fd);
    bool ok = !EngineConfig{}.qwen4exp_prefill_state_handoff;
    for (const std::string& setting : {std::string(),
                                     std::string("qwen4exp_prefill_state_handoff = true\n"),
                                     std::string("qwen4exp_prefill_state_handoff = false\n")}) {
        {
            std::ofstream output(config_path);
            output << "model_path = \"/test/model.gguf\"\n" << setting;
            ok = ok && output.good();
        }
        EngineConfig config;
        const bool loaded = config.load(config_path);
        ok = ok && loaded &&
             config.qwen4exp_prefill_state_handoff ==
                 (setting.find("true") != std::string::npos);
    }
    EngineConfig::write_sample(config_path);
    EngineConfig sample;
    ok = sample.load(config_path) && !sample.qwen4exp_prefill_state_handoff && ok;
    std::remove(config_path);
    std::cout << "  -> " << (ok ? "PASSED" : "FAILED") << "\n";
  return ok;
}

bool test_exl3_ggml_graph_capture_config() {
    std::cout << "[Test] EXL3 GGML graph-capture config...\n";
    char config_path[] = "/tmp/moestream_exl3_graph_capture_config_XXXXXX";
    const int config_fd = mkstemp(config_path);
    if (config_fd < 0)
        return false;
    close(config_fd);
    const auto write_config = [&](const std::string &extra) {
        std::ofstream output(config_path);
        output << "model_path = \"/test/model.gguf\"\n"
                  << extra
                  << "[[gpus]]\n"
                     "device_id = 0\n"
                     "name = \"test\"\n"
                     "role = \"dense+moe\"\n"
                     "vram_gb = 24.0\n"
                     "dense_gb = 4.0\n"
                     "kv_cache_gb = 2.0\n"
                     "moe_cache_gb = 8.0\n";
        return output.good();
    };
    bool ok = write_config("");
    EngineConfig defaults;
    ok = ok && defaults.load(config_path) &&
         !defaults.exl3_ggml_graph_capture;
    ok = ok && write_config("exl3_ggml_graph_capture = true\n");
    EngineConfig invalid;
    ok = ok && invalid.load(config_path) && !invalid.validate();
    ok = ok && write_config(
                  "exl3_ggml_bridge_model_path = \"/test/exl3\"\n"
                  "exl3_ggml_bridge_layer = 0\n"
                  "exl3_ggml_graph_capture = true\n"
                  "exl3_gpu_cache_gb = 1.0\n"
                  "exl3_cpu_expert_cache_gb = 0.0\n");
    EngineConfig enabled;
    ok = ok && enabled.load(config_path) && enabled.validate() &&
         enabled.exl3_ggml_graph_capture;
    std::remove(config_path);
    std::cout << "  -> " << (ok ? "PASSED" : "FAILED") << "\n";
    return ok;
}

bool test_exl3_cpu_dispatch() {
    std::cout << "[Test] EXL3 CPU dispatch and reconstruction...\n";
    constexpr int width = 128;
    std::vector<uint16_t> signs(width, fp32_to_fp16(1.0f));
    std::vector<float> input(width, 1.0f), output(width), weight(width * width);
    for (int bits : {4, 5, 6}) {
        std::vector<uint16_t> trellis(8 * 8 * 16 * bits, 0);
        Exl3Mul1MatrixView matrix{trellis.data(), signs.data(), signs.data(),
                                  nullptr, width, width, bits};
        const auto packed = exl3_mul1_pack_matrix(matrix);
        if (!cpu_quant_matmul(GGUFType::EXL3, packed.data(), input.data(),
                               output.data(), 1, width, width) ||
            !exl3_mul1_reconstruct(matrix, weight.data())) return false;
        const float expected = width * fp16_to_fp32(fp32_to_fp16(
            std::fma(1024.0f, fp16_to_fp32(0x1eee), fp16_to_fp32(0xc931))));
        if (std::abs(output[0] - expected) > 1e-3f) return false;
        for (int column = 1; column < width; ++column)
            if (std::abs(output[column]) > 1e-3f) return false;
        for (int column = 0; column < width; ++column) {
            float total = 0.0f;
            for (int row = 0; row < width; ++row)
                total += weight[row * width + column];
            if (std::abs(output[column] - total) > 1e-3f) return false;
        }
    }
    std::cout << "  -> PASSED\n";
    return true;
}

bool test_exl3_cpu_matrix_cache() {
    std::cout << "[Test] EXL3 DDR matrix cache LRU and held entries...\n";
    const auto matrix = [](uint8_t fill) {
        Exl3Mul1MatrixData value;
        value.trellis.assign(16, fill);
        value.suh.assign(8, fill);
        value.svh.assign(8, fill);
        value.input_features = 128;
        value.output_features = 128;
        value.bits = 4;
        return value;
    };
    Exl3CpuMatrixCache cache(64);
    auto held = cache.insert("expert-a.gate_proj", matrix(1));
    cache.insert("expert-b.gate_proj", matrix(2));
    auto hit = cache.get("expert-a.gate_proj");
    if (!hit || hit->trellis.front() != 1 ||
        cache.stats().resident_bytes != 64)
        return false;
    cache.insert("expert-c.gate_proj", matrix(3));
    if (cache.get("expert-b.gate_proj") ||
        held->trellis.front() != 1 || hit->trellis.front() != 1)
      return false;
    const auto stats = cache.stats();
    const bool passed = stats.evictions == 1 && stats.resident_bytes == 64 &&
                        stats.hits == 1 && stats.misses == 1;
    std::cout << "  " << (passed ? "PASSED" : "FAILED") << "\n";
    return passed;
}

bool test_exl3_gpu_hc(int device) {
    constexpr int width = 37, streams = 3, rank = 17, rows = 3;
    constexpr int expanded = width * streams;
    std::mt19937 random(8001);
    std::uniform_real_distribution<float> distribution(-0.5f, 0.5f);
    std::vector<float> down(expanded * rank), up(rank * expanded), inject(expanded * streams);
    std::vector<float> normed(rows * expanded);
    for (auto *values : {&down, &up, &inject, &normed})
        for (float &value : *values) value = distribution(random);
    const size_t bytes = (down.size() + up.size() + inject.size()) * sizeof(float);
    Exl3GpuExecutor reference(device, 2 * bytes);
    std::vector<float> low(rows * rank), gate(rows * expanded), expected_injection(rows * streams);
    reference.matmul("hc.down", static_cast<uint32_t>(GGUFType::F32),
                     reinterpret_cast<const uint8_t *>(down.data()), down.size() * sizeof(float),
                     normed.data(), low.data(), rows, expanded, rank);
    for (float &value : low) {
        value /= static_cast<float>(streams);
        value /= 1.0f + std::exp(-value);
    }
    reference.matmul("hc.up", static_cast<uint32_t>(GGUFType::F32),
                     reinterpret_cast<const uint8_t *>(up.data()), up.size() * sizeof(float),
                     low.data(), gate.data(), rows, rank, expanded);
    for (float &value : gate) value = 1.0f / (1.0f + std::exp(-value));
    reference.matmul("hc.inject", static_cast<uint32_t>(GGUFType::F32),
                     reinterpret_cast<const uint8_t *>(inject.data()), inject.size() * sizeof(float),
                     normed.data(), expected_injection.data(), rows, expanded, streams);
    const auto expected = [&](bool divide_each_stream) {
        std::vector<float> output(rows * width, 0.0f);
        for (int row = 0; row < rows; ++row) {
            for (int column = 0; column < width; ++column) {
                for (int stream_index = 0; stream_index < streams; ++stream_index) {
                    const size_t offset = (static_cast<size_t>(row) * streams + stream_index) * width + column;
                    float value = normed[offset] * gate[offset];
                    if (divide_each_stream) value /= static_cast<float>(streams);
                    output[row * width + column] += value;
                }
                if (!divide_each_stream) output[row * width + column] /= static_cast<float>(streams);
            }
        }
        return output;
    };
    const auto close = [](const std::vector<float> &actual, const std::vector<float> &reference_values) {
        double error = 0.0, energy = 0.0;
        for (size_t index = 0; index < actual.size(); ++index) {
            if (!std::isfinite(actual[index])) return false;
            const double difference = actual[index] - reference_values[index];
            error += difference * difference;
            energy += static_cast<double>(reference_values[index]) * reference_values[index];
        }
        return std::sqrt(error / std::max(energy, 1e-20)) < 1e-5;
    };
    const Exl3GpuF32Weight down_weight{"hc.down", down.data(), expanded, rank};
    const Exl3GpuF32Weight up_weight{"hc.up", up.data(), rank, expanded};
    const Exl3GpuF32Weight inject_weight{"hc.inject", inject.data(), expanded, streams};
    Exl3GpuExecutor executor(device, 2 * bytes);
    std::vector<float> actual(rows * width), actual_injection(rows * streams);
    executor.hc_mix(down_weight, up_weight, &inject_weight, normed.data(), actual.data(),
                    actual_injection.data(), rows, width, streams, true);
    if (!close(actual, expected(true)) || !close(actual_injection, expected_injection)) return false;
    executor.hc_mix(down_weight, up_weight, nullptr, normed.data(), actual.data(),
                    nullptr, rows, width, streams, false);
    if (!close(actual, expected(false))) return false;
    const auto stats = executor.stats();
    if (stats.hc_mixes != 2 || stats.uploads != 3 || stats.cache_hits != 2) return false;
    bool rejected = false;
    try {
        executor.hc_mix(down_weight, up_weight, nullptr, normed.data(), actual.data(),
                        nullptr, 0, width, streams, false);
    } catch (const std::runtime_error &) { rejected = true; }
    std::cout << "  device=" << device << " HC device chain "
              << (rejected ? "PASSED" : "FAILED") << "\n";
    return rejected;
}

bool test_exl3_gpu_routed_device_async(int device) {
    std::cout << "[Test] EXL3 asynchronous routed cache and shared scratch...\n";
    constexpr int hidden = 128;
    constexpr int intermediate = 128;
    constexpr int bits = 4;
    std::mt19937 random(90210);
    std::uniform_real_distribution<float> distribution(-0.4f, 0.4f);
    std::unordered_map<std::string, std::vector<uint8_t>> packed_matrices;
    for (int expert = 0; expert < 3; ++expert) {
        for (int projection = 0; projection < 3; ++projection) {
            std::vector<uint16_t> trellis(8 * 8 * 16 * bits);
            std::vector<uint16_t> suh(hidden), svh(intermediate);
            for (uint16_t &word : trellis)
                word = static_cast<uint16_t>(random());
            for (uint16_t &value : suh)
                value = fp32_to_fp16(distribution(random));
            for (uint16_t &value : svh)
                value = fp32_to_fp16(distribution(random));
            Exl3Mul1MatrixView matrix{trellis.data(), suh.data(), svh.data(),
                                      nullptr, hidden, intermediate, bits};
            packed_matrices.emplace(
                "model.language_model.layers.0.mlp.experts." +
                    std::to_string(expert) + "." +
                    (projection == 0 ? "gate_proj" :
                     projection == 1 ? "up_proj" : "down_proj"),
                exl3_mul1_pack_matrix(matrix));
        }
    }
    const size_t matrix_bytes = packed_matrices.begin()->second.size();
    for (const auto &[key, matrix] : packed_matrices)
        if (matrix.size() != matrix_bytes)
            return false;

    const auto load = [&](const std::string &key) {
        const auto found = packed_matrices.find(key);
        if (found == packed_matrices.end())
            throw std::runtime_error("missing async EXL3 matrix: " + key);
        return found->second;
    };
    const auto reference = [&](const std::vector<int32_t> &route_ids,
                               const std::vector<float> &route_weights,
                               const std::vector<float> &input,
                               std::vector<float> &output) {
        output.assign(hidden, 0.0f);
        for (size_t route = 0; route < route_ids.size(); ++route) {
            const std::string prefix =
                "model.language_model.layers.0.mlp.experts." +
                std::to_string(route_ids[route]) + ".";
            const auto gate = load(prefix + "gate_proj");
            const auto up = load(prefix + "up_proj");
            const auto down = load(prefix + "down_proj");
            std::vector<float> gate_value(intermediate), up_value(intermediate);
            std::vector<float> activation(intermediate), expert_output(hidden);
            if (!exl3_mul1_matmul_packed(gate.data(), input.data(),
                                        gate_value.data(), 1, hidden,
                                        intermediate) ||
                !exl3_mul1_matmul_packed(up.data(), input.data(),
                                         up_value.data(), 1, hidden,
                                         intermediate))
                return false;
            for (int index = 0; index < intermediate; ++index)
                activation[index] = gate_value[index] /
                    (1.0f + std::exp(-gate_value[index])) * up_value[index];
            if (!exl3_mul1_matmul_packed(down.data(), activation.data(),
                                        expert_output.data(), 1,
                                        intermediate, hidden))
                return false;
            for (int index = 0; index < hidden; ++index)
                output[index] += route_weights[route] * expert_output[index];
        }
        return true;
    };
    const auto close = [](const std::vector<float> &actual,
                          const std::vector<float> &expected) {
        double error = 0.0;
        double energy = 0.0;
        for (size_t index = 0; index < actual.size(); ++index) {
            if (!std::isfinite(actual[index]))
                return false;
            const double difference = actual[index] - expected[index];
            error += difference * difference;
            energy += static_cast<double>(expected[index]) * expected[index];
        }
        return std::sqrt(error / std::max(energy, 1e-20)) < 1e-4;
    };

    struct DeviceResources {
        cudaStream_t streams[2] = {};
        float *inputs[3] = {};
        float *outputs[3] = {};
        float *partials[2] = {};
        std::vector<uint8_t *> packed_weights;
        ~DeviceResources() {
            for (cudaStream_t stream : streams)
                if (stream)
                    cudaStreamSynchronize(stream);
            for (float *pointer : inputs)
                if (pointer)
                    cudaFree(pointer);
            for (float *pointer : outputs)
                if (pointer)
                    cudaFree(pointer);
            for (float *pointer : partials)
                if (pointer)
                    cudaFree(pointer);
            for (uint8_t *pointer : packed_weights)
                if (pointer)
                    cudaFree(pointer);
            for (cudaStream_t stream : streams)
                if (stream)
                    cudaStreamDestroy(stream);
        }
    } device_resources;

    try {
        if (cudaSetDevice(device) != cudaSuccess ||
            cudaStreamCreateWithFlags(&device_resources.streams[0],
                                      cudaStreamNonBlocking) != cudaSuccess ||
            cudaStreamCreateWithFlags(&device_resources.streams[1],
                                      cudaStreamNonBlocking) != cudaSuccess)
            return false;
        std::vector<std::vector<float>> host_inputs(3,
                                                    std::vector<float>(hidden));
        for (auto &input : host_inputs)
            for (float &value : input)
                value = distribution(random);
        for (int index = 0; index < 3; ++index)
            if (cudaMalloc(&device_resources.inputs[index],
                           hidden * sizeof(float)) != cudaSuccess ||
                cudaMalloc(&device_resources.outputs[index],
                           hidden * sizeof(float)) != cudaSuccess)
                return false;
        if (cudaMemcpyAsync(device_resources.inputs[0], host_inputs[0].data(),
                            hidden * sizeof(float), cudaMemcpyHostToDevice,
                            device_resources.streams[0]) != cudaSuccess ||
            cudaMemcpyAsync(device_resources.inputs[1], host_inputs[1].data(),
                            hidden * sizeof(float), cudaMemcpyHostToDevice,
                            device_resources.streams[1]) != cudaSuccess ||
            cudaMemcpyAsync(device_resources.inputs[2], host_inputs[2].data(),
                            hidden * sizeof(float), cudaMemcpyHostToDevice,
                            device_resources.streams[0]) != cudaSuccess)
            return false;

        const std::vector<std::string> prefixes = {
            "model.language_model.layers.0.mlp.experts.0.",
            "model.language_model.layers.0.mlp.experts.1.",
            "model.language_model.layers.0.mlp.experts.2."};
        const std::vector<int32_t> routes_a = {0, 1};
        const std::vector<float> weights_a = {0.3f, 0.7f};
        const std::vector<int32_t> routes_b = {1, 0};
        const std::vector<float> weights_b = {0.6f, 0.4f};
        const std::vector<int32_t> routes_c = {2};
        const std::vector<float> weights_c = {1.0f};
        std::vector<float> expected_a, expected_b, expected_c;
        if (!reference(routes_a, weights_a, host_inputs[0], expected_a) ||
            !reference(routes_b, weights_b, host_inputs[1], expected_b) ||
            !reference(routes_c, weights_c, host_inputs[2], expected_c))
            return false;

        Exl3GpuExecutor executor(device, 6 * matrix_bytes);
        executor.routed_ffn_device(
            prefixes, routes_a.data(), weights_a.data(),
            device_resources.inputs[0], device_resources.outputs[0], 1, 2,
            hidden, intermediate, device_resources.streams[0], false, load);
        executor.routed_ffn_device(
            prefixes, routes_b.data(), weights_b.data(),
            device_resources.inputs[1], device_resources.outputs[1], 1, 2,
            hidden, intermediate, device_resources.streams[1], false, load);
        executor.routed_ffn_device(
            prefixes, routes_c.data(), weights_c.data(),
            device_resources.inputs[2], device_resources.outputs[2], 1, 1,
            hidden, intermediate, device_resources.streams[0], false, load);
        if (cudaStreamSynchronize(device_resources.streams[0]) != cudaSuccess ||
            cudaStreamSynchronize(device_resources.streams[1]) != cudaSuccess)
            return false;
        std::vector<float> actual_a(hidden), actual_b(hidden), actual_c(hidden);
        if (cudaMemcpy(actual_a.data(), device_resources.outputs[0],
                       hidden * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess ||
            cudaMemcpy(actual_b.data(), device_resources.outputs[1],
                       hidden * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess ||
            cudaMemcpy(actual_c.data(), device_resources.outputs[2],
                       hidden * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess)
            return false;
        const Exl3GpuStats stats = executor.stats();
        const bool passed = close(actual_a, expected_a) &&
                            close(actual_b, expected_b) &&
                            close(actual_c, expected_c) &&
                            stats.routed_ffns == 3 && stats.matmuls == 15 &&
                            stats.uploads == 9 && stats.evictions == 3 &&
                            stats.resident_bytes <= 6 * matrix_bytes;
        std::cout << "  device=" << device << " async routed cache and "
                  << "cross-stream scratch "
                  << (passed ? "PASSED" : "FAILED") << "\n";
        if (!passed)
            return false;

        const auto device_view = [&](const std::string &key, int input_width,
                                     int output_width) {
            const auto &packed = load(key);
            Exl3PackedMatrixHeader header{};
            if (packed.size() < sizeof(header))
                throw std::runtime_error("truncated external test matrix");
            std::memcpy(&header, packed.data(), sizeof(header));
            if (!exl3_valid_packed_header(header, packed.size()) ||
                header.input_features != input_width ||
                header.output_features != output_width)
                throw std::runtime_error("invalid external test matrix header");
            uint8_t *device_packed = nullptr;
            if (cudaMalloc(&device_packed, packed.size()) != cudaSuccess)
                throw std::runtime_error("external test matrix allocation failed");
            device_resources.packed_weights.push_back(device_packed);
            if (cudaMemcpy(device_packed, packed.data(), packed.size(),
                           cudaMemcpyHostToDevice) != cudaSuccess)
                throw std::runtime_error("external test matrix upload failed");
            return Exl3CudaMatrixView{
                reinterpret_cast<const uint16_t *>(device_packed +
                                                   header.trellis_offset),
                reinterpret_cast<const uint16_t *>(device_packed +
                                                   header.suh_offset),
                reinterpret_cast<const uint16_t *>(device_packed +
                                                   header.svh_offset),
                (header.flags & kExl3PackedHasBias)
                    ? reinterpret_cast<const uint16_t *>(device_packed +
                                                         header.bias_offset)
                    : nullptr,
                input_width, output_width, static_cast<int>(header.bits)};
        };
        std::vector<Exl3GpuExternalExpert> external_experts(2);
        std::vector<std::weak_ptr<int>> weak_holds(2);
        for (int route = 0; route < 2; ++route) {
            const std::string prefix = prefixes[static_cast<size_t>(route)];
            external_experts[static_cast<size_t>(route)].gate = device_view(
                prefix + "gate_proj", hidden, intermediate);
            external_experts[static_cast<size_t>(route)].up = device_view(
                prefix + "up_proj", hidden, intermediate);
            external_experts[static_cast<size_t>(route)].down = device_view(
                prefix + "down_proj", intermediate, hidden);
            auto hold = std::make_shared<int>(route);
            weak_holds[static_cast<size_t>(route)] = hold;
            external_experts[static_cast<size_t>(route)].hold = hold;
        }
        Exl3GpuExecutor external_executor(device, matrix_bytes);
        external_executor.routed_ffn_device_external(
            external_experts.data(), weights_a.data(), nullptr,
            device_resources.inputs[0], device_resources.outputs[0], 1, 2,
            hidden, intermediate, device_resources.streams[0]);
        external_executor.routed_ffn_device_external(
            external_experts.data(), weights_a.data(), nullptr,
            device_resources.inputs[0], device_resources.outputs[0], 1, 2,
            hidden, intermediate, device_resources.streams[0]);
        external_experts.clear();
        if (cudaStreamSynchronize(device_resources.streams[0]) != cudaSuccess)
            return false;
        std::vector<float> external_output(hidden);
        if (cudaMemcpy(external_output.data(), device_resources.outputs[0],
                       hidden * sizeof(float), cudaMemcpyDeviceToHost) !=
            cudaSuccess)
            return false;
        const Exl3GpuStats external_stats = external_executor.stats();
        const bool external_passed =
            close(external_output, expected_a) &&
            weak_holds[0].expired() && weak_holds[1].expired() &&
            external_stats.routed_ffns == 2 &&
            external_stats.matmuls == 12 && external_stats.uploads == 0;
        std::cout << "  device=" << device << " externally-owned routed weights "
                  << (external_passed ? "PASSED" : "FAILED") << "\n";
        if (!external_passed)
            return false;

        const auto make_packed = [&](int bits) {
            std::vector<uint16_t> trellis(8 * 8 * 16 * bits);
            std::vector<uint16_t> suh(hidden), svh(intermediate);
            for (uint16_t &word : trellis)
                word = static_cast<uint16_t>(random());
            for (uint16_t &value : suh)
                value = fp32_to_fp16(distribution(random));
            for (uint16_t &value : svh)
                value = fp32_to_fp16(distribution(random));
            Exl3Mul1MatrixView matrix{trellis.data(), suh.data(), svh.data(),
                                      nullptr, hidden, intermediate, bits};
            return exl3_mul1_pack_matrix(matrix);
        };
        const auto mixed_gate_packed = make_packed(3);
        const auto mixed_up_packed = make_packed(5);
        const auto upload_view = [&](const std::vector<uint8_t> &packed) {
            Exl3PackedMatrixHeader header{};
            if (packed.size() < sizeof(header))
                throw std::runtime_error("truncated mixed-bit test matrix");
            std::memcpy(&header, packed.data(), sizeof(header));
            if (!exl3_valid_packed_header(header, packed.size()))
                throw std::runtime_error("invalid mixed-bit test matrix");
            uint8_t *device_packed = nullptr;
            if (cudaMalloc(&device_packed, packed.size()) != cudaSuccess)
                throw std::runtime_error("mixed-bit matrix allocation failed");
            device_resources.packed_weights.push_back(device_packed);
            if (cudaMemcpy(device_packed, packed.data(), packed.size(),
                           cudaMemcpyHostToDevice) != cudaSuccess)
                throw std::runtime_error("mixed-bit matrix upload failed");
            return Exl3CudaMatrixView{
                reinterpret_cast<const uint16_t *>(device_packed +
                                                   header.trellis_offset),
                reinterpret_cast<const uint16_t *>(device_packed +
                                                   header.suh_offset),
                reinterpret_cast<const uint16_t *>(device_packed +
                                                   header.svh_offset),
                nullptr, hidden, intermediate, static_cast<int>(header.bits)};
        };
        const auto mixed_gate = upload_view(mixed_gate_packed);
        const auto mixed_up = upload_view(mixed_up_packed);
        if (cudaMalloc(&device_resources.partials[0],
                       intermediate * sizeof(float)) != cudaSuccess ||
            cudaMalloc(&device_resources.partials[1],
                       intermediate * sizeof(float)) != cudaSuccess)
            return false;
        std::vector<float> mixed_gate_output(intermediate);
        std::vector<float> mixed_up_output(intermediate);
        std::vector<float> mixed_activation(intermediate);
        if (!exl3_mul1_matmul_packed(
                mixed_gate_packed.data(), host_inputs[0].data(),
                mixed_gate_output.data(), 1, hidden, intermediate) ||
            !exl3_mul1_matmul_packed(
                mixed_up_packed.data(), host_inputs[0].data(),
                mixed_up_output.data(), 1, hidden, intermediate))
            return false;
        for (int index = 0; index < intermediate; ++index)
            mixed_activation[static_cast<size_t>(index)] =
                mixed_gate_output[static_cast<size_t>(index)] /
                (1.0f + std::exp(-mixed_gate_output[static_cast<size_t>(index)])) *
                mixed_up_output[static_cast<size_t>(index)];
        if (launch_exl3_mul1_gate_up(
                mixed_gate, mixed_up, device_resources.inputs[0],
                device_resources.outputs[1], 1, device_resources.partials[0],
                device_resources.partials[1], device_resources.streams[0]) !=
                cudaSuccess ||
            cudaStreamSynchronize(device_resources.streams[0]) != cudaSuccess)
            return false;
        std::vector<float> mixed_actual(intermediate);
        if (cudaMemcpy(mixed_actual.data(), device_resources.outputs[1],
                       intermediate * sizeof(float), cudaMemcpyDeviceToHost) !=
            cudaSuccess)
            return false;
        const bool mixed_passed = close(mixed_actual, mixed_activation);
        std::cout << "  device=" << device << " mixed-bit gate/up (3/5) "
                  << (mixed_passed ? "PASSED" : "FAILED") << "\n";
        if (!mixed_passed)
            return false;

        Exl3GgmlCachedRoutedFfnState route_state;
        if (!route_state.initialize(&external_executor, prefixes, 3, 2, hidden,
                                    intermediate))
            return false;
        cudaPointerAttributes route_attributes{};
        if (cudaPointerGetAttributes(&route_attributes,
                                     route_state.route_ids_host) !=
                cudaSuccess ||
            route_attributes.type != cudaMemoryTypeHost)
            return false;
        const std::vector<int32_t> host_route_ids = {0, 1, 1, 2, 2, 0};
        const std::vector<float> host_route_weights =
            {0.2f, 0.8f, 0.3f, 0.7f, 0.4f, 0.6f};
        if (!route_state.set_host_routes(host_route_ids.data(),
                                         host_route_weights.data()))
            return false;
        int32_t *device_route_ids = nullptr;
        cudaStream_t route_stream = nullptr;
        if (cudaMalloc(&device_route_ids,
                       host_route_ids.size() * sizeof(int32_t)) != cudaSuccess ||
            cudaStreamCreateWithFlags(&route_stream, cudaStreamNonBlocking) !=
                cudaSuccess) {
            if (device_route_ids)
                cudaFree(device_route_ids);
            return false;
        }
        const bool route_copy_ok =
            cudaMemcpy(device_route_ids, host_route_ids.data(),
                       host_route_ids.size() * sizeof(int32_t),
                       cudaMemcpyHostToDevice) == cudaSuccess &&
            cudaMemcpyAsync(route_state.route_ids_host, device_route_ids,
                            host_route_ids.size() * sizeof(int32_t),
                            cudaMemcpyDeviceToHost, route_stream) ==
                cudaSuccess &&
            cudaStreamSynchronize(route_stream) == cudaSuccess &&
            std::equal(host_route_ids.begin(), host_route_ids.end(),
                       route_state.route_ids_host);
        cudaStreamDestroy(route_stream);
        cudaFree(device_route_ids);
        std::cout << "  device=" << device << " pinned route D2H staging "
                  << (route_copy_ok ? "PASSED" : "FAILED") << "\n";
        return route_copy_ok;
    } catch (const std::exception &error) {
        std::cerr << "  async routed cache error: " << error.what() << "\n";
        return false;
    }
}

bool test_exl3_cuda_kernels() {
    std::cout << "[Test] EXL3 CUDA matmul/reconstruction/cache...\n";
    int device_count = 0;
    int original_device = 0;
    if (cudaGetDevice(&original_device) != cudaSuccess) return false;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0)
        return false;
    const int width = 128;
    std::mt19937 random(4242);
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
    for (int device = 0; device < device_count; ++device) {
        if (cudaSetDevice(device) != cudaSuccess) return false;
        for (int bits = 1; bits <= 8; ++bits) {
            const int input_width = bits <= 3 ? width : 640;
            const int output_width = bits <= 3 ? 256 : 384;
            const int rows = bits == 5 ? 17 : bits == 6 ? 3 : 1;
            std::vector<uint16_t> trellis(input_width / 16 * (output_width / 16) * 16 * bits);
            std::vector<uint16_t> suh(input_width), svh(output_width), bias(output_width);
            for (auto &word : trellis) word = static_cast<uint16_t>(random());
            for (auto &value : suh) value = fp32_to_fp16(distribution(random));
            for (auto &value : svh) value = fp32_to_fp16(distribution(random));
            for (auto &value : bias) value = fp32_to_fp16(distribution(random) * 0.01f);
            Exl3Mul1MatrixView host{trellis.data(), suh.data(), svh.data(), bits % 2 ? bias.data() : nullptr,
                                    input_width, output_width, bits};
            auto packed = exl3_mul1_pack_matrix(host);
            Exl3PackedMatrixHeader header{};
            std::memcpy(&header, packed.data(), sizeof(header));
            std::vector<float> input(rows * input_width), expected(rows * output_width);
            for (auto &value : input) value = distribution(random);
            if (!exl3_mul1_matmul(host, input.data(), expected.data(), rows)) return false;
            void *device_packed = nullptr;
            float *device_input = nullptr, *device_output = nullptr;
            float *partials = nullptr, *device_weight = nullptr;
            float *device_accumulator = nullptr;
            const auto release = [&] {
                cudaFree(device_packed); cudaFree(device_input); cudaFree(device_output);
                cudaFree(partials); cudaFree(device_weight); cudaFree(device_accumulator);
            };
            if (cudaMalloc(&device_packed, packed.size()) != cudaSuccess ||
                cudaMalloc(&device_input, input.size() * sizeof(float)) != cudaSuccess ||
                cudaMalloc(&device_output, expected.size() * sizeof(float)) != cudaSuccess ||
                cudaMalloc(&partials, 2 * expected.size() * exl3_cuda_key_splits(input_width) * sizeof(float)) != cudaSuccess ||
                cudaMalloc(&device_weight, static_cast<size_t>(input_width) * output_width * sizeof(float)) != cudaSuccess ||
                cudaMalloc(&device_accumulator, expected.size() * sizeof(float)) != cudaSuccess) {
                release(); return false;
            }
            cudaMemcpy(device_packed, packed.data(), packed.size(), cudaMemcpyHostToDevice);
            cudaMemcpy(device_input, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice);
            const auto *base = static_cast<const uint8_t *>(device_packed);
            Exl3CudaMatrixView gpu{
                reinterpret_cast<const uint16_t *>(base + header.trellis_offset),
                reinterpret_cast<const uint16_t *>(base + header.suh_offset),
                reinterpret_cast<const uint16_t *>(base + header.svh_offset),
                host.bias ? reinterpret_cast<const uint16_t *>(base + header.bias_offset) : nullptr,
                input_width, output_width, bits};
            if (launch_exl3_mul1_matmul(gpu, device_input, device_output, rows,
                                          partials) != cudaSuccess) {
                release(); return false;
            }
            std::vector<float> actual(expected.size());
            if (cudaMemcpy(actual.data(), device_output, actual.size() * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
                release(); return false;
            }
            const auto close = [](const std::vector<float> &left, const std::vector<float> &right) {
                double error = 0.0, energy = 0.0;
                for (size_t index = 0; index < left.size(); ++index) {
                    if (!std::isfinite(left[index])) return false;
                    const double difference = left[index] - right[index];
                    error += difference * difference;
                    energy += static_cast<double>(right[index]) * right[index];
                }
                return std::sqrt(error / std::max(energy, 1e-20)) < 1e-5;
            };
            bool ok = close(actual, expected);
            std::vector<float> expected_activation = expected;
            for (float &value : expected_activation)
                value = value / (1.0f + std::exp(-value)) * value;
            if (launch_exl3_mul1_gate_up(
                    gpu, gpu, device_input, device_output, rows, partials,
                    partials + expected.size() * exl3_cuda_key_splits(input_width)) != cudaSuccess ||
                cudaMemcpy(actual.data(), device_output,
                           actual.size() * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess)
                ok = false;
            ok = ok && close(actual, expected_activation);
            if (!CudaKernelDispatcher::matmul(GGUFType::EXL3, device_output, device_input,
                                                device_packed, rows, input_width, output_width)) ok = false;
            if (cudaMemcpy(actual.data(), device_output, actual.size() * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess)
                ok = false;
            ok = ok && close(actual, expected);
            std::vector<float> expected_accumulation = expected;
            for (float &value : expected_accumulation) value *= 0.25f;
            cudaMemset(device_accumulator, 0, expected.size() * sizeof(float));
            if (launch_exl3_mul1_matmul_accumulate(
                    gpu, device_input, device_accumulator, 0.75f, rows,
                    partials) != cudaSuccess ||
                launch_exl3_mul1_matmul_accumulate(
                    gpu, device_input, device_accumulator, -0.5f, rows,
                    partials) != cudaSuccess ||
                cudaMemcpy(actual.data(), device_accumulator,
                           actual.size() * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess)
                ok = false;
            ok = ok && close(actual, expected_accumulation);
            std::vector<float> weight(static_cast<size_t>(input_width) * output_width), gpu_weight(weight.size());
            if (!exl3_mul1_reconstruct(host, weight.data()) ||
                launch_exl3_mul1_reconstruct(gpu, device_weight) != cudaSuccess ||
                cudaMemcpy(gpu_weight.data(), device_weight, gpu_weight.size() * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess)
                ok = false;
            ok = ok && close(gpu_weight, weight);
            auto malformed = header;
            malformed.trellis_offset += 2;
            cudaMemcpy(device_packed, &malformed, sizeof(malformed), cudaMemcpyHostToDevice);
            if (exl3_valid_packed_header(malformed, packed.size()) ||
                CudaKernelDispatcher::matmul(GGUFType::EXL3, device_output, device_input,
                                              device_packed, rows, input_width, output_width))
                ok = false;
            if (launch_exl3_mul1_matmul(gpu, device_input, device_output, 0,
                                          partials) != cudaErrorInvalidValue)
                ok = false;
            release();
            Exl3GpuExecutor executor(device, 2 * packed.size());
            for (int attempt = 0; attempt < 4; ++attempt) {
                const std::string key = "expert" + std::to_string(attempt);
                executor.matmul(key, static_cast<uint32_t>(GGUFType::EXL3), packed.data(),
                                  packed.size(), input.data(), actual.data(), rows,
                                  input_width, output_width, false);
                ok = ok && close(actual, expected);
            }
            ok = ok && executor.matmul_cached("expert3", input.data(), actual.data(), rows, input_width, output_width);
            ok = ok && close(actual, expected);
            const auto stats = executor.stats();
            ok = ok && stats.matmuls == 5 && stats.cache_hits == 1 && stats.evictions == 2 &&
                 stats.buffer_reuses == 2 &&
                 stats.resident_bytes <= 2 * packed.size();
            Exl3Mul1MatrixView down_matrix{host.trellis, host.svh, host.suh, nullptr,
                                            output_width, input_width, bits};
            const auto down_packed = exl3_mul1_pack_matrix(down_matrix);
            std::vector<float> activated(output_width), expert_result(input_width);
            for (int column = 0; column < output_width; ++column)
                activated[column] = expected[column] / (1.0f + std::exp(-expected[column])) * expected[column];
            if (!exl3_mul1_matmul(down_matrix, activated.data(), expert_result.data(), 1)) return false;
            std::vector<float> combined(input_width), combined_expected(input_width);
            for (int column = 0; column < input_width; ++column)
                combined_expected[column] = 0.25f * expert_result[column] + 0.75f * expert_result[column];
            const float routing_weights[] = {0.25f, 0.75f};
            size_t load_count = 0;
            const auto load = [&](const std::string &key) {
                ++load_count;
                return key.ends_with("down_proj") ? down_packed : packed;
            };
            const size_t ffn_budget = 4 * packed.size() + 2 * down_packed.size();
            Exl3GpuExecutor ffn_executor(device, ffn_budget);
            for (int attempt = 0; attempt < 3; ++attempt) {
                const std::vector<std::string> prefixes = attempt < 2
                    ? std::vector<std::string>{"expert0.", "expert1."}
                    : std::vector<std::string>{"expert1.", "expert2."};
                ffn_executor.routed_ffn(prefixes, routing_weights, input.data(), combined.data(),
                                         input_width, output_width, load);
                ok = ok && close(combined, combined_expected);
            }
            const auto ffn_stats = ffn_executor.stats();
            ok = ok && load_count == 9 && ffn_stats.routed_ffns == 3 && ffn_stats.matmuls == 18 &&
                 ffn_stats.evictions == 3 && ffn_stats.resident_bytes <= ffn_budget;
            Exl3GpuExecutor held_executor(device, 2 * packed.size());
            bool held_rejected = false;
            try {
                held_executor.routed_ffn({"held."}, routing_weights, input.data(), combined.data(),
                                          input_width, output_width, load);
            } catch (const std::runtime_error &) { held_rejected = true; }
            held_executor.matmul("after-failure", static_cast<uint32_t>(GGUFType::EXL3),
                                   packed.data(), packed.size(), input.data(), actual.data(), rows,
                                   input_width, output_width, false);
            ok = ok && held_rejected && close(actual, expected);
            std::cout << "  device=" << device << " bits=" << bits << " rows=" << rows
                      << " " << (ok ? "PASSED" : "FAILED") << "\n";
            if (!ok) return false;
        }
        const int input_width = 7, output_width = 11, rows = 3;
        std::vector<float> weights(input_width * output_width), input(rows * input_width);
        std::vector<float> expected(rows * output_width, 0.0f), actual(expected.size());
        for (auto &value : weights) value = distribution(random);
        for (auto &value : input) value = distribution(random);
        for (int row = 0; row < rows; ++row)
            for (int column = 0; column < output_width; ++column)
                for (int feature = 0; feature < input_width; ++feature)
                    expected[row * output_width + column] +=
                        input[row * input_width + feature] * weights[column * input_width + feature];
        Exl3GpuExecutor dense_executor(device, weights.size() * sizeof(float));
        dense_executor.matmul("dense", static_cast<uint32_t>(GGUFType::F32),
                              reinterpret_cast<const uint8_t *>(weights.data()), weights.size() * sizeof(float),
                              input.data(), actual.data(), rows, input_width, output_width);
        for (size_t index = 0; index < expected.size(); ++index)
            if (!std::isfinite(actual[index]) || std::abs(actual[index] - expected[index]) > 1e-5f)
                return false;
        if (!dense_executor.matmul_cached("dense", input.data(), actual.data(), rows, input_width, output_width) ||
            dense_executor.matmul_cached("missing", input.data(), actual.data(), rows, input_width, output_width))
            return false;
        bool rejected = false;
        try {
            dense_executor.matmul("over-budget", static_cast<uint32_t>(GGUFType::F32),
                                  reinterpret_cast<const uint8_t *>(weights.data()), weights.size() * sizeof(float),
                                  input.data(), actual.data(), rows, input_width, output_width, false);
        } catch (const std::runtime_error &) { rejected = true; }
        if (!rejected || dense_executor.stats().evictions != 0) return false;
        std::cout << "  device=" << device << " F32 dispatch and pinned-cache budget PASSED\n";
        if (!test_exl3_gpu_hc(device)) return false;
        if (!test_exl3_gpu_routed_device_async(device)) return false;
    }
    return cudaSetDevice(original_device) == cudaSuccess;
}

int main() {
    if (std::getenv("MOESTREAM_TEST_ONLY_EXL3_GGML_CONFIG"))
        return test_exl3_ggml_graph_capture_config() ? 0 : 1;
    if (!test_exl3_cpu_dispatch()) return 1;
    if (!test_exl3_cpu_matrix_cache()) return 1;
    if (std::getenv("MOESTREAM_TEST_EXL3_CPU_ONLY")) return 0;
    if (!test_exl3_cuda_kernels()) return 1;
    if (std::getenv("MOESTREAM_TEST_EXL3_ONLY")) return 0;
    std::cout << "========================================\n";
    std::cout << "  MoeStream Native Test Harness\n";
    std::cout << "========================================\n";

    if (std::getenv("MOESTREAM_TEST_ONLY_Q2_0_VNNI")) {
        return test_qwen_q2_0_vnni_kernel() ? 0 : 1;
    }
    if (std::getenv("MOESTREAM_TEST_ONLY_QWEN_STATE_HANDOFF_CONFIG")) {
        return test_qwen_prefill_state_handoff_config() ? 0 : 1;
    }

    // Diagnostic-only escape hatch (2026-09-10): run just Test 35 in
    // isolation so compute-sanitizer's overhead is tractable -- the full
    // suite's CPU-only tests dominate wall time and add nothing to a
    // race-condition investigation scoped to GgmlQwenSharedRunner.
    if (std::getenv("MOESTREAM_TEST_ONLY_35")) {
        bool ok = test_qwen_shared_expert_runner();
        return ok ? 0 : 1;
    }
    if (std::getenv("MOESTREAM_TEST_ONLY_36")) {
        bool ok = test_qwen_hc_runner();
        return ok ? 0 : 1;
    }
    if (std::getenv("MOESTREAM_TEST_ONLY_37")) {
        bool ok = test_qwen_hc_runner_q8_exact();
        return ok ? 0 : 1;
    }
    if (std::getenv("MOESTREAM_TEST_ONLY_QWEN_CPU_BENCH")) {
        bool ok = bench_qwen_cpu_moe();
        return ok ? 0 : 1;
    }
    if (std::getenv("MOESTREAM_TEST_ONLY_Q3K_CPU_BENCH")) {
        bench_q3k_cpu_only();
        return 0;
    }
#ifdef __linux__
    if (std::getenv("MOESTREAM_TEST_ONLY_Q3K_THP_BENCH")) {
        bench_q3k_thp_cpu_only();
        return 0;
    }
#endif
    if (std::getenv("MOESTREAM_TEST_ONLY_CPU_IQ")) {
        // Keep both diagnostics independent.  In particular, an IQ1_M
        // failure must not hide the Qwen graph test, and a graph failure must
        // not hide the lower-level quantized-kernel result.
        const bool iq_ok = test_cpu_iq_matmul_paths();
        const bool graph_ok = test_qwen_cpu_moe_graph();
        const bool iq1m_graph_ok = test_qwen_cpu_moe_iq1m_graph();
        const bool mixed_graph_ok = test_qwen_cpu_moe_mixed_strided_graph();
        const bool iq2_graph_ok = test_qwen_cpu_moe_iq2_strided_graph();
        return (iq_ok && graph_ok && iq1m_graph_ok && mixed_graph_ok &&
                iq2_graph_ok)
                   ? 0
                   : 1;
    }
    if (std::getenv("MOESTREAM_TEST_ONLY_CUDA_FUSED")) {
        return test_cuda_fused_kernels() ? 0 : 1;
    }
    if (std::getenv("MOESTREAM_TEST_ONLY_COMPACT_CACHE")) {
        return test_compact_vram_slot_planner() ? 0 : 1;
    }

    test_cpu_matrix_ops();
    test_graph_initializer();
    test_attention_extraction();
    test_cuda_execution();
    bool fused_kernels_ok = test_cuda_fused_kernels();
    bool avx512_dequant_ok = test_avx512_dequant();
    bool avx512_dequant_iq_ok = test_avx512_dequant_iq();
    bool cpu_iq_matmul_ok = test_cpu_iq_matmul_paths();
    bool qwen_cpu_moe_graph_ok = test_qwen_cpu_moe_graph();
    bool qwen_cpu_moe_iq1m_graph_ok = test_qwen_cpu_moe_iq1m_graph();
    bool qwen_cpu_moe_mixed_strided_graph_ok =
        test_qwen_cpu_moe_mixed_strided_graph();
    bool qwen_cpu_moe_iq2_strided_graph_ok =
        test_qwen_cpu_moe_iq2_strided_graph();
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
        bench_iq2xxs_vnni_prototype();
        bench_iq3xxs_avx2_prototype();
    } else {
        std::cout << "\n[Test 8] SKIPPED by default (set MOESTREAM_RUN_BENCH=1 to run the throughput bench)\n";
    }
    bool bf16_dense_ok = test_bf16_cublas_dense_matmul();
    bool bf16_matched_dense_ok = test_bf16_matched_cublas_dense_matmul();
    bool bf16_batched_qabsorb_ok = test_bf16_batched_qabsorb();
    bool score_gpu_ok = test_score_gpu();
    bool score_gpu_nope_only_ok = test_score_gpu_nope_only();
    bool wuv_batched_gpu_ok = test_wuv_batched_gpu();
    bool wuv_int8_batched_gpu_ok = test_wuv_int8_batched_gpu();
    bool wuv_int8_batched_prefill_ok = test_wuv_int8_batched_gpu_prefill();
    bool expert_gpu_accum_ok = test_expert_gpu_accum();
    bool dense_bf16_kquant_ok = test_dense_bf16_kquant();
    bool gdn_conv1d_gpu_ok = test_gdn_conv1d_gpu();
    bool gdn_conv_fused_ok = test_gdn_conv_fused();
    bool qwen_shared_expert_ok = test_qwen_shared_expert_runner();
    bool qwen_hc_runner_ok = test_qwen_hc_runner();
    bool qwen_hc_q8_exact_ok = test_qwen_hc_runner_q8_exact();
    bool score_gpu_prefill_ok = test_score_gpu_prefill();
    bool wuv_batched_prefill_ok = test_wuv_batched_gpu_prefill();
    bool real_expert_iq2xxs_ok = test_real_expert_iq2xxs_dequant();
    bool mixed_precision_layer_ok = test_mixed_precision_layer();
    bool unpacked_relaxed_check_ok = test_unpacked_format_relaxed_check();
    bool slab_bytes_max_ok = test_compute_expert_slab_bytes_max();
    bool compact_vram_planner_ok = test_compact_vram_slot_planner();
    bool ram_async_source_hold_ok = test_ram_async_source_hold();
    bool vram_mixed_sizes_ok = test_vram_cache_mixed_sizes();
    bool upload_async_pinned_src_ok = test_upload_async_already_pinned_src();
    bool glm5next_kda_gpu_ok = test_glm5next_kda_gpu();
    bool glm5next_router_gpu_ok = test_glm5next_router_gpu();
    bool glm5next_dense_projection_fusion_ok =
        test_glm5next_dense_projection_fusion();
    bool gpu_device_allowlist_ok = test_gpu_device_allowlist();
    bool qwen_state_handoff_config_ok = test_qwen_prefill_state_handoff_config();
    bool exl3_graph_capture_config_ok =
        test_exl3_ggml_graph_capture_config();

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
    if (!cpu_iq_matmul_ok) {
        std::cout << "❌ CPU IQ1_M/IQ2_XXS/IQ3_XXS matmul test FAILED -- see [Test 7b] output above.\n";
        return 1;
    }
    if (!qwen_cpu_moe_graph_ok) {
        std::cout << "❌ Qwen persistent CPU GGML MoE graph test FAILED -- see [Test 7c] output above.\n";
        return 1;
    }
    if (!qwen_cpu_moe_iq1m_graph_ok) {
        std::cout << "❌ Qwen IQ1_M CPU GGML MoE graph test FAILED -- see [Test 7d] output above.\n";
        return 1;
    }
    if (!qwen_cpu_moe_mixed_strided_graph_ok) {
        std::cout << "❌ Qwen mixed-type strided CPU GGML MoE graph test FAILED -- see [Test 7e] output above.\n";
        return 1;
    }
    if (!qwen_cpu_moe_iq2_strided_graph_ok) {
        std::cout << "❌ Qwen IQ2_XXS strided CPU GGML MoE graph test FAILED -- see [Test 7f] output above.\n";
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
    if (!score_gpu_prefill_ok) {
        std::cout << "❌ Score GPU prefill (S>1, causal mask) correctness test FAILED -- see [Test 16] output above.\n";
        return 1;
    }
    if (!wuv_batched_prefill_ok) {
        std::cout << "❌ WUV batched prefill (S>1 scatter) correctness test FAILED -- see [Test 17] output above.\n";
        return 1;
    }
    if (!score_gpu_ok) {
        std::cout << "❌ Score GPU path correctness test FAILED -- see [Test 13] output above.\n";
        return 1;
    }
    if (!score_gpu_nope_only_ok) {
        std::cout << "❌ Score GPU nope-only/DSA path FAILED -- see [Test 33] output above.\n";
        return 1;
    }
    if (!wuv_batched_gpu_ok) {
        std::cout << "❌ WUV batched GPU path correctness test FAILED -- see [Test 14] output above.\n";
        return 1;
    }
    if (!wuv_int8_batched_gpu_ok) {
        std::cout << "❌ WUV int8 batched GPU path correctness test FAILED -- see [Test 15] output above.\n";
        return 1;
    }
    if (!dense_bf16_kquant_ok) {
        std::cout << "❌ Dense BF16 Q5_K/Q6_K path FAILED -- see [Test 27] output above.\n";
        return 1;
    }
    if (!gdn_conv1d_gpu_ok) {
        std::cout << "❌ GDN causal conv1d+SiLU GPU path FAILED -- see [Test 28] output above.\n";
        return 1;
    }
    if (!gdn_conv_fused_ok) {
        std::cout << "❌ GDN proj+conv1d fused GPU path FAILED -- see [Test 29] output above.\n";
        return 1;
    }
    if (!qwen_shared_expert_ok) {
        std::cout << "❌ Qwen4Exp shared-expert GGML runner FAILED -- see [Test 35] output above.\n";
        return 1;
    }
    if (!qwen_hc_runner_ok) {
        std::cout << "❌ Qwen4Exp HC GGML runner FAILED -- see [Test 36] output above.\n";
        return 1;
    }
    if (!qwen_hc_q8_exact_ok) {
        std::cout << "❌ Qwen4Exp HC exact Q8_0 path FAILED -- see [Test 37] output above.\n";
        return 1;
    }
    if (!expert_gpu_accum_ok) {
        std::cout << "❌ MoE GPU-side expert accumulation FAILED -- see [Test 26] output above.\n";
        return 1;
    }
    if (!wuv_int8_batched_prefill_ok) {
        std::cout << "❌ WUV int8 batched GPU path FAILED at S>1 -- see [Test 22] output above.\n";
        return 1;
    }
    if (!real_expert_iq2xxs_ok) {
        std::cout << "❌ Real-Data IQ2_XXS dequant consistency test FAILED -- see [Test 11] output above.\n";
        return 1;
    }
    if (!mixed_precision_layer_ok) {
        std::cout << "❌ Mixed-precision-within-one-layer test FAILED -- see [Test 18] output above.\n";
        return 1;
    }
    if (!unpacked_relaxed_check_ok) {
        std::cout << "❌ sanity_check_unpacked_format() relaxation test FAILED -- see [Test 19] output above.\n";
        return 1;
    }
    if (!slab_bytes_max_ok) {
        std::cout << "❌ compute_expert_slab_bytes() max-size test FAILED -- see [Test 20] output above.\n";
        return 1;
    }
    if (!compact_vram_planner_ok) {
        std::cout << "❌ compact VRAM slot planner test FAILED -- see [Test 20b] output above.\n";
        return 1;
    }
    if (!ram_async_source_hold_ok) {
        std::cout << "❌ RAM async source hold test FAILED -- see [Test 20c] output above.\n";
        return 1;
    }
    if (!vram_mixed_sizes_ok) {
        std::cout << "❌ VRAM cache mixed-sizes test FAILED -- see [Test 21] output above.\n";
        return 1;
    }
    if (!upload_async_pinned_src_ok) {
        std::cout << "❌ upload_expert_async already-pinned-source test FAILED -- see [Test 23] output above.\n";
        return 1;
    }
    if (!glm5next_kda_gpu_ok) {
        std::cout << "❌ GLM5NEXT KDA GPU path FAILED -- see [Test 30] output above.\n";
        return 1;
    }
    if (!glm5next_router_gpu_ok) {
        std::cout << "❌ GLM5NEXT native router GPU path FAILED -- see [Test 31] output above.\n";
        return 1;
    }
    if (!glm5next_dense_projection_fusion_ok) {
        std::cout << "❌ GLM5NEXT KDA dense projection fusion FAILED -- see [Test 34] output above.\n";
        return 1;
    }
    if (!gpu_device_allowlist_ok) {
        std::cout << "❌ GPU config allow-list test FAILED -- see [Test 32] output above.\n";
        return 1;
    }
    if (!qwen_state_handoff_config_ok) {
        std::cout << "❌ Qwen prefill state handoff config test FAILED.\n";
        return 1;
    }
    if (!exl3_graph_capture_config_ok) {
        std::cout << "❌ EXL3 GGML graph-capture config test FAILED.\n";
        return 1;
    }
    std::cout << "✅ All core engine tests passed successfully!\n";
    return 0;
}
