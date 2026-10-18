// Native test harness: tests_basic_cuda (split from tests/native/main_test.cpp).
#include "native_tests.hpp"

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
    // backends/cpu/ggml_dequant.cpp), so the exact dequant+naive-dot path is both
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
    // sits BELOW kquant_lane8_ok()'s `bpr >= 8` gate (kernels/quant/kquant_matmul.cu) --
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
    // Q8_0 (kernels/quant/q8_0_matmul.cu): pre-existing gap in this suite -- the
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

    // Q5_1 (kernels/quant/q5_1_matmul.cu, new 2026-09-04). No lossy CPU fast path
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
    // per-group scale, no delta -- see kernels/quant/iq1_dequant.cu's own comment).
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
    // layers (kernels/quant/iq4xs_matmul.cu). Fixed 16-entry lookup table + linear
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
    // MXFP4: NEW, currently DORMANT kernel (kernels/quant/mxfp4_matmul.cu) -- see
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

