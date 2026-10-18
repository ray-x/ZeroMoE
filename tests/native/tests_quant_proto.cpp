// Native test harness: tests_quant_proto (split from src/main_test.cpp).
#include "native_tests.hpp"

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

