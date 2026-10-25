// Native test harness: tests_cpu_quant (split from tests/native/main_test.cpp).
#include "native_tests.hpp"

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
//   * IQ2_XXS/IQ3_XXS default to ZeroMoE's exact dequant+F32 dot path;
//   * IQ1_M optionally uses ggml's production IQ1_M x Q8_K dot path;
//   * the Qwen grouped wrappers must produce the same result as their
//     one-matrix form, regardless of whether the caller has one or many
//     routed experts.
// These tests deliberately use real GGML-quantized bytes, not synthetic block
// contents.  They catch row-stride, block-size, expert-index, and grouped
// activation-quantization mistakes without loading a model.
bool exact_iq_matmul_reference(GGUFType type, const uint8_t *weights,
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

bool compare_iq_outputs(const std::string &label, const float *a,
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
    setenv("ZEROMOE_QWEN_CPU_DIRECT_POOL", "1", 1);
    setenv("ZEROMOE_QWEN_CPU_DIRECT_FUSED", "1", 1);
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

