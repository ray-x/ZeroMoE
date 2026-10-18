// Native test harness: tests_cpu_moe (split from src/main_test.cpp).
#include "native_tests.hpp"

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

void bench_q3k_cpu_only() {
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

void bench_q3k_thp_cpu_only() {
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

