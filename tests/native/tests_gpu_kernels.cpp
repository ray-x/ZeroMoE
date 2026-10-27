// Native test harness: tests_gpu_kernels (split from tests/native/main_test.cpp).
#include "native_tests.hpp"

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

// 2026-09-05: GDN causal conv1d+SiLU GPU path (kernels/math/gdn_conv1d.cu,
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
// OFF in every shipped config -- see configs/testing/qwen4exp_3090_only.toml's own
// comment). Written 2026-09-10 to isolate a real, confirmed correctness bug
// found on real hardware: 100% NaN corruption end to end (hidden_states
// 1249280/1249280 NaN, logits 248320/248320 NaN), traced via VERBOSE tracing
// to layer=46 of 48 specifically -- finite input, NaN router weights with a
// suspicious exact stride-4 expert-ID pattern (e292,e288,e284,...), pointing
// at a buffer-aliasing/overwrite bug rather than the missing-sync race first
// suspected (that race was real too -- see run()'s own ggml_backend_synchronize
// fix -- but re-testing after it still reports the non-finite tensor.
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

