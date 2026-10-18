// Native test harness: tests_gpu_attention (split from src/main_test.cpp).
#include "native_tests.hpp"

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

