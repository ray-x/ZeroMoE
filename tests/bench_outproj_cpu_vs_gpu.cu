// bench_outproj_cpu_vs_gpu.cu — standalone, isolated microbenchmark.
//
// Answers one question: for DeepSeek-V4's grouped output-projection shape
// (o_group_count=8, o_lora_rank=1024, group_in=4096, measured on real
// dsv4_attention_gguf via the [DSV4-DEBUG] log line), is CPU or GPU faster
// for the REAL decode access pattern -- one token (S=1) at a time, cycling
// through 43 distinct per-layer weight matrices (so no layer's weight stays
// resident in CPU cache across calls, matching the real forward pass where
// 42 OTHER layers' work happens between two calls to the same layer).
//
// Does NOT touch moestream's real inference path -- completely standalone,
// zero risk to correctness of the live model. Verifies CPU and GPU produce
// numerically matching output for a small case before trusting either
// timing number.
//
// Build (no CMake integration, direct nvcc invocation):
//   nvcc -O3 -std=c++17 -Xcompiler "-O3 -mavx512f -fopenmp" \
//        tests/bench_outproj_cpu_vs_gpu.cu -o bench_outproj -lcublas -fopenmp
// Run:
//   ./bench_outproj
//
// Real shape, from the actual model (DeepSeek-V4-Flash, [DSV4-DEBUG] line):
//   H=64 nope=448 rope=64 qh=512 o_group_count=8 o_lora_rank=1024
//   group_in=(H*qh)/o_groups=4096
// Real run: 43 layers, 40 decode tokens/run -> CPU measured 6.8s total for
// this exact projection (t_attn_outproj minus t_attn_outproj_dequant).

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>
#include <random>
#include <cmath>
#include <omp.h>
#include <immintrin.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// ---- Real model shape ----
static constexpr int O_GROUPS  = 8;
static constexpr int O_RANK    = 1024;
static constexpr int GROUP_IN  = 4096;
static constexpr int N_LAYERS  = 43;
static constexpr int N_TOKENS  = 40;   // matches the real 40-token test runs
static constexpr int WARMUP_CYCLES = 1;

#define CHECK_CUDA(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
        exit(1); \
    } \
} while (0)
#define CHECK_CUBLAS(call) do { \
    cublasStatus_t _s = (call); \
    if (_s != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error %s:%d: status=%d\n", __FILE__, __LINE__, (int)_s); \
        exit(1); \
    } \
} while (0)

// ---- Exact same algorithm as core/attention_layer.cpp's S==1 outproj path
// (dot_f32_avx512 per output row, one combined OMP region over all
// (group, output) pairs) -- kept as a faithful standalone copy, not a
// reimplementation with different tuning, so this benchmark measures
// what's actually running in production. ----
static inline float dot_f32_avx512(const float* a, const float* b, int n) {
    __m512 acc0 = _mm512_setzero_ps(), acc1 = _mm512_setzero_ps();
    __m512 acc2 = _mm512_setzero_ps(), acc3 = _mm512_setzero_ps();
    int i = 0;
    for (; i + 64 <= n; i += 64) {
        acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i),      _mm512_loadu_ps(b + i),      acc0);
        acc1 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i + 16),  _mm512_loadu_ps(b + i + 16),  acc1);
        acc2 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i + 32),  _mm512_loadu_ps(b + i + 32),  acc2);
        acc3 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i + 48),  _mm512_loadu_ps(b + i + 48),  acc3);
    }
    __m512 acc = _mm512_add_ps(_mm512_add_ps(acc0, acc1), _mm512_add_ps(acc2, acc3));
    for (; i + 16 <= n; i += 16) acc = _mm512_fmadd_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i), acc);
    float sum = _mm512_reduce_add_ps(acc);
    for (; i < n; ++i) sum += a[i] * b[i];
    return sum;
}

static void cpu_outproj_one_token(const float* in_row, const float* WOA, float* out) {
    #pragma omp parallel for schedule(static)
    for (int go = 0; go < O_GROUPS * O_RANK; ++go) {
        int g = go / O_RANK, o = go % O_RANK;
        const float* in_group = in_row + (size_t)g * GROUP_IN;
        const float* w_row = WOA + (size_t)g * O_RANK * GROUP_IN + (size_t)o * GROUP_IN;
        out[go] = dot_f32_avx512(in_group, w_row, GROUP_IN);
    }
}

int main() {
    printf("=== outproj CPU vs GPU microbenchmark ===\n");
    printf("Shape: o_groups=%d o_rank=%d group_in=%d, %d layers x %d tokens\n",
           O_GROUPS, O_RANK, GROUP_IN, N_LAYERS, N_TOKENS);
    size_t per_layer_bytes = (size_t)O_GROUPS * O_RANK * GROUP_IN * sizeof(float);
    printf("Per-layer weight: %.1f MB, total resident (%d layers): %.2f GB\n",
           per_layer_bytes / 1e6, N_LAYERS, (per_layer_bytes * N_LAYERS) / 1e9);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-0.05f, 0.05f);

    // ---- Allocate N_LAYERS distinct weight matrices (host) ----
    printf("\nAllocating %d host weight buffers...\n", N_LAYERS);
    std::vector<std::vector<float>> woa_host(N_LAYERS);
    for (int l = 0; l < N_LAYERS; ++l) {
        woa_host[l].resize((size_t)O_GROUPS * O_RANK * GROUP_IN);
        for (auto& v : woa_host[l]) v = dist(rng);
    }
    std::vector<float> in_row(O_GROUPS * GROUP_IN);
    for (auto& v : in_row) v = dist(rng);
    std::vector<float> cpu_out(O_GROUPS * O_RANK);

    // ================= CPU BENCHMARK =================
    // Real access pattern: cycle through all 43 layers' distinct weight
    // buffers per "token" -- no layer's weight survives in cache between
    // two of its own calls (42 other layers' worth of data intervenes),
    // exactly like the real forward pass.
    printf("\n--- CPU (current production code path) ---\n");
    for (int w = 0; w < WARMUP_CYCLES; ++w)
        for (int l = 0; l < N_LAYERS; ++l)
            cpu_outproj_one_token(in_row.data(), woa_host[l].data(), cpu_out.data());

    auto cpu_t0 = std::chrono::high_resolution_clock::now();
    for (int t = 0; t < N_TOKENS; ++t)
        for (int l = 0; l < N_LAYERS; ++l)
            cpu_outproj_one_token(in_row.data(), woa_host[l].data(), cpu_out.data());
    auto cpu_t1 = std::chrono::high_resolution_clock::now();
    double cpu_total_s = std::chrono::duration<double>(cpu_t1 - cpu_t0).count();
    printf("CPU total (%d layers x %d tokens): %.3f s  (%.4f ms/call, %d calls)\n",
           N_LAYERS, N_TOKENS, cpu_total_s,
           1000.0 * cpu_total_s / (N_LAYERS * N_TOKENS), N_LAYERS * N_TOKENS);

    // ================= GPU BENCHMARK =================
    // Upload every layer's weight ONCE (not timed -- this mirrors a real
    // engine's one-time-per-layer VRAM cache load), then time only the
    // repeated small-upload -> GEMV -> small-download steady-state cost,
    // which is what a real decode token would actually pay once weights
    // are already resident.
    printf("\n--- GPU (cuBLAS, weights pre-resident in VRAM) ---\n");
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));
    CHECK_CUBLAS(cublasSetStream(handle, stream));

    std::vector<float*> woa_dev(N_LAYERS);
    for (int l = 0; l < N_LAYERS; ++l) {
        CHECK_CUDA(cudaMalloc(&woa_dev[l], per_layer_bytes));
        CHECK_CUDA(cudaMemcpy(woa_dev[l], woa_host[l].data(), per_layer_bytes, cudaMemcpyHostToDevice));
    }
    float* in_dev = nullptr;
    float* out_dev = nullptr;
    CHECK_CUDA(cudaMalloc(&in_dev, O_GROUPS * GROUP_IN * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&out_dev, O_GROUPS * O_RANK * sizeof(float)));
    // Pinned host staging, matching how this codebase's own real H2D/D2H
    // activation transfers work (pinned buffers, not plain malloc) --
    // an unpinned transfer would understate real achievable H2D/D2H speed.
    float* in_pinned = nullptr;
    float* out_pinned = nullptr;
    CHECK_CUDA(cudaMallocHost(&in_pinned, O_GROUPS * GROUP_IN * sizeof(float)));
    CHECK_CUDA(cudaMallocHost(&out_pinned, O_GROUPS * O_RANK * sizeof(float)));
    std::memcpy(in_pinned, in_row.data(), O_GROUPS * GROUP_IN * sizeof(float));

    const float alpha = 1.0f, beta = 0.0f;

    auto gpu_one_token = [&](int l) {
        CHECK_CUDA(cudaMemcpyAsync(in_dev, in_pinned, O_GROUPS * GROUP_IN * sizeof(float),
                                    cudaMemcpyHostToDevice, stream));
        for (int g = 0; g < O_GROUPS; ++g) {
            // Row-major W[O_RANK, GROUP_IN] reinterpreted as column-major
            // A'(m=GROUP_IN, n=O_RANK); CUBLAS_OP_T gives y = W @ x exactly
            // (see accompanying design notes) -- no data reshuffling needed.
            CHECK_CUBLAS(cublasSgemv(handle, CUBLAS_OP_T,
                GROUP_IN, O_RANK, &alpha,
                woa_dev[l] + (size_t)g * O_RANK * GROUP_IN, GROUP_IN,
                in_dev + (size_t)g * GROUP_IN, 1,
                &beta, out_dev + (size_t)g * O_RANK, 1));
        }
        CHECK_CUDA(cudaMemcpyAsync(out_pinned, out_dev, O_GROUPS * O_RANK * sizeof(float),
                                    cudaMemcpyDeviceToHost, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));
    };

    for (int w = 0; w < WARMUP_CYCLES; ++w)
        for (int l = 0; l < N_LAYERS; ++l) gpu_one_token(l);

    auto gpu_t0 = std::chrono::high_resolution_clock::now();
    for (int t = 0; t < N_TOKENS; ++t)
        for (int l = 0; l < N_LAYERS; ++l) gpu_one_token(l);
    auto gpu_t1 = std::chrono::high_resolution_clock::now();
    double gpu_total_s = std::chrono::duration<double>(gpu_t1 - gpu_t0).count();
    printf("GPU total (%d layers x %d tokens): %.3f s  (%.4f ms/call, %d calls)\n",
           N_LAYERS, N_TOKENS, gpu_total_s,
           1000.0 * gpu_total_s / (N_LAYERS * N_TOKENS), N_LAYERS * N_TOKENS);
    printf("(GPU one-time weight upload, NOT included above -- mirrors real\n"
           " engine behavior of uploading each layer's weight once at cache-fill time)\n");

    // ---- Correctness check: CPU and GPU must agree on the same input ----
    printf("\n--- Correctness check (layer 0, same input) ---\n");
    cpu_outproj_one_token(in_row.data(), woa_host[0].data(), cpu_out.data());
    gpu_one_token(0);
    double max_abs_diff = 0.0, max_rel_diff = 0.0;
    for (int i = 0; i < O_GROUPS * O_RANK; ++i) {
        double diff = std::fabs((double)cpu_out[i] - (double)out_pinned[i]);
        double rel = diff / (std::fabs((double)cpu_out[i]) + 1e-8);
        max_abs_diff = std::max(max_abs_diff, diff);
        max_rel_diff = std::max(max_rel_diff, rel);
    }
    printf("max abs diff=%.6e  max rel diff=%.6e  (%s)\n", max_abs_diff, max_rel_diff,
           (max_abs_diff < 1e-2 && max_rel_diff < 1e-2) ? "MATCH -- results are numerically equivalent"
                                                          : "MISMATCH -- do not trust the timing comparison above");

    // ================= SUMMARY =================
    printf("\n=== Summary (extrapolated to a real 43-layer x 40-token run) ===\n");
    printf("CPU: %.2f s   (real production measurement was 6.8s -- sanity check)\n", cpu_total_s);
    printf("GPU: %.2f s   (steady-state only, weight upload excluded)\n", gpu_total_s);
    printf("Speedup: %.2fx  %s\n", cpu_total_s / gpu_total_s,
           gpu_total_s < cpu_total_s ? "(GPU faster)" : "(CPU faster)");

    for (int l = 0; l < N_LAYERS; ++l) cudaFree(woa_dev[l]);
    cudaFree(in_dev); cudaFree(out_dev);
    cudaFreeHost(in_pinned); cudaFreeHost(out_pinned);
    cublasDestroy(handle);
    cudaStreamDestroy(stream);
    return 0;
}
