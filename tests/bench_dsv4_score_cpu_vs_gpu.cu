// bench_dsv4_score_cpu_vs_gpu.cu — standalone, isolated feasibility/
// correctness microbenchmark for EngineConfig::dsv4_score_gpu.
//
// Same spirit as bench_score_cpu_vs_gpu.cu (GLM-5.2's MLA score bench) but
// for DeepSeek-V4's raw-SWA/HCA score phase (attention_layer.cpp's
// dsv4_attention_gguf, decode S==1 only). Does NOT touch moestream's real
// inference path or link cuda_manager.hpp -- both the CPU reference and the
// GPU implementation below are self-contained reimplementations, kept as
// close as practical to the production math so a divergence here is a real
// signal, not noise from unrelated production plumbing. Answers: (1) is the
// GPU path's output correct (max_abs_diff/max_rel_diff vs. the CPU
// reference, across raw-only, HCA, and sink-present cases), and (2) is it
// actually worth it at real dims, across a sweep of context lengths (i.e.
// growing hca_visible -- the genuinely unbounded part; the raw window
// itself is FIXED by `window`, unlike GLM-5.2's unbounded T).
//
// UNVERIFIED ON REAL HARDWARE (2026-08-31) -- this Mac has no CUDA
// toolchain. Run this on your Linux/CUDA box BEFORE enabling
// dsv4_score_gpu=true in production. If correctness fails, the bug is most
// likely in this file's GPU reimplementation OR in cuda_manager.hpp's
// dsv4_score_gpu_batched_host / kernels/models/dsv4_score_softmax.cu (which this file
// mirrors but does not directly call) -- diff the two once you have a
// concrete failing case.
//
// Build (no CMake integration, direct nvcc invocation):
//   nvcc -O3 -std=c++17 -Xcompiler "-O3 -mavx512f -fopenmp" \
//        tests/bench_dsv4_score_cpu_vs_gpu.cu -o bench_dsv4_score -lcublas -fopenmp
// Run:
//   ./bench_dsv4_score
//
// Adjust H/QH/WINDOW below to match your real checkpoint's own
// [DSV4-DEBUG] log line (H=..., nope=..., rope=..., window=...) before
// trusting the absolute timing numbers -- correctness does not depend on
// these being exact, only the speedup estimate does.

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

static constexpr int H      = 40;   // attention heads
static constexpr int QH     = 192;  // per-head latent width (nope+rope)
static constexpr int WINDOW = 2048; // raw-SWA sliding window (fixed, unlike T)
static constexpr int N_LAYERS = 60;

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

// ---- CPU reference: byte-for-byte the same shape as production's
// dsv4_attention_gguf score loop (attention_layer.cpp) for a single S==1
// decode token -- raw window (ring-order, oldest-to-newest) + HCA prefix +
// optional per-head sink, one joint softmax, weighted-V sum.
static void cpu_reference(
        const float* Q,            // [H,QH]
        const float* kvc_ring,     // [WINDOW,QH]
        int oldest_slot, int past_context_len,
        const float* hca_keys,     // [hca_visible,QH] or nullptr
        int hca_visible,
        const float* sinks,        // [H] or nullptr
        float scale,
        float* context)            // [H,QH] out
{
    const int T = past_context_len + hca_visible;
    #pragma omp parallel for
    for (int h = 0; h < H; ++h) {
        const float* q = Q + (size_t)h * QH;
        std::vector<float> scores((size_t)T + (sinks ? 1 : 0));
        for (int t = 0; t < past_context_len; ++t) {
            int slot = (oldest_slot + t) % WINDOW;
            const float* k = kvc_ring + (size_t)slot * QH;
            float s = 0.0f;
            for (int i = 0; i < QH; ++i) s += q[i] * k[i];
            scores[t] = s * scale;
        }
        for (int i = 0; i < hca_visible; ++i) {
            const float* k = hca_keys + (size_t)i * QH;
            float s = 0.0f;
            for (int d = 0; d < QH; ++d) s += q[d] * k[d];
            scores[past_context_len + i] = s * scale;
        }
        if (sinks) scores[T] = sinks[h];  // unscaled, matches production

        float max_s = -1e30f;
        for (float v : scores) max_s = std::max(max_s, v);
        float sum = 0.0f;
        for (float& v : scores) { v = std::exp(v - max_s); sum += v; }

        float* out = context + (size_t)h * QH;
        for (int i = 0; i < QH; ++i) out[i] = 0.0f;
        for (int t = 0; t < past_context_len; ++t) {
            int slot = (oldest_slot + t) % WINDOW;
            const float* k = kvc_ring + (size_t)slot * QH;
            float w = scores[t] / sum;
            for (int i = 0; i < QH; ++i) out[i] += w * k[i];
        }
        for (int i = 0; i < hca_visible; ++i) {
            const float* k = hca_keys + (size_t)i * QH;
            float w = scores[past_context_len + i] / sum;
            for (int d = 0; d < QH; ++d) out[d] += w * k[d];
        }
        // sink's own softmax mass has no value row -- already excluded above.
    }
}

// ---- GPU softmax kernel: byte-for-byte mirrors
// kernels/models/dsv4_score_softmax.cu's dsv4_score_softmax_kernel. Duplicated here
// (not #included) to keep this bench standalone per the project's existing
// bench_score_cpu_vs_gpu.cu convention -- if you change one, change both,
// or better, diff them after any edit to either.
__global__ void dsv4_score_softmax_kernel(
    float* __restrict__ scores, const float* __restrict__ sinks, int T, float scale)
{
    int h = blockIdx.x;
    extern __shared__ float sh[];
    float* row = scores + (size_t)h * T;
    const float sink_logit = sinks ? sinks[h] : -1e30f;

    float local_max = sink_logit;
    for (int t = threadIdx.x; t < T; t += blockDim.x) {
        float v = row[t] * scale;
        row[t] = v;
        local_max = fmaxf(local_max, v);
    }
    sh[threadIdx.x] = local_max;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sh[threadIdx.x] = fmaxf(sh[threadIdx.x], sh[threadIdx.x + s]);
        __syncthreads();
    }
    float row_max = sh[0];
    __syncthreads();

    float local_sum = (threadIdx.x == 0 && sinks) ? __expf(sink_logit - row_max) : 0.0f;
    for (int t = threadIdx.x; t < T; t += blockDim.x) {
        float e = __expf(row[t] - row_max);
        row[t] = e;
        local_sum += e;
    }
    sh[threadIdx.x] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    float row_sum = sh[0];
    __syncthreads();

    for (int t = threadIdx.x; t < T; t += blockDim.x) row[t] /= row_sum;
}

// GPU path: same shape as CudaManager::dsv4_score_gpu_batched_host --
// compact the (possibly-wrapped) ring + HCA prefix into one contiguous
// [T,QH] buffer, 2 GEMMs + 1 softmax kernel.
static void gpu_score(
        cublasHandle_t handle,
        const float* d_kvc_ring, int oldest_slot, int past_context_len,
        const float* d_hca_keys, int hca_visible,
        const float* d_Q, const float* d_sinks, float scale,
        float* d_kv_scratch, float* d_scores, float* d_context)
{
    const int T = past_context_len + hca_visible;
    const int tail_len = std::min(past_context_len, WINDOW - oldest_slot);
    const int head_len = past_context_len - tail_len;
    CHECK_CUDA(cudaMemcpy(d_kv_scratch, d_kvc_ring + (size_t)oldest_slot * QH,
                           (size_t)tail_len * QH * sizeof(float), cudaMemcpyDeviceToDevice));
    if (head_len > 0) {
        CHECK_CUDA(cudaMemcpy(d_kv_scratch + (size_t)tail_len * QH, d_kvc_ring,
                               (size_t)head_len * QH * sizeof(float), cudaMemcpyDeviceToDevice));
    }
    if (hca_visible > 0) {
        CHECK_CUDA(cudaMemcpy(d_kv_scratch + (size_t)past_context_len * QH, d_hca_keys,
                               (size_t)hca_visible * QH * sizeof(float), cudaMemcpyDeviceToDevice));
    }

    const float alpha = 1.0f, beta = 0.0f;
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, T, H, QH, &alpha,
                              d_kv_scratch, QH, d_Q, QH, &beta, d_scores, T));
    int threads = 256;
    dsv4_score_softmax_kernel<<<H, threads, threads * sizeof(float)>>>(d_scores, d_sinks, T, scale);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, QH, H, T, &alpha,
                              d_kv_scratch, QH, d_scores, T, &beta, d_context, QH));
}

static double max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    double m = 0.0;
    for (size_t i = 0; i < a.size(); ++i) m = std::max(m, (double)std::fabs(a[i] - b[i]));
    return m;
}
static double max_rel_diff(const std::vector<float>& a, const std::vector<float>& b) {
    double m = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double denom = std::max(1e-6, (double)std::fabs(a[i]));
        m = std::max(m, std::fabs((double)a[i] - (double)b[i]) / denom);
    }
    return m;
}

int main() {
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> kvc_ring((size_t)WINDOW * QH);
    for (auto& v : kvc_ring) v = dist(rng);
    std::vector<float> Q((size_t)H * QH);
    for (auto& v : Q) v = dist(rng);
    std::vector<float> sinks(H);
    for (auto& v : sinks) v = dist(rng);
    const float scale = 1.0f / std::sqrt((float)QH);

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    float *d_kvc, *d_hca, *d_Q, *d_sinks, *d_kv_scratch, *d_scores, *d_context;
    CHECK_CUDA(cudaMalloc(&d_kvc, (size_t)WINDOW * QH * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_kvc, kvc_ring.data(), (size_t)WINDOW * QH * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMalloc(&d_Q, (size_t)H * QH * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_Q, Q.data(), (size_t)H * QH * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMalloc(&d_sinks, (size_t)H * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_sinks, sinks.data(), (size_t)H * sizeof(float), cudaMemcpyHostToDevice));
    // HCA cache: size to the largest hca_visible we'll test below.
    const int MAX_HCA = 4096;
    std::vector<float> hca_keys((size_t)MAX_HCA * QH);
    for (auto& v : hca_keys) v = dist(rng);
    CHECK_CUDA(cudaMalloc(&d_hca, (size_t)MAX_HCA * QH * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_hca, hca_keys.data(), (size_t)MAX_HCA * QH * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMalloc(&d_kv_scratch, (size_t)(WINDOW + MAX_HCA) * QH * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_scores, (size_t)H * (WINDOW + MAX_HCA) * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_context, (size_t)H * QH * sizeof(float)));

    struct Case { const char* name; int oldest_slot; int past_context_len; int hca_visible; bool sinks_on; };
    std::vector<Case> cases = {
        {"raw-only, unwrapped, no sink",   0,    512,  0,    false},
        {"raw-only, wrapped, no sink",     WINDOW - 100, WINDOW, 0,    false},
        {"raw-only, wrapped, with sink",   WINDOW - 100, WINDOW, 0,    true},
        {"HCA small, no sink",             WINDOW - 100, WINDOW, 32,   false},
        {"HCA large, with sink",           WINDOW - 100, WINDOW, 2048, true},
    };

    printf("Correctness (H=%d QH=%d WINDOW=%d):\n", H, QH, WINDOW);
    bool all_pass = true;
    for (auto& c : cases) {
        std::vector<float> ctx_cpu((size_t)H * QH), ctx_gpu((size_t)H * QH);
        cpu_reference(Q.data(), kvc_ring.data(), c.oldest_slot, c.past_context_len,
                      c.hca_visible > 0 ? hca_keys.data() : nullptr, c.hca_visible,
                      c.sinks_on ? sinks.data() : nullptr, scale, ctx_cpu.data());
        gpu_score(handle, d_kvc, c.oldest_slot, c.past_context_len,
                  c.hca_visible > 0 ? d_hca : nullptr, c.hca_visible,
                  d_Q, c.sinks_on ? d_sinks : nullptr, scale,
                  d_kv_scratch, d_scores, d_context);
        CHECK_CUDA(cudaMemcpy(ctx_gpu.data(), d_context, (size_t)H * QH * sizeof(float), cudaMemcpyDeviceToHost));
        double ad = max_abs_diff(ctx_cpu, ctx_gpu), rd = max_rel_diff(ctx_cpu, ctx_gpu);
        bool pass = ad < 1e-3 && rd < 0.01;
        all_pass &= pass;
        printf("  %-32s max_abs_diff=%.2e max_rel_diff=%.4f  %s\n",
               c.name, ad, rd, pass ? "PASSED" : "FAILED");
    }
    if (!all_pass) {
        fprintf(stderr, "\nCORRECTNESS FAILED -- do NOT enable dsv4_score_gpu until this passes.\n");
        return 1;
    }

    printf("\nTiming (decode-shaped, single-call latency, %d layers/token estimate):\n", N_LAYERS);
    std::vector<int> hca_sweep = {0, 32, 256, 1024, 4096};
    const int oldest_slot = WINDOW / 3;
    const int past_context_len = WINDOW;
    for (int hca_visible : hca_sweep) {
        std::vector<float> ctx_cpu((size_t)H * QH), ctx_gpu((size_t)H * QH);

        const int reps = 50;
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int r = 0; r < reps; ++r) {
            cpu_reference(Q.data(), kvc_ring.data(), oldest_slot, past_context_len,
                          hca_visible > 0 ? hca_keys.data() : nullptr, hca_visible,
                          sinks.data(), scale, ctx_cpu.data());
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / reps;

        auto t2 = std::chrono::high_resolution_clock::now();
        for (int r = 0; r < reps; ++r) {
            gpu_score(handle, d_kvc, oldest_slot, past_context_len,
                      hca_visible > 0 ? d_hca : nullptr, hca_visible,
                      d_Q, d_sinks, scale, d_kv_scratch, d_scores, d_context);
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        auto t3 = std::chrono::high_resolution_clock::now();
        double gpu_ms = std::chrono::duration<double, std::milli>(t3 - t2).count() / reps;

        printf("  hca_visible=%5d (T=%5d): CPU=%7.3fms/call  GPU=%7.3fms/call  speedup=%.2fx  "
               "(est. %d-layer decode cost: CPU=%.1fms GPU=%.1fms)\n",
               hca_visible, past_context_len + hca_visible, cpu_ms, gpu_ms, cpu_ms / gpu_ms,
               N_LAYERS, cpu_ms * N_LAYERS, gpu_ms * N_LAYERS);
    }

    cublasDestroy(handle);
    cudaFree(d_kvc); cudaFree(d_hca); cudaFree(d_Q); cudaFree(d_sinks);
    cudaFree(d_kv_scratch); cudaFree(d_scores); cudaFree(d_context);
    return 0;
}
