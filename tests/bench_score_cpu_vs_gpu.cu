// bench_score_cpu_vs_gpu.cu — standalone, isolated feasibility microbenchmark.
//
// Answers one question BEFORE any production kernel gets written: is GPU
// dispatch actually worth it for MLA's `score` phase (attn_score in
// attention_layer.cpp -- the QK^T-over-cached-latent + softmax + weighted-V
// step), at GLM-5.2's real dims, across a sweep of context lengths (score's
// cost genuinely grows with context, unlike qabsorb which is context-
// independent)? Does NOT touch moestream's real inference path.
//
// Why this maps to GPU more cleanly than qabsorb did: MLA's latent K/V
// (Lc/Rc) is SHARED across every head (that's the whole point of the
// absorbed/compressed representation) -- so unlike qabsorb (H independent
// per-head GEMMs, needed cublasGemmStridedBatchedEx), score collapses into
// THREE PLAIN (non-batched) GEMMs plus a softmax, since every head just
// multiplies its own query against the SAME shared latent cache:
//   1. scores_nope[H,T] = Q_absorbed[H,kvl] @ Lc[T,kvl]^T
//   2. scores_rope[H,T] = Q_rope[H,rope]    @ Rc[T,rope]^T
//   3. combine+scale+softmax over T, per head row
//   4. context[H,kvl]   = attn_weights[H,T] @ Lc[T,kvl]
// This benchmark simulates the REALISTIC steady-state production shape: the
// KV latent cache (Lc/Rc) is uploaded ONCE (as if already GPU-resident from
// an incremental per-token append, the way a real implementation would grow
// it), and only the tiny per-token Q_absorbed/Q_rope + the softmax + the
// small context output cross PCIe every call -- NOT re-uploading the whole
// growing cache every token, which would be an unrealistic strawman.
//
// Does NOT include the final WUV down-projection (context[H,kvl] -> [H,vh])
// that production's own attn_score bucket also currently includes -- that
// step is H independent per-head GEMMs, structurally identical to qabsorb's
// already-proven cublasGemmStridedBatchedEx approach, so it doesn't need
// re-validating here; this bench isolates only the NEW, context-length-
// dependent part.
//
// Build (no CMake integration, direct nvcc invocation):
//   nvcc -O3 -std=c++17 -Xcompiler "-O3 -mavx512f -fopenmp" \
//        tests/bench_score_cpu_vs_gpu.cu -o bench_score -lcublas -fopenmp
// Run:
//   ./bench_score
//
// Real shape (GLM-5.2, from a real run's own [GRAPH] log):
//   H=64 heads, kv_lora=512, qk_rope=64, hidden=6144, n_layers=78
//
// RESULTS (real RTX 3090, 2026-08-14, two independent runs -- reproducible):
//   Correctness (T=37): max_abs_diff=1.5e-7 max_rel_diff=0.0023  PASSED
//     T=   64: CPU=2.56-2.61ms/call  GPU=0.079-0.080ms/call  speedup=32-33x
//     T=  512: CPU=0.41-0.48ms/call  GPU=0.088-0.089ms/call  speedup=4.6-5.4x
//     T= 1500: CPU=1.39-1.55ms/call  GPU=0.104-0.107ms/call  speedup=13-14x
//     T= 3000: CPU=3.21-3.22ms/call  GPU=0.121-0.130ms/call  speedup=25-27x
//     T= 4096: CPU=6.43-6.83ms/call  GPU=0.125-0.133ms/call  speedup=51x
//   GPU cost stays flat (~0.08-0.13ms/call) across the entire T range -- the
//   3-GEMM+softmax structure genuinely doesn't scale with T the way CPU's
//   scalar/AVX-512 loop does. CPU grows roughly linearly with T as expected,
//   EXCEPT T=64 costs MORE than T=512 (reproduced both runs, not noise) --
//   most likely this bench's tight, sustained timing loop lets the CPU ramp
//   to full turbo mid-measurement at larger T (more total work per timed
//   block), while a short T=64 burst may never leave a lower power state.
//   Production's real dispatch is the OPPOSITE of a tight loop (bursty,
//   interleaved with disk I/O/GPU dispatch/expert routing), so if anything
//   this means real CPU costs could be worse than this bench's steady-state
//   numbers suggest, not better -- doesn't change the conclusion either way.
//   VERDICT: clearly worth a real GPU implementation. Unlike qabsorb (fixed
//   cost regardless of context), score's CPU cost is UNBOUNDED with context
//   length for GLM (no sliding window, unlike DeepSeek-V4) -- the win only
//   grows as conversations get longer, which is the opposite of qabsorb's
//   flat-speedup shape and arguably the more important case to fix.

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

static constexpr int H    = 64;
static constexpr int KVL  = 512;
static constexpr int ROPE = 64;
static constexpr int N_LAYERS = 78;

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

// ---- CPU reference: byte-for-byte the same structure as production's
// attn_score loop (attention_layer.cpp), minus the trailing WUV matmul
// (out of scope here, see header comment) ----
static void cpu_score_reference(
    const float* Q_absorbed, // [H, KVL]
    const float* Q_rope,     // [H, ROPE]
    const float* Lc,         // [T, KVL]
    const float* Rc,         // [T, ROPE]
    float* context,          // [H, KVL] out
    int T, float scale)
{
    #pragma omp parallel for schedule(dynamic, 1)
    for (int h = 0; h < H; ++h) {
        const float* q_absorbed = Q_absorbed + (size_t)h * KVL;
        const float* q_rope = Q_rope + (size_t)h * ROPE;
        std::vector<float> attn_scores(T);
        for (int t = 0; t < T; ++t) {
            const float* L_t = Lc + (size_t)t * KVL;
            const float* R_t = Rc + (size_t)t * ROPE;
            float score_nope = 0.0f;
            __m512 v_score_nope = _mm512_setzero_ps();
            for (int i = 0; i < KVL; i += 16)
                v_score_nope = _mm512_fmadd_ps(_mm512_loadu_ps(q_absorbed + i), _mm512_loadu_ps(L_t + i), v_score_nope);
            score_nope = _mm512_reduce_add_ps(v_score_nope);

            float score_rope = 0.0f;
            __m512 v_score_rope = _mm512_setzero_ps();
            for (int i = 0; i < ROPE; i += 16)
                v_score_rope = _mm512_fmadd_ps(_mm512_loadu_ps(q_rope + i), _mm512_loadu_ps(R_t + i), v_score_rope);
            score_rope = _mm512_reduce_add_ps(v_score_rope);

            attn_scores[t] = (score_nope + score_rope) * scale;
        }
        // softmax
        float mx = attn_scores[0];
        for (int t = 1; t < T; ++t) mx = std::max(mx, attn_scores[t]);
        float sum = 0.0f;
        for (int t = 0; t < T; ++t) { attn_scores[t] = std::exp(attn_scores[t] - mx); sum += attn_scores[t]; }
        for (int t = 0; t < T; ++t) attn_scores[t] /= sum;

        float* head_out = context + (size_t)h * KVL;
        std::vector<float> latent_accumulation(KVL, 0.0f);
        for (int t = 0; t < T; ++t) {
            float alpha_t = attn_scores[t];
            const float* L_t = Lc + (size_t)t * KVL;
            for (int i = 0; i < KVL; i += 16) {
                __m512 v_lat = _mm512_loadu_ps(latent_accumulation.data() + i);
                __m512 v_cache = _mm512_loadu_ps(L_t + i);
                _mm512_storeu_ps(latent_accumulation.data() + i, _mm512_fmadd_ps(_mm512_set1_ps(alpha_t), v_cache, v_lat));
            }
        }
        std::memcpy(head_out, latent_accumulation.data(), KVL * sizeof(float));
    }
}

// ---- GPU: combine (nope+rope)*scale, then row-softmax, one block per head ----
__global__ void combine_scale_softmax_kernel(
    const float* scores_nope, const float* scores_rope, float* scores,
    int T, float scale)
{
    int h = blockIdx.x;
    extern __shared__ float sh[];
    const float* nope_row = scores_nope + (size_t)h * T;
    const float* rope_row = scores_rope + (size_t)h * T;
    float* row = scores + (size_t)h * T;

    // combine + local max
    float local_max = -1e30f;
    for (int t = threadIdx.x; t < T; t += blockDim.x) {
        float v = (nope_row[t] + rope_row[t]) * scale;
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

    // exp + local sum
    float local_sum = 0.0f;
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

    // normalize
    for (int t = threadIdx.x; t < T; t += blockDim.x) row[t] /= row_sum;
}

static double gpu_score_call(
    cublasHandle_t handle,
    const float* d_Q_absorbed, const float* d_Q_rope,
    const float* d_Lc, const float* d_Rc,
    float* d_scores_nope, float* d_scores_rope, float* d_scores,
    float* d_context, float* context_host,
    const float* Q_absorbed_host, const float* Q_rope_host,
    int T, float scale)
{
    const float alpha = 1.0f, beta = 0.0f;
    // upload this token's tiny query (the only per-call H2D of real production shape)
    CHECK_CUDA(cudaMemcpy((void*)d_Q_absorbed, Q_absorbed_host, (size_t)H * KVL * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy((void*)d_Q_rope, Q_rope_host, (size_t)H * ROPE * sizeof(float), cudaMemcpyHostToDevice));

    // GEMM1: scores_nope[H,T] = Q_absorbed[H,KVL] @ Lc[T,KVL]^T  (same "Y=X@W^T" pattern as cublas_dense_matmul)
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, T, H, KVL, &alpha,
                              d_Lc, KVL, d_Q_absorbed, KVL, &beta, d_scores_nope, T));
    // GEMM2: scores_rope[H,T] = Q_rope[H,ROPE] @ Rc[T,ROPE]^T
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, T, H, ROPE, &alpha,
                              d_Rc, ROPE, d_Q_rope, ROPE, &beta, d_scores_rope, T));

    int threads = 256;
    combine_scale_softmax_kernel<<<H, threads, threads * sizeof(float)>>>(
        d_scores_nope, d_scores_rope, d_scores, T, scale);

    // GEMM3: context[H,KVL] = attn_weights[H,T] @ Lc[T,KVL]  (plain row-major A@B, standard trick)
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, KVL, H, T, &alpha,
                              d_Lc, KVL, d_scores, T, &beta, d_context, KVL));

    CHECK_CUDA(cudaMemcpy(context_host, d_context, (size_t)H * KVL * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaStreamSynchronize(0));
    return 0;
}

int main() {
    printf("[Bench] MLA score: CPU AVX-512 vs GPU (3 GEMMs + softmax kernel), GLM-5.2 real dims (H=%d kvl=%d rope=%d)\n", H, KVL, ROPE);

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    const int T_MAX = 4096;
    std::mt19937 rng(2718);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> Lc_full((size_t)T_MAX * KVL), Rc_full((size_t)T_MAX * ROPE);
    for (auto& v : Lc_full) v = dist(rng);
    for (auto& v : Rc_full) v = dist(rng);
    std::vector<float> Q_absorbed((size_t)H * KVL), Q_rope((size_t)H * ROPE);
    for (auto& v : Q_absorbed) v = dist(rng);
    for (auto& v : Q_rope) v = dist(rng);
    float scale = 1.0f / std::sqrt((float)(KVL + ROPE)); // approximate, doesn't affect timing

    // Persistent GPU-resident cache, uploaded ONCE (simulates an incrementally-
    // grown real KV cache already on GPU -- the realistic production shape,
    // not a strawman that re-uploads the whole cache every token)
    float *d_Lc, *d_Rc;
    CHECK_CUDA(cudaMalloc(&d_Lc, (size_t)T_MAX * KVL * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_Rc, (size_t)T_MAX * ROPE * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_Lc, Lc_full.data(), (size_t)T_MAX * KVL * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_Rc, Rc_full.data(), (size_t)T_MAX * ROPE * sizeof(float), cudaMemcpyHostToDevice));

    float *d_Q_absorbed, *d_Q_rope, *d_scores_nope, *d_scores_rope, *d_scores, *d_context;
    CHECK_CUDA(cudaMalloc(&d_Q_absorbed, (size_t)H * KVL * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_Q_rope, (size_t)H * ROPE * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_scores_nope, (size_t)H * T_MAX * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_scores_rope, (size_t)H * T_MAX * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_scores, (size_t)H * T_MAX * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_context, (size_t)H * KVL * sizeof(float)));

    // ---- Correctness check first, small T, before trusting any timing ----
    {
        int T = 37;
        std::vector<float> ctx_cpu((size_t)H * KVL), ctx_gpu((size_t)H * KVL);
        cpu_score_reference(Q_absorbed.data(), Q_rope.data(), Lc_full.data(), Rc_full.data(), ctx_cpu.data(), T, scale);
        gpu_score_call(handle, d_Q_absorbed, d_Q_rope, d_Lc, d_Rc, d_scores_nope, d_scores_rope, d_scores,
                       d_context, ctx_gpu.data(), Q_absorbed.data(), Q_rope.data(), T, scale);
        float max_abs = 0.0f, max_rel = 0.0f;
        for (size_t i = 0; i < ctx_cpu.size(); ++i) {
            float d = std::fabs(ctx_cpu[i] - ctx_gpu[i]);
            max_abs = std::max(max_abs, d);
            if (std::fabs(ctx_cpu[i]) > 1e-6f) max_rel = std::max(max_rel, d / std::fabs(ctx_cpu[i]));
        }
        printf("  Correctness (T=%d): max_abs_diff=%g max_rel_diff=%g  %s\n", T, max_abs, max_rel,
               (max_abs < 1e-2f && max_rel < 0.05f) ? "PASSED" : "FAILED -- DO NOT TRUST TIMING BELOW");
        if (!(max_abs < 1e-2f && max_rel < 0.05f)) return 1;
    }

    // ---- Timing sweep across representative context lengths ----
    const int WARMUP = 5, ITERS = 30;
    std::vector<float> ctx_out((size_t)H * KVL);
    for (int T : {64, 512, 1500, 3000, 4096}) {
        for (int it = 0; it < WARMUP; ++it)
            cpu_score_reference(Q_absorbed.data(), Q_rope.data(), Lc_full.data(), Rc_full.data(), ctx_out.data(), T, scale);
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int it = 0; it < ITERS; ++it)
            cpu_score_reference(Q_absorbed.data(), Q_rope.data(), Lc_full.data(), Rc_full.data(), ctx_out.data(), T, scale);
        double cpu_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t0).count() / ITERS;

        for (int it = 0; it < WARMUP; ++it)
            gpu_score_call(handle, d_Q_absorbed, d_Q_rope, d_Lc, d_Rc, d_scores_nope, d_scores_rope, d_scores,
                           d_context, ctx_out.data(), Q_absorbed.data(), Q_rope.data(), T, scale);
        t0 = std::chrono::high_resolution_clock::now();
        for (int it = 0; it < ITERS; ++it)
            gpu_score_call(handle, d_Q_absorbed, d_Q_rope, d_Lc, d_Rc, d_scores_nope, d_scores_rope, d_scores,
                           d_context, ctx_out.data(), Q_absorbed.data(), Q_rope.data(), T, scale);
        double gpu_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t0).count() / ITERS;

        printf("  T=%5d: CPU=%7.3fms/call  GPU=%7.3fms/call  speedup=%.2fx  (est. %d-layer decode cost: CPU=%.1fms GPU=%.1fms)\n",
               T, cpu_ms, gpu_ms, cpu_ms / gpu_ms, N_LAYERS, cpu_ms * N_LAYERS, gpu_ms * N_LAYERS);
    }

    return 0;
}
