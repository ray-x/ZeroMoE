// qwen4exp_attn.cu -- scale + causal-mask + row-softmax for qwen4exp's dense
// full-attention layers (EngineConfig::qwen4exp_attn_gpu).
//
// The QK^T and scores*V halves are plain cuBLAS strided-batched GEMMs issued
// from CudaManager::qwen4exp_attn_gpu_host (cuda_manager.hpp) -- this kernel
// is the one piece that is not a GEMM, mirroring how cuda/attn_score_softmax.cu
// splits GLM's MLA score phase.
//
// Deliberately a separate file from attn_score_softmax.cu: that kernel COMBINES
// two score matrices (nope+rope), which is MLA-specific. qwen4exp has a single
// score matrix, so reusing it would mean feeding a zero buffer and paying an
// extra full-size read for nothing.
//
// Layout: scores is [H, n_q, T] row-major, so row r = h*n_q + s_local.
// A query at chunk-local row s_local sits at absolute position
// q_base_abs + s_local and may attend to keys [0, q_base_abs + s_local].
// As in attn_score_softmax.cu, masked slots are never written with -inf --
// the loops simply stop at `valid` and the tail is zeroed once, so there is
// no inf arithmetic to reason about and fewer flops are done.

#include <cuda_runtime.h>

__global__ void qwen4exp_scale_softmax_causal_kernel(
    float* __restrict__ scores,
    int T, int n_q, int q_base_abs, float scale)
{
    const int r = blockIdx.x;              // r = h*n_q + s_local
    const int s_local = r % n_q;
    int valid = q_base_abs + s_local + 1;
    if (valid > T) valid = T;
    if (valid < 1) valid = 1;              // a row must attend to >= 1 key

    extern __shared__ float sh[];
    float* row = scores + (size_t)r * T;

    // scale + local max
    float local_max = -1e30f;
    for (int t = threadIdx.x; t < valid; t += blockDim.x) {
        const float v = row[t] * scale;
        row[t] = v;
        local_max = fmaxf(local_max, v);
    }
    sh[threadIdx.x] = local_max;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (threadIdx.x < off) sh[threadIdx.x] = fmaxf(sh[threadIdx.x], sh[threadIdx.x + off]);
        __syncthreads();
    }
    const float row_max = sh[0];
    __syncthreads();

    // exp + local sum
    float local_sum = 0.0f;
    for (int t = threadIdx.x; t < valid; t += blockDim.x) {
        const float e = __expf(row[t] - row_max);
        row[t] = e;
        local_sum += e;
    }
    sh[threadIdx.x] = local_sum;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (threadIdx.x < off) sh[threadIdx.x] += sh[threadIdx.x + off];
        __syncthreads();
    }
    const float inv = 1.0f / (sh[0] > 0.0f ? sh[0] : 1.0f);
    __syncthreads();

    for (int t = threadIdx.x; t < valid; t += blockDim.x) row[t] *= inv;
    // zero the masked tail so the scores*V GEMM (which runs over the full T)
    // contributes nothing from it
    for (int t = valid + threadIdx.x; t < T; t += blockDim.x) row[t] = 0.0f;
}

extern "C" void launch_qwen4exp_scale_softmax_causal(
    float* scores, int H, int n_q, int T, int q_base_abs, float scale,
    cudaStream_t stream)
{
    if (H <= 0 || n_q <= 0 || T <= 0) return;
    int threads = 256;
    if (T < threads) {                     // round down to a power of two >= 32
        threads = 32;
        while (threads * 2 <= T) threads *= 2;
    }
    const int blocks = H * n_q;
    const size_t shmem = (size_t)threads * sizeof(float);
    qwen4exp_scale_softmax_causal_kernel<<<blocks, threads, shmem, stream>>>(
        scores, T, n_q, q_base_abs, scale);
}

__global__ void qwen4exp_sigmoid_gate_kernel(
    float * __restrict__ x, const float * __restrict__ gate, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= 1.0f / (1.0f + expf(-gate[i]));
}

extern "C" void launch_qwen4exp_sigmoid_gate(
    float *x, const float *gate, int n, cudaStream_t stream) {
    if (!x || !gate || n <= 0) return;
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    qwen4exp_sigmoid_gate_kernel<<<blocks, threads, 0, stream>>>(x, gate, n);
}

// Prepare Qwen4Exp's full-attention Q/K/V on device after the three dense
// projections.  The old path downloaded q_full/K/V, did these operations on
// the CPU, and uploaded the same arrays again for the attention GEMMs.  This
// kernel keeps the prefill activation path on CUDA.  Qwen4Exp currently uses
// plain interleaved RoPE (no YaRN extension); the caller gates this fast path
// accordingly.
__global__ void qwen4exp_prepare_qkv_kernel(
    const float * __restrict__ q_full,
    const float * __restrict__ k_raw,
    const float * __restrict__ v_raw,
    float * __restrict__ q,
    float * __restrict__ gate,
    float * __restrict__ k,
    float * __restrict__ v,
    const float * __restrict__ q_norm,
    const float * __restrict__ k_norm,
    int S, int H, int Hkv, int hd, int n_rot, int pos_base,
    float theta_base, float eps) {
    const int q_rows = S * H;
    const int k_rows = S * Hkv;
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    // The CPU reference accumulates RMSNorm's sum of squares in double.  Use
    // the same accumulator width here so a close router boundary is not
    // changed merely by moving normalization to CUDA.
    __shared__ double partial[256];
    __shared__ float rope_cos[256];
    __shared__ float rope_sin[256];

    if (row < q_rows) {
        const int s = row / H;
        const float *src = q_full + (size_t)row * (2 * hd);
        float *dst_q = q + (size_t)row * hd;
        float *dst_gate = gate + (size_t)row * hd;

        double sum = 0.0;
        for (int i = tid; i < hd; i += blockDim.x) {
            const float x = src[i];
            sum += x * x;
            dst_gate[i] = src[hd + i];
        }
        partial[tid] = sum;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) partial[tid] += partial[tid + stride];
            __syncthreads();
        }
        const float inv = 1.0f / sqrtf((float)(partial[0] / (double)hd) + eps);
        for (int i = tid; i < hd; i += blockDim.x)
            dst_q[i] = src[i] * inv * q_norm[i];
        __syncthreads();

        const int rot_pairs = n_rot / 2;
        const float theta_scale = powf(theta_base, -2.0f / (float)hd);
        if (tid == 0) {
            float theta = (float)(pos_base + s);
            for (int j = 0; j < rot_pairs; ++j) {
                rope_cos[j] = cosf(theta);
                rope_sin[j] = sinf(theta);
                theta *= theta_scale;
            }
        }
        __syncthreads();
        for (int j = tid; j < rot_pairs; j += blockDim.x) {
            const float co = rope_cos[j];
            const float si = rope_sin[j];
            const float a = dst_q[2 * j];
            const float b = dst_q[2 * j + 1];
            dst_q[2 * j] = a * co - b * si;
            dst_q[2 * j + 1] = a * si + b * co;
        }
        return;
    }

    const int krow = row - q_rows;
    if (krow >= k_rows) return;
    const int s = krow / Hkv;
    const float *src_k = k_raw + (size_t)krow * hd;
    const float *src_v = v_raw + (size_t)krow * hd;
    float *dst_k = k + (size_t)krow * hd;
    float *dst_v = v + (size_t)krow * hd;

    double sum = 0.0;
    for (int i = tid; i < hd; i += blockDim.x) {
        const float x = src_k[i];
        sum += x * x;
        dst_v[i] = src_v[i];
    }
    partial[tid] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        __syncthreads();
    }
    const float inv = 1.0f / sqrtf((float)(partial[0] / (double)hd) + eps);
    for (int i = tid; i < hd; i += blockDim.x)
        dst_k[i] = src_k[i] * inv * k_norm[i];
    __syncthreads();

    const int rot_pairs = n_rot / 2;
    const float theta_scale = powf(theta_base, -2.0f / (float)hd);
    if (tid == 0) {
        float theta = (float)(pos_base + s);
        for (int j = 0; j < rot_pairs; ++j) {
            rope_cos[j] = cosf(theta);
            rope_sin[j] = sinf(theta);
            theta *= theta_scale;
        }
    }
    __syncthreads();
    for (int j = tid; j < rot_pairs; j += blockDim.x) {
        const float co = rope_cos[j];
        const float si = rope_sin[j];
        const float a = dst_k[2 * j];
        const float b = dst_k[2 * j + 1];
        dst_k[2 * j] = a * co - b * si;
        dst_k[2 * j + 1] = a * si + b * co;
    }
}

extern "C" void launch_qwen4exp_prepare_qkv(
    const float *q_full, const float *k_raw, const float *v_raw,
    float *q, float *gate, float *k, float *v,
    const float *q_norm, const float *k_norm,
    int S, int H, int Hkv, int hd, int n_rot, int pos_base,
    float theta_base, float eps, cudaStream_t stream) {
    if (!q_full || !k_raw || !v_raw || !q || !gate || !k || !v ||
        !q_norm || !k_norm || S <= 0 || H <= 0 || Hkv <= 0 || hd <= 0)
        return;
    const int rows = S * (H + Hkv);
    qwen4exp_prepare_qkv_kernel<<<rows, 256, 0, stream>>>(
        q_full, k_raw, v_raw, q, gate, k, v, q_norm, k_norm,
        S, H, Hkv, hd, n_rot, pos_base, theta_base, eps);
}
