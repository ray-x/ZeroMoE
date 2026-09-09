// qwen4exp_gdn.cu -- chunkwise-parallel gated delta rule (GDN) on the GPU.
//
// This is a direct port of include/qwen4exp_gdn.hpp's qwen4exp_gdn_delta_chunk,
// which is itself DERIVED FROM (and unit-tested against) this project's own
// sequential qwen4exp_gdn_delta_step oracle -- deliberately not transcribed
// from ggml, so no index/layout convention has to be guessed. Read that
// function's header comment for the full derivation; the notation here is
// identical:
//
//   gcs[t]  = cumsum(g_log)[t],   A_t = exp(gcs[t])
//   ratio(t,i) = exp(gcs[t]-gcs[i]) = ecum[t]/ecum[i]   (t>=i, g<=0, so <= 1)
//   M[t][i] = ratio(t,i)*beta[i]*(k_t . k_i)   for i<t   (strictly lower)
//   (I+M) U = V - diag(A_t)(K @ S0)
//   o_t     = q_scale [ A_t (Q @ S0)[t] + sum_{i<=t} P[t][i] u_i ],
//             P[t][i] = ratio(t,i)*beta[i]*(q_t . k_i)   for i<=t
//   S_end   = A_{C-1} S0 + sum_i ratio(C-1,i)*beta_i k_i u_i^T
//
// Everything except the forward substitution is a batched GEMM (issued from
// CudaManager::qwen4exp_gdn_gpu_host); this file holds the pieces that are not.
//
// Heads: qwen4exp has H_v value heads and H_k key/query heads with
// hk = h % H_k, so Q/K are SHARED across H_v/H_k value heads. Rather than pay
// 16 small batched GEMMs to honour that non-uniform stride, gdn_gather_kernel
// materialises per-value-head contiguous [H_v, C, hd] copies once per chunk,
// after which every GEMM has a uniform batch stride and batch = H_v.

#include <cuda_runtime.h>

// Gather this chunk's Q/K/V rows into per-head contiguous [H_v, C, hd] blocks.
// grid (C, H_v), block hd threads.
__global__ void gdn_gather_kernel(
    const float* __restrict__ Q, const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ Qc, float* __restrict__ Kc, float* __restrict__ Vc,
    int s0, int S, int C, int hd, int H_k, int H_v)
{
    const int i = blockIdx.x;          // token within chunk
    const int h = blockIdx.y;          // value head
    if (i >= C || h >= H_v) return;
    const int s = s0 + i;
    if (s >= S) return;
    const int hk = h % H_k;
    const size_t dst = ((size_t)h * C + i) * hd;
    const size_t srcqk = ((size_t)s * H_k + hk) * hd;
    const size_t srcv  = ((size_t)s * H_v + h)  * hd;
    for (int d = threadIdx.x; d < hd; d += blockDim.x) {
        Qc[dst + d] = Q[srcqk + d];
        Kc[dst + d] = K[srcqk + d];
        Vc[dst + d] = V[srcv + d];
    }
}

// Per-head scalars: cumsum of g_log over the chunk, A_t, ecum, and the
// state-update coefficient cf[i] = ratio(C-1,i)*beta[i].
// One block per head, single-threaded cumsum (C <= 64, trivially cheap and
// keeps the prefix sum exactly in the CPU version's order).
//
// fast_exp mirrors the CPU version's own guard: ratio(t,i) is evaluated as
// ecum[t]/ecum[i] only while the chunk's total decay span stays inside float's
// exponent range, otherwise the exact per-pair exp is used, so a strongly
// decaying chunk can never divide by an underflowed zero.
__global__ void gdn_prep_scalars_kernel(
    const float* __restrict__ beta, const float* __restrict__ g_log,
    float* __restrict__ gcs, float* __restrict__ At,
    float* __restrict__ ecum, float* __restrict__ cf,
    float* __restrict__ betac, int* __restrict__ fast_exp_flag,
    int s0, int S, int C, int H_v)
{
    const int h = blockIdx.x;
    if (h >= H_v) return;
    float* g = gcs + (size_t)h * C;
    if (threadIdx.x == 0) {
        float acc = 0.0f;
        for (int t = 0; t < C; ++t) {
            const int s = s0 + t;
            const float gv = (s < S) ? g_log[(size_t)s * H_v + h] : 0.0f;
            acc += gv;
            g[t] = acc;
            betac[(size_t)h * C + t] = (s < S) ? beta[(size_t)s * H_v + h] : 0.0f;
        }
        const float span = g[0] - g[C - 1];
        const int fast = (span < 60.0f) ? 1 : 0;
        fast_exp_flag[h] = fast;
        for (int t = 0; t < C; ++t) {
            At[(size_t)h * C + t] = __expf(g[t]);
            ecum[(size_t)h * C + t] = fast ? __expf(g[t] - g[0]) : 0.0f;
        }
        for (int i = 0; i < C; ++i) {
            const float r = fast ? (ecum[(size_t)h * C + C - 1] / ecum[(size_t)h * C + i])
                                 : __expf(g[C - 1] - g[i]);
            cf[(size_t)h * C + i] = r * betac[(size_t)h * C + i];
        }
    }
}

__device__ __forceinline__ float gdn_ratio(const float* g, const float* e,
                                           int fast, int t, int i) {
    return fast ? (e[t] / e[i]) : __expf(g[t] - g[i]);
}

// M (strictly lower, [H_v,C,C]) from KK, and the RHS U0 = V - diag(A_t)*R.
// grid (C, H_v), block C threads for M plus a strided loop for U0.
__global__ void gdn_build_M_rhs_kernel(
    const float* __restrict__ KK, const float* __restrict__ Vc,
    const float* __restrict__ R, const float* __restrict__ gcs,
    const float* __restrict__ ecum, const float* __restrict__ At,
    const float* __restrict__ betac, const int* __restrict__ fast_flag,
    float* __restrict__ M, float* __restrict__ U0,
    int C, int hd)
{
    const int t = blockIdx.x;
    const int h = blockIdx.y;
    const float* g = gcs + (size_t)h * C;
    const float* e = ecum + (size_t)h * C;
    const float* b = betac + (size_t)h * C;
    const int fast = fast_flag[h];

    float* Mrow = M + ((size_t)h * C + t) * C;
    const float* KKrow = KK + ((size_t)h * C + t) * C;
    for (int i = threadIdx.x; i < C; i += blockDim.x)
        Mrow[i] = (i < t) ? gdn_ratio(g, e, fast, t, i) * b[i] * KKrow[i] : 0.0f;

    const float a = At[(size_t)h * C + t];
    const size_t off = ((size_t)h * C + t) * hd;
    for (int v = threadIdx.x; v < hd; v += blockDim.x)
        U0[off + v] = Vc[off + v] - a * R[off + v];
}

// Forward substitution: U[t] = U0[t] - sum_{i<t} M[t][i] U[i].
// One block per head; U is kept in shared memory (C*hd floats) so the O(C^2)
// dependent reads never leave the SM. Sequential in t by nature -- that is the
// irreducible part of the recurrence -- but every one of the hd columns and
// all H_v heads run in parallel, which is exactly what the CPU form could not do.
extern "C" __global__ void gdn_forward_subst_kernel(
    const float* __restrict__ M, const float* __restrict__ U0,
    float* __restrict__ U, int C, int hd)
{
    extern __shared__ float sU[];      // [C, hd]
    const int h = blockIdx.x;
    const size_t base = (size_t)h * C * hd;
    for (int idx = threadIdx.x; idx < C * hd; idx += blockDim.x)
        sU[idx] = U0[base + idx];
    __syncthreads();

    for (int t = 1; t < C; ++t) {
        const float* Mrow = M + ((size_t)h * C + t) * C;
        for (int v = threadIdx.x; v < hd; v += blockDim.x) {
            float acc = sU[(size_t)t * hd + v];
            for (int i = 0; i < t; ++i) {
                const float m = Mrow[i];
                if (m != 0.0f) acc -= m * sU[(size_t)i * hd + v];
            }
            sU[(size_t)t * hd + v] = acc;
        }
        __syncthreads();
    }
    for (int idx = threadIdx.x; idx < C * hd; idx += blockDim.x)
        U[base + idx] = sU[idx];
}

// P[t][i] = q_scale*ratio(t,i)*beta[i]*QK[t][i] for i<=t, and seed the output
// with q_scale*A_t*Qs[t] so the P@U GEMM can accumulate straight onto it
// (beta=1) instead of needing a separate add pass.
__global__ void gdn_build_P_out_kernel(
    const float* __restrict__ QK, const float* __restrict__ Qs,
    const float* __restrict__ gcs, const float* __restrict__ ecum,
    const float* __restrict__ At, const float* __restrict__ betac,
    const int* __restrict__ fast_flag,
    float* __restrict__ P, float* __restrict__ outc,
    int C, int hd, float q_scale)
{
    const int t = blockIdx.x;
    const int h = blockIdx.y;
    const float* g = gcs + (size_t)h * C;
    const float* e = ecum + (size_t)h * C;
    const float* b = betac + (size_t)h * C;
    const int fast = fast_flag[h];

    float* Prow = P + ((size_t)h * C + t) * C;
    const float* QKrow = QK + ((size_t)h * C + t) * C;
    for (int i = threadIdx.x; i < C; i += blockDim.x)
        Prow[i] = (i <= t) ? q_scale * gdn_ratio(g, e, fast, t, i) * b[i] * QKrow[i] : 0.0f;

    const float a = At[(size_t)h * C + t] * q_scale;
    const size_t off = ((size_t)h * C + t) * hd;
    for (int v = threadIdx.x; v < hd; v += blockDim.x)
        outc[off + v] = a * Qs[off + v];
}

// state *= A_{C-1} (per head), and Kcs[i][k] = cf[i]*Kc[i][k] for the rank-1
// accumulation GEMM that follows. Separate from that GEMM because cuBLAS's
// beta is one scalar for the whole batch while A_{C-1} differs per head.
__global__ void gdn_scale_state_kernel(
    const float* __restrict__ Kc, const float* __restrict__ cf,
    const float* __restrict__ At,
    float* __restrict__ state, float* __restrict__ Kcs,
    int C, int hd)
{
    const int h = blockIdx.y;
    const float alast = At[(size_t)h * C + (C - 1)];
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < hd * hd;
         idx += gridDim.x * blockDim.x)
        state[(size_t)h * hd * hd + idx] *= alast;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < C * hd;
         idx += gridDim.x * blockDim.x) {
        const int i = idx / hd;
        Kcs[(size_t)h * C * hd + idx] = cf[(size_t)h * C + i] * Kc[(size_t)h * C * hd + idx];
    }
}

// Scatter this chunk's [H_v, C, hd] output back into the [S, H_v, hd] buffer.
__global__ void gdn_scatter_kernel(
    const float* __restrict__ outc, float* __restrict__ out,
    int s0, int S, int C, int hd, int H_v)
{
    const int i = blockIdx.x, h = blockIdx.y;
    if (i >= C || h >= H_v) return;
    const int s = s0 + i;
    if (s >= S) return;
    const size_t src = ((size_t)h * C + i) * hd;
    const size_t dst = ((size_t)s * H_v + h) * hd;
    for (int d = threadIdx.x; d < hd; d += blockDim.x) out[dst + d] = outc[src + d];
}

extern "C" {

void launch_gdn_gather(const float* Q, const float* K, const float* V,
                       float* Qc, float* Kc, float* Vc,
                       int s0, int S, int C, int hd, int H_k, int H_v,
                       cudaStream_t st) {
    dim3 grid(C, H_v);
    gdn_gather_kernel<<<grid, 128, 0, st>>>(Q, K, V, Qc, Kc, Vc, s0, S, C, hd, H_k, H_v);
}

void launch_gdn_prep_scalars(const float* beta, const float* g_log,
                             float* gcs, float* At, float* ecum, float* cf,
                             float* betac, int* fast_flag,
                             int s0, int S, int C, int H_v, cudaStream_t st) {
    gdn_prep_scalars_kernel<<<H_v, 32, 0, st>>>(beta, g_log, gcs, At, ecum, cf,
                                                betac, fast_flag, s0, S, C, H_v);
}

void launch_gdn_build_M_rhs(const float* KK, const float* Vc, const float* R,
                            const float* gcs, const float* ecum, const float* At,
                            const float* betac, const int* fast_flag,
                            float* M, float* U0, int C, int hd, int H_v,
                            cudaStream_t st) {
    dim3 grid(C, H_v);
    gdn_build_M_rhs_kernel<<<grid, 128, 0, st>>>(KK, Vc, R, gcs, ecum, At, betac,
                                                 fast_flag, M, U0, C, hd);
}

// Returns false if this (C,hd) needs more shared memory than the device
// allows -- caller then keeps the whole layer on the CPU path.
bool launch_gdn_forward_subst(const float* M, const float* U0, float* U,
                              int C, int hd, int H_v, cudaStream_t st) {
    const size_t shmem = (size_t)C * hd * sizeof(float);
    int dev = 0; cudaGetDevice(&dev);
    int max_shmem = 0;
    cudaDeviceGetAttribute(&max_shmem, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    if (shmem > (size_t)max_shmem) return false;
    if (shmem > 48u * 1024u)
        cudaFuncSetAttribute(gdn_forward_subst_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem);
    gdn_forward_subst_kernel<<<H_v, 128, shmem, st>>>(M, U0, U, C, hd);
    return true;
}

void launch_gdn_build_P_out(const float* QK, const float* Qs, const float* gcs,
                            const float* ecum, const float* At, const float* betac,
                            const int* fast_flag, float* P, float* outc,
                            int C, int hd, int H_v, float q_scale, cudaStream_t st) {
    dim3 grid(C, H_v);
    gdn_build_P_out_kernel<<<grid, 128, 0, st>>>(QK, Qs, gcs, ecum, At, betac,
                                                 fast_flag, P, outc, C, hd, q_scale);
}

void launch_gdn_scale_state(const float* Kc, const float* cf, const float* At,
                            float* state, float* Kcs, int C, int hd, int H_v,
                            cudaStream_t st) {
    dim3 grid(64, H_v);
    gdn_scale_state_kernel<<<grid, 128, 0, st>>>(Kc, cf, At, state, Kcs, C, hd);
}

void launch_gdn_scatter(const float* outc, float* out, int s0, int S, int C,
                        int hd, int H_v, cudaStream_t st) {
    dim3 grid(C, H_v);
    gdn_scatter_kernel<<<grid, 128, 0, st>>>(outc, out, s0, S, C, hd, H_v);
}

}  // extern "C"
