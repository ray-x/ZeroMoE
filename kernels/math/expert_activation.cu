// expert_activation.cu -- SwiGLU (SiLU * up, with optional clamp) for the
// routed-expert FFN, computed ON THE DEVICE.
//
// WHY: expert_ffn_gpu_finish_stage1 used to download BOTH gate and up to the
// host, run silu/elem_mul/clamp on the CPU, and upload the product back --
// two extra PCIe crossings per expert, ~12,000 experts per 512-token prefill.
// Measured dn=13.68s of a 48s prefill, against cmp=0.20s for the expert
// matmuls themselves. llama.cpp keeps the whole FFN in one GPU graph; its
// expert weights cross PCIe once and nothing returns. This closes that gap.
//
// The CPU path's own comment called the SiLU "cheap" -- the ARITHMETIC is;
// the transfers wrapped around it were not.
//
// PRESERVES THE NaN GUARD. check_finite() on the host is unconditional (not
// trace-gated) and is what caught real corruption in this engine, so removing
// the download must not remove the check. The kernel counts non-finite gate/up
// values into two device counters; the caller reads back 8 BYTES instead of
// 2 * S * moe_inter floats and reports identically.
//
// Two activation variants, matching the CPU exactly:
//   fused (GLM5Next ggml_swiglu_clamp): g=min(g,limit); u=clamp(u,+/-limit);
//                                       out = g/(1+exp(-g)) * u
//   default:                            out = clamp(silu(g) * u, +/-clamp)
// expf (not __expf) deliberately: the CPU path uses std::exp, and this is a
// numerically sensitive path with a NaN-cascade history -- the fast
// approximate intrinsic is not worth the divergence.

#include <cuda_runtime.h>
#include <math_constants.h>

__global__ void expert_swiglu_kernel(float* __restrict__ prod,
                                     const float* __restrict__ gate,
                                     const float* __restrict__ up,
                                     int n, float clamp, int fused,
                                     int* __restrict__ bad)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float g0 = gate[i];
    const float u0 = up[i];

    // same accounting check_finite() does on the host
    if (isnan(g0) || isnan(u0)) atomicAdd(&bad[0], 1);
    else if (isinf(g0) || isinf(u0)) atomicAdd(&bad[1], 1);

    float g = g0, u = u0, out;
    if (fused) {
        if (clamp > 0.0f) {
            g = fminf(g, clamp);
            u = fmaxf(-clamp, fminf(clamp, u));
        }
        out = g / (1.0f + expf(-g)) * u;
    } else {
        g = g / (1.0f + expf(-g));
        out = g * u;
        if (clamp > 0.0f) out = fmaxf(-clamp, fminf(clamp, out));
    }
    prod[i] = out;
}

extern "C" void launch_expert_swiglu(float* prod, const float* gate, const float* up,
                                     int n, float clamp, int fused, int* d_bad,
                                     cudaStream_t stream)
{
    if (n <= 0) return;
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    expert_swiglu_kernel<<<blocks, threads, 0, stream>>>(prod, gate, up, n,
                                                         clamp, fused, d_bad);
}
