// q5_1_matmul.cu — fused Q5_1 dequant + matmul: y[S,O] = x[S,I] @ W^T
//
// Added 2026-09-04 for qwen4exp (Qwen3.8-Flash-Next). That checkpoint's
// Unsloth UD-Q4_K_XL build stores ffn_down_exps as Q5_1 on 43 of its 48 MoE
// layers, and kquant_dequant.hpp's gpu_type_supported() is an AND over
// gate/up/down -- so without this kernel those layers' ENTIRE routed-expert
// compute was pinned to the CPU AVX-512 path, at the ~9-10x per-call tax
// this project has measured elsewhere (see CLAUDE.md's cpu_offload_max_
// experts investigation). Measured before this landed: expert_calls=
// 1200GPU/10320CPU with expert= at 93% of request wall time.
//
// Q5_1 is the simplest format in this directory: no grid codebook, no sign
// table, no super-block scale hierarchy. Per 32-element block: an FP16 scale
// d, an FP16 minimum m, a 32-bit qh carrying each weight's 5th bit, and 16
// bytes of packed 4-bit nibbles. Dequant is a plain affine map,
// w = (nibble | hi<<4) * d + m.
//
// EXACT reference (ggml/src/ggml-quants.c, dequantize_row_q5_1):
//     xh_0 = ((qh >> (j +  0)) << 4) & 0x10;   x0 = (qs[j] & 0x0F) | xh_0;
//     xh_1 = ((qh >> (j + 12))     ) & 0x10;   x1 = (qs[j] >>   4) | xh_1;
//     y[j] = x0*d + m;   y[j + 16] = x1*d + m;
// Note (qh >> (j+12)) & 0x10 selects bit (j+16), so element k of the block
// uses bit k of qh for every k in [0,32) -- that uniformity is what the
// inner loop below relies on, and it was derived from the shifts above
// rather than assumed.
//
// The affine form lets each block collapse to two accumulators instead of
// reconstructing 32 weights:
//     sum_k w_k*x_k = d * sum_k q_k*x_k  +  m * sum_k x_k
// which is the same algebraic shortcut ggml's own Q5_1 dot product uses.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

// Real on-disk GGUF layout, matching ggml-common.h's block_q5_1:
//   ggml_half d; ggml_half m; uint8_t qh[4]; uint8_t qs[QK5_1/2];
// static_assert(sizeof(block_q5_1) == 2*sizeof(ggml_half) + sizeof(uint32_t)
//               + QK5_1/2) == 24. qh is declared as 4 bytes (NOT a uint32)
// deliberately: that is how ggml declares it, it keeps the struct's
// alignment requirement at 2, and it means the byte-wise assembly below is
// alignment-safe by construction rather than by an argument about where the
// pointer happens to land. (24 is a multiple of 8, so every block base here
// is in fact 4-byte aligned and a uint32 load would be legal -- but the
// Xid 13 "Misaligned Address" incident documented in q8_0_matmul.cu came
// from exactly this kind of reasoning being wrong for a 34-byte block, so
// the free version is the one worth having.)
struct block_q5_1_k {
    half    d;
    half    m;
    uint8_t qh[4];
    uint8_t qs[16];
};

// One CUDA block per (output row o, token s). Threads stride over the row's
// 32-element quant blocks; each thread fully consumes a block, then the
// partial sums are reduced warp-shuffle-first, shared-memory-only-across-
// warps (the pattern q8_0_matmul.cu and the K-quant kernels already use).
//
// Launch picks blockDim from blocks_per_row rather than always using 256:
// this model's Q5_1 tensor is down_proj at I=640, i.e. only 20 blocks per
// row, so a fixed 256-thread block would leave 92% of its threads idle --
// the same "1 MAC per thread with most threads doing nothing" shape the
// K-quant lane8 restructure was written to fix.
__global__ void q5_1_matmul_kernel(
    float* __restrict__       y,
    const float* __restrict__ x,
    const block_q5_1_k* __restrict__ W,
    int S, int I, int O)
{
    const int o   = blockIdx.x;
    const int s   = blockIdx.y;
    const int tid = threadIdx.x;
    if (o >= O || s >= S) return;

    const int blocks_per_row = I / 32;
    const block_q5_1_k* w_row = W + (size_t)o * blocks_per_row;
    const float*        x_row = x + (size_t)s * I;

    float acc = 0.0f;

    for (int b = tid; b < blocks_per_row; b += blockDim.x) {
        const block_q5_1_k& blk = w_row[b];

        const float d = __half2float(blk.d);
        const float m = __half2float(blk.m);

        // Byte-wise assembly: see the struct comment above.
        const uint32_t qh = (uint32_t)blk.qh[0]        |
                            ((uint32_t)blk.qh[1] <<  8) |
                            ((uint32_t)blk.qh[2] << 16) |
                            ((uint32_t)blk.qh[3] << 24);

        // x_row + b*32 is 128-byte-strided from x_row, and x_row itself is
        // s*I floats into a cudaMalloc'd base. I is always a whole number of
        // 32-element blocks for a quantized tensor, so I%4==0 and these
        // float4 loads are 16-byte aligned. launch_q5_1_matmul re-checks
        // that rather than trusting it.
        const float4* xb = reinterpret_cast<const float4*>(x_row + b * 32);

        float qsum = 0.0f;   // sum_k q_k * x_k   (integer weights)
        float xsum = 0.0f;   // sum_k x_k         (for the shared minimum m)

        #pragma unroll
        for (int g = 0; g < 4; ++g) {
            // Elements [4g, 4g+4) come from the low nibbles, elements
            // [16+4g, 16+4g+4) from the high nibbles of the SAME qs bytes.
            const float4 xlo = xb[g];
            const float4 xhi = xb[g + 4];
            const float* xlo_p = reinterpret_cast<const float*>(&xlo);
            const float* xhi_p = reinterpret_cast<const float*>(&xhi);

            #pragma unroll
            for (int t = 0; t < 4; ++t) {
                const int j = 4 * g + t;
                const uint8_t qsj = blk.qs[j];

                const int lo = (int)(qsj & 0x0F) | (int)(((qh >> j) & 1u) << 4);
                const int hi = (int)(qsj >> 4)   | (int)(((qh >> (j + 16)) & 1u) << 4);

                const float xl = xlo_p[t];
                const float xh = xhi_p[t];

                qsum += (float)lo * xl + (float)hi * xh;
                xsum += xl + xh;
            }
        }

        acc += d * qsum + m * xsum;
    }

    // ---- reduction: warp shuffle, then one shared-memory pass per warp ----
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffff, acc, off);

    if (blockDim.x <= 32) {
        if (tid == 0) y[(size_t)s * O + o] = acc;
        return;
    }

    __shared__ float warp_sums[32];      // blockDim.x <= 1024 -> <= 32 warps
    const int lane = tid & 31;
    const int warp = tid >> 5;
    if (lane == 0) warp_sums[warp] = acc;
    __syncthreads();

    if (warp == 0) {
        const int n_warps = (int)(blockDim.x >> 5);
        float v = (lane < n_warps) ? warp_sums[lane] : 0.0f;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            v += __shfl_down_sync(0xffffffff, v, off);
        if (lane == 0) y[(size_t)s * O + o] = v;
    }
}

extern "C" void launch_q5_1_matmul(float* y, const float* x, const void* weights,
                                   int S, int I, int O)
{
    if (S <= 0 || I <= 0 || O <= 0) return;
    // A Q5_1 row is a whole number of 32-element blocks by construction; if
    // that ever fails the block-stride arithmetic (and the float4 activation
    // loads) would be wrong, so refuse rather than compute garbage. The
    // caller (gpu_matmul_dispatch) treats a silent no-op as "kernel ran", so
    // this is deliberately paired with gpu_type_supported() only ever
    // advertising Q5_1 for real GGUF tensors.
    if (I % 32 != 0) return;

    const int blocks_per_row = I / 32;

    // Round blocks_per_row up to a warp, capped at 256. For this model's
    // down_proj (bpr=20) that is a single 32-thread warp -- no shared memory,
    // no __syncthreads, and 20/32 lanes busy instead of 20/256.
    int threads = ((blocks_per_row + 31) / 32) * 32;
    if (threads < 32)  threads = 32;
    if (threads > 256) threads = 256;

    const dim3 grid((unsigned)O, (unsigned)S);
    q5_1_matmul_kernel<<<grid, threads>>>(
        y, x, reinterpret_cast<const block_q5_1_k*>(weights), S, I, O);
}
