// q8_0_matmul.cu — fused Q8_0 dequant + matmul: y[S,O] = x[S,I] @ W^T
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

struct block_q8_0_k {
    half    d;        // FP16 per-block scale
    int8_t  qs[32];   // 32 int8 values
};  // sizeof == 34

__global__ void q8_0_matmul_kernel_optimized(
    float* __restrict__       y,
    const float* __restrict__ x,
    const block_q8_0_k* __restrict__ W,
    int S, int I, int O)
{
    int o = blockIdx.x;
    int s = blockIdx.y;
    int tid = threadIdx.x;
    
    if (o >= O || s >= S) return;

    int blocks_per_row = I / 32;
    const block_q8_0_k* w_row = W + (size_t)o * blocks_per_row;
    const float* x_row = x + (size_t)s * I;

    float row_sum = 0.0f;

    // 1. Vectorized Memory Access & Unrolled Accumulation
    //
    // FIX (real bug -- caused an Xid 13 "Misaligned Address" GPU exception
    // on real hardware, confirmed by dmesg output across dozens of SMs at
    // once): block_q8_0_k is 34 bytes (2-byte half + 32-byte int8 array),
    // which is the REAL, on-disk GGUF Q8_0 block layout -- it can't be
    // padded to a multiple of 4 without misinterpreting actual model
    // weight bytes. Because 34 is not a multiple of 4, w_row[b].qs lands
    // at a 4-byte-ALIGNED offset only when (o*blocks_per_row + b) is odd,
    // and at a MISALIGNED offset (2 mod 4) whenever it's even -- including
    // the very first block (o=0, b=0). The previous version cast that
    // address straight to `const int32_t*` and dereferenced it, which is a
    // hardware-illegal unaligned 32-bit load on roughly half of all blocks
    // processed. `qs` is read byte-by-byte here instead (int8_t has
    // alignment 1, always safe regardless of block parity) -- the x4
    // float4 read is untouched since x_row's own alignment (16-byte
    // periodic stride from a cudaMalloc'd, heavily-aligned base) was never
    // the problem.
    for (int b = tid; b < blocks_per_row; b += blockDim.x) {
        float d = __half2float(w_row[b].d);

        const float4* xb_f4 = reinterpret_cast<const float4*>(x_row + b * 32);
        const int8_t* qs = w_row[b].qs;

        float dot = 0.0f;

        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            float4 x4 = xb_f4[i];   // Load 4x float activations in one 128-bit transaction

            // qs is already signed int8_t -- no manual sign-extension needed
            dot += (float)qs[i * 4 + 0] * x4.x;
            dot += (float)qs[i * 4 + 1] * x4.y;
            dot += (float)qs[i * 4 + 2] * x4.z;
            dot += (float)qs[i * 4 + 3] * x4.w;
        }
        row_sum += dot * d;
    }

    // 2. Warp-Level Reduction using Shuffle Primitives
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        row_sum += __shfl_down_sync(0xffffffff, row_sum, offset);
    }

    // 3. Block-Level Reduction across Warps
    static __shared__ float s_warp_sums[32]; // Max 32 warps (1024 threads)
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    if (lane_id == 0) {
        s_warp_sums[warp_id] = row_sum;
    }
    __syncthreads();

    // Final reduction by the first warp
    if (warp_id == 0) {
        int num_warps = blockDim.x / 32;
        row_sum = (lane_id < num_warps) ? s_warp_sums[lane_id] : 0.0f;

        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            row_sum += __shfl_down_sync(0xffffffff, row_sum, offset);
        }

        if (lane_id == 0) {
            y[(size_t)s * O + o] = row_sum;
        }
    }
}

// REUSE-ACROSS-S variant (prefill, S>1): one block per `o` ONLY (no S in the
// grid). The weight row for this `o` is read from global memory exactly ONCE,
// staged into shared memory, then reused across all S rows in a loop --
// avoiding S-1 redundant re-reads of the same 34-byte-per-block weight row
// that the original per-(o,s) kernel pays for (each (o,s) block independently
// re-fetches w_row from scratch).
//
// VERIFIED on real RTX 3090 hardware (S=11, I=2048, O=16384 -- attn_q_b's
// real GLM-5.2 shape, confirmed via direct GGUF parse): 0.9546ms/call
// (original) vs 0.8867ms/call (this kernel) -- a real, if modest (~8%), win;
// correctness verified bit-identical (max diff 0) against the original
// across three independent real-hardware runs. See CLAUDE.md section 6 for
// the full investigation: two OTHER proposed fixes for this same kernel --
// warp-cooperative coalesced loads, multi-accumulator ILP -- were ALSO
// real-hardware A/B tested and BOTH disproven (coalescing was actually 47%
// SLOWER, from paying a full warp-shuffle reduction per 32-element block
// instead of once per row; ILP made no measurable difference at all). This
// reuse kernel is the only one of three external suggestions that panned out.
__global__ void q8_0_matmul_kernel_reuse(
    float* __restrict__       y,
    const float* __restrict__ x,
    const block_q8_0_k* __restrict__ W,
    int S, int I, int O)
{
    int o = blockIdx.x;
    int tid = threadIdx.x;
    if (o >= O) return;

    int blocks_per_row = I / 32;
    const block_q8_0_k* w_row = W + (size_t)o * blocks_per_row;

    extern __shared__ block_q8_0_k w_shared[];
    for (int b = tid; b < blocks_per_row; b += blockDim.x) {
        w_shared[b] = w_row[b];
    }
    __syncthreads();

    static __shared__ float s_warp_sums[32];
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int num_warps = blockDim.x / 32;

    for (int s = 0; s < S; ++s) {
        const float* x_row = x + (size_t)s * I;
        float row_sum = 0.0f;

        for (int b = tid; b < blocks_per_row; b += blockDim.x) {
            float d = __half2float(w_shared[b].d);
            const float4* xb_f4 = reinterpret_cast<const float4*>(x_row + b * 32);
            const int8_t* qs = w_shared[b].qs;

            float dot = 0.0f;
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                float4 x4 = xb_f4[i];
                dot += (float)qs[i * 4 + 0] * x4.x;
                dot += (float)qs[i * 4 + 1] * x4.y;
                dot += (float)qs[i * 4 + 2] * x4.z;
                dot += (float)qs[i * 4 + 3] * x4.w;
            }
            row_sum += dot * d;
        }

        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            row_sum += __shfl_down_sync(0xffffffff, row_sum, offset);
        }

        if (lane_id == 0) {
            s_warp_sums[warp_id] = row_sum;
        }
        __syncthreads();

        if (warp_id == 0) {
            float v = (lane_id < num_warps) ? s_warp_sums[lane_id] : 0.0f;
            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                v += __shfl_down_sync(0xffffffff, v, offset);
            }
            if (lane_id == 0) {
                y[(size_t)s * O + o] = v;
            }
        }
        // s_warp_sums gets overwritten next iteration -- every thread must be
        // done reading it (the warp_id==0 branch above) before that happens.
        __syncthreads();
    }
}

extern "C" void launch_q8_0_matmul(float* y, const float* x, const void* weights,
                                    int S, int I, int O) {
    dim3 block(256);
    int blocks_per_row = I / 32;
    size_t smem_bytes = (size_t)blocks_per_row * sizeof(block_q8_0_k);

    // Reuse kernel only pays off once there's more than one row to amortize
    // the weight-staging cost over -- S==1 (decode, the dominant call volume
    // by count) gets zero benefit from it and would only add a wasted
    // shared-memory copy + extra __syncthreads, so decode keeps using the
    // original kernel unconditionally. Also guarded on fitting the default
    // 48KB static shared-memory budget (true for every quant tensor shape
    // confirmed in this engine so far -- I=16384 is the largest real I seen,
    // giving 512*34=17408 bytes -- but checked rather than assumed, falling
    // back safely to the original kernel if a future/different model's shape
    // would exceed it, rather than needing a cudaFuncSetAttribute opt-in).
    constexpr size_t kMaxStaticSmem = 49152; // 48KB
    if (S > 1 && smem_bytes <= kMaxStaticSmem) {
        dim3 grid(O);
        q8_0_matmul_kernel_reuse<<<grid, block, smem_bytes>>>(
            y, x, (const block_q8_0_k*)weights, S, I, O);
    } else {
        dim3 grid(O, S);
        q8_0_matmul_kernel_optimized<<<grid, block>>>(
            y, x, (const block_q8_0_k*)weights, S, I, O);
    }
}