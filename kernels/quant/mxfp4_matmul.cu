// mxfp4_matmul.cu — fused dequant + matmul for MXFP4 weight tensors.
//   y[S,O] = x[S,I] @ W^T,  W packed row-major [O,I].
//
// DORMANT KERNEL: compiled and linked but not yet dispatched to. See
// kquant_dequant.hpp's gpu_type_supported() — MXFP4 stays disabled there
// until this kernel's Test 5 case (src/main_test.cpp) passes on real
// Linux/CUDA hardware, per this project's own IQ4_XS incident history
// (CLAUDE.md): a new kernel was once flipped straight to a real model and
// caused a 100% NaN incident whose real root cause turned out to be an
// unrelated pre-existing concurrency race in shared dispatch machinery
// (already fixed) — not the kernel itself. Landing dormant + test-gating is
// the process fix adopted after that incident.
//
// The per-block dequant below is a verbatim port of ggml's
// dequantize_row_mxfp4 + ggml_e8m0_to_fp32_half (ggml/src/ggml-quants.c,
// ggml/src/ggml-impl.h — verified directly against
// /Users/rayxu/github/llama.cpp, not hand-derived). block_mxfp4 is far
// simpler than IQ4_XS's superblock: a FIXED 16-entry non-linear lookup
// table (kvalues_mxfp4, identical values to kvalues_iq4nl) + ONE E8M0
// power-of-two scale byte, applied to a 32-element block — the block size
// exactly matches a CUDA warp (32 lanes), so this kernel uses a
// warp-per-block cooperative-dequant design rather than IQ4_XS's
// thread-per-256-superblock layout (cuda/iq4xs_matmul.cu). Block struct
// layout matches ggml's block_mxfp4 byte-for-byte (ggml-common.h):
// e(1) + qs[16] = 17 bytes/32 elems (4.25 bits/weight, matches this
// project's own row_bytes() table in kquant_dequant.hpp).

#include <cuda_runtime.h>
#include <cstdint>

#define QK_MXFP4 32

// Field order matches ggml's real block_mxfp4 exactly (all-uint8_t, no
// padding — sizeof == 17, same as the host-side layout GGUF files store).
struct block_mxfp4_cu { uint8_t e; uint8_t qs[QK_MXFP4 / 2]; }; // 17 bytes

// kvalues_fp4 / kvalues_mxfp4 in ggml-common.h — a fixed table, not
// data-derived, so (unlike IQ4_XS's externally-shared kvalues_iq4nl) it's
// safe to statically initialize directly in __constant__ memory here: no
// cudaMemcpyToSymbol/host-accessor indirection needed, and this table isn't
// reused anywhere else in MoEStream's own code (the CPU dequant path calls
// straight into libggml — see src/ggml_dequant.cpp).
__constant__ int8_t c_kvalues_mxfp4[16] = {
    0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12
};

// Verbatim port of ggml_e8m0_to_fp32_half (ggml/src/ggml-impl.h): decodes
// the block's single E8M0 exponent byte into the f32 scale MXFP4 dequant
// multiplies every element by. NOT the same as ggml_e8m0_to_fp32() (a
// different function, used by a different quant format) — this is
// specifically the "_half" variant dequantize_row_mxfp4 calls.
__device__ __forceinline__ float mxfp4_e8m0_to_fp32_half(uint8_t e) {
    uint32_t bits = (e < 2) ? (0x00200000u << e) : (((uint32_t)e - 1) << 23);
    return __uint_as_float(bits);
}

// ---- fused matmul (warp-per-block cooperative dequant + two-stage reduction) ----
__global__ void mxfp4_matmul_kernel(
    float* __restrict__ y, const float* __restrict__ x, const block_mxfp4_cu* __restrict__ W,
    int S, int I, int O)
{
    int o = blockIdx.x;
    int s = blockIdx.y;
    int tid = threadIdx.x;

    if (o >= O || s >= S) return;

    int bpr = I / QK_MXFP4;
    const block_mxfp4_cu* w_row = W + (size_t)o * bpr;
    const float* x_row = x + (size_t)s * I;

    // 256 threads = 8 warps. Each warp owns one 32-element block per
    // iteration (block size == warp size, a direct 1 lane : 1 element
    // mapping), striding by 8 warps across bpr total blocks.
    const int warp_id = tid >> 5;   // 0..7
    const int lane = tid & 31;      // 0..31

    float acc = 0.0f;
    for (int b = warp_id; b < bpr; b += 8) {
        const block_mxfp4_cu& blk = w_row[b];
        const float d = mxfp4_e8m0_to_fp32_half(blk.e);

        // Reference loop (dequantize_row_mxfp4) is: for j in [0,16):
        //   y[j]    = kvalues[qs[j] & 0xF]  * d
        //   y[j+16] = kvalues[qs[j] >> 4]   * d
        // lane<16 covers the first half (low nibbles), lane>=16 the second
        // half (high nibbles of qs[lane-16]) — exactly that loop unrolled
        // across the warp's 32 lanes.
        int nibble = (lane < 16) ? (blk.qs[lane] & 0xF) : (blk.qs[lane - 16] >> 4);
        const float w_val = d * (float)c_kvalues_mxfp4[nibble];

        acc += x_row[(size_t)b * QK_MXFP4 + lane] * w_val;
    }

    // Warp-level shuffle reduction: combines the 32 per-lane accumulators
    // (each already summed across this warp's assigned blocks) into one
    // partial sum for the warp.
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        acc += __shfl_down_sync(0xffffffff, acc, offset);
    }

    __shared__ float warp_sums[8];
    if (lane == 0) {
        warp_sums[warp_id] = acc;
    }
    __syncthreads();

    // Final reduction of the 8 warp sums by the first warp.
    if (warp_id == 0) {
        float sum = (lane < 8) ? warp_sums[lane] : 0.0f;

        #pragma unroll
        for (int offset = 4; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0x000000ff, sum, offset);
        }

        if (tid == 0) {
            y[(size_t)s * O + o] = sum;
        }
    }
}

extern "C" void launch_mxfp4_matmul(float* y, const float* x, const void* w, int S, int I, int O) {
    mxfp4_matmul_kernel<<<dim3(O, S), 256>>>(y, x, (const block_mxfp4_cu*)w, S, I, O);
}
