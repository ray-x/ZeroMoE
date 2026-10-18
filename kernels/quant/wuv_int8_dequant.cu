// wuv_int8_dequant.cu — on-the-fly int8->BF16 dequant kernel for the WUV
// (score GPU path's down-projection) weight cache. See CudaManager::
// upload_wuv_int8/wuv_int8_batched_host (cuda_manager.hpp) for the full
// design/motivation: WUV's persistent BF16 upload only covers a minority
// of layers on GPU1 (CoreEngine) because it competes for a tight, already-
// exhausted VRAM budget with qabsorb/the score mirror (see CLAUDE.md's WUV-
// coverage entries) — storing it as int8 instead roughly halves its
// per-layer footprint, buying real coverage within the SAME existing
// budget, no VRAM displaced from anywhere else. Deliberately does NOT
// touch the proven BF16xBF16 strided-batched cuBLAS GEMM
// (cublas_dense_matmul_bf16_matched_batched) at all -- this kernel's only
// job is to reconstruct a transient BF16 buffer from the persistent int8
// storage right before that unchanged GEMM runs, so the only new
// correctness surface is this one simple elementwise dequant, not a new
// GEMM precision path.
//
// Per-ROW (per (head, output-row) pair, matching the WUV weight's own
// [H,vh,kvl] layout — O=vh rows per head, I=kvl) symmetric scale, not a
// single per-layer or per-head scale — matches this project's own
// established "prefer finer-grained scales over a single tensor-wide one"
// convention for every OTHER lossy quant path already in this codebase
// (Q6_K/Q8_0's own per-32-element-block scales, etc.), and costs only
// H*vh floats/layer (negligible: 64KB/layer at F32 for GLM-5.2's dims).

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

// weight_int8: [H, O, I] row-major int8 (O=vh, I=kvl for WUV's own layout).
// scale: [H, O] row-major float — one scale per output row, dequant is
//   w_bf16[h,o,i] = weight_int8[h,o,i] * scale[h,o].
// out_bf16: [H, O, I] row-major bf16 (as raw uint16_t, matching this
//   codebase's existing fp32_to_bf16/bf16 host-side convention — no
//   __nv_bfloat16 type dependency needed since this is a pure elementwise
//   cast+scale, not arithmetic requiring the type's own ops).
__global__ void dequant_wuv_int8_kernel(
    const int8_t* __restrict__ weight_int8,
    const float* __restrict__ scale,
    uint16_t* __restrict__ out_bf16,
    int H, int O, int I)
{
    // One block per (h,o) row, threads stride across I -- mirrors this
    // project's own combine_scale_softmax_kernel's "one block per head"
    // launch shape (attn_score_softmax.cu), just one level finer (per row
    // within a head instead of per head).
    int row = blockIdx.x;  // row in [0, H*O)
    if (row >= H * O) return;
    int h = row / O;
    int o = row % O;
    float s = scale[h * O + o];
    const int8_t* w_row = weight_int8 + (size_t)row * I;
    uint16_t* out_row = out_bf16 + (size_t)row * I;

    for (int i = threadIdx.x; i < I; i += blockDim.x) {
        float f = (float)w_row[i] * s;
        // fp32->bf16: round-to-nearest-even via the same +0x7fff/round-bit
        // trick this codebase's own host-side fp32_to_bf16 (kquant_dequant.
        // hpp) uses, so GPU and CPU dequant agree bit-for-bit on rounding.
        // Deliberately omits that function's NaN/Inf/overflow-on-round
        // guards -- this value is always (int8 in [-127,127]) * (a finite
        // scale computed from a real weight tensor's own max-abs), a
        // strictly bounded product that can never be NaN/Inf and, being
        // far below BF16's overflow-on-round threshold (~3.4e38), can never
        // trigger that edge case either -- those guards exist for a
        // GENERAL fp32 input this dequant never produces.
        uint32_t bits = __float_as_uint(f);
        uint32_t rounded = bits + 0x7fffu + ((bits >> 16) & 1u);
        out_row[i] = (uint16_t)(rounded >> 16);
    }
}

extern "C" void launch_dequant_wuv_int8(
    const int8_t* weight_int8, const float* scale, uint16_t* out_bf16,
    int H, int O, int I, cudaStream_t stream)
{
    int threads = 128;
    int blocks = H * O;
    dequant_wuv_int8_kernel<<<blocks, threads, 0, stream>>>(
        weight_int8, scale, out_bf16, H, O, I);
}
