// iq4xs_matmul.cu — fused dequant + matmul for IQ4_XS weight tensors.
//   y[S,O] = x[S,I] @ W^T,  W packed row-major [O,I].
//
// The per-super-block dequant below is a verbatim port of ggml's
// dequantize_row_iq4_xs (ggml/src/ggml-quants.c). Unlike the IQ2/IQ3 family
// (cuda/iq2_matmul.cu, cuda/iq3_matmul.cu -- large 256-1024 entry variable
// codebook grids + sign tables), IQ4_XS uses a FIXED 16-entry non-linear
// lookup table (kvalues_iq4nl) + a simple per-32-element linear scale --
// structurally much closer to the K-quant family (cuda/kquant_matmul.cu)
// than to its IQ2/IQ3 siblings, despite the similar naming. Block struct
// layout matches ggml's block_iq4_xs byte-for-byte (ggml-common.h):
// d(2) + scales_h(2) + scales_l[4] + qs[128] = 136 bytes/256 elems
// (4.25 bits/weight, matches this project's own row_bytes() table).
//
// Added because a direct throughput benchmark (main_test.cpp Test 8) found
// this checkpoint's ~4-5 CPU-bound layers (down=IQ4_XS, confirmed via the
// [QUANT] startup report) can NEVER dispatch to GPU regardless of caching/
// pinning -- a fixed floor cost paid every token on every one of those
// layers. Kernel structure (grid(O,S), 256 threads, thread-0 verbatim
// dequant into shared mem, 256-way dot + tree-reduce) is identical to
// kquant_matmul.cu's KQUANT_MATMUL_KERNEL macro, reused here rather than
// duplicated by name (this file doesn't share a translation unit with that
// one, and this project has no relocatable device code / -rdc=true, so a
// standalone macro instantiation here is consistent with how iq3_matmul.cu
// already keeps its own copy rather than sharing iq2_matmul.cu's).
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

#define QK_K 256

// Field order matches ggml's real block_iq4_xs (ggml-common.h) EXACTLY --
// d, scales_h, scales_l, qs, in that order. Getting this wrong (as
// happened once before with block_q3_K's field order, see
// kquant_matmul.cu's own comment on that bug) compiles fine but silently
// reinterprets every field at the wrong byte offset.
struct block_iq4_xs_cu { __half d; uint16_t scales_h; uint8_t scales_l[QK_K/64]; uint8_t qs[QK_K/2]; }; // 2+2+4+128=136

// Fixed 16-entry non-linear lookup table (ggml's kvalues_iq4nl), copied
// once into constant memory. Host data comes from src/ggml_grids.cpp's
// ms_grid_kvalues_iq4nl() accessor -- the actual table VALUES are never
// duplicated/hand-transcribed here, only referenced via that host pointer.
__constant__ int8_t c_kvalues_iq4nl[16];

extern "C" {
const void* ms_grid_kvalues_iq4nl();
}

static bool iq4xs_table_ready = false;
static void init_iq4xs_table() {
    if (iq4xs_table_ready) return;
    cudaMemcpyToSymbol(c_kvalues_iq4nl, ms_grid_kvalues_iq4nl(), 16 * sizeof(int8_t));
    iq4xs_table_ready = true;
}

// Verbatim port of ggml's dequantize_row_iq4_xs. Per 32-element sub-block
// (8 per 256-element super-block): a 6-bit scale index `ls` is assembled
// from a shared nibble in scales_l (2 sub-blocks share one byte) plus 2
// high bits from scales_h (one uint16 covering all 8 sub-blocks, 2 bits
// each) -- `dl = d * (ls - 32)` is the resulting per-sub-block scale. Each
// of the 16 packed bytes in qs then yields TWO output elements via the
// fixed lookup table: low nibble -> y[j+0], high nibble -> y[j+16] -- note
// this is NOT simple sequential unpacking, the low-nibble half fills
// positions 0-15 and the high-nibble half fills 16-31, not interleaved
// per-byte.
__device__ void dequant_iq4_xs_block(const block_iq4_xs_cu* xb, float* y) {
    const uint8_t* qs = xb->qs;
    const float d = __half2float(xb->d);
    for (int ib = 0; ib < QK_K / 32; ++ib) {
        const int ls = ((xb->scales_l[ib / 2] >> (4 * (ib % 2))) & 0xf) |
                        (((xb->scales_h >> (2 * ib)) & 3) << 4);
        const float dl = d * (float)(ls - 32);
        for (int j = 0; j < 16; ++j) {
            y[j +  0] = dl * (float)c_kvalues_iq4nl[qs[j] & 0xf];
            y[j + 16] = dl * (float)c_kvalues_iq4nl[qs[j] >>  4];
        }
        y  += 32;
        qs += 16;
    }
}

// ---- fused matmul: thread-0 verbatim dequant of a super-block + 256-way dot ----
// Identical structure to kquant_matmul.cu's KQUANT_MATMUL_KERNEL macro,
// instantiated directly here (not shared across translation units -- see
// this file's header comment).
__global__ void iq4_xs_matmul_kernel(
    float* __restrict__ y, const float* __restrict__ x, const block_iq4_xs_cu* __restrict__ W,
    int S, int I, int O)
{
    int o = blockIdx.x, s = blockIdx.y, tid = threadIdx.x;
    if (o >= O || s >= S) return;
    int bpr = I / 256;
    const block_iq4_xs_cu* w_row = W + (size_t)o * bpr;
    const float* x_row = x + (size_t)s * I;
    __shared__ float yq[256];
    float acc = 0.0f;
    for (int b = 0; b < bpr; ++b) {
        if (tid == 0) dequant_iq4_xs_block(&w_row[b], yq);
        __syncthreads();
        if (tid < 256) acc += x_row[b * 256 + tid] * yq[tid];
        __syncthreads();
    }
    __shared__ float sm[256];
    sm[tid] = acc; __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) sm[tid] += sm[tid + stride];
        __syncthreads();
    }
    if (tid == 0) y[(size_t)s * O + o] = sm[0];
}

extern "C" void launch_iq4_xs_matmul(float* y, const float* x, const void* w, int S, int I, int O) {
    init_iq4xs_table();
    iq4_xs_matmul_kernel<<<dim3(O, S), 256>>>(y, x, (const block_iq4_xs_cu*)w, S, I, O);
}
