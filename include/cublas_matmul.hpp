// cublas_matmul.hpp — cuBLAS dense matmul for unquantized weight tensors.
//
// Computes the engine's standard dense projection
//     y[S,O] = x[S,I] @ W^T
// where W is stored row-major [O,I] (W[o][i] contiguous in i) — exactly the
// layout the GGUF loader hands us for a 2-D weight tensor (shape [O,I]).
//
// Supported weight types: F32 / F16 / BF16 (via cublasGemmEx). Activations and
// output are always F32. Quantized dense tensors (Q8_0 / Q5_K / Q6_K / …) are
// NOT handled here — return CUBLAS_STATUS_NOT_SUPPORTED so the caller can route
// them to the dedicated dequant-matmul kernels (Step 2).
//
// NOTE on the BF16 case specifically: this mismatched-type combination
// (Atype=CUDA_R_16BF weight, Btype=CUDA_R_32F activation) was originally
// confirmed via Test 9 (src/main_test.cpp) to be REJECTED outright by
// cublasGemmEx on an RTX 3090 (Ampere, sm_86) -- CUBLAS_STATUS_NOT_
// SUPPORTED, a clean "this combination doesn't exist", not a silent
// correctness bug. NOT re-confirmed since -- `CudaManager::dense_matmul_
// host` (cuda_manager.hpp) was later changed to intercept BF16 before it
// can ever reach this function at all (it delegates straight to
// `dense_matmul_bf16_matched_host`, the matched-type BF16xBF16 path
// below, verified correct via Test 10), so nothing in this codebase has
// actually exercised cublasGemmEx with THIS specific mismatched
// combination since that change landed. (A later run's Test 9 result
// that briefly looked like a hardware-dependent silent-miscompute on a
// second GPU turned out to be a stale-test artifact, not a real cuBLAS
// finding -- see Test 9's own comment for the full story; that
// investigation is what prompted hardening this function directly rather
// than continuing to rely on the driver's behavior being verified.) This
// function now refuses to ever issue the call at all (returns
// CUBLAS_STATUS_NOT_SUPPORTED unconditionally for BF16, without touching
// cublasGemmEx) rather than depend on an assumption that was only ever
// checked on one architecture -- belt-and-suspenders given
// `dense_matmul_host` already avoids this path in practice; this closes
// it for any future direct caller of the low-level `dense_matmul()`
// wrapper too, so a call site added later without reading this comment
// fails loudly (a clean "unsupported" return) instead of depending on
// unverified driver behavior.
//
// ALL POINTERS MUST BE DEVICE-RESIDENT. This primitive does no host<->device
// staging — dense activations currently live in host `hidden_states`, so the
// Step-2 dispatch caller is responsible for uploading x and downloading y
// (mirroring how expert_ffn stages d_x today).
//
// ---- Layout / transpose derivation (cuBLAS is column-major) ----
// Row-major target:   y[s,o] = sum_i x[s,i] * W[o,i]
// Reinterpreted as col-major matrices (stride shown = leading dim):
//   X_cm[i,s] = x[s,i]   (rows=I, cols=S, ld=I)
//   W_cm[i,o] = W[o,i]   (rows=I, cols=O, ld=I)
//   Y_cm[o,s] = y[s,o]   (rows=O, cols=S, ld=O)
// Then Y_cm = W_cm^T @ X_cm, i.e. op(A)=W_cm^T (so transa=T, A stored [I,O] ld=I),
// op(B)=X_cm (transb=N, B stored [I,S] ld=I), result [O,S] ld=O. The byte
// layout of col-major Y_cm[O,S] with ld=O is identical to row-major y[S,O], so
// d_Y holds the result in the engine's expected row-major form with no shuffle.
// => GemmEx(OP_T, OP_N, m=O, n=S, k=I, A=d_W lda=I, B=d_X ldb=I, C=d_Y ldc=O)
#pragma once
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "gguf_loader.hpp"  // GGUFType

inline cublasStatus_t cublas_dense_matmul(
    cublasHandle_t handle,
    GGUFType wtype,
    const void* d_W,    // [O,I] row-major, device
    const float* d_X,   // [S,I] row-major, device (F32 activations)
    float* d_Y,         // [S,O] row-major, device (F32 output)
    int S, int I, int O)
{
    cudaDataType atype;
    switch (wtype) {
        case GGUFType::F32:  atype = CUDA_R_32F;  break;
        case GGUFType::F16:  atype = CUDA_R_16F;  break;
        // BF16 weight + F32 activation: confirmed on real hardware (Test 9)
        // to silently miscompute on at least one architecture (Blackwell/
        // sm_120) rather than reject like Ampere/sm_86 does — see the
        // file-header comment above. Never issue this call; use
        // cublas_dense_matmul_bf16_matched (BF16 weight + BF16 activation)
        // instead, via CudaManager::dense_matmul_bf16_matched_host.
        case GGUFType::BF16: return CUBLAS_STATUS_NOT_SUPPORTED;
        default: return CUBLAS_STATUS_NOT_SUPPORTED;  // quant types -> Step 2 kernels
    }

    const float alpha = 1.0f, beta = 0.0f;
    return cublasGemmEx(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        /*m=*/O, /*n=*/S, /*k=*/I,
        &alpha,
        d_W,  atype,        /*lda=*/I,
        d_X,  CUDA_R_32F,   /*ldb=*/I,
        &beta,
        d_Y,  CUDA_R_32F,   /*ldc=*/O,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
}

// Matched-precision variant: weight AND activation are BOTH BF16
// (Atype=Btype=CUDA_R_16BF), output stays F32 (Ctype=CUDA_R_32F,
// compute=CUBLAS_COMPUTE_32F). Added after cublas_dense_matmul's *mixed*
// BF16-weight/F32-activation combination (the original, simpler attempt
// at a BF16 fast-path) was confirmed via a real-hardware isolated test
// (main_test.cpp's Test 9) to be REJECTED outright by cublasGemmEx --
// CUBLAS_STATUS_NOT_SUPPORTED, a hard "this combination doesn't exist",
// not a silent correctness bug. BF16 x BF16 -> F32 (this function) is a
// standard, documented cublasGemmEx combination, unlike the mixed one --
// but per this project's own hard-won lesson (the eviction-race
// corruption incident earlier this session), "should work" gets verified
// via its own isolated test (Test 10, main_test.cpp) on real hardware
// before this is ever wired into the live dense_dispatch.hpp path.
//
// d_X_bf16 must ALREADY be BF16-converted and device-resident. Unlike the
// weight (converted once at load and cached for the process lifetime),
// the activation is different every call, so the caller is responsible
// for converting+uploading it fresh each time -- this function does no
// conversion or staging itself, matching cublas_dense_matmul's own
// "all pointers must be device-resident" contract above.
inline cublasStatus_t cublas_dense_matmul_bf16_matched(
    cublasHandle_t handle,
    const void* d_W_bf16,   // [O,I] row-major, device, BF16
    const void* d_X_bf16,   // [S,I] row-major, device, BF16
    float* d_Y,             // [S,O] row-major, device, F32
    int S, int I, int O)
{
    const float alpha = 1.0f, beta = 0.0f;
    return cublasGemmEx(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        /*m=*/O, /*n=*/S, /*k=*/I,
        &alpha,
        d_W_bf16, CUDA_R_16BF, /*lda=*/I,
        d_X_bf16, CUDA_R_16BF, /*ldb=*/I,
        &beta,
        d_Y, CUDA_R_32F, /*ldc=*/O,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
}
