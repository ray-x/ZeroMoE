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
        case GGUFType::BF16: atype = CUDA_R_16BF; break;
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
