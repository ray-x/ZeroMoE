// dense_dispatch.hpp — single entry point for dense weight matmuls in the
// forward path. Routes y[S,O] = x[S,I] @ W^T (W stored row-major [O,I], looked
// up by name in engine.resident_layers) to the GPU dispatcher when the weight's
// quant has a kernel, else falls back to CPU f32 matmul WITH A LOUD WARNING.
//
// The point: never silently reinterpret a quantized weight buffer as float.
// That reinterpret is the silent-garbage bug this project is eliminating — every
// dense matmul now goes through here so an unsupported quant is visible at the
// first token, not baked into every output.
#pragma once
#include "engine.hpp"
#include "cuda_manager.hpp"
#include "kernels_cpu.hpp"        // matmul
#include "kquant_dequant.hpp"     // cpu_quant_matmul (CPU fallback)
#include "logger.hpp"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>

// Diagnostic kill-switch: forces every dense matmul, norm, and expert FFN onto
// the CPU reference path (libggml dequant + f32 matmul), bypassing ALL GPU
// dispatch. Enabled by EITHER the MOESTREAM_CPU_ONLY env var OR a TOML
// `cpu_only = true` (main.cpp calls enable_cpu_only_mode() after config load).
// Used to isolate whether garbage output comes from a GPU-side issue
// (async-upload race, device/stream, offload corruption, a fused kernel) vs a
// CPU math bug. If output becomes coherent with this set, the GPU path is the
// culprit; if still garbage, the bug is in the CPU math (attention/MoE/
// embedding) and is independent of where compute runs.
inline bool& cpu_only_flag() {
    static bool v = (std::getenv("MOESTREAM_CPU_ONLY") != nullptr);
    return v;
}
inline bool cpu_only_mode() { return cpu_only_flag(); }
inline void enable_cpu_only_mode() { cpu_only_flag() = true; }

inline void dense_proj(StreamingEngine& engine, const std::string& wname,
                       const float* x, float* y, int S, int I, int O) {
    auto it = engine.resident_layers.find(wname);
    if (it == engine.resident_layers.end() || !it->second.host_ptr) {
        LOG_WARN("[DENSE] missing weight %s — zeroing output\n", wname.c_str());
        std::memset(y, 0, (size_t)S * O * sizeof(float));
        return;
    }
    auto& W = it->second;
    // 1) GPU dispatch. Prefer the F32 fast-path buffer (upload_dequantized_f32)
    //    when present — cuBLAS SGEMV instead of W's original quant type's fused
    //    per-block kernel, which has no GEMV specialization and is
    //    disproportionately slow at S=1 decode (attn_output's Q5_K case, see
    //    CLAUDE.md). Falls through to the original device_ptr/meta.type path
    //    (fused kernel for K-quants/IQ, cuBLAS for F32/F16/BF16/Q8_0) when no
    //    fast-path buffer was built for this weight (not attempted, or didn't
    //    fit in the VRAM budget at load time).
    //
    //    REVERTED FROM BF16 (see CLAUDE.md): a real run produced garbage
    //    output ("Hello! I'm the GLM model behind[PAD154879]..." repeating)
    //    after switching this to GGUFType::BF16. No coding bug was found on
    //    static review of the conversion math or dispatch routing; leading
    //    unconfirmed suspicion is cublasGemmEx with mismatched input types
    //    (Atype=CUDA_R_16BF weight, Btype=CUDA_R_32F activations) not being a
    //    genuinely correct/supported combination for this entry point.
    //    Reverted to the known-good F32 path pending real-hardware
    //    investigation -- do not re-enable BF16 here without first
    //    confirming (on Linux/CUDA) that cublas_dense_matmul's BF16 branch
    //    alone produces correct output on a synthetic, non-degenerate test.
    if (!cpu_only_mode() && engine.gpu && W.gpu_fast_ptr &&
        engine.gpu->dense_matmul_host(GGUFType::F32, W.gpu_fast_ptr, x, y, S, I, O)) {
        return;
    }
    if (!cpu_only_mode() && engine.gpu && W.device_ptr &&
        engine.gpu->dense_matmul_host(W.meta.type, W.device_ptr, x, y, S, I, O)) {
        return;
    }
    // 2) CPU reference dequant-matmul: correct for F32/F16/BF16/Q8_0/Q4_0;
    //    loudly zeroes K-quants/IQ2 (dequant bodies pending). Never the old
    //    silent "reinterpret quant as float". This is routine/expected for
    //    many quant types today (no GPU kernel yet) and fires once per weight
    //    per forward pass — DEBUG level, not a real problem on its own.
    LOG_DEBUG("[DENSE] %s type=%u: no GPU dispatch — CPU dequant-matmul fallback\n",
              wname.c_str(), (unsigned)W.meta.type);
    if (!cpu_quant_matmul(W.meta.type, W.host_ptr, x, y, S, I, O)) {
        LOG_ERROR("[DENSE] %s type=%u: CPU dequant unavailable — some outputs zeroed\n",
                  wname.c_str(), (unsigned)W.meta.type);
    }
}

// F32 RMSNorm via GPU (norm weights are offloaded to VRAM), CPU rmsnorm fallback.
inline void rmsnorm_proj(StreamingEngine& engine, const std::string& wname,
                         const float* x, float* y, int S, int D, float eps) {
    auto it = engine.resident_layers.find(wname);
    if (it == engine.resident_layers.end() || !it->second.host_ptr) {
        LOG_WARN("[NORM] missing weight %s — passing input through\n", wname.c_str());
        std::memcpy(y, x, (size_t)S * D * sizeof(float));
        return;
    }
    auto& W = it->second;
    if (!cpu_only_mode() && engine.gpu && W.device_ptr &&
        engine.gpu->rmsnorm_gpu_host(W.device_ptr, x, y, S, D, eps)) {
        return;
    }
    for (int s = 0; s < S; ++s)
        rmsnorm_row(y + (size_t)s * D, x + (size_t)s * D,
                    reinterpret_cast<float*>(W.host_ptr), D, eps);
}
