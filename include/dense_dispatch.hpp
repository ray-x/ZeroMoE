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
#include "pilot_confirmed_set.hpp"  // publish_pilot_confirmed_resident -- execution_layer.cpp's pilot loop writes here
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>
#include <algorithm>
#include <mutex>

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

// Per-layer version of the same idea: `cpu_affinity_layers` (config) names
// specific MoE layers whose routed experts should NEVER be dispatched on
// GPU or promoted/uploaded into VRAM at all -- unlike cpu_only_mode() this
// is scoped to just those layers, everything else still dispatches
// normally. Motivated by layers whose expert set is so large/flat that a
// real run spends far more wall-clock time on the disk read than the
// matmul itself (see CLAUDE.md) -- for those layers, paying for a VRAM
// upload buys nothing (the bottleneck never moves) while it DOES cost real
// VRAM cache capacity taken away from layers where GPU residency actually
// helps, plus real PCIe/DMA time that's pure overhead if the compute was
// going to be I/O-bound either way. Backed by a flat bool vector (not a
// set) for an O(1), branch-predictable check on the hottest possible path
// (once per routed expert per token) -- set once at startup
// (set_cpu_affinity_layers, main.cpp, from EngineConfig::cpu_affinity_layers)
// before any dispatch can run, never mutated afterward, so no locking is
// needed despite being read from multiple threads (prefetcher, decode).
inline std::vector<bool>& cpu_affinity_layer_flags() {
    static std::vector<bool> v;
    return v;
}
inline void set_cpu_affinity_layers(const std::vector<int>& layers, int n_layers) {
    std::vector<bool> v(std::max(0, n_layers), false);
    for (int l : layers) {
        if (l >= 0 && l < (int)v.size()) v[(size_t)l] = true;
    }
    cpu_affinity_layer_flags() = std::move(v);
}
inline bool is_cpu_affinity_layer(int layer_idx) {
    const auto& v = cpu_affinity_layer_flags();
    return layer_idx >= 0 && layer_idx < (int)v.size() && v[(size_t)layer_idx];
}

// Per-layer override for the pilot router's prefetch hint count
// (EngineConfig::prefetch_pilot_k / prefetch_pilot_k_per_layer). Same
// set-once-at-startup, read-from-multiple-threads-without-locking idiom as
// cpu_affinity_layer_flags() above (set_prefetch_pilot_k, main.cpp; read
// from execution_layer.cpp's pilot loop, decode thread only in practice but
// no different in principle from the cpu_affinity flags' own read pattern).
// A missing or negative per-layer entry means "no override for this
// layer" -- falls back to the global default, not to 0 -- distinguishing
// "not configured" from "the user explicitly asked for 0 hints here".
inline std::vector<int>& pilot_k_per_layer_overrides() {
    static std::vector<int> v;
    return v;
}
inline int& pilot_k_default_value() {
    static int v = 8;
    return v;
}
inline void set_prefetch_pilot_k(const std::vector<int>& per_layer, int default_k) {
    pilot_k_per_layer_overrides() = per_layer;
    pilot_k_default_value() = default_k;
}
inline int pilot_k_for_layer(int layer_idx) {
    const auto& v = pilot_k_per_layer_overrides();
    if (layer_idx >= 0 && layer_idx < (int)v.size() && v[(size_t)layer_idx] >= 0) return v[(size_t)layer_idx];
    return pilot_k_default_value();
}

// Same set-once-at-startup / read-everywhere idiom as pilot_k_for_layer
// above, for EngineConfig::prefetch_pilot_depth / prefetch_pilot_k_depth2
// (set_prefetch_pilot_depth, main.cpp; read from execution_layer.cpp's
// pilot loop). No per-layer override for depth-2's k yet (unlike l+1's
// pilot_k_per_layer_overrides) -- pilot_k2_for_layer keeps the same
// per-layer function SHAPE anyway, so one can be added later without
// changing every call site, mirroring pilot_k_for_layer's own signature.
inline int& pilot_depth_value() {
    static int v = 1;
    return v;
}
inline int& pilot_k2_default_value() {
    static int v = 8;
    return v;
}
inline void set_prefetch_pilot_depth(int depth, int k2_default) {
    pilot_depth_value() = depth;
    pilot_k2_default_value() = k2_default;
}
// Whether the pilot should predict+push real depth-2 (l+2) hints, as
// opposed to depth-2 measurement continuing to run for free either way
// (see execution_layer.cpp's pilot loop -- the recall counters are always
// live regardless of this).
inline bool prefetch_pilot_depth2_enabled() {
    return pilot_depth_value() == 2;
}
inline int pilot_k2_for_layer(int layer_idx) {
    (void)layer_idx;  // no per-layer override yet -- see this block's own comment
    return pilot_k2_default_value();
}

inline void dense_proj(StreamingEngine& engine, const std::string& wname,
                       const float* x, float* y, int S, int I, int O) {
    auto it = engine.resident_layers.find(wname);
    if (it == engine.resident_layers.end() || !it->second.host_ptr) {
        LOG_WARN("[DENSE] missing weight %s — zeroing output\n", wname.c_str());
        std::memset(y, 0, (size_t)S * O * sizeof(float));
        return;
    }
    auto& W = it->second;
    // 1) GPU dispatch. Prefer the BF16-matched fast-path buffer
    //    (upload_dequantized_bf16, cuda_manager.hpp) when present — cuBLAS
    //    GEMV (matched BF16 weight/BF16 activation, F32 accumulate/output)
    //    instead of W's original quant type's fused per-block kernel, which
    //    has no GEMV specialization and is disproportionately slow at S=1
    //    decode (attn_output's Q5_K case, see CLAUDE.md). Falls through to
    //    the original device_ptr/meta.type path (fused kernel for
    //    K-quants/IQ, cuBLAS for F32/F16/BF16/Q8_0) when no fast-path
    //    buffer was built for this weight (not attempted, or didn't fit in
    //    the VRAM budget at load time).
    //
    //    Third attempt at this fast-path, per CLAUDE.md's full history: an
    //    F32 version worked but covered too few layers within budget; a
    //    BF16 version using MISMATCHED types (BF16 weight / F32 activation)
    //    caused real NaN corruption, was reverted, and was later confirmed
    //    (Test 9, main_test.cpp, real hardware) to be cleanly REJECTED by
    //    cublasGemmEx regardless — a dead end independent of the earlier
    //    corruption, which was separately root-caused to an unrelated
    //    concurrency race (now fixed) and never actually caused by BF16.
    //    This version uses the MATCHED-type combination (BF16 weight AND
    //    BF16 activation, dense_matmul_bf16_matched_host), verified correct
    //    in isolation via Test 10 (real hardware) before being wired in
    //    here.
    if (!cpu_only_mode() && engine.gpu && W.gpu_fast_ptr &&
        engine.gpu->dense_matmul_bf16_matched_host(W.gpu_fast_ptr, x, y, S, I, O)) {
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
