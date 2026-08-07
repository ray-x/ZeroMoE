// cuda_manager.hpp — Multi-GPU pool manager with topology-aware execution
// Implements the asymmetric dual-GPU pipeline from optimize.md:
//   GPU 0 (CoreEngine): Dense layers + KV cache + activations
//   GPU 1+ (ExpertVault): MoE expert LRU cache in VRAM
#pragma once
#include "engine.hpp"
#include "engine_config.hpp"
#include "perf_timer.hpp"
#include "gpu_topology.hpp"
#include "cublas_matmul.hpp"
#include "kquant_dequant.hpp"  // dequant_tensor_f32, for upload_dequantized_f32
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <cstring>
#include <algorithm>

// External CUDA kernel hooks
extern "C" void launch_gguf_q4_0_dp4a(float* y, const float* x, const void* weights, int S, int I, int O);
extern "C" void launch_q8_0_matmul(float* y, const float* x, const void* weights, int S, int I, int O);
extern "C" void launch_q2_K_matmul(float* y, const float* x, const void* weights, int S, int I, int O);
extern "C" void launch_q3_K_matmul(float* y, const float* x, const void* weights, int S, int I, int O);
extern "C" void launch_q4_K_matmul(float* y, const float* x, const void* weights, int S, int I, int O);
extern "C" void launch_q5_K_matmul(float* y, const float* x, const void* weights, int S, int I, int O);
extern "C" void launch_q6_K_matmul(float* y, const float* x, const void* weights, int S, int I, int O);
extern "C" void launch_iq2_xxs_matmul(float* y, const float* x, const void* weights, int S, int I, int O);
extern "C" void launch_iq2_xs_matmul(float* y, const float* x, const void* weights, int S, int I, int O);
extern "C" void launch_iq2_s_matmul(float* y, const float* x, const void* weights, int S, int I, int O);
extern "C" void launch_iq3_xxs_matmul(float* y, const float* x, const void* weights, int S, int I, int O);
extern "C" void launch_iq4_xs_matmul(float* y, const float* x, const void* weights, int S, int I, int O);
// MXFP4: dormant kernel (see kquant_dequant.hpp's gpu_type_supported()) --
// wired here for Test 5 harness parity even though this switch is only
// reached once gpu_type_supported() itself has already gated the type.
extern "C" void launch_mxfp4_matmul(float* y, const float* x, const void* weights, int S, int I, int O);
extern "C" void launch_rmsnorm(float* y, const float* x, const float* w, int S, int D, float eps);

class CudaManager {
public:
    GpuTopology topology;
    int device_id = 0;  // Core engine GPU device ID
    cublasHandle_t cublas_handle = nullptr;  // cuBLAS handle, bound to core compute_stream

    // Persistent device buffers on core GPU (never freed mid-session)
    float* d_x = nullptr;
    float* d_y = nullptr;
    size_t d_x_cap = 0;
    size_t d_y_cap = 0;
    // Grow-only scratch for the BF16-matched fast-path's per-call activation
    // conversion (dense_matmul_bf16_matched_host) -- separate from d_x/d_y
    // above since those are F32-typed and shared by the fused-kernel /
    // mixed-type dispatch paths; this one holds the freshly-converted BF16
    // activation each call, half the byte size of the equivalent d_x.
    void* d_x_bf16 = nullptr;
    size_t d_x_bf16_cap = 0;
    // Persistent HOST-side scratch for the same per-call activation
    // conversion, grow-only and reused across calls. Originally this was a
    // fresh std::vector<uint16_t> allocated (and freed) on every single
    // call -- real-hardware use showed visibly elevated per-token CPU usage
    // after the BF16 fast-path landed, traced to exactly this: the old
    // dense_matmul_host path did a direct cudaMemcpy straight from x_host
    // with zero host-side CPU work, so the heap alloc/free churn here (once
    // per converted layer, per token, on the single main decode thread) was
    // new cost that didn't exist before. Reusing one grow-only buffer
    // removes that churn; the conversion loop itself stays scalar (not
    // AVX) -- cheap enough at real S=1 decode widths that vectorizing it
    // wasn't judged worth the risk here, unlike this project's other
    // per-element dequant loops which run over far larger data.
    uint16_t* h_x_bf16_scratch = nullptr;
    size_t h_x_bf16_scratch_cap = 0;

    // Pinned host staging buffer for async H2D transfers
    float* h_pinned = nullptr;
    size_t h_pinned_cap = 0;

    // Single-pool arena for bulk dense-tensor offload (main.cpp's startup
    // loop over resident_layers) — one cudaMalloc, offset-based
    // suballocation for every tensor, instead of one cudaMalloc PER tensor
    // (~1182 separate allocations previously). Eliminates per-allocation
    // fragmentation the same way GpuTopology::allocate_expert_cache's slab
    // allocator already does for the expert VRAM cache; ~1182 tensors of
    // wildly varying sizes fragmenting a near-full device was the remaining
    // cause of dense tensors not fitting even after freeing dead allocations
    // elsewhere (see gpu_topology.hpp's removed d_hidden/d_scratch and
    // graph_initializer.cpp's removed GPU KV cache mirror).
    uint8_t* dense_pool = nullptr;
    size_t dense_pool_cap = 0;
    size_t dense_pool_used = 0;

    // Reports actual driver-reported free/used/total VRAM for every known
    // device (not MoEStream's own bookkeeping — this is ground truth from
    // cudaMemGetInfo, so it reflects everything on the GPU including
    // whatever else is running on the machine). Call at checkpoints around
    // allocation-heavy steps to see exactly where headroom runs out, rather
    // than inferring it from which individual tensors failed to offload.
    void report_vram_usage(const char* label) {
        for (auto& dev : topology.devices) {
            if (dev.role == GpuRole::Unused) continue;
            cudaSetDevice(dev.id);
            size_t free_b = 0, total_b = 0;
            if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) continue;
            const char* role_str = (dev.role == GpuRole::CoreEngine) ? "CoreEngine" : "ExpertVault";
            LOG_INFO("[VRAM] %-28s device=%d (%s, %s): used=%.2f GB free=%.2f GB total=%.2f GB\n",
                     label, dev.id, dev.name.c_str(), role_str,
                     (total_b - free_b) / 1e9, free_b / 1e9, total_b / 1e9);
        }
        cudaSetDevice(device_id);  // restore core device as current, matching pre-call state
    }

    // Categorized VRAM breakdown across every known device -- goes beyond
    // report_vram_usage's single used/free/total number by attributing
    // each device's usage to a specific subsystem: CoreEngine = regular
    // dense pool (dense_pool_used, real ground truth from
    // offload_tensor_to_pool, not the pre-allocation estimate) +
    // attn_output's BF16 fast-path buffers (passed in, since only the
    // caller in main.cpp tracks how many layers converted and how many
    // bytes that used); ExpertVault = MoE expert cache pool
    // (dev.expert_cache_size, already tracked by allocate_expert_cache).
    // Whatever's left after subtracting the known subsystems is logged as
    // "other" -- a REMAINDER against real cudaMemGetInfo ground truth
    // (CUDA context, cuBLAS workspace, pinned staging, small scratch
    // buffers like d_x/d_y/d_x_bf16), not a second independent estimate
    // that could silently drift from what's actually on the device.
    //
    // Added specifically because a real run made it hard to tell, from
    // report_vram_usage's one number alone, how much of the CoreEngine
    // GPU's usage the BF16 attn_output fast-path itself was responsible
    // for versus the (much larger) regular dense pool -- see CLAUDE.md's
    // "BF16 fast-path" section for the hand-derived math this log line is
    // meant to make visible directly instead.
    void report_vram_breakdown(const char* label,
                                size_t attn_output_fastpath_bytes,
                                int attn_output_layers_converted) {
        for (auto& dev : topology.devices) {
            if (dev.role == GpuRole::Unused) continue;
            cudaSetDevice(dev.id);
            size_t free_b = 0, total_b = 0;
            if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) continue;
            size_t used_b = total_b - free_b;

            if (dev.role == GpuRole::CoreEngine) {
                // dev.expert_cache_size is normally 0 here -- only nonzero
                // once add_dual_role_expert_cache (gpu_topology.hpp) has
                // given this device its own small MoE cache slice. Included
                // in `accounted` so that slice doesn't get misattributed to
                // "other" once it exists.
                size_t accounted = dense_pool_used + attn_output_fastpath_bytes + dev.expert_cache_size;
                size_t other_b = (used_b > accounted) ? (used_b - accounted) : 0;
                LOG_INFO("[VRAM-BREAKDOWN] %s device=%d (%s, CoreEngine):\n",
                         label, dev.id, dev.name.c_str());
                LOG_INFO("  dense pool (quantized, regular path): %.3f GB\n",
                         dense_pool_used / 1e9);
                LOG_INFO("  attn_output fast-path (BF16):         %.3f GB (%d layer(s))\n",
                         attn_output_fastpath_bytes / 1e9, attn_output_layers_converted);
                if (dev.expert_cache_size > 0) {
                    LOG_INFO("  MoE expert cache (dual-role slice):   %.3f GB\n",
                             dev.expert_cache_size / 1e9);
                }
                LOG_INFO("  other (CUDA ctx / cuBLAS / scratch):  %.3f GB\n", other_b / 1e9);
            } else {
                size_t moe_b = dev.expert_cache_size;
                size_t other_b = (used_b > moe_b) ? (used_b - moe_b) : 0;
                LOG_INFO("[VRAM-BREAKDOWN] %s device=%d (%s, ExpertVault):\n",
                         label, dev.id, dev.name.c_str());
                LOG_INFO("  MoE expert cache pool:                %.3f GB\n", moe_b / 1e9);
                LOG_INFO("  other (CUDA ctx / cuBLAS / scratch):  %.3f GB\n", other_b / 1e9);
            }
            LOG_INFO("  ---> used=%.2f GB free=%.2f GB total=%.2f GB\n",
                     used_b / 1e9, free_b / 1e9, total_b / 1e9);
        }
        cudaSetDevice(device_id);  // restore core device as current, matching pre-call state
    }

    // Free bytes on the core GPU right now (ground truth via cudaMemGetInfo,
    // not MoEStream's own bookkeeping). Used to size the dense-tensor pool
    // when the full requested size doesn't fit — see main.cpp's backoff.
    size_t core_free_vram_bytes() {
        cudaSetDevice(device_id);
        size_t free_b = 0, total_b = 0;
        cudaMemGetInfo(&free_b, &total_b);
        return free_b;
    }

    bool initialize() {
        if (!topology.initialize()) return false;

        device_id = (topology.core_gpu >= 0) ? topology.devices[topology.core_gpu].id : 0;
        cudaSetDevice(device_id);

        // Enable P2P access between device pairs (for cross-GPU vector transfers)
        for (size_t i = 0; i < topology.devices.size(); ++i) {
            for (size_t j = 0; j < topology.devices.size(); ++j) {
                if (i == j) continue;
                int can_peer = 0;
                cudaDeviceCanAccessPeer(&can_peer, topology.devices[i].id, topology.devices[j].id);
                if (can_peer) {
                    cudaSetDevice(topology.devices[i].id);
                    cudaDeviceEnablePeerAccess(topology.devices[j].id, 0);
                    // Ignore errors (may already be enabled)
                }
            }
        }
        cudaSetDevice(device_id);

        // cuBLAS handle for unquantized dense matmul (F32/F16/BF16). Left on the
        // DEFAULT stream (not bound to compute_stream) so dense_matmul_host can
        // mix synchronous H2D/D2H memcpy + dequant kernel launches + cuBLAS on a
        // single stream without cross-stream ordering hazards. Async overlap is
        // a later optimization.
        cublasCreate(&cublas_handle);

        // Allocate 4 MB pinned staging
        cudaMallocHost((void**)&h_pinned, 4 << 20);
        h_pinned_cap = 4 << 20;

        return true;
    }

    ~CudaManager() {
        cudaSetDevice(device_id);
        if (cublas_handle) cublasDestroy(cublas_handle);
        if (d_x) cudaFree(d_x);
        if (d_y) cudaFree(d_y);
        if (d_x_bf16) cudaFree(d_x_bf16);
        if (h_x_bf16_scratch) delete[] h_x_bf16_scratch;
        if (h_pinned) cudaFreeHost(h_pinned);
        if (dense_pool) cudaFree(dense_pool);
    }

    // Allocate the single dense-tensor pool. Call once, sized to the sum of
    // every tensor's aligned byte requirement, BEFORE any
    // offload_tensor_to_pool() calls. Returns false if even this one
    // allocation doesn't fit (caller should re-size and retry, or fall back
    // to per-tensor offload_tensor_to_vram for graceful partial degradation).
    bool init_dense_pool(size_t total_bytes) {
        if (total_bytes == 0) return true;  // nothing to allocate, trivially fine
        cudaSetDevice(device_id);
        cudaError_t err = cudaMalloc((void**)&dense_pool, total_bytes);
        if (err != cudaSuccess) {
            LOG_ERROR("[CUDA] Dense tensor pool allocation failed: %s (requested %.2f GB)\n",
                      cudaGetErrorString(err), total_bytes / 1e9);
            dense_pool = nullptr;
            dense_pool_cap = 0;
            return false;
        }
        dense_pool_cap = total_bytes;
        dense_pool_used = 0;
        return true;
    }

    // Suballocate qt's bytes from the pool (256-byte aligned — matches
    // cudaMalloc's own natural alignment guarantee, small enough to waste
    // negligible space across ~1000 tensors, large enough for coalesced
    // access) and H2D-copy qt.host_ptr into it. False if the pool doesn't
    // have room left — since the pool is normally sized to fit everything
    // requested, this should only trigger if init_dense_pool was sized to a
    // deliberately-reduced budget (see main.cpp's free-VRAM-based backoff).
    bool offload_tensor_to_pool(QuantizedTensor& qt) {
        if (qt.device_ptr) return true;  // already offloaded
        if (!qt.host_ptr || !dense_pool) return false;

        constexpr size_t ALIGN = 256;
        size_t aligned_used = (dense_pool_used + ALIGN - 1) & ~(ALIGN - 1);
        if (aligned_used + qt.meta.size_bytes > dense_pool_cap) return false;

        cudaSetDevice(device_id);
        qt.device_ptr = dense_pool + aligned_used;
        PERF_START(upload);
        cudaMemcpy(qt.device_ptr, qt.host_ptr, qt.meta.size_bytes, cudaMemcpyHostToDevice);
        PERF_END(upload, t_gpu_upload);
        perf().bytes_gpu_uploaded.fetch_add(qt.meta.size_bytes, std::memory_order_relaxed);

        dense_pool_used = aligned_used + qt.meta.size_bytes;
        return true;
    }

    // Offload a dense tensor permanently to the core engine GPU's VRAM.
    // Returns true if it actually ended up device-resident.
    bool offload_tensor_to_vram(QuantizedTensor& qt) {
        if (qt.device_ptr) return true;  // already offloaded
        if (!qt.host_ptr) return false;

        cudaSetDevice(device_id);
        PERF_START(upload);
        cudaError_t err = cudaMalloc((void**)&qt.device_ptr, qt.meta.size_bytes);
        if (err != cudaSuccess) {
            // No budget check exists anywhere in this path — it just tries
            // to malloc every matched tensor until the GPU physically runs
            // out of room, previously with zero error checking, so a
            // tensor that didn't fit silently stayed host-only forever
            // (indistinguishable from "never attempted" in dense_proj's
            // fallback log). Loud now so this is diagnosable: check total
            // resident dense-tensor size against the core GPU's actual free
            // VRAM (minus KV cache / driver overhead) if you see this.
            LOG_ERROR("[CUDA] Failed to offload %.2f MB to VRAM (device %d): %s — "
                      "staying host-resident (CPU dequant-matmul fallback)\n",
                      qt.meta.size_bytes / 1e6, device_id, cudaGetErrorString(err));
            qt.device_ptr = nullptr;
            return false;
        }
        cudaMemcpy(qt.device_ptr, qt.host_ptr, qt.meta.size_bytes, cudaMemcpyHostToDevice);
        PERF_END(upload, t_gpu_upload);
        perf().bytes_gpu_uploaded.fetch_add(qt.meta.size_bytes, std::memory_order_relaxed);
        return true;
    }

    // Dequantize a static per-layer dense weight to BF16 ONCE (host-side, via
    // the same ggml reference dequant_row used by the CPU fallback, then
    // rounded to bf16 via fp32_to_bf16) and upload the result to its OWN
    // standalone cudaMalloc buffer (deliberately NOT a dense_pool
    // suballocation, so this can be called independent of pool sizing/
    // lifetime) — qt.gpu_fast_ptr. dense_proj then dispatches through
    // dense_matmul_bf16_matched_host (cuBLAS, matched BF16 weight/BF16
    // activation) instead of qt's original-quant-type fused kernel
    // (Q2_K..Q6_K/IQ2*/IQ3_XXS), which has no GEMV specialization and is
    // disproportionately slow at S=1 decode — see attn_output's ~623ms/
    // token cost in CLAUDE.md, the motivating case. host_ptr/device_ptr/
    // meta.type are untouched, so the CPU fallback stays correct regardless.
    //
    // RE-ADDED AFTER TWO PRIOR ROUNDS (see CLAUDE.md for the full history):
    // (1) an earlier F32 version of this fast-path worked but only covered
    // ~19% of layers within a real VRAM budget; (2) a bf16 variant using a
    // MISMATCHED-type cublasGemmEx call (BF16 weight, F32 activation) was
    // tried, produced real NaN corruption on real hardware, and was fully
    // reverted -- root-caused MUCH later (via real perf/nsys profiling and
    // a separate, unrelated concurrency-race fix) to NOT actually be
    // caused by BF16 itself; a dedicated isolated test (Test 9,
    // main_test.cpp) subsequently confirmed on real hardware that the
    // mismatched-type combination is cleanly REJECTED by cublasGemmEx
    // (CUBLAS_STATUS_NOT_SUPPORTED), not silently wrong -- so it was never
    // going to work at all, mismatch or not. This version uses the
    // MATCHED-type combination instead (BF16 weight AND BF16 activation),
    // verified correct in isolation first via Test 10 (real hardware,
    // max_rel_diff ~1.18e-5) before being wired in here.
    //
    // Best-effort: returns false (leaving gpu_fast_ptr null, caller keeps
    // using the existing slower path) if the bf16 expansion doesn't fit in
    // max_bytes, the shape is malformed, or the quant type has no CPU
    // reference dequant. bf16 is ~2-3x the size of a K-quant/IQ original
    // (e.g. ~2.9x for Q5_K, vs ~5.8x it would be for F32), so this is still
    // capacity-checked per call rather than assumed to fit — callers should
    // pass real remaining VRAM headroom, not the tensor's own original size.
    bool upload_dequantized_bf16(QuantizedTensor& qt, size_t max_bytes) {
        if (qt.gpu_fast_ptr) return true;  // already converted
        // HARD TYPE GATE: only genuinely lossless-relative-to-storage F32
        // source data may be rounded to BF16 here. Every call site (gate_inp,
        // attn_output) matches by tensor NAME only, never by qt.meta.type --
        // without this gate, any quantized tensor matching those name
        // patterns (Q5_K/Q6_K/Q8_0/IQ2_*/IQ3_XXS, whatever this checkpoint
        // happens to use) would silently get dequantized to F32 then rounded
        // to BF16 here, discarding that quant format's own encoding entirely
        // -- a real precision/behavior change the user explicitly does NOT
        // want for anything already quantized. F32 is the only case where
        // "round to BF16" is the ONLY loss involved (halves storage, no
        // format switch); for every other type, the CPU-fallback dequant
        // reference distinguishes cleanly between quant formats and this
        // fast-path must not collapse that distinction. Gated here (the
        // single choke point both call sites route through) rather than at
        // each call site, so no future caller can bypass it either.
        if (qt.meta.type != GGUFType::F32) return false;
        if (!qt.host_ptr || qt.meta.shape.size() < 2) return false;
        int O = (int)qt.meta.shape[0];
        int I = (int)qt.meta.shape[1];
        if (O <= 0 || I <= 0) return false;
        size_t n = (size_t)O * (size_t)I;
        size_t bf16_bytes = n * sizeof(uint16_t);
        if (bf16_bytes > max_bytes) return false;

        std::vector<float> host_f32(n);
        if (!dequant_tensor_f32(qt.meta.type, qt.host_ptr, O, I, host_f32.data())) {
            LOG_WARN("[CUDA] Fast-path BF16 dequant unsupported for type=%u — "
                     "keeping original quantized GPU path\n", (unsigned)qt.meta.type);
            return false;
        }
        std::vector<uint16_t> host_bf16(n);
        for (size_t i = 0; i < n; ++i) host_bf16[i] = fp32_to_bf16(host_f32[i]);

        cudaSetDevice(device_id);
        cudaError_t err = cudaMalloc((void**)&qt.gpu_fast_ptr, bf16_bytes);
        if (err != cudaSuccess) {
            LOG_ERROR("[CUDA] Fast-path BF16 upload failed (%.2f MB): %s — "
                      "keeping original quantized GPU path\n",
                      bf16_bytes / 1e6, cudaGetErrorString(err));
            qt.gpu_fast_ptr = nullptr;
            return false;
        }
        cudaMemcpy(qt.gpu_fast_ptr, host_bf16.data(), bf16_bytes, cudaMemcpyHostToDevice);
        perf().bytes_gpu_uploaded.fetch_add(bf16_bytes, std::memory_order_relaxed);
        return true;
    }

    // Per-call dispatch for a BF16-fast-path-converted weight (qt.gpu_fast_ptr,
    // set by upload_dequantized_bf16 above). Unlike the weight (converted
    // ONCE at load and cached for the process lifetime), the activation x
    // is different every call, so it must be converted to bf16 fresh each
    // time -- real, new per-call work that the earlier F32 fast-path never
    // had (F32 activations needed no conversion, just upload). Host-side
    // conversion (reusing fp32_to_bf16, the same function Test 10 verified
    // in isolation) keeps this simple: no new CUDA kernel, just a smaller
    // H2D upload than the F32 path would have needed.
    bool dense_matmul_bf16_matched_host(const void* d_W_bf16,
                                         const float* x_host, float* y_host,
                                         int S, int I, int O) {
        if (!d_W_bf16) return false;
        size_t n_x = (size_t)S * I;
        size_t x_bf16_bytes = n_x * sizeof(uint16_t);
        size_t y_bytes = (size_t)S * O * sizeof(float);

        cudaSetDevice(device_id);
        if (x_bf16_bytes > d_x_bf16_cap) {
            if (d_x_bf16) { cudaFree(d_x_bf16); d_x_bf16 = nullptr; }
            d_x_bf16_cap = 0;
            cudaError_t err = cudaMalloc(&d_x_bf16, x_bf16_bytes);
            if (err != cudaSuccess) {
                LOG_ERROR("[CUDA] dense_matmul_bf16_matched_host: d_x_bf16 grow to %.2f MB failed: %s\n",
                          x_bf16_bytes / 1e6, cudaGetErrorString(err));
                return false;
            }
            d_x_bf16_cap = x_bf16_bytes;
        }
        if (!ensure_buffers(0, y_bytes)) return false;  // reuses/grows d_y (F32 output buffer)

        // Grow-only reuse of h_x_bf16_scratch (see member comment) instead
        // of a fresh std::vector every call -- eliminates the heap
        // alloc/free churn that was the real source of elevated per-token
        // CPU usage observed on real hardware after this fast-path landed.
        if (n_x > h_x_bf16_scratch_cap) {
            if (h_x_bf16_scratch) delete[] h_x_bf16_scratch;
            h_x_bf16_scratch = new uint16_t[n_x];
            h_x_bf16_scratch_cap = n_x;
        }
        for (size_t i = 0; i < n_x; ++i) h_x_bf16_scratch[i] = fp32_to_bf16(x_host[i]);

        PERF_START(dense_bf16_up);
        cudaMemcpy(d_x_bf16, h_x_bf16_scratch, x_bf16_bytes, cudaMemcpyHostToDevice);
        PERF_END(dense_bf16_up, t_dense_gpu_up);

        PERF_START(dense_bf16_cmp);
        cublasStatus_t status = cublas_dense_matmul_bf16_matched(
            cublas_handle, d_W_bf16, d_x_bf16, d_y, S, I, O);
        PERF_END(dense_bf16_cmp, t_dense_gpu_compute);
        if (status != CUBLAS_STATUS_SUCCESS) return false;

        PERF_START(dense_bf16_dn);
        cudaMemcpy(y_host, d_y, y_bytes, cudaMemcpyDeviceToHost);
        PERF_END(dense_bf16_dn, t_dense_gpu_down);
        return true;
    }

    // Allocate expert VRAM cache on vault GPU(s) based on model geometry
    void setup_expert_cache(size_t expert_slab_bytes, int n_experts, const EngineConfig& config) {
        topology.allocate_expert_cache(expert_slab_bytes, n_experts, &config);

        // Three-tier caching: Allocate active layer area for temporary expert storage
        // This provides a workspace for current-layer experts without evicting LRU residents
        if (config.use_active_layer_area) {
            if (!topology.allocate_active_layer_area(expert_slab_bytes, config.active_layer_capacity)) {
                LOG_STREAM_WARN() << "[GPU] Active layer area allocation failed - "
                                   "three-tier caching will fall back to standard LRU behavior\n";
            } else {
                LOG_STREAM_INFO() << "[GPU] Three-tier: active layer area enabled for temporary expert storage\n";
            }
        } else {
            LOG_STREAM_INFO() << "[GPU] Three-tier: active layer area disabled by configuration\n";
        }
    }

    // Grow-only persistent device buffers for matmul I/O. Returns false if
    // a grow was needed and the cudaMalloc failed -- previously unchecked,
    // meaning a failed grow silently left d_x/d_y null (or, worse, freed
    // but with a stale cap that made a LATER same-size call skip growing
    // again) while every caller proceeded to memcpy/kernel-launch into it
    // regardless. That's this project's worst-case failure signature
    // (silent corruption from a bad pointer, not an obvious crash) applied
    // to a path this session's prefill-scratch-sizing investigation showed
    // really can grow by hundreds of MB for a long prompt. On failure, the
    // cap is reset to 0 (not left at the stale pre-failure value) so the
    // NEXT call retries the allocation cleanly instead of wrongly assuming
    // a null buffer is already sized.
    bool ensure_buffers(size_t x_bytes, size_t y_bytes) {
        cudaSetDevice(device_id);
        if (x_bytes > d_x_cap) {
            if (d_x) { cudaFree(d_x); d_x = nullptr; }
            d_x_cap = 0;
            cudaError_t err = cudaMalloc((void**)&d_x, x_bytes);
            if (err != cudaSuccess) {
                LOG_ERROR("[CUDA] ensure_buffers: d_x grow to %.2f MB failed: %s\n",
                          x_bytes / 1e6, cudaGetErrorString(err));
                return false;
            }
            d_x_cap = x_bytes;
        }
        if (y_bytes > d_y_cap) {
            if (d_y) { cudaFree(d_y); d_y = nullptr; }
            d_y_cap = 0;
            cudaError_t err = cudaMalloc((void**)&d_y, y_bytes);
            if (err != cudaSuccess) {
                LOG_ERROR("[CUDA] ensure_buffers: d_y grow to %.2f MB failed: %s\n",
                          y_bytes / 1e6, cudaGetErrorString(err));
                return false;
            }
            d_y_cap = y_bytes;
        }
        return true;
    }

    // Frees d_x/d_y/d_x_bf16 back to empty (cap=0), WITHOUT touching
    // h_x_bf16_scratch (host RAM, not VRAM -- not what this is about) or
    // any other persistent buffer. Safe to call any time between requests:
    // the next dense_proj call regrows whatever it actually needs via the
    // SAME grow-only logic above/in dense_matmul_bf16_matched_host, just
    // starting from 0 instead of a stale larger cap. Added specifically
    // because these buffers grow to fit a prefill's full S (dense/attention
    // projections run on the whole batch in one GEMM) and, without this,
    // stay that size for the rest of the session even though decode only
    // ever needs S=1 -- real, measured impact for a long prompt (~120 MB
    // at S=1000, ~240 MB at S=2000 for this project's GLM-5.2 dims) against
    // a CoreEngine GPU that can be left with under 1.2 GB free. Called from
    // generate_text right after the single prefill call completes, before
    // the decode loop begins.
    void shrink_dense_scratch() {
        cudaSetDevice(device_id);
        if (d_x) { cudaFree(d_x); d_x = nullptr; }
        if (d_y) { cudaFree(d_y); d_y = nullptr; }
        if (d_x_bf16) { cudaFree(d_x_bf16); d_x_bf16 = nullptr; }
        d_x_cap = 0;
        d_y_cap = 0;
        d_x_bf16_cap = 0;
    }

    void ensure_pinned(size_t bytes) {
        if (bytes > h_pinned_cap) {
            if (h_pinned) cudaFreeHost(h_pinned);
            cudaMallocHost((void**)&h_pinned, bytes);
            h_pinned_cap = bytes;
        }
    }

    // Execute quantized matmul on core GPU: persistent buffers + pinned + streams
    void execute_matmul(const QuantizedTensor& w, const float* x_host, float* y_host, int S) {
        int I = w.meta.shape[1];
        int O = w.meta.shape[0];
        size_t x_bytes = (size_t)S * I * sizeof(float);
        size_t y_bytes = (size_t)S * O * sizeof(float);

        if (!ensure_buffers(x_bytes, y_bytes)) {
            LOG_ERROR("[CUDA] execute_matmul: buffer grow failed, aborting this call\n");
            return;
        }
        ensure_pinned(std::max(x_bytes, y_bytes));

        cudaSetDevice(device_id);
        auto& core = topology.devices[topology.core_gpu >= 0 ? topology.core_gpu : 0];

        // Async H2D via pinned staging
        PERF_START(upload);
        std::memcpy(h_pinned, x_host, x_bytes);
        cudaMemcpyAsync(d_x, h_pinned, x_bytes, cudaMemcpyHostToDevice, core.copy_stream);
        cudaStreamSynchronize(core.copy_stream);
        PERF_END(upload, t_gpu_upload);
        perf().bytes_gpu_uploaded.fetch_add(x_bytes, std::memory_order_relaxed);

        // Kernel launch
        PERF_START(compute);
        launch_gguf_q4_0_dp4a(d_y, d_x, w.device_ptr, S, I, O);
        cudaStreamSynchronize(core.compute_stream);
        PERF_END(compute, t_gpu_compute);

        // D2H result
        cudaMemcpy(y_host, d_y, y_bytes, cudaMemcpyDeviceToHost);
    }

    // Unquantized dense matmul on the core GPU via cuBLAS:
    //   y[S,O] = x[S,I] @ W^T,  W stored row-major [O,I] at d_W (device).
    // Handles F32/F16/BF16 weights (output always F32); returns false for any
    // quant type so the caller routes those to the dequant-matmul kernels.
    // d_W/d_X/d_Y must be device-resident — this does no host staging. The
    // handle runs on the core compute_stream, so callers staging x on another
    // stream must order/sync first.
    bool dense_matmul(GGUFType wtype,
                      const void* d_W, const float* d_X, float* d_Y,
                      int S, int I, int O) {
        if (!cublas_handle) return false;
        return cublas_dense_matmul(cublas_handle, wtype, d_W, d_X, d_Y, S, I, O)
               == CUBLAS_STATUS_SUCCESS;
    }

    // Host-staged dense matmul for the forward path: uploads x[S,I] to device,
    // dispatches by weight type (cuBLAS for F32/F16/BF16, a fused dequant kernel
    // for Q8_0; Q2_K/Q3_K/Q4_K/Q5_K/Q6_K/IQ2_*/IQ3_XXS land here as they're
    // written), downloads y[S,O]. W must already be device-resident (device_ptr set by
    // offload_tensor_to_vram). Returns false for any type without a GPU kernel
    // yet — callers must treat that as "unsupported" and NOT silently recompute.
    // Everything runs on the default stream with synchronous memcpy, so ordering
    // is correct-by-construction (no async overlap yet).
    //
    // BF16 is special-cased and delegated to dense_matmul_bf16_matched_host
    // BEFORE the generic F32 x-upload below, rather than going through the
    // dense_matmul()/cublas_dense_matmul() path like F32/F16 still do. Why:
    // cublas_dense_matmul's BF16 case calls cublasGemmEx with a MISMATCHED
    // type combination (Atype=CUDA_R_16BF weight, Btype=CUDA_R_32F
    // activation) that Test 9 (src/main_test.cpp) proved cuBLAS rejects
    // outright on real hardware (CUBLAS_STATUS_NOT_SUPPORTED) -- so any
    // dense tensor that's natively BF16 in the GGUF (e.g. a model file
    // pre-converted to BF16 to avoid runtime dequant) would silently fail
    // GPU dispatch here and fall through to the CPU path every time, with
    // no crash and no obvious signal beyond a DEBUG-level "no GPU
    // dispatch" log line. dense_matmul_bf16_matched_host uses the MATCHED
    // combination instead (weight AND activation both BF16), verified
    // correct via Test 10 on real hardware -- this reuses that same,
    // already-verified function rather than adding a second mismatched-
    // type call site.
    bool dense_matmul_host(GGUFType wtype, const void* d_W,
                           const float* x_host, float* y_host,
                           int S, int I, int O) {
        if (!d_W) return false;
        if (wtype == GGUFType::BF16) {
            return dense_matmul_bf16_matched_host(d_W, x_host, y_host, S, I, O);
        }
        size_t x_bytes = (size_t)S * I * sizeof(float);
        size_t y_bytes = (size_t)S * O * sizeof(float);
        if (!ensure_buffers(x_bytes, y_bytes)) return false;  // caller (dense_proj) falls back to CPU
        cudaSetDevice(device_id);

        // Phase split -- added after attn_output's dense_proj call (which
        // routes through here) was found costing ~623ms/token, ~5x its
        // qproj sibling calls, with zero CPU fallback logged (so it IS
        // successfully GPU-dispatching every time) -- this pins down
        // whether the cost is the H2D copy, the kernel itself, or the D2H
        // copy, same method that found the real expert-FFN GPU issue
        // earlier this session.
        PERF_START(dense_up);
        cudaMemcpy(d_x, x_host, x_bytes, cudaMemcpyHostToDevice);
        PERF_END(dense_up, t_dense_gpu_up);

        PERF_START(dense_cmp);
        bool ok = false;
        switch (wtype) {
            case GGUFType::F32:
            case GGUFType::F16:
                // BF16 no longer reaches this switch -- handled by the
                // early-return delegation to dense_matmul_bf16_matched_host
                // above (see that early-return's comment for why).
                ok = dense_matmul(wtype, d_W, d_x, d_y, S, I, O);
                break;
            case GGUFType::Q8_0:
                launch_q8_0_matmul(d_y, d_x, d_W, S, I, O);
                ok = true;
                break;
            case GGUFType::Q2_K:
                launch_q2_K_matmul(d_y, d_x, d_W, S, I, O);
                ok = true;
                break;
            case GGUFType::Q3_K:
                launch_q3_K_matmul(d_y, d_x, d_W, S, I, O);
                ok = true;
                break;
            case GGUFType::Q4_K:
                launch_q4_K_matmul(d_y, d_x, d_W, S, I, O);
                ok = true;
                break;
            case GGUFType::Q5_K:
                launch_q5_K_matmul(d_y, d_x, d_W, S, I, O);
                ok = true;
                break;
            case GGUFType::Q6_K:
                launch_q6_K_matmul(d_y, d_x, d_W, S, I, O);
                ok = true;
                break;
            case GGUFType::IQ2_XXS:
                launch_iq2_xxs_matmul(d_y, d_x, d_W, S, I, O);
                ok = true;
                break;
            case GGUFType::IQ2_XS:
                launch_iq2_xs_matmul(d_y, d_x, d_W, S, I, O);
                ok = true;
                break;
            case GGUFType::IQ2_S:
                launch_iq2_s_matmul(d_y, d_x, d_W, S, I, O);
                ok = true;
                break;
            case GGUFType::IQ3_XXS:
                launch_iq3_xxs_matmul(d_y, d_x, d_W, S, I, O);
                ok = true;
                break;
            case GGUFType::IQ4_XS:
                launch_iq4_xs_matmul(d_y, d_x, d_W, S, I, O);
                ok = true;
                break;
            // MXFP4: reachable here (dense_matmul_host is called directly by
            // src/main_test.cpp's Test 5 harness, bypassing gpu_type_supported()'s
            // gate) so the kernel can be validated on real hardware BEFORE
            // gpu_type_supported() flips it on for the live expert-FFN path.
            case GGUFType::MXFP4:
                launch_mxfp4_matmul(d_y, d_x, d_W, S, I, O);
                ok = true;
                break;
            // IQ1_* / IQ2_M : no GPU fused kernel — CPU dequant (libggml) handles
            // IQ1_* via the caller's cpu_quant_matmul fallback; IQ2_M has no ggml
            // dequant export and loud-zeros.
            default:
                ok = false;
                break;
        }
        // Kernel launches (and dense_matmul's cuBLAS call) are async -- an
        // explicit sync here is required for the timing window to capture
        // real device execution time, not just launch overhead. The
        // downstream D2H cudaMemcpy below would implicitly wait for this
        // anyway (default-stream blocking semantics), but that would fold
        // compute time into t_dense_gpu_down instead of attributing it
        // correctly.
        cudaStreamSynchronize(0);
        PERF_END(dense_cmp, t_dense_gpu_compute);
        if (!ok) return false;

        PERF_START(dense_dn);
        cudaMemcpy(y_host, d_y, y_bytes, cudaMemcpyDeviceToHost);
        PERF_END(dense_dn, t_dense_gpu_down);
        perf().n_dense_gpu_calls.fetch_add(1, std::memory_order_relaxed);
        return true;
    }

    // Host-staged F32 RMSNorm on GPU: y[S,D] = rmsnorm(x[S,D], w[D]).
    // w must be device-resident (F32 norm weights are offloaded to VRAM).
    bool rmsnorm_gpu_host(const void* d_w, const float* x_host, float* y_host,
                          int S, int D, float eps) {
        if (!d_w) return false;
        size_t bytes = (size_t)S * D * sizeof(float);
        if (!ensure_buffers(bytes, bytes)) return false;  // caller falls back to CPU rmsnorm_row
        cudaSetDevice(device_id);
        cudaMemcpy(d_x, x_host, bytes, cudaMemcpyHostToDevice);
        launch_rmsnorm(d_y, d_x, (const float*)d_w, S, D, eps);
        cudaMemcpy(y_host, d_y, bytes, cudaMemcpyDeviceToHost);
        return true;
    }

    // Transfer a small vector between GPUs (expert output → core for residual add)
    // Uses P2P if available, otherwise bounces through pinned host buffer
    void cross_device_transfer(float* dst, int dst_dev,
                               const float* src, int src_dev, size_t bytes) {
        topology.cross_device_transfer(dst, dst_dev, src, src_dev, bytes);
    }

    // Get hardware profile for conditional logic
    HardwareProfile get_profile() const { return topology.profile; }
    int get_core_device() const { return device_id; }
    int get_vault_device(int idx = 0) const {
        if (idx < (int)topology.vault_gpus.size())
            return topology.devices[topology.vault_gpus[idx]].id;
        return device_id; // fallback to core
    }
};
