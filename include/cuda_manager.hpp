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
// EngineConfig::attn_score_gpu (cuda/attn_score_softmax.cu) -- see
// score_gpu_batched_host below for the full design.
extern "C" void launch_combine_scale_softmax(
    const float* scores_nope, const float* scores_rope, float* scores,
    int H, int T, float scale, cudaStream_t stream);

class CudaManager {
public:
    GpuTopology topology;
    int device_id = 0;  // Core engine GPU device ID
    cublasHandle_t cublas_handle = nullptr;  // cuBLAS handle, bound to core compute_stream

    // Dedicated stream + cuBLAS handle for grouped_output_proj_host
    // (dsv4_outproj_gpu) -- deliberately NOT the shared cublas_handle above,
    // which is left on CUDA's legacy default stream (stream 0, see that
    // field's own init comment). Real-hardware A/B found that sharing
    // stream 0 for this call site cost MORE in MoE expert dispatch than it
    // saved in attention: the legacy default stream implicitly
    // synchronizes with EVERY other stream on the device unless per-thread
    // default streams are used, and this call fires 43x/token (once per
    // layer) -- each one a synchronization barrier landing right at the
    // layer boundary where MoE's own async Pass A/Pass B pipeline
    // (core.compute_stream/copy_stream) needs to keep flowing. Measured:
    // attn saved ~3.9s, expert='s own untimed/dark portion grew by ~3.3s,
    // net wash on total throughput. A genuinely separate, non-default
    // stream (created once, never touched by anything else) can never
    // trigger that implicit cross-stream sync, isolating this call site
    // from MoE's dispatch entirely.
    cudaStream_t outproj_stream = nullptr;
    cublasHandle_t outproj_cublas_handle = nullptr;
    float* d_outproj_x = nullptr;
    float* d_outproj_y = nullptr;
    size_t d_outproj_x_cap = 0;
    size_t d_outproj_y_cap = 0;

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

    // MLA qabsorb BF16 fast-path (EngineConfig::attn_qabsorb_gpu) -- see
    // upload_wuk_bf16/qabsorb_bf16_batched_host below for the full design.
    // Per-layer BF16-converted WUK (attn_k_b), one cudaMalloc per layer,
    // populated once at startup by warm_mla_absorb_cache
    // (attention_layer.cpp) and read-only for the rest of the process
    // lifetime -- same "convert once, dispatch many times" shape as
    // QuantizedTensor::gpu_fast_ptr above, just keyed by layer index
    // instead of living on a QuantizedTensor (WUK isn't one -- it's a
    // dequant-cached std::vector<float>, wuk_cache in attention_layer.cpp).
    // Empty vector (not yet sized) means "never attempted"; a null entry
    // at a valid layer index means "attempted and failed/skipped" (VRAM
    // budget exhausted, missing weights, etc.) -- both cases make
    // qabsorb_bf16_batched_host's d_W_bf16-null check return false and the
    // caller falls back to the CPU path for that layer, same discipline as
    // every other best-effort GPU fast-path in this file.
    std::vector<void*> wuk_bf16_by_layer;
    // Grow-only scratch for qabsorb's per-call activation, separate from
    // d_x_bf16/h_x_bf16_scratch above (dense_matmul_bf16_matched_host's own
    // scratch) since qabsorb's shape is [H,S,nope] -- a different size
    // multiplier (H heads) than a single dense_proj call ever uses, and
    // reusing the same buffer would mean re-growing on every alternating
    // call between the two call sites instead of settling at a stable size.
    void* d_qabsorb_x_bf16 = nullptr;
    size_t d_qabsorb_x_bf16_cap = 0;
    float* d_qabsorb_y = nullptr;
    size_t d_qabsorb_y_cap = 0;
    uint16_t* h_qabsorb_x_bf16_scratch = nullptr;
    size_t h_qabsorb_x_bf16_scratch_cap = 0;

    // WUV (attn_v_b) BF16 fast-path -- the score GPU path's own down-
    // projection step, added 2026-08-14 after a real production run showed
    // the CPU per-head OMP-parallel WUV loop costing 0.916ms/call (2.86s
    // over one 40-token decode run), the SAME "OMP team-spawn cost exceeds
    // a tiny per-call workload" pattern qabsorb's own gather fix already
    // hit and fixed at a different call site. Structurally identical to
    // wuk_bf16_by_layer above (H independent per-head [vh,kvl] GEMMs, same
    // strided-batched dispatch), just a different static per-layer weight.
    std::vector<void*> wuv_bf16_by_layer;
    void* d_wuv_ctx_bf16 = nullptr;
    size_t d_wuv_ctx_bf16_cap = 0;
    float* d_wuv_out = nullptr;
    size_t d_wuv_out_cap = 0;
    uint16_t* h_wuv_ctx_bf16_scratch = nullptr;
    size_t h_wuv_ctx_bf16_scratch_cap = 0;

    // MLA score GPU fast-path (EngineConfig::attn_score_gpu) -- see
    // ensure_lc_rc_gpu/upload_lc_rc_rows/score_gpu_batched_host below for the
    // full design. Real-hardware feasibility bench (tests/bench_score_cpu_
    // vs_gpu.cu): 5x-51x speedup over CPU, growing with context length,
    // since unlike qabsorb this phase's CPU cost is UNBOUNDED with context
    // (no sliding window for GLM-5.2). Persistent per-layer GPU mirror of
    // the KV latent cache (engine.Lc/Rc, attention_layer.cpp) -- F32, not
    // BF16, since correctness (not VRAM budget) is the first concern here
    // and the raw compute cost is already tiny per the bench; sized to
    // engine.max_t so it never needs to grow/realloc mid-generation (a
    // realloc would require re-uploading the WHOLE cache, defeating the
    // point of an incrementally-grown mirror). Same null-means-"never
    // attempted-or-failed" discipline as wuk_bf16_by_layer above.
    std::vector<float*> d_lc_by_layer;
    std::vector<float*> d_rc_by_layer;
    std::vector<int> lc_rc_max_t_by_layer;  // allocated capacity per layer, for bounds checks
    // Grow-only scratch, shared across layers/calls (same rationale as
    // qabsorb's own scratch above -- settle at a stable size instead of
    // re-growing every alternating call). Sized to the largest H*T seen so
    // far; T can be up to engine.max_t.
    float* d_score_q_absorbed = nullptr; size_t d_score_q_absorbed_cap = 0;
    float* d_score_q_rope = nullptr;     size_t d_score_q_rope_cap = 0;
    float* d_score_nope_buf = nullptr;   size_t d_score_nope_buf_cap = 0;
    float* d_score_rope_buf = nullptr;   size_t d_score_rope_buf_cap = 0;
    float* d_score_combined = nullptr;   size_t d_score_combined_cap = 0;
    float* d_score_context = nullptr;    size_t d_score_context_cap = 0;

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

        // Dedicated stream + handle for grouped_output_proj_host -- see
        // that field's own comment (above) for why this must NOT share
        // cublas_handle/stream 0. cublasSetStream binds this handle to the
        // new stream for the rest of the process's lifetime; every
        // grouped_output_proj_host call reuses it, never touching stream 0.
        cudaError_t stream_err = cudaStreamCreate(&outproj_stream);
        if (stream_err == cudaSuccess) {
            cublasCreate(&outproj_cublas_handle);
            cublasSetStream(outproj_cublas_handle, outproj_stream);
        } else {
            LOG_STREAM_ERROR() << "[CUDA] outproj_stream creation failed: "
                               << cudaGetErrorString(stream_err)
                               << " -- dsv4_outproj_gpu will be unavailable "
                                  "(grouped_output_proj_host checks for a null handle).\n";
        }

        // Allocate 4 MB pinned staging
        cudaMallocHost((void**)&h_pinned, 4 << 20);
        h_pinned_cap = 4 << 20;

        return true;
    }

    ~CudaManager() {
        cudaSetDevice(device_id);
        if (cublas_handle) cublasDestroy(cublas_handle);
        if (outproj_cublas_handle) cublasDestroy(outproj_cublas_handle);
        if (outproj_stream) cudaStreamDestroy(outproj_stream);
        if (d_outproj_x) cudaFree(d_outproj_x);
        if (d_outproj_y) cudaFree(d_outproj_y);
        if (d_x) cudaFree(d_x);
        if (d_y) cudaFree(d_y);
        if (d_x_bf16) cudaFree(d_x_bf16);
        if (h_x_bf16_scratch) delete[] h_x_bf16_scratch;
        for (void* p : wuk_bf16_by_layer) if (p) cudaFree(p);
        if (d_qabsorb_x_bf16) cudaFree(d_qabsorb_x_bf16);
        if (d_qabsorb_y) cudaFree(d_qabsorb_y);
        if (h_qabsorb_x_bf16_scratch) delete[] h_qabsorb_x_bf16_scratch;
        for (void* p : wuv_bf16_by_layer) if (p) cudaFree(p);
        if (d_wuv_ctx_bf16) cudaFree(d_wuv_ctx_bf16);
        if (d_wuv_out) cudaFree(d_wuv_out);
        if (h_wuv_ctx_bf16_scratch) delete[] h_wuv_ctx_bf16_scratch;
        for (float* p : d_lc_by_layer) if (p) cudaFree(p);
        for (float* p : d_rc_by_layer) if (p) cudaFree(p);
        if (d_score_q_absorbed) cudaFree(d_score_q_absorbed);
        if (d_score_q_rope) cudaFree(d_score_q_rope);
        if (d_score_nope_buf) cudaFree(d_score_nope_buf);
        if (d_score_rope_buf) cudaFree(d_score_rope_buf);
        if (d_score_combined) cudaFree(d_score_combined);
        if (d_score_context) cudaFree(d_score_context);
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

    // MLA qabsorb BF16 fast-path (EngineConfig::attn_qabsorb_gpu), part 1:
    // one-time per-layer weight upload. WUK_host is wuk_cache[layer_idx]
    // (attention_layer.cpp), already dequantized to F32 host memory by
    // ensure_wuk_wuv_cached -- laid out [H,kvl,nope] (H heads, each an
    // independent [kvl,nope] row-major weight, contiguous per-head, matching
    // the CPU matmul() call this replaces: matmul(..., WUK.data()+h*nope*kvl,
    // S, nope, kvl) => per-head W is [O=kvl,I=nope] row-major). Rounds the
    // WHOLE [H,kvl,nope] block to BF16 in one pass and uploads it as one
    // cudaMalloc/cudaMemcpy -- callers already loop layers themselves
    // (warm_mla_absorb_cache), so no separate per-layer capacity-tracking
    // loop is needed here beyond the max_bytes check every fast-path in this
    // file already does.
    //
    // Best-effort, matching upload_dequantized_bf16's own contract: returns
    // false (leaving wuk_bf16_by_layer[layer_idx] null) on any failure --
    // wrong shape, doesn't fit in max_bytes, or the cudaMalloc itself fails
    // -- caller (warm_mla_absorb_cache) just logs and continues; the real
    // attention call falls back to the CPU path for that layer, same
    // discipline as every other best-effort GPU fast-path here.
    bool upload_wuk_bf16(int layer_idx, const std::vector<float>& WUK_host,
                          int H, int nope, int kvl, size_t max_bytes) {
        if (layer_idx < 0) return false;
        size_t n = (size_t)H * (size_t)nope * (size_t)kvl;
        if (n == 0 || WUK_host.size() != n) return false;
        size_t bf16_bytes = n * sizeof(uint16_t);
        if (bf16_bytes > max_bytes) return false;

        if ((int)wuk_bf16_by_layer.size() <= layer_idx) {
            wuk_bf16_by_layer.resize(layer_idx + 1, nullptr);
        }
        if (wuk_bf16_by_layer[layer_idx]) return true;  // already uploaded

        std::vector<uint16_t> host_bf16(n);
        for (size_t i = 0; i < n; ++i) host_bf16[i] = fp32_to_bf16(WUK_host[i]);

        cudaSetDevice(device_id);
        void* d_wuk = nullptr;
        cudaError_t err = cudaMalloc(&d_wuk, bf16_bytes);
        if (err != cudaSuccess) {
            LOG_WARN("[CUDA] qabsorb WUK BF16 upload failed layer=%d (%.2f MB): %s -- "
                     "layer stays on the CPU qabsorb path\n",
                     layer_idx, bf16_bytes / 1e6, cudaGetErrorString(err));
            return false;
        }
        cudaMemcpy(d_wuk, host_bf16.data(), bf16_bytes, cudaMemcpyHostToDevice);
        wuk_bf16_by_layer[layer_idx] = d_wuk;
        perf().bytes_gpu_uploaded.fetch_add(bf16_bytes, std::memory_order_relaxed);
        return true;
    }

    // MLA qabsorb BF16 fast-path, part 2: per-call batched dispatch,
    // Q_absorbed[H,S,kvl] = q_nope_gather[H,S,nope] @ WUK[H,kvl,nope]^T (per
    // head), replacing attention_layer.cpp's #pragma omp parallel for over
    // heads calling the CPU matmul() once per head. ONE H2D upload, ONE
    // cublasGemmStridedBatchedEx call (batches all H heads on-device instead
    // of H separate host-issued cublas calls), ONE D2H download -- avoiding
    // the "per-head loop pays per-head overhead" mistake this codebase's own
    // CPU qabsorb history already hit once (see attention_layer.cpp's
    // qabsorb comment on the OMP-spawn-per-head regression), just for GPU
    // launch/transfer overhead instead of OMP team spawns.
    //
    // x_host: [H,S,nope] row-major F32, host (caller gathers this from Q's
    //   real [S,H,qh] layout -- see mla_attention_absorbed_gguf's call site).
    // y_host: [H,S,kvl] row-major F32, host -- SAME layout Q_absorbed already
    //   uses (Q_absorbed.data() + h*S*kvl), so the caller can write straight
    //   into the existing buffer with no reshuffle.
    // Returns false (caller falls back to the CPU per-head loop) if this
    // layer's WUK was never uploaded (upload_wuk_bf16 never called, or it
    // failed) or any CUDA/cublas call fails -- never a hard requirement.
    bool qabsorb_bf16_batched_host(int layer_idx, const float* x_host, float* y_host,
                                    int H, int S, int nope, int kvl) {
        if (layer_idx < 0 || (size_t)layer_idx >= wuk_bf16_by_layer.size()) return false;
        void* d_wuk = wuk_bf16_by_layer[layer_idx];
        if (!d_wuk || H <= 0 || S <= 0 || nope <= 0 || kvl <= 0) return false;

        size_t n_x = (size_t)H * (size_t)S * (size_t)nope;
        size_t n_y = (size_t)H * (size_t)S * (size_t)kvl;
        size_t x_bf16_bytes = n_x * sizeof(uint16_t);
        size_t y_bytes = n_y * sizeof(float);

        cudaSetDevice(device_id);
        if (x_bf16_bytes > d_qabsorb_x_bf16_cap) {
            if (d_qabsorb_x_bf16) { cudaFree(d_qabsorb_x_bf16); d_qabsorb_x_bf16 = nullptr; }
            d_qabsorb_x_bf16_cap = 0;
            if (cudaMalloc(&d_qabsorb_x_bf16, x_bf16_bytes) != cudaSuccess) {
                LOG_ERROR("[CUDA] qabsorb_bf16_batched_host: d_qabsorb_x_bf16 grow to %.2f MB failed\n",
                          x_bf16_bytes / 1e6);
                return false;
            }
            d_qabsorb_x_bf16_cap = x_bf16_bytes;
        }
        if (y_bytes > d_qabsorb_y_cap) {
            if (d_qabsorb_y) { cudaFree(d_qabsorb_y); d_qabsorb_y = nullptr; }
            d_qabsorb_y_cap = 0;
            if (cudaMalloc((void**)&d_qabsorb_y, y_bytes) != cudaSuccess) {
                LOG_ERROR("[CUDA] qabsorb_bf16_batched_host: d_qabsorb_y grow to %.2f MB failed\n",
                          y_bytes / 1e6);
                return false;
            }
            d_qabsorb_y_cap = y_bytes;
        }
        if (n_x > h_qabsorb_x_bf16_scratch_cap) {
            if (h_qabsorb_x_bf16_scratch) delete[] h_qabsorb_x_bf16_scratch;
            h_qabsorb_x_bf16_scratch = new uint16_t[n_x];
            h_qabsorb_x_bf16_scratch_cap = n_x;
        }
        for (size_t i = 0; i < n_x; ++i) h_qabsorb_x_bf16_scratch[i] = fp32_to_bf16(x_host[i]);

        PERF_START(qabsorb_bf16_up);
        cudaMemcpy(d_qabsorb_x_bf16, h_qabsorb_x_bf16_scratch, x_bf16_bytes, cudaMemcpyHostToDevice);
        PERF_END(qabsorb_bf16_up, t_qabsorb_gpu_up);

        PERF_START(qabsorb_bf16_cmp);
        cublasStatus_t status = cublas_dense_matmul_bf16_matched_batched(
            cublas_handle,
            d_wuk, (long long)nope * kvl,
            d_qabsorb_x_bf16, (long long)S * nope,
            d_qabsorb_y, (long long)S * kvl,
            S, nope, kvl, H);
        PERF_END(qabsorb_bf16_cmp, t_qabsorb_gpu_compute);
        if (status != CUBLAS_STATUS_SUCCESS) return false;

        PERF_START(qabsorb_bf16_dn);
        cudaMemcpy(y_host, d_qabsorb_y, y_bytes, cudaMemcpyDeviceToHost);
        PERF_END(qabsorb_bf16_dn, t_qabsorb_gpu_down);
        perf().n_qabsorb_gpu_calls.fetch_add(1, std::memory_order_relaxed);
        return true;
    }

    // WUV BF16 fast-path, part 1: one-time per-layer upload -- byte-for-
    // byte the same shape/logic as upload_wuk_bf16 above, just a different
    // static weight (WUV/attn_v_b instead of WUK/attn_k_b). Per-head WUV is
    // [vh,kvl] row-major (matches the CPU matmul() call it replaces:
    // matmul(..., WUV.data()+h*kvl*vh, 1, kvl, vh) => per-head W is
    // [O=vh,I=kvl]). Best-effort: false leaves wuv_bf16_by_layer[layer_idx]
    // null, caller falls back to the CPU per-head loop for that layer.
    bool upload_wuv_bf16(int layer_idx, const std::vector<float>& WUV_host,
                          int H, int kvl, int vh, size_t max_bytes) {
        if (layer_idx < 0) return false;
        size_t n = (size_t)H * (size_t)kvl * (size_t)vh;
        if (n == 0 || WUV_host.size() != n) return false;
        size_t bf16_bytes = n * sizeof(uint16_t);
        if (bf16_bytes > max_bytes) return false;

        if ((int)wuv_bf16_by_layer.size() <= layer_idx) {
            wuv_bf16_by_layer.resize(layer_idx + 1, nullptr);
        }
        if (wuv_bf16_by_layer[layer_idx]) return true;  // already uploaded

        std::vector<uint16_t> host_bf16(n);
        for (size_t i = 0; i < n; ++i) host_bf16[i] = fp32_to_bf16(WUV_host[i]);

        cudaSetDevice(device_id);
        void* d_wuv = nullptr;
        cudaError_t err = cudaMalloc(&d_wuv, bf16_bytes);
        if (err != cudaSuccess) {
            LOG_WARN("[CUDA] WUV BF16 upload failed layer=%d (%.2f MB): %s -- "
                     "layer stays on the CPU WUV path\n",
                     layer_idx, bf16_bytes / 1e6, cudaGetErrorString(err));
            return false;
        }
        cudaMemcpy(d_wuv, host_bf16.data(), bf16_bytes, cudaMemcpyHostToDevice);
        wuv_bf16_by_layer[layer_idx] = d_wuv;
        perf().bytes_gpu_uploaded.fetch_add(bf16_bytes, std::memory_order_relaxed);
        return true;
    }

    // WUV BF16 fast-path, part 2: per-call batched dispatch -- byte-for-byte
    // the same shape as qabsorb_bf16_batched_host above (same strided-
    // batched cuBLAS call, same grow-only scratch pattern), just renamed
    // roles: x_host=[H,kvl] is score's own context output (the score GPU
    // path's `context_host`, S folded in since S==1 always here), y_host=
    // [H,vh] is head_contexts. Replaces attention_layer.cpp's CPU per-head
    // OMP-parallel matmul() loop, which real production data showed costing
    // 0.916ms/call -- OMP team-spawn overhead exceeding an ~8.4M-FLOP
    // per-call workload, the same pattern qabsorb's own gather fix already
    // diagnosed and fixed at a different call site. Best-effort: false
    // (this layer's WUV was never uploaded, or a CUDA/cublas call failed)
    // falls back to the existing CPU loop, unchanged.
    bool wuv_bf16_batched_host(int layer_idx, const float* x_host, float* y_host,
                                int H, int kvl, int vh) {
        if (layer_idx < 0 || (size_t)layer_idx >= wuv_bf16_by_layer.size()) return false;
        void* d_wuv = wuv_bf16_by_layer[layer_idx];
        if (!d_wuv || H <= 0 || kvl <= 0 || vh <= 0) return false;

        size_t n_x = (size_t)H * (size_t)kvl;
        size_t n_y = (size_t)H * (size_t)vh;
        size_t x_bf16_bytes = n_x * sizeof(uint16_t);
        size_t y_bytes = n_y * sizeof(float);

        cudaSetDevice(device_id);
        if (x_bf16_bytes > d_wuv_ctx_bf16_cap) {
            if (d_wuv_ctx_bf16) { cudaFree(d_wuv_ctx_bf16); d_wuv_ctx_bf16 = nullptr; }
            d_wuv_ctx_bf16_cap = 0;
            if (cudaMalloc(&d_wuv_ctx_bf16, x_bf16_bytes) != cudaSuccess) {
                LOG_ERROR("[CUDA] wuv_bf16_batched_host: d_wuv_ctx_bf16 grow to %.2f MB failed\n",
                          x_bf16_bytes / 1e6);
                return false;
            }
            d_wuv_ctx_bf16_cap = x_bf16_bytes;
        }
        if (y_bytes > d_wuv_out_cap) {
            if (d_wuv_out) { cudaFree(d_wuv_out); d_wuv_out = nullptr; }
            d_wuv_out_cap = 0;
            if (cudaMalloc((void**)&d_wuv_out, y_bytes) != cudaSuccess) {
                LOG_ERROR("[CUDA] wuv_bf16_batched_host: d_wuv_out grow to %.2f MB failed\n",
                          y_bytes / 1e6);
                return false;
            }
            d_wuv_out_cap = y_bytes;
        }
        if (n_x > h_wuv_ctx_bf16_scratch_cap) {
            if (h_wuv_ctx_bf16_scratch) delete[] h_wuv_ctx_bf16_scratch;
            h_wuv_ctx_bf16_scratch = new uint16_t[n_x];
            h_wuv_ctx_bf16_scratch_cap = n_x;
        }
        for (size_t i = 0; i < n_x; ++i) h_wuv_ctx_bf16_scratch[i] = fp32_to_bf16(x_host[i]);

        PERF_START(wuv_bf16_up);
        cudaMemcpy(d_wuv_ctx_bf16, h_wuv_ctx_bf16_scratch, x_bf16_bytes, cudaMemcpyHostToDevice);
        PERF_END(wuv_bf16_up, t_wuv_gpu_up);

        PERF_START(wuv_bf16_cmp);
        cublasStatus_t status = cublas_dense_matmul_bf16_matched_batched(
            cublas_handle,
            d_wuv, (long long)kvl * vh,
            d_wuv_ctx_bf16, (long long)kvl,
            d_wuv_out, (long long)vh,
            1, kvl, vh, H);
        PERF_END(wuv_bf16_cmp, t_wuv_gpu_compute);
        if (status != CUBLAS_STATUS_SUCCESS) return false;

        PERF_START(wuv_bf16_dn);
        cudaMemcpy(y_host, d_wuv_out, y_bytes, cudaMemcpyDeviceToHost);
        PERF_END(wuv_bf16_dn, t_wuv_gpu_down);
        perf().n_wuv_gpu_calls.fetch_add(1, std::memory_order_relaxed);
        return true;
    }

    // MLA score GPU fast-path, part 1: allocate this layer's persistent KV
    // latent cache mirror ([max_t,kvl] + [max_t,rope], F32), sized ONCE to
    // engine.max_t so it never needs a realloc mid-generation (a realloc
    // would force re-uploading the whole cache from scratch, defeating the
    // point of an incrementally-grown mirror -- see upload_lc_rc_rows).
    // Budget-checked against max_bytes the same way upload_wuk_bf16 is,
    // never assumes headroom. Best-effort: returns false (leaving this
    // layer's entries null) on any failure -- caller (warm_mla_absorb_cache)
    // logs and continues; the real attention call falls back to the CPU
    // score path for that layer, same discipline as every other best-effort
    // GPU fast-path in this file. Idempotent -- a second call for an
    // already-allocated layer is a cheap no-op returning true.
    bool ensure_lc_rc_gpu(int layer_idx, int max_t, int kvl, int rope, size_t max_bytes) {
        if (layer_idx < 0 || max_t <= 0 || kvl <= 0 || rope <= 0) return false;
        if ((int)d_lc_by_layer.size() <= layer_idx) {
            d_lc_by_layer.resize(layer_idx + 1, nullptr);
            d_rc_by_layer.resize(layer_idx + 1, nullptr);
            lc_rc_max_t_by_layer.resize(layer_idx + 1, 0);
        }
        if (d_lc_by_layer[layer_idx] && d_rc_by_layer[layer_idx]) return true;  // already allocated

        size_t lc_bytes = (size_t)max_t * (size_t)kvl * sizeof(float);
        size_t rc_bytes = (size_t)max_t * (size_t)rope * sizeof(float);
        if (lc_bytes + rc_bytes > max_bytes) return false;

        cudaSetDevice(device_id);
        float* d_lc = nullptr;
        float* d_rc = nullptr;
        cudaError_t err1 = cudaMalloc(&d_lc, lc_bytes);
        cudaError_t err2 = (err1 == cudaSuccess) ? cudaMalloc(&d_rc, rc_bytes) : err1;
        if (err1 != cudaSuccess || err2 != cudaSuccess) {
            LOG_WARN("[CUDA] score Lc/Rc GPU mirror alloc failed layer=%d (%.2f MB): %s -- "
                     "layer stays on the CPU score path\n",
                     layer_idx, (lc_bytes + rc_bytes) / 1e6,
                     cudaGetErrorString(err1 != cudaSuccess ? err1 : err2));
            if (d_lc) cudaFree(d_lc);
            if (d_rc) cudaFree(d_rc);
            return false;
        }
        // Zero-init: positions >= the real (still-growing) context length are
        // never read by score_gpu_batched_host's T bound, but zeroing avoids
        // ever feeding uninitialized VRAM bytes into a GEMM if a future
        // caller mistake ever passes too-large a T.
        cudaMemset(d_lc, 0, lc_bytes);
        cudaMemset(d_rc, 0, rc_bytes);
        d_lc_by_layer[layer_idx] = d_lc;
        d_rc_by_layer[layer_idx] = d_rc;
        lc_rc_max_t_by_layer[layer_idx] = max_t;
        perf().bytes_gpu_uploaded.fetch_add(lc_bytes + rc_bytes, std::memory_order_relaxed);
        return true;
    }

    // MLA score GPU fast-path, part 2: push newly-written KV latent cache
    // rows to this layer's GPU mirror. Called right after
    // attention_layer.cpp writes the same rows into engine.Lc[layer_idx]/
    // Rc[layer_idx] (mla_attention_absorbed_gguf's own KV-write step) --
    // host and device buffers share the IDENTICAL [max_t,*] row-major
    // layout and stride, so this is a single contiguous memcpy per array,
    // no reshuffling. n_rows is normally 1 (decode) or the prefill batch
    // size S; pos_base+n_rows must not exceed this layer's allocated max_t
    // (caller's own existing `if (pos >= engine.max_t) continue` bound
    // already guarantees this in practice).
    // Best-effort: false means this layer's mirror was never allocated
    // (ensure_lc_rc_gpu never called or failed) or the row range doesn't
    // fit -- caller should treat this the same as any other GPU fast-path
    // miss, i.e. score_gpu_batched_host will also return false for this
    // layer since its cache is stale/incomplete.
    bool upload_lc_rc_rows(int layer_idx, const float* Lc_host, const float* Rc_host,
                            int pos_base, int n_rows, int kvl, int rope) {
        if (layer_idx < 0 || (size_t)layer_idx >= d_lc_by_layer.size()) return false;
        float* d_lc = d_lc_by_layer[layer_idx];
        float* d_rc = d_rc_by_layer[layer_idx];
        if (!d_lc || !d_rc || n_rows <= 0 || pos_base < 0) return false;
        if (pos_base + n_rows > lc_rc_max_t_by_layer[layer_idx]) return false;

        cudaSetDevice(device_id);
        cudaMemcpy(d_lc + (size_t)pos_base * kvl, Lc_host + (size_t)pos_base * kvl,
                   (size_t)n_rows * kvl * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_rc + (size_t)pos_base * rope, Rc_host + (size_t)pos_base * rope,
                   (size_t)n_rows * rope * sizeof(float), cudaMemcpyHostToDevice);
        return true;
    }

    // MLA score GPU fast-path, part 3: per-token dispatch. Computes
    // context[H,kvl] = softmax((Q_absorbed[H,kvl] @ Lc[T,kvl]^T +
    //                            Q_rope[H,rope] @ Rc[T,rope]^T) * scale)
    //                   @ Lc[T,kvl]
    // via 3 plain (non-batched) GEMMs + one combine/softmax kernel --
    // unlike qabsorb, MLA's latent K/V is SHARED across every head, so this
    // is NOT H independent per-head ops and needs no strided-batched GEMM;
    // every head just multiplies its own query against the one shared
    // cache. See tests/bench_score_cpu_vs_gpu.cu for the real-hardware
    // feasibility numbers this design is based on (5x-51x speedup,
    // verified correctness) -- this is the same 3-GEMM+softmax shape,
    // wired to the persistent per-layer Lc/Rc mirror instead of a
    // benchmark's one-shot upload.
    //
    // context_host: [H,kvl] row-major F32, out -- this is the SAME
    //   quantity the CPU score loop calls `latent_accumulation` per head;
    //   caller still applies the per-head WUV down-projection afterward
    //   (unchanged CPU code, out of scope here -- see this function's own
    //   file-header comment for why: WUV is H independent per-head GEMMs,
    //   structurally identical to qabsorb's already-proven batched
    //   approach, not new work needing separate validation).
    // Returns false (caller falls back to the CPU per-head loop) if this
    // layer's Lc/Rc mirror isn't ready or doesn't cover [0,T), or any CUDA/
    // cublas call fails -- never a hard requirement.
    bool score_gpu_batched_host(int layer_idx, const float* Q_absorbed_host, const float* Q_rope_host,
                                 float* context_host, int H, int T, int kvl, int rope, float scale) {
        if (layer_idx < 0 || (size_t)layer_idx >= d_lc_by_layer.size()) return false;
        float* d_lc = d_lc_by_layer[layer_idx];
        float* d_rc = d_rc_by_layer[layer_idx];
        if (!d_lc || !d_rc || H <= 0 || T <= 0 || kvl <= 0 || rope <= 0) return false;
        if (T > lc_rc_max_t_by_layer[layer_idx]) return false;

        cudaSetDevice(device_id);
        auto grow = [&](float** buf, size_t* cap, size_t need_bytes, const char* label) -> bool {
            if (need_bytes <= *cap) return true;
            if (*buf) { cudaFree(*buf); *buf = nullptr; }
            *cap = 0;
            if (cudaMalloc((void**)buf, need_bytes) != cudaSuccess) {
                LOG_ERROR("[CUDA] score_gpu_batched_host: %s grow to %.2f MB failed\n", label, need_bytes / 1e6);
                return false;
            }
            *cap = need_bytes;
            return true;
        };
        if (!grow(&d_score_q_absorbed, &d_score_q_absorbed_cap, (size_t)H * kvl * sizeof(float), "d_score_q_absorbed")) return false;
        if (!grow(&d_score_q_rope, &d_score_q_rope_cap, (size_t)H * rope * sizeof(float), "d_score_q_rope")) return false;
        if (!grow(&d_score_nope_buf, &d_score_nope_buf_cap, (size_t)H * T * sizeof(float), "d_score_nope_buf")) return false;
        if (!grow(&d_score_rope_buf, &d_score_rope_buf_cap, (size_t)H * T * sizeof(float), "d_score_rope_buf")) return false;
        if (!grow(&d_score_combined, &d_score_combined_cap, (size_t)H * T * sizeof(float), "d_score_combined")) return false;
        if (!grow(&d_score_context, &d_score_context_cap, (size_t)H * kvl * sizeof(float), "d_score_context")) return false;

        const float alpha = 1.0f, beta = 0.0f;
        PERF_START(score_gpu_up);
        cudaMemcpy(d_score_q_absorbed, Q_absorbed_host, (size_t)H * kvl * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_score_q_rope, Q_rope_host, (size_t)H * rope * sizeof(float), cudaMemcpyHostToDevice);
        PERF_END(score_gpu_up, t_score_gpu_up);

        PERF_START(score_gpu_cmp);
        // scores_nope[H,T] = Q_absorbed[H,kvl] @ Lc[T,kvl]^T -- same "Y=X@W^T"
        // pattern as cublas_dense_matmul (Lc plays the role of a [T,kvl] weight).
        cublasStatus_t st1 = cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, T, H, kvl, &alpha,
                                          d_lc, kvl, d_score_q_absorbed, kvl, &beta, d_score_nope_buf, T);
        cublasStatus_t st2 = cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, T, H, rope, &alpha,
                                          d_rc, rope, d_score_q_rope, rope, &beta, d_score_rope_buf, T);
        if (st1 != CUBLAS_STATUS_SUCCESS || st2 != CUBLAS_STATUS_SUCCESS) { PERF_END(score_gpu_cmp, t_score_gpu_compute); return false; }

        launch_combine_scale_softmax(d_score_nope_buf, d_score_rope_buf, d_score_combined, H, T, scale, 0);

        // context[H,kvl] = attn_weights[H,T] @ Lc[T,kvl] -- plain row-major
        // A@B (Lc NOT transposed here, unlike the two GEMMs above), standard
        // row-major-via-column-major trick: swap operand order/dims.
        cublasStatus_t st3 = cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, kvl, H, T, &alpha,
                                          d_lc, kvl, d_score_combined, T, &beta, d_score_context, kvl);
        PERF_END(score_gpu_cmp, t_score_gpu_compute);
        if (st3 != CUBLAS_STATUS_SUCCESS) return false;

        PERF_START(score_gpu_dn);
        cudaMemcpy(context_host, d_score_context, (size_t)H * kvl * sizeof(float), cudaMemcpyDeviceToHost);
        cudaStreamSynchronize(0);
        PERF_END(score_gpu_dn, t_score_gpu_down);
        perf().n_score_gpu_calls.fetch_add(1, std::memory_order_relaxed);
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

    // Same grow-only pattern as ensure_buffers above, but for the DEDICATED
    // outproj staging buffers (d_outproj_x/d_outproj_y) -- kept fully
    // separate from d_x/d_y so grouped_output_proj_host never shares a
    // buffer with anything dispatched on the default stream, matching
    // outproj_stream/outproj_cublas_handle's own isolation.
    bool ensure_outproj_buffers(size_t x_bytes, size_t y_bytes) {
        cudaSetDevice(device_id);
        if (x_bytes > d_outproj_x_cap) {
            if (d_outproj_x) { cudaFree(d_outproj_x); d_outproj_x = nullptr; }
            d_outproj_x_cap = 0;
            cudaError_t err = cudaMalloc((void**)&d_outproj_x, x_bytes);
            if (err != cudaSuccess) {
                LOG_ERROR("[CUDA] ensure_outproj_buffers: d_outproj_x grow to %.2f MB failed: %s\n",
                          x_bytes / 1e6, cudaGetErrorString(err));
                return false;
            }
            d_outproj_x_cap = x_bytes;
        }
        if (y_bytes > d_outproj_y_cap) {
            if (d_outproj_y) { cudaFree(d_outproj_y); d_outproj_y = nullptr; }
            d_outproj_y_cap = 0;
            cudaError_t err = cudaMalloc((void**)&d_outproj_y, y_bytes);
            if (err != cudaSuccess) {
                LOG_ERROR("[CUDA] ensure_outproj_buffers: d_outproj_y grow to %.2f MB failed: %s\n",
                          y_bytes / 1e6, cudaGetErrorString(err));
                return false;
            }
            d_outproj_y_cap = y_bytes;
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

    // One-time upload of a static (never-changes-across-tokens) F32 buffer
    // to a persistent VRAM allocation -- caller owns the returned pointer
    // for the lifetime of the process (mirrors this project's existing
    // "cache dequantized static weights, never free" convention, e.g.
    // attention_layer.cpp's woa_cache/wuk_cache/wuv_cache, just VRAM- rather
    // than host-resident). Returns nullptr on any allocation/copy failure
    // -- caller falls back to the CPU path, never a hard failure.
    float* upload_persistent_f32(const float* host_data, size_t n_floats) {
        cudaSetDevice(device_id);
        float* d_ptr = nullptr;
        size_t bytes = n_floats * sizeof(float);
        cudaError_t err = cudaMalloc((void**)&d_ptr, bytes);
        if (err != cudaSuccess) {
            LOG_STREAM_ERROR() << "[GPU] upload_persistent_f32: cudaMalloc failed for "
                               << bytes / 1e6 << " MB: " << cudaGetErrorString(err) << "\n";
            return nullptr;
        }
        err = cudaMemcpy(d_ptr, host_data, bytes, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            LOG_STREAM_ERROR() << "[GPU] upload_persistent_f32: cudaMemcpy H2D failed for "
                               << bytes / 1e6 << " MB: " << cudaGetErrorString(err) << "\n";
            cudaFree(d_ptr);
            return nullptr;
        }
        return d_ptr;
    }

    // Grouped (block-diagonal) output projection: o_groups independent
    // [1, group_in] x [o_rank, group_in]^T matmuls, y[g] = d_WOA[g] @ x[g].
    // Added specifically for dsv4_attention_gguf's S==1 (decode) outproj
    // step (attention_layer.cpp) -- real-hardware profiling found this the
    // single largest remaining cost in attention (~6.8s of a real 40-token
    // run), root-caused to memory bandwidth, not compute: the weight matrix
    // (o_groups*o_rank*group_in floats, ~134MB for this model) has to be
    // re-read from host RAM on EVERY decode token since no other layer's
    // work fits in cache between two calls to the same layer, and this
    // CPU's RAM bandwidth is the bottleneck (~0.5 FLOP/byte arithmetic
    // intensity -- a textbook memory-bound matrix-vector multiply, not a
    // parallelism problem; confirmed by a standalone CPU-vs-GPU
    // microbenchmark using this exact shape and the real weight-cycling
    // access pattern, tests/bench_outproj_cpu_vs_gpu.cu, before this was
    // wired into the live path -- GPU VRAM bandwidth (~900+ GB/s on
    // consumer cards vs. ~30-50 GB/s CPU DDR) gave a measured 12x+ speedup
    // once the weight is already VRAM-resident, verified numerically
    // matching (max rel diff ~2e-3, normal float32 accumulation-order
    // difference between AVX-512 and cuBLAS, not a bug). d_WOA MUST already
    // be VRAM-resident (via upload_persistent_f32 above, called once per
    // layer on first use) -- this function only runs the GEMV, it never
    // uploads the weight itself.
    bool grouped_output_proj_host(const float* d_WOA, int o_groups, int o_rank, int group_in,
                                  const float* x_host, float* y_host) {
        // outproj_cublas_handle/outproj_stream, NOT cublas_handle/stream 0
        // -- see outproj_stream's own declaration comment for the real-
        // hardware regression (net-wash on total throughput) this fixes.
        if (!d_WOA || !outproj_cublas_handle || !outproj_stream) return false;
        size_t x_bytes = (size_t)o_groups * group_in * sizeof(float);
        size_t y_bytes = (size_t)o_groups * o_rank * sizeof(float);
        if (!ensure_outproj_buffers(x_bytes, y_bytes)) return false;  // caller falls back to CPU
        cudaSetDevice(device_id);

        PERF_START(dense_up);
        cudaMemcpyAsync(d_outproj_x, x_host, x_bytes, cudaMemcpyHostToDevice, outproj_stream);
        PERF_END(dense_up, t_dense_gpu_up);

        PERF_START(dense_cmp);
        const float alpha = 1.0f, beta = 0.0f;
        bool ok = true;
        for (int g = 0; g < o_groups; ++g) {
            // Row-major W[o_rank, group_in] reinterpreted as column-major
            // A'(m=group_in, n=o_rank); CUBLAS_OP_T gives y = W @ x exactly,
            // with no data reshuffling -- same derivation validated in the
            // standalone benchmark above.
            cublasStatus_t st = cublasSgemv(outproj_cublas_handle, CUBLAS_OP_T,
                group_in, o_rank, &alpha,
                d_WOA + (size_t)g * o_rank * group_in, group_in,
                d_outproj_x + (size_t)g * group_in, 1,
                &beta, d_outproj_y + (size_t)g * o_rank, 1);
            if (st != CUBLAS_STATUS_SUCCESS) { ok = false; break; }
        }
        // Sync ONLY outproj_stream -- never stream 0. This is the entire
        // point of the dedicated stream: unlike cudaStreamSynchronize(0),
        // this cannot force any other stream (MoE's compute_stream/
        // copy_stream, or dense_proj's own stream-0 work) to wait or be
        // waited on. Real per-call work is tiny (a few microseconds of
        // GEMV), so synchronous-from-the-caller's-perspective is still
        // fine here -- decode needs the result immediately either way.
        cudaStreamSynchronize(outproj_stream);
        PERF_END(dense_cmp, t_dense_gpu_compute);
        if (!ok) return false;

        PERF_START(dense_dn);
        cudaMemcpyAsync(y_host, d_outproj_y, y_bytes, cudaMemcpyDeviceToHost, outproj_stream);
        cudaStreamSynchronize(outproj_stream);
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
