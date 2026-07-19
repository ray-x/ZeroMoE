// cuda_manager.hpp — Multi-GPU pool manager with topology-aware execution
// Implements the asymmetric dual-GPU pipeline from optimize.md:
//   GPU 0 (CoreEngine): Dense layers + KV cache + activations
//   GPU 1+ (ExpertVault): MoE expert LRU cache in VRAM
#pragma once
#include "engine.hpp"
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

    // Dequantize a static per-layer dense weight to F32 ONCE (host-side, via
    // the same ggml reference dequant_row used by the CPU fallback) and
    // upload the result to its OWN standalone cudaMalloc buffer (deliberately
    // NOT a dense_pool suballocation, so this can be called independent of
    // pool sizing/lifetime) — qt.gpu_fast_ptr. dense_proj then dispatches
    // through cuBLAS (fast SGEMV) instead of qt's original-quant-type fused
    // kernel (Q2_K..Q6_K/IQ2*/IQ3_XXS), which has no GEMV specialization and
    // is disproportionately slow at S=1 decode — see attn_output's ~623ms/
    // token cost in CLAUDE.md, the motivating case. host_ptr/device_ptr/
    // meta.type are untouched, so the CPU fallback stays correct regardless.
    //
    // Best-effort: returns false (leaving gpu_fast_ptr null, caller keeps
    // using the existing slower path) if the F32 expansion doesn't fit in
    // max_bytes, the shape is malformed, or the quant type has no CPU
    // reference dequant. F32 is ~5-6x the size of a K-quant/IQ original (e.g.
    // Q5_K), so this is capacity-checked per call rather than assumed to fit
    // — callers should pass real remaining VRAM headroom, not the tensor's
    // own original size.
    bool upload_dequantized_f32(QuantizedTensor& qt, size_t max_bytes) {
        if (qt.gpu_fast_ptr) return true;  // already converted
        if (!qt.host_ptr || qt.meta.shape.size() < 2) return false;
        int O = (int)qt.meta.shape[0];
        int I = (int)qt.meta.shape[1];
        if (O <= 0 || I <= 0) return false;
        size_t n = (size_t)O * (size_t)I;
        size_t f32_bytes = n * sizeof(float);
        if (f32_bytes > max_bytes) return false;

        std::vector<float> host_f32(n);
        if (!dequant_tensor_f32(qt.meta.type, qt.host_ptr, O, I, host_f32.data())) {
            LOG_WARN("[CUDA] Fast-path F32 dequant unsupported for type=%u — "
                     "keeping original quantized GPU path\n", (unsigned)qt.meta.type);
            return false;
        }

        cudaSetDevice(device_id);
        cudaError_t err = cudaMalloc((void**)&qt.gpu_fast_ptr, f32_bytes);
        if (err != cudaSuccess) {
            LOG_ERROR("[CUDA] Fast-path F32 upload failed (%.2f MB): %s — "
                      "keeping original quantized GPU path\n",
                      f32_bytes / 1e6, cudaGetErrorString(err));
            qt.gpu_fast_ptr = nullptr;
            return false;
        }
        cudaMemcpy(qt.gpu_fast_ptr, host_f32.data(), f32_bytes, cudaMemcpyHostToDevice);
        perf().bytes_gpu_uploaded.fetch_add(f32_bytes, std::memory_order_relaxed);
        return true;
    }

    // Allocate expert VRAM cache on vault GPU(s) based on model geometry
    void setup_expert_cache(size_t expert_slab_bytes, int n_experts) {
        topology.allocate_expert_cache(expert_slab_bytes, n_experts);
    }

    // Grow-only persistent device buffers for matmul I/O
    void ensure_buffers(size_t x_bytes, size_t y_bytes) {
        cudaSetDevice(device_id);
        if (x_bytes > d_x_cap) {
            if (d_x) cudaFree(d_x);
            cudaMalloc((void**)&d_x, x_bytes);
            d_x_cap = x_bytes;
        }
        if (y_bytes > d_y_cap) {
            if (d_y) cudaFree(d_y);
            cudaMalloc((void**)&d_y, y_bytes);
            d_y_cap = y_bytes;
        }
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

        ensure_buffers(x_bytes, y_bytes);
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
    bool dense_matmul_host(GGUFType wtype, const void* d_W,
                           const float* x_host, float* y_host,
                           int S, int I, int O) {
        if (!d_W) return false;
        size_t x_bytes = (size_t)S * I * sizeof(float);
        size_t y_bytes = (size_t)S * O * sizeof(float);
        ensure_buffers(x_bytes, y_bytes);
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
            case GGUFType::BF16:
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
        ensure_buffers(bytes, bytes);
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
