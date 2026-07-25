#pragma once
#include "gguf_loader.hpp"
#include "direct_io.hpp"
#include "io_uring_backend.hpp"
#include "perf_timer.hpp"
#include "logger.hpp"
#include <cuda_runtime.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <thread>
#include <future>
#include <vector>
#include <memory>
#include <unordered_map>

#ifdef __APPLE__
// macOS doesn't have posix_fadvise; provide no-op stubs. The function
// definition must live INSIDE the same #ifndef as the constants (not after
// it, unconditionally) -- gguf_loader.hpp (included above, transitively via
// this file being included first almost everywhere) carries an identical
// stub under the same POSIX_FADV_WILLNEED guard so the two can coexist in
// one translation unit; leaving this one unconditional caused a duplicate-
// definition build error the moment both headers landed together.
#ifndef POSIX_FADV_WILLNEED
#define POSIX_FADV_WILLNEED 3
#define POSIX_FADV_DONTNEED 4
static inline int posix_fadvise(int, off_t, off_t, int) { return 0; }
#endif
#endif

class CudaManager;  // forward decl — defined in cuda_manager.hpp; engine holds a pointer

// Any single expert-slab pread(), or a single posix_memalign call in
// ensure_slab_capacity, exceeding this is logged individually (tagged
// [SLOW-PREAD] / [SLOW-HOST-ALLOC]). ~15ms is a normal full expert-slab
// read at typical measured throughput, so 200ms is already >10x normal --
// unambiguously anomalous, not routine variance. Added alongside
// gpu_topology.hpp's SLOW_CACHE_MUTEX_WAIT_MS and async_prefetcher.hpp's
// SLOW_MUTEX_WAIT_MS to root-cause a real multi-second single-cache-miss
// stall (see RAMSafetyManager::get_expert's loaded_from_disk= log) whose
// per-miss latency histogram had normal p50/p90 but extreme outliers --
// ruling out general disk throughput and pointing at something blocking
// intermittently. These three instrumentation points cover every place
// that stall's ~40s could plausibly be hiding.
constexpr double SLOW_PREAD_MS = 200.0;

struct QuantizedTensor {
    GGUFTensor meta;
    uint8_t* device_ptr = nullptr;
    uint8_t* host_ptr = nullptr;
    // Optional F32-dequantized VRAM copy of this tensor, own cudaMalloc (NOT
    // a dense_pool suballocation). Set by CudaManager::upload_dequantized_f32
    // for static per-layer weights whose fused GPU kernel (Q5_K etc.) has no
    // GEMV specialization and is disproportionately slow at S=1 decode (see
    // attn_output in CLAUDE.md). When set, dense_proj dispatches through this
    // via cuBLAS instead of device_ptr's original-quant-type fused kernel.
    // meta.type/host_ptr/device_ptr are untouched so the CPU fallback path
    // and existing pool bookkeeping stay correct either way.
    uint8_t* gpu_fast_ptr = nullptr;
};

// Q4_0 KV cache block: 32 values in 18 bytes (4.5 bits/value effective)
// Used when kv_quantized=true to compress KV cache by 8x vs FP32
struct BlockQ4_KV {
    uint16_t d;       // FP16 scale (2 bytes)
    uint8_t  qs[16];  // 32 x 4-bit values packed (16 bytes)
};

struct MoEExpertBlock {
    int expert_id;
    QuantizedTensor gate_proj;
    QuantizedTensor up_proj;
    QuantizedTensor down_proj;
    uint8_t* slab_buffer = nullptr;
    size_t slab_capacity = 0;
    uint64_t last_used_clock = 0;
    uint32_t hit_count = 0;  // frequency signal for eviction scoring (power-law routing)

    // True for a lightweight "shadow" block populated by
    // StreamingEngine::populate_vram_shadow (execution_layer.cpp's
    // moe_forward_gguf) instead of a real disk-backed RAM-tier load --
    // gate_proj.meta/up_proj.meta/down_proj.meta are valid (type + the
    // packed slab's uniform per-layer relative offsets), but host_ptr is
    // deliberately null and slab_buffer/slab_capacity are unset, since
    // the fast path this exists for (an expert already confirmed VRAM-
    // resident) never reads them -- GPU dispatch reads weight bytes from
    // ExpertLocation::device_ptr, never from here. Callers MUST check
    // this flag before touching host_ptr; see moe_forward_gguf's Pass A
    // loop for the required emergency-reload fallback if GPU dispatch
    // unexpectedly fails for a shadowed expert.
    bool vram_shadow = false;

    // True once RAMSafetyManager::get_expert/get_experts_batch has
    // CONFIRMED (via GpuTopology::find_expert, which itself polls
    // upload_ready before returning non-null -- never set from the
    // moment an async promotion is merely ISSUED, which would race the
    // in-flight cudaMemcpyAsync still reading this slot's slab_buffer)
    // that this same (layer, expert_id) is ALSO resident and ready in the
    // VRAM tier. A slot with this set is genuinely redundant capacity --
    // the RAM eviction-victim scan (get_expert/get_experts_batch's Step 3)
    // treats it as the lowest-priority ("evict first") candidate
    // regardless of its own recency/hit_count, so double-stored capacity
    // gets reclaimed promptly instead of lingering until unrelated LRU
    // pressure happens to pick it (see CLAUDE.md's TIER-OVERLAP finding).
    // Reset to false whenever a slot is claimed for a different expert --
    // the new occupant isn't known to be VRAM-redundant. Deliberately
    // never proactively un-set if the VRAM copy is later evicted: worst
    // case a slightly-too-eagerly-marked slot causes one extra disk read
    // on its next miss, a performance cost only, never a correctness one.
    bool also_in_vram = false;

    // Non-owning reference to the destination VRAM slot's OWN upload_event
    // (GpuTopology::ExpertLocation::upload_event), set the moment
    // move_on_promotion fires an async H2D copy FROM this slot's
    // slab_buffer and immediately clears this slot's identity/capacity
    // for reuse -- the exact race also_in_vram's own comment above already
    // warns about ("never set from the moment an async promotion is
    // merely ISSUED, which would race the in-flight cudaMemcpyAsync still
    // reading this slot's slab_buffer"), just via a different field
    // (slab_capacity/expert_id instead of also_in_vram). Without this, a
    // LATER miss's eviction scan can pick this exact slot and overwrite
    // (or free+realloc, see ensure_slab_capacity) slab_buffer while the
    // GPU copy engine is still reading the OLD bytes from it -- a genuine
    // use-after-free/WAR hazard, not just a capacity-accounting one. Every
    // RAM-slot victim-selection site (get_expert Step 3, get_experts_batch
    // Phase A/B, demote_from_vram) must check copy_still_pending() before
    // treating this slot as reusable. GpuTopology owns the event's actual
    // lifetime (destroyed only at process teardown, long after RAM-tier
    // operations stop), so holding a bare non-owning pointer here is safe;
    // re-recording on the SAME destination slot for a later, unrelated
    // promotion is also safe to observe here (stream-ordered execution on
    // the same copy_stream guarantees our earlier copy already completed
    // before a later one can even be enqueued), just conservative -- it
    // can make this check report "still pending" slightly longer than
    // strictly necessary, never the reverse.
    cudaEvent_t pending_copy_event = nullptr;

    // Self-healing: true if a previously-issued async H2D copy FROM this
    // slot's slab_buffer may still be reading it. Same "poll before trust"
    // pattern find_expert/evict_lru already use for the VRAM side's
    // upload_ready -- clears the flag the first time it's confirmed done,
    // so this never re-queries a completed event twice.
    bool copy_still_pending() {
        if (!pending_copy_event) return false;
        if (cudaEventQuery(pending_copy_event) == cudaSuccess) {
            pending_copy_event = nullptr;
            return false;
        }
        return true;
    }
};

class StreamingEngine {
public:
    GGUFFile model_file;
    GGUFFile dense_model_file;      // separate dense GGUF (config.dense_model_path); optional
    bool has_dense_model = false;
    std::unordered_map<std::string, QuantizedTensor> resident_layers;
    std::vector<std::vector<MoEExpertBlock>> layers_experts;

    // Router-selection frequency histogram, [n_layers * n_experts], accumulated
    // across a run and dumped to the usage file at shutdown (Phase A 
    // the "learned hot set" signal that drives VRAM preloading on the
    // NEXT run). Atomic so the (future) batched/router-parallel path can bump
    // it without locking; today moe_forward_gguf bumps it single-threaded.
    std::vector<std::atomic<uint64_t>> route_freq;
    std::vector<float*> Lc;   // Host KV latent cache [n_layers+1][max_t * kv_lora]
    std::vector<float*> Rc;   // Host KV rope cache [n_layers+1][max_t * qk_rope]

    // Dense KV cache for non-MLA arches (Hunyuan/Llama): [n_layers][max_t * n_head_kv * head_dim]
    std::vector<float*> Kc;
    std::vector<float*> Vc;

    // GPU-resident KV cache (allocated when VRAM available)
    std::vector<float*> d_Lc;  // Device pointers for latent cache (FP32)
    std::vector<float*> d_Rc;  // Device pointers for rope cache (FP32)
    bool kv_on_device = false; // true if KV cache is in VRAM

    // Q4-compressed KV cache (8x savings over FP32, for large contexts)
    std::vector<BlockQ4_KV*> d_Lc_q4;  // Device Q4 latent cache [n_layers][max_t * kv_lora / 32 blocks]
    std::vector<BlockQ4_KV*> d_Rc_q4;  // Device Q4 rope cache
    bool kv_quantized = false;          // true when using Q4 KV cache

    // GPU-resident hidden state (persistent, never freed mid-session)
    float* d_hidden = nullptr;
    size_t d_hidden_cap = 0;

    int max_t = 0;

    // GPU dense-matmul dispatch handle (set by main.cpp after CudaManager
    // initialization). Forward-declared above (CudaManager itself, to avoid a
    // circular include with cuda_manager.hpp) — cuda_runtime.h is included
    // directly below so this header can free its own device allocations.
    CudaManager* gpu = nullptr;

    // Core GPU device id, captured once (by main.cpp, alongside `gpu` above)
    // rather than read back via gpu->get_core_device() at destruction time —
    // `gpu` may already be destroyed by then depending on local-variable
    // declaration order in run_engine(). -1 means "no GPU was ever attached"
    // (device buffers below were never allocated, destructor skips them).
    int gpu_device_id = -1;

    // Configuration: page cache policy for expert slabs after reading
    // false = leave pages in OS page cache 
    // true = POSIX_FADV_DONTNEED after read (reduces memory pressure under tight RAM)
    bool drop_expert_pages = false;

    // Frees every buffer this class allocated directly: host-side
    // posix_memalign'd tensors (resident dense weights, KV cache, expert
    // slabs) and GPU-side cudaMalloc'd buffers (offloaded dense tensors, KV
    // cache mirrors). Previously there was no destructor at all — every one
    // of these leaked until the OS/CUDA driver reclaimed the whole process
    // at exit, which "worked" for a single run but meant Ctrl-C (or any
    // path that doesn't kill the process) never actually released anything.
    ~StreamingEngine() {
        bool have_gpu = (gpu_device_id >= 0);
        if (have_gpu) cudaSetDevice(gpu_device_id);

        for (auto& [name, qt] : resident_layers) {
            if (qt.host_ptr) free(qt.host_ptr);
            if (have_gpu && qt.device_ptr) cudaFree(qt.device_ptr);
            // Unlike device_ptr (may be a dense_pool suballocation), gpu_fast_ptr
            // is always its own standalone cudaMalloc — always safe/correct to
            // cudaFree individually.
            if (have_gpu && qt.gpu_fast_ptr) cudaFree(qt.gpu_fast_ptr);
        }
        for (auto& layer_slots : layers_experts) {
            for (auto& slot : layer_slots) {
                if (slot.slab_buffer) free(slot.slab_buffer);
            }
        }
        for (float* p : Lc) if (p) free(p);
        for (float* p : Rc) if (p) free(p);
        for (float* p : Kc) if (p) free(p);
        for (float* p : Vc) if (p) free(p);

        if (have_gpu) {
            for (float* p : d_Lc) if (p) cudaFree(p);
            for (float* p : d_Rc) if (p) cudaFree(p);
            for (BlockQ4_KV* p : d_Lc_q4) if (p) cudaFree(p);
            for (BlockQ4_KV* p : d_Rc_q4) if (p) cudaFree(p);
            if (d_hidden) cudaFree(d_hidden);
        }
    }

    void initialize_resident_memory(const std::vector<std::string>& dense_tensor_names) {
        PERF_START(dense_load);
        size_t total_loaded = 0;
        // Dense tensors load from the separate dense GGUF when one is configured
        // (dense_model_path); routed experts still stream from model_file. Both
        // files are complete GGUFs of the same model at different precisions.
        GGUFFile& src = has_dense_model ? dense_model_file : model_file;
        for (const auto& name : dense_tensor_names) {
            if (src.tensors.find(name) == src.tensors.end()) continue;

            auto& t_meta = src.tensors[name];
            QuantizedTensor qt{t_meta, nullptr, nullptr};

            posix_memalign((void**)&qt.host_ptr, 4096, t_meta.size_bytes);
            int tensor_fd = src.fd_for_tensor(t_meta);
            ssize_t bytes_read = direct_io::pread_direct_aware(
                tensor_fd, src.direct_io_active_for(t_meta), src.direct_io_alignment,
                qt.host_ptr, t_meta.size_bytes, t_meta.offset);
            if (bytes_read <= 0 || (uint64_t)bytes_read != t_meta.size_bytes) {
                LOG_WARN("[WARN] Short read for %s (shard %d, offset %lu): got %zd / %lu\n",
                         name.c_str(), t_meta.shard_idx, (unsigned long)t_meta.offset,
                         bytes_read, (unsigned long)t_meta.size_bytes);
                // Mark as unusable — downstream code checks host_ptr != nullptr
                free(qt.host_ptr);
                qt.host_ptr = nullptr;
            } else {
                total_loaded += t_meta.size_bytes;
            }

            // Selective mlock: only pin small latency-sensitive tensors (norms, router)
            bool is_small_critical = (qt.host_ptr &&
                                      (name.find("norm") != std::string::npos ||
                                       name.find("gate_inp") != std::string::npos));
            if (is_small_critical) {
                mlock(qt.host_ptr, t_meta.size_bytes);
            }
            
            resident_layers[name] = qt;
        }
        PERF_END(dense_load, t_dense_load);
        perf().bytes_disk_read.fetch_add(total_loaded, std::memory_order_relaxed);
        LOG_INFO("[PERF] Dense resident load: %.3f s (%.2f GB, %.2f GB/s)\n",
                 perf().t_dense_load.load(),
                 total_loaded / 1e9,
                 perf().t_dense_load > 0 ? (total_loaded / 1e9) / perf().t_dense_load.load() : 0.0);
    }

    // Resolve the packed-slab geometry for a layer: per-projection byte sizes
    // (gb/ub/db) and the source tensor metas (which carry each projection's
    // own quant type + shape). Returns false if this layer has no packed
    // ffn_{gate,up,down}_exps.weight tensors (dense layer or legacy per-expert
    // layout). Shared by load_expert_slab's pread path and the consume_staged
    // (PIPE) path so the two can never disagree on slab layout.
    bool packed_slab_geometry(int layer_idx, size_t& gb, size_t& ub, size_t& db,
                              const GGUFTensor*& gM, const GGUFTensor*& uM, const GGUFTensor*& dM) const {
        std::string base = "blk." + std::to_string(layer_idx) + ".";
        int n_exp = model_file.architecture_config.n_experts;
        if (n_exp <= 0) return false;
        auto gE = model_file.tensors.find(base + "ffn_gate_exps.weight");
        auto uE = model_file.tensors.find(base + "ffn_up_exps.weight");
        auto dE = model_file.tensors.find(base + "ffn_down_exps.weight");
        if (gE == model_file.tensors.end() || uE == model_file.tensors.end() || dE == model_file.tensors.end())
            return false;
        gb = gE->second.size_bytes / n_exp;
        ub = uE->second.size_bytes / n_exp;
        db = dE->second.size_bytes / n_exp;
        gM = &gE->second; uM = &uE->second; dM = &dE->second;
        return true;
    }

    // Ensure target_slot.slab_buffer can hold `total` bytes (grow-only, like
    // load_expert_slab). Used by both the pread path and the consume_staged
    // path so the destination buffer is sized before the bytes arrive.
    void ensure_slab_capacity(MoEExpertBlock& slot, size_t total) {
        PERF_START(alloc);
        if (slot.slab_capacity < total) {
            auto t0 = std::chrono::high_resolution_clock::now();
            if (slot.slab_buffer) free(slot.slab_buffer);
            posix_memalign((void**)&slot.slab_buffer, 4096, total);
            double ms = std::chrono::duration<double, std::milli>(
                std::chrono::high_resolution_clock::now() - t0).count();
            if (ms > 50.0) {
                LOG_WARN("[SLOW-HOST-ALLOC] posix_memalign(%.2fMB) took %.2fms (old_capacity=%.2fMB)\n",
                         total / 1e6, ms, slot.slab_capacity / 1e6);
            }
            slot.slab_capacity = total;
        }
        PERF_END(alloc, t_host_alloc);
    }

    // Set up a MoEExpertBlock's three projection host_ptr/meta from an already-
    // populated packed slab_buffer (gate|up|down at offsets 0, gb, gb+ub).
    // Shared by the pread path and the consume_staged (PIPE) path -- the
    // prefetcher's active_prefetch produces a byte-identical packed layout, so
    // the same projection setup applies whether the bytes came from pread or
    // from the staging pool. Offsets are slab-relative so expert_ffn's
    // (up_meta.offset - gate_meta.offset) delta resolves correctly.
    void populate_packed_slab(MoEExpertBlock& slot, int expert_id,
                              size_t gb, size_t ub, size_t db,
                              const GGUFTensor& gM, const GGUFTensor& uM, const GGUFTensor& dM) {
        uint8_t* base = slot.slab_buffer;
        slot.gate_proj.host_ptr = base;
        slot.gate_proj.meta = gM;
        slot.gate_proj.meta.offset = 0;
        slot.gate_proj.meta.size_bytes = gb;
        slot.up_proj.host_ptr = base + gb;
        slot.up_proj.meta = uM;
        slot.up_proj.meta.offset = gb;
        slot.up_proj.meta.size_bytes = ub;
        slot.down_proj.host_ptr = base + gb + ub;
        slot.down_proj.meta = dM;
        slot.down_proj.meta.offset = gb + ub;
        slot.down_proj.meta.size_bytes = db;
        slot.expert_id = expert_id;
        slot.vram_shadow = false;
    }

    // Metadata-only counterpart to load_expert_slab/populate_packed_slab --
    // ZERO disk I/O, zero RAM-slot involvement. Fixes the real, measured
    // "[TIER-OVERLAP]" finding (see CLAUDE.md): moe_forward_gguf's per-
    // token dispatch used to call RAMSafetyManager::get_experts_batch
    // UNCONDITIONALLY for every routed expert, even ones already GPU-
    // resident, wasting a real disk read and a real RAM slot on data whose
    // BYTES the GPU-dispatch path never actually reads (it reads the
    // packed base tensor's TYPE, uniform across every expert in a layer,
    // and the weight bytes from ExpertLocation::device_ptr -- never from
    // a RAM copy). This fills in exactly that metadata (type + the
    // packed slab's uniform per-expert relative offsets: gate=0, up=gb,
    // down=gb+ub) from packed_slab_geometry, which is a pure tensor-table
    // lookup already parsed at GGUF load time -- same source
    // load_expert_slab itself uses before it ever touches disk. host_ptr
    // is deliberately left null; slot.vram_shadow=true is the required
    // signal callers must check before touching it (see MoEExpertBlock's
    // own comment). Returns false (slot untouched) if this layer has no
    // packed expert tensors at all (e.g. a non-MoE layer).
    bool populate_vram_shadow(MoEExpertBlock& slot, int layer_idx, int expert_id) const {
        size_t gb = 0, ub = 0, db = 0;
        const GGUFTensor *gM = nullptr, *uM = nullptr, *dM = nullptr;
        if (!packed_slab_geometry(layer_idx, gb, ub, db, gM, uM, dM)) return false;
        slot.gate_proj.host_ptr = nullptr;
        slot.gate_proj.meta = *gM;
        slot.gate_proj.meta.offset = 0;
        slot.gate_proj.meta.size_bytes = gb;
        slot.up_proj.host_ptr = nullptr;
        slot.up_proj.meta = *uM;
        slot.up_proj.meta.offset = gb;
        slot.up_proj.meta.size_bytes = ub;
        slot.down_proj.host_ptr = nullptr;
        slot.down_proj.meta = *dM;
        slot.down_proj.meta.offset = gb + ub;
        slot.down_proj.meta.size_bytes = db;
        slot.expert_id = expert_id;
        slot.vram_shadow = true;
        return true;
    }

    // Builds the (up to 3) io_uring read requests needed to load ONE
    // packed-format expert's slab -- but does NOT perform any reads
    // itself. Deliberately kept separate from load_expert_slab (which
    // remains completely unchanged as the proven, correctness-critical
    // fallback for this exact same read) rather than refactored to share
    // code: this lets a batch caller (RAMSafetyManager's Phase B) collect
    // MANY experts' requests into ONE big io_uring submission, reap them
    // all, THEN populate every slab only once every single read in the
    // whole batch has succeeded -- all-or-nothing, so a failure anywhere
    // in the batch can never leave one expert's slab half-populated while
    // its siblings look fine. Reuses the SAME packed_slab_geometry/
    // ensure_slab_capacity/populate_packed_slab helpers load_expert_slab
    // itself uses, so the two paths can never disagree about slab layout.
    //
    // Legacy per-expert format is deliberately NOT supported here (returns
    // false) -- it's already documented as the engine's cold/slow path
    // elsewhere (io_uring_backend.hpp, CLAUDE.md), not worth the added
    // complexity for a first cut; callers fall back to load_expert_slab
    // for it exactly as Phase B always has.
    //
    // Always issues 3 separate reads per expert (gate/up/down), even when
    // they're contiguous on disk -- unlike load_expert_slab's own single-
    // shard coalescing optimization. Simpler, and the coalescing win
    // matters less here: io_uring's real benefit (kernel-managed deep
    // queue depth across the WHOLE batch's reads at once) applies
    // regardless of whether any one expert's 3 reads happen to be
    // contiguous.
    bool build_packed_expert_read_requests(int layer_idx, int expert_id,
                                            MoEExpertBlock& target_slot,
                                            io_uring_backend::ReadRequest* out_reqs,
                                            int& out_n, size_t& out_total) {
        size_t gb, ub, db;
        const GGUFTensor *gM, *uM, *dM;
        if (!packed_slab_geometry(layer_idx, gb, ub, db, gM, uM, dM)) return false;
        out_total = gb + ub + db;
        ensure_slab_capacity(target_slot, out_total);

        uint8_t* dst_gate = target_slot.slab_buffer;
        uint8_t* dst_up   = target_slot.slab_buffer + gb;
        uint8_t* dst_down = target_slot.slab_buffer + gb + ub;
        uint64_t gate_off = gM->offset + (uint64_t)expert_id * gb;
        uint64_t up_off   = uM->offset + (uint64_t)expert_id * ub;
        uint64_t down_off = dM->offset + (uint64_t)expert_id * db;

        out_n = 0;
        out_reqs[out_n++] = {model_file.fd_for_tensor(*gM), model_file.direct_io_active_for(*gM),
                              model_file.direct_io_alignment, dst_gate, gb, gate_off};
        out_reqs[out_n++] = {model_file.fd_for_tensor(*uM), model_file.direct_io_active_for(*uM),
                              model_file.direct_io_alignment, dst_up, ub, up_off};
        out_reqs[out_n++] = {model_file.fd_for_tensor(*dM), model_file.direct_io_active_for(*dM),
                              model_file.direct_io_alignment, dst_down, db, down_off};
        return true;
    }

    void load_expert_slab(int layer_idx, int expert_id, MoEExpertBlock& target_slot) {
        std::string base = "blk." + std::to_string(layer_idx) + ".";
        int n_exp = model_file.architecture_config.n_experts;

        // ---- Packed expert format: ffn_{gate,up,down}_exps.weight ----
        // Each packed tensor is [a, b, n_experts] with the expert index as the
        // LAST (contiguous) dimension, so expert E occupies bytes
        // [E*per_exp, (E+1)*per_exp). ne0 is the input dim, so the slice is
        // already the [O,I] row-major weight the matmul expects — no shuffle.
        // (GLM-5.2 UD ships this way: gate/up = Q2_K, down = Q3_K, packed.)
        size_t gb, ub, db;
        const GGUFTensor *gM, *uM, *dM;
        if (packed_slab_geometry(layer_idx, gb, ub, db, gM, uM, dM)) {
            const size_t total = gb + ub + db;
            ensure_slab_capacity(target_slot, total);

            uint8_t* dst_gate = target_slot.slab_buffer;
            uint8_t* dst_up   = target_slot.slab_buffer + gb;
            uint8_t* dst_down = target_slot.slab_buffer + gb + ub;

            PERF_START(disk);

            // Check for interleaved per-expert layout: if gate+up+down are
            // contiguous on disk (the layout produced by repacked GGUF with
            // interleaved experts), use a single coalesced pread instead of 3.
            // This is the key optimization: one large aligned request saturates
            // NVMe queue depth far better than 3 scattered smaller requests.
            uint64_t gate_expert_offset = gM->offset + (uint64_t)expert_id * gb;
            uint64_t up_expert_offset   = uM->offset + (uint64_t)expert_id * ub;
            uint64_t down_expert_offset = dM->offset + (uint64_t)expert_id * db;
            
            bool is_interleaved = (up_expert_offset == gate_expert_offset + gb) &&
                                  (down_expert_offset == up_expert_offset + ub);

            if (is_interleaved && model_file.fd_for_tensor(*gM) == model_file.fd_for_tensor(*uM) &&
                model_file.fd_for_tensor(*uM) == model_file.fd_for_tensor(*dM)) {
                // Fast Path: Single coalesced pread for the entire expert slab
                // (gate, up, down packed tightly). This fires on repacked GGUFs
                // where tools/gguf_tool pack writes experts in interleaved order.
                auto t0 = std::chrono::high_resolution_clock::now();
                int fd = model_file.fd_for_tensor(*gM);
                ssize_t r = direct_io::pread_direct_aware(
                    fd, model_file.direct_io_active_for(*gM), model_file.direct_io_alignment,
                    dst_gate, total, gate_expert_offset);
                double ms = std::chrono::duration<double, std::milli>(
                    std::chrono::high_resolution_clock::now() - t0).count();
                if (r < 0 || (size_t)r != total) {
                    LOG_WARN("[WARN] interleaved expert coalesced read short: L%d E%d %zd/%zu\n",
                             layer_idx, expert_id, r, total);
                } else if (ms > SLOW_PREAD_MS) {
                    LOG_WARN("[SLOW-PREAD] coalesced L%d E%d: %.2fMB @ %llu took %.2fms (%.2f MB/s)\n",
                             layer_idx, expert_id, total / 1e6, (unsigned long long)gate_expert_offset, ms,
                             ms > 0 ? (total / 1e6) / (ms / 1000.0) : 0.0);
                }
            } else {
                // Fallback: 3 separate reads (original format or tensors on different fds)
                auto read_slice = [this](const GGUFTensor& m, size_t rel_off, size_t bytes, uint8_t* dst, int l, int e, const char* which) {
                    auto t0 = std::chrono::high_resolution_clock::now();
                    int fd = model_file.fd_for_tensor(m);
                    ssize_t r = direct_io::pread_direct_aware(
                        fd, model_file.direct_io_active_for(m), model_file.direct_io_alignment,
                        dst, bytes, m.offset + rel_off);
                    double ms = std::chrono::duration<double, std::milli>(
                        std::chrono::high_resolution_clock::now() - t0).count();
                    if (r < 0 || (size_t)r != bytes)
                        LOG_WARN("[WARN] packed expert short read L%d E%d: %zd/%zu\n", l, e, r, bytes);
                    if (ms > SLOW_PREAD_MS) {
                        LOG_WARN("[SLOW-PREAD] layer=%d expert=%d proj=%s bytes=%.2fMB offset=%llu "
                                 "took=%.2fms (%.2f MB/s)\n",
                                 l, e, which, bytes / 1e6, (unsigned long long)(m.offset + rel_off), ms,
                                 ms > 0 ? (bytes / 1e6) / (ms / 1000.0) : 0.0);
                    }
                };
                read_slice(*gM, (size_t)expert_id * gb, gb, dst_gate, layer_idx, expert_id, "gate");
                read_slice(*uM, (size_t)expert_id * ub, ub, dst_up,   layer_idx, expert_id, "up");
                read_slice(*dM, (size_t)expert_id * db, db, dst_down, layer_idx, expert_id, "down");
            }
            PERF_END(disk, t_disk_read);
            perf().bytes_disk_read.fetch_add(total, std::memory_order_relaxed);

            if (drop_expert_pages) {
                posix_fadvise(model_file.fd_for_tensor(*gM),
                              gM->offset, gM->size_bytes, POSIX_FADV_DONTNEED);
            }

            populate_packed_slab(target_slot, expert_id, gb, ub, db, *gM, *uM, *dM);
            return;
        }

        // ---- Per-expert format: ffn_{gate,up,down}.E.weight (legacy) ----
        std::string gate_name = base + "ffn_gate." + std::to_string(expert_id) + ".weight";
        std::string up_name   = base + "ffn_up."   + std::to_string(expert_id) + ".weight";
        std::string down_name = base + "ffn_down." + std::to_string(expert_id) + ".weight";

        auto gate_it = model_file.tensors.find(gate_name);
        if (gate_it == model_file.tensors.end()) {
            target_slot.expert_id = expert_id;  // neither packed nor per-expert found
            return;
        }
        auto& gate = gate_it->second;
        auto down_it = model_file.tensors.find(down_name);
        if (down_it == model_file.tensors.end()) {
            target_slot.expert_id = expert_id;
            return;
        }
        auto& down = down_it->second;

        // Read each projection INDEPENDENTLY into a TIGHT gate|up|down slab
        // (gate@0, up@gb, down@gb+ub) and assign SLAB-RELATIVE offsets, so the
        // downstream VRAM upload + GPU dispatch (which index gate/up/down by
        // (proj.offset - gate.offset) deltas -- see expert_ffn_gpu_issue /
        // expert_ffn_gpu_finish) work identically to the packed path's slab.
        // This is LAYOUT-AGNOSTIC: it makes no assumption about the relative
        // file offsets of the three projections. The previous variant did ONE
        // combined pread over [gate.offset, down.end) -- correct only when the
        // file lays the three out contiguously as gate<=up<=down, and a real
        // size_t underflow (posix_memalign(SIZE_MAX) -> NULL -> the
        // "missing gate_proj.host_ptr" skip at execution_layer.cpp's Pass A)
        // otherwise -- exactly what a lexicographic tensor sort produces
        // (tools/gguf_tool pack orders them down<gate<up). Reading 3 slices
        // works for any ordering; a contiguous repack (gate/up/down adjacent
        // per expert) lets the kernel coalesce them back toward a single read.
        const size_t gb_pe = gate.size_bytes;
        const GGUFTensor* up_t = nullptr;
        size_t ub_pe = 0;
        auto up_it = model_file.tensors.find(up_name);
        if (up_it != model_file.tensors.end()) {
            up_t = &up_it->second;
            ub_pe = up_t->size_bytes;
        }
        const size_t db_pe = down.size_bytes;
        const size_t total = gb_pe + ub_pe + db_pe;
        ensure_slab_capacity(target_slot, total);

        uint8_t* dst_gate = target_slot.slab_buffer;
        uint8_t* dst_up   = target_slot.slab_buffer + gb_pe;
        uint8_t* dst_down = target_slot.slab_buffer + gb_pe + ub_pe;

        auto read_proj = [&](const GGUFTensor& m, size_t bytes, uint8_t* dst, const char* which) {
            auto t0 = std::chrono::high_resolution_clock::now();
            int fd = model_file.fd_for_tensor(m);
            ssize_t r = direct_io::pread_direct_aware(
                fd, model_file.direct_io_active_for(m), model_file.direct_io_alignment,
                dst, bytes, m.offset);
            double ms = std::chrono::duration<double, std::milli>(
                std::chrono::high_resolution_clock::now() - t0).count();
            if (r < 0 || (size_t)r != bytes)
                LOG_WARN("[WARN] expert short read L%d E%d %s: %zd/%zu\n",
                         layer_idx, expert_id, which, r, bytes);
            if (ms > SLOW_PREAD_MS) {
                LOG_WARN("[SLOW-PREAD] layer=%d expert=%d proj=%s bytes=%.2fMB offset=%llu "
                         "took=%.2fms (%.2f MB/s)\n",
                         layer_idx, expert_id, which, bytes / 1e6, (unsigned long long)m.offset, ms,
                         ms > 0 ? (bytes / 1e6) / (ms / 1000.0) : 0.0);
            }
        };

        PERF_START(disk);
        // Single-read fast path: if up follows gate TIGHTLY and down follows up
        // TIGHTLY, and all three share one shard fd, fetch the WHOLE expert in
        // ONE pread
        // writes (each expert block packed tight, gate/up/down adjacent). One
        // sequential ~total-byte read saturates NVMe far better than three
        // scattered ones, and disk is the bottleneck here. Guarded so it ONLY
        // fires for genuinely tight, same-shard layouts: a scattered legacy
        // layout, OR an expert whose projections got split across two output
        // shards (different fds), fails the guard and falls through to the
        // always-correct 3-independent-read path below.
        const int gate_fd = model_file.fd_for_tensor(gate);
        const bool tight = up_t
            && model_file.fd_for_tensor(*up_t) == gate_fd
            && model_file.fd_for_tensor(down) == gate_fd
            && up_t->offset == gate.offset + gb_pe
            && down.offset == up_t->offset + ub_pe;
        if (tight) {
            auto t0 = std::chrono::high_resolution_clock::now();
            ssize_t r = direct_io::pread_direct_aware(
                gate_fd, model_file.direct_io_active_for(gate), model_file.direct_io_alignment,
                target_slot.slab_buffer, total, gate.offset);
            double ms = std::chrono::duration<double, std::milli>(
                std::chrono::high_resolution_clock::now() - t0).count();
            if (r < 0 || (size_t)r != total)
                LOG_WARN("[WARN] expert slab short read L%d E%d: %zd/%zu\n",
                         layer_idx, expert_id, r, total);
            if (ms > SLOW_PREAD_MS) {
                LOG_WARN("[SLOW-PREAD] layer=%d expert=%d proj=expert-slab bytes=%.2fMB offset=%llu "
                         "took=%.2fms (%.2f MB/s)\n",
                         layer_idx, expert_id, total / 1e6, (unsigned long long)gate.offset, ms,
                         ms > 0 ? (total / 1e6) / (ms / 1000.0) : 0.0);
            }
        } else {
            read_proj(gate, gb_pe, dst_gate, "gate");
            if (up_t) read_proj(*up_t, ub_pe, dst_up, "up");
            read_proj(down, db_pe, dst_down, "down");
        }
        PERF_END(disk, t_disk_read);
        perf().bytes_disk_read.fetch_add(total, std::memory_order_relaxed);

        if (drop_expert_pages) {
            posix_fadvise(model_file.fd_for_tensor(gate), gate.offset, gb_pe, POSIX_FADV_DONTNEED);
            if (up_t) posix_fadvise(model_file.fd_for_tensor(*up_t), up_t->offset, ub_pe, POSIX_FADV_DONTNEED);
            posix_fadvise(model_file.fd_for_tensor(down), down.offset, db_pe, POSIX_FADV_DONTNEED);
        }

        target_slot.gate_proj.host_ptr = target_slot.slab_buffer;
        target_slot.gate_proj.meta = gate;
        target_slot.gate_proj.meta.offset = 0;
        target_slot.gate_proj.meta.size_bytes = gb_pe;
        if (up_t) {
            target_slot.up_proj.host_ptr = target_slot.slab_buffer + gb_pe;
            target_slot.up_proj.meta = *up_t;
            target_slot.up_proj.meta.offset = gb_pe;
            target_slot.up_proj.meta.size_bytes = ub_pe;
        } else {
            target_slot.up_proj.host_ptr = nullptr;
        }
        target_slot.down_proj.host_ptr = target_slot.slab_buffer + gb_pe + ub_pe;
        target_slot.down_proj.meta = down;
        target_slot.down_proj.meta.offset = gb_pe + ub_pe;
        target_slot.down_proj.meta.size_bytes = db_pe;
        target_slot.expert_id = expert_id;
        target_slot.vram_shadow = false;
    }

    // True per-expert slab size in bytes, accounting for each projection's own
    // quant type (e.g. GLM-5.2 UD: gate/up=Q2_K, down=Q3_K — NOT uniformly
    // sized). Mirrors load_expert_slab's sizing exactly so the VRAM cache slot
    // size and the prefetcher's staging buffer size always match what
    // load_expert_slab actually allocates. Returns 0 if layer_idx has no
    // expert tensors at all (e.g. a leading dense layer).
    size_t compute_expert_slab_bytes(int layer_idx) const {
        std::string base = "blk." + std::to_string(layer_idx) + ".";
        int n_exp = model_file.architecture_config.n_experts;

        auto gE = model_file.tensors.find(base + "ffn_gate_exps.weight");
        auto uE = model_file.tensors.find(base + "ffn_up_exps.weight");
        auto dE = model_file.tensors.find(base + "ffn_down_exps.weight");
        if (n_exp > 0 && gE != model_file.tensors.end() &&
            uE != model_file.tensors.end() && dE != model_file.tensors.end()) {
            return gE->second.size_bytes / n_exp + uE->second.size_bytes / n_exp
                 + dE->second.size_bytes / n_exp;
        }

        auto git = model_file.tensors.find(base + "ffn_gate.0.weight");
        auto uit = model_file.tensors.find(base + "ffn_up.0.weight");
        auto dit = model_file.tensors.find(base + "ffn_down.0.weight");
        if (git != model_file.tensors.end() && dit != model_file.tensors.end()) {
            // Sum each projection's OWN size -- layout-agnostic. The earlier
            // (down.end - gate.offset) form assumed gate's file offset was the
            // lowest of the three and underflowed size_t on a scattered repack
            // layout (down<gate<up), handing VRAM-cache/prefetch sizing a
            // ~SIZE_MAX value. Mirrors load_expert_slab's tight-slab total
            // (gb + ub + db) exactly.
            size_t total = git->second.size_bytes + dit->second.size_bytes;
            if (uit != model_file.tensors.end()) total += uit->second.size_bytes;
            return total;
        }

        return 0;
    }

    void issue_prefetch(int layer_idx, const std::vector<int>& predicted_expert_ids) {
        PERF_SCOPE(t_prefetch);
        std::string base = "blk." + std::to_string(layer_idx) + ".";
        for (int exp_id : predicted_expert_ids) {
            std::string es = std::to_string(exp_id);
            // Prefetch each projection by name rather than (gate.offset,
            // gate.size*3): the old hint assumed up/down immediately follow
            // gate contiguously AND share gate's per-expert size -- neither
            // holds for a repacked layout (down<gate<up) nor for mixed-quant
            // experts (gate/up=Q2_K, down=Q3_K). Per-projection is correct for
            // any layout.
            for (const char* proj : {"ffn_gate", "ffn_up", "ffn_down"}) {
                auto it = model_file.tensors.find(base + proj + "." + es + ".weight");
                if (it != model_file.tensors.end()) {
                    posix_fadvise(model_file.fd_for_tensor(it->second),
                                  it->second.offset, it->second.size_bytes, POSIX_FADV_WILLNEED);
                }
            }
        }
    }

    // ---- Learned hot-set profiling (Phase A ) ----
    // route_freq is [n_layers * n_experts]; call once layers/n_experts are known
    // (after build_model_graph). Idempotent.
    void init_route_freq() {
        int nl = model_file.architecture_config.n_layers;
        int ne = model_file.architecture_config.n_experts;
        if (nl <= 0 || ne <= 0) return;
        size_t n = (size_t)nl * (size_t)ne;
        if (route_freq.size() != n) {
            route_freq = std::vector<std::atomic<uint64_t>>(n);
        }
    }
    inline void bump_route_freq(int layer, int expert) {
        if (route_freq.empty()) return;
        int ne = model_file.architecture_config.n_experts;
        if (layer < 0 || expert < 0 || ne <= 0) return;
        size_t idx = (size_t)layer * (size_t)ne + (size_t)expert;
        if (idx < route_freq.size())
            route_freq[idx].fetch_add(1, std::memory_order_relaxed);
    }

    // Write "layer expert count" lines for every (layer,expert) ever selected.
    // Accumulates on top of whatever a PREVIOUS run already wrote here,
    // rather than overwriting it
    // even engage below 5,000 accumulated selections and doesn't reach its
    // full budget until 200,000). The original implementation opened the
    // file in plain "w" mode and wrote only THIS session's in-memory
    // route_freq -- since route_freq itself is zero-initialized fresh at
    // every process start (init_route_freq) and is never seeded FROM the
    // file, every run's shutdown was silently discarding all prior history
    // and replacing it with just that one run's counts (observed: the
    // [PIN] log's "max count" stayed stuck around 24-25 across an entire
    // session's worth of runs -- it was never actually accumulating).
    // Fixed by reading whatever's already on disk first and summing counts
    // per (layer,expert) before writing back out. A user who wants a clean
    // slate (checkpoint swap, or if a matured/pinned profile ever starts
    // hurting instead of helping -- see the pin_vram_lru_fraction/
    // pin_ram_fraction regression writeup) can just delete the file; the
    // next run's load_usage() already treats a missing file as "first run",
    // no separate reset flag needed.
    void dump_usage(const std::string& path) const {
        if (path.empty() || route_freq.empty()) return;
        int ne = model_file.architecture_config.n_experts;
        auto [prior, prior_maxc] = load_usage(path);
        (void)prior_maxc;
        std::vector<uint64_t> merged(route_freq.size(), 0);
        for (auto& e : prior) {
            size_t idx = (size_t)e.layer * (size_t)ne + (size_t)e.expert;
            if (idx < merged.size()) merged[idx] = e.count;
        }
        FILE* f = std::fopen(path.c_str(), "w");
        if (!f) { LOG_WARN("[USAGE] could not write %s\n", path.c_str()); return; }
        for (size_t i = 0; i < route_freq.size(); ++i) {
            uint64_t c = merged[i] + route_freq[i].load(std::memory_order_relaxed);
            if (c == 0) continue;
            std::fprintf(f, "%zu %zu %llu\n", i / (size_t)ne, i % (size_t)ne, (unsigned long long)c);
        }
        std::fclose(f);
        LOG_INFO("[USAGE] wrote routing-frequency histogram -> %s (accumulated across sessions)\n", path.c_str());
    }

    // Read a usage file written by dump_usage. Returns one UsageEntry per
    // (layer,expert) row plus the max count seen (for relative thresholds).
    // Malformed lines are skipped silently; a missing file (first run) returns
    // an empty vector with maxc=0 -- not an error.
    struct UsageEntry { int layer; int expert; uint64_t count; };
    static std::pair<std::vector<UsageEntry>, uint64_t> load_usage(const std::string& path) {
        std::vector<UsageEntry> out;
        uint64_t maxc = 0;
        if (path.empty()) return {out, maxc};
        FILE* f = std::fopen(path.c_str(), "r");
        if (!f) return {out, maxc};
        char line[128];
        while (std::fgets(line, sizeof(line), f)) {
            int layer, expert; unsigned long long cnt;
            if (std::sscanf(line, "%d %d %llu", &layer, &expert, &cnt) == 3) {
                out.push_back({layer, expert, (uint64_t)cnt});
                if (cnt > maxc) maxc = cnt;
            }
        }
        std::fclose(f);
        return {out, maxc};
    }
};