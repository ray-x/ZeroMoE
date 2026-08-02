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
#include <algorithm>

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

    // TEMPORARY protection for a RAM-tier slot referenced by a raw
    // MoEExpertBlock* held across moe_forward_gguf's Pass A -> Pass B gap
    // (execution_layer.cpp's PendingGpuExpert::exp) -- reference-counted,
    // same discipline as GpuTopology::ExpertLocation::temp_pin_count on the
    // VRAM side. Real, measured bug this closes: Pass A resolves `exp`,
    // takes a VRAM hold_for_compute() keyed on the expert_id AT THAT TIME,
    // and queues Pass B to release it later -- but Pass B reads
    // `p.exp->expert_id` FRESH at release time, not a value captured at
    // Pass A. If ANY RAM-tier victim-selection scan (demote_from_vram's
    // VRAM-eviction salvage being the one actually observed firing --
    // config vram_evict_demote_to_ram=true makes this reachable from
    // WITHIN Pass A itself, since an earlier k's synchronous VRAM upload
    // can trigger on_vram_evict/demote_from_vram before this k's Pass B
    // ever runs) reclaims this exact slot for a DIFFERENT expert in that
    // gap, Pass B's release_hold() call ends up searching for the WRONG
    // (new) identity -- "release_hold found no matching resident slot,
    // which should never happen" is the resulting symptom. Worse than a
    // log line: the REAL VRAM slot that was actually held NEVER gets its
    // temp_pin_count decremented when this happens, permanently leaking it
    // as un-evictable -- every occurrence shrinks the usable VRAM cache by
    // one slot for the rest of the process, a real cumulative cause of
    // throughput degrading over a long run. Every RAM-slot victim-
    // selection site (get_expert Step 3, get_experts_batch Phase A,
    // demote_from_vram) must check this alongside copy_still_pending(),
    // same "never pick, no matter how attractive otherwise" priority.
    int ram_temp_pin_count = 0;

    // True when this slot's bytes were served via LockFreePrefetcher::
    // consume_staged (a RAM-tier miss the pilot correctly predicted and
    // staged ahead of time) rather than a synchronous disk pread. Exists
    // solely to make the effect of removing finish_prefetch's old
    // unconditional VRAM-promotion call (async_prefetcher.hpp) directly
    // observable: n_pipe_staged_later_promoted (perf_timer.hpp) counts how
    // many of these later cross promote_hit_threshold and get promoted via
    // the normal hit-count-gated path, proving prefetched experts still
    // reach VRAM -- just only once a real subsequent access confirms it,
    // never on unconfirmed speculation. Reset to false whenever a slot is
    // claimed for a different expert, same lifecycle as also_in_vram.
    bool from_pipe_prefetch = false;

    // True when this slot was claimed by Phase B's speculative disk-batch
    // padding (RAMSafetyManager::get_experts_batch, disk_batch_padding
    // config -- a route_freq-ranked, historically hot but not currently
    // resident expert, loaded to pad queue depth on an already-necessary
    // io_uring submission, not because THIS token actually routed to it).
    // One-shot: cleared the first time this slot is genuinely hit (n_pad_
    // reads_hit_later, perf_timer.hpp) -- deliberately NOT persistent like
    // from_pipe_prefetch above (which tracks promotion eligibility, not a
    // single did-this-pay-off event). Also reset to false whenever a slot
    // is claimed for a different expert, same lifecycle as also_in_vram.
    bool from_padding = false;

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
    // Parallel histogram, same [n_layers * n_experts] shape, but ONLY ever
    // bumped for selections made during prefill (S > 1) -- see
    // bump_route_freq_prefill. route_freq itself keeps accumulating BOTH
    // phases combined, unchanged, for anything that already relies on the
    // combined signal; this is purely additive. Motivated by a real,
    // measured problem: decode contributes far more routing VOLUME per
    // request than prefill does (e.g. 40 decode tokens' worth of top-k
    // selections vs. 11 prefill tokens'), so in the COMBINED histogram a
    // decode-favored expert can outrank a prefill-favored one purely on
    // count, even though prefill needs it every single run -- starving
    // startup [PIN] preload of prefill's own signal. Dumped to a SEPARATE
    // file (EngineConfig::prefill_pin_file) so preload can consult it
    // specifically. See CLAUDE.md.
    std::vector<std::atomic<uint64_t>> route_freq_prefill;
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

    // Set from EngineConfig::unpacked_expert_format at startup (main.cpp),
    // before sanity_check_unpacked_format() is called. See that config
    // field's own comment for the full contract: true means this model
    // directory was produced by `gguf_tool pack --unpack` (per-expert
    // tensors, physically tight gate/up/down per expert) rather than the
    // default packed _exps format, and every fast-path geometry lookup
    // below (expert_layer_geometry, build_expert_read_requests) should
    // resolve against the per-expert tensor convention instead.
    bool unpacked_expert_format = false;

    // Set from EngineConfig::cuda_cufile at startup (main.cpp), AFTER
    // confirming GGUFFile::enable_cufile() actually registered at least one
    // shard AND GpuTopology's expert cache pool(s) registered successfully
    // (cufile_io::available() plus real per-shard/per-device registration
    // results) -- this is the single "is cuFile genuinely usable right
    // now" flag execution_layer.cpp's Phase 3 loading loop checks, distinct
    // from config.cuda_cufile (the raw request) the same way direct_io/
    // io_uring already distinguish "requested" from "actually active".
    bool cuda_cufile_active = false;

    // Set from EngineConfig::prefill_warm_via_prefetcher at startup
    // (main.cpp). See that config field's own comment: routes Phase 2's
    // (moe_forward_gguf's S>1 cold-start VRAM warm-up) candidate loads
    // through the background prefetcher instead of loading them
    // synchronously in the critical path.
    bool prefill_warm_via_prefetcher = false;

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

    // Per-(layer,expert) geometry for the `unpacked_expert_format` layout
    // (gguf_tool pack --unpack): unlike packed_slab_geometry, there is no
    // single shared _exps tensor to slice by expert index -- gate/up/down
    // are THREE SEPARATE tensors per expert, named blk.N.ffn_{gate,up,
    // down}.E.weight. Returns false if any of the three are missing for
    // this exact (layer_idx, expert_id) pair. Pure tensor-table lookup, no
    // I/O -- same cost class as packed_slab_geometry.
    bool unpacked_expert_geometry(int layer_idx, int expert_id, size_t& gb, size_t& ub, size_t& db,
                                   const GGUFTensor*& gM, const GGUFTensor*& uM, const GGUFTensor*& dM) const {
        std::string base = "blk." + std::to_string(layer_idx) + ".";
        std::string es = std::to_string(expert_id);
        auto gIt = model_file.tensors.find(base + "ffn_gate." + es + ".weight");
        auto uIt = model_file.tensors.find(base + "ffn_up." + es + ".weight");
        auto dIt = model_file.tensors.find(base + "ffn_down." + es + ".weight");
        if (gIt == model_file.tensors.end() || uIt == model_file.tensors.end() || dIt == model_file.tensors.end())
            return false;
        gM = &gIt->second; uM = &uIt->second; dM = &dIt->second;
        gb = gM->size_bytes; ub = uM->size_bytes; db = dM->size_bytes;
        return true;
    }

    // True only if gM/uM/dM are on the SAME fd and physically back-to-back
    // with no gap (up starts exactly where gate ends, down exactly where up
    // ends) -- the precondition for reading all three with ONE combined DMA
    // request instead of three scattered ones. tools/gguf_pack.cpp's
    // --unpack layout pass produces exactly this for every expert (see its
    // own Step 4 comment); sanity_check_unpacked_format validates it holds
    // for the WHOLE model at startup, and build_expert_read_requests
    // re-checks it defensively at read time (belt-and-suspenders -- the
    // tensor table is static after load_header, so this should never
    // disagree with the startup check, but a mismatch here means "refuse
    // to read across an unknown gap", never "guess and read garbage").
    bool unpacked_expert_is_tight(const GGUFTensor& gM, const GGUFTensor& uM, const GGUFTensor& dM) const {
        int fd = model_file.fd_for_tensor(gM);
        return model_file.fd_for_tensor(uM) == fd && model_file.fd_for_tensor(dM) == fd
            && uM.offset == gM.offset + gM.size_bytes
            && dM.offset == uM.offset + uM.size_bytes;
    }

    // Unified layer-level geometry: dispatches to packed_slab_geometry (the
    // default format) or, when unpacked_expert_format is set, to
    // unpacked_expert_geometry using expert 0 as the layer's representative
    // (sanity_check_unpacked_format guarantees every expert in a layer
    // shares the same gb/ub/db before the engine ever starts real work, so
    // expert 0 is exactly as valid as any other expert for TYPE/SIZE
    // purposes -- callers needing a SPECIFIC expert's own tensors for a
    // real read still call unpacked_expert_geometry directly, see
    // build_expert_read_requests). Existing callers that only need
    // layer-level sizing/type info (populate_vram_shadow,
    // compute_expert_slab_bytes) gain unpacked-format support through this
    // one switch point with no other change.
    bool expert_layer_geometry(int layer_idx, size_t& gb, size_t& ub, size_t& db,
                                const GGUFTensor*& gM, const GGUFTensor*& uM, const GGUFTensor*& dM) const {
        if (unpacked_expert_format)
            return unpacked_expert_geometry(layer_idx, 0, gb, ub, db, gM, uM, dM);
        return packed_slab_geometry(layer_idx, gb, ub, db, gM, uM, dM);
    }

    // Full-model startup validation for unpacked_expert_format=true. Called
    // once from main.cpp right after the model graph is built (so
    // architecture_config.n_layers/n_experts are known), BEFORE any real
    // generation work starts. Confirms, for every layer that has experts:
    // every expert 0..n_experts-1 has all three tensors present, every
    // expert's projection sizes match the layer's own expert 0 (the
    // uniformity expert_layer_geometry's single-representative-expert
    // shortcut depends on), and every expert is physically tight
    // (unpacked_expert_is_tight) -- the precondition for
    // build_expert_read_requests' single-DMA path. Any violation prints a
    // FATAL line to stderr and returns false immediately (does not keep
    // scanning past the first failure) -- this flag is an explicit
    // performance contract from whoever set it (the whole point of
    // --unpack + this flag is "faster than the default packed format", see
    // CLAUDE.md), so a violation must stop the run rather than silently
    // degrade to the slow per-projection path the flag exists to avoid.
    // A layer with ZERO experts present (a leading dense-only layer, e.g.
    // GLM-5.2's first_dense layers) is legitimate and skipped, not an
    // error; a layer with SOME but not all experts present is a real,
    // fatal incomplete-repack signature (mirrors gguf_tool verify's own
    // completeness check, but enforced by the engine itself at load time
    // for this specific format instead of left to a separate offline step).
    bool sanity_check_unpacked_format() const {
        int n_layers = model_file.architecture_config.n_layers;
        int n_exp = model_file.architecture_config.n_experts;
        if (n_layers <= 0 || n_exp <= 0) {
            fprintf(stderr, "[FATAL] unpacked_expert_format=true but architecture_config has "
                            "n_layers=%d n_experts=%d -- cannot validate, refusing to run\n",
                    n_layers, n_exp);
            return false;
        }
        for (int l = 0; l < n_layers; ++l) {
            size_t layer_gb = 0, layer_ub = 0, layer_db = 0;
            bool have_layer_size = false;
            int found = 0;
            for (int e = 0; e < n_exp; ++e) {
                size_t gb, ub, db;
                const GGUFTensor *gM, *uM, *dM;
                if (!unpacked_expert_geometry(l, e, gb, ub, db, gM, uM, dM)) {
                    if (found == 0) break;  // no experts in this layer at all -- fine, e.g. a dense-only layer
                    fprintf(stderr, "[FATAL] unpacked_expert_format=true: layer %d expert %d is missing "
                                    "one or more of ffn_{gate,up,down}.%d.weight (found %d/%d experts for "
                                    "this layer before this gap) -- incomplete repack, not a valid "
                                    "'aligned unpacked' model. Stopping before any real read.\n",
                            l, e, e, found, n_exp);
                    return false;
                }
                found++;
                if (!have_layer_size) {
                    layer_gb = gb; layer_ub = ub; layer_db = db;
                    have_layer_size = true;
                } else if (gb != layer_gb || ub != layer_ub || db != layer_db) {
                    fprintf(stderr, "[FATAL] unpacked_expert_format=true: layer %d expert %d has "
                                    "projection sizes (%zu,%zu,%zu) bytes but this layer's expert 0 had "
                                    "(%zu,%zu,%zu) -- non-uniform per-expert sizes within one layer are "
                                    "not supported by the single-DMA fast path. Stopping.\n",
                            l, e, gb, ub, db, layer_gb, layer_ub, layer_db);
                    return false;
                }
                if (!unpacked_expert_is_tight(*gM, *uM, *dM)) {
                    fprintf(stderr, "[FATAL] unpacked_expert_format=true: layer %d expert %d's "
                                    "gate/up/down are not physically contiguous on one file (likely split "
                                    "across a repack output-shard boundary) -- the single-DMA fast path "
                                    "requires every expert to be tight. Re-run gguf_tool pack --unpack "
                                    "with a larger --max-shard-gb so no expert straddles a shard boundary, "
                                    "or set unpacked_expert_format=false and use the default --align-only "
                                    "packed format instead. Stopping before any real read.\n",
                            l, e);
                    return false;
                }
            }
            if (found > 0 && found != n_exp) {
                fprintf(stderr, "[FATAL] unpacked_expert_format=true: layer %d has only %d/%d experts "
                                "present -- incomplete repack. Stopping.\n",
                        l, found, n_exp);
                return false;
            }
        }
        LOG_INFO("[UNPACKED] sanity check passed: every MoE layer's experts are complete, uniformly "
                 "sized, and physically tight -- single-DMA reads enabled.\n");
        return true;
    }

    // Detects the specific mismatch that caused a real, measured slowdown:
    // a model genuinely IS in the per-expert ("unpacked") tensor-naming
    // convention (gguf_tool verify would report "Expert format: per-expert"
    // for it), but unpacked_expert_format wasn't set to tell the engine
    // that -- which silently disables BOTH io_uring Phase B batching and
    // the VRAM-shadow bypass fast path for the ENTIRE run (packed_slab_
    // geometry, the shared gate for both, finds none of the packed _exps
    // tensors it looks for), forcing every dispatch through the slow,
    // unbatched fallback with zero visible error. A real run showed this
    // exact signature: 100% "layer-not-packed" on every decode-thread
    // io_uring batch attempt, `disk=` time exceeding its own containing
    // `expert=` wall-clock window (the thread-summed-OMP-fallback tell),
    // and no [CONFIG] indication anything was wrong.
    //
    // Pure tensor-table lookups only (packed_slab_geometry/
    // unpacked_expert_geometry), no I/O -- cheap enough to run
    // unconditionally at startup. Deliberately conservative: returns true
    // ONLY if EVERY MoE layer is missing packed tensors AND has per-expert
    // ones -- the moment even one layer has packed tensors, this returns
    // false immediately, since a MIXED result means something else is
    // wrong (an incomplete/partial repack) that's gguf_tool verify's job
    // to catch, not this check's. Intentionally warn-only at the call
    // site, not a hard refusal like sanity_check_unpacked_format's own --
    // the model may still run correctly here, just slowly, and there's no
    // equivalent contract violation to refuse over.
    bool looks_like_unpacked_format() const {
        int n_layers = model_file.architecture_config.n_layers;
        int first_dense = std::max(0, model_file.architecture_config.first_dense);
        if (n_layers <= 0 || first_dense >= n_layers) return false;
        int total_moe_layers = 0, unpacked_signature = 0;
        for (int l = first_dense; l < n_layers; ++l) {
            ++total_moe_layers;
            size_t gb, ub, db;
            const GGUFTensor *gM, *uM, *dM;
            if (packed_slab_geometry(l, gb, ub, db, gM, uM, dM)) {
                return false; // at least one layer IS packed -- not this scenario
            }
            if (unpacked_expert_geometry(l, 0, gb, ub, db, gM, uM, dM)) ++unpacked_signature;
        }
        return total_moe_layers > 0 && unpacked_signature == total_moe_layers;
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
    // down=gb+ub) from expert_layer_geometry, which is a pure tensor-table
    // lookup already parsed at GGUF load time -- same source
    // load_expert_slab itself uses before it ever touches disk. host_ptr
    // is deliberately left null; slot.vram_shadow=true is the required
    // signal callers must check before touching it (see MoEExpertBlock's
    // own comment). Returns false (slot untouched) if this layer has no
    // packed expert tensors at all (e.g. a non-MoE layer).
    bool populate_vram_shadow(MoEExpertBlock& slot, int layer_idx, int expert_id) const {
        size_t gb = 0, ub = 0, db = 0;
        const GGUFTensor *gM = nullptr, *uM = nullptr, *dM = nullptr;
        if (!expert_layer_geometry(layer_idx, gb, ub, db, gM, uM, dM)) return false;
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

    // Builds the io_uring read request(s) needed to load ONE expert's
    // slab (3 for packed format, exactly 1 for unpacked_expert_format --
    // see below) -- but does NOT perform any reads itself. Deliberately
    // kept separate from load_expert_slab (which remains completely
    // unchanged as the proven, correctness-critical
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
    // A per-expert-tensor (legacy) model with NO unpacked_expert_format
    // guarantee is deliberately NOT supported here (returns false) -- it's
    // already documented as the engine's cold/slow path elsewhere
    // (io_uring_backend.hpp, CLAUDE.md), not worth the added complexity for
    // an unvalidated layout; callers fall back to load_expert_slab for it
    // exactly as Phase B always has.
    //
    // Packed format: always issues 3 separate reads per expert
    // (gate/up/down), even when they're contiguous on disk -- unlike
    // load_expert_slab's own single-shard coalescing optimization. Simpler,
    // and the coalescing win matters less here: io_uring's real benefit
    // (kernel-managed deep queue depth across the WHOLE batch's reads at
    // once) applies regardless of whether any one expert's 3 reads happen
    // to be contiguous.
    //
    // unpacked_expert_format: issues exactly ONE combined read spanning
    // gate+up+down for this expert -- the "single DMA request transfers
    // all three tensors" this format exists for. Safe only because
    // sanity_check_unpacked_format already confirmed EVERY expert in the
    // model is tight at startup; unpacked_expert_is_tight is re-checked
    // here anyway (cheap, pure metadata) as defense-in-depth -- a mismatch
    // here means something is wrong that startup validation didn't catch
    // (should never happen against a static tensor table), and the correct
    // response is "refuse this one read", not "guess and read garbage
    // across an unknown gap".
    bool build_expert_read_requests(int layer_idx, int expert_id,
                                     MoEExpertBlock& target_slot,
                                     io_uring_backend::ReadRequest* out_reqs,
                                     int& out_n, size_t& out_total) {
        if (unpacked_expert_format) {
            size_t gb, ub, db;
            const GGUFTensor *gM, *uM, *dM;
            if (!unpacked_expert_geometry(layer_idx, expert_id, gb, ub, db, gM, uM, dM)) return false;
            if (!unpacked_expert_is_tight(*gM, *uM, *dM)) {
                LOG_WARN("[WARN] unpacked expert L%d E%d failed the tightness re-check at read time "
                         "despite passing startup sanity -- refusing single-DMA read (falling back)\n",
                         layer_idx, expert_id);
                return false;
            }
            out_total = gb + ub + db;
            ensure_slab_capacity(target_slot, out_total);
            out_n = 0;
            out_reqs[out_n++] = {model_file.fd_for_tensor(*gM), model_file.direct_io_active_for(*gM),
                                  model_file.direct_io_alignment, target_slot.slab_buffer, out_total, gM->offset};
            return true;
        }

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
        if (route_freq_prefill.size() != n) {
            route_freq_prefill = std::vector<std::atomic<uint64_t>>(n);
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
    // Called ADDITIONALLY (never instead of) bump_route_freq, only when the
    // caller knows this selection happened during prefill (S > 1) -- see
    // route_freq_prefill's own comment for why this needs to be tracked
    // separately from the combined histogram.
    inline void bump_route_freq_prefill(int layer, int expert) {
        if (route_freq_prefill.empty()) return;
        int ne = model_file.architecture_config.n_experts;
        if (layer < 0 || expert < 0 || ne <= 0) return;
        size_t idx = (size_t)layer * (size_t)ne + (size_t)expert;
        if (idx < route_freq_prefill.size())
            route_freq_prefill[idx].fetch_add(1, std::memory_order_relaxed);
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
        dump_usage_histogram(path, route_freq);
    }
    // Separate output file for the prefill-only histogram -- see
    // route_freq_prefill's own comment. Same accumulate-across-sessions
    // semantics as dump_usage, just a different source vector and
    // (typically) a different path (EngineConfig::prefill_pin_file).
    void dump_usage_prefill(const std::string& path) const {
        dump_usage_histogram(path, route_freq_prefill);
    }

private:
    // Shared body for dump_usage/dump_usage_prefill -- identical
    // accumulate-with-whatever's-already-on-disk logic, just parameterized
    // on WHICH histogram to write, so this non-trivial merge logic exists
    // in exactly one place for both.
    void dump_usage_histogram(const std::string& path,
                               const std::vector<std::atomic<uint64_t>>& histogram) const {
        if (path.empty() || histogram.empty()) return;
        int ne = model_file.architecture_config.n_experts;
        auto [prior, prior_maxc] = load_usage(path);
        (void)prior_maxc;
        std::vector<uint64_t> merged(histogram.size(), 0);
        for (auto& e : prior) {
            size_t idx = (size_t)e.layer * (size_t)ne + (size_t)e.expert;
            if (idx < merged.size()) merged[idx] = e.count;
        }
        FILE* f = std::fopen(path.c_str(), "w");
        if (!f) { LOG_WARN("[USAGE] could not write %s\n", path.c_str()); return; }
        for (size_t i = 0; i < histogram.size(); ++i) {
            uint64_t c = merged[i] + histogram[i].load(std::memory_order_relaxed);
            if (c == 0) continue;
            std::fprintf(f, "%zu %zu %llu\n", i / (size_t)ne, i % (size_t)ne, (unsigned long long)c);
        }
        std::fclose(f);
        LOG_INFO("[USAGE] wrote routing-frequency histogram -> %s (accumulated across sessions)\n", path.c_str());
    }

public:

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