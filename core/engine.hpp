#pragma once
#include "gguf_loader.hpp"
#include "direct_io.hpp"
#include "io_uring_backend.hpp"
#include "perf_timer.hpp"
#include "logger.hpp"
#include "memory_guard.hpp"  // MemoryGuard::allocation_blocked() -- see ensure_slab_capacity
#include "family_ops.hpp" // FamilyOps -- see StreamingEngine::family_ops below
#include "dsv4_compressed_attention.hpp" // Dsv4CompressedLayerState -- see StreamingEngine::dsv4_compressed below
#include <cuda_runtime.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <thread>
#include <future>
#include <vector>
#include <sstream>   // warn_if_expert_tensors_unaligned
#include <memory>
#include <unordered_map>
#include <algorithm>
#include <numeric>
#include <mutex>
#include <chrono>

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
class StreamingEngine;
class Exl3GpuExecutor;
class Exl3CpuMatrixCache;

// GLM5NEXT KDA's production CUDA state is owned by CudaManager, but prompt
// cache save/restore is owned by the generation loop. These narrow bridges
// keep that boundary explicit without making engine.hpp include the complete
// CUDA manager definition.
void sync_glm5next_kda_gpu_states(StreamingEngine& engine);
void invalidate_glm5next_kda_gpu_states(StreamingEngine& engine);

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

// Which side is calling RAMSafetyManager::claim_or_wait (ram_manager.hpp) -- the
// main generation-loop thread's own synchronous miss-handling (moe_forward_gguf's
// per-token dispatch loop -- runs during BOTH prefill's S>1 loop and decode's S=1
// loop, so MainThread does NOT mean "this happened during decode", despite the
// name this replaced; see this enum's own git history / CLAUDE.md for the
// "why is claimed-by-decode nonzero during a --ngen 0, decode-never-ran prefill
// run" investigation that motivated the rename), or a LockFreePrefetcher worker
// thread's speculative claim. Gating behavior on this is a hard correctness
// requirement, not a style choice: claim_or_wait's Hit branch may only run its
// VRAM-promotion side effects (which take GpuTopology::cache_mutex) for
// MainThread -- a PrefetchWorker caller must never touch GpuTopology/cache_mutex
// at all, or a cross-thread AB-BA deadlock against the main thread's own
// cache_mutex-then-ram_mutex nesting (demote_from_vram/finish_sync_promotion,
// both reached from GpuTopology callbacks fired while cache_mutex is still held)
// becomes reachable. See ram_manager.hpp's claim_or_wait for the full reasoning.
// To actually distinguish prefill's own contribution from decode's: read the
// [PREFILL]-scoped "RAM claim" line (generation_loop.hpp) for prefill alone,
// vs. the shutdown "Unified RAM claim" line for the whole session including
// decode -- that phase split already exists via the per-request Snapshot delta
// mechanism, independent of this enum (which only ever distinguishes caller
// THREAD, main vs. background-worker, never generation PHASE).
enum class RamClaimCaller { MainThread, PrefetchWorker };

struct MoEExpertBlock {
    int expert_id;
    QuantizedTensor gate_proj;
    QuantizedTensor up_proj;
    QuantizedTensor down_proj;
    uint8_t* slab_buffer = nullptr;
    size_t slab_capacity = 0;
    // Bytes charged to MemoryGuard while this slab allocation is live.  Keep
    // this per-slot because slab growth can defer the old free while an async
    // H2D reader still owns the old pointer.
    size_t host_reservation_bytes = 0;
    // True once this slab_buffer has been cudaHostRegister'd ONCE, for the
    // lifetime of the allocation (EngineConfig::ram_pinned_budget_gb -- see
    // StreamingEngine::ensure_slab_capacity for the real-hardware reasoning).
    // Distinct from every OTHER cudaHostRegister site in this codebase
    // (upload_expert/upload_expert_async's own Alt-A conversions), which all
    // register and unregister around a SINGLE transfer; this one is paid
    // exactly once per allocation and never per call. MUST be unregistered
    // before the buffer is free()'d/realloc'd (ensure_slab_capacity's own
    // grow path does this) -- freeing still-registered memory is UB.
    bool slab_pinned = false;
    // The slab is a fixed-stride slice of StreamingEngine::graph_ram_arenas,
    // not an individually-owned allocation.  Graph-backed layers keep every
    // expert slice alive for the complete graph execution; release_slab and
    // ensure_slab_capacity must therefore never free, unregister, or regrow
    // this pointer.
    bool slab_from_shared_arena = false;
    uint64_t last_used_clock = 0;
    uint32_t hit_count = 0;  // frequency signal for eviction scoring (power-law routing)
    // EngineConfig::ram_grace_period_tokens (2026-09-04): the DECODE token
    // index (RAMSafetyManager::current_decode_token_idx_ref, NOT the raw
    // operation-tick last_used_clock above) at which this slot was last
    // genuinely (re)loaded with a NEW expert -- set only at the real load
    // sites (RAMSafetyManager::claim_or_wait's Claimed branch, the
    // demote_from_vram salvage-into-RAM path, disk_batch_padding's
    // speculative fill, and preload_ram_slot), never on a plain re-hit or
    // a slot-vacate/free. 0 by default -- harmless even with the feature
    // off, since eviction_score only reads it when ram_grace_period_
    // tokens > 0.
    uint64_t load_token_idx = 0;

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

    // Unified RAM-tier claim state (RAMSafetyManager::claim_or_wait,
    // ram_manager.hpp): true from the moment a caller (decode thread's own miss
    // path, OR a LockFreePrefetcher worker thread) claims this slot for
    // `expert_id` until that same caller's disk read completes and publishes the
    // result. Combined with `expert_id`, this gives the RAM tier's three real
    // states with no new type: expert_id==-1 -> Empty; expert_id==target &&
    // ram_loading -> Loading (claimed, read in progress, slab_buffer/gate_proj/
    // up_proj/down_proj not yet valid); expert_id==target && !ram_loading ->
    // Ready (today's ordinary hit-path semantics, unchanged).
    //
    // Field-ownership rule, the thing that makes the unlocked disk read fast and
    // safe at the same time: ram_loading, expert_id, last_used_clock, hit_count,
    // also_in_vram, ram_temp_pin_count, pending_copy_event are only ever read or
    // written under RAMSafetyManager::ram_mutex. slab_buffer/slab_capacity/
    // gate_proj/up_proj/down_proj are the exclusive claim-owner's PRIVATE
    // territory from the moment claim_or_wait returns Claimed until that same
    // owner publishes (ram_loading=false) -- touched WITHOUT the lock, and no
    // other thread may read or write them meanwhile (a slot can only ever have
    // one claim-owner at a time, enforced by the Empty->Loading transition being
    // the sole mutex-protected decision point). Do not add a diagnostic or scan
    // that reads slab_capacity/slab_buffer without checking Ready first -- that
    // would violate this invariant even if it "only reads".
    //
    // Included in the SAME "never pick this as an eviction victim, score
    // UINT64_MAX" guard as copy_still_pending()/ram_temp_pin_count>0 at all 3
    // RAM-tier victim-selection sites (get_expert Step 3, get_experts_batch Phase
    // A miss-claim, demote_from_vram) -- a slot mid-claim must never be reused
    // for a different expert out from under its owner.
    bool ram_loading = false;

    // True when this slot's bytes were populated by a LockFreePrefetcher worker
    // thread's own claim (RamClaimCaller::PrefetchWorker) rather than the decode
    // thread's own miss-handling claim. Renamed from the pre-unification
    // from_pipe_prefetch (which meant "served via the old separate staging
    // pool's consume_staged") -- under the unified claim/publish design there is
    // no separate staging pool, so this now just means "who populated this
    // slot", used for two things: (1) n_prefetch_staged_later_promoted (perf_timer.
    // hpp) still counts how many prefetch-loaded slots later cross
    // promote_hit_threshold and get promoted to VRAM via the normal hit-count-
    // gated path, proving prefetched experts still reach VRAM only once a real
    // access confirms it; (2) the new n_ram_prefetch_loaded_evicted_unused
    // signal (a RAM-tier victim-selection site reclaiming a slot with
    // loaded_by_prefetch==true and hit_count==0 means prefetch loaded it and it
    // was evicted before any real hit ever touched it -- the unified design's
    // replacement for the old staging-pool-based "wasted prefetch" measurement,
    // now grounded in the real cache's real LRU pressure instead of an
    // artificial round-robin pool). Reset to false whenever a slot is claimed
    // for a different expert, same lifecycle as also_in_vram.
    bool loaded_by_prefetch = false;

    // 2026-08-30 diagnostic: true once this slot's slab_buffer bytes have
    // been handed to a VRAM upload (active-layer pre-warm, active-layer
    // speculative promotion, or a main-cache RAM->VRAM promotion) WITHOUT
    // that read going through claim_or_wait's own Hit path -- i.e. the data
    // was genuinely CONSUMED, but the RAM tier's own hit_count never
    // recorded it, because the eventual dispatch resolved out of VRAM
    // instead of ever consulting this RAM slot again.
    //
    // Exists solely to split n_ram_prefetch_loaded_evicted_unused (the
    // "Unified RAM waste" line, reporting 86.6% on a real 100-token
    // prefill) into its two structurally different causes, which call for
    // completely different fixes and must not be conflated:
    //   (a) BENIGN -- consumed_by_vram_upload==true: the prefetch DID pay
    //       off, the bytes reached VRAM and were used from there; only the
    //       RAM-tier hit_count is blind to it. Nothing to fix.
    //   (b) REAL WASTE -- consumed_by_vram_upload==false: the slot was
    //       fetched, never read by anyone, and reclaimed under capacity
    //       pressure. This is the number that would justify the
    //       prefill-sliding-window capacity work (per-layer cap is 76 while
    //       a 100-token prefill layer needs ~212 unique experts).
    // Same reset lifecycle as also_in_vram/loaded_by_prefetch (cleared
    // whenever the slot is claimed for a different expert).
    bool consumed_by_vram_upload = false;

    // 2026-08-31: how many times THIS REQUEST's prefill router selected this
    // expert (0 = not selected, or not stamped yet). Copied from
    // LayerScratch::expert_freq -- Phase 1's real, already-computed
    // whole-batch router output, not a prediction.
    //
    // WHY this is worth a field: measured on real 5-prompt data, ranking the
    // RAM tier by THIS REQUEST's own prefill counts captures 64.1% of the
    // following decode phase's traffic, against 67.8% for an oracle that
    // already knows decode's answer -- a ratio of 0.946, holding at both
    // 2850 and 5700 slots. Prefill's expert set is also a strict SUPERSET
    // relationship in the right direction (0 prefill-exclusive entries;
    // prefill's set covers 95.7% of decode traffic, r=0.921). So "what
    // prefill just used" is very nearly the best possible answer to "what
    // should survive into decode", and it costs nothing to obtain.
    //
    // eviction_score() folds this in (gated by prefill_retention_weight, 0 =
    // off) so a heavily-used prefill expert outranks a merely-recently-
    // touched one at eviction time. Same reset lifecycle as also_in_vram /
    // loaded_by_prefetch: cleared whenever the slot is claimed for a
    // different expert, so it can never describe the wrong expert.
    uint32_t prefill_demand = 0;

    // True when this slot was claimed by EngineConfig::prefill_dispatch_
    // lookahead_tokens (prefill-only, additive get_experts_batch widening --
    // see that field's own comment, engine_config.hpp) rather than by the
    // token whose real, immediate need triggered the call. Deliberately a
    // SEPARATE flag from loaded_by_prefetch above, not a shared one: that
    // flag is set purely from RamClaimCaller (who==PrefetchWorker), and
    // lookahead widening always goes through RamClaimCaller::MainThread (it's
    // called from the SAME get_experts_batch call as the triggering token's
    // own real needs, just for a different, later token's picks) -- so
    // reusing loaded_by_prefetch would have made lookahead-loaded slots
    // invisible to the "Unified RAM waste" design's own who-based gate, and
    // this project's own "was_it_used" question for this specific mechanism
    // needs its own answer, not a borrowed one. See RAMSafetyManager::
    // claim_or_wait's own n_prefill_lookahead_first_hit comment (the "used"
    // signal) and the eviction-victim-selection sites' n_prefill_lookahead_
    // loaded_evicted_unused comment (the "wasted" signal) for the two halves
    // of this measurement. Reset to false whenever a slot is claimed for a
    // different expert, same lifecycle as also_in_vram/loaded_by_prefetch.
    bool loaded_by_lookahead = false;

    // True when this slot was claimed by Phase B's speculative disk-batch
    // padding (RAMSafetyManager::get_experts_batch, disk_batch_padding
    // config -- a route_freq-ranked, historically hot but not currently
    // resident expert, loaded to pad queue depth on an already-necessary
    // io_uring submission, not because THIS token actually routed to it).
    // One-shot: cleared the first time this slot is genuinely hit (n_pad_
    // reads_hit_later, perf_timer.hpp) -- deliberately NOT persistent like
    // loaded_by_prefetch above (which tracks promotion eligibility, not a
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
    // Standalone "shared" MTP/NextN checkpoint (config.mtp_model_path),
    // Qwen4Exp's own convention -- see EngineConfig::mtp_model_path's own
    // comment. Only its blk.<n_layers>.* tensors are read (resident-loaded
    // whole, via initialize_resident_memory's explicit-source overload --
    // see graph_initializer.cpp's qwen4exp_nextn_manifest); token_embd/
    // output are deliberately absent from this file and resolved from
    // model_file instead wherever the NextN forward pass needs them.
    GGUFFile mtp_model_file;
    bool has_mtp_model = false;

    // Resolved ONCE from ModelFamily (detect_model_family(model_file)) in
    // graph_initializer.cpp, right after architecture_config's other fields
    // are populated -- the single per-family dispatch table layers_forward_
    // gguf (execution_layer.cpp) and the cache-allocation code below read
    // from instead of the old scattered is_mla-bool/arch-string-literal
    // checks. See family_ops.hpp for the full rationale.
    FamilyOps family_ops;

    std::unordered_map<std::string, QuantizedTensor> resident_layers;
    std::vector<std::vector<MoEExpertBlock>> layers_experts;

    // A graph-compatible host representation of a fully resident MoE layer.
    // The legacy layout is one fixed-stride [gate|up|down] slice per expert.
    // Qwen's CPU graph uses the projection-compact layout instead: all gate
    // slices, all up slices, and all down slices are separate fixed-stride
    // planes. This lets CPU MUL_MAT_ID consume compact projection views
    // without allocating a second, per-graph copy after startup.
    // The graph only aliases an arena after every expert in its layer is
    // Ready; partial layers remain on the graph's complete source fallback.
    struct GraphRamLayerArena {
        uint8_t *base = nullptr;
        // Usually equal to base.  The optional diagnostic host offset keeps
        // the allocation pointer separately because cudaHostRegister/free
        // must receive the original allocation, not an interior pointer.
        void *allocation_base = nullptr;
        size_t allocation_bytes = 0;
        size_t bytes = 0;
        // Legacy interleaved expert stride. Zero for projection_compact.
        size_t stride = 0;
        size_t gate_bytes = 0;
        size_t up_bytes = 0;
        size_t down_bytes = 0;
        size_t gate_offset = 0;
        size_t up_offset = 0;
        size_t down_offset = 0;
        size_t gate_stride = 0;
        size_t up_stride = 0;
        size_t down_stride = 0;
        int n_experts = 0;
        bool projection_compact = false;
        bool pinned = false;
        bool mlocked = false;
        size_t host_reservation_bytes = 0;
        // Shared graph arenas are immutable after the startup preload. Cache
        // the complete-slot validation result so decode does not walk all
        // experts again at every layer/token boundary.
        mutable bool ready = false;
    };
    std::vector<GraphRamLayerArena> graph_ram_arenas;
    bool graph_ram_cache_enabled = false;
    // Qwen-only source contract copied from EngineConfig at startup. When
    // true, Qwen host-resident expert graph/dispatch code must use the
    // preloaded DDR tier and must not silently fall back to a GGUF mmap read.
    // Full-graph device-resident layers may use transient GGUF staging.
    // Other model families do not consult this flag.
    bool qwen_ram_preload_all_experts = false;
    std::vector<uint8_t> qwen_ram_vram_only_layers;

    // Per-expert VRAM split budget for full-graph decode (EngineConfig::
    // qwen4exp_graph_expert_vram_budget_gb), copied at startup exactly like
    // qwen_ram_preload_all_experts above. 0.0 = feature off. See
    // ggml_qwen4exp_graph.cpp's plan_expert_vram_split for the consumer.
    double qwen4exp_graph_expert_vram_budget_gb = 0.0;
    // Usage-histogram file the split planner ranks experts from -- defaults
    // to the same pin_file the startup [PIN] preload block already reads,
    // copied here since the graph builder has no other route to
    // EngineConfig (it only receives StreamingEngine&).
    std::string qwen4exp_graph_pin_file;
    std::string exl3_ggml_bridge_model_path;
    int exl3_ggml_bridge_layer = -1;
    std::vector<int> exl3_ggml_bridge_layers;
    bool exl3_ggml_graph_capture = false;
    bool exl3_ggml_routed_graph_capture = false;
    std::shared_ptr<Exl3GpuExecutor> exl3_gpu_executor_owner;
    std::shared_ptr<Exl3CpuMatrixCache> exl3_cpu_matrix_cache;
    Exl3GpuExecutor *exl3_gpu_executor = nullptr;

    // Option B: static hot/cold expert split. The full graph places each
    // selected layer's top-N usage-ranked experts in its fixed CUDA arena and
    // routes complementary experts through the fixed DDR/CPU branch. Unlike
    // qwen4exp_partial_layer, this is not a GPU-only fast path and therefore
    // has no silent dropped-route risk.
    int qwen4exp_option_b_layer = -1;
    int qwen4exp_option_b_top_n = 0;
    float qwen4exp_option_b_min_coverage = 0.80f;
    bool qwen4exp_option_b_right_size_top_n = false;
    int qwen4exp_option_b_global_top_k = 0;
    int prefill_batch_tokens = 0; // EngineConfig::prefill_batch_tokens (0 = family default)
    int qwen4exp_option_b_global_floor_per_layer = 1;
    // Finalized selected layers after startup usage-file filtering.  This is
    // distinct from the config allow-list, which remains available to main's
    // preload/coverage selection pass.
    int qwen4exp_option_b_max_layers = 0;
    bool qwen4exp_option_b_fill_vram = false;
    bool qwen4exp_option_b_concentrated_only = false;
    int qwen4exp_option_b_stage_expert = -1;
    int qwen4exp_option_b_stage_count = 1;
    std::vector<uint8_t> qwen4exp_option_b_hot_mask;
    std::vector<int> qwen4exp_option_b_layers;
    std::unordered_map<int, double> qwen4exp_option_b_layer_coverage;
    std::unordered_map<int, std::vector<uint8_t>>
        qwen4exp_option_b_hot_masks;
    bool qwen4exp_option_b_committed = false;

    // CPUOffload branch Milestone A. This is independent of Option B's
    // planner: it only controls whether the explicit mixed hot-GPU/
    // cold-DDR CPU seam is eligible for a decode layer.
    bool qwen4exp_cpu_offload = false;
    std::vector<int> qwen4exp_cpu_offload_layers;
    int qwen4exp_cpu_offload_force_cold_count = 0;
    int qwen4exp_cpu_offload_force_expert = -1;
    bool qwen4exp_cpu_offload_event_fence = false;
    int qwen4exp_cpu_offload_ring_slots = 2;

    // GPU staging for host-resident decode layers -- see
    // EngineConfig::qwen4exp_gpu_stage_enabled's own comment for the design.
    // qwen4exp_gpu_stage_layers must stay sorted (main.cpp sorts+dedupes it
    // at startup, same convention as qwen4exp_cpu_offload_layers) so the
    // binary_search below is valid.
    bool qwen4exp_gpu_stage_enabled = false;
    std::vector<int> qwen4exp_gpu_stage_layers;

    // Decode-only router-usage tap: batches every layer's real routed
    // `ids` into one on-device accumulator per token (ordinary ggml_concat
    // nodes, no host bridge), then ONE D2H + ONE host callback per token
    // bumps route_freq for all layers at once -- see
    // Qwen4ExpUsageTrackContext (ggml_qwen4exp_graph.cpp) for why this
    // beats a per-layer host-node tap. Off by default: real per-token
    // overhead, only worth paying during a usage survey.
    bool qwen4exp_track_router_usage = false;

    bool qwen4exp_option_b_layer_enabled(int layer) const {
      if (!qwen4exp_option_b_committed)
        return false;
      if (!qwen4exp_option_b_layers.empty())
        return std::binary_search(qwen4exp_option_b_layers.begin(),
                                  qwen4exp_option_b_layers.end(), layer);
      return layer == qwen4exp_option_b_layer;
    }

    bool qwen4exp_option_b_hot_expert(int layer, int expert) const {
      if (!qwen4exp_option_b_layer_enabled(layer) || expert < 0)
        return false;
      const auto it = qwen4exp_option_b_hot_masks.find(layer);
      if (it != qwen4exp_option_b_hot_masks.end())
        return static_cast<size_t>(expert) < it->second.size() &&
               it->second[static_cast<size_t>(expert)] != 0;
      return layer == qwen4exp_option_b_layer &&
               static_cast<size_t>(expert) < qwen4exp_option_b_hot_mask.size() &&
             qwen4exp_option_b_hot_mask[static_cast<size_t>(expert)] != 0;
    }

    bool qwen4exp_cpu_offload_layer_enabled(int layer) const {
      if (!qwen4exp_cpu_offload || layer < 0)
        return false;
      return std::binary_search(qwen4exp_cpu_offload_layers.begin(),
                                qwen4exp_cpu_offload_layers.end(), layer);
    }

    bool qwen4exp_gpu_stage_layer_enabled(int layer) const {
      if (!qwen4exp_gpu_stage_enabled || layer < 0)
        return false;
      return std::binary_search(qwen4exp_gpu_stage_layers.begin(),
                                qwen4exp_gpu_stage_layers.end(), layer);
    }

    // Phase 1 "all-or-nothing" partial-layer GPU dispatch (per-session, not
    // per-token -- see qwen4exp-full-graph-decode-benchmark-baseline memory's
    // follow-on note for why per-token was ruled out: ggml has no in-graph
    // conditional execution, so a genuine per-token GPU-or-CPU choice needs
    // splitting the decode graph at this layer's boundary, which Phase 1
    // deliberately defers). Engine-owned (not FullGraph-owned) because
    // decode_prepare tears down and rebuilds a FRESH FullGraph object after
    // prefill -- this decision must survive that swap.
    int qwen4exp_partial_layer = -1;          // -1 = feature off
    int qwen4exp_partial_layer_top_n = 256;
    float qwen4exp_partial_layer_min_coverage = 1.0f;
    std::vector<int> qwen4exp_partial_layer_gpu_ids; // top-N experts, ranked by usage
    bool qwen4exp_partial_layer_planned = false;      // gpu_ids computed at least once
    bool qwen4exp_partial_layer_committed = false;    // this session: use the GPU-only path

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

    // Raw/SWA MLA KV cache (DeepSeek-V4, CacheShape::MLARawSWA): per-layer
    // ring buffer [sliding_window * n_embd_head], NOT [max_t * ...] --
    // sized by the sliding window, not the full context, which is the
    // actual memory win SWA is for. Ring index is `pos % sliding_window`,
    // applied consistently on every read AND write (dsv4_attention_gguf,
    // attention_layer.cpp) -- unlike Lc/Rc (CacheShape::MLAAbsorbed), there
    // is no separate rope-half cache here: DeepSeek-V4's wkv already
    // produces one per-head-sized latent with the rope slice embedded, so
    // one buffer per layer is enough. sliding_window mirrors
    // ArchitectureConfig::sliding_window (kept on the engine too so the
    // attention function's ring-index math doesn't need to thread
    // ArchitectureConfig through every call for just this one int).
    std::vector<float*> KVc;
    int sliding_window = 0;

    // Per-layer DeepSeek-V4 CSA/HCA compressed-attention state (compressed
    // K cache + ring-buffer compressor state, dsv4_compressed_attention.hpp)
    // -- one entry per layer, sized/configured by alloc_dsv4_swa_cache
    // (models/family_ops.cpp) at load time. A layer with compress_ratio==0
    // (raw SWA, or any non-DeepSeek-V4 architecture) has its entry left
    // default-constructed (Dsv4CompressedLayerState::enabled() == false),
    // never resized/touched -- dsv4_attention_gguf (attention_layer.cpp)
    // checks `enabled()` before using an entry, so an unconfigured slot is
    // inert rather than undefined.
    std::vector<Dsv4CompressedLayerState> dsv4_compressed;

    // DeepSeek-V4.1 shared compressed-attention state. Only source layers
    // own allocated entries; the source maps below point consumers at them.
    std::vector<Dsv41CompressedLayerState> dsv41_compressed;
    std::vector<int> dsv41_kv_source_for_layer;
    std::vector<int> dsv41_index_source_for_layer;
    std::vector<std::vector<int>> dsv41_candidate_positions;
    std::vector<int32_t> dsv41_engram_tok_hist;
    int64_t dsv41_engram_next_pos = 0;
    // Hash ids are shared by every Engram layer in one forward call. Cache
    // the batch so a later Engram layer does not reset/replay the token
    // history that the first layer already advanced.
    std::vector<int64_t> dsv41_engram_hashes;
    int dsv41_engram_hash_pos_base = -1;
    int dsv41_engram_hash_batch = 0;

    // GLM5Next (GLM 5.3 Flash) KDA recurrent-layer state -- one entry per
    // layer, sized/populated by alloc_glm5next_cache (models/family_ops.cpp)
    // at load time. DSA (full-attention) layers get an EMPTY entry and never
    // touch these; KDA layers (glm5next_is_kda_layer) own:
    //   conv_history[l]: [3*d_inner, ssm_conv_kernel-1] -- the last
    //     (kernel-1) pre-conv q|k|v samples per channel, carried across
    //     calls so a decode-time S=1 call still sees its real receptive
    //     field (same contract as cpu_qwen_moe.cpp's conv_history).
    //   recur_state[l]: [n_heads, kda_head_dim, kda_head_dim] -- the
    //     gated-delta-rule recurrent state, [key, value] per head.
    // Both persist for the whole session and are mutated in place by
    // glm5next_kda_gguf (attention_layer.cpp). Sized once at allocation;
    // a layer whose entry is the wrong size is re-initialized (zeroed) by
    // the attention function, matching cpu_qwen_moe.cpp's discipline.
    std::vector<std::vector<float>> glm5next_conv_history;
    std::vector<std::vector<float>> glm5next_recur_state;

    // Qwen4Exp (Qwen3.8-Flash-Next) recurrent-layer + PLE state -- one entry
    // per layer, sized/populated by alloc_qwen4exp_cache (models/
    // family_ops.cpp) at load time. Full-attention layers get an EMPTY entry
    // and never touch these; GDN layers (qwen4exp_is_gdn_layer) own:
    //   qwen4exp_conv_history[l]: [conv_dim, ssm_conv_kernel-1] -- the last
    //     (kernel-1) pre-conv q|k|v samples per channel (conv_dim =
    //     2*key_dim + value_dim from the fused attn_qkv projection), carried
    //     across calls (same contract as glm5next_conv_history above).
    //   qwen4exp_recur_state[l]: [n_v_heads, head_dim, head_dim] -- the
    //     gated-delta-rule recurrent state, [key, value] per value head
    //     (key heads are shared: v-head h reads k-head h % group_count,
    //     matching the converter's tiled V-head reorder --
    //     conversion/qwen.py's _LinearAttentionVReorderBase).
    // PLE n-gram layers (qwen4exp_is_ple_layer) additionally own:
    //   qwen4exp_ple_conv_history[l]: [hc_mult*hidden, hist] where hist =
    //     (ple_conv_kernel-1)*ple_ngram_size -- the dilated causal conv's
    //     per-layer input history (qwen4exp.cpp's build_ple /
    //     build_conv_state_at on the PLE conv's own recurrent row).
    //   qwen4exp_ple_tok_hist: last (ple_ngram_size-1) token ids of the
    //     (single) sequence + the next expected position -- the PLE hash's
    //     n-gram window across decode steps (llama-memory-hybrid-idx's
    //     ple_hist equivalent; single-sequence only, matching this engine's
    //     single-session generation loop).
    // All persist for the whole session and are mutated in place.
    std::vector<std::vector<float>> qwen4exp_conv_history;
    std::vector<std::vector<float>> qwen4exp_recur_state;
    std::vector<std::vector<float>> qwen4exp_ple_conv_history;
    std::vector<int32_t> qwen4exp_ple_tok_hist;
    int64_t qwen4exp_ple_next_pos = 0;

    // Qwen4Exp low-rank HC: the inject vector ([S * hc_mult]) computed by
    // qwen4exp_hc_pre_mix and consumed by qwen4exp_hc_post_combine within
    // the same layer/sub-block bracket (hyper_connections.cpp). Sized per
    // call; reused across layers (pre/post are strictly sequential per
    // sub-block, so one buffer is enough).
    std::vector<float> qwen4exp_hc_inject;

    // Optional Qwen4Exp device handoff for one layer's attention bracket.
    // GDN outputs are GGML tensors; dense full-attention outputs are raw F32
    // CUDA buffers owned by CudaManager. Both are consumed before the next
    // layer advances.
    const void *qwen4exp_device_hc_input = nullptr;
    const void *qwen4exp_device_hc_ffn_input = nullptr;
    const void *qwen4exp_device_attn_output = nullptr;
    bool qwen4exp_device_attn_output_ready = false;
    bool qwen4exp_device_attn_output_raw = false;

    // Hyper-Connections multi-stream residual (DeepSeek-V4, hc_mult > 1
    // only -- empty/unused for every other family, per family_ops.hpp's own
    // "hc_mult==1 is the degenerate case" design). Layout is PRIVATE to
    // hyper_connections.cpp -- nothing outside that file indexes into this
    // buffer directly (layers_forward_gguf and every attention/FFN function
    // still only ever see the plain, single-stream [S,D] hidden_states/
    // scratch buffers). Sized [S * hc_mult * D] floats, resized once per
    // layers_forward_gguf call (S can change between prefill and decode).
    std::vector<float> hc_stream_buf;
    // Reference HC carry: previous FFN pre-mix feeds the next layer's
    // attention, and the final FFN pre-mix collapses V4.1 output.
    std::vector<float> hc_pending_pre_mix;
    std::vector<float> hc_attn_pre_mix;
    std::vector<float> hc_final_pre_mix;

    // Set by the Qwen4Exp full GGML prefill graph after it has
    // produced the final model-level HC output directly into the caller's
    // [S,D] buffer.  The generation loop must not run the legacy HC collapse
    // a second time for that result.
    bool qwen4exp_full_graph_output_ready = false;
    // Set after model-family detection when Qwen full-graph decode is the
    // effective policy. A graph failure is surfaced to the request loop;
    // execution_layer must not continue into the layered token loop.
    bool qwen4exp_full_graph_required = false;
    std::string qwen4exp_full_graph_failure;
    // The Qwen full decode graph may also produce the next-token logits in
    // the same one-submission graph.  The generation loop consumes this once
    // on the following sampling iteration to avoid a separate lm-head H2D/D2H
    // dispatch; false for the layered path and every other model family.
    bool qwen4exp_full_graph_logits_ready = false;
    // Short-lived destination installed by the generation loop while it
    // calls layers_forward_gguf(). The Qwen decode graph writes logits here
    // without retaining a pointer into a caller-local vector.
    float *qwen4exp_full_graph_logits_target = nullptr;
    int qwen4exp_full_graph_logits_target_size = 0;

    // Set by the opt-in GLM5NEXT full GGML prefill graph.  This is separate
    // from the Qwen flag so the shared generation loop cannot apply the
    // legacy HC collapse twice.
    bool glm5next_full_graph_output_ready = false;

    // Set by the opt-in DeepSeek-V4 full GGML prefill graph after it has
    // produced the learned HC collapse and published raw/compressed state
    // for the following decode step.
    bool deepseek_v4_full_graph_output_ready = false;

    // Post-gate/comb-matrix scratch: computed once by hc_pre_collapse
    // (hyper_connections.cpp) and consumed by the LATER hc_post_combine
    // call for the SAME layer/sub_block, after the sub-block itself has run
    // in between -- these two calls aren't adjacent, so the values can't
    // just be locals. Sized [S*hc_mult] / [S*hc_mult*hc_mult]; resized
    // (harmlessly re-resized to the same size) on every hc_pre_collapse
    // call.
    std::vector<float> hc_post_gate_scratch;
    std::vector<float> hc_comb_scratch;

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

    // Process-wide host-memory admission accounting for allocations owned by
    // the engine itself (resident dense tensors, KV caches, and recurrent
    // state). RAM slabs and graph/MTP allocations keep their own lifetimes but
    // charge the same MemoryGuard ledger.
    size_t host_memory_reservation_bytes = 0;
    bool host_memory_allocation_failed = false;

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

    // Per-expert MIXED-PRECISION support, Phase 1 (file-level correctness
    // only -- see CLAUDE.md's per-expert-mixed-precision design writeup for
    // the full phased plan). Populated by sanity_check_unpacked_format(),
    // indexed by layer_idx: true means this layer's experts do NOT all
    // share the same gate/up/down byte size (i.e. genuinely different quant
    // types coexist within the layer), false means they're uniform (the
    // ONLY case expert_layer_geometry's single-representative-expert
    // shortcut, and everything downstream of it -- compute_expert_slab_
    // bytes, RAM/VRAM slot sizing, disk-read batch planning -- is safe to
    // use as-is; none of that has been updated yet to reason per-expert).
    // Empty until sanity_check_unpacked_format() runs.
    std::vector<bool> layer_has_mixed_expert_sizes;

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

    // Set from EngineConfig::ram_pinned_budget_gb at startup (main.cpp);
    // 0 = feature off (every slab stays plain pageable memory, exactly the
    // pre-2026-08-15 behavior). See ensure_slab_capacity for the full
    // reasoning and the real-hardware numbers this exists to capture.
    // ram_pinned_bytes_used is ATOMIC because ensure_slab_capacity is
    // genuinely multi-threaded -- it's reached from LockFreePrefetcher's
    // worker threads via build_expert_read_requests, not just the decode
    // thread (confirmed by grep, not assumed).
    size_t ram_pinned_budget_bytes = 0;
    std::atomic<size_t> ram_pinned_bytes_used{0};

    // Set from EngineConfig::prefill_opportunistic_promote at startup
    // (main.cpp). Only meaningful when prefill_warm_via_prefetcher is also
    // true -- gates LayerScratch::pending_prefill_promotions' cross-layer
    // follow-up (execution_layer.cpp) separately from the hint-pushing
    // itself.
    //
    // REAL A/B, CONCLUSIVE, DEFAULT FLIPPED TO OFF (2026-08-12): four
    // consecutive runs with this ON all landed at prefill=0.56 tok/s; one
    // run with it OFF landed at 0.60, matching the pre-feature baseline
    // (0.59) almost exactly. Root cause confirmed, not just correlated:
    // hint volume (n_prefill_hint_pushed) nearly HALVED when this was ON
    // (420 vs ~930) -- this pass and Phase 2's own per-layer hint-pushing
    // loop compete for the SAME resource (vram_cache->has_free_slot()).
    // Running the opportunistic follow-up FIRST in every Phase 2 call
    // consumes free slots faster than Phase 2's own loop would have on its
    // own, so `has_free_slot()` goes false sooner -- meaning LATER layers'
    // own candidates stop getting hints pushed at all, losing the RAM-
    // prefetch overlap prefill_warm_via_prefetcher exists to provide, and
    // falling back to Phase 3's fully-synchronous path instead. The
    // mechanism itself is safe (every candidate is a confirmed real router
    // pick, never a wrong speculative bet) -- the bug is resource
    // contention with the very feature it was meant to complement, not
    // promotion accuracy. Left implemented (not deleted) rather than
    // ripped out, per this project's own convention for a well-evidenced,
    // cleanly-reasoned negative result -- see CLAUDE.md.
    bool prefill_opportunistic_promote = false;

    // Set from EngineConfig::decode_async_prewarm/decode_async_prewarm_
    // max_batch at startup (main.cpp). Decode-time analogue of the prefill
    // pre-warm above -- see EngineConfig::decode_async_prewarm's own
    // extensive comment (engine_config.hpp) for the real incident that made
    // this default OFF and capped, rather than a straight reinstatement of
    // the first (unbounded) attempt.
    bool decode_async_prewarm = false;
    int decode_async_prewarm_max_batch = 3;
    // Set from EngineConfig::decode_async_prewarm_nonblocking -- see its own
    // comment for the mechanism (find_active_expert poll instead of
    // sync_active_layer_batch's blocking cudaStreamSynchronize).
    bool decode_async_prewarm_nonblocking = false;

    // Opt-in GLM5Next MTP speculative decoding.  The target trunk remains
    // authoritative; the extra NextN block proposes tokens and the target
    // verifies them in bounded batches.  Kept on StreamingEngine (rather
    // than a process-global switch) so a future multi-model caller cannot
    // accidentally enable MTP for Qwen/DeepSeek.
    bool mtp_enabled = false;
    int mtp_draft_tokens = 3;

    // Set from EngineConfig::cpu_offload_max_experts at startup (main.cpp).
    // 0 = off (default). See that field's own extensive comment
    // (engine_config.hpp) for the Fiddler-style rationale and the real
    // per-layer miss data this was sized against.
    int cpu_offload_max_experts = 0;
    // Copied from EngineConfig at startup (main.cpp). Halve retention stamps
    // at the end of each request -- see RAMSafetyManager::decay_prefill_demand.
    bool prefill_retention_decay = true;

    // Set from EngineConfig::prefill_dispatch_lookahead_tokens at startup
    // (main.cpp). 0 = off (default is 8 in EngineConfig itself). See that
    // field's own extensive comment (engine_config.hpp) for the mechanism
    // and the real "phase_b dominates even with prefill_warm_via_prefetcher
    // on" finding this responds to.
    int prefill_dispatch_lookahead_tokens = 0;

    // Set from EngineConfig::prefill_hint_order_by_earliest_token at
    // startup (main.cpp). false = off (default, exact prior expert_freq-
    // descending order). See that field's own comment for the mechanism.
    bool prefill_hint_order_by_earliest_token = false;

    // Set from EngineConfig::pilot_widen_all_tokens at startup (main.cpp).
    // false = off (default). See that field's own comment for the
    // mechanism -- computes the widened prediction + drives the
    // [PILOT-WIDE] measurement; prefetch_widen_all_tokens below decides
    // whether real hints get pushed from it.
    bool pilot_widen_all_tokens = false;

    // Set from EngineConfig::prefetch_widen_all_tokens at startup
    // (main.cpp). false = off (default). See that field's own comment --
    // real disk-prefetch hint push from pilot_widen_all_tokens's widened
    // prediction, hint-push only (no VRAM promotion).
    bool prefetch_widen_all_tokens = false;

    // Set from EngineConfig::attn_qabsorb_gpu/attn_qabsorb_gpu_max_vram_gb
    // at startup (main.cpp) -- gates the MLA absorbed-attention qabsorb
    // BF16 batched GPU GEMM path (attention_layer.cpp / cuda_manager.hpp).
    // See EngineConfig::attn_qabsorb_gpu's own extensive comment
    // (engine_config.hpp) for why this defaults off and is deliberately
    // per-model/hardware configurable rather than a blanket default.
    bool attn_qabsorb_gpu = false;
    double attn_qabsorb_gpu_max_vram_gb = 2.0;

    // Set from EngineConfig::attn_score_gpu/attn_score_gpu_max_vram_gb at
    // startup (main.cpp) -- gates the MLA score (QK^T-over-latent + softmax
    // + weighted-V) GPU path (attention_layer.cpp / cuda_manager.hpp). See
    // EngineConfig::attn_score_gpu's own extensive comment for why this
    // defaults off and is deliberately per-model/hardware configurable --
    // same rationale as attn_qabsorb_gpu, plus: unlike qabsorb, DeepSeek-V4
    // bounds this phase's cost with a sliding window (dsv4_attention_gguf),
    // so it's structurally cheaper there than GLM-5.2's unbounded version
    // this GPU path targets.
    bool attn_score_gpu = false;
    // Set from EngineConfig::qwen4exp_gdn_conv_gpu.
    bool qwen4exp_gdn_conv_gpu = false;
    // Set from EngineConfig::qwen4exp_router_pilot. Qwen-only HC stream-0
    // prediction; conventional GLM/DeepSeek pilot inputs do not use this.
    bool qwen4exp_router_pilot = false;
    int qwen4exp_router_pilot_k = 10;
    // Qwen pilot prefetch action. RAM-resident predictions are copied into
    // the next layer's temporary active-layer area asynchronously; non-RAM
    // predictions use the ordinary disk/RAM hint path.
    bool qwen4exp_router_pilot_prefetch = false;
    // 0 means qwen4exp_router_pilot_k. A performance profile can set both
    // values to 5, while a diagnostic/usage-collection profile can set
    // pilot_k=10 and prefetch_k=5 to retain full top-10 rank reporting.
    int qwen4exp_router_pilot_prefetch_k = 0;
    // Layer indices excluded from Qwen estimation-based prefetch. Prediction
    // and accuracy collection remain enabled for excluded layers.
    std::vector<int> qwen4exp_router_pilot_prefetch_exclude_layers;
    // Optional Qwen-only prediction allow-list. Empty means every target
    // layer not otherwise skipped by the Qwen pilot policy.
    std::vector<int> qwen4exp_router_pilot_predict_layers;
    // Decode-only cache-aware pilot bypass; Qwen-only, other families ignore it.
    bool qwen4exp_router_pilot_skip_fully_cached = false;
    bool qwen4exp_router_pilot_skip_prefetch_excluded_layers = false;
    // Qwen4Exp-only decode opt-in: run the routed FFN through the grouped
    // GGML/CUDA MUL_MAT_ID graph for S==1. Other families never read this
    // switch; false preserves the established per-expert decode path.
    bool qwen4exp_decode_ggml_moe = false;
    // Qwen-only full-graph CUDA-router + CPU-MUL_MAT_ID scheduler topology.
    // It is used only for complete immutable DDR arenas; lazy layers fall
    // back to the normal cache/dispatch path.
    bool qwen4exp_decode_scheduler = false;
    // Qwen-only S==1 HC -> router/MoE device handoff. Kept separate from
    // the grouped decode switch for controlled correctness/performance A/B.
    bool qwen4exp_decode_device_hc = false;
    // Qwen-only S==1 RAM->VRAM batch staging. Other model families never
    // consult this switch; false preserves the direct pinned-slab path.
    bool qwen4exp_decode_batch_stage = false;
    bool qwen4exp_decode_custom_hc_q8 = false;
    // Keep the GPU router for batched Qwen prefill, but opt into it for the
    // S==1 decode boundary only when it wins on the target hardware. The
    // Qwen router pilot remains independently GPU-backed for prediction.
    bool qwen4exp_decode_gpu_router = false;
    // Qwen4Exp-only CPU affinity fast path: group IQ1_M expert projections
    // with the same ggml Q8_K dot kernel used by llama.cpp. It is consulted
    // only when cpu_affinity_layers selects a layer, so other models and the
    // normal GPU route are unchanged.
    bool qwen4exp_cpu_ggml_moe = false;
    // Qwen-only direct DDR-strided grouped CPU projections. Other model
    // families never consult this switch; false keeps the persistent GGML
    // CPU graph as the default.
    bool qwen4exp_cpu_strided_grouped = false;
    // Qwen-only one-team fused gate/up -> SwiGLU -> down projections. This is
    // consulted only by the direct DDR-strided CPU path.
    bool qwen4exp_cpu_fused_grouped = false;
    // Overlap the Qwen CPU routed graph with the GPU shared expert when
    // cpu_affinity_layers is active. Other model families never consult it.
    bool qwen4exp_cpu_gpu_overlap = false;
    int qwen4exp_cpu_gpu_overlap_max_layers = 16;
    bool qwen4exp_cpu_shared_device = false;

    // Set from EngineConfig::glm5next_kda_gpu. The GLM5NEXT KDA recurrent
    // update itself runs on CUDA; the host mirror is synchronized only at
    // explicit prompt-cache/checkpoint boundaries because the live state is
    // kept resident on the device between calls.
    bool glm5next_kda_gpu = false;
    // Set from EngineConfig::glm5next_router_gpu. The native CUDA route
    // returns only compact IDs/weights to the host cache machinery.
    bool glm5next_router_gpu = false;

    // GLM5NEXT-only grouped GPU prefill. This deliberately does not reuse the
    // family-agnostic moe_prefill_batched switch: GLM's mixed expert tensor
    // geometries made that generic path unsafe in earlier revisions. The
    // GLM-specific path has its own routing-consistency checks and fallback.
    bool glm5next_moe_prefill_gpu = false;

    // GLM5NEXT-only KDA dense projection graph fusion. Best effort: any
    // unavailable GGML/CUDA shape or allocation falls back to dense_proj.
    bool glm5next_dense_prefill_fusion = false;

    // Prompt (prefix) state cache -- EngineConfig::prompt_cache.
    //
    // Skips re-prefilling a prompt prefix that this process already processed.
    // Dense KV (Kc/Vc, and MLA's Lc/Rc) is POSITION-indexed and already
    // persists in place across requests, so it needs no snapshot -- what does
    // need snapshotting is every piece of state that is a ROLLING function of
    // the token stream rather than indexed by position:
    //   * qwen4exp_recur_state / glm5next_recur_state (the recurrent state)
    //   * qwen4exp_conv_history / glm5next_conv_history (short-conv history)
    //   * qwen4exp_ple_conv_history + ple_tok_hist + ple_next_pos (PLE n-gram)
    // Missing ANY of these would silently produce wrong output rather than
    // fail, which is why the feature is restricted to the two families whose
    // carried state is fully enumerated here (see prompt_cache_supported()).
    //
    // Only the END-of-prompt state is stored, so a hit requires the cached
    // token list to be a strict prefix of the new prompt -- a divergence
    // earlier than that would need state at the divergence point, which is
    // not retained. One entry: this targets repeated/extending prompts
    // (benchmark reruns, multi-turn continuation), not a general prefix tree.
    struct PromptStateCache {
        bool valid = false;
        std::vector<int> tokens;                      // tokens whose END state this is
        std::vector<std::vector<float>> recur_state;
        std::vector<std::vector<float>> conv_history;
        std::vector<std::vector<float>> ple_conv_history;
        std::vector<int32_t> ple_tok_hist;
        int64_t ple_next_pos = 0;
        std::vector<std::vector<float>> glm_recur_state;
        std::vector<std::vector<float>> glm_conv_history;
        // Post-prefill last hidden row (D floats). Lets an EXACT repeat skip
        // the forward pass altogether and reuse these logits, which makes that
        // path bit-identical to the run that populated it -- a stronger and
        // more checkable guarantee than comparing sampled text, which this
        // hardware's cache-state-dependent dispatch makes non-deterministic.
        std::vector<float> last_hidden;
    };
    PromptStateCache prompt_state_cache;
    bool prompt_cache = false;
    double attn_score_gpu_max_vram_gb = 2.0;
    // Set from EngineConfig::attn_score_gpu_max_context -- see that field's
    // own comment. 0 = no cap (mirror sized to the full context_size).
    int attn_score_gpu_max_context = 0;

    // Set from EngineConfig::dsv4_score_gpu/dsv4_score_gpu_max_vram_gb at
    // startup (main.cpp) -- gates DeepSeek-V4's raw-SWA/HCA score GPU path
    // (attention_layer.cpp / cuda_manager.hpp's dsv4_score_gpu_batched_
    // host). See EngineConfig::dsv4_score_gpu's own extensive comment for
    // why this defaults off (unverified on real hardware -- this Mac has no
    // CUDA toolchain to test with) and why CSA (ratio==4) layers never use
    // it regardless of this flag.
    bool dsv4_score_gpu = false;
    double dsv4_score_gpu_max_vram_gb = 1.0;

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
            // device_ptr is NEVER individually freed here -- the only real
            // assignment site in the whole codebase (CudaManager::
            // offload_tensor_to_pool) always sets it to `dense_pool +
            // aligned_used`, a suballocation of CudaManager's single
            // dense_pool cudaMalloc, never its own standalone allocation.
            // CudaManager's own destructor already frees the whole pool
            // exactly once. Calling cudaFree on an interior (non-base)
            // pointer here was a real, confirmed bug -- compute-sanitizer
            // flagged 861 cudaErrorInvalidValue ("invalid argument")
            // errors from this exact line on a real GLM-5.3 shutdown
            // (reproducible in ~17s via a no-prompt/--ngen 0 run, since
            // this fires purely from model load + teardown, no generation
            // needed) -- harmless in practice (cudaFree on an invalid
            // pointer is a no-op past logging the error, and dense_pool's
            // own later free still succeeds), but a real, avoidable,
            // sanitizer-visible bug all the same. Unlike device_ptr,
            // gpu_fast_ptr (below) is always its own standalone cudaMalloc
            // (see QuantizedTensor's own comment) -- freeing it here
            // individually stays correct.
            if (have_gpu && qt.gpu_fast_ptr) cudaFree(qt.gpu_fast_ptr);
        }
        for (auto& layer_slots : layers_experts) {
            for (auto& slot : layer_slots) {
                if (slot.slab_buffer && !slot.slab_from_shared_arena)
                    free(slot.slab_buffer);
                MemoryGuard::release_host_bytes(slot.host_reservation_bytes);
                slot.host_reservation_bytes = 0;
            }
        }
        // Shared graph arenas own the bytes referenced by their slots.  They
        // are released once per layer, after all slot aliases have stopped
        // being used, and their single CUDA registration is undone here.
        for (auto &arena : graph_ram_arenas) {
            if (!arena.base) continue;
            if (arena.pinned) {
                cudaHostUnregister(arena.allocation_base ? arena.allocation_base
                                                          : arena.base);
                ram_pinned_bytes_used.fetch_sub(arena.bytes,
                                                std::memory_order_relaxed);
            }
            if (arena.mlocked) {
                munlock(arena.allocation_base ? arena.allocation_base
                                               : arena.base,
                        arena.allocation_bytes ? arena.allocation_bytes
                                               : arena.bytes);
            }
            free(arena.allocation_base ? arena.allocation_base : arena.base);
            MemoryGuard::release_host_bytes(arena.host_reservation_bytes);
            arena = {};
        }
        graph_ram_arenas.clear();
        graph_ram_cache_enabled = false;
        // At this point run_engine has already stopped the prefetcher and GPU
        // topology, so deferred slab readers cannot remain in flight.  Free
        // and discharge any old realloc buffers that were parked for safety.
        {
            std::lock_guard<std::mutex> dg(deferred_slab_free_mutex);
            for (const auto& d : deferred_slab_frees) {
                if (d.ptr) free(d.ptr);
                MemoryGuard::release_host_bytes(d.reservation_bytes);
            }
            deferred_slab_frees.clear();
        }
        for (float* p : Lc) if (p) free(p);
        for (float* p : Rc) if (p) free(p);
        for (float* p : Kc) if (p) free(p);
        for (float* p : Vc) if (p) free(p);
        for (float* p : KVc) if (p) free(p);
        MemoryGuard::release_host_bytes(host_memory_reservation_bytes);
        host_memory_reservation_bytes = 0;

        if (have_gpu) {
            for (float* p : d_Lc) if (p) cudaFree(p);
            for (float* p : d_Rc) if (p) cudaFree(p);
            for (BlockQ4_KV* p : d_Lc_q4) if (p) cudaFree(p);
            for (BlockQ4_KV* p : d_Rc_q4) if (p) cudaFree(p);
            if (d_hidden) cudaFree(d_hidden);
        }
    }

    // `src_override`: read from this GGUFFile instead of the usual dense/
    // model_file choice -- used by qwen4exp_nextn_manifest (graph_
    // initializer.cpp) to resident-load a standalone MTP checkpoint
    // (mtp_model_file) without duplicating this whole function. Defaulted
    // to nullptr so the one pre-existing call site (dense_preload_manifest)
    // is byte-for-byte unchanged.
    void initialize_resident_memory(const std::vector<std::string>& dense_tensor_names,
                                    GGUFFile* src_override = nullptr) {
        PERF_START(dense_load);
        size_t total_loaded = 0;
        // Dense tensors load from the separate dense GGUF when one is configured
        // (dense_model_path); routed experts still stream from model_file. Both
        // files are complete GGUFs of the same model at different precisions.
        GGUFFile& src = src_override ? *src_override
                                     : (has_dense_model ? dense_model_file : model_file);
        for (const auto& name : dense_tensor_names) {
            if (src.tensors.find(name) == src.tensors.end()) continue;

            auto& t_meta = src.tensors[name];
            QuantizedTensor qt{t_meta, nullptr, nullptr};

            if (!reserve_host_memory(t_meta.size_bytes)) {
                LOG_ERROR("[MEM-GUARD] resident tensor %s refused: %.2f MB would exceed "
                          "the process host budget; continuing without this tensor\n",
                          name.c_str(), t_meta.size_bytes / 1e6);
                resident_layers[name] = qt;
                continue;
            }
            const int alloc_err = posix_memalign((void**)&qt.host_ptr, 4096,
                                                 t_meta.size_bytes);
            if (alloc_err != 0 || !qt.host_ptr) {
                release_host_memory(t_meta.size_bytes);
                LOG_ERROR("[RESIDENT] posix_memalign failed for %s (errno=%d)\n",
                          name.c_str(), alloc_err);
                resident_layers[name] = qt;
                continue;
            }
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
                release_host_memory(t_meta.size_bytes);
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
    // (prefetch) path so the two can never disagree on slab layout.
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

    // 2026-09: per-LAYER format detection, for mixed packed/unpacked models
    // (Phase 5 of per-expert mixed precision, see CLAUDE.md) -- a real
    // model can now have SOME layers converted to unpacked (per-expert
    // tensors, gguf_tool pack --unpack) while the rest stay in the default
    // packed format, rather than requiring the whole model to be one or
    // the other. unpacked_expert_format (the config flag) is no longer the
    // authority on a specific layer's real format -- it stays a real,
    // meaningful user-facing toggle (the ORIGINAL "every layer is unpacked"
    // expectation, still what most callers should assume when it's set),
    // but the 4 real call sites that make a per-layer packed-vs-unpacked
    // DISPATCH decision (expert_layer_geometry, expert_geometry_for,
    // build_expert_read_requests, and execution_layer.cpp's cuFile
    // eligibility block) now ask THIS function instead, so a layer that
    // genuinely has per-expert tensors is used correctly even if the flag
    // is off, and a layer that's still packed falls back correctly even if
    // the flag is on. Mirrors looks_like_unpacked_format's own per-layer
    // check (packed_slab_geometry succeeding means packed, regardless of
    // the flag; unpacked_expert_geometry succeeding for expert 0 means
    // unpacked) -- reused here at single-layer granularity instead of that
    // function's whole-model aggregate, but with the two checks in the
    // OPPOSITE priority order (unpacked checked first). Deliberate: a real
    // multi-source build can genuinely have BOTH a layer's old packed
    // _exps tensor (from an untouched shard) AND a full set of per-expert
    // unpacked tensors for the same layer (from a supplementary override
    // file) present in the SAME loaded tensor map -- they never collide by
    // NAME (gguf_loader.hpp's last-shard-wins rule only resolves same-name
    // collisions), so both simply coexist unless something here picks one.
    // The per-expert tensors are the more specific, deliberate signal --
    // gguf_tool pack --unpack's own real output never leaves the old
    // packed tensor behind, so seeing both only happens when a caller
    // (this project's own test tooling, or a real "layer this in as an
    // override" build) explicitly added the unpacked set on top of an
    // otherwise-untouched packed shard. Checking packed first would let
    // that leftover packed tensor silently shadow the deliberate override.
    bool layer_is_unpacked(int layer_idx) const {
        size_t gb, ub, db;
        const GGUFTensor *gM, *uM, *dM;
        if (unpacked_expert_geometry(layer_idx, 0, gb, ub, db, gM, uM, dM)) return true;
        return false;
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
    //
    // 2026-09: the "expert 0 represents the whole layer" guarantee above no
    // longer holds unconditionally -- sanity_check_unpacked_format now
    // ACCEPTS a layer whose experts have genuinely different sizes (real
    // mixed precision, Phase 1), flagging it in layer_has_mixed_expert_
    // sizes instead of refusing to start. Every caller of THIS function
    // still assumes one number represents the whole layer (RAM/VRAM slot
    // sizing, disk-read batch planning -- none of that has been made
    // per-expert-aware yet), so returning expert 0's size for a flagged
    // layer would silently under/over-size those callers rather than
    // failing loudly. Refuse instead: false here is a clear, well-defined
    // "not supported yet for this layer" signal a caller can act on,
    // rather than quietly wrong numbers a caller has no way to detect.
    // Per-expert callers (build_expert_read_requests et al., which already
    // call unpacked_expert_geometry directly with a real expert_id) are
    // completely unaffected -- this function is the ONLY thing gated here.
    bool expert_layer_geometry(int layer_idx, size_t& gb, size_t& ub, size_t& db,
                                const GGUFTensor*& gM, const GGUFTensor*& uM, const GGUFTensor*& dM) const {
        // layer_is_unpacked, not the global unpacked_expert_format flag --
        // per-layer dispatch (2026-09, Phase 5, see layer_is_unpacked's own
        // comment): a mixed model can have this SPECIFIC layer packed even
        // when the flag is set, or unpacked even when it's not.
        if (layer_is_unpacked(layer_idx)) {
            if (layer_idx >= 0 && (size_t)layer_idx < layer_has_mixed_expert_sizes.size()
                && layer_has_mixed_expert_sizes[layer_idx]) {
                return false;
            }
            return unpacked_expert_geometry(layer_idx, 0, gb, ub, db, gM, uM, dM);
        }
        return packed_slab_geometry(layer_idx, gb, ub, db, gM, uM, dM);
    }

    // Per-(layer, REAL expert_id) geometry, correctly dispatching by
    // format -- for the caller that already knows exactly which expert it
    // wants, unlike expert_layer_geometry's "expert 0 represents the
    // layer" shortcut above. Packed format: uniform per layer by
    // construction, expert_id is irrelevant, same packed_slab_geometry
    // call either function would make. Unpacked format: resolves THIS
    // expert's own real geometry directly via unpacked_expert_geometry,
    // never expert 0's -- correct for a uniform layer (where they'd agree
    // anyway) AND a genuinely mixed one (where they would not).
    //
    // 2026-09: added after an expert_layer_geometry caller audit (Phase 2
    // of per-expert mixed precision, see CLAUDE.md) found 3 real call
    // sites that already had a specific expert_id in scope but were
    // calling expert_layer_geometry anyway -- silently correct only
    // because every layer was uniform so far, and would have silently
    // populated the WRONG expert's metadata for a genuinely mixed layer
    // once expert_layer_geometry stopped refusing those. Two more call
    // sites called expert_layer_geometry without even checking its return
    // value -- a real, newly-exposed crash risk (dereferencing an
    // uninitialized GGUFTensor* on refusal) once that function started
    // returning false for mixed layers instead of always succeeding.
    // Both classes of caller are fixed to call this function instead.
    bool expert_geometry_for(int layer_idx, int expert_id, size_t& gb, size_t& ub, size_t& db,
                              const GGUFTensor*& gM, const GGUFTensor*& uM, const GGUFTensor*& dM) const {
        // layer_is_unpacked, not the global flag -- see its own comment
        // (2026-09, Phase 5: per-layer mixed packed/unpacked models).
        if (layer_is_unpacked(layer_idx))
            return unpacked_expert_geometry(layer_idx, expert_id, gb, ub, db, gM, uM, dM);
        return packed_slab_geometry(layer_idx, gb, ub, db, gM, uM, dM);
    }

    // Startup diagnostic (called from main.cpp right after the model loads).
    // Reports, LOUDLY and to BOTH the log file and stderr, when this model's
    // expert tensors are not 4096-aligned -- a condition that silently
    // degrades I/O in two different ways depending on WHERE the misalignment
    // is, and which was previously only discoverable after the fact:
    //
    //   * general.alignment < 4096 (file was never repacked, or repacked with
    //     a smaller --align): enable_direct_io refuses O_DIRECT for EVERY
    //     shard, so all expert reads fall back to buffered I/O through the
    //     page cache. On a box whose RAM tier is already sized near physical
    //     memory this is far worse than "a bit slower" -- streaming a
    //     multi-hundred-GB model through the page cache competes directly
    //     with the tier for headroom.
    //
    //   * per-expert stride (gb/ub/db) or base offset not a 4096 multiple:
    //     O_DIRECT still works, but individual expert reads land off a block
    //     boundary, so io_uring_backend's submit_and_wait must stage each one
    //     through an aligned scratch buffer and memcpy it out. On this
    //     project's own reference model that memcpy was ~132GB per session
    //     once measured -- real CPU and memory-bandwidth burn competing with
    //     the compute the engine is trying to overlap I/O against.
    //
    // Deliberately emitted to stderr as well as the log: log_path routes
    // LOG_* to a file (logger.hpp's `FILE* out = file_ ? file_ : stderr`), so
    // a warning that only went through LOG_WARN would be invisible to someone
    // watching the console, which is exactly where a "why is this slow?"
    // investigation usually starts.
    void warn_if_expert_tensors_unaligned() const {
        constexpr uint64_t ALIGN = 4096;
        const auto& c = model_file.architecture_config;

        auto it = model_file.kv_uint.find("general.alignment");
        const uint64_t file_align = (it != model_file.kv_uint.end()) ? it->second : 32;

        int n_checked = 0, n_bad_offset = 0, n_bad_stride = 0, first_bad = -1;
        for (int l = 0; l < c.n_layers; ++l) {
            size_t gb = 0, ub = 0, db = 0;
            const GGUFTensor *gM = nullptr, *uM = nullptr, *dM = nullptr;
            if (!expert_layer_geometry(l, gb, ub, db, gM, uM, dM)) continue;
            ++n_checked;
            const bool bad_offset = (gM->offset % ALIGN) || (uM->offset % ALIGN) || (dM->offset % ALIGN);
            const bool bad_stride = (gb % ALIGN) || (ub % ALIGN) || (db % ALIGN);
            if (bad_offset || bad_stride) {
                if (first_bad < 0) first_bad = l;
                if (bad_offset) ++n_bad_offset;
                if (bad_stride) ++n_bad_stride;
            }
        }

        auto emit = [](const std::string& msg) {
            LOG_WARN("%s", msg.c_str());
            std::fprintf(stderr, "%s", msg.c_str());
            std::fflush(stderr);
        };

        if (file_align < ALIGN) {
            std::ostringstream o;
            o << "[ALIGN-WARN] general.alignment=" << file_align << " (< " << ALIGN
              << "): tensor ranges need aligned-window staging for O_DIRECT; "
                 "the Ling bailingmoe3 loader can use it when cuFile is off, "
                 "while other architectures keep the aligned-file gate. "
                 "Repacking with `tools/gguf_tool pack --align 4096` can "
                 "remove staging copies.\n";
            emit(o.str());
        }

        if (n_checked > 0 && (n_bad_offset > 0 || n_bad_stride > 0)) {
            std::ostringstream o;
            o << "[ALIGN-WARN] Expert tensors are not " << ALIGN << "-aligned in "
              << (n_bad_offset > n_bad_stride ? n_bad_offset : n_bad_stride) << " of " << n_checked
              << " MoE layer(s) (first: layer " << first_bad << "; "
              << n_bad_offset << " with an unaligned base offset, " << n_bad_stride
              << " with an unaligned per-expert stride). Every direct expert read will be staged "
                 "through a scratch buffer and memcpy'd out instead of DMA'd straight into the "
                 "destination -- watch the `io_uring direct staging:` line in the shutdown report to "
                 "see the real cost. Fix: repack with `tools/gguf_tool pack --align 4096`.\n";
            emit(o.str());
        }

        if (file_align >= ALIGN && n_checked > 0 && n_bad_offset == 0 && n_bad_stride == 0) {
            LOG_STREAM_INFO() << "[ALIGN] Expert tensors are " << ALIGN << "-aligned across all "
                              << n_checked << " MoE layer(s) -- direct reads can DMA straight into "
                                 "the destination buffer (no staging memcpy).\n";
        }
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
    // 2026-09: RELAXED, Phase 1 of per-expert mixed precision (see
    // CLAUDE.md). Completeness (every expert present) and tightness (each
    // expert's own gate/up/down physically contiguous, still the real
    // precondition for a single-DMA read of THAT expert) remain hard,
    // fatal requirements -- unchanged from before. What's DROPPED is the
    // old requirement that every expert in a layer share the SAME
    // gate/up/down byte size as expert 0. That requirement was never
    // actually needed for per-expert single-DMA reads (tightness is a
    // per-expert property, independent of what size any OTHER expert in
    // the layer is) -- it existed only because expert_layer_geometry's own
    // "expert 0 represents the whole layer" shortcut needed it to be safe,
    // and nothing downstream of that shortcut has been updated yet to
    // reason per-expert (RAM/VRAM slot sizing, disk-read batch planning --
    // see layer_has_mixed_expert_sizes's own comment). So: a genuinely
    // mixed layer is no longer refused at the FILE level, but is flagged
    // in layer_has_mixed_expert_sizes so expert_layer_geometry can refuse
    // to hand out a wrong "representative" size for it instead of silently
    // returning expert 0's numbers as if they applied to every expert.
    // Verified against a real hand-built 2-expert mixed-type layer (real
    // IQ1_S + real IQ2_XXS/IQ3_XXS bytes, byte-identical round-trip, clean
    // GPU dispatch for both) -- see main_test.cpp Test 18.
    bool sanity_check_unpacked_format() {
        int n_layers = model_file.architecture_config.n_layers;
        int n_exp = model_file.architecture_config.n_experts;
        if (n_layers <= 0 || n_exp <= 0) {
            fprintf(stderr, "[FATAL] unpacked_expert_format=true but architecture_config has "
                            "n_layers=%d n_experts=%d -- cannot validate, refusing to run\n",
                    n_layers, n_exp);
            return false;
        }
        layer_has_mixed_expert_sizes.assign(n_layers, false);
        int mixed_layer_count = 0;
        int packed_layer_count = 0;
        // Genuinely unpacked (found == n_exp), tracked separately from
        // "not packed" -- a dense-only layer (no MoE tensors at all,
        // architecture_config.first_dense) is also "not packed" without
        // being unpacked either. A real bug this session's own Phase 5
        // verification run caught: the summary log below used to compute
        // this as `n_layers - packed_layer_count`, which silently counted
        // every dense-only layer as "unpacked" too (a real run reported
        // "3/78 unpacked" for a file with only ONE real unpacked layer --
        // confirmed via a direct tensor-table check finding exactly one).
        // Purely a log-message inaccuracy, not a dispatch bug -- layer_is_
        // unpacked() itself already correctly returns false for a
        // dense-only layer, same as it does for a packed one -- but worth
        // fixing for real, since a wrong count here would mislead the next
        // person reading this exact line to debug a real mixed model.
        int unpacked_layer_count = 0;
        for (int l = 0; l < n_layers; ++l) {
            // 2026-09, Phase 5 (mixed packed/unpacked models, see
            // layer_is_unpacked's own comment): a layer that genuinely has
            // the default packed _exps tensor is NOT this function's
            // concern at all -- it's not an incomplete unpacked repack,
            // it's simply a different, already-correct format, validated
            // and handled by the existing packed-format code paths
            // entirely independently of this function. Skip it outright
            // rather than let it fall into the "found==0" dense-layer
            // case below (which happens to also return early, but for the
            // wrong reason, and would misreport a real packed layer as an
            // empty one in any future diagnostic that inspects `found`).
            // MUST use layer_is_unpacked() here, not a fresh inline
            // packed_slab_geometry() check -- a real bug this session's
            // own Phase 5 verification run caught: this used to check
            // packed FIRST (its own separate copy of the logic), while
            // layer_is_unpacked() (what every real dispatch/read-planning
            // call site now uses) checks unpacked FIRST -- see that
            // function's own comment for why. The mismatch meant a layer
            // with BOTH a stale packed tensor (untouched shard) and a
            // genuine unpacked override (this function's whole reason to
            // exist) got silently SKIPPED here as "packed", so its real
            // completeness/tightness was NEVER actually validated, even
            // though the real dispatch path correctly used the unpacked
            // data. Confirmed on real hardware: a genuinely mixed test
            // model (one real layer overridden to unpacked, rest
            // untouched packed) logged "0/78 genuinely unpacked" here
            // despite the override working correctly end-to-end (0
            // errors, coherent output) -- the validation was the only
            // thing actually broken, not the dispatch.
            if (!layer_is_unpacked(l)) {
                // Not unpacked -- distinguish genuinely packed (has the
                // _exps tensor) from dense-only (neither format, e.g. a
                // architecture_config.first_dense leading layer) purely
                // for an accurate summary count below; either way there's
                // nothing for this function to validate, so both continue.
                size_t pgb, pub, pdb;
                const GGUFTensor *pgM, *puM, *pdM;
                if (packed_slab_geometry(l, pgb, pub, pdb, pgM, puM, pdM)) ++packed_layer_count;
                continue;
            }
            size_t layer_gb = 0, layer_ub = 0, layer_db = 0;
            bool have_layer_size = false;
            bool layer_mixed = false;
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
                    layer_mixed = true;   // real, expected for a mixed-precision layer -- not fatal
                }
                if (!unpacked_expert_is_tight(*gM, *uM, *dM)) {
                    fprintf(stderr, "[FATAL] unpacked_expert_format=true: layer %d expert %d's "
                                    "gate/up/down are not physically contiguous on one file (likely split "
                                    "across a repack output-shard boundary) -- a single-DMA read of THIS "
                                    "expert requires it to be tight, regardless of what size any other "
                                    "expert in the layer is. Re-run gguf_tool pack --unpack with a larger "
                                    "--max-shard-gb so no expert straddles a shard boundary, or set "
                                    "unpacked_expert_format=false and use the default --align-only packed "
                                    "format instead. Stopping before any real read.\n",
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
            layer_has_mixed_expert_sizes[l] = layer_mixed;
            if (layer_mixed) ++mixed_layer_count;
            if (found == n_exp) ++unpacked_layer_count;
        }
        if (packed_layer_count > 0) {
            LOG_INFO("[UNPACKED] sanity check passed: %d/%d layer(s) are genuinely unpacked (validated "
                     "complete + tight below), %d/%d layer(s) are still the default packed format "
                     "(untouched, handled entirely by the existing packed-format code paths), %d/%d "
                     "layer(s) are dense-only (no MoE tensors, neither format) -- a real mixed model "
                     "(Phase 5, see CLAUDE.md), not a full-model conversion.\n",
                     unpacked_layer_count, n_layers, packed_layer_count, n_layers,
                     n_layers - packed_layer_count - unpacked_layer_count, n_layers);
        }
        if (mixed_layer_count > 0) {
            LOG_INFO("[UNPACKED] every genuinely-unpacked layer's experts are complete and physically "
                     "tight (single-DMA-per-expert reads enabled). %d/%d unpacked layer(s) have "
                     "non-uniform per-expert sizes (real mixed precision) -- layer-level geometry "
                     "shortcuts are disabled for those layers until per-expert-aware sizing lands "
                     "(Phase 2/3); per-expert reads/dispatch are unaffected.\n",
                     mixed_layer_count, unpacked_layer_count);
        } else if (unpacked_layer_count > 0) {
            LOG_INFO("[UNPACKED] sanity check passed: every genuinely-unpacked layer's experts are "
                     "complete, uniformly sized, and physically tight -- single-DMA reads enabled.\n");
        }
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
    // PERSISTENT PINNING (EngineConfig::ram_pinned_budget_gb, 2026-08-15):
    // registers each slab with CUDA exactly ONCE, here at allocation time,
    // instead of per-transfer. Motivation, from real hardware -- a per-call
    // cudaHostRegister conversion of upload_expert_active_async was tried
    // and reverted (see that function's own comment), but its measurements
    // proved the underlying mechanism works and quantified the prize: with
    // a PINNED source, that call site's cudaMemcpyAsync becomes genuinely
    // asynchronous, so the DMA overlaps real compute instead of blocking
    // the decode thread -- its own cost collapsed from 18.96s (blocking, on
    // unpinned memory the driver must stage synchronously through an
    // internal bounce buffer) to 0.15s issue + 5.26s batch-sync wait. The
    // ONLY reason that attempt was a net loss was the 19.10s of per-call
    // register+unregister tax paid to get there. Paying it once per
    // allocation instead removes exactly that tax: projected ~14s/request
    // saved on this config (0.57 -> ~0.72 tok/s).
    //
    // Deliberately BUDGETED, not unconditional: this pins host RAM, which
    // the kernel can then never reclaim under pressure, and host_ram_gb is
    // already 120 of this box's 125GB (see host_ram_gb's own freeze-incident
    // history). Note pinning does not ALLOCATE anything -- these pages are
    // already resident -- it only locks them, so the real risk is reduced OS
    // headroom, not new consumption. Registration throughput measured at
    // ~11 GB/s on this hardware, so even a large budget costs only seconds,
    // once, during preload. Overshoot past the budget is impossible: the
    // reservation is a single atomic fetch_add, undone if it would exceed.
    //
    // Composes safely with this codebase's OTHER cudaHostRegister sites
    // (upload_expert/upload_expert_async): their per-call register on an
    // already-registered slab simply fails, and each only unregisters what
    // its OWN register succeeded at -- so a persistent pin is never torn
    // down underneath them. Expect their n_*_host_register_fail counters to
    // read nonzero (and their own staging path to be skipped) whenever this
    // feature is on; that is the expected, benign interaction, not a bug.
    // Old slab buffers whose free() had to be deferred because a reader
    // still held the pointer (see ensure_slab_capacity). Bounded in
    // practice: only a genuine regrow-under-a-live-reader lands here, which
    // a real run hit 3 times, and each entry is released as soon as the
    // owning block's guards clear.
    struct DeferredSlabFree {
        void* ptr;
        const MoEExpertBlock* owner;
        size_t reservation_bytes;
    };
    std::vector<DeferredSlabFree> deferred_slab_frees;
    std::mutex deferred_slab_free_mutex;

    // Release any deferred buffer whose owning block no longer has a reader.
    // Once that block's temp pin is gone, its load finished, and no async
    // copy is pending, nothing can still be holding the OLD pointer either:
    // every reader of a slab takes one of those holds for the duration of
    // its use, so all three clearing is proof the old pointer is unreachable.
    void drain_deferred_slab_frees() {
        std::lock_guard<std::mutex> dg(deferred_slab_free_mutex);
        size_t w = 0;
        for (size_t i = 0; i < deferred_slab_frees.size(); ++i) {
            const auto& d = deferred_slab_frees[i];
            const MoEExpertBlock* o = d.owner;
            const bool still_busy = o && (o->ram_temp_pin_count > 0 || o->ram_loading ||
                                          const_cast<MoEExpertBlock*>(o)->copy_still_pending());
            if (still_busy) {
                deferred_slab_frees[w++] = d;   // keep waiting
            } else {
                free(d.ptr);
                MemoryGuard::release_host_bytes(d.reservation_bytes);
                perf().n_ram_slab_deferred_freed.fetch_add(1, std::memory_order_relaxed);
            }
        }
        deferred_slab_frees.resize(w);
    }

    // Single safe way to release a slot's slab. Every caller that frees a
    // slab MUST go through this rather than calling free() directly: the
    // three guards are the only thing standing between a regrow/reclaim and
    // an in-flight reader (e.g. a queued ActiveLayerUploadWorkItem holding
    // host_data captured at claim time). Unregister happens immediately --
    // that only drops the page-lock, never the pages -- while the free
    // itself is deferred whenever a reader may still hold the pointer.
    void release_slab(MoEExpertBlock& slot) {
        // A graph-backed layer owns this slice for the lifetime of the
        // engine. Its identity is part of the graph's tensor view and it is
        // never an LRU victim while the graph cache is active.
        if (slot.slab_from_shared_arena) return;
        // 2026-09-04 DOUBLE-FREE FIX. This and ensure_slab_capacity both did
        // read slab_buffer -> decide -> push/free -> clear, with the mutex held
        // ONLY around the push_back. Since ensure_slab_capacity runs inside
        // get_experts_batch's OMP region, two threads could both observe the
        // same non-null pointer and both push it onto deferred_slab_frees; the
        // drain then freed it twice. Observed as
        // `double free or corruption (!prev)` under glibc.malloc.check=3,
        // aborting in ensure_slab_capacity during decode.
        //
        // Fix: CLAIM the pointer (read + null it) while holding the mutex, so
        // exactly one caller can ever own it. The actual free() happens after
        // the lock is dropped -- free() is thread-safe on its own and there is
        // no reason to hold a global mutex across it.
        uint8_t* claimed = nullptr;
        size_t claimed_reservation = 0;
        bool deferred = false;
        {
            std::lock_guard<std::mutex> dg(deferred_slab_free_mutex);
            claimed = slot.slab_buffer;
            if (!claimed) return;
            claimed_reservation = slot.host_reservation_bytes;
            slot.host_reservation_bytes = 0;
            if (slot.slab_pinned) {
                cudaHostUnregister(claimed);
                slot.slab_pinned = false;
                ram_pinned_bytes_used.fetch_sub(slot.slab_capacity, std::memory_order_relaxed);
            }
            const bool in_use = slot.ram_temp_pin_count > 0 || slot.ram_loading ||
                                slot.copy_still_pending();
            if (in_use) {
                deferred_slab_frees.push_back({claimed, &slot, claimed_reservation});
                perf().n_ram_slab_free_deferred.fetch_add(1, std::memory_order_relaxed);
                deferred = true;
            }
            slot.slab_buffer = nullptr;
            slot.slab_capacity = 0;
        }
        if (!deferred) {
            free(claimed);
            MemoryGuard::release_host_bytes(claimed_reservation);
        }
    }

    void ensure_slab_capacity(MoEExpertBlock& slot, size_t total) {
        drain_deferred_slab_frees();
        PERF_START(alloc);
        if (slot.slab_from_shared_arena) {
            if (slot.slab_capacity < total) {
                LOG_ERROR("[GRAPH-RAM] shared arena slice too small: have=%zu need=%zu\n",
                          slot.slab_capacity, total);
            }
            PERF_END(alloc, t_host_alloc);
            return;
        }
        if (slot.slab_capacity < total) {
            auto t0 = std::chrono::high_resolution_clock::now();
            // Reserve before allocating and before releasing the old slab.
            // This accounts for the realloc peak (old + new), so a growth
            // cannot transiently exceed host_ram_gb even when the old buffer
            // is still protected by an async reader.
            bool ledgered = true;
            if (!MemoryGuard::try_reserve_host_bytes(total)) {
                perf().n_mem_guard_alloc_refused.fetch_add(1, std::memory_order_relaxed);
                // Every caller writes `total` bytes into slab_buffer right
                // after this returns and none can report failure, so a
                // refused growth that keeps the old, smaller slab overruns it
                // (SIGSEGV in a 426-token GLM-5.3 prefill). The guard's 20 GiB
                // admission reserve is far above what a few-MB slab needs;
                // grow outside the ledger while the kernel still has a hard
                // 2 GiB of MemAvailable, and only keep the old slab when it
                // is already large enough or the machine is genuinely out.
                size_t avail_kb = 0;
                if (FILE *mf = fopen("/proc/meminfo", "r")) {
                    char line[128];
                    while (fgets(line, sizeof line, mf))
                        if (sscanf(line, "MemAvailable: %zu kB", &avail_kb) == 1)
                            break;
                    fclose(mf);
                }
                if (avail_kb < 2ull * 1024 * 1024) {
                    LOG_ERROR("[MEM-GUARD] RAM slab growth refused: %.2f MB (reserved=%.2f GB, "
                              "budget=%.2f GB, MemAvailable=%.2f GB); slab stays too small\n",
                              total / 1e6, MemoryGuard::reserved_host_bytes() / 1e9,
                              MemoryGuard::host_budget_bytes() / 1e9, avail_kb / 1e6);
                    PERF_END(alloc, t_host_alloc);
                    return;
                }
                ledgered = false;
            }

            // Allocate the replacement while the old slab remains valid.  If
            // malloc fails, the caller keeps a usable old slot and the new
            // reservation is discharged immediately.
            void* nb = nullptr;
            const int pmerr = posix_memalign(&nb, 4096, total);
            if (pmerr != 0 || !nb) {
                if (ledgered) MemoryGuard::release_host_bytes(total);
                LOG_ERROR("[RAM] posix_memalign(%.2f MB) failed (errno=%d) -- this slot is "
                          "left unchanged so the caller can keep using its old slab.\n",
                          total / 1e6, pmerr);
                PERF_END(alloc, t_host_alloc);
                return;
            }

            uint8_t* claimed = nullptr;
            size_t claimed_reservation = 0;
            bool claimed_deferred = false;
            {
                std::lock_guard<std::mutex> dg(deferred_slab_free_mutex);
                claimed = slot.slab_buffer;
                if (claimed) {
                    claimed_reservation = slot.host_reservation_bytes;
                    slot.host_reservation_bytes = 0;
                    const bool in_use = slot.ram_temp_pin_count > 0 ||
                                        slot.ram_loading ||
                                        slot.copy_still_pending();
                    // Unregister before free or deferral; unregister only
                    // removes the CUDA page lock, not the allocation.
                    if (slot.slab_pinned) {
                        cudaHostUnregister(claimed);
                        slot.slab_pinned = false;
                        ram_pinned_bytes_used.fetch_sub(slot.slab_capacity,
                                                        std::memory_order_relaxed);
                        perf().n_ram_slab_unpinned_on_regrow.fetch_add(
                            1, std::memory_order_relaxed);
                    }
                    if (in_use) {
                        deferred_slab_frees.push_back(
                            {claimed, &slot, claimed_reservation});
                        perf().n_ram_slab_free_deferred.fetch_add(
                            1, std::memory_order_relaxed);
                        claimed_deferred = true;
                    }
                    slot.slab_buffer = nullptr;
                    slot.slab_capacity = 0;
                }
                slot.slab_buffer = static_cast<uint8_t*>(nb);
                slot.slab_capacity = total;
                slot.host_reservation_bytes = ledgered ? total : 0;
            }
            if (claimed && !claimed_deferred) {
                free(claimed);
                MemoryGuard::release_host_bytes(claimed_reservation);
            }
            double ms = std::chrono::duration<double, std::milli>(
                std::chrono::high_resolution_clock::now() - t0).count();
            if (ms > 50.0) {
                LOG_WARN("[SLOW-HOST-ALLOC] posix_memalign(%.2fMB) took %.2fms (old_capacity=%.2fMB)\n",
                         total / 1e6, ms, claimed_reservation / 1e6);
            }

            if (slot.slab_buffer && ram_pinned_budget_bytes > 0) {
                // Reserve first, register second -- a plain check-then-add
                // would let concurrent callers (decode thread + prefetch
                // workers) both pass the check and jointly overshoot.
                size_t prev = ram_pinned_bytes_used.fetch_add(total, std::memory_order_relaxed);
                if (prev + total <= ram_pinned_budget_bytes) {
                    PERF_START(ram_slab_pin);
                    cudaError_t perr = cudaHostRegister(slot.slab_buffer, total, cudaHostRegisterPortable);
                    PERF_END(ram_slab_pin, t_ram_slab_pin);
                    if (perr == cudaSuccess) {
                        slot.slab_pinned = true;
                        perf().n_ram_slab_pinned.fetch_add(1, std::memory_order_relaxed);
                        perf().bytes_ram_slab_pinned.fetch_add((int64_t)total, std::memory_order_relaxed);
                    } else {
                        ram_pinned_bytes_used.fetch_sub(total, std::memory_order_relaxed);
                        perf().n_ram_slab_pin_fail.fetch_add(1, std::memory_order_relaxed);
                        // The aggregate counter is useful for the final
                        // report, but it cannot distinguish a driver-wide
                        // registration ceiling from a bad range or an
                        // already-registered allocation.  Keep only the
                        // first few startup examples so a large preload does
                        // not turn this diagnostic into a log bottleneck.
                        static std::atomic<uint64_t> pin_failure_examples{0};
                        const uint64_t example =
                            pin_failure_examples.fetch_add(1, std::memory_order_relaxed);
                        if (example < 4) {
                            LOG_WARN("[RAM-PIN] cudaHostRegister declined slab %.2f MiB "
                                     "after %.2f GiB was accepted: %s\n",
                                     total / (1024.0 * 1024.0),
                                     ram_pinned_bytes_used.load(std::memory_order_relaxed) /
                                         (1024.0 * 1024.0 * 1024.0),
                                     cudaGetErrorString(perr));
                        }
                        cudaGetLastError();  // don't let this leak into an unrelated later check
                    }
                } else {
                    ram_pinned_bytes_used.fetch_sub(total, std::memory_order_relaxed);
                    perf().n_ram_slab_pin_over_budget.fetch_add(1, std::memory_order_relaxed);
                }
            }
        }
        PERF_END(alloc, t_host_alloc);
    }

    // Set up a MoEExpertBlock's three projection host_ptr/meta from an already-
    // populated packed slab_buffer (gate|up|down at offsets 0, gb, gb+ub).
    // Shared by the pread path and the consume_staged (prefetch) path -- the
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
        // expert_geometry_for, not expert_layer_geometry -- this expert_id
        // is real, not a stand-in for "the layer" (see that function's own
        // comment: found by the 2026-09 caller audit).
        if (!expert_geometry_for(layer_idx, expert_id, gb, ub, db, gM, uM, dM)) return false;
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
    // Passed as `replica` to request an INTRA-EXPERT split rather than a
    // single copy: gate+up are read from the primary and down from the
    // mirror, so BOTH drives work on every single expert fetch -- including
    // a solo one, where per-expert striping leaves the other drive fully
    // idle. Real bucketed measurement showed why this matters: with
    // per-expert striping, ~50% of prefetch fetches have no other fetch in
    // flight, and those improved only 15% (8.27 -> 7.04ms) versus 40% at
    // high concurrency, because one drive was moving the whole 11.3MB slab
    // alone.
    //
    // The gate+up / down cut is deliberate, not arbitrary: it puts 6.48MB of
    // 11.3MB (57.3%) on the primary, and this box's primary is the faster
    // drive (2.25 vs 1.98 GB/s), whose ideal share would be 53.2% -- close
    // enough that both legs finish together, which is what actually
    // determines a split read's latency. Predicted solo cost:
    // max(6.48/2.25, 4.82/1.98) = 2.88ms versus 11.3/2.25 = 5.0ms unsplit.
    //
    // Packed format only -- unpacked_expert_format issues ONE combined read
    // for the whole slab by design (that format exists precisely to make it
    // a single DMA), so there is nothing to split and it falls back to the
    // primary.
    static constexpr int REPLICA_SPLIT = -1;

    // `replica` selects which physical copy of the model to read from when a
    // mirror is configured (0 = primary, 1 = mirror on the other device --
    // see GGUFModel::open_mirror). Defaults to 0, so every existing caller
    // is byte-identical to before. A caller that wants striping just varies
    // this per expert; because io_uring allows mixed fds in one submission,
    // a batch built with alternating replicas becomes a single submit that
    // both devices service in parallel.
    bool build_expert_read_requests(int layer_idx, int expert_id,
                                     MoEExpertBlock& target_slot,
                                     io_uring_backend::ReadRequest* out_reqs,
                                     int& out_n, size_t& out_total,
                                     int replica = 0) {
        // A sparse mirror holds only some experts: pick the one drive that has this one.
        replica = model_file.resolve_replica(layer_idx, expert_id, replica);
        // Projection-compact graph arenas have no packed per-expert slab, so
        // point the three startup requests directly at their final planes.
        // Runtime misses are not expected for a complete immutable graph
        // arena and must not write a packed slab over the shared allocation.
        if (graph_ram_layer_projection_compact(layer_idx)) {
            size_t gb = 0, ub = 0, db = 0;
            const GGUFTensor *gM = nullptr, *uM = nullptr, *dM = nullptr;
            const bool unpacked = layer_is_unpacked(layer_idx);
            if (unpacked) {
                if (!unpacked_expert_geometry(layer_idx, expert_id, gb, ub,
                                              db, gM, uM, dM))
                    return false;
            } else if (!packed_slab_geometry(layer_idx, gb, ub, db, gM, uM,
                                             dM)) {
                return false;
            }
            if (!gM || !uM || !dM)
                return false;
            const GraphRamLayerArena &arena =
                graph_ram_arenas[(size_t)layer_idx];
            const int n_exp = model_file.architecture_config.n_experts;
            if (n_exp <= 0 || expert_id < 0 || expert_id >= n_exp ||
                arena.n_experts != n_exp || gb != arena.gate_bytes ||
                ub != arena.up_bytes || db != arena.down_bytes)
                return false;
            const size_t e = (size_t)expert_id;
            if (e > SIZE_MAX / arena.gate_stride ||
                e > SIZE_MAX / arena.up_stride ||
                e > SIZE_MAX / arena.down_stride)
                return false;
            const uint8_t *base = arena.base;
            uint8_t *dst_gate = const_cast<uint8_t *>(
                base + arena.gate_offset + e * arena.gate_stride);
            uint8_t *dst_up = const_cast<uint8_t *>(
                base + arena.up_offset + e * arena.up_stride);
            uint8_t *dst_down = const_cast<uint8_t *>(
                base + arena.down_offset + e * arena.down_stride);
            if (unpacked && replica == REPLICA_SPLIT)
                replica = 0;
            const bool split = !unpacked && replica == REPLICA_SPLIT;
            const int r_gate = split ? 0 : replica;
            const int r_up = split ? 0 : replica;
            const int r_down = split ? 1 : replica;
            out_n = 0;
            if (unpacked) {
                // The per-expert format has independent source tensors.  The
                // compact destination still has three separate planes, so
                // this is necessarily three reads even when the source
                // tensors happen to be tight in the GGUF file.
                out_reqs[out_n++] = {
                    model_file.fd_for_tensor(*gM, r_gate),
                    model_file.direct_io_active_for(*gM, r_gate),
                    model_file.direct_io_alignment, dst_gate, gb, gM->offset};
                out_reqs[out_n++] = {
                    model_file.fd_for_tensor(*uM, r_up),
                    model_file.direct_io_active_for(*uM, r_up),
                    model_file.direct_io_alignment, dst_up, ub, uM->offset};
                out_reqs[out_n++] = {
                    model_file.fd_for_tensor(*dM, r_down),
                    model_file.direct_io_active_for(*dM, r_down),
                    model_file.direct_io_alignment, dst_down, db, dM->offset};
            } else {
                out_reqs[out_n++] = {
                    model_file.fd_for_tensor(*gM, r_gate),
                    model_file.direct_io_active_for(*gM, r_gate),
                    model_file.direct_io_alignment, dst_gate, gb,
                    gM->offset + (uint64_t)expert_id * gb};
                out_reqs[out_n++] = {
                    model_file.fd_for_tensor(*uM, r_up),
                    model_file.direct_io_active_for(*uM, r_up),
                    model_file.direct_io_alignment, dst_up, ub,
                    uM->offset + (uint64_t)expert_id * ub};
                out_reqs[out_n++] = {
                    model_file.fd_for_tensor(*dM, r_down),
                    model_file.direct_io_active_for(*dM, r_down),
                    model_file.direct_io_alignment, dst_down, db,
                    dM->offset + (uint64_t)expert_id * db};
            }
            out_total = gb + ub + db;
            return true;
        }
        // layer_is_unpacked, not the global flag -- see its own comment
        // (2026-09, Phase 5: per-layer mixed packed/unpacked models). This
        // is the function that actually PLANS the real disk read, so
        // dispatching on the wrong per-layer format here would build a
        // read request against the wrong tensor entirely, not just size
        // something wrong -- REPLICA_SPLIT's own single-DMA rationale
        // below still only applies to a layer that's genuinely unpacked.
        if (layer_is_unpacked(layer_idx)) {
            // One combined read; nothing to split across devices.
            if (replica == REPLICA_SPLIT) replica = 0;
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
            out_reqs[out_n++] = {model_file.fd_for_tensor(*gM, replica), model_file.direct_io_active_for(*gM, replica),
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
        // Per-tensor replica selection -- see REPLICA_SPLIT's own comment.
        // For a plain (non-split) replica every leg resolves to the same
        // copy, byte-identical to the previous behavior.
        const bool split = (replica == REPLICA_SPLIT);
        const int r_gate = split ? 0 : replica;
        const int r_up   = split ? 0 : replica;
        const int r_down = split ? 1 : replica;
        out_reqs[out_n++] = {model_file.fd_for_tensor(*gM, r_gate), model_file.direct_io_active_for(*gM, r_gate),
                              model_file.direct_io_alignment, dst_gate, gb, gate_off};
        out_reqs[out_n++] = {model_file.fd_for_tensor(*uM, r_up), model_file.direct_io_active_for(*uM, r_up),
                              model_file.direct_io_alignment, dst_up, ub, up_off};
        out_reqs[out_n++] = {model_file.fd_for_tensor(*dM, r_down), model_file.direct_io_active_for(*dM, r_down),
                              model_file.direct_io_alignment, dst_down, db, down_off};
        return true;
    }

    // Publish projection metadata after a batched read has populated the
    // final graph-arena planes. No bytes are copied here.
    bool populate_graph_ram_projection_views(int layer_idx, int expert_id,
                                             MoEExpertBlock &target_slot) {
        if (!graph_ram_layer_projection_compact(layer_idx) || expert_id < 0)
            return false;
        const GraphRamLayerArena &arena =
            graph_ram_arenas[(size_t)layer_idx];
        const int n_exp = model_file.architecture_config.n_experts;
        if (n_exp <= 0 || expert_id >= n_exp || arena.n_experts != n_exp)
            return false;
        size_t gb = 0, ub = 0, db = 0;
        const GGUFTensor *gM = nullptr, *uM = nullptr, *dM = nullptr;
        const bool unpacked = layer_is_unpacked(layer_idx);
        const bool have_geometry = unpacked
            ? unpacked_expert_geometry(layer_idx, expert_id, gb, ub, db,
                                       gM, uM, dM)
            : packed_slab_geometry(layer_idx, gb, ub, db, gM, uM, dM);
        if (!have_geometry || !gM || !uM || !dM ||
            gb != arena.gate_bytes ||
            ub != arena.up_bytes || db != arena.down_bytes)
            return false;
        const size_t e = (size_t)expert_id;
        if (e > SIZE_MAX / arena.gate_stride ||
            e > SIZE_MAX / arena.up_stride ||
            e > SIZE_MAX / arena.down_stride)
            return false;
        target_slot.gate_proj.host_ptr =
            arena.base + arena.gate_offset + e * arena.gate_stride;
        target_slot.gate_proj.meta = *gM;
        target_slot.gate_proj.meta.offset =
            arena.gate_offset + e * arena.gate_stride;
        target_slot.gate_proj.meta.size_bytes = gb;
        target_slot.up_proj.host_ptr =
            arena.base + arena.up_offset + e * arena.up_stride;
        target_slot.up_proj.meta = *uM;
        target_slot.up_proj.meta.offset =
            arena.up_offset + e * arena.up_stride;
        target_slot.up_proj.meta.size_bytes = ub;
        target_slot.down_proj.host_ptr =
            arena.base + arena.down_offset + e * arena.down_stride;
        target_slot.down_proj.meta = *dM;
        target_slot.down_proj.meta.offset =
            arena.down_offset + e * arena.down_stride;
        target_slot.down_proj.meta.size_bytes = db;
        target_slot.expert_id = expert_id;
        target_slot.vram_shadow = false;
        return true;
    }

    // Populate a projection-compact graph arena directly from the model
    // shards.  The old graph path first read each expert into an interleaved
    // [gate|up|down] slab and load_full_weight() then copied every projection
    // into a compact graph-owned buffer.  This path makes the three disk reads
    // land in their final arena planes, eliminating that multi-gigabyte
    // startup duplicate and preserving the same host pointers for the graph.
    bool load_graph_ram_projection_compact(int layer_idx, int expert_id,
                                           MoEExpertBlock &target_slot) {
        if (!graph_ram_layer_projection_compact(layer_idx) || expert_id < 0)
            return false;
        const GraphRamLayerArena &arena =
            graph_ram_arenas[(size_t)layer_idx];
        const int n_exp = model_file.architecture_config.n_experts;
        if (n_exp <= 0 || expert_id >= n_exp || arena.n_experts != n_exp)
            return false;

        size_t gb = 0, ub = 0, db = 0;
        const GGUFTensor *gM = nullptr, *uM = nullptr, *dM = nullptr;
        const bool unpacked = layer_is_unpacked(layer_idx);
        const bool have_geometry = unpacked
            ? unpacked_expert_geometry(layer_idx, expert_id, gb, ub, db,
                                       gM, uM, dM)
            : packed_slab_geometry(layer_idx, gb, ub, db, gM, uM, dM);
        if (!have_geometry || !gM || !uM || !dM ||
            gb != arena.gate_bytes ||
            ub != arena.up_bytes || db != arena.down_bytes ||
            arena.gate_stride < gb || arena.up_stride < ub ||
            arena.down_stride < db)
            return false;

        const size_t e = (size_t)expert_id;
        if (e > SIZE_MAX / arena.gate_stride ||
            e > SIZE_MAX / arena.up_stride ||
            e > SIZE_MAX / arena.down_stride)
            return false;
        uint8_t *dst_gate = arena.base + arena.gate_offset +
                            e * arena.gate_stride;
        uint8_t *dst_up = arena.base + arena.up_offset + e * arena.up_stride;
        uint8_t *dst_down = arena.base + arena.down_offset +
                            e * arena.down_stride;

        auto read_projection = [&](const GGUFTensor &meta, size_t bytes,
                                   uint8_t *dst, uint64_t source_offset,
                                   const char *which) {
            const auto t0 = std::chrono::high_resolution_clock::now();
            const int fd = model_file.fd_for_tensor(meta);
            const ssize_t got = direct_io::pread_direct_aware(
                fd, model_file.direct_io_active_for(meta),
                model_file.direct_io_alignment, dst, bytes, source_offset);
            const double ms = std::chrono::duration<double, std::milli>(
                                  std::chrono::high_resolution_clock::now() - t0)
                                  .count();
            if (got < 0 || (size_t)got != bytes)
                LOG_WARN("[WARN] compact graph expert short read L%d E%d %s: "
                         "%zd/%zu\n",
                         layer_idx, expert_id, which, got, bytes);
            if (ms > SLOW_PREAD_MS)
                LOG_DEBUG("[SLOW-PREAD] compact graph L%d E%d proj=%s "
                          "bytes=%.2fMB took=%.2fms\n",
                          layer_idx, expert_id, which, bytes / 1e6, ms);
            return got >= 0 && (size_t)got == bytes;
        };

        PERF_START(disk);
        const uint64_t gate_source =
            gM->offset + (unpacked ? 0 : (uint64_t)expert_id * gb);
        const uint64_t up_source =
            uM->offset + (unpacked ? 0 : (uint64_t)expert_id * ub);
        const uint64_t down_source =
            dM->offset + (unpacked ? 0 : (uint64_t)expert_id * db);
        const bool ok = read_projection(*gM, gb, dst_gate, gate_source, "gate") &&
                        read_projection(*uM, ub, dst_up, up_source, "up") &&
                        read_projection(*dM, db, dst_down, down_source, "down");
        PERF_END(disk, t_disk_read);
        perf().bytes_disk_read.fetch_add(gb + ub + db,
                                         std::memory_order_relaxed);
        if (!ok)
            return false;

        if (drop_expert_pages) {
            posix_fadvise(model_file.fd_for_tensor(*gM),
                          gate_source, gb,
                          POSIX_FADV_DONTNEED);
            posix_fadvise(model_file.fd_for_tensor(*uM),
                          up_source, ub,
                          POSIX_FADV_DONTNEED);
            posix_fadvise(model_file.fd_for_tensor(*dM),
                          down_source, db,
                          POSIX_FADV_DONTNEED);
        }

        return populate_graph_ram_projection_views(layer_idx, expert_id,
                                                   target_slot);
    }

    void load_expert_slab(int layer_idx, int expert_id, MoEExpertBlock& target_slot) {
        if (graph_ram_layer_projection_compact(layer_idx)) {
            // This arena is already the final projection layout. Never fall
            // through to the legacy packed loader on a failed read: that
            // would overwrite the gate plane with gate|up|down bytes and
            // leave the graph with silently corrupted weights.
            (void)load_graph_ram_projection_compact(layer_idx, expert_id,
                                                    target_slot);
            return;
        }
        std::string base = "blk." + std::to_string(layer_idx) + ".";
        int n_exp = model_file.architecture_config.n_experts;
        // ---- Packed expert format: ffn_{gate,up,down}_exps.weight ----
        // Each packed tensor is [a, b, n_experts] with the expert index as the
        // LAST (contiguous) dimension, so expert E occupies bytes
        // [E*per_exp, (E+1)*per_exp). ne0 is the input dim, so the slice is
        // already the [O,I] row-major weight the matmul expects — no shuffle.
        // (GLM-5.2 UD ships this way: gate/up = Q2_K, down = Q3_K, packed.)
        //
        // MUST gate on !layer_is_unpacked(layer_idx), not a bare
        // packed_slab_geometry() success check -- a real, confirmed bug
        // (found 2026-09 chasing a real GPU-path NaN on a genuinely mixed
        // per-expert model): a Phase 5 mixed-precision build deliberately
        // leaves the ORIGINAL packed _exps tensor sitting untouched in a
        // symlinked baseline shard alongside a genuine unpacked override
        // (see engine.hpp's layer_is_unpacked() comment -- "both simply
        // coexist... unless something here picks one", and every OTHER
        // dispatch-decision call site already checks layer_is_unpacked()
        // FIRST for exactly this reason). This function was the one real
        // holdout still checking packed_slab_geometry() unconditionally --
        // packed_slab_geometry() SUCCEEDS whenever that old tensor is still
        // present, regardless of whether an unpacked override also exists,
        // so this branch was silently reading a mixed layer's WRONG
        // (baseline-uniform-sized) bytes for every expert -- while every
        // other function downstream (expert_geometry_for, VRAM sizing, GPU
        // kernel type selection) correctly resolved the SAME expert via the
        // real override. That size/type mismatch (RAM slab holds e.g. an
        // IQ2_S expert's bytes at IQ2_S's own offset/size, while everything
        // downstream treats it as the real IQ1_S override -- different
        // size, different quant format) is what produced a real NaN
        // cascade on real hardware once GPU dispatch actually reached a
        // promoted cold expert -- confirmed by an isolated GPU-kernel test
        // against this exact expert's REAL override bytes coming back
        // completely clean (0 NaN), proving the kernel itself was never
        // the problem, only what bytes this function fed it.
        size_t gb, ub, db;
        const GGUFTensor *gM, *uM, *dM;
        if (!layer_is_unpacked(layer_idx) && packed_slab_geometry(layer_idx, gb, ub, db, gM, uM, dM)) {
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
            bool is_direct_this = model_file.direct_io_active_for(gate);
            ssize_t r = direct_io::pread_direct_aware(
                gate_fd, is_direct_this, model_file.direct_io_alignment,
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
    //
    // 2026-09: unpacked-format branch now scans EVERY expert in the layer
    // and returns the MAX combined size, not just expert 0's -- Phase 2 of
    // per-expert mixed precision (see CLAUDE.md). All 5 real callers of
    // this function are budget/capacity-planning uses (never "give me a
    // specific expert's own type"), and 4 of the 5 already take a max
    // across every LAYER to handle GLM-5.2-style cross-layer type
    // heterogeneity (see this function's own comment above and main.cpp's
    // VRAM-pool-sizing loop) -- returning this layer's own max here is the
    // exact same principle applied one level deeper, not a new design, and
    // it's a real, provable safe bound: every caller wants "big enough for
    // whatever this layer might need to hold", and the true max is exactly
    // that, for a uniform layer (max == the one shared size, byte-
    // identical to the old expert-0 lookup) or a genuinely mixed one alike.
    // The packed-format branch below is untouched -- GGUF's one-type-per-
    // tensor constraint already makes it inherently uniform per layer, so
    // there's nothing to scan.
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

        if (n_exp > 0) {
            size_t max_total = 0;
            bool found_any = false;
            for (int e = 0; e < n_exp; ++e) {
                size_t gb, ub, db;
                const GGUFTensor *gM, *uM, *dM;
                if (!unpacked_expert_geometry(layer_idx, e, gb, ub, db, gM, uM, dM)) {
                    if (!found_any) continue;  // no experts at all yet -- keep scanning, e.g. a sparse test file
                    break;                     // real gap after real experts -- sanity_check_unpacked_format
                                                // already refuses this at startup; here, just stop scanning
                                                // rather than let a hole silently under-count.
                }
                found_any = true;
                max_total = std::max(max_total, gb + ub + db);
            }
            if (found_any) return max_total;
        }

        // Legacy fallback (architecture_config.n_experts not yet populated,
        // e.g. a narrow test harness that only sets up expert 0) -- same
        // expert-0-only lookup this function always had, unchanged.
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

    bool prepare_graph_ram_arenas(
        const std::vector<int> &full_layers,
        const std::vector<int> &projection_compact_layers = {}) {
        graph_ram_arenas.assign(layers_experts.size(), {});
        graph_ram_cache_enabled = false;
        const int n_experts = model_file.architecture_config.n_experts;
        if (n_experts <= 0 || full_layers.empty()) return false;

        std::vector<unsigned char> requested(layers_experts.size(), 0);
        for (int layer : full_layers) {
            if (layer >= 0 && (size_t)layer < requested.size())
                requested[(size_t)layer] = 1;
        }
        std::vector<unsigned char> compact_requested(layers_experts.size(), 0);
        for (int layer : projection_compact_layers) {
            if (layer >= 0 && (size_t)layer < compact_requested.size())
                compact_requested[(size_t)layer] = 1;
        }

        size_t allocated_layers = 0;
        for (size_t layer = 0; layer < requested.size(); ++layer) {
            if (!requested[layer] || layers_experts[layer].size() < (size_t)n_experts)
                continue;

            size_t gb = 0, ub = 0, db = 0;
            const GGUFTensor *gM = nullptr, *uM = nullptr, *dM = nullptr;
            if (!expert_layer_geometry((int)layer, gb, ub, db, gM, uM, dM) ||
                gb == 0 || ub == 0 || db == 0 ||
                gb > SIZE_MAX - ub || gb + ub > SIZE_MAX - db)
                continue;

            constexpr size_t alignment = 4096;
            auto align_up = [](size_t value) -> size_t {
                constexpr size_t align = 4096;
                if (value > SIZE_MAX - (align - 1)) return 0;
                return (value + align - 1) & ~(align - 1);
            };
            const bool projection_compact = compact_requested[layer] != 0;
            size_t stride = 0;
            size_t gate_offset = 0;
            size_t up_offset = 0;
            size_t down_offset = 0;
            size_t gate_stride = 0;
            size_t up_stride = 0;
            size_t down_stride = 0;
            size_t bytes = 0;
            if (projection_compact) {
                // Keep each projection plane tightly packed.  The CPU
                // MUL_MAT_ID path consumes nb[2] as the expert-axis stride;
                // making that stride exactly the per-expert byte count gives
                // it the same layout as a normal contiguous [.., expert]
                // tensor and avoids a second graph-owned projection copy.
                // O_DIRECT does not require every destination expert slice
                // to be aligned: direct_io::pread_direct_aware stages an
                // unaligned destination through its aligned scratch buffer.
                gate_stride = gb;
                up_stride = ub;
                down_stride = db;
                if (gate_stride == 0 || up_stride == 0 || down_stride == 0 ||
                    gate_stride > SIZE_MAX / (size_t)n_experts ||
                    up_stride > SIZE_MAX / (size_t)n_experts ||
                    down_stride > SIZE_MAX / (size_t)n_experts)
                    continue;
                up_offset = gate_stride * (size_t)n_experts;
                if (up_offset > SIZE_MAX - up_stride * (size_t)n_experts)
                    continue;
                down_offset = up_offset + up_stride * (size_t)n_experts;
                if (down_offset > SIZE_MAX - down_stride * (size_t)n_experts)
                    continue;
                bytes = down_offset + down_stride * (size_t)n_experts;
            } else {
                if (gb > SIZE_MAX - ub || gb + ub > SIZE_MAX - db)
                    continue;
                const size_t raw = gb + ub + db;
                stride = align_up(raw);
                if (stride == 0 || stride > SIZE_MAX / (size_t)n_experts)
                    continue;
                bytes = stride * (size_t)n_experts;
            }
            // 10 GiB watermark, per explicit user instruction after the
            // 2026-09-09 crash (was 8 GiB) -- aligned to the same floor as
            // every other host-memory admission check in the full-graph
            // path (ggml_glm5next_graph.cpp/ggml_qwen4exp_graph.cpp).
            if (!MemoryGuard::try_reserve_host_bytes(
                    bytes, 10ull * 1024ull * 1024ull * 1024ull)) {
                LOG_WARN("[GRAPH-RAM] layer %zu arena refused: %.2f GiB "
                         "(reserved=%.2f GiB, budget=%.2f GB)\n",
                         layer, (double)bytes / (1024.0 * 1024.0 * 1024.0),
                         (double)MemoryGuard::reserved_host_bytes() / 1.0e9,
                         (double)MemoryGuard::host_budget_bytes() / 1.0e9);
                continue;
            }

            // llama.cpp's CPU-mapped GGUF tensors are normally offset by 32 bytes
            // from their mmap page base.  MoEStream's graph arenas are
            // allocated separately, so reproduce that host pointer phase by
            // default; the logical tensor layout is unchanged.  Set
            // MOESTREAM_QWEN_GRAPH_HOST_OFFSET=0 for the page-aligned A/B.
            size_t host_offset = 32;
            if (const char *v = std::getenv("MOESTREAM_QWEN_GRAPH_HOST_OFFSET")) {
                char *end = nullptr;
                const long parsed = std::strtol(v, &end, 10);
                if (end != v && *end == '\0' && parsed >= 0 && parsed <= 4096)
                    host_offset = static_cast<size_t>(parsed);
            }
            const size_t allocation_bytes = bytes + host_offset;
            void *raw_ptr = nullptr;
            if (posix_memalign(&raw_ptr, alignment, allocation_bytes) != 0 || !raw_ptr) {
                MemoryGuard::release_host_bytes(bytes);
                LOG_WARN("[GRAPH-RAM] layer %zu arena allocation failed: %.2f GiB\n",
                         layer, (double)bytes / (1024.0 * 1024.0 * 1024.0));
                continue;
            }

            GraphRamLayerArena &arena = graph_ram_arenas[layer];
            arena.allocation_base = raw_ptr;
            arena.allocation_bytes = allocation_bytes;
            arena.base = static_cast<uint8_t *>(raw_ptr) + host_offset;
            arena.bytes = bytes;
            arena.stride = stride;
            arena.gate_bytes = gb;
            arena.up_bytes = ub;
            arena.down_bytes = db;
            arena.gate_offset = gate_offset;
            arena.up_offset = up_offset;
            arena.down_offset = down_offset;
            arena.gate_stride = projection_compact ? gate_stride : stride;
            arena.up_stride = projection_compact ? up_stride : stride;
            arena.down_stride = projection_compact ? down_stride : stride;
            arena.n_experts = n_experts;
            arena.projection_compact = projection_compact;
            arena.host_reservation_bytes = bytes;

            // llama.cpp's fast CPU path uses mmap+mlock for its host expert source.
            // Keep MoEStream's explicit preloaded DDR arena as the default,
            // but expose the same page-residency property as a controlled A/B.
            // This is deliberately opt-in: locking the complete graph-RAM
            // set consumes the process's memlock allowance and should not be
            // enabled without an explicit host-memory budget.
            const char *graph_mlock =
                std::getenv("MOESTREAM_QWEN_GRAPH_MLOCK");
            if (graph_mlock && std::atoi(graph_mlock) != 0) {
                if (mlock(arena.allocation_base, arena.allocation_bytes) == 0) {
                    arena.mlocked = true;
                } else if (layer == 0) {
                    LOG_WARN("[GRAPH-RAM] mlock declined: %s\n",
                             std::strerror(errno));
                }
            }

            // Do not register the graph-RAM expert arena by default.  These
            // are large CPU-resident expert weights, not activation staging
            // buffers; pinning the complete arena consumes the host pin
            // budget without helping the CPU MUL_MAT_ID path.  The scheduler
            // separately uses CUDA_Host for its small CPU-only activation,
            // route-ID, and output buffers.  Keep a deliberate opt-in for a
            // future A/B that wants to measure pinned expert weights, and
            // retain the old NO_REGISTER switch as a compatibility override.
            const char * graph_register =
                std::getenv("MOESTREAM_QWEN_GRAPH_REGISTER");
            const char * graph_no_register =
                std::getenv("MOESTREAM_QWEN_GRAPH_NO_REGISTER");
            const bool graph_register_enabled =
                graph_register && std::atoi(graph_register) != 0 &&
                !(graph_no_register && std::atoi(graph_no_register) != 0);
            if (ram_pinned_budget_bytes > 0 && graph_register_enabled) {
                const size_t previous =
                    ram_pinned_bytes_used.fetch_add(bytes, std::memory_order_relaxed);
                if (previous <= ram_pinned_budget_bytes &&
                    bytes <= ram_pinned_budget_bytes - previous) {
                    const cudaError_t err = cudaHostRegister(
                        arena.allocation_base, arena.allocation_bytes,
                        cudaHostRegisterPortable);
                    if (err == cudaSuccess) {
                        arena.pinned = true;
                    } else {
                        ram_pinned_bytes_used.fetch_sub(bytes,
                                                        std::memory_order_relaxed);
                        cudaGetLastError();
                    }
                } else {
                    ram_pinned_bytes_used.fetch_sub(bytes,
                                                    std::memory_order_relaxed);
                }
            }
            if (!graph_register_enabled && layer == 0) {
                LOG_STREAM_INFO()
                    << "[GRAPH-RAM] expert-arena cudaHostRegister disabled by "
                       "default; set MOESTREAM_QWEN_GRAPH_REGISTER=1 for an "
                       "explicit A/B\n";
            }

            for (int expert = 0; expert < n_experts; ++expert) {
                MoEExpertBlock &slot = layers_experts[layer][(size_t)expert];
                // This path is startup-only and is called before preload. Do
                // not overwrite a live individually-owned slab if a caller
                // violates that ordering; leave the layer out of the graph
                // view rather than leaking or aliasing the wrong allocation.
                if (slot.slab_buffer && !slot.slab_from_shared_arena) {
                    LOG_WARN("[GRAPH-RAM] layer %zu has an existing RAM slab; "
                             "shared view disabled for this layer\n", layer);
                    if (arena.pinned) {
                        cudaHostUnregister(arena.allocation_base);
                        ram_pinned_bytes_used.fetch_sub(
                            arena.bytes, std::memory_order_relaxed);
                    }
                    if (arena.mlocked) {
                        munlock(arena.allocation_base, arena.allocation_bytes);
                    }
                    free(arena.allocation_base);
                    MemoryGuard::release_host_bytes(arena.host_reservation_bytes);
                    arena = {};
                    goto next_graph_ram_layer;
                }
                // A compact arena has no single packed per-expert slab.
                // `slab_buffer` is consumed by the generic RAM->VRAM
                // promotion paths as a contiguous [gate|up|down] source.
                // Advertising arena.base here would make those paths upload
                // the beginning of the gate plane as if it were this
                // expert's packed record, corrupting graph-visible VRAM
                // weights.  Leave the packed-slab handle empty; the three
                // projection host_ptr fields below are the only valid source
                // representation for a compact graph slot.
                if (projection_compact) {
                    slot.slab_buffer = nullptr;
                    slot.slab_capacity = 0;
                } else {
                    slot.slab_buffer = arena.base + (size_t)expert * arena.stride;
                    slot.slab_capacity = arena.stride;
                }
                slot.host_reservation_bytes = 0;
                slot.slab_pinned = arena.pinned;
                slot.slab_from_shared_arena = true;
            }
            ++allocated_layers;
            if (projection_compact) {
                LOG_STREAM_INFO()
                    << "[GRAPH-RAM] layer " << layer << " arena: "
                    << (double)bytes / (1024.0 * 1024.0 * 1024.0)
                    << " GiB, projection-compact, gate_stride=" << gate_stride
                                  << ", up_stride=" << up_stride
                                  << ", down_stride=" << down_stride << " bytes, "
                                  << (arena.pinned ? "pinned" : "pageable")
                                  << "\n";
            } else {
                LOG_STREAM_INFO()
                    << "[GRAPH-RAM] layer " << layer << " arena: "
                    << (double)bytes / (1024.0 * 1024.0 * 1024.0)
                    << " GiB, interleaved, stride=" << stride << " bytes, "
                                  << (arena.pinned ? "pinned" : "pageable")
                                  << "\n";
            }
        next_graph_ram_layer:
            continue;
        }

        graph_ram_cache_enabled = allocated_layers > 0;
        if (graph_ram_cache_enabled)
            LOG_STREAM_INFO() << "[GRAPH-RAM] prepared " << allocated_layers
                              << " complete layer arena(s); startup preload will "
                                 "populate existing graph-visible slots\n";
        return graph_ram_cache_enabled;
    }

    bool graph_ram_layer_allocated(int layer) const {
        return graph_ram_cache_enabled && layer >= 0 &&
               (size_t)layer < graph_ram_arenas.size() &&
               graph_ram_arenas[(size_t)layer].base != nullptr;
    }

    bool graph_ram_layer_ready(int layer) const {
        if (!graph_ram_layer_allocated(layer)) return false;
        const GraphRamLayerArena &arena = graph_ram_arenas[(size_t)layer];
        if (arena.ready) return true;
        if (arena.n_experts <= 0 || (size_t)arena.n_experts > layers_experts[(size_t)layer].size())
            return false;
        for (int expert = 0; expert < arena.n_experts; ++expert) {
            const MoEExpertBlock &slot = layers_experts[(size_t)layer][(size_t)expert];
            const bool slab_identity = arena.projection_compact
                                           ? slot.slab_buffer == nullptr
                                           : slot.slab_buffer ==
                                                 arena.base +
                                                     (size_t)expert * arena.stride;
            const bool slab_capacity = arena.projection_compact
                                           ? slot.slab_capacity == 0
                                           : slot.slab_capacity >=
                                                 arena.gate_bytes +
                                                     arena.up_bytes +
                                                     arena.down_bytes;
            const bool projection_identity =
                arena.projection_compact
                    ? slot.gate_proj.host_ptr ==
                              arena.base + arena.gate_offset +
                                  (size_t)expert * arena.gate_stride &&
                          slot.up_proj.host_ptr ==
                              arena.base + arena.up_offset +
                                  (size_t)expert * arena.up_stride &&
                          slot.down_proj.host_ptr ==
                              arena.base + arena.down_offset +
                                  (size_t)expert * arena.down_stride
                    : true;
            if (!slot.slab_from_shared_arena || slot.expert_id != expert ||
                !slab_identity || !slab_capacity || !projection_identity ||
                slot.ram_loading || !slot.gate_proj.host_ptr ||
                !slot.up_proj.host_ptr || !slot.down_proj.host_ptr)
                return false;
        }
        arena.ready = true;
        return true;
    }

    bool qwen_ram_vram_only_layer(int layer) const {
        return layer >= 0 &&
               (size_t)layer < qwen_ram_vram_only_layers.size() &&
               qwen_ram_vram_only_layers[(size_t)layer] != 0;
    }

    bool graph_ram_layer_projection_compact(int layer) const {
        return graph_ram_layer_allocated(layer) &&
               graph_ram_arenas[(size_t)layer].projection_compact;
    }

    const GraphRamLayerArena *graph_ram_arena(int layer) const {
        return graph_ram_layer_allocated(layer)
                   ? &graph_ram_arenas[(size_t)layer]
                   : nullptr;
    }

    // GGML-CUDA's quantized MUL_MAT_ID kernels express nb[2] in units of
    // the source type's block size.  The VRAM cache uses one physical slot
    // stride for every layer, while dynamic-quant models can have different
    // row sizes (for example IQ2_XXS and IQ1_M in the same Qwen checkpoint).
    // A direct view of an interleaved resident slot is valid only when that
    // common stride is a whole number of rows for every routed projection.
    // Return the smallest such byte alignment.  This is deliberately based
    // on the GGUF tensor metadata, so it remains correct for future mixed
    // quantization layouts without hard-coding a list of block sizes.
    size_t expert_cache_layer_stride_alignment(int layer) const {
        size_t alignment = 1;
        const int n_exp = model_file.architecture_config.n_experts;
        if (n_exp <= 0 || layer < 0) return alignment;

        const std::string base = "blk." + std::to_string(layer) + ".";
        auto add_row_alignment = [&](const GGUFTensor& t, size_t divisor) {
            if (divisor == 0 || t.shape.size() < 2) return;
            const size_t rows = (size_t)t.shape[1];
            if (rows == 0) return;
            const size_t per_expert = t.size_bytes / divisor;
            const size_t row_bytes = per_expert / rows;
            if (row_bytes == 0) return;
            const size_t g = std::gcd(alignment, row_bytes);
            if (alignment > SIZE_MAX / (row_bytes / g)) {
                alignment = SIZE_MAX;
                return;
            }
            alignment *= row_bytes / g;
        };

        auto add = [&](const std::string& name, size_t divisor) {
            auto it = model_file.tensors.find(name);
            if (it != model_file.tensors.end()) add_row_alignment(it->second, divisor);
        };
        add(base + "ffn_gate_exps.weight", (size_t)n_exp);
        add(base + "ffn_up_exps.weight", (size_t)n_exp);
        add(base + "ffn_down_exps.weight", (size_t)n_exp);
        if (layer_is_unpacked(layer)) {
            for (int e = 0; e < n_exp; ++e) {
                const std::string es = std::to_string(e);
                add(base + "ffn_gate." + es + ".weight", 1);
                add(base + "ffn_up." + es + ".weight", 1);
                add(base + "ffn_down." + es + ".weight", 1);
            }
        }
        return alignment;
    }

    size_t expert_cache_stride_alignment() const {
        size_t alignment = 1;
        const int n_exp = model_file.architecture_config.n_experts;
        if (n_exp <= 0) return alignment;

        auto add_row_alignment = [&](const GGUFTensor& t, size_t divisor) {
            if (divisor == 0 || t.shape.size() < 2) return;
            const size_t rows = (size_t)t.shape[1];
            if (rows == 0) return;
            const size_t per_expert = t.size_bytes / divisor;
            const size_t row_bytes = per_expert / rows;
            if (row_bytes == 0) return;
            const size_t g = std::gcd(alignment, row_bytes);
            if (alignment > SIZE_MAX / (row_bytes / g)) {
                alignment = SIZE_MAX;
                return;
            }
            alignment *= row_bytes / g;
        };

        // Normally layers_experts is populated by graph_initializer and also
        // includes the optional GLM NextN/MTP routed block.  Keep the target
        // count as a fallback for narrow unit-test engines that call this
        // helper before constructing the cache topology.
        const int cache_layers = layers_experts.empty()
                                     ? model_file.architecture_config.n_layers
                                     : static_cast<int>(layers_experts.size());
        for (int l = 0; l < cache_layers; ++l) {
            const std::string base = "blk." + std::to_string(l) + ".";
            auto add = [&](const std::string& name, size_t divisor) {
                auto it = model_file.tensors.find(name);
                if (it != model_file.tensors.end()) add_row_alignment(it->second, divisor);
            };

            add(base + "ffn_gate_exps.weight", (size_t)n_exp);
            add(base + "ffn_up_exps.weight", (size_t)n_exp);
            add(base + "ffn_down_exps.weight", (size_t)n_exp);

            // Repacked per-expert GGUFs have one tensor per expert.  Their
            // row alignment is the same constraint, but there is no expert
            // axis to divide out.
            if (layer_is_unpacked(l)) {
                for (int e = 0; e < n_exp; ++e) {
                    const std::string es = std::to_string(e);
                    add(base + "ffn_gate." + es + ".weight", 1);
                    add(base + "ffn_up." + es + ".weight", 1);
                    add(base + "ffn_down." + es + ".weight", 1);
                }
            }
            if (alignment == SIZE_MAX) return alignment;
        }
        return alignment;
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
        // layers_experts is the authoritative routed-layer topology.  It also
        // includes an opt-in GLM NextN/MTP block; omitting that layer here
        // would make its router writes silently fall outside the histogram.
        int nl = static_cast<int>(layers_experts.size());
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

    // Charge a long-lived engine-owned host allocation to the same ledger
    // used by RAM slabs, graph buffers, and MTP scratch.  Callers must invoke
    // this before allocating and release the charge if allocation fails.
    bool reserve_host_memory(size_t bytes) {
        if (bytes == 0) return true;
        if (!MemoryGuard::try_reserve_host_bytes(bytes)) {
            host_memory_allocation_failed = true;
            return false;
        }
        host_memory_reservation_bytes += bytes;
        return true;
    }

    void release_host_memory(size_t bytes) {
        if (bytes == 0) return;
        const size_t released = std::min(bytes, host_memory_reservation_bytes);
        host_memory_reservation_bytes -= released;
        MemoryGuard::release_host_bytes(released);
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
