// gpu_topology.hpp — Hardware topology detection and multi-GPU role assignment
// Implements Profiles A/B/C from optimize.md
#pragma once
#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <functional>
#include "perf_timer.hpp"
#include "logger.hpp"
#include "engine_config.hpp"  // GpuConfig -- allocate_expert_cache's real per-vault-GPU cap
#include "pilot_confirmed_set.hpp"  // was_pilot_confirmed_resident -- evict_lru race measurement, see that function's comment
#include "cufile_io.hpp"  // upload_expert_cufile's direct disk->VRAM read path

// Any wait to acquire cache_mutex exceeding this is logged individually
// (tagged [SLOW-CACHE-MUTEX-WAIT]) -- companion to async_prefetcher.hpp's
// SLOW_MUTEX_WAIT_MS for staging_mutex. Both added together to root-cause a
// real, reproducible multi-second (up to 7.5s observed) single-cache-miss
// stall whose per-miss latency histogram ruled out raw disk throughput as
// the cause (median/p90 were completely normal -- only a handful of calls
// were catastrophic, the signature of blocking on contention, not slow I/O).
// cache_mutex is shared between the main decode thread's inline hit-
// triggered promotions (RAMSafetyManager::get_expert Step 1 -- fires on
// nearly every hit when promote_hit_threshold is low) and the async
// prefetcher worker thread's own background promotions/evictions.
constexpr double SLOW_CACHE_MUTEX_WAIT_MS = 50.0;

// Acquire cache_mutex into `varname` while timing how long the acquisition
// itself blocked -- logs + accumulates into perf().t_cache_mutex_wait on a
// slow wait. A wait here (as opposed to slow work AFTER acquiring) directly
// proves cross-thread contention, since cache_mutex is recursive (same-
// thread re-entry, e.g. upload_expert calling find_expert internally, is
// always near-instant and won't trip the threshold).
#define GPU_TOPO_TIMED_LOCK(varname) \
    auto _lock_wait_t0_##varname = std::chrono::high_resolution_clock::now(); \
    std::lock_guard<std::recursive_mutex> varname(cache_mutex); \
    do { \
        double _wait_ms = std::chrono::duration<double, std::milli>( \
            std::chrono::high_resolution_clock::now() - _lock_wait_t0_##varname).count(); \
        PerfCounters::atomic_add(perf().t_cache_mutex_wait, _wait_ms / 1000.0); \
        if (_wait_ms > SLOW_CACHE_MUTEX_WAIT_MS) { \
            LOG_WARN("[SLOW-CACHE-MUTEX-WAIT] %s blocked %.2fms acquiring cache_mutex\n", \
                     __func__, _wait_ms); \
        } \
    } while (0)

enum class GpuRole {
    CoreEngine,   // Dense layers + KV cache + activation state (newest arch / lower VRAM)
    ExpertVault,  // MoE expert LRU cache (highest VRAM capacity)
    Unused
};

struct GpuDevice {
    int id = -1;
    std::string name;
    size_t total_vram = 0;
    size_t available_vram = 0;
    int sm_major = 0;
    int sm_minor = 0;
    GpuRole role = GpuRole::Unused;
    cudaStream_t compute_stream = nullptr;
    cudaStream_t copy_stream = nullptr;
    // Dedicated stream for the decode-critical-path active-layer batch
    // pre-warm (upload_expert_active_async/sync_active_layer_batch) ONLY --
    // added 2026-08-19 after a real-hardware diagnostic (n_active_prewarm_
    // saw_copy_stream_busy, perf_timer.hpp) showed copy_stream busy 62.7%
    // of the time (8932/14247 calls) at the exact moment this call site's
    // H2D fires. copy_stream is also shared by upload_expert_async's
    // reactive/pilot RAM->VRAM promotions and demote_from_vram's D2H
    // evictions -- background housekeeping that has no reason to make
    // THIS token's own imminent expert dispatch wait behind it, since a
    // CUDA stream executes its enqueued ops strictly in issue order
    // regardless of the 2 independent copy engines this hardware actually
    // has (asyncEngineCount=2 on both GPUs here, confirmed in CLAUDE.md's
    // demote_from_vram investigation) that could otherwise run one H2D and
    // one D2H concurrently. Giving this specific traffic its own stream
    // lets it overlap with, rather than serialize behind, that other
    // copy_stream work. Two EARLIER fix attempts at this call site
    // (cudaHostRegister-per-call, then a persistent bounce-buffer pool --
    // see upload_expert_active_async's own comment) were both about making
    // the DMA source pinned/genuinely-async and were reverted after real
    // regressions; neither of them touched WHICH stream the copy went on,
    // so this is a genuinely different, previously-untried lever, not a
    // re-run of a closed investigation.
    cudaStream_t active_layer_stream = nullptr;

    // Dedicated stream for SPECULATIVE pilot pre-fills only (EngineConfig::
    // speculative_promote_to_active_layer's async issue path,
    // RAMSafetyManager::promote_ram_to_active_layer -> upload_expert_
    // active_async(speculative=true)), added 2026-08-26 -- separates this
    // traffic from active_layer_stream's OWN, more time-critical
    // consumers (decode_async_prewarm's real-need pre-warm, prefill's
    // Phase-2 batch warm-up, and the reactive synchronous upload_expert_
    // active path all still use active_layer_stream). A speculative l+1
    // prediction is lower-priority than any of those (it's a guess about
    // NEXT layer, not a confirmed need for the layer about to dispatch),
    // so it should never be able to queue ahead of and delay them on a
    // shared stream -- CUDA streams execute enqueued work strictly in
    // issue order, the exact mechanism active_layer_stream's own creation
    // comment above documents a real, measured win from. Whether THIS
    // split also pays off needs its own real A/B (this project's own
    // history has both proven wins, e.g. active_layer_stream itself, and
    // clean no-benefit results, e.g. demote_from_vram/upload_expert_
    // active's own reverted attempts, from the identical-looking change
    // at different call sites) -- see CLAUDE.md for the result.
    cudaStream_t speculative_prefetch_stream = nullptr;

    // 2026-08-26 follow-up: a SECOND speculative stream, alternated with
    // the one above via speculative_stream_rr -- built after a real
    // diagnostic (n_speculative_issue_saw_own_stream_busy/_idle,
    // perf_timer.hpp) DISPROVED the original hypothesis for this feature
    // (cross-mechanism contention with active_layer_stream: confirmed
    // 0.0% busy, never once caught) and instead found the real cost is
    // SELF-contention -- 22.7-31.4% of speculative issues found the
    // PREVIOUS speculative issue's own DMA still draining on the SAME
    // stream, from several pilot candidates (up to max_k per pilot, two
    // pilots) bursting through in one transition. Splitting that burst
    // across two independent streams should directly cut same-stream
    // queuing -- see speculative_stream_rr's own comment for the
    // alternation logic. Real A/B result in CLAUDE.md.
    cudaStream_t speculative_prefetch_stream2 = nullptr;

    // Round-robin counter picking which of the two speculative streams
    // above a given upload_expert_active_async(speculative=true) call
    // uses -- plain (not atomic), safe because every caller already holds
    // GPU_TOPO_TIMED_LOCK for the whole call.
    uint64_t speculative_stream_rr = 0;

    // 2026-08-29 re-test: dedicated stream for upload_expert_active's
    // SYNCHRONOUS fallback cudaMemcpy specifically -- see n_active_upload_
    // saw_active_layer_stream_busy's own comment (perf_timer.hpp) for the
    // real diagnostic that motivated this (22.5% of these calls found
    // active_layer_stream busy at the exact moment this genuinely urgent,
    // real-dispatch fallback fired, and CUDA's legacy default/NULL stream
    // -- what this call used before -- implicitly waits for EVERY other
    // blocking-type stream on the device to drain before it can even
    // start). A broader version of this exact fix (bundled with a
    // cudaHostRegister change) was already tried once and reverted with
    // "no effect" (0.70->0.68 tok/s, within noise) -- see this function's
    // own comment above the cudaMemcpy call for that history. Re-tried
    // narrowly (stream only, no register change) because the system has
    // changed substantially since that null result (ActiveLayerUploadWorkers,
    // multi-GPU active-layer routing, ram_pinned_budget_gb=45, the cache_
    // mutex batching fix) -- any of which could plausibly have changed how
    // much real active_layer_stream backlog exists at the moment this
    // fires. cudaMemcpyAsync+an explicit cudaStreamSynchronize right after
    // preserves this function's existing fully-synchronous contract
    // (caller still blocks until the copy is done) -- only WHICH stream
    // the copy waits behind changes, not whether the caller waits for it.
    // See CLAUDE.md for the real A/B result.
    cudaStream_t active_layer_sync_fallback_stream = nullptr;

    // 2026-10-22: dedicated NON-BLOCKING stream for demand-time active-layer
    // uploads (upload_expert_active_async's `demand` mode). The existing
    // synchronous fallback blocks the decode thread inside
    // cudaStreamSynchronize; this stream lets the DMA run while the CPU
    // resolves the next expert, with expert_ffn_gpu_issue fencing only the
    // default compute stream on this slot's own upload_event. It MUST be
    // created non-blocking: every other stream here is a legacy blocking
    // stream, and a blocking demand stream would make the default stream
    // implicitly wait for every queued demand H2D, defeating the overlap.
    cudaStream_t active_layer_demand_stream = nullptr;

    // Persistent VRAM allocations
    uint8_t* expert_cache_pool = nullptr;  // For ExpertVault role
    size_t expert_cache_size = 0;

    // True once expert_cache_pool has been successfully cuFileBufRegister'd
    // (cufile_io.hpp) -- one registration call per cudaMalloc'd pool, not
    // per-expert-slot (see allocate_expert_cache/add_dual_role_expert_cache,
    // the two sites that set this). upload_expert_cufile checks this before
    // ever attempting a cuFileRead into a slot in THIS device's pool --
    // false is the safe default (falls back to RAM-mediated upload), never
    // assumed true just because cuFile is compiled in / requested.
    bool cufile_pool_registered = false;
};

enum class HardwareProfile {
    SingleGPU,    // Profile A: everything on one card
    DualGPU,      // Profile B: Core + Vault split
    TripleGPU     // Profile C: Core + spanned Vault
};

struct ExpertLocation {
    int gpu_id = -1;           // which GPU holds this expert (-1 = not cached)
    uint8_t* device_ptr = nullptr;  // pointer within that GPU's cache pool
    size_t offset = 0;         // offset within the pool (for slab reuse)
    size_t size = 0;
    // Static layout region owning this slot. In the legacy cache all slots
    // are interchangeable and this stays -1. In the compact cache each
    // layer gets slots sized to that layer's real packed expert slab, so
    // eviction must only reuse a slot from the requested layer's region.
    int pool_layer = -1;
    uint64_t last_used_clock = 0;
    uint32_t hit_count = 0;    // frequency signal for eviction scoring (power-law routing)
    int layer_idx = -1;
    int expert_id = -1;

    // Async promotion support: while an upload_expert_async() copy is in
    // flight, upload_ready is false and the slot must not be used for GPU
    // compute yet (device_ptr contents are only partially written).
    bool upload_ready = true;
    cudaEvent_t upload_event = nullptr;

    // True only while a DEMAND-time active-layer H2D issued by
    // upload_expert_active_async(demand=true) is still pending. Unlike a
    // speculative/prefill prewarm, this copy is needed by a real dispatch
    // that is already being queued: expert_ffn_gpu_issue inserts a
    // cudaStreamWaitEvent on the consumer stream before its matmuls, so the
    // CPU can keep resolving later experts while the DMA runs. The async
    // issue's own temp_pin_count hold keeps the slot alive until compute has
    // consumed it; release_hold() publishes upload_ready and drops that hold
    // after the compute synchronization. Cleared by every readiness path
    // (find_active_expert's poll, sync_active_layer_batch, explicit
    // completion) so it can never outlive the transfer.
    bool demand_async_pending = false;

    // Persistent per-active-slot CUDA events used to measure the actual H2D
    // interval.  They are never created in the per-transfer path.  The main
    // VRAM cache keeps using upload_event alone.
    cudaEvent_t upload_start_event = nullptr;
    bool upload_timing_pending = false;
    bool upload_timing_recorded = false;
    bool upload_timing_async = false;
    bool upload_source_pinned = false;
    size_t last_upload_bytes = 0;
    // Non-negative only for an active-layer upload issued by the Qwen
    // prediction path.  This lets the CUDA-event completion accounting split
    // predicted copies from generic active-layer prewarm without changing the
    // latter's behavior.
    int64_t qwen_pilot_prediction_ns = -1;
    int64_t qwen_pilot_issue_ns = -1;

    // 2026-08-28: guards a real race the worker-thread upload path
    // introduces (ActiveLayerUploadWorkers, active_layer_upload_workers.
    // hpp) that does NOT exist for any synchronous or same-thread-async
    // caller today. find_active_expert's poll trusts cudaEventQuery(
    // upload_event) -- but cudaEventQuery on an event that has NEVER been
    // recorded returns cudaSuccess immediately (CUDA's own documented
    // semantics: nothing to wait for). Every EXISTING caller issues
    // cudaMemcpyAsync + cudaEventRecord back-to-back on the SAME thread
    // before returning, so no poll could ever observe an unrecorded event
    // for a slot it's able to look up. A worker-thread hand-off breaks
    // that invariant: the decode thread marks this slot claimed
    // (upload_ready=false) and returns BEFORE a worker has necessarily
    // gotten around to the actual cudaMemcpyAsync/cudaEventRecord call --
    // a poll landing in that window would otherwise see "event query
    // succeeded" and wrongly treat an unstarted copy as complete. Default
    // true (matches every existing synchronous-record caller, zero
    // behavior change for them); set false ONLY when a slot is claimed
    // for worker-deferred dispatch, and flipped back to true by the
    // worker itself, under cache_mutex, immediately after it has
    // genuinely called cudaEventRecord -- never before. Every reader of
    // upload_event on active_layer_slots must check this FIRST, before
    // trusting cudaEventQuery/cudaEventSynchronize at all: find_active_
    // expert's poll, cleanup_active_layer_area's opportunistic stale-hold
    // release, AND upload_expert_active's own duplicate-transfer pre-check
    // (gpu_topology.cpp) -- the last of these was found MISSING this guard
    // on real hardware (a real, severe NaN-cascade corruption bug: a
    // cudaEventSynchronize on a not-yet-recorded event returns immediately,
    // spuriously marking a still-stale slot "ready") and fixed the same day
    // this field was added; see that fix's own comment for the full
    // bisection that found it.
    bool copy_dispatched = true;

    // Active-layer batch pre-warm only (upload_expert_active_async,
    // gpu_topology.cpp, 2026-08-12): set true when a slot is proactively
    // filled BEFORE any token's Phase-3 dispatch has touched it this
    // layer. Real-hardware A/B found pre-warming interacts badly with a
    // low active_layer_promote_hit_threshold (this config: 1) --
    // reactive upload_expert_active's own first touch was ALWAYS a
    // find_active_expert MISS (cheap, no promotion; promotion only fired
    // on a genuine SECOND touch by a different token needing the same
    // expert). Pre-warming makes the FIRST real touch a HIT instead,
    // immediately crossing threshold=1 and promoting EVERY pre-warmed
    // expert on first use, not just genuinely repeat-visited ones --
    // measured cost: vram_demote and find_active_expert's own scan cost
    // both roughly TRIPLED (1.2s->3.8-3.9s), more than erasing active_up's
    // own real 63% reduction (4.87s->1.78s), net prefill wall time WORSE
    // (16.28s->18.89s). Fixed by having find_active_expert's hit branch
    // skip counting this ONE upcoming hit toward the promote threshold
    // when this flag is set, then clearing it -- restores the original
    // "first touch is free" semantics pre-warming would otherwise silently
    // break, without changing hit_count accounting itself.
    bool prewarm_skip_next_promote = false;

    // Waste-measurement pair for decode_async_prewarm/prefill's Phase-2
    // batch pre-warm (2026-08-15) -- built specifically to answer "how much
    // of the ~205GB/request this feature moves is later actually used"
    // before spending any further engineering effort tuning HOW MUCH gets
    // pre-warmed (see t_active_layer_upload_async's own comment for why two
    // real transfer-mechanism fixes were tried and reverted, converging on
    // "the volume itself is the real cost, not the mechanism"). was_async_
    // prewarmed marks a slot's CURRENT content as having arrived via
    // upload_expert_active_async (never set by the reactive upload_expert_
    // active sync path -- that's already a confirmed real need, not a
    // speculative one). prewarm_touched is set true the instant find_active
    // _expert finds a real dispatch match for this slot's content,
    // regardless of hit_count/promotion outcome -- deliberately NOT the
    // same signal as hit_count==0, because decode_async_prewarm's own
    // prewarm_skip_next_promote branch (see find_active_expert's own
    // comment) intentionally does NOT increment hit_count on a prewarmed
    // slot's first real touch, which is exactly the common, INTENDED,
    // successful case this counter must not misclassify as "wasted".
    // Checked (and both reset) at every reclaim site (cleanup_active_layer_
    // area, upload_expert_active_async's own victim-eviction of an occupied
    // slot, upload_expert_active's equivalent, and a successful promotion)
    // -- was_async_prewarmed && !prewarm_touched at reclaim time means the
    // data was fetched but no real dispatch ever needed it before it was
    // discarded.
    bool was_async_prewarmed = false;
    bool prewarm_touched = false;

    // Speculative pilot pre-fill only (EngineConfig::speculative_promote_to_
    // active_layer, 2026-08-26, default off): set true when a slot is
    // proactively filled by DEPTH-1/ATTN-PILOT's one-layer-ahead prediction
    // (RAMSafetyManager::promote_ram_to_active_layer), BEFORE any real
    // dispatch has confirmed this expert is actually needed -- structurally
    // the same role as was_async_prewarmed above (skip-next-promote,
    // waste-tracked via prewarm_touched), but kept as its own separate
    // flag/counter pair rather than reusing was_async_prewarmed directly:
    // this is a genuinely different source (a PREDICTION, not a real
    // router's already-computed, 100%-confirmed-need batch the way
    // decode_async_prewarm's own pre-warm is) -- conflating the two would
    // hide which source actually dominates a run's waste, the same
    // "don't share buckets across structurally distinct call sites"
    // discipline this project already follows elsewhere (see
    // t_active_layer_upload_async's own comment). record_active_layer_
    // prewarm_reclaim checks this alongside was_async_prewarmed and resets
    // both, so no new reclaim call sites were needed.
    bool pilot_speculative_fill = false;

    // 2026-08-28: which pilot RANK (0-7, the k in DEPTH-1/ATTN-PILOT's own
    // per-transition k-loop, execution_layer.cpp) caused THIS speculative
    // fill -- set only alongside pilot_speculative_fill above, -1
    // otherwise. Exists specifically to answer "of the MoEs promoted at
    // rank K specifically, what fraction were actually touched by a real
    // dispatch before reclaim" -- a genuinely different question from the
    // pilot's own per-rank PREDICTION-accuracy metric (n_pilot_ram_only_
    // hits_by_rank, which compares a prediction against the real router's
    // outcome regardless of whether that prediction was ever promoted or
    // used). record_active_layer_prewarm_reclaim reads this alongside
    // prewarm_touched to attribute the used/wasted outcome to the correct
    // rank, then resets it to -1.
    int8_t promoted_at_rank = -1;

    // Phase A a pinned slot is never chosen as an eviction
    // victim, regardless of eviction_score. Set once a RAM-cache expert's
    // hit_count crosses RAMSafetyManager::pin_threshold (proven-hot within
    // this conversation), or at startup for the pre-loaded hot set from a
    // usage-file profile (cross-conversation). Only mutated under
    // cache_mutex, same as every other field here. PERMANENT -- never
    // cleared once set (except defensively when a slot is freed/reused).
    bool pinned = false;

    // Reference-counted, TEMPORARY eviction protection -- distinct from
    // `pinned` above, which is permanent. Held for the scope of exactly one
    // decode-time GPU dispatch (see expert_ffn_resolve_gpu_loc's cold-miss
    // branch in execution_layer.cpp): incremented right after a synchronous
    // upload_expert() call confirms this slot's data is present, decremented
    // only after expert_ffn_gpu_finish's cudaStreamSynchronize(0) confirms
    // every kernel reading this slot (gate/up/down) has actually finished.
    // Reference-counted rather than a boolean so it can never incorrectly
    // un-protect a slot that's ALSO independently, permanently `pinned`, and
    // so nothing breaks if this is ever called re-entrantly for the same
    // slot. A slot with temp_pin_count > 0 must never be chosen as an
    // eviction victim, same as `pinned` -- see evict_lru.
    int temp_pin_count = 0;

    // Set once a cudaMemcpy H2D into this slot's device_ptr has genuinely
    // failed (see upload_expert's error path). A real run showed evict_lru
    // repeatedly re-selecting the SAME still-"empty" slot after a failed
    // write left it looking eligible again -- ~1800 preload attempts all
    // funneled into one broken address instead of ever trying a different
    // slot, since a failed upload never marks the slot occupied. Once set,
    // this slot is permanently excluded from both the "prefer empty slot"
    // fast path and the scored eviction scan, for the rest of the process
    // (this is a hardware/driver-level or address-computation problem, not
    // something that resolves itself by retrying) -- so the OTHER, likely-
    // healthy slots in the pool get a real chance instead of the whole
    // pool's effective capacity collapsing to zero the moment one address
    // goes bad. Never cleared, mirroring `pinned`'s "permanent" convention.
    bool memcpy_broken = false;
    // 2026-09-01: how many times an H2D copy into THIS slot has failed.
    // memcpy_broken is now only set at >=2, because a single failure does
    // NOT prove the slot is at fault: a bad HOST pointer fails against
    // whatever destination it is handed, so it produces one failure each on
    // several DIFFERENT healthy slots (observed on a real run: one host_src
    // failing against three distinct dst, retiring three good slots). A
    // genuinely faulty VRAM address, by contrast, fails repeatedly on the
    // SAME slot and still trips the guard on its second attempt.
    int memcpy_fail_count = 0;

    // Blended recency+frequency score used for eviction: a frequently-hit
    // expert resists eviction even after a burst of newer accesses touches
    // other slots (plain LRU would evict a hot-but-not-most-recent expert).
    uint64_t eviction_score() const {
        return last_used_clock + (uint64_t)hit_count * FREQ_BONUS_TICKS;
    }
    static constexpr uint64_t FREQ_BONUS_TICKS = 64;

    // 2026-08-26 (user-requested): a speculative pilot fill (pilot_
    // speculative_fill=true) that has NEVER yet been confirmed by a real
    // dispatch (prewarm_touched still false) is still just a guess -- it
    // should always be preferred as an active-layer eviction victim over
    // ANY other content, real or speculative-but-already-confirmed,
    // regardless of how its own eviction_score() compares. Once a real
    // dispatch touches it (prewarm_touched flips true, see find_active_
    // expert), it's proven useful and competes on normal eviction_score
    // footing from then on, same as everything else -- this is NOT a
    // permanent "speculative content is always cheap" rule, only "an
    // UNCONFIRMED guess is always cheap". Mirrors also_in_vram's own
    // "force to the front of the eviction queue" trick (ram_manager.hpp),
    // applied here to the active-layer pool's own internal victim
    // selection instead of the RAM tier's.
    bool is_unconfirmed_speculative_fill() const {
        return pilot_speculative_fill && !prewarm_touched;
    }

    // Comparator for active-layer victim-selection scans: does `this`
    // deserve to be picked as the victim ahead of `other`? An unconfirmed
    // speculative fill always outranks anything that isn't also one;
    // between two candidates of the same kind, falls back to the normal
    // recency+frequency eviction_score().
    bool preferred_victim_over(const ExpertLocation& other) const {
        bool a = is_unconfirmed_speculative_fill(), b = other.is_unconfirmed_speculative_fill();
        if (a != b) return a;
        return eviction_score() < other.eviction_score();
    }
};

class GpuTopology {
public:
    std::vector<GpuDevice> devices;
    HardwareProfile profile = HardwareProfile::SingleGPU;
    int core_gpu = -1;         // device index for CoreEngine
    std::vector<int> vault_gpus; // device indices for ExpertVault(s)

    // Real CUDA device_id (nvidia-smi ordinal, NOT a devices[] index -- that
    // index doesn't exist yet when main.cpp needs to set this, before
    // topology.initialize() has even enumerated devices[]) the config wants
    // as CoreEngine, from config.gpus[dense_gpu_idx].device_id -- i.e. the
    // SAME source of truth CudaManager::device_id was already (incorrectly)
    // being seeded from. -1 means "no explicit preference, use the SM-
    // architecture/VRAM auto-detect heuristic" (the only behavior that
    // existed before this field). Set by main.cpp BEFORE calling
    // gpu.initialize(), consulted by assign_roles() once devices[] exists.
    //
    // Was previously impossible to honor: assign_roles() took no parameters
    // at all and unconditionally picked whichever device had the newest SM
    // architecture as CoreEngine -- a real run showed a user's explicit
    // role="dense" on the OLDER-architecture GPU (an Ampere 3090, vs a
    // Blackwell 5060 Ti configured role="moe") get silently overridden:
    // CudaManager::initialize() sets device_id correctly from config, then
    // IMMEDIATELY overwrites it one line later with topology.core_gpu's
    // auto-detected result, discarding the config-correct value. The TOML's
    // per-GPU role field was fully parsed and even correctly logged
    // ("[CONFIG] GPU 0 ... — dense"), just never actually consulted by the
    // one function that decides which physical GPU plays which role.
    int forced_core_device_id = -1;

    // Optional allow-list supplied by EngineConfig. CUDA still enumerates all
    // physical adapters (the runtime requires that), but only these device
    // IDs may receive a MoEStream role or allocation. An entry omitted from
    // the TOML is therefore genuinely unused rather than silently becoming
    // an ExpertVault through the old hardware-count heuristic. Empty keeps
    // the legacy "use every adapter" behavior for callers that do not use a
    // config file.
    std::vector<int> configured_device_ids;

    // Expert VRAM LRU cache: maps (layer, expert_id) -> location
    std::vector<ExpertLocation> expert_vram_cache;
    size_t expert_vram_cache_cap = 0;  // max entries
    // Nonzero only for the legacy uniform-stride layout. A compact layout
    // deliberately sets this to zero so callers cannot assume that every
    // layer's slots are equally sized or physically consecutive.
    size_t expert_slab_size = 0;
    size_t expert_max_slab_size = 0;   // largest real/aligned slab, for staging
    int expert_n_experts = 0;
    std::vector<size_t> expert_layer_slab_bytes;
    // Explicit CPU-affinity layers do not dispatch routed experts to VRAM.
    // Retain the list so compact-cache reindexing after a topology update
    // preserves the same redistributed physical ownership.
    std::vector<int> expert_cpu_affinity_layers;
    bool expert_compact_layout = false;
    int expert_full_layer_prefix = 0;
    uint64_t cache_clock = 0;

    // ========== Three-tier caching: Active layer area ==========
    // Active layer area: temporary workspace for current-layer experts
    // Reclaimed-per-request, never participates in LRU eviction
    uint8_t* active_layer_pool = nullptr;
    size_t active_layer_capacity = 0;     // Maximum experts active layer can hold
    std::vector<ExpertLocation> active_layer_slots;  // Pre-allocated slot metadata

    // 2026-08-28: multi-GPU active-layer pool routing. Each participating
    // GPU gets its OWN cudaMalloc allocation (own ActiveLayerGpuSlice
    // entry); active_layer_slots above stays ONE combined vector spanning
    // ALL participating GPUs' slots contiguously (this GPU's slots first,
    // that GPU's next, etc.) -- find_active_expert/active_layer_has/
    // cleanup_active_layer_area continue to do a flat linear scan over the
    // WHOLE vector, completely unchanged, since residency lookup doesn't
    // care which GPU a slot lives on (a slot already carries its own
    // gpu_id, same as every other ExpertLocation). Only allocation and
    // victim-selection need to be GPU-aware -- see allocate_active_layer_
    // area_multi and upload_expert_active(_async)'s new target_gpu_id
    // parameter. Empty = legacy single-pool mode (today's behavior,
    // byte-identical) -- active_layer_pool/active_layer_capacity above are
    // still populated in that case too, exactly as before, so any code
    // that still reads them directly (there is none left after this
    // change, but kept for safety) continues to work.
    struct ActiveLayerGpuSlice {
        int gpu_id = -1;
        int weight = 0;
        size_t slot_start = 0;   // first index into active_layer_slots owned by this GPU
        size_t slot_count = 0;   // contiguous slot count this GPU owns
        uint8_t* pool_base = nullptr;  // this GPU's own cudaMalloc'd base pointer
    };
    std::vector<ActiveLayerGpuSlice> active_layer_gpu_slices;

    // Hit count (while resident in the active layer area) at which an
    // expert gets promoted into the real VRAM LRU cache (device-to-device
    // copy, see promote_active_to_main) instead of just being freed once
    // it's no longer this pass's current layer. Set from EngineConfig::
    // active_layer_promote_hit_threshold in main.cpp. Default 1 (promote on
    // ANY repeat access) rather than mirroring RAMSafetyManager's
    // promote_hit_threshold=1-8 range: the active layer area only holds
    // decode-time cold misses for the CURRENT layer, a much narrower window
    // than the RAM tier's whole-conversation lifetime, so a repeat hit here
    // is already a comparatively strong signal.
    int active_layer_promote_threshold = 1;

    // EngineConfig::decode_promote_all_misses: at a decode layer's end, promote
    // EVERY real active-layer miss into the main VRAM cache instead of only the
    // ones with a repeat hit. In decode an active slot is freed when its layer
    // window closes, long before the same expert can be touched again, so the
    // repeat-hit rule never fires and the main cache stays frozen at its
    // startup contents (measured: 0 hits / 0 promotions over a whole decode).
    // Promotions are COPY-style (the RAM tier keeps its slab: on_promoted is
    // not called) and take their victim from the SAME GPU as the active slot,
    // so the copy is a plain device-to-device one with no host bounce.
    bool decode_promote_all_misses = false;

    // Mirrors RAMSafetyManager::in_prefill (same set site, execution_layer.
    // cpp's layers_forward_gguf, S>1 = prefill). REAL-HARDWARE FINDING
    // (2026-08-12): active_layer_promote_threshold=1 means literally any
    // second touch of a pre-warmed/reactively-uploaded active-layer expert
    // promotes it into the main VRAM cache -- fine at decode's steady one-
    // token-at-a-time pace, but during prefill's own dense multi-token
    // burst this fires constantly, and since the main cache is ALREADY
    // essentially 100% full by partway through every prefill pass, each
    // promotion forces a real eviction (vram_demote) of some OTHER
    // resident to make room. Measured: vram_demote and find_active_
    // expert's own scan cost both climbed to ~40% of prefill's total wall
    // time chasing this churn, repeatedly, across several attempted
    // mitigations (see CLAUDE.md) that each still allowed SOME promotion
    // during prefill. The actual fix: don't promote from the active layer
    // into the main cache AT ALL while in_prefill is true -- let prefill's
    // own experts live in the active-layer pool (a real, separate ~256-
    // slot VRAM allowance that doesn't compete with the main cache's fixed
    // budget at all) for the whole prefill pass, protecting decode's own
    // pinned/hot main-cache content from any prefill-driven churn by
    // construction, not by chasing every downstream side effect of
    // letting it happen. cleanup_active_layer_area's own end-of-layer
    // reclaim already frees/reuses active-layer slots every layer
    // regardless (no explicit prefill-specific swap-out step needed --
    // nothing prefill-only ever reaches the main cache in the first
    // place). Decode's own promotion behavior (in_prefill=false) is
    // completely unchanged -- this only gates the prefill case.
    bool in_prefill = false;
    // Layer currently being executed by the batched prefill. When set,
    // evict_lru prefers unpinned experts from completed layers before it
    // considers current/future-layer residents.
    int prefill_current_layer = -1;
    // Suppresses hit accounting while the startup manifest itself is being
    // replayed. The preload loop calls upload_expert again for duplicates;
    // those cache lookups are setup work, not routed prompt usage.
    bool startup_preload_phase = false;

    // Speculative pilot pre-fill (EngineConfig::speculative_promote_to_
    // active_layer, 2026-08-26, default off, set once by main.cpp). When
    // true, cleanup_active_layer_area additionally protects slots tagged
    // for current_layer_idx+1 -- a speculative fill made one layer ahead of
    // when it will actually be dispatched (RAMSafetyManager::promote_ram_
    // to_active_layer, called from execution_layer.cpp's l+1 pilot-
    // prediction loop DURING layer l's own processing, before layer l+1
    // has even started). Without this widening, such an entry would be
    // wiped out by THIS SAME layer's own end-of-layer cleanup call the
    // instant it's inserted (tagged l+1, which never matches
    // current_layer_idx==l) -- before layer l+1's real dispatch ever gets a
    // chance to look it up. Safe to be a plain, unconditional-looking "+1"
    // match in cleanup_active_layer_area's own code (not itself re-checking
    // this flag there) BECAUSE nothing is ever tagged current_layer_idx+1
    // in the first place unless this flag is on -- see promote_ram_to_
    // active_layer's own gate. With this on, a speculative l+1 entry gets
    // the exact same real lifetime a genuine l+1 active-layer entry already
    // gets (survives through l's cleanup via the +1 exemption, then through
    // l+1's own cleanup via a direct match, reclaimed only at l+2's
    // cleanup if nothing consumed it) -- never touches the main VRAM cache
    // at all unless a real repeat hit later promotes it, exactly mirroring
    // decode_async_prewarm's own prewarm_skip_next_promote discipline.
    bool speculative_active_layer_fill = false;

    // Periodic decay of hit_count so a once-hot expert doesn't stay
    // permanently sticky after routing patterns shift (e.g. topic change).
    static constexpr uint32_t HIT_DECAY_INTERVAL = 4096;
    uint64_t hits_since_decay = 0;

    // Guards all reads/writes of expert_vram_cache slot state (layer_idx,
    // expert_id, upload_ready, last_used_clock, hit_count) — the prefetcher
    // worker thread and the main decode thread both touch these with no
    // prior synchronization, which is a real data race (two threads can
    // claim/relabel the same slot concurrently). Recursive because
    // upload_expert()/upload_expert_async() call find_expert()/evict_lru()
    // internally on the same thread while already holding the lock.
    mutable std::recursive_mutex cache_mutex;

    // Cache-churn diagnostics: per-(layer,expert) swap in/out counts for the
    // VRAM tier, so a shutdown report can answer "which experts are being
    // repeatedly evicted and reloaded, and are they pinned or not".
    // Keyed dynamically (not a dense n_layers*
    // n_experts array like StreamingEngine::route_freq) since GpuTopology
    // doesn't know those dimensions itself. Mutated only from evict_lru/
    // upload_expert/upload_expert_async, all of which already hold
    // cache_mutex for their whole body (GPU_TOPO_TIMED_LOCK), so no
    // additional synchronization is needed here.
    struct SwapStats { uint64_t swap_in = 0; uint64_t swap_out = 0; };
    std::unordered_map<uint64_t, SwapStats> vram_swap_stats;

    // Startup-preload accounting. The normal swap counters say how much
    // churn occurred, but cannot answer whether a preloaded identity was
    // ever actually routed before it was evicted. These records are marked
    // by main.cpp after a successful startup upload, touched by
    // finish_expert_hit(), and finalized by evict_lru(). All mutation sites
    // already hold cache_mutex, except the startup-only mark call.
    struct StartupPreloadStats {
        bool used = false;
        bool eviction_counted = false;
    };
    std::unordered_map<uint64_t, StartupPreloadStats> startup_preload_stats;
    uint64_t startup_preload_marked = 0;
    uint64_t startup_preload_used = 0;
    uint64_t startup_preload_evicted_before_use = 0;
    uint64_t startup_preload_evicted_before_use_prefill = 0;
    uint64_t startup_preload_evicted_after_use = 0;
    uint64_t startup_preload_evicted_after_use_prefill = 0;

    void mark_startup_preloaded(int layer, int expert_id) {
        const uint64_t key = swap_key(layer, expert_id);
        const auto result = startup_preload_stats.emplace(key, StartupPreloadStats{});
        if (result.second) ++startup_preload_marked;
    }

    void note_startup_preload_hit(int layer, int expert_id) {
        if (startup_preload_phase) return;
        auto it = startup_preload_stats.find(swap_key(layer, expert_id));
        if (it != startup_preload_stats.end() && !it->second.used) {
            it->second.used = true;
            ++startup_preload_used;
        }
    }

    void note_startup_preload_eviction(int layer, int expert_id) {
        auto it = startup_preload_stats.find(swap_key(layer, expert_id));
        if (it == startup_preload_stats.end() || it->second.eviction_counted)
            return;
        it->second.eviction_counted = true;
        if (it->second.used) {
            ++startup_preload_evicted_after_use;
            if (in_prefill) ++startup_preload_evicted_after_use_prefill;
        } else {
            ++startup_preload_evicted_before_use;
            if (in_prefill) ++startup_preload_evicted_before_use_prefill;
        }
    }

    static uint64_t swap_key(int layer, int expert_id) {
        return ((uint64_t)(uint32_t)layer << 32) | (uint32_t)expert_id;
    }

    // Optional VRAM-eviction salvage hook (vram_evict_demote_to_ram): called
    // by upload_expert() right after evict_lru() picks a REAL victim (one
    // that held a genuine prior (layer,expert_id) identity) and BEFORE that
    // slot's device_ptr is overwritten by the incoming H2D copy -- this is
    // the last moment the outgoing expert's bytes still exist anywhere.
    // Set once by main.cpp, after both GpuTopology and RAMSafetyManager
    // exist, to a lambda that D2H-copies device_ptr into a RAM-tier slot
    // (RAMSafetyManager::demote_from_vram) instead of letting the data be
    // silently dropped the way evict_lru always did before this feature.
    //
    // Deliberately wired into upload_expert() ONLY, never
    // upload_expert_async(). RAMSafetyManager::layers_experts has a hard,
    // load-bearing invariant documented on its own SwapStats member: it is
    // "mutated only from get_expert/get_experts_batch, both called
    // exclusively from the single decode thread ... never the prefetcher
    // worker thread -- so no lock is needed". upload_expert_async is the
    // PREFETCHER thread's own speculative-promotion path; routing a
    // layers_experts write through it would introduce a brand-new writer
    // thread to a structure with zero synchronization today, and this
    // project has already been burned multiple times (see CLAUDE.md) by
    // races introduced in exactly this "VRAM eviction / cross-thread
    // reuse" territory. upload_expert, by contrast, is only ever called
    // from the decode thread (decode-time cold GPU dispatch, prefill's
    // S>=4 cold-batch path, and the startup [PIN] preload block all run on
    // the main thread) -- so wiring the salvage there needs no new locking
    // at all, at the cost of not covering the prefetcher's own async
    // promotion evictions (a real, documented, accepted gap -- see
    // ram_manager.hpp's demote_from_vram).
    std::function<void(int layer, int expert_id, const uint8_t* device_ptr,
                        size_t bytes, int gpu_id)> on_vram_evict = nullptr;

    // O(1) identity -> slot-index lookup, replacing find_expert's linear
    // scan over expert_vram_cache (up to expert_vram_cache_cap entries,
    // e.g. 1642 in real runs) with a hash lookup. Deliberately maps to a
    // SLOT INDEX, not a raw ExpertLocation*/address+size copy:
    // expert_vram_cache[idx] stays the single source of truth for
    // address/size/tier/state (upload_ready, pinned, temp_pin_count) --
    // this map only answers "which slot, if any, currently claims to hold
    // (layer, expert_id)". Same key encoding as vram_swap_stats
    // (swap_key) for consistency; this map's lifecycle is different
    // though -- vram_swap_stats accumulates forever, this one is kept in
    // sync with real occupancy (inserted on upload_expert/
    // upload_expert_async, erased on evict_lru claiming a real victim).
    //
    // Scope: converting expert_vram_cache's linear scans to O(1) lookups.
    // find_expert (the highest-traffic read site) was converted first;
    // hold_for_compute/release_hold were converted next, once find_expert's
    // pattern was proven correct on real hardware -- a real run found their
    // OWN unconverted O(n) scans (2x per GPU-dispatched expert, ~63K scans
    // of a 1322-slot vector in one 40-token request) were the dominant
    // remaining cause of a large expert=/gpu=+cpu= timing gap, entirely
    // invisible to lock-contention timers since this is uncontended
    // CPU work done WHILE holding the lock, not time spent waiting for it
    // -- see their own comments for the real numbers. pin_expert/
    // decay_hit_counts still do their own linear scans -- both are much
    // lower-frequency (pin_expert only on a promotion-to-pin event,
    // decay_hit_counts only every HIT_DECAY_INTERVAL hits), so converting
    // them is lower priority; do so if they're ever shown to matter the
    // same way. evict_lru's own job (find the MINIMUM eviction_score across
    // all slots) is a different access pattern this index doesn't help at
    // all -- it stays a linear scan regardless of any future index work.
    //
    // Safety design: find_expert NEVER trusts an index hit blindly -- it
    // re-validates the resolved slot's actual (layer_idx, expert_id,
    // device_ptr) against what was asked for before returning it. If a
    // hit doesn't validate (the index and expert_vram_cache have
    // desynced -- should be impossible by construction given the two
    // maintenance sites above, but a maintenance-site bug is exactly the
    // failure class this project has been burned by before), it's logged
    // loudly (perf().n_stale_expert_index_hits, a shutdown-visible
    // counter) and find_expert falls back to the original linear scan,
    // which also opportunistically repairs the index entry it finds --
    // so even a completely broken index degrades to "no worse/slower
    // than before this change", never to wrong data.
    std::unordered_map<uint64_t, size_t> expert_index;

    // evict_lru optimization -- DIFFERENT mechanism from expert_index above,
    // not contradicted by that comment's "evict_lru's own job ... stays a
    // linear scan regardless of any future index work": expert_index can't
    // help evict_lru because finding the MINIMUM eviction_score genuinely
    // requires comparing every candidate, but the SET of candidates worth
    // comparing can be shrunk -- a PERMANENTLY pinned slot (ExpertLocation::
    // pinned, never un-set once true) can NEVER be a valid victim, so there
    // is no reason for evict_lru to visit it at all, every single call.
    // Real measured cost this was added to fix: a run with 600 pinned slots
    // out of 1642 logged 15,191,100 n_pinned_eviction_skips across 24,384
    // real evictions (~600/eviction, matching exactly) -- i.e. every single
    // evict_lru call was paying to scan-and-skip the SAME 600 slots, a
    // direct, measured contributor to Cache mutex wait (evict_lru runs
    // under cache_mutex).
    //
    // Holds every slot index NOT currently permanently pinned. Rebuilt from
    // scratch in reindex_expert_vram_cache (safe: that function only ever
    // runs pre-pinning, before prefetcher.start()/the [PIN] block, so there
    // is never partial pin state to preserve across a rebuild -- confirmed
    // by call-site order in main.cpp). Shrunk (swap-and-pop, order doesn't
    // matter for a scan) by pin_expert the moment a slot's pinned flag
    // flips true. Never needs to grow back, since pinning is permanent --
    // no unpin() exists in this codebase.
    std::vector<size_t> unpinned_slot_indices;

    // Prefill ring time-shares the main GPU's expert cache.  The cache is
    // copied to host and its CUDA pool is freed before the prompt-shaped
    // graph allocates the speculative ring; FullGraph::destroy restores it
    // after the ring buffers are released, so decode still gets the full
    // configured cache.  Keeping the copy on the host is what lets the ring
    // use the cache's VRAM without permanently growing either allocation.
    int detached_cache_device = -1;
    size_t detached_cache_bytes = 0;
    void *detached_cache_host = nullptr;
    bool detached_cache_host_reserved = false;
    std::vector<std::pair<size_t, ExpertLocation>> detached_cache_slots;

    // Pinned host buffer for cross-device transfers
    float* h_xfer_pinned = nullptr;
    size_t h_xfer_cap = 0;
    size_t host_pinned_reservation_bytes = 0;

    // Fixed-size pinned staging pool for expert H2D uploads (upload_expert/
    // upload_expert_async) -- see init_bounce_pool's own comment for why
    // this is a small, never-resized pool rather than pinning the RAM
    // tier's own slab_buffer directly.
    struct BounceBuffer {
        uint8_t* host_ptr = nullptr;
        cudaEvent_t in_flight = nullptr;  // records the last DMA that read this buffer
    };
    static constexpr int BOUNCE_POOL_SIZE = 4;
    std::vector<BounceBuffer> bounce_pool;
    size_t bounce_next = 0;

    // upload_expert_async's cudaHostRegister conversion (2026-08-13, Alt A,
    // same mechanism as upload_expert's own -- see that function's comment
    // for the real-hardware numbers that motivated it there). Unlike
    // upload_expert, this function's DMA is still in flight when it
    // returns, so host_data can't be unregistered synchronously -- doing
    // so would risk unregistering pinned memory a still-in-flight DMA is
    // reading from, the exact class of "completion timing" mistake this
    // codebase's own documented decode-path corruption incident already
    // taught it to avoid. Deferred instead: each successful registration's
    // (host pointer, slot's upload_event) pair is queued here, and
    // reap_pending_host_unregisters() -- called opportunistically, with
    // `lock` already held, at the top of every upload_expert_async call --
    // unregisters only the ones whose event has ACTUALLY completed
    // (cudaEventQuery == cudaSuccess), never speculatively. Safe to check
    // slot->upload_event even though it's a per-SLOT event that may be
    // re-recorded by a LATER, unrelated promotion before this entry is
    // reaped: cudaMemcpyAsync + cudaEventRecord are both issued on the same
    // dev.copy_stream in program order, and CUDA streams execute enqueued
    // work in issue order, so confirming ANY recording of this event from
    // this point forward (this one or a later one) also proves THIS entry's
    // own earlier DMA -- issued strictly before that recording on the same
    // stream -- has already completed. Never records the same host pointer
    // twice while a prior registration for it is still pending (see the
    // guard in upload_expert_async itself) -- cudaHostRegister on an
    // already-registered range fails, which would otherwise misattribute a
    // failure to a genuinely fine call.
    struct PendingHostUnregister { void* host_ptr; cudaEvent_t event; };
    std::vector<PendingHostUnregister> pending_host_unregisters;

    // Drains pending_host_unregisters of every entry whose DMA has
    // genuinely completed. Caller must already hold `lock` (both call
    // sites -- upload_expert_async and the destructor -- satisfy this).
    void reap_pending_host_unregisters() {
        size_t write = 0;
        for (size_t i = 0; i < pending_host_unregisters.size(); ++i) {
            auto& p = pending_host_unregisters[i];
            if (cudaEventQuery(p.event) == cudaSuccess) {
                PERF_START(upload_async_host_unregister);
                cudaHostUnregister(p.host_ptr);
                PERF_END(upload_async_host_unregister, t_upload_async_host_unregister);
                perf().n_upload_async_deferred_unregister_reaped.fetch_add(1, std::memory_order_relaxed);
            } else {
                pending_host_unregisters[write++] = p;
            }
        }
        pending_host_unregisters.resize(write);
    }

    bool initialize() {
        int device_count = 0;
        if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
            LOG_STREAM_ERROR() << "[GPU] No CUDA devices found.\n";
            return false;
        }

        devices.resize(device_count);
        for (int i = 0; i < device_count; ++i) {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, i);
            devices[i].id = i;
            devices[i].name = prop.name;
            devices[i].total_vram = prop.totalGlobalMem;
            devices[i].available_vram = prop.totalGlobalMem; // will be reduced as we allocate
            devices[i].sm_major = prop.major;
            devices[i].sm_minor = prop.minor;
        }

        // Assign roles based on hardware profile
        assign_roles();

        // Create streams for each active device
        for (auto& dev : devices) {
            if (dev.role == GpuRole::Unused) continue;
            cudaSetDevice(dev.id);
            // Found via a real perf record/report CPU profile: ~40-60% of
            // ALL sampled CPU cycles across the whole process landed in
            // unsymbolicated code (unresolved even after a debug-symbol
            // rebuild, so almost certainly the closed-source CUDA driver,
            // not this project's own code), tightly correlated with
            // cudaStreamSynchronize call chains ending in repeated
            // clock_gettime calls (83 occurrences in one profile) -- the
            // textbook signature of the driver's DEFAULT spin-wait/polling
            // synchronization mode busy-looping a CPU core while waiting
            // for a stream to finish, rather than actually blocking. This
            // must be set BEFORE the device's CUDA context is created (the
            // very next line, cudaStreamCreate, is that first context-
            // creating call in this whole program -- cudaGetDeviceProperties
            // above does not create one) or it silently has no effect.
            // Trades slightly higher per-sync wake-up latency (a real OS
            // blocking wait instead of a spin) for freeing up the CPU
            // entirely while waiting -- a good trade here specifically
            // since the CPU is already a contended resource elsewhere in
            // this pipeline (OMP dequant work, the prefetcher thread
            // previously diagnosed as CPU-starved) and this codebase's own
            // synchronous cudaStreamSynchronize calls (dense_matmul_host,
            // expert_ffn_gpu_finish) already correctly measure their own
            // wall-clock duration either way, so no existing timer numbers
            // become invalid -- only the CPU-side cost of THAT wait changes.
            // REVERSED (2026-09-18): the BlockingSync-by-default reasoning
            // above was measured on the legacy per-layer dispatch path
            // (many synchronous cudaStreamSynchronize calls interleaved with
            // CPU-contended OMP dequant work), and that finding still holds
            // there. But on the full-graph Qwen4Exp decode path (ggml CUDA
            // graph capture/replay, see MOESTREAM_ENABLE_CUDA_GRAPHS),
            // BlockingSync's per-call OS wake-up/context-switch latency
            // (~40us median) was directly measured -- via nsys cuda_api_sum,
            // matched 22/48-GPU-resident-layer config against llama.cpp's own
            // reference (~3.9us median for the equivalent call) -- to cost
            // ~4-5ms of every ~50ms decode step with NO corresponding GPU
            // work to show for it (kernel-busy + memcpy time per token
            // together only account for ~1.8ms of that gap). Switching to
            // Auto closed the entire remaining throughput gap to llama.cpp on
            // that path: full-graph decode went from ~20 tok/s to ~26 tok/s
            // (llama.cpp's own reference: ~25.2 tok/s), reproduced twice.
            // llama.cpp itself never overrides CUDA scheduling mode at all,
            // i.e. its equivalent is already cudaDeviceScheduleAuto.
            // Defaulting to Auto here matches that and wins on the now-
            // primary full-graph path; MOESTREAM_CUDA_SCHEDULE=blocking
            // remains available for anyone who needs to A/B the legacy path
            // against the old default.
            const char * schedule_env = std::getenv("MOESTREAM_CUDA_SCHEDULE");
            // Host-node mode (MOESTREAM_QWEN4EXP_HOST_NODE=1) runs the CPU MoE from a CUDA host callback while the
            // generation thread waits in cudaStreamSynchronize. With the Auto (spin) schedule that waiting thread
            // burns a logical CPU the 12-thread OpenMP team needs, stalling every barrier: measured 18.2 vs 22.6
            // tok/s (+24%) with a blocking wait (see tests/option_b_split_reverify_howto.md section 16). An explicit
            // MOESTREAM_CUDA_SCHEDULE still wins; every other mode keeps Auto.
            const char * host_node_env = std::getenv("MOESTREAM_QWEN4EXP_HOST_NODE");
            if ((!schedule_env || !schedule_env[0]) && host_node_env && std::strcmp(host_node_env, "1") == 0)
                schedule_env = "blocking";
            unsigned schedule_flags = cudaDeviceScheduleAuto;
            const char * schedule_name = "auto";
            if (schedule_env && std::strcmp(schedule_env, "blocking") == 0) {
                schedule_flags = cudaDeviceScheduleBlockingSync;
                schedule_name = "blocking";
            } else if (schedule_env && std::strcmp(schedule_env, "spin") == 0) {
                schedule_flags = cudaDeviceScheduleSpin;
                schedule_name = "spin";
            }
            const cudaError_t schedule_err = cudaSetDeviceFlags(schedule_flags);
            if (schedule_err != cudaSuccess) {
                LOG_WARN("[CUDA-SCHEDULE] device=%d mode=%s set failed: %s\n",
                         dev.id, schedule_name, cudaGetErrorString(schedule_err));
            } else {
                LOG_INFO("[CUDA-SCHEDULE] device=%d mode=%s\n", dev.id, schedule_name);
            }
            cudaStreamCreate(&dev.compute_stream);
            cudaStreamCreate(&dev.copy_stream);
            cudaStreamCreate(&dev.active_layer_stream);
            cudaStreamCreate(&dev.speculative_prefetch_stream);
            cudaStreamCreate(&dev.speculative_prefetch_stream2);
            cudaStreamCreate(&dev.active_layer_sync_fallback_stream);
            // Non-blocking by construction -- see this field's own comment.
            cudaStreamCreateWithFlags(&dev.active_layer_demand_stream,
                                      cudaStreamNonBlocking);
        }

        // Allocate pinned host transfer buffer (256 KB for small cross-GPU
        // vectors).  CUDA host allocations are real host memory and are not
        // visible to the RAM-slab accounting, so charge them to the shared
        // process budget before asking the driver for the pages.
        constexpr size_t kXferBytes = 256 * 1024;
        const bool xfer_admitted = MemoryGuard::try_reserve_host_bytes(kXferBytes);
        if (xfer_admitted) {
            if (cudaMallocHost((void**)&h_xfer_pinned, kXferBytes) == cudaSuccess &&
                h_xfer_pinned) {
                h_xfer_cap = kXferBytes;
                host_pinned_reservation_bytes += kXferBytes;
            } else {
                cudaGetLastError();
                MemoryGuard::release_host_bytes(kXferBytes);
            }
        } else {
            LOG_STREAM_WARN() << "[MEM-GUARD] cross-device pinned transfer buffer refused\n";
        }

        print_topology();
        return true;
    }

    ~GpuTopology() {
        // Drain pending_host_unregisters BEFORE the upload_event destroy
        // loop below -- reap_pending_host_unregisters() reads those same
        // events to decide what's safe to unregister, so it must run while
        // they're still valid. A normal, orderly shutdown should have
        // little or nothing left pending here (every real DMA had ample
        // time to complete); any genuine leftovers are force-unregistered
        // afterward rather than leaked, since the process teardown below
        // (cudaFree on the pools these pointers' DMAs targeted) is about to
        // happen regardless of whether the copy is technically still
        // in-flight -- best-effort cleanup on the way out, not a new
        // correctness contract for mid-run behavior.
        reap_pending_host_unregisters();
        for (auto& p : pending_host_unregisters) cudaHostUnregister(p.host_ptr);
        pending_host_unregisters.clear();
        if (detached_cache_host) {
            cudaFreeHost(detached_cache_host);
            detached_cache_host = nullptr;
        }
        if (detached_cache_host_reserved) {
            MemoryGuard::release_host_bytes(detached_cache_bytes);
            detached_cache_host_reserved = false;
        }
        for (auto& loc : expert_vram_cache) {
            if (loc.upload_event) { cudaSetDevice(loc.gpu_id); cudaEventDestroy(loc.upload_event); }
        }
        // Mirrors the main-cache loop above -- active_layer_slots' own
        // upload_event, added 2026-08-26 (see allocate_active_layer_area).
        for (auto& loc : active_layer_slots) {
            if (loc.upload_event) { cudaSetDevice(loc.gpu_id); cudaEventDestroy(loc.upload_event); }
            if (loc.upload_start_event) { cudaSetDevice(loc.gpu_id); cudaEventDestroy(loc.upload_start_event); }
        }
        for (auto& dev : devices) {
            if (dev.compute_stream) { cudaSetDevice(dev.id); cudaStreamDestroy(dev.compute_stream); }
            if (dev.copy_stream) { cudaSetDevice(dev.id); cudaStreamDestroy(dev.copy_stream); }
            if (dev.active_layer_stream) { cudaSetDevice(dev.id); cudaStreamDestroy(dev.active_layer_stream); }
            if (dev.speculative_prefetch_stream) { cudaSetDevice(dev.id); cudaStreamDestroy(dev.speculative_prefetch_stream); }
            if (dev.speculative_prefetch_stream2) { cudaSetDevice(dev.id); cudaStreamDestroy(dev.speculative_prefetch_stream2); }
            if (dev.active_layer_sync_fallback_stream) { cudaSetDevice(dev.id); cudaStreamDestroy(dev.active_layer_sync_fallback_stream); }
            if (dev.active_layer_demand_stream) { cudaSetDevice(dev.id); cudaStreamDestroy(dev.active_layer_demand_stream); }
            if (dev.expert_cache_pool) {
                cudaSetDevice(dev.id);
                if (dev.cufile_pool_registered) cufile_io::deregister_buffer(dev.expert_cache_pool);
                cudaFree(dev.expert_cache_pool);
            }
        }
        if (h_xfer_pinned) cudaFreeHost(h_xfer_pinned);
        for (auto& bb : bounce_pool) {
            if (bb.in_flight) cudaEventDestroy(bb.in_flight);
            if (bb.host_ptr) cudaFreeHost(bb.host_ptr);
        }
        // active_layer_pool is deliberately NEVER cuFileBufRegister'd -- see
        // upload_expert_cufile's own comment: evict_lru() (which it reuses
        // for slot selection, same as upload_expert) only ever returns a
        // slot inside expert_cache_pool/add_dual_role's pool, never this
        // one, so there is nothing to deregister here.
        if (active_layer_pool) { cudaSetDevice(active_layer_slots.front().gpu_id); cudaFree(active_layer_pool); }
        MemoryGuard::release_host_bytes(host_pinned_reservation_bytes);
        host_pinned_reservation_bytes = 0;
    }

    // Allocate expert VRAM cache on vault GPUs
    // Fixed-size (BOUNCE_POOL_SIZE), pinned (cudaHostAllocPortable so any
    // vault GPU can DMA from it, not just whichever device was
    // cudaSetDevice-active at alloc time), allocated exactly ONCE here and
    // never resized. Deliberately NOT the same pattern as the prefetcher's
    // own staging pool (async_prefetcher.hpp), which grows its pinned
    // buffers on demand -- a real run showed that grow-on-demand pattern
    // stalling for hundreds of ms per cudaFreeHost+cudaMallocHost regrow
    // while holding a mutex ([SLOW-STAGING-REGROW], see CLAUDE.md). A fixed
    // count sized once to the model's own expert_slab_bytes can't reproduce
    // that failure mode. Also deliberately NOT pinning the RAM tier's own
    // slab_buffer directly -- that's a multi-GB, ever-growing LRU cache,
    // not a small transient staging area, and would risk the same regrow
    // stall at a much larger scale plus lock down host RAM that's already
    // tightly budgeted (see host_ram_gb's freeze-incident history).
    void init_bounce_pool(size_t slab_bytes) {
        if (!bounce_pool.empty() || slab_bytes == 0) return;
        if (slab_bytes > std::numeric_limits<size_t>::max() / BOUNCE_POOL_SIZE) {
            LOG_STREAM_WARN() << "[GPU] Bounce buffer size overflow -- disabled\n";
            return;
        }
        const size_t total_host_bytes = slab_bytes * BOUNCE_POOL_SIZE;
        if (!MemoryGuard::try_reserve_host_bytes(total_host_bytes)) {
            LOG_STREAM_WARN() << "[MEM-GUARD] bounce-buffer pool refused: "
                              << total_host_bytes / 1e6
                              << " MB; expert uploads will use pageable sources\n";
            return;
        }
        host_pinned_reservation_bytes += total_host_bytes;
        bounce_pool.resize(BOUNCE_POOL_SIZE);
        int allocated = 0;
        for (auto& bb : bounce_pool) {
            cudaError_t err = cudaHostAlloc((void**)&bb.host_ptr, slab_bytes, cudaHostAllocPortable);
            if (err != cudaSuccess) {
                LOG_STREAM_ERROR() << "[GPU] Bounce buffer cudaHostAlloc(" << slab_bytes / 1e6
                          << " MB) failed: " << cudaGetErrorString(err)
                          << " -- expert uploads fall back to direct pageable-source copy.\n";
                bb.host_ptr = nullptr;
                continue;
            }
            ++allocated;
            cudaEventCreate(&bb.in_flight);
        }
        if (allocated < BOUNCE_POOL_SIZE) {
            const size_t unused = (size_t)(BOUNCE_POOL_SIZE - allocated) * slab_bytes;
            MemoryGuard::release_host_bytes(unused);
            host_pinned_reservation_bytes -= unused;
        }
        LOG_STREAM_INFO() << "[GPU] Bounce buffer pool: " << BOUNCE_POOL_SIZE << " x "
                  << slab_bytes / 1e6 << " MB pinned (portable) staging buffers for expert H2D uploads\n";
    }

    // Round-robins bounce_pool; waits only if THIS buffer's previous DMA
    // hasn't drained yet (double buffering -- with BOUNCE_POOL_SIZE > 1 the
    // wait is normally a no-op, since real compute happens between calls).
    // Returns nullptr (caller falls back to copying straight from the
    // pageable source) if the pool failed to allocate.
    BounceBuffer* acquire_bounce_buffer() {
        if (bounce_pool.empty()) return nullptr;
        BounceBuffer& bb = bounce_pool[bounce_next];
        bounce_next = (bounce_next + 1) % bounce_pool.size();
        if (!bb.host_ptr) return nullptr;
        // Direct test of a real hypothesis -- see t_bounce_buffer_wait's own
        // comment in PerfCounters for the full reasoning: this exact wait
        // was never timed before (by anything, ever), and is suspected to
        // be most of this codebase's own long-standing "untimed expert=
        // gap". Now the only caller is upload_expert_async (upload_expert
        // was converted off the bounce pool entirely -- see gpu_topology.hpp's
        // upload_expert), so this should read small if that hypothesis holds.
        if (bb.in_flight) {
            PERF_START(bounce_buffer_wait);
            cudaEventSynchronize(bb.in_flight);
            PERF_END(bounce_buffer_wait, t_bounce_buffer_wait);
        }
        return &bb;
    }

    // Shared by every active-layer reclaim site (cleanup_active_layer_area,
    // upload_expert_active_async/upload_expert_active's own victim-eviction
    // of an occupied slot, and a successful promotion in find_active_expert)
    // -- see ExpertLocation::was_async_prewarmed/prewarm_touched's own
    // comment for the full design. Call BEFORE overwriting/clearing a
    // slot's identity, with the slot's CURRENT (about-to-be-replaced)
    // content -- not after, since prewarm_touched only means anything
    // relative to the content being discarded. Resets both flags itself,
    // so callers don't need their own explicit reset alongside this.
    void record_active_layer_prewarm_reclaim(ExpertLocation& slot) {
        // 2026-08-28: per-GPU used/wasted breakdown -- covers BOTH branches
        // below (was_async_prewarmed and pilot_speculative_fill), since the
        // question "does this GPU's routed slice have a different real
        // hit-rate" applies regardless of which mechanism populated the
        // slot. See n_active_layer_used_by_gpu's own comment (perf_timer.hpp).
        if ((slot.was_async_prewarmed || slot.pilot_speculative_fill)
            && slot.gpu_id >= 0 && slot.gpu_id < 8) {
            if (slot.prewarm_touched) {
                perf().n_active_layer_used_by_gpu[slot.gpu_id].fetch_add(1, std::memory_order_relaxed);
            } else {
                perf().n_active_layer_wasted_by_gpu[slot.gpu_id].fetch_add(1, std::memory_order_relaxed);
            }
        }
        if (slot.was_async_prewarmed) {
            if (slot.prewarm_touched) {
                perf().n_active_layer_prewarm_used.fetch_add(1, std::memory_order_relaxed);
            } else {
                perf().n_active_layer_prewarm_wasted.fetch_add(1, std::memory_order_relaxed);
                perf().bytes_active_layer_prewarm_wasted.fetch_add((int64_t)slot.size, std::memory_order_relaxed);
            }
        } else if (slot.pilot_speculative_fill) {
            // Own counter pair -- see ExpertLocation::pilot_speculative_fill's
            // own comment for why this is deliberately not folded into the
            // was_async_prewarmed branch above.
            if (slot.prewarm_touched) {
                perf().n_pilot_active_fill_used.fetch_add(1, std::memory_order_relaxed);
            } else {
                perf().n_pilot_active_fill_wasted.fetch_add(1, std::memory_order_relaxed);
                perf().bytes_pilot_active_fill_wasted.fetch_add((int64_t)slot.size, std::memory_order_relaxed);
            }
            // Per-rank breakdown -- see ExpertLocation::promoted_at_rank's
            // own comment for why this answers a different question than
            // the pilot's own per-rank prediction-accuracy metric.
            if (slot.promoted_at_rank >= 0 && slot.promoted_at_rank < 8) {
                if (slot.prewarm_touched) {
                    perf().n_pilot_active_fill_used_by_rank[slot.promoted_at_rank].fetch_add(
                        1, std::memory_order_relaxed);
                } else {
                    perf().n_pilot_active_fill_wasted_by_rank[slot.promoted_at_rank].fetch_add(
                        1, std::memory_order_relaxed);
                }
            }
        }
        slot.was_async_prewarmed = false;
        slot.pilot_speculative_fill = false;
        slot.prewarm_touched = false;
        slot.promoted_at_rank = -1;
    }

    // Build a balanced compact-cache plan for one physical VRAM pool. The
    // plan gives every MoE layer the same number of resident expert slots as
    // far as the byte budget permits, instead of charging every slot at the
    // largest slab size in the model. The round-robin fill keeps coverage
    // balanced across layers; the final remainder is used wherever it fits.
    // A pure function so its arithmetic can be tested without allocating a
    // CUDA pool.
    static std::vector<int> compact_layer_slot_counts(
        size_t cache_bytes, int n_experts_total,
        const std::vector<size_t>& layer_slab_bytes,
        int full_layer_prefix = 0,
        const std::vector<int>& excluded_layers = {}) {
        std::vector<int> counts(layer_slab_bytes.size(), 0);
        if (cache_bytes == 0 || n_experts_total <= 0) return counts;

        const auto is_excluded = [&](size_t layer) {
            return std::find(excluded_layers.begin(), excluded_layers.end(),
                             static_cast<int>(layer)) != excluded_layers.end();
        };

        size_t used = 0;
        // A complete layer is useful to the GGML external-weight path, while
        // a partial layer still needs the ordinary compact repack. Reserve
        // complete prefix layers first when the caller explicitly requests
        // that layout. The option is deliberately independent of preload
        // ranking: this function only decides physical slot ownership.
        const int prefix_end = std::min(
            std::max(0, full_layer_prefix), (int)layer_slab_bytes.size());
        for (int l = 0; l < prefix_end; ++l) {
            if (is_excluded((size_t)l)) continue;
            const size_t bytes = layer_slab_bytes[(size_t)l];
            if (bytes == 0) continue;  // dense layer: no routed experts
            if (bytes > cache_bytes - used ||
                (size_t)n_experts_total > (cache_bytes - used) / bytes)
                break;
            counts[(size_t)l] = n_experts_total;
            used += bytes * (size_t)n_experts_total;
        }
        // Give each active layer one slot first when possible. This avoids a
        // large layer disappearing entirely from the VRAM tier just because
        // an earlier layer consumed the budget during the balanced fill.
        for (size_t l = 0; l < layer_slab_bytes.size(); ++l) {
            if (is_excluded(l)) continue;
            const size_t bytes = layer_slab_bytes[l];
            if (bytes == 0 || counts[l] != 0 || bytes > cache_bytes - used)
                continue;
            counts[l] = 1;
            used += bytes;
        }

        bool progress = true;
        while (progress) {
            progress = false;
            for (size_t l = 0; l < layer_slab_bytes.size(); ++l) {
                if (is_excluded(l)) continue;
                const size_t bytes = layer_slab_bytes[l];
                if (bytes == 0 || counts[l] >= n_experts_total ||
                    bytes > cache_bytes - used)
                    continue;
                ++counts[l];
                used += bytes;
                progress = true;
            }
        }
        return counts;
    }

    void allocate_expert_cache(size_t expert_slab_bytes, int n_experts_total,
                                const EngineConfig* config = nullptr,
                                const std::vector<size_t>& layer_slab_bytes = {},
                                bool disable_auto_cache = false) {
        expert_max_slab_size = expert_slab_bytes;
        expert_n_experts = n_experts_total;
        expert_layer_slab_bytes = layer_slab_bytes;
        expert_cpu_affinity_layers = config ? config->cpu_affinity_layers
                                            : std::vector<int>{};
        expert_full_layer_prefix = config
            ? std::max(0, config->vram_cache_full_layer_prefix)
            : 0;
        // A geometry vector is an explicit request for the compact layout.
        // Do not require the raw layer sizes to differ: the old global
        // stride may have been inflated by the LCM/row-alignment calculation
        // even when every layer's real slab is the same size. In that case
        // compact mode is still what removes the alignment-induced waste.
        expert_compact_layout = false;
        for (size_t bytes : expert_layer_slab_bytes) {
            if (bytes != 0) {
                expert_compact_layout = true;
                break;
            }
        }
        // The scalar is still the legacy fallback and the staging-buffer
        // maximum. It is intentionally hidden from consumers that need a
        // contiguous resident-layer view when compact mode is active.
        expert_slab_size = expert_compact_layout ? 0 : expert_slab_bytes;
        if (expert_slab_bytes == 0) return;
        init_bounce_pool(expert_slab_bytes);

        for (int vi : vault_gpus) {
            auto& dev = devices[vi];
            // Reserve 1 GB for driver overhead
            size_t reserve = 1ULL << 30;
            size_t hw_cap = (dev.total_vram > reserve) ? dev.total_vram - reserve : 0;

            // Real, ENFORCED cap: honor this GPU's configured moe_cache_gb
            // if a matching [[gpus]] entry set one, clamped to what's
            // physically possible (hw_cap). Previously this was completely
            // IGNORED for every vault-role GPU -- this function always grabbed
            // (total_vram - 1GB) regardless of moe_cache_gb, silently making
            // that TOML field decorative. Confirmed on a real run: changing
            // moe_cache_gb from 23.0 -> 18.5 on a vault GPU produced an
            // IDENTICAL resulting pool size (24.2123 GB, 1642 slots) either
            // way, and starved allocate_active_layer_area of the VRAM
            // headroom that config change was trying to free for it (see
            // CLAUDE.md). By the time this runs, EngineConfig::validate()
            // has already filled in a real positive moe_cache_gb for every
            // dense+moe/moe-role GPU (either the user's explicit value, or
            // an auto-derived default -- see its own comment), so a missing
            // config or no matching entry is the only way requested_bytes
            // stays 0, in which case this preserves the exact old greedy-
            // grab behavior unchanged (safe default for any caller that
            // doesn't pass config, e.g. tests).
            size_t requested_bytes = 0;
            bool matched_config = false;
            if (config) {
                for (auto& gc : config->gpus) {
                    if (gc.device_id == dev.id) {
                        matched_config = true;
                        if (gc.moe_cache_gb > 0)
                            requested_bytes = (size_t)(gc.moe_cache_gb * 1e9);
                        break;
                    }
                }
            }
            // Most callers intentionally retain the historical meaning of
            // moe_cache_gb <= 0: auto-size the cache.  Full-graph startup is
            // the exception: it may have no VRAM left for a legacy decode
            // fallback after reserving the resident graph, and in that case
            // an explicit zero must really disable this allocation.  Without
            // this separate switch, a graph fallback of 0 GB silently turns
            // back into "total_vram - 1GB" here and starves the graph's own
            // allocator before it is built.
            const bool explicitly_disabled = disable_auto_cache &&
                (matched_config || config != nullptr) && requested_bytes == 0;
            size_t cache_bytes = explicitly_disabled
                ? 0
                : (requested_bytes > 0 ? std::min(requested_bytes, hw_cap) : hw_cap);

            if (explicitly_disabled) {
                dev.expert_cache_pool = nullptr;
                dev.expert_cache_size = 0;
                LOG_STREAM_INFO() << "[GPU] Expert cache disabled on " << dev.name
                                  << " (explicit graph fallback budget is 0 GB)\n";
                continue;
            }

            cudaSetDevice(dev.id);
            if (cudaMalloc((void**)&dev.expert_cache_pool, cache_bytes) != cudaSuccess) {
                // Retry with 80% of the REQUESTED budget (not raw
                // total_vram) -- same shrink-and-retry safety net as
                // before, just scoped to the real budget instead of the
                // whole card.
                cache_bytes = (size_t)(cache_bytes * 0.8);
                if (cudaMalloc((void**)&dev.expert_cache_pool, cache_bytes) != cudaSuccess) {
                    LOG_STREAM_ERROR() << "[GPU] Failed to allocate expert cache on device " << dev.id << "\n";
                    dev.expert_cache_pool = nullptr;
                    cache_bytes = 0;
                }
            }
            dev.expert_cache_size = cache_bytes;
            dev.available_vram -= cache_bytes;

            LOG_STREAM_INFO() << "[GPU] Expert cache on " << dev.name << " (GPU " << dev.id
                      << "): " << cache_bytes / 1e9 << " GB"
                      << (requested_bytes > 0 ? " (moe_cache_gb enforced)" : " (auto: total_vram - 1GB)")
                      << "\n";

            // ONE cuFileBufRegister call for the whole pool -- every future
            // upload_expert_cufile() into any slot inside it is covered by
            // this single registration, no per-slot/per-expert call needed
            // (see cufile_io.hpp's own comment on register_buffer). No-op
            // (cufile_io::available()==false) when cuFile wasn't compiled
            // in -- cufile_pool_registered simply stays false, and
            // upload_expert_cufile's own check on it is what actually gates
            // whether that path is ever attempted for this device.
            if (dev.expert_cache_pool && cufile_io::available()) {
                dev.cufile_pool_registered = cufile_io::register_buffer(dev.expert_cache_pool, cache_bytes);
                if (!dev.cufile_pool_registered) {
                    LOG_STREAM_WARN() << "[CUFILE] Buffer registration failed for GPU " << dev.id
                              << "'s expert cache pool -- direct disk->VRAM reads unavailable for "
                                 "this device, falls back to RAM-mediated upload.\n";
                } else {
                    LOG_STREAM_INFO() << "[CUFILE] Buffer registered: GPU " << dev.id
                              << " expert cache pool (" << cache_bytes / 1e9 << " GB).\n";
                }
            }
        }

        reindex_expert_vram_cache(expert_slab_bytes);
    }

    // Slot-index layout ONLY (device_ptr/offset/size per slot) -- no
    // cudaMalloc. Extracted out of allocate_expert_cache's tail so it can
    // be safely RE-RUN after vault_gpus grows (add_dual_role_expert_cache
    // below appends a device mid-startup, after that device's pool has
    // already been separately cudaMalloc'd). Deterministic given
    // vault_gpus + each device's expert_cache_pool/expert_cache_size, so
    // re-running it in full after an append reproduces IDENTICAL
    // device_ptr/offset values for every pre-existing slot (vault_gpus is
    // only ever appended to here, never reordered) and lays out new slots
    // for whatever was newly appended. Only safe to call before anything
    // has started USING expert_vram_cache (true at startup, before the
    // prefetcher/decode loop begins) -- not safe to call mid-session.
    void reindex_expert_vram_cache(size_t expert_slab_bytes) {
        expert_max_slab_size = expert_slab_bytes;
        expert_slab_size = expert_compact_layout ? 0 : expert_slab_bytes;
        if (expert_slab_bytes == 0) return;

        if (expert_compact_layout && expert_n_experts > 0 &&
            !expert_layer_slab_bytes.empty()) {
            expert_vram_cache_cap = 0;
            for (int vi : vault_gpus) {
                const auto counts = compact_layer_slot_counts(
                    devices[vi].expert_cache_size, expert_n_experts,
                    expert_layer_slab_bytes, expert_full_layer_prefix,
                    expert_cpu_affinity_layers);
                for (int count : counts) expert_vram_cache_cap += (size_t)count;
            }
            expert_vram_cache.resize(expert_vram_cache_cap);

            size_t slot_index = 0;
            for (int vi : vault_gpus) {
                auto& dev = devices[vi];
                const auto counts = compact_layer_slot_counts(
                    dev.expert_cache_size, expert_n_experts,
                    expert_layer_slab_bytes, expert_full_layer_prefix,
                    expert_cpu_affinity_layers);
                size_t offset = 0;
                for (size_t layer = 0; layer < counts.size(); ++layer) {
                    const size_t layer_bytes = expert_layer_slab_bytes[layer];
                    for (int e = 0; e < counts[layer]; ++e) {
                        auto& slot = expert_vram_cache[slot_index++];
                        slot.gpu_id = dev.id;
                        slot.device_ptr = dev.expert_cache_pool + offset;
                        slot.offset = offset;
                        slot.size = layer_bytes;
                        slot.pool_layer = (int)layer;
                        slot.layer_idx = -1;
                        slot.expert_id = -1;
                        slot.last_used_clock = 0;
                        slot.hit_count = 0;
                        if (!slot.upload_event) {
                            cudaSetDevice(dev.id);
                            cudaEventCreate(&slot.upload_event);
                        }
                        offset += layer_bytes;
                    }
                }
                LOG_STREAM_INFO() << "[GPU] Compact expert pool: GPU " << dev.id
                                  << " uses " << slot_index << " total slots so far, "
                                  << offset / 1e9 << " GB laid out across "
                                  << expert_layer_slab_bytes.size() << " layer regions\n";
            }

            unpinned_slot_indices.clear();
            unpinned_slot_indices.reserve(expert_vram_cache_cap);
            for (size_t i = 0; i < expert_vram_cache_cap; ++i)
                unpinned_slot_indices.push_back(i);

            size_t total_cache_bytes = 0;
            for (int vi : vault_gpus) total_cache_bytes += devices[vi].expert_cache_size;
            LOG_STREAM_INFO() << "[GPU] Compact expert VRAM cache: "
                              << expert_vram_cache_cap << " variable-size slots ("
                              << total_cache_bytes / 1e9 << " GB pool) across "
                              << vault_gpus.size() << " GPU(s)"
                              << (expert_cpu_affinity_layers.empty()
                                      ? "\n"
                                      : " (CPU-affinity layers excluded from "
                                        "physical slot ownership)\n");
            return;
        }

        // REAL BUG FIX: total usable capacity must be the SUM of each
        // device's OWN floor(expert_cache_size / expert_slab_bytes), not
        // floor(SUM of all devices' capacities / expert_slab_bytes). The
        // old aggregate-then-divide approach let a slot's [offset,
        // offset+slab) range straddle a per-device boundary: the loop
        // below only checked a slot's STARTING offset against the current
        // device's capacity, never whether the slot's FULL extent fit
        // before that device's real cudaMalloc'd buffer ended. Confirmed
        // on a real run: a slot's dst was only 3.72MB before its device's
        // actual end, but the slab needed 10.78MB -- cudaMemcpy correctly
        // refused with "invalid argument" rather than silently writing
        // past the allocation, but the slot should never have been laid
        // out that way to begin with. Each device's own leftover
        // remainder (less than one full slab) is now deliberately wasted
        // rather than risking a straddling slot -- at most
        // expert_slab_bytes-1 bytes lost per vault GPU, negligible next
        // to the sizes involved. Only actually manifests with >=2 vault
        // GPUs (a single vault's own capacity is never subdivided against
        // anything else), which is why this went unnoticed until the
        // dual-role feature made a second, differently-sized vault common.
        expert_vram_cache_cap = 0;
        for (int vi : vault_gpus) expert_vram_cache_cap += devices[vi].expert_cache_size / expert_slab_bytes;
        // resize() to a LARGER cap preserves existing elements (including
        // their already-created upload_event) at their existing indices --
        // only the newly-appended tail is default-constructed.
        expert_vram_cache.resize(expert_vram_cache_cap);

        size_t offset = 0;
        int vault_idx = 0;
        for (size_t i = 0; i < expert_vram_cache_cap; ++i) {
            // Advance to the next vault GPU once the CURRENT one can no
            // longer fit a FULL slot -- checks the slot's whole extent
            // (offset + expert_slab_bytes), not just whether offset alone
            // has already reached the boundary (the actual fix; see this
            // function's header comment). Resets offset to 0 rather than
            // subtracting the device's capacity, since any leftover
            // remainder in the device being left behind is deliberately
            // abandoned, not carried forward.
            while (vault_idx < (int)vault_gpus.size() &&
                   offset + expert_slab_bytes > devices[vault_gpus[vault_idx]].expert_cache_size) {
                offset = 0;
                vault_idx++;
            }
            if (vault_idx >= (int)vault_gpus.size()) break;

            auto& dev = devices[vault_gpus[vault_idx]];
            expert_vram_cache[i].gpu_id = dev.id;
            expert_vram_cache[i].device_ptr = dev.expert_cache_pool + offset;
            expert_vram_cache[i].offset = offset;
            expert_vram_cache[i].size = expert_slab_bytes;
            expert_vram_cache[i].pool_layer = -1;
            expert_vram_cache[i].layer_idx = -1;
            expert_vram_cache[i].expert_id = -1;
            // Guard against re-creating (and leaking) an event that a
            // PRIOR call to this function already created for this slot --
            // the only reason this function can run more than once.
            if (!expert_vram_cache[i].upload_event) {
                cudaSetDevice(dev.id);
                cudaEventCreate(&expert_vram_cache[i].upload_event);
            }
            offset += expert_slab_bytes;
        }

        // Diagnostic: log each vault GPU's actual pool base address + byte
        // range every time this runs (including on a second, dual-role-
        // triggered call) -- added to root-cause a real run where H2D
        // preload uploads failed almost universally with cudaErrorInvalidValue
        // ("invalid argument") ONLY once a second (dual-role) vault GPU was
        // added, never on a single-vault run. That error is a classic
        // signature of a device_ptr not belonging to the currently
        // cudaSetDevice'd GPU -- this makes it possible to directly compare
        // a failing upload's logged device_ptr/gpu_id (see upload_expert's
        // own error log) against the ACTUAL valid ranges below, rather than
        // guess whether the two-vault slot layout computed the wrong
        // address for some slots.
        for (int vi : vault_gpus) {
            auto& d = devices[vi];
            LOG_STREAM_INFO() << "[GPU] Vault pool range: GPU " << d.id << " base=" << (void*)d.expert_cache_pool
                      << " end=" << (void*)(d.expert_cache_pool + d.expert_cache_size)
                      << " (" << d.expert_cache_size / 1e9 << " GB)\n";
        }

        // Rebuild unpinned_slot_indices to cover every slot -- always safe
        // here (never anything pinned yet at this point in startup, see
        // this vector's own comment), and correctly re-covers any newly
        // appended tail on a second (dual-role) call too.
        unpinned_slot_indices.clear();
        unpinned_slot_indices.reserve(expert_vram_cache_cap);
        for (size_t i = 0; i < expert_vram_cache_cap; ++i) unpinned_slot_indices.push_back(i);

        // Raw total for display only -- NOT used for capacity math anymore
        // (see this function's header comment for why the aggregate-then-
        // divide approach was wrong); this can legitimately be a bit
        // larger than expert_vram_cache_cap * expert_slab_bytes, since it
        // includes each device's own wasted remainder.
        size_t total_cache_bytes = 0;
        for (int vi : vault_gpus) total_cache_bytes += devices[vi].expert_cache_size;
        LOG_STREAM_INFO() << "[GPU] Expert VRAM cache: " << expert_vram_cache_cap
                  << " slots (" << total_cache_bytes / 1e9 << " GB total) across "
                  << vault_gpus.size() << " GPU(s)\n";
    }

    // Rebuild the two lookup structures after the slot set changes.  Only
    // slots that still own a device pointer are indexable; detached slots
    // intentionally stay in expert_vram_cache so restoring them preserves
    // their original slot indices and identities.
    void rebuild_cache_indices() {
        expert_index.clear();
        unpinned_slot_indices.clear();
        unpinned_slot_indices.reserve(expert_vram_cache.size());
        for (size_t i = 0; i < expert_vram_cache.size(); ++i) {
            ExpertLocation &loc = expert_vram_cache[i];
            if (!loc.device_ptr) continue;
            if (loc.layer_idx >= 0 && loc.expert_id >= 0)
                expert_index[swap_key(loc.layer_idx, loc.expert_id)] = i;
            if (!loc.pinned) unpinned_slot_indices.push_back(i);
        }
    }

    // Copy one GPU's resident expert cache to host and free its CUDA pool.
    // The prompt graph's ring is built while the pool is gone; the bytes are
    // copied back by restore_detached_cache() after the ring is freed.
    // Returns true when the cache is (already) detached.
    bool detach_cache_for_prefill(int device_id) {
        GPU_TOPO_TIMED_LOCK(lock);
        if (detached_cache_device >= 0)
            return detached_cache_device == device_id;

        GpuDevice *dev = nullptr;
        for (auto &d : devices)
            if (d.id == device_id) {
                dev = &d;
                break;
            }
        if (!dev || !dev->expert_cache_pool || dev->expert_cache_size == 0)
            return false;
        const size_t bytes = dev->expert_cache_size;
        if (!MemoryGuard::try_reserve_host_bytes(bytes)) {
            LOG_STREAM_WARN() << "[GPU] prefill cache borrow: host reservation refused for "
                              << bytes / 1e9 << " GB on GPU " << dev->id << "\n";
            return false;
        }

        cudaSetDevice(dev->id);
        void *host = nullptr;
        cudaError_t err = cudaHostAlloc(&host, bytes, cudaHostAllocDefault);
        if (err != cudaSuccess || !host) {
            MemoryGuard::release_host_bytes(bytes);
            LOG_STREAM_WARN() << "[GPU] prefill cache borrow: pinned host copy failed on GPU "
                              << dev->id << " (" << bytes / 1e9 << " GB): "
                              << cudaGetErrorString(err) << "\n";
            return false;
        }
        err = cudaMemcpy(host, dev->expert_cache_pool, bytes,
                         cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            cudaFreeHost(host);
            MemoryGuard::release_host_bytes(bytes);
            LOG_STREAM_WARN() << "[GPU] prefill cache borrow: D2H copy failed on GPU "
                              << dev->id << ": " << cudaGetErrorString(err) << "\n";
            return false;
        }

        std::vector<std::pair<size_t, ExpertLocation>> saved;
        saved.reserve(expert_vram_cache.size());
        for (size_t i = 0; i < expert_vram_cache.size(); ++i) {
            if (expert_vram_cache[i].gpu_id != dev->id) continue;
            saved.emplace_back(i, expert_vram_cache[i]);
        }

        if (dev->cufile_pool_registered) {
            cufile_io::deregister_buffer(dev->expert_cache_pool);
            dev->cufile_pool_registered = false;
        }
        err = cudaFree(dev->expert_cache_pool);
        if (err != cudaSuccess) {
            // The slot pointers are still valid; keep the cache usable.
            if (cufile_io::available())
                dev->cufile_pool_registered =
                    cufile_io::register_buffer(dev->expert_cache_pool,
                                               dev->expert_cache_size);
            cudaFreeHost(host);
            MemoryGuard::release_host_bytes(bytes);
            LOG_STREAM_WARN() << "[GPU] prefill cache borrow: cudaFree failed on GPU "
                              << dev->id << ": " << cudaGetErrorString(err) << "\n";
            return false;
        }

        for (auto &entry : saved) {
            ExpertLocation &loc = expert_vram_cache[entry.first];
            loc.device_ptr = nullptr;
            loc.layer_idx = -1;
            loc.expert_id = -1;
            loc.upload_ready = true;
        }
        dev->expert_cache_pool = nullptr;
        dev->expert_cache_size = 0;
        dev->available_vram += bytes;

        detached_cache_device = dev->id;
        detached_cache_bytes = bytes;
        detached_cache_host = host;
        detached_cache_host_reserved = true;
        detached_cache_slots = std::move(saved);
        rebuild_cache_indices();

        LOG_STREAM_INFO() << "[GPU] prefill cache borrow: detached GPU " << dev->id
                          << " expert cache (" << bytes / 1e9 << " GB, "
                          << detached_cache_slots.size()
                          << " slots) to host for the prompt ring\n";
        return true;
    }

    // Restore the cache detached by detach_cache_for_prefill().  Must run
    // only after the prompt graph (including its ring buffers) is gone.
    bool restore_detached_cache() {
        GPU_TOPO_TIMED_LOCK(lock);
        if (detached_cache_device < 0)
            return true;

        GpuDevice *dev = nullptr;
        for (auto &d : devices)
            if (d.id == detached_cache_device) {
                dev = &d;
                break;
            }
        if (!dev) {
            LOG_STREAM_WARN() << "[GPU] prefill cache borrow: detached device "
                              << detached_cache_device << " disappeared\n";
            return false;
        }

        cudaSetDevice(dev->id);
        void *pool = nullptr;
        cudaError_t err = cudaMalloc(&pool, detached_cache_bytes);
        if (err != cudaSuccess || !pool) {
            LOG_STREAM_WARN() << "[GPU] prefill cache borrow: restore cudaMalloc failed on GPU "
                              << dev->id << " (" << detached_cache_bytes / 1e9
                              << " GB): " << cudaGetErrorString(err)
                              << "; decode keeps the smaller cache\n";
            return false;
        }
        if (detached_cache_host) {
            err = cudaMemcpy(pool, detached_cache_host, detached_cache_bytes,
                             cudaMemcpyHostToDevice);
            if (err != cudaSuccess) {
                cudaFree(pool);
                LOG_STREAM_WARN() << "[GPU] prefill cache borrow: restore H2D failed on GPU "
                                  << dev->id << ": " << cudaGetErrorString(err)
                                  << "; decode keeps the smaller cache\n";
                return false;
            }
        }

        for (auto &entry : detached_cache_slots) {
            ExpertLocation loc = entry.second;
            if (loc.offset > detached_cache_bytes ||
                loc.size > detached_cache_bytes - loc.offset) {
                LOG_STREAM_WARN() << "[GPU] prefill cache borrow: invalid slot offset "
                                  << loc.offset << " size " << loc.size
                                  << " during restore\n";
                continue;
            }
            loc.device_ptr = (uint8_t *)pool + loc.offset;
            loc.upload_ready = true;
            expert_vram_cache[entry.first] = loc;
        }

        dev->expert_cache_pool = (uint8_t *)pool;
        dev->expert_cache_size = detached_cache_bytes;
        dev->available_vram = dev->available_vram > detached_cache_bytes
                                  ? dev->available_vram - detached_cache_bytes
                                  : 0;
        if (cufile_io::available())
            dev->cufile_pool_registered =
                cufile_io::register_buffer(dev->expert_cache_pool,
                                           dev->expert_cache_size);

        if (detached_cache_host) {
            cudaFreeHost(detached_cache_host);
            detached_cache_host = nullptr;
        }
        if (detached_cache_host_reserved) {
            MemoryGuard::release_host_bytes(detached_cache_bytes);
            detached_cache_host_reserved = false;
        }
        rebuild_cache_indices();

        const size_t slots = detached_cache_slots.size();
        LOG_STREAM_INFO() << "[GPU] prefill cache borrow: restored GPU " << dev->id
                          << " expert cache (" << detached_cache_bytes / 1e9
                          << " GB, " << slots << " slots)\n";
        detached_cache_device = -1;
        detached_cache_bytes = 0;
        detached_cache_slots.clear();
        return true;
    }

    bool cache_detached_for_prefill() const {
        return detached_cache_device >= 0;
    }

    // Give a device that's NOT already a pure ExpertVault (typically the
    // CoreEngine GPU) an ADDITIONAL, explicitly-capped expert-cache slice,
    // then re-lays-out the shared slot index to include it. Unlike
    // allocate_expert_cache's pure-vault path (which grabs "everything
    // minus a flat 1GB reserve" -- safe there because a pure vault GPU has
    // no other VRAM duty to protect), cap_bytes here MUST already reflect
    // real remaining free VRAM on THIS device after its other duties
    // (dense pool, BF16 fast-paths) are sized -- this function trusts the
    // caller's cap completely and does no reservation math of its own.
    // Once added, the existing find_expert/get_expert/evict_lru/
    // upload_expert machinery needs NO changes -- it already handles
    // multiple vault_gpus generically (the same code already supports the
    // TripleGPU profile spanning several ExpertVault devices), so a
    // dual-role device is indistinguishable from any other vault GPU to
    // that code; only role stays CoreEngine for VRAM-usage reporting.
    bool add_dual_role_expert_cache(int device_index, size_t cap_bytes, size_t expert_slab_bytes) {
        size_t minimum_compact_slab = expert_slab_bytes;
        if (expert_compact_layout) {
            minimum_compact_slab = SIZE_MAX;
            for (size_t bytes : expert_layer_slab_bytes) {
                if (bytes != 0) minimum_compact_slab = std::min(minimum_compact_slab, bytes);
            }
            if (minimum_compact_slab == SIZE_MAX) minimum_compact_slab = expert_slab_bytes;
        }
        if (expert_slab_bytes == 0 || cap_bytes < minimum_compact_slab) {
            LOG_STREAM_INFO() << "[GPU] Dual-role expert cache on device " << devices[device_index].id
                      << " skipped -- cap (" << cap_bytes / 1e9 << " GB) smaller than one expert slab ("
                      << minimum_compact_slab / 1e9 << " GB).\n";
            return false;
        }
        auto& dev = devices[device_index];
        cudaSetDevice(dev.id);
        cudaError_t err = cudaMalloc((void**)&dev.expert_cache_pool, cap_bytes);
        if (err != cudaSuccess) {
            LOG_STREAM_ERROR() << "[GPU] Dual-role expert cache allocation failed on device " << dev.id
                      << " (" << cap_bytes / 1e9 << " GB): " << cudaGetErrorString(err) << "\n";
            dev.expert_cache_pool = nullptr;
            return false;
        }
        dev.expert_cache_size = cap_bytes;
        dev.available_vram -= cap_bytes;
        if (std::find(vault_gpus.begin(), vault_gpus.end(), device_index) == vault_gpus.end()) {
            vault_gpus.push_back(device_index);
        }
        LOG_STREAM_INFO() << "[GPU] Dual-role expert cache on " << dev.name << " (GPU " << dev.id
                  << ", also " << ((dev.role == GpuRole::CoreEngine) ? "CoreEngine" : "role") << "): "
                  << cap_bytes / 1e9 << " GB\n";
        // Same registration as allocate_expert_cache's own pool -- this
        // device's pool is appended to vault_gpus above, so evict_lru() can
        // return a slot inside it exactly like any other vault GPU's pool
        // (see this function's own header comment: "a dual-role device is
        // indistinguishable from any other vault GPU"), and
        // upload_expert_cufile must be able to cuFileRead into it too.
        if (cufile_io::available()) {
            dev.cufile_pool_registered = cufile_io::register_buffer(dev.expert_cache_pool, cap_bytes);
            if (!dev.cufile_pool_registered) {
                LOG_STREAM_WARN() << "[CUFILE] Buffer registration failed for GPU " << dev.id
                          << "'s dual-role expert cache -- direct disk->VRAM reads unavailable for "
                             "this device.\n";
            } else {
                LOG_STREAM_INFO() << "[CUFILE] Buffer registered: GPU " << dev.id
                          << " dual-role expert cache (" << cap_bytes / 1e9 << " GB).\n";
            }
        }
        reindex_expert_vram_cache(expert_slab_bytes);
        return true;
    }

    // Shared tail of find_expert's identity-match case, used by both the
    // index-hit fast path and the linear-scan fallback below so the
    // upload_ready-poll / hit-count / decay logic exists in exactly one
    // place. Caller has already confirmed loc's identity matches
    // (layer, expert_id) and loc.device_ptr is non-null. Returns nullptr
    // only when an async promotion is still mid-copy (data not safe to
    // read yet) -- everything else about "this is the right slot" is
    // already settled by the time this is called.
    ExpertLocation* finish_expert_hit(ExpertLocation& loc, int layer, int expert_id) {
        if (!loc.upload_ready) {
            // BUG (found via zero [MOE-VRAM] READY events ever logged,
            // despite constant PROMOTE traffic): this queries an event
            // that lives on the VAULT gpu, but the calling thread's
            // ambient CUDA "current device" is whatever it was last
            // left at by unrelated calls elsewhere (e.g. dense_proj's
            // GPU dispatch on the CORE gpu) — cudaEventQuery on an
            // event from a different device than the current context
            // does not reliably return cudaSuccess even once the copy
            // has actually completed, so upload_ready silently never
            // flipped true and every promoted expert stayed CPU-only
            // forever. Must pin the device explicitly before querying.
            cudaSetDevice(loc.gpu_id);
            if (loc.upload_event && cudaEventQuery(loc.upload_event) == cudaSuccess) {
                loc.upload_ready = true;
                LOG_DEBUG("[MOE-VRAM]  READY   layer=%d expert=%d async H2D copy complete -> "
                          "GPU dispatch now available\n", layer, expert_id);
            } else {
                return nullptr; // async copy still in flight — not safe to use yet
            }
        }
        // A successful cache lookup is the first real routed use of this
        // startup-loaded identity. Keep this separate from hit_count because
        // the question here is binary: was the preloaded expert ever needed
        // before eviction?
        note_startup_preload_hit(layer, expert_id);
        loc.last_used_clock = ++cache_clock;
        if (loc.hit_count < UINT32_MAX) loc.hit_count++;
        if (++hits_since_decay >= HIT_DECAY_INTERVAL) decay_hit_counts();
        return &loc;
    }

    // Look up or load an expert into VRAM cache. Returns device pointer or
    // nullptr — including when a background async promotion is still in
    // flight (so callers never read a partially-copied slab).
    //
    // O(1) via expert_index first; falls back to (and self-heals via) the
    // original O(n) linear scan if the index misses or, more importantly,
    // if it hits but doesn't validate against the slot's real contents --
    // see expert_index's own comment for why that path exists and why
    // it's safe even if the index has a latent maintenance bug.
    ExpertLocation* find_expert(int layer, int expert_id) {
        GPU_TOPO_TIMED_LOCK(lock);
        // Whole-function cost -- likely the highest-call-count function in
        // the entire decode-time hot path (execution_layer.cpp's shadow
        // pre-check calls this once per topk-expert per token per layer,
        // regardless of eventual GPU/CPU outcome), yet only its lock-
        // ACQUISITION wait was ever measured (t_cache_mutex_wait via
        // GPU_TOPO_TIMED_LOCK), never the O(1) expert_index lookup + hit-
        // count bookkeeping done once the lock is already held. Added after
        // hold_for_compute/release_hold/find_active_expert (measured
        // directly, found to cost only 0.65s combined) and t_gpu_upload
        // (measured, found to be 0.00s this run) both ruled out the leading
        // hypotheses without closing the real ~65-70s expert=/accounted-for
        // gap -- this function was the one remaining GpuTopology call this
        // frequent that had never been measured as a whole.
        PERF_SCOPE(t_find_expert);
        uint64_t key = swap_key(layer, expert_id);
        auto it = expert_index.find(key);
        if (it != expert_index.end()) {
            size_t idx = it->second;
            if (idx < expert_vram_cache.size()) {
                ExpertLocation& loc = expert_vram_cache[idx];
                if (loc.layer_idx == layer && loc.expert_id == expert_id && loc.device_ptr) {
                    return finish_expert_hit(loc, layer, expert_id);
                }
            }
            // Index pointed somewhere that doesn't actually match --
            // should be unreachable given evict_lru/upload_expert/
            // upload_expert_async are the only three places that mutate
            // both the index and expert_vram_cache's identity fields, and
            // both erase/insert under the same cache_mutex a genuine
            // eviction/upload takes. Treat it as a real bug signal (loud
            // log + counter), not a silent no-op, but don't let it break
            // correctness -- fall through to the linear scan below like
            // a plain index miss.
            LOG_ERROR("[MOE-VRAM] find_expert: stale expert_index entry for layer=%d expert=%d "
                      "(pointed at slot %zu, which doesn't match) -- falling back to linear scan\n",
                      layer, expert_id, idx);
            perf().n_stale_expert_index_hits.fetch_add(1, std::memory_order_relaxed);
            expert_index.erase(it);
        }
        for (size_t i = 0; i < expert_vram_cache.size(); ++i) {
            ExpertLocation& loc = expert_vram_cache[i];
            if (loc.layer_idx == layer && loc.expert_id == expert_id && loc.device_ptr) {
                expert_index[key] = i; // opportunistic self-heal
                return finish_expert_hit(loc, layer, expert_id);
            }
        }
        return nullptr;
    }

    // Batch sibling of find_expert().  Grouped MoE dispatch already has the
    // complete routed expert set for a layer, but the old loop acquired
    // cache_mutex and re-entered the expert-index lookup once per expert.
    // Keep one lock over the layer's main-cache probes while preserving the
    // exact per-hit semantics (upload-ready polling, startup-hit accounting,
    // hit/LRU updates, decay, and stale-index self-healing).  The returned
    // pointers obey the same contract as find_expert(): callers must still
    // validate/hold the identity before enqueueing work because an upload or
    // eviction can occur after this function releases cache_mutex.
    void find_expert_batch(int layer, const int *expert_ids, int count,
                           ExpertLocation **out) {
        if (!expert_ids || !out || count <= 0) return;
        GPU_TOPO_TIMED_LOCK(lock);
        PERF_SCOPE(t_find_expert);
        for (int i = 0; i < count; ++i) {
            out[i] = nullptr;
            const int expert_id = expert_ids[i];
            const uint64_t key = swap_key(layer, expert_id);
            auto it = expert_index.find(key);
            if (it != expert_index.end()) {
                const size_t idx = it->second;
                if (idx < expert_vram_cache.size()) {
                    ExpertLocation &loc = expert_vram_cache[idx];
                    if (loc.layer_idx == layer && loc.expert_id == expert_id &&
                        loc.device_ptr) {
                        out[i] = finish_expert_hit(loc, layer, expert_id);
                        continue;
                    }
                }
                LOG_ERROR(
                    "[MOE-VRAM] find_expert_batch: stale expert_index entry "
                    "for layer=%d expert=%d (slot %zu) -- linear scan\n",
                    layer, expert_id, idx);
                perf().n_stale_expert_index_hits.fetch_add(
                    1, std::memory_order_relaxed);
                expert_index.erase(it);
            }
            for (size_t idx = 0; idx < expert_vram_cache.size(); ++idx) {
                ExpertLocation &loc = expert_vram_cache[idx];
                if (loc.layer_idx == layer && loc.expert_id == expert_id &&
                    loc.device_ptr) {
                    expert_index[key] = idx;
                    out[i] = finish_expert_hit(loc, layer, expert_id);
                    break;
                }
            }
        }
    }

    // Side-effect-free residency probe used by the Qwen pilot policy. Unlike
    // find_expert(), this deliberately does not update hit counters or LRU
    // clocks: it answers only whether every expert in a layer is already in
    // the durable main VRAM cache. When that is true, a decode-time pilot
    // cannot improve placement and its HC snapshot/router synchronization is
    // pure overhead. Active-layer entries do not count as durable residency.
    bool layer_fully_resident(int layer, int expert_count) const {
        if (layer < 0 || expert_count <= 0) return false;
        GPU_TOPO_TIMED_LOCK(lock);
        for (int expert = 0; expert < expert_count; ++expert) {
            const auto it = expert_index.find(swap_key(layer, expert));
            if (it == expert_index.end() || it->second >= expert_vram_cache.size())
                return false;
            const ExpertLocation &loc = expert_vram_cache[it->second];
            if (loc.layer_idx != layer || loc.expert_id != expert ||
                !loc.device_ptr || !loc.upload_ready)
                return false;
        }
        return true;
    }

    // Side-effect-free O(1) identity check for prediction/prefetch policy.
    // Unlike find_expert(), this does not update hit/LRU counters, poll an
    // upload event, or change the current CUDA device.  An identity whose
    // async upload is still pending is deliberately reported as present:
    // another transfer has already claimed the slot, so a predictor must not
    // enqueue a duplicate copy.  The eventual consumer uses the normal
    // upload-ready/active-layer wait path before reading the bytes.
    bool main_layer_has(int layer, int expert_id) const {
        if (layer < 0 || expert_id < 0) return false;
        GPU_TOPO_TIMED_LOCK(lock);
        const uint64_t key = swap_key(layer, expert_id);
        const auto it = expert_index.find(key);
        if (it != expert_index.end() &&
            it->second < expert_vram_cache.size()) {
            const ExpertLocation &loc = expert_vram_cache[it->second];
            if (loc.layer_idx == layer && loc.expert_id == expert_id &&
                loc.device_ptr)
                return true;
        }
        // Keep a defensive fallback for a transient stale index.  This is
        // rare and preserves the old correctness behavior without making the
        // normal prediction path scan the entire cache.
        for (const ExpertLocation &loc : expert_vram_cache) {
            if (loc.layer_idx == layer && loc.expert_id == expert_id &&
                loc.device_ptr)
                return true;
        }
        return false;
    }

    // Resolve a complete layer's contiguous resident view with one cache
    // lock. This is intentionally separate from repeated find_expert() calls:
    // the GGML external-weight path may inspect 512 experts for one layer,
    // and taking the cache mutex once per expert would erase the benefit of
    // avoiding the transfer. When `holds` is supplied, only unpinned slots
    // receive a temporary compute hold; pinned slots are already eviction-
    // safe. The returned hold pairs are released by the caller after the
    // queued graph completes.
    bool find_contiguous_resident_layer(
        int layer, int expert_count, int gpu_id, size_t stride,
        const uint8_t **base,
        std::vector<std::pair<int, const uint8_t *>> *holds = nullptr) {
        if (base) *base = nullptr;
        if (!base || layer < 0 || expert_count <= 0 || gpu_id < 0 ||
            stride == 0)
            return false;

        GPU_TOPO_TIMED_LOCK(lock);
        std::vector<size_t> indices;
        indices.reserve((size_t)expert_count);
        const uint8_t *first = nullptr;
        for (int expert = 0; expert < expert_count; ++expert) {
            const auto it = expert_index.find(swap_key(layer, expert));
            if (it == expert_index.end() || it->second >= expert_vram_cache.size()) {
                return false;
            }
            const size_t index = it->second;
            const ExpertLocation &loc = expert_vram_cache[index];
            if (loc.layer_idx != layer || loc.expert_id != expert ||
                !loc.device_ptr || !loc.upload_ready || loc.gpu_id != gpu_id) {
                return false;
            }
            if (expert == 0) {
                first = loc.device_ptr;
            } else if (loc.device_ptr != first + (size_t)expert * stride) {
                return false;
            }
            indices.push_back(index);
        }

        if (holds) holds->clear();
        for (size_t index : indices) {
            ExpertLocation &loc = expert_vram_cache[index];
            if (!loc.pinned) {
                if (loc.temp_pin_count < UINT32_MAX) ++loc.temp_pin_count;
                if (holds)
                    holds->push_back({loc.expert_id, loc.device_ptr});
            }
        }
        *base = first;
        return first != nullptr;
    }

    // Mark a currently VRAM-resident expert's slot as pinned (never chosen
    // as an eviction victim by evict_lru). No-op (returns false) if the
    // expert isn't resident -- caller must upload it first. Used both by
    // RAMSafetyManager::get_expert's within-conversation promotion (a host
    // slot's hit_count crosses pin_threshold) and main.cpp's startup
    // hot-set preload from a usage-file profile (cross-conversation).
    // Deliberately does NOT bump hit_count/last_used_clock the way
    // find_expert does -- pinning is a policy action, not a real access.
    bool pin_expert(int layer, int expert_id) {
        GPU_TOPO_TIMED_LOCK(lock);
        for (auto& loc : expert_vram_cache) {
            if (loc.layer_idx == layer && loc.expert_id == expert_id && loc.device_ptr) {
                if (!loc.pinned) {
                    loc.pinned = true;
                    perf().n_pinned_resident.fetch_add(1, std::memory_order_relaxed);
                    // Remove this slot from evict_lru's scan set -- see
                    // unpinned_slot_indices's own comment. Swap-and-pop:
                    // scan order is irrelevant to evict_lru's min-score
                    // search, so this is a correct O(n) removal (n here is
                    // unpinned_slot_indices's size, not the whole cache;
                    // pin_expert itself is low-frequency, unlike evict_lru).
                    size_t idx = (size_t)(&loc - expert_vram_cache.data());
                    for (size_t i = 0; i < unpinned_slot_indices.size(); ++i) {
                        if (unpinned_slot_indices[i] == idx) {
                            unpinned_slot_indices[i] = unpinned_slot_indices.back();
                            unpinned_slot_indices.pop_back();
                            break;
                        }
                    }
                }
                return true;
            }
        }
        return false;
    }

    // TEMPORARY eviction protection for the scope of one decode-time GPU
    // dispatch -- see ExpertLocation::temp_pin_count's comment for the full
    // reasoning (distinct from pin_expert's PERMANENT pinning above;
    // reference-counted so the two can never incorrectly interfere).
    // Caller contract: hold_for_compute() must be called only after a
    // synchronous upload_expert() (not the _async variant) has confirmed
    // this slot's data is actually present -- this function does not upload
    // anything itself, it only marks an ALREADY-resident slot un-evictable.
    // Returns false if the expert isn't resident (caller error -- upload
    // first). Every successful hold_for_compute() must be matched by
    // exactly one release_hold() once the caller's GPU kernels are
    // confirmed done reading this slot (cudaStreamSynchronize), or the slot
    // leaks as permanently un-evictable.
    //
    // device_ptr disambiguates WHICH physical slot to protect when the
    // caller's (layer, expert_id) could otherwise match more than one --
    // real, reachable case: an expert can be resident in BOTH the main
    // expert_vram_cache AND the active_layer_slots pool simultaneously
    // (the same double-residency this project's [TIER-OVERLAP] tracking
    // already knows can happen). Matching identity alone here would place
    // the hold on WHICHEVER pool's entry is scanned first, protecting the
    // WRONG slot when the caller's actual `loc` (the one the GPU kernel
    // will actually read) is the other one -- leaving the real in-use slot
    // completely unprotected against concurrent reuse (evict_lru,
    // upload_expert_active's victim picker, or cleanup_active_layer_area)
    // while a kernel is still reading it. The two pools' backing memory
    // (expert_cache_pool vs active_layer_pool, separate cudaMalloc
    // regions) never overlaps, so device_ptr equality is an unambiguous
    // disambiguator. `from_active_layer` (see below) tells us which ONE
    // pool to check -- no need to also scan the other, since a device_ptr
    // can only ever belong to the pool it was actually allocated from.
    // BUG FIXED, TAKE 1 (insufficient on its own -- see TAKE 2 below): this
    // used to be an unconditional O(n) linear scan of expert_vram_cache (up
    // to 1322 slots in a real run) on EVERY call, even though find_expert
    // already solved this exact problem with the expert_index O(1) map.
    // Adding the same index lookup here was a real fix for calls resolved
    // via the main cache -- but a real re-test showed ZERO measurable
    // improvement (expert= still ~101s, untimed= unchanged) because on this
    // workload the DOMINANT resolution path is upload_expert_active/
    // find_active_expert (the active-layer pool, confirmed by
    // untimed=17.36s being almost entirely active_up=16.80s) -- and
    // expert_index only ever maps expert_vram_cache entries, so for an
    // active-layer-resident expert the index lookup ALWAYS misses, meaning
    // every call still fell through to the full O(1322) linear scan of
    // expert_vram_cache (which was NEVER going to find it there), THEN the
    // active_layer_slots scan that actually finds it. Net effect: paying
    // for BOTH the pointless O(1322) scan-to-nothing AND the real O(256)
    // scan, on every hold AND every release, for the dominant case --
    // strictly worse than doing nothing extra.
    //
    // BUG FIXED, TAKE 2 (the actual fix): the caller (expert_ffn_resolve_
    // gpu_loc, execution_layer.cpp) already knows UNAMBIGUOUSLY which pool
    // `loc` came from at the exact moment it calls this function -- it's
    // whichever branch just resolved `loc` (gpu_ctx->loc/upload_expert =
    // main cache; find_active_expert/upload_expert_active = active layer).
    // Since the two pools' backing memory (expert_cache_pool vs
    // active_layer_pool, separate cudaMalloc regions) never overlaps, a
    // GIVEN device_ptr value can only ever be found in exactly the ONE pool
    // it actually belongs to -- so once the caller tells us which pool that
    // is, there is no need to guess-scan the other pool at all, not even as
    // a fallback (barring the identity-changed-race case this function
    // already defensively returns false for). from_active_layer is a
    // REQUIRED parameter (no default) so every call site has to make this
    // choice deliberately rather than silently falling back to the slow
    // dual-scan path. expert_index (O(1)) is still used for the main-cache
    // case; active_layer_slots has no index of its own (capped at
    // active_layer_capacity, typically 256 -- small enough that a linear
    // scan of ONLY that pool, with the O(1322) main-cache scan skipped
    // entirely, is cheap).
    bool hold_for_compute(int layer, int expert_id, const uint8_t* device_ptr, bool from_active_layer) {
        GPU_TOPO_TIMED_LOCK(lock);
        PERF_SCOPE(t_hold_for_compute);
        if (from_active_layer) {
            for (auto& loc : active_layer_slots) {
                if (loc.layer_idx == layer && loc.expert_id == expert_id && loc.device_ptr == device_ptr) {
                    loc.temp_pin_count++;
                    return true;
                }
            }
            return false;
        }
        uint64_t key = swap_key(layer, expert_id);
        auto it = expert_index.find(key);
        if (it != expert_index.end()) {
            size_t idx = it->second;
            if (idx < expert_vram_cache.size()) {
                ExpertLocation& loc = expert_vram_cache[idx];
                if (loc.layer_idx == layer && loc.expert_id == expert_id && loc.device_ptr == device_ptr) {
                    loc.temp_pin_count++;
                    return true;
                }
            }
        }
        for (auto& loc : expert_vram_cache) {
            if (loc.layer_idx == layer && loc.expert_id == expert_id && loc.device_ptr == device_ptr) {
                loc.temp_pin_count++;
                return true;
            }
        }
        return false;
    }
    void release_hold(int layer, int expert_id, const uint8_t* device_ptr, bool from_active_layer) {
        GPU_TOPO_TIMED_LOCK(lock);
        PERF_SCOPE(t_release_hold);
        if (from_active_layer) {
            for (auto& loc : active_layer_slots) {
                if (loc.layer_idx == layer && loc.expert_id == expert_id && loc.device_ptr == device_ptr) {
                    // A demand async upload owns TWO holds: the async issue's
                    // (+1, released here only after the consumer's compute
                    // has synchronized) and this compute hold's (+1). Publish
                    // the slot once both are gone. find_active_expert's poll
                    // may already have released the async hold and cleared
                    // the flag; in that case this decrement releases only
                    // the compute hold. This method is called only after
                    // compute completion by contract, so the DMA is
                    // guaranteed complete here as well.
                    if (loc.demand_async_pending) {
                        loc.demand_async_pending = false;
                        loc.upload_ready = true;
                        if (loc.temp_pin_count > 0) loc.temp_pin_count--;
                    }
                    if (loc.temp_pin_count > 0) loc.temp_pin_count--;
                    return;
                }
            }
            LOG_ERROR("[MOE-VRAM]  release_hold (active-layer) found no matching resident slot for "
                      "layer=%d expert=%d -- a temp-pinned slot's identity changed while held, which "
                      "should never happen\n", layer, expert_id);
            return;
        }
        uint64_t key = swap_key(layer, expert_id);
        auto it = expert_index.find(key);
        if (it != expert_index.end()) {
            size_t idx = it->second;
            if (idx < expert_vram_cache.size()) {
                ExpertLocation& loc = expert_vram_cache[idx];
                if (loc.layer_idx == layer && loc.expert_id == expert_id && loc.device_ptr == device_ptr) {
                    if (loc.temp_pin_count > 0) loc.temp_pin_count--;
                    return;
                }
            }
        }
        for (auto& loc : expert_vram_cache) {
            if (loc.layer_idx == layer && loc.expert_id == expert_id && loc.device_ptr == device_ptr) {
                if (loc.temp_pin_count > 0) loc.temp_pin_count--;
                return;
            }
        }
        // Defensive cross-tier recovery. A few legacy/deferred callers do
        // not carry the source-pool bit alongside a raw device pointer. If
        // the main-cache identity is gone but the exact pointer is still a
        // live active-layer slot, release that slot instead of leaking its
        // temporary pin and reporting a false identity-change error. The
        // pointer comparison is required: layer/expert alone is not enough
        // because the same identity may be re-materialized in the other
        // pool during a promotion.
        for (auto& loc : active_layer_slots) {
            if (loc.layer_idx == layer && loc.expert_id == expert_id &&
                loc.device_ptr == device_ptr) {
                if (loc.demand_async_pending) {
                    loc.demand_async_pending = false;
                    loc.upload_ready = true;
                    if (loc.temp_pin_count > 0) loc.temp_pin_count--;
                }
                if (loc.temp_pin_count > 0) loc.temp_pin_count--;
                return;
            }
        }
        // No match: the slot's identity changed between hold and release,
        // which should be IMPOSSIBLE by construction (evict_lru must never
        // pick/clear a temp_pin_count > 0 slot -- see its own guard). Loud,
        // not silent, since a mismatched hold/release is exactly the class
        // of bug this mechanism exists to prevent.
        LOG_ERROR("[MOE-VRAM]  release_hold found no matching resident slot for layer=%d expert=%d "
                  "-- a temp-pinned slot's identity changed while held, which should never happen\n",
                  layer, expert_id);
    }

    // Evict lowest recency+frequency score and return a free slot for a new
    // expert upload. The slot's device_ptr is pre-assigned (slab offset),
    // ready for cudaMemcpy.
    //
    // out_old_layer/out_old_expert, if non-null, receive the identity that
    // was JUST evicted (-1/-1 if the claimed slot was already empty, i.e.
    // no real eviction happened). Default nullptr preserves every existing
    // call site unchanged. Added for vram_evict_demote_to_ram (see
    // on_vram_evict above): upload_expert needs to know what identity it
    // just displaced so it can hand the outgoing bytes to the RAM tier
    // BEFORE overwriting this slot with the incoming H2D copy -- evict_lru
    // already computes this identity internally (for vram_swap_stats/
    // expert_index bookkeeping) right before clearing it, so this just
    // surfaces a value that was already being thrown away.
    // dual_role_gpu_id: the CoreEngine GPU's device index, ONLY when it
    // also hosts a genuine (>0 slot) dual-role vault slice AND at least
    // one OTHER vault GPU exists too (vault_gpus.size() > 1) -- the second
    // condition matters: a single-GPU dense+moe deployment has core_gpu
    // IN vault_gpus but nothing else to fall back to, so treating it as
    // "the dual-role GPU to exclude by default" would leave evict_lru's
    // default (unrestricted) scan with nothing to consider at all. Used
    // by evict_lru's own default-exclusion below and by the pilot-driven
    // GPU1-targeted promotion call site (execution_layer.cpp) to know
    // which device index to explicitly target. Returns -1 (no dual-role
    // GPU to treat specially) otherwise.
    int dual_role_gpu_id() const {
        if (core_gpu < 0 || vault_gpus.size() <= 1) return -1;
        return (std::find(vault_gpus.begin(), vault_gpus.end(), core_gpu) != vault_gpus.end())
            ? core_gpu : -1;
    }

    // restrict_gpu_id: -1 (default) = the general/unrestricted pool, EXACTLY
    // as evict_lru behaved before 2026-08-16's Idea-1/Direction-1 work --
    // every existing caller (upload_expert, upload_expert_async's own
    // default, promote_active_to_main, startup PIN preload, reactive
    // promote_ram_to_vram, etc.) keeps calling this with no new argument
    // and sees NO behavior change at all. Any value >= 0 instead means
    // "ONLY consider slots physically on this one device" -- used
    // exclusively by the explicit GPU1-targeted promotion paths
    // (RAMSafetyManager::promote_ram_to_vram_gpu1, reactive_promote_gpu1_
    // every_n's routing) to stage something specifically into the
    // dual-role slice.
    //
    // REGRESSION FOUND AND FIXED 2026-08-16: an earlier version of this
    // function made restrict_gpu_id=-1 ALSO exclude the dual-role GPU by
    // default (to make GPU1 "prediction-only" for the since-reverted Idea
    // 1 speculative mechanism) -- but that changed the DEFAULT for every
    // other caller too, not just the new one, silently zeroing out GPU1's
    // dual-role slice for the rest of the session every time (confirmed on
    // real hardware: [VRAM-SLOTS] showed exactly GPU0's own slot count,
    // 1322/1410, i.e. GPU1's ~88 slots sat completely empty all session,
    // whenever the new opt-in features were off -- which is the default,
    // so this was silently active in every "baseline" run since it landed,
    // not just the Idea-1 test runs). Real, measurable throughput cost,
    // not noise -- caught by the user questioning a persistent ~0.60-0.64
    // vs 0.68 tok/s gap that survived a reboot. Since Direction 1 (routing
    // REAL reactive demand to GPU1, not speculation) is the surviving
    // approach, the whole "GPU1 must be prediction-only" premise this
    // exclusion existed for is obsolete anyway -- removed outright rather
    // than gated behind yet another flag.
    // required_layer: -1 (default) = any layer (legacy behavior). In compact
    // mode a non-negative value restricts the victim to the region whose
    // slots are sized for that layer. This is required because a compact
    // slot is not interchangeable with a slot from another layer.
    ExpertLocation* evict_lru(size_t needed_bytes, int* out_old_layer = nullptr,
                              int* out_old_expert = nullptr, int restrict_gpu_id = -1,
                              int required_layer = -1) {
        GPU_TOPO_TIMED_LOCK(lock);
        // Whole-function cost -- never directly timed before. Unlike a plain
        // comparison-only scan, this loop can call cudaEventQuery (a real
        // synchronous CUDA driver round-trip, not a cheap CPU-only op) for
        // every candidate slot with upload_ready==false -- a genuinely
        // different cost profile than hold_for_compute/release_hold/
        // find_active_expert's pure-comparison scans, which measured at only
        // 0.65s combined. Called from upload_expert, upload_expert_async
        // (itself untimed until this same investigation pass -- see its own
        // t_upload_expert_async), and promote_active_to_main.
        PERF_SCOPE(t_evict_lru);
        ExpertLocation* victim = nullptr;
        // Iterates unpinned_slot_indices, NOT expert_vram_cache directly --
        // see that vector's own comment. Permanently pinned slots are
        // excluded from this set at the source (pin_expert removes them),
        // so this loop never spends a single iteration on one of the 600+
        // pinned slots a real run can have, unlike before.
        for (size_t _idx : unpinned_slot_indices) {
            auto& loc = expert_vram_cache[_idx];
            if (loc.memcpy_broken) continue;  // permanently excluded -- see its own comment
            if (restrict_gpu_id >= 0 && loc.gpu_id != restrict_gpu_id) {
                continue;  // targeted scan: this device ONLY -- restrict_gpu_id=-1 (default) never filters at all
            }
            if (expert_compact_layout && required_layer >= 0 &&
                loc.pool_layer != required_layer) {
                continue;  // compact slots are reusable only within their layer region
            }
            if (needed_bytes > loc.size) {
                continue;  // never truncate an expert into a smaller compact slot
            }
            if (loc.layer_idx == -1) { victim = &loc; break; } // empty slot preferred
            // Defensive only, should be unreachable by construction: if a
            // slot in unpinned_slot_indices is somehow ALSO pinned (a
            // maintenance-site desync between pin_expert and this vector --
            // the same failure class expert_index's own validation guards
            // against), never trust the vector blindly -- still check and
            // skip. A nonzero n_pinned_eviction_skips now means "desync
            // detected", not "protection working as intended" -- see this
            // counter's dump_report() line for the updated meaning.
            //
            // Never evict a slot whose async upload is still in flight
            // -- this second part is a real correctness fix, not just
            // policy: before this, an in-flight cudaMemcpyAsync into a
            // not-yet-ready slot could be chosen as ANOTHER eviction's
            // victim and overwritten mid-copy (the old code only checked
            // eviction_score, which a fresh in-flight slot satisfies same
            // as any other -- last_used_clock/hit_count don't reflect
            // "copy not done yet").
            // Never evict a permanently pinned slot, a TEMPORARILY held one
            // (decode-time GPU dispatch with kernels still in flight --
            // see ExpertLocation::temp_pin_count), or one whose async
            // upload is still in flight.
            //
            // BUG FOUND (real hardware: report_slot_summary() showed 619 of
            // 1642 VRAM slots -- 37.7% -- permanently stuck "mid-upload" at
            // session end, eroding the effectively-evictable cache well
            // below its nominal size and directly correlating with a hit-
            // rate regression). Root cause: upload_ready only ever gets
            // re-checked (cudaEventQuery, same pattern as below) inside
            // find_expert -- if a speculatively-promoted expert (pilot
            // prefetch's async upload_expert_async) is never subsequently
            // looked up via find_expert, its upload_ready flag stays false
            // FOREVER even though the H2D copy itself finished long ago,
            // and evict_lru here was trusting that stale flag blindly
            // rather than re-checking it -- silently leaking cache
            // capacity every time a speculative promotion goes unconfirmed.
            // Fixed by giving evict_lru the SAME poll-before-trust logic
            // find_expert already uses, so a genuinely-completed slot
            // becomes a real eviction candidate here too, not just usable
            // once something happens to look it up again.
            if (loc.pinned || loc.temp_pin_count > 0) {
                if (loc.pinned) perf().n_pinned_eviction_skips.fetch_add(1, std::memory_order_relaxed);
                if (loc.temp_pin_count > 0) perf().n_temp_pin_eviction_skips.fetch_add(1, std::memory_order_relaxed);
                continue;
            }
            // 2026-08-17: exclude a candidate the pilot's most recent
            // skip-fetch decision JUST confirmed resident -- see
            // pilot_confirmed_set.hpp and n_pilot_protected_eviction_skips_
            // vram's own comment (perf_timer.hpp) for why this can never
            // starve the scan (the confirmed set holds at most 16 entries,
            // all for one single most-recent layer transition). Checked
            // AFTER pinned/temp_pin_count (those are hard correctness
            // requirements; this is a soft preference), so a pilot-
            // confirmed slot that's ALSO pinned still hits the branch above
            // first with the right counter.
            if (loc.layer_idx != -1 && was_pilot_confirmed_resident(loc.layer_idx, loc.expert_id)) {
                perf().n_pilot_protected_eviction_skips_vram.fetch_add(1, std::memory_order_relaxed);
                continue;
            }
            if (!loc.upload_ready) {
                cudaSetDevice(loc.gpu_id);
                if (loc.upload_event && cudaEventQuery(loc.upload_event) == cudaSuccess) {
                    loc.upload_ready = true;
                    perf().n_stale_upload_ready_reclaimed.fetch_add(1, std::memory_order_relaxed);
                } else {
                    perf().n_upload_pending_eviction_skips.fetch_add(1, std::memory_order_relaxed);
                    continue;  // genuinely still copying -- must not evict
                }
            }
            // Prefill advances strictly by layer. Once a layer is complete,
            // its unpinned runtime experts are the safest first victims: they
            // cannot be needed by this forward pass again. Within the same
            // age class, retain the existing recency+frequency policy.
            if (!victim) {
                victim = &loc;
            } else {
                const bool loc_completed =
                    in_prefill && prefill_current_layer >= 0 &&
                    loc.layer_idx < prefill_current_layer;
                const bool victim_completed =
                    in_prefill && prefill_current_layer >= 0 &&
                    victim->layer_idx < prefill_current_layer;
                if (loc_completed != victim_completed ||
                    (loc_completed == victim_completed &&
                     loc.eviction_score() < victim->eviction_score())) {
                    victim = &loc;
                }
            }
        }
        if (victim) {
            // Record the swap-out BEFORE clearing identity below -- an empty
            // slot being claimed (layer_idx == -1) isn't a real eviction of
            // anything, so it's deliberately excluded from churn stats.
            if (victim->layer_idx != -1) {
                if (out_old_layer) *out_old_layer = victim->layer_idx;
                if (out_old_expert) *out_old_expert = victim->expert_id;
                // Measurement only (pilot_confirmed_set.hpp) -- was this
                // EXACT (layer, expert) confirmed already-resident by the
                // pilot's most recent skip-fetch decision, moments before
                // ordinary LRU pressure evicted it anyway? Checked here,
                // not folded into eviction_score() above -- this counts
                // the race, it doesn't (yet) prevent it.
                if (was_pilot_confirmed_resident(victim->layer_idx, victim->expert_id)) {
                    perf().n_evicted_was_pilot_confirmed_vram.fetch_add(1, std::memory_order_relaxed);
                }
                note_startup_preload_eviction(victim->layer_idx, victim->expert_id);
                vram_swap_stats[swap_key(victim->layer_idx, victim->expert_id)].swap_out++;
                // Drop the index entry for the identity being evicted --
                // without this, expert_index would keep pointing this
                // slot's index at the OLD (layer,expert_id) after it's
                // reused for something else, a stale entry find_expert's
                // validation would have to catch on every subsequent
                // lookup for the old identity instead of just missing
                // cleanly.
                expert_index.erase(swap_key(victim->layer_idx, victim->expert_id));
            }
            victim->layer_idx = -1;
            victim->expert_id = -1;
            victim->last_used_clock = 0;
            victim->hit_count = 0;
            victim->upload_ready = true;
            victim->pinned = false;       // defensive: a freed slot must never carry a stale pin
            victim->temp_pin_count = 0;   // defensive: same for the temporary hold count
        }
        return victim;
    }

    // Halve every slot's hit_count periodically so frequency signal reflects
    // recent behavior instead of accumulating forever (bounded "hotness" decay).
    void decay_hit_counts() {
        for (auto& loc : expert_vram_cache) loc.hit_count /= 2;
        hits_since_decay = 0;
    }

    // True if at least one VRAM cache slot is completely unused. Used to
    // gate cold-start "warm preload" promotions: filling an empty slot is
    // free (no eviction), unlike promoting into a full cache which always
    // costs displacing some other expert.
    bool has_free_slot() const {
        GPU_TOPO_TIMED_LOCK(lock);
        for (const auto& loc : expert_vram_cache) {
            if (loc.layer_idx == -1) return true;
        }
        return false;
    }

    // Upload an expert slab from host memory into a VRAM cache slot.
    // Returns the slot if successful, nullptr if cache is full/not available.
    // target_gpu_id: optional hint restricting this upload to a specific
    // vault GPU (passed straight through to evict_lru's own restrict_
    // gpu_id, -1 = unrestricted). Added 2026-09 for pin_core_gpu_top_n
    // (main.cpp's startup [PIN] preload) -- before this, every caller of
    // upload_expert always went through the unrestricted scan, which in
    // practice means "whichever vault GPU evict_lru's scan order happens
    // to check first with a free slot", almost always the much bigger
    // primary vault GPU. Callers that pass a real target_gpu_id and get
    // nullptr back should be prepared to retry with -1 (evict_lru finding
    // no victim on a SPECIFIC, possibly-tiny GPU is an expected outcome,
    // not a hard failure of the whole upload).
    ExpertLocation* upload_expert(int layer, int expert_id,
                                   const uint8_t* host_data, size_t bytes,
                                   int target_gpu_id = -1,
                                   bool startup_plain_copy = false) {
        if (expert_vram_cache_cap == 0 || !host_data || bytes == 0) return nullptr;
        GPU_TOPO_TIMED_LOCK(lock);

        // Check if already cached
        ExpertLocation* existing = find_expert(layer, expert_id);
        if (existing) return existing;

        // Get a slot (evict LRU if needed)
        int old_layer = -1, old_expert = -1;
        ExpertLocation* slot = evict_lru(bytes, &old_layer, &old_expert,
                                          target_gpu_id, layer);
        if (!slot || !slot->device_ptr) {
            // Previously silent -- a real run showed this path (or the
            // cudaMemcpy failure below) responsible for a startup preload
            // losing ~99% of its attempts with zero diagnostic evidence
            // either way. Should be near-impossible on an otherwise-empty
            // cache (every slot pinned/temp-pinned/upload-pending
            // simultaneously) -- if this fires during startup preload with
            // target_gpu_id==-1 (unrestricted), something upstream
            // (vault_gpus/unpinned_slot_indices bookkeeping, likely around
            // a dual-role reindex) is broken. A target_gpu_id-RESTRICTED
            // miss is a different, expected story -- the caller asked for
            // a SPECIFIC (possibly small) GPU's own slots specifically, so
            // "that one GPU is already full" is routine, not alarming
            // (e.g. pin_core_gpu_top_n exceeding the CoreEngine GPU's own
            // small slice once its slots run out) -- DEBUG, not ERROR, and
            // its own counter so it's not conflated with the real-bug case.
            if (target_gpu_id < 0) {
                perf().n_upload_expert_no_victim.fetch_add(1, std::memory_order_relaxed);
                LOG_ERROR("[MOE-VRAM]  upload_expert: evict_lru found NO victim for layer=%d expert=%d "
                          "(cache_cap=%zu, unpinned_slots=%zu) -- every candidate slot is pinned, "
                          "temp-pinned, or upload-pending, or the pool is empty/misconfigured\n",
                          layer, expert_id, expert_vram_cache_cap, unpinned_slot_indices.size());
            } else {
                perf().n_upload_expert_no_victim_restricted.fetch_add(1, std::memory_order_relaxed);
                LOG_DEBUG("[MOE-VRAM]  upload_expert: no victim on restricted target_gpu_id=%d for "
                          "layer=%d expert=%d (that GPU's own slots are full -- caller should retry "
                          "unrestricted)\n", target_gpu_id, layer, expert_id);
            }
            return nullptr;
        }

        // Salvage the outgoing expert's bytes into the RAM tier before this
        // slot's VRAM content is overwritten below (vram_evict_demote_to_ram
        // -- see on_vram_evict's own comment for why this is only wired
        // into this synchronous, decode-thread-only path). old_layer==-1
        // means evict_lru claimed an already-empty slot -- nothing to
        // salvage. Must run strictly before the cudaMemcpy H2D a few lines
        // down, since that's the last point slot->device_ptr still holds
        // the OLD expert's data.
        if (on_vram_evict && old_layer != -1 && old_expert >= 0) {
            on_vram_evict(old_layer, old_expert, slot->device_ptr, slot->size, slot->gpu_id);
        }

        // Copy host → device (synchronous for correctness). Transiently
        // cudaHostRegister/cudaHostUnregister the RAM tier's own slab_buffer
        // bytes directly for the duration of this ONE DMA, instead of
        // memcpy-ing them into a separate pinned bounce buffer first (Alt A
        // of a cudaHostRegister investigation -- see t_host_register_stage's
        // own comment in PerfCounters for the real-hardware numbers: ~25%
        // less combined overhead than the memcpy approach, 1.070ms/call vs
        // 1.428ms/call @ ~8.26MB). Nothing stays pinned beyond this one
        // synchronous call -- safe specifically because upload_expert is
        // fully blocking, so the unregister below can never race an
        // in-flight read the way it would on an async path (see
        // upload_expert_async's own comment for why THAT function is
        // deliberately NOT converted to this same approach: its DMA is
        // still in flight when the function returns, so unregistering
        // host_data before the async copy actually completes would be a
        // real, silent correctness violation -- not a resurrection of the
        // codebase's own documented decode-path corruption incident, but
        // the same *class* of "completion timing" mistake).
        size_t copy_bytes = std::min(bytes, slot->size);
        cudaSetDevice(slot->gpu_id);

        PERF_START(host_register);
        cudaError_t reg_err = cudaSuccess;
        bool registered = false;
        if (!startup_plain_copy) {
            reg_err = cudaHostRegister((void*)host_data, copy_bytes, cudaHostRegisterPortable);
            registered = (reg_err == cudaSuccess);
        }
        PERF_END(host_register, t_host_register_stage);
        if (!registered && !startup_plain_copy) {
            // cudaErrorHostMemoryAlreadyRegistered is now an EXPECTED, benign
            // outcome, not the "real double-registration bug" this comment
            // originally assumed it could only mean: with EngineConfig::
            // ram_pinned_budget_gb set, the RAM tier persistently pins its
            // own slabs once at allocation (see ensure_slab_capacity), so
            // this per-call register correctly finds them already pinned.
            // The copy below then DMAs straight from that already-page-locked
            // source -- exactly what we want -- and `registered` staying
            // false is also correct, since this call must not unregister a
            // pin it does not own.
            if (reg_err == cudaErrorHostMemoryAlreadyRegistered) {
                cudaGetLastError();  // benign -- don't leak into a later unrelated check
                perf().n_upload_expert_src_already_pinned.fetch_add(1, std::memory_order_relaxed);
            } else {
                perf().n_host_register_fail.fetch_add(1, std::memory_order_relaxed);
                LOG_WARN("[MOE-VRAM]  upload_expert: cudaHostRegister failed for layer=%d expert=%d "
                          "(%zu bytes): %s -- falling back to direct pageable-source copy\n",
                          layer, expert_id, copy_bytes, cudaGetErrorString(reg_err));
            }
        }

        // Issued on the vault GPU's own copy_stream (via cudaMemcpyAsync +
        // an immediate cudaStreamSynchronize) instead of a plain cudaMemcpy,
        // which implicitly runs on the legacy default stream (stream 0).
        // Same fix, same real-hardware evidence, as demote_from_vram's own
        // D2H copy (see CLAUDE.md): stream-0 ops force a full wait for
        // everything queued on every OTHER stream on the device (e.g.
        // compute_stream's in-flight expert FFN kernels) before they can
        // even start. cudaStreamSynchronize below still blocks this
        // function until the copy genuinely completes before returning --
        // identical synchronous-completion contract to the old cudaMemcpy,
        // NOT the kind of async change that caused the documented decode-
        // path corruption incident (that was about a completion EVENT
        // being trusted before the reading kernels had finished, letting a
        // background thread reclaim the slot mid-compute -- this function
        // still fully blocks before returning, so no such event-vs-actual-
        // completion gap exists here).
        PERF_START(gpu_sync_upload);
        cudaStream_t upload_stream = devices[slot->gpu_id].copy_stream;
        cudaError_t err;
        if (startup_plain_copy) {
            // Startup preload runs before inference and has no compute to
            // overlap. Keep this path independent of pageable-source
            // async-copy staging and transient host registration; those
            // operations are useful for runtime promotion but produced
            // invalid-argument failures during large one-time preloads on
            // this machine. The caller still gets the same synchronous
            // completion contract.
            err = cudaMemcpy(slot->device_ptr, host_data, copy_bytes, cudaMemcpyHostToDevice);
        } else if (upload_stream) {
            err = cudaMemcpyAsync(slot->device_ptr, host_data, copy_bytes, cudaMemcpyHostToDevice, upload_stream);
            if (err == cudaSuccess) err = cudaStreamSynchronize(upload_stream);
        } else {
            perf().n_upload_expert_stream_unavailable.fetch_add(1, std::memory_order_relaxed);
            err = cudaMemcpy(slot->device_ptr, host_data, copy_bytes, cudaMemcpyHostToDevice);
        }
        PERF_END(gpu_sync_upload, t_gpu_upload);
        if (err == cudaSuccess && slot->gpu_id >= 0 && (size_t)slot->gpu_id < PerfCounters::PERF_MAX_GPU_SLOTS) {
            perf().bytes_h2d_upload_by_gpu[slot->gpu_id].fetch_add((int64_t)copy_bytes, std::memory_order_relaxed);
            perf().n_h2d_upload_by_gpu[slot->gpu_id].fetch_add(1, std::memory_order_relaxed);
        }

        // Unregister BEFORE checking err/returning -- must run on every path
        // (success or failure) since a registration that's never undone
        // would otherwise leak, and freeing/reallocating this slab later
        // (ensure_slab_capacity) while still registered would be undefined
        // behavior.
        if (registered) {
            PERF_START(host_unregister);
            cudaHostUnregister((void*)host_data);
            PERF_END(host_unregister, t_host_register_stage);
            perf().bytes_host_registered.fetch_add((int64_t)copy_bytes, std::memory_order_relaxed);
        }

        if (err != cudaSuccess) {
            // Previously silent -- see evict_lru's own no-victim log above
            // for why this visibility was added. A real cudaMemcpy failure
            // here (vs. the no-victim case) points at VRAM exhaustion, a
            // stale/invalid device_ptr, or a corrupted CUDA context rather
            // than a cache-bookkeeping bug.
            perf().n_upload_expert_memcpy_fail.fetch_add(1, std::memory_order_relaxed);
            LOG_ERROR("[MOE-VRAM]  upload_expert: cudaMemcpy H2D failed layer=%d expert=%d "
                      "gpu=%d bytes=%zu dst=%p host_src=%p: %s\n",
                      layer, expert_id, slot->gpu_id, copy_bytes, (void*)slot->device_ptr,
                      (void*)host_data, cudaGetErrorString(err));
            // This synchronous startup path treats a failed preload as a
            // rejected residency hint and continues with the expert in RAM.
            // Clear the handled CUDA error before the next startup upload or
            // the first GGML graph call; otherwise CUDA can report the same
            // stale invalid-argument state from an unrelated later API call.
            cudaGetLastError();
            // Permanently exclude this slot from future eviction candidacy --
            // a real run showed evict_lru re-selecting the SAME still-"empty"
            // slot on every subsequent call (a failed write never marks a
            // slot occupied, so it stays the "first empty slot" forever),
            // funneling ~1800 preload attempts into one broken address
            // instead of ever reaching the pool's other, presumably-healthy
            // slots. Whatever is wrong with this address (bad device_ptr
            // computation, or a genuinely faulty VRAM region) won't fix
            // itself by retrying -- see ExpertLocation::memcpy_broken.
            ++slot->memcpy_fail_count;
            if (slot->memcpy_fail_count >= 2) {
                slot->memcpy_broken = true;
                LOG_ERROR("[MOE-VRAM]  upload_expert: permanently excluding slot at dst=%p from future "
                          "eviction candidacy (gpu=%d) -- this SAME address has now failed %d times, "
                          "so the fault really is destination-side\n",
                          (void*)slot->device_ptr, slot->gpu_id, slot->memcpy_fail_count);
            } else {
                // First failure on this slot: most likely a bad SOURCE
                // pointer being handed to a healthy destination. Retiring
                // the slot here would punish the wrong object and shrink
                // usable VRAM cache for a host-side bug (which is exactly
                // what this workaround was doing before 2026-09-01).
                LOG_ERROR("[MOE-VRAM]  upload_expert: H2D failed once at dst=%p (gpu=%d) -- NOT retiring "
                          "the slot; a bad host_src fails against any destination, so one failure is not "
                          "evidence this address is faulty. Retires only on a second failure here.\n",
                          (void*)slot->device_ptr, slot->gpu_id);
            }
            return nullptr;
        }

        perf().bytes_gpu_uploaded.fetch_add((int64_t)copy_bytes, std::memory_order_relaxed);
        perf().n_expert_promote.fetch_add(1, std::memory_order_relaxed);

        slot->layer_idx = layer;
        slot->expert_id = expert_id;
        slot->last_used_clock = ++cache_clock;
        vram_swap_stats[swap_key(layer, expert_id)].swap_in++;
        expert_index[swap_key(layer, expert_id)] = (size_t)(slot - expert_vram_cache.data());
        return slot;
    }

    // Direct disk->VRAM read via NVIDIA cuFile (GPUDirect Storage), skipping
    // the host-RAM hop entirely -- the "genuine double miss" fast path (see
    // RAMSafetyManager::is_ram_resident and execution_layer.cpp's own
    // gating: this must NEVER be called for an expert that's already
    // host-RAM-resident, only for one that's in neither tier). Mirrors
    // upload_expert's eviction/hold discipline EXACTLY -- same
    // find_expert/evict_lru/on_vram_evict/bookkeeping sequence, deliberately
    // NOT reinvented here -- the only thing that changes is the actual byte
    // transfer: 1-3 synchronous cuFileRead calls straight into
    // slot->device_ptr instead of one cudaMemcpy from a host buffer.
    // (gate_off,gb)/(up_off,ub)/(down_off,db) are this expert's three
    // projections' absolute file offsets + byte sizes (same convention
    // load_expert_slab/build_expert_read_requests already use) -- when they
    // happen to be contiguous (always true for unpacked_expert_format, per
    // its own startup sanity check; sometimes true for packed format) this
    // issues ONE combined cuFileRead instead of three, same "single DMA"
    // reasoning as the unpacked-format work.
    //
    // Caller contract (not re-validated here, to keep this symmetric with
    // upload_expert's own "trust the caller" style): `h.valid` must already
    // be true, and the caller should already know (via
    // devices[?].cufile_pool_registered) that SOME vault device has a
    // registered pool -- but evict_lru() below is device-agnostic (it can
    // return a slot on ANY vault GPU, per the existing shared
    // expert_vram_cache indexing), so if it happens to pick a slot on a
    // device whose pool was NOT successfully registered (only possible in a
    // multi-vault-GPU setup where cuFileBufRegister failed on just one of
    // them), this returns nullptr after having already evicted that slot's
    // PREVIOUS occupant for nothing -- a real, narrow inefficiency (one
    // wasted eviction cycle in an already-rare edge case), not a
    // correctness issue: evict_lru already left the slot in a valid "empty"
    // state before this check runs, and the caller's existing
    // ram_needed_ids fallback still resolves this expert correctly on the
    // very next path. Not worth adding device-targeting to the shared,
    // already-delicate evict_lru for this narrow a case -- see CLAUDE.md's
    // extensive history of eviction-logic regressions before touching that
    // function for any reason.
    ExpertLocation* upload_expert_cufile(int layer, int expert_id,
                                          const cufile_io::FileHandle& h,
                                          uint64_t gate_off, size_t gb,
                                          uint64_t up_off, size_t ub,
                                          uint64_t down_off, size_t db) {
        size_t bytes = gb + ub + db;
        if (expert_vram_cache_cap == 0 || bytes == 0 || !h.valid) return nullptr;
        GPU_TOPO_TIMED_LOCK(lock);

        ExpertLocation* existing = find_expert(layer, expert_id);
        if (existing) return existing;

        int old_layer = -1, old_expert = -1;
        ExpertLocation* slot = evict_lru(bytes, &old_layer, &old_expert, -1, layer);
        if (!slot || !slot->device_ptr) {
            perf().n_cufile_read_fail.fetch_add(1, std::memory_order_relaxed);
            LOG_ERROR("[CUFILE]  upload_expert_cufile: evict_lru found NO victim for layer=%d "
                      "expert=%d (cache_cap=%zu, unpinned_slots=%zu)\n",
                      layer, expert_id, expert_vram_cache_cap, unpinned_slot_indices.size());
            return nullptr;
        }
        if (!devices[slot->gpu_id].cufile_pool_registered) {
            // See this function's own header comment -- evict_lru is
            // device-agnostic and already left this slot correctly empty;
            // nothing to undo, just decline to write into an unregistered
            // pool.
            perf().n_cufile_read_fail.fetch_add(1, std::memory_order_relaxed);
            return nullptr;
        }

        // Salvage the outgoing expert into the RAM tier -- identical to
        // upload_expert's own call, same reasoning (see on_vram_evict's
        // comment). Must run before the reads below overwrite this slot's
        // old VRAM content.
        if (on_vram_evict && old_layer != -1 && old_expert >= 0) {
            on_vram_evict(old_layer, old_expert, slot->device_ptr, slot->size, slot->gpu_id);
        }

        cudaSetDevice(slot->gpu_id);
        // Own timer (t_cufile_read), deliberately NOT sharing t_gpu_upload
        // (the cudaMemcpy H2D counter) -- this session's own history (see
        // CLAUDE.md's disk=/expert= investigation) is a direct lesson in
        // how conflating two genuinely different mechanisms into one
        // counter hides exactly the kind of regression a real A/B needs to
        // see separately.
        PERF_START(cufile_read);
        bool contiguous = (up_off == gate_off + gb) && (down_off == up_off + ub);
        bool ok;
        if (contiguous) {
            ssize_t r = cufile_io::read(h, slot->device_ptr, bytes, (off_t)gate_off, 0);
            ok = (r >= 0 && (size_t)r == bytes);
        } else {
            ssize_t r1 = cufile_io::read(h, slot->device_ptr, gb, (off_t)gate_off, 0);
            ssize_t r2 = cufile_io::read(h, slot->device_ptr, ub, (off_t)up_off, (off_t)gb);
            ssize_t r3 = cufile_io::read(h, slot->device_ptr, db, (off_t)down_off, (off_t)(gb + ub));
            ok = r1 >= 0 && (size_t)r1 == gb && r2 >= 0 && (size_t)r2 == ub && r3 >= 0 && (size_t)r3 == db;
        }
        PERF_END(cufile_read, t_cufile_read);
        if (!ok) {
            perf().n_cufile_read_fail.fetch_add(1, std::memory_order_relaxed);
            LOG_ERROR("[CUFILE]  upload_expert_cufile: cuFileRead failed layer=%d expert=%d gpu=%d "
                      "bytes=%zu dst=%p\n", layer, expert_id, slot->gpu_id, bytes, (void*)slot->device_ptr);
            // Same permanent-exclusion discipline as upload_expert's own
            // cudaMemcpy-failure path -- see ExpertLocation::memcpy_broken
            // and upload_expert's identical comment on why a failed
            // address is excluded rather than retried.
            slot->memcpy_broken = true;
            return nullptr;
        }

        perf().bytes_cufile_read.fetch_add((int64_t)bytes, std::memory_order_relaxed);
        perf().n_expert_promote.fetch_add(1, std::memory_order_relaxed);
        perf().n_cufile_read_ok.fetch_add(1, std::memory_order_relaxed);

        slot->layer_idx = layer;
        slot->expert_id = expert_id;
        slot->last_used_clock = ++cache_clock;
        vram_swap_stats[swap_key(layer, expert_id)].swap_in++;
        expert_index[swap_key(layer, expert_id)] = (size_t)(slot - expert_vram_cache.data());
        return slot;
    }

    // Fire-and-forget background promotion: for "this expert has proven hot"
    // signals where the caller does NOT need the data resident immediately
    // (unlike upload_expert()). Uses the vault GPU's dedicated copy_stream so
    // it never blocks the CPU decode critical path. The slot is marked
    // not-ready until the async copy completes (checked via cudaEventQuery in
    // find_expert), so no compute can race a partially-written slab.
    // target_gpu_id: -1 (default, every existing caller) = the general
    // pool (see evict_lru's own default-exclusion). >= 0 targets ONE
    // specific device -- currently only used by the pilot-driven GPU1
    // promotion path (RAMSafetyManager::promote_ram_to_vram_gpu1).
    // src_already_pinned: caller-provided hint that host_data is already
    // PERSISTENTLY page-locked (MoEExpertBlock::slab_pinned -- set once by
    // ensure_slab_capacity under EngineConfig::ram_pinned_budget_gb). When
    // true, this function skips the speculative cudaHostRegister attempt
    // below entirely instead of trying it and catching the resulting
    // cudaErrorHostMemoryAlreadyRegistered -- functionally identical (both
    // paths end up doing a direct, unregistered-by-this-call DMA from
    // host_data), but a real, confirmed difference under compute-sanitizer:
    // sanitizer flags ANY non-success CUDA API return as an "error" even
    // when the caller handles it correctly, so the old always-attempt
    // pattern produced one flagged (but harmless) error per promotion once
    // ram_pinned_budget_gb was raised into its "most slabs already pinned"
    // range (>=30GB -- see that field's own comment) -- confirmed via a
    // real compute-sanitizer run against a live GLM-5.3 generation: 1375
    // errors, all this exact call/class, zero actual memory-safety issues,
    // output coherent throughout. Passing the hint when known removes the
    // doomed call instead of just tolerating its result -- default false so
    // any caller that doesn't know its pin state keeps the old, still-safe,
    // reactive fallback.
    ExpertLocation* upload_expert_async(int layer, int expert_id,
                                         const uint8_t* host_data, size_t bytes,
                                         int target_gpu_id = -1,
                                         bool src_already_pinned = false) {
        if (expert_vram_cache_cap == 0 || !host_data || bytes == 0) return nullptr;
        GPU_TOPO_TIMED_LOCK(lock);
        // Whole-function cost -- had ZERO instrumentation of any kind before
        // this (not even a memcpy sub-timer like upload_expert's
        // t_gpu_upload), despite calling both find_expert and evict_lru
        // internally (each now separately timed too -- this will overlap/
        // nest with their own PERF_SCOPEs, which is fine, same deliberate
        // pattern as t_prefill_total vs t_attn elsewhere in this codebase).
        // This is the RAM->VRAM reactive-hit MOVE-promotion path
        // (RAMSafetyManager::get_expert/get_experts_batch, fire-and-forget)
        // AND moe_forward_gguf's Phase 2 cold-start VRAM warm-up call, so it
        // runs on the decode thread, not just the prefetcher worker thread.
        PERF_SCOPE(t_upload_expert_async);

        ExpertLocation* existing = find_expert(layer, expert_id);
        // TEMPORARY diagnostic (2026-08-16, Idea 1 churn investigation --
        // see doc/gpu-utilization-vram-analysis.md): target_gpu_id >= 0
        // means this call came from promote_ram_to_vram_gpu1 specifically.
        // Distinguishes "found already resident (no new transfer -- either
        // a prior k4 promotion still sitting there, or already on GPU0)"
        // from "genuinely fresh upload" for THAT call site only, without
        // touching the general-purpose counters every other caller shares.
        if (target_gpu_id >= 0) {
            if (existing) perf().n_predict_gpu1_already_resident.fetch_add(1, std::memory_order_relaxed);
            else perf().n_predict_gpu1_fresh_upload.fetch_add(1, std::memory_order_relaxed);
        }
        if (existing) return existing;

        ExpertLocation* slot = evict_lru(bytes, nullptr, nullptr, target_gpu_id, layer);
        if (!slot || !slot->device_ptr) return nullptr;

        size_t copy_bytes = std::min(bytes, slot->size);
        GpuDevice& dev = devices[slot->gpu_id];
        cudaSetDevice(dev.id);
        slot->upload_ready = false;

        // Opportunistic cleanup of earlier calls' deferred unregisters --
        // see pending_host_unregisters' own comment. Cheap (a linear scan
        // of a normally-small, actively-draining list) and keeps the list
        // from growing unbounded even on a run that never happens to hit
        // the other reap call site (the destructor).
        reap_pending_host_unregisters();

        // cudaHostRegister the RAM tier's own slab_buffer bytes directly,
        // DMA straight from host_data -- same Alt-A approach as
        // upload_expert (see that function's comment for the real-hardware
        // numbers), replacing the CPU-side bounce-buffer memcpy entirely
        // for the common case, not just relocating it. Falls back to the
        // OLD bounce-buffer staging on any registration failure (pool
        // exhaustion equivalent -- e.g. cudaErrorHostMemoryAlreadyRegistered
        // if this exact range is already mid-registration from an earlier,
        // not-yet-reaped call) or if this slot has no upload_event to defer
        // against (should not happen for a real pool slot -- see
        // allocate_expert_cache's unconditional event creation -- but
        // registering with no way to safely know when to unregister would
        // be a real leak, not a degraded-but-safe path, so this is treated
        // as a hard precondition, not an optimization to skip opportunistically).
        const uint8_t* dma_src = host_data;
        BounceBuffer* bb = nullptr;
        bool registered = false;
        // Set when host_data turns out to be PERSISTENTLY pinned already
        // (EngineConfig::ram_pinned_budget_gb -- see ensure_slab_capacity).
        // In that case this call must NOT stage through a bounce buffer:
        // the source is already page-locked, so a direct DMA from it is
        // both correct and strictly faster, and there is nothing for THIS
        // call to unregister afterwards (it never registered anything --
        // the persistent owner does that, once, at allocation). Without
        // this branch a persistent pin actively DEGRADES this path, since
        // the failed register would fall through to the bounce-buffer
        // memcpy below -- a real, measured regression (stage_cpy 0.00s ->
        // 1.28s on the first real run with pinning enabled).
        bool already_pinned = src_already_pinned;
        if (already_pinned) {
            // Caller already told us -- skip the doomed cudaHostRegister
            // call entirely (see this function's own header comment for
            // why this is more than just tidiness: it removes a real,
            // compute-sanitizer-flagged CUDA API error, not just tolerates
            // it after the fact).
            perf().n_upload_async_src_already_pinned.fetch_add(1, std::memory_order_relaxed);
        } else if (slot->upload_event) {
            PERF_START(upload_async_host_register);
            cudaError_t reg_err = cudaHostRegister((void*)host_data, copy_bytes, cudaHostRegisterPortable);
            registered = (reg_err == cudaSuccess);
            PERF_END(upload_async_host_register, t_upload_async_host_register);
            if (!registered) {
                already_pinned = (reg_err == cudaErrorHostMemoryAlreadyRegistered);
                if (already_pinned) {
                    cudaGetLastError();  // benign -- don't leak into a later unrelated check
                    perf().n_upload_async_src_already_pinned.fetch_add(1, std::memory_order_relaxed);
                } else {
                    perf().n_upload_async_host_register_fail.fetch_add(1, std::memory_order_relaxed);
                }
            }
        }
        if (!registered && !already_pinned) {
            // Same pinned bounce-buffer staging as upload_expert used to --
            // see its comment (now historical there, still live here as the
            // fallback). Ping-ponging across bounce_pool's N buffers is
            // what gives this path real double buffering: while buffer[i]'s
            // DMA is still draining on copy_stream, the NEXT call's memcpy
            // into buffer[i+1] can proceed immediately without waiting on it.
            bb = acquire_bounce_buffer();
            if (bb) {
                // Same staging cost bucket upload_expert's sync path used to
                // share this with, back when it also used the bounce pool --
                // kept as the same counter since it's still the same
                // operation, just upload_expert_async-only now.
                PERF_START(bounce_stage_async);
                memcpy(bb->host_ptr, host_data, copy_bytes);
                PERF_END(bounce_stage_async, t_bounce_stage_memcpy);
                perf().bytes_bounce_staged.fetch_add((int64_t)copy_bytes, std::memory_order_relaxed);
                dma_src = bb->host_ptr;
            }
        }
        cudaMemcpyAsync(slot->device_ptr, dma_src, copy_bytes, cudaMemcpyHostToDevice, dev.copy_stream);
        if (slot->gpu_id >= 0 && (size_t)slot->gpu_id < PerfCounters::PERF_MAX_GPU_SLOTS) {
            perf().bytes_h2d_upload_by_gpu[slot->gpu_id].fetch_add((int64_t)copy_bytes, std::memory_order_relaxed);
            perf().n_h2d_upload_by_gpu[slot->gpu_id].fetch_add(1, std::memory_order_relaxed);
        }
        if (slot->upload_event) cudaEventRecord(slot->upload_event, dev.copy_stream);
        if (bb && bb->in_flight) cudaEventRecord(bb->in_flight, dev.copy_stream);
        if (registered) {
            // Unregister deferred to reap_pending_host_unregisters() (this
            // function's own next call, or the destructor) once
            // slot->upload_event confirms the DMA above has actually
            // completed -- see pending_host_unregisters' own comment for
            // why checking that event is safe even though it's a per-slot
            // event that may be re-recorded before this entry is reaped.
            pending_host_unregisters.push_back({(void*)host_data, slot->upload_event});
            perf().bytes_upload_async_host_registered.fetch_add((int64_t)copy_bytes, std::memory_order_relaxed);
        }

        // Record promotion metrics immediately (bytes enqueued + count).
        perf().bytes_gpu_uploaded.fetch_add((int64_t)copy_bytes, std::memory_order_relaxed);
        perf().n_expert_promote.fetch_add(1, std::memory_order_relaxed);

        slot->layer_idx = layer;
        slot->expert_id = expert_id;
        slot->last_used_clock = ++cache_clock;
        vram_swap_stats[swap_key(layer, expert_id)].swap_in++;
        expert_index[swap_key(layer, expert_id)] = (size_t)(slot - expert_vram_cache.data());
        return slot;
    }

    // ========== Three-Tier Caching: Active Layer Area ==========

    // Allocate active layer area for temporary current-layer expert storage
    // Returns true if successful, false on allocation failure
    bool allocate_active_layer_area(size_t expert_slab_bytes, int max_experts_per_layer);

    // 2026-08-28: multi-GPU variant. `requests` with fewer than 2 entries
    // (i.e. at most one GPU opted in via active_layer_weight > 0) falls
    // through to the plain single-GPU allocate_active_layer_area above,
    // UNCHANGED -- this is what keeps the feature fully opt-in and
    // byte-identical to legacy when not configured. Each request's own
    // capacity_slots (falling back to legacy_max_experts_per_layer if 0)
    // sizes that GPU's own allocation independently.
    struct ActiveLayerGpuRequest { int gpu_id; int weight; int capacity_slots; };
    bool allocate_active_layer_area_multi(size_t expert_slab_bytes, int legacy_max_experts_per_layer,
                                           const std::vector<ActiveLayerGpuRequest>& requests);

    // Upload expert to active layer area (temporary, never cached)
    // Returns slot pointer or nullptr if area is full/allocation failed
    // This is decode-thread-only and never participates in LRU eviction
    // speculative (2026-08-26, default false): set true ONLY by
    // RAMSafetyManager::promote_ram_to_active_layer (pilot one-layer-ahead
    // prediction) -- marks the returned slot pilot_speculative_fill=true
    // and prewarm_skip_next_promote=true instead of the plain reactive-
    // cold-miss defaults, mirroring upload_expert_active_async's own
    // treatment of a genuine pre-warm. Every OTHER existing caller (real
    // dispatch-time cold miss) passes the default false, unchanged.
    // target_gpu_id (2026-08-28, default -1): when >= 0, restricts BOTH
    // the pre-check (is this identity already present/in-flight -- see the
    // duplicate-transfer fix's own comment) and victim-selection to slots
    // belonging to that GPU's own active_layer_gpu_slices entry, instead of
    // scanning the whole (possibly multi-GPU) active_layer_slots vector.
    // -1 (legacy default) scans everything, unchanged -- the correct value
    // whenever active_layer_gpu_slices is empty (single-GPU mode) or a
    // caller has no routing decision to make.
    ExpertLocation* upload_expert_active(int layer, int expert_id,
                                        const uint8_t* host_data, size_t bytes,
                                        bool speculative = false, int target_gpu_id = -1);

    // Async sibling of upload_expert_active, for BATCH pre-warming a whole
    // layer's known-needed active-layer misses before Phase 3's per-token
    // dispatch loop starts (execution_layer.cpp's moe_forward_gguf, S>1),
    // instead of one blocking cudaMemcpy per cold miss scattered through
    // that loop -- see t_active_layer_upload_async's own comment in
    // perf_timer.hpp for the real-hardware finding that motivated this
    // (active_up costing ~3x the full-run per-call average during
    // prefill's own tight burst).
    //
    // CORRECTNESS CONTRACT, read before calling: the returned slot's
    // upload_ready is left FALSE and its temp_pin_count is bumped by 1 --
    // the DMA has only been ISSUED on this GPU's copy_stream, not
    // necessarily completed. find_active_expert does NOT check
    // upload_ready (confirmed by reading it -- it only matches identity),
    // so a slot returned by this function must NEVER be exposed to a
    // find_active_expert lookup, by any thread, before
    // sync_active_layer_batch() has confirmed the whole batch this slot
    // belongs to. In THIS codebase that's satisfied structurally, not just
    // by convention: find_active_expert is decode-thread-only (prefetch
    // workers never touch VRAM/GpuTopology -- see claim_or_wait's own
    // CRITICAL rule), and every intended caller issues a full batch then
    // calls sync_active_layer_batch() BEFORE returning to Phase 3 on that
    // SAME thread -- so Phase 3 can never run, on any layer, before its
    // own layer's batch has already been synced. Do not call this and
    // then let control reach any find_active_expert call site (on any
    // thread) without an intervening sync_active_layer_batch() for this
    // exact slot -- that would be a real use-of-incomplete-data race, not
    // a hypothetical one. temp_pin_count already protects against a
    // SECOND call in the same batch picking this same slot before the
    // first's copy lands, same discipline upload_expert_active's own
    // victim loop already uses.
    // speculative (2026-08-26, default false): set true ONLY by
    // RAMSafetyManager::promote_ram_to_active_layer (pilot one-layer-ahead
    // prediction) -- marks the returned slot pilot_speculative_fill=true
    // instead of was_async_prewarmed=true, and counts into n_pilot_active_
    // fill_attempted instead of n_active_layer_uploads_async_issued, so
    // this source's own used/wasted rate is never blended with decode_
    // async_prewarm/prefill's Phase-2 batch pre-warm (both existing,
    // structurally different callers of this same function). Caller is
    // responsible for eventually calling sync_active_layer_batch on the
    // returned slot, same contract as every other caller.
    // target_gpu_id: same meaning as upload_expert_active's own parameter
    // of the same name -- see its comment.
    // demand (2026-10-22, default true in the executor): a real decode-time cold miss that
    // is about to be dispatched THIS call, not a batch pre-warm. The copy is
    // issued on active_layer_demand_stream (non-blocking) instead of
    // active_layer_stream, the slot is marked demand_async_pending, and the
    // caller must let expert_ffn_gpu_issue fence its compute stream on
    // upload_event before reading the slot. Unlike the prewarm modes this
    // does NOT set was_async_prewarmed/prewarm_skip_next_promote: this is a
    // confirmed need, so its first real touch keeps the normal promotion
    // semantics. The existing async-batch contract (upload_ready=false and
    // one temp_pin_count hold) still applies; release_hold() publishes the
    // slot after the consumer's compute has completed.
    ExpertLocation* upload_expert_active_async(int layer, int expert_id,
                                                const uint8_t* host_data, size_t bytes,
                                                bool speculative = false, int target_gpu_id = -1,
                                                int64_t qwen_pilot_prediction_ns = -1,
                                                bool demand = false);

    // Single-slot Layer-13 proof path.  The graph-RAM arena is projection
    // compact, so the three pinned DDR planes are copied into one active
    // slot as [gate | up | down] on the dedicated active-layer stream.  The
    // completion event is recorded after all three DMA legs; callers fence
    // the GGML CUDA stream with slot->upload_event and never synchronize the
    // host before graph submission.
    ExpertLocation* upload_expert_active_projections_async(
        int layer, int expert_id, const uint8_t * gate_data, size_t gate_bytes,
        const uint8_t * up_data, size_t up_bytes, const uint8_t * down_data,
        size_t down_bytes, int target_gpu_id = -1);

    // Called only after the graph stream has completed the dependent work.
    // This publishes the slot for the next proof transfer without adding a
    // second host-side wait on the active-layer stream.
    void complete_active_layer_upload_after_compute(ExpertLocation *slot);

    // 2026-08-28: victim-selection ONLY, no copy -- for ActiveLayerUploadWorkers
    // (active_layer_upload_workers.hpp), which does the actual register+
    // cudaMemcpyAsync+cudaEventRecord on a background worker thread instead
    // of inline here. Mirrors upload_expert_active_async's own victim-
    // selection exactly (temp_pin_count>0 slots skipped, an unconfirmed
    // speculative fill always outranks a real slot as a victim), but marks
    // the chosen slot copy_dispatched=false instead of issuing anything --
    // see ExpertLocation::copy_dispatched's own comment for why this is
    // load-bearing, not optional. Callers (both real caller sites already
    // filter via active_layer_has() before reaching this point, so no
    // duplicate-transfer pre-check is repeated here) must, immediately
    // after this returns a non-null slot: (1) hand {host_data, bytes,
    // slot->device_ptr, slot->gpu_id, slot} to ActiveLayerUploadWorkers::
    // submit, and (2) protect the RAM-tier source via ram_pin() (NOT
    // pending_copy_event yet -- that transition only happens once the
    // worker confirms cudaEventRecord actually ran, via RAMSafetyManager::
    // ram_mark_pending_copy_and_unpin) until the worker takes over.
    // Returns nullptr on the same conditions upload_expert_active_async
    // would (pool not allocated, invalid params, pool full).
    ExpertLocation* claim_active_layer_slot_for_worker(int layer, int expert_id, size_t bytes,
                                                         bool speculative, int target_gpu_id);

    // Companion to claim_active_layer_slot_for_worker: called by
    // ActiveLayerUploadWorkers' own worker thread once it has genuinely
    // issued the cudaMemcpyAsync + cudaEventRecord for a claimed slot. Under
    // one brief cache_mutex critical section: flips copy_dispatched back to
    // true (see that field's own comment -- this is what makes it safe for
    // find_active_expert/cleanup_active_layer_area to start trusting
    // cudaEventQuery on this slot's event again) and, if the register call
    // succeeded, queues the deferred unregister onto the EXISTING pending_
    // host_unregisters mechanism (reused verbatim -- see its own comment;
    // already reaped opportunistically by upload_expert_async and the
    // destructor, no new drain path needed). If register_succeeded is
    // false, nothing is queued -- there is nothing to unregister.
    void finalize_worker_dispatch(ExpertLocation* slot, void* host_ptr, bool register_succeeded) {
        if (!slot) return;
        GPU_TOPO_TIMED_LOCK(lock);
        slot->copy_dispatched = true;
        if (register_succeeded) {
            pending_host_unregisters.push_back({host_ptr, slot->upload_event});
        }
    }

    // 2026-08-29: batched sibling of finalize_worker_dispatch, added to
    // reduce cache_mutex ACQUISITION COUNT from ActiveLayerUploadWorkers'
    // two background threads -- real hardware measurement (this session)
    // found "Cache mutex wait" at ~2.4-3.0s/request (5.7-6.9% of decode
    // wall), root-caused to these two threads' frequent brief acquisitions
    // contending with the decode thread's own longer critical sections
    // elsewhere in this file (evict_lru, upload_expert, ...). Decoupling
    // copy_dispatched/temp_pin_count onto a separate lock was considered
    // and REJECTED after a full audit: temp_pin_count is read/written by
    // 10+ functions across this file and is a hard safety invariant
    // (victim-selection's "never evict a temp-pinned slot" guard) --
    // splitting it risks a new TOCTOU race class in the one file with the
    // most documented lock-ordering history in this codebase. This is the
    // safer alternative: don't touch what's protected or by whom, just
    // reduce how MANY separate acquisitions the worker pool needs by
    // batching several already-completed items' bookkeeping into one.
    // Safe because it changes nothing about WHEN a copy is considered done
    // (each item's cudaStreamSynchronize already confirmed completion
    // before ActiveLayerUploadWorkers ever calls this) or the meaning of
    // copy_dispatched itself -- purely fewer lock/unlock cycles for the
    // exact same set of writes, no new invariant. register_succeeded is
    // always false from this pool (confirmed via grep of every call site
    // -- this pool never calls cudaHostRegister, see its own header
    // comment), so pending_host_unregisters is never touched by this path;
    // any FUTURE caller needing that branch should keep using the
    // single-item version above, not this one.
    void finalize_worker_dispatch_batch(const std::vector<ExpertLocation*>& slots) {
        if (slots.empty()) return;
        GPU_TOPO_TIMED_LOCK(lock);
        for (ExpertLocation* slot : slots) {
            if (slot) slot->copy_dispatched = true;
        }
    }

    // Companion to claim_active_layer_slot_for_worker's OTHER exit path --
    // the claim succeeded but ActiveLayerUploadWorkers::try_submit then
    // rejected the item (queue at capacity). Releases the claim's temp_pin_
    // count hold so the slot is immediately eligible for a fresh claim by
    // anything else (a later worker submission, or the ordinary synchronous
    // path) -- deliberately does NOT restore layer_idx/expert_id/upload_
    // ready to any prior state (this slot's true previous occupant was
    // already reclaim-accounted by record_active_layer_prewarm_reclaim
    // inside claim_active_layer_slot_for_worker and is gone either way).
    // copy_dispatched is left false, so any reader still correctly treats
    // this half-claimed slot as a miss until it's picked as a victim again
    // (by anything) and genuinely overwritten.
    void abandon_worker_claim(ExpertLocation* slot) {
        if (!slot) return;
        GPU_TOPO_TIMED_LOCK(lock);
        if (slot->temp_pin_count > 0) slot->temp_pin_count--;
    }

    // Companion to upload_expert_active_async: ONE cudaStreamSynchronize
    // for the whole batch (the actual point of batching -- N async issues
    // amortized behind a single wait instead of N separate blocking
    // calls), then marks every victim upload_ready=true and releases its
    // temp_pin_count. Safe to call with an empty vector (no-op). All
    // victims MUST share the same GPU (true by construction -- the active
    // layer area is allocated on exactly one GPU, allocate_active_layer_area).
    void sync_active_layer_batch(const std::vector<ExpertLocation*>& victims);

    // Cleanup active layer area after processing a layer's experts
    // Called from moe_forward_gguf after each layer completion.
    // on_promoted (optional, forwarded to promote_active_to_main): fired
    // once per successful end-of-layer promotion with (layer, expert_id) --
    // see promote_active_to_main's own comment for why this exists.
    void cleanup_active_layer_area(int current_layer_idx,
                                    const std::function<void(int, int)>& on_promoted = nullptr);

    // Look up (layer, expert_id) in the active layer area. On a hit, bumps
    // hit_count/last_used_clock (like find_expert's finish_expert_hit) and
    // counts n_active_layer_hits; if hit_count then meets
    // active_layer_promote_threshold, immediately promotes the expert into
    // the real VRAM LRU cache (promote_active_to_main) and frees this
    // active-layer slot (MOVE semantics, mirroring RAMSafetyManager's
    // move_on_promotion) so it's available for a new cold upload right
    // away rather than waiting for the next cleanup_active_layer_area call.
    // Returns nullptr on a miss. IMPORTANT for callers: on a promoted hit,
    // the RETURNED pointer belongs to the main expert_vram_cache pool, NOT
    // the active-layer pool, even though this function's own name/the
    // common case suggest otherwise -- on_promoted (optional, forwarded to
    // promote_active_to_main) fires exactly when this happens, so a caller
    // that needs to disambiguate which pool `loc` came from (e.g. for
    // hold_for_compute's from_active_layer parameter) MUST use it rather
    // than assume every non-null return is active-layer-resident. This was
    // a real, previously-unfixed bug (see execution_layer.cpp's call site
    // comment) -- every promoted hit was silently mislabeled, making every
    // subsequent hold_for_compute call search the wrong pool and fail.
    ExpertLocation* find_active_expert(int layer, int expert_id,
                                        const std::function<void(int, int)>& on_promoted = nullptr);

    // Like find_active_expert(), but if the matching active-layer upload is
    // already dispatched and still in flight, wait for that one CUDA event
    // before retrying. This prevents a consumer from issuing a duplicate
    // host upload for an expert that was already predicted and copied.
    // Worker-claimed slots whose upload has not been dispatched are still
    // reported as misses rather than blocking on an unbounded worker.
    ExpertLocation* find_active_expert_wait(
        int layer, int expert_id,
        const std::function<void(int, int)>& on_promoted = nullptr);

    // Side-effect-free residency peek for the active layer area -- unlike
    // find_active_expert, this NEVER bumps hit_count/last_used_clock and
    // NEVER triggers promote_active_to_main. Exists specifically for
    // upload_expert_active_async's Phase-2 batch pre-warm caller
    // (execution_layer.cpp), which must check "is this candidate already
    // active-layer-resident" for MANY candidates before Phase 3 has
    // legitimately dispatched to any of them yet -- calling the real
    // find_active_expert speculatively there would inflate hit_count for
    // experts not actually used yet this token, and (with a threshold as
    // low as 1, this codebase's own GLM default) could fire a REAL
    // promotion/D2D-copy/slot-free as a side effect of a mere existence
    // check, corrupting the same active_layer_slots state the pre-warm
    // batch is simultaneously choosing victims from. Mirrors
    // RAMSafetyManager::is_ram_resident's own "peek, don't touch" pattern.
    bool active_layer_has(int layer, int expert_id) const {
        GPU_TOPO_TIMED_LOCK(lock);
        for (const auto& slot : active_layer_slots) {
            if (slot.layer_idx == layer && slot.expert_id == expert_id) return true;
        }
        return false;
    }

    // Promote an expert already resident in the active layer area (VRAM)
    // into the real VRAM LRU cache via a device-to-device copy -- same
    // victim-selection/salvage path as upload_expert(), just sourcing from
    // device memory instead of a host buffer. Used by find_active_expert
    // (on-hit promotion) and cleanup_active_layer_area (end-of-layer
    // promotion for anything not caught by an inline hit first). Returns
    // the new main-cache slot, or nullptr if the main cache is unavailable/
    // the copy fails.
    //
    // on_promoted (optional): fired with (layer, expert_id) immediately
    // after a successful promotion, still under cache_mutex. FIX (real,
    // measured "[TIER-OVERLAP]" leak -- see CLAUDE.md's "Cold-Miss GPU
    // Dispatch Never Freed the RAM Slot" entry, the "still-open gap" it
    // left behind): this D2D copy is SYNCHRONOUS (cudaMemcpy, not Async),
    // so by the time this function returns, the destination is
    // unconditionally durable main-cache-resident -- the exact same
    // "synchronous therefore safe to mark immediately" reasoning already
    // used for the cold-batch/decode-miss-fallback upload_expert() sites in
    // execution_layer.cpp, and NOT subject to the temporary-active-layer-
    // pool caveat that made marking also_in_vram at ACTIVE-LAYER UPLOAD
    // time wrong (that pool is genuinely temporary; the MAIN cache this
    // promotes INTO is not). GpuTopology has no RAM-tier type to depend on
    // directly, so this callback is the caller's hook to apply that
    // marking (via RAMSafetyManager::mark_ram_also_in_vram) without
    // creating a reverse header dependency.
    // src_gpu_id (2026-08-28, default -1): which GPU device_src actually
    // lives on. Required now that multi-GPU active-layer pools mean the
    // promotion SOURCE and the main-cache victim evict_lru picks can be on
    // DIFFERENT GPUs -- previously always implicitly the same GPU (the
    // active-layer pool was single-GPU), so a plain same-device cudaMemcpy
    // was always correct; it silently is NOT once a second GPU is in play.
    // -1 means "assume same GPU as the destination victim" (the historical
    // behavior, still correct for every legacy single-GPU-pool call) --
    // every NEW call site should pass the source slot's own real gpu_id.
    ExpertLocation* promote_active_to_main(int layer, int expert_id,
                                            const uint8_t* device_src, size_t bytes,
                                            const std::function<void(int, int)>& on_promoted = nullptr,
                                            int src_gpu_id = -1,
                                            bool same_gpu_victim = false);

    // Transfer a small result vector between GPUs (e.g., expert output → core)
    void cross_device_transfer(float* dst, int dst_device,
                               const float* src, int src_device,
                               size_t bytes) {
        // Try peer-to-peer first
        int can_peer = 0;
        cudaDeviceCanAccessPeer(&can_peer, dst_device, src_device);
        if (can_peer) {
            cudaMemcpyPeer(dst, dst_device, src, src_device, bytes);
        } else {
            // Bounce through pinned host buffer
            size_t xfer_bytes = std::min(bytes, h_xfer_cap);
            cudaSetDevice(src_device);
            cudaMemcpy(h_xfer_pinned, src, xfer_bytes, cudaMemcpyDeviceToHost);
            cudaSetDevice(dst_device);
            cudaMemcpy(dst, h_xfer_pinned, xfer_bytes, cudaMemcpyHostToDevice);
        }
    }

private:
    // Consume one active-layer start/end event pair after its completion event
    // is known to have fired.  CUDA event time is the copy-engine interval,
    // not the CPU issue/wait time or a session-wide bytes/time ratio.
    void record_active_layer_upload_event_timing(ExpertLocation& slot,
                                                 bool sync_path);

    void assign_roles() {
        int n = devices.size();
        if (n == 0) return;

        // CUDA must enumerate every physical adapter, but a config may
        // intentionally select only a subset. Keep the full inventory for
        // diagnostics while restricting roles and allocations to the
        // configured device IDs. Previously an omitted adapter silently
        // became an ExpertVault whenever the machine had more than one GPU.
        std::vector<int> selected;
        if (configured_device_ids.empty()) {
            selected.reserve(n);
            for (int i = 0; i < n; ++i) selected.push_back(i);
        } else {
            for (int i = 0; i < n; ++i) {
                if (std::find(configured_device_ids.begin(), configured_device_ids.end(),
                              devices[i].id) != configured_device_ids.end())
                    selected.push_back(i);
            }
        }
        if (selected.empty()) {
            LOG_STREAM_ERROR() << "[GPU] Configured device allow-list does not match any "
                                  "CUDA device; no GPU roles assigned\n";
            return;
        }
        for (auto& dev : devices) dev.role = GpuRole::Unused;
        core_gpu = -1;
        vault_gpus.clear();

        if (selected.size() == 1) {
            // Profile A: single GPU does everything
            profile = HardwareProfile::SingleGPU;
            devices[selected[0]].role = GpuRole::CoreEngine;
            core_gpu = selected[0];
            vault_gpus.push_back(selected[0]); // same device serves as vault too
        } else {
            // Multi-GPU: honor an explicit config choice if one was given
            // (forced_core_device_id, set by main.cpp from
            // config.gpus[dense_gpu_idx].device_id) -- otherwise fall back
            // to the auto-detect heuristic this function always used
            // before: assign Core to newest arch (or smallest VRAM if same
            // arch), Vault to the rest.
            int best_core = -1;
            if (forced_core_device_id >= 0) {
                for (int i : selected) {
                    if (devices[i].id == forced_core_device_id) { best_core = i; break; }
                }
                if (best_core < 0) {
                    LOG_STREAM_WARN() << "[GPU] forced_core_device_id=" << forced_core_device_id
                              << " does not match any enumerated device -- falling back to "
                                 "SM-architecture/VRAM auto-detect for CoreEngine assignment\n";
                }
            }
            if (best_core < 0) {
                best_core = selected[0];
                for (size_t si = 1; si < selected.size(); ++si) {
                    int i = selected[si];
                    bool newer = devices[i].sm_major > devices[best_core].sm_major;
                    bool same_gen_less_vram = (devices[i].sm_major == devices[best_core].sm_major &&
                                               devices[i].total_vram < devices[best_core].total_vram);
                    if (newer || same_gen_less_vram) best_core = i;
                }
            }

            devices[best_core].role = GpuRole::CoreEngine;
            core_gpu = best_core;

            for (int i : selected) {
                if (i != best_core) {
                    devices[i].role = GpuRole::ExpertVault;
                    vault_gpus.push_back(i);
                }
            }

            profile = (vault_gpus.size() >= 2) ? HardwareProfile::TripleGPU : HardwareProfile::DualGPU;
        }
    }

    void print_topology() {
        const char* profile_names[] = {"Single GPU (A)", "Dual GPU (B)", "Triple+ GPU (C)"};
        LOG_STREAM_INFO() << "[GPU] Hardware Profile: " << profile_names[(int)profile] << "\n";
        for (const auto& dev : devices) {
            const char* role_str = (dev.role == GpuRole::CoreEngine) ? "CoreEngine" :
                                   (dev.role == GpuRole::ExpertVault) ? "ExpertVault" : "Unused";
            LOG_STREAM_INFO() << "[GPU]   " << dev.name << " (GPU " << dev.id << ") — "
                      << dev.total_vram / 1e9 << " GB VRAM — sm_"
                      << dev.sm_major << dev.sm_minor << " — " << role_str << "\n";
        }
    }

public:
    // Real per-GPU VRAM usage via cudaMemGetInfo -- NOT the same thing as
    // "how full is the expert cache logically" (expert_cache_pool is one
    // fixed cudaMalloc reserved entirely at load time, see initialize()
    // above; nvidia-smi/this query reports that whole reservation as "used"
    // from the moment it's allocated, regardless of how many of its slots
    // actually hold real expert data yet -- see log_expert_heatmap() below
    // for the logical fill state instead). Meant to be called periodically
    // (e.g. once per generated token) to track real usage over the
    // lifetime of a run, distinct from the one-time startup topology dump.
    void log_vram_usage(const char* label) const {
        std::ostringstream os;
        os << "[VRAM] " << label << ":";
        for (const auto& dev : devices) {
            if (dev.role == GpuRole::Unused) continue;
            cudaSetDevice(dev.id);
            size_t free_b = 0, total_b = 0;
            if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) continue;
            double used_gb = (double)(total_b - free_b) / 1e9;
            double total_gb = (double)total_b / 1e9;
            os << "  GPU" << dev.id << "(" << dev.name << ")=" << std::fixed << std::setprecision(2)
               << used_gb << "/" << total_gb << "GB(" << (100.0 * (total_b - free_b) / total_b) << "%)";
        }
        os << "\n";
        LOG_DEBUG("%s", os.str().c_str());
    }

    // Text heatmap of the VRAM expert cache's LOGICAL fill state, per layer
    // -- how many of this layer's currently-resident slots there are, and
    // their total hit_count (recency+frequency signal), scaled to a single
    // ASCII intensity character per layer so the whole model depth fits on
    // one line. Answers "which layers are actually hot in VRAM right now",
    // as opposed to log_vram_usage()'s "how much raw memory is reserved".
    void log_expert_heatmap(int n_layers) const {
        if (n_layers <= 0) return;
        std::vector<int> resident_count(n_layers, 0);
        std::vector<uint64_t> hit_sum(n_layers, 0);
        {
            GPU_TOPO_TIMED_LOCK(lock);
            for (const auto& loc : expert_vram_cache) {
                if (loc.layer_idx < 0 || loc.layer_idx >= n_layers) continue;
                resident_count[loc.layer_idx]++;
                hit_sum[loc.layer_idx] += loc.hit_count;
            }
        }
        uint64_t max_hits = 1;
        for (auto h : hit_sum) max_hits = std::max(max_hits, h);
        static const char ramp[] = " .:-=+*#%@";
        constexpr int ramp_len = sizeof(ramp) - 2; // exclude trailing NUL, keep last real char reachable
        std::ostringstream os;
        os << "[VRAM-HEATMAP] per-layer resident-slot hit density (0-9, ' '=empty, '@'=hottest):\n[VRAM-HEATMAP] ";
        for (int l = 0; l < n_layers; ++l) {
            int idx = resident_count[l] == 0 ? 0
                : 1 + (int)((double)hit_sum[l] / (double)max_hits * (ramp_len - 1));
            os << ramp[idx];
        }
        os << "\n";
        LOG_DEBUG("%s", os.str().c_str());
    }

    // Shutdown-time summary of VRAM-tier cache churn: which experts were
    // swapped in/out the most, and -- the actionable part -- which of those
    // are NOT currently pinned, i.e. paying repeated H2D/eviction cost with
    // no protection. pinned tier, which by design never pays this cost more
    // than once per expert (upload at startup, never evicted). A high-churn,
    // unpinned entry here is a concrete candidate for a lower pin_threshold
    // or a larger pin_topk_per_layer preload. Call once at shutdown, after
    // generation has finished (no locking -- nothing else is touching
    // expert_vram_cache/vram_swap_stats by then).
    void report_cache_churn(int top_n = 15) const {
        if (startup_preload_marked > 0) {
            LOG_STREAM_INFO()
                << "[CACHE-CHURN] Startup VRAM preload: " << startup_preload_marked
                << " identities marked, " << startup_preload_used << " used at least once, "
                << (startup_preload_marked - startup_preload_used)
                << " never used; evicted before first use="
                << startup_preload_evicted_before_use << " (during prefill="
                << startup_preload_evicted_before_use_prefill << "), evicted after use="
                << startup_preload_evicted_after_use << " (during prefill="
                << startup_preload_evicted_after_use_prefill << ").\n";
        }
        if (vram_swap_stats.empty()) {
            LOG_STREAM_INFO() << "[CACHE-CHURN] VRAM tier: no swap activity recorded.\n";
            return;
        }
        uint64_t total_in = 0, total_out = 0;
        for (const auto& kv : vram_swap_stats) { total_in += kv.second.swap_in; total_out += kv.second.swap_out; }
        LOG_STREAM_INFO() << "[CACHE-CHURN] VRAM tier: " << vram_swap_stats.size()
                           << " distinct experts touched, " << total_in << " swap-ins / "
                           << total_out << " swap-outs this session.\n";

        std::vector<std::pair<uint64_t, SwapStats>> sorted(vram_swap_stats.begin(), vram_swap_stats.end());
        std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) {
            return (a.second.swap_in + a.second.swap_out) > (b.second.swap_in + b.second.swap_out);
        });

        LOG_STREAM_INFO() << "[CACHE-CHURN] Top " << top_n
                           << " most-churned VRAM experts NOT currently pinned "
                              "(paying repeated swap cost with no protection):\n";
        int shown = 0;
        for (const auto& kv : sorted) {
            if (shown >= top_n) break;
            int layer = (int)(kv.first >> 32), expert_id = (int)(kv.first & 0xFFFFFFFFu);
            bool resident_now = false, pinned_now = false;
            for (const auto& loc : expert_vram_cache) {
                if (loc.layer_idx == layer && loc.expert_id == expert_id) {
                    resident_now = true; pinned_now = loc.pinned; break;
                }
            }
            if (pinned_now) continue;  // already protected -- not an actionable entry
            LOG_STREAM_INFO() << "  layer=" << layer << " expert=" << expert_id
                               << ": swap_in=" << kv.second.swap_in << " swap_out=" << kv.second.swap_out
                               << (resident_now ? " (resident now, unpinned)" : " (not resident now)") << "\n";
            shown++;
        }
        if (shown == 0) {
            LOG_STREAM_INFO() << "  (every high-churn expert is already pinned)\n";
        }
    }

    // How many VRAM cache slots are ACTUALLY holding an expert right now,
    // vs. total capacity -- distinct from expert_vram_cache_cap (capacity)
    // and from report_cache_churn's cumulative swap-activity counters
    // (which say nothing about current occupancy). Added specifically to
    // answer "how many MoE experts do we have loaded" with a real number
    void report_slot_summary() const {
        size_t filled = 0, pinned_n = 0, mid_upload = 0;
        for (const auto& loc : expert_vram_cache) {
            if (loc.layer_idx == -1) continue;
            ++filled;
            if (loc.pinned) ++pinned_n;
            if (!loc.upload_ready) ++mid_upload;
        }
        // Compact mode gives each layer its own variable-sized region, so
        // filled * expert_slab_size would report zero there (the scalar is
        // intentionally zero) instead of the real resident bytes. Sum each
        // slot's actual geometry; this is also correct for legacy slots.
        size_t bytes_used = 0;
        for (const auto& loc : expert_vram_cache) {
            if (loc.layer_idx != -1) bytes_used += loc.size;
        }
        LOG_STREAM_INFO() << "[VRAM-SLOTS] " << filled << " / " << expert_vram_cache_cap
                           << " slots filled (" << bytes_used / 1e9 << " GB of experts resident), "
                           << pinned_n << " pinned, " << mid_upload << " mid-upload\n";
    }
};
