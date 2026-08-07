#pragma once
#include "engine.hpp"
#include "perf_timer.hpp"
#include "gpu_topology.hpp"
#include "logger.hpp"
#include "kquant_dequant.hpp"  // gpu_type_supported
#include "dense_dispatch.hpp"  // is_cpu_affinity_layer (cpu_affinity_layers config)
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>  // std::find (get_experts_batch's claimed_this_batch guard)
#include <functional>  // std::greater (apply_layer_adaptive_capacity's per-layer sort)
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <mutex>
#include <fcntl.h>
#include <unistd.h>
#include <cuda_runtime.h>  // demote_from_vram's D2H salvage copy; already transitively
                           // pulled in via gpu_topology.hpp, included directly here too
                           // so the dependency isn't implicit
#ifdef __APPLE__
#include <mach/mach.h>
#include <sys/sysctl.h>
#endif

// Any wait to acquire ram_mutex exceeding this is logged individually (tagged
// [SLOW-RAM-MUTEX-WAIT]) -- mirrors gpu_topology.hpp's SLOW_CACHE_MUTEX_WAIT_MS
// for GpuTopology::cache_mutex. ram_mutex is shared between the decode thread's
// own miss-handling (get_expert/get_experts_batch/claim_or_wait/wait_for_ram_
// slot) and a LockFreePrefetcher worker thread's own speculative claim -- see
// claim_or_wait's own comment for the full unified-claim design this protects.
constexpr double SLOW_RAM_MUTEX_WAIT_MS = 50.0;

// How long wait_for_ram_slot may poll for an in-flight claim (owned by
// another caller, almost always a prefetch worker) before logging
// [SLOW-RAM-WAIT] -- mirrors async_prefetcher.hpp's old SLOW_PICKUP_MS
// threshold. Logged once, not on every poll iteration; waiting continues past
// this point rather than falling back to a duplicate read (see wait_for_ram_
// slot's own comment for why that's the correct, deadlock/corruption-free
// choice given ram_expert_index's one-slot-per-key invariant).
constexpr double SLOW_RAM_WAIT_MS = 250.0;

// Acquire ram_mutex into `varname` while timing how long the acquisition
// itself blocked -- mirrors gpu_topology.hpp's GPU_TOPO_TIMED_LOCK exactly.
// Deliberately a PLAIN std::mutex, not recursive (unlike cache_mutex): every
// function that takes ram_mutex is a self-contained, non-composing critical
// section by design (see claim_or_wait's own comment) -- a plain mutex hangs
// loudly on the first same-thread double-lock a future change might
// accidentally introduce, where a recursive mutex would silently mask it.
#define RAM_TIMED_LOCK(varname) \
    auto _lock_wait_t0_##varname = std::chrono::high_resolution_clock::now(); \
    std::lock_guard<std::mutex> varname(ram_mutex); \
    do { \
        double _wait_ms = std::chrono::duration<double, std::milli>( \
            std::chrono::high_resolution_clock::now() - _lock_wait_t0_##varname).count(); \
        PerfCounters::atomic_add(perf().t_ram_mutex_wait, _wait_ms / 1000.0); \
        if (_wait_ms > SLOW_RAM_MUTEX_WAIT_MS) { \
            LOG_WARN("[SLOW-RAM-MUTEX-WAIT] %s blocked %.2fms acquiring ram_mutex\n", \
                     __func__, _wait_ms); \
        } \
    } while (0)

// Outcome of RAMSafetyManager::claim_or_wait -- see that function's own
// comment for the full design.
enum class RamClaimStatus { Hit, Loading, Claimed };
struct RamClaimResult {
    RamClaimStatus status;
    MoEExpertBlock* slot = nullptr;  // valid for Hit/Claimed; nullptr for Loading
    size_t slot_idx = 0;             // valid for Hit/Claimed only
};

class RAMSafetyManager {
public:
    int slots_per_layer = 0;
    // Per-layer override for the uniform slots_per_layer cap above --
    // populated by apply_layer_adaptive_capacity (ram_cache_layer_adaptive
    // config flag) when real usage history exists; empty otherwise. Never
    // read directly -- always go through cap_for_layer(layer).
    std::vector<int> slots_per_layer_by_layer;
    uint64_t global_clock = 0;
    GpuTopology* vram_cache = nullptr;  // GPU expert VRAM cache (optional, set by CudaManager)

    // Guards all mutation of the RAM tier's per-slot claim state (expert_id,
    // ram_loading, last_used_clock, hit_count, also_in_vram, ram_temp_pin_count,
    // pending_copy_event) and ram_expert_index -- shared between the decode
    // thread's own miss-handling and a LockFreePrefetcher worker thread's own
    // speculative claim (async_prefetcher.hpp/async_prefetcher.cpp, via a
    // RAMSafetyManager* pointer set post-construction in main.cpp -- NOT a
    // header include, to avoid a circular dependency: LockFreePrefetcher now
    // calls INTO RAMSafetyManager::claim_or_wait, the reverse of the old
    // consume_staged design where RAMSafetyManager called into the
    // prefetcher's own staging pool). See claim_or_wait's own comment for the
    // full unified claim/publish design this mutex protects, and MoEExpertBlock::
    // ram_loading's own comment for the field-ownership rule it enforces.
    mutable std::mutex ram_mutex;

    // io_uring ring OWNED BY THE DECODE THREAD ITSELF -- distinct from the
    // prefetcher's own per-worker-thread rings (io_uring_backend.hpp: a
    // Ring is not shareable across threads). Used by get_experts_batch's
    // Phase B to batch an ENTIRE layer's pending cache-miss set into ONE
    // io_uring submission (colibri's own real miss-servicing design,
    // glm.c's pipe_dispatch/URING_REQ_MAX) instead of N OS threads each
    // blocking on their own pread -- this is the SYNCHRONOUS,
    // decode-critical path that handles the large majority of every real
    // run's misses (the prefetcher only ever serves a small
    // fraction of them), so it's where the actual disk-throughput ceiling
    // lives, unlike the earlier (Prefetcher-only, deliberately lower-risk)
    // io_uring scope. init_decode_uring() sets this up; falls back to the
    // existing OMP-parallel load_expert_slab loop, unconditionally
    // correct, whenever the ring is unavailable OR a batch fails for any
    // reason (short read, unsupported layer format, etc.).
    io_uring_backend::Ring decode_uring_ring;
    bool use_decode_uring = false;

    void init_decode_uring(bool enable, int io_uring_workers = 8) {
        if (!enable) return;
        use_decode_uring = decode_uring_ring.init();
        if (use_decode_uring) {
            // io_uring_workers (config, EngineConfig -- default 8, matching
            // colibri's own field-tested PIPE_WORKERS default) bounds the
            // io-wq kernel worker pool this ring dispatches to. Only
            // meaningful now that IOSQE_ASYNC (io_uring_backend.hpp)
            // actually forces reads onto io-wq instead of letting them
            // complete inline -- this is what lets a submission's reads
            // genuinely overlap on the device. Deliberately sized to real
            // NVMe queue-depth saturation (~4-16 concurrent requests for
            // reads this size), not to Phase B's own max batch shape
            // (topk*3, commonly ~24) -- more workers than the device can
            // actually service concurrently doesn't help throughput, it
            // just adds idle-but-real OS threads competing with everything
            // else this process runs (omp_max_threads, expert_fetch_
            // threads, prefetch_worker_threads). Best-effort: a no-op on
            // kernels that don't support the opcode, same fallback-never-
            // fails discipline as every other io_uring call in this
            // codebase.
            decode_uring_ring.register_iowq_workers((unsigned)std::max(1, io_uring_workers));
            LOG_STREAM_INFO() << "[IO-URING] Enabled on the decode thread's own ring -- "
                                  "batching Phase B's per-layer miss set through io_uring.\n";
        } else {
            LOG_STREAM_WARN() << "[IO-URING] io_uring=true requested but the decode-thread "
                                  "Ring::init() failed -- Phase B falls back to the OMP-parallel "
                                  "pread loop.\n";
        }
    }

    // Set by layers_forward_gguf at the top of every call (S > 1 = prefill,
    // S == 1 = decode) purely so the miss path below can split its prefetch
    // hit/miss counters by phase -- diagnostic only, changes no behavior.
    // Added to test a specific hypothesis: the pilot prefetcher predicts
    // from only the LAST token's hidden state (execution_layer.cpp), which
    // structurally can't cover prefill's full per-layer expert fan-out
    // across ALL prompt tokens, so a near-zero aggregate prefetch-served rate
    // could be "prefill can't be helped by this design at all" rather than
    // "decode isn't benefiting either" -- this split is how the next real
    // run tells those apart instead of guessing from the aggregate number.
    bool in_prefill = false;

    // Pin a VRAM-resident expert's slot once its re-hit count crosses this
    // threshold, so a burst of one-off cold experts can't evict it (Phase A
    // "stop evicting hot experts within a conversation"). 0 disables pinning.
    uint32_t pin_threshold = 8;

    // Explicit host RAM (DDR) budget for the MoE expert cache, in GB.
    // Set from EngineConfig::host_ram_gb (TOML `host_ram_gb` or --ram CLI flag).
    // 0 = auto-detect (88% of currently-available system memory).
    double configured_ram_gb = 0.0;

    // When true, logs every host-cache HIT (very high volume — one line per
    // routed expert per token). MISS/LOAD and VRAM PROMOTE events are always
    // logged since they're comparatively rare (disk I/O, one-time promotion).
    bool verbose_cache_log = false;

    // How many host-cache re-hits before triggering an async H2D promotion
    // into the VRAM expert cache. Default 1 (promote on first re-hit).
    uint32_t promote_hit_threshold = 1;

    // Three-tier caching: MOVE instead of COPY on RAM→VRAM promotion
    // When true, immediately frees RAM slot after VRAM promotion starts,
    // eliminating the 7.7GB tier overlap waste.
    bool move_on_promotion = true;

    // Pad Phase B's real cache-miss submission with speculative reads of
    // historically-hot, not-currently-resident experts for the SAME layer
    // -- see EngineConfig::disk_batch_padding's own comment for the full
    // rationale/mechanism/known-limitation. Mirrored here (set from
    // main.cpp) rather than reaching into EngineConfig directly, matching
    // this class's existing pattern for every other config-derived field
    // above (pin_threshold, promote_hit_threshold, move_on_promotion, ...).
    bool disk_batch_padding = false;
    int disk_batch_padding_target = 8;

    // Per-layer, pre-sorted-descending-by-historical-count expert ID
    // ranking, used as disk_batch_padding's candidate source. Populated
    // ONCE at startup by main.cpp from the SAME .moestream_usage-derived
    // `by_layer` ranking [PIN] preload itself already computes and trusts
    // (including its prefill_pin_file per-layer override, if configured) --
    // deliberately NOT StreamingEngine::route_freq, which is zero-
    // initialized every process start and only reflects THIS run's own
    // routing so far. That live signal turned out self-defeating on a real
    // run: an expert only earns a nonzero live count once routed to a few
    // times, by which point it's almost always already RAM/VRAM-resident
    // via the normal miss->promotion path -- exactly what padding's own
    // residency filters exclude -- so the "hot but not yet cached"
    // candidate pool it offered was nearly empty by construction, and
    // padding never issued a single speculative read for a whole real run
    // (see CLAUDE.md). This persisted, cross-run ranking has real
    // historical depth from token 1, independent of how much THIS run has
    // routed so far -- the same property that makes it right for [PIN]
    // preload's identical "find genuinely hot experts" purpose. Empty (no
    // entry for a given layer, or the whole map empty) is a normal,
    // graceful degradation -- padding simply finds no candidates for that
    // layer and does nothing, same as running with pin_file unset.
    std::unordered_map<int, std::vector<int>> route_rank_by_layer;

    // OMP thread count for get_experts_batch's parallel disk-read phase.
    // Tuned empirically against the real model file
    // on the target machine: buffered reads scaled ~8.5x from 1->4 threads,
    // then flattened (8 threads: no further gain) -- see engine_config.hpp's
    // expert_fetch_threads field for the full measurement.
    int expert_fetch_threads = 4;

    // Frequency-aware eviction: blend recency with hit_count so a
    // power-law "always hot" expert survives bursts of cold accesses to
    // other experts. Decayed periodically to avoid permanent stickiness.
    static constexpr uint64_t FREQ_BONUS_TICKS = 64;
    static constexpr uint32_t HIT_DECAY_INTERVAL = 4096;
    uint64_t hits_since_decay = 0;

    // Cache-churn diagnostics: per-(layer,expert) swap in (disk load) / out
    // (eviction) counts for the host RAM tier -- the counterpart to
    // GpuTopology::vram_swap_stats. Unlike the VRAM tier, this tier has no
    // pinning mechanism at all (RAMSafetyManager never protects a slot from
    // eviction), so the report below is churn-only, not an "unpinned"
    // callout. Mutated only from get_expert/get_experts_batch, both called
    // exclusively from the single decode thread (src/execution_layer.cpp)
    // -- never the prefetcher worker thread -- so no lock is needed.
    struct SwapStats { uint64_t swap_in = 0; uint64_t swap_out = 0; };
    std::unordered_map<uint64_t, SwapStats> ram_swap_stats;
    static uint64_t swap_key(int layer, int expert_id) {
        return ((uint64_t)(uint32_t)layer << 32) | (uint32_t)expert_id;
    }

    // O(1) identity -> slot-index lookup for the RAM tier, mirroring
    // GpuTopology::expert_index exactly (same key encoding via swap_key).
    // Added after a real run showed get_experts_batch alone costing 59.74s
    // of a 104.40s expert= total (t_ram_get_experts_batch), traced to
    // Step 1/Phase A's per-lookup UNINDEXED linear scan over layer_slots
    // (up to `cap` MoEExpertBlock entries -- a much larger, more cache-
    // unfriendly struct than GpuTopology::ExpertLocation) run on EVERY
    // topk lookup, hit or miss, with misses paying for a SECOND full scan
    // during victim selection on top.
    //
    // TAKE 1 (measured insufficient -- a real lesson from this exact file's
    // history repeating): the first version of find_ram_slot fell back to a
    // full linear scan whenever the index had no entry at all, mirroring
    // GpuTopology::find_expert's own fallback-on-miss behavior. A real
    // re-run showed ZERO measurable improvement (get_batch=60.65s vs
    // 59.74s before) -- because ~56% of lookups are genuine misses, and a
    // miss NEEDS the "confirm not present" check on every single call; only
    // the ~44% hit case became O(1), while both the miss confirmation AND
    // the (always-linear, unchanged) victim-selection scan kept paying full
    // O(cap) either way. Root cause of why the fallback existed at all:
    // main.cpp's Tier-3 startup RAM preload writes directly into
    // engine.layers_experts via engine.load_expert_slab, bypassing
    // RAMSafetyManager entirely -- so trusting a miss as genuine absence
    // would have been WRONG for any of the ~1575 preloaded experts on
    // their first real lookup (index has no entry for them despite being
    // genuinely resident). Fixed by making main.cpp's preload loop insert
    // into ram_expert_index too (the actual missing mutation site -- see
    // that call site's own comment), which makes the index a COMPLETE,
    // authoritative record of RAM-tier residency, letting a genuine index
    // miss be trusted as genuine absence with no scan at all. This is the
    // TAKE 2 fix below.
    //
    // Mutated at every site that changes a RAM slot's expert_id identity:
    // get_expert (Step 1's MOVE-on-promotion clear, Step 3's evict+load),
    // get_experts_batch (Phase A's MOVE-on-promotion clear, miss-claim+
    // load), demote_from_vram (VRAM-eviction salvage write), and main.cpp's
    // Tier-3 startup preload (see above). Verified via full-codebase grep
    // that these are the ONLY sites that ever write engine.layers_experts[
    // layer][*].expert_id -- graph_initializer.cpp's startup zero-init sets
    // it to -1 (empty, never needs an index entry) and execution_layer.cpp's
    // pilot-prefetch "already resident" check is read-only.
    std::unordered_map<uint64_t, size_t> ram_expert_index;

    // Shared lookup used by every RAM-tier hit-check site (get_expert Step
    // 1, get_experts_batch Phase A, demote_from_vram's already-resident
    // pre-check). TAKE 2: an index MISS is now trusted as genuine absence
    // (O(1), no scan) -- safe only because ram_expert_index is a COMPLETE
    // record of residency (see its own comment for the full mutation-site
    // list, including the main.cpp preload fix this required). An index
    // HIT is still re-validated against the slot's real contents before
    // being trusted, and falls back to a defensive linear scan (self-
    // healing the index) only in that case -- should be unreachable by
    // construction, but a maintenance-site bug is exactly the failure
    // class this project has been burned by before (see CLAUDE.md), and
    // that specific case can't safely trust a lookup that already proved
    // itself wrong once. Returns the slot index within layer_slots
    // (bounded by cap) or -1 if genuinely not resident.
    ssize_t find_ram_slot(std::vector<MoEExpertBlock>& layer_slots, size_t cap, int layer, int expert_id) {
        uint64_t key = swap_key(layer, expert_id);
        auto it = ram_expert_index.find(key);
        if (it == ram_expert_index.end()) {
            return -1;  // trusted absence -- see this function's own comment
        }
        size_t idx = it->second;
        if (idx < cap && layer_slots[idx].expert_id == expert_id) {
            return (ssize_t)idx;
        }
        // Stale entry -- should be unreachable by construction given every
        // mutation site is covered, but treat it as a real bug signal, not
        // a silent no-op: fall back to a linear scan (this ONE case still
        // needs it, since idx already proved untrustworthy) and self-heal.
        ram_expert_index.erase(it);
        for (size_t si = 0; si < cap; ++si) {
            if (layer_slots[si].expert_id == expert_id) {
                ram_expert_index[key] = si; // opportunistic self-heal
                return (ssize_t)si;
            }
        }
        return -1;
    }

    // Side-effect-free RAM-tier residency peek -- unlike find_ram_slot
    // (used by the actual hit-servicing paths, which bump hit_count/
    // recency as part of "using" the hit), this exists purely so a CALLER
    // can decide which read strategy to use BEFORE committing to one: the
    // cuFile direct-disk-to-VRAM path (GpuTopology::upload_expert_cufile,
    // via execution_layer.cpp's Phase 3 loading loop) must only ever be
    // attempted for a genuine double miss (not VRAM-resident AND not
    // RAM-resident) -- re-reading from disk via cuFile when a perfectly
    // good copy already sits in host RAM would waste real NVMe bandwidth
    // for no benefit. Trusts ram_expert_index exactly as find_ram_slot's
    // own index-hit fast path does (an index miss is genuine absence, an
    // index hit is real residency -- see that function's own comment for
    // why this index is a complete, self-healing record) but deliberately
    // skips the stale-entry re-validation/linear-scan fallback: a false
    // positive here (index says resident but the slot turns out stale) just
    // means the caller takes the slower RAM-mediated path instead of
    // cuFile for that one call -- get_experts_batch's own find_ram_slot
    // will re-validate for real and correctly re-fetch if needed, so this
    // peek being occasionally too conservative is safe, never wrong.
    //
    // Hardened to take ram_mutex once a second writer thread (a prefetcher
    // worker, via claim_or_wait) genuinely exists: reading ram_expert_index
    // (an unordered_map) unlocked while another thread concurrently inserts/
    // erases from it is real UB (rehash/iterator invalidation), not merely
    // "occasionally stale" the way this function's own comment above assumed
    // under the old single-decode-thread-only invariant. The lock cost here
    // is negligible (one .count() call).
    bool is_ram_resident(int layer, int expert_id) const {
        RAM_TIMED_LOCK(lock);
        return ram_expert_index.count(swap_key(layer, expert_id)) > 0;
    }

    // Undo a Phase B speculative-padding claim (disk_batch_padding) whose
    // geometry lookup or actual read failed -- no existing equivalent:
    // every OTHER RAM-tier failure path in this file falls back to "redo
    // the whole batch via the OMP loop," never a per-item revert, because
    // a real miss's failure always retries via that fallback. A padding
    // candidate has no such retry (it's optional, speculative), so a
    // failed one must be returned to genuinely-free (expert_id=-1) rather
    // than left half-claimed with garbage/unpopulated slab bytes -- a
    // half-claimed slot would make a LATER real hit trust is_ram_resident/
    // find_ram_slot's claim of residency and read uninitialized data into
    // a matmul, exactly the "fluent but wrong" failure class this
    // project's own history has been burned by before.
    void revert_padding_claim(MoEExpertBlock& slot, int layer, int expert_id) {
        RAM_TIMED_LOCK(lock);
        slot.expert_id = -1;
        ram_expert_index.erase(swap_key(layer, expert_id));
        slot.last_used_clock = 0;
        slot.hit_count = 0;
        slot.from_padding = false;
        slot.ram_loading = false;
    }

    // ACTIVE reclaim for a caller that just confirmed (via some OTHER path,
    // e.g. GpuTopology::promote_active_to_main's on_promoted callback) that
    // a durable VRAM copy now exists for this exact (layer, expert_id) --
    // closes the "still-open gap" documented in CLAUDE.md's "Cold-Miss GPU
    // Dispatch Never Freed the RAM Slot" entry. FIRST fix here (superseded,
    // history kept for context) only marked also_in_vram=true (an eviction-
    // PRIORITY hint, not an active free) -- a real run confirmed this fix
    // does correctly fire (hundreds of "(promoted to main cache)" DEBUG
    // lines) but [TIER-OVERLAP] barely moved (608 vs a 614-720 baseline on
    // a matching config), because "evict first" only matters the NEXT time
    // that layer's RAM tier actually needs to evict something -- with a
    // generous host_ram_gb and layer-adaptive capacity, a short run may
    // never put real eviction pressure on the specific layers involved, so
    // the marked-redundant slot just sits there, correctly deprioritized
    // but never picked as a victim. This version reclaims the slot RIGHT
    // NOW instead of waiting for eviction pressure that may never come.
    //
    // Mirrors apply_promotion_bookkeeping's MOVE-vs-COPY split (same
    // move_on_promotion config, same counters) but is NOT a call to that
    // function: apply_promotion_bookkeeping exists for an ASYNC upload
    // (upload_expert_async) whose completion isn't yet certain, so its MOVE
    // branch has to carry `promoted->upload_event` forward as
    // pending_copy_event so a later reader can defensively wait/poll. Our
    // caller's copy (GpuTopology::promote_active_to_main's D2D memcpy) is
    // SYNCHRONOUS and has already returned successfully by the time
    // on_promoted fires -- there is no pending-copy race to protect
    // against, and `promoted`'s own upload_event (if any) would be a STALE
    // value left over from whatever THIS SPECIFIC SLOT was previously used
    // for, unrelated to the transfer that just happened -- borrowing it
    // here would be wrong, not just unnecessary. pending_copy_event is left
    // untouched. COPY branch (move_on_promotion=false) sets also_in_vram
    // immediately, unlike apply_promotion_bookkeeping's own COPY branch --
    // that one can't (its caller has no confirmation the copy landed); ours
    // just did.
    //
    // Two overloads: the direct one for a caller that already holds the
    // MoEExpertBlock& (currently only the lookup overload below actually
    // calls this one -- see that overload's own comment; kept as a direct
    // entry point in case a future caller has a block in hand already,
    // matching apply_promotion_bookkeeping's own shape); the lookup one for
    // a caller that only has (layer, expert_id) (moe_forward_gguf's
    // cleanup_active_layer_area call site, a sweep with no single block in
    // scope). No-op if the expert isn't currently RAM-resident (may have
    // already been evicted/reused by the time this fires -- benign).
    //
    // ram_temp_pin_count guard, added after a real crash: this used to also
    // be called directly from execution_layer.cpp's find_active_expert
    // callback, on gpu_ctx->exp -- the SAME MoEExpertBlock currently being
    // dispatched for the in-flight k, about to be pushed into
    // scratch.pending_gpu for Pass B to read p.exp->expert_id fresh later.
    // Clearing expert_id here raced that read and produced a real, observed
    // "release_hold found no matching resident slot ... expert=-1" error
    // (see execution_layer.cpp's own comment at that reverted call site for
    // the full incident) -- the exact same "identity read late, not
    // captured" shape ram_temp_pin_count exists to prevent for RAM-tier
    // victim-selection SCANS, but this function bypasses that check
    // entirely since it's a direct field clear, not a scan. That call site
    // was reverted to marking-only (safe: also_in_vram=true doesn't destroy
    // identity). The remaining caller (cleanup_active_layer_area, which
    // only runs after a layer's Pass A/B are fully drained) has never shown
    // this failure mode in practice, but checking ram_temp_pin_count here
    // too costs nothing and closes the same category of gap defensively
    // rather than relying on "this call site happens not to be reachable
    // mid-flight" holding forever -- same discipline this project already
    // applies everywhere else a slot's identity can be cleared.
    // NOTE: this used to be one function that the (StreamingEngine&) overload
    // below called directly; now split into a private, non-locking _locked
    // core plus two independently-locking public overloads. ram_mutex is a
    // plain, non-recursive std::mutex (see RAM_TIMED_LOCK's own comment for
    // why), so the lookup overload can no longer call the block overload
    // directly (that would double-lock on the same thread and hang) -- both
    // now call this shared core instead, each taking the lock exactly once.
private:
    void finish_sync_promotion_locked(MoEExpertBlock& blk, int layer, int expert_id) {
        if (move_on_promotion && blk.ram_temp_pin_count <= 0) {
            blk.expert_id = -1;
            ram_expert_index.erase(swap_key(layer, expert_id));
            blk.last_used_clock = 0;
            blk.hit_count = 0;
            blk.also_in_vram = false;
            blk.slab_capacity = 0;
            blk.ram_loading = false;
            perf().n_move_on_promotion.fetch_add(1, std::memory_order_relaxed);
        } else {
            blk.also_in_vram = true;
            perf().n_copy_on_promotion.fetch_add(1, std::memory_order_relaxed);
        }
    }
public:
    void finish_sync_promotion(MoEExpertBlock& blk, int layer, int expert_id) {
        RAM_TIMED_LOCK(lock);
        finish_sync_promotion_locked(blk, layer, expert_id);
    }

    void finish_sync_promotion(StreamingEngine& engine, int layer, int expert_id) {
        if (layer < 0 || layer >= (int)engine.layers_experts.size()) return;
        RAM_TIMED_LOCK(lock);
        auto& layer_slots = engine.layers_experts[layer];
        size_t cap = layer_slots.empty() ? 0
            : std::min(layer_slots.size(), (size_t)std::max(1, cap_for_layer(layer)));
        ssize_t idx = find_ram_slot(layer_slots, cap, layer, expert_id);
        if (idx >= 0) finish_sync_promotion_locked(layer_slots[idx], layer, expert_id);
    }

    // Shared MOVE-vs-COPY promotion bookkeeping for a RAM->VRAM async
    // upload issued OUTSIDE get_expert/get_experts_batch's own internal
    // reactive-hit branches (which each already inline this same logic
    // themselves -- see their own comments). Added to close the "third,
    // independent tier-overlap source" documented in CLAUDE.md:
    // moe_forward_gguf's Phase 2 prefill cold-start VRAM warm-up
    // (execution_layer.cpp) calls upload_expert_async directly on a
    // freshly-loaded RAM block, entirely bypassing get_expert's own
    // promotion path -- so neither also_in_vram nor MOVE-freeing ever
    // happened for it, silently leaving a genuine RAM/VRAM double-
    // residency with no way for either tier's eviction scoring to know.
    // MOVE branch mirrors get_expert's Step 1 exactly (including the
    // ram_expert_index erase this session's RAM-tier indexing made a
    // real invariant -- see that member's own comment); COPY branch
    // deliberately does NOT set also_in_vram here (unlike get_expert's
    // own COPY branch, which only sets it after find_expert POLLS AND
    // CONFIRMS the copy landed) -- this caller has no such confirmation
    // point, and also_in_vram's own contract requires the copy be
    // genuinely done, not merely issued, before it's safe to mark. Given
    // move_on_promotion defaults true throughout this project's configs,
    // the COPY branch is the narrow, less-common case; it still gets a
    // correct swap-stat/counter update, just not the also_in_vram
    // optimization -- a real but small residual gap, not a correctness
    // issue (worst case: this slot is evicted slightly later than
    // strictly necessary, same "harmless, perf-only" tradeoff already
    // accepted elsewhere for also_in_vram).
    void apply_promotion_bookkeeping(MoEExpertBlock& blk, int layer, int expert_id,
                                      ExpertLocation* promoted) {
        RAM_TIMED_LOCK(lock);
        if (move_on_promotion) {
            blk.expert_id = -1;
            ram_expert_index.erase(swap_key(layer, expert_id));
            blk.last_used_clock = 0;
            blk.hit_count = 0;
            blk.also_in_vram = false;
            blk.slab_capacity = 0;
            blk.ram_loading = false;
            blk.pending_copy_event = promoted ? promoted->upload_event : nullptr;
            perf().n_move_on_promotion.fetch_add(1, std::memory_order_relaxed);
        } else {
            perf().n_copy_on_promotion.fetch_add(1, std::memory_order_relaxed);
        }
    }

    // Locked increment/decrement of MoEExpertBlock::ram_temp_pin_count --
    // replaces the old direct, unlocked `exp->ram_temp_pin_count++`/`--`
    // field pokes in execution_layer.cpp (moe_forward_gguf's Pass A -> Pass B
    // hold/release). ram_temp_pin_count is read, unlocked before this fix, by
    // every RAM-tier victim-selection scan (get_expert/get_experts_batch/
    // demote_from_vram) -- once a prefetcher worker thread genuinely exists
    // as a second caller into the RAM tier's claim/eviction logic, an
    // unlocked field write on the decode thread racing an unlocked read on a
    // prefetch worker thread is real UB, not just "safe because only one
    // thread ever touched this before". See MoEExpertBlock::ram_temp_pin_
    // count's own comment for what this field protects against.
    void ram_pin(MoEExpertBlock& blk) {
        RAM_TIMED_LOCK(lock);
        ++blk.ram_temp_pin_count;
    }
    void ram_unpin(MoEExpertBlock& blk) {
        RAM_TIMED_LOCK(lock);
        if (blk.ram_temp_pin_count > 0) --blk.ram_temp_pin_count;
    }

    static uint64_t eviction_score(const MoEExpertBlock& b) {
        return b.last_used_clock + (uint64_t)b.hit_count * FREQ_BONUS_TICKS;
    }

    // Halve hit_count across every layer's cache slots. Cheap since total
    // slot count (slots_per_layer * n_layers) is small and this runs rarely.
    void decay_hit_counts(StreamingEngine& engine) {
        for (auto& layer_slots : engine.layers_experts)
            for (auto& slot : layer_slots) slot.hit_count /= 2;
        hits_since_decay = 0;
    }

    double get_mem_available_gb() {
#ifdef __APPLE__
        // macOS: use host_statistics64 to get free + inactive pages
        mach_port_t host = mach_host_self();
        vm_statistics64_data_t vm_stat;
        mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
        if (host_statistics64(host, HOST_VM_INFO64, (host_info64_t)&vm_stat, &count) == KERN_SUCCESS) {
            uint64_t page_size = sysconf(_SC_PAGESIZE);
            uint64_t free_bytes = ((uint64_t)vm_stat.free_count + vm_stat.inactive_count) * page_size;
            return free_bytes / 1e9;
        }
        // Fallback: total physical memory
        int64_t memsize = 0;
        size_t len = sizeof(memsize);
        sysctlbyname("hw.memsize", &memsize, &len, nullptr, 0);
        return memsize / 1e9 * 0.7; // assume 70% available
#else
        std::ifstream file("/proc/meminfo");
        std::string line;
        double kb = 0;
        while (std::getline(file, line)) {
            if (line.find("MemAvailable:") == 0) {
                sscanf(line.c_str(), "MemAvailable: %lf", &kb);
                break;
            }
        }
        return kb / 1e6;
#endif
    }

    void calculate_safe_cache_capacity(StreamingEngine& engine, int max_ctx, double user_ram_gb = 0.0) {
        auto& c = engine.model_file.architecture_config;
        double requested = user_ram_gb > 0 ? user_ram_gb : configured_ram_gb;
        double ram_gb = requested > 0 ? requested : get_mem_available_gb() * 0.88;
        if (ram_gb < 4.0) ram_gb = 8.0;

        double resident_bytes = 0;
        for (const auto& [name, tensor] : engine.resident_layers) resident_bytes += tensor.meta.size_bytes;

        // Per-expert slab size, accounting for each projection's own quant
        // type (e.g. GLM-5.2 UD: gate/up=Q2_K, down=Q3_K -- NOT uniform).
        // Scans every layer via compute_expert_slab_bytes (the same helper
        // main.cpp uses for VRAM slab sizing) instead of assuming layer 0
        // has expert tensors: models with leading dense layers
        // (c.first_dense > 0, true for GLM-5.2) have plain
        // ffn_gate.weight/ffn_up.weight/ffn_down.weight at blk.0, not
        // blk.0.ffn_gate_exps.weight -- a hardcoded blk.0 lookup silently
        // returned 0 here, which propagated through the capmax division
        // below (available_for_experts / (layers * 0)) into a near-infinite
        // or UB static_cast<int>, which always floored slots_per_layer to 1
        // via std::max(1, ...). That 1-slot number was itself never the
        // real bug -- graph_initializer.cpp always allocates n_experts
        // slots per layer regardless -- but get_expert() below now actually
        // enforces slots_per_layer as a real cap, so this value has to be
        // correct or the cache is artificially starved to 1 expert/layer.
        double expert_bytes = 0;
        for (int l = 0; l < c.n_layers && expert_bytes == 0; ++l) {
            expert_bytes = (double)engine.compute_expert_slab_bytes(l);
        }

        double pc_b = 2.5e9;
        double ws_b = 64.0 * expert_bytes;
        double kv_b = (double)(c.n_layers + 1) * max_ctx * (c.kv_lora + c.qk_rope) * 4.0;
        double kvb_b = (double)max_ctx * c.n_heads * (c.qk_nope + c.v_head) * 4.0;

        double slack = 1.2e9 + pc_b + ws_b + kv_b + kvb_b;
        double available_for_experts = (ram_gb * 1e9) - resident_bytes - slack;

        int active_moe_layers = c.n_layers;
        int capmax = (available_for_experts > 0 && expert_bytes > 0)
            ? static_cast<int>(available_for_experts / (active_moe_layers * expert_bytes)) : 1;
        slots_per_layer = std::max(1, std::min(capmax, c.n_experts));

        LOG_STREAM_INFO() << "[RAM] Budget: " << ram_gb << " GB | Resident: " << resident_bytes/1e9 << " GB | Slack: " << slack/1e9 << " GB\n";
        LOG_STREAM_INFO() << "[RAM] LRU Cache sized to " << slots_per_layer << " experts per layer.\n";
    }

    // ram_cache_layer_adaptive: redistributes the FIXED total RAM-tier
    // budget (slots_per_layer * total_layers, already computed above and
    // never grown by this function -- this only reshapes how it's split)
    // across layers proportional to each layer's real historical need,
    // instead of the uniform per-layer split every other RAM-tier code
    // path used before this. "Need" for layer L = how many of L's own
    // historically-ranked experts (by selection count, from the SAME
    // per-layer-independent ranking main.cpp's [PIN] block already
    // computes from the usage file) are required to cover
    // `coverage_target` of L's total recorded selections -- a concentrated
    // layer (few experts dominate) needs few slots to capture most of its
    // value; a flat layer (traffic spread near-evenly over many experts)
    // needs proportionally more for the same coverage. Real numbers from
    // this project's own accumulated usage file motivated this (75 layers,
    // 672 selections/layer after several accumulated sessions): the most
    // concentrated layer needed only 31 ranked experts for 80% coverage,
    // the flattest needed 92 -- a ~3x range the uniform 40/layer split
    // used until now doesn't account for at all, while RAM-SLOTS was
    // simultaneously reporting 100% full (a real, binding constraint, not
    // a hypothetical one).
    //
    // Budget-safety is the hard requirement here, not distribution
    // precision -- this function must NEVER let the redistributed total
    // exceed what calculate_safe_cache_capacity already determined is
    // safe, since that function's whole purpose is preventing the kind of
    // unbounded RAM growth this project had a real, severe incident with
    // before (see CLAUDE.md). Three passes: (1) raw proportional shares
    // for layers with usage data; (2) clamp each to
    // [min_slots_per_layer, n_experts] -- a soft floor/ceiling that can
    // shift the sum away from the pool; (3) an UNCONDITIONAL rescale if
    // the clamped sum still exceeds the pool, which may push some layers
    // below the soft floor in pathological all-near-floor cases -- budget
    // safety intentionally wins over the floor guarantee. Layers with NO
    // usage data (never routed to yet, or the usage file predates them)
    // are left at the uniform `slots_per_layer` default, contributing
    // exactly their normal uniform share to the total either way, outside
    // the redistribution pool. First-ever run (no usage file, or an
    // empty/all-zero one) leaves slots_per_layer_by_layer empty entirely
    // -- cap_for_layer() then falls back to the plain uniform
    // `slots_per_layer` scalar for every layer, IDENTICAL to this
    // feature's disabled behavior, by construction (safe default: no
    // history yet means no redistribution decision can be trusted yet).
    void apply_layer_adaptive_capacity(const std::string& usage_file, int total_layers, int n_experts,
                                        double coverage_target = 0.8, int min_slots_per_layer = 8) {
        slots_per_layer_by_layer.clear();
        if (usage_file.empty() || slots_per_layer <= 0 || total_layers <= 0) return;
        auto [entries, maxc] = StreamingEngine::load_usage(usage_file);
        (void)maxc;
        if (entries.empty()) return;

        std::unordered_map<int, std::vector<uint64_t>> by_layer;
        for (auto& e : entries) {
            if (e.layer < 0 || e.layer >= total_layers) continue;
            by_layer[e.layer].push_back(e.count);
        }
        if (by_layer.empty()) return;

        std::vector<double> need(total_layers, 0.0);
        double need_sum = 0.0;
        for (auto& [l, v] : by_layer) {
            std::sort(v.begin(), v.end(), std::greater<uint64_t>());
            uint64_t total = 0;
            for (auto c : v) total += c;
            if (total == 0) continue;
            uint64_t cum = 0;
            double rank = 0;
            for (auto c : v) {
                cum += c; rank += 1.0;
                if ((double)cum >= coverage_target * (double)total) break;
            }
            need[l] = rank;
            need_sum += rank;
        }
        if (need_sum <= 0.0) return;

        int n_with_data = (int)by_layer.size();
        double pool = (double)slots_per_layer * n_with_data;

        slots_per_layer_by_layer.assign(total_layers, slots_per_layer);
        std::vector<double> clamped(total_layers, 0.0);
        double clamped_sum = 0.0;
        for (int l = 0; l < total_layers; ++l) {
            if (need[l] <= 0.0) continue;
            double raw = need[l] / need_sum * pool;
            clamped[l] = std::min((double)n_experts, std::max((double)min_slots_per_layer, raw));
            clamped_sum += clamped[l];
        }
        if (clamped_sum > pool && clamped_sum > 0.0) {
            double scale = pool / clamped_sum;
            for (int l = 0; l < total_layers; ++l) {
                if (need[l] <= 0.0) continue;
                clamped[l] *= scale;  // may go below min_slots_per_layer -- budget safety wins
            }
        }
        int lo = -1, hi = -1;
        for (int l = 0; l < total_layers; ++l) {
            if (need[l] <= 0.0) continue;
            int v = std::max(1, (int)std::lround(clamped[l]));
            slots_per_layer_by_layer[l] = v;
            if (lo == -1 || v < lo) lo = v;
            if (hi == -1 || v > hi) hi = v;
        }

        LOG_STREAM_INFO() << "[RAM] Layer-adaptive capacity: redistributed " << (long long)pool
                  << " slots across " << n_with_data << " layer(s) with usage history by need "
                  << "(coverage target " << (coverage_target * 100.0) << "%), range [" << lo << ", "
                  << hi << "] slots/layer (uniform default was " << slots_per_layer << "/layer)\n";
    }

    // Real per-layer active cap: slots_per_layer_by_layer[layer] if
    // apply_layer_adaptive_capacity populated it for this layer, else the
    // uniform slots_per_layer scalar -- every RAM-tier code path that used
    // to read `slots_per_layer` directly as a per-layer cap now goes
    // through this instead, so the adaptive/uniform choice lives in
    // exactly one place.
    int cap_for_layer(int layer) const {
        if (layer >= 0 && layer < (int)slots_per_layer_by_layer.size() && slots_per_layer_by_layer[layer] > 0)
            return slots_per_layer_by_layer[layer];
        return slots_per_layer;
    }

    // vram_evict_demote_to_ram: called (via GpuTopology::on_vram_evict) right
    // before a VRAM slot's outgoing expert is overwritten by an incoming
    // H2D upload. Salvages the outgoing bytes into a RAM-tier slot via a
    // D2H copy instead of letting evict_lru drop them outright -- before
    // this, a VRAM-evicted expert not ALSO already RAM-resident went fully
    // cold, so its next use always paid a full disk read even though the
    // identical bytes existed in VRAM a moment earlier. A PCIe D2H copy of
    // one expert slab (tens of MB) costs on the order of a millisecond;
    // the NVMe read it can save on reuse costs 15-50ms+ per this project's
    // own measured [MOE-CACHE] MISS timings -- asymmetric enough to be
    // worth the extra bandwidth even though not every salvaged expert will
    // actually be reused before its own turn to be evicted from RAM too.
    //
    // Only ever called from GpuTopology::upload_expert, which (per
    // on_vram_evict's own comment in gpu_topology.hpp) only ever runs on
    // the decode thread -- so this can safely mutate
    // engine.layers_experts[layer] with no new locking, preserving that
    // structure's existing "single-writer-thread" invariant (see
    // ram_swap_stats's own comment above) instead of requiring a new mutex
    // across the whole RAM tier. NOT wired into upload_expert_async (the
    // prefetcher thread's own promotion path) for exactly that reason --
    // see gpu_topology.hpp for the full explanation.
    void demote_from_vram(StreamingEngine& engine, int layer, int expert_id,
                           const uint8_t* device_ptr, size_t bytes, int gpu_id) {
        auto& layer_slots = engine.layers_experts[layer];
        if (layer_slots.empty()) {
            perf().n_vram_evict_demote_skipped.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        size_t cap = std::min(layer_slots.size(), (size_t)std::max(1, cap_for_layer(layer)));

        // Held for the whole rest of this function, including the D2H
        // cudaMemcpy below -- decode-thread-only (see the comment further
        // down on where this is called from), and cache_mutex is ALREADY
        // held by this function's own caller (GpuTopology's on_vram_evict,
        // fired from inside evict_lru/upload_expert/promote_active_to_main),
        // so this is the cache_mutex -> ram_mutex nesting direction the
        // unified claim design's deadlock analysis requires to stay
        // decode-thread-only (see claim_or_wait's own comment for the full
        // reasoning -- a PrefetchWorker-role caller must never acquire
        // cache_mutex while holding ram_mutex, or this nesting would become
        // a cross-thread AB-BA deadlock).
        RAM_TIMED_LOCK(lock);

        // Skip if already RAM-resident -- nothing to gain from a duplicate,
        // and it would just waste a D2H copy plus displace some OTHER
        // expert's RAM slot for no benefit.
        if (find_ram_slot(layer_slots, cap, layer, expert_id) >= 0) {
            perf().n_vram_evict_demote_skipped.fetch_add(1, std::memory_order_relaxed);
            return;
        }

        // Packed format or a validated unpacked_expert_format model both
        // have a well-defined per-layer (gb,ub,db) geometry via
        // expert_layer_geometry -- an UNvalidated legacy per-expert model
        // (unpacked_expert_format=false but per-expert tensor names present
        // anyway) has no single geometry to reconstruct host_ptr/meta
        // offsets from, so demotion is a silent (counted, not logged) no-op
        // for that case, matching how the prefetch fast path already degrades.
        size_t gb, ub, db;
        const GGUFTensor *gM, *uM, *dM;
        if (!engine.expert_layer_geometry(layer, gb, ub, db, gM, uM, dM)) {
            perf().n_vram_evict_demote_skipped.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        size_t total = gb + ub + db;
        // BUG FOUND on real hardware (a real run showed 2825 failed / 48
        // salvaged demotes, ~51% of attempts -- far too high to be edge-
        // case noise): this used to require total == bytes, on the WRONG
        // assumption that "both the VRAM slab and the RAM packed slab are
        // sized from this SAME geometry function" -- true of the SHAPE of
        // the calculation, false of the RESULT. GpuTopology's VRAM cache
        // uses ONE UNIFORM slot size (expert_slab_bytes, computed by
        // main.cpp's compute_expert_slab_bytes as the MAX per-expert size
        // across EVERY layer, since all slots must be the same fixed
        // size) -- but THIS layer's own packed_slab_geometry can be
        // smaller whenever its quant types are cheaper than whatever
        // layer set that max (this checkpoint has 3 different gate/up/
        // down type combinations across layers, e.g. down=IQ3_XXS vs the
        // larger down=IQ4_XS/IQ2_S -- a real, expected per-layer size
        // difference, confirmed via the [QUANT] startup report). `bytes`
        // (== the caller's slot->size) is therefore >= total for the
        // common case, not ==. The VRAM slot's ACTUAL VALID DATA has
        // always only ever occupied `total` bytes anyway -- upload_expert
        // itself already copies `std::min(caller_bytes, slot->size)` in,
        // which for a real caller (load_expert_slab, sized via this same
        // per-layer geometry) IS exactly `total` -- so `total` was always
        // the right byte count to D2H below; only the equality CHECK was
        // wrong. Fixed to the real invariant: total must never EXCEED the
        // uniformly-sized slot (would mean actual corruption upstream),
        // but total < bytes is the expected, common, safe case.
        if (total > bytes) {
            LOG_WARN("[MOE-RAM] demote_from_vram: geometry_bytes exceeds vram slot size "
                      "layer=%d expert=%d vram_bytes=%zu geometry_bytes=%zu -- skipped\n",
                      layer, expert_id, bytes, total);
            perf().n_vram_evict_demote_failed.fetch_add(1, std::memory_order_relaxed);
            return;
        }

        // Pick an eviction victim in THIS layer's RAM tier -- same
        // also_in_vram-aware scoring as Step 3 below. The slot being
        // filled here is, by definition, about to become
        // also_in_vram==false (its data just left VRAM, this D2H copy is
        // now the only one), so it's never short-circuited to score 0 by
        // that same rule on this call. copy_still_pending() and
        // ram_temp_pin_count > 0 both take priority over both: a slot with
        // an unconfirmed async H2D copy still reading its OLD slab_buffer
        // (MoEExpertBlock::pending_copy_event), OR one a raw pointer is
        // currently held across moe_forward_gguf's Pass A -> Pass B gap
        // (MoEExpertBlock::ram_temp_pin_count -- see its own comment for
        // the real release_hold-identity-mismatch bug this closes), must
        // never be picked -- scored as UINT64_MAX so either is only ever
        // chosen if literally every candidate has one of these problems.
        size_t victim_idx = 0;
        uint64_t victim_score = (layer_slots[0].copy_still_pending() || layer_slots[0].ram_temp_pin_count > 0
                                  || layer_slots[0].ram_loading)
            ? UINT64_MAX : (layer_slots[0].also_in_vram ? 0 : eviction_score(layer_slots[0]));
        for (size_t i = 1; i < cap; ++i) {
            uint64_t score = (layer_slots[i].copy_still_pending() || layer_slots[i].ram_temp_pin_count > 0
                               || layer_slots[i].ram_loading)
                ? UINT64_MAX : (layer_slots[i].also_in_vram ? 0 : eviction_score(layer_slots[i]));
            if (score < victim_score) { victim_score = score; victim_idx = i; }
        }
        MoEExpertBlock& victim = layer_slots[victim_idx];
        int displaced = victim.expert_id;
        // Measurement only (pilot_confirmed_set.hpp) -- see
        // GpuTopology::evict_lru's identical check for the full
        // explanation of what this counts and why. This D2H-salvage path
        // is a RAM-tier victim selection exactly like claim_or_wait's own
        // Claimed-branch victim selection, just triggered by an incoming
        // VRAM eviction instead of a RAM-tier miss.
        if (displaced != -1 && was_pilot_confirmed_resident(layer, displaced)) {
            perf().n_evicted_was_pilot_confirmed_ram.fetch_add(1, std::memory_order_relaxed);
        }
        if (displaced != -1 && victim.loaded_by_prefetch && victim.hit_count == 0) {
            perf().n_ram_prefetch_loaded_evicted_unused.fetch_add(1, std::memory_order_relaxed);
        }
        // Erase the outgoing identity's index entry now -- this slot is
        // committed to being overwritten by `expert_id` from here on,
        // whether the D2H copy below ultimately succeeds or fails (see the
        // failure branch further down, which also leaves the slot with a
        // NEW, empty identity rather than the old `displaced` one).
        if (displaced != -1) ram_expert_index.erase(swap_key(layer, displaced));

        // Defensive last resort: only actually blocks if EVERY candidate
        // above scored UINT64_MAX (all pending) and this one still got
        // picked -- cudaEventSynchronize on an event that's very likely
        // already signaled by now returns near-instantly in the common
        // case this branch is even entered.
        if (victim.pending_copy_event) {
            cudaEventSynchronize(victim.pending_copy_event);
            victim.pending_copy_event = nullptr;
        }

        // Previously completely untimed -- see t_vram_demote_copy's own
        // comment in perf_timer.hpp. Called from GpuTopology's on_vram_evict
        // callback, itself invoked from inside evict_lru while
        // GpuTopology::cache_mutex is held by the caller (upload_expert,
        // upload_expert_async, or promote_active_to_main) -- so this D2H
        // copy's time also directly extends every other thread's
        // t_cache_mutex_wait, not just this call's own latency.
        // ensure_slab_capacity is NOT included in this timer: it's a
        // possible posix_memalign/free pair, a different (and separately
        // worth investigating, if this timer turns out small) cost from the
        // cudaMemcpy itself.
        cudaSetDevice(gpu_id);
        engine.ensure_slab_capacity(victim, total);
        PERF_START(vram_demote_copy);
        cudaError_t err = cudaMemcpy(victim.slab_buffer, device_ptr, total, cudaMemcpyDeviceToHost);
        PERF_END(vram_demote_copy, t_vram_demote_copy);
        if (err != cudaSuccess) {
            LOG_WARN("[MOE-RAM] demote_from_vram: cudaMemcpy D2H failed layer=%d expert=%d (%s) -- slot left empty\n",
                      layer, expert_id, cudaGetErrorString(err));
            // Leave the slot genuinely empty rather than half-populated --
            // this project's standing discipline (see MoEExpertBlock's own
            // also_in_vram comment, the cudaMalloc-failure-checking fix
            // elsewhere in this codebase): a failed operation must never
            // leave state that looks valid but isn't.
            victim.expert_id = -1;
            victim.also_in_vram = false;
            victim.loaded_by_prefetch = false;
            victim.ram_loading = false;
            perf().n_vram_evict_demote_failed.fetch_add(1, std::memory_order_relaxed);
            return;
        }

        if (displaced != -1) ram_swap_stats[swap_key(layer, displaced)].swap_out++;
        engine.populate_packed_slab(victim, expert_id, gb, ub, db, *gM, *uM, *dM);
        victim.also_in_vram = false;  // it just left VRAM -- this RAM copy is the only one now
        victim.loaded_by_prefetch = false;  // fresh identity, synchronously salvaged, not prefetch-loaded
        victim.ram_loading = false;  // synchronous salvage, already fully populated above
        victim.last_used_clock = ++global_clock;
        victim.hit_count = 0;
        ram_swap_stats[swap_key(layer, expert_id)].swap_in++;
        ram_expert_index[swap_key(layer, expert_id)] = victim_idx;
        perf().n_vram_evict_demoted.fetch_add(1, std::memory_order_relaxed);

        LOG_DEBUG("[MOE-RAM] DEMOTE layer=%d expert=%d slot=%zu displaced=%d (%.2f MB salvaged from VRAM via D2H)\n",
                  layer, expert_id, victim_idx, displaced, total / 1e6);
    }

    // Unified claim/lookup entrypoint for the RAM tier -- the single locked
    // decision point BOTH the decode thread's own miss-handling (get_expert/
    // get_experts_batch) and a LockFreePrefetcher worker thread's speculative
    // claim (async_prefetcher.cpp's active_prefetch) go through. This is the
    // mechanism that eliminates "Cache-hit-beat-fetch" and the "Phase-A/
    // Phase-B ordering gap" (see CLAUDE.md) by construction: there is exactly
    // ONE locked check-and-decide per (layer, expert), never two independent
    // systems racing to reach the same conclusion.
    //
    // CRITICAL, load-bearing rule -- do not relax without re-reading the
    // deadlock analysis this was designed against: the Hit branch's
    // VRAM-promotion side effects (which take GpuTopology::cache_mutex) fire
    // ONLY when who==Decode. A PrefetchWorker caller landing in the Hit
    // branch must NEVER touch GpuTopology/cache_mutex -- doing so would open
    // a cross-thread AB-BA deadlock against the decode thread's own
    // cache_mutex-then-ram_mutex nesting (demote_from_vram/finish_sync_
    // promotion above, both reached from GpuTopology callbacks fired while
    // cache_mutex is still held -- see gpu_topology.cpp's promote_active_to_
    // main/find_active_expert, both firing on_promoted before releasing their
    // own GPU_TOPO_TIMED_LOCK). With this rule, ram_mutex->cache_mutex
    // nesting only ever happens on the decode thread, matching cache_mutex->
    // ram_mutex nesting's own decode-thread-only nature above -- no
    // cross-thread cycle is possible.
    //
    // Returns Hit (slot is the resident block, ready to use -- for Decode,
    // hit_count/last_used_clock/promotion bookkeeping already applied; for
    // PrefetchWorker, this is a pure "already resident, don't fetch" signal,
    // nothing mutated beyond a counter), Loading (someone else already
    // claimed this exact (layer,expert) and is mid-read -- slot is nullptr;
    // Decode should call wait_for_ram_slot, PrefetchWorker should just skip),
    // or Claimed (caller now EXCLUSIVELY owns this slot: expert_id/
    // ram_loading are set under the lock, but slab_buffer/gate_proj/up_proj/
    // down_proj are NOT yet valid -- caller must populate them OUTSIDE the
    // lock, then call publish_ram_slot on success or abandon_ram_claim on ANY
    // failure -- every populate exit path MUST resolve the claim to a
    // terminal state, or a decode-thread miss for this exact expert would
    // poll past [SLOW-RAM-WAIT] and wait forever; see MoEExpertBlock::
    // ram_loading's own field-ownership comment).
    //
    // extra_exclude (optional): additional slot indices this specific caller
    // knows must never be picked as a victim, beyond what ram_loading/
    // copy_still_pending()/ram_temp_pin_count already exclude -- used ONLY by
    // get_experts_batch, to protect a slot its OWN Hit branch already
    // MOVE-freed EARLIER in the SAME batch (expert_id==-1 but not yet safe to
    // reuse until the caller's out[k] pointer for that earlier k has been
    // read) from being picked as a LATER k's victim within the SAME call.
    // This is call-local, single-thread bookkeeping layered on top of
    // claim_or_wait's own cross-thread ram_loading protection, not a
    // replacement for it -- see get_experts_batch's own claimed_this_batch
    // comment for the full incident this prevents.
    RamClaimResult claim_or_wait(StreamingEngine& engine, int layer, int expert_id, RamClaimCaller who,
                                  const std::vector<size_t>* extra_exclude = nullptr) {
        RAM_TIMED_LOCK(lock);
        auto& layer_slots = engine.layers_experts[layer];
        size_t cap = layer_slots.empty() ? 0
            : std::min(layer_slots.size(), (size_t)std::max(1, cap_for_layer(layer)));

        ssize_t idx = find_ram_slot(layer_slots, cap, layer, expert_id);
        if (idx >= 0) {
            MoEExpertBlock& slot = layer_slots[(size_t)idx];
            if (slot.ram_loading) {
                if (who == RamClaimCaller::PrefetchWorker) {
                    perf().n_ram_claim_already_inflight_prefetch.fetch_add(1, std::memory_order_relaxed);
                }
                return {RamClaimStatus::Loading, nullptr, 0};
            }

            // -- Ready: an ordinary cache hit --
            slot.last_used_clock = ++global_clock;
            if (slot.hit_count < UINT32_MAX) slot.hit_count++;
            if (slot.from_padding) {
                perf().n_pad_reads_hit_later.fetch_add(1, std::memory_order_relaxed);
                slot.from_padding = false;
            }
            if (++hits_since_decay >= HIT_DECAY_INTERVAL) decay_hit_counts(engine);
            perf().n_expert_cache_hit.fetch_add(1, std::memory_order_relaxed);
            perf().n_ram_claim_hit.fetch_add(1, std::memory_order_relaxed);

            if (who == RamClaimCaller::PrefetchWorker) {
                // Structurally cannot re-fetch an already-resident expert --
                // this IS the by-construction elimination of the old
                // "Cache-hit-beat-fetch" waste category. No GpuTopology/
                // cache_mutex touch here, see this function's own CRITICAL
                // rule above.
                perf().n_ram_claim_already_resident_prefetch.fetch_add(1, std::memory_order_relaxed);
                return {RamClaimStatus::Hit, &slot, (size_t)idx};
            }

            if (verbose_cache_log) {
                LOG_DEBUG("[MOE-CACHE] HIT   layer=%d expert=%d hit_count=%u\n",
                          layer, expert_id, slot.hit_count);
            }

            // Async, non-blocking promotion to VRAM -- decode-thread-only
            // side effect (see this function's CRITICAL rule above). Same
            // logic as the pre-unification get_expert Step 1 / get_experts_
            // batch Phase A hit path.
            bool gpu_capable = slot.slab_buffer
                && gpu_type_supported(slot.gate_proj.meta.type)
                && gpu_type_supported(slot.up_proj.meta.type)
                && gpu_type_supported(slot.down_proj.meta.type);
            if (vram_cache && gpu_capable && slot.hit_count >= promote_hit_threshold
                && !is_cpu_affinity_layer(layer)) {
                ExpertLocation* resident = vram_cache->find_expert(layer, expert_id);
                if (!resident) {
                    LOG_DEBUG("[MOE-VRAM]  PROMOTE layer=%d expert=%d hit_count=%u -> async H2D upload (%.2f MB)\n",
                              layer, expert_id, slot.hit_count, slot.slab_capacity / 1e6);
                    ExpertLocation* promoted = vram_cache->upload_expert_async(
                        layer, expert_id, slot.slab_buffer, slot.slab_capacity);
                    if (promoted && slot.loaded_by_prefetch) {
                        perf().n_prefetch_staged_later_promoted.fetch_add(1, std::memory_order_relaxed);
                    }
                    if (move_on_promotion) {
                        slot.expert_id = -1;
                        ram_expert_index.erase(swap_key(layer, expert_id));
                        slot.last_used_clock = 0;
                        slot.hit_count = 0;
                        slot.also_in_vram = false;
                        slot.slab_capacity = 0;
                        slot.pending_copy_event = promoted ? promoted->upload_event : nullptr;
                        LOG_DEBUG("[MOE-VRAM]  MOVE: freed RAM slot immediately after promotion (layer=%d expert=%d)\n",
                                  layer, expert_id);
                        perf().n_move_on_promotion.fetch_add(1, std::memory_order_relaxed);
                    } else {
                        perf().n_copy_on_promotion.fetch_add(1, std::memory_order_relaxed);
                    }
                } else {
                    slot.also_in_vram = true;
                    if (pin_threshold > 0 && slot.hit_count >= pin_threshold) {
                        vram_cache->pin_expert(layer, expert_id);
                    }
                }
            }
            return {RamClaimStatus::Hit, &slot, (size_t)idx};
        }

        // -- Not found: claim a fresh victim slot -- same eviction-scoring
        // logic as before unification, now also excluding ram_loading slots
        // (a slot mid-claim by someone else must never be reused).
        int lru_idx = -1;
        uint64_t lru_score = 0;
        for (size_t i = 0; i < cap; ++i) {
            bool excluded = extra_exclude
                && std::find(extra_exclude->begin(), extra_exclude->end(), i) != extra_exclude->end();
            bool never_pick = excluded || layer_slots[i].copy_still_pending() || layer_slots[i].ram_temp_pin_count > 0
                || layer_slots[i].ram_loading;
            uint64_t score = never_pick ? UINT64_MAX
                : (layer_slots[i].also_in_vram ? 0 : eviction_score(layer_slots[i]));
            if (lru_idx < 0 || score < lru_score) { lru_score = score; lru_idx = (int)i; }
        }
        if (lru_idx < 0) {
            LOG_WARN("[RAM-CLAIM] layer=%d: no eviction-eligible slot available (every slot pending/loading) "
                     "-- falling back to slot 0, which may collide with an in-flight claim\n", layer);
            lru_idx = 0;
        }
        MoEExpertBlock& victim = layer_slots[(size_t)lru_idx];
        int evicted_expert_id = victim.expert_id;
        if (evicted_expert_id != -1) {
            if (was_pilot_confirmed_resident(layer, evicted_expert_id)) {
                perf().n_evicted_was_pilot_confirmed_ram.fetch_add(1, std::memory_order_relaxed);
            }
            if (victim.loaded_by_prefetch && victim.hit_count == 0) {
                perf().n_ram_prefetch_loaded_evicted_unused.fetch_add(1, std::memory_order_relaxed);
            }
            ram_swap_stats[swap_key(layer, evicted_expert_id)].swap_out++;
            ram_expert_index.erase(swap_key(layer, evicted_expert_id));
        }
        if (victim.pending_copy_event) {
            cudaEventSynchronize(victim.pending_copy_event);
            victim.pending_copy_event = nullptr;
        }
        if (victim.expert_id != -1 && engine.drop_expert_pages) {
            std::string base = "blk." + std::to_string(layer) + ".";
            std::string es = std::to_string(victim.expert_id);
            for (const char* proj : {"ffn_gate", "ffn_up", "ffn_down"}) {
                auto it = engine.model_file.tensors.find(base + proj + "." + es + ".weight");
                if (it != engine.model_file.tensors.end())
                    posix_fadvise(engine.model_file.fd_for_tensor(it->second),
                                  it->second.offset, it->second.size_bytes, POSIX_FADV_DONTNEED);
            }
        }
        victim.also_in_vram = false;
        victim.loaded_by_prefetch = (who == RamClaimCaller::PrefetchWorker);
        victim.from_padding = false;
        victim.expert_id = expert_id;
        victim.ram_loading = true;
        ram_expert_index[swap_key(layer, expert_id)] = (size_t)lru_idx;
        victim.last_used_clock = ++global_clock;
        victim.hit_count = 0;

        perf().n_expert_cache_miss.fetch_add(1, std::memory_order_relaxed);
        if (in_prefill) perf().n_expert_cache_miss_prefill.fetch_add(1, std::memory_order_relaxed);
        else perf().n_expert_cache_miss_decode.fetch_add(1, std::memory_order_relaxed);
        if (who == RamClaimCaller::Decode) perf().n_ram_claim_by_decode.fetch_add(1, std::memory_order_relaxed);
        else perf().n_ram_claim_by_prefetch.fetch_add(1, std::memory_order_relaxed);

        return {RamClaimStatus::Claimed, &victim, (size_t)lru_idx};
    }

    // Decode-thread-only: called after claim_or_wait returns Loading (someone
    // else -- a prefetch worker, almost always -- already claimed this exact
    // (layer,expert) and is mid-read). Polls until that claim's owner
    // publishes (ram_loading becomes false) rather than duplicate-reading:
    // ram_expert_index's one-slot-per-key invariant means a second claim for
    // the same key is never possible by construction, so there is exactly one
    // owner who WILL eventually publish or abandon (every populate-step exit
    // path resolves to a terminal state via publish_ram_slot/abandon_ram_
    // claim -- see MoEExpertBlock::ram_loading's own comment) -- indefinite
    // waiting is therefore safe, not a hang risk, and is the mechanism that
    // eliminates the old Phase-A/Phase-B ordering gap: the decode thread now
    // waits for the SAME in-flight read a prefetch worker already started,
    // instead of racing past it and paying for a duplicate synchronous read.
    // Returns nullptr if the in-flight claim was abandoned (populate failed)
    // rather than published -- caller should re-attempt claim_or_wait, which
    // will now see a genuinely free slot.
    MoEExpertBlock* wait_for_ram_slot(StreamingEngine& engine, int layer, int expert_id) {
        auto t0 = std::chrono::high_resolution_clock::now();
        bool warned = false;
        for (;;) {
            {
                RAM_TIMED_LOCK(lock);
                auto& layer_slots = engine.layers_experts[layer];
                size_t cap = layer_slots.empty() ? 0
                    : std::min(layer_slots.size(), (size_t)std::max(1, cap_for_layer(layer)));
                ssize_t idx = find_ram_slot(layer_slots, cap, layer, expert_id);
                if (idx < 0) return nullptr;  // claim was abandoned -- caller should retry claim_or_wait
                if (!layer_slots[(size_t)idx].ram_loading) {
                    double wait_ms = std::chrono::duration<double, std::milli>(
                        std::chrono::high_resolution_clock::now() - t0).count();
                    perf().n_ram_wait_inflight_decode.fetch_add(1, std::memory_order_relaxed);
                    PerfCounters::atomic_add(perf().t_ram_wait_inflight_ms_sum, wait_ms);
                    return &layer_slots[(size_t)idx];
                }
            }
            double elapsed_ms = std::chrono::duration<double, std::milli>(
                std::chrono::high_resolution_clock::now() - t0).count();
            if (!warned && elapsed_ms > SLOW_RAM_WAIT_MS) {
                LOG_WARN("[SLOW-RAM-WAIT] layer=%d expert=%d waited %.2fms so far for an in-flight claim "
                         "(owned by another caller, likely a prefetch worker) to publish\n",
                         layer, expert_id, elapsed_ms);
                warned = true;
            }
            usleep(150);
        }
    }

    // Publish a successful populate: retake the lock, mark Ready. Caller
    // (decode thread's own miss path, or a prefetch worker) must have already
    // filled slab_buffer/gate_proj/up_proj/down_proj (via StreamingEngine::
    // build_expert_read_requests + populate_packed_slab, or load_expert_slab)
    // BEFORE calling this -- ram_loading=false is the signal that makes those
    // fields visible to other threads/hit-checks. `slot` is the exact
    // MoEExpertBlock& claim_or_wait returned for this claim -- slot addresses
    // are stable for the process lifetime (graph_initializer.cpp allocates
    // every layer's full n_experts slots up front and never resizes), so
    // holding this reference across the unlocked populate step is safe.
    void publish_ram_slot(MoEExpertBlock& slot) {
        RAM_TIMED_LOCK(lock);
        slot.ram_loading = false;
    }

    // Undo a claim whose populate step failed (geometry lookup failed, every
    // read attempt failed) -- returns the slot to genuinely Empty rather than
    // leaving it stuck in Loading forever, which would hang any future
    // decode-thread miss for this exact expert via wait_for_ram_slot (claim
    // ownership is strictly single-owner by construction). Mirrors the
    // pre-existing revert_padding_claim pattern, generalized to any claim,
    // not just speculative padding.
    void abandon_ram_claim(int layer, MoEExpertBlock& slot) {
        RAM_TIMED_LOCK(lock);
        int expert_id = slot.expert_id;
        slot.expert_id = -1;
        ram_expert_index.erase(swap_key(layer, expert_id));
        slot.last_used_clock = 0;
        slot.hit_count = 0;
        slot.ram_loading = false;
        slot.loaded_by_prefetch = false;
    }

    // Startup-only: replaces main.cpp's old direct field pokes into
    // engine.layers_experts[layer][slot_idx] + manual ram_expert_index insert
    // (the Tier-3 RAM preload block) with a locked, single entry point --
    // closes a real gap: that preload used to be safe only because
    // LockFreePrefetcher's worker threads happened to not exist yet at that
    // point in main.cpp's startup sequence (an implicit timing argument, not
    // an enforced invariant). Marks the slot Ready immediately -- this is a
    // synchronous, already-completed load (engine.load_expert_slab has
    // already filled slab_buffer/gate_proj/up_proj/down_proj by the time this
    // is called), no Loading transition needed.
    void preload_ram_slot(StreamingEngine& engine, int layer, int expert_id, size_t slot_idx) {
        RAM_TIMED_LOCK(lock);
        auto& layer_slots = engine.layers_experts[layer];
        MoEExpertBlock& slot = layer_slots[slot_idx];
        slot.last_used_clock = ++global_clock;
        slot.hit_count = 0;
        slot.ram_loading = false;
        ram_expert_index[swap_key(layer, expert_id)] = slot_idx;
    }

    // Rewritten for the unified claim/publish design: the old Step 1 (hit
    // scan)/Step 2/Step 3 (victim-selection) logic, and the separate prefetch
    // staged-consume branch, all now live inside claim_or_wait (shared with
    // get_experts_batch and LockFreePrefetcher's active_prefetch) -- there is
    // no more separate staging pool to consult here, so a genuine miss after
    // claim_or_wait's Claimed branch always means a real synchronous read is
    // needed (nobody else -- including a prefetch worker -- is already
    // fetching this exact expert; claim_or_wait itself is what guarantees
    // that, not an extra check here).
    MoEExpertBlock* get_expert(StreamingEngine& engine, int layer, int expert_id) {
        // Whole-function cost -- see this function's own PERF_SCOPE history:
        // the RAM tier (this function and its batched sibling) is the
        // largest remaining scope in the whole decode-time hot path.
        PERF_SCOPE(t_ram_get_expert);
        for (;;) {
            RamClaimResult result = claim_or_wait(engine, layer, expert_id, RamClaimCaller::Decode);
            if (result.status == RamClaimStatus::Hit) {
                return result.slot;
            }
            if (result.status == RamClaimStatus::Loading) {
                MoEExpertBlock* slot = wait_for_ram_slot(engine, layer, expert_id);
                if (slot) return slot;
                continue;  // the in-flight claim was abandoned -- retry the claim ourselves
            }

            // -- Claimed: this caller now exclusively owns the slot; populate
            // it OUTSIDE the lock via the same synchronous loader every
            // other real-miss path already uses --
            MoEExpertBlock& victim = *result.slot;
            PERF_START(expert_disk_load);
            engine.load_expert_slab(layer, expert_id, victim);
            double load_ms = std::chrono::duration<double, std::milli>(
                std::chrono::high_resolution_clock::now() - _perf_expert_disk_load).count();
            ram_swap_stats[swap_key(layer, expert_id)].swap_in++;
            publish_ram_slot(victim);

            LOG_DEBUG("[MOE-CACHE] MISS  layer=%d expert=%d slot=%zu loaded_from_disk=%.2fms via=pread "
                      "(%.2f MB, %.2f GB/s)\n",
                      layer, expert_id, result.slot_idx, load_ms,
                      victim.slab_capacity / 1e6,
                      load_ms > 0 ? (victim.slab_capacity / 1e9) / (load_ms / 1000.0) : 0.0);

            // Note: no eager VRAM promotion here — a cold disk load hasn't
            // proven hot yet (hit_count starts at 0). It will be promoted
            // asynchronously by claim_or_wait's Hit branch once it's
            // actually re-hit (promote_hit_threshold).
            return &victim;
        }
    }

    // Batched sibling of get_expert(), for fetching one token's ENTIRE topk
    // selection within a single layer at once. Motivated by a direct
    // measurement: buffered reads scale ~8.5x from 1->4 concurrent
    // threads on this hardware (1.53 -> 13.12 GB/s), which single-expert
    // get_expert()'s strictly-sequential one-pread-at-a-time loop (the
    // caller's original `for (k) get_expert(...)`) never exploits — every
    // miss pays its full read latency before the next one even starts.
    //
    // Rewritten to three phases for the unified claim/publish design:
    //   Phase A (sequential): for each k, call claim_or_wait and classify
    //     into Hit (out[k] set immediately, no I/O), MineToFetch (this call
    //     won the Empty->Loading claim -- added to `pending`), or WaitOnOther
    //     (someone else, almost always a prefetch worker, already claimed
    //     this exact expert and is mid-read -- added to `waiting`, NOT
    //     resolved inline here: see Phase C below for why).
    //   Phase B (parallel, UNCHANGED from before unification): for `pending`,
    //     run the actual read via io_uring batch (+ disk_batch_padding) or
    //     the OMP-parallel load_expert_slab fallback, across
    //     expert_fetch_threads threads. Safe with zero extra synchronization
    //     because each pending entry owns a DISTINCT, already-claimed slot.
    //   Phase C (sequential, NEW): resolve `waiting` via wait_for_ram_slot.
    //     Deliberately AFTER Phase B, not inline in Phase A -- Phase A must
    //     stay non-blocking (a WaitOnOther k's wait shouldn't stall every
    //     OTHER k in the batch behind it), and running Phase C after Phase B
    //     lets its wait overlap with Phase B's own real disk I/O: by the time
    //     Phase B's reads finish, the prefetch worker's in-flight fetch for a
    //     Phase C entry has very often finished too.
    //
    // out[k] is valid (non-null, correct identity) the moment this call
    // returns for every k — every phase has already completed by then, so
    // callers see the exact same synchronous-completion contract
    // get_expert() has always had, just serviced faster.
    void get_experts_batch(StreamingEngine& engine, int layer, const int* expert_ids, int topk,
                           MoEExpertBlock** out) {
        // See get_expert's own comment on t_ram_get_expert -- same rationale,
        // this is get_expert's batched sibling and the higher-traffic of the
        // two on the real decode-time call path (called once per non-shadowed
        // subset per token/layer, vs get_expert's per-expert calls elsewhere).
        PERF_SCOPE(t_ram_get_experts_batch);
        auto& layer_slots = engine.layers_experts[layer];
        size_t cap = layer_slots.empty() ? 0
            : std::min(layer_slots.size(), (size_t)std::max(1, cap_for_layer(layer)));

        struct PendingLoad {
            size_t slot_idx; int expert_id; int k; double load_ms;
        };
        std::vector<PendingLoad> pending;
        pending.reserve(topk);
        struct WaitEntry { int expert_id; int k; };
        std::vector<WaitEntry> waiting;

        // Slots this batch has already claimed as a miss's victim, PLUS any
        // slot this batch's OWN Hit branch MOVE-freed earlier in the SAME
        // loop -- passed to claim_or_wait as extra_exclude so its victim-
        // selection can't reuse a slot a LATER k in this SAME batch hasn't
        // read from yet. claim_or_wait's own ram_loading flag already
        // protects a Claimed slot; this vector exists specifically for the
        // MOVE-promotion-freed case, which looks genuinely Empty
        // (expert_id==-1) to claim_or_wait despite an EARLIER k's out[k]
        // still pointing at its (unread) gate_proj/up_proj/down_proj/
        // slab_buffer -- see the push_back below for the exact incident this
        // prevents.
        std::vector<size_t> claimed_this_batch;
        claimed_this_batch.reserve(topk);

        // --- Phase A: sequential claim, via the unified entrypoint shared
        // with get_expert and LockFreePrefetcher's active_prefetch. No I/O,
        // no blocking -- WaitOnOther entries are deferred to Phase C so one
        // slow in-flight prefetch can't stall the rest of this batch. ---
        for (int k = 0; k < topk; ++k) {
            int expert_id = expert_ids[k];
            RamClaimResult result = claim_or_wait(engine, layer, expert_id, RamClaimCaller::Decode,
                                                   &claimed_this_batch);
            if (result.status == RamClaimStatus::Hit) {
                out[k] = result.slot;
                if (result.slot->expert_id == -1) {
                    // claim_or_wait's Hit branch just MOVE-promoted and freed
                    // this slot -- out[k] still points at its (still valid,
                    // not-yet-read-by-the-caller) gate_proj/up_proj/
                    // down_proj/slab_buffer. Protect it from being picked as
                    // a LATER k's victim in this SAME batch (which would
                    // corrupt or replace that data before this k's own
                    // caller ever reads it) -- see claimed_this_batch's own
                    // comment above.
                    claimed_this_batch.push_back(result.slot_idx);
                }
                continue;
            }
            if (result.status == RamClaimStatus::Loading) {
                waiting.push_back({expert_id, k});
                continue;
            }
            // Claimed -- this call now exclusively owns the slot; the actual
            // read happens in Phase B below. out[k] is set NOW, not after
            // Phase B populates the bytes -- the slot pointer itself is
            // already stable and correct (this call owns it exclusively
            // until publish_ram_slot), and out[k]'s own contract only
            // requires validity once this whole function returns, by which
            // point Phase B has always already completed synchronously.
            // REAL BUG this fixes (found on real hardware): the original
            // rewrite of this function dropped this assignment entirely --
            // every Hit and WaitOnOther k got a correct out[k], but every
            // genuine cache-miss k (the common case) left out[k] holding
            // whatever garbage was already in the caller's stack-allocated
            // array, read later as a dangling/stale MoEExpertBlock* (NaN
            // gate/up/down matmul output) or occasionally a literal NULL
            // (caught by execution_layer.cpp's defensive null check, which
            // silently drops that expert's contribution instead of crashing
            // -- an equally real correctness bug, just a quieter one).
            out[k] = &layer_slots[result.slot_idx];
            claimed_this_batch.push_back(result.slot_idx);
            pending.push_back({result.slot_idx, expert_id, k, 0.0});
        }

        // --- Phase B: batch the WHOLE layer's pending set into one io_uring submission when
        // the WHOLE layer's pending set into one io_uring submission when
        // available (colibri's own real miss-servicing design -- kernel-
        // managed deep queue depth across many concurrent reads at once,
        // not N OS threads each blocking on their own pread). All-or-
        // nothing: if the ring is unavailable, any pending expert isn't
        // packed-format, the batch would exceed the ring's capacity, or
        // ANY single read in the batch comes back short/failed, this
        // falls through to the ORIGINAL OMP-parallel load_expert_slab
        // loop for every pending expert (unconditionally correct,
        // untouched) -- never a partial mix of the two paths, and never a
        // half-populated slab: load_expert_slab always overwrites its
        // slab_buffer from scratch, so a failed uring attempt's partial
        // writes are safely clobbered on fallback. Each pending entry owns
        // a DISTINCT, already-claimed slot (guaranteed by
        // claimed_this_batch above), so both paths are safe with no
        // additional synchronization.
        int n_pending = (int)pending.size();
        if (n_pending > 0) {
            // A per-item "recheck right before the read" fix was tried here
            // (and in the OMP fallback below) in an earlier, PRE-unification
            // design that still had a separate prefetcher staging pool --
            // measured on real hardware to be a net negative (near-zero
            // rescue rate, plus a real counter-corruption regression) and
            // fully reverted. Moot now regardless: there is no separate
            // staging pool left to recheck against -- every `pending` entry
            // here already went through claim_or_wait, which is itself the
            // single, unified check (see that function's own comment for why
            // this eliminates the ordering-gap class of problem entirely,
            // rather than papering over it with a second check).
            bool uring_ok = false;
            // Root-cause-attributed outcome of THIS batch's uring attempt --
            // see perf_timer.hpp's n_decode_uring_* comment for why this was
            // added (a real run showed the ring initialize successfully at
            // startup, yet get_batch=/disk= differed by ~3.7x, the
            // signature of the OMP-fallback thread-summing firing far more
            // than expected -- this makes the actual per-batch outcome
            // directly visible instead of inferred from that ratio).
            if (!(use_decode_uring && decode_uring_ring.available())) {
                perf().n_decode_uring_not_available.fetch_add(1, std::memory_order_relaxed);
            }
            if (use_decode_uring && decode_uring_ring.available()) {
                std::vector<io_uring_backend::ReadRequest> reqs;
                reqs.reserve((size_t)n_pending * 3);
                bool all_packed = true;
                for (int i = 0; i < n_pending && all_packed; ++i) {
                    io_uring_backend::ReadRequest tmp[3];
                    int cnt = 0; size_t total = 0;
                    MoEExpertBlock& victim = layer_slots[pending[i].slot_idx];
                    if (!engine.build_expert_read_requests(layer, pending[i].expert_id, victim,
                                                             tmp, cnt, total)) {
                        all_packed = false;
                        break;
                    }
                    for (int j = 0; j < cnt; ++j) reqs.push_back(tmp[j]);
                }

                // --- Speculative Phase B padding (disk_batch_padding) ---
                // Rides THIS already-necessary io_uring submission with extra
                // reads of historically-hot, not-currently-resident experts
                // for the SAME layer, pushing queue depth toward disk_batch_
                // padding_target -- see EngineConfig::disk_batch_padding's own
                // comment for the measured queue-depth effect this exists to
                // capture, and the known ramp-up-only limitation of this V1
                // (free-slots-only, never evicts a resident). Only attempted
                // when the REAL portion is already structurally sound and
                // about to proceed to uring on its own -- never spent on a
                // batch that's falling back to OMP anyway.
                size_t real_reqs_count = reqs.size();
                std::vector<int> pad_ids;
                std::vector<size_t> pad_slots;
                std::vector<int> pad_req_counts;
                // Independent, non-short-circuited term-by-term visibility
                // into the padding gate below -- see n_pad_dbg_calls' own
                // comment (perf_timer.hpp) for why this was added. Cheap:
                // 6 relaxed atomic increments per real Phase B call.
                perf().n_pad_dbg_calls.fetch_add(1, std::memory_order_relaxed);
                if (!disk_batch_padding) perf().n_pad_dbg_padding_false.fetch_add(1, std::memory_order_relaxed);
                if (!all_packed) perf().n_pad_dbg_not_all_packed.fetch_add(1, std::memory_order_relaxed);
                if (reqs.empty()) perf().n_pad_dbg_reqs_empty.fetch_add(1, std::memory_order_relaxed);
                if (in_prefill) perf().n_pad_dbg_in_prefill.fetch_add(1, std::memory_order_relaxed);
                if (n_pending >= disk_batch_padding_target) perf().n_pad_dbg_target_not_reached.fetch_add(1, std::memory_order_relaxed);
                if (disk_batch_padding && all_packed && !reqs.empty() && !in_prefill
                    && n_pending < disk_batch_padding_target) {
                    perf().n_pad_gate_entered.fetch_add(1, std::memory_order_relaxed);
                    int want = disk_batch_padding_target - n_pending;
                    auto rr_it = route_rank_by_layer.find(layer);
                    if (want > 0 && rr_it == route_rank_by_layer.end()) {
                        perf().n_pad_no_ranking_for_layer.fetch_add(1, std::memory_order_relaxed);
                    }
                    if (want > 0 && rr_it != route_rank_by_layer.end()) {
                        // Walk the persisted, pre-sorted-by-historical-count
                        // ranking (see route_rank_by_layer's own comment for
                        // why this is used instead of live route_freq),
                        // filtering out: this token's OWN real routed set
                        // (without this exclusion a real miss could ALSO be
                        // picked as a "candidate" and double-claimed into a
                        // SECOND RAM slot, corrupting ram_expert_index's
                        // one-slot-per-key invariant), anything already
                        // RAM-resident, and anything already VRAM-resident
                        // (pinned or reactively cached -- the ranking's top
                        // entries are exactly the kind of expert most likely
                        // to already be VRAM-pinned, since pinning uses this
                        // SAME data; without this check padding would
                        // routinely load an already-VRAM-resident expert
                        // into RAM too: zero new benefit, pure duplicated
                        // capacity, a real [TIER-OVERLAP] source). Already
                        // sorted descending, so taking the first `want`
                        // survivors needs no separate sort/resize step.
                        std::vector<int> ranked;
                        ranked.reserve((size_t)want);
                        for (int e : rr_it->second) {
                            if ((int)ranked.size() >= want) break;
                            bool is_this_tokens_own = false;
                            for (int kk = 0; kk < topk; ++kk) {
                                if (expert_ids[kk] == e) { is_this_tokens_own = true; break; }
                            }
                            if (is_this_tokens_own) continue;
                            if (is_ram_resident(layer, e)) continue;
                            if (vram_cache && vram_cache->find_expert(layer, e)) continue;
                            ranked.push_back(e);
                        }
                        if (ranked.empty()) {
                            perf().n_pad_no_candidates_after_filter.fetch_add(1, std::memory_order_relaxed);
                        }

                        // Free (expert_id==-1), not-already-claimed-this-batch
                        // slots only -- V1 NEVER evicts a real resident. The
                        // scan + claim below are done under ram_mutex in ONE
                        // critical section (not just the field mutations) --
                        // now that a prefetch worker thread can concurrently
                        // claim slots via claim_or_wait, reading
                        // layer_slots[i].expert_id unlocked here would race
                        // that thread's own writes.
                        std::vector<size_t> free_slots;
                        std::vector<int> claimed_candidate_ids;
                        {
                            RAM_TIMED_LOCK(pad_lock);
                            free_slots.reserve(ranked.size());
                            for (size_t i = 0; i < cap && free_slots.size() < ranked.size(); ++i) {
                                if (layer_slots[i].expert_id != -1) continue;
                                if (std::find(claimed_this_batch.begin(), claimed_this_batch.end(), i)
                                    != claimed_this_batch.end()) continue;
                                free_slots.push_back(i);
                            }
                            if (free_slots.size() < ranked.size()) {
                                perf().n_pad_no_free_slot.fetch_add(ranked.size() - free_slots.size(),
                                                                     std::memory_order_relaxed);
                            }

                            size_t n_pad = std::min(ranked.size(), free_slots.size());
                            claimed_candidate_ids.reserve(n_pad);
                            for (size_t p = 0; p < n_pad; ++p) {
                                int candidate_id = ranked[p];
                                size_t slot_idx = free_slots[p];
                                MoEExpertBlock& pv = layer_slots[slot_idx];

                                // Claim exactly like a real miss's own claim
                                // sequence above -- mostly a no-op for a
                                // genuinely-free slot (no evicted identity, no
                                // pending copy), mirrored verbatim for
                                // consistency/future-proofing.
                                claimed_this_batch.push_back(slot_idx);
                                pv.also_in_vram = false;
                                pv.loaded_by_prefetch = false;
                                if (pv.pending_copy_event) {
                                    cudaEventSynchronize(pv.pending_copy_event);
                                    pv.pending_copy_event = nullptr;
                                }
                                pv.expert_id = candidate_id;
                                pv.ram_loading = true;
                                ram_expert_index[swap_key(layer, candidate_id)] = slot_idx;
                                pv.last_used_clock = ++global_clock;
                                pv.hit_count = 0;
                                pv.from_padding = true;
                                claimed_candidate_ids.push_back(candidate_id);
                            }
                            // free_slots and claimed_candidate_ids are built
                            // in lockstep above (one push_back each per
                            // iteration of the SAME loop), so they always end
                            // up the same size -- free_slots[p] is exactly
                            // claimed_candidate_ids[p]'s slot below.
                        }
                        for (size_t p = 0; p < claimed_candidate_ids.size(); ++p) {
                            int candidate_id = claimed_candidate_ids[p];
                            size_t slot_idx = free_slots[p];
                            MoEExpertBlock& pv = layer_slots[slot_idx];

                            io_uring_backend::ReadRequest ptmp[3];
                            int pcnt = 0; size_t ptotal = 0;
                            if (!engine.build_expert_read_requests(layer, candidate_id, pv,
                                                                     ptmp, pcnt, ptotal)) {
                                revert_padding_claim(pv, layer, candidate_id);
                                continue;
                            }
                            for (int j = 0; j < pcnt; ++j) reqs.push_back(ptmp[j]);
                            pad_ids.push_back(candidate_id);
                            pad_slots.push_back(slot_idx);
                            pad_req_counts.push_back(pcnt);
                            perf().n_pad_reads_issued.fetch_add(1, std::memory_order_relaxed);
                        }

                        // Defense-in-depth: should be provably unreachable --
                        // disk_batch_padding_target is clamped to [1,21] in
                        // EngineConfig::validate() specifically so n_pending+
                        // want <= 21, keeping total requests <= 21*3=63 <=
                        // Ring::CAPACITY(64) regardless of packed(3x)/
                        // unpacked(1x) format -- but padding must NEVER be
                        // the reason a real batch that would otherwise have
                        // fit gets pushed into the OMP fallback, so this is
                        // enforced directly rather than only trusted.
                        while (reqs.size() > io_uring_backend::Ring::CAPACITY && !pad_ids.empty()) {
                            int drop_id = pad_ids.back();
                            size_t drop_slot = pad_slots.back();
                            int drop_cnt = pad_req_counts.back();
                            pad_ids.pop_back();
                            pad_slots.pop_back();
                            pad_req_counts.pop_back();
                            reqs.resize(reqs.size() - (size_t)drop_cnt);
                            revert_padding_claim(layer_slots[drop_slot], layer, drop_id);
                            perf().n_pad_reads_issued.fetch_sub(1, std::memory_order_relaxed);
                        }
                    }
                }

                if (!all_packed) {
                    perf().n_decode_uring_not_packed.fetch_add(1, std::memory_order_relaxed);
                } else if (reqs.empty()) {
                    // Should be unreachable (n_pending > 0 implies at least
                    // one request), but if it ever happens, don't silently
                    // miscount it as any of the other buckets.
                } else if (reqs.size() > io_uring_backend::Ring::CAPACITY) {
                    perf().n_decode_uring_too_large.fetch_add(1, std::memory_order_relaxed);
                }
                if (all_packed && !reqs.empty() && reqs.size() <= io_uring_backend::Ring::CAPACITY) {
                    std::vector<ssize_t> results(reqs.size());
                    auto t0 = std::chrono::high_resolution_clock::now();
                    bool submitted = decode_uring_ring.submit_and_wait(reqs.data(), (int)reqs.size(), results.data());
                    double batch_ms = std::chrono::duration<double, std::milli>(
                        std::chrono::high_resolution_clock::now() - t0).count();
                    // Previously completely untimed -- batch_ms existed only
                    // for the LOG_DEBUG line below, never fed into
                    // t_disk_read the way the OMP-fallback path's
                    // load_expert_slab (engine.hpp) already does internally.
                    // Since this whole submit_and_wait call sits inside
                    // get_experts_batch's own PERF_SCOPE(t_ram_get_experts_
                    // batch), the real disk I/O cost was silently being
                    // attributed to "scanning" (ram_scan=/get_batch= in the
                    // [REQUEST] line) instead of "disk=" -- discovered after
                    // two separate scan-focused index fixes to
                    // get_experts_batch produced ZERO measurable change to
                    // get_batch= across three real runs, which should have
                    // been the tell that the dominant cost wasn't the scan
                    // at all. Added unconditionally (matching load_expert_
                    // slab's own PERF_END, which fires regardless of a
                    // short/failed read) since a failed or partial uring
                    // attempt still consumed real wall-clock time.
                    PerfCounters::atomic_add(perf().t_disk_read, batch_ms / 1000.0);
                    // Narrowed to [0, real_reqs_count) -- see the padding
                    // block above. When padding contributed nothing,
                    // real_reqs_count == reqs.size(), so this is byte-
                    // identical to checking the whole array, exactly as
                    // before. This is what stops a padding-request failure
                    // from wastefully triggering a full OMP re-read of
                    // already-successfully-read real data.
                    bool all_ok = submitted;
                    if (submitted) {
                        for (size_t r = 0; r < real_reqs_count; ++r) {
                            if (results[r] != (ssize_t)reqs[r].len) { all_ok = false; break; }
                        }
                    }
                    if (!submitted) {
                        perf().n_decode_uring_submit_failed.fetch_add(1, std::memory_order_relaxed);
                    } else if (!all_ok) {
                        perf().n_decode_uring_short_read.fetch_add(1, std::memory_order_relaxed);
                    }

                    // --- Padding outcome, independent of the real portion's
                    // all_ok (a padding failure must never cause the real,
                    // already-successfully-read misses to be wastefully
                    // redone via the OMP fallback below) ---
                    if (!pad_ids.empty()) {
                        if (!submitted) {
                            // io_uring_enter itself failed -- nothing was
                            // read at all (Ring::submit_and_wait's own
                            // contract: either every request is submitted
                            // and every result written, or nothing is).
                            // Revert every padding claim; the OMP fallback
                            // below correctly services pending[] alone.
                            for (size_t p = 0; p < pad_ids.size(); ++p) {
                                revert_padding_claim(layer_slots[pad_slots[p]], layer, pad_ids[p]);
                            }
                        } else {
                            size_t r = real_reqs_count;
                            uint64_t pad_bytes = 0;
                            for (size_t p = 0; p < pad_ids.size(); ++p) {
                                int cand_cnt = pad_req_counts[p];
                                bool cand_ok = true;
                                for (int j = 0; j < cand_cnt; ++j) {
                                    if (results[r + (size_t)j] != (ssize_t)reqs[r + (size_t)j].len) {
                                        cand_ok = false;
                                        break;
                                    }
                                }
                                MoEExpertBlock& pv = layer_slots[pad_slots[p]];
                                if (cand_ok) {
                                    size_t gb, ub, db;
                                    const GGUFTensor *gM, *uM, *dM;
                                    engine.expert_layer_geometry(layer, gb, ub, db, gM, uM, dM);
                                    engine.populate_packed_slab(pv, pad_ids[p], gb, ub, db, *gM, *uM, *dM);
                                    publish_ram_slot(pv);
                                    pad_bytes += gb + ub + db;
                                    perf().n_pad_reads_ok.fetch_add(1, std::memory_order_relaxed);
                                } else {
                                    revert_padding_claim(pv, layer, pad_ids[p]);
                                    perf().n_pad_reads_failed.fetch_add(1, std::memory_order_relaxed);
                                }
                                r += (size_t)cand_cnt;
                            }
                            if (pad_bytes > 0) {
                                perf().bytes_disk_read_speculative.fetch_add((int64_t)pad_bytes,
                                                                              std::memory_order_relaxed);
                            }
                        }
                    }

                    if (all_ok) {
                        size_t total_bytes = 0;
                        for (int i = 0; i < n_pending; ++i) {
                            size_t gb, ub, db;
                            const GGUFTensor *gM, *uM, *dM;
                            engine.expert_layer_geometry(layer, gb, ub, db, gM, uM, dM);
                            MoEExpertBlock& victim = layer_slots[pending[i].slot_idx];
                            engine.populate_packed_slab(victim, pending[i].expert_id, gb, ub, db, *gM, *uM, *dM);
                            publish_ram_slot(victim);
                            pending[i].load_ms = batch_ms; // shared batch time; per-expert split isn't meaningful here
                            total_bytes += gb + ub + db;
                            ram_swap_stats[swap_key(layer, pending[i].expert_id)].swap_in++;
                        }
                        perf().bytes_disk_read.fetch_add(total_bytes, std::memory_order_relaxed);
                        uring_ok = true;
                        perf().n_decode_uring_ok.fetch_add(1, std::memory_order_relaxed);
                        LOG_DEBUG("[MOE-CACHE] MISS  layer=%d batch=%d experts via=io_uring took=%.2fms "
                                  "(%.2f MB, %.2f GB/s)\n",
                                  layer, n_pending, batch_ms, total_bytes / 1e6,
                                  batch_ms > 0 ? (total_bytes / 1e9) / (batch_ms / 1000.0) : 0.0);
                    }
                }
            }
            if (!uring_ok) {
                int nthreads = std::max(1, std::min(expert_fetch_threads, n_pending));
                #pragma omp parallel for num_threads(nthreads) schedule(dynamic, 1)
                for (int i = 0; i < n_pending; ++i) {
                    PendingLoad& pl = pending[i];
                    MoEExpertBlock& victim = layer_slots[pl.slot_idx];
                    auto t0 = std::chrono::high_resolution_clock::now();
                    engine.load_expert_slab(layer, pl.expert_id, victim);
                    pl.load_ms = std::chrono::duration<double, std::milli>(
                        std::chrono::high_resolution_clock::now() - t0).count();
                }
                // Logging, swap-in accounting, and publish_ram_slot all stay
                // sequential, outside the parallel region -- ram_swap_stats
                // is a plain unordered_map (unsafe to mutate concurrently),
                // and publish_ram_slot's own lock would otherwise contend
                // across every OMP thread for no benefit (each slot is
                // already exclusively owned by exactly one iteration).
                for (int i = 0; i < n_pending; ++i) {
                    PendingLoad& pl = pending[i];
                    MoEExpertBlock& victim = layer_slots[pl.slot_idx];
                    ram_swap_stats[swap_key(layer, pl.expert_id)].swap_in++;
                    publish_ram_slot(victim);
                    LOG_DEBUG("[MOE-CACHE] MISS  layer=%d expert=%d slot=%zu loaded_from_disk=%.2fms via=pread "
                              "(%.2f MB, %.2f GB/s) [batch=%d/%d]\n",
                              layer, pl.expert_id, pl.slot_idx, pl.load_ms,
                              victim.slab_capacity / 1e6,
                              pl.load_ms > 0 ? (victim.slab_capacity / 1e9) / (pl.load_ms / 1000.0) : 0.0,
                              i + 1, n_pending);
                }
            }
        }

        // --- Phase C: resolve WaitOnOther entries from Phase A. Deliberately
        // AFTER Phase B (see this function's own comment on why) -- by now,
        // whatever prefetch worker claimed each of these has very often
        // already finished, so this wait is usually short or immediate. ---
        for (const WaitEntry& w : waiting) {
            MoEExpertBlock* slot = wait_for_ram_slot(engine, layer, w.expert_id);
            if (slot) {
                out[w.k] = slot;
                continue;
            }
            // The in-flight claim was abandoned (its populate step failed) --
            // retry via the normal claim path, which will now see a
            // genuinely free slot. Not batched with the rest of `pending`
            // above since this is expected to be rare (a real read failure),
            // not worth complicating Phase B's already-intricate uring/OMP
            // split for.
            RamClaimResult result = claim_or_wait(engine, layer, w.expert_id, RamClaimCaller::Decode);
            if (result.status == RamClaimStatus::Hit) {
                out[w.k] = result.slot;
            } else if (result.status == RamClaimStatus::Claimed) {
                MoEExpertBlock& victim = *result.slot;
                engine.load_expert_slab(layer, w.expert_id, victim);
                ram_swap_stats[swap_key(layer, w.expert_id)].swap_in++;
                publish_ram_slot(victim);
                out[w.k] = &victim;
            } else {
                // Claimed by yet another in-flight fetch immediately after
                // the retry -- extremely unlikely back-to-back, but wait
                // once more rather than leaving out[w.k] unset.
                out[w.k] = wait_for_ram_slot(engine, layer, w.expert_id);
            }
        }
    }

    // Text heatmap of the HOST RAM expert cache's logical fill state, per
    // layer -- same style/ramp as GpuTopology::log_expert_heatmap, so the
    // two can be read side by side (one line each) to compare which layers
    // are hot in VRAM vs which are hot in host RAM. Only scans the first
    // `slots_per_layer` slots of each layer's array (the REAL active
    // capacity per the RAM-budget fix -- engine.layers_experts[layer] is
    // always physically sized to n_experts for slot-address stability, but
    // slots beyond slots_per_layer are permanently unused, see get_expert()
    // above and CLAUDE.md's "tiered expert cache" section).
    void log_host_cache_heatmap(StreamingEngine& engine) const {
        auto& layers_experts = engine.layers_experts;
        int n_layers = (int)layers_experts.size();
        if (n_layers <= 0) return;
        std::vector<int> resident_count(n_layers, 0);
        std::vector<uint64_t> hit_sum(n_layers, 0);
        for (int l = 0; l < n_layers; ++l) {
            auto& slots = layers_experts[l];
            size_t cap = slots.empty() ? 0 : std::min(slots.size(), (size_t)std::max(1, cap_for_layer(l)));
            for (size_t i = 0; i < cap; ++i) {
                if (slots[i].expert_id < 0) continue;
                resident_count[l]++;
                hit_sum[l] += slots[i].hit_count;
            }
        }
        uint64_t max_hits = 1;
        for (auto h : hit_sum) max_hits = std::max(max_hits, h);
        static const char ramp[] = " .:-=+*#%@";
        constexpr int ramp_len = sizeof(ramp) - 2;
        std::ostringstream os;
        os << "[RAM-HEATMAP]  per-layer resident-slot hit density (0-9, ' '=empty, '@'=hottest):\n[RAM-HEATMAP]  ";
        for (int l = 0; l < n_layers; ++l) {
            int idx = resident_count[l] == 0 ? 0
                : 1 + (int)((double)hit_sum[l] / (double)max_hits * (ramp_len - 1));
            os << ramp[idx];
        }
        os << "\n";
        LOG_DEBUG("%s", os.str().c_str());
    }

    // Shutdown-time summary of host-RAM-tier cache churn: which experts were
    // loaded from disk and evicted again the most. Unlike GpuTopology's
    // report_cache_churn, there's no "pinned" qualifier to check here -- this
    // tier has no pinning mechanism at all, so every entry is inherently
    // unprotected. A high count here represents real, repeated disk-read
    // cost (each swap-in past the first is a re-fetch of data this process
    // already read once before). Call once at shutdown; no locking needed
    // (single decode thread only, see ram_swap_stats' own comment).
    void report_cache_churn(int top_n = 15) const {
        if (ram_swap_stats.empty()) {
            LOG_STREAM_INFO() << "[CACHE-CHURN] Host RAM tier: no swap activity recorded.\n";
            return;
        }
        uint64_t total_in = 0, total_out = 0;
        for (const auto& kv : ram_swap_stats) { total_in += kv.second.swap_in; total_out += kv.second.swap_out; }
        LOG_STREAM_INFO() << "[CACHE-CHURN] Host RAM tier: " << ram_swap_stats.size()
                           << " distinct experts touched, " << total_in << " loads / "
                           << total_out << " evictions this session (no pinning exists for this tier).\n";

        std::vector<std::pair<uint64_t, SwapStats>> sorted(ram_swap_stats.begin(), ram_swap_stats.end());
        std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) {
            return (a.second.swap_in + a.second.swap_out) > (b.second.swap_in + b.second.swap_out);
        });

        LOG_STREAM_INFO() << "[CACHE-CHURN] Top " << std::min((size_t)top_n, sorted.size())
                           << " most-churned host-RAM experts:\n";
        for (int i = 0; i < top_n && i < (int)sorted.size(); ++i) {
            int layer = (int)(sorted[i].first >> 32), expert_id = (int)(sorted[i].first & 0xFFFFFFFFu);
            LOG_STREAM_INFO() << "  layer=" << layer << " expert=" << expert_id
                               << ": loaded=" << sorted[i].second.swap_in
                               << " evicted=" << sorted[i].second.swap_out << "\n";
        }
    }

    // How many host-RAM LRU slots are ACTUALLY holding an expert right
    // now, vs. total capacity (slots_per_layer * n_layers -- the REAL
    // active cap from the RAM-budget fix, not engine.layers_experts[l]'s
    // raw pre-allocated size, which is always n_experts for slot-address
    // stability but mostly unused beyond slots_per_layer -- see
    // get_expert()'s own comment on this). Companion to GpuTopology::
    // report_slot_summary; together they answer "how many MoE experts do
    // we have loaded" for both tiers.
    void report_slot_summary(StreamingEngine& engine) const {
        size_t filled = 0, capacity = 0, bytes_used = 0;
        for (int l = 0; l < (int)engine.layers_experts.size(); ++l) {
            auto& layer_slots = engine.layers_experts[l];
            size_t cap = layer_slots.empty() ? 0
                : std::min(layer_slots.size(), (size_t)std::max(1, cap_for_layer(l)));
            capacity += cap;
            for (size_t i = 0; i < cap; ++i) {
                if (layer_slots[i].expert_id != -1) {
                    ++filled;
                    bytes_used += layer_slots[i].slab_capacity;
                }
            }
        }
        // The bare `slots_per_layer` scalar printed below can be STALE
        // relative to what cap_for_layer() actually used throughout this
        // session, once ram_cache_layer_adaptive is on: calculate_safe_
        // cache_capacity runs a SECOND time per-request inside
        // generate_text (using the real prompt's max_ctx, vs main.cpp's
        // startup call using config.context_size as a placeholder), which
        // overwrites this scalar -- but apply_layer_adaptive_capacity is
        // only ever called once, at startup, right after the FIRST call.
        // slots_per_layer_by_layer therefore keeps reflecting the startup
        // baseline even after a later request's recompute changes the bare
        // scalar (real example that motivated this fix: startup baseline
        // 29/layer, redistributed to 2175 total across 75 layers; a later
        // per-request recompute changed the bare scalar to 40, which used
        // to print here looking like a live, meaningful "cap" even though
        // cap_for_layer() was correctly still using the 29-baseline
        // redistribution the whole time -- capacity/hit-rate numbers were
        // never wrong, only this log line's own trailing text was
        // confusing). Print the real min/max of what's actually in effect
        // instead of the scalar whenever adaptive mode is active.
        LOG_STREAM_INFO() << "[RAM-SLOTS] " << filled << " / " << capacity
                           << " slots filled (" << bytes_used / 1e9 << " GB of experts resident), ";
        if (slots_per_layer_by_layer.empty()) {
            LOG_STREAM_INFO() << slots_per_layer << " slots/layer cap (uniform)\n";
        } else {
            int lo = slots_per_layer, hi = slots_per_layer;
            for (int v : slots_per_layer_by_layer) { if (v > 0) { lo = std::min(lo, v); hi = std::max(hi, v); } }
            LOG_STREAM_INFO() << "[" << lo << ", " << hi << "] slots/layer cap (layer-adaptive active, "
                               << "startup baseline was " << slots_per_layer << "/layer -- see apply_"
                               << "layer_adaptive_capacity's comment if this baseline looks stale "
                               << "vs. the current [RAM] LRU Cache log line)\n";
        }
        report_tier_overlap(engine);
    }

    // How many (layer, expert) pairs are resident in BOTH the RAM LRU tier
    // and the VRAM cache simultaneously, right now. Unlike alternatives (whose
    // VRAM-pinned tier and RAM-LRU tier are architecturally exclusive --
    // CUDA_RELEASE_HOST frees an expert's host copy once it's VRAM-
    // uploaded), MoEStream's tiers are NOT exclusive by design:
    // moe_forward_gguf's per-token dispatch calls
    // RAMSafetyManager::get_experts_batch UNCONDITIONALLY for every routed
    // expert (execution_layer.cpp) to obtain its MoEExpertBlock metadata/
    // host_ptr, REGARDLESS of whether that same expert is also
    // GPU-dispatched via a separate vram_cache->find_expert lookup -- so
    // any actively-used expert ends up resident in RAM as a side effect of
    // simply being routed to, whether or not it's ALSO VRAM-resident. This
    // means the naive "VRAM GB + RAM GB" sum overstates how much DISTINCT
    // expert data is actually cached -- overlapping bytes are paid for
    // twice. Read-only: iterates expert_vram_cache's public fields
    // directly (NOT vram_cache->find_expert, which mutates hit_count/
    // last_used_clock/cache_clock as a side effect -- calling it here
    // would corrupt real eviction-scoring state for a mere report).
    void report_tier_overlap(StreamingEngine& engine) const {
        if (!vram_cache) return;
        std::unordered_set<uint64_t> vram_resident;
        vram_resident.reserve(vram_cache->expert_vram_cache.size());
        for (auto& loc : vram_cache->expert_vram_cache) {
            if (loc.layer_idx == -1) continue;
            vram_resident.insert(((uint64_t)(uint32_t)loc.layer_idx << 32) | (uint32_t)loc.expert_id);
        }
        size_t overlap = 0, overlap_bytes = 0;
        for (int layer = 0; layer < (int)engine.layers_experts.size(); ++layer) {
            auto& layer_slots = engine.layers_experts[layer];
            size_t cap = layer_slots.empty() ? 0
                : std::min(layer_slots.size(), (size_t)std::max(1, cap_for_layer(layer)));
            for (size_t i = 0; i < cap; ++i) {
                if (layer_slots[i].expert_id == -1) continue;
                uint64_t key = ((uint64_t)(uint32_t)layer << 32) | (uint32_t)layer_slots[i].expert_id;
                if (vram_resident.count(key)) {
                    ++overlap;
                    overlap_bytes += layer_slots[i].slab_capacity;
                }
            }
        }
        LOG_STREAM_INFO() << "[TIER-OVERLAP] " << overlap << " expert(s) resident in BOTH RAM and VRAM "
                           << "simultaneously (" << overlap_bytes / 1e9 << " GB double-counted -- "
                           << "paid for in both tiers' totals above, not genuinely distinct capacity)\n";
    }

    // Per-layer VRAM / RAM-only / cold expert-tier breakdown -- the
    // compact, actionable middle ground between the whole-cache aggregate
    // totals (report_slot_summary/report_tier_overlap above) and a
    // literal full per-expert dump (n_experts * n_layers entries -- 19968
    // for a 256-expert/78-layer checkpoint, too much raw data to be
    // useful as text). For each MoE layer: `vram` = currently VRAM-
    // resident (device_ptr set, ready for GPU dispatch with zero I/O);
    // `ram_only` = RAM-resident but NOT also VRAM-resident (would need
    // only an H2D copy, not a disk read, if promoted); `cold` = resident
    // nowhere -- would need a real disk read if routed to right now.
    // Same read-only construction as report_tier_overlap (direct field
    // reads, not find_expert, so this doesn't perturb real hit_count/
    // last_used_clock eviction-scoring state just by reporting on it).
    void report_tier_map(StreamingEngine& engine) const {
        if (!vram_cache) return;
        int n_experts = engine.model_file.architecture_config.n_experts;
        if (n_experts <= 0) return;

        std::unordered_map<int, int> vram_count_by_layer;
        std::unordered_set<uint64_t> vram_resident;
        vram_resident.reserve(vram_cache->expert_vram_cache.size());
        for (auto& loc : vram_cache->expert_vram_cache) {
            if (loc.layer_idx == -1) continue;
            vram_resident.insert(((uint64_t)(uint32_t)loc.layer_idx << 32) | (uint32_t)loc.expert_id);
            vram_count_by_layer[loc.layer_idx]++;
        }

        LOG_STREAM_INFO() << "[EXPERT-TIER-MAP] per-layer expert location (of " << n_experts
                           << " experts/layer): vram / ram-only / cold (would need a disk read)\n";
        for (int layer = 0; layer < (int)engine.layers_experts.size(); ++layer) {
            auto& layer_slots = engine.layers_experts[layer];
            if (layer_slots.empty()) continue;  // not a MoE layer (leading dense)
            size_t cap = std::min(layer_slots.size(), (size_t)std::max(1, cap_for_layer(layer)));

            int vram_n = vram_count_by_layer.count(layer) ? vram_count_by_layer[layer] : 0;
            int ram_only_n = 0;
            for (size_t i = 0; i < cap; ++i) {
                if (layer_slots[i].expert_id == -1) continue;
                uint64_t key = ((uint64_t)(uint32_t)layer << 32) | (uint32_t)layer_slots[i].expert_id;
                if (!vram_resident.count(key)) ++ram_only_n;
            }
            int cold_n = n_experts - vram_n - ram_only_n;
            LOG_STREAM_INFO() << "  layer=" << layer << "  vram=" << vram_n
                               << "  ram_only=" << ram_only_n
                               << "  cold=" << (cold_n > 0 ? cold_n : 0) << "\n";
        }
    }
};