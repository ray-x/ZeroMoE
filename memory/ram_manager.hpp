#pragma once
#include "dense_dispatch.hpp" // is_cpu_affinity_layer (cpu_affinity_layers config)
#include "engine.hpp"
#include "gpu_topology.hpp"
#include "kquant_dequant.hpp" // gpu_type_supported
#include "logger.hpp"
#include "perf_timer.hpp"
#include <algorithm> // std::find (get_experts_batch's claimed_this_batch guard)
#include <cmath>
#include <cuda_runtime.h> // demote_from_vram's D2H salvage copy; already transitively
#include <fcntl.h>
#include <fstream>
#include <functional> // std::greater (apply_layer_adaptive_capacity's per-layer sort)
#include <iostream>
#include <limits>
#include <mutex>
#include <sstream>
#include <string>
#include <unistd.h>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
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
// [SLOW-RAM-WAIT]. This one is deliberately still a FIXED constant, unlike
// async_prefetcher.hpp's [SLOW-PICKUP] threshold which was made self-
// calibrating on 2026-09-02: the two measure different things. [SLOW-PICKUP]
// measures queue-drain latency, whose benign value scales with how many
// hints were pushed ahead of you -- so a constant there rots whenever batch
// size changes. This measures the wait for ONE already-in-flight read to
// publish; that has no queue-depth term, and a single expert read exceeding
// 250ms is anomalous regardless of how busy the rest of the system is. Logged once, not on every poll iteration; waiting continues past
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
#define RAM_TIMED_LOCK(varname)                                                \
  auto _lock_wait_t0_##varname = std::chrono::high_resolution_clock::now();    \
  std::lock_guard<std::mutex> varname(ram_mutex);                              \
  do {                                                                         \
    double _wait_ms = std::chrono::duration<double, std::milli>(               \
                          std::chrono::high_resolution_clock::now() -          \
                          _lock_wait_t0_##varname)                             \
                          .count();                                            \
    PerfCounters::atomic_add(perf().t_ram_mutex_wait, _wait_ms / 1000.0);      \
    if (_wait_ms > SLOW_RAM_MUTEX_WAIT_MS) {                                   \
      LOG_WARN(                                                                \
          "[SLOW-RAM-MUTEX-WAIT] %s blocked %.2fms acquiring ram_mutex\n",     \
          __func__, _wait_ms);                                                 \
    }                                                                          \
  } while (0)

// Outcome of RAMSafetyManager::claim_or_wait -- see that function's own
// comment for the full design.
enum class RamClaimStatus { Hit, Loading, Claimed };
struct RamClaimResult {
  RamClaimStatus status;
  MoEExpertBlock *slot = nullptr; // valid for Hit/Claimed; nullptr for Loading
  size_t slot_idx = 0;            // valid for Hit/Claimed only
};

// Forward-declared only -- the full definition (active_layer_upload_
// workers.hpp) itself includes this header for RAMSafetyManager/
// MoEExpertBlock/GpuTopology, so a full include here would be circular.
// Mirrors LockFreePrefetcher::ram_mgr's own established precedent (see that
// field's comment, async_prefetcher.hpp) for "the pointer lives on whichever
// side CALLS INTO the other, set once post-construction in main.cpp" --
// here execution_layer.cpp's dispatch code calls INTO ActiveLayerUpload
// Workers::try_submit via ram_mgr.upload_workers, so the pointer lives here.
class ActiveLayerUploadWorkers;

class RAMSafetyManager {
public:
  // 2026-08-28: null when active_layer_upload_worker_threads == 0 (the
  // default) -- every call site that reads this MUST null-check before
  // dereferencing. Set once, post-construction, in main.cpp (same pattern
  // as LockFreePrefetcher::ram_mgr) right after the pool itself is
  // constructed. Never owned here -- main.cpp owns the pool's lifetime.
  ActiveLayerUploadWorkers *upload_workers = nullptr;

  int slots_per_layer = 0;
  // Per-layer override for the uniform slots_per_layer cap above --
  // populated by apply_layer_adaptive_capacity (ram_cache_layer_adaptive
  // config flag) when real usage history exists; empty otherwise. Never
  // read directly -- always go through cap_for_layer(layer).
  std::vector<int> slots_per_layer_by_layer;

  // Byte-aware base capacity.  Each entry is bounded using that layer's
  // own maximum expert slab size, so a small-expert layer is not charged the
  // model-wide largest slab.  This remains conservative for an unpacked layer
  // whose experts have mixed sizes: compute_expert_slab_bytes() returns that
  // layer's maximum because RAM slots are grow-only.
  std::vector<int> byte_slots_per_layer_by_layer;
  std::vector<size_t> layer_expert_slab_bytes;
  size_t ram_expert_budget_bytes = 0;

  // Optional global logical-slot limit from EngineConfig.  Unlike
  // ram_pinned_budget_gb (which controls CUDA host-page registration), this
  // limits the number of expert slots resident in the RAM tier across all
  // layers.  0 means no explicit count limit.
  int configured_total_experts = 0;

  // --- Prefill sliding-window capacity (2026-08-30, P1+P2 of the scoped
  // plan; P3 "trim on window exit" is deliberately NOT built yet) --------
  //
  // WHY: calculate_safe_cache_capacity divides the RAM budget by
  // ALL n_layers, because DECODE traverses every layer per token so every
  // layer needs concurrent residency. PREFILL is the opposite shape: it
  // processes layers strictly sequentially (layer 1 for all tokens, then
  // layer 2, ...), so exactly ONE layer is live at a time while the other
  // ~74 sit idle holding budget they cannot use. Real measured cost of
  // that mismatch on a 100-token prefill: the per-layer cap is 76 while a
  // layer's real unique-expert demand is ~212, so the prefetcher evicts
  // its OWN not-yet-consumed work -- 8493/10335 (82.2%) of all prefetch
  // reads were fetched, never read by anyone, and reclaimed under
  // capacity pressure (see n_ram_prefetch_evicted_untouched), which is
  // ~97GB of the ~118GB prefetched being read and thrown away.
  //
  // WHAT: an optional THIRD override tier consulted by cap_for_layer,
  // taking precedence over slots_per_layer_by_layer (ram_cache_layer_
  // adaptive) and the uniform slots_per_layer. Set per layer from that
  // layer's OWN real, already-computed Phase 1 demand (count of
  // expert_freq>0) -- exact, not a prediction: Phase 1 runs its batched
  // router over every token BEFORE Phase 3 dispatches any of them.
  // 0 = no override (the default for every layer, and the permanent
  // value when the feature is off), so an empty/zeroed vector is
  // byte-identical to pre-2026-08-30 behavior.
  //
  // SAFETY WITHOUT P3 (the trim step): raising a cap makes previously-
  // ineligible slots claimable, and nothing lowers memory back down until
  // P3 exists -- so total extra allocation is bounded HERE instead, by a
  // hard extra-slot budget (prefill_window_extra_slots_budget). Once
  // spent, later layers silently fall back to their normal cap rather
  // than growing. That makes P1+P2 safe to actually run and measure
  // (partial coverage, bounded memory) before committing to P3's own,
  // genuinely riskier free()-while-workers-may-be-reading path. This
  // project has a real scar from exactly the failure this bounds --
  // see host_ram_gb's own swap-thrashing history (CLAUDE.md).
  bool prefill_window_enabled = false;
  std::vector<int>
      prefill_window_cap_by_layer; // 0 = no override for that layer
  size_t prefill_window_extra_slots_budget =
      0; // hard ceiling on TOTAL extra slots granted (all layers)
  size_t prefill_window_extra_slots_used =
      0; // running total against the budget above

  uint64_t global_clock = 0;
  GpuTopology *vram_cache =
      nullptr; // GPU expert VRAM cache (optional, set by CudaManager)

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
  // io_uring submission instead of N OS threads each
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
    if (!enable)
      return;
    use_decode_uring = decode_uring_ring.init();
    if (use_decode_uring) {
      // io_uring_workers bounds the
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
      decode_uring_ring.register_iowq_workers(
          (unsigned)std::max(1, io_uring_workers));
      LOG_STREAM_INFO()
          << "[IO-URING] Enabled on the decode thread's own ring -- "
             "batching Phase B's per-layer miss set through io_uring.\n";
    } else {
      LOG_STREAM_WARN()
          << "[IO-URING] io_uring=true requested but the decode-thread "
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

  // "Direction 1" (2026-08-16, follow-up to the Idea 1 speculative-
  // prediction investigation -- see doc/gpu-utilization-vram-analysis.md):
  // route a fraction of the REACTIVE (proven-hot, hit_count >=
  // promote_hit_threshold) promotion stream to GPU1's dual-role slice
  // instead of always the general/GPU0 pool. Deliberately NOT the same
  // mechanism as the reverted speculative predict_gpu1_* feature -- this
  // only ever moves something that's ALREADY proven hot this session,
  // never a guess, so the waste risk that killed the speculative version
  // doesn't apply here the same way. 0 (default) = off, every reactive
  // promotion goes through the general pool exactly as before. N > 0 =
  // every Nth QUALIFYING reactive promotion (this branch's own counter,
  // decode-thread-only so no atomics needed) targets the dual-role GPU
  // specifically instead.
  int reactive_promote_gpu1_every_n = 0;
  uint64_t reactive_promote_counter = 0;

  // Three-tier caching: MOVE instead of COPY on RAM→VRAM promotion
  // When true, immediately frees RAM slot after VRAM promotion starts,
  // eliminating the 7.7GB tier overlap waste.
  bool move_on_promotion = true;

  // Only consulted when move_on_promotion=false -- see EngineConfig's own
  // field comment (engine_config.hpp) for the full reasoning. 0.0
  // (default) means "off", identical to plain COPY behavior.
  double ram_copy_reclaim_threshold = 0.0;

  // Pad Phase B's real cache-miss submission with speculative reads of
  // historically-hot, not-currently-resident experts for the SAME layer
  // -- see EngineConfig::disk_batch_padding's own comment for the full
  // rationale/mechanism/known-limitation. Mirrored here (set from
  // main.cpp) rather than reaching into EngineConfig directly, matching
  // this class's existing pattern for every other config-derived field
  // above (pin_threshold, promote_hit_threshold, move_on_promotion, ...).
  bool disk_batch_padding = false;
  int disk_batch_padding_target = 8;
  // See EngineConfig::disk_batch_padding_evict_dup's own comment for the
  // full safety argument. Mirrored here for the same reason as the two
  // fields above.
  bool disk_batch_padding_evict_dup = false;

  // Per-layer, pre-sorted-descending-by-historical-count expert ID
  // ranking, used as disk_batch_padding's candidate source. Populated
  // ONCE at startup by main.cpp from the SAME .zeromoe_usage-derived
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
  struct SwapStats {
    uint64_t swap_in = 0;
    uint64_t swap_out = 0;
  };
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
  ssize_t find_ram_slot(std::vector<MoEExpertBlock> &layer_slots, size_t cap,
                        int layer, int expert_id) {
    uint64_t key = swap_key(layer, expert_id);
    auto it = ram_expert_index.find(key);
    if (it == ram_expert_index.end()) {
      return -1; // trusted absence -- see this function's own comment
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

  // Like is_ram_resident, but also confirms the slot is FULLY POPULATED
  // (!ram_loading) and returns it -- for a caller that wants to act on
  // the data (e.g. prefill's opportunistic Phase-2 VRAM promotion,
  // execution_layer.cpp's LayerScratch::pending_prefill_promotions),
  // never merely note its presence. Must NEVER block: a still-Loading
  // slot (claimed by a prefetch worker, read in flight) returns nullptr
  // exactly like a genuinely-absent one -- the caller's job is to try
  // again later, not wait here (that's what wait_for_ram_slot is for,
  // and this function is deliberately not that). Reuses find_ram_slot's
  // own trusted-index-hit/self-healing-on-stale-hit logic under the
  // lock, so this is not const the way is_ram_resident is.
  MoEExpertBlock *
  find_ram_slot_if_ready(std::vector<MoEExpertBlock> &layer_slots, size_t cap,
                         int layer, int expert_id) {
    RAM_TIMED_LOCK(lock);
    ssize_t idx = find_ram_slot(layer_slots, cap, layer, expert_id);
    if (idx < 0)
      return nullptr;
    MoEExpertBlock &slot = layer_slots[(size_t)idx];
    if (slot.ram_loading)
      return nullptr; // still in flight -- try again next layer
    return &slot;
  }

  // Same lookup as find_ram_slot_if_ready(), but retain the RAM-tier
  // temporary pin before releasing ram_mutex. This closes the lookup-to-pin
  // window for callers that queue an asynchronous DMA directly from the
  // returned slab_buffer: a prefetch worker may otherwise evict or regrow the
  // slot between the two separate lock acquisitions. Capture the pointer and
  // capacity under the same lock as the pin; a regrow may defer freeing the
  // old allocation while this caller still owns its raw pointer. The caller
  // owns one matching ram_unpin() and must keep it until the DMA has completed.
  MoEExpertBlock *find_ram_slot_if_ready_pinned(
      std::vector<MoEExpertBlock> &layer_slots, size_t cap, int layer,
      int expert_id, const uint8_t **slab_out, size_t *capacity_out) {
    RAM_TIMED_LOCK(lock);
    ssize_t idx = find_ram_slot(layer_slots, cap, layer, expert_id);
    if (idx < 0)
      return nullptr;
    MoEExpertBlock &slot = layer_slots[(size_t)idx];
    if (slot.ram_loading || !slot.slab_buffer || !slot.slab_pinned)
      return nullptr;
    if (slab_out) *slab_out = slot.slab_buffer;
    if (capacity_out) *capacity_out = slot.slab_capacity;
    ++slot.ram_temp_pin_count;
    return &slot;
  }

  // Same atomic lookup-and-hold as find_ram_slot_if_ready_pinned(), but for
  // pageable RAM slabs too. The caller must keep the temporary hold until it
  // has finished reading the source: for a staged path that is through the
  // CPU memcpy, while a direct asynchronous H2D path must keep it through
  // its completion event. A prefetch worker must not evict or regrow the slab
  // during either interval.
  MoEExpertBlock *find_ram_slot_if_ready_held(
      std::vector<MoEExpertBlock> &layer_slots, size_t cap, int layer,
      int expert_id, const uint8_t **slab_out, size_t *capacity_out) {
    RAM_TIMED_LOCK(lock);
    ssize_t idx = find_ram_slot(layer_slots, cap, layer, expert_id);
    if (idx < 0)
      return nullptr;
    MoEExpertBlock &slot = layer_slots[(size_t)idx];
    if (slot.ram_loading || !slot.slab_buffer)
      return nullptr;
    if (slab_out) *slab_out = slot.slab_buffer;
    if (capacity_out) *capacity_out = slot.slab_capacity;
    ++slot.ram_temp_pin_count;
    return &slot;
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
  void revert_padding_claim(MoEExpertBlock &slot, int layer, int expert_id) {
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
  // Fraction [0,1] of this layer's REAL, enforced RAM-tier slot budget
  // (cap_for_layer(layer), the same cap every other RAM-tier scan
  // respects -- not layer_slots.size(), which is pre-sized to n_experts
  // for slot-address stability but mostly unused) currently occupied
  // (expert_id != -1). O(cap) scan -- only called from
  // finish_sync_promotion, itself only invoked once per CONFIRMED sync
  // VRAM promotion (not once per token/hit), so this is a bounded,
  // infrequent cost, not a hot-path one. Deliberately a direct scan
  // rather than an incrementally-maintained counter: several RAM-tier
  // code paths overwrite a slot's identity directly (evict A, claim B)
  // without an explicit intermediate "become empty" step, which would
  // make a naive increment/decrement counter silently drift wrong over a
  // long session -- a scan is always correct, and this call site's low
  // frequency means the cost of re-deriving it is negligible next to that
  // risk.
  double layer_ram_occupancy(std::vector<MoEExpertBlock> &layer_slots,
                             int layer) {
    size_t cap = layer_slots.empty()
                     ? 0
                     : std::min(layer_slots.size(),
                                (size_t)std::max(1, cap_for_layer(layer)));
    if (cap == 0)
      return 0.0;
    size_t filled = 0;
    for (size_t i = 0; i < cap; i++)
      if (layer_slots[i].expert_id != -1)
        filled++;
    return (double)filled / (double)cap;
  }

  // layer_slots_for_occupancy: optional, non-owning. When non-null (and
  // ram_copy_reclaim_threshold > 0), a COPY-mode promotion whose layer has
  // crossed the configured occupancy threshold is reclaimed immediately
  // (same field-clearing as MOVE, not a new mechanism) instead of being
  // left occupied-but-redundant to depend entirely on reactive eviction
  // pressure that may never arrive for a lightly-trafficked layer. Only
  // ever passed non-null from call sites independently verified safe for
  // this (see CLAUDE.md for the full correctness trace: freeing here only
  // clears expert_id/slab_capacity/etc -- exactly what MOVE already does
  // at these same sites today -- never slab_buffer itself or the
  // gate_proj/up_proj/down_proj host pointers into it, so a same-call CPU
  // fallback reading those afterward, a real reachable path on GPU-issue
  // failure, stays valid).
  void finish_sync_promotion_locked(
      MoEExpertBlock &blk, int layer, int expert_id,
      std::vector<MoEExpertBlock> *layer_slots_for_occupancy = nullptr) {
    // A graph-backed slot aliases an immutable shared arena.  It must retain
    // its identity and capacity even when the same expert is also promoted to
    // VRAM; treating it as MOVE would invalidate the arena metadata while the
    // CPUOffload worker may still use the DDR view.
    bool should_free = !blk.slab_from_shared_arena &&
                       move_on_promotion && blk.ram_temp_pin_count <= 0;
    if (!should_free && !move_on_promotion &&
        ram_copy_reclaim_threshold > 0.0 && blk.ram_temp_pin_count <= 0 &&
        layer_slots_for_occupancy) {
      if (layer_ram_occupancy(*layer_slots_for_occupancy, layer) >=
          ram_copy_reclaim_threshold) {
        should_free = true;
        perf().n_copy_reclaimed_threshold.fetch_add(1,
                                                    std::memory_order_relaxed);
      }
    }
    if (should_free) {
      blk.expert_id = -1;
      ram_expert_index.erase(swap_key(layer, expert_id));
      blk.last_used_clock = 0;
      blk.hit_count = 0;
      blk.also_in_vram = false;
      // MOVE retires the expert identity, not this slot's allocation. Keep
      // its real capacity so the next load reuses the buffer instead of
      // freeing/registering the same-sized slab again.
      blk.ram_loading = false;
      perf().n_move_on_promotion.fetch_add(1, std::memory_order_relaxed);
    } else {
      blk.also_in_vram = true;
      perf().n_copy_on_promotion.fetch_add(1, std::memory_order_relaxed);
    }
  }

public:
  void finish_sync_promotion(
      MoEExpertBlock &blk, int layer, int expert_id,
      std::vector<MoEExpertBlock> *layer_slots_for_occupancy = nullptr) {
    RAM_TIMED_LOCK(lock);
    finish_sync_promotion_locked(blk, layer, expert_id,
                                 layer_slots_for_occupancy);
  }

  void finish_sync_promotion(StreamingEngine &engine, int layer,
                             int expert_id) {
    if (layer < 0 || layer >= (int)engine.layers_experts.size())
      return;
    RAM_TIMED_LOCK(lock);
    auto &layer_slots = engine.layers_experts[layer];
    size_t cap = layer_slots.empty()
                     ? 0
                     : std::min(layer_slots.size(),
                                (size_t)std::max(1, cap_for_layer(layer)));
    ssize_t idx = find_ram_slot(layer_slots, cap, layer, expert_id);
    if (idx >= 0)
      finish_sync_promotion_locked(layer_slots[idx], layer, expert_id,
                                   &layer_slots);
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
  void apply_promotion_bookkeeping(MoEExpertBlock &blk, int layer,
                                   int expert_id, ExpertLocation *promoted) {
    RAM_TIMED_LOCK(lock);
    if (move_on_promotion && !blk.slab_from_shared_arena) {
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
  void ram_pin(MoEExpertBlock &blk) {
    RAM_TIMED_LOCK(lock);
    ++blk.ram_temp_pin_count;
  }
  void ram_unpin(MoEExpertBlock &blk) {
    RAM_TIMED_LOCK(lock);
    if (blk.ram_temp_pin_count > 0)
      --blk.ram_temp_pin_count;
  }

  // Mark an asynchronously uploaded RAM slab as actually consumed by the
  // VRAM path.  This is a small bookkeeping helper, but the field is part of
  // the RAM-tier state protected by ram_mutex; callers must not write it
  // directly after an unlocked lookup while prefetch workers are active.
  void mark_consumed_by_vram_upload(MoEExpertBlock &blk) {
    RAM_TIMED_LOCK(lock);
    blk.consumed_by_vram_upload = true;
  }

  // 2026-08-28: transfers a RAM slot's protection from ram_temp_pin_count
  // (a synchronous-window-only guard, meant for a caller that will itself
  // block until the in-flight copy completes) to pending_copy_event (self-
  // healing, cudaEventQuery-polled -- see MoEExpertBlock::pending_copy_
  // event's own comment) in one locked step, for a caller that issued an
  // async H2D copy FROM this slot's slab_buffer but has NO synchronous
  // confirmation point of its own (decode_async_prewarm's own active-layer
  // batch, when EngineConfig::decode_async_prewarm_nonblocking is set --
  // see that flag's own comment for why removing the blocking wait on the
  // destination side is safe; this is what makes the SOURCE side equally
  // safe without one). Mirrors apply_promotion_bookkeeping's own COPY-
  // branch pending_copy_event assignment, generalized to a non-promotion
  // caller. Both writes happen under the same lock acquisition so no
  // window ever exists where NEITHER guard protects this slot from a
  // concurrent victim-selection scan (every scan already OR's copy_still_
  // pending() and ram_temp_pin_count>0 together as independent exclusion
  // reasons -- see MoEExpertBlock::pending_copy_event's own comment).
  void ram_mark_pending_copy_and_unpin(MoEExpertBlock &blk, cudaEvent_t event) {
    RAM_TIMED_LOCK(lock);
    blk.pending_copy_event = event;
    if (blk.ram_temp_pin_count > 0)
      --blk.ram_temp_pin_count;
  }

  // EngineConfig::prefill_retention_weight, set once at startup (main.cpp).
  // 0 (default) = OFF: eviction_score below is then byte-identical to its
  // pre-2026-08-31 form, so this whole mechanism is a strict no-op unless
  // explicitly enabled. Function-local-static storage, mirroring the
  // established cpu_only_mode()/is_cpu_affinity_layer() global-switch
  // pattern in dense_dispatch.hpp rather than threading a config reference
  // through eviction_score's many call sites.
  static uint64_t &prefill_retention_weight_ref() {
    static uint64_t w = 0;
    return w;
  }
  static void set_prefill_retention_weight(uint64_t w) { prefill_retention_weight_ref() = w; }
  static int &prefill_retention_top_n_ref()      { static int v = 0; return v; }
  static int &prefill_retention_min_demand_ref() { static int v = 1; return v; }
  static double &prefill_retention_flat_cut_ref()  { static double v = 0.636; return v; }
  static double &prefill_retention_peaky_cut_ref() { static double v = 0.693; return v; }
  static void set_prefill_retention_shape(int top_n, int min_demand, double flat_cut, double peaky_cut) {
    prefill_retention_top_n_ref() = top_n;
    prefill_retention_min_demand_ref() = min_demand;
    prefill_retention_flat_cut_ref() = flat_cut;
    prefill_retention_peaky_cut_ref() = peaky_cut;
  }

  // EngineConfig::ram_grace_period_tokens/ram_grace_period_bonus
  // (2026-09-04). eviction_score is static (no `this`), so -- same reason
  // prefill_retention_weight_ref() above is a function-local static
  // rather than a member field -- current_decode_token_idx_ref() mirrors
  // that pattern instead of threading a parameter through eviction_
  // score's many call sites. Updated once per DECODE token (S==1) from
  // layers_forward_gguf (execution_layer.cpp), right alongside the
  // existing ram_mgr.in_prefill = (S > 1) assignment; left untouched
  // during prefill (S>1) calls, which has a gentle, deliberate side
  // effect: anything loaded during prefill has load_token_idx==0 (or
  // whatever the LAST decode step was, if this is a later turn), so it
  // is naturally grace-protected only for the first ram_grace_period_
  // tokens decode steps of the request that follows -- not a permanent
  // bonus for the whole prefill phase.
  static uint64_t &ram_grace_period_tokens_ref() { static uint64_t v = 0; return v; }
  static uint64_t &ram_grace_period_bonus_ref()  { static uint64_t v = 0; return v; }
  static uint64_t &current_decode_token_idx_ref() { static uint64_t v = 0; return v; }
  static void set_ram_grace_period(uint64_t tokens, uint64_t bonus) {
    ram_grace_period_tokens_ref() = tokens;
    ram_grace_period_bonus_ref() = bonus;
  }
  static void set_current_decode_token_idx(uint64_t idx) {
    current_decode_token_idx_ref() = idx;
  }

  static uint64_t eviction_score(const MoEExpertBlock &b) {
    // prefill->decode retention handoff: an expert THIS REQUEST's prefill
    // actually routed to is far more likely to be wanted again during the
    // decode that follows (measured: prefill-ranked retention captures 94.6%
    // of what a decode-oracle would -- see MoEExpertBlock::prefill_demand).
    // Recency alone cannot see that: a prefill expert used heavily early in
    // the pass looks "old" by last_used_clock and gets evicted just before
    // decode starts asking for it. This adds a demand-proportional term so
    // proven-needed experts outrank merely-recently-touched ones.
    //
    // Additive (not a hard pin) on purpose: it biases the ordering, it does
    // not make a slot un-evictable -- an unbounded/hard priority here could
    // starve the reactive path the way this project's own over-pinning
    // regressions already have (see pinned_vram_capacity's tuning history).
    uint64_t score = b.last_used_clock + (uint64_t)b.hit_count * FREQ_BONUS_TICKS +
                     (uint64_t)b.prefill_demand * prefill_retention_weight_ref();
    // Grace-period bonus: last_used_clock above is a raw operation-tick
    // counter, not a token counter -- a burst of several loads within ONE
    // decode token (this model's multiple speculative predictors can each
    // trigger a RAM load per layer transition) can make a slot look
    // "stale" relative to slots loaded microseconds later in the SAME
    // token, before it ever got a real chance to be reused. This term is
    // token-counted instead, so it is immune to that intra-token burst
    // effect. See ram_grace_period_tokens's own comment (engine_config.hpp)
    // for why this is additive, not a hard exemption.
    uint64_t grace_tokens = ram_grace_period_tokens_ref();
    if (grace_tokens > 0) {
      uint64_t now = current_decode_token_idx_ref();
      uint64_t age = (now >= b.load_token_idx) ? (now - b.load_token_idx) : 0;
      if (age < grace_tokens) score += ram_grace_period_bonus_ref();
    }
    return score;
  }

  // Halve hit_count across every layer's cache slots. Cheap since total
  // slot count (slots_per_layer * n_layers) is small and this runs rarely.
  void decay_hit_counts(StreamingEngine &engine) {
    for (auto &layer_slots : engine.layers_experts)
      for (auto &slot : layer_slots)
        slot.hit_count /= 2;
    hits_since_decay = 0;
  }

  double get_mem_available_gb() {
#ifdef __APPLE__
    // macOS: use host_statistics64 to get free + inactive pages
    mach_port_t host = mach_host_self();
    vm_statistics64_data_t vm_stat;
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    if (host_statistics64(host, HOST_VM_INFO64, (host_info64_t)&vm_stat,
                          &count) == KERN_SUCCESS) {
      uint64_t page_size = sysconf(_SC_PAGESIZE);
      uint64_t free_bytes =
          ((uint64_t)vm_stat.free_count + vm_stat.inactive_count) * page_size;
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

  void calculate_safe_cache_capacity(StreamingEngine &engine, int max_ctx,
                                     double user_ram_gb = 0.0) {
    auto &c = engine.model_file.architecture_config;
    // Expert tiers follow the allocated routed-layer topology.  GLM MTP adds
    // one streamed MoE block beyond c.n_layers; target KV sizing below must
    // remain based on c.n_layers, but expert byte/count budgets must include
    // every entry that can actually be claimed.
    const int cache_layers = std::max(0, (int)engine.layers_experts.size());
    double requested = user_ram_gb > 0 ? user_ram_gb : configured_ram_gb;
    double ram_gb = requested > 0 ? requested : get_mem_available_gb() * 0.88;
    if (ram_gb < 4.0)
      ram_gb = 8.0;

    double resident_bytes = 0;
    for (const auto &[name, tensor] : engine.resident_layers)
      resident_bytes += tensor.meta.size_bytes;

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
    //
    // 2026-08-29: takes the TRUE MAXIMUM per-expert slab size across
    // ALL layers now, not just the FIRST layer with any expert tensors
    // -- found via a real multi-turn memory investigation (see
    // CLAUDE.md) that a single sampled layer's size is not a safe
    // stand-in for every layer's. The legacy slots_per_layer scalar still
    // uses this maximum as a conservative fallback, while cap_for_layer()
    // now uses each layer's own byte-aware cap. Every real
    // RAM-tier allocation (ensure_slab_capacity, engine.hpp) is
    // grow-only -- a slot never shrinks back down once it has held a
    // larger expert, so over a session's LRU churn a slot can ratchet
    // up to the largest expert it has EVER been asked to hold. Sizing
    // slots_per_layer off anything smaller than the model-wide max
    // means that ratchet can push REAL total RAM-tier usage past
    // host_ram_gb's intended cap without this function's own math ever
    // noticing -- confirmed on real hardware: a 108GB-configured run
    // reached 111.9GB RSS after just one prompt. Using the true max
    // here instead makes each layer's cap * its own maximum slab size a
    // genuine, provable upper bound on total possible usage even under
    // full worst-case ratcheting. The fix is a startup-only calculation
    // change, not a new runtime
    // enforcement mechanism, so it carries none of the correctness
    // risk a live/reactive budget check over the hot allocation path
    // would.
    // Keep the model-wide maximum for the shared workspace estimate and the
    // legacy scalar diagnostic, but also retain every layer's own maximum.
    // The latter is the important part: charging every layer as if it had
    // the largest slab anywhere in the model can waste a substantial part of
    // the DDR budget on models whose MoE layers have different tensor shapes.
    layer_expert_slab_bytes.assign((size_t)cache_layers, 0);
    double expert_bytes = 0;
    int active_moe_layers = 0;
    double full_moe_bytes = 0;
    for (int l = 0; l < cache_layers; ++l) {
      size_t layer_bytes = engine.compute_expert_slab_bytes(l);
      layer_expert_slab_bytes[(size_t)l] = layer_bytes;
      if (layer_bytes == 0)
        continue;
      ++active_moe_layers;
      expert_bytes = std::max(expert_bytes, (double)layer_bytes);
      full_moe_bytes += (double)c.n_experts * (double)layer_bytes;
    }

    double pc_b = 2.5e9;
    double ws_b = 64.0 * expert_bytes;
    double kv_b =
        (double)(c.n_layers + 1) * max_ctx * (c.kv_lora + c.qk_rope) * 4.0;
    double kvb_b = (double)max_ctx * c.n_heads * (c.qk_nope + c.v_head) * 4.0;

    double slack = 1.2e9 + pc_b + ws_b + kv_b + kvb_b;
    double available_for_experts = (ram_gb * 1e9) - resident_bytes - slack;

    // This scalar remains for older diagnostics and for code paths that do
    // not have a per-layer entry.  Correct the old dense-layer overcount here
    // as well: a leading dense layer has no RAM expert slots to budget.
    int capmax = (available_for_experts > 0 && expert_bytes > 0)
                     ? static_cast<int>(available_for_experts /
                                        (active_moe_layers * expert_bytes))
                     : 1;
    slots_per_layer = std::max(1, std::min(capmax, c.n_experts));

    ram_expert_budget_bytes = available_for_experts > 0
                                  ? (size_t)available_for_experts
                                  : 0;
    byte_slots_per_layer_by_layer.assign((size_t)cache_layers, 0);

    // Compute a common safe residency fraction from BOTH constraints:
    //   * the available byte budget after dense/KV/workspace headroom; and
    //   * the optional requested total number of RAM-tier expert slots.
    // Applying the fraction per layer preserves coverage across the model,
    // while using each layer's own slab size prevents small layers from being
    // charged at the largest slab size found elsewhere.
    const double full_moe_slots =
        (double)active_moe_layers * (double)std::max(0, c.n_experts);
    double count_fraction = 1.0;
    if (configured_total_experts > 0 && full_moe_slots > 0.0)
      count_fraction = std::min(
          1.0, (double)configured_total_experts / full_moe_slots);
    double byte_fraction =
        (full_moe_bytes > 0.0 && available_for_experts > 0.0)
            ? std::min(1.0, available_for_experts / full_moe_bytes)
            : 0.0;
    double residency_fraction = std::min(count_fraction, byte_fraction);

    size_t byte_slots_total = 0;
    size_t byte_budget_used = 0;
    if (residency_fraction > 0.0 && c.n_experts > 0) {
      for (int l = 0; l < cache_layers; ++l) {
        size_t layer_bytes = layer_expert_slab_bytes[(size_t)l];
        if (layer_bytes == 0)
          continue;
        int cap = std::min(
            c.n_experts,
            (int)std::floor((double)c.n_experts * residency_fraction));
        if (cap < 1 && ram_expert_budget_bytes >= layer_bytes)
          cap = 1;

        // Round-down allocation is normally already safe.  Keep this guard
        // explicit because size_t arithmetic here is the final invariant
        // protecting a grow-only RAM slab from exceeding the byte budget.
        size_t remaining = byte_budget_used < ram_expert_budget_bytes
                                ? ram_expert_budget_bytes - byte_budget_used
                                : 0;
        if (layer_bytes > 0)
          cap = std::min(cap, (int)std::min<size_t>(
                                   (size_t)c.n_experts,
                                   remaining / layer_bytes));
        byte_slots_per_layer_by_layer[(size_t)l] = std::max(0, cap);
        byte_slots_total += (size_t)std::max(0, cap);
        byte_budget_used += (size_t)std::max(0, cap) * layer_bytes;
      }
    }

    LOG_STREAM_INFO() << "[RAM] Budget: " << ram_gb
                      << " GB | Resident: " << resident_bytes / 1e9
                      << " GB | Slack: " << slack / 1e9 << " GB\n";
    LOG_STREAM_INFO() << "[RAM] Byte-aware MoE capacity: "
                      << byte_slots_total << " expert slots / "
                      << byte_slots_total * 1.0 / std::max(1, active_moe_layers)
                      << " average slots per MoE layer ("
                      << byte_budget_used / 1e9 << " GB budgeted expert slabs, "
                      << "full model " << full_moe_bytes / 1e9 << " GB";
    if (configured_total_experts > 0)
      LOG_STREAM_INFO() << ", requested total " << configured_total_experts;
    LOG_STREAM_INFO() << ")\n";
    LOG_STREAM_INFO() << "[RAM] LRU Cache scalar fallback: "
                      << slots_per_layer << " experts per layer.\n";
  }

  // ram_cache_layer_adaptive: redistributes the already byte-bounded RAM-tier
  // capacity across layers. The pool is denominated in real slab bytes (and
  // also respects configured_total_experts), so moving capacity toward a
  // smaller-expert layer cannot accidentally exceed the DDR budget.
  // It distributes capacity proportional to each layer's real historical
  // need, instead of the uniform per-layer split every other RAM-tier code
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
  // are left at their byte-aware base cap, contributing
  // exactly their normal safe share to the total either way, outside
  // the redistribution pool. First-ever run (no usage file, or an
  // empty/all-zero one) leaves slots_per_layer_by_layer empty entirely
  // -- cap_for_layer() then uses the byte-aware base caps, identical to
  // this feature's disabled behavior, by construction (safe default: no
  // history yet means no redistribution decision can be trusted yet).
  void apply_layer_adaptive_capacity(const std::string &usage_file,
                                     int total_layers, int n_experts,
                                     double coverage_target = 0.8,
                                     int min_slots_per_layer = 8) {
    slots_per_layer_by_layer.clear();
    if (usage_file.empty() || slots_per_layer <= 0 || total_layers <= 0)
      return;
    auto [entries, maxc] = StreamingEngine::load_usage(usage_file);
    (void)maxc;
    if (entries.empty())
      return;

    std::unordered_map<int, std::vector<uint64_t>> by_layer;
    for (auto &e : entries) {
      if (e.layer < 0 || e.layer >= total_layers)
        continue;
      by_layer[e.layer].push_back(e.count);
    }
    if (by_layer.empty())
      return;

    std::vector<double> need(total_layers, 0.0);
    double need_sum = 0.0;
    for (auto &[l, v] : by_layer) {
      std::sort(v.begin(), v.end(), std::greater<uint64_t>());
      uint64_t total = 0;
      for (auto c : v)
        total += c;
      if (total == 0)
        continue;
      uint64_t cum = 0;
      double rank = 0;
      for (auto c : v) {
        cum += c;
        rank += 1.0;
        if ((double)cum >= coverage_target * (double)total)
          break;
      }
      need[l] = rank;
      need_sum += rank;
    }
    if (need_sum <= 0.0)
      return;

    int n_with_data = (int)by_layer.size();
    auto layer_base_cap = [&](int layer) {
      if (layer >= 0 &&
          layer < (int)byte_slots_per_layer_by_layer.size() &&
          layer < (int)layer_expert_slab_bytes.size() &&
          layer_expert_slab_bytes[(size_t)layer] > 0)
        return byte_slots_per_layer_by_layer[(size_t)layer];
      return slots_per_layer;
    };
    auto layer_bytes = [&](int layer) -> size_t {
      if (layer >= 0 && layer < (int)layer_expert_slab_bytes.size() &&
          layer_expert_slab_bytes[(size_t)layer] > 0)
        return layer_expert_slab_bytes[(size_t)layer];
      // This fallback is only for callers that construct a manager without
      // first running calculate_safe_cache_capacity.  Normal engine startup
      // always has real per-layer slab sizes here.
      return 1;
    };

    slots_per_layer_by_layer.assign(total_layers, slots_per_layer);
    for (int l = 0; l < total_layers; ++l)
      slots_per_layer_by_layer[(size_t)l] = layer_base_cap(l);

    size_t pool_slots = 0;
    double pool_bytes = 0.0;
    for (const auto &[l, _] : by_layer) {
      int cap = std::max(0, std::min(n_experts, layer_base_cap(l)));
      pool_slots += (size_t)cap;
      pool_bytes += (double)cap * (double)layer_bytes(l);
    }
    if (pool_slots == 0 || pool_bytes <= 0.0)
      return;

    std::vector<double> clamped(total_layers, 0.0);
    double clamped_sum_bytes = 0.0;
    for (int l = 0; l < total_layers; ++l) {
      if (need[l] <= 0.0)
        continue;
      double raw = need[l] / need_sum * pool_bytes / (double)layer_bytes(l);
      clamped[l] = std::min((double)layer_base_cap(l),
                            std::min((double)n_experts,
                            std::max((double)min_slots_per_layer, raw)));
      clamped_sum_bytes += clamped[l] * (double)layer_bytes(l);
    }
    if (clamped_sum_bytes > pool_bytes && clamped_sum_bytes > 0.0) {
      double scale = pool_bytes / clamped_sum_bytes;
      for (int l = 0; l < total_layers; ++l) {
        if (need[l] <= 0.0)
          continue;
        clamped[l] *=
            scale; // may go below min_slots_per_layer -- byte safety wins
      }
    }

    // Rebalancing toward smaller layers can increase the number of slots
    // while staying within the same byte pool.  Preserve the explicit global
    // expert-count limit as a second hard constraint.
    double clamped_sum_slots = 0.0;
    for (int l = 0; l < total_layers; ++l)
      if (need[l] > 0.0)
        clamped_sum_slots += clamped[l];
    if (clamped_sum_slots > (double)pool_slots && clamped_sum_slots > 0.0) {
      double scale = (double)pool_slots / clamped_sum_slots;
      for (int l = 0; l < total_layers; ++l) {
        if (need[l] <= 0.0)
          continue;
        clamped[l] *= scale;
      }
    }

    int lo = -1, hi = -1;
    size_t allocated_slots = 0;
    size_t allocated_bytes = 0;
    for (int l = 0; l < total_layers; ++l) {
      if (need[l] <= 0.0)
        continue;
      int v = std::max(1, (int)std::floor(clamped[l]));
      v = std::min(v, std::min(n_experts, layer_base_cap(l)));
      slots_per_layer_by_layer[l] = v;
      allocated_slots += (size_t)v;
      allocated_bytes += (size_t)v * layer_bytes(l);
      if (lo == -1 || v < lo)
        lo = v;
      if (hi == -1 || v > hi)
        hi = v;
    }

    // Floors/rounding can add a few units back after the two floating-point
    // rescaling passes.  Remove only from used layers, and only above one
    // slot, until both hard pools are satisfied.
    while ((allocated_slots > pool_slots || allocated_bytes > (size_t)pool_bytes) &&
           allocated_slots > 0) {
      int victim = -1;
      size_t victim_bytes = 0;
      for (int l = 0; l < total_layers; ++l) {
        if (need[l] <= 0.0 || slots_per_layer_by_layer[(size_t)l] <= 1)
          continue;
        size_t b = layer_bytes(l);
        if (victim < 0 || b > victim_bytes) {
          victim = l;
          victim_bytes = b;
        }
      }
      if (victim < 0)
        break;
      --slots_per_layer_by_layer[(size_t)victim];
      --allocated_slots;
      allocated_bytes -= victim_bytes;
    }

    LOG_STREAM_INFO() << "[RAM] Layer-adaptive capacity: redistributed "
                      << allocated_slots << " slots / " << allocated_bytes / 1e9
                      << " GB across " << n_with_data
                      << " layer(s) with usage history by need "
                      << "(coverage target " << (coverage_target * 100.0)
                      << "%), range [" << lo << ", " << hi
                      << "] slots/layer (byte-aware; uniform fallback was "
                      << slots_per_layer << "/layer)\n";
  }

  // Real per-layer active cap: byte-aware base cap, optionally reduced by
  // slots_per_layer_by_layer when adaptive history is available. Every
  // RAM-tier code path goes through this instead of reading a scalar cap
  // directly. Precedence (2026-08-30): prefill window > adaptive history >
  // byte-aware base > uniform fallback. The optional prefill window is a
  // separately budgeted transient expansion; normal decode/cache capacity
  // always uses the byte/count-bounded base. See prefill_window_cap_by_layer's
  // own comment for the full rationale.
  //
  // Deliberately still lock-free/const: this is read on the hot path from
  // BOTH the main thread and prefetch worker threads (via claim_or_wait),
  // exactly as before. The window vector is sized ONCE at startup and
  // thereafter only has individual int elements overwritten under
  // ram_mutex by set_prefill_window_cap -- never resized, so a concurrent
  // reader can never see a dangling/reallocated buffer. A reader racing
  // an update sees either the old or new int, both of which are valid
  // caps (a torn read is not possible for an aligned int on x86-64), and
  // every consumer already clamps with std::min(layer_slots.size(), cap)
  // -- so a briefly-stale cap is safe by construction, not just by luck.
  // 2026-08-31: prefill->decode retention handoff. Called once per layer at
  // the END of that layer's prefill pass, when expert_freq is final and the
  // layer's experts are still resident. Stamps each RAM slot with how often
  // THIS request's prefill actually routed to it, which eviction_score then
  // uses to decide what survives into decode.
  //
  // Deliberately a separate pass rather than stamping inside the dispatch
  // loop: dispatch only sees experts it actually dispatches, while a slot
  // can also be resident via prefetch/padding without having been dispatched
  // yet -- and those are exactly the ones most at risk of being evicted
  // before decode gets to them. Cheap: one bounded scan per layer, no I/O,
  // no allocation.
  // (see call site in execution_layer.cpp for placement rationale)
  void stamp_prefill_demand(StreamingEngine &engine, int layer,
                            const std::vector<int> &expert_freq) {
    if (prefill_retention_weight_ref() == 0) return;  // feature off -> no-op
    if (layer < 0 || layer >= (int)engine.layers_experts.size()) return;

    // --- Tiered retention width, from THIS layer's own concentration ---
    //
    // A fixed "keep top N per layer" is the wrong shape: per-layer routing
    // concentration varies a lot (measured on real data, top-20 share ran
    // 46.6% on the flattest layer vs 78.4% on the peakiest). Spending the
    // same N slots on both over-serves the flat layer -- whose individual
    // experts are each less likely to be reused -- and under-serves the
    // peaky one, where a few experts carry most of the traffic.
    //
    // Concentration is computed here from expert_freq alone, so it needs NO
    // history and works on the very first run -- which is what makes this
    // answerable at layer 3 without knowing anything about layers 4..77.
    //
    // Measured payoff at equal N: tiered retention delivered 9.72 units of
    // decode traffic per RAM slot vs 8.17 for a flat N (~19% better use of
    // the same memory).
    int n_experts_seen = 0;
    long total_picks = 0;
    for (size_t e = 0; e < expert_freq.size(); ++e) {
      if (expert_freq[e] > 0) { ++n_experts_seen; total_picks += expert_freq[e]; }
    }
    if (n_experts_seen == 0 || total_picks == 0) return;

    int base_n = prefill_retention_top_n_ref();
    int keep_n = n_experts_seen;   // 0 = "no cap", keep everything stamped
    if (base_n > 0) {
      // top-20 share == the same metric the offline analysis banded on.
      std::vector<int> f;
      f.reserve(n_experts_seen);
      for (size_t e = 0; e < expert_freq.size(); ++e)
        if (expert_freq[e] > 0) f.push_back(expert_freq[e]);
      std::partial_sort(f.begin(), f.begin() + std::min<size_t>(20, f.size()), f.end(),
                        std::greater<int>());
      long top20 = 0;
      for (size_t i = 0; i < std::min<size_t>(20, f.size()); ++i) top20 += f[i];
      double share = (double)top20 / (double)total_picks;
      // Terciles measured offline on real data; a layer below the flat cut
      // gets 1/3 of the width, a mid layer 2/3, a peaky layer the full N.
      if (share < prefill_retention_flat_cut_ref())        keep_n = std::max(1, base_n / 3);
      else if (share < prefill_retention_peaky_cut_ref())  keep_n = std::max(1, (2 * base_n) / 3);
      else                                                 keep_n = base_n;
    }

    // Rank this layer's experts by real prefill demand; only the top keep_n
    // (and only those clearing the minimum-demand threshold) get stamped.
    // A threshold is comparable ACROSS layers in a way a count is not, so it
    // is what keeps a flat layer from hoarding slots on near-one-off picks.
    const int min_demand = prefill_retention_min_demand_ref();
    std::vector<std::pair<int,int>> ranked;   // (freq, expert_id)
    ranked.reserve(n_experts_seen);
    for (size_t e = 0; e < expert_freq.size(); ++e)
      if (expert_freq[e] >= min_demand && expert_freq[e] > 0)
        ranked.push_back({expert_freq[e], (int)e});
    if (ranked.empty()) return;
    if ((int)ranked.size() > keep_n)
      std::partial_sort(ranked.begin(), ranked.begin() + keep_n, ranked.end(),
                        std::greater<std::pair<int,int>>());
    else
      std::sort(ranked.begin(), ranked.end(), std::greater<std::pair<int,int>>());

    std::unordered_map<int,int> keep;   // expert_id -> demand
    keep.reserve((size_t)std::min<int>(keep_n, (int)ranked.size()) * 2);
    for (int i = 0; i < (int)ranked.size() && i < keep_n; ++i)
      keep[ranked[i].second] = ranked[i].first;

    RAM_TIMED_LOCK(lock);
    auto &layer_slots = engine.layers_experts[layer];
    size_t cap = layer_slots.empty()
                     ? 0
                     : std::min(layer_slots.size(),
                                (size_t)std::max(1, cap_for_layer(layer)));
    uint64_t stamped = 0;
    for (size_t i = 0; i < cap; ++i) {
      MoEExpertBlock &slot = layer_slots[i];
      auto it = keep.find(slot.expert_id);
      if (it == keep.end()) continue;
      slot.prefill_demand = (uint32_t)it->second;
      ++stamped;
    }
    perf().n_prefill_retention_stamped.fetch_add(stamped, std::memory_order_relaxed);
  }

  // Inter-request decay. Called once when a request finishes, so a later
  // prompt's real demand can outrank an earlier prompt's stale stamps.
  //
  // Deliberately a DECAY, not a bulk release: releasing and rebuilding would
  // risk dumping experts the next prompt actually needs, and forces an
  // awkward "how many slots should I leave free" decision. Halving instead
  // lets old demand fade while anything reinforced by successive prompts
  // survives, so the steady state self-regulates. Mirrors the existing
  // decay_hit_counts pattern rather than inventing a second convention.
  void decay_prefill_demand(StreamingEngine &engine) {
    if (prefill_retention_weight_ref() == 0) return;
    RAM_TIMED_LOCK(lock);
    uint64_t n = 0;
    for (auto &layer_slots : engine.layers_experts)
      for (auto &slot : layer_slots)
        if (slot.prefill_demand) { slot.prefill_demand /= 2; ++n; }
    perf().n_prefill_retention_decayed.fetch_add(n, std::memory_order_relaxed);
  }


  int base_cap_for_layer(int layer) const {
    int cap = slots_per_layer;
    if (layer >= 0 &&
        layer < (int)byte_slots_per_layer_by_layer.size() &&
        layer < (int)layer_expert_slab_bytes.size() &&
        layer_expert_slab_bytes[(size_t)layer] > 0) {
      cap = byte_slots_per_layer_by_layer[(size_t)layer];
    }
    if (layer >= 0 && layer < (int)slots_per_layer_by_layer.size() &&
        slots_per_layer_by_layer[(size_t)layer] > 0)
      cap = std::min(cap, slots_per_layer_by_layer[(size_t)layer]);
    return cap;
  }

  int cap_for_layer(int layer) const {
    if (layer >= 0 && layer < (int)prefill_window_cap_by_layer.size() &&
        prefill_window_cap_by_layer[layer] > 0)
      return prefill_window_cap_by_layer[layer];
    return base_cap_for_layer(layer);
  }

  // P1+P2 setup: called once at startup (main.cpp) after
  // calculate_safe_cache_capacity has settled slots_per_layer. Sizes the
  // per-layer override vector and converts the configured extra-memory
  // allowance into a real slot count using the model's OWN largest expert
  // slab (the same model-wide max calculate_safe_cache_capacity uses, so
  // the two budgets are denominated identically and can't drift apart).
  void init_prefill_window(StreamingEngine &engine, bool enabled,
                           double extra_gb) {
    auto &c = engine.model_file.architecture_config;
    const int cache_layers = std::max(0, (int)engine.layers_experts.size());
    prefill_window_enabled = enabled;
    prefill_window_cap_by_layer.assign((size_t)cache_layers, 0);
    prefill_window_extra_slots_used = 0;
    prefill_window_extra_slots_budget = 0;
    if (!enabled || extra_gb <= 0.0)
      return;
    double expert_bytes = 0;
    for (int l = 0; l < cache_layers; ++l)
      expert_bytes =
          std::max(expert_bytes, (double)engine.compute_expert_slab_bytes(l));
    if (expert_bytes <= 0)
      return;
    prefill_window_extra_slots_budget =
        (size_t)((extra_gb * 1e9) / expert_bytes);
    LOG_STREAM_INFO()
        << "[RAM] Prefill window: ON -- base cap " << slots_per_layer
        << "/layer, may grow to a layer's own real Phase-1 demand while it is "
           "the "
           "active prefill layer, funded by a hard extra budget of "
        << extra_gb << " GB = " << prefill_window_extra_slots_budget
        << " extra slot(s) in flight at once (P3 trim-on-exit IS built: a "
           "layer's "
           "extra slots are freed and RETURNED to this budget the moment the "
           "window "
           "moves off it, so at most ONE layer holds extra slots at any "
           "instant)\n";
  }

  // P3 (2026-08-30): free layer `layer`'s EXTRA slots (everything above
  // its base cap) and return them to the global extra-slot budget.
  // Caller MUST already hold ram_mutex -- this is the _locked half, split
  // out precisely because ram_mutex is a plain, NON-recursive std::mutex
  // (see RAM_TIMED_LOCK's own comment): calling the public setter's
  // lock-taking path from inside it would deadlock instantly.
  //
  // SAFETY: this is the one genuinely dangerous part of the whole prefill
  // window design -- it free()s slabs that a prefetch worker thread or an
  // in-flight async H2D DMA may still be reading. It therefore reuses the
  // EXACT same three guards every existing RAM-tier victim-selection site
  // already uses, verbatim and in the same order, rather than inventing
  // new safety reasoning for this call site:
  //   - ram_loading         : a worker has claimed this slot and its read
  //                           is in progress (would be a torn/partial read)
  //   - ram_temp_pin_count>0: someone holds a raw pointer into slab_buffer
  //                           across a call boundary (Pass A->B, an
  //                           ActiveLayerUploadWorkers item, a CPU dequant)
  //   - copy_still_pending(): an async H2D copy is still sourcing FROM
  //                           this buffer (pending_copy_event unsignaled)
  // A slot failing any guard is SKIPPED, and the cap is then held high
  // enough to keep it reachable -- deliberately leaving a little memory
  // un-reclaimed rather than orphaning a live, still-referenced slot.
  // That asymmetry is the whole point: over-trimming is a use-after-free
  // (this project already has a real SIGSEGV from exactly that class of
  // bug, 2026-08-30), under-trimming is just a few MB held a bit longer.
  size_t trim_prefill_window_locked(StreamingEngine &engine, int layer) {
    if (layer < 0 || layer >= (int)prefill_window_cap_by_layer.size())
      return 0;
    int window_cap = prefill_window_cap_by_layer[layer];
    if (window_cap <= 0)
      return 0; // no window granted here
    int base = base_cap_for_layer(layer);
    if (window_cap <= base) {
      prefill_window_cap_by_layer[layer] = 0;
      return 0;
    }

    auto &layer_slots = engine.layers_experts[layer];
    size_t hi = std::min(layer_slots.size(), (size_t)window_cap);

    // 2026-08-31: trim COLDEST-FIRST, not by slot index.
    //
    // Real measurement motivating this (current .zeromoe_usage vs
    // .zeromoe_usage_prefill): prefill's and decode's expert sets are
    // only PARTIALLY correlated -- 47.9% overlap, r=0.655 on shared
    // entries -- but crucially **59.0% of decode's real traffic is
    // covered by experts prefill already touched**. So the experts this
    // trim leaves behind are not arbitrary leftovers: they are a warm,
    // materially useful cache for the decode phase that follows.
    //
    // The first version of this loop walked [base, window_cap) in INDEX
    // order and freed whatever it found, which could discard a hot
    // expert while keeping a cold one purely by slot position -- pure
    // luck as to whether decode's warm set survived. Instead, rank the
    // trimmable slots by the SAME eviction_score() every other RAM-tier
    // victim scan uses (recency + frequency), free the coldest, and
    // RETAIN the hottest. Same number of slots freed, same memory
    // returned to the budget -- strictly better choice of which ones.
    //
    // NOTE this deliberately reuses eviction_score rather than inventing
    // a prefill-specific ranking: it is the codebase's one established
    // definition of "least valuable to keep", and a second, divergent
    // notion of hotness in the same tier would be a real maintenance
    // trap.
    struct TrimCand {
      size_t idx;
      uint64_t score;
    };
    static thread_local std::vector<TrimCand> trim_cands;
    trim_cands.clear();
    int lowest_safe_cap_scan = base;
    for (size_t i = (size_t)base; i < hi; ++i) {
      MoEExpertBlock &slot = layer_slots[i];
      // Same three guards, checked FIRST and unchanged -- an in-use
      // slot is never a trim candidate regardless of how cold it is.
      if (slot.ram_loading || slot.ram_temp_pin_count > 0 ||
          slot.copy_still_pending()) {
        lowest_safe_cap_scan = (int)i + 1; // must stay reachable
        continue;
      }
      // An empty slot (never populated) is free to reclaim and carries
      // no decode value -- score 0 sorts it to the front.
      trim_cands.push_back(
          {i, slot.expert_id == -1 ? 0 : eviction_score(slot)});
    }
    // Coldest first.
    std::sort(
        trim_cands.begin(), trim_cands.end(),
        [](const TrimCand &a, const TrimCand &b) { return a.score < b.score; });

    // How many must actually go: everything above the base cap. Freeing
    // the coldest `need_to_free` leaves the hottest ones resident, and
    // the surviving hot slots are then kept reachable by holding the cap
    // above the highest such index (same rule as an un-freeable slot).
    size_t trimmable = trim_cands.size();
    size_t need_to_free = trimmable; // reclaim all extra capacity
    size_t freed = 0, skipped = (size_t)(hi - (size_t)base) - trimmable;
    for (size_t n = 0; n < trim_cands.size(); ++n) {
      size_t i = trim_cands[n].idx;
      MoEExpertBlock &slot = layer_slots[i];
      if (n >= need_to_free) { // retained (hottest) -- keep reachable
        if ((int)i + 1 > lowest_safe_cap_scan)
          lowest_safe_cap_scan = (int)i + 1;
        perf().n_prefill_window_slots_retained_hot.fetch_add(
            1, std::memory_order_relaxed);
        continue;
      }
      if (slot.expert_id != -1) {
        ram_expert_index.erase(swap_key(layer, slot.expert_id));
        slot.expert_id = -1;
      }
      if (slot.slab_buffer) {
        // Mirrors ensure_slab_capacity's own free path exactly --
        // unregistering BEFORE free() is mandatory (freeing still-
        // page-locked memory is UB), and the pinned-bytes accounting
        // must be given back or ram_pinned_budget_gb leaks downward.
        // 2026-09-01: was a direct free(), the SAME blind spot the
        // original upload bug had. The three guards are checked when
        // choosing a trim candidate, but a reference those guards cannot
        // see (a queued ActiveLayerUploadWorkItem's host_data, captured at
        // claim time) would still be left dangling -- permanently, since
        // nothing reallocates here. Real signature: ONE host_src failing
        // repeatedly against three DIFFERENT dst. release_slab() re-checks
        // the guards and defers the free while any reader may hold it.
        engine.release_slab(slot);
      }
      slot.also_in_vram = false;
      slot.loaded_by_prefetch = false;
      slot.loaded_by_lookahead = false;
      slot.consumed_by_vram_upload = false;
      slot.prefill_demand = 0;  // 2026-08-31: same reset lifecycle
      slot.from_padding = false;
      slot.hit_count = 0;
      slot.last_used_clock = 0;
      ++freed;
    }
    int lowest_safe_cap = lowest_safe_cap_scan;

    int new_cap = lowest_safe_cap;
    size_t returned = (size_t)(window_cap - new_cap);
    prefill_window_cap_by_layer[layer] = (new_cap <= base) ? 0 : new_cap;
    prefill_window_extra_slots_used =
        prefill_window_extra_slots_used > returned
            ? prefill_window_extra_slots_used - returned
            : 0;
    perf().n_prefill_window_trimmed_layers.fetch_add(1,
                                                     std::memory_order_relaxed);
    perf().n_prefill_window_slots_freed.fetch_add(freed,
                                                  std::memory_order_relaxed);
    perf().n_prefill_window_slots_skipped_inuse.fetch_add(
        skipped, std::memory_order_relaxed);
    return returned;
  }

  // P2 (+P3): grant layer `layer` a cap equal to its own real Phase-1
  // demand, bounded by (a) the model's expert count, (b) whatever remains
  // of the global extra-slot budget.
  //
  // P3 makes this a genuine SLIDING window: before granting, every OTHER
  // layer's window is trimmed and its slots returned to the budget, so at
  // most ONE layer holds extra slots at any instant. That is what turns
  // the budget from "total extra memory ever granted" (P1+P2, which
  // capped coverage at ~9 of 75 layers) into "extra memory in flight at
  // once" -- the same 12GB now covers every layer, one at a time, instead
  // of being permanently consumed by the first few.
  //
  // Never LOWERS an existing cap for the layer being granted (a revisited
  // layer keeps what it has); lowering happens only via the trim path
  // above, which is guard-checked.
  int set_prefill_window_cap(StreamingEngine &engine, int layer,
                             int desired_cap) {
    if (!prefill_window_enabled)
      return cap_for_layer(layer);
    if (layer < 0 || layer >= (int)prefill_window_cap_by_layer.size())
      return cap_for_layer(layer);
    int n_experts = engine.model_file.architecture_config.n_experts;
    RAM_TIMED_LOCK(lock);

    // --- P3: slide the window off layers that are genuinely DONE ---
    // Keeps a 3-layer band [layer-1, layer+1] rather than trimming
    // everything but the active layer. Both neighbours can legitimately
    // still have in-flight work referencing their RAM slabs at this
    // instant, and NOT every such reference is covered by the three
    // eviction guards:
    //   layer+1: the speculative pilot / prefetch hints for the NEXT
    //            layer are pushed during THIS layer and land in l+1's
    //            slots -- trimming them both wastes the fetch and races
    //            the worker that is about to publish into it.
    //   layer-1: ActiveLayerUploadWorkers items submitted while l-1 was
    //            active can still be sitting in the queue, holding a
    //            host_data pointer CAPTURED AT CLAIM TIME. Freeing that
    //            slab makes the worker's own cudaMemcpyAsync fail with
    //            "invalid argument" -- which is exactly what a first
    //            trim-everything-else version of this loop produced on
    //            real hardware (6774 errors, concentrated on the last
    //            layers, with 0 slots reported skipped -- i.e. the
    //            guards genuinely did not see these references).
    // Trimming only l-2 and older keeps the memory bound essentially
    // unchanged (3 layers' extra slots in flight instead of 1) while
    // staying clear of every in-flight reference this path can't see.
    for (int l = 0; l < (int)prefill_window_cap_by_layer.size(); ++l) {
      if (l >= layer - 1 && l <= layer + 1)
        continue;
      if (prefill_window_cap_by_layer[l] <= 0)
        continue;
      trim_prefill_window_locked(engine, l);
    }

    int base = base_cap_for_layer(layer);
    int current = prefill_window_cap_by_layer[layer] > 0
                      ? prefill_window_cap_by_layer[layer]
                      : base;
    int want = std::min(desired_cap, n_experts);
    if (want <= current) {
      perf().n_prefill_window_not_needed.fetch_add(1,
                                                   std::memory_order_relaxed);
      return current;
    }
    size_t extra_wanted = (size_t)(want - current);
    size_t remaining =
        prefill_window_extra_slots_budget > prefill_window_extra_slots_used
            ? prefill_window_extra_slots_budget -
                  prefill_window_extra_slots_used
            : 0;
    if (remaining == 0) {
      perf().n_prefill_window_denied_budget.fetch_add(
          1, std::memory_order_relaxed);
      return current;
    }
    size_t granted_extra = std::min(extra_wanted, remaining);
    int granted_cap = current + (int)granted_extra;
    prefill_window_cap_by_layer[layer] = granted_cap;
    prefill_window_extra_slots_used += granted_extra;
    perf().n_prefill_window_granted.fetch_add(1, std::memory_order_relaxed);
    if (layer < 128)
      perf().prefill_window_layer_granted[layer].store(
          1, std::memory_order_relaxed);
    perf().n_prefill_window_extra_slots_granted.fetch_add(
        granted_extra, std::memory_order_relaxed);
    if (granted_extra < extra_wanted) {
      perf().n_prefill_window_partial.fetch_add(1, std::memory_order_relaxed);
    }
    return granted_cap;
  }

  // P3: release EVERY layer's window (end of prefill, or shutdown). Public
  // wrapper -- takes the lock itself, so must NOT be called from anywhere
  // already holding ram_mutex.
  void clear_all_prefill_windows(StreamingEngine &engine) {
    if (!prefill_window_enabled)
      return;
    RAM_TIMED_LOCK(lock);
    for (int l = 0; l < (int)prefill_window_cap_by_layer.size(); ++l) {
      if (prefill_window_cap_by_layer[l] > 0)
        trim_prefill_window_locked(engine, l);
    }
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
  void demote_from_vram(StreamingEngine &engine, int layer, int expert_id,
                        const uint8_t *device_ptr, size_t bytes, int gpu_id) {
    auto &layer_slots = engine.layers_experts[layer];
    if (layer_slots.empty()) {
      perf().n_vram_evict_demote_skipped.fetch_add(1,
                                                   std::memory_order_relaxed);
      return;
    }
    size_t cap =
        std::min(layer_slots.size(), (size_t)std::max(1, cap_for_layer(layer)));

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
      perf().n_vram_evict_demote_skipped.fetch_add(1,
                                                   std::memory_order_relaxed);
      return;
    }

    // Packed format or a validated unpacked_expert_format model both
    // have a well-defined per-(layer,expert) geometry via
    // expert_geometry_for -- an UNvalidated legacy per-expert model
    // (unpacked_expert_format=false but per-expert tensor names present
    // anyway) has no single geometry to reconstruct host_ptr/meta
    // offsets from, so demotion is a silent (counted, not logged) no-op
    // for that case, matching how the prefetch fast path already degrades.
    // expert_geometry_for, not expert_layer_geometry -- expert_id is
    // real here (2026-09 caller audit).
    size_t gb, ub, db;
    const GGUFTensor *gM, *uM, *dM;
    if (!engine.expert_geometry_for(layer, expert_id, gb, ub, db, gM, uM, dM)) {
      perf().n_vram_evict_demote_skipped.fetch_add(1,
                                                   std::memory_order_relaxed);
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
      LOG_WARN(
          "[MOE-RAM] demote_from_vram: geometry_bytes exceeds vram slot size "
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
    // 2026-08-17: pilot-confirmed candidates excluded the same way
    // copy_still_pending/ram_temp_pin_count/ram_loading already are --
    // see GpuTopology::evict_lru's identical exclusion for the safety
    // argument (the confirmed set is tiny and single-layer-scoped, so
    // this can't starve the scan). Counted separately below via
    // n_pilot_protected_eviction_skips_ram, distinct from the
    // was_pilot_confirmed_resident check AFTER victim selection further
    // down (which now measures the rare residual case where every
    // OTHER slot was also ineligible).
    auto pilot_protected = [&](size_t i) {
      bool hit = was_pilot_confirmed_resident(layer, layer_slots[i].expert_id);
      if (hit)
        perf().n_pilot_protected_eviction_skips_ram.fetch_add(
            1, std::memory_order_relaxed);
      return hit;
    };
    size_t victim_idx = 0;
    uint64_t victim_score =
        (layer_slots[0].copy_still_pending() ||
         layer_slots[0].ram_temp_pin_count > 0 || layer_slots[0].ram_loading ||
         pilot_protected(0))
            ? UINT64_MAX
            : (layer_slots[0].also_in_vram ? 0
                                           : eviction_score(layer_slots[0]));
    for (size_t i = 1; i < cap; ++i) {
      uint64_t score =
          (layer_slots[i].copy_still_pending() ||
           layer_slots[i].ram_temp_pin_count > 0 ||
           layer_slots[i].ram_loading || pilot_protected(i))
              ? UINT64_MAX
              : (layer_slots[i].also_in_vram ? 0
                                             : eviction_score(layer_slots[i]));
      if (score < victim_score) {
        victim_score = score;
        victim_idx = i;
      }
    }
    MoEExpertBlock &victim = layer_slots[victim_idx];
    int displaced = victim.expert_id;
    // Measurement of the RESIDUAL race -- see GpuTopology::evict_lru's
    // identical check for the full explanation of what this counts and
    // why. This D2H-salvage path is a RAM-tier victim selection exactly
    // like claim_or_wait's own Claimed-branch victim selection, just
    // triggered by an incoming VRAM eviction instead of a RAM-tier
    // miss. Should now be near-zero (see the exclusion above) -- only
    // fires if literally every slot in this layer was ineligible.
    if (displaced != -1 && was_pilot_confirmed_resident(layer, displaced)) {
      perf().n_evicted_was_pilot_confirmed_ram.fetch_add(
          1, std::memory_order_relaxed);
    }
    if (displaced != -1 && victim.loaded_by_prefetch && victim.hit_count == 0) {
      perf().n_ram_prefetch_loaded_evicted_unused.fetch_add(
          1, std::memory_order_relaxed);
      // 2026-08-30 split -- see MoEExpertBlock::consumed_by_vram_upload
      // (engine.hpp) for why the aggregate above is ambiguous on its own.
      if (victim.consumed_by_vram_upload) {
        perf().n_ram_prefetch_evicted_consumed_by_vram.fetch_add(
            1, std::memory_order_relaxed);
      } else {
        perf().n_ram_prefetch_evicted_untouched.fetch_add(
            1, std::memory_order_relaxed);
        if (layer >= 0 && layer < PerfCounters::kMaxTrackedLayersForMiss)
          perf().n_ram_prefetch_evicted_untouched_by_layer[layer].fetch_add(
              1, std::memory_order_relaxed);
      }
    }
    if (displaced != -1 && victim.loaded_by_lookahead &&
        victim.hit_count == 0) {
      perf().n_prefill_lookahead_loaded_evicted_unused.fetch_add(
          1, std::memory_order_relaxed);
    }
    // Erase the outgoing identity's index entry now -- this slot is
    // committed to being overwritten by `expert_id` from here on,
    // whether the D2H copy below ultimately succeeds or fails (see the
    // failure branch further down, which also leaves the slot with a
    // NEW, empty identity rather than the old `displaced` one).
    if (displaced != -1)
      ram_expert_index.erase(swap_key(layer, displaced));

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
    //
    // Issued on the vault GPU's own dedicated copy_stream (via
    // cudaMemcpyAsync + an immediate cudaStreamSynchronize) instead of
    // the plain cudaMemcpy this used before, which implicitly runs on
    // the legacy default stream (stream 0). A real microbenchmark (see
    // CLAUDE.md) confirmed stream-0 ops force a full wait for
    // EVERYTHING queued on every other stream on the device (e.g.
    // compute_stream's in-flight expert FFN kernels) before they can
    // even start, and block every other stream until they finish --
    // cost scaled linearly with queue depth in that benchmark (0.87ms
    // isolated -> 15.28ms at a 60-kernel queue, vs a dedicated stream
    // staying flat at ~0.9ms regardless of queue depth). This function
    // is decode-thread-only (see its own header comment) and this
    // change does NOT alter that: cudaStreamSynchronize still blocks
    // the caller until the copy is genuinely done before any of the
    // "mark this RAM slot Ready" bookkeeping below runs -- identical
    // synchronous-completion contract to the old cudaMemcpy, just off
    // the contended default stream. Falls back to the old plain
    // cudaMemcpy if vram_cache/the device's copy_stream is ever
    // unavailable (should not happen in practice -- this function is
    // only ever reached via a GpuTopology callback, so the SAME
    // GpuTopology instance is necessarily vram_cache -- but matches
    // this codebase's own no-silent-degrade discipline elsewhere).
    cudaSetDevice(gpu_id);
    engine.ensure_slab_capacity(victim, total);
    cudaStream_t demote_stream = (vram_cache && gpu_id >= 0 &&
                                  (size_t)gpu_id < vram_cache->devices.size())
                                     ? vram_cache->devices[gpu_id].copy_stream
                                     : nullptr;
    // Diagnostic ONLY, added to directly test WHY the stream-0 fix
    // showed no measurable real-hardware win despite a confirmed-real
    // mechanism in isolation (see CLAUDE.md): query whether the
    // device's OTHER streams (compute_stream, copy_stream) actually
    // have genuinely pending/in-flight work at the exact moment this
    // D2H fires. cudaStreamQuery is non-blocking -- cudaSuccess means
    // idle/nothing pending, cudaErrorNotReady means real in-flight
    // work exists. If both come back idle almost always, that directly
    // confirms there was rarely anything to avoid waiting for at this
    // specific call site, explaining the null result without
    // contradicting the (separately, definitively proven) underlying
    // CUDA mechanism.
    if (vram_cache && gpu_id >= 0 &&
        (size_t)gpu_id < vram_cache->devices.size()) {
      cudaError_t cq =
          cudaStreamQuery(vram_cache->devices[gpu_id].compute_stream);
      cudaError_t pq = cudaStreamQuery(vram_cache->devices[gpu_id].copy_stream);
      if (cq == cudaErrorNotReady)
        perf().n_demote_saw_compute_stream_busy.fetch_add(
            1, std::memory_order_relaxed);
      else
        perf().n_demote_saw_compute_stream_idle.fetch_add(
            1, std::memory_order_relaxed);
      if (pq == cudaErrorNotReady)
        perf().n_demote_saw_copy_stream_busy.fetch_add(
            1, std::memory_order_relaxed);
      else
        perf().n_demote_saw_copy_stream_idle.fetch_add(
            1, std::memory_order_relaxed);
      // cudaStreamQuery can itself leave a spurious error in the
      // per-thread last-error slot on some driver versions when it
      // returns cudaErrorNotReady -- clear it so it's never
      // misattributed to the real cudaMemcpyAsync/cudaStreamSynchronize
      // calls a few lines below.
      cudaGetLastError();
    }
    PERF_START(vram_demote_copy);
    auto demote_call_t0 = std::chrono::high_resolution_clock::now();
    cudaError_t err;
    if (demote_stream) {
      err = cudaMemcpyAsync(victim.slab_buffer, device_ptr, total,
                            cudaMemcpyDeviceToHost, demote_stream);
      if (err == cudaSuccess)
        err = cudaStreamSynchronize(demote_stream);
    } else {
      perf().n_vram_demote_stream_unavailable.fetch_add(
          1, std::memory_order_relaxed);
      err = cudaMemcpy(victim.slab_buffer, device_ptr, total,
                       cudaMemcpyDeviceToHost);
    }
    PERF_END(vram_demote_copy, t_vram_demote_copy);
    // Per-GPU/per-size breakdown -- direct test of the two leading
    // candidates for the residual gap left after ruling out cross-stream
    // contention and accounting for cudaDeviceScheduleBlockingSync's own
    // wake-up tax (see CLAUDE.md, and PerfCounters::t_vram_demote_by_gpu's
    // own comment). Recorded regardless of success/failure below so a
    // real failure doesn't silently skew the average toward only the
    // fast/successful calls.
    if (gpu_id >= 0 && gpu_id < PerfCounters::PERF_MAX_GPU_SLOTS) {
      double call_ms =
          std::chrono::duration<double>(
              std::chrono::high_resolution_clock::now() - demote_call_t0)
              .count();
      PerfCounters::atomic_add(perf().t_vram_demote_by_gpu[gpu_id], call_ms);
      perf().bytes_vram_demote_by_gpu[gpu_id].fetch_add(
          (int64_t)total, std::memory_order_relaxed);
      perf().n_vram_demote_by_gpu[gpu_id].fetch_add(1,
                                                    std::memory_order_relaxed);
    }
    if (err != cudaSuccess) {
      LOG_WARN("[MOE-RAM] demote_from_vram: cudaMemcpy D2H failed layer=%d "
               "expert=%d (%s) -- slot left empty\n",
               layer, expert_id, cudaGetErrorString(err));
      // Leave the slot genuinely empty rather than half-populated --
      // this project's standing discipline (see MoEExpertBlock's own
      // also_in_vram comment, the cudaMalloc-failure-checking fix
      // elsewhere in this codebase): a failed operation must never
      // leave state that looks valid but isn't.
      victim.expert_id = -1;
      victim.also_in_vram = false;
      victim.loaded_by_prefetch = false;
      victim.loaded_by_lookahead = false;
      victim.consumed_by_vram_upload =
          false; // 2026-08-30: same reset lifecycle (engine.hpp)
      victim.prefill_demand = 0;  // 2026-08-31: same reset lifecycle
      victim.ram_loading = false;
      perf().n_vram_evict_demote_failed.fetch_add(1, std::memory_order_relaxed);
      return;
    }

    if (displaced != -1)
      ram_swap_stats[swap_key(layer, displaced)].swap_out++;
    engine.populate_packed_slab(victim, expert_id, gb, ub, db, *gM, *uM, *dM);
    victim.also_in_vram =
        false; // it just left VRAM -- this RAM copy is the only one now
    victim.loaded_by_prefetch =
        false; // fresh identity, synchronously salvaged, not prefetch-loaded
    victim.loaded_by_lookahead = false; // ditto
    victim.consumed_by_vram_upload =
        false; // 2026-08-30: same reset lifecycle (engine.hpp)
    victim.prefill_demand = 0;  // 2026-08-31: same reset lifecycle
    victim.ram_loading =
        false; // synchronous salvage, already fully populated above
    victim.last_used_clock = ++global_clock;
    victim.hit_count = 0;
    victim.load_token_idx = current_decode_token_idx_ref();
    ram_swap_stats[swap_key(layer, expert_id)].swap_in++;
    ram_expert_index[swap_key(layer, expert_id)] = victim_idx;
    perf().n_vram_evict_demoted.fetch_add(1, std::memory_order_relaxed);

    LOG_DEBUG("[MOE-RAM] DEMOTE layer=%d expert=%d slot=%zu displaced=%d (%.2f "
              "MB salvaged from VRAM via D2H)\n",
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
  // Shared "upload this already-populated RAM slot into VRAM" mechanics,
  // used by BOTH promotion callers: claim_or_wait's Hit branch (reactive,
  // gated on a live re-hit crossing promote_hit_threshold) and
  // promote_ram_to_vram below (speculative, pilot-prediction-driven).
  // Caller MUST already hold ram_mutex (RAM_TIMED_LOCK) and MUST be on
  // the decode thread -- this touches GpuTopology::upload_expert_async
  // and vram_cache->pin_expert, both of which take cache_mutex, and per
  // this class's own standing cache_mutex -> ram_mutex nesting rule, a
  // PrefetchWorker thread reaching this would be the exact cross-thread
  // AB-BA deadlock claim_or_wait's own CRITICAL comment (above) already
  // warns about -- see that comment for the full incident this must
  // never reproduce. Both current callers satisfy this by construction
  // (claim_or_wait's Hit branch already gates its promotion block on
  // `who == RamClaimCaller::MainThread` implicitly via being main-thread-
  // only logic in that branch; promote_ram_to_vram is only ever invoked
  // from execution_layer.cpp's pilot-prediction loop, itself main-thread
  // code, during either prefill or decode -- see RamClaimCaller's own
  // comment, engine.hpp, for why "MainThread" is not phase-specific) --
  // a future caller MUST preserve that, not just copy the call.
  //
  // allow_move (default true, preserving the pre-existing reactive-path
  // behavior verbatim): when false, ALWAYS copy (leave the RAM slot
  // intact) regardless of the global move_on_promotion setting.
  // promote_ram_to_vram (speculative, pilot-prediction-driven) passes
  // false -- unlike the reactive path, which only ever fires after a
  // PROVEN re-hit (hit_count >= promote_hit_threshold), this path fires
  // on a prediction with real-hardware-measured accuracy as low as
  // ~44% at its best rank (see CLAUDE.md's "ram-only" per-rank metric).
  // A wrong bet under MOVE semantics is actively harmful, not merely
  // wasteful: it deletes the RAM copy at the same moment it takes a
  // scarce VRAM slot, so if that speculative VRAM slot itself gets
  // evicted before use (likely, given VRAM is now under more pressure),
  // the expert ends up resident NOWHERE -- a cheap RAM-tier hit that
  // would have resolved the NEXT real access for free instead becomes a
  // full disk re-read. Confirmed on real hardware: enabling top_k=3
  // speculative promotion with the (at the time) shared MOVE semantics
  // dropped cache-hit rate from 88% to 69% and grew disk time 20x
  // (0.07s -> 1.44s) on an otherwise-identical run -- this parameter is
  // the fix for that regression, not a defensive guess.
  // target_gpu_id: -1 (default, every pre-existing caller) = let
  // upload_expert_async pick from the general pool as before. >= 0
  // forces this promotion onto ONE specific device -- currently only
  // passed by promote_ram_to_vram_gpu1 below. Threaded straight through
  // to upload_expert_async's own same-named parameter (GpuTopology).
  void do_promote_ram_slot_to_vram(int layer, int expert_id,
                                   MoEExpertBlock &slot, bool allow_move = true,
                                   int target_gpu_id = -1) {
    if (!vram_cache)
      return;
    // Defense-in-depth: checked HERE (not just by each caller) so a
    // future promotion caller can't forget this the way the
    // cpu_affinity_layers CLAUDE.md entry warns is easy to do -- this
    // is effectively a 6th VRAM-entry point beyond the five that grep
    // sweep already found and gated; centralizing the check in the one
    // function that actually calls upload_expert_async/pin_expert
    // means there is nowhere else a cpu_affinity layer's expert could
    // leak into VRAM through THIS path, regardless of caller.
    if (is_cpu_affinity_layer(layer))
      return;
    bool gpu_capable = slot.slab_buffer &&
                       gpu_type_supported(slot.gate_proj.meta.type) &&
                       gpu_type_supported(slot.up_proj.meta.type) &&
                       gpu_type_supported(slot.down_proj.meta.type);
    if (!gpu_capable)
      return;
    ExpertLocation *resident = vram_cache->find_expert(layer, expert_id);
    if (!resident) {
      // real_total, not slab_capacity -- see promote_ram_to_active_layer's
      // own comment for the full incident (same bug, this being the
      // dominant/main-cache promotion path makes it the higher-impact of
      // the two: slab_capacity is this SLOT's grow-only allocation size,
      // which legitimately outlives whatever smaller expert currently
      // occupies it after an LRU reuse, so uploading it copies real data
      // followed by a previous occupant's stale tail bytes into a VRAM
      // location downstream dispatch treats as this expert's own tight
      // gate|up|down layout.
      const size_t real_total = slot.gate_proj.meta.size_bytes +
                                 slot.up_proj.meta.size_bytes +
                                 slot.down_proj.meta.size_bytes;
      LOG_DEBUG("[MOE-VRAM]  PROMOTE layer=%d expert=%d hit_count=%u -> async "
                "H2D upload (%.2f MB)\n",
                layer, expert_id, slot.hit_count, real_total / 1e6);
      ExpertLocation *promoted =
          vram_cache->upload_expert_async(layer, expert_id, slot.slab_buffer,
                                          real_total, target_gpu_id,
                                          slot.slab_pinned);
      // 2026-08-30 diagnostic: this slot's bytes were genuinely
      // consumed (uploaded to VRAM), even though the RAM tier's own
      // hit_count may never record a later use -- see MoEExpertBlock::
      // consumed_by_vram_upload (engine.hpp).
      if (promoted)
        slot.consumed_by_vram_upload = true;
      if (promoted && slot.loaded_by_prefetch) {
        perf().n_prefetch_staged_later_promoted.fetch_add(
            1, std::memory_order_relaxed);
      }
      // Shared graph-arena slots are immutable DDR sources for the CPU graph;
      // keep them in place even when the global policy is MOVE.
      if (move_on_promotion && allow_move && !slot.slab_from_shared_arena) {
        slot.expert_id = -1;
        ram_expert_index.erase(swap_key(layer, expert_id));
        slot.last_used_clock = 0;
        slot.hit_count = 0;
        slot.also_in_vram = false;
        // 2026-09-01 BUG FIX: this used to also do `slot.slab_capacity = 0;`
        // while leaving slab_buffer ALLOCATED -- i.e. it lied about the
        // buffer's real size.
        //
        // Why that was harmful: ensure_slab_capacity decides whether to
        // regrow by testing `slot.slab_capacity < total`. With capacity
        // falsely zeroed, EVERY later claim on this slot looked like it
        // needed a regrow, so it free()d and re-posix_memalign'd a buffer
        // that was already the right size. That multiplied free/realloc
        // traffic enormously, and each one is a chance to pull the buffer
        // out from under a queued ActiveLayerUploadWorkItem still holding
        // the old pointer -> "cudaMemcpyAsync ... invalid argument".
        //
        // Measured, and this is what pinned it down: the residual upload
        // failures appeared ONLY under MOVE (4 and 17 errors across two
        // runs) and NEVER under COPY (0 and 0) -- and this line is the one
        // behaviour unique to the MOVE branch.
        //
        // MOVE's real intent is "this slot no longer holds a valid expert",
        // which expert_id = -1 already expresses completely. The buffer is
        // still allocated and still exactly slab_capacity bytes, so the
        // honest value is the one it already has. Leaving it correct lets
        // the next claim REUSE the buffer instead of needlessly churning it.
        slot.pending_copy_event = promoted ? promoted->upload_event : nullptr;
        LOG_DEBUG("[MOE-VRAM]  MOVE: freed RAM slot immediately after "
                  "promotion (layer=%d expert=%d)\n",
                  layer, expert_id);
        perf().n_move_on_promotion.fetch_add(1, std::memory_order_relaxed);
      } else {
        // FIX (real correctness gap, not just missed perf): this branch
        // used to set NOTHING here, unlike the MOVE branch above -- so a
        // COPY-mode promotion's slab_buffer had ZERO protection against a
        // totally unrelated eviction reclaiming this exact slot (or
        // ensure_slab_capacity free+realloc'ing it) while
        // upload_expert_async's DMA was still reading FROM it, a genuine
        // use-after-free/WAR hazard (see MoEExpertBlock::pending_copy_event's
        // own comment in engine.hpp -- explicitly safe/intended to
        // re-record here for exactly this case). Was previously only
        // reachable through the narrow pilot-speculative path
        // (allow_move=false); move_on_promotion=false routes EVERY
        // promotion through this branch, making this gap far more
        // exercised.
        slot.pending_copy_event = promoted ? promoted->upload_event : nullptr;
        // 2026-08-19: RE-EXAMINED the "deliberately does NOT set
        // also_in_vram=true here, would be unsafe" reasoning this
        // comment used to carry, and it doesn't hold up against the
        // current code -- every single site that reads also_in_vram
        // for a reclaim decision (this file's own victim-selection
        // scans at ~line 981/1391, and disk_batch_padding_evict_dup's
        // candidate filter at ~line 1981) ALWAYS checks
        // copy_still_pending() as an independent, unconditional
        // gate before treating a slot as reclaimable -- also_in_vram
        // is only ever used as a cheap SCORING/filter hint on top of
        // that, never as the sole safety signal. So a slot marked
        // also_in_vram=true here, while copy_still_pending() is
        // still true (pending_copy_event just set above), remains
        // fully protected from eviction/reuse exactly as before --
        // marking it early changes WHEN it becomes eligible for
        // reclaim (the instant the async copy is confirmed done,
        // via the same copy_still_pending() poll every candidate
        // scan already performs) not WHETHER it's protected while
        // still in flight.
        //
        // This closes a real, previously-documented capacity gap
        // (see CLAUDE.md/config comments on the "also_in_vram gap"):
        // without this, a COPY-mode promotion (the pilot/speculative
        // ram_vram_promote_top_k path is ALWAYS COPY-mode,
        // regardless of the global move_on_promotion setting -- see
        // promote_ram_to_vram's allow_move=false) only got flagged
        // as cheap-to-evict on a LATER re-promotion call that finds
        // it already VRAM-resident (the "already resident" branch
        // below) -- gated behind slot.hit_count >= promote_hit_
        // threshold, which a once-per-token-per-layer routing
        // pattern can easily never reach within a typical
        // generation. Real effect measured on a 40-token GLM-5.2
        // decode run before this fix: ~1900+ COPY-mode promotions
        // per session, the vast majority never marked also_in_vram,
        // so the same expert bytes sat fully duplicated in BOTH the
        // RAM tier and VRAM for the rest of the run, shrinking the
        // RAM tier's effective distinct-expert capacity (and
        // driving more real evictions/disk re-reads than a properly
        // reclaimed tier would need) for no compensating benefit.
        if (promoted) {
          slot.also_in_vram = true;
        }
        perf().n_copy_on_promotion.fetch_add(1, std::memory_order_relaxed);
      }
    } else {
      slot.also_in_vram = true;
      if (pin_threshold > 0 && slot.hit_count >= pin_threshold) {
        vram_cache->pin_expert(layer, expert_id);
      }
    }
  }

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
  // pin_on_hit (2026-08-30, default false -- every EXISTING caller's
  // behavior is completely unchanged): when true, a Hit result for
  // who==MainThread is pinned (ram_temp_pin_count incremented) BEFORE
  // this function's own lock is released, not deferred to the caller.
  // Real, hardware-confirmed bug this closes: a caller pinning the
  // returned slot itself, AFTER this function already returned, has a
  // genuine unlock-then-relock window in between (claim_or_wait's own
  // RAM_TIMED_LOCK releases on return; the caller's own separate ram_pin()
  // call is a SECOND, later lock acquisition) -- a concurrent thread's own
  // victim-selection scan can win that race and reclaim the slot before
  // the caller's pin ever takes effect, despite find_ram_slot having just
  // validated the slot's identity correctly moments earlier. Confirmed via
  // a real S=100 reproduction (speculative_promote_to_active_layer +
  // active_layer_upload_worker_threads, both adding enough concurrent
  // RAM-tier claim/victim-selection pressure to make this race likely
  // within one prefill pass): claim_or_wait genuinely returned Hit for a
  // real expert_id, backed by a slot find_ram_slot had JUST validated, yet
  // by the time the caller's own separate ram_pin() call ran, the slot's
  // real identity had already reverted to -1/slab_capacity=0 -- proving
  // the race lives in the gap BETWEEN this function's return and the
  // caller's own later pin, not (as first suspected) in the gap between
  // Phase A resolution and eventual dispatch, which an earlier,
  // insufficient version of this fix targeted. Deliberately scoped to a
  // new parameter (not applied unconditionally to every existing
  // who==MainThread caller) so get_expert()'s own simpler two callers
  // (execution_layer.cpp) are untouched -- they'd need their own matching
  // release logic added to safely receive a new pin obligation, which
  // hasn't been audited. Only get_experts_batch's Phase A/C (the exact
  // call sites that reproduced this) pass true; the caller (execution_
  // layer.cpp's ram_needed_k dispatch loop) is exactly where the matching
  // release lives.
  RamClaimResult
  claim_or_wait(StreamingEngine &engine, int layer, int expert_id,
                RamClaimCaller who,
                const std::vector<size_t> *extra_exclude = nullptr,
                bool pin_on_hit = false) {
    RAM_TIMED_LOCK(lock);
    auto &layer_slots = engine.layers_experts[layer];
    size_t cap = layer_slots.empty()
                     ? 0
                     : std::min(layer_slots.size(),
                                (size_t)std::max(1, cap_for_layer(layer)));

    ssize_t idx = find_ram_slot(layer_slots, cap, layer, expert_id);
    if (idx >= 0) {
      MoEExpertBlock &slot = layer_slots[(size_t)idx];
      if (slot.ram_loading) {
        if (who == RamClaimCaller::PrefetchWorker) {
          perf().n_ram_claim_already_inflight_prefetch.fetch_add(
              1, std::memory_order_relaxed);
        }
        return {RamClaimStatus::Loading, nullptr, 0};
      }

      // -- Ready: an ordinary cache hit --
      // EngineConfig::prefill_dispatch_lookahead_tokens instrumentation:
      // captured BEFORE the increment below, since hit_count==0 only
      // means "never hit before" up to this exact point -- this is the
      // FIRST real reuse of a slot that was loaded purely by the
      // lookahead widening (see MoEExpertBlock::loaded_by_lookahead's
      // own comment), i.e. the future token this fetch was FOR actually
      // reached it and found it resident (a genuine "prefetch worked"
      // event), as opposed to the slot instead being evicted before
      // anyone ever claimed it again (n_prefill_lookahead_loaded_
      // evicted_unused, ram_manager.hpp's eviction-victim-selection
      // sites). Together the two give a real used-vs-wasted split for
      // this mechanism, mirroring loaded_by_prefetch/"Unified RAM
      // waste"'s own existing design exactly, just for a different
      // caller/mechanism that design's own who==PrefetchWorker gate
      // doesn't cover.
      if (slot.loaded_by_lookahead && slot.hit_count == 0) {
        perf().n_prefill_lookahead_first_hit.fetch_add(
            1, std::memory_order_relaxed);
      }
      slot.last_used_clock = ++global_clock;
      if (slot.hit_count < UINT32_MAX)
        slot.hit_count++;
      if (slot.from_padding) {
        perf().n_pad_reads_hit_later.fetch_add(1, std::memory_order_relaxed);
        slot.from_padding = false;
      }
      if (++hits_since_decay >= HIT_DECAY_INTERVAL)
        decay_hit_counts(engine);
      perf().n_expert_cache_hit.fetch_add(1, std::memory_order_relaxed);
      perf().n_ram_claim_hit.fetch_add(1, std::memory_order_relaxed);

      if (who == RamClaimCaller::PrefetchWorker) {
        // Structurally cannot re-fetch an already-resident expert --
        // this IS the by-construction elimination of the old
        // "Cache-hit-beat-fetch" waste category. No GpuTopology/
        // cache_mutex touch here, see this function's own CRITICAL
        // rule above.
        perf().n_ram_claim_already_resident_prefetch.fetch_add(
            1, std::memory_order_relaxed);
        return {RamClaimStatus::Hit, &slot, (size_t)idx};
      }

      if (verbose_cache_log) {
        LOG_DEBUG("[MOE-CACHE] HIT   layer=%d expert=%d hit_count=%u\n", layer,
                  expert_id, slot.hit_count);
      }

      // Async, non-blocking promotion to VRAM -- decode-thread-only
      // side effect (see this function's CRITICAL rule above). Same
      // logic as the pre-unification get_expert Step 1 / get_experts_
      // batch Phase A hit path. Reactive: gated on a LIVE re-hit
      // crossing promote_hit_threshold. See promote_ram_to_vram()
      // below for the OTHER (pilot-prediction-driven, speculative)
      // caller of the shared do_promote_ram_slot_to_vram helper.
      if (vram_cache && slot.hit_count >= promote_hit_threshold &&
          !is_cpu_affinity_layer(layer)) {
        // "Direction 1" GPU1 routing -- see reactive_promote_gpu1_
        // every_n's own comment above. dual_role_gpu_id() re-derived
        // per call (cheap int compare + optional lock, same cost
        // already paid elsewhere in this codebase for the same
        // check) rather than cached, since this must stay correct
        // even if topology somehow changed (it doesn't at runtime
        // today, but nothing here should assume that silently).
        int target_gpu_id = -1;
        if (reactive_promote_gpu1_every_n > 0) {
          ++reactive_promote_counter;
          if (reactive_promote_counter %
                  (uint64_t)reactive_promote_gpu1_every_n ==
              0) {
            int dual_role = vram_cache->dual_role_gpu_id();
            if (dual_role >= 0) {
              target_gpu_id = dual_role;
              perf().n_reactive_promote_routed_gpu1.fetch_add(
                  1, std::memory_order_relaxed);
            }
          }
        }
        do_promote_ram_slot_to_vram(layer, expert_id, slot, /*allow_move=*/true,
                                    target_gpu_id);
      }
      // pin_on_hit: raw increment, NOT a call to ram_pin() -- this
      // function already holds `lock` for its own duration (ram_mutex
      // is a plain, non-recursive std::mutex; ram_pin() would
      // deadlock here, same reasoning as promote_ram_to_active_
      // layer_claim_for_worker's own identical raw-increment comment
      // above). This is the atomic, race-closing pin -- see this
      // function's own pin_on_hit parameter comment.
      if (pin_on_hit && who == RamClaimCaller::MainThread)
        ++slot.ram_temp_pin_count;
      return {RamClaimStatus::Hit, &slot, (size_t)idx};
    }

    // -- Not found: claim a fresh victim slot -- same eviction-scoring
    // logic as before unification, now also excluding ram_loading slots
    // (a slot mid-claim by someone else must never be reused).
    int lru_idx = -1;
    uint64_t lru_score = 0;
    for (size_t i = 0; i < cap; ++i) {
      bool excluded = extra_exclude &&
                      std::find(extra_exclude->begin(), extra_exclude->end(),
                                i) != extra_exclude->end();
      // 2026-08-17: pilot-confirmed exclusion, same safety argument
      // as the D2H-salvage victim scan above and GpuTopology::
      // evict_lru (see either's comment) -- counted separately so
      // this specific path's protection rate is visible too.
      bool pilot_protected =
          was_pilot_confirmed_resident(layer, layer_slots[i].expert_id);
      if (pilot_protected)
        perf().n_pilot_protected_eviction_skips_ram.fetch_add(
            1, std::memory_order_relaxed);
      bool never_pick = excluded || layer_slots[i].copy_still_pending() ||
                        layer_slots[i].ram_temp_pin_count > 0 ||
                        layer_slots[i].ram_loading || pilot_protected;
      uint64_t score = never_pick ? UINT64_MAX
                                  : (layer_slots[i].also_in_vram
                                         ? 0
                                         : eviction_score(layer_slots[i]));
      if (lru_idx < 0 || score < lru_score) {
        lru_score = score;
        lru_idx = (int)i;
      }
    }
    if (lru_idx < 0) {
      LOG_WARN("[RAM-CLAIM] layer=%d: no eviction-eligible slot available "
               "(every slot pending/loading) "
               "-- falling back to slot 0, which may collide with an in-flight "
               "claim\n",
               layer);
      lru_idx = 0;
    }
    MoEExpertBlock &victim = layer_slots[(size_t)lru_idx];
    int evicted_expert_id = victim.expert_id;
    if (evicted_expert_id != -1) {
      if (was_pilot_confirmed_resident(layer, evicted_expert_id)) {
        perf().n_evicted_was_pilot_confirmed_ram.fetch_add(
            1, std::memory_order_relaxed);
      }
      if (victim.loaded_by_prefetch && victim.hit_count == 0) {
        perf().n_ram_prefetch_loaded_evicted_unused.fetch_add(
            1, std::memory_order_relaxed);
        // 2026-08-30 split -- see MoEExpertBlock::consumed_by_vram_
        // upload (engine.hpp). Mirrors the identical split at the
        // D2H-salvage victim scan above.
        if (victim.consumed_by_vram_upload) {
          perf().n_ram_prefetch_evicted_consumed_by_vram.fetch_add(
              1, std::memory_order_relaxed);
        } else {
          perf().n_ram_prefetch_evicted_untouched.fetch_add(
              1, std::memory_order_relaxed);
          if (layer >= 0 && layer < PerfCounters::kMaxTrackedLayersForMiss)
            perf().n_ram_prefetch_evicted_untouched_by_layer[layer].fetch_add(
                1, std::memory_order_relaxed);
        }
      }
      if (victim.loaded_by_lookahead && victim.hit_count == 0) {
        perf().n_prefill_lookahead_loaded_evicted_unused.fetch_add(
            1, std::memory_order_relaxed);
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
      for (const char *proj : {"ffn_gate", "ffn_up", "ffn_down"}) {
        auto it =
            engine.model_file.tensors.find(base + proj + "." + es + ".weight");
        if (it != engine.model_file.tensors.end())
          posix_fadvise(engine.model_file.fd_for_tensor(it->second),
                        it->second.offset, it->second.size_bytes,
                        POSIX_FADV_DONTNEED);
      }
    }
    victim.also_in_vram = false;
    victim.loaded_by_prefetch = (who == RamClaimCaller::PrefetchWorker);
    // Per-layer denominator for the untouched-waste rate -- see
    // n_ram_prefetch_evicted_untouched_by_layer's own comment.
    if (who == RamClaimCaller::PrefetchWorker && layer >= 0 &&
        layer < PerfCounters::kMaxTrackedLayersForMiss)
      perf().n_ram_prefetch_loaded_by_layer[layer].fetch_add(
          1, std::memory_order_relaxed);
    victim.consumed_by_vram_upload =
        false; // 2026-08-30: same reset lifecycle (engine.hpp)
    victim.prefill_demand = 0;  // 2026-08-31: same reset lifecycle
    // Default reset -- claim_or_wait itself has no concept of "lookahead",
    // that's a get_experts_batch-only distinction (a specific candidate
    // INDEX within one call, not a caller identity). get_experts_batch's
    // own Phase A sets this true afterward, right after this Claimed
    // result comes back, for exactly the candidates its own is_lookahead
    // array marks -- see that call site's own comment.
    victim.loaded_by_lookahead = false;
    victim.from_padding = false;
    victim.expert_id = expert_id;
    victim.ram_loading = true;
    ram_expert_index[swap_key(layer, expert_id)] = (size_t)lru_idx;
    victim.last_used_clock = ++global_clock;
    victim.hit_count = 0;
    victim.load_token_idx = current_decode_token_idx_ref();

    perf().n_expert_cache_miss.fetch_add(1, std::memory_order_relaxed);
    if (in_prefill)
      perf().n_expert_cache_miss_prefill.fetch_add(1,
                                                   std::memory_order_relaxed);
    else
      perf().n_expert_cache_miss_decode.fetch_add(1, std::memory_order_relaxed);
    if (who == RamClaimCaller::MainThread)
    perf().n_ram_claim_by_main_thread.fetch_add(1, std::memory_order_relaxed);
    else
      perf().n_ram_claim_by_prefetch.fetch_add(1, std::memory_order_relaxed);

    return {RamClaimStatus::Claimed, &victim, (size_t)lru_idx};
  }

  // Pilot-prediction-driven speculative RAM->VRAM promotion (contrast:
  // claim_or_wait's Hit-branch promotion above is REACTIVE, gated on a
  // live re-hit crossing promote_hit_threshold; this is instead called
  // from execution_layer.cpp's l+1 pilot-prediction loop for a pick
  // classified into the "already_ram" bucket -- see CLAUDE.md's
  // per-rank "ram-only" metric this is calibrated against). Caller
  // already determined via a separate, EARLIER is_ram_resident() check
  // that this expert should be RAM-resident, but that was a distinct
  // locked operation, not the same one as this call -- by the time this
  // runs, the slot could already be gone (see comment above
  // do_promote_ram_slot_to_vram: RAM-tier eviction only ever happens on
  // this same decode thread, so nothing else can race it away between
  // the two calls, but a caller from a hypothetical future context
  // should not assume that) or could already be VRAM-resident (e.g. the
  // SAME layer transition's reactive path above already promoted it
  // moments earlier, if hit_count also happened to cross
  // promote_hit_threshold this cycle) -- both are handled as harmless
  // no-ops via do_promote_ram_slot_to_vram's own resident/gone checks,
  // not treated as errors.
  //
  // CRITICAL: decode-thread-only, same rule as claim_or_wait's own Hit
  // branch (see that function's CRITICAL comment on cache_mutex/
  // ram_mutex nesting) -- do_promote_ram_slot_to_vram touches
  // GpuTopology via upload_expert_async/pin_expert, both of which take
  // cache_mutex. execution_layer.cpp's pilot-prediction loop is
  // decode-thread-only code, satisfying this by construction; do not
  // call this from a LockFreePrefetcher worker thread.
  void promote_ram_to_vram(StreamingEngine &engine, int layer, int expert_id) {
    RAM_TIMED_LOCK(lock);
    auto &layer_slots = engine.layers_experts[layer];
    size_t cap = layer_slots.empty()
                     ? 0
                     : std::min(layer_slots.size(),
                                (size_t)std::max(1, cap_for_layer(layer)));
    ssize_t idx = find_ram_slot(layer_slots, cap, layer, expert_id);
    if (idx < 0) {
      perf().n_pilot_promote_slot_gone.fetch_add(1, std::memory_order_relaxed);
      return; // evicted/moved between the caller's residency check and this
              // call
    }
    MoEExpertBlock &slot = layer_slots[(size_t)idx];
    if (slot.ram_loading)
      return; // still mid-read -- nothing valid to promote yet
    perf().n_pilot_promote_attempted.fetch_add(1, std::memory_order_relaxed);
    // allow_move=false: always COPY here, regardless of move_on_promotion
    // -- see do_promote_ram_slot_to_vram's own comment on allow_move for
    // why a speculative, prediction-driven promotion must never delete
    // its RAM fallback the way the reactive (proven-re-hit) path safely
    // can.
    do_promote_ram_slot_to_vram(layer, expert_id, slot, /*allow_move=*/false);
  }

  // Speculative-promote-into-active-layer sibling of promote_ram_to_vram
  // above (2026-08-26, EngineConfig::speculative_promote_to_active_layer,
  // default off) -- SAME caller (execution_layer.cpp's l+1 pilot-
  // prediction loop), SAME "already_ram" population, but targets the
  // CHEAP, self-contained active-layer pool (GpuTopology::upload_expert_
  // active(..., speculative=true)) instead of the main VRAM LRU cache.
  // Directly answers the user-raised concern that speculative main-cache
  // promotion pays a real eviction cost (of some UNRELATED resident, via
  // the shared global pool) even when the prediction is individually
  // correct, since the alternative -- doing nothing and letting real
  // dispatch discover the need -- would have reached VRAM via the active
  // layer for free anyway. This routes the speculative guess through
  // that SAME free path preemptively, so a correct guess saves the DMA
  // latency at zero main-cache cost, and a wrong guess just expires
  // harmlessly (see GpuTopology::speculative_active_layer_fill's own
  // comment for the lifetime this gets).
  //
  // CRITICAL: decode-thread-only, same rule as promote_ram_to_vram above
  // (upload_expert_active also takes cache_mutex).
  // Returns the newly-issued (not-yet-upload_ready) slot on success, or
  // nullptr on any no-op/failure outcome -- the caller (execution_layer.
  // cpp's pilot-prediction loop) is responsible for collecting non-null
  // returns across this whole transition and calling GpuTopology::
  // sync_active_layer_batch on them ONCE, batching potentially several
  // DMAs behind one cudaStreamSynchronize instead of blocking on each
  // individually (the synchronous upload_expert_active version of this
  // function, tried first, cost a real, measured ~3.1s/request of pure
  // decode-thread-blocking DMA time -- see CLAUDE.md's real A/B for the
  // full numbers that motivated this async conversion). Until the
  // caller's batch sync runs, the returned slot's upload_ready is FALSE
  // and it must never be exposed to a find_active_expert lookup on any
  // thread -- same contract as upload_expert_active_async's own header
  // comment (gpu_topology.hpp), satisfied here because this whole
  // function is decode-thread-only and the caller syncs before returning
  // control to anything that could dispatch against layer+1.
  // target_gpu_id (2026-08-28, default -1): forwarded to upload_expert_
  // active_async -- see its own comment for the multi-GPU routing
  // meaning. -1 (legacy default) means "no routing decision, let the
  // pool scan everything" -- correct whenever active_layer_multi_gpu_
  // enabled() is false.
  ExpertLocation *promote_ram_to_active_layer(StreamingEngine &engine,
                                              int layer, int expert_id,
                                              int target_gpu_id = -1,
                                              int64_t qwen_pilot_prediction_ns = -1) {
    if (!vram_cache || !vram_cache->active_layer_pool)
      return nullptr;
    RAM_TIMED_LOCK(lock);
    // Same defense-in-depth as do_promote_ram_slot_to_vram -- centralize
    // the cpu_affinity_layers check here too rather than trust every
    // future caller to remember it (see that function's own comment).
    if (is_cpu_affinity_layer(layer))
      return nullptr;
    auto &layer_slots = engine.layers_experts[layer];
    size_t cap = layer_slots.empty()
                     ? 0
                     : std::min(layer_slots.size(),
                                (size_t)std::max(1, cap_for_layer(layer)));
    ssize_t idx = find_ram_slot(layer_slots, cap, layer, expert_id);
    if (idx < 0) {
      perf().n_pilot_promote_slot_gone.fetch_add(1, std::memory_order_relaxed);
      return nullptr; // evicted/moved between the caller's residency check and
                      // this call
    }
    MoEExpertBlock &slot = layer_slots[(size_t)idx];
    if (slot.ram_loading)
      return nullptr; // still mid-read -- nothing valid to upload yet
    bool gpu_capable = slot.slab_buffer &&
                       gpu_type_supported(slot.gate_proj.meta.type) &&
                       gpu_type_supported(slot.up_proj.meta.type) &&
                       gpu_type_supported(slot.down_proj.meta.type);
    if (!gpu_capable)
      return nullptr;
    // Already active-layer-resident (e.g. the OTHER pilot already filled
    // it this same transition, before this call's own cross-pilot dedup
    // could catch it via a different (layer,expert) key collision path,
    // or a genuine real cold-miss beat this speculative call to it) --
    // side-effect-free peek, never bumps hit_count/last_used_clock or
    // triggers a promotion, exactly why active_layer_has exists.
    if (vram_cache->active_layer_has(layer, expert_id)) {
      perf().n_pilot_active_fill_already_present.fetch_add(
          1, std::memory_order_relaxed);
      return nullptr;
    }
    // Already durably main-cache-resident (via reactive promotion, an
    // earlier pilot main-cache promotion, or startup [PIN] preload) --
    // nothing to gain from also active-layer-filling it.
    if (vram_cache->find_expert(layer, expert_id)) {
      perf().n_pilot_active_fill_already_main_cache.fetch_add(
          1, std::memory_order_relaxed);
      return nullptr;
    }
    // 2026-08-27: real coverage measurement -- is THIS speculative
    // fill's own source slab already covered by ram_pinned_budget_gb's
    // persistent pin? See n_speculative_fill_src_pinned/_unpinned's own
    // comment (perf_timer.hpp) for why this matters: upload_expert_
    // active_async has no pin-awareness of its own, unlike upload_
    // expert_async (the main-cache path this same candidate population
    // would have used with the redirect feature off).
    if (slot.slab_pinned) {
      perf().n_speculative_fill_src_pinned.fetch_add(1,
                                                     std::memory_order_relaxed);
      perf().n_active_layer_upload_src_pinned.fetch_add(
          1, std::memory_order_relaxed);
    } else {
      perf().n_speculative_fill_src_unpinned.fetch_add(
          1, std::memory_order_relaxed);
      perf().n_active_layer_upload_src_unpinned.fetch_add(
          1, std::memory_order_relaxed);
      perf().mark_unpinned_touch(layer, expert_id);
    }
    // Real, confirmed bug (found 2026-09 chasing a real GPU-path NaN on a
    // genuinely mixed per-expert model): slab_capacity is this SLOT's
    // grow-only ALLOCATION size, not this expert's real byte total --
    // ensure_slab_capacity (engine.hpp) never shrinks it, so a slot that
    // previously held a bigger expert (LRU-reused across differently-
    // sized experts, exactly the scenario Phase 5 mixed-precision layers
    // introduce) legitimately keeps a larger slab_capacity than the
    // CURRENT, smaller expert's real data occupies. Uploading
    // slab_capacity bytes copies real gate/up/down data followed by
    // stale leftover bytes from whatever the slot's PREVIOUS occupant's
    // tail was -- into a GPU active-layer slot that downstream dispatch
    // treats as this (smaller) expert's own tight gate|up|down layout,
    // corrupting the interpretation of where each projection's real
    // bytes end. Use the real total from this slot's own already-correct
    // per-projection metadata (populated by load_expert_slab) instead --
    // same computation the RAM-tier's other real-total call sites use.
    const size_t real_total = slot.gate_proj.meta.size_bytes +
                               slot.up_proj.meta.size_bytes +
                               slot.down_proj.meta.size_bytes;
    // upload_expert_active_async returns before the H2D DMA completes. The
    // RAM mutex being held here protects only until this function returns;
    // transfer that protection to the source slot's completion event before
    // releasing the mutex, exactly like the main-cache async promotion path
    // below. Without this, a RAM victim scan can free/regrow slot.slab_buffer
    // while the active-layer copy is still reading it.
    ++slot.ram_temp_pin_count;
    ExpertLocation *issued = vram_cache->upload_expert_active_async(
        layer, expert_id, slot.slab_buffer, real_total,
        /*speculative=*/true, target_gpu_id, qwen_pilot_prediction_ns);
    if (!issued) {
      --slot.ram_temp_pin_count;
      perf().n_pilot_active_fill_pool_full.fetch_add(1,
                                                     std::memory_order_relaxed);
      return nullptr;
    }
    // 2026-08-30 diagnostic -- see MoEExpertBlock::consumed_by_vram_
    // upload (engine.hpp). This path is a PRIME contributor to the
    // ambiguity being measured: it deliberately never touches the RAM
    // slot's hit_count (see the comment immediately below), so a
    // successfully-consumed slot looks identical to a never-read one at
    // eviction time without this flag.
    slot.consumed_by_vram_upload = true;
    slot.pending_copy_event = issued->upload_event;
    --slot.ram_temp_pin_count;
    // Deliberately does NOT touch the RAM slot at all (no also_in_vram,
    // no MOVE) -- matches the established, deliberate discipline for
    // EVERY active-layer fill (see execution_layer.cpp's own comment at
    // the real dispatch-time upload_expert_active call site): the
    // active layer is explicitly temporary, and marking the RAM source
    // "redundant, evict first" before any real hit confirms this
    // expert survives past its layer window would risk losing the only
    // copy of what a real run showed is the common case (most active-
    // layer fills are never promoted to durable residency).
    return issued;
  }

  // Return value of promote_ram_to_active_layer_claim_for_worker below --
  // deliberately plain pointers/values only (no ActiveLayerUploadWorkers
  // reference), since that type is only forward-declared here (see its
  // own comment above) -- the actual ActiveLayerUploadWorkers::try_submit
  // call must happen in a translation unit that has the FULL definition
  // (active_layer_upload_workers.hpp), i.e. the caller (execution_layer.
  // cpp), not here.
  struct RamPromoteForWorkerResult {
    ExpertLocation *slot = nullptr; // claimed active-layer slot; nullptr =
                                    // don't proceed via worker
    MoEExpertBlock *blk = nullptr;  // RAM source block -- ram_temp_pin_count
                                    // already bumped when slot != nullptr
    const uint8_t *host_data = nullptr;
    size_t host_bytes = 0;
  };

  // 2026-08-28: sibling of promote_ram_to_active_layer above, for the
  // ActiveLayerUploadWorkers path -- same pre-checks (cpu_affinity, slot
  // lookup, ram_loading, gpu_capable, already-active/already-main-cache,
  // pinned-vs-unpinned accounting), but claims the destination slot via
  // claim_active_layer_slot_for_worker (victim selection only, no copy)
  // instead of issuing the copy inline, and explicitly protects the RAM
  // source via ram_temp_pin_count (the same field ram_pin/ram_unpin
  // manipulate, but incremented directly here since this function already
  // holds `lock` for its own duration -- ram_mutex is a plain, non-
  // recursive mutex, so calling the separate ram_pin() method from in
  // here would deadlock). The caller is responsible for: (1) submitting
  // the returned {slot, blk, host_data, host_bytes} to ActiveLayerUpload
  // Workers::try_submit, and (2) on a submit failure specifically (queue
  // full -- the claim itself already succeeded), calling vram_cache->
  // abandon_worker_claim(slot) AND ram_unpin(*blk) to release both holds
  // cleanly. On the worker's own success path, ram_temp_pin_count is
  // released by the worker calling ram_mark_pending_copy_and_unpin once
  // it has genuinely dispatched the copy -- never by this function or its
  // caller in that case.
  RamPromoteForWorkerResult
  promote_ram_to_active_layer_claim_for_worker(StreamingEngine &engine,
                                               int layer, int expert_id,
                                               int target_gpu_id = -1) {
    RamPromoteForWorkerResult result;
    if (!vram_cache || !vram_cache->active_layer_pool)
      return result;
    RAM_TIMED_LOCK(lock);
    if (is_cpu_affinity_layer(layer))
      return result;
    auto &layer_slots = engine.layers_experts[layer];
    size_t cap = layer_slots.empty()
                     ? 0
                     : std::min(layer_slots.size(),
                                (size_t)std::max(1, cap_for_layer(layer)));
    ssize_t idx = find_ram_slot(layer_slots, cap, layer, expert_id);
    if (idx < 0) {
      perf().n_pilot_promote_slot_gone.fetch_add(1, std::memory_order_relaxed);
      return result;
    }
    MoEExpertBlock &slot = layer_slots[(size_t)idx];
    if (slot.ram_loading)
      return result;
    bool gpu_capable = slot.slab_buffer &&
                       gpu_type_supported(slot.gate_proj.meta.type) &&
                       gpu_type_supported(slot.up_proj.meta.type) &&
                       gpu_type_supported(slot.down_proj.meta.type);
    if (!gpu_capable)
      return result;
    if (vram_cache->active_layer_has(layer, expert_id)) {
      perf().n_pilot_active_fill_already_present.fetch_add(
          1, std::memory_order_relaxed);
      return result;
    }
    if (vram_cache->find_expert(layer, expert_id)) {
      perf().n_pilot_active_fill_already_main_cache.fetch_add(
          1, std::memory_order_relaxed);
      return result;
    }
    if (slot.slab_pinned) {
      perf().n_speculative_fill_src_pinned.fetch_add(1,
                                                     std::memory_order_relaxed);
      perf().n_active_layer_upload_src_pinned.fetch_add(
          1, std::memory_order_relaxed);
    } else {
      perf().n_speculative_fill_src_unpinned.fetch_add(
          1, std::memory_order_relaxed);
      perf().n_active_layer_upload_src_unpinned.fetch_add(
          1, std::memory_order_relaxed);
      perf().mark_unpinned_touch(layer, expert_id);
    }
    // real_total, not slab_capacity -- same bug/reasoning as
    // promote_ram_to_active_layer's own upload_expert_active_async call
    // (see its comment): this slot's grow-only allocation can legitimately
    // outlive whatever smaller expert currently occupies it.
    const size_t real_total = slot.gate_proj.meta.size_bytes +
                               slot.up_proj.meta.size_bytes +
                               slot.down_proj.meta.size_bytes;
    ExpertLocation *claimed = vram_cache->claim_active_layer_slot_for_worker(
        layer, expert_id, real_total, /*speculative=*/true,
        target_gpu_id);
    if (!claimed) {
      perf().n_pilot_active_fill_pool_full.fetch_add(1,
                                                     std::memory_order_relaxed);
      return result;
    }
    ++slot.ram_temp_pin_count;
    // 2026-08-30 diagnostic -- see MoEExpertBlock::consumed_by_vram_
    // upload (engine.hpp). Set at CLAIM time rather than after the
    // worker's copy completes: the claim is what commits this slot's
    // bytes to being read, and the worker thread has no clean path back
    // to this flag under ram_mutex anyway. A claim that later fails to
    // submit is rare and would only very slightly over-count the benign
    // bucket -- acceptable for a diagnostic whose whole purpose is
    // separating two populations that currently differ by ~86 points.
    slot.consumed_by_vram_upload = true;
    result.slot = claimed;
    result.blk = &slot;
    result.host_data = slot.slab_buffer;
    result.host_bytes = real_total;
    return result;
  }

  // GPU1-targeted sibling of promote_ram_to_vram above, for the "Idea 1"
  // prediction-based dual-role-slice loading investigation (2026-08-16,
  // see doc/gpu-utilization-vram-analysis.md and the memory note this
  // session produced). Same "already_ram" population, same COPY-only
  // safety discipline (allow_move=false -- a rank-4+ prediction is lower
  // confidence than the existing top-1/2 mechanism, so it must never
  // delete its own RAM fallback), same decode-thread-only contract as
  // promote_ram_to_vram (see that function's own CRITICAL comment) --
  // the only difference is threading target_gpu_id through so the
  // upload lands specifically in the dual-role slice (via evict_lru's
  // restrict_gpu_id) instead of competing for the primary vault pool.
  // Caller (execution_layer.cpp) is responsible for confirming
  // target_gpu_id is a real dual-role GPU (GpuTopology::dual_role_
  // gpu_id() >= 0) before ever calling this -- this function itself
  // doesn't re-derive that, since the caller already needs the value to
  // decide whether to call at all.
  void promote_ram_to_vram_gpu1(StreamingEngine &engine, int layer,
                                int expert_id, int target_gpu_id) {
    RAM_TIMED_LOCK(lock);
    auto &layer_slots = engine.layers_experts[layer];
    size_t cap = layer_slots.empty()
                     ? 0
                     : std::min(layer_slots.size(),
                                (size_t)std::max(1, cap_for_layer(layer)));
    ssize_t idx = find_ram_slot(layer_slots, cap, layer, expert_id);
    if (idx < 0) {
      perf().n_predict_gpu1_promote_slot_gone.fetch_add(
          1, std::memory_order_relaxed);
      return;
    }
    MoEExpertBlock &slot = layer_slots[(size_t)idx];
    if (slot.ram_loading)
      return;
    perf().n_predict_gpu1_promote_attempted.fetch_add(
        1, std::memory_order_relaxed);
    do_promote_ram_slot_to_vram(layer, expert_id, slot, /*allow_move=*/false,
                                target_gpu_id);
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
  // will now see a genuinely free slot. pin_on_ready atomically retains the
  // returned Ready slot for a caller that will continue using its slab after
  // this function returns; that caller owns the matching ram_unpin().
  MoEExpertBlock *wait_for_ram_slot(StreamingEngine &engine, int layer,
                                    int expert_id,
                                    bool pin_on_ready = false) {
    auto t0 = std::chrono::high_resolution_clock::now();
    bool warned = false;
    for (;;) {
      {
        RAM_TIMED_LOCK(lock);
        auto &layer_slots = engine.layers_experts[layer];
        size_t cap = layer_slots.empty()
                         ? 0
                         : std::min(layer_slots.size(),
                                    (size_t)std::max(1, cap_for_layer(layer)));
        ssize_t idx = find_ram_slot(layer_slots, cap, layer, expert_id);
        if (idx < 0)
          return nullptr; // claim was abandoned -- caller should retry
                          // claim_or_wait
        if (!layer_slots[(size_t)idx].ram_loading) {
          double wait_ms = std::chrono::duration<double, std::milli>(
                               std::chrono::high_resolution_clock::now() - t0)
                               .count();
          perf().n_ram_wait_inflight_main_thread.fetch_add(
              1, std::memory_order_relaxed);
          PerfCounters::atomic_add(perf().t_ram_wait_inflight_ms_sum, wait_ms);
          MoEExpertBlock &slot = layer_slots[(size_t)idx];
          // The caller may need to keep the returned slab stable after this
          // lookup. Take that hold under the same lock that observed Ready;
          // a separate ram_pin() after return would leave a real eviction
          // window between the two lock acquisitions.
          if (pin_on_ready)
            ++slot.ram_temp_pin_count;
          return &slot;
        }
      }
      double elapsed_ms = std::chrono::duration<double, std::milli>(
                              std::chrono::high_resolution_clock::now() - t0)
                              .count();
      if (!warned && elapsed_ms > SLOW_RAM_WAIT_MS) {
        LOG_WARN(
            "[SLOW-RAM-WAIT] layer=%d expert=%d waited %.2fms so far for an "
            "in-flight claim "
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
  void publish_ram_slot(MoEExpertBlock &slot) {
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
  void abandon_ram_claim(int layer, MoEExpertBlock &slot) {
    RAM_TIMED_LOCK(lock);
    int expert_id = slot.expert_id;
    slot.expert_id = -1;
    ram_expert_index.erase(swap_key(layer, expert_id));
    slot.last_used_clock = 0;
    slot.hit_count = 0;
    slot.ram_loading = false;
    slot.loaded_by_prefetch = false;
    slot.loaded_by_lookahead = false;
    slot.consumed_by_vram_upload =
        false; // 2026-08-30: same reset lifecycle (engine.hpp)
    slot.prefill_demand = 0;  // 2026-08-31: same reset lifecycle
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
  // seed_hit_count: EngineConfig::ram_preload_seed_hit_count's own value,
  // already clamped to promote_hit_threshold-1 by the caller (main.cpp) --
  // see that flag's comment for why the clamp matters. 0 (default,
  // matches every pre-2026-09-04 call site) reproduces the original
  // behavior exactly.
  void preload_ram_slot(StreamingEngine &engine, int layer, int expert_id,
                        size_t slot_idx, uint32_t seed_hit_count = 0) {
    RAM_TIMED_LOCK(lock);
    auto &layer_slots = engine.layers_experts[layer];
    MoEExpertBlock &slot = layer_slots[slot_idx];
    slot.last_used_clock = ++global_clock;
    slot.hit_count = seed_hit_count;
    slot.load_token_idx = current_decode_token_idx_ref();
    slot.ram_loading = false;
    ram_expert_index[swap_key(layer, expert_id)] = slot_idx;
  }

  // 2026-08-29 (EngineConfig::ram_pin_eager_preregister): called ONCE
  // from main.cpp, right after the [PIN] preload block finishes writing
  // into engine.layers_experts and BEFORE prefetcher.start() -- the same
  // "no concurrent thread touches this state yet" timing invariant Tier
  // 1/2/3 preload itself already relies on (see prefetcher.start()'s own
  // call-site comment), so no locking is needed here either, matching
  // how Tier 3's own preload loop (main.cpp) calls engine.load_expert_
  // slab/ensure_slab_capacity directly without RAM_TIMED_LOCK.
  //
  // For every RAM-tier slot the historically-ranked preload above left
  // EMPTY (expert_id==-1) within that layer's own real cap_for_layer
  // budget, eagerly grows and persistently pins a GENERIC buffer sized
  // to that layer's own real expert size (engine.compute_expert_slab_
  // bytes -- every expert within one layer shares this exact size, so
  // ANY future real load into this slot will find slab_capacity already
  // sufficient) -- consuming whatever's left of ram_pinned_budget_gb
  // beyond what real historically-ranked experts already claimed. The
  // slot's expert_id is deliberately left at -1 (never given a fake
  // identity, never inserted into ram_expert_index) so every existing
  // residency check (is_ram_resident, find_expert-style lookups) keeps
  // treating it as genuinely empty until a real future load overwrites
  // it through the completely unchanged ensure_slab_capacity/load_
  // expert_slab path -- that real load will simply find slab_capacity
  // already >= total and skip both the realloc and the register call.
  //
  // Budget is checked BEFORE allocating (not left to ensure_slab_
  // capacity's own internal check alone) specifically so a slot beyond
  // the budget never even pays a wasted posix_memalign -- an idle,
  // unregistered generic buffer sitting in host RAM for no benefit would
  // just be pure waste against this project's own careful host_ram_gb
  // margin management. Stops entirely (not just this layer) the moment
  // the global budget can't fit one more slab, since ram_pinned_budget_
  // bytes is shared across every layer, not per-layer.
  void eager_preregister_remaining_slots(StreamingEngine &engine) {
    if (engine.ram_pinned_budget_bytes == 0)
      return;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (size_t layer = 0; layer < engine.layers_experts.size(); ++layer) {
      size_t total = engine.compute_expert_slab_bytes((int)layer);
      if (total == 0)
        continue; // dense layer, no experts
      int cap = std::min(cap_for_layer((int)layer),
                         (int)engine.layers_experts[layer].size());
      for (int slot_idx = 0; slot_idx < cap; ++slot_idx) {
        MoEExpertBlock &slot = engine.layers_experts[layer][slot_idx];
        if (slot.expert_id != -1)
          continue; // real preload already landed here
        if (slot.slab_capacity >= total)
          continue; // already big enough somehow
        size_t used =
            engine.ram_pinned_bytes_used.load(std::memory_order_relaxed);
        if (used + total > engine.ram_pinned_budget_bytes) {
          // Global budget exhausted -- no further layer/slot can
          // succeed either, stop the whole pass rather than
          // paying wasted allocations to confirm that repeatedly.
          double ms = std::chrono::duration<double, std::milli>(
                          std::chrono::high_resolution_clock::now() - t0)
                          .count();
          perf().t_ram_eager_preregister.fetch_add(ms / 1000.0,
                                                   std::memory_order_relaxed);
          return;
        }
        size_t cap_before = slot.slab_capacity;
        bool was_pinned = slot.slab_pinned;
        engine.ensure_slab_capacity(slot, total);
        // Only count it if this call actually grew+registered the
        // slot (mirrors ensure_slab_capacity's own success signal
        // -- slab_pinned flips true only on a real cudaHostRegister
        // success, per that function's own body).
        if (slot.slab_capacity > cap_before && slot.slab_pinned &&
            !was_pinned) {
          perf().n_ram_eager_preregistered.fetch_add(1,
                                                     std::memory_order_relaxed);
          perf().bytes_ram_eager_preregistered.fetch_add(
              (int64_t)total, std::memory_order_relaxed);
        }
      }
    }
    double ms = std::chrono::duration<double, std::milli>(
                    std::chrono::high_resolution_clock::now() - t0)
                    .count();
    perf().t_ram_eager_preregister.fetch_add(ms / 1000.0,
                                             std::memory_order_relaxed);
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
  // pin_for_caller=false preserves the normal lookup contract. When true,
  // the returned block carries one RAM-tier temporary hold taken atomically
  // for Hit/Loading results (or before publishing a newly loaded result); the
  // caller must release it with ram_unpin() after its last source read.
  MoEExpertBlock *get_expert(StreamingEngine &engine, int layer,
                             int expert_id, bool pin_for_caller = false) {
    // Whole-function cost -- see this function's own PERF_SCOPE history:
    // the RAM tier (this function and its batched sibling) is the
    // largest remaining scope in the whole decode-time hot path.
    PERF_SCOPE(t_ram_get_expert);
    for (;;) {
      RamClaimResult result =
          claim_or_wait(engine, layer, expert_id, RamClaimCaller::MainThread,
                        nullptr, pin_for_caller);
      if (result.status == RamClaimStatus::Hit) {
        return result.slot;
      }
      if (result.status == RamClaimStatus::Loading) {
        MoEExpertBlock *slot =
            wait_for_ram_slot(engine, layer, expert_id, pin_for_caller);
        if (slot)
          return slot;
        continue; // the in-flight claim was abandoned -- retry the claim
                  // ourselves
      }

      // -- Claimed: this caller now exclusively owns the slot; populate
      // it OUTSIDE the lock via the same synchronous loader every
      // other real-miss path already uses --
      MoEExpertBlock &victim = *result.slot;
      // claim_or_wait marked this slot Loading, so it cannot be evicted while
      // this lock-free load runs. Take the caller-lifetime hold before
      // publishing Ready; unlike a later pin, this has no publish-to-pin gap.
      if (pin_for_caller)
        ram_pin(victim);
      PERF_START(expert_disk_load);
      engine.load_expert_slab(layer, expert_id, victim);
      double load_ms = std::chrono::duration<double, std::milli>(
                           std::chrono::high_resolution_clock::now() -
                           _perf_expert_disk_load)
                           .count();
      ram_swap_stats[swap_key(layer, expert_id)].swap_in++;
      publish_ram_slot(victim);

      LOG_DEBUG("[MOE-CACHE] MISS  layer=%d expert=%d slot=%zu "
                "loaded_from_disk=%.2fms via=pread "
                "(%.2f MB, %.2f GB/s)\n",
                layer, expert_id, result.slot_idx, load_ms,
                victim.slab_capacity / 1e6,
                load_ms > 0 ? (victim.slab_capacity / 1e9) / (load_ms / 1000.0)
                            : 0.0);

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
  // out_was_real_fetch (optional, nullable): parallel to expert_ids/out,
  // set true for exactly the k's Phase A classified Claimed (a genuine
  // disk-cold miss owned by THIS call) and false for Hit/Loading. Added
  // for EngineConfig::cpu_offload_max_experts -- the caller needs to know
  // WHICH of a batch's entries actually paid for a real SSD read before it
  // can decide which ones are eligible to redirect to CPU compute instead
  // of a VRAM upload (a Hit/Loading entry was never a candidate for that
  // decision in the first place, it never touched disk this call).
  // caller_k (nullable, parallel array to expert_ids, size `topk`):
  // execution_layer.cpp's own ram_needed_k array, reused directly rather
  // than asking the caller to build a second parallel array -- a value
  // of -1 already means exactly "this entry is a prefill_dispatch_
  // lookahead_tokens addition, not the calling token's own real need"
  // (see that field's own comment, engine_config.hpp). When caller_k[k]
  // < 0 and Phase A's claim_or_wait for expert_ids[k] returns Claimed (a
  // genuine new fetch, not an existing Hit/Loading), the resulting slot
  // is tagged MoEExpertBlock::loaded_by_lookahead=true so its later
  // used-vs-evicted-unused outcome can be measured (see that field's own
  // comment for the two counters this feeds). nullptr (every pre-existing
  // caller) = exact prior behavior, no tagging.
  void get_experts_batch(StreamingEngine &engine, int layer,
                         const int *expert_ids, int topk, MoEExpertBlock **out,
                         bool *out_was_real_fetch = nullptr,
                         const int *caller_k = nullptr) {
    // See get_expert's own comment on t_ram_get_expert -- same rationale,
    // this is get_expert's batched sibling and the higher-traffic of the
    // two on the real decode-time call path (called once per non-shadowed
    // subset per token/layer, vs get_expert's per-expert calls elsewhere).
    PERF_SCOPE(t_ram_get_experts_batch);
    auto &layer_slots = engine.layers_experts[layer];
    size_t cap = layer_slots.empty()
                     ? 0
                     : std::min(layer_slots.size(),
                                (size_t)std::max(1, cap_for_layer(layer)));

    struct PendingLoad {
      size_t slot_idx;
      int expert_id;
      int k;
      double load_ms;
    };
    std::vector<PendingLoad> pending;
    pending.reserve(topk);
    struct WaitEntry {
      int expert_id;
      int k;
    };
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
    PERF_START(ram_batch_phase_a);
    for (int k = 0; k < topk; ++k) {
      int expert_id = expert_ids[k];
      // pin_on_hit=true: see claim_or_wait's own parameter comment
      // for the full 2026-08-30 investigation -- a Hit result must be
      // pinned ATOMICALLY, inside claim_or_wait's own lock, not via a
      // separate ram_pin() call here. A first attempt at this fix
      // instead added a diagnostic right here comparing out[k]->
      // expert_id against the requested expert_id, expecting it to
      // catch the race directly -- it fired constantly, but on a
      // FALSE lead: a Hit result can legitimately be a MOVE-promoted
      // slot (do_promote_ram_slot_to_vram's allow_move branch, just
      // above in this same claim_or_wait call, expert_id reset to -1
      // deliberately as part of that promotion, see the comment right
      // below), so the comparison isn't a valid correctness check in
      // general and was removed. The REAL race, only found by tracing
      // through exactly WHERE claim_or_wait releases its own lock
      // relative to any caller-side pin, is documented on claim_or_
      // wait's own pin_on_hit parameter -- confirmed fixed on real
      // hardware (the same S=100 config that reliably crashed now
      // completes cleanly) once the pin happens atomically here,
      // before this function's lock is ever released.
      RamClaimResult result =
          claim_or_wait(engine, layer, expert_id, RamClaimCaller::MainThread,
                        &claimed_this_batch, /*pin_on_hit=*/true);
      if (out_was_real_fetch)
        out_was_real_fetch[k] = false; // default; Claimed below overrides
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
      // 2026-08-30: same pin-at-resolution fix as the Hit branch
      // above -- this slot is EXCLUSIVELY owned by this call already
      // (claim_or_wait's Claimed status guarantees that), so this pin
      // can't be "protecting against yourself" -- it protects against
      // an UNRELATED thread's victim-selection scan for a DIFFERENT
      // (layer,expert) mistakenly treating this one as eligible
      // before ram_loading/publish_ram_slot's own state machine would
      // normally exclude it, and continues protecting straight
      // through to the caller's own eventual dispatch, after
      // ram_loading flips false. Released by the caller once done.
      ram_pin(*out[k]);
      if (out_was_real_fetch)
        out_was_real_fetch[k] = true;
      if (caller_k && caller_k[k] < 0)
        layer_slots[result.slot_idx].loaded_by_lookahead = true;
      claimed_this_batch.push_back(result.slot_idx);
      pending.push_back({result.slot_idx, expert_id, k, 0.0});
    }
    PERF_END(ram_batch_phase_a, t_ram_batch_phase_a);

    // Real per-layer disk-fetch-miss measurement (decode-only) -- see
    // PerfCounters::n_decode_real_fetch_by_layer's own comment for why
    // (gathering data to inform a CPU-offload cap design before it's
    // built). `pending` here is exactly Phase A's real-fetch population
    // for this call.
    if (!in_prefill && layer >= 0 &&
        layer < PerfCounters::kMaxTrackedLayersForMiss) {
      perf().n_decode_calls_by_layer[layer].fetch_add(
          1, std::memory_order_relaxed);
      uint32_t n_fetch = (uint32_t)pending.size();
      if (n_fetch > 0) {
        perf().n_decode_real_fetch_by_layer[layer].fetch_add(
            n_fetch, std::memory_order_relaxed);
      }
      // Distribution (pooled across layers) + per-layer max -- see
      // n_decode_fetch_count_histogram's own comment (perf_timer.hpp)
      // for why the mean alone isn't enough to size a cap.
      int bucket =
          std::min((int)n_fetch, PerfCounters::kFetchHistogramBuckets - 1);
      perf().n_decode_fetch_count_histogram[bucket].fetch_add(
          1, std::memory_order_relaxed);
      auto &layer_max = perf().n_decode_max_fetch_by_layer[layer];
      uint32_t cur_max = layer_max.load(std::memory_order_relaxed);
      while (n_fetch > cur_max &&
             !layer_max.compare_exchange_weak(cur_max, n_fetch,
                                              std::memory_order_relaxed)) {
        // cur_max reloaded by compare_exchange_weak on failure; retry.
      }
    }

    // --- Phase B: batch the WHOLE layer's pending set into one io_uring
    // submission when the WHOLE layer's pending set into one io_uring
    // submission when available. All-or-
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
    PERF_START(ram_batch_phase_b);
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
        perf().n_decode_uring_not_available.fetch_add(
            1, std::memory_order_relaxed);
      }
      if (use_decode_uring && decode_uring_ring.available()) {
        std::vector<io_uring_backend::ReadRequest> reqs;
        reqs.reserve((size_t)n_pending * 3);
        bool all_packed = true;
        // Stripe this batch's experts across both replicas when a
        // mirror is configured (GGUFModel::open_mirror). io_uring
        // allows mixed fds in ONE submission, so the single
        // submit_and_wait below is serviced by both devices' queues
        // concurrently -- the decode thread's own blocking read
        // (phase_b, the largest single decode-time cost) finishes in
        // roughly the time the slower half takes, not the sum.
        // Degrades to replica 0 for everything on a single-drive
        // setup, so no branch is needed here.
        // Intra-expert split rather than per-expert alternation --
        // see StreamingEngine::REPLICA_SPLIT. Decode batches here
        // average only ~2.3 experts, so alternating whole experts
        // left one drive largely idle on the very reads the decode
        // thread blocks on; splitting puts both drives on each one.
        const int stripe_replica =
            engine.model_file.has_mirror() ? StreamingEngine::REPLICA_SPLIT : 0;
        for (int i = 0; i < n_pending && all_packed; ++i) {
          io_uring_backend::ReadRequest tmp[3];
          int cnt = 0;
          size_t total = 0;
          MoEExpertBlock &victim = layer_slots[pending[i].slot_idx];
          if (!engine.build_expert_read_requests(layer, pending[i].expert_id,
                                                 victim, tmp, cnt, total,
                                                 stripe_replica)) {
            all_packed = false;
            break;
          }
          for (int j = 0; j < cnt; ++j)
            reqs.push_back(tmp[j]);
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
        if (!disk_batch_padding)
          perf().n_pad_dbg_padding_false.fetch_add(1,
                                                   std::memory_order_relaxed);
        if (!all_packed)
          perf().n_pad_dbg_not_all_packed.fetch_add(1,
                                                    std::memory_order_relaxed);
        if (reqs.empty())
          perf().n_pad_dbg_reqs_empty.fetch_add(1, std::memory_order_relaxed);
        if (in_prefill)
          perf().n_pad_dbg_in_prefill.fetch_add(1, std::memory_order_relaxed);
        if (n_pending >= disk_batch_padding_target)
          perf().n_pad_dbg_target_not_reached.fetch_add(
              1, std::memory_order_relaxed);
        if (disk_batch_padding && all_packed && !reqs.empty() && !in_prefill &&
            n_pending < disk_batch_padding_target) {
          perf().n_pad_gate_entered.fetch_add(1, std::memory_order_relaxed);
          int want = disk_batch_padding_target - n_pending;
          auto rr_it = route_rank_by_layer.find(layer);
          if (want > 0 && rr_it == route_rank_by_layer.end()) {
            perf().n_pad_no_ranking_for_layer.fetch_add(
                1, std::memory_order_relaxed);
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
              if ((int)ranked.size() >= want)
                break;
              bool is_this_tokens_own = false;
              for (int kk = 0; kk < topk; ++kk) {
                if (expert_ids[kk] == e) {
                  is_this_tokens_own = true;
                  break;
                }
              }
              if (is_this_tokens_own)
                continue;
              if (is_ram_resident(layer, e))
                continue;
              if (vram_cache && vram_cache->find_expert(layer, e))
                continue;
              ranked.push_back(e);
            }
            if (ranked.empty()) {
              perf().n_pad_no_candidates_after_filter.fetch_add(
                  1, std::memory_order_relaxed);
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
              for (size_t i = 0; i < cap && free_slots.size() < ranked.size();
                   ++i) {
                if (layer_slots[i].expert_id != -1)
                  continue;
                if (std::find(claimed_this_batch.begin(),
                              claimed_this_batch.end(),
                              i) != claimed_this_batch.end())
                  continue;
                free_slots.push_back(i);
              }
              if (free_slots.size() < ranked.size()) {
                perf().n_pad_no_free_slot.fetch_add(ranked.size() -
                                                        free_slots.size(),
                                                    std::memory_order_relaxed);
              }

              // --- Eviction fallback (disk_batch_padding_evict_dup):
              // once genuinely free slots run out, reach for a RAM
              // slot whose expert is ALSO VRAM-resident instead of
              // giving up -- see that config field's own comment
              // (engine_config.hpp) for the full safety argument
              // (also_in_vram slots are the SAME score-0/cheapest
              // eviction candidates claim_or_wait's own real-miss
              // victim selection already relies on; this just lets
              // padding reach for that same, already-established-
              // safe population too). Same never_pick exclusions as
              // that canonical victim scan: copy_still_pending(),
              // ram_temp_pin_count>0, ram_loading, already claimed
              // this batch. Reclaimed slots are bookkept exactly
              // like a real miss's victim claim (swap_out stat,
              // index erase) THEN pushed into the SAME free_slots
              // vector as expert_id=-1 -- the unchanged claim loop
              // below can't tell the difference from a genuinely
              // free slot at that point.
              if (disk_batch_padding_evict_dup &&
                  free_slots.size() < ranked.size()) {
                size_t n_genuinely_free = free_slots.size();
                for (size_t i = 0; i < cap && free_slots.size() < ranked.size();
                     ++i) {
                  MoEExpertBlock &cand = layer_slots[i];
                  if (!cand.also_in_vram)
                    continue;
                  if (cand.ram_loading)
                    continue;
                  if (cand.copy_still_pending())
                    continue;
                  if (cand.ram_temp_pin_count > 0)
                    continue;
                  if (std::find(claimed_this_batch.begin(),
                                claimed_this_batch.end(),
                                i) != claimed_this_batch.end())
                    continue;
                  if (std::find(free_slots.begin(), free_slots.end(), i) !=
                      free_slots.end())
                    continue;
                  int evicted_id = cand.expert_id;
                  ram_swap_stats[swap_key(layer, evicted_id)].swap_out++;
                  ram_expert_index.erase(swap_key(layer, evicted_id));
                  cand.expert_id = -1;
                  free_slots.push_back(i);
                }
                size_t n_evicted = free_slots.size() - n_genuinely_free;
                if (n_evicted > 0) {
                  perf().n_pad_evict_dup_used.fetch_add(
                      n_evicted, std::memory_order_relaxed);
                }
                if (free_slots.size() < ranked.size()) {
                  perf().n_pad_evict_dup_unavailable.fetch_add(
                      ranked.size() - free_slots.size(),
                      std::memory_order_relaxed);
                }
              }

              size_t n_pad = std::min(ranked.size(), free_slots.size());
              claimed_candidate_ids.reserve(n_pad);
              for (size_t p = 0; p < n_pad; ++p) {
                int candidate_id = ranked[p];
                size_t slot_idx = free_slots[p];
                MoEExpertBlock &pv = layer_slots[slot_idx];

                // Claim exactly like a real miss's own claim
                // sequence above -- mostly a no-op for a
                // genuinely-free slot (no evicted identity, no
                // pending copy), mirrored verbatim for
                // consistency/future-proofing.
                claimed_this_batch.push_back(slot_idx);
                pv.also_in_vram = false;
                pv.loaded_by_prefetch = false;
                pv.loaded_by_lookahead = false;
                pv.consumed_by_vram_upload =
                    false; // 2026-08-30: same reset lifecycle (engine.hpp)
                pv.prefill_demand = 0;  // 2026-08-31: same reset lifecycle
                if (pv.pending_copy_event) {
                  cudaEventSynchronize(pv.pending_copy_event);
                  pv.pending_copy_event = nullptr;
                }
                pv.expert_id = candidate_id;
                pv.ram_loading = true;
                ram_expert_index[swap_key(layer, candidate_id)] = slot_idx;
                pv.last_used_clock = ++global_clock;
                pv.hit_count = 0;
                pv.load_token_idx = current_decode_token_idx_ref();
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
              MoEExpertBlock &pv = layer_slots[slot_idx];

              io_uring_backend::ReadRequest ptmp[3];
              int pcnt = 0;
              size_t ptotal = 0;
              // Padding rides the SAME submission as the real
              // misses above, so it continues the same
              // alternation rather than restarting at 0 --
              // otherwise a padded batch would skew back toward
              // the primary drive.
              if (!engine.build_expert_read_requests(layer, candidate_id, pv,
                                                     ptmp, pcnt, ptotal,
                                                     stripe_replica)) {
                revert_padding_claim(pv, layer, candidate_id);
                continue;
              }
              for (int j = 0; j < pcnt; ++j)
                reqs.push_back(ptmp[j]);
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
            while (reqs.size() > io_uring_backend::Ring::CAPACITY &&
                   !pad_ids.empty()) {
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
          perf().n_decode_uring_not_packed.fetch_add(1,
                                                     std::memory_order_relaxed);
        } else if (reqs.empty()) {
          // Should be unreachable (n_pending > 0 implies at least
          // one request), but if it ever happens, don't silently
          // miscount it as any of the other buckets.
        } else if (reqs.size() > io_uring_backend::Ring::CAPACITY) {
          perf().n_decode_uring_too_large.fetch_add(1,
                                                    std::memory_order_relaxed);
        }
        if (all_packed && !reqs.empty() &&
            reqs.size() <= io_uring_backend::Ring::CAPACITY) {
          std::vector<ssize_t> results(reqs.size());
          auto t0 = std::chrono::high_resolution_clock::now();
          bool submitted = decode_uring_ring.submit_and_wait(
              reqs.data(), (int)reqs.size(), results.data(),
              engine.drop_expert_pages);
          double batch_ms = std::chrono::duration<double, std::milli>(
                                std::chrono::high_resolution_clock::now() - t0)
                                .count();
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
              if (results[r] != (ssize_t)reqs[r].len) {
                all_ok = false;
                break;
              }
            }
          }
          if (!submitted) {
            perf().n_decode_uring_submit_failed.fetch_add(
                1, std::memory_order_relaxed);
          } else if (!all_ok) {
            perf().n_decode_uring_short_read.fetch_add(
                1, std::memory_order_relaxed);
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
                revert_padding_claim(layer_slots[pad_slots[p]], layer,
                                     pad_ids[p]);
              }
            } else {
              size_t r = real_reqs_count;
              uint64_t pad_bytes = 0;
              for (size_t p = 0; p < pad_ids.size(); ++p) {
                int cand_cnt = pad_req_counts[p];
                bool cand_ok = true;
                for (int j = 0; j < cand_cnt; ++j) {
                  if (results[r + (size_t)j] !=
                      (ssize_t)reqs[r + (size_t)j].len) {
                    cand_ok = false;
                    break;
                  }
                }
                MoEExpertBlock &pv = layer_slots[pad_slots[p]];
                // expert_geometry_for, not expert_layer_geometry --
                // pad_ids[p] is a real expert_id, not a stand-in for "the
                // layer" (2026-09 caller audit). Its return value is now
                // checked too -- previously wasn't, a real bug the audit
                // found: expert_layer_geometry can return false for a
                // mixed layer, which would have left gM/uM/dM
                // uninitialized right before *gM/*uM/*dM below.
                size_t gb, ub, db;
                const GGUFTensor *gM, *uM, *dM;
                if (cand_ok && engine.expert_geometry_for(layer, pad_ids[p], gb, ub, db, gM, uM, dM)) {
                  engine.populate_packed_slab(pv, pad_ids[p], gb, ub, db, *gM,
                                              *uM, *dM);
                  publish_ram_slot(pv);
                  pad_bytes += gb + ub + db;
                  perf().n_pad_reads_ok.fetch_add(1, std::memory_order_relaxed);
                } else {
                  revert_padding_claim(pv, layer, pad_ids[p]);
                  perf().n_pad_reads_failed.fetch_add(
                      1, std::memory_order_relaxed);
                }
                r += (size_t)cand_cnt;
              }
              if (pad_bytes > 0) {
                perf().bytes_disk_read_speculative.fetch_add(
                    (int64_t)pad_bytes, std::memory_order_relaxed);
              }
            }
          }

          if (all_ok) {
            size_t total_bytes = 0;
            bool geometry_ok = true;
            for (int i = 0; i < n_pending; ++i) {
              size_t gb, ub, db;
              const GGUFTensor *gM, *uM, *dM;
              // expert_geometry_for, not expert_layer_geometry --
              // pending[i].expert_id is real (2026-09 caller audit). Its
              // return value is now checked too -- previously wasn't, a
              // real bug the audit found: expert_layer_geometry can
              // return false for a mixed layer, which would have left
              // gM/uM/dM uninitialized right before *gM/*uM/*dM below.
              // On failure, stop publishing (uring_ok stays false) and
              // let the OMP fallback path just below handle this batch --
              // whatever slots already got published above are safe, real
              // successes; RAM-tier claim/publish is designed to tolerate
              // a slot being resolved more than once.
              if (!engine.expert_geometry_for(layer, pending[i].expert_id, gb, ub, db, gM, uM, dM)) {
                geometry_ok = false;
                break;
              }
              MoEExpertBlock &victim = layer_slots[pending[i].slot_idx];
              engine.populate_packed_slab(victim, pending[i].expert_id, gb, ub,
                                          db, *gM, *uM, *dM);
              publish_ram_slot(victim);
              pending[i].load_ms = batch_ms; // shared batch time; per-expert
                                             // split isn't meaningful here
              total_bytes += gb + ub + db;
              ram_swap_stats[swap_key(layer, pending[i].expert_id)].swap_in++;
            }
            if (geometry_ok) {
              perf().bytes_disk_read.fetch_add(total_bytes,
                                               std::memory_order_relaxed);
              uring_ok = true;
              perf().n_decode_uring_ok.fetch_add(1, std::memory_order_relaxed);
              LOG_DEBUG("[MOE-CACHE] MISS  layer=%d batch=%d experts "
                        "via=io_uring took=%.2fms "
                        "(%.2f MB, %.2f GB/s)\n",
                        layer, n_pending, batch_ms, total_bytes / 1e6,
                        batch_ms > 0 ? (total_bytes / 1e9) / (batch_ms / 1000.0)
                                     : 0.0);
            }
          }
        }
      }
      if (!uring_ok) {
        int nthreads = std::max(1, std::min(expert_fetch_threads, n_pending));
#pragma omp parallel for num_threads(nthreads) schedule(dynamic, 1)
        for (int i = 0; i < n_pending; ++i) {
          PendingLoad &pl = pending[i];
          MoEExpertBlock &victim = layer_slots[pl.slot_idx];
          auto t0 = std::chrono::high_resolution_clock::now();
          engine.load_expert_slab(layer, pl.expert_id, victim);
          pl.load_ms = std::chrono::duration<double, std::milli>(
                           std::chrono::high_resolution_clock::now() - t0)
                           .count();
        }
        // Logging, swap-in accounting, and publish_ram_slot all stay
        // sequential, outside the parallel region -- ram_swap_stats
        // is a plain unordered_map (unsafe to mutate concurrently),
        // and publish_ram_slot's own lock would otherwise contend
        // across every OMP thread for no benefit (each slot is
        // already exclusively owned by exactly one iteration).
        for (int i = 0; i < n_pending; ++i) {
          PendingLoad &pl = pending[i];
          MoEExpertBlock &victim = layer_slots[pl.slot_idx];
          ram_swap_stats[swap_key(layer, pl.expert_id)].swap_in++;
          publish_ram_slot(victim);
          LOG_DEBUG("[MOE-CACHE] MISS  layer=%d expert=%d slot=%zu "
                    "loaded_from_disk=%.2fms via=pread "
                    "(%.2f MB, %.2f GB/s) [batch=%d/%d]\n",
                    layer, pl.expert_id, pl.slot_idx, pl.load_ms,
                    victim.slab_capacity / 1e6,
                    pl.load_ms > 0
                        ? (victim.slab_capacity / 1e9) / (pl.load_ms / 1000.0)
                        : 0.0,
                    i + 1, n_pending);
        }
      }
    }
    PERF_END(ram_batch_phase_b, t_ram_batch_phase_b);

    // --- Phase C: resolve WaitOnOther entries from Phase A. Deliberately
    // AFTER Phase B (see this function's own comment on why) -- by now,
    // whatever prefetch worker claimed each of these has very often
    // already finished, so this wait is usually short or immediate. ---
    PERF_START(ram_batch_phase_c);
    // 2026-08-30: every out[w.k] assignment below gets the SAME pin-at-
    // resolution treatment as Phase A above, for the same reason -- see
    // this function's own header comment. Phase C resolves entries that
    // were Loading (owned by ANOTHER caller/thread) back in Phase A, so
    // by the time any of these branches sets out[w.k], that slot has
    // just become a normal published/resident slot with no protection
    // of its own -- exactly the state a concurrent victim-selection
    // scan is free to reclaim, same as any other resident expert,
    // unless pinned here.
    for (const WaitEntry &w : waiting) {
      MoEExpertBlock *slot =
          wait_for_ram_slot(engine, layer, w.expert_id,
                            /*pin_on_ready=*/true);
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
      // pin_on_hit=true -- same atomic-pin fix as Phase A's own Hit
      // branch above; see claim_or_wait's own parameter comment.
      RamClaimResult result =
          claim_or_wait(engine, layer, w.expert_id, RamClaimCaller::MainThread,
                        nullptr, /*pin_on_hit=*/true);
      if (result.status == RamClaimStatus::Hit) {
        out[w.k] = result.slot;
      } else if (result.status == RamClaimStatus::Claimed) {
        MoEExpertBlock &victim = *result.slot;
        // The slot is still Loading and therefore already excluded from
        // eviction. Pin before the load so the hold remains valid after the
        // subsequent publish_ram_slot transition.
        ram_pin(victim);
        engine.load_expert_slab(layer, w.expert_id, victim);
        ram_swap_stats[swap_key(layer, w.expert_id)].swap_in++;
        publish_ram_slot(victim);
        out[w.k] = &victim;
      } else {
        // Claimed by yet another in-flight fetch immediately after
        // the retry -- extremely unlikely back-to-back, but wait
        // once more rather than leaving out[w.k] unset.
        out[w.k] = wait_for_ram_slot(engine, layer, w.expert_id,
                                     /*pin_on_ready=*/true);
      }
    }
    PERF_END(ram_batch_phase_c, t_ram_batch_phase_c);
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
  void log_host_cache_heatmap(StreamingEngine &engine) const {
    auto &layers_experts = engine.layers_experts;
    int n_layers = (int)layers_experts.size();
    if (n_layers <= 0)
      return;
    std::vector<int> resident_count(n_layers, 0);
    std::vector<uint64_t> hit_sum(n_layers, 0);
    for (int l = 0; l < n_layers; ++l) {
      auto &slots = layers_experts[l];
      size_t cap =
          slots.empty()
              ? 0
              : std::min(slots.size(), (size_t)std::max(1, cap_for_layer(l)));
      for (size_t i = 0; i < cap; ++i) {
        if (slots[i].expert_id < 0)
          continue;
        resident_count[l]++;
        hit_sum[l] += slots[i].hit_count;
      }
    }
    uint64_t max_hits = 1;
    for (auto h : hit_sum)
      max_hits = std::max(max_hits, h);
    static const char ramp[] = " .:-=+*#%@";
    constexpr int ramp_len = sizeof(ramp) - 2;
    std::ostringstream os;
    os << "[RAM-HEATMAP]  per-layer resident-slot hit density (0-9, ' '=empty, "
          "'@'=hottest):\n[RAM-HEATMAP]  ";
    for (int l = 0; l < n_layers; ++l) {
      int idx = resident_count[l] == 0
                    ? 0
                    : 1 + (int)((double)hit_sum[l] / (double)max_hits *
                                (ramp_len - 1));
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
      LOG_STREAM_INFO()
          << "[CACHE-CHURN] Host RAM tier: no swap activity recorded.\n";
      return;
    }
    uint64_t total_in = 0, total_out = 0;
    for (const auto &kv : ram_swap_stats) {
      total_in += kv.second.swap_in;
      total_out += kv.second.swap_out;
    }
    LOG_STREAM_INFO()
        << "[CACHE-CHURN] Host RAM tier: " << ram_swap_stats.size()
        << " distinct experts touched, " << total_in
        << " decode-thread loads / " << total_out
        << " evictions this session (no pinning exists for this tier).\n"
        << "[CACHE-CHURN] NOTE: 'loads' counts ONLY the decode thread's own "
           "load paths -- publish_ram_slot, which every PREFETCH WORKER uses "
           "to complete a load, does not increment it, while 'evictions' "
           "counts every slot reclaim regardless of cause. So loads/evictions "
           "here is NOT a churn ratio and will look alarmingly lopsided "
           "(e.g. 1243 vs 7396) on any run with real prefetch activity -- "
           "cross-check against 'Prefetch real fetches' (14521 that same run) "
           "before concluding the tier is thrashing. Verified 2026-09-01 after "
           "this exact line prompted a false 6:1-churn investigation.\n";

    std::vector<std::pair<uint64_t, SwapStats>> sorted(ram_swap_stats.begin(),
                                                       ram_swap_stats.end());
    std::sort(sorted.begin(), sorted.end(), [](const auto &a, const auto &b) {
      return (a.second.swap_in + a.second.swap_out) >
             (b.second.swap_in + b.second.swap_out);
    });

    LOG_STREAM_INFO() << "[CACHE-CHURN] Top "
                      << std::min((size_t)top_n, sorted.size())
                      << " most-churned host-RAM experts:\n";
    for (int i = 0; i < top_n && i < (int)sorted.size(); ++i) {
      int layer = (int)(sorted[i].first >> 32),
          expert_id = (int)(sorted[i].first & 0xFFFFFFFFu);
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
  void report_slot_summary(StreamingEngine &engine) const {
    size_t filled = 0, capacity = 0, bytes_used = 0;
    for (int l = 0; l < (int)engine.layers_experts.size(); ++l) {
      auto &layer_slots = engine.layers_experts[l];
      size_t cap = layer_slots.empty()
                       ? 0
                       : std::min(layer_slots.size(),
                                  (size_t)std::max(1, cap_for_layer(l)));
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
                      << " slots filled (" << bytes_used / 1e9
                      << " GB of experts resident), ";
    if (byte_slots_per_layer_by_layer.empty() &&
        slots_per_layer_by_layer.empty()) {
      LOG_STREAM_INFO() << slots_per_layer << " slots/layer cap (uniform)\n";
    } else {
      int lo = std::numeric_limits<int>::max(), hi = 0;
      for (int l = 0; l < (int)engine.layers_experts.size(); ++l) {
        int v = base_cap_for_layer(l);
        if (v > 0) {
          lo = std::min(lo, v);
          hi = std::max(hi, v);
        }
      }
      if (lo == std::numeric_limits<int>::max())
        lo = 0;
      LOG_STREAM_INFO()
          << "[" << lo << ", " << hi
          << "] slots/layer cap (byte-aware/layer-adaptive active, "
          << "startup baseline was " << slots_per_layer
          << "/layer -- see apply_"
          << "layer_adaptive_capacity's comment if this baseline looks stale "
          << "vs. the current [RAM] LRU Cache log line)\n";
    }
    report_tier_overlap(engine);
  }

  // How many (layer, expert) pairs are resident in BOTH the RAM LRU tier
  // and the VRAM cache simultaneously, right now. Unlike alternatives (whose
  // VRAM-pinned tier and RAM-LRU tier are architecturally exclusive --
  // CUDA_RELEASE_HOST frees an expert's host copy once it's VRAM-
  // uploaded), ZeroMoE's tiers are NOT exclusive by design:
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
  void report_tier_overlap(StreamingEngine &engine) const {
    if (!vram_cache)
      return;
    std::unordered_set<uint64_t> vram_resident;
    vram_resident.reserve(vram_cache->expert_vram_cache.size());
    for (auto &loc : vram_cache->expert_vram_cache) {
      if (loc.layer_idx == -1)
        continue;
      vram_resident.insert(((uint64_t)(uint32_t)loc.layer_idx << 32) |
                           (uint32_t)loc.expert_id);
    }
    size_t overlap = 0, overlap_bytes = 0;
    for (int layer = 0; layer < (int)engine.layers_experts.size(); ++layer) {
      auto &layer_slots = engine.layers_experts[layer];
      size_t cap = layer_slots.empty()
                       ? 0
                       : std::min(layer_slots.size(),
                                  (size_t)std::max(1, cap_for_layer(layer)));
      for (size_t i = 0; i < cap; ++i) {
        if (layer_slots[i].expert_id == -1)
          continue;
        uint64_t key = ((uint64_t)(uint32_t)layer << 32) |
                       (uint32_t)layer_slots[i].expert_id;
        if (vram_resident.count(key)) {
          ++overlap;
          overlap_bytes += layer_slots[i].slab_capacity;
        }
      }
    }
    LOG_STREAM_INFO() << "[TIER-OVERLAP] " << overlap
                      << " expert(s) resident in BOTH RAM and VRAM "
                      << "simultaneously (" << overlap_bytes / 1e9
                      << " GB double-counted -- "
                      << "paid for in both tiers' totals above, not genuinely "
                         "distinct capacity)\n";
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
  void report_tier_map(StreamingEngine &engine) const {
    if (!vram_cache)
      return;
    int n_experts = engine.model_file.architecture_config.n_experts;
    if (n_experts <= 0)
      return;

    std::unordered_map<int, int> vram_count_by_layer;
    std::unordered_set<uint64_t> vram_resident;
    vram_resident.reserve(vram_cache->expert_vram_cache.size());
    for (auto &loc : vram_cache->expert_vram_cache) {
      if (loc.layer_idx == -1)
        continue;
      vram_resident.insert(((uint64_t)(uint32_t)loc.layer_idx << 32) |
                           (uint32_t)loc.expert_id);
      vram_count_by_layer[loc.layer_idx]++;
    }

    LOG_STREAM_INFO()
        << "[EXPERT-TIER-MAP] per-layer expert location (of " << n_experts
        << " experts/layer): vram / ram-only / cold (would need a disk read)\n";
    for (int layer = 0; layer < (int)engine.layers_experts.size(); ++layer) {
      auto &layer_slots = engine.layers_experts[layer];
      if (layer_slots.empty())
        continue; // not a MoE layer (leading dense)
      size_t cap = std::min(layer_slots.size(),
                            (size_t)std::max(1, cap_for_layer(layer)));

      int vram_n =
          vram_count_by_layer.count(layer) ? vram_count_by_layer[layer] : 0;
      int ram_only_n = 0;
      for (size_t i = 0; i < cap; ++i) {
        if (layer_slots[i].expert_id == -1)
          continue;
        uint64_t key = ((uint64_t)(uint32_t)layer << 32) |
                       (uint32_t)layer_slots[i].expert_id;
        if (!vram_resident.count(key))
          ++ram_only_n;
      }
      int cold_n = n_experts - vram_n - ram_only_n;
      LOG_STREAM_INFO() << "  layer=" << layer << "  vram=" << vram_n
                        << "  ram_only=" << ram_only_n
                        << "  cold=" << (cold_n > 0 ? cold_n : 0) << "\n";
    }
  }
};

// Shared pilot -> active-layer promotion contract. The caller owns the
// prediction policy (including any accuracy gate); this helper owns the
// existing speculative-fill policy, rank cutoff, multi-GPU routing, and the
// cache-side residency/upload checks. The graph never owns the slot lifetime:
// RAMSafetyManager/GpuTopology remain the single cache authority.
inline void promote_pilot_prediction_to_active_layer(
    StreamingEngine &engine, RAMSafetyManager &ram, int layer, int expert_id,
    int rank, int max_k) {
  if (!speculative_promote_to_active_layer_enabled() || max_k <= 0 ||
      rank < 0 || rank >= max_k)
    return;
  const int routed_gpu = active_layer_multi_gpu_enabled()
                             ? route_active_layer_gpu(rank, max_k)
                             : -1;
  ram.promote_ram_to_active_layer(engine, layer, expert_id, routed_gpu);
}
