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
    uint64_t last_used_clock = 0;
    uint32_t hit_count = 0;    // frequency signal for eviction scoring (power-law routing)
    int layer_idx = -1;
    int expert_id = -1;

    // Async promotion support: while an upload_expert_async() copy is in
    // flight, upload_ready is false and the slot must not be used for GPU
    // compute yet (device_ptr contents are only partially written).
    bool upload_ready = true;
    cudaEvent_t upload_event = nullptr;

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

    // Blended recency+frequency score used for eviction: a frequently-hit
    // expert resists eviction even after a burst of newer accesses touches
    // other slots (plain LRU would evict a hot-but-not-most-recent expert).
    uint64_t eviction_score() const {
        return last_used_clock + (uint64_t)hit_count * FREQ_BONUS_TICKS;
    }
    static constexpr uint64_t FREQ_BONUS_TICKS = 64;
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

    // Expert VRAM LRU cache: maps (layer, expert_id) -> location
    std::vector<ExpertLocation> expert_vram_cache;
    size_t expert_vram_cache_cap = 0;  // max entries
    size_t expert_slab_size = 0;       // bytes per expert slot
    uint64_t cache_clock = 0;

    // ========== Three-tier caching: Active layer area ==========
    // Active layer area: temporary workspace for current-layer experts
    // Reclaimed-per-request, never participates in LRU eviction
    uint8_t* active_layer_pool = nullptr;
    size_t active_layer_capacity = 0;     // Maximum experts active layer can hold
    std::vector<ExpertLocation> active_layer_slots;  // Pre-allocated slot metadata

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

    // Pinned host buffer for cross-device transfers
    float* h_xfer_pinned = nullptr;
    size_t h_xfer_cap = 0;

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
            cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
            cudaStreamCreate(&dev.compute_stream);
            cudaStreamCreate(&dev.copy_stream);
        }

        // Allocate pinned host transfer buffer (64 KB for small cross-GPU vectors)
        cudaMallocHost((void**)&h_xfer_pinned, 256 * 1024);
        h_xfer_cap = 256 * 1024;

        print_topology();
        return true;
    }

    ~GpuTopology() {
        for (auto& loc : expert_vram_cache) {
            if (loc.upload_event) { cudaSetDevice(loc.gpu_id); cudaEventDestroy(loc.upload_event); }
        }
        for (auto& dev : devices) {
            if (dev.compute_stream) { cudaSetDevice(dev.id); cudaStreamDestroy(dev.compute_stream); }
            if (dev.copy_stream) { cudaSetDevice(dev.id); cudaStreamDestroy(dev.copy_stream); }
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
        bounce_pool.resize(BOUNCE_POOL_SIZE);
        for (auto& bb : bounce_pool) {
            cudaError_t err = cudaHostAlloc((void**)&bb.host_ptr, slab_bytes, cudaHostAllocPortable);
            if (err != cudaSuccess) {
                LOG_STREAM_ERROR() << "[GPU] Bounce buffer cudaHostAlloc(" << slab_bytes / 1e6
                          << " MB) failed: " << cudaGetErrorString(err)
                          << " -- expert uploads fall back to direct pageable-source copy.\n";
                bb.host_ptr = nullptr;
                continue;
            }
            cudaEventCreate(&bb.in_flight);
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
        if (bb.in_flight) cudaEventSynchronize(bb.in_flight);
        return &bb;
    }

    void allocate_expert_cache(size_t expert_slab_bytes, int n_experts_total,
                                const EngineConfig* config = nullptr) {
        expert_slab_size = expert_slab_bytes;
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
            if (config) {
                for (auto& gc : config->gpus) {
                    if (gc.device_id == dev.id && gc.moe_cache_gb > 0) {
                        requested_bytes = (size_t)(gc.moe_cache_gb * 1e9);
                        break;
                    }
                }
            }
            size_t cache_bytes = requested_bytes > 0 ? std::min(requested_bytes, hw_cap) : hw_cap;

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
        expert_slab_size = expert_slab_bytes;
        if (expert_slab_bytes == 0) return;

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
        if (expert_slab_bytes == 0 || cap_bytes < expert_slab_bytes) {
            LOG_STREAM_INFO() << "[GPU] Dual-role expert cache on device " << devices[device_index].id
                      << " skipped -- cap (" << cap_bytes / 1e9 << " GB) smaller than one expert slab ("
                      << expert_slab_bytes / 1e9 << " GB).\n";
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
    ExpertLocation* evict_lru(size_t needed_bytes, int* out_old_layer = nullptr, int* out_old_expert = nullptr) {
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
        (void)needed_bytes; // All slots are fixed-size in slab allocator
        ExpertLocation* victim = nullptr;
        // Iterates unpinned_slot_indices, NOT expert_vram_cache directly --
        // see that vector's own comment. Permanently pinned slots are
        // excluded from this set at the source (pin_expert removes them),
        // so this loop never spends a single iteration on one of the 600+
        // pinned slots a real run can have, unlike before.
        for (size_t _idx : unpinned_slot_indices) {
            auto& loc = expert_vram_cache[_idx];
            if (loc.memcpy_broken) continue;  // permanently excluded -- see its own comment
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
            if (!victim || loc.eviction_score() < victim->eviction_score()) victim = &loc;
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
    ExpertLocation* upload_expert(int layer, int expert_id,
                                   const uint8_t* host_data, size_t bytes) {
        if (expert_vram_cache_cap == 0 || !host_data || bytes == 0) return nullptr;
        GPU_TOPO_TIMED_LOCK(lock);

        // Check if already cached
        ExpertLocation* existing = find_expert(layer, expert_id);
        if (existing) return existing;

        // Get a slot (evict LRU if needed)
        int old_layer = -1, old_expert = -1;
        ExpertLocation* slot = evict_lru(bytes, &old_layer, &old_expert);
        if (!slot || !slot->device_ptr) {
            // Previously silent -- a real run showed this path (or the
            // cudaMemcpy failure below) responsible for a startup preload
            // losing ~99% of its attempts with zero diagnostic evidence
            // either way. Should be near-impossible on an otherwise-empty
            // cache (every slot pinned/temp-pinned/upload-pending
            // simultaneously) -- if this fires during startup preload,
            // something upstream (vault_gpus/unpinned_slot_indices
            // bookkeeping, likely around a dual-role reindex) is broken.
            perf().n_upload_expert_no_victim.fetch_add(1, std::memory_order_relaxed);
            LOG_ERROR("[MOE-VRAM]  upload_expert: evict_lru found NO victim for layer=%d expert=%d "
                      "(cache_cap=%zu, unpinned_slots=%zu) -- every candidate slot is pinned, "
                      "temp-pinned, or upload-pending, or the pool is empty/misconfigured\n",
                      layer, expert_id, expert_vram_cache_cap, unpinned_slot_indices.size());
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
        if (on_vram_evict && old_layer != -1) {
            on_vram_evict(old_layer, old_expert, slot->device_ptr, slot->size, slot->gpu_id);
        }

        // Copy host → device (synchronous for correctness). Stages through
        // a pinned bounce buffer first (see init_bounce_pool) so the real
        // DMA reads from page-locked memory instead of host_data's pageable
        // RAM-tier slab_buffer -- avoids the CUDA driver's own internal
        // pageable-copy staging, which is slower and not truly overlapped.
        size_t copy_bytes = std::min(bytes, slot->size);
        cudaSetDevice(slot->gpu_id);
        BounceBuffer* bb = acquire_bounce_buffer();
        const uint8_t* dma_src = host_data;
        if (bb) {
            memcpy(bb->host_ptr, host_data, copy_bytes);
            dma_src = bb->host_ptr;
        }
        PERF_START(gpu_sync_upload);
        cudaError_t err = cudaMemcpy(slot->device_ptr, dma_src, copy_bytes, cudaMemcpyHostToDevice);
        PERF_END(gpu_sync_upload, t_gpu_upload);
        if (bb && bb->in_flight) cudaEventRecord(bb->in_flight, 0);
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
            // Permanently exclude this slot from future eviction candidacy --
            // a real run showed evict_lru re-selecting the SAME still-"empty"
            // slot on every subsequent call (a failed write never marks a
            // slot occupied, so it stays the "first empty slot" forever),
            // funneling ~1800 preload attempts into one broken address
            // instead of ever reaching the pool's other, presumably-healthy
            // slots. Whatever is wrong with this address (bad device_ptr
            // computation, or a genuinely faulty VRAM region) won't fix
            // itself by retrying -- see ExpertLocation::memcpy_broken.
            slot->memcpy_broken = true;
            LOG_ERROR("[MOE-VRAM]  upload_expert: permanently excluding slot at dst=%p from future "
                      "eviction candidacy (gpu=%d) -- this address has now failed at least once\n",
                      (void*)slot->device_ptr, slot->gpu_id);
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
        ExpertLocation* slot = evict_lru(bytes, &old_layer, &old_expert);
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
        if (on_vram_evict && old_layer != -1) {
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
    ExpertLocation* upload_expert_async(int layer, int expert_id,
                                         const uint8_t* host_data, size_t bytes) {
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
        if (existing) return existing;

        ExpertLocation* slot = evict_lru(bytes);
        if (!slot || !slot->device_ptr) return nullptr;

        size_t copy_bytes = std::min(bytes, slot->size);
        GpuDevice& dev = devices[slot->gpu_id];
        cudaSetDevice(dev.id);
        slot->upload_ready = false;
        // Same pinned bounce-buffer staging as upload_expert -- see its
        // comment. Ping-ponging across bounce_pool's N buffers is what
        // gives this async path real double buffering: while buffer[i]'s
        // DMA is still draining on copy_stream, the NEXT call's memcpy into
        // buffer[i+1] can proceed immediately without waiting on it.
        BounceBuffer* bb = acquire_bounce_buffer();
        const uint8_t* dma_src = host_data;
        if (bb) {
            memcpy(bb->host_ptr, host_data, copy_bytes);
            dma_src = bb->host_ptr;
        }
        cudaMemcpyAsync(slot->device_ptr, dma_src, copy_bytes, cudaMemcpyHostToDevice, dev.copy_stream);
        if (slot->upload_event) cudaEventRecord(slot->upload_event, dev.copy_stream);
        if (bb && bb->in_flight) cudaEventRecord(bb->in_flight, dev.copy_stream);

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

    // Upload expert to active layer area (temporary, never cached)
    // Returns slot pointer or nullptr if area is full/allocation failed
    // This is decode-thread-only and never participates in LRU eviction
    ExpertLocation* upload_expert_active(int layer, int expert_id,
                                        const uint8_t* host_data, size_t bytes);

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
    ExpertLocation* promote_active_to_main(int layer, int expert_id,
                                            const uint8_t* device_src, size_t bytes,
                                            const std::function<void(int, int)>& on_promoted = nullptr);

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
    void assign_roles() {
        int n = devices.size();
        if (n == 0) return;

        if (n == 1) {
            // Profile A: single GPU does everything
            profile = HardwareProfile::SingleGPU;
            devices[0].role = GpuRole::CoreEngine;
            core_gpu = 0;
            vault_gpus.push_back(0); // same device serves as vault too
        } else {
            // Multi-GPU: honor an explicit config choice if one was given
            // (forced_core_device_id, set by main.cpp from
            // config.gpus[dense_gpu_idx].device_id) -- otherwise fall back
            // to the auto-detect heuristic this function always used
            // before: assign Core to newest arch (or smallest VRAM if same
            // arch), Vault to the rest.
            int best_core = -1;
            if (forced_core_device_id >= 0) {
                for (int i = 0; i < n; ++i) {
                    if (devices[i].id == forced_core_device_id) { best_core = i; break; }
                }
                if (best_core < 0) {
                    LOG_STREAM_WARN() << "[GPU] forced_core_device_id=" << forced_core_device_id
                              << " does not match any enumerated device -- falling back to "
                                 "SM-architecture/VRAM auto-detect for CoreEngine assignment\n";
                }
            }
            if (best_core < 0) {
                best_core = 0;
                for (int i = 1; i < n; ++i) {
                    bool newer = devices[i].sm_major > devices[best_core].sm_major;
                    bool same_gen_less_vram = (devices[i].sm_major == devices[best_core].sm_major &&
                                               devices[i].total_vram < devices[best_core].total_vram);
                    if (newer || same_gen_less_vram) best_core = i;
                }
            }

            devices[best_core].role = GpuRole::CoreEngine;
            core_gpu = best_core;

            for (int i = 0; i < n; ++i) {
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
        size_t bytes_used = filled * expert_slab_size;
        LOG_STREAM_INFO() << "[VRAM-SLOTS] " << filled << " / " << expert_vram_cache_cap
                           << " slots filled (" << bytes_used / 1e9 << " GB of experts resident), "
                           << pinned_n << " pinned, " << mid_upload << " mid-upload\n";
    }
};
