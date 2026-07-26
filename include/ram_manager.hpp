#pragma once
#include "engine.hpp"
#include "perf_timer.hpp"
#include "gpu_topology.hpp"
#include "async_prefetcher.hpp"  // LockFreePrefetcher (consume_staged in the miss path)
#include "logger.hpp"
#include "kquant_dequant.hpp"  // gpu_type_supported
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
#include <fcntl.h>
#include <unistd.h>
#include <cuda_runtime.h>  // demote_from_vram's D2H salvage copy; already transitively
                           // pulled in via gpu_topology.hpp, included directly here too
                           // so the dependency isn't implicit
#ifdef __APPLE__
#include <mach/mach.h>
#include <sys/sysctl.h>
#endif

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
    // Topology-enabled prefetcher whose staging pool the cache-MISS path
    // consumes (Phase A "PIPE" overlap). Set by main.cpp alongside vram_cache.
    // nullptr = miss path always falls back to synchronous load_expert_slab.
    LockFreePrefetcher* prefetcher = nullptr;

    // io_uring ring OWNED BY THE DECODE THREAD ITSELF -- distinct from the
    // prefetcher's own per-worker-thread rings (io_uring_backend.hpp: a
    // Ring is not shareable across threads). Used by get_experts_batch's
    // Phase B to batch an ENTIRE layer's pending cache-miss set into ONE
    // io_uring submission (colibri's own real miss-servicing design,
    // glm.c's pipe_dispatch/URING_REQ_MAX) instead of N OS threads each
    // blocking on their own pread -- this is the SYNCHRONOUS,
    // decode-critical path that handles the large majority of every real
    // run's misses (the prefetcher's PIPE only ever serves a small
    // fraction of them), so it's where the actual disk-throughput ceiling
    // lives, unlike the earlier (Prefetcher-only, deliberately lower-risk)
    // io_uring scope. init_decode_uring() sets this up; falls back to the
    // existing OMP-parallel load_expert_slab loop, unconditionally
    // correct, whenever the ring is unavailable OR a batch fails for any
    // reason (short read, unsupported layer format, etc.).
    io_uring_backend::Ring decode_uring_ring;
    bool use_decode_uring = false;

    void init_decode_uring(bool enable) {
        if (!enable) return;
        use_decode_uring = decode_uring_ring.init();
        if (use_decode_uring) {
            LOG_STREAM_INFO() << "[IO-URING] Enabled on the decode thread's own ring -- "
                                  "batching Phase B's per-layer miss set through io_uring.\n";
        } else {
            LOG_STREAM_WARN() << "[IO-URING] io_uring=true requested but the decode-thread "
                                  "Ring::init() failed -- Phase B falls back to the OMP-parallel "
                                  "pread loop.\n";
        }
    }

    // Set by layers_forward_gguf at the top of every call (S > 1 = prefill,
    // S == 1 = decode) purely so the miss path below can split its PIPE
    // hit/miss counters by phase -- diagnostic only, changes no behavior.
    // Added to test a specific hypothesis: the pilot prefetcher predicts
    // from only the LAST token's hidden state (execution_layer.cpp), which
    // structurally can't cover prefill's full per-layer expert fan-out
    // across ALL prompt tokens, so a near-zero aggregate PIPE-served rate
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

        // Skip if already RAM-resident -- nothing to gain from a duplicate,
        // and it would just waste a D2H copy plus displace some OTHER
        // expert's RAM slot for no benefit.
        for (size_t si = 0; si < cap; ++si) {
            if (layer_slots[si].expert_id == expert_id) {
                perf().n_vram_evict_demote_skipped.fetch_add(1, std::memory_order_relaxed);
                return;
            }
        }

        // Only the packed-slab (ffn_*_exps) layout is supported -- mirrors
        // the same guard the PIPE consume_staged path already uses. A
        // legacy per-expert-tensor model has no single (gb,ub,db) geometry
        // to reconstruct host_ptr/meta offsets from, so demotion is a
        // silent (counted, not logged) no-op for it, matching how the PIPE
        // fast path already degrades for that case.
        size_t gb, ub, db;
        const GGUFTensor *gM, *uM, *dM;
        if (!engine.packed_slab_geometry(layer, gb, ub, db, gM, uM, dM)) {
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
        // that same rule on this call. copy_still_pending() takes priority
        // over both: a slot with an unconfirmed async H2D copy still
        // reading its OLD slab_buffer must never be picked (see
        // MoEExpertBlock::pending_copy_event) -- scored as UINT64_MAX so
        // it's only ever chosen if literally every candidate has this
        // problem, an extremely rare case the defensive sync below still
        // handles safely.
        size_t victim_idx = 0;
        uint64_t victim_score = layer_slots[0].copy_still_pending() ? UINT64_MAX
            : (layer_slots[0].also_in_vram ? 0 : eviction_score(layer_slots[0]));
        for (size_t i = 1; i < cap; ++i) {
            uint64_t score = layer_slots[i].copy_still_pending() ? UINT64_MAX
                : (layer_slots[i].also_in_vram ? 0 : eviction_score(layer_slots[i]));
            if (score < victim_score) { victim_score = score; victim_idx = i; }
        }
        MoEExpertBlock& victim = layer_slots[victim_idx];
        int displaced = victim.expert_id;

        // Defensive last resort: only actually blocks if EVERY candidate
        // above scored UINT64_MAX (all pending) and this one still got
        // picked -- cudaEventSynchronize on an event that's very likely
        // already signaled by now returns near-instantly in the common
        // case this branch is even entered.
        if (victim.pending_copy_event) {
            cudaEventSynchronize(victim.pending_copy_event);
            victim.pending_copy_event = nullptr;
        }

        cudaSetDevice(gpu_id);
        engine.ensure_slab_capacity(victim, total);
        cudaError_t err = cudaMemcpy(victim.slab_buffer, device_ptr, total, cudaMemcpyDeviceToHost);
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
            perf().n_vram_evict_demote_failed.fetch_add(1, std::memory_order_relaxed);
            return;
        }

        if (displaced != -1) ram_swap_stats[swap_key(layer, displaced)].swap_out++;
        engine.populate_packed_slab(victim, expert_id, gb, ub, db, *gM, *uM, *dM);
        victim.also_in_vram = false;  // it just left VRAM -- this RAM copy is the only one now
        victim.last_used_clock = ++global_clock;
        victim.hit_count = 0;
        ram_swap_stats[swap_key(layer, expert_id)].swap_in++;
        perf().n_vram_evict_demoted.fetch_add(1, std::memory_order_relaxed);

        LOG_DEBUG("[MOE-RAM] DEMOTE layer=%d expert=%d slot=%zu displaced=%d (%.2f MB salvaged from VRAM via D2H)\n",
                  layer, expert_id, victim_idx, displaced, total / 1e6);
    }

    MoEExpertBlock* get_expert(StreamingEngine& engine, int layer, int expert_id) {
        auto& layer_slots = engine.layers_experts[layer];

        // graph_initializer.cpp always allocates n_experts (e.g. 256) slots
        // per layer up front, for slot-address stability -- but that is NOT
        // the real cache capacity. Bound the active/scanned/evictable range
        // to slots_per_layer (the actual RAM-budget-derived cap from
        // calculate_safe_cache_capacity) so eviction kicks in once the real
        // budget is full, instead of only once ALL n_experts slots for a
        // layer have each been individually touched -- which, uncapped,
        // silently grows host RAM usage far past the configured budget
        // (every distinct expert ever routed to gets a permanent slot until
        // the array is entirely full) and was the actual cause of the
        // escalating per-token slowdown / OS-level swapping this was added
        // to diagnose, not any single compute kernel being slow.
        size_t cap = layer_slots.empty() ? 0
            : std::min(layer_slots.size(), (size_t)std::max(1, cap_for_layer(layer)));

        // --- Step 1: Check host RAM LRU cache ---
        for (size_t si = 0; si < cap; ++si) {
            auto& slot = layer_slots[si];
            if (slot.expert_id == expert_id) {
                slot.last_used_clock = ++global_clock;
                if (slot.hit_count < UINT32_MAX) slot.hit_count++;
                if (++hits_since_decay >= HIT_DECAY_INTERVAL) decay_hit_counts(engine);
                perf().n_expert_cache_hit.fetch_add(1, std::memory_order_relaxed);

                if (verbose_cache_log) {
                    LOG_DEBUG("[MOE-CACHE] HIT   layer=%d expert=%d hit_count=%u\n",
                              layer, expert_id, slot.hit_count);
                }

                // Async, non-blocking promotion to VRAM — gated on proven
                // hotness (re-hit threshold) so a single one-off touch
                // doesn't evict a more consistently useful expert from the
                // small VRAM cache, and never stalls this token's critical
                // path on a PCIe copy. The threshold is configurable via
                // EngineConfig.promote_hit_threshold.
                //
                // ALSO gated on every projection's quant type actually having
                // a GPU kernel: expert_ffn only takes the GPU path when
                // gate/up/down are ALL GPU-supported (all-or-nothing), so
                // promoting an expert with even one unsupported projection
                // (e.g. a Dynamic-quant model where down=IQ3_XXS but
                // gate/up=IQ2_XXS) wastes a VRAM cache slot and PCIe H2D
                // bandwidth on data that will never actually be computed on
                // GPU — it just displaces a slot a genuinely GPU-capable
                // expert could have used.
                bool gpu_capable = gpu_type_supported(slot.gate_proj.meta.type)
                                 && gpu_type_supported(slot.up_proj.meta.type)
                                 && gpu_type_supported(slot.down_proj.meta.type);
                if (vram_cache && slot.slab_buffer && gpu_capable && slot.hit_count >= promote_hit_threshold) {
                    // find_expert has side effects (bumps the VRAM slot's own
                    // hit_count/last_used_clock), so call it once and branch
                    // on the result rather than calling it again below.
                    ExpertLocation* resident = vram_cache->find_expert(layer, expert_id);
                    if (!resident) {
                        LOG_DEBUG("[MOE-VRAM]  PROMOTE layer=%d expert=%d hit_count=%u -> async H2D upload (%.2f MB)\n",
                                  layer, expert_id, slot.hit_count, slot.slab_capacity / 1e6);
                        ExpertLocation* promoted = vram_cache->upload_expert_async(
                            layer, expert_id, slot.slab_buffer, slot.slab_capacity);

                        // FIX: Three-tier caching - MOVE instead of COPY
                        // Immediately free RAM slot after VRAM promotion starts
                        // This eliminates the 7.7GB tier overlap waste
                        if (move_on_promotion) {
                            slot.expert_id = -1;  // Mark RAM slot as free
                            slot.last_used_clock = 0;
                            slot.hit_count = 0;
                            slot.also_in_vram = false;
                            // Keep slab_buffer allocated (slot.slab_buffer remains valid)
                            // but mark it as empty so the slot can be reused
                            slot.slab_capacity = 0;
                            // Real bug this fixes: upload_expert_async above is
                            // fire-and-forget (async cudaMemcpyAsync still
                            // reading FROM slab_buffer on a background stream)
                            // -- clearing this slot's identity here makes it
                            // immediately eligible for a LATER miss to reuse
                            // (overwrite/free+realloc) slab_buffer before that
                            // copy lands, a genuine use-after-free/WAR race.
                            // Recording the destination VRAM slot's own
                            // upload_event here lets every victim-selection
                            // site (MoEExpertBlock::copy_still_pending) refuse
                            // to reuse this slot until the copy is confirmed
                            // done.
                            slot.pending_copy_event = promoted ? promoted->upload_event : nullptr;
                            LOG_DEBUG("[MOE-VRAM]  MOVE: freed RAM slot immediately after promotion (layer=%d expert=%d)\n",
                                      layer, expert_id);
                            // Track three-tier metrics
                            perf().n_move_on_promotion.fetch_add(1, std::memory_order_relaxed);
                        } else {
                            // Track COPY operations (fallback for testing)
                            perf().n_copy_on_promotion.fetch_add(1, std::memory_order_relaxed);
                        }
                    } else {
                        // find_expert only ever returns non-null after
                        // confirming (polling upload_ready/cudaEventQuery
                        // internally) the H2D copy has genuinely completed
                        // -- so this is the safe, confirmed point to mark
                        // this RAM slot as redundant capacity (Point 1 fix:
                        // reclaim double-stored capacity promptly instead of
                        // waiting for unrelated LRU pressure -- see
                        // MoEExpertBlock::also_in_vram's own comment).
                        // Deliberately NOT set in the upload_expert_async
                        // branch above: that copy may still be in flight,
                        // and this slot's slab_buffer is its own source --
                        // marking it redundant before the copy finishes
                        // would be unsafe once anything acts on the flag.
                        slot.also_in_vram = true;
                        if (pin_threshold > 0 && slot.hit_count >= pin_threshold) {
                            // Already VRAM-resident AND proven hot enough within
                            // THIS conversation (no cross-prompt profile needed)
                            // -- pin it so evict_lru never reactively kicks it
                            // out for a one-hit-wonder later in the same
                            // generation. Phase A "stop evicting
                            // hot experts within a conversation".
                            vram_cache->pin_expert(layer, expert_id);
                        }
                    }
                }
                return &slot;
            }
        }

        // --- Step 2: Check GPU VRAM cache (if enabled) ---
        // If the expert is in VRAM but got evicted from host RAM, we still need
        // the host copy for the current CPU execution path. When GPU execution is
        // fully wired, this path would return a device pointer directly.
        // For now, VRAM cache serves as a "hot store" for future GPU dispatch.

        perf().n_expert_cache_miss.fetch_add(1, std::memory_order_relaxed);
        if (in_prefill) perf().n_expert_cache_miss_prefill.fetch_add(1, std::memory_order_relaxed);
        else perf().n_expert_cache_miss_decode.fetch_add(1, std::memory_order_relaxed);

        // --- Step 3: Evict lowest recency+frequency score and load from disk ---
        // A slot confirmed also_in_vram (Point 1 fix) is treated as the
        // lowest-priority ("evict first") candidate regardless of its own
        // recency/hit_count -- its data is genuinely redundant capacity,
        // already safely resident and dispatch-ready in the VRAM tier.
        // copy_still_pending() takes priority over also_in_vram: a slot
        // with an unconfirmed async H2D copy still reading its OLD
        // slab_buffer must never be picked (see MoEExpertBlock::
        // pending_copy_event) -- scored as UINT64_MAX, only chosen if
        // every candidate has this problem, which the defensive sync
        // below still handles safely.
        int lru_idx = 0;
        uint64_t lru_score = layer_slots[0].copy_still_pending() ? UINT64_MAX
            : (layer_slots[0].also_in_vram ? 0 : eviction_score(layer_slots[0]));
        for (size_t i = 1; i < cap; ++i) {
            uint64_t score = layer_slots[i].copy_still_pending() ? UINT64_MAX
                : (layer_slots[i].also_in_vram ? 0 : eviction_score(layer_slots[i]));
            if (score < lru_score) { lru_score = score; lru_idx = (int)i; }
        }

        MoEExpertBlock& victim = layer_slots[lru_idx];
        int evicted_expert_id = victim.expert_id;
        if (evicted_expert_id != -1) {
            ram_swap_stats[swap_key(layer, evicted_expert_id)].swap_out++;
        }
        victim.also_in_vram = false;  // fresh identity in this slot momentarily
        // Defensive last resort (see demote_from_vram's identical comment
        // above) -- only actually blocks if every candidate was pending.
        if (victim.pending_copy_event) {
            cudaEventSynchronize(victim.pending_copy_event);
            victim.pending_copy_event = nullptr;
        }

        if (victim.expert_id != -1 && engine.drop_expert_pages) {
            // Drop each projection's OWN pages by name -- the old
            // (down.end - gate.offset) span underflowed size_t on a scattered
            // repack layout (down<gate<up) and advised a ~SIZE_MAX range.
            std::string base = "blk." + std::to_string(layer) + ".";
            std::string es = std::to_string(victim.expert_id);
            for (const char* proj : {"ffn_gate", "ffn_up", "ffn_down"}) {
                auto it = engine.model_file.tensors.find(base + proj + "." + es + ".weight");
                if (it != engine.model_file.tensors.end())
                    posix_fadvise(engine.model_file.fd_for_tensor(it->second),
                                  it->second.offset, it->second.size_bytes, POSIX_FADV_DONTNEED);
            }
        }

        // Phase A "PIPE" overlap: try to serve this miss from the
        // prefetcher's staging pool (already pread by the worker thread
        // while a previous layer computed) before falling back to a
        // synchronous pread on THIS (the decode) thread. The prefetcher
        // only ever produces the packed-format layout, so this path is
        // skipped for legacy per-expert-tensor models (packed_slab_geometry
        // returns false) or if `prefetcher` isn't wired (nullptr).
        PERF_START(expert_disk_load);
        bool served_from_pipe = false;
        if (prefetcher) {
            size_t gb, ub, db;
            const GGUFTensor *gM, *uM, *dM;
            if (engine.packed_slab_geometry(layer, gb, ub, db, gM, uM, dM)) {
                size_t total = gb + ub + db;
                engine.ensure_slab_capacity(victim, total);
                PERF_START(consume);
                size_t got_bytes = 0;
                bool ok = prefetcher->consume_staged(layer, expert_id, victim.slab_buffer, victim.slab_capacity, &got_bytes);
                PERF_END(consume, t_consume_staged);
                if (ok && got_bytes == total) {
                    engine.populate_packed_slab(victim, expert_id, gb, ub, db, *gM, *uM, *dM);
                    served_from_pipe = true;
                    perf().n_pipe_hits.fetch_add(1, std::memory_order_relaxed);
                    if (in_prefill) perf().n_pipe_hits_prefill.fetch_add(1, std::memory_order_relaxed);
                    else perf().n_pipe_hits_decode.fetch_add(1, std::memory_order_relaxed);
                } else if (!ok && prefetcher->has_pending_hint(layer, expert_id)) {
                    perf().n_pipe_miss_still_queued.fetch_add(1, std::memory_order_relaxed);
                }
            }
        }
        if (!served_from_pipe) {
            engine.load_expert_slab(layer, expert_id, victim);
        }
        double load_ms = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - _perf_expert_disk_load).count();
        victim.last_used_clock = ++global_clock;
        victim.hit_count = 0;  // fresh identity in this slot, reset frequency signal
        ram_swap_stats[swap_key(layer, expert_id)].swap_in++;

        LOG_DEBUG("[MOE-CACHE] MISS  layer=%d expert=%d slot=%d evicted=%d loaded_from_disk=%.2fms via=%s (%.2f MB, %.2f GB/s)\n",
                  layer, expert_id, lru_idx, evicted_expert_id, load_ms,
                  served_from_pipe ? "PIPE" : "pread",
                  victim.slab_capacity / 1e6,
                  load_ms > 0 ? (victim.slab_capacity / 1e9) / (load_ms / 1000.0) : 0.0);

        // Note: no eager VRAM promotion here — a cold disk load hasn't proven
        // hot yet (hit_count starts at 0). It will be promoted asynchronously
        // by Step 1 above once it's actually re-hit (PROMOTE_HIT_THRESHOLD).
        return &victim;
    }

    // Batched sibling of get_expert(), for fetching one token's ENTIRE topk
    // selection within a single layer at once. Motivated by a direct
    // measurement: buffered reads scale ~8.5x from 1->4 concurrent
    // threads on this hardware (1.53 -> 13.12 GB/s), which single-expert
    // get_expert()'s strictly-sequential one-pread-at-a-time loop (the
    // caller's original `for (k) get_expert(...)`) never exploits — every
    // miss pays its full read latency before the next one even starts.
    //
    // Split into two phases:
    //   Phase A (sequential): for each k, either service a host-cache HIT
    //     (identical logic to get_expert's Step 1 — no I/O, must stay
    //     single-threaded since it's the only part touching shared LRU
    //     state) or CLAIM a victim slot for a miss (Step 3's eviction pick
    //     + identity/recency stamp, moved BEFORE the actual read) and try
    //     the PIPE staged-consume fast path (a mutex+memcpy, not a real
    //     disk read, so it stays sequential too — cheap either way).
    //   Phase B (parallel): for whatever PIPE didn't already have staged,
    //     run the actual pread via engine.load_expert_slab across
    //     expert_fetch_threads OMP threads. Safe with ZERO extra
    //     synchronization because Phase A already gave each miss its OWN,
    //     explicitly-distinct slot (see claimed_this_batch below) before
    //     any thread starts reading
    //     pool uses (claim identity serially, then load in parallel with
    //     no further per-job locking).
    //
    // out[k] is valid (non-null, correct identity) the moment this call
    // returns for every k — Phase B has already completed by then, so
    // callers see the exact same synchronous-completion contract
    // get_expert() has always had, just serviced faster.
    void get_experts_batch(StreamingEngine& engine, int layer, const int* expert_ids, int topk,
                           MoEExpertBlock** out) {
        auto& layer_slots = engine.layers_experts[layer];
        size_t cap = layer_slots.empty() ? 0
            : std::min(layer_slots.size(), (size_t)std::max(1, cap_for_layer(layer)));

        struct PendingLoad { size_t slot_idx; int expert_id; int evicted_expert_id; int k; double load_ms; };
        std::vector<PendingLoad> pending;
        pending.reserve(topk);
        // Slots this batch has already claimed as a miss's victim -- an
        // EXPLICIT correctness guard (not just probabilistic) against two
        // misses in the same batch landing on the same slot. In practice
        // this is already astronomically unlikely on its own: claiming
        // stamps last_used_clock to a fresh ++global_clock immediately, so
        // a just-claimed slot's eviction_score becomes the local maximum
        // and won't be picked again by a later k's scan. But two Phase-B
        // threads writing the same slab_buffer concurrently would silently
        // corrupt data with no crash to catch it, so this is enforced
        // directly rather than relied upon.
        std::vector<size_t> claimed_this_batch;
        claimed_this_batch.reserve(topk);

        // --- Phase A: sequential hit-check + miss-claim (+ PIPE attempt) ---
        for (int k = 0; k < topk; ++k) {
            int expert_id = expert_ids[k];
            bool found = false;
            for (size_t si = 0; si < cap; ++si) {
                auto& slot = layer_slots[si];
                if (slot.expert_id == expert_id) {
                    // -- identical to get_expert's Step 1 --
                    slot.last_used_clock = ++global_clock;
                    if (slot.hit_count < UINT32_MAX) slot.hit_count++;
                    if (++hits_since_decay >= HIT_DECAY_INTERVAL) decay_hit_counts(engine);
                    perf().n_expert_cache_hit.fetch_add(1, std::memory_order_relaxed);
                    if (verbose_cache_log) {
                        LOG_DEBUG("[MOE-CACHE] HIT   layer=%d expert=%d hit_count=%u\n",
                                  layer, expert_id, slot.hit_count);
                    }
                    bool gpu_capable = gpu_type_supported(slot.gate_proj.meta.type)
                                     && gpu_type_supported(slot.up_proj.meta.type)
                                     && gpu_type_supported(slot.down_proj.meta.type);
                    if (vram_cache && slot.slab_buffer && gpu_capable && slot.hit_count >= promote_hit_threshold) {
                        ExpertLocation* resident = vram_cache->find_expert(layer, expert_id);
                        if (!resident) {
                            LOG_DEBUG("[MOE-VRAM]  PROMOTE layer=%d expert=%d hit_count=%u -> async H2D upload (%.2f MB)\n",
                                      layer, expert_id, slot.hit_count, slot.slab_capacity / 1e6);
                            ExpertLocation* promoted = vram_cache->upload_expert_async(
                                layer, expert_id, slot.slab_buffer, slot.slab_capacity);

                            // Same MOVE-vs-COPY fix as get_expert's Step 1 above --
                            // this batched hit path was missing it entirely, which
                            // meant every re-hit promotion routed through
                            // get_experts_batch (the only path prefill/decode's
                            // topk fetch actually calls) left the just-promoted
                            // expert double-resident in RAM AND VRAM forever
                            // (n_move_on_promotion/n_copy_on_promotion stayed at 0
                            // even with thousands of real promotions), silently
                            // wasting RAM-tier slots on redundant capacity -- see
                            // [TIER-OVERLAP] in the perf report.
                            if (move_on_promotion) {
                                slot.expert_id = -1;
                                slot.last_used_clock = 0;
                                slot.hit_count = 0;
                                slot.also_in_vram = false;
                                slot.slab_capacity = 0;
                                // See get_expert's identical comment -- upload_expert_async
                                // is fire-and-forget, so this slot must stay
                                // protected from reuse until the copy is
                                // confirmed done (MoEExpertBlock::copy_still_pending).
                                slot.pending_copy_event = promoted ? promoted->upload_event : nullptr;
                                LOG_DEBUG("[MOE-VRAM]  MOVE: freed RAM slot immediately after promotion (layer=%d expert=%d)\n",
                                          layer, expert_id);
                                perf().n_move_on_promotion.fetch_add(1, std::memory_order_relaxed);
                                // REAL BUG FIX: this slot now looks "empty"
                                // (expert_id=-1, last_used_clock=0), giving it
                                // eviction_score()==0 -- the guaranteed lowest
                                // possible score, making it the MOST
                                // attractive victim for a LATER k's miss-
                                // handling later in this SAME Phase A loop.
                                // But out[k] was already pointed at this exact
                                // slot (a few lines below) for THIS k's own
                                // Phase 3 dispatch, which runs AFTER this
                                // whole function returns -- a later miss
                                // reusing it (ensure_slab_capacity's free+
                                // realloc, then loading a DIFFERENT expert's
                                // bytes into gate_proj/up_proj/down_proj/
                                // slab_buffer) would corrupt or replace the
                                // data THIS k's own out[k] pointer expects to
                                // read, entirely undetected -- a genuine
                                // same-thread, same-batch read-after-free/
                                // wrong-data hazard, not just a missed GPU-
                                // dispatch opportunity. claimed_this_batch
                                // already exists to prevent exactly this
                                // class of collision for miss-claimed slots;
                                // a hit-driven promotion's vacated slot needs
                                // the SAME protection, since victim-selection
                                // can't otherwise tell "promoted THIS batch,
                                // do not touch" from "genuinely free."
                                claimed_this_batch.push_back(si);
                            } else {
                                perf().n_copy_on_promotion.fetch_add(1, std::memory_order_relaxed);
                            }
                        } else {
                            // Same "confirmed, not just issued" reasoning as
                            // get_expert's Step 1 -- see also_in_vram's own
                            // comment on MoEExpertBlock.
                            slot.also_in_vram = true;
                            if (pin_threshold > 0 && slot.hit_count >= pin_threshold) {
                                vram_cache->pin_expert(layer, expert_id);
                            }
                        }
                    }
                    out[k] = &slot;
                    found = true;
                    break;
                }
            }
            if (found) continue;

            // -- miss: claim a victim slot (Step 3's eviction pick, minus
            // an already-claimed-this-batch exclusion), same as get_expert
            // but WITHOUT the actual read yet --
            perf().n_expert_cache_miss.fetch_add(1, std::memory_order_relaxed);
            if (in_prefill) perf().n_expert_cache_miss_prefill.fetch_add(1, std::memory_order_relaxed);
            else perf().n_expert_cache_miss_decode.fetch_add(1, std::memory_order_relaxed);

            int lru_idx = -1;
            uint64_t lru_score = 0;
            for (size_t i = 0; i < cap; ++i) {
                if (std::find(claimed_this_batch.begin(), claimed_this_batch.end(), i) != claimed_this_batch.end())
                    continue;
                // Same also_in_vram-first eviction preference as get_expert's
                // Step 3 -- see MoEExpertBlock::also_in_vram's own comment.
                // copy_still_pending() takes priority over both: never pick
                // a slot whose async H2D copy hasn't been confirmed done yet
                // (see MoEExpertBlock::pending_copy_event).
                uint64_t score = layer_slots[i].copy_still_pending() ? UINT64_MAX
                    : (layer_slots[i].also_in_vram ? 0 : eviction_score(layer_slots[i]));
                if (lru_idx < 0 || score < lru_score) { lru_score = score; lru_idx = (int)i; }
            }
            if (lru_idx < 0) {
                // cap < remaining misses this batch -- every active slot for
                // this layer is already claimed by an EARLIER k in the SAME
                // batch. Only possible if slots_per_layer < topk (an
                // extremely small RAM budget), never expected in normal
                // operation. Falls back to slot 0, which means this k WILL
                // collide with whichever earlier k also claimed slot 0 --
                // logged loudly since that's a real correctness-relevant
                // event worth knowing about if it ever fires, not silently
                // swallowed the way get_expert's own single-expert path
                // (which has no equivalent "already claimed" concept at all)
                // would never surface it.
                LOG_WARN("[BATCH-FETCH] layer=%d: ran out of distinct cache slots mid-batch "
                         "(slots_per_layer=%d < topk=%d) -- falling back to slot 0, which may "
                         "collide with an earlier pick in this same batch\n",
                         layer, slots_per_layer, topk);
                lru_idx = 0;
            }
            claimed_this_batch.push_back((size_t)lru_idx);

            MoEExpertBlock& victim = layer_slots[lru_idx];
            int evicted_expert_id = victim.expert_id;
            if (evicted_expert_id != -1) {
                ram_swap_stats[swap_key(layer, evicted_expert_id)].swap_out++;
            }
            victim.also_in_vram = false;  // fresh identity in this slot momentarily
            // Defensive last resort (see get_expert Step 3's identical
            // comment) -- only actually blocks if every candidate was
            // pending. Must run before ensure_slab_capacity below, which
            // is what would actually touch/free/realloc slab_buffer.
            if (victim.pending_copy_event) {
                cudaEventSynchronize(victim.pending_copy_event);
                victim.pending_copy_event = nullptr;
            }

            if (victim.expert_id != -1 && engine.drop_expert_pages) {
                // Per-projection DONTNEED by name -- the old (down.end -
                // gate.offset) span underflowed size_t on a scattered repack
                // layout (down<gate<up) and advised a ~SIZE_MAX range.
                std::string base = "blk." + std::to_string(layer) + ".";
                std::string es = std::to_string(victim.expert_id);
                for (const char* proj : {"ffn_gate", "ffn_up", "ffn_down"}) {
                    auto it = engine.model_file.tensors.find(base + proj + "." + es + ".weight");
                    if (it != engine.model_file.tensors.end())
                        posix_fadvise(engine.model_file.fd_for_tensor(it->second),
                                      it->second.offset, it->second.size_bytes, POSIX_FADV_DONTNEED);
                }
            }

            // Claim identity + freshen recency IMMEDIATELY (not after the
            // read) -- this is what makes it safe for a LATER k in this
            // batch to scan past this slot without re-picking it (belt-
            // and-suspenders alongside claimed_this_batch above).
            victim.expert_id = expert_id;
            victim.last_used_clock = ++global_clock;
            victim.hit_count = 0;

            PERF_START(expert_disk_load);
            bool served_from_pipe = false;
            if (prefetcher) {
                size_t gb, ub, db;
                const GGUFTensor *gM, *uM, *dM;
                if (engine.packed_slab_geometry(layer, gb, ub, db, gM, uM, dM)) {
                    size_t total = gb + ub + db;
                    engine.ensure_slab_capacity(victim, total);
                    PERF_START(consume);
                    size_t got_bytes = 0;
                    bool ok = prefetcher->consume_staged(layer, expert_id, victim.slab_buffer, victim.slab_capacity, &got_bytes);
                    PERF_END(consume, t_consume_staged);
                    if (ok && got_bytes == total) {
                        engine.populate_packed_slab(victim, expert_id, gb, ub, db, *gM, *uM, *dM);
                        served_from_pipe = true;
                        perf().n_pipe_hits.fetch_add(1, std::memory_order_relaxed);
                        if (in_prefill) perf().n_pipe_hits_prefill.fetch_add(1, std::memory_order_relaxed);
                        else perf().n_pipe_hits_decode.fetch_add(1, std::memory_order_relaxed);
                    } else if (!ok && prefetcher->has_pending_hint(layer, expert_id)) {
                        // consume_staged found no matching slot (its own
                        // n_pipe_miss_no_match already fired internally) --
                        // check whether the pilot DID predict this exact
                        // expert and the hint is simply still sitting unpopped
                        // in the SPSC queue, vs no trace of it anywhere. See
                        // n_pipe_miss_still_queued's own comment.
                        perf().n_pipe_miss_still_queued.fetch_add(1, std::memory_order_relaxed);
                    }
                }
            }

            out[k] = &victim;
            if (served_from_pipe) {
                ram_swap_stats[swap_key(layer, expert_id)].swap_in++;
                double load_ms = std::chrono::duration<double, std::milli>(
                    std::chrono::high_resolution_clock::now() - _perf_expert_disk_load).count();
                LOG_DEBUG("[MOE-CACHE] MISS  layer=%d expert=%d slot=%d evicted=%d loaded_from_disk=%.2fms via=PIPE (%.2f MB, %.2f GB/s)\n",
                          layer, expert_id, lru_idx, evicted_expert_id, load_ms,
                          victim.slab_capacity / 1e6,
                          load_ms > 0 ? (victim.slab_capacity / 1e9) / (load_ms / 1000.0) : 0.0);
            } else {
                pending.push_back({(size_t)lru_idx, expert_id, evicted_expert_id, k, 0.0});
            }
        }

        // --- Phase B: for whatever PIPE didn't already have staged, batch
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
            bool uring_ok = false;
            if (use_decode_uring && decode_uring_ring.available()) {
                std::vector<io_uring_backend::ReadRequest> reqs;
                reqs.reserve((size_t)n_pending * 3);
                bool all_packed = true;
                for (int i = 0; i < n_pending && all_packed; ++i) {
                    io_uring_backend::ReadRequest tmp[3];
                    int cnt = 0; size_t total = 0;
                    MoEExpertBlock& victim = layer_slots[pending[i].slot_idx];
                    if (!engine.build_packed_expert_read_requests(layer, pending[i].expert_id, victim,
                                                                    tmp, cnt, total)) {
                        all_packed = false;
                        break;
                    }
                    for (int j = 0; j < cnt; ++j) reqs.push_back(tmp[j]);
                }
                if (all_packed && !reqs.empty() && reqs.size() <= io_uring_backend::Ring::CAPACITY) {
                    std::vector<ssize_t> results(reqs.size());
                    auto t0 = std::chrono::high_resolution_clock::now();
                    bool submitted = decode_uring_ring.submit_and_wait(reqs.data(), (int)reqs.size(), results.data());
                    double batch_ms = std::chrono::duration<double, std::milli>(
                        std::chrono::high_resolution_clock::now() - t0).count();
                    bool all_ok = submitted;
                    if (submitted) {
                        for (size_t r = 0; r < reqs.size(); ++r) {
                            if (results[r] != (ssize_t)reqs[r].len) { all_ok = false; break; }
                        }
                    }
                    if (all_ok) {
                        size_t total_bytes = 0;
                        for (int i = 0; i < n_pending; ++i) {
                            size_t gb, ub, db;
                            const GGUFTensor *gM, *uM, *dM;
                            engine.packed_slab_geometry(layer, gb, ub, db, gM, uM, dM);
                            MoEExpertBlock& victim = layer_slots[pending[i].slot_idx];
                            engine.populate_packed_slab(victim, pending[i].expert_id, gb, ub, db, *gM, *uM, *dM);
                            pending[i].load_ms = batch_ms; // shared batch time; per-expert split isn't meaningful here
                            total_bytes += gb + ub + db;
                            ram_swap_stats[swap_key(layer, pending[i].expert_id)].swap_in++;
                        }
                        perf().bytes_disk_read.fetch_add(total_bytes, std::memory_order_relaxed);
                        uring_ok = true;
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
                // Logging (and swap-in accounting -- ram_swap_stats is a plain
                // unordered_map, unsafe to mutate from the parallel region above)
                // stays sequential, outside the parallel region.
                for (int i = 0; i < n_pending; ++i) {
                    PendingLoad& pl = pending[i];
                    MoEExpertBlock& victim = layer_slots[pl.slot_idx];
                    ram_swap_stats[swap_key(layer, pl.expert_id)].swap_in++;
                    LOG_DEBUG("[MOE-CACHE] MISS  layer=%d expert=%d slot=%zu evicted=%d loaded_from_disk=%.2fms via=pread "
                              "(%.2f MB, %.2f GB/s) [batch=%d/%d]\n",
                              layer, pl.expert_id, pl.slot_idx, pl.evicted_expert_id, pl.load_ms,
                              victim.slab_capacity / 1e6,
                              pl.load_ms > 0 ? (victim.slab_capacity / 1e9) / (pl.load_ms / 1000.0) : 0.0,
                              i + 1, n_pending);
                }
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