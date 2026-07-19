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
#include <fcntl.h>
#include <unistd.h>
#ifdef __APPLE__
#include <mach/mach.h>
#include <sys/sysctl.h>
#endif

class RAMSafetyManager {
public:
    int slots_per_layer = 0;
    uint64_t global_clock = 0;
    GpuTopology* vram_cache = nullptr;  // GPU expert VRAM cache (optional, set by CudaManager)
    // Topology-enabled prefetcher whose staging pool the cache-MISS path
    // consumes (Phase A "PIPE" overlap). Set by main.cpp alongside vram_cache.
    // nullptr = miss path always falls back to synchronous load_expert_slab.
    LockFreePrefetcher* prefetcher = nullptr;

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

    // Frequency-aware eviction: blend recency with hit_count so a
    // power-law "always hot" expert survives bursts of cold accesses to
    // other experts. Decayed periodically to avoid permanent stickiness.
    static constexpr uint64_t FREQ_BONUS_TICKS = 64;
    static constexpr uint32_t HIT_DECAY_INTERVAL = 4096;
    uint64_t hits_since_decay = 0;

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
            : std::min(layer_slots.size(), (size_t)std::max(1, slots_per_layer));

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
                        vram_cache->upload_expert_async(layer, expert_id, slot.slab_buffer, slot.slab_capacity);
                    } else if (pin_threshold > 0 && slot.hit_count >= pin_threshold) {
                        // Already VRAM-resident AND proven hot enough within
                        // THIS conversation (no cross-prompt profile needed)
                        // -- pin it so evict_lru never reactively kicks it
                        // out for a one-hit-wonder later in the same
                        // generation. Phase A colibri-port: "stop evicting
                        // hot experts within a conversation".
                        vram_cache->pin_expert(layer, expert_id);
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

        // --- Step 3: Evict lowest recency+frequency score and load from disk ---
        int lru_idx = 0;
        for (size_t i = 1; i < cap; ++i) {
            if (eviction_score(layer_slots[i]) < eviction_score(layer_slots[lru_idx])) lru_idx = i;
        }

        MoEExpertBlock& victim = layer_slots[lru_idx];
        int evicted_expert_id = victim.expert_id;

        if (victim.expert_id != -1 && engine.drop_expert_pages) {
            std::string base = "blk." + std::to_string(layer) + ".";
            std::string gate_name = base + "ffn_gate." + std::to_string(victim.expert_id) + ".weight";
            std::string down_name = base + "ffn_down." + std::to_string(victim.expert_id) + ".weight";
            auto git = engine.model_file.tensors.find(gate_name);
            auto dit = engine.model_file.tensors.find(down_name);
            if (git != engine.model_file.tensors.end() && dit != engine.model_file.tensors.end()) {
                size_t total_chunk = (dit->second.offset + dit->second.size_bytes) - git->second.offset;
                posix_fadvise(engine.model_file.fd_for_tensor(git->second), git->second.offset, total_chunk, POSIX_FADV_DONTNEED);
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
            size_t cap = slots.empty() ? 0 : std::min(slots.size(), (size_t)std::max(1, slots_per_layer));
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
};