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
#include <mutex>
#include "perf_timer.hpp"
#include "logger.hpp"

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

    // Phase A colibri-port: a pinned slot is never chosen as an eviction
    // victim, regardless of eviction_score. Set once a RAM-cache expert's
    // hit_count crosses RAMSafetyManager::pin_threshold (proven-hot within
    // this conversation), or at startup for the pre-loaded hot set from a
    // usage-file profile (cross-conversation). Only mutated under
    // cache_mutex, same as every other field here.
    bool pinned = false;

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

    // Expert VRAM LRU cache: maps (layer, expert_id) -> location
    std::vector<ExpertLocation> expert_vram_cache;
    size_t expert_vram_cache_cap = 0;  // max entries
    size_t expert_slab_size = 0;       // bytes per expert slot
    uint64_t cache_clock = 0;

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

    // Pinned host buffer for cross-device transfers
    float* h_xfer_pinned = nullptr;
    size_t h_xfer_cap = 0;

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
            if (dev.expert_cache_pool) { cudaSetDevice(dev.id); cudaFree(dev.expert_cache_pool); }
        }
        if (h_xfer_pinned) cudaFreeHost(h_xfer_pinned);
    }

    // Allocate expert VRAM cache on vault GPUs
    void allocate_expert_cache(size_t expert_slab_bytes, int n_experts_total) {
        expert_slab_size = expert_slab_bytes;
        if (expert_slab_bytes == 0) return;

        for (int vi : vault_gpus) {
            auto& dev = devices[vi];
            // Reserve 1 GB for driver overhead, allocate the rest
            size_t reserve = 1ULL << 30;
            size_t cache_bytes = (dev.total_vram > reserve) ? dev.total_vram - reserve : 0;

            cudaSetDevice(dev.id);
            if (cudaMalloc((void**)&dev.expert_cache_pool, cache_bytes) != cudaSuccess) {
                // Retry with 80% of VRAM
                cache_bytes = (size_t)(dev.total_vram * 0.8);
                if (cudaMalloc((void**)&dev.expert_cache_pool, cache_bytes) != cudaSuccess) {
                    LOG_STREAM_ERROR() << "[GPU] Failed to allocate expert cache on device " << dev.id << "\n";
                    dev.expert_cache_pool = nullptr;
                    cache_bytes = 0;
                }
            }
            dev.expert_cache_size = cache_bytes;
            dev.available_vram -= cache_bytes;

            LOG_STREAM_INFO() << "[GPU] Expert cache on " << dev.name << " (GPU " << dev.id
                      << "): " << cache_bytes / 1e9 << " GB\n";
        }

        // Size the LRU index and pre-assign slab offsets
        size_t total_cache = 0;
        for (int vi : vault_gpus) total_cache += devices[vi].expert_cache_size;
        if (expert_slab_bytes > 0) {
            expert_vram_cache_cap = total_cache / expert_slab_bytes;
        }
        expert_vram_cache.resize(expert_vram_cache_cap);

        // Pre-assign device_ptr for each slot (fixed slab allocation)
        size_t offset = 0;
        int vault_idx = 0;
        for (size_t i = 0; i < expert_vram_cache_cap; ++i) {
            // Find which vault GPU this slot belongs to
            while (vault_idx < (int)vault_gpus.size() &&
                   offset >= devices[vault_gpus[vault_idx]].expert_cache_size) {
                offset -= devices[vault_gpus[vault_idx]].expert_cache_size;
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
            cudaSetDevice(dev.id);
            cudaEventCreate(&expert_vram_cache[i].upload_event);
            offset += expert_slab_bytes;
        }

        LOG_STREAM_INFO() << "[GPU] Expert VRAM cache: " << expert_vram_cache_cap
                  << " slots (" << total_cache / 1e9 << " GB total)\n";
    }

    // Look up or load an expert into VRAM cache. Returns device pointer or
    // nullptr — including when a background async promotion is still in
    // flight (so callers never read a partially-copied slab).
    ExpertLocation* find_expert(int layer, int expert_id) {
        std::lock_guard<std::recursive_mutex> lock(cache_mutex);
        for (auto& loc : expert_vram_cache) {
            if (loc.layer_idx == layer && loc.expert_id == expert_id && loc.device_ptr) {
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
        std::lock_guard<std::recursive_mutex> lock(cache_mutex);
        for (auto& loc : expert_vram_cache) {
            if (loc.layer_idx == layer && loc.expert_id == expert_id && loc.device_ptr) {
                if (!loc.pinned) {
                    loc.pinned = true;
                    perf().n_pinned_resident.fetch_add(1, std::memory_order_relaxed);
                }
                return true;
            }
        }
        return false;
    }

    // Evict lowest recency+frequency score and return a free slot for a new
    // expert upload. The slot's device_ptr is pre-assigned (slab offset),
    // ready for cudaMemcpy.
    ExpertLocation* evict_lru(size_t needed_bytes) {
        std::lock_guard<std::recursive_mutex> lock(cache_mutex);
        (void)needed_bytes; // All slots are fixed-size in slab allocator
        ExpertLocation* victim = nullptr;
        for (auto& loc : expert_vram_cache) {
            if (loc.layer_idx == -1) { victim = &loc; break; } // empty slot preferred
            // Never evict a pinned (proven-hot, Phase A colibri-port) slot.
            // Also never evict a slot whose async upload is still in flight
            // -- this second part is a real correctness fix, not just
            // policy: before this, an in-flight cudaMemcpyAsync into a
            // not-yet-ready slot could be chosen as ANOTHER eviction's
            // victim and overwritten mid-copy (the old code only checked
            // eviction_score, which a fresh in-flight slot satisfies same
            // as any other -- last_used_clock/hit_count don't reflect
            // "copy not done yet").
            if (loc.pinned || !loc.upload_ready) {
                if (loc.pinned) perf().n_pinned_eviction_skips.fetch_add(1, std::memory_order_relaxed);
                continue;
            }
            if (!victim || loc.eviction_score() < victim->eviction_score()) victim = &loc;
        }
        if (victim) {
            victim->layer_idx = -1;
            victim->expert_id = -1;
            victim->last_used_clock = 0;
            victim->hit_count = 0;
            victim->upload_ready = true;
            victim->pinned = false;  // defensive: a freed slot must never carry a stale pin
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
        std::lock_guard<std::recursive_mutex> lock(cache_mutex);
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
        std::lock_guard<std::recursive_mutex> lock(cache_mutex);

        // Check if already cached
        ExpertLocation* existing = find_expert(layer, expert_id);
        if (existing) return existing;

        // Get a slot (evict LRU if needed)
        ExpertLocation* slot = evict_lru(bytes);
        if (!slot || !slot->device_ptr) return nullptr;

        // Copy host → device (synchronous for correctness)
        size_t copy_bytes = std::min(bytes, slot->size);
        cudaSetDevice(slot->gpu_id);
        PERF_START(gpu_sync_upload);
        cudaError_t err = cudaMemcpy(slot->device_ptr, host_data, copy_bytes, cudaMemcpyHostToDevice);
        PERF_END(gpu_sync_upload, t_gpu_upload);
        if (err != cudaSuccess) return nullptr;

        perf().bytes_gpu_uploaded.fetch_add((int64_t)copy_bytes, std::memory_order_relaxed);
        perf().n_expert_promote.fetch_add(1, std::memory_order_relaxed);

        slot->layer_idx = layer;
        slot->expert_id = expert_id;
        slot->last_used_clock = ++cache_clock;
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
        std::lock_guard<std::recursive_mutex> lock(cache_mutex);

        ExpertLocation* existing = find_expert(layer, expert_id);
        if (existing) return existing;

        ExpertLocation* slot = evict_lru(bytes);
        if (!slot || !slot->device_ptr) return nullptr;

        size_t copy_bytes = std::min(bytes, slot->size);
        GpuDevice& dev = devices[slot->gpu_id];
        cudaSetDevice(dev.id);
        slot->upload_ready = false;
        cudaMemcpyAsync(slot->device_ptr, host_data, copy_bytes, cudaMemcpyHostToDevice, dev.copy_stream);
        if (slot->upload_event) cudaEventRecord(slot->upload_event, dev.copy_stream);

        // Record promotion metrics immediately (bytes enqueued + count).
        perf().bytes_gpu_uploaded.fetch_add((int64_t)copy_bytes, std::memory_order_relaxed);
        perf().n_expert_promote.fetch_add(1, std::memory_order_relaxed);

        slot->layer_idx = layer;
        slot->expert_id = expert_id;
        slot->last_used_clock = ++cache_clock;
        return slot;
    }

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
            // Multi-GPU: assign Core to newest arch (or smallest VRAM if same arch)
            // Assign Vault to highest VRAM cards
            int best_core = 0;
            for (int i = 1; i < n; ++i) {
                bool newer = devices[i].sm_major > devices[best_core].sm_major;
                bool same_gen_less_vram = (devices[i].sm_major == devices[best_core].sm_major &&
                                           devices[i].total_vram < devices[best_core].total_vram);
                if (newer || same_gen_less_vram) best_core = i;
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
            std::lock_guard<std::recursive_mutex> lock(cache_mutex);
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
};
