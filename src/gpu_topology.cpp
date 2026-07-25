// gpu_topology.cpp - Active layer area implementation for three-tier VRAM caching
#include "gpu_topology.hpp"
#include "perf_timer.hpp"
#include "logger.hpp"
#include <cuda_runtime.h>
#include <algorithm>

bool GpuTopology::allocate_active_layer_area(size_t expert_slab_bytes, int max_experts_per_layer) {
    if (expert_slab_bytes == 0 || max_experts_per_layer <= 0) {
        LOG_STREAM_ERROR() << "[GPU] Invalid active layer area parameters: slab_bytes="
                           << expert_slab_bytes << " max_experts=" << max_experts_per_layer << "\n";
        return false;
    }

    // Allocate on primary vault GPU (or core GPU if no vault)
    int target_gpu = vault_gpus.empty() ? core_gpu : vault_gpus[0];
    if (target_gpu < 0 || target_gpu >= (int)devices.size()) {
        LOG_STREAM_ERROR() << "[GPU] Invalid target GPU for active layer area: " << target_gpu << "\n";
        return false;
    }

    auto& dev = devices[target_gpu];
    cudaSetDevice(target_gpu);

    size_t total_bytes = expert_slab_bytes * max_experts_per_layer;
    cudaError_t err = cudaMalloc((void**)&active_layer_pool, total_bytes);
    if (err != cudaSuccess) {
        LOG_STREAM_ERROR() << "[GPU] Failed to allocate active layer area: "
                           << cudaGetErrorString(err) << " (requested "
                           << total_bytes / 1e9 << " GB)\n";
        return false;
    }

    active_layer_capacity = max_experts_per_layer;

    // Pre-allocate slot metadata
    active_layer_slots.clear();
    active_layer_slots.reserve(max_experts_per_layer);
    for (int i = 0; i < max_experts_per_layer; ++i) {
        ExpertLocation slot;
        slot.gpu_id = target_gpu;
        slot.device_ptr = active_layer_pool + (i * expert_slab_bytes);
        slot.size = expert_slab_bytes;
        slot.offset = i * expert_slab_bytes;
        slot.layer_idx = -1;
        slot.expert_id = -1;
        slot.last_used_clock = 0;
        slot.hit_count = 0;
        slot.upload_ready = true;
        slot.pinned = false;
        slot.temp_pin_count = 0;
        slot.upload_event = nullptr;
        active_layer_slots.push_back(slot);
    }

    LOG_STREAM_INFO() << "[GPU] Active layer area allocated: "
                      << max_experts_per_layer << " experts ("
                      << total_bytes / 1e9 << " GB) on GPU " << target_gpu << "\n";
    return true;
}

ExpertLocation* GpuTopology::upload_expert_active(int layer, int expert_id,
                                                 const uint8_t* host_data, size_t bytes) {
    if (active_layer_pool == nullptr || active_layer_capacity == 0) {
        return nullptr;  // Active layer area not allocated
    }

    if (!host_data || bytes == 0) {
        LOG_STREAM_ERROR() << "[GPU] Invalid upload_expert_active parameters\n";
        return nullptr;
    }

    GPU_TOPO_TIMED_LOCK(lock);

    // Find empty slot or lowest-eviction_score victim within active layer
    // area -- eviction_score (recency + hit_count, same blend the main
    // cache uses) rather than plain oldest-clock, so a slot that's already
    // taken one hit (but not yet crossed active_layer_promote_threshold)
    // isn't evicted ahead of a colder one just for being slightly older.
    ExpertLocation* victim = nullptr;

    for (auto& slot : active_layer_slots) {
        if (slot.layer_idx == -1) {
            victim = &slot;  // Prefer empty slot
            break;
        }
        if (!victim || slot.eviction_score() < victim->eviction_score()) {
            victim = &slot;
        }
    }

    if (!victim) {
        LOG_STREAM_WARN() << "[GPU] Active layer area full - cannot upload expert\n";
        return nullptr;
    }

    // Synchronous upload (decode-thread-only, no async needed)
    size_t copy_bytes = std::min(bytes, victim->size);
    cudaSetDevice(victim->gpu_id);
    cudaError_t err = cudaMemcpy(victim->device_ptr, host_data, copy_bytes,
                                cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        LOG_STREAM_ERROR() << "[GPU] Failed to upload expert to active layer: "
                           << cudaGetErrorString(err) << "\n";
        return nullptr;
    }

    victim->layer_idx = layer;
    victim->expert_id = expert_id;
    victim->last_used_clock = ++cache_clock;
    victim->hit_count = 0;  // fresh identity in this slot -- find_active_expert accumulates from here
    victim->upload_ready = true;

    LOG_DEBUG("[GPU] Active layer upload: layer=%d expert=%d (%.2f MB)\n",
              layer, expert_id, copy_bytes / 1e6);

    // Track three-tier metrics
    perf().n_active_layer_uploads.fetch_add(1, std::memory_order_relaxed);

    return victim;
}

void GpuTopology::cleanup_active_layer_area(int current_layer_idx) {
    if (active_layer_pool == nullptr) {
        return;  // Not allocated, nothing to cleanup
    }

    GPU_TOPO_TIMED_LOCK(lock);

    int slots_freed = 0;
    for (auto& slot : active_layer_slots) {
        if (slot.layer_idx != current_layer_idx && slot.layer_idx != -1) {
            // This slot's layer just finished (execution has moved on to
            // current_layer_idx) -- promote it into the real VRAM LRU cache
            // if it proved hot enough while resident (find_active_expert
            // already promotes+frees on an inline hit crossing the
            // threshold; this is the end-of-layer catch for anything that
            // hit fewer times than that but at least once), otherwise this
            // is exactly "swap out the one not used": just free it.
            if (slot.hit_count >= active_layer_promote_threshold) {
                promote_active_to_main(slot.layer_idx, slot.expert_id,
                                       slot.device_ptr, slot.size);
            }
            // Clear slot regardless of whether promotion happened above --
            // promote_active_to_main only copies the bytes elsewhere, it
            // never touches this slot's own identity.
            slot.layer_idx = -1;
            slot.expert_id = -1;
            slot.last_used_clock = 0;
            slot.hit_count = 0;
            slot.upload_ready = true;
            slot.temp_pin_count = 0;
            slots_freed++;
        }
    }

    if (slots_freed > 0) {
        LOG_DEBUG("[GPU] Active layer cleanup: freed %d slots (current layer=%d)\n",
                  slots_freed, current_layer_idx);
        // Track three-tier metrics
        perf().n_active_layer_cleanup.fetch_add(slots_freed, std::memory_order_relaxed);
    }
}

ExpertLocation* GpuTopology::find_active_expert(int layer, int expert_id) {
    if (active_layer_pool == nullptr) return nullptr;
    GPU_TOPO_TIMED_LOCK(lock);

    for (auto& slot : active_layer_slots) {
        if (slot.layer_idx == layer && slot.expert_id == expert_id) {
            slot.last_used_clock = ++cache_clock;
            if (slot.hit_count < UINT32_MAX) slot.hit_count++;
            perf().n_active_layer_hits.fetch_add(1, std::memory_order_relaxed);

            if (slot.hit_count >= active_layer_promote_threshold) {
                // MOVE, not COPY: promote now rather than waiting for this
                // layer's cleanup_active_layer_area call, then immediately
                // free this slot so it's available for a new cold upload
                // right away -- mirrors RAMSafetyManager's move_on_promotion.
                ExpertLocation* promoted = promote_active_to_main(layer, expert_id,
                                                                   slot.device_ptr, slot.size);
                if (promoted) {
                    slot.layer_idx = -1;
                    slot.expert_id = -1;
                    slot.last_used_clock = 0;
                    slot.hit_count = 0;
                    slot.upload_ready = true;
                    slot.temp_pin_count = 0;
                    return promoted;
                }
                // Main cache unavailable/copy failed -- fall through and
                // keep serving this token's compute from the active slot.
            }
            return &slot;
        }
    }
    return nullptr;
}

ExpertLocation* GpuTopology::promote_active_to_main(int layer, int expert_id,
                                                     const uint8_t* device_src, size_t bytes) {
    if (expert_vram_cache_cap == 0 || !device_src || bytes == 0) return nullptr;
    GPU_TOPO_TIMED_LOCK(lock);

    ExpertLocation* existing = find_expert(layer, expert_id);
    if (existing) return existing;  // already resident via some other path

    int old_layer = -1, old_expert = -1;
    ExpertLocation* slot = evict_lru(bytes, &old_layer, &old_expert);
    if (!slot || !slot->device_ptr) return nullptr;

    // Same VRAM->RAM salvage as upload_expert() -- must run before the
    // device-to-device copy below overwrites this slot's outgoing data.
    if (on_vram_evict && old_layer != -1) {
        on_vram_evict(old_layer, old_expert, slot->device_ptr, slot->size, slot->gpu_id);
    }

    size_t copy_bytes = std::min(bytes, slot->size);
    cudaSetDevice(slot->gpu_id);
    cudaError_t err = cudaMemcpy(slot->device_ptr, device_src, copy_bytes, cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) {
        LOG_STREAM_ERROR() << "[GPU] Active-layer promotion D2D copy failed: "
                           << cudaGetErrorString(err) << "\n";
        return nullptr;
    }

    perf().n_active_layer_promoted.fetch_add(1, std::memory_order_relaxed);

    slot->layer_idx = layer;
    slot->expert_id = expert_id;
    slot->last_used_clock = ++cache_clock;
    vram_swap_stats[swap_key(layer, expert_id)].swap_in++;
    expert_index[swap_key(layer, expert_id)] = (size_t)(slot - expert_vram_cache.data());
    return slot;
}