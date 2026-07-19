#pragma once
#include "engine.hpp"
#include "gpu_topology.hpp"
#include "kquant_dequant.hpp"  // gpu_type_supported
#include <atomic>
#include <thread>
#include <mutex>
#include <vector>
#include <string>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>
#include <cuda_runtime.h>

struct PrefetchHint {
    int layer_idx;
    int expert_id;
};

// Staging slot: pinned host memory for an expert slab ready for async GPU upload
struct PrefetchSlot {
    uint8_t* pinned_buf = nullptr;  // cudaMallocHost'd buffer
    uint8_t* device_ptr = nullptr;  // GPU destination (if upload fired)
    size_t capacity = 0;
    size_t loaded_bytes = 0;
    int layer_idx = -1;
    int expert_id = -1;
    std::atomic<bool> ready{false};      // true when pread is complete
    std::atomic<bool> on_device{false};  // true when async upload to GPU is done
};

class LockFreePrefetcher {
private:
    static constexpr size_t QUEUE_SIZE = 4096;
    static constexpr int NUM_STAGING_SLOTS = 16;  // double-buffer pool

    PrefetchHint queue[QUEUE_SIZE];
    std::atomic<unsigned int> head{0}; 
    std::atomic<unsigned int> tail{0}; 
    
    std::thread worker_thread;
    std::atomic<bool> running{false};
    StreamingEngine* engine = nullptr;
    GpuTopology* gpu_topo = nullptr;  // For async VRAM uploads

    // Pinned double-buffer pool for active prefetch
    PrefetchSlot staging[NUM_STAGING_SLOTS];
    std::atomic<int> staging_write_idx{0};  // next slot to write into
    // Guards all mutation of a staging slot's identity (layer_idx/expert_id),
    // ready/on_device flags, AND the pinned_buf lifecycle (acquire_staging_slot
    // recycle/regrow vs consume_staged's memcpy-out). Without it the decode
    // thread's consume_staged could memcpy from a pinned_buf the worker thread
    // is concurrently cudaFreeHost-ing via acquire_staging_slot, and could read
    // a half-written (layer,expert) identity. finish_prefetch publishes
    // identity+ready under this lock; consume_staged + acquire_staging_slot take
    // it to read/recycle consistently.
    std::mutex staging_mutex;

public:
    void start(StreamingEngine* eng, GpuTopology* topo = nullptr) {
        engine = eng;
        gpu_topo = topo;
        // Async H2D promotion goes through GpuTopology::upload_expert_async,
        // which uses the vault GPU's own dedicated copy_stream (see
        // gpu_topology.hpp) — no separate stream needed here.
        running.store(true, std::memory_order_release);
        worker_thread = std::thread(&LockFreePrefetcher::worker_loop, this);
    }

    void stop() {
        running.store(false, std::memory_order_release);
        if (worker_thread.joinable()) {
            worker_thread.join();
        }
    }

    ~LockFreePrefetcher() {
        for (int i = 0; i < NUM_STAGING_SLOTS; ++i) {
            if (staging[i].pinned_buf) cudaFreeHost(staging[i].pinned_buf);
        }
    }

    void push_hint(int layer_idx, int expert_id) {
        unsigned int current_head = head.load(std::memory_order_relaxed);
        unsigned int current_tail = tail.load(std::memory_order_acquire);

        if (current_head - current_tail >= QUEUE_SIZE) {
            return;
        }

        queue[current_head & (QUEUE_SIZE - 1)] = {layer_idx, expert_id};
        head.store(current_head + 1, std::memory_order_release);
    }

    // Consume a prefetched slab: memcpy the staged bytes into the caller's
    // buffer under staging_mutex (so the worker can't recycle/free the pinned
    // slot mid-copy), then release the staging slot for reuse. Returns false
    // (no-op) if no ready staged slab matches this (layer, expert).
    //
    // This is the Phase A "PIPE" overlap path: a hit here serves a cache miss
    // WITHOUT a synchronous pread on the decode thread, because the prefetcher
    // worker already pread the bytes while a previous layer computed. The
    // previous check_staged() returned a raw pointer into the recycling pool
    // and was dead code anyway; this copy-out-under-lock is the safe, live
    // replacement. Caller is responsible for then setting up the per-projection
    // host_ptr/meta from `dst` (see StreamingEngine::populate_slab_from_buffer).
    bool consume_staged(int layer_idx, int expert_id, uint8_t* dst, size_t dst_cap, size_t* out_bytes) {
        if (!dst) return false;
        std::lock_guard<std::mutex> lock(staging_mutex);
        for (int i = 0; i < NUM_STAGING_SLOTS; ++i) {
            if (staging[i].layer_idx == layer_idx &&
                staging[i].expert_id == expert_id &&
                staging[i].ready.load(std::memory_order_acquire)) {
                size_t bytes = staging[i].loaded_bytes;
                if (dst_cap < bytes) return false;  // caller buffer too small -> caller falls back to load_expert_slab
                std::memcpy(dst, staging[i].pinned_buf, bytes);
                if (out_bytes) *out_bytes = bytes;
                // Release the staging slot for reuse. (We do NOT touch the VRAM
                // copy's completion -- that is tracked independently by
                // GpuTopology via the slot's upload_event in find_expert.)
                staging[i].ready.store(false, std::memory_order_release);
                staging[i].layer_idx = -1;
                staging[i].expert_id = -1;
                return true;
            }
        }
        return false;
    }

private:
    void worker_loop() {
        while (running.load(std::memory_order_acquire) || 
               head.load(std::memory_order_acquire) != tail.load(std::memory_order_relaxed)) {
            
            unsigned int current_tail = tail.load(std::memory_order_relaxed);
            unsigned int current_head = head.load(std::memory_order_acquire);

            if (current_tail == current_head) {
                usleep(200);
                continue;
            }

            PrefetchHint hint = queue[current_tail & (QUEUE_SIZE - 1)];
            tail.store(current_tail + 1, std::memory_order_release);
            active_prefetch(hint.layer_idx, hint.expert_id);
        }
    }

    // Active prefetch: pread into pinned host memory (not just fadvise), then
    // (if a VRAM vault is configured) hand the pinned buffer to
    // GpuTopology::upload_expert_async for the actual H2D copy — that
    // function owns the slot-claim + stamp + memcpy sequence under
    // cache_mutex, so this never races the main thread's own promotions.
    void active_prefetch(int layer_idx, int expert_id) {
        if (!engine) return;

        // Check if already staged (under the same lock that publish/consume
        // use, so we never read a half-written identity).
        {
            std::lock_guard<std::mutex> lock(staging_mutex);
            for (int i = 0; i < NUM_STAGING_SLOTS; ++i) {
                if (staging[i].layer_idx == layer_idx &&
                    staging[i].expert_id == expert_id) return; // already in flight
            }
        }

        std::string base = "blk." + std::to_string(layer_idx) + ".";
        int n_exp = engine->model_file.architecture_config.n_experts;

        // ---- Packed format: ffn_{gate,up,down}_exps.weight. Expert E's slice
        // is [E*per_exp, (E+1)*per_exp) within each tensor — mirrors
        // StreamingEngine::load_expert_slab's packed branch exactly, including
        // per-projection quant-type sizing (GLM-5.2 UD: gate/up=Q2_K,
        // down=Q3_K, NOT uniformly sized). The old code here only read
        // `size_bytes*3/n_experts` bytes starting at the GATE tensor's base
        // offset (expert 0's slice, regardless of `expert_id`) and called that
        // "the expert" — silently mislabeling every promoted slot.
        auto gE = engine->model_file.tensors.find(base + "ffn_gate_exps.weight");
        auto uE = engine->model_file.tensors.find(base + "ffn_up_exps.weight");
        auto dE = engine->model_file.tensors.find(base + "ffn_down_exps.weight");
        if (n_exp > 0 && gE != engine->model_file.tensors.end() &&
            uE != engine->model_file.tensors.end() && dE != engine->model_file.tensors.end()) {
            // This entire prefetch cycle (disk pread + pinned staging) exists
            // ONLY to feed the async VRAM promotion below (check_staged()'s
            // staging buffer has no other consumer anywhere in the codebase)
            // — so when a VRAM vault is configured but this layer's expert
            // quant types can never dispatch to GPU (all three projections
            // must be supported; a Dynamic-quant model can have this true for
            // every expert in every layer), skip it entirely rather than
            // burning disk I/O + pinned memory for a promotion that would
            // never fire anyway.
            if (gpu_topo && !(gpu_type_supported(gE->second.type) &&
                               gpu_type_supported(uE->second.type) &&
                               gpu_type_supported(dE->second.type))) {
                return;
            }
            const size_t gb = gE->second.size_bytes / n_exp;
            const size_t ub = uE->second.size_bytes / n_exp;
            const size_t db = dE->second.size_bytes / n_exp;
            const size_t total = gb + ub + db;

            PrefetchSlot& slot = acquire_staging_slot(total);
            if (!slot.pinned_buf) { issue_fadvise_fallback(layer_idx, expert_id); return; }

            int gfd = engine->model_file.fd_for_tensor(gE->second);
            int ufd = engine->model_file.fd_for_tensor(uE->second);
            int dfd = engine->model_file.fd_for_tensor(dE->second);
            ssize_t rg = pread(gfd, slot.pinned_buf,          gb, gE->second.offset + (size_t)expert_id * gb);
            ssize_t ru = pread(ufd, slot.pinned_buf + gb,     ub, uE->second.offset + (size_t)expert_id * ub);
            ssize_t rd = pread(dfd, slot.pinned_buf + gb + ub, db, dE->second.offset + (size_t)expert_id * db);

            if (rg == (ssize_t)gb && ru == (ssize_t)ub && rd == (ssize_t)db) {
                finish_prefetch(slot, layer_idx, expert_id, total);
            } else {
                issue_fadvise_fallback(layer_idx, expert_id);
            }
            return;
        }

        // ---- Legacy per-expert format: ffn_{gate,up,down}.E.weight ----
        std::string gate_name = base + "ffn_gate." + std::to_string(expert_id) + ".weight";
        std::string down_name = base + "ffn_down." + std::to_string(expert_id) + ".weight";
        auto git = engine->model_file.tensors.find(gate_name);
        auto dit = engine->model_file.tensors.find(down_name);
        if (git == engine->model_file.tensors.end() || dit == engine->model_file.tensors.end()) {
            issue_fadvise_fallback(layer_idx, expert_id);
            return;
        }
        auto& gate = git->second;
        auto& down = dit->second;
        size_t total = (down.offset + down.size_bytes) - gate.offset;

        PrefetchSlot& slot = acquire_staging_slot(total);
        if (!slot.pinned_buf) { issue_fadvise_fallback(layer_idx, expert_id); return; }

        int fd = engine->model_file.fd_for_tensor(gate);
        ssize_t bytes_read = pread(fd, slot.pinned_buf, total, gate.offset);

        if (bytes_read == (ssize_t)total) {
            finish_prefetch(slot, layer_idx, expert_id, total);
        } else {
            issue_fadvise_fallback(layer_idx, expert_id);
        }
    }

    // Round-robin staging slot, grown to `needed_bytes` if too small. Takes
    // staging_mutex so the regrow (which can cudaFreeHost the pinned_buf) can't
    // race a concurrent consume_staged memcpy-out of the same slot.
    PrefetchSlot& acquire_staging_slot(size_t needed_bytes) {
        int slot_idx = staging_write_idx.fetch_add(1, std::memory_order_relaxed) % NUM_STAGING_SLOTS;
        std::lock_guard<std::mutex> lock(staging_mutex);
        PrefetchSlot& slot = staging[slot_idx];
        if (slot.capacity < needed_bytes) {
            if (slot.pinned_buf) cudaFreeHost(slot.pinned_buf);
            cudaMallocHost((void**)&slot.pinned_buf, needed_bytes);
            slot.capacity = needed_bytes;
        }
        return slot;
    }

    // Fire the async H2D promotion FIRST (if a VRAM vault is configured), THEN
    // publish identity+ready under staging_mutex. Ordering matters: if
    // identity were published before the upload fired, a racing consume_staged
    // could clear the slot before the upload stamped it. The upload itself
    // goes through GpuTopology's own mutexed claim path (cache_mutex), so it
    // is safe to call outside staging_mutex; only the slot-state publish is
    // locked here.
    void finish_prefetch(PrefetchSlot& slot, int layer_idx, int expert_id, size_t total_bytes) {
        slot.loaded_bytes = total_bytes;

        ExpertLocation* vram_slot = nullptr;
        if (gpu_topo) {
            vram_slot = gpu_topo->upload_expert_async(
                layer_idx, expert_id, slot.pinned_buf, slot.loaded_bytes);
        }

        {
            std::lock_guard<std::mutex> lock(staging_mutex);
            slot.layer_idx = layer_idx;
            slot.expert_id = expert_id;
            if (vram_slot) {
                slot.device_ptr = vram_slot->device_ptr;
                slot.on_device.store(true, std::memory_order_release);
            }
            slot.ready.store(true, std::memory_order_release);
        }
    }

    void issue_fadvise_fallback(int layer_idx, int expert_id) {
        std::string gate_name = "blk." + std::to_string(layer_idx) + ".ffn_gate." + std::to_string(expert_id) + ".weight";
        auto it = engine->model_file.tensors.find(gate_name);
        if (it != engine->model_file.tensors.end()) {
            auto& t_meta = it->second;
            posix_fadvise(engine->model_file.fd_for_tensor(t_meta), t_meta.offset, t_meta.size_bytes * 3, POSIX_FADV_WILLNEED);
        }
    }
};