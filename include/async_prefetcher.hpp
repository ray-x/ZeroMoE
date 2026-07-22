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

// Any staging_mutex acquisition wait, or any staging-slot regrow, exceeding
// this is logged individually (tagged [SLOW-MUTEX-WAIT] / [SLOW-STAGING-
// REGROW]) -- added to root-cause a real, reproducible multi-second (up to
// 7.5s observed) single-miss stall that a per-miss latency histogram showed
// was NOT explained by raw disk throughput (p50/p90 were completely normal,
// only a handful of outlier calls were catastrophic). 50ms is already ~3x a
// normal ~15ms expert-slab read, so anything over this is unambiguously
// anomalous, not ordinary variance.
constexpr double SLOW_MUTEX_WAIT_MS = 50.0;

// How long a hint may sit in the queue before the worker thread actually
// starts processing it (pickup delay) before it's logged as anomalous
// ([SLOW-PICKUP]). Set higher than SLOW_MUTEX_WAIT_MS deliberately: even a
// perfectly healthy worker processes hints strictly sequentially, and the
// pilot pushes up to 8 hints per layer -- at a normal ~15ms/expert read,
// the LAST hint in one layer's batch can legitimately wait ~100ms+ just
// from the worker churning through the ones ahead of it, with zero
// contention or backlog involved. 250ms gives real margin above that
// benign case while still catching genuine backlog/starvation well before
// it reaches the multi-second severity seen in the SLOW-PREAD investigation.
// Added specifically to distinguish "the worker thread can't get scheduled
// promptly" (CPU contention with OMP dequant threads) from "the worker
// thread is scheduled fine but can't drain hints as fast as they're
// produced" (pure throughput/concurrency ceiling) -- the two point at very
// different fixes (nothing to do with thread count in the first case,
// more I/O parallelism in the second).
constexpr double SLOW_PICKUP_MS = 250.0;

struct PrefetchHint {
    int layer_idx;
    int expert_id;
    std::chrono::high_resolution_clock::time_point pushed_at;
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

        queue[current_head & (QUEUE_SIZE - 1)] = {layer_idx, expert_id, std::chrono::high_resolution_clock::now()};
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
        auto _wait_t0 = std::chrono::high_resolution_clock::now();
        std::lock_guard<std::mutex> lock(staging_mutex);
        double _wait_ms = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - _wait_t0).count();
        PerfCounters::atomic_add(perf().t_staging_mutex_wait, _wait_ms / 1000.0);
        if (_wait_ms > SLOW_MUTEX_WAIT_MS) {
            LOG_WARN("[SLOW-MUTEX-WAIT] consume_staged blocked %.2fms acquiring staging_mutex "
                     "(layer=%d expert=%d) -- worker thread likely holding it, check for a nearby "
                     "[SLOW-STAGING-REGROW] line\n", _wait_ms, layer_idx, expert_id);
        }
        // Failure-reason split (n_pipe_miss_no_match / n_pipe_miss_not_ready,
        // perf_timer.hpp) -- added to diagnose a real run where Prefetch
        // (PIPE) served collapsed to 0.1%: matching on identity FIRST,
        // independent of ready, lets a caller distinguish "the pilot never
        // predicted this expert (or the slot got recycled before this miss
        // arrived)" from "the pilot predicted correctly but the read simply
        // hadn't finished yet" -- two different root causes needing two
        // different fixes. Behavior change from before: previously, a slot
        // with a matching identity but ready==false was silently skipped
        // and the scan continued (in case some OTHER slot also matched,
        // extremely unlikely in practice since round-robin write assignment
        // means at most one slot should ever hold a given identity at a
        // time); now the first identity match returns immediately either
        // way. Correctness-neutral either way -- ready==false still always
        // falls back to a synchronous load, exactly as before.
        for (int i = 0; i < NUM_STAGING_SLOTS; ++i) {
            if (staging[i].layer_idx == layer_idx && staging[i].expert_id == expert_id) {
                if (!staging[i].ready.load(std::memory_order_acquire)) {
                    perf().n_pipe_miss_not_ready.fetch_add(1, std::memory_order_relaxed);
                    return false;  // predicted right, but the read isn't done -- caller falls back
                }
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
        perf().n_pipe_miss_no_match.fetch_add(1, std::memory_order_relaxed);
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
            // Backlog depth AT the moment of pickup (includes the hint just
            // popped) -- a direct, independent signal for "is the worker
            // falling behind hint production", complementing the pickup-
            // delay measurement below (a growing backlog and a growing
            // delay should move together; if they don't, that itself is
            // informative -- e.g. a big backlog with small delay would mean
            // hints are being produced in a sudden burst but still drained
            // fast once picked up).
            unsigned int backlog = current_head - current_tail;
            tail.store(current_tail + 1, std::memory_order_release);

            double pickup_ms = std::chrono::duration<double, std::milli>(
                std::chrono::high_resolution_clock::now() - hint.pushed_at).count();
            PerfCounters::atomic_add(perf().t_prefetch_pickup_delay, pickup_ms / 1000.0);
            perf().n_prefetch_hints_processed.fetch_add(1, std::memory_order_relaxed);
            PerfCounters::atomic_max(perf().n_prefetch_backlog_peak, (uint64_t)backlog);
            if (pickup_ms > SLOW_PICKUP_MS) {
                LOG_WARN("[SLOW-PICKUP] layer=%d expert=%d waited %.2fms in queue before the worker "
                         "thread picked it up (queue backlog at pickup: %u hints) -- worker is falling "
                         "behind hint production, either CPU-starved or I/O-throughput-bound\n",
                         hint.layer_idx, hint.expert_id, pickup_ms, backlog);
            }

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
            auto _wait_t0 = std::chrono::high_resolution_clock::now();
            std::lock_guard<std::mutex> lock(staging_mutex);
            double _wait_ms = std::chrono::duration<double, std::milli>(
                std::chrono::high_resolution_clock::now() - _wait_t0).count();
            PerfCounters::atomic_add(perf().t_staging_mutex_wait, _wait_ms / 1000.0);
            if (_wait_ms > SLOW_MUTEX_WAIT_MS) {
                LOG_WARN("[SLOW-MUTEX-WAIT] active_prefetch blocked %.2fms acquiring staging_mutex "
                         "(layer=%d expert=%d)\n", _wait_ms, layer_idx, expert_id);
            }
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
        auto _wait_t0 = std::chrono::high_resolution_clock::now();
        std::lock_guard<std::mutex> lock(staging_mutex);
        double _wait_ms = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - _wait_t0).count();
        PerfCounters::atomic_add(perf().t_staging_mutex_wait, _wait_ms / 1000.0);
        if (_wait_ms > SLOW_MUTEX_WAIT_MS) {
            LOG_WARN("[SLOW-MUTEX-WAIT] acquire_staging_slot blocked %.2fms acquiring staging_mutex (slot=%d)\n",
                     _wait_ms, slot_idx);
        }
        PrefetchSlot& slot = staging[slot_idx];
        if (slot.capacity < needed_bytes) {
            // PRIME SUSPECT for the multi-second single-miss stall: pinned
            // (page-locked) host memory alloc/free is a documented-slow CUDA
            // driver operation (can run tens to thousands of ms, especially
            // under memory pressure or fragmentation) -- and this whole pair
            // runs WHILE HOLDING staging_mutex, so any concurrent
            // consume_staged() call on the decode thread blocks for the full
            // duration. A fixed pool of only NUM_STAGING_SLOTS=16 slots
            // rotating round-robin across MoE layers whose packed slab size
            // can differ layer-to-layer (different quant types e.g. IQ4_XS
            // vs IQ3_XXS down-projections) means this regrow can trigger far
            // more often than "only once at startup" -- every time the
            // round-robin lands on a slot last sized for a DIFFERENT layer's
            // (smaller) slab.
            auto _regrow_t0 = std::chrono::high_resolution_clock::now();
            if (slot.pinned_buf) cudaFreeHost(slot.pinned_buf);
            cudaMallocHost((void**)&slot.pinned_buf, needed_bytes);
            double _regrow_ms = std::chrono::duration<double, std::milli>(
                std::chrono::high_resolution_clock::now() - _regrow_t0).count();
            PerfCounters::atomic_add(perf().t_staging_regrow, _regrow_ms / 1000.0);
            perf().n_staging_regrow.fetch_add(1, std::memory_order_relaxed);
            if (_regrow_ms > SLOW_MUTEX_WAIT_MS) {
                LOG_WARN("[SLOW-STAGING-REGROW] slot=%d cudaFreeHost+cudaMallocHost took %.2fms "
                         "(old_capacity=%.2fMB new=%.2fMB) -- held staging_mutex for the whole "
                         "duration, blocking any concurrent consume_staged\n",
                         slot_idx, slot.capacity / 1e6, needed_bytes / 1e6, _regrow_ms);
            }
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
            auto _wait_t0 = std::chrono::high_resolution_clock::now();
            std::lock_guard<std::mutex> lock(staging_mutex);
            double _wait_ms = std::chrono::duration<double, std::milli>(
                std::chrono::high_resolution_clock::now() - _wait_t0).count();
            PerfCounters::atomic_add(perf().t_staging_mutex_wait, _wait_ms / 1000.0);
            if (_wait_ms > SLOW_MUTEX_WAIT_MS) {
                LOG_WARN("[SLOW-MUTEX-WAIT] finish_prefetch blocked %.2fms acquiring staging_mutex "
                         "(layer=%d expert=%d)\n", _wait_ms, layer_idx, expert_id);
            }
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