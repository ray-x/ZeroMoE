// memory_guard.hpp -- background host-memory watchdog.
//
// WHY THIS EXISTS (all observed on real hardware 2026-09-04/05):
//   * host_ram_gb=96 on GLM while the page cache still held another model:
//     `posix_memalign(11.30MB)` took 1893 ms, 53 times. CPU pegged at 100%,
//     GPUs idle, prefill never finished. Nothing in the engine said "you are
//     out of memory" -- it just got slower until it died.
//   * host_ram_gb=120 historically caused 989K major faults/request.
//   * zram is the only swap here and quantized weights compress at 1.000x,
//     so this machine degrades off a CLIFF rather than a slope -- by the time
//     throughput visibly drops, it is already thrashing.
//
// Two jobs, deliberately separated:
//   1. WARN early (mem_guard_warn_mb) so a run that is heading for trouble
//      says so in the log while it is still healthy.
//   2. STOP growing (mem_guard_stop_mb) before the allocator starts taking
//      seconds per call. Refusing an allocation is a bounded, recoverable
//      cost -- the RAM tier simply re-reads that expert later. Thrashing is
//      not: it takes the whole run down and produces no result at all.
//
// MemAvailable (not MemFree) is the right signal: MemFree excludes reclaimable
// page cache and reads near-zero on a healthy machine, so gating on it would
// fire constantly. MemAvailable is the kernel's own estimate of what can be
// handed out without swapping.

#pragma once
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include "logger.hpp"
// Deliberately does NOT include perf_timer.hpp: that header includes this one
// (for the shutdown report), and the cycle only survives on #pragma once.

// VRAM headroom check for DYNAMIC (grow-at-runtime) device buffers.
//
// NOTE ON llama.cpp's 0.9: that fraction is applied at MODEL LOAD to decide
// how many layers to offload. Applying the same fraction-of-TOTAL per dynamic
// allocation does not work in this engine and was measured making things 2.6x
// SLOWER (14.97 -> 5.79 tok/s): MoEStream deliberately fills VRAM -- the
// expert cache alone (moe_cache_gb) sits at ~23.7 of 24 GB by design -- so a
// 90%-of-total test reads "already over budget" and refuses EVERY dynamic
// allocation forever (11,632 refusals in one run), permanently disabling the
// BF16 / attention / GDN fast paths.
//
// What actually matters for these buffers is FREE headroom: keep a reserve so
// CUDA contexts, cuBLAS workspaces and driver staging -- none of which go
// through our budgets -- always have room. Hence a floor on remaining free
// memory rather than a ceiling on used memory.
inline bool vram_has_headroom(size_t bytes, size_t vfree, size_t vtotal,
                              size_t reserve_bytes) {
    (void)vtotal;
    if (vfree == 0) return true;              // unknown -- do not block
    return bytes + reserve_bytes <= vfree;
}

// Pure arithmetic used by the admission ledger and its component tests.  Keep
// this separate from /proc/meminfo so a test can prove the important boundary
// cases without depending on the machine's current free memory.
inline bool host_budget_allows(size_t budget, size_t current, size_t bytes,
                               size_t watermark_bytes = 0) {
    if (budget == 0) return true;
    if (current > budget || bytes > budget - current) return false;
    return watermark_bytes <= budget - current - bytes;
}

class MemoryGuard {
public:
    // `host_ram_gb` is the process-wide host-memory ceiling, not merely a
    // suggestion for the RAM expert cache. Resident tensors, KV/state, RAM
    // slabs, graph source, and MTP scratch all share this ledger. A value of
    // zero preserves the historical auto-detect behavior.
    static void configure_host_budget(size_t bytes) {
        host_budget_bytes_.store(bytes, std::memory_order_relaxed);
        LOG_STREAM_INFO() << "[MEM-GUARD] host allocation budget: "
                          << (bytes ? (double)bytes / 1e9 : 0.0)
                          << (bytes ? " GB (hard process cap)"
                                    : " (auto; no explicit process cap)")
                          << "\n";
    }

    static size_t host_budget_bytes() {
        return host_budget_bytes_.load(std::memory_order_relaxed);
    }

    static size_t reserved_host_bytes() {
        return reserved_bytes_.load(std::memory_order_relaxed);
    }

    // Reads /proc/meminfo MemAvailable in MB. Returns -1 if unreadable (in
    // which case the guard disables itself rather than guessing).
    static long read_available_mb() {
        FILE* f = std::fopen("/proc/meminfo", "r");
        if (!f) return -1;
        char line[256];
        long kb = -1;
        while (std::fgets(line, sizeof(line), f)) {
            if (std::strncmp(line, "MemAvailable:", 13) == 0) {
                kb = std::strtol(line + 13, nullptr, 10);
                break;
            }
        }
        std::fclose(f);
        return kb < 0 ? -1 : kb / 1024;
    }

    void start(long warn_mb, long stop_mb, int poll_ms) {
        if (running_.load()) return;
        warn_mb_ = warn_mb;
        stop_mb_ = stop_mb;
        poll_ms_ = poll_ms > 0 ? poll_ms : 500;
        stop_limit_mb_.store(stop_mb_, std::memory_order_relaxed);
        if (read_available_mb() < 0) {
            LOG_WARN("[MEM-GUARD] /proc/meminfo unreadable -- guard disabled\n");
            return;
        }
        running_.store(true);
        thread_ = std::thread([this] { loop(); });
        LOG_STREAM_INFO() << "[MEM-GUARD] started: warn below " << warn_mb_
                          << " MB, STOP allocating below " << stop_mb_
                          << " MB, polling every " << poll_ms_ << " ms\n";
    }

    void stop() {
        if (!running_.exchange(false)) return;
        if (thread_.joinable()) thread_.join();
    }
    ~MemoryGuard() { stop(); }

    // Hot-path check. Any code about to grow host memory calls this first.
    // Deliberately a plain relaxed atomic read: this sits in the RAM tier's
    // allocation path, and the cost of being one poll interval stale is far
    // lower than the cost of synchronising.
    static bool allocation_blocked() {
        return blocked_.load(std::memory_order_relaxed);
    }

    // Synchronous, size-aware admission check for a new host allocation.
    // The background flag is only a low-latency early signal; a large
    // allocation must also account for its own bytes because the watchdog
    // may be one poll interval behind MemAvailable.
    static bool allocation_allowed(size_t bytes) {
        if (bytes == 0) return true;
        if (allocation_blocked()) return false;
        const size_t budget = host_budget_bytes();
        if (budget > 0) {
            const size_t current = reserved_host_bytes();
            if (!host_budget_allows(budget, current, bytes))
                return false;
        }
        const long available_mb = read_available_mb();
        if (available_mb < 0) return true; // preserve the guard's fail-open rule
        const size_t available = static_cast<size_t>(available_mb) * 1024ull * 1024ull;
        const long stop_mb = stop_limit_mb_.load(std::memory_order_relaxed);
        const size_t reserve = stop_mb > 0
                                   ? static_cast<size_t>(stop_mb) * 1024ull * 1024ull
                                   : 0;
        return available > reserve && bytes <= available - reserve;
    }

    // Reserve host bytes for allocations that remain live. This is used by
    // graph-owned buffers, whose allocations are outside the RAM slab
    // allocator. It closes the race where several source allocations each
    // pass an independent MemAvailable check and collectively cross the
    // safety margin. `watermark_bytes` is an additional one-time safety
    // margin for allocations that are not part of the reservation (GGML
    // contexts, I/O fluctuation, and transient buffers).
    static bool try_reserve_host_bytes(size_t bytes,
                                       size_t watermark_bytes = 0) {
        if (bytes == 0) return true;
        if (!allocation_allowed(bytes)) return false;
        const long available_mb = read_available_mb();
        if (available_mb < 0) {
            // Preserve the historical fail-open behavior for the live system
            // signal, but still enforce an explicit host_ram_gb admission cap
            // when /proc/meminfo is unavailable.
            const size_t host_budget = host_budget_bytes_.load(std::memory_order_relaxed);
            size_t current = reserved_bytes_.load(std::memory_order_relaxed);
            for (;;) {
                if (host_budget > 0 &&
                    !host_budget_allows(host_budget, current, bytes,
                                        watermark_bytes))
                    return false;
                if (reserved_bytes_.compare_exchange_weak(
                        current, current + bytes, std::memory_order_relaxed,
                        std::memory_order_relaxed))
                    return true;
            }
        }
        const size_t available = static_cast<size_t>(available_mb) * 1024ull * 1024ull;
        const long stop_mb = stop_limit_mb_.load(std::memory_order_relaxed);
        const size_t stop_reserve = stop_mb > 0
                                        ? static_cast<size_t>(stop_mb) * 1024ull * 1024ull
                                        : 0;
        if (available <= stop_reserve ||
            watermark_bytes > available - stop_reserve)
            return false;
        const size_t limit = available - stop_reserve - watermark_bytes;
        size_t current = reserved_bytes_.load(std::memory_order_relaxed);
        for (;;) {
            // `available` is the kernel's current free/reclaimable memory and
            // already reflects the bytes represented by `current`. Do not add
            // the ledger a second time here: the ledger's separate hard-cap
            // check below enforces current + bytes <= host_ram_gb, while this
            // check only asks whether the new allocation fits in today's real
            // MemAvailable low-water budget.
            if (bytes > limit) return false;
            const size_t host_budget = host_budget_bytes_.load(std::memory_order_relaxed);
            if (host_budget > 0 &&
                !host_budget_allows(host_budget, current, bytes,
                                    watermark_bytes))
                return false;
            if (reserved_bytes_.compare_exchange_weak(
                    current, current + bytes, std::memory_order_relaxed,
                    std::memory_order_relaxed))
                return true;
        }
    }

    static void release_host_bytes(size_t bytes) {
        if (!bytes) return;
        // A failed/partial initialization must never make the process-wide
        // counter wrap.  The CAS also makes release safe when graph teardown,
        // a worker, and the engine destructor overlap during error handling.
        size_t current = reserved_bytes_.load(std::memory_order_relaxed);
        for (;;) {
            if (current < bytes) {
                reserved_bytes_.store(0, std::memory_order_relaxed);
                return;
            }
            if (reserved_bytes_.compare_exchange_weak(
                    current, current - bytes, std::memory_order_relaxed,
                    std::memory_order_relaxed))
                return;
        }
    }

    static long last_available_mb() { return last_mb_.load(std::memory_order_relaxed); }
    static uint64_t block_events() { return n_block_.load(std::memory_order_relaxed); }
    static long min_seen_mb() { return min_mb_.load(std::memory_order_relaxed); }
    static long stop_limit_mb() { return stop_limit_mb_.load(std::memory_order_relaxed); }

private:
    void loop() {
        bool was_blocked = false, was_warned = false;
        while (running_.load()) {
            const long mb = read_available_mb();
            if (mb >= 0) {
                last_mb_.store(mb, std::memory_order_relaxed);
                long prev = min_mb_.load(std::memory_order_relaxed);
                while (mb < prev && !min_mb_.compare_exchange_weak(prev, mb)) {}

                const bool block = (mb < stop_mb_);
                blocked_.store(block, std::memory_order_relaxed);
                if (block && !was_blocked) {
                    n_block_.fetch_add(1, std::memory_order_relaxed);
                    LOG_ERROR("[MEM-GUARD] MemAvailable %ld MB < stop threshold %ld MB -- "
                              "BLOCKING further host allocations. The RAM tier will re-read "
                              "these experts instead of growing; that is bounded, whereas "
                              "allocating into thrash is not.\n", mb, stop_mb_);
                } else if (!block && was_blocked) {
                    LOG_STREAM_INFO() << "[MEM-GUARD] recovered: MemAvailable " << mb
                                      << " MB >= " << stop_mb_ << " MB, allocations re-enabled\n";
                }
                was_blocked = block;

                if (!block && mb < warn_mb_ && !was_warned) {
                    LOG_WARN("[MEM-GUARD] MemAvailable %ld MB is below the warn threshold "
                             "%ld MB -- host_ram_gb may be too large for this model on this "
                             "machine (allocation latency rises sharply past this point)\n",
                             mb, warn_mb_);
                    was_warned = true;
                } else if (mb >= warn_mb_) {
                    was_warned = false;
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(poll_ms_));
        }
    }

    std::thread thread_;
    std::atomic<bool> running_{false};
    long warn_mb_ = 4096, stop_mb_ = 4096;
    int poll_ms_ = 500;
    static inline std::atomic<bool> blocked_{false};
    static inline std::atomic<long> last_mb_{-1};
    static inline std::atomic<long> min_mb_{1L << 40};
    static inline std::atomic<uint64_t> n_block_{0};
    static inline std::atomic<long> stop_limit_mb_{4096};
    static inline std::atomic<size_t> reserved_bytes_{0};
    static inline std::atomic<size_t> host_budget_bytes_{0};
};
