#pragma once
#include "memory_guard.hpp"
#include "logger.hpp"
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <atomic>
#include <map>
#include <string>
#include <sstream>
#include <iomanip>
#include <vector>
#include <algorithm>

#ifdef __APPLE__
#include <mach/mach.h>
#include <sys/resource.h>
#else
#include <sys/resource.h>
#endif

#include "perf/perf_counters.hpp"

// RAII scoped timer that accumulates into an atomic<double>
struct ScopedTimer {
    std::atomic<double>& target;
    std::chrono::high_resolution_clock::time_point start;

    explicit ScopedTimer(std::atomic<double>& t)
        : target(t), start(std::chrono::high_resolution_clock::now()) {}

    ~ScopedTimer() {
        auto end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(end - start).count();
        // Relaxed atomic add (non-lock-free double requires CAS loop)
        double old = target.load(std::memory_order_relaxed);
        while (!target.compare_exchange_weak(old, old + elapsed,
                    std::memory_order_relaxed, std::memory_order_relaxed)) {}
    }
};

// Global instance
inline PerfCounters& perf() {
    static PerfCounters instance;
    return instance;
}

// Mirror-fd registry -- lets io_uring_backend.hpp's submit_and_wait
// classify which physical drive serviced a completed read (bytes_disk_
// read_primary/_mirror above) WITHOUT that low-level header needing to
// depend on gguf_loader.hpp/GGUFModel at all. Same function-local-static,
// set-once-read-everywhere pattern this project uses throughout (cpu_only_
// flag, cpu_affinity_layer_flags, etc.) -- GGUFModel::open_mirror registers
// each shard's mirror fd here the moment it opens successfully; every
// OTHER fd (primary shards, the base file_fd) is implicitly "primary" by
// not appearing in this list. A linear scan is fine: real shard counts are
// single digits.
inline std::vector<int>& mirror_fd_registry() {
    static std::vector<int> v;
    return v;
}
inline void register_mirror_fd(int fd) {
    if (fd >= 0) mirror_fd_registry().push_back(fd);
}
inline bool is_mirror_fd(int fd) {
    auto& v = mirror_fd_registry();
    return std::find(v.begin(), v.end(), fd) != v.end();
}

// Convenience macros
// `##` suppresses expansion of its operands, so the old
// `_perf_timer_##__LINE__` spelled the identifier `_perf_timer___LINE__` on
// every call instead of a line-specific name: two PERF_SCOPE calls in one
// scope were a redeclaration error. Expand the counter first, then paste.
#define PERF_SCOPE_CAT_(a, b) a##b
#define PERF_SCOPE_CAT(a, b) PERF_SCOPE_CAT_(a, b)
#define PERF_SCOPE(counter) \
  ScopedTimer PERF_SCOPE_CAT(_perf_timer_, __COUNTER__)(perf().counter)
#define PERF_START(name) auto _perf_##name = std::chrono::high_resolution_clock::now()
#define PERF_END(name, counter) do { \
    auto _perf_end = std::chrono::high_resolution_clock::now(); \
    double _elapsed = std::chrono::duration<double>(_perf_end - _perf_##name).count(); \
    double _old = perf().counter.load(std::memory_order_relaxed); \
    while (!perf().counter.compare_exchange_weak(_old, _old + _elapsed, \
                std::memory_order_relaxed, std::memory_order_relaxed)) {} \
} while(0)

// ---- Diagnostic-only, per-tensor-name dense_proj timing ----
// Added to check whether specific attention projection weights
// (attn_q_a/attn_q_b/attn_kv_a_mqa) cost enough through their fused
// K-quant/IQ kernel to justify a BF16 fast-path extension -- the same
// question attn_output's ~623ms/token profiling answered for that tensor,
// but attn_q_a/q_b/kv_a_mqa are individually much smaller and share
// attention_layer.cpp's existing t_attn_qproj/t_attn_kvproj buckets with
// each other AND with non-matmul work (RoPE), so those buckets alone can't
// isolate any one tensor's own cost.
//
// Deliberately independent of PerfCounters' Snapshot/operator-/snapshot()
// triple above -- every field added there requires updating three
// positions in lockstep (a mismatch already caused a real narrowing-
// conversion build break once this session, see CLAUDE.md). This is a
// narrower, possibly-temporary investigation, not a permanent session-wide
// counter, so it gets its own much simpler single-threaded accumulator
// instead: safe because dense_proj/attention are only ever called from the
// single main decode thread (same assumption this project already relies
// on elsewhere, e.g. the wuk_cache/wuv_cache static caches).
struct NamedTimerStats { double total_ms = 0.0; uint64_t calls = 0; };

inline std::map<std::string, NamedTimerStats>& named_timer_stats() {
    static std::map<std::string, NamedTimerStats> stats;
    return stats;
}

// RAII scope timer: construct at the top of the block being measured,
// destructor records elapsed ms under `name`. e.g.:
//   { NamedTimerScope _t("attn_q_a"); dense_proj(engine, p + "attn_q_a.weight", ...); }
struct NamedTimerScope {
    std::string name;
    std::chrono::high_resolution_clock::time_point t0;
    explicit NamedTimerScope(std::string n)
        : name(std::move(n)), t0(std::chrono::high_resolution_clock::now()) {}
    ~NamedTimerScope() {
        double ms = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t0).count();
        auto& s = named_timer_stats()[name];
        s.total_ms += ms;
        s.calls += 1;
    }
};

// ---- end-of-run reports (defined in src/perf_report_stats.cpp) ----
void report_named_timers();
void report_layer_miss_stats();
void report_pilot_wide_stats();
void report_prefill_lookahead_stats();
void report_prefill_window_stats();
void report_moe_bucket_stats();
void report_pilot_recall_by_layer();
void report_qwen4exp_pilot_stats();
