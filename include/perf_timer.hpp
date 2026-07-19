#pragma once
#include "logger.hpp"
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <atomic>

#ifdef __APPLE__
#include <mach/mach.h>
#include <sys/resource.h>
#else
#include <sys/resource.h>
#endif

struct PerfCounters {
    // Cumulative timings (seconds)
    std::atomic<double> t_disk_read{0};     // Expert slab pread from disk
    std::atomic<double> t_host_alloc{0};    // posix_memalign + setup
    std::atomic<double> t_gpu_upload{0};    // cudaMemcpy H2D
    std::atomic<double> t_gpu_download{0};  // cudaMemcpy D2H -- was bundled into
                                             // whatever counter happened to wrap the
                                             // next call (usually none, since
                                             // expert_ffn's GPU path had ZERO
                                             // PERF_START/PERF_END until this was
                                             // added); tracked separately from
                                             // t_gpu_upload since expert_ffn does 3
                                             // D2H copies per expert vs 2 H2D.
    std::atomic<double> t_gpu_compute{0};   // Kernel execution time (real device
                                             // time, via an explicit sync point right
                                             // after each kernel launch -- launches
                                             // are async, so timing the launch call
                                             // alone would only capture negligible
                                             // CPU-side dispatch overhead, not actual
                                             // GPU execution time).
    std::atomic<double> t_cpu_expert{0};    // expert_ffn's CPU dequant-matmul
                                             // fallback path (cpu_quant_matmul x3),
                                             // for direct comparison against the GPU
                                             // path's total (upload+compute+download).
    std::atomic<double> t_cpu_dequant{0};   // Time spent in dequant_row (K-quant/
                                             // IQ bit-unpacking) inside
                                             // cpu_quant_matmul, separate from the
                                             // dot product -- answers whether the
                                             // CPU expert cost is dominated by
                                             // unpacking ggml's reference
                                             // (unvectorized) dequant or by the
                                             // actual matmul, since only the dot
                                             // product was AVX-512 vectorized so
                                             // far. NOTE: summed across every OMP
                                             // worker thread (aggregate CPU-core-
                                             // seconds), NOT wall-clock -- with N
                                             // threads this can read up to ~N x
                                             // larger than the wall-clock portion
                                             // of t_cpu_expert it's part of. Only
                                             // the RATIO of this to t_cpu_dot is
                                             // meaningful, not the absolute value
                                             // against t_cpu_expert.
    std::atomic<double> t_cpu_dot{0};       // Same caveat as t_cpu_dequant: time
                                             // in the post-dequant AVX-512 dot
                                             // product (dot_f32_avx512), summed
                                             // across OMP worker threads.
    std::atomic<double> t_attn{0};          // MLA attention forward (total)
    // Per-phase breakdown of t_attn, added to root-cause an intermittent
    // severe stall (~90s in one token) that showed the CPU idle/cool at the
    // time (ruled out: thermal throttling, memory pressure, other
    // processes, disk I/O) -- meaning something inside attention is
    // BLOCKING without burning CPU, most likely a GPU call. attn_q_a/q_b,
    // attn_kv_a_mqa, and attn_output all go through dense_proj, which can
    // dispatch to GPU if the weight is VRAM-resident -- a blocking
    // cudaMemcpy/cudaStreamSynchronize stalling there (e.g. contention with
    // the async expert-promotion traffic on the same or a P2P-linked GPU)
    // would show up as elevated wall-clock time here with the CPU idle,
    // exactly matching what was observed.
    std::atomic<double> t_attn_norm{0};     // attn_norm rmsnorm
    std::atomic<double> t_attn_qproj{0};    // attn_q_a -> q_a_norm -> attn_q_b + RoPE
    std::atomic<double> t_attn_kvproj{0};   // attn_kv_a_mqa + norm/RoPE + Lc/Rc cache write
    std::atomic<double> t_attn_absorb{0};   // attn_k_b/attn_v_b cache lookup-or-fill (should be ~0 after layer's first call)
    std::atomic<double> t_attn_score{0};    // the absorbed score/softmax/value-aggregation loop
    std::atomic<double> t_attn_outproj{0};  // attn_output dense_proj

    // Phase split for CudaManager::dense_matmul_host -- ALL dense_proj GPU
    // calls share these (not just attention's), added after confirming
    // attn_output's dense_proj call *does* successfully GPU-dispatch every
    // time (zero "[DENSE] ... CPU dequant-matmul fallback" lines logged)
    // yet still costs ~623ms/token, dwarfing the qproj dense_proj calls
    // (~120ms combined for two calls) -- this splits WHERE inside that
    // successful GPU call the time actually goes, same method that found
    // the real issue in the expert-FFN GPU path earlier this session.
    std::atomic<double> t_dense_gpu_up{0};
    std::atomic<double> t_dense_gpu_compute{0};
    std::atomic<double> t_dense_gpu_down{0};
    std::atomic<uint64_t> n_dense_gpu_calls{0};
    std::atomic<double> t_moe_route{0};     // Router + top-k selection
    std::atomic<double> t_moe_expert{0};    // Expert FFN computation
    std::atomic<double> t_prefetch{0};      // fadvise/prefetch overhead
    std::atomic<double> t_dense_load{0};    // Initial resident tensor loading
    std::atomic<double> t_consume_staged{0};// Time in consume_staged (memcpy a
                                             // prefetched slab out of the staging
                                             // pool into the cache slot) -- the
                                             // Phase A "PIPE" overlap path. Near-
                                             // zero means the prefetcher isn't
                                             // feeding the miss path yet.

    // Counters
    std::atomic<uint64_t> n_expert_cache_hit{0};
    std::atomic<uint64_t> n_expert_cache_miss{0};
    std::atomic<uint64_t> n_expert_promote{0};
    std::atomic<uint64_t> n_tokens_generated{0};
    std::atomic<uint64_t> n_expert_gpu_calls{0};  // expert_ffn calls that took the GPU-resident path
    std::atomic<uint64_t> n_expert_cpu_calls{0};  // expert_ffn calls that took the CPU fallback path
    std::atomic<int64_t> bytes_disk_read{0};
    std::atomic<int64_t> bytes_gpu_uploaded{0};
    // Phase A (colibri-port) counters:
    std::atomic<uint64_t> n_pipe_hits{0};          // cache misses served from the prefetcher's staging pool (no pread) -- the higher this is vs n_expert_cache_miss, the more disk I/O was hidden behind compute
    std::atomic<uint64_t> n_pinned_eviction_skips{0}; // times evict_lru skipped a pinned/in-flight slot -- non-zero under pressure proves the pinning policy is protecting the hot set
    std::atomic<uint64_t> n_pinned_resident{0};    // experts currently held as non-evictable VRAM residents (pinned). Gauge-ish: incremented on pin, decremented on unpin.

    // Plain-value point-in-time copy of the cumulative counters above, so
    // callers can diff two snapshots to get a per-token or per-request
    // breakdown without disturbing the global running totals.
    struct Snapshot {
        double t_disk_read=0, t_host_alloc=0, t_gpu_upload=0, t_gpu_download=0, t_gpu_compute=0, t_cpu_expert=0;
        double t_cpu_dequant=0, t_cpu_dot=0;
        double t_attn=0;
        double t_attn_norm=0, t_attn_qproj=0, t_attn_kvproj=0, t_attn_absorb=0, t_attn_score=0, t_attn_outproj=0;
        double t_dense_gpu_up=0, t_dense_gpu_compute=0, t_dense_gpu_down=0;
        double t_moe_route=0, t_moe_expert=0, t_prefetch=0, t_dense_load=0, t_consume_staged=0;
        uint64_t n_expert_cache_hit=0, n_expert_cache_miss=0, n_expert_promote=0, n_tokens_generated=0;
        uint64_t n_expert_gpu_calls=0, n_expert_cpu_calls=0, n_dense_gpu_calls=0;
        int64_t bytes_disk_read=0, bytes_gpu_uploaded=0;
        uint64_t n_pipe_hits=0, n_pinned_eviction_skips=0, n_pinned_resident=0;

        Snapshot operator-(const Snapshot& b) const {
            return Snapshot{
                t_disk_read - b.t_disk_read, t_host_alloc - b.t_host_alloc,
                t_gpu_upload - b.t_gpu_upload, t_gpu_download - b.t_gpu_download,
                t_gpu_compute - b.t_gpu_compute, t_cpu_expert - b.t_cpu_expert,
                t_cpu_dequant - b.t_cpu_dequant, t_cpu_dot - b.t_cpu_dot,
                t_attn - b.t_attn,
                t_attn_norm - b.t_attn_norm, t_attn_qproj - b.t_attn_qproj,
                t_attn_kvproj - b.t_attn_kvproj, t_attn_absorb - b.t_attn_absorb,
                t_attn_score - b.t_attn_score, t_attn_outproj - b.t_attn_outproj,
                t_dense_gpu_up - b.t_dense_gpu_up, t_dense_gpu_compute - b.t_dense_gpu_compute,
                t_dense_gpu_down - b.t_dense_gpu_down,
                t_moe_route - b.t_moe_route,
                t_moe_expert - b.t_moe_expert, t_prefetch - b.t_prefetch,
                t_dense_load - b.t_dense_load, t_consume_staged - b.t_consume_staged,
                    n_expert_cache_hit - b.n_expert_cache_hit,
                    n_expert_cache_miss - b.n_expert_cache_miss,
                    n_expert_promote - b.n_expert_promote,
                    n_tokens_generated - b.n_tokens_generated,
                    n_expert_gpu_calls - b.n_expert_gpu_calls,
                    n_expert_cpu_calls - b.n_expert_cpu_calls,
                    n_dense_gpu_calls - b.n_dense_gpu_calls,
                    bytes_disk_read - b.bytes_disk_read,
                    bytes_gpu_uploaded - b.bytes_gpu_uploaded,
                    n_pipe_hits - b.n_pipe_hits,
                    n_pinned_eviction_skips - b.n_pinned_eviction_skips,
                    n_pinned_resident - b.n_pinned_resident
            };
        }
    };

    Snapshot snapshot() const {
        return Snapshot{
            t_disk_read.load(), t_host_alloc.load(), t_gpu_upload.load(), t_gpu_download.load(),
            t_gpu_compute.load(), t_cpu_expert.load(),
            t_cpu_dequant.load(), t_cpu_dot.load(),
            t_attn.load(),
            t_attn_norm.load(), t_attn_qproj.load(), t_attn_kvproj.load(),
            t_attn_absorb.load(), t_attn_score.load(), t_attn_outproj.load(),
            t_dense_gpu_up.load(), t_dense_gpu_compute.load(), t_dense_gpu_down.load(),
            t_moe_route.load(), t_moe_expert.load(), t_prefetch.load(), t_dense_load.load(), t_consume_staged.load(),
            n_expert_cache_hit.load(), n_expert_cache_miss.load(), n_expert_promote.load(), n_tokens_generated.load(),
            n_expert_gpu_calls.load(), n_expert_cpu_calls.load(), n_dense_gpu_calls.load(),
            bytes_disk_read.load(), bytes_gpu_uploaded.load(),
            n_pipe_hits.load(), n_pinned_eviction_skips.load(), n_pinned_resident.load()
        };
    }

    // CAS-loop add for atomic<double> (no native fetch_add for double until
    // C++20's atomic<double>::fetch_add, and this project targets a mix of
    // GCC versions -- keep the same pattern ScopedTimer/PERF_END already use).
    static void atomic_add(std::atomic<double>& target, double v) {
        double old = target.load(std::memory_order_relaxed);
        while (!target.compare_exchange_weak(old, old + v,
                    std::memory_order_relaxed, std::memory_order_relaxed)) {}
    }

    // Prints a single-line latency breakdown for a diffed window. Called once
    // PER GENERATED TOKEN (see generation_loop.hpp) -- DEBUG level, since at
    // INFO this would print one line per token for the entire response
    // (hundreds of lines for a normal reply) with no way to turn it off.
    // gpu_up/gpu_dn/gpu_cmp/cpu_exp show whether GPU round-trip overhead or
    // CPU fallback dominates THIS token; use `log_level = "DEBUG"` to see
    // the per-token pattern (useful for "is this latency- or throughput-
    // bound", which an aggregate summary averages away).
    static void print_delta(const char* label, double wall_ms, const Snapshot& d) {
        uint64_t hits = d.n_expert_cache_hit, misses = d.n_expert_cache_miss;
        LOG_DEBUG(
            "[%s] wall=%8.2fms  attn=%7.2fms [norm=%6.2f qproj=%6.2f kvproj=%6.2f absorb=%6.2f "
            "score=%6.2f outproj=%6.2f] route=%6.2fms expert=%7.2fms "
            "gpu_up=%6.2fms gpu_dn=%6.2fms gpu_cmp=%6.2fms cpu_exp=%7.2fms (dequant=%6.2fms dot=%6.2fms) "
            "dense_gpu[up=%6.2f cmp=%6.2f dn=%6.2f calls=%llu] "
            "disk=%6.2fms  cache=%lluH/%lluM (%.0f%%) gpu_calls=%llu cpu_calls=%llu "
            "pipe=%llu pin_skip=%llu pin=%llu\n",
            label, wall_ms,
            d.t_attn * 1000.0,
            d.t_attn_norm * 1000.0, d.t_attn_qproj * 1000.0, d.t_attn_kvproj * 1000.0,
            d.t_attn_absorb * 1000.0, d.t_attn_score * 1000.0, d.t_attn_outproj * 1000.0,
            d.t_moe_route * 1000.0, d.t_moe_expert * 1000.0,
            d.t_gpu_upload * 1000.0, d.t_gpu_download * 1000.0, d.t_gpu_compute * 1000.0,
            d.t_cpu_expert * 1000.0, d.t_cpu_dequant * 1000.0, d.t_cpu_dot * 1000.0,
            d.t_dense_gpu_up * 1000.0, d.t_dense_gpu_compute * 1000.0, d.t_dense_gpu_down * 1000.0,
            (unsigned long long)d.n_dense_gpu_calls,
            d.t_disk_read * 1000.0,
            (unsigned long long)hits, (unsigned long long)misses,
            (hits + misses) > 0 ? 100.0 * hits / (hits + misses) : 0.0,
            (unsigned long long)d.n_expert_gpu_calls, (unsigned long long)d.n_expert_cpu_calls,
            (unsigned long long)d.n_pipe_hits, (unsigned long long)d.n_pinned_eviction_skips,
            (unsigned long long)d.n_pinned_resident);
    }

    // Concise INFO-level summary for one finished request (prompt + all
    // generated tokens) -- unlike print_delta (DEBUG, one line per token),
    // this is the always-visible answer to "how fast was that and where did
    // the time go", printed once when generation completes rather than only
    // at process exit (dump_report, which a killed/long-running process may
    // never reach, and which mixes every request together over the whole
    // program lifetime rather than isolating this one).
    static void print_summary(const char* label, double wall_ms, int n_tokens, const Snapshot& d) {
        double gpu_round_trip = d.t_gpu_upload + d.t_gpu_compute + d.t_gpu_download;
        uint64_t hits = d.n_expert_cache_hit, misses = d.n_expert_cache_miss;
        uint64_t gcalls = d.n_expert_gpu_calls, ccalls = d.n_expert_cpu_calls;
        double tok_s = wall_ms > 0 ? n_tokens / (wall_ms / 1000.0) : 0.0;
        double dense_gpu_total = d.t_dense_gpu_up + d.t_dense_gpu_compute + d.t_dense_gpu_down;
        LOG_INFO(
            "[%s] %d tokens in %.2fs (%.2f tok/s)  attn=%.2fs [norm=%.2f qproj=%.2f kvproj=%.2f "
            "absorb=%.2f score=%.2f outproj=%.2f] route=%.2fs "
            "expert=%.2fs (gpu=%.2fs [up=%.2fs cmp=%.2fs dn=%.2fs] cpu=%.2fs [dequant=%.2fs dot=%.2fs]) "
            "dense_gpu=%.2fs [up=%.2fs cmp=%.2fs dn=%.2fs calls=%llu] "
            "disk=%.2fs  cache=%.0f%% hit  expert_calls=%lluGPU/%lluCPU "
            "pipe=%llu(pin_skip=%llu pin=%llu)\n",
            label, n_tokens, wall_ms / 1000.0, tok_s,
            d.t_attn,
            d.t_attn_norm, d.t_attn_qproj, d.t_attn_kvproj, d.t_attn_absorb, d.t_attn_score, d.t_attn_outproj,
            d.t_moe_route, d.t_moe_expert,
            gpu_round_trip, d.t_gpu_upload, d.t_gpu_compute, d.t_gpu_download, d.t_cpu_expert,
            d.t_cpu_dequant, d.t_cpu_dot,
            dense_gpu_total, d.t_dense_gpu_up, d.t_dense_gpu_compute, d.t_dense_gpu_down,
            (unsigned long long)d.n_dense_gpu_calls,
            d.t_disk_read,
            (hits + misses) > 0 ? 100.0 * hits / (hits + misses) : 0.0,
            (unsigned long long)gcalls, (unsigned long long)ccalls,
            (unsigned long long)d.n_pipe_hits, (unsigned long long)d.n_pinned_eviction_skips,
            (unsigned long long)d.n_pinned_resident);
    }

    void dump_report() const {
        double total = t_disk_read + t_host_alloc + t_gpu_upload + t_gpu_compute +
                       t_attn + t_moe_route + t_moe_expert;
        double gpu_round_trip = t_gpu_upload + t_gpu_compute + t_gpu_download;
        LOG_INFO("\n========== MoEStream Performance Report ==========\n");
        LOG_INFO("Dense load (init):   %8.3f s\n", t_dense_load.load());
        LOG_INFO("Disk read (expert):  %8.3f s  (%6.2f GB/s, %.2f GB total)\n",
                 t_disk_read.load(),
                 t_disk_read > 0 ? (bytes_disk_read / 1e9) / t_disk_read : 0.0,
                 bytes_disk_read / 1e9);
        LOG_INFO("Host alloc/setup:    %8.3f s\n", t_host_alloc.load());
        LOG_INFO("GPU upload (H2D):    %8.3f s  (%6.2f GB/s, %.2f GB total)\n",
                 t_gpu_upload.load(),
                 t_gpu_upload > 0 ? (bytes_gpu_uploaded / 1e9) / t_gpu_upload : 0.0,
                 bytes_gpu_uploaded / 1e9);
        LOG_INFO("Promotions (H2D):    %lu uploads / %.2f GB total\n",
            (unsigned long)n_expert_promote.load(),
            bytes_gpu_uploaded.load() / 1e9);
        LOG_INFO("GPU compute:         %8.3f s\n", t_gpu_compute.load());
        LOG_INFO("GPU download (D2H):  %8.3f s\n", t_gpu_download.load());
        LOG_INFO("GPU round-trip total:%8.3f s  (upload+compute+download, expert_ffn only)\n", gpu_round_trip);
        LOG_INFO("CPU expert (fallback):%7.3f s  (dequant=%.3fs dot=%.3fs)\n",
                 t_cpu_expert.load(), t_cpu_dequant.load(), t_cpu_dot.load());
        LOG_INFO("Attention (MLA):     %8.3f s  (norm=%.3f qproj=%.3f kvproj=%.3f absorb=%.3f score=%.3f outproj=%.3f)\n",
                 t_attn.load(), t_attn_norm.load(), t_attn_qproj.load(), t_attn_kvproj.load(),
                 t_attn_absorb.load(), t_attn_score.load(), t_attn_outproj.load());
        LOG_INFO("Dense GPU dispatch:  %8.3f s  (up=%.3f cmp=%.3f dn=%.3f, %lu calls, %.3fms/call avg)\n",
                 t_dense_gpu_up.load() + t_dense_gpu_compute.load() + t_dense_gpu_down.load(),
                 t_dense_gpu_up.load(), t_dense_gpu_compute.load(), t_dense_gpu_down.load(),
                 (unsigned long)n_dense_gpu_calls.load(),
                 n_dense_gpu_calls > 0
                     ? 1000.0 * (t_dense_gpu_up.load() + t_dense_gpu_compute.load() + t_dense_gpu_down.load()) / n_dense_gpu_calls.load()
                     : 0.0);
        LOG_INFO("MoE routing:         %8.3f s\n", t_moe_route.load());
        LOG_INFO("MoE expert FFN:      %8.3f s\n", t_moe_expert.load());
        LOG_INFO("Prefetch overhead:   %8.3f s\n", t_prefetch.load());
        LOG_INFO("--------------------------------------------------\n");
        uint64_t gcalls = n_expert_gpu_calls, ccalls = n_expert_cpu_calls;
        LOG_INFO("Expert path split:   %lu GPU calls / %lu CPU calls (%.1f%% GPU)\n",
                 (unsigned long)gcalls, (unsigned long)ccalls,
                 (gcalls + ccalls) > 0 ? 100.0 * gcalls / (gcalls + ccalls) : 0.0);
        if (gcalls > 0)
            LOG_INFO("  avg GPU round-trip: %6.3f ms/call (up=%.3f cmp=%.3f dn=%.3f)\n",
                     1000.0 * gpu_round_trip / gcalls,
                     1000.0 * t_gpu_upload / gcalls, 1000.0 * t_gpu_compute / gcalls,
                     1000.0 * t_gpu_download / gcalls);
        if (ccalls > 0)
            LOG_INFO("  avg CPU expert:     %6.3f ms/call\n", 1000.0 * t_cpu_expert / ccalls);
        uint64_t hits = n_expert_cache_hit, misses = n_expert_cache_miss;
        LOG_INFO("Expert cache:        %lu hit / %lu miss (%.1f%% hit rate)\n",
                 (unsigned long)hits, (unsigned long)misses,
                 (hits + misses) > 0 ? 100.0 * hits / (hits + misses) : 0.0);
        LOG_INFO("Prefetch (PIPE):     %lu misses served from staging / %lu total misses (%.1f%% served)\n",
                 (unsigned long)n_pipe_hits.load(), (unsigned long)misses,
                 misses > 0 ? 100.0 * n_pipe_hits.load() / misses : 0.0);
        LOG_INFO("Pinned residents:    %lu current / %lu eviction-skips\n",
                 (unsigned long)n_pinned_resident.load(),
                 (unsigned long)n_pinned_eviction_skips.load());
        LOG_INFO("Tokens generated:    %lu\n", (unsigned long)n_tokens_generated.load());
        if (n_tokens_generated > 0 && total > 0) {
            LOG_INFO("Throughput:          %.2f tok/s\n",
                     n_tokens_generated.load() / total);
        }
        LOG_INFO("RSS:                 %.2f GB\n", rss_gb());
        LOG_INFO("===================================================\n\n");
    }

    static double rss_gb() {
#ifdef __APPLE__
        struct mach_task_basic_info info;
        mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
        if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t)&info, &count) == KERN_SUCCESS)
            return info.resident_size / 1e9;
        return 0;
#else
        struct rusage r;
        getrusage(RUSAGE_SELF, &r);
        return r.ru_maxrss / (1024.0 * 1024.0); // Linux: KB
#endif
    }
};

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

// Convenience macros
#define PERF_SCOPE(counter) ScopedTimer _perf_timer_##__LINE__(perf().counter)
#define PERF_START(name) auto _perf_##name = std::chrono::high_resolution_clock::now()
#define PERF_END(name, counter) do { \
    auto _perf_end = std::chrono::high_resolution_clock::now(); \
    double _elapsed = std::chrono::duration<double>(_perf_end - _perf_##name).count(); \
    double _old = perf().counter.load(std::memory_order_relaxed); \
    while (!perf().counter.compare_exchange_weak(_old, _old + _elapsed, \
                std::memory_order_relaxed, std::memory_order_relaxed)) {} \
} while(0)
