#pragma once
#include "logger.hpp"
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <atomic>
#include <map>
#include <string>
#include <vector>
#include <algorithm>

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
    // Untimed setup before the first named sub-phase (arch-config reads,
    // rope_theta lookup via get_config_float, string prefix build) -- added
    // after the six phases below plus t_attn itself failed to fully
    // reconcile (sub-phases summed to only ~35-40% of t_attn across
    // multiple real runs, even with WUK/WUV's one-time cost -- see absorb's
    // own comment -- confirmed via a same-process two-request test to be
    // warm on the second call). Static review of every line in this
    // preamble suggested it should be microseconds, not the tens of
    // seconds needed to explain the gap -- this timer stayed near-zero on
    // every real run, correctly ruling this out as the explanation (see
    // t_prefill_total below for the real one).
    std::atomic<double> t_attn_preamble{0};
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

    // Added to root-cause a real, reproducible multi-second-per-token stall
    // (up to 7.5s for a SINGLE cache-miss service call, confirmed via a
    // per-miss loaded_from_disk= latency histogram showing normal p50/p90
    // but extreme outliers) that could NOT be explained by raw disk
    // throughput (median/p90 individual reads were completely normal --
    // only a handful of calls were catastrophically slow, which rules out
    // "the drive is slow" and points at something blocking intermittently).
    // These isolate the two most likely culprits: (1) cudaMallocHost/
    // cudaFreeHost inside the prefetcher's staging-slot regrow, which is a
    // documented-slow CUDA operation (pinned/page-locked memory allocation)
    // that this codebase was calling WHILE HOLDING staging_mutex -- so any
    // concurrent consume_staged() call (the decode thread servicing a cache
    // miss) would block for the full regrow duration; and (2) generic
    // cross-thread contention on GpuTopology::cache_mutex, shared between
    // the main decode thread's inline hit-triggered promotions
    // (RAMSafetyManager::get_expert Step 1, especially aggressive when
    // promote_hit_threshold is low) and the prefetcher worker thread's own
    // background promotions/evictions.
    std::atomic<double> t_staging_regrow{0};      // time inside acquire_staging_slot's cudaFreeHost+cudaMallocHost pair, held under staging_mutex
    std::atomic<uint64_t> n_staging_regrow{0};    // how many times a staging slot needed to grow (different layers' slab sizes rotating through a small fixed pool of slots is the expected trigger)
    std::atomic<double> t_staging_mutex_wait{0};  // total time ANY caller (consume_staged/acquire_staging_slot/finish_prefetch) spent BLOCKED acquiring staging_mutex -- high values here directly prove cross-thread contention on this lock, separate from time spent doing real work once held
    std::atomic<double> t_cache_mutex_wait{0};    // total time ANY caller spent BLOCKED acquiring GpuTopology::cache_mutex (find_expert/pin_expert/evict_lru/has_free_slot/upload_expert/upload_expert_async)

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
    std::atomic<uint64_t> n_pinned_eviction_skips{0}; // MEANING CHANGED once GpuTopology::unpinned_slot_indices landed: evict_lru no longer visits permanently-pinned slots at all (they're excluded from the scan at the source), so this now only increments on the DEFENSIVE fallback check -- should read 0 on a healthy run; nonzero means unpinned_slot_indices desynced from the real `pinned` flags somewhere (see gpu_topology.hpp's evict_lru). Previously (before that optimization) this counted every pinned slot skipped during a normal scan, and nonzero was the EXPECTED/healthy signal -- the polarity of "what a nonzero value means" flipped with that change.
    std::atomic<uint64_t> n_temp_pin_eviction_skips{0}; // times evict_lru skipped a slot TEMPORARILY held for an in-flight decode-time GPU dispatch (ExpertLocation::temp_pin_count > 0) -- distinct from n_pinned_eviction_skips (permanent pins); nonzero proves the compute-in-flight protection is actually engaging
    std::atomic<uint64_t> n_upload_pending_eviction_skips{0}; // times evict_lru skipped a slot whose async H2D upload was CONFIRMED still genuinely in flight (re-polled via cudaEventQuery, not just trusting a stale flag) -- the real "must not evict" count
    std::atomic<uint64_t> n_stale_upload_ready_reclaimed{0}; // times evict_lru found upload_ready==false but the event had ACTUALLY already completed (a speculative pilot-prefetch promotion that was never subsequently confirmed via find_expert) and flipped it back to true itself -- nonzero proves real cache capacity was being silently leaked before this fix; see gpu_topology.hpp's evict_lru comment
    std::atomic<uint64_t> n_stale_expert_index_hits{0}; // times find_expert's O(1) expert_index pointed at a slot that didn't actually match the requested (layer,expert_id) -- should be ALWAYS ZERO by construction (evict_lru/upload_expert/upload_expert_async are the only mutation sites, all under cache_mutex); nonzero means the index and expert_vram_cache have desynced somewhere and find_expert is safely falling back to its linear scan, but the root cause needs investigating
    std::atomic<uint64_t> n_pinned_resident{0};    // experts currently held as non-evictable VRAM residents (pinned). Gauge-ish: incremented on pin, decremented on unpin.
    std::atomic<uint64_t> n_pilot_hints_skipped{0}; // speculative pilot-prefetch hints skipped because the expert was ALREADY host-cache-resident (a guaranteed instant hit needing no read) -- measures how much prefetch-worker bandwidth this avoids wasting; compare against n_pipe_hits/n_expert_cache_miss before/after to see the effect
    std::atomic<uint64_t> n_vram_evict_demoted{0};  // VRAM evictions salvaged into the RAM tier via a D2H copy instead of being dropped outright (vram_evict_demote_to_ram=true) -- nonzero proves the salvage path is engaging; compare against Expert cache's disk-miss count for the real win (a demoted expert re-hit later serves from RAM, no disk read at all)
    std::atomic<uint64_t> n_vram_evict_demote_skipped{0}; // demote attempts that declined for a benign reason (already RAM-resident, or the layer uses the legacy per-expert-tensor format with no packed_slab_geometry) -- not an error, just nothing to do
    std::atomic<uint64_t> n_vram_evict_demote_failed{0};  // demote attempts that hit a real problem (cudaMemcpy D2H failure, size mismatch) -- should stay at/near 0; a nonzero count is worth investigating but never corrupts anything, since a failed demote just leaves the RAM slot empty rather than half-populated
    // PIPE consume_staged failure-reason split -- added after a real run
    // showed Prefetch (PIPE) served collapse to 0.1% (16/11247 misses),
    // far below this project's own previously-documented ~24-25% baseline,
    // with no obvious code regression found on inspection. These two
    // counters separate two very different root causes that n_pipe_hits
    // alone can't distinguish, so the next real run's data (not more
    // guessing) can determine which fix actually applies:
    std::atomic<uint64_t> n_pipe_miss_no_match{0};    // consume_staged found NO staging slot at all for this (layer,expert) -- either the pilot never predicted this exact expert, or it did but the slot was already round-robin-recycled for a later hint before this miss arrived. Points at PREDICTION ACCURACY / staging POOL SIZE as the lever, not raw read speed.
    std::atomic<uint64_t> n_pipe_miss_not_ready{0};    // consume_staged found a matching (layer,expert) slot but its pread hadn't finished yet -- the pilot's prediction was CORRECT, this was purely a timing loss. Points at making the staged read start earlier / finish faster (or growing NUM_STAGING_SLOTS so more predictions survive the wait) as the lever, not prediction accuracy.

    // Sum of every request's OWN measured wall time (request_wall_ms from
    // generate_text, the same number the [REQUEST] line itself prints),
    // accumulated across the whole session. Deliberately NOT the same as
    // session_elapsed_s (engine-ready-to-shutdown wall-clock) -- that
    // figure silently includes any idle time between turns (REPL waiting
    // on the next prompt, a script pausing between requests), which isn't
    // processing time at all. Found via a real run whose aggregate
    // "Throughput" line implied a ~415s stall that turned out to be pure
    // idle time between one finished request and the next -- the true
    // per-request numbers were fine throughout. This is the correct
    // denominator for a "how fast is generation, only counting actual
    // work" throughput figure; session_elapsed_s remains useful too, but
    // strictly as "how long did the whole process run," not as a speed
    // metric on its own.
    std::atomic<double> t_active_processing{0};
    std::atomic<uint64_t> n_requests{0};

    // Whole-request prefill pass duration (all 78 layers, one call to
    // layers_forward_gguf covering attention + MoE routing + MoE expert FFN
    // + everything else for the full prompt) -- ROOT CAUSE of the
    // "sub-phases summed to only ~35-40% of t_attn" mystery documented on
    // t_attn_preamble above. generation_loop.hpp used to feed this entire
    // duration into t_attn itself (`PERF_END(prefill, t_attn)`, comment:
    // "counts as part of attention/compute time") -- which double-counted
    // attention's own real contribution (mla_attention_absorbed_gguf's own
    // per-layer PERF_SCOPE(t_attn) already captures that correctly, called
    // from within this same layers_forward_gguf pass) AND, far more
    // significantly, folded in everything else the prefill pass does that
    // has NOTHING to do with attention -- MoE routing, MoE expert FFN
    // (routinely the single largest cost bucket in this project), and any
    // cold-cache disk-bound expert loading triggered by the FIRST-ever
    // visit to every layer. On a cold-cache run (thin/absent
    // .moestream_usage history) that one prefill call can be dominated by
    // disk I/O across all 78 layers -- exactly the kind of large, real,
    // non-attention cost that was silently inflating "Attention (MLA)" in
    // dump_report() and made a genuine, still-partially-open GPU-stall
    // investigation earlier this session harder to pin down than it needed
    // to be. Given its own standalone counter (not folded into t_attn,
    // and deliberately NOT part of the Snapshot/operator-/snapshot()
    // triple below -- this is a once-per-request total, not a per-token
    // delta, same reasoning as t_active_processing above).
    std::atomic<double> t_prefill_total{0};

    // Hint-push-to-pickup delay: how long a prefetch hint sits in the
    // queue before the worker thread actually starts processing it.
    // Added to distinguish two very different explanations for low PIPE
    // coverage (only 10.1% of misses served from staging in a real
    // measured run): (a) the worker thread isn't getting scheduled
    // promptly, i.e. CPU-starved by the OMP-parallelized dequant work
    // saturating all cores -- fixed by nothing thread-count-related, needs
    // CPU headroom instead; vs (b) the worker IS scheduled fine but simply
    // can't drain hints as fast as they're produced -- a genuine
    // concurrency/throughput ceiling, where more worker threads (or faster
    // per-read completion, e.g. O_DIRECT) would actually help. A large
    // t_prefetch_pickup_delay with a small n_prefetch_backlog_peak would
    // suggest (a); a large backlog growing over time suggests (b).
    std::atomic<double> t_prefetch_pickup_delay{0};
    std::atomic<uint64_t> n_prefetch_hints_processed{0};
    std::atomic<uint64_t> n_prefetch_backlog_peak{0};  // max (head - tail) ever observed at pickup time -- a gauge/high-water-mark, not cumulative

    // Plain-value point-in-time copy of the cumulative counters above, so
    // callers can diff two snapshots to get a per-token or per-request
    // breakdown without disturbing the global running totals.
    struct Snapshot {
        double t_disk_read=0, t_host_alloc=0, t_gpu_upload=0, t_gpu_download=0, t_gpu_compute=0, t_cpu_expert=0;
        double t_cpu_dequant=0, t_cpu_dot=0;
        double t_attn=0;
        double t_attn_preamble=0;
        double t_attn_norm=0, t_attn_qproj=0, t_attn_kvproj=0, t_attn_absorb=0, t_attn_score=0, t_attn_outproj=0;
        double t_dense_gpu_up=0, t_dense_gpu_compute=0, t_dense_gpu_down=0;
        double t_moe_route=0, t_moe_expert=0, t_prefetch=0, t_dense_load=0, t_consume_staged=0;
        double t_staging_regrow=0, t_staging_mutex_wait=0, t_cache_mutex_wait=0;
        uint64_t n_expert_cache_hit=0, n_expert_cache_miss=0, n_expert_promote=0, n_tokens_generated=0;
        uint64_t n_expert_gpu_calls=0, n_expert_cpu_calls=0, n_dense_gpu_calls=0;
        int64_t bytes_disk_read=0, bytes_gpu_uploaded=0;
        uint64_t n_pipe_hits=0, n_pinned_eviction_skips=0, n_pinned_resident=0, n_staging_regrow=0;

        Snapshot operator-(const Snapshot& b) const {
            return Snapshot{
                t_disk_read - b.t_disk_read, t_host_alloc - b.t_host_alloc,
                t_gpu_upload - b.t_gpu_upload, t_gpu_download - b.t_gpu_download,
                t_gpu_compute - b.t_gpu_compute, t_cpu_expert - b.t_cpu_expert,
                t_cpu_dequant - b.t_cpu_dequant, t_cpu_dot - b.t_cpu_dot,
                t_attn - b.t_attn,
                t_attn_preamble - b.t_attn_preamble,
                t_attn_norm - b.t_attn_norm, t_attn_qproj - b.t_attn_qproj,
                t_attn_kvproj - b.t_attn_kvproj, t_attn_absorb - b.t_attn_absorb,
                t_attn_score - b.t_attn_score, t_attn_outproj - b.t_attn_outproj,
                t_dense_gpu_up - b.t_dense_gpu_up, t_dense_gpu_compute - b.t_dense_gpu_compute,
                t_dense_gpu_down - b.t_dense_gpu_down,
                t_moe_route - b.t_moe_route,
                t_moe_expert - b.t_moe_expert, t_prefetch - b.t_prefetch,
                t_dense_load - b.t_dense_load, t_consume_staged - b.t_consume_staged,
                t_staging_regrow - b.t_staging_regrow,
                t_staging_mutex_wait - b.t_staging_mutex_wait,
                t_cache_mutex_wait - b.t_cache_mutex_wait,
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
                    n_pinned_resident - b.n_pinned_resident,
                    n_staging_regrow - b.n_staging_regrow
            };
        }
    };

    Snapshot snapshot() const {
        return Snapshot{
            t_disk_read.load(), t_host_alloc.load(), t_gpu_upload.load(), t_gpu_download.load(),
            t_gpu_compute.load(), t_cpu_expert.load(),
            t_cpu_dequant.load(), t_cpu_dot.load(),
            t_attn.load(),
            t_attn_preamble.load(),
            t_attn_norm.load(), t_attn_qproj.load(), t_attn_kvproj.load(),
            t_attn_absorb.load(), t_attn_score.load(), t_attn_outproj.load(),
            t_dense_gpu_up.load(), t_dense_gpu_compute.load(), t_dense_gpu_down.load(),
            t_moe_route.load(), t_moe_expert.load(), t_prefetch.load(), t_dense_load.load(), t_consume_staged.load(),
            t_staging_regrow.load(), t_staging_mutex_wait.load(), t_cache_mutex_wait.load(),
            n_expert_cache_hit.load(), n_expert_cache_miss.load(), n_expert_promote.load(), n_tokens_generated.load(),
            n_expert_gpu_calls.load(), n_expert_cpu_calls.load(), n_dense_gpu_calls.load(),
            bytes_disk_read.load(), bytes_gpu_uploaded.load(),
            n_pipe_hits.load(), n_pinned_eviction_skips.load(), n_pinned_resident.load(),
            n_staging_regrow.load()
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

    // CAS-loop max update for atomic<uint64_t> -- tracks a high-water-mark
    // (e.g. n_prefetch_backlog_peak) rather than a running total. Only
    // writes if v is actually greater than the current value.
    static void atomic_max(std::atomic<uint64_t>& target, uint64_t v) {
        uint64_t old = target.load(std::memory_order_relaxed);
        while (v > old && !target.compare_exchange_weak(old, v,
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
            "[%s] wall=%8.2fms  attn=%7.2fms [preamble=%6.2f norm=%6.2f qproj=%6.2f kvproj=%6.2f absorb=%6.2f "
            "score=%6.2f outproj=%6.2f] route=%6.2fms expert=%7.2fms "
            "gpu_up=%6.2fms gpu_dn=%6.2fms gpu_cmp=%6.2fms cpu_exp=%7.2fms (dequant=%6.2fms dot=%6.2fms) "
            "dense_gpu[up=%6.2f cmp=%6.2f dn=%6.2f calls=%llu] "
            "disk=%6.2fms  cache=%lluH/%lluM (%.0f%%) gpu_calls=%llu cpu_calls=%llu "
            "pipe=%llu pin_skip=%llu pin=%llu\n",
            label, wall_ms,
            d.t_attn * 1000.0,
            d.t_attn_preamble * 1000.0,
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
            "[%s] %d tokens in %.2fs (%.2f tok/s)  attn=%.2fs [preamble=%.2f norm=%.2f qproj=%.2f kvproj=%.2f "
            "absorb=%.2f score=%.2f outproj=%.2f] route=%.2fs "
            "expert=%.2fs (gpu=%.2fs [up=%.2fs cmp=%.2fs dn=%.2fs] cpu=%.2fs [dequant=%.2fs dot=%.2fs]) "
            "dense_gpu=%.2fs [up=%.2fs cmp=%.2fs dn=%.2fs calls=%llu] "
            "disk=%.2fs  cache=%.0f%% hit  expert_calls=%lluGPU/%lluCPU "
            "pipe=%llu(pin_skip=%llu pin=%llu)\n",
            label, n_tokens, wall_ms / 1000.0, tok_s,
            d.t_attn,
            d.t_attn_preamble,
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

    // session_elapsed_s: REAL wall-clock seconds from "engine ready to
    // accept prompts" to shutdown (see main.cpp's call site) -- used ONLY
    // for the final Throughput line. Deliberately NOT derived from summing
    // individual perf counters (t_disk_read + t_gpu_upload + ... etc.):
    // that was this function's ORIGINAL approach, and it quietly became
    // wrong as concurrency was added to this codebase (4-thread parallel
    // expert fetch, GPU/CPU Pass A/B overlap, PIPE async prefetch all
    // running concurrently with compute) -- summing durations that
    // genuinely overlap in real time systematically OVER-counts elapsed
    // time (confirmed on a real run: the old summed "total" came out to
    // 244.8s for a request whose ACTUAL wall-clock duration, per the
    // `[REQUEST]` line's own wall_ms-based calculation, was 118.08s -- more
    // than 2x inflated), which silently made every "Throughput: X tok/s"
    // line this session under-report real speed by a growing margin as
    // more overlap was added. 0 is a valid default (e.g. a test harness
    // that never calls generate_text) -- Tokens/Throughput lines simply
    // don't print in that case, same as the old total>0 guard did.
    void dump_report(double session_elapsed_s = 0.0) const {
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
        LOG_INFO("Attention (MLA):     %8.3f s  (preamble=%.3f norm=%.3f qproj=%.3f kvproj=%.3f absorb=%.3f score=%.3f outproj=%.3f)\n",
                 t_attn.load(), t_attn_preamble.load(), t_attn_norm.load(), t_attn_qproj.load(), t_attn_kvproj.load(),
                 t_attn_absorb.load(), t_attn_score.load(), t_attn_outproj.load());
        LOG_INFO("Prefill (whole pass):%8.3f s  (ALL layers' attention+MoE routing+MoE FFN combined, one call per "
                 "request -- was previously double-counted into Attention (MLA) above; see t_prefill_total's own "
                 "comment. If Attention (MLA) still doesn't reconcile against its own sub-phases after subtracting "
                 "prefill's own attention share, that's a genuinely separate gap worth investigating.)\n",
                 t_prefill_total.load());
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
        LOG_INFO("Prefetch (PIPE):     %lu misses served from staging / %lu total misses (%.1f%% served) "
                 "-- of the unserved: %lu no matching staged slot (wrong/stale prediction) / "
                 "%lu matched but pread not finished yet (pure timing loss)\n",
                 (unsigned long)n_pipe_hits.load(), (unsigned long)misses,
                 misses > 0 ? 100.0 * n_pipe_hits.load() / misses : 0.0,
                 (unsigned long)n_pipe_miss_no_match.load(), (unsigned long)n_pipe_miss_not_ready.load());
        LOG_INFO("Pilot hints skipped: %lu (already host-cache-resident -- avoided wasting prefetch bandwidth)\n",
                 (unsigned long)n_pilot_hints_skipped.load());
        LOG_INFO("VRAM->RAM demote:    %lu salvaged / %lu skipped (benign) / %lu failed "
                 "(vram_evict_demote_to_ram: instead of dropping an evicted expert's bytes, D2H "
                 "copy them back into the RAM tier before the VRAM slot is overwritten)\n",
                 (unsigned long)n_vram_evict_demoted.load(),
                 (unsigned long)n_vram_evict_demote_skipped.load(),
                 (unsigned long)n_vram_evict_demote_failed.load());
        LOG_INFO("Prefetch pickup:     avg %.2fms over %lu hints, peak queue backlog %lu "
                 "(large delay + small backlog => worker CPU-starved; growing backlog => worker throughput-bound)\n",
                 n_prefetch_hints_processed > 0
                     ? 1000.0 * t_prefetch_pickup_delay.load() / n_prefetch_hints_processed.load() : 0.0,
                 (unsigned long)n_prefetch_hints_processed.load(),
                 (unsigned long)n_prefetch_backlog_peak.load());
        LOG_INFO("Pinned residents:    %lu current / %lu desync-detected "
                 "(evict_lru no longer scans pinned slots at all -- this should read 0; "
                 "nonzero means unpinned_slot_indices desynced from the real pinned flags)\n",
                 (unsigned long)n_pinned_resident.load(),
                 (unsigned long)n_pinned_eviction_skips.load());
        LOG_INFO("Temp-pin skips:      %lu (decode-time cold-GPU-dispatch slots protected mid-compute)\n",
                 (unsigned long)n_temp_pin_eviction_skips.load());
        LOG_INFO("Upload-pending skips: %lu genuinely-in-flight (confirmed via re-poll) / "
                 "%lu stale-flag reclaimed (a speculative promotion's copy had actually "
                 "finished but was never re-confirmed -- see gpu_topology.hpp's evict_lru fix)\n",
                 (unsigned long)n_upload_pending_eviction_skips.load(),
                 (unsigned long)n_stale_upload_ready_reclaimed.load());
        LOG_INFO("Stale index hits:    %lu (find_expert's O(1) expert_index disagreed with "
                 "expert_vram_cache's real contents -- should always be 0; nonzero means the "
                 "index/array desynced somewhere, see gpu_topology.hpp)\n",
                 (unsigned long)n_stale_expert_index_hits.load());
        LOG_INFO("Staging mutex wait:  %8.3f s  (staging slot regrow: %lu events, %.3f s -- "
                 "cudaMallocHost/cudaFreeHost held under the SAME lock, see [SLOW-STAGING-REGROW])\n",
                 t_staging_mutex_wait.load(), (unsigned long)n_staging_regrow.load(), t_staging_regrow.load());
        LOG_INFO("Cache mutex wait:    %8.3f s  (GpuTopology::cache_mutex, shared between the decode "
                 "thread's hit-triggered promotions and the prefetcher worker thread)\n",
                 t_cache_mutex_wait.load());
        LOG_INFO("Tokens generated:    %lu\n", (unsigned long)n_tokens_generated.load());
        double active_s = t_active_processing.load();
        if (n_tokens_generated > 0 && active_s > 0) {
            LOG_INFO("Throughput:          %.2f tok/s  (%.2fs active processing across %lu request(s) -- "
                     "sum of each [REQUEST] line's own wall time, EXCLUDING any idle time between "
                     "prompts/turns -- this is the real generation-speed number, mixes prefill+decode; "
                     "see the per-request [REQUEST] line for a single turn's own rate, or per-token "
                     "[TOKEN N] DEBUG lines excluding token 0 for a decode-only rate)\n",
                     n_tokens_generated.load() / active_s, active_s, (unsigned long)n_requests.load());
        }
        if (n_tokens_generated > 0 && session_elapsed_s > 0) {
            LOG_INFO("Session wall-clock:  %.2f tok/s  (%.2fs engine-ready to shutdown -- INCLUDES any "
                     "idle time between turns, e.g. a REPL waiting on the next prompt; use the "
                     "Throughput line above for real generation speed. A large gap between these two "
                     "numbers means the process sat idle, not that generation was slow.)\n",
                     n_tokens_generated.load() / session_elapsed_s, session_elapsed_s);
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

// Call once at shutdown (mirrors report_cache_churn()'s placement/style).
// Largest-total-first so the dominant contributor is obvious at a glance.
inline void report_named_timers() {
    auto& stats = named_timer_stats();
    if (stats.empty()) return;
    std::vector<std::pair<std::string, NamedTimerStats>> sorted(stats.begin(), stats.end());
    std::sort(sorted.begin(), sorted.end(),
              [](const auto& a, const auto& b) { return a.second.total_ms > b.second.total_ms; });
    LOG_INFO("[NAMED-TIMING] Per-tensor-name dense_proj cost (diagnostic -- "
             "added to evaluate BF16 fast-path candidates):\n");
    for (auto& [name, s] : sorted) {
        LOG_INFO("  %-20s total=%.2f ms  calls=%llu  avg=%.4f ms/call\n",
                 name.c_str(), s.total_ms, (unsigned long long)s.calls,
                 s.calls ? s.total_ms / (double)s.calls : 0.0);
    }
}
