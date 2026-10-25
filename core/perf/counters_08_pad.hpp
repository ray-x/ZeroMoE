#pragma once
// Counter slice 8 of 8 split mechanically from perf_timer.hpp. Slices form a
// linear inheritance chain so constants declared in an earlier slice stay visible
// in later ones; `PerfCounters` derives from the last slice.
#include <array>
#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>
#include "perf/counters_07_prefill.hpp"

struct PerfCountersPart08 : PerfCountersPart07 {
    std::atomic<uint64_t> n_pad_no_candidates_after_filter{0};
    // Bumped the moment the OUTER gate (disk_batch_padding && all_packed &&
    // !reqs.empty() && !in_prefill && n_pending<target) passes, before any
    // candidate-selection logic runs -- if THIS stays 0 across a whole run
    // despite disk_batch_padding=true, the bug is in the outer gate itself
    // (or this Phase B branch never runs at all), not in candidate
    // selection -- separates that failure class from the two counters
    // above, which only fire once the outer gate has already passed.
    std::atomic<uint64_t> n_pad_gate_entered{0};
    // Independent, non-short-circuited visibility into EACH term of the
    // padding gate (disk_batch_padding && all_packed && !reqs.empty() &&
    // !in_prefill && n_pending<target) -- added after n_pad_gate_entered
    // itself stayed at 0 across a whole real run despite the ENCLOSING
    // use_decode_uring/Ring::available() branch confirmed running 3400/
    // 3400 times successfully (Decode-thread io_uring batches: 100%
    // batched). Every other explanation (reference-vs-copy on ram_mgr,
    // config.load()/validate() ordering relative to run_engine(), a
    // duplicate/shadowed member) was directly traced through the real
    // call chain and ruled out -- these five counters exist to show which
    // SPECIFIC term is false, since the combined && expression alone gives
    // no way to tell without them.
    std::atomic<uint64_t> n_pad_dbg_calls{0};            // total times checked (denominator)
    std::atomic<uint64_t> n_pad_dbg_padding_false{0};
    std::atomic<uint64_t> n_pad_dbg_not_all_packed{0};
    std::atomic<uint64_t> n_pad_dbg_reqs_empty{0};
    std::atomic<uint64_t> n_pad_dbg_in_prefill{0};
    std::atomic<uint64_t> n_pad_dbg_target_not_reached{0};  // n_pending >= disk_batch_padding_target

    // Bytes read by padding's speculative requests specifically -- kept
    // SEPARATE from bytes_disk_read (the real-miss total) so bandwidth
    // attribution between "real demand" and "speculation" stays visible,
    // rather than silently inflating the existing real-traffic number.
    // 2026-08-18: this separation, while correct on the BYTES side, was
    // pairing with a t_disk_read that already counts padding's own wall-
    // clock (same io_uring submission, one shared batch_ms) -- an
    // asymmetric real-bytes-over-real+padding-time division that silently
    // DEFLATES the reported real-only GB/s instead. See dump_report's own
    // "Disk read (expert)" line for the fix (a second, combined-bytes
    // throughput figure using this same counter) and its comment for the
    // real magnitude this was hiding.
    std::atomic<int64_t> bytes_disk_read_speculative{0};

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
    // .zeromoe_usage history) that one prefill call can be dominated by
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
    // Added to distinguish two very different explanations for low prefetch
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

    // Two-lane priority queue (LockFreePrefetcher, async_prefetcher.hpp):
    // HIGH lane = l+1 pilot hints + prefill Phase-2 warm hints (PilotSource
    // != Depth2 at push_hint time); LOW lane = l+2 pilot hints
    // (PilotSource::Depth2). worker_loop always drains HIGH to empty before
    // ever looking at LOW -- strict priority, not weighted-fair -- on the
    // reasoning that a l+2 hint is both lower-confidence AND further from
    // being needed than any l+1/prefill hint, so it should never make a
    // higher-value hint wait. n_prefetch_hi_lane_processed/lo_lane_processed
    // split n_prefetch_hints_processed above by which lane a hint actually
    // came from -- the direct, measured confirmation that priority ordering
    // is really happening (rather than assumed from the code shape alone),
    // mirroring this project's own "measure, don't assume" house style.
    // n_prefetch_lo_lane_starved_ticks counts every worker_loop iteration
    // that found the LOW lane non-empty but skipped it because HIGH was
    // non-empty that same tick -- a real, accepted starvation risk of
    // strict (not round-robin) priority; if this climbs alongside a real
    // drop in l+2 fetch-worthy accuracy or completion rate, that's the
    // signal a fairness knob (e.g. service LOW once every K HIGH pops)
    // would actually be worth adding, rather than guessing it's needed
    // up front.
    std::atomic<uint64_t> n_prefetch_hi_lane_processed{0};
    std::atomic<uint64_t> n_prefetch_lo_lane_processed{0};
    std::atomic<uint64_t> n_prefetch_lo_lane_starved_ticks{0};
};
