#pragma once
// Counter slice 5 of 8 split mechanically from perf_timer.hpp. Slices form a
// linear inheritance chain so constants declared in an earlier slice stay visible
// in later ones; `PerfCounters` derives from the last slice.
#include <array>
#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>
#include "perf/counters_04_pilot.hpp"

struct PerfCountersPart05 : PerfCountersPart04 {
    std::atomic<uint64_t> n_prefill_lookahead_loaded_evicted_unused{0};

    // EngineConfig::pilot_widen_all_tokens -- three different comparisons
    // against the SAME widened (union-of-all-tokens) prediction, see
    // LayerScratch::pilot_wide_pred_ids's own comment.
    // "recall" = accuracy vs the last token's own real picks, apples-to-
    // apples with DEPTH-1's own n_pilot_recall_hits/checks*topk convention
    // -- overlap_last / (checks*topk), i.e. n_pilot_wide_recall_hits_last /
    // n_pilot_wide_recall_topk_sum. NOTE (2026-08-30): the shipped
    // report line used to divide by n_pilot_wide_pred_size_sum instead --
    // that's PRECISION on the padded/widened set (how much of the wide
    // set's own size was a real hit), not recall against the last token's
    // fixed topk-sized target -- a real mislabeling caught after reporting
    // an apples-to-oranges "11%" figure that looked like a regression vs
    // DEPTH-1's 81.8% when the correctly-computed recall was actually
    // ~92%, higher than DEPTH-1 (expected: the widened set is a strict
    // superset of what DEPTH-1 alone would predict, since the last token is
    // one of the S tokens folded into the union). Fixed by adding a
    // dedicated topk_sum denominator instead of reusing pred_size_sum.
    // "coverage" = how much of the WHOLE layer's real unique-expert demand
    // the widened prediction's own set actually overlaps -- the number
    // that answers "if DEPTH-1 sampled more tokens, would it actually
    // help" (measured real, 91.2-91.3% on a 16-token prompt -- see
    // CLAUDE.md). "precision" = overlap_last / pred_size_sum, kept as a
    // secondary, correctly-labeled stat (how much of the wide set itself
    // was useful for the one token it's compared against) -- this is the
    // number the old code mislabeled as "recall".
    std::atomic<uint64_t> n_pilot_wide_recall_hits_last{0};
    std::atomic<uint64_t> n_pilot_wide_recall_checks{0};
    std::atomic<uint64_t> n_pilot_wide_recall_topk_sum{0}; // sum of the last token's own real topk size, across checks -- the correct recall denominator
    std::atomic<uint64_t> n_pilot_wide_pred_size_sum{0};   // sum of the widened prediction's own set size, across checks -- for "avg predicted-set size" and the precision denominator
    std::atomic<uint64_t> n_pilot_wide_coverage_hits{0};   // sum, across checks, of how many of the widened prediction's own picks were ALSO in the real whole-layer unique set
    std::atomic<uint64_t> n_pilot_wide_real_unique_sum{0}; // sum, across checks, of the real whole-layer unique-expert count (the denominator for coverage %)

    // EngineConfig::prefetch_widen_all_tokens (2026-08-30) -- real disk-
    // prefetch hint push wired from the widened prediction above, gated
    // separately from pilot_widen_all_tokens (which only computes/measures
    // the prediction; this flag decides whether not-yet-resident members
    // of it get pushed as real hints). Deliberately hint-push ONLY, no
    // VRAM promotion -- see the call site's own comment for why.
    std::atomic<uint64_t> n_pilot_wide_hints_pushed{0};
    std::atomic<uint64_t> n_pilot_wide_hints_skipped_resident{0};

    // MEASUREMENT ONLY, prefill-only (S>1) -- real token-sharing rate for
    // MoE expert dispatch, gathered to answer "would batching tokens that
    // route to the same expert into one GEMM per (layer,expert) actually
    // help" before committing to that (much larger, much riskier)
    // restructuring of moe_forward_gguf's Phase 3 per-token dispatch loop.
    // llama.cpp/ggml's ggml_mul_mat_id groups every (token,k) pair by
    // expert id and issues ONE batched GEMM per expert -- ZeroMoE's own
    // Phase 3 loop instead calls dispatch_expert_k once per (token,k), even
    // though scratch.expert_freq (Phase 1, already computed for the WHOLE
    // layer's S-token batch) already has everything needed to answer this
    // without any new computation: expert_freq[e] IS the real bucket size
    // (how many (s,k) picks this layer would land in one batched call to
    // expert e), and sum(expert_freq) == S*topk is today's real per-layer
    // dispatch count. Zero effect on dispatch/caching -- read-only tally of
    // an array that's already fully populated before Phase 3 ever runs.
    std::atomic<uint64_t> n_moe_bucket_unique_experts{0};   // sum, across every sampled prefill layer, of count(expert_freq[e]>0)
    std::atomic<uint64_t> n_moe_bucket_total_dispatches{0}; // sum, across every sampled prefill layer, of S*topk (today's real per-token dispatch count)
    std::atomic<uint64_t> n_moe_bucket_layers_sampled{0};   // how many (layer, prefill-call) instances contributed to the two sums above

    // Per-RANK breakdown of DEPTH-2(a)'s accuracy -- LayerScratch::
    // pilot_pred2_ids[k] is already ranked by the pilot's own gate score
    // (k=0 is the single highest-confidence pick, built by a repeated
    // argmax -- see execution_layer.cpp's pilot loop), but
    // n_pilot2_recall_hits/n_pilot2_recall_hint_hits above only sum across
    // the whole top-K, so they can't say whether accuracy concentrates in
    // the high-confidence ranks or is spread flat across all 8 -- the real
    // question behind "could we skip fetching the low-confidence tail and
    // only prefetch the top M ranks." Indexed by rank k (0..7).
    // *_checks_by_rank[k] is the per-rank denominator (how many times rank
    // k was even populated -- pilot_k2 can be < 8 for some layers via
    // prefetch_pilot_k_per_layer, so this isn't always a flat divide by
    // n_pilot2_recall_checks). *_hint_*_by_rank[k] mirrors this restricted
    // to the fetch-worthy subset, same "only correctness that could change
    // real I/O matters" reasoning as the aggregate hint counters above,
    // just broken out per rank instead of summed into one total.
    std::array<std::atomic<uint64_t>, 8> n_pilot2_recall_hits_by_rank{};
    std::array<std::atomic<uint64_t>, 8> n_pilot2_recall_checks_by_rank{};
    std::array<std::atomic<uint64_t>, 8> n_pilot2_recall_hint_hits_by_rank{};
    std::array<std::atomic<uint64_t>, 8> n_pilot2_recall_hint_checks_by_rank{};

    // Real prefetch-worker fetch stats (only nonzero once
    // prefetch_worker_threads > 0 and a worker actually starts a read --
    // separate from n_pilot_pred_needs_fetch above, which counts hints
    // PUSHED to the queue regardless of whether any worker ever picks
    // them up). n_prefetch_hits (existing counter, below) is reused as-is for
    // "consumed" -- it already counts exactly "staged reads later matched
    // and used by a real miss". t_prefetch_read_ms_sum is a SUM across
    // possibly-concurrent worker threads -- valid only for computing an
    // average ms/fetch, NOT a wall-clock GB/s figure (see this project's
    // own documented OMP-thread-sum-vs-wall-clock lesson for t_cpu_dequant
    // and the original t_disk_read uring-timing investigation).
    // 2026-09-02: LockFreePrefetcher::push_hint silently returns when its
    // ring is full -- a dropped hint is a GUARANTEED future cache miss, and
    // until now it was completely uninstrumented. This is the direct signal
    // for "hints are produced faster than the workers can drain them", the
    // one real pathology the [SLOW-PICKUP] pickup-delay warning cannot see
    // once that warning became queue-depth-aware (a genuine throughput
    // ceiling makes the drain time long, which is exactly what a depth-aware
    // threshold then treats as expected). Nonzero here means real work was
    // thrown away; it should be 0 on a healthy run.
    std::atomic<uint64_t> n_prefetch_hints_dropped_queue_full{0};
    std::atomic<uint64_t> n_prefetch_issued{0};
    std::atomic<uint64_t> n_prefetch_completed{0};
    std::atomic<uint64_t> n_prefetch_failed{0};
    std::atomic<uint64_t> bytes_prefetched{0};
    std::atomic<double>   t_prefetch_read_ms_sum{0};

    // Concurrency-bucketed fetch latency: at the moment EACH real prefetch
    // fetch's submit_and_wait is about to fire, how many OTHER prefetch-
    // worker fetches were simultaneously in flight (sampled from
    // LockFreePrefetcher::n_fetches_in_flight right before the read,
    // excluding this fetch's own increment)? Bucketed 0/1/2/3/4/5+.
    // Exists to directly test, under REAL production conditions (real GPU
    // dispatch + attention compute + real concurrent workers, not a
    // synthetic proxy), whether genuine multi-submitter contention is what
    // separates an isolated single-fetch benchmark's ~5ms median
    // (gguf_tool dma, real request shape) from this project's own real
    // "avg ms/fetch" figure -- see CLAUDE.md's disk-throughput
    // investigation for the isolated-benchmark side of this comparison. If
    // bucket 0's own avg matches the isolated benchmark while higher
    // buckets are progressively worse, that CONFIRMS real concurrent
    // contention as the cause; if even bucket 0 is elevated, the cause is
    // something else entirely (not concurrency). Deliberately does NOT
    // capture decode-thread Phase B activity (a separate, batched call
    // site in ram_manager.hpp) -- RAMSafetyManager has no pointer back to
    // LockFreePrefetcher by design (see CLAUDE.md's "Unified RAM-tier
    // claim/publish design" entry on why that reverse pointer was
    // removed), so this only sees prefetch-worker-vs-prefetch-worker
    // contention, not decode-thread-vs-prefetch-worker.
    static constexpr int PREFETCH_CONCURRENCY_BUCKETS = 6;  // 0,1,2,3,4,5+
    std::array<std::atomic<uint64_t>, PREFETCH_CONCURRENCY_BUCKETS> n_prefetch_fetch_by_concurrency{};
    std::array<std::atomic<double>,   PREFETCH_CONCURRENCY_BUCKETS> t_prefetch_fetch_ms_by_concurrency{};

    // DEPTH-2-only breakdown of the five counters directly above --
    // subsets, not separate totals (n_prefetch_issued already includes
    // n_prefetch_issued_depth2's count, etc.). Exists to make depth-2's
    // (EngineConfig::prefetch_pilot_depth=2) marginal real disk-I/O cost a
    // MEASURED number: real bytes read, real time spent, real fetches
    // issued/completed/failed, all attributable specifically to l+2 pilot
    // hints (async_prefetcher.hpp's PilotSource::Depth2) rather than an
    // estimate extrapolated from the recall-only counters above (which
    // count predictions, not actual disk activity). "Depth-1's own share"
    // is recoverable as (aggregate - this) but isn't tracked as its own
    // explicit counter -- see PilotSource's own comment for why: the
    // aggregate also includes Phase 2 prefill-warm hints, which aren't
    // depth-1 pilot hints either, so a literal "_depth1" counter would
    // invite exactly that misreading.
    std::atomic<uint64_t> n_prefetch_issued_depth2{0};
    std::atomic<uint64_t> n_prefetch_completed_depth2{0};
    std::atomic<uint64_t> n_prefetch_failed_depth2{0};
    std::atomic<uint64_t> bytes_prefetched_depth2{0};
    std::atomic<double>   t_prefetch_read_ms_sum_depth2{0};
    // A depth-2 hint's worker claim found the expert already resident or
    // already being fetched by someone else -- a hint that cost queue
    // space and worker attention but triggered no real read either way
    // (async_prefetcher.hpp's active_prefetch, claim_or_wait's Hit/Loading
    // branches). Distinct from n_pilot2_pred_already_* (perf_timer.hpp,
    // execution_layer.cpp) -- those measure residency AT PREDICTION TIME
    // (before a hint is even pushed, already excluded via is_hint_worthy);
    // this measures residency AT CLAIM TIME (moments later, when a worker
    // actually got to it) -- a real, if usually small, race window this
    // number quantifies directly instead of assuming away.
    std::atomic<uint64_t> n_prefetch_claims_beaten_depth2{0};

    std::atomic<uint64_t> n_vram_evict_demoted{0};  // VRAM evictions salvaged into the RAM tier via a D2H copy instead of being dropped outright (vram_evict_demote_to_ram=true) -- nonzero proves the salvage path is engaging; compare against Expert cache's disk-miss count for the real win (a demoted expert re-hit later serves from RAM, no disk read at all)
    std::atomic<uint64_t> n_vram_evict_demote_skipped{0}; // demote attempts that declined for a benign reason (already RAM-resident, or the layer uses the legacy per-expert-tensor format with no packed_slab_geometry) -- not an error, just nothing to do
    std::atomic<uint64_t> n_vram_evict_demote_failed{0};  // demote attempts that hit a real problem (cudaMemcpy D2H failure, size mismatch) -- should stay at/near 0; a nonzero count is worth investigating but never corrupts anything, since a failed demote just leaves the RAM slot empty rather than half-populated
    std::atomic<uint64_t> n_vram_demote_stream_unavailable{0};  // demote_from_vram fell back to the
                                             // old plain (stream-0) cudaMemcpy because vram_cache or
                                             // the target device's copy_stream wasn't available --
                                             // should read 0 in practice (this function is only ever
                                             // reached via a GpuTopology callback, so vram_cache is
                                             // necessarily set); nonzero means the stream-0 tax this
                                             // was meant to avoid is silently back for those calls.
    // Diagnostic pair added to directly test why the stream-0 fix (see
    // t_bounce_buffer_wait's sibling entry in CLAUDE.md) showed no
    // measurable real-hardware win for demote_from_vram despite a
    // confirmed-real mechanism in an isolated microbenchmark -- see the
    // cudaStreamQuery call site in ram_manager.hpp for the full reasoning.
    // If *_idle dominates *_busy, that's direct evidence there was rarely
    // any genuinely concurrent other-stream work to avoid waiting for at
    // this specific call site.
    std::atomic<uint64_t> n_demote_saw_compute_stream_busy{0};
    std::atomic<uint64_t> n_demote_saw_compute_stream_idle{0};
    std::atomic<uint64_t> n_demote_saw_copy_stream_busy{0};
    std::atomic<uint64_t> n_demote_saw_copy_stream_idle{0};

    // Same diagnostic, same reasoning, different call site:
    // GpuTopology::upload_expert_active (gpu_topology.cpp). That call
    // site's own async conversion (cudaMemcpyAsync+dedicated copy_stream)
    // was reverted earlier this session after showing ZERO measured
    // benefit on a decode-dominated full run -- but a real per-request
    // [PREFILL] breakdown later showed prefill's OWN active_up cost runs
    // ~3x the full-run per-call average (2026-08-12), and prefill is
    // exactly when the async prefetcher's worker threads are doing real
    // concurrent io_uring disk reads -- genuinely different conditions
    // from whatever the original demote_from_vram-derived 0%-busy finding
    // measured. This re-checks stream occupancy AT THIS call site,
    // instead of assuming that other finding transfers. If busy% is real
    // here (unlike demote_from_vram's near-0%), that's the evidence
    // needed to justify re-trying async for THIS call site specifically.
    std::atomic<uint64_t> n_active_upload_saw_compute_stream_busy{0};
    std::atomic<uint64_t> n_active_upload_saw_compute_stream_idle{0};
    std::atomic<uint64_t> n_active_upload_saw_copy_stream_busy{0};
    std::atomic<uint64_t> n_active_upload_saw_copy_stream_idle{0};
    // 2026-08-28: the pre-existing check just above only queries compute_
    // stream/copy_stream -- it never checks active_layer_stream or
    // speculative_prefetch_stream(2), the actual streams decode_async_
    // prewarm's and the speculative pilot fill's own async DMAs queue onto.
    // This matters structurally, not just as more coverage: this call site
    // (upload_expert_active's SYNCHRONOUS fallback) issues a plain
    // cudaMemcpy with NO stream argument -- CUDA's legacy default (NULL)
    // stream has real, documented implicit-synchronization semantics: it
    // will not begin until ALL prior work already queued on every OTHER
    // cudaStreamCreate'd (blocking-type, which every stream in this
    // codebase is -- grepped, none use cudaStreamNonBlocking) stream on
    // that device has drained. If active_layer_stream/speculative_
    // prefetch_stream(2) have a real backlog of queued (lower-urgency,
    // predicted) transfers at the moment this genuinely urgent, real-
    // dispatch fallback fires, this legacy-stream semantics would force it
    // to wait behind ALL of them -- a real, structural priority-inversion
    // risk this project's own prior "0% cross-stream contention" findings
    // never actually tested for, since none of them checked these specific
    // streams from this specific call site. Non-blocking; cudaSuccess =
    // idle, cudaErrorNotReady = real queued work on that stream right now.
    std::atomic<uint64_t> n_active_upload_saw_active_layer_stream_busy{0};
    std::atomic<uint64_t> n_active_upload_saw_active_layer_stream_idle{0};
    std::atomic<uint64_t> n_active_upload_saw_speculative_stream_busy{0};
    std::atomic<uint64_t> n_active_upload_saw_speculative_stream_idle{0};
    // 2026-08-28: direct answer to "do we double-queue the same MoE
    // expert's transfer" -- fires when upload_expert_active's fallback
    // (reached specifically when find_active_expert's poll-before-trust
    // check found this exact identity already resident but its async DMA
    // still in flight, treated as a miss) is about to pick a victim slot
    // and issue a FRESH synchronous copy for a (layer,expert_id) that is
    // STILL PRESENT (by identity, regardless of upload_ready) somewhere
    // else in active_layer_slots -- i.e. the earlier async issue (decode_
    // async_prewarm or speculative fill) is still genuinely in flight, and
    // this call is about to move the exact same bytes a SECOND time,
    // wasting real DDR->VRAM bandwidth (see the bandwidth-achieved report
    // line) and potentially evicting a different, useful active-layer
    // resident to make room for the duplicate. Instrumentation only for
    // now -- does not change behavior.
    std::atomic<uint64_t> n_active_layer_sync_duplicate_transfer{0};
    std::atomic<int64_t> bytes_active_layer_sync_duplicate_transfer{0};
    // 2026-08-28: the FIX for the duplicate-transfer/priority-inversion bug
    // above -- upload_expert_active now checks for an already-in-flight (or
    // already-landed) identity FIRST, before any victim-selection. `_ready`
    // counts the cheap case (landed in the narrow window between find_
    // active_expert's poll and this call -- no wait needed at all); the
    // plain counter counts the real wait-on-specific-event case (bounded,
    // targeted cudaEventSynchronize on the ALREADY-in-flight transfer's own
    // event, instead of starting a redundant second copy). Both should
    // track close to the OLD n_active_layer_sync_duplicate_transfer figure
    // (494 in the run that found this bug) -- that counter is left in place
    // post-fix specifically to prove it drops to ~0 once this engages.
    std::atomic<uint64_t> n_active_layer_wait_on_inflight{0};
    std::atomic<uint64_t> n_active_layer_wait_on_inflight_ready{0};
    std::atomic<double> t_active_layer_wait_on_inflight{0};

    // Same diagnostic again, THIRD call site: GpuTopology::upload_expert_
    // active_async (gpu_topology.cpp), the batch-pre-warm path -- not
    // upload_expert_active's own single-call sync path the two counter
    // blocks above already cover. Added specifically because a real
    // [REQUEST] line (2026-08-18) showed active_layer_prewarm_issue (this
    // call site's own PERF_START/PERF_END scope) as the single largest
    // bucket in the whole request (~15.8s/59.5s, 26.6%) on a run where
    // 'Three-tier caching' showed ZERO calls through the sync path above --
    // every active-layer transfer that run went through THIS async batch
    // path instead, so the sync path's own (0%-population, silent-by-guard)
    // diagnostic was uninformative for it. The source comment at this call
    // site already root-caused the cost as genuinely necessary H2D
    // bandwidth (two prior fix attempts here both just moved the cost
    // between issue=/sync= without reducing it) -- this re-confirms that
    // conclusion with a direct measurement rather than trusting it applies
    // unchanged: a real busy%% here would mean there IS other decode-thread
    // GPU work queued on the same stream this DMA could be waiting behind
    // (a scheduling fix, not a bandwidth one); near-0%% would confirm the
    // existing conclusion still holds at THIS call site's own real volume.
    std::atomic<uint64_t> n_active_prewarm_saw_compute_stream_busy{0};
    std::atomic<uint64_t> n_active_prewarm_saw_compute_stream_idle{0};
    std::atomic<uint64_t> n_active_prewarm_saw_copy_stream_busy{0};
    std::atomic<uint64_t> n_active_prewarm_saw_copy_stream_idle{0};

    // Follow-up diagnostic: after decomposing demote_from_vram's cost into
    // pure-DMA-bandwidth + cudaDeviceScheduleBlockingSync's real wake-up
    // tax, a real residual (~23% of the total, per a real-hardware A/B)
    // was still unexplained. These test the two leading remaining
    // candidates directly: (1) multi-GPU PCIe asymmetry (this box has a
    // PCIe x16 3090 and a PCIe x8 5060 Ti -- a demote landing on the
    // slower link would genuinely cost more, not "overhead") and (2)
    // whether the specifically-EVICTED population's average slab size
    // differs from the corpus-wide ~8.26MB figure used to benchmark. Fixed
    // 8-slot array indexed by real CUDA device ordinal (same convention as
    // devices[gpu_id] elsewhere in this codebase) -- generous headroom
    // over this box's real 2-GPU config.
    static constexpr int PERF_MAX_GPU_SLOTS = 8;
    std::atomic<double> t_vram_demote_by_gpu[PERF_MAX_GPU_SLOTS]{};
    std::atomic<int64_t> bytes_vram_demote_by_gpu[PERF_MAX_GPU_SLOTS]{};
    std::atomic<uint64_t> n_vram_demote_by_gpu[PERF_MAX_GPU_SLOTS]{};
    // 2026-08-28: same idea, H2D direction, UNIFIED across all four upload
    // call sites (upload_expert, upload_expert_async, upload_expert_active,
    // upload_expert_active_async) -- direct answer to "does all DDR->VRAM
    // traffic go to one GPU". The active-layer pool is structurally 100%
    // GPU0 by construction (allocate_active_layer_area targets vault_gpus[0]
    // if any vault GPU exists, gpu_topology.cpp) -- this counter confirms
    // that empirically rather than asserting it, and (more usefully) also
    // covers the main VRAM cache paths (upload_expert/upload_expert_async),
    // which genuinely CAN target either GPU since moe_cache_gb is sized
    // per-GPU.
    std::atomic<int64_t> bytes_h2d_upload_by_gpu[PERF_MAX_GPU_SLOTS]{};
    std::atomic<uint64_t> n_h2d_upload_by_gpu[PERF_MAX_GPU_SLOTS]{};
    // REMOVED: n_prefetch_miss_no_match/not_ready/still_queued and
    // t_prefetch_not_ready_elapsed_ms_sum/t_prefetch_still_queued_elapsed_ms_sum
    // used to live here -- a consume_staged failure-reason split for the
    // old separate staging pool, which no longer exists (see the "Unified
    // RAM-tier claim/publish state machine" comment further down).

    // REMOVED: n_staging_stale_reclaimed/n_staging_cross_token_stale_
    // reclaimed/n_staging_fallback_events used to live here, measuring
    // acquire_staging_slot's own staleness-signal reclaim logic for the old
    // separate staging pool. Both the pool and that reclaim logic no longer
    // exist -- a prefetch worker now claims a real RAM-tier slot directly
    // via RAMSafetyManager::claim_or_wait, subject to the SAME LRU eviction
    // scoring every other RAM-tier claim uses, not a separate round-robin
    // policy needing its own staleness heuristics.

    // Added when finish_prefetch's unconditional, unconfirmed VRAM
    // promotion (async_prefetcher.hpp) was removed -- that call used to
    // fire on EVERY successful staged read regardless of whether the
    // expert was ever actually used, evicting real VRAM residents on pure
    // speculation. VRAM promotion for a prefetch-loaded expert (MoEExpertBlock::
    // loaded_by_prefetch, engine.hpp) now only ever happens via the normal
    // hit-count-gated path (promote_hit_threshold) once a REAL subsequent
    // access confirms it. This counter is the direct evidence that path
    // still works for prefetch-origin experts post-fix -- nonzero proves
    // prefetched experts do still reach VRAM, just on confirmed reuse
    // rather than blind speculation. Compare against n_prefetch_hits: a
    // healthy run should show a meaningful fraction of prefetch hits
    // eventually promoted, not zero.
    std::atomic<uint64_t> n_prefetch_staged_later_promoted{0};


    // Three-tier caching performance counters
    std::atomic<uint64_t> n_active_layer_uploads{0};    // uploads to active layer area (temporary workspace)
    // Subset of n_active_layer_uploads issued through the opt-in demand
    // async path (upload_expert_active_async(demand=true)). Kept separate
    // so a run log can prove the new path actually engaged; does NOT
    // include prefill/speculative/worker prewarm traffic.
    std::atomic<uint64_t> n_active_layer_uploads_demand_async{0};
    std::atomic<uint64_t> n_active_layer_hits{0};      // cache hits served from active layer area
    // 2026-08-26: a find_active_expert lookup matched a slot by identity
    // but its async DMA (upload_expert_active_async) hadn't landed yet --
    // treated as a miss (poll, don't block), see find_active_expert's own
    // comment. Nonzero is expected/healthy (proves the poll path is
    // engaging, not a bug); a HIGH rate relative to n_active_layer_hits
    // would mean speculative fills are routinely still in flight when
    // real dispatch reaches them -- i.e. not enough real slack time
    // between issue and use for this specific candidate.
    std::atomic<uint64_t> n_active_layer_poll_not_ready{0};
    // 2026-08-28: same event, split by WHICH mechanism issued the slot --
    // added specifically to test a real hypothesis about why decode_async_
    // prewarm_nonblocking's first real run showed a throughput regression
    // (0.72 vs the ~0.85-0.87 baseline) despite active_layer_prewarm_issue
    // itself looking fine. Unlike the speculative pilot fill (a full layer
    // of real lead time before its prediction is used), decode_async_
    // prewarm issues a candidate for the SAME token it's about to dispatch,
    // moments later -- near-zero natural lead time. If THIS split shows a
    // much higher not-ready rate than the speculative one, that's the
    // mechanism: removing the blocking sync doesn't give these transfers
    // anywhere near enough time to land before use, so most of them abort
    // (poll miss) and fall through to a FRESH synchronous fetch anyway --
    // paying for the abandoned async DMA's copy-engine time AND the new
    // synchronous one, strictly worse than the old guaranteed-ready design.
    std::atomic<uint64_t> n_active_layer_poll_not_ready_decode_prewarm{0};
    std::atomic<uint64_t> n_active_layer_poll_not_ready_speculative{0};
    // 2026-08-26: cleanup_active_layer_area's own opportunistic poll-and-
    // release of a stale async-issue hold nothing ever consulted via
    // find_active_expert before its layer window closed -- prevents a
    // permanent temp_pin_count leak now that the speculative path no
    // longer goes through a guaranteed batch sync. See that function's
    // own comment.
    std::atomic<uint64_t> n_active_layer_stale_hold_released{0};
};
