#pragma once
// Counter slice 4 of 8 split mechanically from perf_timer.hpp. Slices form a
// linear inheritance chain so constants declared in an earlier slice stay visible
// in later ones; `PerfCounters` derives from the last slice.
#include <array>
#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>
#include "perf/counters_03_qwen.hpp"

struct PerfCountersPart04 : PerfCountersPart03 {

    // Prediction-quality benchmarking (always measured, independent of
    // prefetch_worker_threads -- the pilot loop runs unconditionally
    // today). Classifies each pilot top-K pick, at prediction time, into
    // exactly one of four buckets: already VRAM-pinned, already
    // VRAM-LRU/reactive, already RAM-resident, or genuinely "needs a
    // fetch" (the only bucket that ever calls push_hint). The raw
    // n_pilot_recall_hits/checks/topk_sum above measure accuracy over the
    // FULL top-K including already-cached picks, which is the wrong
    // denominator for "is prefetch worth enabling" -- an already-cached
    // pick is correctly skipped regardless of whether it was predicted
    // right, so its correctness doesn't matter to that question. These
    // two counters instead measure accuracy restricted to the
    // needs_fetch subset only -- the number that actually determines
    // whether enabling prefetch can help a given model.
    std::atomic<uint64_t> n_pilot_pred_already_ram{0};
    std::atomic<uint64_t> n_pilot_pred_already_vram_pinned{0};
    std::atomic<uint64_t> n_pilot_pred_already_vram_lru{0};
    std::atomic<uint64_t> n_pilot_pred_needs_fetch{0};
    // ATTN-PILOT's own real-hint wiring (2026-08-25) -- exact mirror of the
    // four above, plus a real push count (n_pilot_pred above has no direct
    // equivalent since DEPTH-1's own hint-push is unconditional whenever
    // is_hint_worthy; ATTN-PILOT's is additionally gated by EngineConfig::
    // prefetch_attn_pilot/prefetch_attn_pilot_k, so it needs one).
    std::atomic<uint64_t> n_pilot_attn_pred_already_ram{0};
    std::atomic<uint64_t> n_pilot_attn_pred_already_vram_pinned{0};
    std::atomic<uint64_t> n_pilot_attn_pred_already_vram_lru{0};
    std::atomic<uint64_t> n_pilot_attn_pred_needs_fetch{0};
    std::atomic<uint64_t> n_pilot_attn_hints_pushed{0};

    // Measurement for the "evict what the pilot just confirmed is safe"
    // race (dense_dispatch.hpp's was_pilot_confirmed_resident). Nonzero
    // here is direct evidence eviction_score()'s pure recency+frequency
    // scoring is undoing the pilot's own already-cached determination --
    // the expert was correctly judged safe moments ago, then evicted by
    // ordinary LRU pressure before its real dispatch ever consumed it,
    // forcing exactly the synchronous re-read the pilot's skip-fetch
    // decision assumed wouldn't be needed. Split by tier since VRAM
    // eviction with vram_evict_demote_to_ram=on is cheaper (salvaged into
    // RAM, not necessarily a disk re-read) than a genuine RAM-tier
    // eviction (which always means a real disk read next time).
    std::atomic<uint64_t> n_evicted_was_pilot_confirmed_vram{0};
    std::atomic<uint64_t> n_evicted_was_pilot_confirmed_ram{0};

    // ACTIVE protection counterpart to the measurement-only pair above
    // (2026-08-17): victim-selection scans in evict_lru (gpu_topology.hpp)
    // and both RAM-tier eviction sites (ram_manager.hpp) now exclude a
    // was_pilot_confirmed_resident() candidate from consideration the same
    // way an already-pinned/temp-pinned/in-flight slot is excluded, instead
    // of only counting the race after the fact. Safe by construction, not
    // just by convention: pilot_confirmed_set.hpp's own set holds at most
    // PILOT_CONFIRMED_CAP (16) entries, all for the SAME single most-recent
    // layer transition, wholesale-replaced every transition -- so even in
    // the worst case this excludes at most ~16 of a layer's ~90-100+ slots
    // (or a negligible fraction of the whole 1000+ slot VRAM pool), never
    // enough to starve a victim scan on its own. These counters confirm
    // the exclusion is actually engaging (nonzero => real saves), and their
    // sum should track close to (not exceed) what n_evicted_was_pilot_
    // confirmed_* used to report before this fix, now redirected into
    // whatever the SECOND-best candidate was instead.
    std::atomic<uint64_t> n_pilot_protected_eviction_skips_vram{0};
    std::atomic<uint64_t> n_pilot_protected_eviction_skips_ram{0};

    // Measurement for a DIFFERENT question than the pilot-confirm-evict
    // pair above: when a real dispatch resolves an expert via an
    // INDEPENDENT cache hit -- RAM-tier LRU (ram_manager.hpp's get_expert/
    // get_experts_batch Step 1/Phase A) or the VRAM shadow-fast-path
    // (execution_layer.cpp) -- BYPASSING consume_staged entirely, had that
    // exact (layer,expert) ALSO been recently touched by the async
    // prefetcher (expert_fetch_tracker.hpp)? Split by which cache tier won
    // the race AND by what state the prefetch was caught in:
    //   *_had_staged_fetch  -- the CLEANEST signal: the fetch had fully
    //     SUCCEEDED and was still sitting ready, unconsumed, in staging --
    //     pure redundancy, nothing to do with pool size or timing, an
    //     entirely independent cache hit just resolved the dispatch first.
    //   *_had_inflight_fetch -- a worker was still actively reading this
    //     expert's bytes when the independent hit resolved the dispatch;
    //     the fetch (once it finishes) will be wasted the same way.
    //   *_had_evicted_fetch -- the fetch had ALREADY been evicted from
    //     staging by the time this hit happened -- both mechanisms
    //     (pool churn AND cache-path redundancy) compounded on this one.
    // This is the direct answer to "of the ~98% of correct predictions
    // that go unused, how much is explained by an independent cache hit
    // beating the prefetcher to the dispatch" -- see CLAUDE.md/session
    // history for the investigation this closes.
    std::atomic<uint64_t> n_vram_hit_had_inflight_fetch{0};
    std::atomic<uint64_t> n_vram_hit_had_staged_fetch{0};
    std::atomic<uint64_t> n_vram_hit_had_evicted_fetch{0};
    std::atomic<uint64_t> n_ram_hit_had_inflight_fetch{0};
    std::atomic<uint64_t> n_ram_hit_had_staged_fetch{0};
    std::atomic<uint64_t> n_ram_hit_had_evicted_fetch{0};

    // REMOVED: n_ram_load_had_prefetch_*/n_phase_gap_staged_*/
    // t_phase_gap_ms_sum used to live here, fed by expert_fetch_tracker.
    // hpp's trace_ram_load_if_recently_fetched/trace_phase_gap_if_staged_
    // after_check -- both measured races against LockFreePrefetcher's old
    // separate staging pool, which no longer exists (prefetch and the
    // decode thread's own miss-handling now claim/populate/publish through
    // the SAME locked RAM-tier slot table, RAMSafetyManager::claim_or_wait,
    // ram_manager.hpp). A "recheck right before the real read" fix aimed at
    // the ordering gap these counters diagnosed was tried at both former
    // Phase-B call sites and reverted from both (near-zero benefit plus a
    // real counter-corruption regression) before the unification made the
    // whole mechanism moot. See "Unified RAM claim"/"Unified RAM wait" in
    // this file's shutdown report for the replacement instrumentation.
    // Elapsed time (microseconds) sums for the *_had_staged_fetch buckets
    // specifically -- how long the fetch had been sitting ready before an
    // independent hit beat it to the dispatch. (in-flight/evicted buckets
    // don't need this to answer the core question, kept lean.)
    std::atomic<double> t_vram_hit_staged_fetch_elapsed_us_sum{0};
    std::atomic<double> t_ram_hit_staged_fetch_elapsed_us_sum{0};

    // Hypothesis test (expert_fetch_tracker.hpp's current_token_step): of
    // the n_{vram,ram}_hit_had_{staged,evicted}_fetch events above, how
    // many happened during the SAME token's own forward pass as the fetch
    // that produced them (step_delta==0, consistent with a genuine
    // same-transition race) vs a LATER, unrelated token's hit
    // rediscovering an EARLIER token's now-stale, never-consumed fetch
    // record (step_delta>0)? The *_same_step counters are subsets of the
    // corresponding n_*_hit_had_*_fetch totals above; the
    // *_step_delta_sum values (divided by the relevant total count) give
    // the average number of token-steps back a match was found, across
    // ALL matches (not just the same-step subset) -- large values here
    // directly confirm the "stale record" hypothesis, near-zero values
    // would instead confirm a same-transition race.
    std::atomic<uint64_t> n_vram_staged_fetch_same_step{0};
    std::atomic<uint64_t> n_vram_evicted_fetch_same_step{0};
    std::atomic<uint64_t> n_ram_staged_fetch_same_step{0};
    std::atomic<uint64_t> n_ram_evicted_fetch_same_step{0};
    std::atomic<double> t_vram_staged_fetch_step_delta_sum{0};
    std::atomic<double> t_ram_staged_fetch_step_delta_sum{0};
    std::atomic<double> t_vram_evicted_fetch_step_delta_sum{0};
    std::atomic<double> t_ram_evicted_fetch_step_delta_sum{0};

    std::atomic<uint64_t> n_pilot_recall_hint_hits{0};        // of the needs_fetch subset, how many matched the real router's later choice
    std::atomic<uint64_t> n_pilot_recall_hint_checks_denom{0}; // denominator for the above (sum of needs_fetch picks actually checked against a real routing decision)

    // Per-RANK breakdown for l+1 (depth-1), exact mirror of
    // n_pilot2_recall_hits_by_rank/etc. below -- added specifically because
    // every "Pilot recall by rank" line printed until now was DEPTH-2 only;
    // l+1's own rank-0/1/2/... accuracy curve (the number that actually
    // decides how many top ranks are safe to promote RAM->VRAM
    // speculatively, since l+1 is the layer close enough to matter for
    // that decision) had never been measured. Same "does accuracy
    // concentrate in the high-confidence ranks" question as the DEPTH-2
    // version, one hop closer where the pilot's own prediction is
    // structurally more accurate (real post-FFN hidden_states, not a
    // naive-reuse approximation).
    std::array<std::atomic<uint64_t>, 8> n_pilot_recall_hits_by_rank{};
    std::array<std::atomic<uint64_t>, 8> n_pilot_recall_checks_by_rank{};
    // ATTN-PILOT by-rank (2026-08-25) -- exact mirror of the pair above,
    // see n_pilot_attn_recall_hits's own comment for the full design.
    std::array<std::atomic<uint64_t>, 8> n_pilot_attn_recall_hits_by_rank{};
    std::array<std::atomic<uint64_t>, 8> n_pilot_attn_recall_checks_by_rank{};
    std::array<std::atomic<uint64_t>, 8> n_pilot_attn_recall_hint_hits_by_rank{};
    std::array<std::atomic<uint64_t>, 8> n_pilot_attn_recall_hint_checks_by_rank{};
    std::array<std::atomic<uint64_t>, 8> n_pilot_recall_hint_hits_by_rank{};
    std::array<std::atomic<uint64_t>, 8> n_pilot_recall_hint_checks_by_rank{};

    // Per-RANK accuracy restricted to the "already_ram" bucket specifically
    // (RAM-resident, NOT yet VRAM-resident at prediction time) -- the ONE
    // population a RAM->VRAM promotion decision actually needs, and
    // genuinely distinct from both numbers above: raw recall (mixes all
    // four residency buckets together) and fetch-worthy accuracy (measures
    // ONLY the needs_fetch bucket, i.e. the disk-prefetch path's own
    // decision-relevant number, which says nothing about already-RAM
    // picks). See LayerScratch::pilot_pred_was_ram_only's own comment
    // (execution_layer.cpp) for the classification this reads.
    std::array<std::atomic<uint64_t>, 8> n_pilot_ram_only_hits_by_rank{};
    std::array<std::atomic<uint64_t>, 8> n_pilot_ram_only_checks_by_rank{};

    // Real per-layer disk-fetch-miss data, gathered specifically to inform a
    // CPU-offload cap design (Fiddler-style CPU/GPU split for SSD-cold-miss
    // experts, arxiv 2402.07033) BEFORE building that feature -- measures
    // how many of a layer's topk-routed experts are genuinely disk-cold (not
    // RAM- or VRAM-resident, i.e. would need a real SSD fetch) per decode
    // token, since that population size directly bounds how useful any
    // CPU-offload cap choice (1-8) can be. Decode-only (in_prefill==false):
    // prefill's batched routing isn't the population this feature would act
    // on. Incremented in RAMSafetyManager::get_experts_batch's own Phase A,
    // once per call (n_decode_calls_by_layer, the denominator) and once per
    // real Claimed miss within that call (n_decode_real_fetch_by_layer) --
    // fixed to 128 slots, generous headroom over every model in this
    // project's own config set (GLM-5.2/DeepSeek-V4 both ~78-80 layers).
    static constexpr int kMaxTrackedLayersForMiss = 128;
    std::array<std::atomic<uint32_t>, kMaxTrackedLayersForMiss> n_decode_calls_by_layer{};
    std::array<std::atomic<uint32_t>, kMaxTrackedLayersForMiss> n_decode_real_fetch_by_layer{};

    // Qwen prediction timeline.  The *_ns arrays are one-shot handoff slots
    // between the decode thread's next-layer pilot and that target layer;
    // they are diagnostics only and are consumed atomically when the target
    // reaches the corresponding boundary.
    std::array<std::atomic<int64_t>, kMaxTrackedLayersForMiss>
        qwen_pilot_prediction_ready_ns{};
    std::array<std::atomic<int64_t>, kMaxTrackedLayersForMiss>
        qwen_pilot_prediction_for_fetch_ns{};
    std::array<std::atomic<int64_t>, kMaxTrackedLayersForMiss>
        qwen_pilot_router_ready_ns{};
    std::atomic<double> t_qwen_pilot_queue_to_ready{0};
    std::atomic<double> t_qwen_pilot_ready_to_router{0};
    std::atomic<double> t_qwen_pilot_router_to_fetch{0};
    std::atomic<double> t_qwen_pilot_ready_to_fetch{0};
    std::atomic<uint64_t> n_qwen_pilot_queue_to_ready{0};
    std::atomic<uint64_t> n_qwen_pilot_ready_to_router{0};
    std::atomic<uint64_t> n_qwen_pilot_router_to_fetch{0};
    std::atomic<uint64_t> n_qwen_pilot_ready_to_fetch{0};
    // Active-layer copies tagged as Qwen prediction fills.  The event time
    // is the actual GPU copy-engine interval, not request-total time.
    std::atomic<double> t_qwen_pilot_active_ready_to_issue{0};
    std::atomic<double> t_qwen_pilot_active_dma{0};
    std::atomic<uint64_t> n_qwen_pilot_active_ready_to_issue{0};
    std::atomic<uint64_t> n_qwen_pilot_active_dma{0};
    std::atomic<int64_t> bytes_qwen_pilot_active_dma{0};

    // Per-layer split of n_ram_prefetch_evicted_untouched (declared
    // earlier in this struct; placed HERE only because it needs
    // kMaxTrackedLayersForMiss, declared just above). This is
    // what makes the prefill-window measurement DECISIVE rather than
    // suggestive: without P3's trim the extra-slot budget only stretches to
    // the first handful of layers, so the SESSION-WIDE waste number can only
    // move a few percent even if the mechanism works perfectly. Reading it
    // per layer instead turns one partially-covered run into a real
    // controlled experiment -- windowed layers should collapse toward ~0
    // untouched-waste while un-windowed layers stay at the ~82% baseline,
    // within the SAME run, same prompt, same hardware, no cross-run noise.
    std::array<std::atomic<uint64_t>, kMaxTrackedLayersForMiss> n_ram_prefetch_evicted_untouched_by_layer{};
    std::array<std::atomic<uint64_t>, kMaxTrackedLayersForMiss> n_ram_prefetch_loaded_by_layer{};


    // Distribution/max companion to the mean-only pair above -- the mean
    // alone can't distinguish a CPU-offload cap of 1 from a cap of 2 (does a
    // layer/token often see 2+ SIMULTANEOUS real-fetch misses in one call,
    // or almost never past 1?). n_decode_fetch_count_histogram[N] = how many
    // decode-time get_experts_batch calls (any layer, pooled) saw EXACTLY N
    // real-fetch misses (N=0..7, index 8 = "8 or more"/cap-full).
    // n_decode_max_fetch_by_layer[layer] = the single highest real-fetch
    // count ever seen in one call for that layer -- flags an outlier layer
    // (e.g. layer 77's 0.8/token mean, ~3x the overall average) a single
    // global cap might serve poorly.
    static constexpr int kFetchHistogramBuckets = 9;  // 0,1,2,3,4,5,6,7,8+
    std::array<std::atomic<uint32_t>, kFetchHistogramBuckets> n_decode_fetch_count_histogram{};
    std::array<std::atomic<uint32_t>, kMaxTrackedLayersForMiss> n_decode_max_fetch_by_layer{};

    // Qwen4Exp decode-only MoE layer wall time.  This is deliberately a
    // model-gated diagnostic rather than part of Snapshot(): it measures the
    // complete moe_forward_gguf call for S==1, including routing, cache
    // resolution, expert execution, and any CPU/GPU handoff.  The split is
    // what we need to choose a CPU-affinity set from evidence instead of
    // assuming that low router confidence alone predicts a faster path.
    std::array<std::atomic<double>, kMaxTrackedLayersForMiss>
        t_qwen_decode_layer{};
    std::array<std::atomic<double>, kMaxTrackedLayersForMiss>
        t_qwen_decode_cpu_layer{};
    std::array<std::atomic<double>, kMaxTrackedLayersForMiss>
        t_qwen_decode_gpu_layer{};
    std::array<std::atomic<uint64_t>, kMaxTrackedLayersForMiss>
        n_qwen_decode_layer{};

    // EngineConfig::cpu_offload_max_experts -- how many of the real (Claimed)
    // misses classified above were actually redirected to CPU compute this
    // run, vs. left to take the normal VRAM-upload path because the per-
    // call cap was already spent. See execution_layer.cpp's force_cpu_
    // offload_k / expert_ffn_resolve_gpu_loc's own early-return check.
    std::atomic<uint64_t> n_cpu_offload_forced{0};
    std::atomic<uint64_t> n_cpu_offload_left_uncapped{0};

    // Per-LAYER recall breakdown for all three pilots (2026-08-25) -- the
    // global n_pilot*_recall_hits/checks/topk_sum counters above answer
    // "how accurate is this pilot overall", but a real per-layer question
    // ("is layer 4 much more/less predictable than layer 40, so a config
    // that speculates more aggressively on the reliable layers and less on
    // the noisy ones might beat one flat global cutoff") needs a per-layer
    // breakdown none of those give. Reuses kMaxTrackedLayersForMiss (128,
    // declared above -- must come first in the class body since this uses
    // it as an array size) for the same generous-headroom reason that
    // constant already documents. hits_by_layer/topk_sum_by_layer[l] sum
    // exactly like their global counterparts (accuracy = hits/topk_sum),
    // just scoped to one target layer's own checks; checks_by_layer[l] is
    // the raw sample count for that layer (roughly topk_sum/topk, kept as
    // its own counter rather than derived so a reader doesn't have to
    // assume topk is constant). Incremented in execution_layer.cpp's three
    // recall-check blocks (DEPTH-1, ATTN-PILOT, DEPTH-2), right alongside
    // each one's existing global counters -- indexed by the TARGET layer
    // (layer_idx the prediction was FOR, l+1 for DEPTH-1/ATTN-PILOT, l+2
    // for DEPTH-2), not the layer the prediction was made FROM.
    std::array<std::atomic<uint32_t>, kMaxTrackedLayersForMiss> n_pilot_recall_hits_by_layer{};
    std::array<std::atomic<uint32_t>, kMaxTrackedLayersForMiss> n_pilot_recall_topk_sum_by_layer{};
    // TEMPORARY DIAGNOSTIC (2026-09-13): n_pilot_recall_hits_by_rank above
    // (rank-only, summed across ALL layers) and n_pilot_recall_hits_by_layer
    // above (layer-only, summed across ALL ranks) are both marginals --
    // neither answers "for layer L specifically, is rank 0 more reliable
    // than rank 3." Real question: this session's own per-layer routing
    // concentration table shows top-8 share ranging ~16%-36% across
    // layers, so the high-confidence-rank cutoff plausibly varies by
    // layer too. Genuine 2D cross table: [layer][rank].
    // kMaxTrackedLayersForMiss x 8 = 1024 atomics, trivial.
    std::array<std::array<std::atomic<uint32_t>, 8>, kMaxTrackedLayersForMiss>
        n_pilot_recall_hits_by_layer_rank{};
    std::array<std::array<std::atomic<uint32_t>, 8>, kMaxTrackedLayersForMiss>
        n_pilot_recall_checks_by_layer_rank{};
    std::array<std::atomic<uint32_t>, kMaxTrackedLayersForMiss> n_pilot_recall_checks_by_layer{};
    std::array<std::atomic<uint32_t>, kMaxTrackedLayersForMiss> n_pilot_attn_recall_hits_by_layer{};
    std::array<std::atomic<uint32_t>, kMaxTrackedLayersForMiss> n_pilot_attn_recall_topk_sum_by_layer{};
    std::array<std::atomic<uint32_t>, kMaxTrackedLayersForMiss> n_pilot_attn_recall_checks_by_layer{};
    std::array<std::atomic<uint32_t>, kMaxTrackedLayersForMiss> n_pilot2_recall_hits_by_layer{};
    std::array<std::atomic<uint32_t>, kMaxTrackedLayersForMiss> n_pilot2_recall_topk_sum_by_layer{};
    std::array<std::atomic<uint32_t>, kMaxTrackedLayersForMiss> n_pilot2_recall_checks_by_layer{};
    std::array<std::atomic<uint32_t>, kMaxTrackedLayersForMiss> n_qwen_pilot_recall_hits_by_layer{};
    std::array<std::atomic<uint32_t>, kMaxTrackedLayersForMiss> n_qwen_pilot_recall_pred_size_sum_by_layer{};
    std::array<std::atomic<uint32_t>, kMaxTrackedLayersForMiss> n_qwen_pilot_recall_checks_by_layer{};
    std::array<std::atomic<uint32_t>, kMaxTrackedLayersForMiss> n_qwen_pilot_prefetch_recall_hits_by_layer{};
    std::array<std::atomic<uint32_t>, kMaxTrackedLayersForMiss> n_qwen_pilot_prefetch_recall_pred_size_sum_by_layer{};
    std::array<std::atomic<uint32_t>, kMaxTrackedLayersForMiss> n_qwen_pilot_prefetch_recall_checks_by_layer{};
    std::array<std::array<std::atomic<uint32_t>, kQwenPilotMaxRanks>,
               kMaxTrackedLayersForMiss>
        n_qwen_pilot_rank_hits_by_layer{};
    std::array<std::array<std::atomic<uint32_t>, kQwenPilotMaxRanks>,
               kMaxTrackedLayersForMiss>
        n_qwen_pilot_rank_checks_by_layer{};

    // DEPTH-2 measurement (approach (a), "naive reuse" only) -- exact
    // mirror of the eight counters above, one hop further out.
    // execution_layer.cpp's pilot loop predicts l+2's routing by reusing
    // the SAME pre-(l+1)-attention hidden_states vector the l+1 pilot
    // already computes from -- i.e. missing l+1's own FFN contribution
    // entirely, the single term already proven fatal to omit at depth-1
    // (see that block's own comment on the 0%-recall failure the
    // post-FFN timing fix corrected). MEASUREMENT ONLY: never wired to
    // async_io_queue.push_hint or publish_pilot_confirmed_resident, so
    // these counters have zero effect on cache behavior or dispatch --
    // they exist purely to get a real per-token answer to "is depth-2
    // worth building at all" instead of the aggregate .zeromoe_usage
    // proxy check.
    std::atomic<uint64_t> n_pilot2_recall_hits{0};
    std::atomic<uint64_t> n_pilot2_recall_checks{0};
    std::atomic<uint64_t> n_pilot2_recall_topk_sum{0};
    std::atomic<uint64_t> n_pilot2_pred_already_ram{0};
    std::atomic<uint64_t> n_pilot2_pred_already_vram_pinned{0};
    std::atomic<uint64_t> n_pilot2_pred_already_vram_lru{0};
    std::atomic<uint64_t> n_pilot2_pred_needs_fetch{0};
    std::atomic<uint64_t> n_pilot2_recall_hint_hits{0};
    std::atomic<uint64_t> n_pilot2_recall_hint_checks_denom{0};

    // Real push_hint(l+2, ...) calls -- nonzero only once
    // EngineConfig::prefetch_pilot_depth=2 AND a pick both clears the
    // is_hint_worthy check (genuinely cold) and falls within
    // prefetch_pilot_k_depth2's rank cutoff (execution_layer.cpp's pilot
    // loop). Distinguishes "we WOULD have pushed a hint here" (the
    // measurement counters above, always live) from "we actually did"
    // (this one, only live when depth-2 is turned on for real).
    std::atomic<uint64_t> n_pilot2_hints_pushed{0};

    // EngineConfig::prefill_dispatch_lookahead_tokens -- real counts for the
    // additive, prefill-only widening of each token's own get_experts_batch
    // call to also opportunistically warm the RAM tier for upcoming tokens'
    // not-yet-resident picks. n_prefill_lookahead_added is real work done
    // (a genuine extra fetch that wouldn't otherwise have happened this
    // call); the three _skipped_ counters break down why a scanned
    // candidate was NOT added, so a report line that doesn't sum to the
    // total number of candidates scanned is itself a bug signal (see this
    // project's own standing "a report line that doesn't reconcile is a
    // bug, not an inconvenience" lesson).
    std::atomic<uint64_t> n_prefill_lookahead_added{0};
    std::atomic<uint64_t> n_prefill_lookahead_skipped_vram{0};   // already VRAM-resident
    std::atomic<uint64_t> n_prefill_lookahead_skipped_ram{0};    // already RAM-resident
    std::atomic<uint64_t> n_prefill_lookahead_skipped_dup{0};    // already in THIS call's own combined list

    // The "was it actually used" half of the lookahead measurement --
    // MoEExpertBlock::loaded_by_lookahead's own comment has the full design.
    // n_prefill_lookahead_first_hit (claim_or_wait's Hit branch): the future
    // token this fetch was FOR reached it and found it resident -- a real
    // "prefetch worked" event. n_prefill_lookahead_loaded_evicted_unused
    // (the two RAM-tier eviction-victim-selection sites, mirroring
    // n_ram_prefetch_loaded_evicted_unused's own design exactly): evicted
    // again before anyone ever claimed it a second time -- pure wasted disk
    // I/O. Neither is a full accounting of n_prefill_lookahead_added above
    // (a slot that's still resident, un-evicted, at shutdown falls into
    // neither bucket) -- read the sum of these two against n_prefill_
    // lookahead_added as a lower bound on the used/wasted split, not an
    // exact partition.
    std::atomic<uint64_t> n_prefill_lookahead_first_hit{0};
};
