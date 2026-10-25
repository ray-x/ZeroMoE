#pragma once
// Counter slice 3 of 8 split mechanically from perf_timer.hpp. Slices form a
// linear inheritance chain so constants declared in an earlier slice stay visible
// in later ones; `PerfCounters` derives from the last slice.
#include <array>
#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>
#include "perf/counters_02_gdn.hpp"

struct PerfCountersPart03 : PerfCountersPart02 {
    std::atomic<double> t_evict_lru{0};
    std::atomic<double> t_upload_expert_async{0};
    std::atomic<double> t_ram_get_expert{0};
    std::atomic<double> t_ram_get_experts_batch{0};

    // get_experts_batch's own internal Phase A/B/C breakdown -- previously
    // t_ram_get_experts_batch was a single PERF_SCOPE around the WHOLE
    // function (claim scan + disk I/O + wait-on-other-claim, all folded
    // into one number), and comparing it against the separately-scoped,
    // multi-source-fed t_disk_read global (see n_uring_batch_ok's own
    // comment on why that comparison is apples-to-oranges: t_disk_read is
    // fed by prefetch worker threads and prefill warm-up too, not just this
    // function's own Phase B) repeatedly looked like an unexplained gap
    // without ever localizing WHERE inside get_experts_batch itself the
    // time actually goes. These three are mutually exclusive and scoped
    // ONLY to this one function, so t_ram_batch_phase_a + _b + _c should sum
    // to very close to t_ram_get_experts_batch's own delta for the same
    // window -- a real, direct decomposition instead of an inferred one.
    std::atomic<double> t_ram_batch_phase_a{0};  // claim/scan loop (claim_or_wait per k, no I/O)
    std::atomic<double> t_ram_batch_phase_b{0};  // issuing + waiting on the real disk read (uring submit_and_wait or OMP fallback)
    std::atomic<double> t_ram_batch_phase_c{0};  // wait_for_ram_slot for WaitOnOther entries (an in-flight prefetch claim)

    // Added to answer a specific question raised while validating a
    // "switch expert uploads to cudaMemcpyAsync" suggestion: BOTH
    // upload_expert (sync) and upload_expert_async stage the expert slab
    // through a small fixed pinned bounce_pool via a plain CPU memcpy
    // BEFORE either cudaMemcpy/cudaMemcpyAsync is even issued -- required
    // because RAMSafetyManager's own slab_buffer is deliberately left
    // UNPINNED (see init_bounce_pool's comment: pinning a multi-GB
    // ever-growing LRU cache was rejected as too risky/expensive). That
    // CPU memcpy previously had ZERO timer of its own -- t_gpu_upload's
    // PERF_START/PERF_END wraps only the cudaMemcpy call, placed AFTER
    // this memcpy already ran, so its cost was invisible to every
    // existing report. This matters because it's identical regardless of
    // whether the DMA that follows is sync or async: if this counter
    // turns out to dominate whatever "host DDR -> VRAM" cost a profiler
    // attributed to the upload call as a whole, converting the DMA call
    // alone to cudaMemcpyAsync would not touch it at all. Same
    // measure-before-concluding discipline as t_expert_gpu_upload's split
    // from t_gpu_upload above.
    std::atomic<double> t_bounce_stage_memcpy{0};

    // Alt A of the cudaHostRegister investigation the comment above
    // describes -- upload_expert (the SYNC decode-hot-path only; see
    // gpu_topology.hpp's own comment on why upload_expert_async is
    // deliberately NOT converted the same way) now transiently
    // cudaHostRegister/cudaHostUnregister's the RAM tier's own slab_buffer
    // bytes directly for the duration of one DMA, instead of memcpy-ing
    // them into the small fixed bounce_pool first. Measured on real
    // hardware (RTX 3090, ~8.26MB representative slab, matching this
    // session's real average): 1.070ms/call combined register+DMA+
    // unregister vs the memcpy approach's 1.428ms/call -- a ~25%
    // reduction, with nothing pinned longer than this one synchronous
    // call. Deliberately its OWN counter, not folded into
    // t_bounce_stage_memcpy: that counter now exclusively reflects
    // upload_expert_async's still-unconverted memcpy path, and conflating
    // a register-cost measurement with a memcpy-cost measurement under one
    // shared name would recreate exactly the class of attribution bug
    // t_expert_gpu_upload was split out to fix. Includes BOTH the
    // cudaHostRegister AND cudaHostUnregister call combined (not split
    // further), to stay a single number directly comparable to
    // t_bounce_stage_memcpy's own precedent.
    std::atomic<double> t_host_register_stage{0};

    // Added to test a specific hypothesis raised while explaining an
    // unexpectedly LARGE real-hardware win from the Alt A change above:
    // acquire_bounce_buffer()'s cudaEventSynchronize(bb.in_flight) call
    // (gpu_topology.hpp) -- waiting for a PREVIOUS DMA using that bounce
    // slot to drain before returning it for reuse -- sat before any timer
    // in every version of this code, old and new alike, and with only
    // BOUNCE_POOL_SIZE=4 buffers shared between upload_expert (called
    // constantly on the decode-hot-path) and upload_expert_async
    // (background promotions), contention for that small pool could easily
    // produce large, fully-invisible stalls. A real A/B (same 80-token
    // prompt) showed expert='s own sum-of-sub-timers gap go from ~40.1s
    // unaccounted (67.15s expert= vs 27.05s summed) to ~1.0s (27.77s vs
    // 26.81s) the moment upload_expert stopped calling acquire_bounce_buffer
    // at all (Alt A converted it to cudaHostRegister instead) -- far larger
    // than Alt A's own isolated per-call savings could explain, and a
    // striking match for the "untimed" gap this codebase's history
    // documents chasing across MANY prior rounds (t_hold_for_compute/
    // t_release_hold/t_find_active_expert, then t_find_expert/t_evict_lru/
    // t_upload_expert_async/t_ram_get_expert/t_ram_get_experts_batch), each
    // of which came back near-zero without closing it -- none of those
    // rounds ever instrumented this exact call, because it's a cross-call
    // resource-contention wait (two DIFFERENT functions racing for the same
    // 4-slot pool), not a cost internal to any single function a
    // by-function timing sweep would think to check. This timer directly
    // tests that hypothesis: now only reachable from upload_expert_async
    // (upload_expert no longer contends for the pool at all), so if the
    // hypothesis is right this should read small/near-zero going forward.
    std::atomic<double> t_bounce_buffer_wait{0};

    // io_uring O_DIRECT staging elision (io_uring_backend.hpp's
    // submit_and_wait). Every direct read used to be routed through an
    // aligned scratch buffer and memcpy'd out, unconditionally -- correct,
    // but unnecessary whenever the request's offset, length and destination
    // are ALREADY alignment-clean, which is the case for essentially every
    // MoE expert read here by construction (per-expert GGUF offsets are
    // exact 4096 multiples, tensor sizes are too, and slab_buffer is
    // posix_memalign(4096)). n_uring_direct_unstaged should therefore
    // dominate n_uring_direct_staged on a healthy run;
    // bytes_uring_stage_memcpy_saved is the real CPU + memory-bandwidth
    // burn removed from contending with concurrent compute.
    std::atomic<uint64_t> n_uring_direct_unstaged{0};
    std::atomic<uint64_t> n_uring_direct_staged{0};
    std::atomic<int64_t> bytes_uring_stage_memcpy_saved{0};

    // Persistent RAM-slab pinning (EngineConfig::ram_pinned_budget_gb) --
    // see StreamingEngine::ensure_slab_capacity's own comment for the full
    // design and the real-hardware measurements that motivated it.
    // t_ram_slab_pin is a ONE-TIME-PER-ALLOCATION cost paid during preload/
    // first-touch, NOT a per-transfer cost like every other host-register
    // timer in this file -- do not compare it against those directly.
    // n_ram_slab_pin_over_budget counts slabs that stayed pageable purely
    // because the configured budget was already spent (expected and benign
    // whenever the budget is smaller than the full RAM tier -- it is the
    // direct measure of what fraction of the tier the budget actually
    // covers). n_ram_slab_unpinned_on_regrow should be ~0 after warmup:
    // slabs are grow-only and sized to the model's largest expert, so a
    // regrow (which forces unregister+free+realloc+re-register) should only
    // happen during the very first fills.
    std::atomic<double> t_ram_slab_pin{0};
    std::atomic<uint64_t> n_ram_slab_pinned{0};
    std::atomic<int64_t> bytes_ram_slab_pinned{0};
    std::atomic<uint64_t> n_ram_slab_pin_fail{0};
    std::atomic<uint64_t> n_ram_slab_pin_over_budget{0};
    std::atomic<uint64_t> n_ram_slab_unpinned_on_regrow{0};
    // 2026-08-29: RAMSafetyManager::eager_preregister_remaining_slots --
    // real nsys profiling (this session) found cudaHostRegister averaging
    // ~5x slower during a live prefill (5.15ms/call, 7 threads contending)
    // than during single-threaded startup preload (0.97ms/call) -- both
    // the wall-clock relocation AND the driver-contention tax are avoided
    // by eagerly registering empty, not-yet-assigned RAM-tier slots (sized
    // to their own layer's real expert size via compute_expert_slab_bytes)
    // for the REMAINDER of ram_pinned_budget_gb at startup, single-
    // threaded, before any request is timed -- not just the historically-
    // ranked experts pin_file/prefill_pin_file already cover. A real
    // future load into one of these slots then finds slab_capacity
    // already >= total and skips both the realloc and the register call
    // entirely (ensure_slab_capacity's own existing growth-only check).
    std::atomic<uint64_t> n_ram_eager_preregistered{0};
    std::atomic<int64_t> bytes_ram_eager_preregistered{0};
    std::atomic<double> t_ram_eager_preregister{0};
    // Per-call register sites finding the source ALREADY persistently
    // pinned. Expected and benign whenever ram_pinned_budget_gb > 0 -- these
    // calls then DMA straight from the page-locked source (the best case),
    // rather than either registering it again or falling back to bounce-
    // buffer staging. Tracked separately from the *_host_register_fail
    // counters precisely so a healthy persistent-pin run doesn't look like
    // a run full of registration failures.
    std::atomic<uint64_t> n_upload_expert_src_already_pinned{0};
    std::atomic<uint64_t> n_upload_async_src_already_pinned{0};

    // Counters
    std::atomic<uint64_t> n_expert_cache_hit{0};
    std::atomic<uint64_t> n_expert_cache_miss{0};
    std::atomic<uint64_t> n_expert_promote{0};
    std::atomic<uint64_t> n_tokens_generated{0};
    // Sum of prompt_ids.size() across every request this session -- the
    // missing piece for a real prefill-only tok/s figure (t_prefill_total
    // already exists but had no matching token count to divide by). See
    // dump_report()'s "Rates:" line.
    std::atomic<uint64_t> n_prompt_tokens{0};
    // MTP (NextN speculative decoding, mtp_speculative.cpp): one "cycle" is
    // one draft-then-verify round (K drafts proposed, some prefix accepted,
    // one correction token always emitted). Shutdown-report-only (not
    // threaded through Snapshot/[REQUEST] deltas, matching the precedent
    // set elsewhere for a structurally distinct, low-call-volume subsystem)
    // -- exists so a validation run can see MTP is actually doing useful
    // speculative work, not just running without crashing. n_mtp_tokens_
    // emitted double-counts against n_tokens_generated by design (both are
    // incremented from the same emit callback) -- it exists so acceptance
    // rate is directly computable (accepted / drafts_proposed) without
    // relying on n_tokens_generated, whose count includes non-MTP runs too.
    std::atomic<uint64_t> n_mtp_cycles{0};
    std::atomic<uint64_t> n_mtp_drafts_proposed{0};
    std::atomic<uint64_t> n_mtp_drafts_accepted{0};
    std::atomic<uint64_t> n_mtp_tokens_emitted{0};
    std::atomic<uint64_t> n_expert_gpu_calls{0};  // expert_ffn calls that took the GPU-resident path
    std::atomic<uint64_t> n_expert_cpu_calls{0};  // expert_ffn calls that took the CPU fallback path
    std::atomic<int64_t> bytes_disk_read{0};
    // Per-drive split of bytes_disk_read above -- added to answer "is the
    // dual-SSD striped read actually balanced in practice" with real
    // numbers (see mirror_fd_registry()/is_mirror_fd()/register_mirror_fd()
    // below, and io_uring_backend.hpp's submit_and_wait, the one real call
    // site that increments these). Scoped to the io_uring path only (this
    // project's own report shows ~100% of real reads go through it on this
    // hardware) -- the rare sequential-pread fallback isn't separately
    // attributed, so these two won't sum to EXACTLY bytes_disk_read on a
    // run that ever falls back.
    std::atomic<int64_t> bytes_disk_read_primary{0};
    std::atomic<int64_t> bytes_disk_read_mirror{0};
    std::atomic<int64_t> bytes_gpu_uploaded{0};
    // Bytes actually routed through t_bounce_stage_memcpy's CPU memcpy
    // (only when acquire_bounce_buffer() succeeds -- on bounce-pool
    // exhaustion dma_src falls back to the unpinned host_data directly
    // and this memcpy is skipped) -- may be slightly less than
    // bytes_gpu_uploaded for that reason. upload_expert_async only (see
    // t_bounce_stage_memcpy's own comment for why upload_expert no longer
    // contributes here).
    std::atomic<int64_t> bytes_bounce_staged{0};
    // Bytes actually routed through t_host_register_stage's
    // cudaHostRegister (upload_expert only) -- may be slightly less than
    // this function's own share of bytes_gpu_uploaded if a registration
    // ever fails and falls back to the direct pageable-source copy (see
    // n_host_register_fail).
    std::atomic<int64_t> bytes_host_registered{0};
    std::atomic<uint64_t> n_host_register_fail{0};  // cudaHostRegister failed (e.g.
                                             // cudaErrorHostMemoryAlreadyRegistered) --
                                             // should read ~0 on a healthy run; falls
                                             // back to the direct pageable-source copy
                                             // path on failure, same degrade behavior
                                             // the old bounce-pool-exhausted case had.

    // upload_expert_async's own cudaHostRegister conversion (2026-08-13) --
    // mirrors upload_expert's t_host_register_stage/n_host_register_fail/
    // bytes_host_registered above, kept as SEPARATE counters rather than
    // shared ones -- same "structurally distinct call site, don't hide
    // which path dominates" reasoning as upload_expert_active's own
    // separate counters (see t_active_layer_host_register's comment).
    // Unlike upload_expert (fully synchronous, register/DMA/unregister all
    // in one call), upload_expert_async's DMA is still in flight when the
    // function returns -- host_data can't be unregistered immediately
    // without risking a use-after-unregister race against the in-flight
    // copy. t_upload_async_host_register times ONLY the register call
    // itself (paid synchronously, before the async DMA is issued); the
    // matching cudaHostUnregister is deferred until slot->upload_event
    // confirms the DMA actually completed (see GpuTopology::
    // pending_host_unregisters/reap_pending_host_unregisters) and is timed
    // separately as t_upload_async_host_unregister so a slow reap doesn't
    // get misattributed to the register call that triggered it.
    std::atomic<double> t_upload_async_host_register{0};
    std::atomic<double> t_upload_async_host_unregister{0};
    std::atomic<int64_t> bytes_upload_async_host_registered{0};
    std::atomic<uint64_t> n_upload_async_host_register_fail{0};
    // Deferred unregisters actually reaped (event confirmed complete) vs.
    // still pending the next time reap ran -- a healthy run should show
    // this tracking closely behind n_expert_promote's own async-path
    // share; a persistently large pending count would mean DMAs are
    // completing much slower than new promotions are being issued, i.e.
    // registrations piling up faster than they can be safely freed.
    std::atomic<uint64_t> n_upload_async_deferred_unregister_reaped{0};
    // Phase A () counters:
    std::atomic<uint64_t> n_prefetch_hits{0};          // cache misses served from the prefetcher's staging pool (no pread) -- the higher this is vs n_expert_cache_miss, the more disk I/O was hidden behind compute
    std::atomic<uint64_t> n_pinned_eviction_skips{0}; // MEANING CHANGED once GpuTopology::unpinned_slot_indices landed: evict_lru no longer visits permanently-pinned slots at all (they're excluded from the scan at the source), so this now only increments on the DEFENSIVE fallback check -- should read 0 on a healthy run; nonzero means unpinned_slot_indices desynced from the real `pinned` flags somewhere (see gpu_topology.hpp's evict_lru). Previously (before that optimization) this counted every pinned slot skipped during a normal scan, and nonzero was the EXPECTED/healthy signal -- the polarity of "what a nonzero value means" flipped with that change.
    std::atomic<uint64_t> n_temp_pin_eviction_skips{0}; // times evict_lru skipped a slot TEMPORARILY held for an in-flight decode-time GPU dispatch (ExpertLocation::temp_pin_count > 0) -- distinct from n_pinned_eviction_skips (permanent pins); nonzero proves the compute-in-flight protection is actually engaging
    std::atomic<uint64_t> n_upload_pending_eviction_skips{0}; // times evict_lru skipped a slot whose async H2D upload was CONFIRMED still genuinely in flight (re-polled via cudaEventQuery, not just trusting a stale flag) -- the real "must not evict" count
    std::atomic<uint64_t> n_stale_upload_ready_reclaimed{0}; // times evict_lru found upload_ready==false but the event had ACTUALLY already completed (a speculative pilot-prefetch promotion that was never subsequently confirmed via find_expert) and flipped it back to true itself -- nonzero proves real cache capacity was being silently leaked before this fix; see gpu_topology.hpp's evict_lru comment
    std::atomic<uint64_t> n_stale_expert_index_hits{0}; // times find_expert's O(1) expert_index pointed at a slot that didn't actually match the requested (layer,expert_id) -- should be ALWAYS ZERO by construction (evict_lru/upload_expert/upload_expert_async are the only mutation sites, all under cache_mutex); nonzero means the index and expert_vram_cache have desynced somewhere and find_expert is safely falling back to its linear scan, but the root cause needs investigating
    std::atomic<uint64_t> n_pinned_resident{0};    // experts currently held as non-evictable VRAM residents (pinned). Gauge-ish: incremented on pin, decremented on unpin.
    std::atomic<uint64_t> n_pilot_hints_skipped{0}; // speculative pilot-prefetch hints skipped because the expert was ALREADY host-cache-resident (a guaranteed instant hit needing no read) -- measures how much prefetch-worker bandwidth this avoids wasting; compare against n_prefetch_hits/n_expert_cache_miss before/after to see the effect

    // Pure PREDICTION-ACCURACY recall: overlap between the pilot's
    // predicted top-K (LayerScratch::pilot_pred_ids, captured before any
    // already-resident filtering) and the REAL router's top-K when that
    // same layer's moe_forward_gguf actually runs, at the same "last
    // token" position both sides use. Deliberately measured completely
    // separately from "Prefetch served %" -- that number also
    // depends on worker-thread speed and staging-pool size (both already
    // tested this session with little effect), so it can't tell you
    // whether the REMAINING gap is prediction accuracy or something else
    // downstream. This can. n_pilot_recall_hits / (n_pilot_recall_checks *
    // topk) is the recall fraction, directly comparable to a reference
    // implementation's own reported recall number.
    std::atomic<uint64_t> n_pilot_recall_hits{0};
    std::atomic<uint64_t> n_pilot_recall_checks{0};   // number of (layer, "last token") comparisons made
    std::atomic<uint64_t> n_pilot_recall_topk_sum{0}; // sum of topk across those checks (denominator, since topk could vary in principle)

    // Qwen4Exp-only router-pilot measurements. This predictor uses the
    // device-authoritative HC stream-0 state as a deliberately explicit
    // proxy for the next layer's HC router input; it is not the conventional
    // hidden->RMSNorm->gate pilot above. These counters are separate so a
    // Qwen experiment cannot change the GLM/DeepSeek pilot report or policy.
    std::atomic<uint64_t> n_qwen_pilot_predictions{0};
    std::atomic<uint64_t> n_qwen_pilot_gpu_predictions{0};
    std::atomic<uint64_t> n_qwen_pilot_host_fallbacks{0};
    std::atomic<uint64_t> n_qwen_pilot_recall_hits{0};
    std::atomic<uint64_t> n_qwen_pilot_recall_checks{0};
    std::atomic<uint64_t> n_qwen_pilot_recall_pred_size_sum{0};
    // Accuracy of the candidates actually eligible for Qwen prefetch. This
    // is intentionally separate from the diagnostic top-10 predictor metric:
    // the runtime may predict top-10 but prefetch only the first N ranks, and
    // may disable prefetch for selected layers.
    std::atomic<uint64_t> n_qwen_pilot_prefetch_recall_hits{0};
    std::atomic<uint64_t> n_qwen_pilot_prefetch_recall_checks{0};
    std::atomic<uint64_t> n_qwen_pilot_prefetch_recall_pred_size_sum{0};
    std::atomic<uint32_t> n_qwen_pilot_prefetch_k_observed{0};
    std::atomic<uint64_t> n_qwen_pilot_pred_already_ram{0};
    std::atomic<uint64_t> n_qwen_pilot_pred_already_vram{0};
    std::atomic<uint64_t> n_qwen_pilot_pred_needs_fetch{0};
    // Predictive Partial Pinning empirical filter (2026-09-19): for the
    // ONE configured partial layer (ZEROMOE_QWEN4EXP_PARTIAL_LAYER),
    // per real decode token, M = |E_routed \ (S_pinned U E_staged)| --
    // the count of routed experts NOT covered by the static usage-ranked
    // VRAM pin set even after adding the pilot's own top-K predicted
    // experts as a hypothetical "staged" set. Tracked at three staging
    // depths (K=0 -- no pilot help at all, K=configured prefetch_k, K=5
    // -- the full diagnostic top-5 reporting set) to show the marginal
    // value of staging more predictions, not just today's configured
    // value. Bucketed at M=0 (complete hit), M=1, M>=2. See
    // qwen4exp-legacy-vs-fullgraph-routing-divergence.md and the Phase 2
    // session for why this has to be measured, not assumed.
    static constexpr int kPPPStagingScenarios = 3; // K=0, K=prefetch_k, K=5
    std::array<std::atomic<uint64_t>, kPPPStagingScenarios> n_ppp_m0{};
    std::array<std::atomic<uint64_t>, kPPPStagingScenarios> n_ppp_m1{};
    std::array<std::atomic<uint64_t>, kPPPStagingScenarios> n_ppp_m2plus{};
    // Sum of the real M value for every M>=2 token, so the M>=2 bucket's
    // average can be reported exactly instead of assumed.
    std::array<std::atomic<uint64_t>, kPPPStagingScenarios> n_ppp_m2plus_sum{};
    std::atomic<uint64_t> n_ppp_tokens_checked{0};
    std::atomic<uint64_t> n_qwen_pilot_hints_pushed{0};
    std::atomic<uint64_t> n_qwen_pilot_hints_skipped_resident{0};
    // Qwen estimation-based next-layer active-area prefetch. These are
    // separate from ordinary disk/RAM hint counters because an active-area
    // copy is an actual H2D transfer issued by cudaMemcpyAsync.
    std::atomic<uint64_t> n_qwen_pilot_active_prefetch_issued{0};
    std::atomic<uint64_t> n_qwen_pilot_active_prefetch_failed{0};
    std::atomic<uint64_t> n_qwen_pilot_prefetch_excluded{0};
    std::atomic<uint64_t> n_qwen_pilot_prefetch_rank_skipped{0};
    std::atomic<uint64_t> n_qwen_pilot_sync_fail{0};
    std::atomic<uint64_t> n_qwen_pilot_skipped_fully_cached{0};
    std::atomic<uint64_t> n_qwen_pilot_skipped_prefetch_excluded{0};
    std::atomic<uint64_t> n_qwen_pilot_skipped_predict_policy{0};
    // Qwen-only adaptive pilot bypass.  Unlike layer_fully_resident(), this
    // tracks the actual routed top-k set seen by decode, not all experts in
    // the layer.  A short streak prevents one lucky token from suppressing
    // the pilot before the route has demonstrated cache stability.
    std::atomic<uint64_t> n_qwen_pilot_adaptive_residency_checks{0};
    std::atomic<uint64_t> n_qwen_pilot_adaptive_residency_misses{0};
    std::atomic<uint64_t> n_qwen_pilot_adaptive_skips{0};
    static constexpr int kQwenPilotMaxRanks = 16;
    // Per-predicted-rank membership accuracy. A hit means the expert at
    // predicted rank r belongs to the real router's top-k set; this is the
    // useful definition for loading, where order among the required experts
    // does not matter.
    std::array<std::atomic<uint64_t>, kQwenPilotMaxRanks>
        n_qwen_pilot_rank_hits{};
    std::array<std::atomic<uint64_t>, kQwenPilotMaxRanks>
        n_qwen_pilot_rank_checks{};
    std::array<std::atomic<uint64_t>, kQwenPilotMaxRanks>
        n_qwen_pilot_prefetch_rank_hits{};
    std::array<std::atomic<uint64_t>, kQwenPilotMaxRanks>
        n_qwen_pilot_prefetch_rank_checks{};

    // ATTN-PILOT measurement (2026-08-25) -- predicts a layer from the
    // PREVIOUS layer's post-attention (pre-FFN) state, instead of DEPTH-1's
    // post-FFN state or DEPTH-2's naive 2-layer reuse. See execution_layer.
    // cpp's firing site (moe_forward_gguf, right after the attention-
    // residual check_finite call) for the full design -- one signal reads
    // as an early/cheaper DEPTH-1 (predicts layer T earlier, at the cost of
    // missing T-1's FFN) or a more-accurate/later DEPTH-2 (predicts layer T
    // from T-1's REAL attention instead of T-2's stale state) depending on
    // which existing baseline it's compared against. Exact mirror of n_
    // pilot_recall_hits/checks/topk_sum above, same accounting.
    std::atomic<uint64_t> n_pilot_attn_recall_hits{0};
    std::atomic<uint64_t> n_pilot_attn_recall_checks{0};
    std::atomic<uint64_t> n_pilot_attn_recall_topk_sum{0};
    // Fetch-worthy accuracy, exact mirror of n_pilot_recall_hint_hits/
    // checks_denom above -- restricted to picks that were NOT already
    // cached anywhere at prediction time (scratch.pilot_attn_pred_was_hint).
    std::atomic<uint64_t> n_pilot_attn_recall_hint_hits{0};
    std::atomic<uint64_t> n_pilot_attn_recall_hint_checks_denom{0};

    // UNION coverage (2026-08-25) -- real answer to "if DEPTH-1, DEPTH-2,
    // and ATTN-PILOT are ALL enabled and predicting a layer, is there
    // still a real disk-cold miss when we get there?" execution_layer.cpp's
    // union-coverage check merges whichever of the three predictions are
    // pending for a given layer (deduplicated) and measures overlap
    // against the real router's top-K, same accounting as n_pilot_recall_
    // hits/checks/topk_sum above. n_pilot_union_pred_size_sum is the sum of
    // each check's own union SIZE (before dedup against the real top-K) --
    // divide by n_pilot_union_recall_checks for the average number of
    // DISTINCT experts across all enabled predictors, to see how much
    // "extra" prefetch volume the union costs relative to any one
    // mechanism's own top-K width.
    std::atomic<uint64_t> n_pilot_union_recall_hits{0};
    std::atomic<uint64_t> n_pilot_union_recall_checks{0};
    std::atomic<uint64_t> n_pilot_union_recall_topk_sum{0};
    std::atomic<uint64_t> n_pilot_union_pred_size_sum{0};
};
