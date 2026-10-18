#pragma once
// Counter slice 2 of 8 split mechanically from perf_timer.hpp. Slices form a
// linear inheritance chain so constants declared in an earlier slice stay visible
// in later ones; `PerfCounters` derives from the last slice.
#include <array>
#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>
#include "perf/counters_01_attn.hpp"

struct PerfCountersPart02 : PerfCountersPart01 {
    std::atomic<uint64_t> n_expert_gpu_activation{0};

    // Per-stage breakdown INSIDE a GDN layer. Added 2026-09-04 after the
    // chunkwise-GDN GPU port cut the recurrence to 2.03s but moved the
    // 82s t_attn_qwen_gdn bucket by only 2.2s -- i.e. the recurrence was
    // never the bulk of that bucket, and the bucket had never been broken
    // down. These are nested inside t_attn_qwen_gdn, not a partition of it.
    std::atomic<double> t_gdn_qkv_proj{0};   // attn_qkv + attn_gate dense_proj
    std::atomic<double> t_gdn_conv{0};       // causal short conv + silu
    std::atomic<double> t_gdn_l2{0};         // per-head L2 norm of Q/K
    std::atomic<double> t_gdn_gates{0};      // ssm_alpha + ssm_beta + sigmoid
    std::atomic<double> t_gdn_delta{0};      // the recurrence itself
    std::atomic<double> t_gdn_outnorm{0};    // gated per-head RMSNorm
    std::atomic<double> t_gdn_outproj{0};    // ssm_out dense_proj
    std::atomic<double> t_gdn_full{0};      // complete device-resident GDN graph
    // Path accounting for the complete Qwen GDN graph. The timer above is
    // shared by device-input and host-output variants, so these counters make
    // a decode report identify the path that actually ran.
    std::atomic<uint64_t> n_gdn_full_device_success{0};
    std::atomic<uint64_t> n_gdn_full_host_success{0};
    std::atomic<uint64_t> n_gdn_full_device_failure{0};
    std::atomic<uint64_t> n_gdn_full_split_fallback{0};
    // Isolates the CPU-side WUV down-projection loop (attention_layer.cpp's
    // score_gpu_ok branch, run AFTER the GPU call returns) from
    // t_score_gpu_up/compute/down above, which only cover the GPU call
    // itself -- added specifically to root-cause a real, large gap seen on
    // the first real production run (score= bucket 4.21s vs. only 0.745s of
    // measured GPU up+cmp+dn for the same 3120 calls): the leading
    // hypothesis is this loop's own #pragma omp parallel for, spawned fresh
    // every single call, paying the same "team-spawn cost exceeds a tiny
    // workload" tax the qabsorb gather fix (2026-08-14) already found and
    // fixed at a different call site -- this counter will confirm or rule
    // that out directly instead of guessing again.
    std::atomic<double> t_score_gpu_wuv{0};
    // Incremented by the CALLER (attention_layer.cpp) whenever
    // score_gpu_batched_host returns false and it falls back to the CPU
    // per-head AVX-512 loop for that layer/token -- e.g. attn_score_gpu is
    // on but this layer's Lc/Rc GPU mirror was never allocated (VRAM
    // budget) or S>1 (prefill, out of scope for this GPU path -- see
    // attention_layer.cpp's own call-site comment). Should track close to 0
    // for decode once warm_mla_absorb_cache has finished allocating every
    // eligible layer's mirror; a steady nonzero DECODE rate across a whole
    // run means some layer never got a working GPU fast-path.
    std::atomic<uint64_t> n_score_gpu_fallback_cpu{0};

    // DSV4 raw-SWA/HCA score GPU fast-path (EngineConfig::dsv4_score_gpu,
    // cuda_manager.hpp's dsv4_score_gpu_batched_host) -- same shutdown-
    // report-only discipline as the MLA score counters above. Scoped to
    // ratio==0/ratio==128 layers only; CSA (ratio==4) never touches these.
    std::atomic<double> t_dsv4_score_gpu_up{0};
    std::atomic<double> t_dsv4_score_gpu_compute{0};
    std::atomic<double> t_dsv4_score_gpu_down{0};
    std::atomic<uint64_t> n_dsv4_score_gpu_calls{0};
    // Incremented by the CALLER (attention_layer.cpp) whenever
    // dsv4_score_gpu_batched_host returns false and it falls back to the
    // existing CPU per-head AVX-512 loop for that layer/token -- e.g.
    // dsv4_score_gpu is on but this layer's KVc/HCA GPU mirror was never
    // allocated (VRAM budget), the layer is CSA (ratio==4, always CPU by
    // design), or S>1 (prefill, out of scope for this GPU path -- see
    // attention_layer.cpp's own call-site comment). A steady nonzero DECODE
    // rate on a ratio==0/128 layer across a whole run means that layer
    // never got a working GPU fast-path.
    std::atomic<uint64_t> n_dsv4_score_gpu_fallback_cpu{0};

    // WUV BF16 fast-path (score GPU path's own down-projection, added
    // 2026-08-14 to replace the CPU per-head OMP-parallel loop t_score_gpu_
    // wuv above measured at 0.916ms/call -- see wuv_bf16_batched_host's own
    // comment, cuda_manager.hpp). Same shutdown-report-only discipline.
    std::atomic<double> t_wuv_gpu_up{0};
    std::atomic<double> t_wuv_gpu_compute{0};
    std::atomic<double> t_wuv_gpu_down{0};
    std::atomic<uint64_t> n_wuv_gpu_calls{0};
    // 2026-08-29: int8-persistent WUV coverage fix (upload_wuv_int8/
    // wuv_int8_batched_host, cuda_manager.hpp) -- t_wuv_int8_dequant is the
    // NEW cost this path adds on top of up/compute/down above (an
    // on-the-fly int8->BF16 dequant kernel, run once per call since each
    // call targets a different layer's WUV -- see that function's own
    // comment for why this isn't reused across calls). n_wuv_int8_layers
    // is a startup-only gauge (not a rate), set once after upload finishes.
    std::atomic<double> t_wuv_int8_dequant{0};
    std::atomic<uint64_t> n_wuv_int8_layers{0};
    // Incremented by the CALLER whenever wuv_bf16_batched_host returns
    // false (this layer's WUV BF16 upload never succeeded) and the CPU
    // per-head loop runs instead. Should track toward 0 once warm_mla_
    // absorb_cache finishes uploading every eligible layer's WUV.
    std::atomic<uint64_t> n_wuv_gpu_fallback_cpu{0};
    // GGML Qwen4Exp router boundary diagnostics. The host route path must
    // wait for the device router before it can select DDR expert addresses;
    // split that wait from the graph submission/readback for comparison with
    // llama.cpp's scheduler trace.
    std::atomic<double> t_ggml_router_host_call{0};
    std::atomic<double> t_ggml_router_host_sync{0};
    std::atomic<double> t_ggml_router_host_get{0};
    std::atomic<uint64_t> n_ggml_router_host_calls{0};
    std::atomic<uint64_t> n_ggml_router_device_calls{0};
    // Native GLM5NEXT router bridge: router activation is uploaded, the
    // device-resident router matmul/top-k runs, and only compact IDs/weights
    // are read back for the host cache bookkeeping.
    std::atomic<double> t_glm5next_router_gpu_up{0};
    std::atomic<double> t_glm5next_router_gpu_cmp{0};
    std::atomic<double> t_glm5next_router_gpu_dn{0};
    std::atomic<uint64_t> n_glm5next_router_gpu_calls{0};
    std::atomic<uint64_t> n_glm5next_router_gpu_fallback_cpu{0};
    std::atomic<double> t_moe_route{0};     // Router (dense_proj + gating activation) ONLY --
                                             // despite this counter's own name, top-k SELECTION
                                             // (the argsort/partial_sort + bias + pilot-recall
                                             // bookkeeping loop) is NOT inside this scope; see
                                             // t_moe_topk_select below, added specifically because
                                             // this mismatch between name and real scope left that
                                             // whole block invisible to every prior [REQUEST] line.
    std::atomic<double> t_moe_expert{0};    // Expert FFN computation
    // Added to chase a real, confirmed gap: gpu+silu+cpu+scan+ram_scan+
    // lock_wait summed to ~35s against expert=56s on a real run (~21s/30%
    // of the WHOLE request unaccounted for by any existing sub-timer,
    // despite every previously-hypothesized cause -- scan_total, promote_up,
    // stage_cpy, reg_stage, bb_wait -- already reading ~0 on that same run).
    // Direct source read of moe_forward_gguf (execution_layer.cpp) found two
    // large blocks with NO timer of their own: Phase 1 (batch-wide top-k
    // selection -- std::partial_sort over all n_experts, bias add, pilot
    // recall bookkeeping, route_freq histogram bumps) runs OUTSIDE
    // t_moe_route's own PERF_SCOPE despite that counter's name/comment
    // claiming to include it; and Phase 3 (the actual per-token dispatch
    // loop: shadow-fast-path pre-check, dispatch_expert_k's own branching/
    // gpu_ctx setup/expert_ffn_resolve_gpu_loc call, Pass B's expert_ffn_
    // gpu_finish + accumulate) has real loop-level orchestration cost
    // outside the nested gpu=/cpu=/ram_scan= sub-timers it also contains.
    // t_moe_topk_select is a clean, non-overlapping scope (Phase 1 only,
    // between t_moe_route's closing brace and Phase 2). t_moe_phase3_dispatch
    // is DELIBERATELY a diagnostic upper bound, not a clean partition --
    // same convention as ram_scan_total's own comment -- since it wraps the
    // whole per-token loop INCLUDING the already-separately-timed gpu=/cpu=/
    // ram_scan= work nested inside it; its purpose is to show whether the
    // per-token loop's OWN overhead (everything in it minus those nested
    // timers) is large, not to be summed alongside them as a fifth
    // independent bucket.
    std::atomic<double> t_moe_topk_select{0};
    std::atomic<double> t_moe_phase3_dispatch{0};
    // Follow-up round, same investigation: t_moe_phase3_dispatch isolated
    // ~18s of untimed cost to Phase 3's own loop after topk_select was
    // ruled out (0.03s). These three wrap the whole body of the three
    // functions Phase 3's dispatch_expert_k lambda calls for every routed
    // expert (expert_ffn_resolve_gpu_loc, expert_ffn_gpu_issue,
    // expert_ffn_gpu_finish) -- each is ALSO a diagnostic upper bound
    // (overlaps already-nested timers: resolve_gpu_loc contains hold_for_
    // compute/release_hold/find_active_expert, already in scan_total; issue
    // contains t_expert_gpu_upload; finish contains t_gpu_compute/
    // t_gpu_download) -- same "measure the whole function first, subtract
    // known nested pieces" approach as t_moe_phase3_dispatch itself. Whatever
    // residual remains after subtracting gpu=/scan_total from these three
    // localizes the ~18s further than "somewhere in Phase 3."
    std::atomic<double> t_resolve_gpu_loc_total{0};
    std::atomic<double> t_gpu_issue_total{0};
    std::atomic<double> t_gpu_finish_total{0};
    // Third round: resolve_gpu_loc/gpu_issue/gpu_finish's own "extra" beyond
    // their already-nested timers came back tiny (0.02+0.34+0.38=0.74s
    // combined on a real run) -- nowhere near the ~17-18s phase3_loop
    // residual, ruling out the per-routed-expert dispatch functions
    // themselves. Three more candidates, all structurally inside
    // moe_forward_gguf's t_moe_expert scope but none previously timed:
    // accumulate (the AVX-512 weighted-combine step, called once per
    // routed expert AND once for the shared expert -- clean, non-
    // overlapping scope, unlike the three above); the shared-expert dense
    // FFN block (GLM-5.2/DeepSeek-V3-style always-on expert, 3 dense_proj
    // calls once per token per layer -- its own GPU time already lands in
    // the SEPARATE dense_gpu= bucket, but since this code runs INSIDE
    // t_moe_expert's scope, that time ALSO counts toward expert='s total
    // without appearing in any of expert='s own sub-timers, a real
    // structural double-attribution the report never made visible); and
    // cleanup_active_layer_area (runs once per layer, AFTER Phase 3's own
    // loop -- outside t_moe_phase3_dispatch's scope but still inside
    // t_moe_expert's).
    std::atomic<double> t_accumulate_total{0};
    std::atomic<double> t_shared_expert_total{0};
    std::atomic<double> t_layer_cleanup_total{0};
    // Fifth round, same investigation (2026-08-29): the Fourth round's
    // t_active_layer_upload_async/t_active_layer_batch_sync explained MOST
    // of the ~15-18s residual (down to ~8.4s decode-only on a real matched-
    // prompt run, per the [PREFILL]-subtracted-from-[REQUEST] arithmetic
    // this whole investigation uses) but not all of it -- and every prior
    // round's own timers (resolve_gpu_loc/gpu_issue/gpu_finish/accumulate/
    // shared_expert/layer_cleanup/active_layer_prewarm_*) are already
    // wired into the [REQUEST] line, ruling out "just not reported yet" as
    // the explanation this time. Two more candidates, both genuinely
    // untimed until now, found by direct source read of moe_forward_gguf's
    // per-token dispatch loop: (1) the shadow-fast-path pre-check loop
    // (`for k in topk`, execution_layer.cpp) that decides VRAM-shadowed vs.
    // needs-a-real-fetch for every k BEFORE get_experts_batch's blocking
    // disk read -- does a real find_expert() call (cache_mutex) +
    // populate_vram_shadow() per k, entirely outside any existing timer;
    // (2) dispatch_expert_k's own lambda body has a SECOND, untimed
    // find_expert() fallback call (when a shadow hint isn't available) plus
    // ram_pin()/error-path bookkeeping, all outside resolve_gpu_loc's own
    // scope (which only starts once dispatch_expert_k calls into it).
    // Both are upper bounds like every timer in this investigation --
    // t_dispatch_expert_k_total nests resolve_gpu_loc_total/gpu_issue_total
    // (gpu_finish does NOT nest here -- it runs in the SEPARATE Pass B
    // loop, not inside dispatch_expert_k, confirmed by direct source read
    // before assuming otherwise); t_shadow_precheck_total additionally
    // nests t_dispatch_expert_k_total for whichever k's are shadowed
    // (dispatched inline, before get_experts_batch runs).
    //
    // RESULT (2026-08-29, real matched-11-token-prompt hardware run,
    // decode-only i.e. REQUEST-minus-PREFILL arithmetic): adding
    // t_shadow_precheck_total + t_ram_needed_dispatch_total (below) closed
    // the residual from ~8.35s (20.3% of decode wall) down to ~1.87s
    // (4.3%) -- confirmed via a clean, non-overlapping additive
    // decomposition: shadow_precheck + ram_scan + ram_needed_dispatch +
    // gpu_finish + accumulate + shared_expert + layer_cleanup +
    // active_layer_prewarm_issue/sync summed to 29.87s against
    // phase3_loop's own 31.74s. Cross-checked independently:
    // ram_needed_dispatch (14.76s) matched resolve_loc+gpu_issue combined
    // (14.78s) almost exactly, confirming that loop's own bookkeeping
    // overhead (beyond the two already-timed calls it makes) is
    // genuinely ~0, not a hidden cost. CONCLUSION: this investigation is
    // CLOSED -- the ~8-18s that looked like "unexplained Phase 3
    // orchestration overhead" across every prior round was never a real
    // hidden inefficiency, it was an INSTRUMENTATION GAP: the per-k
    // dispatch loops wrapping already-timed sub-functions (resolve_gpu_loc,
    // gpu_issue) had never themselves been measured, so their real (and
    // already-legitimate) cost had nowhere to be attributed until these
    // two timers existed. No code behavior changed; this was purely a
    // measurement fix. See CLAUDE.md for the full writeup.
    std::atomic<double> t_shadow_precheck_total{0};
    std::atomic<double> t_dispatch_expert_k_total{0};
    // The OTHER (mutually exclusive) half of dispatch_expert_k calls:
    // ram_needed_k's own loop, AFTER get_experts_batch resolves the real
    // RAM-tier misses. Together with t_shadow_precheck_total, this
    // partitions ~all dispatch_expert_k calls for a token/layer into two
    // non-overlapping top-level segments -- see t_shadow_precheck_total's
    // own comment and this field's own call site for the additive
    // decomposition this enables.
    std::atomic<double> t_ram_needed_dispatch_total{0};
    std::atomic<double> t_prefetch{0};      // fadvise/prefetch overhead
    std::atomic<double> t_dense_load{0};    // Initial resident tensor loading
    std::atomic<double> t_consume_staged{0};// Time in consume_staged (memcpy a
                                             // prefetched slab out of the staging
                                             // pool into the cache slot) -- the
                                             // Phase A "prefetch" overlap path. Near-
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
    // A slot round-robin-recycled (acquire_staging_slot) while STILL
    // ready==true and holding a real, never-consumed (layer,expert)
    // identity -- i.e. a correctly-predicted, correctly-read, correctly-
    // staged expert that simply never matched a miss before its slot got
    // reused. Added specifically to separate "pool thrashing" from other
    // explanations for the gap between Pilot recall (prediction accuracy
    // alone, see n_pilot_recall_hits) and Prefetch served %
    // (n_prefetch_hits) -- a high count here means the pool genuinely needs
    // to be bigger (or predictions need to happen closer to time-of-use);
    // a low count here despite a large recall/served gap means the
    // explanation is something else entirely (e.g. the predicted expert
    // quietly became a cache HIT via some other path before the miss
    // check ever ran, so consume_staged was never even called for it).
    std::atomic<double> t_staging_mutex_wait{0};  // no writer left (the old staging_mutex/PrefetchSlot pool this measured is gone -- see the "Unified RAM-tier claim/publish" comment below) -- kept alive only because it's still read by the live per-request [REQUEST] delta-report struct; always 0 now
    std::atomic<double> t_cache_mutex_wait{0};    // total time ANY caller spent BLOCKED acquiring GpuTopology::cache_mutex (find_expert/pin_expert/evict_lru/has_free_slot/upload_expert/upload_expert_async)

    // Unified RAM-tier claim/publish state machine (RAMSafetyManager::
    // claim_or_wait, ram_manager.hpp) -- replaces the old staging-pool-based
    // prefetch counters above (n_prefetch_hits/n_prefetch_miss_*/n_staged_evicted_unused/
    // n_staging_*/t_staging_*) now that prefetch and the decode thread's own
    // miss-handling claim/populate/publish through the SAME locked slot table
    // instead of two disconnected systems. See ram_manager.hpp's ram_mutex and
    // claim_or_wait for the full design.
    std::atomic<double> t_ram_mutex_wait{0};  // total time ANY caller (decode thread or a prefetch worker) spent BLOCKED acquiring RAMSafetyManager::ram_mutex -- mirrors t_cache_mutex_wait above for the RAM tier
    std::atomic<uint64_t> n_ram_claim_hit{0};              // claim_or_wait found the slot already Ready -- an ordinary cache hit, either caller
    std::atomic<uint64_t> n_ram_claim_by_main_thread{0};   // claim_or_wait's caller was the main generation-loop thread's own miss-handling (runs during BOTH prefill and decode -- see RamClaimCaller's own comment, engine.hpp, for why this is NOT decode-specific) and it won the Empty->Loading claim (i.e. no prefetch had already fetched this expert)
    std::atomic<uint64_t> n_ram_claim_by_prefetch{0};      // claim_or_wait's caller was a LockFreePrefetcher worker thread and it won the Empty->Loading claim
    std::atomic<uint64_t> n_ram_claim_already_resident_prefetch{0};  // a prefetch worker's claim attempt found the expert already Ready -- prefetch structurally cannot re-fetch an already-resident expert now (this is the direct, by-construction elimination of the old "Cache-hit-beat-fetch" waste category)
    std::atomic<uint64_t> n_ram_claim_already_inflight_prefetch{0};  // a prefetch worker's claim attempt found the expert already Loading (claimed by someone else, decode or another prefetch worker) -- a no-op dedup, same intent as the old "already in flight" staging-pool check, now checking the real shared cache
    std::atomic<uint64_t> n_ram_wait_inflight_main_thread{0};   // the decode thread's own miss-handling found the expert already Loading (claimed by a prefetch worker) and had to wait_for_ram_slot() for it -- this is the direct, by-construction elimination of the old "Phase-A/Phase-B ordering gap": no more check-once-then-race, the decode thread waits for the exact in-flight read instead
    std::atomic<double> t_ram_wait_inflight_ms_sum{0};     // sum of wait_for_ram_slot's own wait duration across all n_ram_wait_inflight_main_thread events
    std::atomic<uint64_t> n_ram_prefetch_loaded_evicted_unused{0};  // a RAM-tier victim-selection site reclaimed a slot with loaded_by_prefetch==true and hit_count==0 -- prefetch loaded it and normal LRU eviction reclaimed it before any real hit ever touched it. The unified design's replacement for the old staging-pool-based "wasted prefetch" measurement, now grounded in the real cache's real LRU pressure instead of an artificial round-robin pool.
    // 2026-08-30: the two halves of n_ram_prefetch_loaded_evicted_unused
    // above -- see MoEExpertBlock::consumed_by_vram_upload (engine.hpp) for
    // the full rationale. These SUM to n_ram_prefetch_loaded_evicted_unused
    // exactly (every increment of it also increments exactly one of these),
    // so a mismatch in the report is itself a bug signal.
    //   _consumed  = BENIGN: bytes reached VRAM and were used from there,
    //                only the RAM-tier hit_count is blind to it.
    //   _untouched = REAL WASTE: fetched, never read by anyone, reclaimed
    //                under capacity pressure. THIS is the number that
    //                justifies (or doesn't) the prefill sliding-window
    //                capacity work.
    std::atomic<uint64_t> n_ram_prefetch_evicted_consumed_by_vram{0};
    std::atomic<uint64_t> n_ram_prefetch_evicted_untouched{0};
    // EngineConfig::prefill_window_cap (2026-08-30, P1+P2) -- see
    // RAMSafetyManager::prefill_window_cap_by_layer's own comment.
    // granted/partial/denied_budget together show whether the hard extra-slot
    // budget (the thing standing in for the not-yet-built P3 trim) was the
    // binding constraint on coverage, which is exactly what decides whether
    // P3 is worth building.
    std::atomic<uint64_t> n_prefill_window_granted{0};
    std::atomic<uint64_t> n_prefill_window_partial{0};        // granted, but clipped by the remaining budget
    std::atomic<uint64_t> n_prefill_window_denied_budget{0};  // budget fully spent, layer kept its base cap
    std::atomic<uint64_t> n_prefill_window_not_needed{0};     // demand already <= current cap (short prompt)
    std::atomic<uint64_t> n_prefill_window_extra_slots_granted{0};
    // P3 (trim-on-window-exit). slots_skipped_inuse is the safety signal:
    // slots the trim could NOT free because a guard said they were still
    // live (ram_loading / temp-pinned / async-copy-pending). A small
    // nonzero value is healthy and expected (real concurrency); it just
    // means a little memory stayed held one layer longer. A LARGE value
    // would mean the window is sliding faster than work drains, which
    // would cap how much budget actually gets recycled.
    std::atomic<uint64_t> n_prefill_window_trimmed_layers{0};
    std::atomic<uint64_t> n_prefill_window_slots_freed{0};
    std::atomic<uint64_t> n_prefill_window_slots_skipped_inuse{0};
    // 2026-08-31: slots the hotness-ordered trim deliberately KEPT because
    // they were among the hottest -- real decode value, since 59.0% of
    // decode's measured traffic is covered by experts prefill touched (see
    // trim_prefill_window_locked's own comment). Currently expected to be 0
    // (the trim reclaims all extra capacity); nonzero only if a future
    // change retains some extra capacity across the window slide.
    std::atomic<uint64_t> n_prefill_window_slots_retained_hot{0};
    // 2026-08-31 prefill->decode retention handoff: RAM slots stamped with
    // this request's own prefill demand (MoEExpertBlock::prefill_demand).
    std::atomic<uint64_t> n_prefill_retention_stamped{0};
    std::atomic<uint64_t> n_prefill_retention_decayed{0};
    // Which layers actually received a window -- the grouping key for the
    // per-layer waste comparison in report_prefill_window_stats().
    std::array<std::atomic<uint8_t>, 128> prefill_window_layer_granted{};

    // Three previously COMPLETELY UNTIMED cudaMemcpy calls on the decode-time
    // hot path, added after a real investigation found expert=101.90s in a
    // [REQUEST] line but gpu=+cpu= (the only two sub-timers previously
    // printed) summed to only 16.28s -- an 84% gap with no instrumentation
    // to explain it. All three run while GpuTopology::cache_mutex is HELD
    // (see GPU_TOPO_TIMED_LOCK at each call site), so a slow one of these
    // doesn't just cost its own thread time -- it directly extends
    // t_cache_mutex_wait for every OTHER thread (the background prefetcher,
    // primarily) blocked on the same lock. Diagnostic-only: adding these
    // does not change behavior, only visibility.
    std::atomic<double> t_active_layer_upload{0};       // GpuTopology::upload_expert_active's H2D cudaMemcpy (decode-time active-layer hot path, gpu_topology.cpp)
    std::atomic<double> t_active_layer_promote_copy{0}; // GpuTopology::promote_active_to_main's D2D cudaMemcpy (active-layer -> main VRAM cache promotion, gpu_topology.cpp)
    std::atomic<double> t_vram_demote_copy{0};          // RAMSafetyManager::demote_from_vram's D2H cudaMemcpy (VRAM-evict-to-RAM salvage, ram_manager.hpp) -- fires on ANY eviction when vram_evict_demote_to_ram=true, including ones triggered mid-promotion via the on_vram_evict callback inside promote_active_to_main's own critical section

    // Direct timing of hold_for_compute/release_hold/find_active_expert's
    // own work done WHILE HOLDING cache_mutex (excludes the lock ACQUISITION
    // wait itself, already captured separately by t_cache_mutex_wait via
    // GPU_TOPO_TIMED_LOCK's own inline timer -- these start counting only
    // after the lock is already held). Added after TAKE 2's pool-hint fix to
    // hold_for_compute/release_hold (see gpu_topology.hpp's own comment on
    // that function) showed NO measurable change in a real before/after run
    // -- expert= and untimed= stayed essentially identical (108.45s/17.98s
    // vs the pre-fix 101.59s/17.36s, well within run-to-run noise) despite
    // being a real, reviewed fix for a real O(n)-scan problem. Rather than
    // guess a THIRD time whether these three functions (the only remaining
    // untimed work on the decode-time GPU dispatch hot path inside expert=)
    // are the actual remaining cause of the gap, measure them directly. Uses
    // PERF_SCOPE (RAII), not PERF_START/PERF_END, since all three functions
    // have multiple early-return paths that manual START/END pairing would
    // have to duplicate at every one.
    std::atomic<double> t_hold_for_compute{0};
    std::atomic<double> t_release_hold{0};
    std::atomic<double> t_find_active_expert{0};

    // Second round of the same investigation, after t_hold_for_compute/
    // t_release_hold/t_find_active_expert (measured, 0.65s combined) and
    // t_gpu_upload (measured, 0.00s) both ruled out the leading hypotheses
    // without closing the real ~65-70s gap between expert= and every
    // accounted-for sub-timer, across three separate real runs with a
    // remarkably STABLE gap size regardless of what got measured -- pointing
    // at ONE specific still-untimed mechanism rather than scattered small
    // costs. These four cover every remaining GpuTopology/RAMSafetyManager
    // function on the decode-time hot path that had no timer of its own:
    // find_expert (the highest-call-count function in the system -- called
    // once per topk-expert per token per layer for the shadow pre-check,
    // execution_layer.cpp), evict_lru (calls real cudaEventQuery driver
    // round-trips per not-yet-ready candidate, not just comparisons),
    // upload_expert_async (had ZERO instrumentation of any kind -- not even
    // a memcpy sub-timer -- despite being the RAM->VRAM reactive-hit MOVE-
    // promotion path), and get_expert/get_experts_batch's own bodies (only
    // their internal disk-read/consume-staged sub-calls were ever timed,
    // never the RAM-tier hit/miss scan or victim-selection logic around
    // them -- the single largest unwrapped scope on the whole path).
    std::atomic<double> t_find_expert{0};
};
