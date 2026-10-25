#pragma once
// Counter slice 7 of 8 split mechanically from perf_timer.hpp. Slices form a
// linear inheritance chain so constants declared in an earlier slice stay visible
// in later ones; `PerfCounters` derives from the last slice.
#include <array>
#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>
#include "perf/counters_06_active.hpp"

struct PerfCountersPart07 : PerfCountersPart06 {
    // 2026-08-26: counts a promotion that should_promote (DEPTH-1) / the
    // rank cutoff (ATTN-PILOT) already selected, but that dense_dispatch.
    // hpp's layer_promote_allowed then VETOED because this specific
    // layer's own live-measured accuracy is below ram_vram_promote_
    // layer_gate_threshold -- see EngineConfig::ram_vram_promote_layer_
    // gate's own comment. Zero whenever that gate is off (the default),
    // or before any layer has accumulated min_samples of live data yet.
    // Split by which pilot's own per-layer counters triggered the veto,
    // same reasoning as splitting dedup counts would be tempting but
    // wasn't needed for -- this one benefits from the split because the
    // two pilots' per-layer accuracy genuinely differs (see CLAUDE.md).
    std::atomic<uint64_t> n_pilot_promote_layer_gate_vetoed_depth1{0};
    std::atomic<uint64_t> n_pilot_promote_layer_gate_vetoed_attn{0};

    // "Idea 1" prediction-based GPU1 dual-role-slice loading (2026-08-16),
    // sibling counters to the two above -- same meaning, scoped to
    // RAMSafetyManager::promote_ram_to_vram_gpu1 (the rank-4+, GPU1-
    // targeted mechanism) instead of the existing rank-0/1 general-pool
    // one. Kept separate rather than sharing buckets specifically so this
    // new, narrower mechanism's own behavior is visible on its own, not
    // blended into a counter another call site already owns.
    std::atomic<uint64_t> n_predict_gpu1_promote_attempted{0};
    std::atomic<uint64_t> n_predict_gpu1_promote_slot_gone{0};
    // Incremented in execution_layer.cpp's pilot loop the one time per
    // layer-transition the per-transition cap (predict_gpu1_cap_per_layer)
    // is what actually held a rank back from promoting, not slot-gone or
    // rank mismatch -- answers "is the cap ever actually binding" without
    // needing a full waste-tracking apparatus for this first, narrow test.
    std::atomic<uint64_t> n_predict_gpu1_cap_reached{0};
    // TEMPORARY diagnostic (2026-08-16, churn investigation): of every
    // attempted promotion, how many found the target already resident
    // (no new transfer needed -- either still sitting from an earlier k4
    // promotion, or already on GPU0) vs how many triggered a genuinely
    // fresh H2D upload. A high fresh-upload share relative to attempted,
    // against GPU1's own small slot count, is direct, quantified evidence
    // of repeat-upload churn (the same expert being transferred multiple
    // times across one request because it keeps getting evicted before
    // reuse) -- see GpuTopology::upload_expert_async's own comment.
    std::atomic<uint64_t> n_predict_gpu1_already_resident{0};
    std::atomic<uint64_t> n_predict_gpu1_fresh_upload{0};

    // "Direction 1" (2026-08-16): how many REACTIVE (proven-hot) promotions
    // got routed to GPU1's dual-role slice via reactive_promote_gpu1_every_n
    // instead of the general pool. Real-demand-driven, not speculative --
    // kept separate from the reverted predict_gpu1_* counters above since
    // this is a structurally different (safer) mechanism.
    std::atomic<uint64_t> n_reactive_promote_routed_gpu1{0};

    // upload_expert()'s two silent failure modes -- both previously
    // returned nullptr with ZERO logging, making a real run's "why did
    // preload/promotion fail" undiagnosable. evict_lru returning no victim
    // at all (should be near-impossible on a startup-empty cache -- every
    // slot would have to be pinned/temp-pinned/upload-pending
    // simultaneously) vs the cudaMemcpy H2D itself failing (a genuine CUDA
    // error, logged with its own cudaGetErrorString()) are two very
    // different root causes and need to be told apart.
    std::atomic<uint64_t> n_upload_expert_no_victim{0};
    // Same "no victim found" outcome, but from a target_gpu_id-RESTRICTED
    // upload_expert call (e.g. pin_core_gpu_top_n) -- expected once that
    // one GPU's own slots fill up, not a bug signal like the unrestricted
    // counter above. Kept separate so a nonzero value here never gets
    // conflated with the real-bug case in dump_report().
    std::atomic<uint64_t> n_upload_expert_no_victim_restricted{0};
    std::atomic<uint64_t> n_upload_expert_memcpy_fail{0};
    std::atomic<uint64_t> n_upload_expert_stream_unavailable{0};  // upload_expert fell back to
                                             // the old plain (stream-0) cudaMemcpy because the
                                             // target device's copy_stream wasn't set -- should
                                             // read 0 (copy_stream is created once at device
                                             // init); nonzero means the stream-0 tax this was
                                             // meant to avoid is silently back for those calls.

    // How often expert_ffn_resolve_gpu_loc's uniform hold_for_compute()
    // call (execution_layer.cpp) fails for an apparently-VRAM-resident
    // expert -- i.e. its identity changed in the narrow window between the
    // residency check and the hold, forcing a safety fallback to CPU
    // rather than trust a possibly-stale pointer. Was only a LOG_WARN with
    // no counter, so a real hypothesis ("does a smaller reactive/unpinned
    // VRAM pool churn hard enough to make this fire more often, driving up
    // CPU fallback calls as pinned_vram_capacity grows") was previously
    // untestable except by eyeballing warning volume in the raw log.
    std::atomic<uint64_t> n_hold_for_compute_fail{0};

    // Full attribution for WHY expert_ffn_resolve_gpu_loc sends a call to
    // the CPU fallback path, split at the two return-nullptr sites in
    // execution_layer.cpp. Added after a real run showed
    // n_hold_for_compute_fail at 0 despite a rising CPU-fallback call
    // count -- that ruled out the eviction-race hypothesis, but nothing
    // else in this space was measured at all, so the next most likely
    // causes needed real counters instead of more guessing:
    //   type_unsupported: this expert's quant type has no GPU kernel for
    //     at least one of gate/up/down -- a FIXED, model-intrinsic cost,
    //     entirely independent of cache/pin sizing. A high, stable-across-
    //     runs value here means CPU-fallback volume is dominated by
    //     capability gaps (see gpu_unsupported_reason), not caching.
    //   upload_failed: the type WAS supported and an upload was genuinely
    //     attempted (cold-batch, active-layer, or cold-GPU) but never
    //     achieved VRAM residency -- see upload_expert's own
    //     n_upload_expert_no_victim/n_upload_expert_memcpy_fail for the
    //     specific cudaMemcpy/evict_lru cause when this one is nonzero.
    //   no_host_buffer: type was supported but gpu_ctx->slab_buffer was
    //     null, so NO upload was ever structurally attempted (every guard
    //     in expert_ffn_resolve_gpu_loc requires a non-null slab_buffer).
    //     Split out from upload_failed after a real run showed the WHOLE
    //     upload_failed bucket (2668 calls) with only ONE actual
    //     upload_expert error logged -- the two are NOT the same thing.
    //     The dominant source of a null slab_buffer here is a shadow-fast-
    //     pathed expert (execution_layer.cpp's exp->vram_shadow, which
    //     never carries a host copy by design) that raced and lost its
    //     VRAM residency between the shadow pre-check and this dispatch --
    //     see the caller's own "emergency RAM reload" comment, previously
    //     assumed "rare" but never actually counted.
    std::atomic<uint64_t> n_cpu_fallback_type_unsupported{0};
    std::atomic<uint64_t> n_cpu_fallback_upload_failed{0};
    std::atomic<uint64_t> n_cpu_fallback_no_host_buffer{0};
    // Expected/benign: gpu_ctx->slab_capacity==0 with slab_buffer non-null
    // -- the signature of a RAM slot that was JUST move_on_promotion'd to
    // VRAM (get_expert/get_experts_batch). gate_proj/up_proj/down_proj
    // host pointers are untouched by the MOVE, so the CPU fallback this
    // triggers still reads valid data -- a real but brief missed-GPU-
    // dispatch cost, not a failure. Was the dominant, previously-
    // misattributed contributor to n_cpu_fallback_upload_failed until
    // split out.
    std::atomic<uint64_t> n_cpu_fallback_just_promoted{0};
    // config cpu_affinity_layers: this layer is explicitly pinned to CPU --
    // expert_ffn_resolve_gpu_loc returns nullptr before even checking
    // residency/type, so every routed expert in this layer lands here
    // every call, by design (not an error or an unexpected fallback path).
    std::atomic<uint64_t> n_cpu_fallback_affinity_layer{0};

    // cuFile (GPUDirect Storage) direct disk->VRAM path (config
    // cuda_cufile, GpuTopology::upload_expert_cufile). n_cufile_read_ok/
    // fail are the terminal outcome of an ATTEMPTED cuFileRead (mirrors
    // n_upload_expert_no_victim/n_upload_expert_memcpy_fail's split for the
    // cudaMemcpy path). n_cufile_skipped_ram_resident is the one that
    // matters most for confirming this feature's actual contract is being
    // honored (not just assumed): it fires every time a layer was cuFile-
    // eligible but the expert turned out to already be RAM-resident, i.e.
    // the "never re-read from disk if RAM already has it" rule correctly
    // routed to the existing RAM-mediated promotion path instead. A run
    // where this stays at 0 despite cuFile being active for real traffic
    // would be suspicious -- some RAM-tier hits are expected on any warm
    // cache.
    std::atomic<uint64_t> n_cufile_read_ok{0};
    std::atomic<uint64_t> n_cufile_read_fail{0};
    std::atomic<uint64_t> n_cufile_skipped_ram_resident{0};
    // Deliberately separate from bytes_gpu_uploaded (the cudaMemcpy H2D
    // counter, shared by expert-cache promotions and dense-tensor offload)
    // -- same "don't conflate two different mechanisms" reasoning as
    // t_cufile_read above.
    std::atomic<int64_t> bytes_cufile_read{0};

    // How often the shadow-fast-path's own emergency RAM reload
    // (execution_layer.cpp, exp->vram_shadow branch) actually fires --
    // its own comment called this "rare" (a shadow-fast-pathed expert
    // raced and lost its VRAM residency, or expert_ffn_gpu_issue's scratch
    // grow failed) but that was never actually counted. success = the
    // emergency ram_mgr.get_expert() reload found real host data (the
    // expert's contribution is still computed, just via CPU instead of the
    // originally-expected GPU path); hard_fail = even the emergency reload
    // failed, so this expert's contribution is skipped ENTIRELY for this
    // token (a real, if rare, quality/correctness-relevant gap, not just a
    // perf one).
    std::atomic<uint64_t> n_shadow_race_reload_success{0};
    std::atomic<uint64_t> n_shadow_race_reload_hard_fail{0};

    // fp32_to_bf16 (kquant_dequant.hpp) rounding a FINITE f32 input up into
    // BF16's reserved Inf/NaN exponent pattern -- requires |f| within a
    // fraction of a percent of FLT_MAX (~3.4e38), astronomically larger
    // than any real LLM weight or activation, so this should be
    // essentially unreachable in practice. Clamped to the largest finite
    // BF16 magnitude (same sign) rather than silently letting an Inf
    // propagate into cuBLAS and poison every downstream value with no
    // visible cause -- matches this project's standing "never silently
    // reinterpret/propagate" discipline (see dense_dispatch.hpp's own
    // header comment). Should ALWAYS read 0; nonzero is a genuine,
    // actionable signal (a runaway activation/numerical instability
    // upstream), not routine rounding noise.
    std::atomic<uint64_t> n_bf16_overflow_clamped{0};

    // Prefill-vs-decode split of the same hit/miss counts above. Added to
    // test a specific hypothesis: the pilot predicts from only the LAST
    // token's hidden state, which can't cover prefill's full per-layer
    // expert fan-out across ALL prompt tokens at once. (This split used to
    // also feed an n_prefetch_hits_prefill/decode pair for the old separate
    // staging pool's own hit rate -- removed along with that pool; these
    // two miss counters themselves are still live, now fed by claim_or_wait,
    // ram_manager.hpp.)
    std::atomic<uint64_t> n_expert_cache_miss_prefill{0};
    std::atomic<uint64_t> n_expert_cache_miss_decode{0};

    // Phase 2 (moe_forward_gguf's S>1 cold-start VRAM warm-up) instrumentation
    // -- previously only visible via `grep -c "\[MOE-VRAM\]  WARM"` on the raw
    // log, no summary counter at all. Legacy synchronous path (default,
    // config prefill_warm_via_prefetcher=false): attempted = every candidate
    // that passed the free-slot/not-already-resident/gpu-capable gates and
    // reached the actual get_expert+upload_expert_async call; succeeded =
    // upload_expert_async returned a real slot; failed = it returned nullptr
    // (VRAM exhausted mid-loop, or get_expert's own slab_buffer came back
    // null). t_prefill_warm_time wraps the WHOLE Phase 2 loop body
    // (find_expert checks, get_expert's own potential disk read,
    // upload_expert_async issue) -- this is what actually answers "how much
    // of Prefill (whole pass) is Phase 2 vs. real attention/FFN compute,"
    // instead of inferring it indirectly from a WARM count. When
    // prefill_warm_via_prefetcher=true, this loop instead just pushes hints
    // (see n_prefill_hint_pushed below) -- expect t_prefill_warm_time to
    // collapse toward ~0 in that mode, since the actual read/promote work
    // moves onto the prefetcher's background worker thread; that collapse
    // is itself the confirmation the mechanism actually changed, same
    // diagnostic pattern as t_cufile_read vs t_gpu_upload.
    std::atomic<uint64_t> n_prefill_warm_attempted{0};
    std::atomic<uint64_t> n_prefill_warm_succeeded{0};
    std::atomic<uint64_t> n_prefill_warm_failed{0};
    std::atomic<double> t_prefill_warm_time{0};
    // prefill_warm_via_prefetcher=true path: counts hints pushed via
    // async_io_queue.push_hint, NOT a success/failure outcome (that's the
    // whole point of handing it to the background worker instead of
    // blocking here to find out) -- these hints flow through the exact
    // same queue/worker pipeline the pilot's own hints already use, so
    // their eventual claim outcome shows up in the "Unified RAM claim"
    // report line, not a dedicated counter of their own. (There used to
    // also be an n_prefetch_miss_still_queued counter here, measuring the old
    // separate staging pool's own miss-vs-queue-state split -- removed
    // along with that pool; LockFreePrefetcher::has_pending_hint itself is
    // still a valid diagnostic peek at the SPSC queue, just no longer fed
    // into a dedicated counter.)
    std::atomic<uint64_t> n_prefill_hint_pushed{0};
    // LayerScratch::pending_prefill_promotions' own opportunistic-promotion
    // outcomes (execution_layer.cpp): a pushed hint that later landed in
    // RAM (checked cross-layer, decode-thread-only) gets promoted to VRAM
    // right there via upload_expert_async, same as the legacy synchronous
    // Phase-2 path always did inline -- these are NOT speculative bets
    // (every entry is a confirmed real router pick for this exact prefill
    // batch), so "promoted" here has no accuracy/wrong-guess dimension the
    // way decode's own ram_vram_promote_top_k does; it's purely a timing
    // question. n_prefill_pending_still_loading is expected to dominate
    // early (most checks land before the read finishes) -- only a genuine
    // problem if n_prefill_pending_aged_out stays high too (both is a
    // signal the check cadence/cap is too tight to ever catch up).
    std::atomic<uint64_t> n_prefill_pending_promoted{0};
    std::atomic<uint64_t> n_prefill_pending_still_loading{0};
    std::atomic<uint64_t> n_prefill_pending_aged_out{0};
    std::atomic<uint64_t> n_prefill_pending_no_free_slot{0};
    // Found already VRAM-resident by the time this follow-up pass checked
    // it -- something ELSE (most likely Phase 3's own synchronous dispatch
    // for that entry's layer, possibly a reactive hit-count promotion)
    // beat this opportunistic pass to it. Not a failure -- the expert IS
    // VRAM-resident either way -- but a real accounting gap without this:
    // n_prefill_hint_pushed should equal the sum of promoted+still_loading+
    // aged_out+no_free_slot+already_resident (modulo whatever's still
    // legitimately pending when the process exits); without this counter
    // a large silent remainder was unexplained.
    std::atomic<uint64_t> n_prefill_pending_already_resident{0};
    // A pushed hint whose (layer,expert) never even got a tracking slot --
    // LayerScratch::pending_prefill_promotions' 32-entry cap was already
    // full at push time. This is the OTHER real gap the first version of
    // this accounting missed (already_resident above was the first): a
    // real run showed 146 promoted + 96 already_resident + 0 still_loading
    // + 0 aged_out + 14 no_free_slot = 256 accounted of 420 hints pushed --
    // 164 (39%) unexplained until this counter existed. High values here
    // mean the cap itself, not fetch speed or free-slot availability, is
    // the binding constraint -- the fetches ARE landing fast (0 aged_out/
    // still_loading in that same run says so), there just isn't room to
    // track more than 32 at once given how many hints one layer's Phase 2
    // can push before the next layer's follow-up pass drains any.
    std::atomic<uint64_t> n_prefill_pending_cap_dropped{0};

    // io_uring batch outcome, per active_prefetch() call that attempted it
    // (io_uring=true AND Ring::available()). Added because the fallback-on-
    // any-failure design (io_uring_backend.hpp) is deliberately silent at
    // the per-call level -- without this, there is NO way to tell from a
    // real run's numbers whether io_uring is actually doing anything or
    // quietly failing every single batch and always taking the sequential
    // pread path. n_uring_batch_ok + n_uring_batch_fallback == every
    // attempted batch; a near-0 ok fraction despite [IO-URING] Enabled at
    // startup means Ring::init() succeeded but real reads are failing --
    // worth a closer look, not a "it's just not helping" shrug.
    std::atomic<uint64_t> n_uring_batch_ok{0};
    std::atomic<uint64_t> n_uring_batch_fallback{0};

    // Same idea as n_uring_batch_ok/n_uring_batch_fallback above, but for
    // RAMSafetyManager::get_experts_batch's OWN decode-thread uring attempt
    // (ram_manager.hpp Phase B) -- a SEPARATE call site from the
    // prefetcher's, sharing nothing but the pattern. Added after a real run
    // showed the decode thread's own ring log "[IO-URING] Enabled..." at
    // startup, yet `get_batch=` (a genuine, non-thread-summed wall-clock
    // PERF_SCOPE) and `disk=` (which SHOULD sit almost on top of get_batch=
    // when uring genuinely handles every batch, since uring's own batch_ms
    // is what populates disk= from inside get_batch's scope) differed by
    // ~3.7x -- exactly the signature of the OMP-parallel fallback's thread-
    // summed t_disk_read firing far more than expected, despite successful
    // ring initialization at startup. "Ring initialized OK" is not proof
    // every batch afterward actually used it -- this makes that visible
    // instead of inferred from a timing ratio. Split by root cause, not
    // just ok/fail, since each failure mode points at a different fix:
    //   ok:            uring handled this batch's ENTIRE pending set,
    //                  successfully, every request the right length.
    //   not_available: use_decode_uring==false or Ring::available()==false
    //                  at THIS call -- uring was never even attempted this
    //                  batch, despite (or independent of) startup init.
    //   not_packed:    build_expert_read_requests failed for at least one
    //                  pending expert THIS batch. For the default packed
    //                  format, almost certainly packed_slab_geometry
    //                  finding one of the three _exps tensors missing for
    //                  that expert's layer, i.e. a SPECIFIC LAYER'S repack
    //                  is incomplete (see CLAUDE.md's "Silent Incomplete
    //                  Repack" entry) even though gguf_tool verify or the
    //                  overall model otherwise looks fine -- run gguf_tool
    //                  verify on the model directory directly if this is
    //                  ever nonzero. For unpacked_expert_format=true this
    //                  should never fire at all (sanity_check_unpacked_
    //                  format already refused to start the run otherwise);
    //                  if it ever does, see the [WARN] it logs alongside.
    //   too_large:     this batch's request count exceeded Ring::CAPACITY
    //                  (64) -- should be unreachable at topk=8 (max 24
    //                  requests/batch); nonzero here means topk or the
    //                  pending-miss count is larger than assumed.
    //   submit_failed: decode_uring_ring.submit_and_wait itself returned
    //                  false -- a genuine io_uring_enter/kernel-level
    //                  failure, not a data problem.
    //   short_read:    submit_and_wait succeeded, but at least one
    //                  request's result length didn't match what was
    //                  asked for -- a real, individual read failure
    //                  (short read / I/O error on that specific request).
    std::atomic<uint64_t> n_decode_uring_ok{0};
    std::atomic<uint64_t> n_decode_uring_not_available{0};
    std::atomic<uint64_t> n_decode_uring_not_packed{0};
    std::atomic<uint64_t> n_decode_uring_too_large{0};
    std::atomic<uint64_t> n_decode_uring_submit_failed{0};
    std::atomic<uint64_t> n_decode_uring_short_read{0};

    // Phase B speculative-padding counters (disk_batch_padding config,
    // RAMSafetyManager::get_experts_batch) -- see that config field's own
    // comment for the mechanism. Each has a specific diagnostic meaning:
    //   issued:        a candidate was claimed (RAM slot stamped, request
    //                  appended) and submitted as part of this batch.
    //   ok:            that candidate's read came back the right length --
    //                  its slot is now genuinely, correctly resident.
    //   failed:        that candidate's read failed (short read / I/O
    //                  error) -- the claim was reverted (revert_padding_
    //                  claim), the slot is back to genuinely free. Should
    //                  be rare; a real, individual read failure, not a
    //                  padding-logic bug.
    //   hit_later:      a padded slot was genuinely routed to by a LATER
    //                  real token before being reclaimed for something
    //                  else -- the actual payoff this feature exists for.
    //                  hit_later/issued is the real win-rate to watch.
    //   no_free_slot:  padding WANTED a candidate this call but found no
    //                  genuinely-free RAM slot to claim it into (V1 never
    //                  evicts a resident). Watch this ratio closely in
    //                  steady state: this project's own real [RAM-SLOTS]
    //                  history shows the RAM tier saturates to ~100% full
    //                  within the first handful of decode tokens and stays
    //                  there -- a high ratio here means padding is mostly
    //                  only engaging during each run's brief ramp-up
    //                  window, not the full run (see CLAUDE.md).
    std::atomic<uint64_t> n_pad_reads_issued{0};
    std::atomic<uint64_t> n_pad_reads_ok{0};
    std::atomic<uint64_t> n_pad_reads_failed{0};
    std::atomic<uint64_t> n_pad_reads_hit_later{0};
    std::atomic<uint64_t> n_pad_no_free_slot{0};
    // disk_batch_padding_evict_dup's own two outcomes -- see that config
    // field's own comment (engine_config.hpp) for the full safety argument.
    // n_pad_evict_dup_used: padding candidates filled by reclaiming an
    // also_in_vram RAM slot instead of a genuinely free one (subset of what
    // n_pad_no_free_slot would otherwise have counted as unfillable).
    // n_pad_evict_dup_unavailable: STILL couldn't reach target even after
    // considering eviction (no also_in_vram slots left either) -- both stay
    // 0 when disk_batch_padding_evict_dup=false (the default).
    std::atomic<uint64_t> n_pad_evict_dup_used{0};
    std::atomic<uint64_t> n_pad_evict_dup_unavailable{0};
    // Diagnostic pair added specifically because a real run showed BOTH
    // n_pad_reads_issued AND n_pad_no_free_slot stay at exactly 0 for a
    // whole 40-token run despite route_rank_by_layer confirmed populated
    // (100+ entries/layer in .zeromoe_usage) -- meaning the candidate
    // list (`ranked`) itself was empty every single call, not just short
    // of free slots. These two counters distinguish the two ways that can
    // happen: no_ranking_for_layer fires if route_rank_by_layer had NO
    // entry at all for that layer (population bug, or a layer-numbering
    // mismatch between .zeromoe_usage and get_experts_batch's own
    // `layer` parameter); no_candidates_after_filter fires if the ranking
    // DID exist but every single entry got excluded (this-token's-own-set
    // / RAM-resident / VRAM-resident) before `want` candidates were found.
    std::atomic<uint64_t> n_pad_no_ranking_for_layer{0};
};
