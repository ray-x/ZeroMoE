#pragma once
// Counter slice 6 of 8 split mechanically from perf_timer.hpp. Slices form a
// linear inheritance chain so constants declared in an earlier slice stay visible
// in later ones; `PerfCounters` derives from the last slice.
#include <array>
#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>
#include "perf/counters_05_active.hpp"

struct PerfCountersPart06 : PerfCountersPart05 {

    // 2026-08-26: diagnostic pair for the speculative path specifically --
    // see the cudaStreamQuery call site's own comment (gpu_topology.cpp,
    // upload_expert_active_async). Directly tests whether the ~2s/request
    // active_layer_prewarm_issue growth (far more than the ~5% wrong-guess
    // rate alone predicts) is explained by real cross-mechanism stream
    // contention at issue time.
    std::atomic<uint64_t> n_speculative_issue_saw_active_layer_stream_busy{0};
    std::atomic<uint64_t> n_speculative_issue_saw_active_layer_stream_idle{0};
    std::atomic<uint64_t> n_speculative_issue_saw_own_stream_busy{0};
    std::atomic<uint64_t> n_speculative_issue_saw_own_stream_idle{0};
    std::atomic<uint64_t> n_active_layer_cleanup{0};    // active layer slots reclaimed after layer completion
    std::atomic<uint64_t> n_active_layer_promoted{0};   // active-layer experts promoted into the real VRAM LRU cache (hit_count crossed active_layer_promote_threshold), vs. just freed at cleanup

    // 2026-08-28: decode-only occupancy sample -- how many active_layer_
    // slots are genuinely non-empty (layer_idx != -1) at the moment each
    // cleanup_active_layer_area call fires, gated !in_prefill so prefill's
    // much larger whole-batch footprint never pollutes this. Answers a
    // direct, previously-unmeasured question (a user asked "why 256 slots
    // when decode only ever has ~8+8=16 active at once") with a real
    // number instead of an assumption -- see the [ACTIVE-LAYER-OCCUPANCY]
    // report line and CLAUDE.md's active-layer sizing discussion.
    std::atomic<uint64_t> active_layer_decode_occupancy_max{0};
    std::atomic<uint64_t> active_layer_decode_occupancy_sum{0};
    std::atomic<uint64_t> active_layer_decode_occupancy_samples{0};
    // Same sample, taken during prefill (in_prefill==true) instead --
    // companion measurement to size the pool for prefill's own real
    // worst case, not just decode's.
    std::atomic<uint64_t> active_layer_prefill_occupancy_max{0};
    std::atomic<uint64_t> active_layer_prefill_occupancy_sum{0};
    std::atomic<uint64_t> active_layer_prefill_occupancy_samples{0};

    // GpuTopology::upload_expert_active_async / sync_active_layer_batch
    // (2026-08-12, prefill active_up investigation): batch pre-warm of a
    // layer's known active-layer misses (from the SAME already-computed
    // router output Phase 2's disk-prefetch push already uses) via N
    // cudaMemcpyAsync issues + ONE shared cudaStreamSynchronize, instead
    // of N separate blocking cudaMemcpy calls scattered through Phase 3's
    // per-token dispatch loop. t_active_layer_upload_async measures only
    // the async issue calls (should be small/cheap, no wait); t_active_
    // layer_batch_sync measures the one shared wait per layer -- compare
    // their SUM against the old t_active_layer_upload (sync path,
    // per-call blocking) on an equivalent population to see whether
    // batching actually amortizes per-call overhead as intended, or just
    // moves the same total cost around for no net change.
    std::atomic<double> t_active_layer_upload_async{0};
    std::atomic<double> t_active_layer_batch_sync{0};

    // Actual active-layer H2D intervals measured by persistent CUDA events.
    // Split by async-vs-reactive-sync and by persistent host pin coverage.
    // These times exclude CPU cudaMemcpyAsync issue/staging and the later
    // stream wait; bytes/time is therefore a per-transfer copy-engine rate.
    std::atomic<double> t_active_layer_dma_async_pinned{0};
    std::atomic<double> t_active_layer_dma_async_unpinned{0};
    std::atomic<double> t_active_layer_dma_sync_pinned{0};
    std::atomic<double> t_active_layer_dma_sync_unpinned{0};
    std::atomic<int64_t> bytes_active_layer_dma_async_pinned{0};
    std::atomic<int64_t> bytes_active_layer_dma_async_unpinned{0};
    std::atomic<int64_t> bytes_active_layer_dma_sync_pinned{0};
    std::atomic<int64_t> bytes_active_layer_dma_sync_unpinned{0};
    std::atomic<uint64_t> n_active_layer_dma_async_pinned{0};
    std::atomic<uint64_t> n_active_layer_dma_async_unpinned{0};
    std::atomic<uint64_t> n_active_layer_dma_sync_pinned{0};
    std::atomic<uint64_t> n_active_layer_dma_sync_unpinned{0};
    // 2026-08-28: total bytes carried by every cudaMemcpyAsync issued at
    // this call site (upload_expert_active_async), across BOTH callers --
    // decode_async_prewarm AND the speculative pilot fill (they share this
    // one function/timer). Added to directly answer "is DDR->VRAM actually
    // hitting a bandwidth ceiling" with a real measured GB/s (bytes /
    // t_active_layer_upload_async) instead of reasoning from an older
    // session's own byte-count estimate -- see the "Active-layer async
    // upload achieved bandwidth" report line.
    std::atomic<int64_t> bytes_active_layer_upload_async{0};
    std::atomic<uint64_t> n_active_layer_uploads_async_issued{0};
    // 2026-08-28: multi-GPU active-layer routing decisions, by target GPU
    // -- see route_active_layer_gpu's own comment (dense_dispatch.hpp).
    // Covers BOTH callers (DEPTH-1/ATTN-PILOT speculative fill and decode_
    // async_prewarm's own real-need candidates) in one array, since the
    // question this answers ("which GPU is traffic actually landing on")
    // doesn't care which mechanism decided to send it there -- the
    // existing bytes_h2d_upload_by_gpu/n_h2d_upload_by_gpu counters
    // already give the REALIZED per-GPU byte/call totals across all
    // upload paths; this one specifically isolates the ROUTING DECISION
    // itself (issued before the actual upload call, so it also counts a
    // routed candidate that later fails to find a free slot on its
    // target GPU -- compare against the realized totals to see how often
    // that happens).
    std::atomic<uint64_t> n_active_layer_routed_by_gpu[8]{};
    // 2026-08-27: real coverage measurement for upload_expert_active_async's
    // OWN source-pinning gap -- unlike upload_expert_async (main VRAM cache
    // path), this function does no persistent-pin check and no transient
    // cudaHostRegister (that was tried once and reverted -- too much per-
    // call overhead at this call site's volume, see CLAUDE.md). It's only
    // genuinely async when host_data happens to already fall inside
    // ram_pinned_budget_gb's persistent pin budget; otherwise the driver
    // secretly stages it synchronously. This checks MoEExpertBlock::
    // slab_pinned (already tracked by the RAM tier, zero new CUDA calls)
    // right before each of this function's three real callers (prefill's
    // own Phase-2 batch warm-up, decode_async_prewarm's own batch, and
    // RAMSafetyManager::promote_ram_to_active_layer's speculative fill) --
    // answers "what fraction of THIS specific call site's traffic is
    // actually fast" directly, instead of inferring it from the general
    // RAM-tier-wide pin coverage report.
    std::atomic<uint64_t> n_active_layer_upload_src_pinned{0};
    std::atomic<uint64_t> n_active_layer_upload_src_unpinned{0};
    // 2026-08-28: direct answer to "is active_layer_prewarm_issue's cost
    // real PCIe transfer time, or a CPU-bound driver-internal staging
    // memcpy for unpinned sources" -- real hardware nvidia-smi telemetry
    // (dmon -s t) showed GPU0's PCIe RX averaging only ~2.2-2.6 GB/s
    // during decode (peak ~9-13 GB/s, touched rarely), nowhere near
    // saturated -- ruling out "we're at the wire" as the explanation for
    // this bucket's real cost. Splits the EXACT SAME cudaMemcpyAsync call
    // this bucket already times (PERF_START/END around the call only, see
    // t_active_layer_upload_async's own comment) by whether the source was
    // known-pinned at call time. If unpinned calls cost dramatically more
    // per-call than pinned ones, that's the decisive, direct confirmation:
    // the driver's own internal staged-copy fallback for unpinned memory
    // is a real CPU/host-memory-bandwidth-bound cost hiding inside a call
    // that LOOKS async, not a PCIe-wire cost.
    std::atomic<double> t_active_layer_upload_async_pinned{0};
    std::atomic<double> t_active_layer_upload_async_unpinned{0};
    std::atomic<uint64_t> n_active_layer_upload_async_pinned_calls{0};
    std::atomic<uint64_t> n_active_layer_upload_async_unpinned_calls{0};
    // 2026-08-28: ActiveLayerUploadWorkers (active_layer_upload_workers.hpp)
    // -- background worker-thread pool that moves the register+copy+event
    // sequence for an UNPINNED-source active-layer upload off the decode
    // thread entirely. n_active_layer_worker_submitted/queue_full: decode-
    // thread-side submission outcome (queue_full means the caller had to
    // abandon its claim and fall back -- see GpuTopology::abandon_worker_
    // claim's own comment). t_active_layer_worker_register/_calls/_fail:
    // the worker thread's own cudaHostRegister cost, now paid off the
    // decode thread -- compare against t_upload_async_host_register (the
    // analogous main-cache-path counter) for an apples-to-apples read on
    // whether this call site's register cost matches that one's. n_active_
    // layer_worker_processed: total work items a worker thread actually
    // completed (claim -> register -> copy -> event -> both hand-offs), the
    // denominator for the pool's own real throughput.
    std::atomic<uint64_t> n_active_layer_worker_submitted{0};
    std::atomic<uint64_t> n_active_layer_worker_queue_full{0};
    std::atomic<uint64_t> n_active_layer_worker_claim_failed{0};
    std::atomic<double> t_active_layer_worker_register{0};
    std::atomic<uint64_t> n_active_layer_worker_register_calls{0};
    std::atomic<uint64_t> n_active_layer_worker_register_fail{0};
    // 2026-08-28: the worker's own SYNCHRONOUS unregister (waits for this
    // item's own copy to finish, then unregisters right there) -- see
    // ActiveLayerUploadWorkers::process's own comment for why this is
    // deliberately NOT deferred through GpuTopology::pending_host_
    // unregisters (a real hardware bug, found and fixed: that queue is
    // only reaped from upload_expert_async's own call site, which this
    // pool's much-higher-volume callers could starve indefinitely).
    std::atomic<double> t_active_layer_worker_unregister{0};
    std::atomic<uint64_t> n_active_layer_worker_unregister_calls{0};
    std::atomic<uint64_t> n_active_layer_worker_processed{0};
    // 2026-08-28: defensive hardening added while chasing a real, twice-
    // reproduced hardware crash (NVRM Xid 31 MMU fault) -- see
    // ActiveLayerUploadWorkers::process's own comment. Both should read 0
    // in a healthy run; either being nonzero is a direct, immediate signal
    // (unlike the crash's own original symptom, which surfaced only as an
    // unrelated "unspecified launch failure" much later, once the CUDA
    // context was already silently corrupted).
    std::atomic<uint64_t> n_active_layer_worker_invalid_item{0};
    std::atomic<uint64_t> n_active_layer_worker_copy_fail{0};
    // 2026-08-31: slots invalidated because their H2D copy failed. These are
    // uploads that previously would have been PUBLISHED holding uninitialized
    // VRAM, producing NaN. Nonzero here means the underlying upload bug is
    // still occurring -- but is now contained (dispatch re-fetches instead of
    // computing on garbage), not silently corrupting output.
    std::atomic<uint64_t> n_active_layer_worker_slot_invalidated{0};
    // 2026-08-31 root-cause fix: slab regrows that had to DEFER freeing the
    // old buffer because a reader still held it (this is the race that used
    // to produce a dangling host_src -> "cudaMemcpyAsync ... invalid
    // argument" -> NaN). n_ram_slab_free_deferred counts how often the race
    // was caught; n_ram_slab_deferred_freed counts buffers later released.
    // The two should converge -- a persistent gap means entries are stuck,
    // i.e. some block's guards never clear.
    std::atomic<uint64_t> n_ram_slab_free_deferred{0};
    std::atomic<uint64_t> n_ram_slab_deferred_freed{0};
    // Same pair, but scoped ONLY to the speculative pilot-fill caller
    // (promote_ram_to_active_layer) -- lets V1-vs-V2's own speculative-
    // promotion coverage be compared apples-to-apples against upload_
    // expert_async's own n_upload_async_src_already_pinned counter (the
    // main-cache path V1 uses for the identical candidate population).
    std::atomic<uint64_t> n_speculative_fill_src_pinned{0};
    std::atomic<uint64_t> n_speculative_fill_src_unpinned{0};

    // 2026-08-27 probe: is the ~46-54% UNPINNED half of active-layer
    // traffic a small set of experts repeatedly missing pin coverage
    // (dynamic pin-swap would help a lot -- keep just that small set
    // pinned and close most of the gap cheaply), or genuinely spread
    // across the whole ~95GB RAM-tier working set (dynamic pin-swap
    // would have to churn constantly and likely wouldn't pay for
    // itself)? Answers this BEFORE committing to that bigger, riskier
    // build. Fixed atomic "seen" grid, not a mutex+set, to match this
    // file's own lock-free-only convention -- 128 layers x 320 experts
    // (generous headroom over GLM-5.2's real up-to-256/layer) x 1 byte
    // = 40KB, trivial. `mark_unpinned_touch(layer, expert_id)` is the
    // one entry point every call site below uses; it flips the "seen"
    // bit and only counts a NEW distinct key toward n_unpinned_touch_
    // distinct, while n_unpinned_touch_total counts every occurrence
    // regardless. Report: distinct/total IS the concentration ratio --
    // close to n_active_layer_upload_src_unpinned's own total means
    // spread out (every miss is a different expert); much smaller means
    // concentrated (the same experts keep missing pin coverage).
    static constexpr int kMaxExpertsPerLayerForPinProbe = 320;
    std::array<std::atomic<uint8_t>,
               (size_t)kMaxTrackedLayersForMiss * kMaxExpertsPerLayerForPinProbe>
        unpinned_touch_seen{};
    std::atomic<uint64_t> n_unpinned_touch_distinct{0};
    std::atomic<uint64_t> n_unpinned_touch_total{0};
    void mark_unpinned_touch(int layer, int expert_id) {
        n_unpinned_touch_total.fetch_add(1, std::memory_order_relaxed);
        if (layer < 0 || layer >= kMaxTrackedLayersForMiss ||
            expert_id < 0 || expert_id >= kMaxExpertsPerLayerForPinProbe) {
            return;  // out of tracked range -- undercounts distinct, never crashes
        }
        size_t idx = (size_t)layer * kMaxExpertsPerLayerForPinProbe + (size_t)expert_id;
        if (unpinned_touch_seen[idx].exchange(1, std::memory_order_relaxed) == 0) {
            n_unpinned_touch_distinct.fetch_add(1, std::memory_order_relaxed);
        }
    }

    std::atomic<uint64_t> n_active_layer_uploads_async_synced{0};
    // Pre-warm candidates skipped because they weren't yet RAM-resident
    // (find_ram_slot_if_ready returned null) -- these just fall through to
    // Phase 3's existing reactive path unchanged, never a correctness
    // issue, but a high count here means phase_b's own prefetch coverage,
    // not this batch pre-warm, is the binding constraint for those cases.
    std::atomic<uint64_t> n_active_layer_prewarm_not_ram_ready{0};
    // See ExpertLocation::was_async_prewarmed/prewarm_touched's own comment
    // (gpu_topology.hpp) for the full design. bytes_active_layer_prewarm_
    // wasted lets the report show a real GB figure, not just a call count --
    // the number that actually determines whether narrowing decode_async_
    // prewarm's own eagerness is worth the real prefetch-coverage it would
    // trade away.
    std::atomic<uint64_t> n_active_layer_prewarm_used{0};
    std::atomic<uint64_t> n_active_layer_prewarm_wasted{0};
    std::atomic<int64_t> bytes_active_layer_prewarm_wasted{0};

    // Speculative pilot pre-fill into the active layer (EngineConfig::
    // speculative_promote_to_active_layer, 2026-08-26, default off) -- see
    // ExpertLocation::pilot_speculative_fill's own comment (gpu_topology.hpp)
    // for why this is a separate counter pair from the active_layer_prewarm_
    // * trio above, despite the structurally identical used/wasted shape.
    // n_pilot_active_fill_attempted: real upload_expert_active(speculative=
    // true) calls that actually issued a DMA (i.e. the active layer pool had
    // a claimable slot). n_pilot_active_fill_used/_wasted mirror
    // n_active_layer_prewarm_used/_wasted exactly (was this slot's content
    // ever actually consulted by a real dispatch before being reclaimed) --
    // the single most direct answer to "are the extra speculatively-filled
    // experts actually used" for THIS mechanism specifically, unlike the
    // main-cache promotion path's own indirect recall-accuracy proxy.
    // n_pilot_active_fill_already_present / _already_main_cache: the caller
    // (RAMSafetyManager::promote_ram_to_active_layer) found the expert
    // already active-layer- or main-cache-resident and skipped the upload
    // entirely -- a real, harmless no-op, not a failure.
    // n_pilot_active_fill_pool_full: upload_expert_active returned nullptr
    // (the active layer's own, much cheaper, self-contained victim pool was
    // momentarily unavailable -- never touches the main VRAM cache).
    std::atomic<uint64_t> n_pilot_active_fill_attempted{0};
    // 2026-08-28: direct answer to "is speculative_active_layer_max_k
    // actually the binding constraint at this value, or has the tier/
    // threshold structure (ram_vram_promote_dynamic_*) already capped this
    // rank first" -- fires only when dynamic_layer_top_k's own per-layer
    // width already approved this rank (should_promote true) but max_k
    // clipped it anyway. Motivated directly by a user question: raising
    // max_k without also checking whether MORE experts actually got
    // promoted risks measuring noise from an already-saturated knob. Real
    // data (see this counter's own report line) showed max_k=4/5/8 all
    // produced statistically flat total promotion-attempt counts (~2900-
    // 3000, vs max_k=1's 1024) -- this counter is what confirms WHY,
    // directly, instead of inferring it from the flat attempted totals.
    std::atomic<uint64_t> n_speculative_max_k_binding_depth1{0};
    std::atomic<uint64_t> n_speculative_max_k_binding_attn{0};
    std::atomic<uint64_t> n_pilot_active_fill_used{0};
    std::atomic<uint64_t> n_pilot_active_fill_wasted{0};
    // 2026-08-28: same used/wasted split as the pair above, broken down by
    // WHICH rank (k) caused the promotion -- direct answer to "of the MoEs
    // promoted at rank K specifically, what fraction were actually used" --
    // see ExpertLocation::promoted_at_rank's own comment for why this is a
    // genuinely different question from the pilot's own per-rank
    // PREDICTION-accuracy metric (n_pilot_ram_only_hits_by_rank), which
    // measures prediction-vs-real-router agreement regardless of whether
    // that prediction was ever promoted or used.
    std::atomic<uint64_t> n_pilot_active_fill_used_by_rank[8]{};
    std::atomic<uint64_t> n_pilot_active_fill_wasted_by_rank[8]{};
    // 2026-08-28: same used/wasted split, broken down by WHICH GPU the
    // slot landed on instead of by rank -- direct answer to "does a
    // routed GPU's slice have a different real hit-rate than the
    // others'", the actual empirical check the multi-GPU routing design
    // was built to allow (not just assume the split is fine). Covers
    // decode_async_prewarm's own uploads too (was_async_prewarmed
    // branch), not just the speculative-fill one -- both branches of
    // record_active_layer_prewarm_reclaim write here.
    std::atomic<uint64_t> n_active_layer_used_by_gpu[8]{};
    std::atomic<uint64_t> n_active_layer_wasted_by_gpu[8]{};
    std::atomic<int64_t> bytes_pilot_active_fill_wasted{0};
    std::atomic<uint64_t> n_pilot_active_fill_already_present{0};
    std::atomic<uint64_t> n_pilot_active_fill_already_main_cache{0};
    std::atomic<uint64_t> n_pilot_active_fill_pool_full{0};
    std::atomic<uint64_t> n_pilot_active_fill_dedup_skipped{0};

    // Decode-time analogue of the batch pre-warm above (EngineConfig::
    // decode_async_prewarm, execution_layer.cpp's Phase-3 ram_needed_k
    // dispatch loop) -- SEPARATE counters from n_active_layer_uploads_
    // async_issued/synced above, even though both call sites share the
    // same underlying upload_expert_active_async/sync_active_layer_batch
    // functions and both still also feed those combined totals. Exists
    // specifically because the shared/combined counters can't tell "did
    // decode's own pre-warm ever engage" from "prefill's own Phase-2
    // pre-warm produced this number, decode contributed zero" -- a real
    // ambiguity hit in practice: a run with decode_async_prewarm NOT set
    // showed the exact same n_active_layer_uploads_async_issued/synced
    // total as a completely unrelated prior run without this feature at
    // all (expected, since prefill's own contribution is deterministic
    // under a frozen-usage-stats/temp=0 test setup), which made it
    // impossible to confirm from the shared counters alone whether the
    // flag had actually been left off or had simply found zero eligible
    // candidates. These counters remove that ambiguity going forward.
    std::atomic<uint64_t> n_decode_async_prewarm_issued{0};   // successful upload_expert_active_async calls from decode's own loop
    std::atomic<uint64_t> n_decode_async_prewarm_synced{0};   // == issued, mirrors n_active_layer_uploads_async_synced's victims.size() accounting
    std::atomic<uint64_t> n_decode_async_prewarm_chunks{0};   // sync_active_layer_batch calls from decode's loop (chunk count -- avg issued/chunks should be <= decode_async_prewarm_max_batch)
    std::atomic<uint64_t> n_decode_async_prewarm_no_slot{0};  // upload_expert_active_async returned nullptr (active-layer pool full/alloc failed) -- falls through to dispatch_expert_k's existing sync path, not a correctness issue
    // Should be ~0 on a healthy run -- counts the post-pin TOCTOU/identity
    // guards (see that code's own comment) actually catching a reclaimed or
    // reidentified slot. Nonzero isn't itself a bug (the guards existing
    // means it's handled safely, falling through to the legacy sync path
    // for that one candidate) but IS a real signal of how often a
    // concurrent prefetch worker is racing this exact window, worth
    // watching especially on a first real enable of this feature.
    std::atomic<uint64_t> n_decode_async_prewarm_stale_caught{0};

    // upload_expert_active's own cudaHostRegister conversion (mirrors
    // upload_expert's t_host_register_stage/n_host_register_fail/
    // bytes_host_registered, kept as SEPARATE counters rather than shared
    // ones -- this is a structurally distinct call site (active-layer pool,
    // not the main VRAM cache) and conflating the two would hide which path
    // is actually dominant for a given model/config, exactly the kind of
    // attribution gap that caused today's "why doesn't the upload_expert fix
    // help GLM" investigation in the first place -- see CLAUDE.md).
    // t_active_layer_upload itself (pre-existing) still measures only the
    // DMA call, now via cudaHostRegister+cudaMemcpyAsync+cudaHostUnregister
    // instead of a raw pageable-source cudaMemcpy.
    std::atomic<double> t_active_layer_host_register{0};      // cudaHostRegister+cudaHostUnregister time (register call only; DMA itself stays in t_active_layer_upload)
    std::atomic<uint64_t> n_active_layer_host_register_fail{0}; // fell back to direct pageable-source copy
    std::atomic<int64_t> bytes_active_layer_host_registered{0}; // bytes actually routed through a successful register
    std::atomic<uint64_t> n_active_layer_stream_unavailable{0}; // fell back to stream-0 cudaMemcpy (copy_stream missing)
    std::atomic<uint64_t> n_active_layer_upload_fail{0};       // cudaMemcpy/cudaMemcpyAsync H2D itself failed

    std::atomic<uint64_t> n_move_on_promotion{0};       // RAM→VRAM promotions that used MOVE (freed RAM slot) instead of COPY
    std::atomic<uint64_t> n_copy_on_promotion{0};       // RAM→VRAM promotions that used COPY (left RAM slot occupied)
    std::atomic<uint64_t> n_copy_reclaimed_threshold{0}; // of the COPY-mode promotions above, how many were
                                             // ADDITIONALLY reclaimed right away anyway because
                                             // ram_copy_reclaim_threshold was set and that layer's RAM
                                             // occupancy had crossed it (finish_sync_promotion) -- these
                                             // are already included in n_move_on_promotion's count (the
                                             // field-clearing is identical to MOVE), this counter exists
                                             // only to distinguish "genuinely configured MOVE" from
                                             // "COPY that got proactively reclaimed under pressure".

    // Pilot-prediction-driven speculative promotion (RAMSafetyManager::
    // promote_ram_to_vram, gated by ram_vram_promote_top_k/adaptive) --
    // DISTINCT counters from n_move_on_promotion/n_copy_on_promotion above,
    // which are shared with the reactive hit-count promotion path (both
    // funnel through the same do_promote_ram_slot_to_vram helper, so an
    // actual successful promotion from EITHER source still increments
    // those two). These two track this path's own call volume specifically:
    // how often it was even attempted (slot still RAM-resident, not mid-
    // read) vs. how often the slot was already gone by the time this ran
    // (evicted/moved between the pilot loop's is_ram_resident() check and
    // this call -- expected to be rare, since RAM-tier eviction only
    // happens on this same decode thread, but not impossible if this
    // exact expert also got promoted-and-MOVE-freed by the reactive path
    // moments earlier in the SAME layer transition).
    std::atomic<uint64_t> n_pilot_promote_attempted{0};
    std::atomic<uint64_t> n_pilot_promote_slot_gone{0};
    // 2026-08-25: found via a real, user-flagged suspicion ("9900 attempted
    // looks like double loading") that turned out correct -- DEPTH-1 and
    // ATTN-PILOT both predict the SAME target layer (l+1) from two
    // different inputs, so their top-K sets routinely overlap; each ran its
    // own promote_ram_to_vram call for the SAME (layer,expert) within the
    // SAME layer transition, back-to-back on the same thread. Because
    // GpuTopology::find_expert only reports a slot resident once its async
    // H2D copy is CONFIRMED complete (cudaEventQuery), a second call issued
    // moments after the first (DMA still in flight) sees "not resident yet"
    // and genuinely re-issues a SECOND evict_lru + upload_expert_async for
    // the exact same expert -- a real wasted VRAM slot/DMA, not just a
    // cheap no-op re-check. Counts calls skipped by the new per-transition
    // dedup (execution_layer.cpp's vram_promoted_this_transition) BEFORE
    // ever reaching promote_ram_to_vram, so a skip here means one fewer
    // call landing in n_pilot_promote_attempted above, not a separate outcome.
    std::atomic<uint64_t> n_pilot_promote_dedup_skipped{0};
};
