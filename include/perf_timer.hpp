#pragma once
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

struct PerfCounters {
    // Cumulative timings (seconds)
    std::atomic<double> t_disk_read{0};     // Expert slab pread from disk
    std::atomic<double> t_host_alloc{0};    // posix_memalign + setup
    std::atomic<double> t_gpu_upload{0};    // cudaMemcpy H2D -- shared by expert cache
                                             // PROMOTION uploads (GpuTopology::upload_expert,
                                             // whole ~12-15MB slabs) AND dense_proj's own
                                             // one-time startup dense-tensor-offload uploads.
                                             // Deliberately NOT touched by expert_ffn's own
                                             // per-call activation upload anymore -- see
                                             // t_expert_gpu_upload below for why that was split
                                             // out.
    std::atomic<double> t_expert_gpu_upload{0};  // cudaMemcpy H2D of the small activation
                                             // vector before an expert's gate/up/down kernels
                                             // (execution_layer.cpp's expert_ffn_gpu_issue) --
                                             // split OUT of t_gpu_upload after a real
                                             // investigation found "avg GPU round-trip: up=X ms/call" in
                                             // dump_report()/the [REQUEST] line's "gpu=...
                                             // [up=...]" breakdown was computing
                                             // t_gpu_upload/gcalls where t_gpu_upload ALSO
                                             // included cache-PROMOTION slab uploads (12553 of
                                             // them in one real run, ~2ms each) sharing the same
                                             // counter as expert_ffn's own much-smaller per-call
                                             // activation upload (a few KB) -- i.e. the reported
                                             // per-call "up" figure was silently dominated by an
                                             // unrelated bucket, not a genuine measurement of
                                             // expert_ffn's own GPU dispatch overhead. This is
                                             // the SAME class of attribution bug as the earlier
                                             // Attention-vs-Prefill and summed-Throughput fixes
                                             // in this project's history -- fix the counter
                                             // before trusting conclusions drawn from it.
    std::atomic<double> t_cufile_read{0};   // cuFileRead (GpuTopology::upload_expert_cufile) --
                                             // deliberately its OWN counter, not folded into
                                             // t_gpu_upload (the cudaMemcpy H2D one) -- see that
                                             // function's own comment for why conflating two
                                             // genuinely different transfer mechanisms into one
                                             // counter is exactly the mistake this project's own
                                             // disk=/expert= investigation was burned by once
                                             // already.
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
    std::atomic<double> t_expert_silu_activation{0};  // expert_ffn_gpu_finish's CPU-side
                                             // silu_inplace/elem_mul/swiglu_clamp_inplace
                                             // on the downloaded gate_buf/up_buf, between
                                             // the gate/up download and the down_proj
                                             // re-upload -- previously had ZERO timer of
                                             // its own (the surrounding comment called it
                                             // "cheap", but that was never actually
                                             // measured). Added as the next concrete
                                             // hypothesis after three prior rounds of
                                             // instrumenting GpuTopology/RAMSafetyManager
                                             // functions (t_hold_for_compute/t_release_
                                             // hold/t_find_active_expert, then t_find_
                                             // expert/t_evict_lru/t_upload_expert_async/
                                             // t_ram_get_expert/t_ram_get_experts_batch)
                                             // all came back near-zero without closing the
                                             // real gap between expert= and every prior
                                             // sub-timer -- this is a genuinely different
                                             // function (execution_layer.cpp's own GPU
                                             // dispatch path, not the cache/eviction layer)
                                             // that none of those rounds touched.
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
    // warm_mla_absorb_cache's own dedicated timer (attention_layer.cpp,
    // 2026-08-12) -- DELIBERATELY separate from t_attn_absorb above: the
    // whole point is moving this one-time, model-only dequant cost OUT of
    // the first prefill call's own timing window, so it must NOT feed the
    // same counter prefill's [PREFILL] breakdown reads from. Not part of
    // Snapshot/operator-/snapshot() (same convention as t_dense_load) --
    // this is a startup-only, once-per-process cost, not something a
    // per-request delta needs to report.
    std::atomic<double> t_mla_absorb_warmup{0};
    std::atomic<double> t_attn_score{0};    // the absorbed score/softmax/value-aggregation loop
    // Batched q_absorbed GEMM (attention_layer.cpp, 2026-08-12) -- ONE real
    // matmul per head covering all S tokens at once, moved OUT of the
    // per-(s,h) loop t_attn_score measures (was previously invisible,
    // folded into attn_score's own total as S separate per-token GEMVs).
    // Given its own counter specifically so a real run can see this
    // sub-step's cost split out from the causal score/softmax/accumulate
    // loop that still follows it. Part of Snapshot (unlike t_mla_absorb_
    // warmup above) -- this is a real per-request cost, not a one-time
    // startup cost.
    std::atomic<double> t_attn_qabsorb_batch{0};
    std::atomic<double> t_attn_outproj_dequant{0};  // dsv4_attention_gguf's
                                             // one-time-per-layer WOA
                                             // (attn_output_a) dequant_row
                                             // call, gated by WOA.empty() --
                                             // split out from t_attn_outproj
                                             // (which now measures only the
                                             // recurring per-call work) to
                                             // distinguish a one-time-43x
                                             // cost from a per-call-1720x
                                             // one after outproj came back
                                             // at 11.96s/14.67s (82% of all
                                             // attention) on a real run --
                                             // see t_attn_outproj's own
                                             // comment for the leading
                                             // hypothesis this is meant to
                                             // confirm/refute.
    std::atomic<double> t_attn_outproj{0};  // attn_output dense_proj
                                             // PLUS dsv4_attention_gguf's
                                             // recurring grouped matmul()
                                             // loop (attn_output_a per o_group
                                             // -- see t_attn_outproj_dequant's
                                             // comment). Leading hypothesis
                                             // for why outproj dominates: that
                                             // loop calls the AVX-512+OMP
                                             // matmul() kernel o_groups times
                                             // PER TOKEN PER LAYER, every one
                                             // with S=1 (a single decode
                                             // token) -- spinning up a full
                                             // OpenMP thread team for a
                                             // single tiny matmul, up to
                                             // ~1720 times across a 40-token
                                             // run, is a classic overhead-
                                             // dominates-the-work anti-
                                             // pattern. This split isolates
                                             // that loop from the one-time
                                             // dequant to test that
                                             // hypothesis directly.

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

    // MLA qabsorb BF16 batched fast-path (EngineConfig::attn_qabsorb_gpu,
    // CudaManager::qabsorb_bf16_batched_host, cuda_manager.hpp).
    // Shutdown-report-only, NOT threaded through the live per-request
    // Snapshot (matching the precedent set by upload_expert_active's own
    // separate counters, and t_active_layer_upload's -- a structurally
    // distinct call site's instrumentation shouldn't share buckets with
    // dense_proj's t_dense_gpu_* above, which would hide whether qabsorb is
    // engaging at all behind that already-large, unrelated total).
    std::atomic<double> t_qabsorb_gpu_up{0};
    std::atomic<double> t_qabsorb_gpu_compute{0};
    std::atomic<double> t_qabsorb_gpu_down{0};
    std::atomic<uint64_t> n_qabsorb_gpu_calls{0};
    // Incremented by the CALLER (attention_layer.cpp) whenever
    // qabsorb_bf16_batched_host returns false and it falls back to the CPU
    // per-head matmul() loop for that layer/call -- e.g. attn_qabsorb_gpu is
    // on but this layer's WUK BF16 upload never succeeded (VRAM budget,
    // missing weights). Should track close to 0 once warm_mla_absorb_cache
    // has finished uploading every eligible layer; a steady nonzero rate
    // across a whole run means some layer(s) never got a working GPU
    // fast-path and are silently paying the CPU cost every call.
    std::atomic<uint64_t> n_qabsorb_gpu_fallback_cpu{0};

    // MLA score GPU fast-path (EngineConfig::attn_score_gpu) -- same
    // shutdown-report-only discipline as the qabsorb counters above.
    std::atomic<double> t_score_gpu_up{0};
    std::atomic<double> t_score_gpu_compute{0};
    std::atomic<double> t_score_gpu_down{0};
    std::atomic<uint64_t> n_score_gpu_calls{0};
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

    // WUV BF16 fast-path (score GPU path's own down-projection, added
    // 2026-08-14 to replace the CPU per-head OMP-parallel loop t_score_gpu_
    // wuv above measured at 0.916ms/call -- see wuv_bf16_batched_host's own
    // comment, cuda_manager.hpp). Same shutdown-report-only discipline.
    std::atomic<double> t_wuv_gpu_up{0};
    std::atomic<double> t_wuv_gpu_compute{0};
    std::atomic<double> t_wuv_gpu_down{0};
    std::atomic<uint64_t> n_wuv_gpu_calls{0};
    // Incremented by the CALLER whenever wuv_bf16_batched_host returns
    // false (this layer's WUV BF16 upload never succeeded) and the CPU
    // per-head loop runs instead. Should track toward 0 once warm_mla_
    // absorb_cache finishes uploading every eligible layer's WUV.
    std::atomic<uint64_t> n_wuv_gpu_fallback_cpu{0};
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
    std::atomic<uint64_t> n_ram_claim_by_decode{0};        // claim_or_wait's caller was the decode thread's own miss-handling and it won the Empty->Loading claim (i.e. no prefetch had already fetched this expert)
    std::atomic<uint64_t> n_ram_claim_by_prefetch{0};      // claim_or_wait's caller was a LockFreePrefetcher worker thread and it won the Empty->Loading claim
    std::atomic<uint64_t> n_ram_claim_already_resident_prefetch{0};  // a prefetch worker's claim attempt found the expert already Ready -- prefetch structurally cannot re-fetch an already-resident expert now (this is the direct, by-construction elimination of the old "Cache-hit-beat-fetch" waste category)
    std::atomic<uint64_t> n_ram_claim_already_inflight_prefetch{0};  // a prefetch worker's claim attempt found the expert already Loading (claimed by someone else, decode or another prefetch worker) -- a no-op dedup, same intent as the old "already in flight" staging-pool check, now checking the real shared cache
    std::atomic<uint64_t> n_ram_wait_inflight_decode{0};   // the decode thread's own miss-handling found the expert already Loading (claimed by a prefetch worker) and had to wait_for_ram_slot() for it -- this is the direct, by-construction elimination of the old "Phase-A/Phase-B ordering gap": no more check-once-then-race, the decode thread waits for the exact in-flight read instead
    std::atomic<double> t_ram_wait_inflight_ms_sum{0};     // sum of wait_for_ram_slot's own wait duration across all n_ram_wait_inflight_decode events
    std::atomic<uint64_t> n_ram_prefetch_loaded_evicted_unused{0};  // a RAM-tier victim-selection site reclaimed a slot with loaded_by_prefetch==true and hit_count==0 -- prefetch loaded it and normal LRU eviction reclaimed it before any real hit ever touched it. The unified design's replacement for the old staging-pool-based "wasted prefetch" measurement, now grounded in the real cache's real LRU pressure instead of an artificial round-robin pool.

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
    std::atomic<uint64_t> n_expert_gpu_calls{0};  // expert_ffn calls that took the GPU-resident path
    std::atomic<uint64_t> n_expert_cpu_calls{0};  // expert_ffn calls that took the CPU fallback path
    std::atomic<int64_t> bytes_disk_read{0};
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
    // worth building at all" instead of the aggregate .moestream_usage
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
    std::atomic<uint64_t> n_active_layer_hits{0};      // cache hits served from active layer area
    std::atomic<uint64_t> n_active_layer_cleanup{0};    // active layer slots reclaimed after layer completion
    std::atomic<uint64_t> n_active_layer_promoted{0};   // active-layer experts promoted into the real VRAM LRU cache (hit_count crossed active_layer_promote_threshold), vs. just freed at cleanup

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
    std::atomic<uint64_t> n_active_layer_uploads_async_issued{0};
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

    // upload_expert()'s two silent failure modes -- both previously
    // returned nullptr with ZERO logging, making a real run's "why did
    // preload/promotion fail" undiagnosable. evict_lru returning no victim
    // at all (should be near-impossible on a startup-empty cache -- every
    // slot would have to be pinned/temp-pinned/upload-pending
    // simultaneously) vs the cudaMemcpy H2D itself failing (a genuine CUDA
    // error, logged with its own cudaGetErrorString()) are two very
    // different root causes and need to be told apart.
    std::atomic<uint64_t> n_upload_expert_no_victim{0};
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
    // (100+ entries/layer in .moestream_usage) -- meaning the candidate
    // list (`ranked`) itself was empty every single call, not just short
    // of free slots. These two counters distinguish the two ways that can
    // happen: no_ranking_for_layer fires if route_rank_by_layer had NO
    // entry at all for that layer (population bug, or a layer-numbering
    // mismatch between .moestream_usage and get_experts_batch's own
    // `layer` parameter); no_candidates_after_filter fires if the ranking
    // DID exist but every single entry got excluded (this-token's-own-set
    // / RAM-resident / VRAM-resident) before `want` candidates were found.
    std::atomic<uint64_t> n_pad_no_ranking_for_layer{0};
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

    // Plain-value point-in-time copy of the cumulative counters above, so
    // callers can diff two snapshots to get a per-token or per-request
    // breakdown without disturbing the global running totals.
    struct Snapshot {
        double t_disk_read=0, t_host_alloc=0, t_gpu_upload=0, t_expert_gpu_upload=0, t_gpu_download=0, t_gpu_compute=0, t_expert_silu_activation=0, t_cpu_expert=0;
        double t_cpu_dequant=0, t_cpu_dot=0;
        double t_attn=0;
        double t_attn_preamble=0;
        double t_attn_norm=0, t_attn_qproj=0, t_attn_kvproj=0, t_attn_absorb=0, t_attn_score=0, t_attn_outproj=0;
        double t_attn_outproj_dequant=0;
        double t_dense_gpu_up=0, t_dense_gpu_compute=0, t_dense_gpu_down=0;
        double t_moe_route=0, t_moe_expert=0, t_prefetch=0, t_dense_load=0, t_consume_staged=0;
        double t_staging_regrow=0, t_staging_mutex_wait=0, t_cache_mutex_wait=0;
        double t_active_layer_upload=0, t_active_layer_promote_copy=0, t_vram_demote_copy=0;
        double t_hold_for_compute=0, t_release_hold=0, t_find_active_expert=0;
        double t_find_expert=0, t_evict_lru=0, t_upload_expert_async=0;
        double t_ram_get_expert=0, t_ram_get_experts_batch=0;
        double t_ram_batch_phase_a=0, t_ram_batch_phase_b=0, t_ram_batch_phase_c=0;
        double t_bounce_stage_memcpy=0, t_host_register_stage=0, t_bounce_buffer_wait=0;
        uint64_t n_expert_cache_hit=0, n_expert_cache_miss=0, n_expert_promote=0, n_tokens_generated=0;
        uint64_t n_expert_gpu_calls=0, n_expert_cpu_calls=0, n_dense_gpu_calls=0;
        int64_t bytes_disk_read=0, bytes_gpu_uploaded=0, bytes_bounce_staged=0, bytes_host_registered=0;
        uint64_t n_prefetch_hits=0, n_pinned_eviction_skips=0, n_pinned_resident=0, n_staging_regrow=0;
        // Added to answer "is phase_b's remaining prefill cost a coverage
        // gap (never hinted) or a worker-capacity gap (hinted, but no
        // prefetch worker got to it before the decode thread did)" -- see
        // claim_or_wait's own comment for what Hit/by_decode/by_prefetch
        // mean. Scoped via the SAME Snapshot diff as everything else here
        // (e.g. via the new [PREFILL] line, generation_loop.hpp), not just
        // the whole-program-lifetime dump_report() totals these already
        // fed before this addition.
        uint64_t n_ram_claim_hit=0, n_ram_claim_by_decode=0, n_ram_claim_by_prefetch=0;
        uint64_t n_ram_wait_inflight_decode=0;
        double t_ram_wait_inflight_ms_sum=0;
        double t_attn_qabsorb_batch=0;
        double t_moe_topk_select=0, t_moe_phase3_dispatch=0;
        double t_resolve_gpu_loc_total=0, t_gpu_issue_total=0, t_gpu_finish_total=0;
        double t_accumulate_total=0, t_shared_expert_total=0, t_layer_cleanup_total=0;
        // Fourth round, same investigation: these two ALREADY EXISTED
        // (feeding the shutdown-only "Active-layer BATCH pre-warm ...
        // issue=Xs sync=Ys" line) but were never threaded into the
        // per-request Snapshot/[REQUEST] line at all -- a real, live shot
        // shutdown-report figure (issue=18.308s on one real run) was
        // invisible in every [REQUEST] line to date despite explaining most
        // of the ~15-18s phase3_loop residual left after ruling out
        // resolve_gpu_loc/gpu_issue/gpu_finish/accumulate/shared_expert/
        // layer_cleanup. t_active_layer_upload_async's own comment expected
        // this to be "small/cheap, no wait" -- worth confirming or
        // disproving directly against the real per-request number, not just
        // the whole-session total.
        double t_active_layer_upload_async=0, t_active_layer_batch_sync=0;

        Snapshot operator-(const Snapshot& b) const {
            return Snapshot{
                t_disk_read - b.t_disk_read, t_host_alloc - b.t_host_alloc,
                t_gpu_upload - b.t_gpu_upload, t_expert_gpu_upload - b.t_expert_gpu_upload,
                t_gpu_download - b.t_gpu_download,
                t_gpu_compute - b.t_gpu_compute,
                t_expert_silu_activation - b.t_expert_silu_activation,
                t_cpu_expert - b.t_cpu_expert,
                t_cpu_dequant - b.t_cpu_dequant, t_cpu_dot - b.t_cpu_dot,
                t_attn - b.t_attn,
                t_attn_preamble - b.t_attn_preamble,
                t_attn_norm - b.t_attn_norm, t_attn_qproj - b.t_attn_qproj,
                t_attn_kvproj - b.t_attn_kvproj, t_attn_absorb - b.t_attn_absorb,
                t_attn_score - b.t_attn_score, t_attn_outproj - b.t_attn_outproj,
                t_attn_outproj_dequant - b.t_attn_outproj_dequant,
                t_dense_gpu_up - b.t_dense_gpu_up, t_dense_gpu_compute - b.t_dense_gpu_compute,
                t_dense_gpu_down - b.t_dense_gpu_down,
                t_moe_route - b.t_moe_route,
                t_moe_expert - b.t_moe_expert, t_prefetch - b.t_prefetch,
                t_dense_load - b.t_dense_load, t_consume_staged - b.t_consume_staged,
                t_staging_regrow - b.t_staging_regrow,
                t_staging_mutex_wait - b.t_staging_mutex_wait,
                t_cache_mutex_wait - b.t_cache_mutex_wait,
                t_active_layer_upload - b.t_active_layer_upload,
                t_active_layer_promote_copy - b.t_active_layer_promote_copy,
                t_vram_demote_copy - b.t_vram_demote_copy,
                t_hold_for_compute - b.t_hold_for_compute,
                t_release_hold - b.t_release_hold,
                t_find_active_expert - b.t_find_active_expert,
                t_find_expert - b.t_find_expert,
                t_evict_lru - b.t_evict_lru,
                t_upload_expert_async - b.t_upload_expert_async,
                t_ram_get_expert - b.t_ram_get_expert,
                t_ram_get_experts_batch - b.t_ram_get_experts_batch,
                t_ram_batch_phase_a - b.t_ram_batch_phase_a,
                t_ram_batch_phase_b - b.t_ram_batch_phase_b,
                t_ram_batch_phase_c - b.t_ram_batch_phase_c,
                t_bounce_stage_memcpy - b.t_bounce_stage_memcpy,
                t_host_register_stage - b.t_host_register_stage,
                t_bounce_buffer_wait - b.t_bounce_buffer_wait,
                    n_expert_cache_hit - b.n_expert_cache_hit,
                    n_expert_cache_miss - b.n_expert_cache_miss,
                    n_expert_promote - b.n_expert_promote,
                    n_tokens_generated - b.n_tokens_generated,
                    n_expert_gpu_calls - b.n_expert_gpu_calls,
                    n_expert_cpu_calls - b.n_expert_cpu_calls,
                    n_dense_gpu_calls - b.n_dense_gpu_calls,
                    bytes_disk_read - b.bytes_disk_read,
                    bytes_gpu_uploaded - b.bytes_gpu_uploaded,
                    bytes_bounce_staged - b.bytes_bounce_staged,
                    bytes_host_registered - b.bytes_host_registered,
                    n_prefetch_hits - b.n_prefetch_hits,
                    n_pinned_eviction_skips - b.n_pinned_eviction_skips,
                    n_pinned_resident - b.n_pinned_resident,
                    n_staging_regrow - b.n_staging_regrow,
                    n_ram_claim_hit - b.n_ram_claim_hit,
                    n_ram_claim_by_decode - b.n_ram_claim_by_decode,
                    n_ram_claim_by_prefetch - b.n_ram_claim_by_prefetch,
                    n_ram_wait_inflight_decode - b.n_ram_wait_inflight_decode,
                    t_ram_wait_inflight_ms_sum - b.t_ram_wait_inflight_ms_sum,
                    t_attn_qabsorb_batch - b.t_attn_qabsorb_batch,
                    t_moe_topk_select - b.t_moe_topk_select,
                    t_moe_phase3_dispatch - b.t_moe_phase3_dispatch,
                    t_resolve_gpu_loc_total - b.t_resolve_gpu_loc_total,
                    t_gpu_issue_total - b.t_gpu_issue_total,
                    t_gpu_finish_total - b.t_gpu_finish_total,
                    t_accumulate_total - b.t_accumulate_total,
                    t_shared_expert_total - b.t_shared_expert_total,
                    t_layer_cleanup_total - b.t_layer_cleanup_total,
                    t_active_layer_upload_async - b.t_active_layer_upload_async,
                    t_active_layer_batch_sync - b.t_active_layer_batch_sync
            };
        }
    };

    Snapshot snapshot() const {
        return Snapshot{
            t_disk_read.load(), t_host_alloc.load(), t_gpu_upload.load(), t_expert_gpu_upload.load(),
            t_gpu_download.load(),
            t_gpu_compute.load(),
            t_expert_silu_activation.load(),
            t_cpu_expert.load(),
            t_cpu_dequant.load(), t_cpu_dot.load(),
            t_attn.load(),
            t_attn_preamble.load(),
            t_attn_norm.load(), t_attn_qproj.load(), t_attn_kvproj.load(),
            t_attn_absorb.load(), t_attn_score.load(), t_attn_outproj.load(),
            t_attn_outproj_dequant.load(),
            t_dense_gpu_up.load(), t_dense_gpu_compute.load(), t_dense_gpu_down.load(),
            t_moe_route.load(), t_moe_expert.load(), t_prefetch.load(), t_dense_load.load(), t_consume_staged.load(),
            t_staging_regrow.load(), t_staging_mutex_wait.load(), t_cache_mutex_wait.load(),
            t_active_layer_upload.load(), t_active_layer_promote_copy.load(), t_vram_demote_copy.load(),
            t_hold_for_compute.load(), t_release_hold.load(), t_find_active_expert.load(),
            t_find_expert.load(), t_evict_lru.load(), t_upload_expert_async.load(),
            t_ram_get_expert.load(), t_ram_get_experts_batch.load(),
            t_ram_batch_phase_a.load(), t_ram_batch_phase_b.load(), t_ram_batch_phase_c.load(),
            t_bounce_stage_memcpy.load(), t_host_register_stage.load(), t_bounce_buffer_wait.load(),
            n_expert_cache_hit.load(), n_expert_cache_miss.load(), n_expert_promote.load(), n_tokens_generated.load(),
            n_expert_gpu_calls.load(), n_expert_cpu_calls.load(), n_dense_gpu_calls.load(),
            bytes_disk_read.load(), bytes_gpu_uploaded.load(), bytes_bounce_staged.load(), bytes_host_registered.load(),
            n_prefetch_hits.load(), n_pinned_eviction_skips.load(), n_pinned_resident.load(),
            n_staging_regrow.load(),
            n_ram_claim_hit.load(), n_ram_claim_by_decode.load(), n_ram_claim_by_prefetch.load(),
            n_ram_wait_inflight_decode.load(), t_ram_wait_inflight_ms_sum.load(),
            t_attn_qabsorb_batch.load(),
            t_moe_topk_select.load(), t_moe_phase3_dispatch.load(),
            t_resolve_gpu_loc_total.load(), t_gpu_issue_total.load(), t_gpu_finish_total.load(),
            t_accumulate_total.load(), t_shared_expert_total.load(), t_layer_cleanup_total.load(),
            t_active_layer_upload_async.load(), t_active_layer_batch_sync.load()
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
            (unsigned long long)d.n_prefetch_hits, (unsigned long long)d.n_pinned_eviction_skips,
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
        // t_expert_gpu_upload (NOT t_gpu_upload -- see that counter's own
        // comment) so this "gpu=... [up=...]" breakdown reflects
        // expert_ffn's own per-call GPU round trip, not cache-promotion
        // slab uploads bleeding in from a shared bucket.
        double gpu_round_trip = d.t_expert_gpu_upload + d.t_gpu_compute + d.t_gpu_download;
        // Diagnostic addition (see t_active_layer_upload's own comment): these
        // three previously had ZERO instrumentation, and were the leading
        // hypothesis for why expert= vastly exceeded gpu=+cpu= in every prior
        // [REQUEST] line. lock_wait (t_cache_mutex_wait) is separate from
        // "untimed" -- it's SEPARATELY instrumented already, just never
        // printed here before; surfaced alongside so a large "untimed" total
        // caused by lock contention (as opposed to genuinely necessary copy
        // time) is visible in the SAME line instead of requiring a
        // shutdown-only dump_report() correlation.
        double untimed_total = d.t_active_layer_upload + d.t_active_layer_promote_copy + d.t_vram_demote_copy;
        // hold_for_compute/release_hold/find_active_expert's own work-under-
        // lock cost -- see t_hold_for_compute's own comment in PerfCounters
        // for why this was added (TAKE 2's fix to the first two produced no
        // measurable change in expert=/untimed=, so rather than guess again,
        // these are now measured directly instead of inferred from the
        // residual gap).
        double scan_total = d.t_hold_for_compute + d.t_release_hold + d.t_find_active_expert;
        // t_gpu_upload (NOT t_expert_gpu_upload, which is already in
        // gpu_round_trip above) -- the SYNCHRONOUS cache-promotion H2D slab
        // upload inside upload_expert() (gpu_topology.hpp), called from the
        // decode thread's cold-miss GPU dispatch path
        // (execution_layer.cpp's expert_ffn_resolve_gpu_loc, both the S>=4
        // cold-batch branch and the decode-time active-layer-failed
        // fallback branch). Was already being accumulated correctly but had
        // NEVER been printed anywhere except the per-token DEBUG line
        // (print_delta) and the shutdown-only dump_report() -- invisible in
        // every [REQUEST] line to date. Added as the next concrete
        // hypothesis after scan= (hold_for_compute/release_hold/
        // find_active_expert) was directly measured and found negligible
        // (0.65s out of a 104.92s expert= total) -- unlike that dead end,
        // this is a real, substantial cudaMemcpy (whole ~12-15MB expert
        // slabs, not small activation vectors) on a call path confirmed
        // to run on the decode thread, and it was never even a candidate
        // in prior investigation because it was invisible in the one line
        // being compared against gpu=+cpu=.
        double promote_up = d.t_gpu_upload;
        // Companion to promote_up: the CPU-side bounce-buffer staging memcpy,
        // now upload_expert_async-only (see t_bounce_stage_memcpy's own
        // comment) -- printed alongside promote_up so a real run can
        // directly compare "DMA time" vs "CPU staging time" instead of only
        // seeing their sum attributed to one bucket.
        double stage_cpy = d.t_bounce_stage_memcpy;
        // upload_expert's own replacement mechanism (see
        // t_host_register_stage's own comment) -- the number that should
        // now dominate promote_up's companion cost on the sync decode-hot
        // path, in place of what stage_cpy used to measure there.
        double reg_stage = d.t_host_register_stage;
        // Direct test of the "was this the long-mysterious untimed gap"
        // hypothesis -- see t_bounce_buffer_wait's own comment. Now
        // upload_expert_async-only; should read small if removing
        // upload_expert's competition for the same 4-slot pool was really
        // what closed the gap.
        double bb_wait = d.t_bounce_buffer_wait;
        // Second round: find_expert/evict_lru/upload_expert_async
        // (GpuTopology) and get_expert/get_experts_batch (RAMSafetyManager)
        // were the last remaining decode-time-hot-path functions with no
        // timer of their own -- see t_find_expert's comment in PerfCounters
        // for the full reasoning. Deliberately NOT mutually exclusive with
        // each other (upload_expert_async calls find_expert+evict_lru
        // internally, so their scopes nest/overlap) -- same intentional
        // pattern as t_prefill_total vs t_attn elsewhere in this file; this
        // sum is a diagnostic upper bound on "RAM/VRAM lookup+eviction
        // machinery", not a clean partition.
        double ram_scan_total = d.t_find_expert + d.t_evict_lru + d.t_upload_expert_async
                               + d.t_ram_get_expert + d.t_ram_get_experts_batch;
        uint64_t hits = d.n_expert_cache_hit, misses = d.n_expert_cache_miss;
        uint64_t gcalls = d.n_expert_gpu_calls, ccalls = d.n_expert_cpu_calls;
        double tok_s = wall_ms > 0 ? n_tokens / (wall_ms / 1000.0) : 0.0;
        double dense_gpu_total = d.t_dense_gpu_up + d.t_dense_gpu_compute + d.t_dense_gpu_down;
        LOG_INFO(
            "[%s] %d tokens in %.2fs (%.2f tok/s)  attn=%.2fs [preamble=%.2f norm=%.2f qproj=%.2f kvproj=%.2f "
            "absorb=%.2f qabsorb=%.2f score=%.2f outproj=%.2f [outproj_dequant=%.2f]] route=%.2fs "
            "expert=%.2fs (topk_select=%.2fs gpu=%.2fs [up=%.2fs cmp=%.2fs dn=%.2fs] silu=%.2fs cpu=%.2fs [dequant=%.2fs dot=%.2fs] "
            "untimed=%.2fs [active_up=%.2fs promote_copy=%.2fs vram_demote=%.2fs] "
            "scan=%.2fs [hold=%.2fs release=%.2fs find_active=%.2fs] promote_up=%.2fs stage_cpy=%.2fs reg_stage=%.2fs bb_wait=%.2fs "
            "ram_scan=%.2fs [find_expert=%.2fs evict_lru=%.2fs upload_async=%.2fs get_expert=%.2fs "
            "get_batch=%.2fs [phase_a=%.2fs phase_b=%.2fs phase_c=%.2fs]] "
            "lock_wait=%.2fs phase3_loop=%.2fs(upper-bound,overlaps gpu/cpu/ram_scan) "
            "[resolve_loc=%.2fs(upper-bound,overlaps scan) gpu_issue=%.2fs(upper-bound,overlaps gpu_up) "
            "gpu_finish=%.2fs(upper-bound,overlaps gpu_cmp+gpu_dn) accumulate=%.2fs "
            "shared_expert=%.2fs(overlaps dense_gpu) layer_cleanup=%.2fs "
            "active_layer_prewarm_issue=%.2fs active_layer_prewarm_sync=%.2fs]) "
            "dense_gpu=%.2fs [up=%.2fs cmp=%.2fs dn=%.2fs calls=%llu] "
            "disk=%.2fs  cache=%.0f%% hit  expert_calls=%lluGPU/%lluCPU "
            "pipe=%llu(pin_skip=%llu pin=%llu)\n",
            label, n_tokens, wall_ms / 1000.0, tok_s,
            d.t_attn,
            d.t_attn_preamble,
            d.t_attn_norm, d.t_attn_qproj, d.t_attn_kvproj, d.t_attn_absorb, d.t_attn_qabsorb_batch,
            d.t_attn_score, d.t_attn_outproj,
            d.t_attn_outproj_dequant,
            d.t_moe_route, d.t_moe_expert,
            d.t_moe_topk_select,
            gpu_round_trip, d.t_expert_gpu_upload, d.t_gpu_compute, d.t_gpu_download,
            d.t_expert_silu_activation,
            d.t_cpu_expert,
            d.t_cpu_dequant, d.t_cpu_dot,
            untimed_total, d.t_active_layer_upload, d.t_active_layer_promote_copy, d.t_vram_demote_copy,
            scan_total, d.t_hold_for_compute, d.t_release_hold, d.t_find_active_expert, promote_up, stage_cpy, reg_stage, bb_wait,
            ram_scan_total, d.t_find_expert, d.t_evict_lru, d.t_upload_expert_async,
            d.t_ram_get_expert, d.t_ram_get_experts_batch,
            d.t_ram_batch_phase_a, d.t_ram_batch_phase_b, d.t_ram_batch_phase_c,
            d.t_cache_mutex_wait, d.t_moe_phase3_dispatch,
            d.t_resolve_gpu_loc_total, d.t_gpu_issue_total, d.t_gpu_finish_total,
            d.t_accumulate_total, d.t_shared_expert_total, d.t_layer_cleanup_total,
            d.t_active_layer_upload_async, d.t_active_layer_batch_sync,
            dense_gpu_total, d.t_dense_gpu_up, d.t_dense_gpu_compute, d.t_dense_gpu_down,
            (unsigned long long)d.n_dense_gpu_calls,
            d.t_disk_read,
            (hits + misses) > 0 ? 100.0 * hits / (hits + misses) : 0.0,
            (unsigned long long)gcalls, (unsigned long long)ccalls,
            (unsigned long long)d.n_prefetch_hits, (unsigned long long)d.n_pinned_eviction_skips,
            (unsigned long long)d.n_pinned_resident);
    }

    // session_elapsed_s: REAL wall-clock seconds from "engine ready to
    // accept prompts" to shutdown (see main.cpp's call site) -- used ONLY
    // for the final Throughput line. Deliberately NOT derived from summing
    // individual perf counters (t_disk_read + t_gpu_upload + ... etc.):
    // that was this function's ORIGINAL approach, and it quietly became
    // wrong as concurrency was added to this codebase (4-thread parallel
    // expert fetch, GPU/CPU Pass A/B overlap, async prefetch all
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
        // t_expert_gpu_upload, NOT t_gpu_upload -- see t_expert_gpu_upload's
        // own comment for the real conflation bug this split fixes (cache-
        // promotion slab uploads were bleeding into this "expert_ffn only"
        // total, inflating the reported per-call GPU upload figure).
        double gpu_round_trip = t_expert_gpu_upload + t_gpu_compute + t_gpu_download;
        LOG_INFO("\n========== MoEStream Performance Report ==========\n");
        LOG_INFO("Dense load (init):   %8.3f s\n", t_dense_load.load());
        LOG_INFO("Disk read (expert):  %8.3f s  (%6.2f GB/s, %.2f GB total)\n",
                 t_disk_read.load(),
                 t_disk_read > 0 ? (bytes_disk_read / 1e9) / t_disk_read : 0.0,
                 bytes_disk_read / 1e9);
        LOG_INFO("Host alloc/setup:    %8.3f s\n", t_host_alloc.load());
        LOG_INFO("GPU upload (H2D):    %8.3f s  (%6.2f GB/s, %.2f GB total) -- cache PROMOTION "
                 "slab uploads only (RAM->VRAM); expert_ffn's own small per-call activation "
                 "upload is now tracked separately, see 'GPU round-trip total' below\n",
                 t_gpu_upload.load(),
                 t_gpu_upload > 0 ? (bytes_gpu_uploaded / 1e9) / t_gpu_upload : 0.0,
                 bytes_gpu_uploaded / 1e9);
        LOG_INFO("Bounce stage memcpy: %8.3f s  (%6.2f GB/s, %.2f GB total) -- CPU-side memcpy into "
                 "the pinned bounce buffer, BEFORE the DMA above -- upload_expert_async's FALLBACK path "
                 "only now (2026-08-13: its common-case path converted to the same cudaHostRegister "
                 "approach as upload_expert, see 'Upload-async host register' below; this line should "
                 "collapse toward ~0 on a healthy run, nonzero mainly from cudaHostRegister failures). "
                 "Compare this against 'GPU upload (H2D)' above: if this still dominates, the register "
                 "conversion isn't engaging for some reason worth checking (see that line's own "
                 "failure count)\n",
                 t_bounce_stage_memcpy.load(),
                 t_bounce_stage_memcpy > 0 ? (bytes_bounce_staged / 1e9) / t_bounce_stage_memcpy : 0.0,
                 bytes_bounce_staged / 1e9);
        LOG_INFO("Host register stage: %8.3f s  (%6.2f GB/s, %.2f GB total) -- upload_expert's "
                 "replacement for the bounce-buffer memcpy above: transient cudaHostRegister/"
                 "cudaHostUnregister of the RAM tier's own slab_buffer bytes directly (Alt A of the "
                 "cudaHostRegister investigation) -- %llu registration failure(s) (should be ~0; "
                 "falls back to the direct pageable-source copy on failure, same as bounce-pool "
                 "exhaustion previously did)\n",
                 t_host_register_stage.load(),
                 t_host_register_stage > 0 ? (bytes_host_registered / 1e9) / t_host_register_stage : 0.0,
                 bytes_host_registered / 1e9,
                 (unsigned long long)n_host_register_fail.load());
        LOG_INFO("Upload-async host register: %8.3f s register + %8.3f s deferred-unregister  "
                 "(%6.2f GB/s, %.2f GB total, %llu reaped) -- upload_expert_async's OWN cudaHostRegister "
                 "conversion (2026-08-13), same Alt-A mechanism as 'Host register stage' above but a "
                 "SEPARATE call site: the register call is synchronous (paid here, before the async DMA "
                 "is issued), but the matching cudaHostUnregister can't run until the DMA -- still in "
                 "flight when upload_expert_async returns -- actually completes, so it's deferred and "
                 "reaped opportunistically (GpuTopology::pending_host_unregisters); this replaces the "
                 "'Bounce stage memcpy' line above for upload_expert_async's common case, not just adds "
                 "to it -- %llu registration failure(s) (should be ~0; falls back to the old bounce-"
                 "buffer staging on failure)\n",
                 t_upload_async_host_register.load(), t_upload_async_host_unregister.load(),
                 t_upload_async_host_register > 0 ? (bytes_upload_async_host_registered / 1e9) / t_upload_async_host_register : 0.0,
                 bytes_upload_async_host_registered / 1e9,
                 (unsigned long long)n_upload_async_deferred_unregister_reaped.load(),
                 (unsigned long long)n_upload_async_host_register_fail.load());
        if (n_ram_slab_pinned.load() > 0 || n_ram_slab_pin_fail.load() > 0
            || n_ram_slab_pin_over_budget.load() > 0) {
            LOG_INFO("RAM slab persistent pin: %8.3f s to pin %llu slab(s) / %.2f GB "
                     "(%.2f GB/s) -- ONE-TIME per allocation (EngineConfig::ram_pinned_budget_gb), "
                     "not per transfer: lets upload_expert_active_async's cudaMemcpyAsync run "
                     "genuinely async (overlapping compute) instead of the driver staging it "
                     "synchronously through its own internal bounce buffer. %llu slab(s) stayed "
                     "pageable (budget exhausted -- the direct measure of what fraction of the RAM "
                     "tier this budget covers), %llu registration failure(s) (should be ~0), %llu "
                     "unregister+repin(s) from a slab regrow (should be ~0 after warmup)\n",
                     t_ram_slab_pin.load(), (unsigned long long)n_ram_slab_pinned.load(),
                     bytes_ram_slab_pinned.load() / 1e9,
                     t_ram_slab_pin > 0 ? (bytes_ram_slab_pinned / 1e9) / t_ram_slab_pin : 0.0,
                     (unsigned long long)n_ram_slab_pin_over_budget.load(),
                     (unsigned long long)n_ram_slab_pin_fail.load(),
                     (unsigned long long)n_ram_slab_unpinned_on_regrow.load());
        }
        {
            uint64_t un = n_uring_direct_unstaged.load(), st = n_uring_direct_staged.load();
            if (un + st > 0) {
                LOG_INFO("io_uring direct staging: %llu unstaged / %llu staged (%.1f%% read straight into "
                         "the caller's buffer) -- %.2f GB of scratch->dst memcpy avoided. Staging is only "
                         "needed when a request's offset/length/destination aren't all alignment-clean; "
                         "for this model's expert reads they always are, so a high unstaged share is the "
                         "expected healthy state (a low one means something upstream lost its alignment).\n",
                         (unsigned long long)un, (unsigned long long)st,
                         (un + st) > 0 ? 100.0 * un / (un + st) : 0.0,
                         bytes_uring_stage_memcpy_saved.load() / 1e9);
            }
        }
        LOG_INFO("Promotions (H2D):    %lu uploads / %.2f GB total\n",
            (unsigned long)n_expert_promote.load(),
            bytes_gpu_uploaded.load() / 1e9);
        LOG_INFO("Expert activation upload (H2D): %8.3f s  (expert_ffn's own per-call upload, "
                 "isolated from cache-promotion uploads above -- %llu calls, %.4f ms/call avg)\n",
                 t_expert_gpu_upload.load(), (unsigned long long)n_expert_gpu_calls.load(),
                 n_expert_gpu_calls > 0 ? 1000.0 * t_expert_gpu_upload / n_expert_gpu_calls : 0.0);
        LOG_INFO("GPU compute:         %8.3f s\n", t_gpu_compute.load());
        LOG_INFO("GPU download (D2H):  %8.3f s\n", t_gpu_download.load());
        LOG_INFO("GPU round-trip total:%8.3f s  (upload+compute+download, expert_ffn only)\n", gpu_round_trip);
        LOG_INFO("Expert SiLU/mul/clamp (CPU): %8.3f s  (silu_inplace+elem_mul+swiglu_clamp_inplace on the "
                 "downloaded gate/up buffers, between GPU download and the down_proj re-upload -- %llu calls, "
                 "%.4f ms/call avg; previously untimed, see t_expert_silu_activation's own comment)\n",
                 t_expert_silu_activation.load(), (unsigned long long)n_expert_gpu_calls.load(),
                 n_expert_gpu_calls > 0 ? 1000.0 * t_expert_silu_activation / n_expert_gpu_calls : 0.0);
        LOG_INFO("CPU expert (fallback):%7.3f s  (dequant=%.3fs dot=%.3fs)\n",
                 t_cpu_expert.load(), t_cpu_dequant.load(), t_cpu_dot.load());
        LOG_INFO("Attention (MLA):     %8.3f s  (preamble=%.3f norm=%.3f qproj=%.3f kvproj=%.3f absorb=%.3f score=%.3f "
                 "outproj=%.3f [outproj_dequant=%.3f])\n",
                 t_attn.load(), t_attn_preamble.load(), t_attn_norm.load(), t_attn_qproj.load(), t_attn_kvproj.load(),
                 t_attn_absorb.load(), t_attn_score.load(), t_attn_outproj.load(), t_attn_outproj_dequant.load());
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
                     1000.0 * t_expert_gpu_upload / gcalls, 1000.0 * t_gpu_compute / gcalls,
                     1000.0 * t_gpu_download / gcalls);
        if (ccalls > 0)
            LOG_INFO("  avg CPU expert:     %6.3f ms/call\n", 1000.0 * t_cpu_expert / ccalls);
        uint64_t hits = n_expert_cache_hit, misses = n_expert_cache_miss;
        LOG_INFO("Expert cache:        %lu hit / %lu miss (%.1f%% hit rate)\n",
                 (unsigned long)hits, (unsigned long)misses,
                 (hits + misses) > 0 ? 100.0 * hits / (hits + misses) : 0.0);
        // REMOVED: "Prefetch" report block used to live here
        // (n_prefetch_hits/n_prefetch_miss_*/n_prefetch_hits_prefill/decode/
        // t_prefetch_not_ready_elapsed_ms_sum/t_prefetch_still_queued_elapsed_ms_sum)
        // -- all measured the old separate staging pool's own consume/miss
        // bookkeeping, which no longer exists under the unified claim/
        // publish design (RAMSafetyManager::claim_or_wait, ram_manager.hpp).
        // See "Unified RAM claim"/"Unified RAM wait" further down for the
        // replacement instrumentation. n_prefetch_staged_later_promoted is still
        // real (see MoEExpertBlock::loaded_by_prefetch's own comment) but no
        // longer has a meaningful "of how many prefetch hits" denominator to
        // report a percentage against, so it's not printed as its own line
        // here anymore -- read it directly from a raw counter dump if
        // needed.
        LOG_INFO("Pilot hints skipped: %lu (already host-cache-resident -- avoided wasting prefetch bandwidth)\n",
                 (unsigned long)n_pilot_hints_skipped.load());
        {
            uint64_t pw_att = n_prefill_warm_attempted.load();
            uint64_t pw_ok = n_prefill_warm_succeeded.load();
            uint64_t pw_fail = n_prefill_warm_failed.load();
            uint64_t pw_hint = n_prefill_hint_pushed.load();
            if (pw_att || pw_hint) {
                LOG_INFO("Prefill warm-up:     %lu attempted / %lu succeeded / %lu failed, %.3f s "
                         "(Phase 2's cold-start VRAM fill -- synchronous path; only fires when "
                         "prefill_warm_via_prefetcher=false) / %lu hint(s) pushed to the prefetcher "
                         "instead (prefill_warm_via_prefetcher=true -- served/miss visible via the "
                         "Pilot recall/prefetch lines above, since these share the same queue)\n",
                         (unsigned long)pw_att, (unsigned long)pw_ok, (unsigned long)pw_fail,
                         t_prefill_warm_time.load(), (unsigned long)pw_hint);
            }
            uint64_t pp_promo = n_prefill_pending_promoted.load();
            uint64_t pp_load = n_prefill_pending_still_loading.load();
            uint64_t pp_age = n_prefill_pending_aged_out.load();
            uint64_t pp_slot = n_prefill_pending_no_free_slot.load();
            uint64_t pp_res = n_prefill_pending_already_resident.load();
            uint64_t pp_cap = n_prefill_pending_cap_dropped.load();
            if (pp_promo || pp_load || pp_age || pp_slot || pp_res || pp_cap) {
                LOG_INFO("Prefill opportunistic VRAM promotion: %lu promoted / %lu already resident by the "
                         "time checked (something else, e.g. Phase 3's own dispatch, beat this pass to it) "
                         "/ %lu still loading (checked again next layer) / %lu aged out (never landed in "
                         "time, Phase 3's own synchronous dispatch handled it as usual) / %lu found no free "
                         "VRAM slot left / %lu never tracked at all (pending_prefill_promotions' 32-entry "
                         "cap was already full at push time -- high here means the cap, not fetch speed, "
                         "is the binding constraint) -- cross-layer follow-up for prefill_warm_via_"
                         "prefetcher's own pushed hints, not a speculative bet (every entry here is a "
                         "confirmed real router pick for this exact prefill batch). Should sum to "
                         "~n_prefill_hint_pushed above.\n",
                         (unsigned long)pp_promo, (unsigned long)pp_res, (unsigned long)pp_load,
                         (unsigned long)pp_age, (unsigned long)pp_slot, (unsigned long)pp_cap);
            }
        }
        {
            uint64_t rhits = n_pilot_recall_hits.load(), rchecks = n_pilot_recall_checks.load();
            uint64_t rdenom = n_pilot_recall_topk_sum.load();
            if (rchecks > 0) {
                LOG_INFO("Pilot recall:        %lu/%lu (%.1f%%) predicted-top-K overlap with the REAL "
                         "router's top-K, over %lu (layer, last-token) checks -- independent of worker "
                         "speed/staging pool size, measures prediction accuracy alone\n",
                         (unsigned long)rhits, (unsigned long)rdenom,
                         rdenom > 0 ? 100.0 * rhits / rdenom : 0.0, (unsigned long)rchecks);
                uint64_t already_ram = n_pilot_pred_already_ram.load();
                uint64_t already_pin = n_pilot_pred_already_vram_pinned.load();
                uint64_t already_lru = n_pilot_pred_already_vram_lru.load();
                uint64_t needs_fetch = n_pilot_pred_needs_fetch.load();
                uint64_t total_pred = already_ram + already_pin + already_lru + needs_fetch;
                LOG_INFO("Pilot prediction coverage: %lu/%lu top-K picks needed a fetch (already cached: "
                         "%lu vram-pin + %lu vram-lru + %lu ram)\n",
                         (unsigned long)needs_fetch, (unsigned long)total_pred,
                         (unsigned long)already_pin, (unsigned long)already_lru, (unsigned long)already_ram);
                uint64_t hhits = n_pilot_recall_hint_hits.load(), hdenom = n_pilot_recall_hint_checks_denom.load();
                LOG_INFO("Pilot accuracy (fetch-worthy only): %lu/%lu (%.1f%%) -- the number that actually "
                         "determines whether enabling prefetch can help this model\n",
                         (unsigned long)hhits, (unsigned long)hdenom,
                         hdenom > 0 ? 100.0 * hhits / hdenom : 0.0);

                // Per-RANK breakdown for l+1 -- see n_pilot_recall_hits_by_
                // rank's own comment (PerfCounters) for why this was added:
                // every prior "Pilot recall by rank" line was DEPTH-2 only.
                // Same two-line format (overall / fetch-worthy), one hop
                // closer where the pilot's prediction is structurally more
                // accurate than DEPTH-2's naive-reuse approximation.
                std::string rank_line_overall1 = "Pilot recall by rank (DEPTH-1, overall):     ";
                std::string rank_line_hint1    = "Pilot recall by rank (DEPTH-1, fetch-worthy): ";
                // Third line: accuracy restricted to the already_ram bucket
                // specifically -- see n_pilot_ram_only_hits_by_rank's own
                // comment (PerfCounters) for why this is a genuinely
                // different population from both lines above. THIS is the
                // number that should calibrate a RAM->VRAM promotion cutoff,
                // not the other two.
                std::string rank_line_ramonly1 = "Pilot recall by rank (DEPTH-1, ram-only):     ";
                char buf1[64];
                for (int rk = 0; rk < 8; ++rk) {
                    uint64_t rh1 = n_pilot_recall_hits_by_rank[rk].load();
                    uint64_t rc1 = n_pilot_recall_checks_by_rank[rk].load();
                    snprintf(buf1, sizeof(buf1), "k%d=%.1f%%(n=%lu) ", rk,
                             rc1 > 0 ? 100.0 * rh1 / rc1 : 0.0, (unsigned long)rc1);
                    rank_line_overall1 += buf1;
                    uint64_t hh1 = n_pilot_recall_hint_hits_by_rank[rk].load();
                    uint64_t hc1 = n_pilot_recall_hint_checks_by_rank[rk].load();
                    snprintf(buf1, sizeof(buf1), "k%d=%.1f%%(n=%lu) ", rk,
                             hc1 > 0 ? 100.0 * hh1 / hc1 : 0.0, (unsigned long)hc1);
                    rank_line_hint1 += buf1;
                    uint64_t rmh1 = n_pilot_ram_only_hits_by_rank[rk].load();
                    uint64_t rmc1 = n_pilot_ram_only_checks_by_rank[rk].load();
                    snprintf(buf1, sizeof(buf1), "k%d=%.1f%%(n=%lu) ", rk,
                             rmc1 > 0 ? 100.0 * rmh1 / rmc1 : 0.0, (unsigned long)rmc1);
                    rank_line_ramonly1 += buf1;
                }
                LOG_INFO("%s\n", rank_line_overall1.c_str());
                LOG_INFO("%s\n", rank_line_hint1.c_str());
                LOG_INFO("%s\n", rank_line_ramonly1.c_str());
            }
        }
        {
            // DEPTH-2 report -- approach (a), "naive reuse", measurement
            // only (see n_pilot2_recall_hits's own comment). Not wired to
            // any real prefetch path; exists to answer whether depth-2 is
            // worth building at all.
            uint64_t rhits2 = n_pilot2_recall_hits.load(), rchecks2 = n_pilot2_recall_checks.load();
            uint64_t rdenom2 = n_pilot2_recall_topk_sum.load();
            if (rchecks2 > 0) {
                LOG_INFO("Pilot recall (DEPTH-2, EXPERIMENTAL, measurement-only): %lu/%lu (%.1f%%) predicted-top-K "
                         "overlap with the REAL router's top-K two layers out, over %lu (layer, last-token) "
                         "checks -- naive reuse of the SAME vector the depth-1 pilot uses, missing l+1's own FFN "
                         "contribution entirely\n",
                         (unsigned long)rhits2, (unsigned long)rdenom2,
                         rdenom2 > 0 ? 100.0 * rhits2 / rdenom2 : 0.0, (unsigned long)rchecks2);
                uint64_t already_ram2 = n_pilot2_pred_already_ram.load();
                uint64_t already_pin2 = n_pilot2_pred_already_vram_pinned.load();
                uint64_t already_lru2 = n_pilot2_pred_already_vram_lru.load();
                uint64_t needs_fetch2 = n_pilot2_pred_needs_fetch.load();
                uint64_t total_pred2 = already_ram2 + already_pin2 + already_lru2 + needs_fetch2;
                LOG_INFO("Pilot prediction coverage (DEPTH-2): %lu/%lu top-K picks needed a fetch (already cached: "
                         "%lu vram-pin + %lu vram-lru + %lu ram)\n",
                         (unsigned long)needs_fetch2, (unsigned long)total_pred2,
                         (unsigned long)already_pin2, (unsigned long)already_lru2, (unsigned long)already_ram2);
                uint64_t hhits2 = n_pilot2_recall_hint_hits.load(), hdenom2 = n_pilot2_recall_hint_checks_denom.load();
                LOG_INFO("Pilot accuracy (DEPTH-2, fetch-worthy only): %lu/%lu (%.1f%%) -- the number that "
                         "determines whether depth-2 prefetch (prefetch_pilot_depth=2) can help\n",
                         (unsigned long)hhits2, (unsigned long)hdenom2,
                         hdenom2 > 0 ? 100.0 * hhits2 / hdenom2 : 0.0);
                uint64_t pushed2 = n_pilot2_hints_pushed.load();
                LOG_INFO("Pilot DEPTH-2 real hints pushed: %lu (nonzero only when prefetch_pilot_depth=2 -- "
                         "compare against %lu fetch-worthy picks above to see how much prefetch_pilot_k_depth2's "
                         "rank cutoff is holding back)\n",
                         (unsigned long)pushed2, (unsigned long)hdenom2);

                // Per-RANK breakdown -- see n_pilot2_recall_hits_by_rank's
                // own comment. Answers "does accuracy concentrate in the
                // pilot's own highest-confidence ranks" -- the question
                // that decides whether truncating to fewer fetched ranks
                // is a sound idea, as opposed to naively scaling K by the
                // aggregate fetch-worthy percentage above (which says
                // nothing about the SHAPE of the accuracy-by-rank curve).
                // Two lines (overall / fetch-worthy) mirroring the two
                // aggregate numbers just printed, one %% per rank instead
                // of one %% total.
                std::string rank_line_overall = "Pilot recall by rank (DEPTH-2, overall):     ";
                std::string rank_line_hint    = "Pilot recall by rank (DEPTH-2, fetch-worthy): ";
                char buf[64];
                for (int rk = 0; rk < 8; ++rk) {
                    uint64_t rh = n_pilot2_recall_hits_by_rank[rk].load();
                    uint64_t rc = n_pilot2_recall_checks_by_rank[rk].load();
                    snprintf(buf, sizeof(buf), "k%d=%.1f%%(n=%lu) ", rk,
                             rc > 0 ? 100.0 * rh / rc : 0.0, (unsigned long)rc);
                    rank_line_overall += buf;
                    uint64_t hh = n_pilot2_recall_hint_hits_by_rank[rk].load();
                    uint64_t hc = n_pilot2_recall_hint_checks_by_rank[rk].load();
                    snprintf(buf, sizeof(buf), "k%d=%.1f%%(n=%lu) ", rk,
                             hc > 0 ? 100.0 * hh / hc : 0.0, (unsigned long)hc);
                    rank_line_hint += buf;
                }
                LOG_INFO("%s\n", rank_line_overall.c_str());
                LOG_INFO("%s\n", rank_line_hint.c_str());
            }
        }
        {
            uint64_t issued = n_prefetch_issued.load();
            if (issued > 0) {
                uint64_t completed = n_prefetch_completed.load();
                uint64_t failed = n_prefetch_failed.load();
                double mb = bytes_prefetched.load() / 1e6;
                double avg_ms = completed > 0 ? t_prefetch_read_ms_sum.load() / completed : 0.0;
                // NOTE: this used to also report a "later actually used"
                // fraction here, computed from n_prefetch_hits -- that counter
                // fed the old separate staging pool's own consume-tracking
                // and is now permanently 0 (nothing increments it under the
                // unified claim/publish design), which would make this line
                // print a misleading, always-0.0% "later actually used"
                // forever rather than a real measurement. The genuinely
                // correct modern version of that same question -- was a
                // prefetch worker's real fetch later actually used before
                // eviction -- is already answered by "Unified RAM waste"
                // below (its complement: n_ram_claim_by_prefetch minus
                // n_ram_prefetch_loaded_evicted_unused), so it's not
                // recomputed redundantly here.
                LOG_INFO("Prefetch real fetches: %lu issued, %lu completed / %lu failed | %.2f MB read, "
                         "avg %.2f ms/fetch\n",
                         (unsigned long)issued, (unsigned long)completed, (unsigned long)failed, mb, avg_ms);

                // Concurrency-bucketed latency -- see n_prefetch_fetch_by_
                // concurrency's own comment for the full rationale. Read
                // this against the isolated single-fetch benchmark
                // (tools/gguf_tool dma --experts 1, real request shape):
                // bucket "0 others" should be the most directly comparable
                // number. If it's still elevated vs the isolated figure,
                // concurrency isn't the (whole) explanation; if it matches
                // and higher buckets climb, concurrent contention is
                // confirmed as a real, measured cause.
                {
                    std::ostringstream oss;
                    oss << "Prefetch fetch latency by concurrency (N other prefetch-worker fetches "
                           "in flight at submit time): ";
                    bool any = false;
                    for (int b = 0; b < PREFETCH_CONCURRENCY_BUCKETS; ++b) {
                        uint64_t c = n_prefetch_fetch_by_concurrency[b].load();
                        if (c == 0) continue;
                        any = true;
                        double avg_b = t_prefetch_fetch_ms_by_concurrency[b].load() / (double)c;
                        oss << (b == PREFETCH_CONCURRENCY_BUCKETS - 1 ? (std::to_string(b) + "+") : std::to_string(b))
                            << "=" << std::fixed << std::setprecision(2) << avg_b << "ms(n=" << c << ") ";
                    }
                    if (any) LOG_INFO("%s\n", oss.str().c_str());
                }

                // DEPTH-2's marginal share of the above -- see
                // n_prefetch_issued_depth2's own comment for why this is a
                // subset, not a separate total, and why "depth-1's share"
                // isn't computed here as a literal subtraction (the
                // aggregate also includes Phase 2 prefill-warm hints).
                // Only printed once depth-2 has actually issued at least
                // one real fetch -- silent (not a misleading 0-line) when
                // prefetch_pilot_depth=1, the default.
                uint64_t issued2 = n_prefetch_issued_depth2.load();
                if (issued2 > 0) {
                    uint64_t completed2 = n_prefetch_completed_depth2.load();
                    uint64_t failed2 = n_prefetch_failed_depth2.load();
                    uint64_t beaten2 = n_prefetch_claims_beaten_depth2.load();
                    double mb2 = bytes_prefetched_depth2.load() / 1e6;
                    double avg_ms2 = completed2 > 0 ? t_prefetch_read_ms_sum_depth2.load() / completed2 : 0.0;
                    LOG_INFO("Prefetch real fetches (DEPTH-2 only): %lu issued (%.1f%% of all real fetches), "
                             "%lu completed / %lu failed / %lu claim-beaten (already resident/loading by the "
                             "time a worker got to it) | %.2f MB read (%.1f%% of all prefetched bytes), "
                             "avg %.2f ms/fetch\n",
                             (unsigned long)issued2, issued > 0 ? 100.0 * issued2 / issued : 0.0,
                             (unsigned long)completed2, (unsigned long)failed2, (unsigned long)beaten2,
                             mb2, mb > 0 ? 100.0 * mb2 / mb : 0.0, avg_ms2);
                }
            }
        }
        LOG_INFO("VRAM->RAM demote:    %lu salvaged / %lu skipped (benign) / %lu failed "
                 "(vram_evict_demote_to_ram: instead of dropping an evicted expert's bytes, D2H "
                 "copy them back into the RAM tier before the VRAM slot is overwritten) -- "
                 "%lu fell back to legacy stream-0 cudaMemcpy (should be 0; nonzero means the "
                 "dedicated-stream fix silently didn't engage for those calls)\n",
                 (unsigned long)n_vram_evict_demoted.load(),
                 (unsigned long)n_vram_evict_demote_skipped.load(),
                 (unsigned long)n_vram_evict_demote_failed.load(),
                 (unsigned long)n_vram_demote_stream_unavailable.load());
        {
            uint64_t cb = n_demote_saw_compute_stream_busy.load(), ci = n_demote_saw_compute_stream_idle.load();
            uint64_t pb = n_demote_saw_copy_stream_busy.load(), pi = n_demote_saw_copy_stream_idle.load();
            uint64_t ct = cb + ci, pt = pb + pi;
            LOG_INFO("Demote stream-busy check: compute_stream busy %lu/%lu (%.1f%%) / copy_stream busy %lu/%lu (%.1f%%) "
                     "at the moment demote_from_vram's D2H fired -- direct test of whether there was genuinely "
                     "concurrent other-stream work to avoid waiting for (see n_demote_saw_* comment)\n",
                     (unsigned long)cb, (unsigned long)ct, ct > 0 ? 100.0 * cb / ct : 0.0,
                     (unsigned long)pb, (unsigned long)pt, pt > 0 ? 100.0 * pb / pt : 0.0);
        }
        {
            uint64_t acb = n_active_upload_saw_compute_stream_busy.load();
            uint64_t aci = n_active_upload_saw_compute_stream_idle.load();
            uint64_t apb = n_active_upload_saw_copy_stream_busy.load();
            uint64_t api = n_active_upload_saw_copy_stream_idle.load();
            uint64_t act = acb + aci, apt = apb + api;
            if (act > 0 || apt > 0) {
                LOG_INFO("Active-layer upload stream-busy check: compute_stream busy %lu/%lu (%.1f%%) / "
                         "copy_stream busy %lu/%lu (%.1f%%) at the moment upload_expert_active's H2D fired "
                         "-- same diagnostic as the Demote check above, different call site (GLM's dominant "
                         "dispatch path). A real busy%% here, unlike demote_from_vram's near-0%%, would "
                         "justify re-trying the async conversion already reverted for this call site\n",
                         (unsigned long)acb, (unsigned long)act, act > 0 ? 100.0 * acb / act : 0.0,
                         (unsigned long)apb, (unsigned long)apt, apt > 0 ? 100.0 * apb / apt : 0.0);
            }
        }
        {
            // Direct test of the two leading candidates for demote_from_vram's
            // remaining ~23% unexplained residual (see CLAUDE.md): per-GPU
            // ms/call + GB/s (multi-GPU PCIe asymmetry -- this box's 3090 is
            // x16, 5060 Ti is x8) and overall avg slab size for the
            // specifically-EVICTED population (vs the ~8.26MB corpus-wide
            // figure used to benchmark).
            int64_t total_bytes = 0; uint64_t total_calls = 0;
            for (int g = 0; g < PERF_MAX_GPU_SLOTS; g++) {
                uint64_t n = n_vram_demote_by_gpu[g].load();
                if (n == 0) continue;
                double t = t_vram_demote_by_gpu[g].load();
                int64_t b = bytes_vram_demote_by_gpu[g].load();
                total_bytes += b; total_calls += n;
                LOG_INFO("  demote gpu=%d: %lu calls, %.3f ms/call, %.2f GB/s, avg slab %.2f MB\n",
                         g, (unsigned long)n, t * 1000.0 / n,
                         t > 0 ? (b / 1e9) / t : 0.0, (b / 1e6) / (double)n);
            }
            if (total_calls > 0) {
                LOG_INFO("Demote per-GPU/size breakdown: %lu total calls, avg slab %.2f MB "
                         "(compare against the ~8.26MB corpus-wide figure used to benchmark)\n",
                         (unsigned long)total_calls, (total_bytes / 1e6) / (double)total_calls);
            }
        }
        LOG_INFO("Three-tier caching:  %lu active-layer uploads / %lu active-layer hits / %lu cleanups / "
                 "%lu promoted-to-main (MOVE vs COPY: %lu promotions freed RAM slots vs %lu left occupied, "
                 "%lu of the freed ones were COPY-mode reclaimed under ram_copy_reclaim_threshold pressure)\n",
                 (unsigned long)n_active_layer_uploads.load(),
                 (unsigned long)n_active_layer_hits.load(),
                 (unsigned long)n_active_layer_cleanup.load(),
                 (unsigned long)n_active_layer_promoted.load(),
                 (unsigned long)n_move_on_promotion.load(),
                 (unsigned long)n_copy_on_promotion.load(),
                 (unsigned long)n_copy_reclaimed_threshold.load());
        {
            uint64_t used = n_active_layer_prewarm_used.load(), wasted = n_active_layer_prewarm_wasted.load();
            uint64_t total_pw = used + wasted;
            LOG_INFO("Active-layer prewarm waste: %llu/%llu (%.1f%%) speculatively pre-warmed slot(s) "
                     "(decode_async_prewarm + prefill's Phase-2 batch warm-up) reclaimed WITHOUT any real "
                     "dispatch ever touching them first -- %.2f GB moved for nothing. Answers whether "
                     "narrowing this feature's own eagerness (fewer, higher-confidence candidates) is worth "
                     "pursuing before any further transfer-mechanism tuning -- see t_active_layer_upload_"
                     "async's own comment for why two such mechanism-level fixes were already tried and "
                     "reverted here.\n",
                     (unsigned long long)wasted, (unsigned long long)total_pw,
                     total_pw > 0 ? 100.0 * wasted / total_pw : 0.0,
                     bytes_active_layer_prewarm_wasted.load() / 1e9);
        }
        LOG_INFO("Active-layer upload (H2D): %8.3f s  (%6.2f GB/s, %.2f GB total) -- upload_expert_active's "
                 "own cudaHostRegister conversion, same Alt-A treatment as 'Host register stage' above but a "
                 "SEPARATE call site (the active-layer pool, not the main VRAM cache) -- %llu registration "
                 "failure(s), %llu stream-0 fallback(s), %llu cudaMemcpy failure(s) (all should be ~0)\n",
                 t_active_layer_host_register.load(),
                 t_active_layer_host_register > 0
                     ? (bytes_active_layer_host_registered / 1e9) / t_active_layer_host_register : 0.0,
                 bytes_active_layer_host_registered / 1e9,
                 (unsigned long long)n_active_layer_host_register_fail.load(),
                 (unsigned long long)n_active_layer_stream_unavailable.load(),
                 (unsigned long long)n_active_layer_upload_fail.load());
        {
            uint64_t ai = n_active_layer_uploads_async_issued.load();
            uint64_t as = n_active_layer_uploads_async_synced.load();
            uint64_t anr = n_active_layer_prewarm_not_ram_ready.load();
            if (ai > 0 || as > 0 || anr > 0) {
                LOG_INFO("Active-layer BATCH pre-warm (async): %llu issued / %llu synced, issue=%.3fs "
                         "sync=%.3fs (should sum to LESS than an equivalent population's plain "
                         "'Active-layer upload (H2D)'-style sync cost if batching is actually amortizing "
                         "per-call overhead) / %llu skipped (not yet RAM-resident, Phase 3's reactive path "
                         "handled them as before)\n",
                         (unsigned long long)ai, (unsigned long long)as,
                         t_active_layer_upload_async.load(), t_active_layer_batch_sync.load(),
                         (unsigned long long)anr);
            }
        }
        {
            uint64_t qc = n_qabsorb_gpu_calls.load();
            uint64_t qf = n_qabsorb_gpu_fallback_cpu.load();
            if (qc > 0 || qf > 0) {
                double up = t_qabsorb_gpu_up.load(), cmp = t_qabsorb_gpu_compute.load(), dn = t_qabsorb_gpu_down.load();
                LOG_INFO("MLA qabsorb BF16 batched GPU path (EngineConfig::attn_qabsorb_gpu): %llu call(s), "
                         "up=%.3fs cmp=%.3fs dn=%.3fs (%.3fms/call avg) / %llu CPU fallback(s) (this layer's "
                         "WUK BF16 upload never succeeded -- should track toward 0 once warm_mla_absorb_cache "
                         "finishes at startup)\n",
                         (unsigned long long)qc, up, cmp, dn,
                         qc > 0 ? ((up + cmp + dn) * 1000.0 / qc) : 0.0,
                         (unsigned long long)qf);
            }
        }
        {
            uint64_t sc = n_score_gpu_calls.load();
            uint64_t sf = n_score_gpu_fallback_cpu.load();
            if (sc > 0 || sf > 0) {
                double up = t_score_gpu_up.load(), cmp = t_score_gpu_compute.load(), dn = t_score_gpu_down.load();
                double wuv = t_score_gpu_wuv.load();
                LOG_INFO("MLA score GPU path (EngineConfig::attn_score_gpu): %llu call(s), "
                         "up=%.3fs cmp=%.3fs dn=%.3fs wuv=%.3fs (gpu %.3fms/call avg, wuv %.3fms/call avg) / "
                         "%llu CPU fallback(s) (S>1 prefill, or this layer's Lc/Rc GPU mirror never succeeded -- "
                         "should track toward 0 for decode once warm_mla_absorb_cache finishes at startup)\n",
                         (unsigned long long)sc, up, cmp, dn, wuv,
                         sc > 0 ? ((up + cmp + dn) * 1000.0 / sc) : 0.0,
                         sc > 0 ? (wuv * 1000.0 / sc) : 0.0,
                         (unsigned long long)sf);
            }
        }
        {
            uint64_t wc = n_wuv_gpu_calls.load();
            uint64_t wf = n_wuv_gpu_fallback_cpu.load();
            if (wc > 0 || wf > 0) {
                double up = t_wuv_gpu_up.load(), cmp = t_wuv_gpu_compute.load(), dn = t_wuv_gpu_down.load();
                LOG_INFO("WUV BF16 batched GPU path (score GPU path's down-projection): %llu call(s), "
                         "up=%.3fs cmp=%.3fs dn=%.3fs (%.3fms/call avg) / %llu CPU fallback(s) (this layer's "
                         "WUV BF16 upload never succeeded -- should track toward 0 once warm_mla_absorb_cache "
                         "finishes at startup; CPU fallback cost visible in t_score_gpu_wuv above)\n",
                         (unsigned long long)wc, up, cmp, dn,
                         wc > 0 ? ((up + cmp + dn) * 1000.0 / wc) : 0.0,
                         (unsigned long long)wf);
            }
        }
        {
            uint64_t di = n_decode_async_prewarm_issued.load();
            uint64_t ds = n_decode_async_prewarm_synced.load();
            uint64_t dc = n_decode_async_prewarm_chunks.load();
            uint64_t dn = n_decode_async_prewarm_no_slot.load();
            uint64_t dst = n_decode_async_prewarm_stale_caught.load();
            if (di > 0 || ds > 0 || dc > 0 || dn > 0 || dst > 0) {
                LOG_INFO("Decode-time BATCH pre-warm (async, EngineConfig::decode_async_prewarm): "
                         "%llu issued / %llu synced in %llu chunk(s) (avg %.2f/chunk, capped at "
                         "decode_async_prewarm_max_batch), %llu skipped (active-layer pool full/alloc "
                         "failed -- fell through to dispatch_expert_k's existing synchronous path), "
                         "%llu stale/reclaimed slot(s) caught by the post-pin identity check (should be "
                         "~0 -- see that code's own comment; SEPARATE from the combined 'Active-layer "
                         "BATCH pre-warm' counters above, which this also feeds into but can't "
                         "distinguish from prefill's own contribution alone)\n",
                         (unsigned long long)di, (unsigned long long)ds, (unsigned long long)dc,
                         dc > 0 ? (double)di / (double)dc : 0.0,
                         (unsigned long long)dn, (unsigned long long)dst);
            }
        }
        LOG_INFO("Pilot RAM->VRAM promotion: %lu attempted / %lu found slot already gone "
                 "(ram_vram_promote_top_k/adaptive -- MOVE/COPY counts above include this path's "
                 "successes too, shared with the reactive hit-count promotion)\n",
                 (unsigned long)n_pilot_promote_attempted.load(),
                 (unsigned long)n_pilot_promote_slot_gone.load());
        LOG_INFO("Prefetch pickup:     avg %.2fms over %lu hints, peak queue backlog %lu "
                 "(large delay + small backlog => worker CPU-starved; growing backlog => worker throughput-bound)\n",
                 n_prefetch_hints_processed > 0
                     ? 1000.0 * t_prefetch_pickup_delay.load() / n_prefetch_hints_processed.load() : 0.0,
                 (unsigned long)n_prefetch_hints_processed.load(),
                 (unsigned long)n_prefetch_backlog_peak.load());
        {
            uint64_t hi = n_prefetch_hi_lane_processed.load(), lo = n_prefetch_lo_lane_processed.load();
            uint64_t starved = n_prefetch_lo_lane_starved_ticks.load();
            if (hi + lo > 0) {
                LOG_INFO("Prefetch lanes:      %lu HIGH (l+1/prefill) / %lu LOW (l+2) processed, "
                         "%lu tick(s) LOW was ready but skipped for HIGH (strict priority -- expected "
                         "to be nonzero whenever both lanes are active; watch for l+2 starvation if this "
                         "is a large fraction of lo-lane pushes)\n",
                         (unsigned long)hi, (unsigned long)lo, (unsigned long)starved);
            }
        }
        {
            uint64_t uok = n_uring_batch_ok.load(), ufb = n_uring_batch_fallback.load();
            if (uok + ufb > 0) {
                LOG_INFO("io_uring batches:    %lu batched / %lu fell back to sequential preads (%.1f%% batched)\n",
                         (unsigned long)uok, (unsigned long)ufb,
                         100.0 * uok / (uok + ufb));
            }
        }
        {
            uint64_t dok = n_decode_uring_ok.load();
            uint64_t dna = n_decode_uring_not_available.load();
            uint64_t dnp = n_decode_uring_not_packed.load();
            uint64_t dtl = n_decode_uring_too_large.load();
            uint64_t dsf = n_decode_uring_submit_failed.load();
            uint64_t dsr = n_decode_uring_short_read.load();
            uint64_t dtotal = dok + dna + dnp + dtl + dsf + dsr;
            if (dtotal > 0) {
                LOG_INFO("Decode-thread io_uring batches: %lu/%lu batched (%.1f%%) -- fallback cause: "
                         "%lu ring-not-available / %lu layer-not-packed (check gguf_tool verify!) / "
                         "%lu batch-too-large / %lu submit-failed / %lu short-read\n",
                         (unsigned long)dok, (unsigned long)dtotal,
                         100.0 * dok / dtotal,
                         (unsigned long)dna, (unsigned long)dnp, (unsigned long)dtl,
                         (unsigned long)dsf, (unsigned long)dsr);
            }
        }
        {
            // Gated on ANY of these being nonzero, not just issued>0 -- a
            // run where padding is enabled but never issues a single read
            // is exactly the "why didn't this help" case these diagnostics
            // exist for; gating only on issued>0 would hide every counter
            // that explains why, precisely when they're most needed.
            uint64_t pi = n_pad_reads_issued.load();
            uint64_t pns = n_pad_no_free_slot.load();
            uint64_t pge = n_pad_gate_entered.load();
            uint64_t pnr = n_pad_no_ranking_for_layer.load();
            uint64_t pnc = n_pad_no_candidates_after_filter.load();
            uint64_t pev = n_pad_evict_dup_used.load();
            uint64_t pev_un = n_pad_evict_dup_unavailable.load();
            if (pi > 0 || pns > 0 || pge > 0 || pnr > 0 || pnc > 0 || pev > 0 || pev_un > 0) {
                LOG_INFO("Disk-batch padding:  %lu issued (%lu ok / %lu failed), %lu later genuinely "
                         "hit (%.1f%% payoff), %.2f GB speculative bytes read, %lu no-free-slot skips "
                         "(%lu of which were filled anyway by evicting an also-VRAM-resident RAM slot, "
                         "disk_batch_padding_evict_dup; %lu still had none available even then)\n",
                         (unsigned long)pi, (unsigned long)n_pad_reads_ok.load(),
                         (unsigned long)n_pad_reads_failed.load(), (unsigned long)n_pad_reads_hit_later.load(),
                         pi > 0 ? 100.0 * n_pad_reads_hit_later.load() / pi : 0.0,
                         bytes_disk_read_speculative.load() / 1e9,
                         (unsigned long)pns, (unsigned long)pev, (unsigned long)pev_un);
                LOG_INFO("Disk-batch padding diag: %lu gate-entered, %lu layer-had-no-ranking, "
                         "%lu ranking-had-zero-survivors-after-filter\n",
                         (unsigned long)pge, (unsigned long)pnr, (unsigned long)pnc);
            }
            uint64_t pdc = n_pad_dbg_calls.load();
            if (pdc > 0) {
                LOG_INFO("Disk-batch padding gate terms (of %lu real Phase B calls): "
                         "%lu disk_batch_padding=false, %lu !all_packed, %lu reqs.empty(), "
                         "%lu in_prefill, %lu n_pending>=target\n",
                         (unsigned long)pdc,
                         (unsigned long)n_pad_dbg_padding_false.load(),
                         (unsigned long)n_pad_dbg_not_all_packed.load(),
                         (unsigned long)n_pad_dbg_reqs_empty.load(),
                         (unsigned long)n_pad_dbg_in_prefill.load(),
                         (unsigned long)n_pad_dbg_target_not_reached.load());
            }
        }
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
        if (n_upload_expert_no_victim.load() || n_upload_expert_memcpy_fail.load()) {
            LOG_INFO("upload_expert fails: %lu no-victim (evict_lru found nothing evictable) / %lu "
                     "cudaMemcpy H2D failures -- both should normally be 0; nonzero during startup "
                     "preload means pinning/pre-warming silently lost most of its attempts (see "
                     "[MOE-VRAM] upload_expert error lines for the specific cause per attempt)\n",
                     (unsigned long)n_upload_expert_no_victim.load(),
                     (unsigned long)n_upload_expert_memcpy_fail.load());
        }
        LOG_INFO("upload_expert stream fallback: %lu (fell back to legacy stream-0 cudaMemcpy "
                 "instead of the dedicated copy_stream; should be 0 -- nonzero means the "
                 "dedicated-stream fix silently didn't engage for those calls)\n",
                 (unsigned long)n_upload_expert_stream_unavailable.load());
        LOG_INFO("Hold-for-compute fails: %lu (an apparently-VRAM-resident expert's identity changed "
                 "between the residency check and hold_for_compute -- forces a safety fallback to CPU "
                 "for that call rather than trust a possibly-stale pointer. Correctness-safe either way; "
                 "high/rising values across runs with a shrinking unpinned/reactive VRAM pool point at "
                 "eviction churn as the driver of any CPU-fallback-call increase, not a bug)\n",
                 (unsigned long)n_hold_for_compute_fail.load());
        LOG_INFO("CPU fallback cause:  %lu type-unsupported (fixed, model-intrinsic) / %lu no-host-buffer "
                 "(shadow-fast-path likely raced and lost VRAM residency -- see shadow-race line below) / "
                 "%lu just-promoted (expected/benign -- RAM slot was just MOVE-promoted to VRAM, needs one "
                 "more reactive hit before it's GPU-usable) / %lu upload-failed (a genuine cudaMemcpy/"
                 "evict_lru attempt failed -- see upload_expert fails line above) / %lu cpu_affinity_layers "
                 "(expected -- config explicitly pinned this layer to CPU, never even attempted a GPU "
                 "residency check) of the total CPU-dispatched calls\n",
                 (unsigned long)n_cpu_fallback_type_unsupported.load(),
                 (unsigned long)n_cpu_fallback_no_host_buffer.load(),
                 (unsigned long)n_cpu_fallback_just_promoted.load(),
                 (unsigned long)n_cpu_fallback_upload_failed.load(),
                 (unsigned long)n_cpu_fallback_affinity_layer.load());
        {
            uint64_t cf_ok = n_cufile_read_ok.load();
            uint64_t cf_fail = n_cufile_read_fail.load();
            uint64_t cf_skip = n_cufile_skipped_ram_resident.load();
            if (cf_ok || cf_fail || cf_skip) {
                LOG_INFO("cuFile (GDS):        %lu direct disk->VRAM read(s) OK (%.2f GB, %.3f s) / "
                         "%lu failed (fell back to RAM-mediated upload) / %lu skipped because the "
                         "expert was already RAM-resident (correct -- cuFile never re-reads disk when "
                         "a RAM copy already exists)\n",
                         (unsigned long)cf_ok, bytes_cufile_read.load() / 1e9, t_cufile_read.load(),
                         (unsigned long)cf_fail, (unsigned long)cf_skip);
            }
        }
        if (n_shadow_race_reload_success.load() || n_shadow_race_reload_hard_fail.load()) {
            LOG_INFO("Shadow-race reload:  %lu succeeded (expert still computed, via CPU instead of the "
                     "originally-expected GPU path) / %lu hard-failed (expert's contribution skipped "
                     "ENTIRELY for that token -- a real quality gap, not just perf) -- previously assumed "
                     "\"rare\" but never actually counted\n",
                     (unsigned long)n_shadow_race_reload_success.load(),
                     (unsigned long)n_shadow_race_reload_hard_fail.load());
        }
        if (n_bf16_overflow_clamped.load()) {
            LOG_WARN("BF16 overflow clamped: %lu (a finite F32 weight or activation rounded up into "
                     "BF16's Inf/NaN exponent pattern and was clamped to the largest finite BF16 "
                     "magnitude instead of propagating -- should be essentially unreachable for real "
                     "LLM values; nonzero here means something upstream produced a value within a "
                     "fraction of a percent of FLT_MAX, worth investigating directly)\n",
                     (unsigned long)n_bf16_overflow_clamped.load());
        }
        LOG_INFO("Staging mutex wait:  %8.3f s  (staging slot regrow: %lu events, %.3f s -- "
                 "cudaMallocHost/cudaFreeHost held under the SAME lock, see [SLOW-STAGING-REGROW])\n",
                 t_staging_mutex_wait.load(), (unsigned long)n_staging_regrow.load(), t_staging_regrow.load());
        // REMOVED: "Staged-unused evict" and "Stale-slot reclaim" report
        // lines used to live here (n_staged_evicted_unused/n_staging_stale_
        // reclaimed/n_staging_cross_token_stale_reclaimed/n_staging_
        // fallback_events) -- both measured the old separate staging pool's
        // own round-robin eviction/reclaim behavior, which no longer exists
        // (LockFreePrefetcher writes directly into the RAM tier now, see
        // ram_manager.hpp's claim_or_wait). The replacement signal is
        // "Unified RAM waste" further down (n_ram_prefetch_loaded_evicted_
        // unused), measured against the real cache's real LRU eviction
        // instead of an artificial round-robin pool.
        {
            uint64_t vram_race = n_evicted_was_pilot_confirmed_vram.load();
            uint64_t ram_race = n_evicted_was_pilot_confirmed_ram.load();
            LOG_INFO("Pilot-confirm evict: %lu VRAM / %lu RAM -- experts the pilot's most recent skip-fetch "
                     "decision had JUST confirmed already-resident (RAM or VRAM-unpinned), then ordinary LRU "
                     "eviction picked as a victim anyway before decode's real dispatch reached them (see "
                     "was_pilot_confirmed_resident, pilot_confirmed_set.hpp). Nonzero here is direct evidence "
                     "eviction_score()'s pure recency+frequency scoring undoes the pilot's own already-cached "
                     "determination, forcing a synchronous re-read the pilot assumed wouldn't be needed -- "
                     "measurement only, does not yet change eviction behavior.\n",
                     (unsigned long)vram_race, (unsigned long)ram_race);
        }
        {
            uint64_t v_if = n_vram_hit_had_inflight_fetch.load(), v_st = n_vram_hit_had_staged_fetch.load();
            uint64_t v_ev = n_vram_hit_had_evicted_fetch.load();
            uint64_t r_if = n_ram_hit_had_inflight_fetch.load(), r_st = n_ram_hit_had_staged_fetch.load();
            uint64_t r_ev = n_ram_hit_had_evicted_fetch.load();
            uint64_t total = v_if + v_st + v_ev + r_if + r_st + r_ev;
            if (total > 0) {
                double v_st_avg_ms = v_st > 0 ? t_vram_hit_staged_fetch_elapsed_us_sum.load() / 1000.0 / (double)v_st : 0.0;
                double r_st_avg_ms = r_st > 0 ? t_ram_hit_staged_fetch_elapsed_us_sum.load() / 1000.0 / (double)r_st : 0.0;
                LOG_INFO("Cache-hit-beat-fetch: %lu total -- of every INDEPENDENT cache hit (bypassed "
                         "consume_staged entirely) whose expert had ALSO been recently touched by the "
                         "prefetcher: VRAM tier: %lu in-flight / %lu still-staged-unconsumed (avg %.1fms old, "
                         "the CLEANEST redundancy signal -- the fetch fully succeeded and was still sitting "
                         "there) / %lu already-evicted | RAM tier: %lu in-flight / %lu still-staged-unconsumed "
                         "(avg %.1fms old) / %lu already-evicted -- this is the direct measurement of why "
                         "correctly-predicted fetches go unused: an independent cache hit resolved the "
                         "dispatch before the prefetched copy was ever consulted.\n",
                         (unsigned long)total, (unsigned long)v_if, (unsigned long)v_st, v_st_avg_ms,
                         (unsigned long)v_ev, (unsigned long)r_if, (unsigned long)r_st, r_st_avg_ms,
                         (unsigned long)r_ev);

                // Hypothesis test: was the above a SAME-TOKEN race, or an
                // earlier token's abandoned fetch being rediscovered by a
                // later, unrelated token's own hit? step_delta==0 across
                // (nearly) all matches would mean same-token race; a large
                // average step_delta with same_step near 0 would confirm
                // the stale-record hypothesis instead.
                uint64_t v_st_same = n_vram_staged_fetch_same_step.load(), v_ev_same = n_vram_evicted_fetch_same_step.load();
                uint64_t r_st_same = n_ram_staged_fetch_same_step.load(), r_ev_same = n_ram_evicted_fetch_same_step.load();
                double v_st_step_avg = v_st > 0 ? t_vram_staged_fetch_step_delta_sum.load() / (double)v_st : 0.0;
                double v_ev_step_avg = v_ev > 0 ? t_vram_evicted_fetch_step_delta_sum.load() / (double)v_ev : 0.0;
                double r_st_step_avg = r_st > 0 ? t_ram_staged_fetch_step_delta_sum.load() / (double)r_st : 0.0;
                double r_ev_step_avg = r_ev > 0 ? t_ram_evicted_fetch_step_delta_sum.load() / (double)r_ev : 0.0;
                LOG_INFO("Cache-hit-beat-fetch token-step test: same-token-step (race within one layer "
                         "transition) vs cross-token-step (an earlier token's fetch, never consumed, "
                         "rediscovered later by an unrelated token's own hit) -- VRAM staged: %lu/%lu "
                         "same-step, avg %.1f steps back | VRAM evicted: %lu/%lu same-step, avg %.1f steps "
                         "back | RAM staged: %lu/%lu same-step, avg %.1f steps back | RAM evicted: %lu/%lu "
                         "same-step, avg %.1f steps back -- near-zero same-step counts + large avg steps-back "
                         "confirms the stale-record hypothesis; same-step counts close to the total would "
                         "instead mean a genuine same-transition race.\n",
                         (unsigned long)v_st_same, (unsigned long)v_st, v_st_step_avg,
                         (unsigned long)v_ev_same, (unsigned long)v_ev, v_ev_step_avg,
                         (unsigned long)r_st_same, (unsigned long)r_st, r_st_step_avg,
                         (unsigned long)r_ev_same, (unsigned long)r_ev, r_ev_step_avg);
            }
        }
        // REMOVED: "RAM-load-vs-prefetch" and "Phase-A/Phase-B gap" report
        // blocks used to live here, fed by n_ram_load_had_prefetch_*/
        // n_phase_gap_staged_*/t_phase_gap_ms_sum -- both measured races
        // against the old separate staging pool that no longer exists. See
        // "Unified RAM claim"/"Unified RAM wait" right below for the
        // replacement instrumentation (RAMSafetyManager::claim_or_wait,
        // ram_manager.hpp).
        {
            uint64_t hit = n_ram_claim_hit.load();
            uint64_t by_decode = n_ram_claim_by_decode.load();
            uint64_t by_prefetch = n_ram_claim_by_prefetch.load();
            uint64_t already_resident = n_ram_claim_already_resident_prefetch.load();
            uint64_t already_inflight = n_ram_claim_already_inflight_prefetch.load();
            uint64_t waited = n_ram_wait_inflight_decode.load();
            uint64_t evicted_unused = n_ram_prefetch_loaded_evicted_unused.load();
            uint64_t total_claims = hit + by_decode + by_prefetch;
            if (total_claims > 0 || already_resident > 0 || already_inflight > 0) {
                LOG_INFO("Unified RAM claim:   %lu hit, %lu claimed-by-decode, %lu claimed-by-prefetch "
                         "(of %lu total claim_or_wait calls) -- prefetch's own claim step consults the "
                         "SAME index a decode-thread hit-check uses, so it structurally cannot re-fetch an "
                         "already-resident expert: %lu prefetch claim attempts found the expert already "
                         "Ready (this is the direct, by-construction elimination of the old "
                         "Cache-hit-beat-fetch waste category), %lu found it already Loading (a no-op "
                         "dedup against whoever else is fetching it).\n",
                         (unsigned long)hit, (unsigned long)by_decode, (unsigned long)by_prefetch,
                         (unsigned long)total_claims, (unsigned long)already_resident,
                         (unsigned long)already_inflight);
                if (waited > 0) {
                    LOG_INFO("Unified RAM wait:    %lu decode-thread miss(es) found the expert already "
                             "Loading (claimed by a prefetch worker) and waited for that exact in-flight "
                             "read instead of racing past it -- avg wait %.2fms, total %.3fs. This is the "
                             "direct, by-construction elimination of the old Phase-A/Phase-B ordering gap: "
                             "no more check-once-then-race, the decode thread now waits for the SAME read "
                             "the prefetcher already started.\n",
                             (unsigned long)waited, t_ram_wait_inflight_ms_sum.load() / (double)waited,
                             t_ram_wait_inflight_ms_sum.load() / 1000.0);
                }
                if (by_prefetch > 0) {
                    LOG_INFO("Unified RAM waste:   %lu/%lu (%.1f%%) prefetch-loaded slot(s) were reclaimed "
                             "by normal LRU eviction before any real hit ever touched them -- the "
                             "unified design's replacement for the old staging-pool-based wasted-prefetch "
                             "measurement, now grounded in the real cache's real LRU pressure instead of "
                             "an artificial round-robin pool.\n",
                             (unsigned long)evicted_unused, (unsigned long)by_prefetch,
                             by_prefetch > 0 ? 100.0 * evicted_unused / by_prefetch : 0.0);
                }
            }
        }
        LOG_INFO("RAM mutex wait:      %8.3f s  (RAMSafetyManager::ram_mutex, shared between the decode "
                 "thread's own miss-handling and the prefetcher worker thread's claims)\n",
                 t_ram_mutex_wait.load());
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
        // Explicit overall/decode/prefill split -- the Throughput line above
        // mixes prefill+decode into one denominator (see its own comment),
        // which understates real steady-state decode speed and gives no way
        // to directly compare prefill-only cost across runs/config changes
        // without manually subtracting t_prefill_total by hand each time.
        // decode = generated tokens over wall time EXCLUDING prefill;
        // prefill = prompt tokens over prefill's own wall time (both derived
        // from counters that already existed independently -- n_prompt_
        // tokens is the only new one, added specifically to complete this
        // line).
        {
            double decode_s = active_s - t_prefill_total.load();
            double overall_rate = active_s > 0 ? n_tokens_generated.load() / active_s : 0.0;
            double decode_rate = decode_s > 0 ? n_tokens_generated.load() / decode_s : 0.0;
            double prefill_rate = t_prefill_total > 0 ? n_prompt_tokens.load() / t_prefill_total.load() : 0.0;
            if (n_tokens_generated > 0 && active_s > 0) {
                LOG_INFO("Rates:               overall=%.2f tok/s | decode=%.2f tok/s (%lu tokens / %.2fs) | "
                         "prefill=%.2f tok/s (%lu tokens / %.2fs)\n",
                         overall_rate, decode_rate, (unsigned long)n_tokens_generated.load(), decode_s,
                         prefill_rate, (unsigned long)n_prompt_tokens.load(), t_prefill_total.load());
            }
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
