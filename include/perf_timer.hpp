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
    // A slot round-robin-recycled (acquire_staging_slot) while STILL
    // ready==true and holding a real, never-consumed (layer,expert)
    // identity -- i.e. a correctly-predicted, correctly-read, correctly-
    // staged expert that simply never matched a miss before its slot got
    // reused. Added specifically to separate "pool thrashing" from other
    // explanations for the gap between Pilot recall (prediction accuracy
    // alone, see n_pilot_recall_hits) and Prefetch (PIPE) served %
    // (n_pipe_hits) -- a high count here means the pool genuinely needs
    // to be bigger (or predictions need to happen closer to time-of-use);
    // a low count here despite a large recall/served gap means the
    // explanation is something else entirely (e.g. the predicted expert
    // quietly became a cache HIT via some other path before the miss
    // check ever ran, so consume_staged was never even called for it).
    std::atomic<uint64_t> n_staged_evicted_unused{0};
    std::atomic<double> t_staging_mutex_wait{0};  // total time ANY caller (consume_staged/acquire_staging_slot/finish_prefetch) spent BLOCKED acquiring staging_mutex -- high values here directly prove cross-thread contention on this lock, separate from time spent doing real work once held
    std::atomic<double> t_cache_mutex_wait{0};    // total time ANY caller spent BLOCKED acquiring GpuTopology::cache_mutex (find_expert/pin_expert/evict_lru/has_free_slot/upload_expert/upload_expert_async)

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
    // Phase A () counters:
    std::atomic<uint64_t> n_pipe_hits{0};          // cache misses served from the prefetcher's staging pool (no pread) -- the higher this is vs n_expert_cache_miss, the more disk I/O was hidden behind compute
    std::atomic<uint64_t> n_pinned_eviction_skips{0}; // MEANING CHANGED once GpuTopology::unpinned_slot_indices landed: evict_lru no longer visits permanently-pinned slots at all (they're excluded from the scan at the source), so this now only increments on the DEFENSIVE fallback check -- should read 0 on a healthy run; nonzero means unpinned_slot_indices desynced from the real `pinned` flags somewhere (see gpu_topology.hpp's evict_lru). Previously (before that optimization) this counted every pinned slot skipped during a normal scan, and nonzero was the EXPECTED/healthy signal -- the polarity of "what a nonzero value means" flipped with that change.
    std::atomic<uint64_t> n_temp_pin_eviction_skips{0}; // times evict_lru skipped a slot TEMPORARILY held for an in-flight decode-time GPU dispatch (ExpertLocation::temp_pin_count > 0) -- distinct from n_pinned_eviction_skips (permanent pins); nonzero proves the compute-in-flight protection is actually engaging
    std::atomic<uint64_t> n_upload_pending_eviction_skips{0}; // times evict_lru skipped a slot whose async H2D upload was CONFIRMED still genuinely in flight (re-polled via cudaEventQuery, not just trusting a stale flag) -- the real "must not evict" count
    std::atomic<uint64_t> n_stale_upload_ready_reclaimed{0}; // times evict_lru found upload_ready==false but the event had ACTUALLY already completed (a speculative pilot-prefetch promotion that was never subsequently confirmed via find_expert) and flipped it back to true itself -- nonzero proves real cache capacity was being silently leaked before this fix; see gpu_topology.hpp's evict_lru comment
    std::atomic<uint64_t> n_stale_expert_index_hits{0}; // times find_expert's O(1) expert_index pointed at a slot that didn't actually match the requested (layer,expert_id) -- should be ALWAYS ZERO by construction (evict_lru/upload_expert/upload_expert_async are the only mutation sites, all under cache_mutex); nonzero means the index and expert_vram_cache have desynced somewhere and find_expert is safely falling back to its linear scan, but the root cause needs investigating
    std::atomic<uint64_t> n_pinned_resident{0};    // experts currently held as non-evictable VRAM residents (pinned). Gauge-ish: incremented on pin, decremented on unpin.
    std::atomic<uint64_t> n_pilot_hints_skipped{0}; // speculative pilot-prefetch hints skipped because the expert was ALREADY host-cache-resident (a guaranteed instant hit needing no read) -- measures how much prefetch-worker bandwidth this avoids wasting; compare against n_pipe_hits/n_expert_cache_miss before/after to see the effect

    // Pure PREDICTION-ACCURACY recall: overlap between the pilot's
    // predicted top-K (LayerScratch::pilot_pred_ids, captured before any
    // already-resident filtering) and the REAL router's top-K when that
    // same layer's moe_forward_gguf actually runs, at the same "last
    // token" position both sides use. Deliberately measured completely
    // separately from "Prefetch (PIPE) served %" -- that number also
    // depends on worker-thread speed and staging-pool size (both already
    // tested this session with little effect), so it can't tell you
    // whether the REMAINING gap is prediction accuracy or something else
    // downstream. This can. n_pilot_recall_hits / (n_pilot_recall_checks *
    // topk) is the recall fraction, directly comparable to a reference
    // implementation's own reported recall number.
    std::atomic<uint64_t> n_pilot_recall_hits{0};
    std::atomic<uint64_t> n_pilot_recall_checks{0};   // number of (layer, "last token") comparisons made
    std::atomic<uint64_t> n_pilot_recall_topk_sum{0}; // sum of topk across those checks (denominator, since topk could vary in principle)
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
    std::atomic<uint64_t> n_pipe_miss_not_ready{0};    // consume_staged found a matching (layer,expert) slot but its pread hadn't finished yet -- the pilot's prediction was CORRECT, this was purely a timing loss. Points at making the staged read start earlier / finish faster (or growing prefetch_staging_slots so more predictions survive the wait) as the lever, not prediction accuracy.

    // Added when finish_prefetch's unconditional, unconfirmed VRAM
    // promotion (async_prefetcher.hpp) was removed -- that call used to
    // fire on EVERY successful staged read regardless of whether the
    // expert was ever actually used, evicting real VRAM residents on pure
    // speculation. VRAM promotion for a PIPE-served expert (MoEExpertBlock::
    // from_pipe_prefetch, engine.hpp) now only ever happens via the normal
    // hit-count-gated path (promote_hit_threshold) once a REAL subsequent
    // access confirms it. This counter is the direct evidence that path
    // still works for prefetch-origin experts post-fix -- nonzero proves
    // prefetched experts do still reach VRAM, just on confirmed reuse
    // rather than blind speculation. Compare against n_pipe_hits: a
    // healthy run should show a meaningful fraction of PIPE hits
    // eventually promoted, not zero.
    std::atomic<uint64_t> n_pipe_staged_later_promoted{0};


    // Three-tier caching performance counters
    std::atomic<uint64_t> n_active_layer_uploads{0};    // uploads to active layer area (temporary workspace)
    std::atomic<uint64_t> n_active_layer_hits{0};      // cache hits served from active layer area
    std::atomic<uint64_t> n_active_layer_cleanup{0};    // active layer slots reclaimed after layer completion
    std::atomic<uint64_t> n_active_layer_promoted{0};   // active-layer experts promoted into the real VRAM LRU cache (hit_count crossed active_layer_promote_threshold), vs. just freed at cleanup
    std::atomic<uint64_t> n_move_on_promotion{0};       // RAM→VRAM promotions that used MOVE (freed RAM slot) instead of COPY
    std::atomic<uint64_t> n_copy_on_promotion{0};       // RAM→VRAM promotions that used COPY (left RAM slot occupied)

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
    // expert fan-out across ALL prompt tokens at once -- so an aggregate
    // PIPE-served rate near 0% could mean "decode is fine, prefill just
    // structurally can't be helped" rather than "the whole mechanism is
    // broken." Compare n_pipe_hits_decode/n_expert_cache_miss_decode
    // against the aggregate rate to see if decode alone looks any better.
    std::atomic<uint64_t> n_expert_cache_miss_prefill{0};
    std::atomic<uint64_t> n_expert_cache_miss_decode{0};
    std::atomic<uint64_t> n_pipe_hits_prefill{0};
    std::atomic<uint64_t> n_pipe_hits_decode{0};

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
    // blocking here to find out) -- whether they were actually served is
    // visible via the EXISTING n_pipe_hits_prefill/n_pipe_miss_still_queued
    // counters, since these hints flow through the exact same queue/worker
    // pipeline PILOT's own hints already use.
    std::atomic<uint64_t> n_prefill_hint_pushed{0};

    // Of n_pipe_miss_no_match above, how many still had a hint for this
    // EXACT (layer,expert) sitting unpopped in the prefetcher's SPSC queue
    // at the moment of the miss (LockFreePrefetcher::has_pending_hint) --
    // i.e. the pilot predicted correctly, but the single worker thread
    // hadn't even STARTED that read yet. A high fraction here points at a
    // worker-thread throughput/scheduling bottleneck (N hints queued per
    // layer transition, one thread servicing them one at a time) as the
    // fix, not prediction accuracy or staging pool size.
    std::atomic<uint64_t> n_pipe_miss_still_queued{0};

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
        double t_disk_read=0, t_host_alloc=0, t_gpu_upload=0, t_expert_gpu_upload=0, t_gpu_download=0, t_gpu_compute=0, t_cpu_expert=0;
        double t_cpu_dequant=0, t_cpu_dot=0;
        double t_attn=0;
        double t_attn_preamble=0;
        double t_attn_norm=0, t_attn_qproj=0, t_attn_kvproj=0, t_attn_absorb=0, t_attn_score=0, t_attn_outproj=0;
        double t_dense_gpu_up=0, t_dense_gpu_compute=0, t_dense_gpu_down=0;
        double t_moe_route=0, t_moe_expert=0, t_prefetch=0, t_dense_load=0, t_consume_staged=0;
        double t_staging_regrow=0, t_staging_mutex_wait=0, t_cache_mutex_wait=0;
        double t_active_layer_upload=0, t_active_layer_promote_copy=0, t_vram_demote_copy=0;
        double t_hold_for_compute=0, t_release_hold=0, t_find_active_expert=0;
        double t_find_expert=0, t_evict_lru=0, t_upload_expert_async=0;
        double t_ram_get_expert=0, t_ram_get_experts_batch=0;
        uint64_t n_expert_cache_hit=0, n_expert_cache_miss=0, n_expert_promote=0, n_tokens_generated=0;
        uint64_t n_expert_gpu_calls=0, n_expert_cpu_calls=0, n_dense_gpu_calls=0;
        int64_t bytes_disk_read=0, bytes_gpu_uploaded=0;
        uint64_t n_pipe_hits=0, n_pinned_eviction_skips=0, n_pinned_resident=0, n_staging_regrow=0;

        Snapshot operator-(const Snapshot& b) const {
            return Snapshot{
                t_disk_read - b.t_disk_read, t_host_alloc - b.t_host_alloc,
                t_gpu_upload - b.t_gpu_upload, t_expert_gpu_upload - b.t_expert_gpu_upload,
                t_gpu_download - b.t_gpu_download,
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
            t_disk_read.load(), t_host_alloc.load(), t_gpu_upload.load(), t_expert_gpu_upload.load(),
            t_gpu_download.load(),
            t_gpu_compute.load(), t_cpu_expert.load(),
            t_cpu_dequant.load(), t_cpu_dot.load(),
            t_attn.load(),
            t_attn_preamble.load(),
            t_attn_norm.load(), t_attn_qproj.load(), t_attn_kvproj.load(),
            t_attn_absorb.load(), t_attn_score.load(), t_attn_outproj.load(),
            t_dense_gpu_up.load(), t_dense_gpu_compute.load(), t_dense_gpu_down.load(),
            t_moe_route.load(), t_moe_expert.load(), t_prefetch.load(), t_dense_load.load(), t_consume_staged.load(),
            t_staging_regrow.load(), t_staging_mutex_wait.load(), t_cache_mutex_wait.load(),
            t_active_layer_upload.load(), t_active_layer_promote_copy.load(), t_vram_demote_copy.load(),
            t_hold_for_compute.load(), t_release_hold.load(), t_find_active_expert.load(),
            t_find_expert.load(), t_evict_lru.load(), t_upload_expert_async.load(),
            t_ram_get_expert.load(), t_ram_get_experts_batch.load(),
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
            "absorb=%.2f score=%.2f outproj=%.2f] route=%.2fs "
            "expert=%.2fs (gpu=%.2fs [up=%.2fs cmp=%.2fs dn=%.2fs] cpu=%.2fs [dequant=%.2fs dot=%.2fs] "
            "untimed=%.2fs [active_up=%.2fs promote_copy=%.2fs vram_demote=%.2fs] "
            "scan=%.2fs [hold=%.2fs release=%.2fs find_active=%.2fs] promote_up=%.2fs "
            "ram_scan=%.2fs [find_expert=%.2fs evict_lru=%.2fs upload_async=%.2fs get_expert=%.2fs get_batch=%.2fs] "
            "lock_wait=%.2fs) "
            "dense_gpu=%.2fs [up=%.2fs cmp=%.2fs dn=%.2fs calls=%llu] "
            "disk=%.2fs  cache=%.0f%% hit  expert_calls=%lluGPU/%lluCPU "
            "pipe=%llu(pin_skip=%llu pin=%llu)\n",
            label, n_tokens, wall_ms / 1000.0, tok_s,
            d.t_attn,
            d.t_attn_preamble,
            d.t_attn_norm, d.t_attn_qproj, d.t_attn_kvproj, d.t_attn_absorb, d.t_attn_score, d.t_attn_outproj,
            d.t_moe_route, d.t_moe_expert,
            gpu_round_trip, d.t_expert_gpu_upload, d.t_gpu_compute, d.t_gpu_download, d.t_cpu_expert,
            d.t_cpu_dequant, d.t_cpu_dot,
            untimed_total, d.t_active_layer_upload, d.t_active_layer_promote_copy, d.t_vram_demote_copy,
            scan_total, d.t_hold_for_compute, d.t_release_hold, d.t_find_active_expert, promote_up,
            ram_scan_total, d.t_find_expert, d.t_evict_lru, d.t_upload_expert_async,
            d.t_ram_get_expert, d.t_ram_get_experts_batch,
            d.t_cache_mutex_wait,
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
                     1000.0 * t_expert_gpu_upload / gcalls, 1000.0 * t_gpu_compute / gcalls,
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
        if (n_pipe_hits.load() > 0) {
            uint64_t promoted = n_pipe_staged_later_promoted.load(), pipe_hits = n_pipe_hits.load();
            LOG_INFO("  PIPE->VRAM later:  %lu/%lu (%.1f%%) PIPE-served experts later confirmed-promoted "
                     "to VRAM via a real subsequent hit (no more unconfirmed speculative promotion)\n",
                     (unsigned long)promoted, (unsigned long)pipe_hits,
                     pipe_hits > 0 ? 100.0 * promoted / pipe_hits : 0.0);
        }
        {
            uint64_t pf_miss = n_expert_cache_miss_prefill.load(), dc_miss = n_expert_cache_miss_decode.load();
            uint64_t pf_hit = n_pipe_hits_prefill.load(), dc_hit = n_pipe_hits_decode.load();
            if (pf_miss + dc_miss > 0) {
                LOG_INFO("  PIPE by phase:     prefill %lu/%lu served (%.1f%%)  |  decode %lu/%lu served (%.1f%%) "
                         "-- decode-only rate isolates whether the single-token pilot design can help at all here\n",
                         (unsigned long)pf_hit, (unsigned long)pf_miss,
                         pf_miss > 0 ? 100.0 * pf_hit / pf_miss : 0.0,
                         (unsigned long)dc_hit, (unsigned long)dc_miss,
                         dc_miss > 0 ? 100.0 * dc_hit / dc_miss : 0.0);
            }
            uint64_t still_queued = n_pipe_miss_still_queued.load();
            uint64_t no_match = n_pipe_miss_no_match.load();
            if (no_match > 0) {
                LOG_INFO("  no-match detail:   %lu/%lu still had a hint waiting unpopped in the prefetch queue "
                         "(worker-thread throughput gap) vs %lu with no trace at all (prediction/pool-size gap)\n",
                         (unsigned long)still_queued, (unsigned long)no_match,
                         (unsigned long)(no_match - std::min(no_match, still_queued)));
            }
        }
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
                         "Pilot recall/PIPE lines above, since these share the same queue)\n",
                         (unsigned long)pw_att, (unsigned long)pw_ok, (unsigned long)pw_fail,
                         t_prefill_warm_time.load(), (unsigned long)pw_hint);
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
            }
        }
        LOG_INFO("VRAM->RAM demote:    %lu salvaged / %lu skipped (benign) / %lu failed "
                 "(vram_evict_demote_to_ram: instead of dropping an evicted expert's bytes, D2H "
                 "copy them back into the RAM tier before the VRAM slot is overwritten)\n",
                 (unsigned long)n_vram_evict_demoted.load(),
                 (unsigned long)n_vram_evict_demote_skipped.load(),
                 (unsigned long)n_vram_evict_demote_failed.load());
        LOG_INFO("Three-tier caching:  %lu active-layer uploads / %lu active-layer hits / %lu cleanups / "
                 "%lu promoted-to-main (MOVE vs COPY: %lu promotions freed RAM slots vs %lu left occupied)\n",
                 (unsigned long)n_active_layer_uploads.load(),
                 (unsigned long)n_active_layer_hits.load(),
                 (unsigned long)n_active_layer_cleanup.load(),
                 (unsigned long)n_active_layer_promoted.load(),
                 (unsigned long)n_move_on_promotion.load(),
                 (unsigned long)n_copy_on_promotion.load());
        LOG_INFO("Prefetch pickup:     avg %.2fms over %lu hints, peak queue backlog %lu "
                 "(large delay + small backlog => worker CPU-starved; growing backlog => worker throughput-bound)\n",
                 n_prefetch_hints_processed > 0
                     ? 1000.0 * t_prefetch_pickup_delay.load() / n_prefetch_hints_processed.load() : 0.0,
                 (unsigned long)n_prefetch_hints_processed.load(),
                 (unsigned long)n_prefetch_backlog_peak.load());
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
            if (pi > 0 || pns > 0 || pge > 0 || pnr > 0 || pnc > 0) {
                LOG_INFO("Disk-batch padding:  %lu issued (%lu ok / %lu failed), %lu later genuinely "
                         "hit (%.1f%% payoff), %.2f GB speculative bytes read, %lu no-free-slot skips\n",
                         (unsigned long)pi, (unsigned long)n_pad_reads_ok.load(),
                         (unsigned long)n_pad_reads_failed.load(), (unsigned long)n_pad_reads_hit_later.load(),
                         pi > 0 ? 100.0 * n_pad_reads_hit_later.load() / pi : 0.0,
                         bytes_disk_read_speculative.load() / 1e9,
                         (unsigned long)pns);
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
        LOG_INFO("Staged-unused evict: %lu (correctly predicted+read+staged experts recycled by round-robin "
                 "before any miss consumed them -- high means the staging pool is genuinely too small; low "
                 "despite a recall/served gap points elsewhere, e.g. the expert became a cache hit via "
                 "another path before ever reaching the miss check)\n",
                 (unsigned long)n_staged_evicted_unused.load());
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
