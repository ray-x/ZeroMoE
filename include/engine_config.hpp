// engine_config.hpp — Runtime configuration for MoEStream multi-GPU inference
// Loaded from TOML config files. Specifies model paths, GPU roles, memory layout.
#pragma once
#include "logger.hpp"
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <unordered_map>
#include <cstdlib>
#include <cuda_runtime.h>
#include <optional>

// Per-GPU configuration
struct GpuConfig {
    int device_id = -1;              // nvidia-smi GPU index
    std::string name;                // Human label, e.g. "NVIDIA GeForce RTX 3090"
    std::string role;                // "dense", "moe", "dense+moe"
    std::string model_path;          // GGUF file/dir for this GPU (optional override)
    double vram_gb = 0.0;            // Usable VRAM budget (0 = auto-detect)
    double dense_gb = 0.0;           // GB reserved for dense layers
    double kv_cache_gb = 0.0;        // GB reserved for KV cache
    int kv_bits = 4;                 // KV cache quantization (4 or 32)
    double moe_cache_gb = 0.0;       // GB reserved for expert LRU cache
};

// Full engine configuration
struct EngineConfig {
    // Model paths
    std::string model_path;          // Primary model GGUF (experts + architecture)
    std::string dense_model_path;    // Optional: separate GGUF for dense layers (e.g. 4-bit)

    // Generation parameters
    int context_size = 4096;         // Max context window (tokens)
    int max_new_tokens = 1024;       // Max tokens to generate per response
    float temperature = 0.7f;
    float top_p = 0.90f;
    bool drop_pages = false;         // fadvise DONTNEED after expert read
    bool cpu_only = false;           // force ALL compute to CPU (no GPU dispatch)

    // Phase 2 (moe_forward_gguf's S>1 cold-start VRAM warm-up) currently
    // loads each candidate expert SYNCHRONOUSLY, one at a time, inside the
    // per-layer critical path -- ram_mgr.get_expert() (which can trigger a
    // real, blocking disk read) followed by upload_expert_async. Measured
    // on a real run: 325 such warm-ups for an 11-token prompt, entirely
    // serialized into "Prefill (whole pass)"'s own wall-clock time. When
    // true, Phase 2 instead pushes a hint per candidate to the SAME
    // async_io_queue (LockFreePrefetcher) PILOT prediction already uses
    // for decode -- a cheap, non-blocking lock-free queue push -- and lets
    // the prefetcher's background worker thread(s) do the actual read +
    // promote, overlapped with whatever this layer/the next layer computes
    // next, instead of blocking this thread on it. Trade-off, not a strict
    // win: the legacy synchronous path GUARANTEES the expert is resident by
    // the time THIS layer's own Phase 3 dispatch runs a moment later; the
    // async version might not finish in time for that, falling through to
    // the ordinary (already-correct) miss path for this specific layer --
    // never a correctness issue, only a question of whether the aggregate
    // "stop blocking" win outweighs the lost same-layer guarantee. Default
    // false (unchanged legacy behavior) -- A/B test via t_prefill_warm_time
    // (should collapse toward ~0 when this is on) and overall Prefill
    // wall-clock before trusting it. See CLAUDE.md.
    bool prefill_warm_via_prefetcher = false;

    // Only meaningful when prefill_warm_via_prefetcher is also true --
    // separately gates the cross-layer opportunistic RAM->VRAM promotion
    // follow-up (LayerScratch::pending_prefill_promotions,
    // execution_layer.cpp) from the hint-pushing itself.
    //
    // REAL A/B, CONCLUSIVE, DEFAULT OFF (2026-08-12): four runs ON all
    // landed at prefill=0.56 tok/s; OFF landed at 0.60, matching the
    // pre-feature baseline. Root cause confirmed: this pass and Phase 2's
    // own hint-pushing loop compete for the same free-VRAM-slot budget --
    // running eager promotion first starves later layers of hints
    // entirely (n_prefill_hint_pushed nearly halved, 420 vs ~930 ON vs
    // OFF), losing the RAM-prefetch overlap the whole feature exists to
    // provide. See engine.hpp's mirror of this field for the full
    // writeup, and CLAUDE.md.
    bool prefill_opportunistic_promote = false;

    // Decode-time analogue of prefill's own upload_expert_active_async +
    // sync_active_layer_batch pre-warm (see execution_layer.cpp's Phase-3
    // ram_needed_k dispatch loop): batches a TOKEN's own real active-layer
    // VRAM misses into small groups of async H2D issues + one shared
    // cudaStreamSynchronize per group, instead of dispatch_expert_k's
    // existing fully-sequential blocking cudaMemcpy per miss. Real DMA-level
    // win confirmed via tests/bench_dma_sync_vs_async.cu on both GPUs in
    // this box (2-16%, biggest at small sizes on the faster link) -- see
    // that file's header comment.
    //
    // DEFAULT OFF, and NOT a simple reinstatement of the first attempt: an
    // earlier, unbounded version of this (batching a whole token's ~8
    // candidates in one shot) caused a REAL production incident on real
    // hardware -- host RAM exhaustion, swap thrashing, and an eventual
    // freeze (SLOW-HOST-ALLOC times climbing past 1s, SLOW-PICKUP prefetch
    // backlog exploding to 30+ seconds, matching a classic memory-pressure
    // death spiral). No confirmed single root-cause line was found in the
    // recovered debug.log (specifically: no "[RAM-CLAIM] ... falling back
    // to slot 0" ever fired, ruling out the first suspected mechanism), but
    // one real code bug WAS found and fixed as part of re-adding this
    // (upload_expert_active_async was being called without checking
    // slab_capacity > 0, only slab_buffer != nullptr -- see the "[GPU]
    // Invalid upload_expert_active_async parameters" ERROR line that
    // incident's log showed). Given host_ram_gb is commonly tuned close to
    // physical RAM on this project's own real hardware (e.g. 120GB of
    // 128GB, ~8GB margin -- see CLAUDE.md's GLM-5.2 settled config), even a
    // small, otherwise-harmless shift in PEAK simultaneous in-flight memory
    // (several slab-buffer resizes clustering together under batching,
    // instead of serializing one-at-a-time the way the old fully-sequential
    // sync path forced) is plausible as a contributing factor on a config
    // running that close to the ceiling. decode_async_prewarm_max_batch
    // (below) exists specifically to bound that peak footprint;
    // decode_async_prewarm itself defaults OFF so this can be re-validated
    // deliberately (start small: low max_batch, generous host_ram_gb
    // headroom, watch SLOW-HOST-ALLOC/SLOW-PICKUP) rather than silently
    // re-enabled by an upgrade. UNVERIFIED ON REAL HARDWARE as of this
    // writing -- the safer, capped version has not yet been tested end to
    // end on a real run.
    bool decode_async_prewarm = false;

    // Caps how many of a token's active-layer VRAM misses get batched into
    // ONE async-issue-then-sync group at a time when decode_async_prewarm
    // is on -- e.g. 8 real misses become ceil(8/3)=3 groups of <=3 instead
    // of one group of 8. Directly bounds peak simultaneous in-flight
    // footprint (RAM slots pinned + active-layer slots mid-DMA at once) at
    // the cost of losing some of the batching win for tokens with more
    // misses than this -- a deliberate trade after the unbounded version's
    // real incident (see decode_async_prewarm's own comment). 3 chosen to
    // roughly match tests/bench_dma_sync_vs_async.cu's own tested n=4
    // (closest tested value that's still conservatively smaller), not
    // derived from any real per-layer-miss-count data on this specific
    // knob -- a reasonable starting guess, not a tuned value.
    int decode_async_prewarm_max_batch = 3;

    // MLA absorbed-attention qabsorb (Q_absorbed = q_nope @ WUK) BF16 batched
    // GPU fast-path (CudaManager::upload_wuk_bf16/qabsorb_bf16_batched_host,
    // cuda_manager.hpp), replacing the CPU per-head matmul() loop
    // (attention_layer.cpp) with one cublasGemmStridedBatchedEx call across
    // all H heads. Real motivation: qabsorb measured at ~60ms/decode-token
    // on GLM-5.2 real hardware (see CLAUDE.md's decode-vs-prefill attn
    // breakdown), 100% CPU with zero GPU path today -- WUK/WUV are cached
    // F32 host vectors (ensure_wuk_wuv_cached), never dispatched through
    // dense_proj's existing quant/GPU pipeline at all.
    //
    // Deliberately per-model/hardware configurable, not a blanket default,
    // per direct user observation: DeepSeek-V4 (dsv4_attention_gguf, the
    // structurally SEPARATE raw/SWA MLA path with no absorption at all --
    // no WUK/WUV, this flag has literally no effect there) already runs its
    // CPU attention path much faster relative to GLM-5.2's absorbed
    // mla_attention_absorbed_gguf, so a GPU win measured on one model/GPU
    // pairing should not be assumed to generalize -- always re-verify with a
    // real same-prompt A/B (compare CPU-only decode tok/s against this flag
    // on, watching the "MLA qabsorb BF16 batched GPU path" shutdown report
    // line for real call counts / fallback rate) before trusting it for a
    // given model+hardware combination. Default OFF (matches this project's
    // standing convention for anything not yet real-hardware-verified).
    bool attn_qabsorb_gpu = false;

    // Per-layer VRAM budget cap for attn_qabsorb_gpu's one-time WUK BF16
    // upload (warm_mla_absorb_cache, attention_layer.cpp) -- a real,
    // measured free-VRAM query (cudaMemGetInfo) at startup, not an assumed
    // headroom figure, is still capped by this ceiling so the upload can
    // never eat into VRAM some OTHER subsystem (expert cache, dense pool)
    // was counting on, even on a box with lots of apparent free VRAM at the
    // moment this particular warmup step runs. A layer whose WUK doesn't
    // fit under the remaining budget just stays on the CPU qabsorb path
    // (n_qabsorb_gpu_fallback_cpu) -- never a hard failure.
    double attn_qabsorb_gpu_max_vram_gb = 2.0;

    // attn_score_gpu: 3-GEMM + softmax kernel replacing the CPU AVX-512
    // per-head loop for MLA's score phase (QK^T-over-latent + softmax +
    // weighted-V, mla_attention_absorbed_gguf's attn_score block --
    // attention_layer.cpp). Structurally simpler than qabsorb: MLA's latent
    // K/V (Lc/Rc) is SHARED across every head, so this collapses into 3
    // PLAIN (non-batched) GEMMs -- no cublasGemmStridedBatchedEx needed.
    // Real isolated bench at GLM-5.2's actual dims (H=64 kvl=512 rope=64,
    // tests/bench_score_cpu_vs_gpu.cu): 5x-51x speedup, GROWING with
    // context length (CPU cost is O(T), GPU stays flat ~0.08-0.13ms/call
    // regardless of T) -- unlike qabsorb, whose cost is context-independent.
    // Decode (S=1) only: prefill (S>1) has a DIFFERENT past_context_len per
    // row within the same batch, out of scope for this first cut, and
    // decode is where score's unbounded-with-context cost actually bites.
    // Same DeepSeek-V4-vs-GLM-5.2 configurability rationale as
    // attn_qabsorb_gpu -- V4's dsv4_attention_gguf bounds this phase with a
    // sliding window (see CLAUDE.md), so it's already cheap there and this
    // flag has no reason to default on for that model. Default OFF (matches
    // this project's standing convention for anything not yet real-
    // hardware-verified end-to-end).
    bool attn_score_gpu = false;

    // Per-layer VRAM budget cap for attn_score_gpu's persistent KV latent
    // cache GPU mirror (d_lc_by_layer/d_rc_by_layer, cuda_manager.hpp),
    // allocated once at startup (warm_mla_absorb_cache) sized to
    // engine.max_t so it never needs a mid-generation realloc. Same
    // cudaMemGetInfo-checked, never-assume-headroom discipline as
    // attn_qabsorb_gpu_max_vram_gb. A layer whose mirror doesn't fit just
    // stays on the CPU score path (n_score_gpu_fallback_cpu) -- never a
    // hard failure. Note this budget is typically larger in absolute GB
    // than qabsorb's own (the mirror holds the WHOLE context, not one
    // layer's static weights) -- e.g. at context_size=4096, kv_lora=512:
    // 4096*512*4 bytes = 8.4MB just for Lc per layer, before Rc.
    double attn_score_gpu_max_vram_gb = 2.0;

    // Caps the KV latent cache GPU mirror's ALLOCATED size independently of
    // context_size/engine.max_t -- 0 (default) means "no cap, size to the
    // full context_size" (the original behavior). A nonzero value smaller
    // than context_size lets the mirror (and the WUV BF16 upload that
    // competes with it for the SAME attn_score_gpu_max_vram_gb budget pool)
    // use much less VRAM, at the cost of the score GPU path silently
    // falling back to CPU (score_gpu_batched_host's existing `T >
    // lc_rc_max_t_by_layer[layer_idx]` bounds check, already tested) for
    // any decode position beyond this cap -- the MODEL's real context_size
    // is completely unaffected, only how much of it gets the GPU speedup.
    // Added after a real production run on a VRAM-scarce GPU (RTX 5060 Ti,
    // 1.82GB free at this point in startup) showed qabsorb_gpu (0.98GB) +
    // the full-context_size=4096 mirror (0.74GB F32) left only 0.10GB for
    // WUV's own 1.22GB need (78 layers), so only 6/78 layers got WUV
    // coverage -- capping the mirror instead of shrinking context_size
    // (which would also cap the model's real usable conversation length)
    // recovers most of that VRAM for WUV without that downside. The real
    // ceiling on WUV coverage while qabsorb_gpu stays enabled is fixed by
    // WUV's own footprint alone (~50/78 layers = 64% on that same GPU, even
    // at cap=0 mirror) -- this can get close to that ceiling, not exceed
    // it; disabling attn_qabsorb_gpu is the only way to reach 78/78.
    int attn_score_gpu_max_context = 0;

    // Per-layer version of cpu_only: these specific MoE layers never
    // dispatch on GPU and their routed experts are never uploaded/promoted
    // into VRAM (no shadow-fast-path, no cold-batch/decode-miss synchronous
    // upload, no RAM-tier reactive-hit promotion, no startup VRAM pin/
    // preload) -- experts for these layers load into host RAM only and
    // always compute on CPU. Intended for layers whose expert set is so
    // large/flat that a real run spends far more wall-clock time on the
    // disk read than the matmul itself: for those layers a VRAM upload buys
    // nothing (the bottleneck never moves) while it costs real VRAM cache
    // capacity that layers where GPU residency DOES help could otherwise
    // use, plus real PCIe/DMA time that's pure overhead when compute was
    // always going to be I/O-bound. Empty (default) = no layers affected,
    // identical to today's behavior. See CLAUDE.md.
    std::vector<int> cpu_affinity_layers;

    // How many candidate experts the Phase 1 pilot router pushes as
    // speculative prefetch hints per layer transition (execution_layer.cpp's
    // pilot loop -- was a hardcoded `std::min(8, c.topk)`). Global default;
    // see prefetch_pilot_k_per_layer below for a per-layer override. Lower
    // this if [PREFETCH-LAYER]'s per-layer report shows a layer's pushed
    // hints are mostly evicted unused (staging pool churn) or mostly
    // already-resident (n_pilot_pred_already_* -- no read needed at all) --
    // in both cases the hint bought nothing but still cost a worker-thread
    // read + a staging slot that a DIFFERENT layer's genuinely useful hint
    // could have used instead.
    int prefetch_pilot_k = 8;

    // Per-layer override for the above, one entry per layer index (e.g.
    // `prefetch_pilot_k_per_layer = [8,8,8,0,0,4,...]`), reusing the same
    // `[int]`-array TOML parser cpu_affinity_layers uses. A layer whose
    // index is out of range, or whose entry is negative, falls back to
    // prefetch_pilot_k; entries past the model's real layer count are
    // ignored. Empty (default) = every layer uses prefetch_pilot_k
    // uniformly, identical to today's behavior. Intended for layers whose
    // [PREFETCH-LAYER] report shows the pilot's hints there are rarely
    // "really required" (see that report's hint-outcome line) -- set those
    // layers lower (or 0, disabling prefetch for them entirely) to free up
    // staging-slot and worker-thread bandwidth for layers where prefetch
    // actually pays off, without needing a single global value that fits
    // every layer's very different routing concentration equally badly.
    std::vector<int> prefetch_pilot_k_per_layer;

    // How many layers ahead the pilot predicts AND pushes real prefetch
    // hints for. 1 (default) = only l+1, the original production pilot
    // (measured ~74% fetch-worthy recall on GLM-5.2/UD-IQ2_M). 2 = ALSO
    // predicts and hints l+2, reusing the SAME vector the l+1 pilot
    // already computes from (approach "(a)" -- free, one more gate pass,
    // no extra compute; an FFN-reconstruction alternative was measured
    // statistically tied with this despite costing real compute and was
    // removed, see execution_layer.cpp's pilot loop). ADDITIVE, not a
    // replacement: l+1 hints are always predicted and pushed first, l+2
    // hints (governed by prefetch_pilot_k_depth2 below) are pushed after,
    // on top -- since the shared prefetch queue is a FIFO ring buffer,
    // this naturally gives l+1's more-accurate hints first claim on queue
    // capacity. The l+2 MEASUREMENT (recall counters, "Pilot recall by
    // rank" report) always runs regardless of this setting -- it's cheap
    // (one gate pass) -- only the real push_hint/eviction-guard behavior
    // is gated by it, so you can leave this at 1 and still see whether
    // depth-2 would be worth enabling before ever turning it on. Valid
    // values: 1 or 2; anything else falls back to 1 with a startup
    // warning. UNVERIFIED ON REAL HARDWARE for wall-clock impact: l+2's
    // recall (~55% fetch-worthy vs l+1's ~74%) is measured, but whether
    // enabling this actually improves throughput -- versus adding queue
    // contention or spending prefetch bandwidth on the ~45% wrong
    // guesses -- has not been A/B'd on a real run yet.
    int prefetch_pilot_depth = 1;

    // How many of l+2's predicted top-K (ranked by the pilot's own
    // confidence score -- rank 0 is the single highest-confidence pick,
    // see the "Pilot recall by rank (DEPTH-2...)" shutdown report) are
    // actually pushed as real prefetch hints. Independent of
    // prefetch_pilot_k (which governs l+1 only) -- l+2's own measured
    // accuracy is meaningfully lower, so a smaller cutoff here often
    // makes sense once you've seen your own per-rank curve; fetching the
    // low-confidence tail ranks costs the same prefetch bandwidth as the
    // high-confidence ones for a much lower hit rate. Only takes effect
    // when prefetch_pilot_depth = 2. Defaults to 8 (same ceiling as
    // prefetch_pilot_k) -- lower it once the per-rank report shows where
    // your model's own curve drops off.
    int prefetch_pilot_k_depth2 = 8;

    // O_DIRECT expert-slab reads (Linux only). Opt-in, requires the model
    // to have been repacked with tools/gguf_tool pack (or otherwise have
    // general.alignment >= 4096) -- see GGUFFile::enable_direct_io. A
    // plain/un-repacked GGUF silently falls back to buffered reads with a
    // logged warning, never a hard failure.
    bool direct_io = false;

    // NVIDIA cuFile (GPUDirect Storage): DMA a genuine "double miss" expert
    // (not VRAM-resident AND not host-RAM-resident -- see
    // RAMSafetyManager::is_ram_resident) straight from NVMe into a VRAM
    // cache slot, skipping the host-RAM hop entirely. Requires direct_io=true
    // (cuFile needs the SAME O_DIRECT file descriptor direct_io already
    // opens -- validated below, not silently assumed) AND the machine
    // actually having NVIDIA GPUDirect Storage + the nvidia-fs kernel module
    // installed (a SEPARATE install from the base CUDA Toolkit -- see
    // CMakeLists.txt's MOESTREAM_HAVE_CUFILE detection). Requesting this
    // without GDS installed (or on a non-Linux/no-cuFile build) logs a
    // warning and silently falls back to the existing RAM-mediated upload
    // path -- never a hard failure, same contract as direct_io/io_uring.
    // If an expert IS already RAM-resident, this is never used for it
    // regardless of this setting -- cuFile only ever replaces a disk read
    // that would otherwise happen anyway, never a RAM->VRAM promotion of
    // data already in host memory. See CLAUDE.md.
    bool cuda_cufile = false;

    // The model directory was produced by `gguf_tool pack --unpack`: every
    // routed expert's gate/up/down are three SEPARATE tensors
    // (blk.N.ffn_{gate,up,down}.E.weight) instead of one shared packed
    // _exps tensor per layer, but laid out physically TIGHT and
    // CONTIGUOUS per expert (gate immediately followed by up immediately
    // followed by down, no padding -- see tools/gguf_pack.cpp's own
    // layout pass) so the whole expert can be read with ONE combined DMA
    // request instead of three scattered ones. This is NOT auto-detected
    // from the tensor names alone -- a plain, non-repacked per-expert GGUF
    // uses the exact same tensor-naming convention but has no contiguity
    // or alignment guarantee at all, so treating it as this fast layout
    // would read garbage across whatever gap actually exists between
    // projections. Setting this true makes the engine ASSUME the
    // contiguous-and-aligned invariant holds and validate it explicitly at
    // startup (StreamingEngine::sanity_check_unpacked_format, called right
    // after the model graph is built) -- any violation is a FATAL error
    // printed to stderr and the run refuses to start, rather than silently
    // falling back to the slow per-projection path. Independent of
    // direct_io/io_uring (compose with either); has no effect at all on a
    // model using the default packed _exps format (--align-only, the
    // recommended default) -- leave this false for that format. See
    // CLAUDE.md's "Offline GGUF Tool" section for why --unpack alone,
    // without this flag, previously made performance WORSE rather than
    // better (it silently disabled both the VRAM-shadow fast path and
    // io_uring batching, which are gated by the packed _exps tensor name).
    bool unpacked_expert_format = false;

    // Batch the async prefetcher's per-expert reads (up to 3: gate/up/down)
    // through io_uring instead of sequential blocking preads (Linux only,
    // kernel >= 5.6 for IORING_OP_READ). Opt-in, independent of direct_io --
    // composes with it either way (see io_uring_backend.hpp). Falls back to
    // the existing sequential path with a logged warning on any failure
    // (old kernel, io_uring disabled by seccomp/container policy, etc.),
    // never a hard failure. Prefetcher-only: does not touch the synchronous
    // decode-thread miss path (StreamingEngine::load_expert_slab).
    bool io_uring = false;

    // Bounds the io-wq kernel worker pool each io_uring Ring registers
    // (IORING_REGISTER_IOWQ_MAX_WORKERS, io_uring_backend.hpp::Ring::
    // register_iowq_workers) -- only meaningful together with IOSQE_ASYNC
    // (unconditionally set on every submitted read; see that flag's own
    // comment), which is what actually forces reads onto io-wq instead of
    // letting them execute inline inside io_uring_enter(). This is a real
    // OS thread count, not free: colibri's own reference implementation
    // (a separate engine, c/uring.h/glm.c) defaults to 8 (PIPE_WORKERS),
    // reflecting where most consumer/prosumer NVMe drives reach near-peak
    // throughput for reads this size (~4-13MB) -- going much higher rarely
    // helps further and starts competing for CPU with everything else this
    // process already runs (omp_max_threads' own OMP team, expert_fetch_
    // threads, prefetch_worker_threads) since io-wq workers are a SEPARATE
    // kernel-managed pool, not counted against omp_max_threads' budget.
    // Applied to the decode thread's own ring (RAMSafetyManager::
    // init_decode_uring) directly; each prefetch worker's own ring stays
    // fixed at 3 (every active_prefetch() submission is exactly one
    // hint's gate/up/down, never more, so more workers there than 3
    // couldn't parallelize anything further regardless of this setting).
    // 2026-08-14: batching multiple hints into one prefetch-worker
    // submission (PREFETCH_BATCH_HINTS) was tried and reverted -- real A/B
    // showed zero blended-throughput change and a real phase_c regression
    // (head-of-line blocking). See async_prefetcher.hpp's top-of-file
    // comment for the full finding before re-attempting this.
    int io_uring_workers = 8;

    // Pad Phase B's real cache-miss submission (RAMSafetyManager::
    // get_experts_batch) with speculative reads of historically-hot but
    // not-currently-resident experts for the SAME layer, riding the SAME
    // already-necessary io_uring submission real misses trigger -- a real
    // hardware benchmark (tools/gguf_dma.cpp) measured disk throughput
    // climbing from ~2.1 GB/s at 1 concurrent request-group to ~2.5-2.6
    // GB/s at 8-16, while real production logs show the actual batch size
    // reaching Phase B averages only ~2.7 experts/call -- this pads toward
    // disk_batch_padding_target without the bandwidth contention a
    // SEPARATE competing prefetcher thread already caused (see CLAUDE.md's
    // prefetcher investigation). Requires io_uring=true (validated below).
    // NEVER fires when there's no real cache miss this call (n_pending==0)
    // -- only ever enlarges an already-necessary disk trip, never creates
    // a new one purely for speculation. V1 only ever claims genuinely FREE
    // RAM slots (expert_id==-1) -- it never evicts a real resident, so it
    // carries zero cache-quality regression risk, at the cost of likely
    // engaging mostly during a run's early ramp-up window before the RAM
    // tier saturates (see n_pad_no_free_slot in the shutdown report).
    // Default false (opt-in, unverified on real hardware).
    bool disk_batch_padding = false;

    // Target TOTAL experts (real+speculative) per Phase B submission.
    // Clamped in validate() to [1,21] -- 21 = floor(io_uring_backend::
    // Ring::CAPACITY(64) / 3), the worst-case (packed-format, 3 requests
    // per expert) bound that keeps total requests within the ring's hard
    // capacity regardless of packed/unpacked format -- the same constant
    // tools/gguf_dma.cpp's own --experts sweep clamp uses, for the
    // identical reason.
    int disk_batch_padding_target = 8;

    // V1 of disk_batch_padding (above) only ever claims genuinely FREE RAM
    // slots -- a real, confirmed limitation once the RAM tier saturates: a
    // real GLM-5.2 run showed 9138 "no-free-slot" skips against only 8
    // successful pads in one request, because a RAM tier sized to stay
    // under a fixed host_ram_gb budget runs essentially full almost the
    // entire time, not just during early ramp-up. This flag lets padding
    // ALSO reclaim a RAM slot whose expert is already VRAM-resident
    // (also_in_vram==true) once free slots run out, instead of giving up.
    // Safe for the SAME reason claim_or_wait's own real-miss victim scoring
    // already treats also_in_vram slots as score=0 (the cheapest possible
    // eviction, see eviction_score's own comment): decode's real dispatch
    // resolves VRAM residency FIRST (GpuTopology::find_expert), before ever
    // consulting the RAM tier, so this expert's real dispatch doesn't touch
    // this RAM slot at all while it stays VRAM-resident -- the RAM copy is
    // provably redundant at the moment of eviction. The one residual risk
    // (VRAM independently evicts its own copy later, and a FUTURE RAM-tier
    // lookup then misses and re-fetches from disk, having lost the RAM
    // copy here) is the exact same tradeoff a real miss's own victim
    // selection already accepts today -- padding is a strictly lower-
    // priority, purely speculative consumer of that same tradeoff, not a
    // new risk class. Default false (opt-in, unverified on real hardware --
    // needs a real A/B against disk_batch_padding_evict_dup=false on the
    // same prompt/config before treating this as a default-on win).
    bool disk_batch_padding_evict_dup = false;

    // Number of concurrent async prefetcher worker threads pulling hints
    // off the SPSC queue and servicing them (disk read + optional async
    // H2D promotion). Was hardcoded to 1 -- real-run prefetch instrumentation
    // (n_prefetch_miss_still_queued) showed a real, measurable fraction of
    // otherwise-correctly-predicted misses failing to be served purely
    // because a single worker thread hadn't even STARTED that hint's read
    // yet (pilot_k=8 hints pushed per layer transition, one thread
    // servicing them strictly one at a time). Each thread gets its own
    // io_uring ring when io_uring=true (a Ring is not shareable across
    // threads). Each worker claims/populates/publishes real RAM-tier slots
    // directly via RAMSafetyManager::claim_or_wait (see async_prefetcher.
    // hpp's active_prefetch) -- there is no separate staging pool anymore
    // (see below), so this only controls how many concurrent speculative
    // reads can be in flight at once, not any pool-sizing tradeoff. Kept
    // modest by default -- these threads mostly block on I/O rather than
    // compute, but more of them means more concurrent ram_mutex/cache_mutex
    // contention too.
    int prefetch_worker_threads = 2;

    // NOTE: this used to also have a `prefetch_staging_slots` knob sizing a
    // separate pinned-host round-robin staging pool the prefetcher held
    // prefetched-but-not-yet-consumed expert slabs in. That whole pool was
    // removed when the prefetcher was unified with the RAM-tier cache
    // (RAMSafetyManager::claim_or_wait) -- a prefetch worker now writes
    // straight into a real RAM-tier slot, the exact same structure the
    // decode thread's own cache hits read from, so there is no second pool
    // left to size. See CLAUDE.md for the investigation that motivated the
    // unification (the old pool's own round-robin eviction was shown to be
    // a secondary, structural waste source independent of prediction
    // accuracy) and ram_manager.hpp's claim_or_wait for the design that
    // replaced it.

    // Host RAM (DDR) budget for the MoE expert LRU cache, in GB.
    // 0 = auto-detect (88% of currently-available system memory).
    double host_ram_gb = 0.0;

    // Persistent RAM-slab pinning budget, in GB. 0 (default) = OFF, every
    // slab stays plain pageable memory (pre-2026-08-15 behavior).
    //
    // What it buys: cudaMemcpyAsync from UNPINNED host memory cannot
    // actually run asynchronously -- the CUDA driver stages it through its
    // own internal pinned buffer, synchronously, on the calling thread.
    // That is exactly what made upload_expert_active_async's supposedly
    // "small/cheap, no wait" issue call cost 18.96s across ~13.9k calls on
    // a real GLM-5.2 run. Registering each slab ONCE (here) rather than per
    // transfer lets that DMA overlap real compute: measured 0.15s issue +
    // 5.26s batch-sync when the source was pinned, vs 18.96s blocking when
    // it wasn't -- projected ~14s/request, 0.57 -> ~0.72 tok/s.
    //
    // Why a BUDGET rather than just pinning everything: pinned pages can
    // never be reclaimed by the kernel, and host_ram_gb is routinely tuned
    // close to physical RAM (120 of 125GB on the box this was developed on
    // -- see host_ram_gb's own freeze-incident history). Pinning does not
    // allocate anything new (these pages are already resident), so the risk
    // is reduced OS headroom, not extra consumption -- but that is still a
    // real risk worth capping and stepping up deliberately rather than
    // enabling wholesale. Start well below host_ram_gb (e.g. 40), confirm
    // stability and the [REQUEST] line's own active_layer_prewarm_issue=
    // drop, then raise. Registration costs ~11 GB/s, one time, during
    // preload -- a large budget adds seconds to startup, not to any request.
    double ram_pinned_budget_gb = 0.0;

    // Optional SECOND physical copy of the same model, on a DIFFERENT drive
    // (e.g. "/data/models/UD-IQ2_M_align" while model_path is on another
    // NVMe). Empty (default) = single-drive behavior, unchanged.
    //
    // Expert reads dominate both startup preload and decode-time stalls, and
    // one NVMe saturates around 2.2-2.5 GB/s for this model's real access
    // shape. Two independent devices have independent queues, so striping is
    // additive bandwidth, not a redistribution of one ceiling -- measured on
    // the development box: primary 2.25-2.52 GB/s, mirror 1.98-2.15 GB/s,
    // combined ceiling ~1.85x. The copy must be byte-identical (all reads use
    // the PRIMARY's tensor offsets); GGUFModel::open_mirror enforces this with
    // a per-shard size check and refuses to stripe any shard that differs.
    std::string mirror_model_path;

    // Log every host RAM cache HIT for the MoE expert cache (high volume —
    // one line per routed expert per token). MISS/LOAD and VRAM PROMOTE/READY
    // events are always logged regardless of this flag since they're rare.
    bool verbose_moe_log = false;

    // How many host-cache re-hits before triggering an async H2D promotion
    // into the GPU expert cache. Lower values promote earlier (lower latency),
    // higher values reduce unnecessary PCIe traffic.
    int promote_hit_threshold = 1;

    // Phase A: pin a VRAM-resident expert (never evict it via
    // evict_lru) once its host-cache hit_count crosses this threshold --
    // "stop evicting hot experts within a conversation", no cross-prompt
    // profile needed. 0 disables pinning entirely (pure reactive LRU).
    int pin_threshold = 8;

    // Cross-conversation VRAM residency: at startup, read this usage file
    // (written by StreamingEngine::dump_usage at the end of a previous run)
    // and preload the top pin_topk_per_layer hottest experts per layer into
    // VRAM, pinned. Empty path disables preload (a usage file is still
    // written at shutdown for a future run). pin_topk_per_layer=0 auto-sizes
    // to fill available VRAM expert-cache capacity instead of a fixed
    // per-layer count.
    std::string pin_file = ".moestream_usage";
    int pin_topk_per_layer = 0;

    // Separate usage file tracking ONLY prefill-phase (S > 1) routing
    // selections -- see StreamingEngine::route_freq_prefill's own comment
    // for why decode's higher routing volume can crowd out prefill's own
    // needs in the combined pin_file histogram. Empty (default) = not
    // used; startup preload falls back entirely to pin_file's ranking for
    // every layer. When set, preload prefers this file's ranking for any
    // layer it has data for, falling back to pin_file only for layers this
    // file has no entries for (e.g. its very first run, before any
    // prefill has been recorded into it yet). Written at shutdown
    // alongside pin_file, same accumulate-across-sessions semantics.
    std::string prefill_pin_file;

    // Debug/A-B-testing aid: when true, skip the shutdown writes to
    // pin_file AND prefill_pin_file entirely -- both files are still READ
    // at startup (preload behaves exactly as configured), just never
    // overwritten. Without this, both usage files accumulate additively
    // on every single run (see dump_usage_histogram's own comment) --
    // confirmed on real hardware to genuinely drift the startup pin/
    // preload ranking run-to-run even under a literally identical,
    // temperature=0 prompt (two runs a few minutes apart logged
    // .moestream_usage growing from 12958 to 12988 known experts, +45 on
    // the single hottest entry's count -- GPU reduction order isn't
    // bit-exact call-to-call, and this model's IQ2_XXS routing margins are
    // thin enough for that jitter to occasionally flip a close top-8 tie).
    // That drift is real signal to accumulate across genuinely different
    // sessions, but it's exactly the wrong thing to have moving underneath
    // you mid-A/B when isolating a SINGLE variable (e.g. a pin-merge logic
    // change) -- set this true, keep re-running against the same frozen
    // files, and only the variable actually under test can explain a
    // result. Default false (unchanged accumulate-forever behavior).
    bool freeze_usage_stats = false;

    // How the pinned_vram_capacity budget gets DISTRIBUTED across layers at
    // startup preload. Two options, both driven by the same pin_file usage
    // history -- this only changes which (layer,expert) entries win a
    // pinned slot, not how many total slots exist:
    //   "per_layer_topk" (default): every layer gets the SAME topk count
    //     (pin_topk_per_layer, or pinned_vram_capacity/n_layers if that's
    //     smaller) -- the long-standing behavior, unchanged. Safe default:
    //     never leaves any layer with zero protection, at the cost of
    //     wasting pin slots on layers whose routing is close to uniform
    //     (see the [PIN-CONCENTRATION] log -- a layer near the low end of
    //     that coverage metric gains almost nothing from a fixed top-K).
    //   "global_topn": pool EVERY (layer,expert,count) triple across ALL
    //     layers into one ranking, sorted by raw count descending, and pin
    //     greedily in that order until pinned_vram_capacity is spent --
    //     valid specifically because every layer sees the same number of
    //     routing decisions per token in this engine (one top-k selection
    //     per layer per token), so raw counts genuinely are comparable
    //     across layers, unlike e.g. comparing byte sizes of differently-
    //     shaped tensors. Concentrates the budget on genuinely hot experts
    //     (whichever layer they're in) instead of spreading it evenly, but
    //     a layer whose historical traffic is flat can end up with FEWER
    //     than pin_global_floor_per_layer pins purely because its top
    //     experts never out-rank other layers' hotter ones globally.
    // Kept switchable (not a hard replacement) because a different model,
    // or a very different prompt distribution than whatever wrote the
    // current pin_file, could have a flatter or more skewed routing profile
    // than what motivated adding "global_topn" -- see CLAUDE.md.
    std::string pin_strategy = "per_layer_topk";
    // Only used when pin_strategy=="global_topn": every layer with at least
    // this many known (layer,expert) entries in pin_file is GUARANTEED at
    // least this many pinned slots, taken off the top of pinned_vram_capacity
    // before the remainder is spent on the global ranking -- so a flat
    // layer never drops to fully-zero protection just because none of its
    // experts out-rank other layers' hottest ones. 0 disables the floor
    // (a pure global ranking, some layers may get zero pins).
    int pin_global_floor_per_layer = 1;

    // Multi-tier startup preload, extending the single top-K-pinned tier
    // above into a graduated scheme driven by the SAME usage file (which
    // already records every expert ever routed to, not just the top-K --
    // StreamingEngine::route_freq/dump_usage cover the full histogram).
    // Per layer, ranked by historical selection count, AFTER the
    // pin_topk_per_layer-sized pinned prefix:
    //   next round(pin_vram_lru_fraction * avg_vram_slots_per_layer) ranks
    //     -> VRAM, but NOT pinned -- a regular reactively-evictable slot,
    //     just pre-warmed instead of starting cold;
    //   next round(pin_ram_fraction * ram_mgr.slots_per_layer) ranks ->
    //     host RAM tier, preloaded (a genuinely new capability -- the RAM
    //     tier previously always started completely cold, filling only
    //     reactively during real generation).
    double pin_vram_lru_fraction = 0.0;
    double pin_ram_fraction = 0.0;

    // Three-tier VRAM caching configuration
    // Tier overlap fix: MOVE instead of COPY on RAM→VRAM promotion
    bool move_on_promotion = true;
    // Only consulted when move_on_promotion=false (COPY semantics). 0.0
    // (default) preserves plain COPY behavior: a RAM slot stays occupied
    // for as long as reactive eviction pressure allows, relying entirely
    // on also_in_vram-based scoring to reclaim it once some OTHER expert's
    // miss needs the slot -- which may never happen for a lightly-trafficked
    // layer, on a long enough session. Set to a fraction in (0.0, 1.0] to
    // bound that: once a layer's RAM-tier occupancy (filled slots /
    // cap_for_layer(layer)) reaches this threshold, a newly-CONFIRMED
    // promotion (finish_sync_promotion, called the instant a sync VRAM
    // upload lands) is proactively reclaimed immediately -- same field-
    // clearing as MOVE, not a new mechanism -- instead of being left
    // occupied-but-redundant. Gives graceful degradation on a tighter
    // host_ram_gb budget (e.g. a bigger model, or less RAM) without fully
    // reverting to MOVE and losing COPY's own real, measured throughput win
    // (see CLAUDE.md) on setups with enough headroom to not need this at
    // all. Only ever applied at the ONE call site verified safe for it
    // (execution_layer.cpp's finish_sync_promotion sites, confirmed to
    // never read slab_buffer/host_ptr fields again in the same call after
    // this fires) -- deliberately NOT applied to do_promote_ram_slot_to_vram's
    // own COPY branches, which are reached from a different, NOT
    // confirmed-safe context (see CLAUDE.md for the correctness reasoning).
    double ram_copy_reclaim_threshold = 0.0;
    // Maximum number of permanently pinned experts (was ~600, targeting 1024)
    int pinned_vram_capacity = 1024;
    // Active layer area: temporary workspace for current-layer experts
    int active_layer_capacity = 256;
    bool use_active_layer_area = true;
    // Hit count (while resident in the active layer area, i.e. re-accessed
    // before this layer's window closes) at which an active-layer expert
    // gets promoted into the real VRAM LRU cache (GpuTopology::
    // promote_active_to_main, a device-to-device copy) instead of just
    // being freed once its layer finishes. Previously this whole promotion
    // path didn't exist at all -- the active layer area only ever held a
    // temporary hold-and-discard buffer with no way for a proven-hot
    // expert to escape into the real cache, and no hit-counting to know
    // "proven hot" in the first place (n_active_layer_hits was declared but
    // never incremented anywhere). Default 1 (promote on any repeat access)
    // -- much lower than RAMSafetyManager::promote_hit_threshold because
    // the active layer area's window (one layer, current token) is far
    // narrower than the RAM tier's whole-conversation lifetime, so a repeat
    // hit here is already a comparatively strong signal.
    int active_layer_promote_hit_threshold = 1;

    // When a VRAM cache slot is evicted (GpuTopology::evict_lru, via
    // upload_expert), salvage the outgoing expert's bytes into the RAM
    // tier with a D2H copy instead of letting them be silently dropped.
    // Previously a VRAM-evicted expert not ALSO already RAM-resident went
    // fully cold -- its next use always paid a full disk read even though
    // the identical bytes existed in VRAM a moment earlier. A PCIe D2H
    // copy of one expert slab costs roughly a millisecond vs. a 15-50ms+
    // NVMe re-read (this project's own measured [MOE-CACHE] MISS timings),
    // so this is a real, asymmetric win on paper -- but it's new,
    // untested on real hardware, and touches the same VRAM-eviction/
    // cross-thread-reuse territory that has produced multiple real
    // silent-corruption incidents earlier in this project's history (see
    // CLAUDE.md). Scoped deliberately narrow to avoid the worst of that
    // risk: only wired into upload_expert (decode-thread-only), never
    // upload_expert_async (the prefetcher thread's own promotion path),
    // so RAMSafetyManager::layers_experts keeps its existing
    // single-writer-thread invariant with no new locking required -- see
    // gpu_topology.hpp's on_vram_evict and ram_manager.hpp's
    // demote_from_vram for the full reasoning. Default false: opt-in only,
    // A/B test against a run with this disabled before trusting it (watch
    // dump_report()'s "VRAM->RAM demote:" line and Expert cache hit
    // rate/disk-read totals for the real effect).
    bool vram_evict_demote_to_ram = false;

    // dsv4_attention_gguf's grouped output-projection weight (attn_output_a,
    // "WOA") gets a ONE-TIME persistent VRAM upload per layer, and its
    // decode-time (S==1) matmul runs via cuBLAS GEMV on GPU instead of the
    // CPU AVX-512+OMP path. Root-caused on real hardware: this projection's
    // ~134MB/layer weight has to be re-read from host RAM on every single
    // decode token (no other layer's work fits in cache between two calls
    // to the same layer), making it memory-bandwidth-bound on CPU (~6.8s of
    // a real 40-token run) -- moving the weight to VRAM once and reusing it
    // trades CPU's ~30-50GB/s DDR bandwidth for GPU's ~900+GB/s, measured
    // 12x+ faster in a standalone microbenchmark (tests/
    // bench_outproj_cpu_vs_gpu.cu) using this exact shape and access
    // pattern, with numerically verified-matching output before this was
    // wired into the live path. Default false (opt-in) because the VRAM
    // cost is real and persistent, not reclaimable like the RAM/VRAM expert
    // caches: all 43 layers' WOA resident simultaneously is
    // ~134MB*43=~5.8GB, committed for the whole process lifetime the moment
    // each layer is first used, on the SAME CoreEngine GPU that also holds
    // dense weights/KV cache/BF16 fast-path buffers -- on a tight VRAM
    // budget (see this file's own dense_gb/kv_cache_gb sizing) this could
    // starve something else rather than helping. Falls back to the existing
    // CPU path per-layer (not a hard failure) if the VRAM upload fails.
    bool dsv4_outproj_gpu = false;

    // Pilot-driven RAM->VRAM speculative promotion. Calibrated against the
    // "ram-only" per-rank metric (PerfCounters::n_pilot_ram_only_hits_by_
    // rank / n_pilot_ram_only_checks_by_rank, printed as "Pilot recall by
    // rank (DEPTH-1, ram-only)") -- the ONE metric that measures the actual
    // population this decision acts on: pilot's l+1 rank-k picks that are
    // already RAM-resident but NOT yet VRAM-resident at prediction time.
    // Deliberately NOT calibrated against the "overall" or "fetch-worthy"
    // metrics printed alongside it -- those measure disjoint populations
    // (all 4 residency buckets mixed, and the needs-fetch-only bucket that
    // drives disk prefetch, respectively) that say nothing about whether
    // promoting an already-RAM expert into VRAM is worth its cost.
    //
    // Exactly two mutually exclusive modes; setting both in TOML disables
    // BOTH (with a startup warning) rather than silently picking one --
    // see the validation block below.
    //
    // Static mode: unconditionally promote pilot's rank [0, top_k) picks
    // whenever classified into the "already_ram" bucket. Default 0 = off.
    int ram_vram_promote_top_k = 0;

    // Adaptive mode: self-calibrating from the session's OWN live "ram-
    // only" counters (no static per-model tuning needed -- these counters
    // are process-lifetime, i.e. naturally reset each session). Walks
    // rank = 0, 1, 2... promoting while BOTH:
    //   n_pilot_ram_only_checks_by_rank[rank] >= ram_vram_promote_min_samples
    //   AND (hits/checks) at that rank >= ram_vram_promote_adaptive_threshold
    // stopping at the first rank that fails either condition (contiguous-
    // prefix cutoff -- a later rank clearing the bar after an earlier one
    // failed it is not enough, matching the observed monotonic-decay shape
    // of this metric in every real capture so far). The sample-count check
    // doubles as the cold-start guard: early in a session, or for a rank
    // the "already_ram" bucket rarely reaches, there's no separate branch
    // -- it just fails the same floor check an established-but-inaccurate
    // rank would fail, and the cutoff stops there until more data arrives.
    bool ram_vram_promote_adaptive = false;
    double ram_vram_promote_adaptive_threshold = 0.30; // 30%, provisional --
        // re-tune once more than one real capture exists (single-run data
        // so far: k0=44.2% k1=43.0% k2=30.5% k3=25.3% k4=22.8% k5=16.7%,
        // GLM-... see CLAUDE.md; NOT yet validated across a fresh-vs-repeat
        // prompt spread the way the fetch-worthy metric was).
    int ram_vram_promote_min_samples = 30; // per-rank floor before that
        // rank's own accuracy is trusted enough to act on.

    // Redistributes RAMSafetyManager's fixed total RAM-tier budget
    // (uniform slots_per_layer * n_layers) across layers proportional to
    // each layer's real historical need instead of splitting it evenly.
    // Motivated by direct measurement on this project's own accumulated
    // .moestream_usage file (75 layers, 672 selections/layer after
    // several sessions): the most routing-concentrated layer needed only
    // 31 of its own top-ranked experts to cover 80% of its traffic, the
    // flattest needed 92 -- a ~3x range the uniform per-layer split
    // ignores, while the RAM tier was simultaneously observed 100% full
    // (a real, binding constraint) in every real run this session.
    // Default false: opt-in, A/B test against the uniform baseline before
    // trusting it -- see ram_manager.hpp's apply_layer_adaptive_capacity
    // for the full algorithm and its budget-safety guarantees (the
    // redistributed total can never exceed what calculate_safe_cache_
    // capacity already determined is safe, by construction). Uses the
    // SAME `pin_file` usage history as the [PIN] preload block -- no
    // separate file/config needed. First-ever run (no usage history yet)
    // is a complete no-op, identical to leaving this false, since there's
    // no real signal yet to redistribute by.
    bool ram_cache_layer_adaptive = false;

    // How many OMP threads RAMSafetyManager::get_experts_batch uses to
    // parallelize the disk reads for one token's genuine (non-prefetch-served)
    // host-cache misses within a single layer's topk selection. Default 4,
    // chosen from a direct empirical measurement on the target machine
    // buffered reads scaled ~8.5x from 1->4 threads (1.53 -> 13.12 GB/s)
    // then flattened (8 threads: 12.30 GB/s, no further gain) -- 4 captures
    // essentially the full available win on this hardware's real queue
    // depth without over-provisioning threads that would only add
    // contention against the OMP-parallel CPU dequant path running
    // elsewhere. Re-run iobench.c on a different machine/drive to re-tune.
    int expert_fetch_threads = 4;

    // Process-wide default OpenMP team size for EVERY parallel region that
    // does NOT specify its own num_threads(...) clause. Was previously
    // unset entirely -- expert_fetch_threads above only caps ONE specific
    // region (ram_manager.hpp's Phase B disk-read loop, the only one with
    // an explicit num_threads()); every other #pragma omp parallel for in
    // the codebase (attention_layer.cpp's dense/attention matmuls,
    // kernels_cpu.cpp's CPU expert-fallback dequant+dot path, kquant_
    // dequant.hpp's reference dequant) silently defaulted to OpenMP's own
    // runtime default (all available cores), with nothing to stop several
    // of these regions running concurrently and oversubscribing the
    // machine. A real run showed the OS itself become unresponsive to
    // other commands -- traced to the CPU expert-fallback path
    // specifically: expert_ffn_cpu calls cpu_quant_matmul three times per
    // call (gate/up/down), each hitting one of these uncapped regions, and
    // this path's call volume had grown to ~2000+/run (from ~1000 a few
    // sessions earlier) as a direct, deliberate side effect of raising
    // pinned_vram_capacity -- more pinning means more "just promoted, not
    // yet GPU-usable" experts falling back to CPU (see perf_timer.hpp's
    // n_cpu_fallback_just_promoted), each burst spinning up to every core,
    // three times, completely uncoordinated with whatever else was
    // running. 0 = auto (hardware_concurrency() - 2, floor 1) -- leaves
    // headroom for the OS and any other process by default rather than
    // requiring every deployment to work out its own safe number; set
    // explicitly to override. Applied once via omp_set_num_threads() at
    // the very start of run_engine(), before any parallel region can run
    // -- explicit num_threads() clauses (like expert_fetch_threads' own)
    // still override this default locally, so raising expert_fetch_threads
    // above this value is still honored for that one region specifically.
    int omp_max_threads = 0;

    // Dedicated OMP team size for JUST attention_layer.cpp's causal
    // score/softmax/value-aggregate loop (mla_attention_absorbed_gguf's
    // `#pragma omp parallel for collapse(2)`) -- separate from omp_max_
    // threads above, which this region would otherwise share with every
    // other parallel region in the process. Motivated by a real-hardware
    // finding (2026-08-13): raising expert_fetch_threads/omp_max_threads
    // 8->10 for a real, measured disk-I/O win (decode 0.68->0.70 tok/s)
    // simultaneously raised score's own per-token cost from ~74.8 to
    // ~83.5ms/token in that SAME real run -- score is the one attention
    // sub-step with no GPU path at all (see CLAUDE.md), so it competes
    // directly with expert_fetch_threads' background disk-read threads for
    // the same physical cores once omp_max_threads is raised alongside it.
    // 0 (default) = no override, falls back to whatever omp_max_threads'
    // own effective value already is (see dense_dispatch.hpp's
    // attn_score_threads() for the exact fallback) -- a config that never
    // sets this behaves identically to before this field existed. Set
    // LOWER than omp_max_threads to trade some of score's own parallelism
    // back for less contention against expert_fetch_threads;
    // UNVERIFIED which value (if any) is actually a net win here -- this
    // is a real, previously-uninvestigated lever, not yet A/B tested.
    int attn_score_threads = 0;

    // When true, main.cpp's REPL feeds stdin content to the tokenizer
    // VERBATIM -- skips the GLM-3/4-style chat-template wrap
    // ([gMASK]<sop><|user|>\n{msg}<|assistant|>\n) it would otherwise apply
    // whenever the tokenizer detects chat-template tokens in the vocab.
    // Two real uses: (1) fair prefill-time benchmarking against another
    // engine on an identical token count, since the wrap alone can more
    // than double a short prompt's token count; (2) required for
    // api/api_server.py, which already builds its own <|role|>content-
    // style prompt before ever reaching this engine -- without this flag
    // that pre-formatted text gets wrapped a SECOND time, producing
    // nested/duplicated special tokens (a real correctness bug, not just
    // extra tokens -- see CLAUDE.md). Default false: the REPL's own
    // template-detection behavior is unchanged for normal interactive use.
    bool raw_prompt = false;

    // Optional system-role message, prepended ahead of the user turn when a
    // chat template is in use (GGUFTokenizer::has_chat_template /
    // has_deepseek_chat_template). Empty by default -- confirmed
    // byte-for-byte against the real DeepSeek-V4-Flash jinja template
    // (models/templates/deepseek-ai-DeepSeek-V4-Flash-0731.jinja in a real
    // llama.cpp checkout) that an absent system message legitimately
    // produces {bos}<｜User｜>{msg}<｜Assistant｜></think> with no system
    // text at all -- exactly what this engine already sent, so "missing
    // system prompt" alone is not a template bug. A base model with no
    // language-steering instruction will happily answer an
    // ambiguous/short English prompt in Chinese; that's normal
    // DeepSeek/Qwen/GLM-family behavior without one. Set this (or
    // --system-prompt) to add one, e.g. "Respond in English unless asked
    // otherwise."
    std::string system_prompt;

    // Logging. Empty log_path (default) keeps everything on stderr, matching
    // prior behavior. log_level filters by severity: VERBOSE shows
    // high-volume structural traces (e.g. per-layer hidden-state norms
    // across the full model depth) on top of everything below; DEBUG shows
    // every per-token/per-expert diagnostic (cache hits/misses, VRAM
    // promotions, CPU dequant fallback notices); INFO (default) hides those
    // and shows only startup info and real warnings/errors.
    std::string log_path;
    std::string log_level = "INFO";

    // Tensor-trace debugging (include/tensor_trace.hpp), a llama.cpp
    // eval-callback analog for this engine's imperative forward pass.
    // trace_filter is a comma-separated list of tags (e.g. "HC,MLA") matched
    // as an anchored regex prefix against each VERBOSE trace call site's own
    // tag ("HC", "MLA", "EXPERT-GPU", "SILU-TRACE-GPU", ...); empty (default)
    // matches every tag, i.e. no filtering beyond log_level=VERBOSE itself.
    // trace_abort_on_nan hard-aborts the process at the FIRST tensor a
    // NaN/Inf checker flags, instead of logging an error and continuing --
    // useful for stopping right where a numerical bug originates rather than
    // scrolling through pages of downstream fallout.
    std::string trace_filter;
    bool trace_abort_on_nan = false;

    // GPU configuration
    std::vector<GpuConfig> gpus;

    // Derived: which GPU index serves each role (set by validate())
    int dense_gpu_idx = -1;
    std::vector<int> moe_gpu_idxs;

    bool load(const std::string& path);
    bool validate();
    static void write_sample(const std::string& path);
    void print() const;
};

// ============================================================
// Minimal TOML parser — supports the subset we need:
//   key = "string"
//   key = 123          (int)
//   key = 1.5          (float)
//   key = true/false   (bool)
//   [section]          (table header — ignored, we use flat keys)
//   [[array]]          (array of tables — each creates a new entry)
//   # comments
// ============================================================
namespace toml {

inline std::string trim(const std::string& s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    size_t b = s.find_last_not_of(" \t\r\n");
    return (a == std::string::npos) ? "" : s.substr(a, b - a + 1);
}

inline std::string strip_comment(const std::string& line) {
    bool in_string = false;
    for (size_t i = 0; i < line.size(); ++i) {
        if (line[i] == '"') in_string = !in_string;
        if (line[i] == '#' && !in_string) return line.substr(0, i);
    }
    return line;
}

struct Value {
    std::string str;
    double num = 0;
    bool boolean = false;
    std::vector<int> int_array;  // only populated when type == IntArray
    enum Type { String, Number, Bool, IntArray } type = String;
};

struct Table {
    std::unordered_map<std::string, Value> values;

    bool has(const std::string& key) const { return values.count(key) > 0; }

    // Every get_*() above silently falls back to `def` on a type mismatch,
    // by design (same "def" spirit noted on get_int_array) -- but that
    // makes a present-but-wrong-type key (e.g. `prefetch_pilot_depth =
    // true` where an integer was expected) INDISTINGUISHABLE from the key
    // being absent entirely, so a real typo silently vanishes with no
    // warning anywhere. This lets a call site tell the two apart and warn
    // specifically on "present but wrong type", without changing any
    // existing get_*()'s own default-on-mismatch behavior (many call sites
    // rely on that leniency and shouldn't need to change).
    std::optional<Value::Type> get_type(const std::string& key) const {
        auto it = values.find(key);
        return (it != values.end()) ? std::optional<Value::Type>(it->second.type) : std::nullopt;
    }

    std::string get_str(const std::string& key, const std::string& def = "") const {
        auto it = values.find(key);
        return (it != values.end()) ? it->second.str : def;
    }
    int get_int(const std::string& key, int def = 0) const {
        auto it = values.find(key);
        return (it != values.end() && it->second.type == Value::Number) ? (int)it->second.num : def;
    }
    double get_double(const std::string& key, double def = 0.0) const {
        auto it = values.find(key);
        return (it != values.end() && it->second.type == Value::Number) ? it->second.num : def;
    }
    float get_float(const std::string& key, float def = 0.0f) const {
        return (float)get_double(key, def);
    }
    bool get_bool(const std::string& key, bool def = false) const {
        auto it = values.find(key);
        return (it != values.end() && it->second.type == Value::Bool) ? it->second.boolean : def;
    }
    // e.g. `cpu_affinity_layers = [4, 5, 7]`. Missing key or wrong type
    // (not a bracketed array) returns an empty vector, same "def" spirit
    // as the other get_*'s -- callers treat empty as "no layers affected".
    std::vector<int> get_int_array(const std::string& key) const {
        auto it = values.find(key);
        return (it != values.end() && it->second.type == Value::IntArray) ? it->second.int_array
                                                                            : std::vector<int>{};
    }
};

struct Document {
    Table root;                        // Top-level key = value pairs
    std::vector<Table> gpu_tables;     // [[gpus]] array-of-tables entries
};

inline Value parse_value(const std::string& raw) {
    Value v;
    std::string s = trim(raw);
    if (s.empty()) return v;

    // Quoted string
    if (s.front() == '"' && s.back() == '"') {
        v.type = Value::String;
        v.str = s.substr(1, s.size() - 2);
        return v;
    }
    // Boolean
    if (s == "true" || s == "false") {
        v.type = Value::Bool;
        v.boolean = (s == "true");
        v.str = s;
        return v;
    }
    // Integer array: `[4, 5, 7]`. Deliberately simple (no nested arrays, no
    // floats, no multi-line arrays) -- the only current consumer
    // (cpu_affinity_layers) is a short flat list of layer indices, and this
    // parser is documented as "the subset we need", not general TOML.
    if (s.front() == '[' && s.back() == ']') {
        v.type = Value::IntArray;
        v.str = s;
        std::string inner = s.substr(1, s.size() - 2);
        std::istringstream elems(inner);
        std::string tok;
        while (std::getline(elems, tok, ',')) {
            std::string t = trim(tok);
            if (t.empty()) continue;
            try {
                v.int_array.push_back(std::stoi(t));
            } catch (...) {
                // Malformed element -- skip it rather than abort the whole
                // config load; a missing layer index just means that layer
                // isn't CPU-pinned, never a correctness/crash risk.
            }
        }
        return v;
    }
    // Number
    try {
        v.num = std::stod(s);
        v.type = Value::Number;
        v.str = s;
    } catch (...) {
        v.type = Value::String;
        v.str = s;
    }
    return v;
}

inline Document parse(const std::string& content) {
    Document doc;
    Table* current = &doc.root;
    bool in_gpu_array = false;

    std::istringstream stream(content);
    std::string line;
    while (std::getline(stream, line)) {
        line = trim(strip_comment(line));
        if (line.empty()) continue;

        // [[gpus]] — array of tables
        if (line == "[[gpus]]") {
            in_gpu_array = true;
            doc.gpu_tables.emplace_back();
            current = &doc.gpu_tables.back();
            continue;
        }

        // [section] — named table (not gpus)
        if (line.front() == '[' && line.back() == ']') {
            in_gpu_array = false;
            current = &doc.root;  // flatten non-gpu sections into root
            continue;
        }

        // key = value
        size_t eq = line.find('=');
        if (eq == std::string::npos) continue;
        std::string key = trim(line.substr(0, eq));
        std::string val_str = trim(line.substr(eq + 1));
        current->values[key] = parse_value(val_str);
    }
    return doc;
}

inline Document parse_file(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) {
        std::cerr << "[CONFIG] Cannot open: " << path << "\n";
        return {};
    }
    std::string content((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    return parse(content);
}

} // namespace toml

// ============================================================
// EngineConfig implementation
// ============================================================
inline bool EngineConfig::load(const std::string& path) {
    auto doc = toml::parse_file(path);

    model_path       = doc.root.get_str("model_path");
    dense_model_path = doc.root.get_str("dense_model_path");
    context_size     = doc.root.get_int("context_size", context_size);
    max_new_tokens   = doc.root.get_int("max_new_tokens", max_new_tokens);
    temperature      = doc.root.get_float("temperature", temperature);
    top_p            = doc.root.get_float("top_p", top_p);
    drop_pages       = doc.root.get_bool("drop_pages", drop_pages);
    cpu_only         = doc.root.get_bool("cpu_only", cpu_only);
    cpu_affinity_layers = doc.root.get_int_array("cpu_affinity_layers");
    prefetch_pilot_k    = doc.root.get_int("prefetch_pilot_k", prefetch_pilot_k);
    prefetch_pilot_k_per_layer = doc.root.get_int_array("prefetch_pilot_k_per_layer");
    // Real incident this caught: a config had `prefetch_pilot_depth = true`
    // (a boolean, presumably intended as an enable/disable toggle -- an easy
    // mistake given how many OTHER flags in this same config format ARE
    // plain booleans, e.g. io_uring/direct_io/disk_batch_padding). get_int()
    // silently ignores a non-Number value and falls back to its default
    // (1), so that config had been silently running DEPTH-1-only the whole
    // time with zero indication anything was wrong -- caught only by
    // manually reading the raw TOML, not by any startup log. Closed via
    // get_type(): a present-but-wrong-type key is now distinguishable from
    // an absent one and warned about explicitly, before get_int()'s own
    // silent fallback ever runs.
    if (auto t = doc.root.get_type("prefetch_pilot_depth"); t && *t != toml::Value::Number) {
        LOG_WARN("[CONFIG] prefetch_pilot_depth is set but is not a number (e.g. a boolean like "
                 "`true` was found) -- this key takes an integer, 1 or 2 (e.g. "
                 "prefetch_pilot_depth = 2), not a toggle -- ignoring, falling back to 1\n");
    }
    prefetch_pilot_depth = doc.root.get_int("prefetch_pilot_depth", prefetch_pilot_depth);
    if (prefetch_pilot_depth != 1 && prefetch_pilot_depth != 2) {
        LOG_WARN("[CONFIG] prefetch_pilot_depth=%d is invalid (must be 1 or 2) -- falling back to 1\n",
                 prefetch_pilot_depth);
        prefetch_pilot_depth = 1;
    }
    prefetch_pilot_k_depth2 = doc.root.get_int("prefetch_pilot_k_depth2", prefetch_pilot_k_depth2);
    direct_io        = doc.root.get_bool("direct_io", direct_io);
    cuda_cufile      = doc.root.get_bool("cuda_cufile", cuda_cufile);
    prefill_warm_via_prefetcher = doc.root.get_bool("prefill_warm_via_prefetcher", prefill_warm_via_prefetcher);
    prefill_opportunistic_promote = doc.root.get_bool("prefill_opportunistic_promote", prefill_opportunistic_promote);
    decode_async_prewarm = doc.root.get_bool("decode_async_prewarm", decode_async_prewarm);
    decode_async_prewarm_max_batch = doc.root.get_int("decode_async_prewarm_max_batch", decode_async_prewarm_max_batch);
    if (decode_async_prewarm_max_batch < 1) {
        LOG_WARN("[CONFIG] decode_async_prewarm_max_batch must be >= 1 (got %d) -- clamping to 1\n",
                 decode_async_prewarm_max_batch);
        decode_async_prewarm_max_batch = 1;
    }
    attn_qabsorb_gpu = doc.root.get_bool("attn_qabsorb_gpu", attn_qabsorb_gpu);
    attn_qabsorb_gpu_max_vram_gb = doc.root.get_double("attn_qabsorb_gpu_max_vram_gb", attn_qabsorb_gpu_max_vram_gb);
    if (attn_qabsorb_gpu_max_vram_gb < 0.0) {
        LOG_WARN("[CONFIG] attn_qabsorb_gpu_max_vram_gb must be >= 0 (got %.2f) -- clamping to 0\n",
                 attn_qabsorb_gpu_max_vram_gb);
        attn_qabsorb_gpu_max_vram_gb = 0.0;
    }
    attn_score_gpu = doc.root.get_bool("attn_score_gpu", attn_score_gpu);
    attn_score_gpu_max_vram_gb = doc.root.get_double("attn_score_gpu_max_vram_gb", attn_score_gpu_max_vram_gb);
    if (attn_score_gpu_max_vram_gb < 0.0) {
        LOG_WARN("[CONFIG] attn_score_gpu_max_vram_gb must be >= 0 (got %.2f) -- clamping to 0\n",
                 attn_score_gpu_max_vram_gb);
        attn_score_gpu_max_vram_gb = 0.0;
    }
    attn_score_gpu_max_context = doc.root.get_int("attn_score_gpu_max_context", attn_score_gpu_max_context);
    if (attn_score_gpu_max_context < 0) {
        LOG_WARN("[CONFIG] attn_score_gpu_max_context must be >= 0 (got %d) -- clamping to 0 (no cap)\n",
                 attn_score_gpu_max_context);
        attn_score_gpu_max_context = 0;
    }
    unpacked_expert_format = doc.root.get_bool("unpacked_expert_format", unpacked_expert_format);
    io_uring         = doc.root.get_bool("io_uring", io_uring);
    io_uring_workers = doc.root.get_int("io_uring_workers", io_uring_workers);
    disk_batch_padding        = doc.root.get_bool("disk_batch_padding", disk_batch_padding);
    disk_batch_padding_target = doc.root.get_int("disk_batch_padding_target", disk_batch_padding_target);
    disk_batch_padding_evict_dup = doc.root.get_bool("disk_batch_padding_evict_dup", disk_batch_padding_evict_dup);
    prefetch_worker_threads = doc.root.get_int("prefetch_worker_threads", prefetch_worker_threads);
    host_ram_gb      = doc.root.get_double("host_ram_gb", host_ram_gb);
    mirror_model_path = doc.root.get_str("mirror_model_path", mirror_model_path);
    ram_pinned_budget_gb = doc.root.get_double("ram_pinned_budget_gb", ram_pinned_budget_gb);
    verbose_moe_log  = doc.root.get_bool("verbose_moe_log", verbose_moe_log);
    promote_hit_threshold = doc.root.get_int("promote_hit_threshold", promote_hit_threshold);
    pin_threshold    = doc.root.get_int("pin_threshold", pin_threshold);
    pin_file         = doc.root.get_str("pin_file", pin_file);
    prefill_pin_file = doc.root.get_str("prefill_pin_file", prefill_pin_file);
    freeze_usage_stats = doc.root.get_bool("freeze_usage_stats", freeze_usage_stats);
    pin_topk_per_layer = doc.root.get_int("pin_topk_per_layer", pin_topk_per_layer);
    pin_strategy = doc.root.get_str("pin_strategy", pin_strategy);
    pin_global_floor_per_layer = doc.root.get_int("pin_global_floor_per_layer", pin_global_floor_per_layer);
    pin_vram_lru_fraction = doc.root.get_double("pin_vram_lru_fraction", pin_vram_lru_fraction);
    pin_ram_fraction      = doc.root.get_double("pin_ram_fraction", pin_ram_fraction);
    vram_evict_demote_to_ram = doc.root.get_bool("vram_evict_demote_to_ram", vram_evict_demote_to_ram);
    dsv4_outproj_gpu = doc.root.get_bool("dsv4_outproj_gpu", dsv4_outproj_gpu);
    ram_vram_promote_top_k = doc.root.get_int("ram_vram_promote_top_k", ram_vram_promote_top_k);
    ram_vram_promote_adaptive = doc.root.get_bool("ram_vram_promote_adaptive", ram_vram_promote_adaptive);
    ram_vram_promote_adaptive_threshold = doc.root.get_double("ram_vram_promote_adaptive_threshold", ram_vram_promote_adaptive_threshold);
    ram_vram_promote_min_samples = doc.root.get_int("ram_vram_promote_min_samples", ram_vram_promote_min_samples);
    ram_cache_layer_adaptive = doc.root.get_bool("ram_cache_layer_adaptive", ram_cache_layer_adaptive);
    expert_fetch_threads = doc.root.get_int("expert_fetch_threads", expert_fetch_threads);
    omp_max_threads = doc.root.get_int("omp_max_threads", omp_max_threads);
    attn_score_threads = doc.root.get_int("attn_score_threads", attn_score_threads);
    // NOTE: these four were previously missing from parsing entirely --
    // every configs/*.toml's `pinned_vram_capacity`/`move_on_promotion`/
    // `active_layer_capacity`/`use_active_layer_area` lines were silently
    // no-ops, only "working" because each one happened to match its
    // compiled default (1024/true/256/true). Any run that tried to
    // actually CHANGE one of these via TOML got the default instead, with
    // no warning.
    pinned_vram_capacity = doc.root.get_int("pinned_vram_capacity", pinned_vram_capacity);
    move_on_promotion    = doc.root.get_bool("move_on_promotion", move_on_promotion);
    ram_copy_reclaim_threshold = doc.root.get_double("ram_copy_reclaim_threshold", ram_copy_reclaim_threshold);
    active_layer_capacity = doc.root.get_int("active_layer_capacity", active_layer_capacity);
    use_active_layer_area = doc.root.get_bool("use_active_layer_area", use_active_layer_area);
    active_layer_promote_hit_threshold = doc.root.get_int("active_layer_promote_hit_threshold",
                                                            active_layer_promote_hit_threshold);
    raw_prompt           = doc.root.get_bool("raw_prompt", raw_prompt);
    system_prompt        = doc.root.get_str("system_prompt", system_prompt);
    log_path         = doc.root.get_str("log_path", log_path);
    log_level        = doc.root.get_str("log_level", log_level);
    trace_filter     = doc.root.get_str("trace_filter", trace_filter);
    trace_abort_on_nan = doc.root.get_bool("trace_abort_on_nan", trace_abort_on_nan);

    for (auto& gt : doc.gpu_tables) {
        GpuConfig gc;
        gc.device_id    = gt.get_int("device_id", -1);
        gc.name         = gt.get_str("name");
        gc.role         = gt.get_str("role", "dense+moe");
        gc.model_path   = gt.get_str("model_path");
        gc.vram_gb      = gt.get_double("vram_gb");
        gc.dense_gb     = gt.get_double("dense_gb");
        gc.kv_cache_gb  = gt.get_double("kv_cache_gb");
        gc.kv_bits      = gt.get_int("kv_bits", 4);
        gc.moe_cache_gb = gt.get_double("moe_cache_gb");
        gpus.push_back(gc);
    }

    if (model_path.empty()) {
        std::cerr << "[CONFIG] Error: model_path is required.\n";
        return false;
    }

    std::cerr << "[CONFIG] Loaded from " << path << " (" << gpus.size() << " GPU entries)\n";
    return true;
}

inline bool EngineConfig::validate() {
    int hw_count = 0;
    cudaGetDeviceCount(&hw_count);

    std::vector<GpuConfig> valid_gpus;
    for (auto& gc : gpus) {
        if (gc.device_id < 0 || gc.device_id >= hw_count) {
            std::cerr << "[CONFIG] GPU " << gc.device_id << " not present, skipping.\n";
            continue;
        }
        if (gc.vram_gb <= 0) {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, gc.device_id);
            gc.vram_gb = prop.totalGlobalMem / 1e9;
            if (gc.name.empty()) gc.name = prop.name;
        }
        valid_gpus.push_back(gc);
    }
    gpus = valid_gpus;

    if (gpus.empty()) {
        std::cerr << "[CONFIG] No valid GPUs configured.\n";
        return false;
    }

    // Assign roles
    dense_gpu_idx = -1;
    moe_gpu_idxs.clear();
    for (size_t i = 0; i < gpus.size(); ++i) {
        const std::string& r = gpus[i].role;
        if (r == "dense" || r == "dense+moe") {
            if (dense_gpu_idx < 0) dense_gpu_idx = (int)i;
        }
        if (r == "moe" || r == "dense+moe") {
            moe_gpu_idxs.push_back((int)i);
        }
    }

    if (dense_gpu_idx < 0) {
        std::cerr << "[CONFIG] No GPU with role 'dense' or 'dense+moe'. "
                  << "Assigning GPU 0 as dense+moe.\n";
        gpus[0].role = "dense+moe";
        dense_gpu_idx = 0;
        if (std::find(moe_gpu_idxs.begin(), moe_gpu_idxs.end(), 0) == moe_gpu_idxs.end())
            moe_gpu_idxs.push_back(0);
    }

    if (dense_model_path.empty()) dense_model_path = model_path;
    for (auto& gc : gpus) {
        if (gc.model_path.empty()) gc.model_path = model_path;
    }

    // Auto-compute memory layout if not specified
    for (auto& gc : gpus) {
        if (gc.role == "dense" || gc.role == "dense+moe") {
            double available = gc.vram_gb - 1.0;
            if (gc.dense_gb <= 0) gc.dense_gb = std::min(8.0, available * 0.5);
            if (gc.kv_cache_gb <= 0) gc.kv_cache_gb = std::min(5.0, available * 0.3);
            if (gc.role == "dense+moe" && gc.moe_cache_gb <= 0) {
                gc.moe_cache_gb = available - gc.dense_gb - gc.kv_cache_gb;
                if (gc.moe_cache_gb < 0) gc.moe_cache_gb = 0;
            }
        }
        if (gc.role == "moe") {
            if (gc.moe_cache_gb <= 0) gc.moe_cache_gb = gc.vram_gb - 1.0;
        }
    }

    if (cuda_cufile && !direct_io) {
        std::cerr << "[CONFIG] cuda_cufile=true requires direct_io=true (cuFile registers the SAME "
                  << "O_DIRECT file descriptor direct_io opens -- see GGUFFile::enable_cufile) -- "
                  << "disabling cuda_cufile for this run. Set direct_io=true as well to actually use it.\n";
        cuda_cufile = false;
    }

    if (pin_strategy != "per_layer_topk" && pin_strategy != "global_topn") {
        std::cerr << "[CONFIG] Unknown pin_strategy '" << pin_strategy
                  << "' (expected \"per_layer_topk\" or \"global_topn\") -- "
                  << "falling back to \"per_layer_topk\".\n";
        pin_strategy = "per_layer_topk";
    }

    if (ram_vram_promote_top_k > 0 && ram_vram_promote_adaptive) {
        std::cerr << "[CONFIG] ram_vram_promote_top_k and ram_vram_promote_adaptive are "
                  << "mutually exclusive but both were set -- disabling BOTH rather than "
                  << "guessing which one you meant. Set exactly one.\n";
        ram_vram_promote_top_k = 0;
        ram_vram_promote_adaptive = false;
    }

    if (disk_batch_padding && !io_uring) {
        std::cerr << "[CONFIG] disk_batch_padding=true requires io_uring=true (padding rides the "
                  << "decode thread's own io_uring ring, RAMSafetyManager::decode_uring_ring) -- "
                  << "disabling disk_batch_padding for this run.\n";
        disk_batch_padding = false;
    }
    disk_batch_padding_target = std::max(1, std::min(21, disk_batch_padding_target));

    return true;
}

inline void EngineConfig::print() const {
    LOG_STREAM_INFO() << "[CONFIG] ========== Engine Configuration ==========\n";
    LOG_STREAM_INFO() << "[CONFIG] Model: " << model_path << "\n";
    if (!dense_model_path.empty() && dense_model_path != model_path)
        LOG_STREAM_INFO() << "[CONFIG] Dense model: " << dense_model_path << "\n";
    LOG_STREAM_INFO() << "[CONFIG] Context: " << context_size << " tokens | Max gen: " << max_new_tokens << "\n";
    LOG_STREAM_INFO() << "[CONFIG] Temperature: " << temperature << " | Top-p: " << top_p << "\n";
    LOG_STREAM_INFO() << "[CONFIG] Drop pages: " << (drop_pages ? "yes" : "no") << "\n";
    LOG_STREAM_INFO() << "[CONFIG] Direct I/O (O_DIRECT expert reads): " << (direct_io ? "requested" : "off")
              << (direct_io ? " (falls back to buffered per-shard if unsupported/unaligned -- see [DIRECT-IO] log)" : "") << "\n";
    if (unpacked_expert_format)
        LOG_STREAM_INFO() << "[CONFIG] Unpacked expert format: ON (single combined DMA per expert; "
                              "startup sanity check will FATAL if the model isn't actually tight/aligned)\n";
    LOG_STREAM_INFO() << "[CONFIG] cuFile (GPUDirect Storage direct disk->VRAM): " << (cuda_cufile ? "requested" : "off")
              << (cuda_cufile ? " (falls back to RAM-mediated upload if GDS/nvidia-fs unavailable -- see [CUFILE] log)" : "") << "\n";
    LOG_STREAM_INFO() << "[CONFIG] Prefill warm-up via prefetcher: " << (prefill_warm_via_prefetcher ? "ON" : "off (legacy synchronous)")
              << "\n";
    if (prefill_warm_via_prefetcher)
        LOG_STREAM_INFO() << "[CONFIG] Prefill opportunistic VRAM promotion: "
                  << (prefill_opportunistic_promote ? "ON" : "off (hints land in RAM only until Phase 3's own synchronous dispatch)")
                  << "\n";
    LOG_STREAM_INFO() << "[CONFIG] Decode async pre-warm (batched active-layer H2D for decode-time misses): "
              << (decode_async_prewarm ? "ON" : "off (legacy sequential blocking upload)")
              << (decode_async_prewarm ? (", max_batch=" + std::to_string(decode_async_prewarm_max_batch)) : std::string())
              << "\n";
    LOG_STREAM_INFO() << "[CONFIG] MLA qabsorb BF16 batched GPU path: "
              << (attn_qabsorb_gpu ? "ON" : "off (CPU per-head matmul() loop)")
              << (attn_qabsorb_gpu ? (", max_vram=" + std::to_string(attn_qabsorb_gpu_max_vram_gb) + "GB")
                                    : std::string())
              << " (no effect on DeepSeek-V4 -- dsv4_attention_gguf has no absorption/WUK at all)\n";
    LOG_STREAM_INFO() << "[CONFIG] MLA score GPU path (3-GEMM+softmax, decode only): "
              << (attn_score_gpu ? "ON" : "off (CPU AVX-512 per-head loop)")
              << (attn_score_gpu ? (", max_vram=" + std::to_string(attn_score_gpu_max_vram_gb) + "GB")
                                  : std::string())
              << " (DeepSeek-V4's windowed score is already cheap on CPU -- see attn_score_gpu's own comment)\n";
    LOG_STREAM_INFO() << "[CONFIG] io_uring (batched prefetcher reads): " << (io_uring ? "requested" : "off")
              << (io_uring ? " (falls back to sequential preads if unsupported -- see [IO-URING] log)" : "") << "\n";
    if (io_uring) {
        LOG_STREAM_INFO() << "[CONFIG] io_uring workers (decode-thread ring's io-wq pool): " << io_uring_workers << "\n";
    }
    if (disk_batch_padding)
        LOG_STREAM_INFO() << "[CONFIG] Disk batch padding: ON (target " << disk_batch_padding_target
                           << " experts/submission, speculative reads from route_freq"
                           << (disk_batch_padding_evict_dup
                                   ? ", CAN evict also-VRAM-resident RAM slots once free slots run out"
                                   : ", free-slots-only (V1)")
                           << ")\n";
    LOG_STREAM_INFO() << "[CONFIG] Prefetch worker threads: " << prefetch_worker_threads << "\n";
    LOG_STREAM_INFO() << "[CONFIG] Prefetch pilot_k (hints/layer transition): default=" << prefetch_pilot_k
                       << (prefetch_pilot_k_per_layer.empty() ? "" : ", " + std::to_string(prefetch_pilot_k_per_layer.size())
                           + " layer override value(s) set (see [PREFETCH-LAYER] report)")
                       << "\n";
    LOG_STREAM_INFO() << "[CONFIG] Prefetch pilot depth: " << prefetch_pilot_depth
                       << (prefetch_pilot_depth == 2
                               ? (" (l+2 hints ON, top-" + std::to_string(prefetch_pilot_k_depth2)
                                  + " ranks pushed -- UNVERIFIED for wall-clock impact, see 'Pilot recall (DEPTH-2...' "
                                  + "and 'Pilot recall by rank' in the shutdown report)")
                               : " (l+2 measurement-only -- see 'Pilot recall (DEPTH-2...' in the shutdown report "
                                 "before enabling)")
                       << "\n";
    if (cpu_only) LOG_STREAM_INFO() << "[CONFIG] CPU-only mode: ON (all GPU dispatch disabled)\n";
    if (!cpu_affinity_layers.empty()) {
        LOG_STREAM_INFO() << "[CONFIG] CPU-affinity layers (never GPU-dispatched or VRAM-uploaded): ";
        for (size_t i = 0; i < cpu_affinity_layers.size(); ++i)
            LOG_STREAM_INFO() << (i > 0 ? ", " : "") << cpu_affinity_layers[i];
        LOG_STREAM_INFO() << "\n";
    }
    LOG_STREAM_INFO() << "[CONFIG] Host RAM (DDR) budget: "
              << (host_ram_gb > 0 ? std::to_string(host_ram_gb) + " GB (explicit)" : "auto-detect") << "\n";
    LOG_STREAM_INFO() << "[CONFIG] Promote H2D threshold: " << promote_hit_threshold << " hits\n";
    LOG_STREAM_INFO() << "[CONFIG] Pin threshold: " << pin_threshold
              << (pin_threshold > 0 ? " hits" : " (pinning disabled)") << "\n";
    if (!prefill_pin_file.empty())
        LOG_STREAM_INFO() << "[CONFIG] Prefill pin file: " << prefill_pin_file
                  << " (preferred for prefill-time preload ranking; falls back to pin_file per-layer "
                     "where this has no data)\n";
    LOG_STREAM_INFO() << "[CONFIG] Pin file: " << (pin_file.empty() ? "(preload disabled)" : pin_file)
              << " | strategy=" << pin_strategy
              << (pin_strategy == "global_topn"
                      ? (" (floor " + std::to_string(pin_global_floor_per_layer) + "/layer, remainder ranked globally)")
                      : (" (top-" + (pin_topk_per_layer > 0 ? std::to_string(pin_topk_per_layer) : std::string("auto")) + "/layer)"))
              << ", next " << (pin_vram_lru_fraction * 100.0) << "% of VRAM tier capacity (unpinned)"
              << ", next " << (pin_ram_fraction * 100.0) << "% of RAM tier capacity\n";
    if (freeze_usage_stats)
        LOG_STREAM_INFO() << "[CONFIG] Usage-file writes: FROZEN (pin_file/prefill_pin_file read at startup "
                              "as configured, but NOT overwritten at shutdown -- for controlled A/B testing "
                              "against a fixed ranking)\n";
    LOG_STREAM_INFO() << "[CONFIG] VRAM evict -> RAM demote: "
              << (vram_evict_demote_to_ram ? "ON (D2H-salvage evicted experts into the RAM tier)" : "off") << "\n";
    LOG_STREAM_INFO() << "[CONFIG] DeepSeek-V4 outproj GPU dispatch: "
              << (dsv4_outproj_gpu ? "ON (cuBLAS GEMV, weight VRAM-resident once per layer, ~5.8GB total)" : "off") << "\n";
    LOG_STREAM_INFO() << "[CONFIG] RAM->VRAM speculative promotion: "
              << (ram_vram_promote_top_k > 0
                      ? ("static top-" + std::to_string(ram_vram_promote_top_k))
                      : (ram_vram_promote_adaptive
                             ? ("adaptive (threshold " + std::to_string(ram_vram_promote_adaptive_threshold * 100.0)
                                + "%, min_samples " + std::to_string(ram_vram_promote_min_samples) + "/rank)")
                             : std::string("off")))
              << "\n";
    LOG_STREAM_INFO() << "[CONFIG] RAM cache layer-adaptive: "
              << (ram_cache_layer_adaptive ? "ON (per-layer capacity by historical need, not uniform)" : "off") << "\n";
    LOG_STREAM_INFO() << "[CONFIG] Expert fetch threads: " << expert_fetch_threads << "\n";
    LOG_STREAM_INFO() << "[CONFIG] OMP max threads (process-wide default): "
              << (omp_max_threads > 0 ? std::to_string(omp_max_threads) : std::string("auto"))
              << "\n";
    LOG_STREAM_INFO() << "[CONFIG] Attn score threads (dedicated, attention_layer.cpp's causal "
                          "score loop only): "
              << (attn_score_threads > 0 ? std::to_string(attn_score_threads)
                                         : std::string("no override (uses the process-wide default above)"))
              << "\n";
    LOG_STREAM_INFO() << "[CONFIG] Pinned VRAM capacity: " << pinned_vram_capacity
              << " | move_on_promotion: " << (move_on_promotion ? "MOVE" : "COPY")
              << (!move_on_promotion && ram_copy_reclaim_threshold > 0.0
                  ? (" (ram_copy_reclaim_threshold=" + std::to_string(ram_copy_reclaim_threshold) + ")")
                  : std::string(""))
              << "\n";
    LOG_STREAM_INFO() << "[CONFIG] Active layer area: "
              << (use_active_layer_area ? std::to_string(active_layer_capacity) + " slots, promote at "
                                           + std::to_string(active_layer_promote_hit_threshold) + " hit(s)"
                                         : "disabled") << "\n";
    if (raw_prompt)
        LOG_STREAM_INFO() << "[CONFIG] Raw prompt mode: ON -- REPL will NOT apply the chat "
                              "template, stdin content is tokenized verbatim\n";
    LOG_STREAM_INFO() << "[CONFIG] Verbose MoE cache logging: " << (verbose_moe_log ? "on" : "off") << "\n";
    LOG_STREAM_INFO() << "[CONFIG] Log level: " << log_level
               << " | Log path: " << (log_path.empty() ? "(stderr)" : log_path) << "\n";
    LOG_STREAM_INFO() << "[CONFIG] Trace filter: " << (trace_filter.empty() ? "(all)" : trace_filter)
               << " | Trace abort-on-nan: " << (trace_abort_on_nan ? "on" : "off") << "\n";
    for (size_t i = 0; i < gpus.size(); ++i) {
        const auto& g = gpus[i];
        LOG_STREAM_INFO() << "[CONFIG] GPU " << g.device_id << " (" << g.name << ") — " << g.role
                  << " — " << g.vram_gb << " GB total\n";
        if (g.dense_gb > 0)
            LOG_STREAM_INFO() << "[CONFIG]   Dense: " << g.dense_gb << " GB\n";
        if (g.kv_cache_gb > 0)
            LOG_STREAM_INFO() << "[CONFIG]   KV Cache: " << g.kv_cache_gb << " GB (" << g.kv_bits << "-bit)\n";
        if (g.moe_cache_gb > 0)
            LOG_STREAM_INFO() << "[CONFIG]   MoE Cache: " << g.moe_cache_gb << " GB\n";
    }
    LOG_STREAM_INFO() << "[CONFIG] =============================================\n";
}

inline void EngineConfig::write_sample(const std::string& path) {
    std::ofstream f(path);
    f << R"(# MoEStream Engine Configuration
# See configs/ for example profiles (single_gpu, dual_gpu, triple_gpu)

model_path = "/models/GLM-5.2-UD-IQ2_M/"
# dense_model_path = "/models/GLM-5.2-Q4_K/"  # optional: separate dense weights

context_size   = 131072   # max context window (tokens)
max_new_tokens = 4096
temperature    = 0.7
top_p          = 0.95
drop_pages     = false    # true = evict expert pages from OS cache after use

# cpu_affinity_layers: MoE layer indices that should NEVER dispatch on GPU
# or have their experts uploaded/promoted into VRAM -- experts for these
# layers load into host RAM only and always compute on CPU. Use this for
# layers whose I/O time dominates their compute time (a real run's own
# per-layer expert=/disk= breakdown tells you which -- a layer where disk
# read time already exceeds what a GPU matmul would save isn't worth a VRAM
# upload, and that upload just steals cache capacity from layers where it
# DOES help). Empty (default) = no layers affected.
# cpu_affinity_layers = [4, 5, 7]

# direct_io: read routed-expert slabs with O_DIRECT instead of buffered
# reads (Linux only). Requires the model to have been repacked with
# tools/gguf_tool pack (general.alignment >= 4096) -- a plain/un-repacked
# GGUF logs a warning and falls back to buffered reads automatically,
# never a hard failure. See tools/README.md for the repack workflow.
direct_io      = false

# cuda_cufile: NVIDIA cuFile (GPUDirect Storage) -- DMA a genuine "double
# miss" expert (not VRAM-resident AND not already in the host RAM tier)
# straight from NVMe into a VRAM cache slot, skipping the host-RAM hop
# entirely. Requires direct_io=true above (cuFile registers the SAME
# O_DIRECT file descriptor) AND GPUDirect Storage + the nvidia-fs kernel
# module actually installed on this machine (a separate install from the
# base CUDA Toolkit -- see CMakeLists.txt's cuFile detection at configure
# time, and the `-> cuFile (GPUDirect Storage):` build-summary line). Never
# used when an expert is already RAM-resident, regardless of this setting --
# only ever replaces a disk read that would happen anyway. Falls back to the
# existing RAM-mediated upload path on any failure (GDS not installed,
# driver/handle/buffer registration failure) -- never a hard failure.
cuda_cufile    = false

# prefill_warm_via_prefetcher: Phase 2 (prefill's cold-start VRAM warm-up)
# normally loads candidates synchronously, one at a time, blocking inside
# the per-layer critical path (a real run showed 325 such warm-ups for an
# 11-token prompt, entirely serialized into the reported prefill time).
# When true, it instead pushes hints to the same background prefetcher
# decode's own PILOT prediction already uses, so the actual read+promote
# work happens on a worker thread instead of blocking here. Trade-off, not
# a strict win -- see EngineConfig's own comment. Default false (unchanged
# legacy behavior); A/B via t_prefill_warm_time / overall Prefill wall time
# before trusting it.
prefill_warm_via_prefetcher = false
# prefill_opportunistic_promote: only meaningful when
# prefill_warm_via_prefetcher=true above. When true, a hint that lands in
# RAM before Phase 3 needs it gets opportunistically promoted to VRAM
# right then (cross-layer follow-up, decode-thread), instead of waiting
# for Phase 3's own synchronous dispatch to do it. REAL A/B, CONCLUSIVE,
# DEFAULT OFF: four runs ON landed at prefill=0.56 tok/s; OFF landed at
# 0.60 (matches the pre-feature baseline). Root cause: this pass competes
# with Phase 2's own hint-pushing loop for the same free-VRAM-slot budget
# -- eager promotion starves LATER layers of hints entirely (hint volume
# nearly halved, 420 vs ~930), losing the RAM-prefetch overlap the whole
# feature exists to provide. Not a promotion-accuracy problem (every
# candidate is still a confirmed real pick, never a wrong bet) -- a
# resource-contention one. See CLAUDE.md.
prefill_opportunistic_promote = false

# decode_async_prewarm: decode-time analogue of prefill's own
# upload_expert_active_async batch pre-warm -- batches a token's own real
# active-layer VRAM misses into small async-issue-then-sync groups (size
# capped by decode_async_prewarm_max_batch below) instead of dispatch_expert_
# k's existing fully-sequential blocking cudaMemcpy per miss. DEFAULT OFF:
# an earlier, UNBOUNDED version of this (batch the whole token's ~8
# candidates in one shot) caused a real production incident -- host RAM
# exhaustion, swap thrashing, eventual freeze -- on a config running
# host_ram_gb close to physical RAM. UNVERIFIED ON REAL HARDWARE as of this
# writing even in this capped form; enable deliberately, with real host_ram_gb
# headroom, and watch [SLOW-HOST-ALLOC]/[SLOW-PICKUP] for signs of memory
# pressure before trusting it for a real run. See CLAUDE.md and
# tests/bench_dma_sync_vs_async.cu.
decode_async_prewarm = false
# decode_async_prewarm_max_batch: caps how many misses get batched into one
# async-issue-then-sync group at a time when decode_async_prewarm=true --
# bounds peak simultaneous in-flight footprint (RAM slots pinned + active-
# layer slots mid-DMA at once), at the cost of losing some batching benefit
# for tokens with more misses than this. A starting guess (roughly matches
# the bench file's own tested n=4), not a tuned value.
decode_async_prewarm_max_batch = 3

# attn_qabsorb_gpu: batched BF16 GPU GEMM (cublasGemmStridedBatchedEx, all
# attention heads in one call) for MLA's absorbed-query step (Q_absorbed =
# q_nope @ WUK), replacing the CPU per-head matmul() loop. Real measured cost
# on GLM-5.2: ~60ms/decode-token, 100% CPU today. No effect on DeepSeek-V4
# (dsv4_attention_gguf has no absorption/WUK at all -- structurally separate
# path). Per-model/hardware configurable rather than a blanket default:
# DeepSeek-V4's own CPU attention path already runs much faster relative to
# GLM-5.2's absorbed path on this hardware, so a win measured on one
# model+GPU pairing should not be assumed to generalize -- always re-verify
# with a real same-prompt A/B, watching the "MLA qabsorb BF16 batched GPU
# path" shutdown report line for real call counts / CPU-fallback rate.
attn_qabsorb_gpu = false
# attn_qabsorb_gpu_max_vram_gb: hard ceiling on the one-time startup WUK BF16
# upload (warm_mla_absorb_cache), on top of a real cudaMemGetInfo headroom
# check -- a layer whose WUK doesn't fit under this budget just stays on the
# CPU qabsorb path (see n_qabsorb_gpu_fallback_cpu in the shutdown report),
# never a hard failure.
attn_qabsorb_gpu_max_vram_gb = 2.0

# attn_score_gpu: 3-GEMM + softmax GPU path for MLA's score phase (decode
# only), replacing the CPU AVX-512 per-head loop -- real isolated bench at
# GLM-5.2's actual dims showed 5x-51x speedup, GROWING with context length
# (unlike qabsorb, this phase's CPU cost is unbounded with context for
# GLM-5.2), see tests/bench_score_cpu_vs_gpu.cu and CLAUDE.md for the full
# writeup. Same per-model caveat as attn_qabsorb_gpu: DeepSeek-V4 bounds
# this phase with a sliding window, so it's already cheap there -- always
# re-verify with a real same-prompt A/B, watching the "MLA score GPU path"
# shutdown report line for real call counts / CPU-fallback rate.
attn_score_gpu = false
# attn_score_gpu_max_vram_gb: hard ceiling on the persistent per-layer KV
# latent cache GPU mirror (d_lc_by_layer/d_rc_by_layer), allocated once at
# startup sized to context_size so it never needs a mid-generation realloc
# -- on top of a real cudaMemGetInfo headroom check. A layer whose mirror
# doesn't fit under this budget just stays on the CPU score path (see
# n_score_gpu_fallback_cpu in the shutdown report), never a hard failure.
attn_score_gpu_max_vram_gb = 2.0
# attn_score_gpu_max_context: caps the KV latent cache GPU mirror's
# ALLOCATED size independently of context_size -- 0 (default) means no cap
# (mirror sized to the full context_size). On a VRAM-scarce GPU, lowering
# this frees room for WUV's own BF16 upload (which competes for the SAME
# attn_score_gpu_max_vram_gb budget) WITHOUT reducing the model's real
# context_size -- the score GPU path just falls back to CPU for any decode
# position beyond this cap, a correctness-safe, already-tested fallback.
# See attn_score_gpu_max_context's own comment (engine_config.hpp) for the
# real VRAM math this was added to solve.
attn_score_gpu_max_context = 0

# unpacked_expert_format: set true ONLY for a model directory produced by
# `gguf_tool pack --unpack` (per-expert tensors, gate/up/down laid out
# physically tight and contiguous per expert -- NOT the default
# --align-only packed format). Enables a single combined DMA read per
# expert (gate+up+down in one request) on both the decode thread's
# io_uring batching AND the VRAM-shadow bypass fast path -- without this
# flag, an --unpack'd model silently loses BOTH of those (they're gated by
# the packed _exps tensor name, which --unpack removes) and runs slower
# than the default format, not faster. A startup sanity check validates
# every layer's every expert is complete, uniformly sized, and physically
# tight before any generation starts -- any violation is a FATAL error to
# stderr and the run refuses to start (never a silent slow fallback for
# this specific flag). Leave false for the default/recommended
# --align-only packed format.
unpacked_expert_format = false

# io_uring: batch the async prefetcher's per-expert reads (up to 3:
# gate/up/down) through one io_uring submission instead of sequential
# blocking preads (Linux only, kernel >= 5.6). Opt-in, independent of
# direct_io -- combine both for O_DIRECT reads batched via io_uring, or use
# either alone. Falls back to sequential preads automatically on any
# failure (old kernel, sandboxed/container environment, etc.), never a hard
# failure. Only affects the background prefetcher, not the synchronous
# decode-thread miss path.
io_uring       = false

# io_uring_workers: caps the io-wq kernel worker pool the decode thread's
# own io_uring ring registers (IORING_REGISTER_IOWQ_MAX_WORKERS) -- only
# meaningful together with io_uring=true. This is a real OS thread count,
# not free: default 8 matches where most consumer/prosumer NVMe drives
# reach near-peak throughput for reads this size (~4-13MB); going much
# higher rarely helps further and starts competing for CPU with everything
# else this process runs (omp_max_threads, expert_fetch_threads,
# prefetch_worker_threads), since io-wq workers are a separate kernel-
# managed pool not counted against omp_max_threads' own budget. No-op
# (falls back to kernel default sizing) on kernels whose headers predate
# this opcode, never a hard failure.
io_uring_workers = 8

# disk_batch_padding: pad Phase B's real cache-miss submission
# (RAMSafetyManager::get_experts_batch) with speculative reads of
# historically-hot but not-currently-resident experts for the SAME layer,
# riding the SAME already-necessary io_uring submission real misses
# trigger. Requires io_uring=true. A real benchmark (tools/gguf_dma.cpp)
# measured disk throughput climbing from ~2.1 GB/s at 1 concurrent
# request-group to ~2.5-2.6 GB/s at 8-16, while production's real Phase B
# batch size averages only ~2.7 experts/call -- this tries to close that
# gap without the bandwidth contention a separate competing prefetcher
# thread already caused (see CLAUDE.md). V1 only ever claims genuinely
# FREE RAM slots -- it never evicts a real resident, so it's zero risk to
# cache quality, at the cost of likely engaging mostly during a run's
# early ramp-up window before the RAM tier saturates (watch
# n_pad_no_free_slot in the shutdown report). Default false, unverified on
# real hardware -- A/B against disk_batch_padding=false before trusting it.
disk_batch_padding = false

# disk_batch_padding_target: target TOTAL experts (real+speculative) per
# Phase B submission. Clamped to [1,21] -- 21 = floor(Ring::CAPACITY(64)/3),
# the worst-case (packed-format, 3 requests/expert) bound that keeps total
# requests within the ring's hard capacity regardless of format.
disk_batch_padding_target = 8

# disk_batch_padding_evict_dup: V1 of disk_batch_padding above only ever
# claims genuinely FREE RAM slots -- once the RAM tier saturates (the normal
# steady state under any fixed host_ram_gb budget, not just early ramp-up)
# it can barely engage at all (watch n_pad_no_free_slot vs n_pad_reads_issued
# in the shutdown report). This lets padding ALSO reclaim a RAM slot whose
# expert is already VRAM-resident (a provably redundant RAM copy -- decode's
# real dispatch resolves VRAM residency first and never touches this RAM
# slot while it stays VRAM-resident) once free slots run out. Default false,
# unverified on real hardware -- A/B against disk_batch_padding_evict_dup=
# false (same disk_batch_padding=true baseline) before trusting it.
disk_batch_padding_evict_dup = false

# prefetch_worker_threads: number of concurrent worker threads servicing
# the async prefetcher's hint queue (disk read + optional async H2D
# promotion). Default 2. Each gets its own io_uring ring when io_uring=true.
# More threads means hints get STARTED sooner (fixes a real "worker hadn't
# even begun this read yet" gap seen in prefetch instrumentation), at the cost
# of more concurrent contention on ram_mutex/cache_mutex. (There used to
# also be a prefetch_staging_slots knob here, sizing a separate pinned-host
# staging pool the prefetcher held prefetched-but-not-yet-consumed slabs
# in -- removed when the prefetcher was unified with the RAM-tier cache: a
# worker now writes straight into a real cache slot via
# RAMSafetyManager::claim_or_wait, the same structure the decode thread's
# own cache hits read from, so there is no second pool left to size.)
prefetch_worker_threads = 2

# prefetch_pilot_k: how many candidate experts the pilot router pushes as
# speculative prefetch hints per layer transition. Default 8 (was a
# hardcoded std::min(8, topk)). Lower this (globally or per-layer, see
# prefetch_pilot_k_per_layer below) if the [PREFETCH-LAYER] shutdown report
# shows a layer's hints are rarely "really required" -- each unneeded hint
# still costs a worker-thread read and a staging slot that a genuinely
# useful hint (this layer's or another's) could have used instead.
prefetch_pilot_k = 8

# prefetch_pilot_k_per_layer: per-layer override for prefetch_pilot_k, one
# value per layer index (e.g. [8,8,8,0,0,4] disables prefetch for layers 3
# and 4 and halves it for layer 5). Missing/negative entries fall back to
# prefetch_pilot_k. Default empty = every layer uses prefetch_pilot_k
# uniformly. Different layers have very different routing concentration
# (see [PIN-CONCENTRATION] at startup) -- a flat layer's pilot prediction
# is little better than a guess, so hints there are disproportionately
# likely to be wasted; this lets you turn those down without touching
# layers where prediction is genuinely good.
# prefetch_pilot_k_per_layer = [8, 8, 8, 0, 0, 4]

# prefetch_pilot_depth: how many layers ahead the pilot predicts AND pushes
# real prefetch hints for. 1 (default) = only l+1, the original pilot. 2 =
# ALSO predicts and hints l+2 (free -- reuses l+1's own input vector, no
# extra compute). l+2's measurement (recall counters, "Pilot recall by
# rank" report) always runs regardless of this setting, so you can check
# whether depth-2 looks worth it BEFORE flipping this to 2. On GLM-5.2,
# l+2's own fetch-worthy recall measured ~55% vs l+1's ~74% -- meaningfully
# worse, real, not free of cost once wired into real dispatch (queue
# contention, wrong-guess bandwidth). UNVERIFIED ON REAL HARDWARE for
# wall-clock impact -- only the recall numbers are measured so far.
prefetch_pilot_depth = 1

# prefetch_pilot_k_depth2: how many of l+2's predicted top-K (ranked by the
# pilot's own confidence, rank 0 = highest) are actually pushed as real
# hints. Independent of prefetch_pilot_k. Only takes effect when
# prefetch_pilot_depth = 2. Check the fetch-worthy 'Pilot recall by rank
# (DEPTH-2...' shutdown-report line first -- if accuracy drops off
# sharply past rank 2-3, there's little point pushing ranks 4-7 too.
prefetch_pilot_k_depth2 = 8

# Host RAM (DDR) budget for the MoE expert LRU cache, in GB.
# 0 = auto-detect (88% of currently-available system memory).
host_ram_gb    = 0
verbose_moe_log = false
promote_hit_threshold = 1

# Phase A VRAM expert residency, not reactive-only eviction.
# pin_threshold: once a VRAM-resident expert's host-cache hit_count crosses
# this, it's pinned (never evicted) for the rest of the process -- stops
# thrashing hot experts out within one conversation. 0 disables pinning.
pin_threshold = 8
# pin_file: usage-frequency file (written at shutdown, read at next startup)
# used to preload the measured hot set into VRAM before the first prompt.
# Empty string disables preload (a usage file is still written at shutdown).
pin_file = ".moestream_usage"
# prefill_pin_file: separate usage file tracking ONLY prefill-phase routing
# selections, not decode's. Decode contributes far more routing volume per
# request than prefill does, so in the combined pin_file histogram a
# decode-favored expert can outrank one prefill genuinely needs every run --
# this file, when set, is preferred for prefill's own preload ranking
# (falling back to pin_file per-layer where this file has no data yet).
# Empty (default) = not used, falls back entirely to pin_file.
# prefill_pin_file = ".moestream_usage_prefill"
# freeze_usage_stats: debug/A-B-testing aid. When true, pin_file and
# prefill_pin_file are both still READ at startup exactly as configured,
# but NEITHER is overwritten at shutdown -- stops the additive
# accumulate-forever behavior from shifting the startup pin/preload
# ranking underneath you between runs. Real drift was confirmed on
# hardware even under a literally identical, temperature=0 prompt (GPU
# reduction order isn't bit-exact call-to-call, and this model's IQ2_XXS
# routing margins are thin enough for that jitter to occasionally flip a
# close top-8 tie) -- turn this on whenever comparing two configs/code
# paths against each other, so only the thing actually under test can
# explain the result. Leave off (default) for normal use, where that
# accumulation across genuinely different sessions is the whole point.
freeze_usage_stats = false
# pin_topk_per_layer: how many of each layer's hottest experts to preload.
# 0 = auto-size to fill available VRAM expert-cache capacity. Only used when
# pin_strategy="per_layer_topk" below.
pin_topk_per_layer = 0
# pin_strategy: how pinned_vram_capacity gets DISTRIBUTED across layers.
#   "per_layer_topk" (default): every layer gets the same topk count
#     (pin_topk_per_layer above). Safe -- no layer ever gets zero pins --
#     but wastes budget on layers whose routing is close to uniform (see
#     the [PIN-CONCENTRATION] log line for whether that's actually your
#     model/workload).
#   "global_topn": pool every (layer,expert,count) triple across ALL layers
#     and pin the globally-hottest ones first, until pinned_vram_capacity is
#     spent -- concentrates the budget where concentration is real, at the
#     cost of a flat layer possibly getting fewer than
#     pin_global_floor_per_layer pins. Valid because every layer sees the
#     same number of routing decisions per token in this engine, so raw
#     counts ARE comparable across layers.
# Switchable, not a hard replacement -- a different model or a very
# different prompt distribution than whatever wrote pin_file could have a
# flatter or more skewed profile than whichever strategy was tuned for. See
# CLAUDE.md.
pin_strategy = "per_layer_topk"
# pin_global_floor_per_layer: only used when pin_strategy="global_topn".
# Every layer with known usage history is guaranteed at least this many
# pinned slots (taken off the top of pinned_vram_capacity) before the
# remainder is spent on the global ranking -- prevents a flat layer from
# dropping to fully-zero protection. 0 = pure global ranking, no floor.
pin_global_floor_per_layer = 1
# pin_vram_lru_fraction: preload the next-hottest experts (ranked just below
# pin_topk_per_layer) into VRAM but leave UNPINNED -- a normal reactively-
# evictable slot, just pre-warmed instead of starting cold. Fraction of the
# VRAM tier's own real per-layer capacity, e.g. 0.1 = 10%, 0.3 = 30%.
# 0.0 = disabled (default -- real testing found nonzero defaults here made
# a short benchmark run SLOWER, more disk read, not less; A/B test your
# own workload before enabling
pin_vram_lru_fraction = 0.0
# pin_ram_fraction: same idea, next tier down, preloaded into the host RAM
# tier (previously always started completely cold). Fraction of
# ram_mgr.slots_per_layer. 0.0 = disabled (default, same reasoning as above).
pin_ram_fraction = 0.0

# vram_evict_demote_to_ram: when a VRAM cache slot is evicted, salvage its
# bytes into the RAM tier via a D2H copy instead of dropping them (so a
# later re-request serves from RAM instead of paying a full disk read
# again). New, untested on real hardware, opt-in only -- default false.
vram_evict_demote_to_ram = false

# dsv4_outproj_gpu: run DeepSeek-V4 attention's grouped output projection
# (attn_output_a) on GPU via cuBLAS instead of CPU -- the weight is
# memory-bandwidth-bound on CPU (re-read from host RAM every decode token),
# measured 12x+ faster once VRAM-resident. Costs ~5.8GB VRAM, committed
# permanently once each layer's weight is first used (not reclaimable like
# the expert caches) -- only enable if you have real headroom on the
# CoreEngine GPU (see dense_gb/kv_cache_gb sizing above). Default false.
dsv4_outproj_gpu = false

# ram_vram_promote_top_k / ram_vram_promote_adaptive: pilot-driven RAM->VRAM
# speculative promotion, calibrated against the "ram-only" per-rank metric
# (see the "Pilot recall by rank (DEPTH-1, ram-only" line in the run log) -- NOT the
# overall or fetch-worthy metrics printed next to it, which measure
# different populations. Mutually exclusive: setting both disables both
# (with a startup warning). Both default off/0 -- no behavior change from
# the pre-existing reactive-hit-count promotion until one is set.
#
# Static: unconditionally promote pilot's rank [0, top_k) "already_ram"
# picks. 0 = off.
ram_vram_promote_top_k = 0
# Adaptive: self-calibrate from this session's own live counters -- promote
# rank 0, 1, 2... while that rank's accuracy >= adaptive_threshold AND its
# sample count >= min_samples, stopping at the first rank that fails either
# (contiguous-prefix cutoff, doubles as the cold-start guard).
ram_vram_promote_adaptive = false
ram_vram_promote_adaptive_threshold = 0.30
ram_vram_promote_min_samples = 30

# ram_cache_layer_adaptive: give layers with flatter/less-predictable
# routing more RAM-tier cache slots and concentrated layers fewer, instead
# of splitting the total budget evenly -- uses the same pin_file usage
# history as the [PIN] preload above. No-op until that history exists.
# New, untested on real hardware, opt-in only -- default false.
ram_cache_layer_adaptive = false

# expert_fetch_threads: OMP threads used to parallelize disk reads for one
# token's genuine (non-prefetch-served) cache misses within a layer's topk
# selection. Default 4
# against your own model file/drive (buffered reads, increasing thread
# count until throughput stops scaling) rather than assuming this value.
expert_fetch_threads = 4

# omp_max_threads: process-wide default OpenMP team size for every
# parallel region that doesn't set its OWN num_threads(...) -- previously
# unset entirely, so attention/dense matmuls, the CPU expert-fallback
# dequant+dot path, and the reference dequant path all defaulted to
# OpenMP's own runtime default (every available core), with nothing
# stopping several of them running concurrently and oversubscribing the
# machine. A real run showed the OS itself become unresponsive to other
# commands from exactly this -- the CPU expert-fallback path alone can
# fire thousands of times per run, each call spinning up to every core,
# three times (gate/up/down), fully uncoordinated with anything else
# running. 0 = auto (hardware_concurrency() - 2, floor 1) -- leaves
# headroom for the OS/other processes by default; set explicitly to
# override. expert_fetch_threads above still applies to its own one
# region regardless of this value (explicit num_threads() always wins
# over the process default).
omp_max_threads = 0

# attn_score_threads: dedicated OMP team size for JUST attention_layer.cpp's
# causal score/softmax/value-aggregate loop -- separate from omp_max_threads
# above, which this region would otherwise share with every other parallel
# region. Motivated by a real-hardware finding (2026-08-13): raising
# expert_fetch_threads/omp_max_threads 8->10 for a real, measured disk-I/O
# win (decode 0.68->0.70 tok/s) simultaneously raised this loop's own
# per-token cost ~74.8->83.5ms/token in the SAME real run -- this is the one
# attention sub-step with no GPU path at all, so it directly competes with
# expert_fetch_threads' background disk-read threads for the same physical
# cores once omp_max_threads is raised alongside it. 0 = no override, falls
# back to whatever omp_max_threads' own effective value already is. Set
# LOWER than omp_max_threads to trade some of this loop's own parallelism
# back for less contention -- UNVERIFIED which value (if any) is a net win,
# a real, previously-uninvestigated lever, not yet A/B tested.
attn_score_threads = 0

# Three-tier VRAM caching. pinned_vram_capacity is a global ceiling on
# permanently-pinned experts (see pin_topk_per_layer above -- the effective
# per-layer pin count is min(pin_topk_per_layer, floor(pinned_vram_capacity
# / total_layers)), uniform across layers with no remainder distribution,
# so exact round numbers aren't always reachable). move_on_promotion frees
# the RAM slot immediately on a RAM->VRAM promotion instead of leaving the
# expert double-resident in both tiers (see [TIER-OVERLAP] in the perf
# report if this regresses).
pinned_vram_capacity = 1024
move_on_promotion = true

# Only consulted when move_on_promotion=false (COPY semantics). Real A/B on
# real hardware (see CLAUDE.md) showed COPY can be a genuine throughput win
# over MOVE when host_ram_gb has enough headroom for the model -- but on a
# tighter budget (bigger model, less RAM), COPY-mode RAM slots only get
# reclaimed reactively (whenever some OTHER expert's miss needs the exact
# same slot), which can leave redundant RAM+VRAM copies sitting unreclaimed
# for a whole session on a lightly-trafficked layer. Set to e.g. 0.9 to
# proactively reclaim a layer's redundant slots once that layer's RAM
# occupancy crosses 90%, instead of relying purely on reactive pressure.
# 0.0 (default) preserves plain COPY behavior, no threshold applied.
ram_copy_reclaim_threshold = 0.0

# Active layer area: a small dedicated VRAM buffer holding the CURRENT
# layer's decode-time cold-miss experts, so a not-yet-proven-hot expert can
# still get a GPU dispatch without evicting a real LRU resident for it.
# active_layer_capacity slots, each sized to one expert slab -- costs
# roughly active_layer_capacity x largest-expert-slab-size of VRAM,
# additional to and separate from each GPU's own moe_cache_gb; make sure
# the vault GPU has that much genuinely free or allocation silently fails
# and this whole feature falls back to plain LRU behavior (see the
# "[GPU] Active layer area allocation failed" log line -- also check
# [GPU] Active layer area allocated for the actual GB requested).
# active_layer_promote_hit_threshold: hit count (repeat access from a LATER
# token while still resident) at which an active-layer expert is promoted
# into the real VRAM LRU cache instead of being freed once its layer's
# window closes -- default 1 since this window is much narrower than the
# RAM tier's whole-conversation lifetime.
use_active_layer_area = true
active_layer_capacity = 256
active_layer_promote_hit_threshold = 1

# Logging. Empty log_path (default) keeps everything on stderr.
# log_level: "VERBOSE" | "DEBUG" | "INFO" (default) | "WARN" | "ERROR"
# VERBOSE adds high-volume structural traces (per-layer hidden-state norms
# across the full model depth) on top of DEBUG's per-token/per-expert
# diagnostics (cache hits/misses, VRAM promotions, CPU dequant fallback
# notices) -- both are very high volume, intended for active debugging only.
# log_path = "output.log"
log_level = "INFO"

# Tensor-trace debugging (a llama.cpp eval-callback analog for this engine's
# imperative forward pass). Requires log_level = "VERBOSE" to have any
# effect. trace_filter narrows VERBOSE tensor-trace output to specific tags
# (comma-separated, matched as an anchored prefix regex against each trace
# call site's own tag, e.g. "HC", "MLA", "EXPERT-GPU", "SILU-TRACE-GPU") --
# empty (default) shows every tag. trace_abort_on_nan hard-aborts the
# process at the FIRST tensor a NaN/Inf checker flags, instead of logging
# and continuing.
# trace_filter = "HC,MLA"
trace_abort_on_nan = false

# GPU 0: Core Engine — Dense layers + KV cache
[[gpus]]
device_id    = 0
name         = "NVIDIA GeForce RTX 5060 Ti"
role         = "dense"          # "dense", "moe", or "dense+moe"
vram_gb      = 16.0             # 0 = auto-detect
dense_gb     = 7.5              # GB for dense layers (attn, norms, embed, lm_head)
kv_cache_gb  = 4.5              # GB for KV cache
kv_bits      = 4                # KV cache quantization: 4 or 32
moe_cache_gb = 0                # no expert cache on this GPU

# GPU 1: Expert Vault — MoE LRU cache
[[gpus]]
device_id    = 1
name         = "NVIDIA GeForce RTX 3090"
role         = "moe"
vram_gb      = 24.0
moe_cache_gb = 23.0

# GPU 2: Expert Vault (optional — skipped if not present)
[[gpus]]
device_id    = 2
name         = "NVIDIA GeForce RTX 3090"
role         = "moe"
vram_gb      = 24.0
moe_cache_gb = 23.0
)";
    f.close();
    std::cerr << "[CONFIG] Sample config written to " << path << "\n";
}
