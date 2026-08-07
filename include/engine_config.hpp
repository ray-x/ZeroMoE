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

    // Logging. Empty log_path (default) keeps everything on stderr, matching
    // prior behavior. log_level filters by severity: VERBOSE shows
    // high-volume structural traces (e.g. per-layer hidden-state norms
    // across the full model depth) on top of everything below; DEBUG shows
    // every per-token/per-expert diagnostic (cache hits/misses, VRAM
    // promotions, CPU dequant fallback notices); INFO (default) hides those
    // and shows only startup info and real warnings/errors.
    std::string log_path;
    std::string log_level = "INFO";

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
    direct_io        = doc.root.get_bool("direct_io", direct_io);
    cuda_cufile      = doc.root.get_bool("cuda_cufile", cuda_cufile);
    prefill_warm_via_prefetcher = doc.root.get_bool("prefill_warm_via_prefetcher", prefill_warm_via_prefetcher);
    unpacked_expert_format = doc.root.get_bool("unpacked_expert_format", unpacked_expert_format);
    io_uring         = doc.root.get_bool("io_uring", io_uring);
    io_uring_workers = doc.root.get_int("io_uring_workers", io_uring_workers);
    disk_batch_padding        = doc.root.get_bool("disk_batch_padding", disk_batch_padding);
    disk_batch_padding_target = doc.root.get_int("disk_batch_padding_target", disk_batch_padding_target);
    prefetch_worker_threads = doc.root.get_int("prefetch_worker_threads", prefetch_worker_threads);
    host_ram_gb      = doc.root.get_double("host_ram_gb", host_ram_gb);
    verbose_moe_log  = doc.root.get_bool("verbose_moe_log", verbose_moe_log);
    promote_hit_threshold = doc.root.get_int("promote_hit_threshold", promote_hit_threshold);
    pin_threshold    = doc.root.get_int("pin_threshold", pin_threshold);
    pin_file         = doc.root.get_str("pin_file", pin_file);
    prefill_pin_file = doc.root.get_str("prefill_pin_file", prefill_pin_file);
    pin_topk_per_layer = doc.root.get_int("pin_topk_per_layer", pin_topk_per_layer);
    pin_strategy = doc.root.get_str("pin_strategy", pin_strategy);
    pin_global_floor_per_layer = doc.root.get_int("pin_global_floor_per_layer", pin_global_floor_per_layer);
    pin_vram_lru_fraction = doc.root.get_double("pin_vram_lru_fraction", pin_vram_lru_fraction);
    pin_ram_fraction      = doc.root.get_double("pin_ram_fraction", pin_ram_fraction);
    vram_evict_demote_to_ram = doc.root.get_bool("vram_evict_demote_to_ram", vram_evict_demote_to_ram);
    ram_cache_layer_adaptive = doc.root.get_bool("ram_cache_layer_adaptive", ram_cache_layer_adaptive);
    expert_fetch_threads = doc.root.get_int("expert_fetch_threads", expert_fetch_threads);
    omp_max_threads = doc.root.get_int("omp_max_threads", omp_max_threads);
    // NOTE: these four were previously missing from parsing entirely --
    // every configs/*.toml's `pinned_vram_capacity`/`move_on_promotion`/
    // `active_layer_capacity`/`use_active_layer_area` lines were silently
    // no-ops, only "working" because each one happened to match its
    // compiled default (1024/true/256/true). Any run that tried to
    // actually CHANGE one of these via TOML got the default instead, with
    // no warning.
    pinned_vram_capacity = doc.root.get_int("pinned_vram_capacity", pinned_vram_capacity);
    move_on_promotion    = doc.root.get_bool("move_on_promotion", move_on_promotion);
    active_layer_capacity = doc.root.get_int("active_layer_capacity", active_layer_capacity);
    use_active_layer_area = doc.root.get_bool("use_active_layer_area", use_active_layer_area);
    active_layer_promote_hit_threshold = doc.root.get_int("active_layer_promote_hit_threshold",
                                                            active_layer_promote_hit_threshold);
    raw_prompt           = doc.root.get_bool("raw_prompt", raw_prompt);
    log_path         = doc.root.get_str("log_path", log_path);
    log_level        = doc.root.get_str("log_level", log_level);

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
    LOG_STREAM_INFO() << "[CONFIG] io_uring (batched prefetcher reads): " << (io_uring ? "requested" : "off")
              << (io_uring ? " (falls back to sequential preads if unsupported -- see [IO-URING] log)" : "") << "\n";
    if (io_uring) {
        LOG_STREAM_INFO() << "[CONFIG] io_uring workers (decode-thread ring's io-wq pool): " << io_uring_workers << "\n";
    }
    if (disk_batch_padding)
        LOG_STREAM_INFO() << "[CONFIG] Disk batch padding: ON (target " << disk_batch_padding_target
                           << " experts/submission, speculative reads from route_freq)\n";
    LOG_STREAM_INFO() << "[CONFIG] Prefetch worker threads: " << prefetch_worker_threads << "\n";
    LOG_STREAM_INFO() << "[CONFIG] Prefetch pilot_k (hints/layer transition): default=" << prefetch_pilot_k
                       << (prefetch_pilot_k_per_layer.empty() ? "" : ", " + std::to_string(prefetch_pilot_k_per_layer.size())
                           + " layer override value(s) set (see [PREFETCH-LAYER] report)")
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
    LOG_STREAM_INFO() << "[CONFIG] VRAM evict -> RAM demote: "
              << (vram_evict_demote_to_ram ? "ON (D2H-salvage evicted experts into the RAM tier)" : "off") << "\n";
    LOG_STREAM_INFO() << "[CONFIG] RAM cache layer-adaptive: "
              << (ram_cache_layer_adaptive ? "ON (per-layer capacity by historical need, not uniform)" : "off") << "\n";
    LOG_STREAM_INFO() << "[CONFIG] Expert fetch threads: " << expert_fetch_threads << "\n";
    LOG_STREAM_INFO() << "[CONFIG] OMP max threads (process-wide default): "
              << (omp_max_threads > 0 ? std::to_string(omp_max_threads) : std::string("auto"))
              << "\n";
    LOG_STREAM_INFO() << "[CONFIG] Pinned VRAM capacity: " << pinned_vram_capacity
              << " | move_on_promotion: " << (move_on_promotion ? "MOVE" : "COPY") << "\n";
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
