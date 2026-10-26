// engine_config.hpp — Runtime configuration for ZeroMoE multi-GPU inference
// Loaded from TOML config files. Specifies model paths, GPU roles, memory
// layout.
#pragma once
#include "logger.hpp"
#include "resident_memory_policy.hpp"
#include "graph_settings_config.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

// Per-GPU configuration
struct GpuConfig {
  int device_id = -1;        // nvidia-smi GPU index
  std::string name;          // Human label, e.g. "NVIDIA GeForce RTX 3090"
  std::string role;          // "dense", "moe", "dense+moe"
  std::string model_path;    // GGUF file/dir for this GPU (optional override)
  double vram_gb = 0.0;      // Usable VRAM budget (0 = auto-detect)
  double dense_gb = 0.0;     // GB reserved for dense layers
  double kv_cache_gb = 0.0;  // GB reserved for KV cache
  int kv_bits = 4;           // KV cache quantization (4 or 32)
  double moe_cache_gb = 0.0; // GB reserved for expert LRU cache
  // > 0: size the cache to FILL the VRAM that dense weights, KV and graph workspace do not claim, ignoring
  // moe_cache_gb (so a copied or stale config cannot waste memory). Dual-role GPU: cache = free VRAM after the
  // dense weights are resident - this many GB. Vault GPU: cache = total VRAM - this many GB (it includes the
  // driver's ~1 GB). Set it to what the graph needs after the cache plus a margin: the startup log prints the
  // resulting free VRAM and the graph's own measured peak ([MIMO2-VRAM]).
  double moe_cache_reserve_gb = 0.0;

  // 2026-08-28 (multi-GPU active-layer pool, see EngineConfig::
  // active_layer_routing_policy's own comment for the full design):
  // per-GPU opt-in into the active-layer temp pool (current-layer misses
  // + DEPTH-1/ATTN-PILOT one-layer-ahead speculative fills), separate
  // from this GPU's `role`/`moe_cache_gb` (the PERSISTENT main cache --
  // this is deliberately a different pool, since a GPU might reasonably
  // want to participate in one but not the other). Default 0 = does not
  // participate. If NO configured GPU has a nonzero weight, the whole
  // multi-GPU mechanism is inert and behavior is BYTE-IDENTICAL to
  // legacy: allocate_active_layer_area falls back to its original single-
  // pool-on-vault_gpus[0]-or-core_gpu logic, using the global (legacy)
  // active_layer_capacity for sizing -- zero risk for a config that
  // doesn't opt in. active_layer_weight is a RELATIVE routing weight
  // (any positive integer scale; only ratios between participating GPUs
  // matter, e.g. 40/40/20 or 4/4/2 are equivalent) -- see
  // route_active_layer_gpu's own comment (dense_dispatch.hpp) for how
  // it's used. active_layer_capacity_slots is this GPU's OWN pool size
  // (in expert-slab-sized slots, same unit as the legacy global
  // active_layer_capacity) -- sized independently per GPU, not derived
  // from the weight (a GPU could have a small weight but still want a
  // roomy pool, or vice versa -- these answer different questions:
  // weight decides HOW OFTEN this GPU is chosen, capacity decides how
  // much it can hold once chosen).
  int active_layer_weight = 0;
  int active_layer_capacity_slots = 0;
};

// Full engine configuration
struct EngineConfig {
  // Root graph_* keys, collected verbatim and validated by the typed resolver.
  zeromoe::graphsettings::ConfigValues graph_settings;
  bool serve_api = false;
  std::string api_host = "127.0.0.1";
  int api_port = 8080;
  std::string api_model = "zeromoe";
  // Model paths
  std::string model_path; // Primary model GGUF (experts + architecture)
  std::string model_format = "gguf";
  std::string exl3_backend = "native";
  std::string exl3_ggml_bridge_model_path;
  int exl3_ggml_bridge_layer = -1;
  std::vector<int> exl3_ggml_bridge_layers;
  bool exl3_ggml_graph_capture = false;
  bool exl3_ggml_routed_graph_capture = false;
  std::string exllamav3_python;
  std::string exllamav3_runner_path;
  std::string exllamav3_cuda_schedule = "auto";
  bool model_load_only = false;
  std::string metadata_model_path;
  std::string prompt;
  bool exl3_gpu = false;
  bool exl3_gpu_hc = false;
  int exl3_gpu_device = 0;
  double exl3_gpu_cache_gb = 8.0;
  int exl3_gpu_expert_layer_begin = 0;
  int exl3_gpu_expert_split_layer = -1;
  int exl3_gpu_split_device = 1;
  double exl3_gpu_split_cache_gb = 8.0;
  double exl3_cpu_expert_cache_gb = 24.0;
  bool exl3_cpu_expert_int8_vnni = false;
  std::string
      dense_model_path; // Optional: separate GGUF for dense layers (e.g. 4-bit)
  // Optional: a standalone "shared" MTP/NextN checkpoint (Qwen4Exp's own
  // mtp-*-shared-*.gguf convention) -- carries ONLY the dedicated draft-
  // block tensors (blk.<n_layers>.*: attention, MoE, HC, nextn.* bridge/
  // head) and deliberately omits token_embd.weight/output.weight, which
  // this engine instead reuses directly from the already-loaded base
  // model (see StreamingEngine::mtp_model_file / has_mtp_model). Distinct
  // from GLM5NEXT/GLM_DSA's own NextN tensors, which already live inside
  // model_path itself and need no second file.
  std::string mtp_model_path;

  // Generation parameters
  int context_size = 4096;   // Max context window (tokens)
  int max_new_tokens = 1024; // Max tokens to generate per response
  float temperature = 0.7f;
  float top_p = 0.90f;
  bool drop_pages = false; // fadvise DONTNEED after expert read
  bool cpu_only = false;   // force ALL compute to CPU (no GPU dispatch)

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

  // 2026-08-28: removes the ONE remaining blocking wait left in the whole
  // active-layer pre-warm system -- decode_async_prewarm's own per-chunk
  // sync_active_layer_batch() call (a real cudaStreamSynchronize on the
  // decode thread). Motivated directly by a user challenge: correctly
  // prefetching more MoE experts wasn't moving decode throughput AT ALL,
  // and neither disk nor PCIe bandwidth were anywhere near saturated --
  // i.e. the bottleneck isn't bytes/sec, it's SERIALIZATION. The
  // speculative_promote_to_active_layer path (see its own history in
  // CLAUDE.md) already proved the fix for exactly this shape of problem:
  // find_active_expert's poll-before-trust check (cudaEventQuery on the
  // slot's own upload_event) treats "DMA not landed yet" as a plain MISS
  // rather than blocking -- and across dozens of real runs that poll came
  // back "not ready" 0.0% of the time, meaning the lead time was always
  // enough. That mechanism is fully generic (gpu_topology.cpp's find_
  // active_expert never gated it on pilot_speculative_fill) -- it already
  // works correctly for a decode_async_prewarm-issued slot with ZERO code
  // changes on the destination side. The only reason decode_async_prewarm
  // couldn't already rely on it is the SOURCE side: sync_active_layer_
  // batch was also what told this loop it was safe to ram_unpin() the RAM
  // slot the DMA reads from. Fixed by transferring that protection to
  // MoEExpertBlock::pending_copy_event (RAMSafetyManager::ram_mark_
  // pending_copy_and_unpin, ram_manager.hpp) instead -- the exact same
  // self-healing, poll-checked mechanism apply_promotion_bookkeeping's own
  // COPY branch already uses for main-cache promotions, generalized to a
  // caller with no synchronous confirmation point of its own. Removing
  // the blocking sync doesn't reduce total DMA-engine busy time (that's
  // already confirmed bytes-bound, not call-count-bound, by the max_batch
  // 3->6 experiment -- see CLAUDE.md) -- what it removes is CPU wall-clock
  // time the decode thread spent doing NOTHING but waiting for a
  // cudaStreamSynchronize to return, when it could have been issuing the
  // NEXT chunk's DMA (or advancing to dispatch_expert_k for an already-
  // ready expert) instead. This is a genuinely different overlap
  // opportunity from the one three earlier sessions already ruled out
  // (GPU *compute* stream idle-ness) -- it never depends on compute being
  // slow enough to hide DMA behind, only on there being SOME other CPU-
  // side work available to do instead of blocking. Default false
  // (unchanged blocking behavior) so this is a clean, revertible A/B, not
  // a silent behavior change; only takes effect when decode_async_prewarm
  // is also true. UNVERIFIED ON REAL HARDWARE as of this writing.
  bool decode_async_prewarm_nonblocking = false;

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

  // glm5next_kda_gpu: run GLM5NEXT's per-key-channel recurrent KDA
  // delta-rule update on CUDA. The token dependency remains sequential, but
  // the H*head_dim*head_dim state update no longer occupies a CPU loop.
  // Host state is synchronized at the call boundary so prompt-cache and
  // checkpoint semantics remain unchanged. Falls back to the CPU reference
  // on any CUDA allocation/copy/launch failure. Default ON for GLM5NEXT;
  // other model families never read this flag.
  bool glm5next_kda_gpu = true;

  // glm5next_router_gpu: run GLM5NEXT's learned router matmul, gating,
  // bias-aware top-k, and weight preparation on CUDA. The compact route table
  // is copied back for the existing expert-cache bookkeeping. Falls back to
  // the CPU router when the weight is not device-resident or CUDA fails.
  bool glm5next_router_gpu = true;

  // GLM5NEXT-only grouped GPU expert prefill. Unlike the generic
  // moe_prefill_batched switch, this is explicitly family-gated and keeps
  // the known-unsafe generic GLM/DeepSeek toggle independent. It groups the
  // current prefill token/expert pairs and issues grouped GPU expert FFNs;
  // cache residency and routing validation remain on the host for safety.
  // Default off so existing GLM, Qwen, and DeepSeek configurations are
  // unchanged until this opt-in is present.
  bool glm5next_moe_prefill_gpu = false;

  // GLM5NEXT-only dense KDA projection fusion. The eight KDA projection
  // matmuls are submitted as one GGML CUDA graph for S>1. The graph is
  // best-effort and falls back to the existing dense_proj calls on any
  // mismatch or backend failure. Default off to preserve all existing model
  // families and provide a clean A/B.
  bool glm5next_dense_prefill_fusion = false;

  // qwen4exp_attn_gpu / qwen4exp_gdn_gpu: DEPRECATED NO-OPS. The layered
  // dense-attention and chunkwise-GDN CUDA paths were removed (Qwen4Exp runs
  // as a ggml full graph; kernels are in obsolete/cuda). The keys are still
  // parsed so existing configs load; the values are ignored.
  bool qwen4exp_attn_gpu = false;
  bool qwen4exp_gdn_gpu = false;

  // qwen4exp_gdn_conv_gpu: GDN's causal short conv + SiLU on the GPU
  // (cuda/gdn_conv1d.cu), replacing attn_causal_conv1d_silu's CPU AVX-512
  // path. MEASURED 2026-09-05: conv was 0.43-0.46s of a ~5.0s p=512 prefill
  // (the single largest GDN sub-bucket) -- already fixed once for CPU
  // cache-locality (see CLAUDE.md's "conv1d loop order: 29.4x" entry) and
  // now purely CPU-DRAM-bandwidth-bound. The op has no sequential
  // dependency beyond the kernel width, so it's embarrassingly parallel
  // across (token, channel) -- a clean fit for GPU HBM bandwidth instead.
  // Falls back to the byte-identical CPU path on any allocation/H2D/D2H
  // failure. Also used by the GLM5Next KDA path.
  bool qwen4exp_gdn_conv_gpu = false;

  // Qwen4Exp-only router pilot. The pilot projects the current layer's
  // device-authoritative HC stream-0 snapshot through the next layer's gate
  // and records the predicted expert set against the real next-layer route.
  // This is deliberately separate from the conventional hidden->norm->gate
  // pilot used by GLM/DeepSeek, whose inputs and routing semantics differ.
  // Default off so every other model, and Qwen configs that do not opt in,
  // retain their existing behavior.
  bool qwen4exp_router_pilot = false;
  int qwen4exp_router_pilot_k = 10;
  // Optional next-layer prefetch from the Qwen pilot. When enabled, ranks
  // that are already resident in RAM are copied into the next layer's
  // temporary active-layer area with cudaMemcpyAsync; colder predictions are
  // sent as ordinary disk/RAM hints. The active-layer copy is event-tracked
  // and the target-layer dispatch waits on that exact expert if necessary.
  bool qwen4exp_router_pilot_prefetch = false;
  // Keep prediction/accuracy K independent from the number of experts loaded
  // speculatively. 0 means use qwen4exp_router_pilot_k for compatibility.
  int qwen4exp_router_pilot_prefetch_k = 0;
  // Qwen layer indices for which estimation-based prefetch is disabled.
  // Prediction and accuracy collection still run for these layers.
  std::vector<int> qwen4exp_router_pilot_prefetch_exclude_layers;
  // Optional Qwen-only prediction allow-list. When non-empty, the pilot runs
  // only for these target layers. This is independent of production routing,
  // usage-file pinning, and all other model families.
  std::vector<int> qwen4exp_router_pilot_predict_layers;
  // Decode-only Qwen optimization: skip the advisory pilot snapshot/router
  // pass when the target layer's complete expert set is resident OR the
  // actual routed top-k hot set has stayed resident for two tokens. The
  // latter avoids scanning all experts and is reset by an exact-route miss.
  // Other model families do not read this option.
  bool qwen4exp_router_pilot_skip_fully_cached = false;
  // Decode-only optimization: when a target layer is explicitly excluded
  // from estimation-based prefetch, skip the pilot itself as well. This keeps
  // the exclusion list useful for performance; set false during an accuracy-
  // collection run when excluded layers should still be measured.
  bool qwen4exp_router_pilot_skip_prefetch_excluded_layers = false;

  // Qwen4Exp-only decode path: use the grouped GGML/CUDA MUL_MAT_ID routed
  // FFN for S==1, collapsing the per-token/per-expert cache and dispatch
  // loop into one graph submission per MoE layer. Separate from the older
  // CPU-oriented moe_decode_batched experiment, which is not appropriate
  // for Qwen IQ1_M. Default off for every model/config.
  bool qwen4exp_decode_ggml_moe = false;
  // Qwen4Exp-only full-graph decode topology experiment. A complete
  // preloaded DDR arena may use one scheduler graph containing the CUDA
  // router and CPU MUL_MAT_ID expert path. Layers without a complete arena
  // retain the ordinary lazy-loading path; other model families ignore this.
  bool qwen4exp_decode_scheduler = false;
  // Qwen4Exp-only decode handoff: keep the HC pre-mix result on the GGML
  // CUDA backend for the S==1 router and grouped MoE graph. This is separate
  // from qwen4exp_decode_ggml_moe so the latter can be benchmarked with the
  // established host-router boundary. Default off for every model/config.
  bool qwen4exp_decode_device_hc = false;
  // Qwen-only S==1 transfer policy. Batch RAM-resident expert slabs through
  // the pinned staging arena before the CUDA upload, reducing the per-expert
  // memcpy/unpack submission count. Prefill and every other model family
  // retain the existing direct-slab path unless the diagnostic environment
  // override is used.
  bool qwen4exp_decode_batch_stage = false;
  // Qwen-only exact-Q8 HC decode kernels. Off by default and ignored by all
  // other model families.
  bool qwen4exp_decode_custom_hc_q8 = false;
  // Qwen-only HC [D,S] result readback for CPU-routed layers. When enabled,
  // the producer event is waited on by a reusable nonblocking CUDA copy
  // stream and the D2H copy lands in pinned staging memory before the CPU
  // consumer reads it. Off by default until the runtime A/B is validated.
  bool qwen4exp_hc_async_readback = false;
  // Qwen-only S==1 router policy. The GPU router is useful for batched
  // prefill, but its graph launch/synchronization/readback boundary is slower
  // than the AVX-512 host router for decode on the RTX 3090. Keep this opt-in
  // so enabling the GPU router for prefill does not force the slower decode
  // boundary. The Qwen prediction pilot has its own GPU-router path.
  bool qwen4exp_decode_gpu_router = false;
  // Qwen4Exp-only grouped CPU fallback for explicitly selected
  // cpu_affinity_layers. Other model families never consult this switch.
  bool qwen4exp_cpu_ggml_moe = false;
  // Qwen-only direct DDR-strided CPU implementation: reuse one Q8_K
  // activation quantization for the selected experts and fuse IQ1_M gate/up.
  // This is separate from qwen4exp_cpu_ggml_moe so the persistent GGML graph
  // remains the default, verified CPU implementation.
  bool qwen4exp_cpu_strided_grouped = false;
  // Qwen-only opt-in for the one-team CPU projection path. When the direct
  // DDR-strided path is active, fuse gate/up, SwiGLU, activation quantization,
  // and down into one OpenMP team for IQ1_M/IQ2_XXS -> IQ4_NL layers. This is
  // deliberately separate from the strided-path switch so existing configs
  // keep their CPU implementation unchanged until both are enabled.
  bool qwen4exp_cpu_fused_grouped = false;
  // Qwen-only CPU/GPU overlap for cpu_affinity_layers: run the routed CPU
  // MUL_MAT_ID graph on a persistent worker while the always-on shared expert
  // runs on CUDA. Disabled by default for an explicit A/B and ignored by all
  // other model families. The bounded layer count prevents the two-graph
  // shared-only/host-add handoff from becoming slower than the normal CPU
  // path when a large fraction of the trunk is offloaded.
  bool qwen4exp_cpu_gpu_overlap = false;
  int qwen4exp_cpu_gpu_overlap_max_layers = 16;
  // Qwen-only CPU routed-result handoff: after the CPU MUL_MAT_ID graph
  // finishes, feed its [D] result directly to the device shared-expert graph
  // instead of using the host-facing shared-expert fallback.  This is kept
  // separate from the two-graph overlap experiment above so it can be tested
  // without adding a shared-only launch or a second device add graph.
  bool qwen4exp_cpu_shared_device = false;

  // CPUOffload branch Milestone A: split selected decode layers into the
  // existing VRAM expert path and a persistent DDR-backed CPU worker. This is
  // deliberately separate from Option B's static pinning switch so the
  // experiment can be enabled without changing the baseline configuration.
  bool qwen4exp_cpu_offload = false;
  std::vector<int> qwen4exp_cpu_offload_layers;
  // Force the first N routed slots cold for a deterministic A/B while keeping
  // the router's IDs and weights unchanged.  A fixed expert ID can be forced
  // instead with qwen4exp_cpu_offload_force_expert.
  int qwen4exp_cpu_offload_force_cold_count = 0;
  int qwen4exp_cpu_offload_force_expert = -1;
  // Milestone B: queue the CPU result through a pinned host/device ring and
  // fence the shared CUDA stream on the copy event.  Milestone A remains the
  // default until the event protocol has been A/B validated.
  bool qwen4exp_cpu_offload_event_fence = false;
  int qwen4exp_cpu_offload_ring_slots = 2;

  // GPU staging for host-resident decode layers: reuses the CUDA-graph
  // host-node bridge (same technique as qwen4exp_cpu_offload above) to
  // gather a listed layer's real per-token top-k routed experts from the
  // preloaded DDR arena into a small, fixed, topk-sized compact VRAM buffer
  // each decode token, so its MoE FFN compute can run as ordinary GPU
  // MUL_MAT_ID nodes instead of always falling to CPU AVX-512. Decode
  // (S==1) only -- see qwen4exp_full_graph_decode_run/build_gpu_stage_node
  // in ggml_qwen4exp_graph.cpp for why prefill is out of scope. Independent
  // of qwen4exp_cpu_offload: a layer should be listed in at most one of the
  // two lists (build_moe checks staging first).
  bool qwen4exp_gpu_stage_enabled = false;
  std::vector<int> qwen4exp_gpu_stage_layers;

  // Decode-only router-usage tap -- see StreamingEngine::
  // qwen4exp_track_router_usage's own comment. Real per-token overhead
  // (one D2H + one host callback per token); only worth enabling for a
  // usage survey, never for production decode.
  bool qwen4exp_track_router_usage = false;

  // prompt_cache: reuse the previous request's end-of-prompt state when the
  // new prompt EXTENDS it (or repeats it), so only the new suffix is
  // prefilled. Restricted to families whose carried recurrent/conv/PLE state
  // is fully enumerated (Qwen4Exp, GLM5NEXT) -- see
  // StreamingEngine::PromptStateCache. Default off.
  bool prompt_cache = false;

  // Optional GLM5NEXT NextN/MTP speculative decoding.  The target trunk
  // remains authoritative; keep this disabled unless the model/config has a
  // validated draft block.
  bool mtp_enabled = false;
  int mtp_draft_tokens = 3;
  // Qwen4Exp only: run MTP on the full-graph target with a graph-native draft
  // and target-verified speculation (requires mtp_enabled and mtp_model_path).
  // Replaces the layered Qwen4ExpMtpSpeculator; never falls back to it.
  bool mtp_graph = false;

  // moe_gpu_accum: accumulate routed-expert outputs on the DEVICE and download
  // one [S,D] per layer, instead of downloading every expert's [n,D] output so
  // a CPU loop can do the weighted add. Measured motivation: 1.685M
  // expert-token rows x 2560 floats = ~16.9 GB of D2H per 3511-token prefill
  // (dn=16.81s) against cmp=0.27s of actual expert matmul. Requires the
  // generic moe_prefill_batched path or the GLM5NEXT-specific
  // glm5next_moe_prefill_gpu path (both hook the grouped dispatch path).
  bool moe_gpu_accum = false;

  // dense_bf16_kquant: extend dense_bf16_prefill_gemm from Q8_0 to Q5_K/Q6_K.
  // Separate knob because the Q8_0 half is verified on GLM-5.3 while this half
  // is verified on qwen4exp only, and because GLM's 309 Q5_K + 81 Q6_K dense
  // tensors are large enough that on a VRAM-tight GPU the watermark refuses
  // most of them anyway. Requires dense_bf16_prefill_gemm.
  bool dense_bf16_kquant = false;

  // expert_gpu_activation: compute the routed-expert SwiGLU on the device.
  // The old path downloaded gate AND up to the host, ran silu/mul/clamp on
  // the CPU, then uploaded the product back -- two extra PCIe crossings per
  // expert, ~12,000 experts per 512-token prefill, measured dn=13.68s of a
  // 48s prefill against cmp=0.20s for the expert matmuls. The device-side
  // NaN/Inf guard is preserved (8-byte readback, not 2*S*moe_inter floats).
  // Automatically yields to the host path whenever a SILU/SWIGLU/EXPERT-GPU
  // trace is active, since those traces exist to inspect this exact data.
  bool expert_gpu_activation = false;

  // Memory guard (include/memory_guard.hpp): a background watchdog on
  // /proc/meminfo MemAvailable. Warns early, and BLOCKS further host-memory
  // growth below the stop threshold so the allocator never enters the
  // multi-second-per-call thrash observed on this machine at host_ram_gb=96
  // with a full page cache. MemAvailable (not MemFree) is the signal --
  // MemFree excludes reclaimable page cache and reads near-zero when healthy.
  // Minimum FREE VRAM (MB) to preserve on a device when growing a dynamic
  // buffer. NOT a fraction of total: this engine intentionally fills VRAM
  // (the expert cache alone sits at ~96% by design), so a llama.cpp-style
  // 0.9-of-total watermark refuses every dynamic allocation forever -- that
  // was measured at 11,632 refusals and a 2.6x slowdown. A free-memory floor
  // leaves room for CUDA contexts / cuBLAS workspaces / driver staging, which
  // allocate outside our budgets.
  long vram_min_free_mb = 256;

  bool mem_guard = true;
  long mem_guard_warn_mb = 32768;  // warn with 32 GiB left
  long mem_guard_stop_mb = 20480;  // retain at least 20 GiB for the host
  int  mem_guard_poll_ms = 500;

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

  // dsv4_score_gpu: cuBLAS GPU path for DeepSeek-V4's raw-SWA and HCA
  // attention score phase (QK^T + softmax + weighted-V), replacing the
  // CPU AVX-512 per-head loop -- scoped to ratio==0 (raw-SWA) and
  // ratio==128 (HCA) layers only. CSA (ratio==4) layers ALWAYS stay on
  // CPU regardless of this flag: their Lightning-Indexer-selected blocks
  // are a small, data-dependent, per-query gather that would need a
  // second softmax merged back into this one (an online-softmax/log-sum-
  // exp merge across the CPU/GPU boundary) -- real correctness-risk
  // surface area for an already-small workload, deliberately out of scope
  // here. Same rationale as attn_score_gpu for WHY this wasn't built
  // sooner (see that field's own comment: "DeepSeek-V4 bounds this phase
  // with a sliding window, so it's already cheap there") -- that's still
  // true of the raw window alone, but HCA's compressed-block count grows
  // with total context the same unbounded way GLM-5.2's own MLA score
  // does (capacity = context_size/128), and the raw window has to share
  // ONE joint softmax with those HCA rows regardless, so accelerating
  // both together is the natural (and only correctness-simple) unit of
  // work once HCA is in play. Decode (S==1) only, same scope restriction
  // as attn_score_gpu. Default OFF: UNVERIFIED ON REAL HARDWARE -- this
  // Mac has no CUDA toolchain to test with; see
  // tests/bench_dsv4_score_cpu_vs_gpu.cu for the bench to run before
  // flipping this on, matching this project's standing convention for
  // anything not yet real-hardware-verified end-to-end.
  bool dsv4_score_gpu = false;

  // Per-layer VRAM budget cap for dsv4_score_gpu's two persistent GPU
  // mirrors (d_dsv4_kvc_by_layer + d_dsv4_hca_by_layer, cuda_manager.hpp),
  // checked independently per mirror (a layer whose KVc ring doesn't fit
  // still gets no GPU path even if its HCA cache would). Same
  // cudaMemGetInfo-checked, never-assume-headroom discipline as
  // attn_score_gpu_max_vram_gb. Smaller in practice than attn_score_gpu's
  // own budget: the KVc ring is fixed-size (sliding_window*qh*4 bytes,
  // not context_size-scaled) and the HCA cache is already 128x smaller
  // than an equivalent full-context mirror (cache_capacity =
  // context_size/128).
  double dsv4_score_gpu_max_vram_gb = 1.0;

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

  // qwen4exp_graph_expert_vram_budget_gb: full-graph decode (S==1) ONLY --
  // generalizes cpu_affinity_layers/device_expert_layers' whole-layer
  // residency choice into a genuine PER-EXPERT split within a layer. When
  // >0, the graph builder reads the existing usage-ranking data (pin_file,
  // the same ".zeromoe_usage" histogram pin_topk_per_layer/global_topn
  // already consult) and greedily spends this many GB of VRAM on the
  // globally hottest routed experts, across every layer NOT already
  // claimed by cpu_affinity_layers or device_expert_layers/
  // device_expert_partial_gate_up (those two mechanisms are checked first
  // and always win). A layer that gets some but not all of its experts
  // placed this way runs TWO ggml_mul_mat_id calls per projection instead
  // of one (one over the VRAM-resident subset, one over the rest, masked
  // and merged) -- see ggml_qwen4exp_graph.cpp's plan_expert_vram_split/
  // build_moe. Default 0.0 = feature off, identical to today's behavior.
  double qwen4exp_graph_expert_vram_budget_gb = 0.0;

  // Option B: static power-law placement for one Qwen layer.  The full
  // graph keeps the selected experts in a permanent CUDA graph arena and
  // leaves the complementary experts in the fixed-stride DDR arena.  A
  // value below zero disables the option; this is deliberately separate
  // from the older partial-layer environment experiment, whose GPU-only
  // path has different correctness semantics.
  int qwen4exp_option_b_layer = -1;
  int qwen4exp_option_b_top_n = 0;
  float qwen4exp_option_b_min_coverage = 0.80f;
  // When true, top_n is only a ceiling: each selected layer's real hot-set
  // size shrinks to the smallest N (per that layer's own real usage ranking)
  // that still reaches qwen4exp_option_b_min_coverage, instead of every
  // layer always paying for exactly top_n experts regardless of how
  // concentrated its own routing is. Off by default -- existing configs
  // that rely on a fixed, deterministic per-layer VRAM cost keep that
  // behavior unchanged.
  bool qwen4exp_option_b_right_size_top_n = false;
  // Global pool mode (0 = off, the per-layer behaviour above). When > 0 the
  // selected layers' (layer, expert) pairs are ranked together by usage and
  // the top K of the whole pool become the static hot sets, so concentrated
  // layers get many hot experts and flat layers few (see
  // include/option_b_global_hot.hpp). qwen4exp_option_b_top_n then acts only as
  // a per-layer ceiling (set it to the model's expert count for no ceiling);
  // min_coverage, right_size_top_n and fill_vram are ignored. Each eligible
  // layer with usage keeps its own hottest experts up to the requested
  // floor, clamped to the ceiling and available K/layer budget. Listing all
  // layers with floor >= 1 and K >= the eligible layer count gives 0/N/0.
  int qwen4exp_option_b_global_top_k = 0;
  int qwen4exp_option_b_global_floor_per_layer = 1;
  // Optional explicit layer allow-list for the multi-layer Option B test.
  // When non-empty, only these CPU-side layers receive the static hot set;
  // expert ranking within each selected layer still comes from pin_file.
  std::vector<int> qwen4exp_option_b_selected_layers;
  // In concentrated-only mode, cap the number of automatically selected
  // layers after ranking them by top-N historical traffic coverage.
  int qwen4exp_option_b_max_layers = 0;
  // Start selected layers at top_n and use remaining graph-arena capacity for
  // additional lower-ranked experts, subject to the VRAM safety floor.
  bool qwen4exp_option_b_fill_vram = false;
  // Explicitly omit the legacy GPU LRU for Option-B full-graph runs. This
  // preserves a configured zero cache instead of treating it as auto-size;
  // if graph execution is declined, the legacy fallback has no GPU cache.
  bool qwen4exp_option_b_disable_legacy_cache = false;
  // When true and qwen4exp_option_b_layer is -1, rank every layer from
  // pin_file and apply Option B only where top_n experts cover at least the
  // requested historical route share. Flat layers remain on the CPU path.
  bool qwen4exp_option_b_concentrated_only = false;
  // Layer-13 single-slot staging proof.  -1 keeps Option B byte-identical;
  // a non-negative expert is appended to the static GPU split and refreshed
  // from the pinned DDR arena before each decode graph replay.
  int qwen4exp_option_b_stage_expert = -1;
  // Number of cold experts to force into the active-layer proof.  The
  // default preserves the original one-slot laboratory; values up to the
  // model top-k are useful for stressing concurrent H2D staging.
  int qwen4exp_option_b_stage_count = 1;

  // Configurable full-graph switches.  Environment variables remain valid
  // overrides for existing A/B scripts; these fields make the master-based
  // Option B profile self-contained.
  // Qwen4Exp execution policy: full-graph prefill/decode are the defaults.
  // A config may explicitly disable them only for a documented exemption or
  // a diagnostic; the runtime must not silently select layered decode after
  // a requested full graph fails to build or execute.
  bool qwen4exp_full_graph = true;
  bool qwen4exp_full_graph_decode = true;
  bool qwen4exp_prefill_state_handoff = false;

  // Decode-time (S==1) CPU-offload cap for genuine disk-cold misses:
  // instead of a REAL cache miss (not RAM- nor VRAM-resident, would need
  // an SSD read) always continuing on to a synchronous VRAM upload, up to
  // this many of a single token/layer's real misses instead skip the
  // upload and are dispatched via the existing CPU dequant+matmul
  // fallback (expert_ffn_cpu) once their weight bytes land in the RAM
  // tier -- the SSD->DDR read itself is unavoidable either way (CPU
  // compute reads from the same RAM-tier slab_buffer the GPU path would
  // have uploaded), but the DDR->VRAM DMA (this codebase's own measured
  // dominant decode-time cost, active_layer_prewarm_issue+sync, ~17-18s/
  // request -- see CLAUDE.md) is skipped entirely for the capped experts.
  // Fiddler-style CPU/GPU orchestration (arxiv 2402.07033): for a single-
  // token decode GEMV, a cold expert's PCIe transfer can cost more than
  // just computing it on CPU directly. 0 (default) = off, identical to
  // today's behavior. Valid range 0-8 (8 = every real miss this token/
  // layer goes to CPU, i.e. CPU-only for cold misses specifically -- NOT
  // the same as cpu_only_mode(), which forces EVERYTHING including warm
  // VRAM-resident hits to CPU). Real per-layer/per-token data gathered
  // BEFORE this field was wired to anything (new report_layer_miss_
  // stats()/[LAYER-MISS] instrumentation, perf_timer.hpp) showed a real
  // GLM-5.2 decode-time population of 0/1/2/3 simultaneous misses in one
  // call at 74.4%/21.9%/3.3%/0.4% respectively (never 4+ on that run) --
  // cap=1 already covers 96.3% of real miss events, cap=2 covers 99.6%.
  // See CLAUDE.md's "CPU-offload feasibility" entries for the full
  // writeup and that data. Scoped to decode only (S==1) by construction
  // -- prefill's own dispatch path (moe_forward_gguf's S>1 branch) is
  // structurally separate and never reads this field; the Fiddler
  // argument is decode-specific anyway (a batched S>1 GEMM amortizes the
  // PCIe cost per-token far better than a lone S=1 GEMV does).
  int cpu_offload_max_experts = 0;

  // Prefill-only (S>1), additive/purely-a-widening-of-the-existing-call
  // config: when moe_forward_gguf's Phase 3 loop resolves token s's own
  // genuine cache misses via RAMSafetyManager::get_experts_batch, it ALSO
  // opportunistically folds in the next prefill_dispatch_lookahead_tokens
  // tokens' own not-yet-resident picks into that SAME combined io_uring-
  // batched call -- purely to warm the RAM tier ahead of when THEIR OWN
  // token's turn comes up later in the loop, never dispatched by token s
  // itself. Motivated by a real, measured finding: with prefill_warm_via_
  // prefetcher already on, Phase 2's own whole-layer hint push still left
  // phase_b (blocking, prefetcher-missed disk reads) at ~35% of a long
  // (237-token) prefill's wall time, because Phase 2 pushes ALL of a
  // layer's hints ONCE, right before Phase 3 starts consuming them
  // immediately -- the very first tokens in a layer get almost no lead
  // time by construction, regardless of prompt length. Widening EVERY
  // token's own get_experts_batch call to also pull in upcoming tokens'
  // needs gives those specific experts real additional wall-clock lead
  // time (however many tokens' worth of dispatch happen before their own
  // turn) without touching dispatch ORDER, dispatch_expert_k, or any
  // per-token scratch shape at all -- a deliberately narrower, lower-risk
  // alternative to actually reordering/deferring dispatch across tokens
  // (which would require turning scratch.expert_ptrs/vram_loc_hint/
  // vram_shadow_blocks/pending_gpu into per-chunk 2D state -- a real
  // restructuring of the single most incident-prone loop in this
  // codebase, not attempted here). 0 = off (exact prior behavior, each
  // token's call scoped to only its own topk picks). Decode is
  // unaffected by construction -- moe_forward_gguf's S==1 branch never
  // reads this field, and S==1 means "the next N tokens" wouldn't be a
  // real batch of independent tokens waiting to be routed anyway.
  // VERIFIED ON REAL HARDWARE -- and a real, clean REGRESSION, not a win.
  // Same 237-token prompt: 225.28s/1.05 tok/s (off) -> 260.23s/0.91 tok/s
  // (8 tokens ahead). [PREFILL-LOOKAHEAD] showed 1.12M candidates scanned,
  // only 37,555 (3.4%) genuinely new -- but fetching those pushed ~176GB
  // MORE real disk traffic onto a drive already sitting near its ~4 GB/s
  // bandwidth ceiling (confirmed independently the same session via a
  // prefetch_worker_threads 6->8 A/B that also found no headroom there).
  // disk= grew 74.78s->182.44s (309.5GB@4.14GB/s -> 485.2GB@2.66GB/s --
  // the achieved RATE dropped too, plausibly from breaking up what would
  // otherwise be larger/more sequential reads into more numerous, smaller,
  // competing ones) and phase_b (blocking) got WORSE (77.35s->110.10s),
  // not better. Root cause: on THIS hardware, disk bandwidth is a hard
  // ceiling, not a scheduling/timing problem -- adding more speculative
  // fetch demand onto an already-saturated resource competes with fetches
  // that were already correctly timed, rather than getting free overlap.
  // Same failure shape as this codebase's own repeated VRAM-promotion
  // over-eagerness regressions (ram_vram_promote_top_k, the reverted
  // prefill_opportunistic_promote), one tier down (disk instead of VRAM).
  // Default OFF. Kept implemented (not deleted) -- correct, 0 errors,
  // real instrumentation ([PREFILL-LOOKAHEAD]) -- matching this project's
  // convention for a well-evidenced negative result. Would need a genuine
  // bandwidth increase (faster/more drives) or a smaller working set
  // (bigger cache budgets) before this mechanism could plausibly help;
  // not something a config flip alone can fix.
  int prefill_dispatch_lookahead_tokens = 0;

  // Prefill-only (S>1). Phase 2's cold-start warm-up (moe_forward_gguf)
  // pushes hint(s) for a layer's WHOLE real candidate set (every expert
  // any token in the batch needs, from the already-computed real router
  // output -- 100% certain, not a guess) before Phase 3 dispatches
  // anything. The default order is purely by scratch.expert_freq
  // descending (most-shared expert first) -- but Phase 3 then dispatches
  // tokens 0, 1, 2, ... in sequence, so a LATER token (say, token 200 of
  // 237) already gets real lead time "for free" (however long Phase 3
  // took to reach it) regardless of push order, while the EARLIEST
  // tokens (0-10ish) get almost none, since Phase 3 starts consuming
  // essentially the instant Phase 2 finishes pushing. When true, sorts
  // candidates by LayerScratch::first_needed_by_token ascending instead
  // (the earliest-needing token's experts pushed first) -- directly
  // targeting the tokens that currently get the least lead time, at the
  // cost of pushing some high-frequency-but-late-needed experts later
  // than before. Mutually orthogonal to WHICH experts get hinted (same
  // candidate set either way, from the same real expert_freq>0 filter)
  // -- purely a push-ORDER experiment.
  //
  // VERIFIED ON REAL HARDWARE, default flipped to true (2026-08-30): 2x2
  // real A/B on a 16-token prompt (freeze_usage_stats=true + a real
  // prefill_pin_file, so both sides read the identical frozen startup
  // state -- no usage-file-drift confound between runs), --ngen 0 so only
  // prefill is measured. Consistent, reproducible win both repeats:
  //   prefill wall   16.49s/16.62s -> 14.92s/14.88s  (~9% faster)
  //   tok/s          0.97/0.96     -> 1.07/1.08       (~11% higher)
  //   phase_b (main-thread blocking read)  2.86s/2.79s -> 0.43s/0.43s
  //   phase_c (cheap wait-for-in-flight)   0.22s/0.23s -> 0.74s/0.77s
  //   by-main-thread claims (real disk read on the main thread)
  //                  333/322       -> 40/43            (~88% down)
  //   by-prefetch claims (a worker won the claim first, as intended)
  //                  2493/2529     -> 2767/2758         (absorbed the rest)
  //   cache hit      74% both sides (unchanged)
  // Mechanism confirmed, not just correlated: this is a "who does the
  // fetching" win, not a "fewer fetches" win -- total real claim volume
  // (hit+main+prefetch) and cache-hit % are essentially identical between
  // baseline and reordered; what moved is which thread pays for each real
  // read. Earlier-needing tokens' experts now get pushed first, giving
  // background prefetch workers real lead time to win the claim before
  // Phase 3's own dispatch loop reaches them, so far fewer real reads fall
  // to the main thread's own blocking phase_b path. (disk= in the
  // [PREFILL] line reports main-thread-only bytes -- confirmed via source,
  // bytes_disk_read/t_disk_read are only touched by get_experts_batch's
  // own Phase B; prefetch-worker reads go into a separate bytes_prefetched
  // counter -- so the 8x disk= drop is this same mechanism, not
  // double-counting or a reduction in total bytes moved.) No observed
  // downside on this test. Single short prompt though -- re-verify at a
  // longer prompt length before assuming the magnitude holds at scale.
  bool prefill_hint_order_by_earliest_token = true;

  // Prefill-only (S>1). Computes the widened (all-tokens) prediction this
  // field's own real-hint-push sibling below (prefetch_widen_all_tokens)
  // consumes, AND always drives the [PILOT-WIDE] recall/coverage/
  // precision measurement regardless of whether real hints are pushed --
  // same "computation always runs, hint-push separately gated" split as
  // ATTN-PILOT's own prefetch_attn_pilot. DEPTH-1's own production
  // prediction samples only the LAST token's hidden state per layer
  // transition (see that block's own "Use only last token... saves
  // compute" comment) -- real accuracy against that one token is high
  // (81.8% measured), but its COVERAGE of what the whole batch actually
  // needs is tiny (topk=8 out of ~230 real unique experts/layer on a
  // 237-token prompt, ~3.5%). This runs the SAME per-token router
  // projection (rmsnorm + dense_proj + gating + bias, byte-identical
  // math to DEPTH-1's own) for EVERY token in the batch instead of just
  // the last one, then takes the UNION of each token's own top-k picks
  // as a wider candidate set -- directly answering "if DEPTH-1 sampled
  // more tokens, how much of the real per-layer demand would it cover"
  // with a real measured number instead of an estimate.
  //
  // VERIFIED ON REAL HARDWARE (2026-08-30, coverage/recall measurement
  // only -- the real-hint-push sibling below is separately unverified):
  // 16-token prompt, freeze_usage_stats=true (no usage-file-drift
  // confound), 91.2-91.3% coverage of the real whole-layer unique-expert
  // demand (up from DEPTH-1's own ~3.5% single-token-coverage estimate at
  // long prompt lengths), true recall vs the last token's own picks
  // ~92.3-92.5% (higher than DEPTH-1's 81.8% -- expected, since the
  // widened set is a strict superset of what DEPTH-1 alone would find).
  // See n_pilot_wide_recall_hits_last's own comment (perf_timer.hpp) for
  // a real mislabeling this session found and fixed in how that number
  // used to be reported.
  //
  // Honest caveat, not yet resolved: at S=16, the widened candidate set
  // (avg 66.6) is nearly as large as the real whole-layer demand itself
  // (avg 65.5) -- doesn't yet prove the mechanism stays cheap/valuable at
  // real prompt-length scale. Real extra compute cost: S dense_proj
  // calls instead of 1 (trivial at S=16; would be ~237 calls/transition
  // at a long prompt -- needs real cost-vs-benefit reconsideration at
  // that scale, not assumed to transfer). See LayerScratch::
  // pilot_wide_pred_ids's own comment for the recall/coverage
  // measurement this feeds. Default off (real compute cost every layer
  // transition when on).
  bool pilot_widen_all_tokens = false;

  // Prefill-only (S>1), requires pilot_widen_all_tokens=true (a no-op
  // otherwise -- the whole widened-prediction block above is skipped
  // entirely when that's off, so there's nothing here to push). Real
  // disk-prefetch hint push wired from the widened prediction's own
  // not-yet-resident members -- mirrors DEPTH-1/ATTN-PILOT's own
  // already_ram/already_vram residency check + async_io_queue.push_hint
  // call exactly (execution_layer.cpp), just applied to the much wider
  // candidate set pilot_widen_all_tokens computes instead of DEPTH-1's
  // fixed 8. Deliberately HINT-PUSH ONLY, no VRAM promotion wired --
  // this codebase has repeated, real evidence (ram_vram_promote_top_k,
  // the reverted prefill_opportunistic_promote, the ATTN-PILOT VRAM-
  // promotion stacking regressions) that widening a SPECULATIVE VRAM-
  // promotion population too far causes real cache churn/regression;
  // widening the much cheaper, disk-tier-only DISK-prefetch population
  // is the safer, more directly-motivated lever for this specific gap
  // ("layer 6 predicts 8 MoEs but uses 1,896 in total" -- the coverage
  // problem this field exists to close). A future session could
  // consider VRAM promotion for a NARROW, high-confidence slice of the
  // widened set, but that would need its own dedicated tuning/A-B, not a
  // simple reuse of this flag. UNVERIFIED ON REAL HARDWARE -- see this
  // field's own real-hardware note once tested; n_pilot_wide_hints_
  // pushed/skipped_resident (perf_timer.hpp) and the "real hint push"
  // suffix on the [PILOT-WIDE] report line are the counters to check.
  bool prefetch_widen_all_tokens = false;

  // Prefill sliding-window RAM capacity (2026-08-30, P1+P2 of a scoped
  // plan -- P3 "trim on window exit" deliberately NOT built yet). See
  // RAMSafetyManager::prefill_window_cap_by_layer's own comment
  // (ram_manager.hpp) for the full design and the real measurement that
  // motivated it.
  //
  // Short version: calculate_safe_cache_capacity divides the RAM budget
  // by ALL n_layers (correct for decode, which traverses every layer per
  // token), giving 76 slots/layer -- but prefill runs layers strictly
  // sequentially, so exactly one layer is live while ~74 sit on unusable
  // budget, and that one live layer really needs ~212 unique experts for
  // a 100-token prompt. Measured consequence: the prefetcher evicts its
  // OWN not-yet-consumed work, 82.2% of all prefetch reads (~97GB of
  // ~118GB) fetched and thrown away unread.
  //
  // When on, a layer's cap grows to its own REAL Phase-1 demand (count of
  // expert_freq>0 -- exact, already computed before Phase 3 dispatches
  // anything, not a prediction) while it is the active prefill layer.
  //
  // DEFAULT OFF, and off is a strict no-op (the override vector stays
  // all-zero, cap_for_layer falls through to exactly its pre-2026-08-30
  // two-tier behavior). Note this is inert for short prompts even when
  // ON: a 16-token prefill needs ~65 experts/layer, already under the 76
  // base cap, so nothing is granted.
  bool prefill_window_cap = false;

  // Tokens per prefill graph batch (llama.cpp's n_ubatch). 0 = the model family's default (GLM 2048, Qwen whole prompt). A prompt longer than the
  // batch runs in several batches that continue each other; a value too large for the device is reduced at startup of the
  // prefill to what fits in VRAM and host RAM (see FullGraphModel::max_prefill_tokens and the family's prepare check).
  int prefill_batch_tokens = 0;

  // Hard ceiling on TOTAL extra RAM (across ALL layers combined) the
  // window above may claim beyond the normal per-layer caps. This is the
  // safety mechanism standing in for the unbuilt P3 trim: nothing lowers
  // a raised cap back down yet, so without this bound an un-trimmed
  // window would grow toward 75 layers x 212 slots ~= 243GB and swap-kill
  // the box (this project has a real scar from exactly that -- see
  // host_ram_gb's swap-thrashing history in CLAUDE.md). Once spent,
  // later layers silently keep their base cap (n_prefill_window_denied_
  // budget). Sized in GB and converted to a slot count against the
  // model's own largest expert slab, so it stays denominated the same way
  // calculate_safe_cache_capacity's budget is.
  double prefill_window_extra_gb = 8.0;

  // prefill->decode retention handoff (2026-08-31). Weight added to a RAM
  // slot's eviction_score per unit of THIS REQUEST's prefill demand, so
  // experts prefill actually routed to survive into the decode that follows
  // instead of being aged out by pure recency.
  //
  // Motivation is measured, not assumed: on real 5-prompt data, ranking the
  // RAM tier by this request's own prefill counts captures 64.1% of the
  // following decode's traffic vs 67.8% for an oracle that already knows
  // decode's answer (ratio 0.946, stable at both 2850 and 5700 slots).
  // Prefill's expert set had ZERO prefill-exclusive entries and covered
  // 95.7% of decode traffic at r=0.921 -- so "what prefill just used" is
  // very nearly the best obtainable predictor of what decode will want, and
  // Phase 1 has already computed it for free.
  //
  // 0 (DEFAULT) = OFF, a strict no-op: eviction_score reverts to exactly its
  // previous last_used_clock + hit_count*FREQ_BONUS_TICKS form. Scale
  // reference: FREQ_BONUS_TICKS is 64, so a weight of 64 makes one prefill
  // selection worth about one real cache hit. Additive bias, NOT a hard pin
  // -- a hard priority here could starve the reactive path the way this
  // project's own over-pinning regressions already have.
  int prefill_retention_weight = 0;

  // Tiered retention width. 0 = keep every stamped expert (no per-layer cap).
  // A layer's width scales with its OWN routing concentration, measured at
  // runtime from expert_freq (so it needs no history and works on the first
  // run): flat layer -> N/3, median -> 2N/3, peaky -> N. Measured payoff at
  // equal N on real data: 9.72 vs 8.17 units of decode traffic per RAM slot
  // (~19% better use of the same memory) versus a flat N everywhere.
  int prefill_retention_top_n = 0;
  // Minimum prefill demand to be stamped at all. A threshold is comparable
  // ACROSS layers in a way a per-layer count is not, which is what stops a
  // flat layer from hoarding slots on near-one-off picks.
  int prefill_retention_min_demand = 1;
  // Tercile cut-points on a layer's top-20 share, from real-data analysis.
  double prefill_retention_flat_cut = 0.636;
  double prefill_retention_peaky_cut = 0.693;
  // Halve every stamp when a request ends, so a later prompt's real demand
  // can outrank an earlier prompt's stale stamps without a bulk purge.
  bool prefill_retention_decay = true;

  // Grace period for a freshly-(re)loaded RAM-tier slot: for
  // ram_grace_period_tokens DECODE tokens after a slot is genuinely
  // loaded (not just re-hit), eviction_score gets +ram_grace_period_bonus
  // added on top of its normal recency+frequency terms. Motivation, from
  // a real 2026-09-04 investigation: RAMSafetyManager's own recency term
  // (last_used_clock) is a raw operation-tick counter, not a token
  // counter -- during a burst of many loads within ONE token (this
  // model's multiple speculative predictors -- DEPTH-1, ATTN-PILOT,
  // prefetch_pilot_k's wide hint-push -- can each trigger several RAM
  // loads per layer transition), a slot loaded a few ticks ago can
  // already look "stale" relative to slots loaded microseconds later in
  // the SAME token, well before it ever got a real chance to be reused.
  // A real run showed the RAM tier (5702-6740 slots) absorbing 5800-8100
  // evictions across a single 56-token request -- close to a full,
  // sometimes more-than-full, turnover within ONE request. This gives a
  // freshly-loaded slot a token-counted (not tick-counted, so immune to
  // intra-token burst effects) floor of protection instead.
  //
  // 0 (DEFAULT) = OFF, strict no-op: eviction_score is byte-identical to
  // its pre-2026-09-04 form. Deliberately ADDITIVE, not a hard exemption
  // (unlike ram_loading/ram_temp_pin_count/copy_still_pending, which DO
  // hard-block eviction) -- a small per-layer pool (as few as 30-40
  // slots) receiving 8+ speculative loads in one token could otherwise
  // deadlock (every slot grace-protected, nothing left to evict) if this
  // were a hard floor; additive-only means it strongly biases the
  // ordering without ever making a slot un-evictable, matching this
  // project's own established, repeatedly-validated "bias, don't block"
  // philosophy (see prefill_retention_weight's own comment above).
  // Scale reference: FREQ_BONUS_TICKS is 64 (one real cache hit).
  //
  // 2026-09-04 REAL A/B, TESTED AND NEGATIVE -- keep OFF. Same prompt,
  // three arms (off / tokens=10,bonus=256 / tokens=20,bonus=1000 -- a 4x
  // stronger dial): RAM tier evictions 5864 / 5859 / 5862, waste 42.1% /
  // 42.3% / 42.2%, decode-only tok/s 1.109 / 1.112 / 1.104 -- all flat
  // within noise, even at 4x the bonus and 2x the window. Root cause:
  // this same session's RAM tier touched ~6045 DISTINCT experts against
  // only 5702 total slots -- the working set already exceeds capacity,
  // so a meaningful share of eviction is mathematically forced regardless
  // of which slot a scoring bias picks as the victim first. This mirrors
  // the 2026-08-27 dynamic-pin-swap finding for the VRAM active-layer
  // pool above (also ruled out for the identical reason: "the real
  // constraint is the working set itself being too large and too flat...
  // not a targeting inefficiency a smarter algorithm could fix"). A
  // grace period can only reorder WHO gets evicted, not reduce the total
  // COUNT -- don't re-attempt this without first growing real capacity
  // (host_ram_gb / ram_cache_layer_adaptive), which is the only lever
  // that could actually change this math.
  int ram_grace_period_tokens = 0;
  // Bonus added to eviction_score while a slot is within its grace
  // window. No effect if ram_grace_period_tokens <= 0.
  int ram_grace_period_bonus = 0;

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

  // prefetch_pilot_k_adaptive (2026-09-13): user request -- "test the most
  // confident numbers, not a guessed flat cut". Real data motivating this
  // (see CLAUDE.md's DeepSeek-V4.1 session memory for the full numbers):
  // fetch-worthy DEPTH-1 accuracy decays hard by rank (k0=93%, k1=86%,
  // k2=65%, k3=50%, k4=37%, k5=27-30%), and the LOW ranks are also the
  // MAJORITY of the fetch-worthy population (more picks need a fresh
  // fetch at low confidence than at high, since high-confidence picks are
  // more often already resident) -- 22.8% of all real prefetches this
  // session were reclaimed by LRU before any hit ever touched them, a
  // direct cost of always fetching all the way to prefetch_pilot_k
  // regardless of a rank's own live accuracy. Instead of guessing a
  // smaller flat number (or the extreme of disabling prefetch entirely),
  // this computes each layer's own effective fetch width from its live,
  // already-collected [PILOT-BY-LAYER-RANK] accuracy (PerfCounters::
  // n_pilot_recall_hits_by_layer_rank / n_pilot_recall_checks_by_layer_
  // rank, dense_dispatch.hpp's adaptive_pilot_k_for_layer) -- a
  // contiguous-prefix walk from rank 0 (matches ram_vram_promote_
  // adaptive's own proven shape below): keep extending the fetch width
  // while the NEXT rank's own live accuracy is still >=
  // prefetch_pilot_k_adaptive_threshold, stop at the first rank that
  // fails. Deliberately only ever NARROWS the existing prefetch_pilot_k/
  // prefetch_pilot_k_per_layer ceiling for that layer, never widens past
  // it -- same "do less for unreliable ranks, never more for reliable
  // ones" safety direction as ram_vram_promote_layer_gate. Off by
  // default (false); every existing config/model that doesn't set this
  // keeps its exact current fetch-width behavior.
  bool prefetch_pilot_k_adaptive = false;
  double prefetch_pilot_k_adaptive_threshold =
      0.50; // roughly the
            // k2/k3 boundary in the real fetch-worthy decay curve above --
            // keeps the ranks that clear "more likely right than wrong".
  int prefetch_pilot_k_adaptive_min_samples =
      30; // per (layer,rank)
          // floor before that rank's own accuracy is trusted -- below
          // this, cold-start falls back to the existing static width
          // (pilot_k_for_layer's ceiling) unchanged, same shape as every
          // other adaptive/gate mechanism in this file.

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

  // prefetch_attn_pilot / prefetch_attn_pilot_k (2026-08-25): real hint
  // wiring for ATTN-PILOT -- predicts a layer from the PREVIOUS layer's
  // post-attention/pre-FFN state, instead of DEPTH-1's post-FFN state or
  // DEPTH-2's naive 2-layer-stale reuse. Real recall (measurement always
  // runs regardless of this flag, same as DEPTH-2's own convention):
  // 75.8% overall for the SAME target layer DEPTH-1 predicts (84.8%), or
  // 75.8% when read as reaching 2 layers past the last known-complete
  // layer the way DEPTH-2 does (72.1%) -- see CLAUDE.md's pilot
  // investigation for the full writeup and the real recall table.
  // Default off. prefetch_attn_pilot_k caps how many of the top-ranked
  // picks get a real hint pushed, same shape as prefetch_pilot_k_depth2.
  bool prefetch_attn_pilot = false;
  int prefetch_attn_pilot_k = 8;

  // attn_pilot_vram_promote_top_k (2026-08-25): SEPARATE, independent of
  // prefetch_attn_pilot/_k above -- of ATTN-PILOT's own already-RAM-
  // resident top-ranked picks (regardless of which mechanism put them in
  // RAM), how many get speculatively promoted straight to VRAM via the
  // SAME proven RAMSafetyManager::promote_ram_to_vram (COPY-only,
  // allow_move=false) mechanism DEPTH-1's own ram_vram_promote_top_k
  // already uses -- NOT a new promotion pathway, and NOT the same cutoff
  // (ram_vram_promote_top_k is settled at 2 for DEPTH-1's own accuracy/
  // call-volume profile; ATTN-PILOT's is different, so that tuned value
  // has no reason to carry over). 0 (default) = off. Watch cache-hit rate
  // alongside decode throughput when raising this -- ram_vram_promote_
  // top_k=3 caused a real over-aggressive-promotion regression for
  // DEPTH-1 once before (VRAM churn -> lower cache-hit), the same risk
  // class applies here.
  int attn_pilot_vram_promote_top_k = 0;

  // attn_pilot_vram_promote_dynamic (2026-08-26): opts ATTN-PILOT's own
  // promotion decision INTO the shared ram_vram_promote_dynamic_layer
  // policy below (min/base/max_k tiers, judged against ATTN-PILOT's OWN
  // per-layer accuracy counters, not DEPTH-1's) instead of the fixed
  // attn_pilot_vram_promote_top_k cutoff above. Only takes effect when
  // ram_vram_promote_dynamic_layer is ALSO true -- this flag alone (with
  // dynamic_layer off) changes nothing, matching every other "off unless
  // the shared mode is also on" flag in this project. Default false --
  // opt in explicitly rather than silently changing ATTN-PILOT's
  // behavior the moment dynamic_layer is turned on for DEPTH-1.
  bool attn_pilot_vram_promote_dynamic = false;

  // O_DIRECT expert-slab reads (Linux only). Opt-in, requires the model
  // to have been repacked with tools/gguf_tool pack (or otherwise have
  // general.alignment >= 4096) -- see GGUFFile::enable_direct_io. A
  // plain/un-repacked GGUF silently falls back to buffered reads with a
  // logged warning, never a hard failure.
  bool direct_io = false;

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
  // OS thread count, not free
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

  // Process-wide host RAM (DDR) budget, in GB. This is a hard admission cap
  // shared by the MoE expert tier, resident/KV/recurrent state, full-graph
  // host arenas, and transient MTP scratch. The RAM manager automatically
  // uses only the remainder after those other allocations are charged.
  // 0 = auto-detect (88% of currently-available system memory).
  double host_ram_gb = 0.0;

  // Optional hard limit for the logical number of MoE expert slots in the
  // host-RAM tier, counted across all MoE layers.  0 = let the byte-aware
  // host_ram_gb calculation decide. The effective value is still reduced
  // when N would exceed the safe DDR budget.  This controls RAM-tier
  // residency/preload, not CUDA host-page registration
  // (ram_pinned_budget_gb below).
  int ram_cache_total_experts = 0;

  // Preload every expert described by the model into the RAM tier, including
  // experts absent from the historical pin files. This is useful when DDR is
  // large enough to hold the complete MoE and avoids a first-request disk
  // miss for an expert with no usage history. Default false preserves the
  // usage-ranked preload behavior.
  bool ram_preload_all_experts = false;
  bool ram_preload_all_experts_explicit = false;

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
  bool ram_pinned_budget_gb_explicit = false;

  // 2026-08-29 (default off, UNVERIFIED ON REAL HARDWARE -- new this
  // session, SECOND implementation after a first attempt was tried,
  // measured, and reverted): extends ram_pinned_budget_gb's own startup
  // Tier 3 RAM preload further down its OWN real, historically-ranked
  // candidate list (v, main.cpp) when there's still real budget room
  // left after pin_ram_fraction's fixed per-layer quota runs out --
  // instead of leaving that remainder to be discovered lazily, and
  // expensively, during a live request.
  //
  // Motivation, real nsys profiling (this session): the SAME
  // cudaHostRegister call averages 0.97ms/call during single-threaded
  // startup preload vs. 5.15ms/call (5.3x slower) during a live
  // prefill, where 7 threads (the decode thread + all prefetch workers)
  // contend for the same driver-level call concurrently -- confirmed
  // via real per-call timestamps to be a sharp step function exactly at
  // the prefill boundary (not a gradual "more memory already pinned"
  // trend), and via real thread-ID data that startup calls are 100%
  // single-threaded while live-prefill calls spread across 7 threads.
  //
  // FIRST implementation (reverted, see this flag's own call site in
  // main.cpp and RAMSafetyManager::eager_preregister_remaining_slots,
  // ram_manager.hpp, both kept unused): eagerly grew+registered GENERIC
  // (identity-agnostic) empty slots for the remaining budget. Real A/B
  // showed this made things WORSE (active_up 0.76-0.81s -> 1.18-1.20s,
  // source-pin coverage 62.7% -> 47.7%) -- it was still greedy
  // pre-allocation against the same finite, shared budget the ranking-
  // informed preload and live demand-driven registration were already
  // using efficiently, just UNINFORMED (no signal which specific
  // expert would ever land in a given slot), so it displaced
  // better-targeted allocations with worse ones for the same spend.
  //
  // This SECOND implementation fixes that by staying informed: same
  // budget, same startup timing, same real ranking data the preload
  // already has -- just doesn't stop using it as early. No effect if
  // false or ram_pinned_budget_gb<=0.
  bool ram_pin_eager_preregister = false;

  // Seed a Tier-3 RAM-preloaded slot's hit_count from its REAL historical
  // count in the pin_file ranking (v[i].count in main.cpp), instead of the
  // hardcoded 0 every RAM-tier slot starts at today. Directly targets a
  // real, measured gap: RAMSafetyManager::eviction_score's frequency term
  // (hit_count * FREQ_BONUS_TICKS) is the ONLY thing that keeps a hot
  // expert from being evicted as easily as a cold one -- but hit_count only
  // ever grows from RE-HITS WITHIN THIS SESSION, so a genuinely
  // popular expert that just got preloaded from .zeromoe_usage looks
  // identical to a marginal one until it happens to get touched twice
  // during THIS request. A real 2026-09-04 run showed the RAM tier turning
  // over almost exactly once per request (5925 evictions / 5702 slots)
  // with 39.5% of what the prefetcher loaded never read before eviction --
  // consistent with "no memory of real popularity" being a real cost, not
  // just a hypothesis.
  //
  // Clamped to promote_hit_threshold-1, never the raw count, on purpose:
  // seeding hit_count >= promote_hit_threshold would make the slot look
  // ALREADY re-hit before it's ever touched this session, firing a
  // reactive VRAM promotion (ram_manager.hpp's own hit_count>=
  // promote_hit_threshold check) on the very first real access instead of
  // the second -- a different, separately-tuned, historically
  // regression-prone lever (see pinned_vram_capacity/ram_vram_promote_*'s
  // own extensive over-promotion tuning history in CLAUDE.md) that this
  // flag is not meant to touch. Clamping keeps the effect scoped to RAM
  // retention only: a real re-hit is still what crosses the threshold,
  // exactly as today, just starting from a "known warm" baseline instead
  // of a blank one. Additive, not a hard pin -- still decays via
  // decay_hit_counts like any other hit_count, still fully evictable if
  // nothing re-hits it. Off by default: byte-identical to pre-2026-09-04
  // behavior (seed=0) when false. UNVERIFIED ON REAL HARDWARE -- run an
  // A/B (same prompt, same pin_file, freeze_usage_stats=true) toggling
  // only this flag and compare the [CACHE-CHURN] Host RAM tier eviction
  // count / Unified RAM waste % / cache hit rate / decode tok/s.
  bool ram_preload_seed_hit_count = false;

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
  // Optional "layer expert" list (one pair per line). When set, the mirror is
  // treated as a SPARSE copy holding only the `down` slices of these experts
  // (see bench/dsv41_pack_cold_mirror.py); every other expert reads from the
  // primary alone. Empty = the mirror is a complete copy.
  std::string mirror_coverage_file;

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
  std::string pin_file = ".zeromoe_usage";
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
  // .zeromoe_usage growing from 12958 to 12988 known experts, +45 on
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

  // Where token_embd.weight lives. The embedding lookup is a one-row read done on the CPU from the host copy
  // (embed_token, MTP, the CPU fallbacks), so a VRAM copy is only dead weight unless the family's graph embeds on the
  // GPU. "auto" = CPU/DDR for every model unless its graph says otherwise (frees ~0.7 GB on a 150k-vocab Q8_0 model);
  // "gpu" forces the device copy; "cpu" forces the host-only placement.
  std::string embedding_device = "auto";

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

  // Experimental llama.cpp-style placement: fill the VRAM expert cache in
  // ascending layer order, starting at layer 0, until the cache is full.
  // The value is an exclusive layer boundary; zero keeps the normal
  // usage-ranked preload strategy.
  int vram_preload_prefix_layers = 0;

  // Optional physical cache-layout policy. When nonzero, the compact expert
  // allocator gives layers [0, value) complete expert-ID regions first,
  // provided they fit in the configured VRAM cache, then distributes any
  // remaining bytes across later layers. This is separate from the preload
  // policy above: it is an opt-in layout change whose benefit is that a
  // complete layer can be consumed by GGML without a per-token repack or
  // PCIe transfer. Zero preserves the balanced compact layout.
  int vram_cache_full_layer_prefix = 0;

  // Only used when pin_strategy=="global_topn". Of the experts already
  // selected for pinning by the ranking above, the top N by real
  // historical count (globally, across every layer -- not per-layer) are
  // deliberately placed on the CoreEngine/dual-role GPU (GpuTopology::
  // core_gpu) instead of wherever GpuTopology::upload_expert's default
  // unrestricted victim scan would otherwise put them (in practice, almost
  // always the much bigger primary vault GPU, since it has vastly more
  // free slots and evict_lru's unrestricted scan finds one there first --
  // confirmed by reading the code: upload_expert never passed a
  // target_gpu_id before this, so 100% of pinned experts landed on the
  // vault GPU regardless of hotness, leaving the CoreEngine GPU's own
  // small dual-role expert-cache slice populated purely by incidental LRU/
  // active-layer overflow, never by deliberately-placed hot experts).
  // Motivation: a smaller/narrower-bandwidth CoreEngine GPU benefits most
  // from LOCAL hits on its own resident experts (no PCIe/DDR round-trip,
  // and it can dispatch the matmul itself instead of sitting idle waiting
  // on a transfer) -- but only if what's resident there is actually likely
  // to be reused, which incidental overflow doesn't guarantee. This
  // doesn't add a new VRAM reservation or change WHICH experts get pinned
  // overall (same pinned_vram_capacity budget, same ranking) -- it only
  // redirects WHERE a subset of the pins that were already going to happen
  // physically land. If the CoreEngine GPU's own slice is too small to fit
  // all N (a real, expected outcome once N exceeds its slot count),
  // upload_expert falls back to the default unrestricted placement for
  // whichever ones don't fit -- this can never reduce total pin coverage
  // below what pin_core_gpu_top_n=0 would have achieved, only redirect
  // some of it. 0 (default) disables this -- unchanged, exactly the old
  // behavior. UNVERIFIED -- see CLAUDE.md for the real A/B this needs
  // before trusting it beyond "doesn't regress correctness".
  int pin_core_gpu_top_n = 0;

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
  bool pin_ram_fraction_explicit = false;
  // pin_global_tiers: place the VRAM-LRU pre-warm and the host-RAM preload
  // from ONE global usage ranking (like pin_strategy=global_topn already does
  // for the pinned tier) instead of a fixed per-layer count. Same total
  // preload budget as the per-layer rule, redistributed to the globally
  // hottest remaining experts, bounded per layer by its VRAM region and RAM
  // cap. The RAM preload then starts AFTER the layer's VRAM entries (disjoint
  // tiers) instead of at rank 0. Only for graph families whose miss path
  // skips the RAM tier for VRAM-resident experts (DeepSeek-V4.1).
  bool pin_global_tiers = false;

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
  // Decode only: promote every real active-layer miss into the main VRAM cache
  // when its layer window closes (COPY-style, same-GPU victim), instead of only
  // those with a repeat hit -- which never happens in decode, leaving the main
  // cache frozen at its startup contents. See GpuTopology::
  // decode_promote_all_misses. Off by default (A/B).
  bool decode_promote_all_misses = false;

  // 2026-08-26: redirect DEPTH-1/ATTN-PILOT's speculative one-layer-ahead
  // RAM->VRAM promotion (ram_vram_promote_top_k/dynamic_layer,
  // attn_pilot_vram_promote_*) into the active-layer pool instead of the
  // main VRAM LRU cache. Motivated directly by a real, measured mechanism:
  // main-cache promotion pays a real eviction cost from the ENTIRE shared
  // 78-layer pool even when the prediction is individually correct (a
  // real run showed rank-3's own recall accuracy at 93.3%, yet cache hit
  // rate still fell 67.5%->65.3% when threshold_4 widened promotion --
  // see CLAUDE.md), because the alternative (do nothing, let real
  // dispatch discover the need) would have reached VRAM via the active
  // layer for free, no main-cache eviction at all. This routes the guess
  // through that same free path preemptively: RAMSafetyManager::
  // promote_ram_to_active_layer calls GpuTopology::upload_expert_active
  // (speculative=true) instead of do_promote_ram_slot_to_vram, and
  // GpuTopology::cleanup_active_layer_area is widened (gated on this same
  // flag, see speculative_active_layer_fill's own comment) to protect a
  // slot tagged one layer ahead of the layer that just finished, giving
  // it the exact same real lifetime a genuine active-layer entry already
  // gets -- survives through its own predicted layer's dispatch window,
  // reclaimed (never promoted on first touch, mirrors decode_async_
  // prewarm's own prewarm_skip_next_promote fix) only if nothing ever
  // consulted it. Default false -- new, unverified on real hardware; see
  // the "Pilot speculative active-layer fill" shutdown report line
  // (perf_timer.hpp) for the real used/wasted rate once tested.
  bool speculative_promote_to_active_layer = false;

  // 2026-08-26: separate, narrower rank cutoff for the active-layer
  // redirect above, independent of ram_vram_promote_top_k/dynamic_layer's
  // own width -- see dense_dispatch.hpp's speculative_active_layer_max_k_
  // value's own comment for the real A/B that motivated this (redirect
  // was mechanically correct but still net-negative on decode throughput
  // at the full dynamic-tier width, since added DMA volume competes for
  // the same bandwidth-bound active-layer transfer pipe decode_async_
  // prewarm also uses). Applied as an ADDITIONAL min() on top of
  // whichever width should_promote/attn_promote_cutoff already decided,
  // never a widening. Default 2 -- unverified on real hardware.
  int speculative_active_layer_max_k = 2;

  // 2026-08-28: multi-GPU active-layer pool routing. Direct response to a
  // real, measured finding: the active-layer pool (which carries the
  // dominant share of DDR->VRAM transfer volume, confirmed via real
  // per-GPU byte counters) is hard-coded to a single GPU (vault_gpus[0]),
  // and that GPU's own PCIe link (measured at ~13.5-14.1 GB/s achieved,
  // consistent with its real PCIe Gen3 x16 ceiling) is the confirmed
  // bottleneck -- while a second GPU can sit almost entirely idle for
  // this traffic (measured: 99.5%/0.5% split). Since expert-FFN dispatch
  // is ALREADY fully GPU-agnostic per-expert (activation/output already
  // round-trip through HOST memory regardless of which GPU holds the
  // weights -- confirmed by reading expert_ffn_gpu_issue/_finish, no P2P
  // copy needed), routing SOME active-layer candidates to a second GPU's
  // OWN small pool adds a genuinely separate, non-contending PCIe link's
  // worth of bandwidth, not just relocated cost.
  //
  // Opt-in and additive: with no GPU's active_layer_weight set (default),
  // this is completely inert and behavior is byte-identical to the
  // legacy single-pool design. Set active_layer_weight/
  // active_layer_capacity_slots on 2+ [[gpus]] entries to enable.
  //
  // Deliberately NOT hardcoded to any specific policy or GPU count --
  // see route_active_layer_gpu's own comment (dense_dispatch.hpp) for why
  // the actual routing DECISION is a small, swappable function keyed off
  // this string, not inlined logic, so a different policy can be added
  // later (e.g. "proportional", a smooth weighted-round-robin across all
  // participating GPUs regardless of priority) without touching any
  // caller. "tiered" (the only policy implemented so far): groups GPUs
  // into capability tiers by weight, reserves the LOWEST-priority tail of
  // each layer transition's candidate list (proportional to the weak
  // tier's combined weight share) for the weak tier specifically, and
  // round-robins the higher-priority majority across the strong tier
  // (GPUs of equal/near-equal weight share that majority evenly, rather
  // than one always getting better ranks than an equally-capable sibling).
  std::string active_layer_routing_policy = "tiered";

  // 2026-08-28, default 0 (off, zero cost/risk when unset): number of
  // background worker threads in ActiveLayerUploadWorkers (active_layer_
  // upload_workers.hpp) -- moves an UNPINNED-source active-layer H2D
  // upload's cudaHostRegister/cudaMemcpyAsync/cudaEventRecord sequence off
  // the decode thread entirely, onto a small pool of dedicated worker
  // threads. Real, isolated microbenchmark (tests/bench_register_
  // parallel.cu, run on this project's own real hardware BEFORE this
  // feature was implemented) confirmed genuine, if imperfect, parallelism
  // across concurrent cudaHostRegister/cudaMemcpyAsync/cudaHostUnregister
  // calls on DIFFERENT host buffers (N=2 -> 1.85x, N=4 -> 3.00x, N=8 ->
  // 5.39x wall-clock speedup vs. sequential) -- this is NOT internally
  // serialized by the CUDA driver on this hardware, so a worker pool is
  // expected to genuinely help, not just relocate the same total cost.
  // An already-pinned source's inline call is left completely untouched
  // (still near-free, ~0.005ms/call measured -- routing it through a
  // worker would only add cross-thread handoff cost for a call that was
  // never the problem); only unpinned-source calls are eligible to route
  // through this pool. Start small (2-4) -- see bench_register_parallel's
  // own diminishing-returns-past-N=4 result before assuming a bigger pool
  // is automatically better; this also competes with every other CPU-
  // bound thread already active during decode (OMP dequant fallback,
  // prefetch workers), so an overly large pool risks a DIFFERENT
  // regression this project has hit before (the expert_fetch_threads/
  // omp_max_threads OS-unresponsiveness incident) rather than the one
  // it's meant to fix.
  // UNVERIFIED ON REAL HARDWARE as of this writing.
  int active_layer_upload_worker_threads = 0;

  // 2026-08-27, default false (preserves prior behavior exactly): see
  // dense_dispatch.hpp's speculative_active_layer_promote_on_confirm_
  // enabled() for the full real-hardware motivation. A speculative
  // active-layer fill's first real touch confirms the pilot's guess
  // (~94-95% correct) but is currently SKIPPED for promotion purposes
  // (same discipline as decode_async_prewarm's own 100%-guaranteed
  // touch, which correctly must be skipped) -- meaning promotion to
  // the persistent main cache is effectively impossible for this path,
  // since the active-layer window is gone before a later token could
  // ever supply the second touch active_layer_promote_hit_threshold
  // needs. Real measured cost: 30-39% more promotion-ATTEMPT volume in
  // V2 (speculative_promote_to_active_layer=true) than V1's own
  // persistent main-cache promotion for the same real underlying need
  // (2371->3079, 2143->2982 across two matched A/B pairs), since a
  // correctly-predicted expert must be re-fetched from scratch every
  // time a later token needs it again instead of staying resident.
  // When true, let a speculative fill's first confirming touch count
  // toward promotion. Known, real risk: this reopens the ORIGINAL
  // problem speculative_promote_to_active_layer was built to avoid
  // (main-cache eviction damage from speculative guesses), just at a
  // much lower wrong-guess rate.
  //
  // TESTED ON REAL HARDWARE (2026-08-27) -- a clean, decisive NEGATIVE
  // result, the predicted risk confirmed exactly. promoted-to-main
  // jumped from ~10 (baseline) to 2385-2542 (~240x), matching the real
  // ~94-95% confirmation rate almost exactly -- the mechanism itself
  // works as designed. But decode dropped 0.88->0.76-0.77 (13-14%
  // regression), and one of two runs threw a REAL error: "[ERROR]
  // [GPU] Failed to async-upload expert to active layer: invalid
  // argument" / "cudaMemcpy H2D failed" -- a genuine stability issue,
  // not just a slowdown, from flooding the main cache with promotion
  // volume at a rate this codebase's eviction/upload path apparently
  // isn't robust to. Default false (leave off) -- do not enable
  // without first hardening whatever surfaced that error, and even
  // then expect a real throughput cost from reopened eviction damage,
  // not a win.
  bool speculative_active_layer_promote_on_confirm = false;

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

  // cpu_dot_ggml_avx2: switches the CPU-fallback dot product for
  // IQ2_XXS/IQ3_XXS (GLM-5.2's actual routed-expert types) from the
  // existing FMA path to a ported upstream llama.cpp/ggml AVX2 kernel --
  // real isolated bench showed ~1.7-1.9x faster per-call, but IQ3_XXS
  // carries a real, not-yet-resolved accuracy caveat (20-row worst-case
  // 1.73% relative error). Default off -- see kquant_dequant.hpp's
  // cpu_dot_ggml_avx2_flag() for the full writeup and CLAUDE.md's
  // "CPU-offload feasibility" entries for the investigation this came
  // from. Exists to A/B the real end-to-end decode effect, not yet a
  // recommended production setting.
  bool cpu_dot_ggml_avx2 = false;

  // Qwen4Exp-only CPU expert fallback: use ggml's IQ1_M x Q8_K kernel
  // (the same family used by llama.cpp) instead of row dequant + FP32 dot.
  // Off by default so other model families and their existing CPU paths are
  // unchanged; enable only after an end-to-end Qwen CPU-layer A/B.
  bool cpu_dot_ggml_iq1m = false;

  // Group a prefill batch's tokens BY EXPERT before dispatch (one call per
  // expert over all its tokens) instead of one S=1 call per (token, expert).
  // This generic switch remains OFF by default because moe_forward_gguf is
  // family-agnostic and mixed expert tensor geometries need model-specific
  // validation. Set false to retain the byte-for-byte-unchanged per-token
  // loop. GLM5NEXT has a separate glm5next_moe_prefill_gpu opt-in so this
  // generic switch can stay off in the GLM profile.
  bool moe_prefill_batched = false;

  // Decode-time (S==1) counterpart to moe_prefill_batched -- see
  // dense_dispatch.hpp's own accessor for the full writeup and the real
  // measured numbers. Routes every topk expert of a decode token straight
  // to the existing CPU dequant-matmul path (dispatch_expert_k's own
  // CPU-fallback branch, execution_layer.cpp), skipping the VRAM-cache
  // find_expert/hold_for_compute/GPU-pool-issue machinery entirely for a
  // genuinely RAM-resident expert.
  //
  // MUST STAY FALSE FOR QWEN3.8-FLASH-NEXT (qwen4exp) -- VERIFIED ON REAL
  // HARDWARE, 2026-09-10: a real, reproducible ~41% DECODE REGRESSION
  // (5.03/4.97 tok/s baseline vs 2.93 tok/s with this on), not a wash and
  // not a win. That model's routed experts (UD-IQ1_M quant) already have
  // a fast, working GPU kernel (480GPU/0CPU dispatch at baseline), so
  // forcing CPU dequant-then-matmul instead is strictly worse for it. See
  // memory: moe-decode-batched-regression.md. Only a plausible win when
  // GPU dispatch was never going to be used anyway (a quant type with NO
  // GPU kernel at all, or an all-`cpu_affinity_layers` config) -- do not
  // enable for qwen4exp UD-IQ1_M, and do not assume this generalizes to
  // any other model/quant without re-measuring first.
  //
  // Real trade-off even where it might help, not free: this also
  // permanently skips expert_ffn_resolve_gpu_loc's own decode-time
  // cold-miss VRAM promotion for every expert while enabled. Never
  // engages for a vram_shadow block (no host bytes without a synchronous
  // reload) -- falls through to the existing per-expert path unchanged
  // for those. Default off.
  bool moe_decode_batched = false;

  // Optional legacy stream path: read prefill expert weights from a whole-
  // shard mmap through a pinned, double-buffered GPU chunk instead of the RAM
  // tier + VRAM LRU. Leave false when MoE weights must come from resident
  // system RAM. See dense_dispatch.hpp's accessor.
  bool moe_prefill_stream = false;
  // Experts staged per chunk. Larger = fewer, bigger H2D transfers and more
  // parallel memcpy per OMP region, at 2 x chunk x slab_bytes of pinned host
  // AND VRAM. 64 x ~1.3 MB x 2 buffers is ~170 MB of each on qwen4exp.
  int moe_prefill_stream_chunk = 32;

  // Run the routed-expert FFN through ggml's MUL_MAT_ID CUDA backend instead
  // of Phase 3b's per-expert-bucket loop. Requires moe_prefill_batched and a
  // build with ZEROMOE_USE_GGML_CUDA; resident RAM slabs are preferred and
  // mmap is used only when the optional stream path is enabled. Falls back
  // per layer whenever any precondition fails. See
  // include/ggml_moe_backend.hpp and dense_dispatch.hpp's own accessor.
  bool moe_ggml_mul_mat_id = false;
  bool moe_ggml_shared_expert = false;
  // Use three direct planar H2D copies per expert instead of the packed
  // H2D+device-unpack path. Qwen4Exp currently benefits from this on PCIe
  // Gen3 because it removes an unpack kernel between small DMA operations;
  // default false preserves other model families' behavior.
  bool moe_ggml_planar_copies = false;

  // Qwen4Exp-only device router. Requires the GGML MoE path and HC GGML
  // handoff. The router consumes HC's device-resident [S,D] result and
  // returns only top-k IDs/weights to the host; false keeps the existing
  // host router for a clean A/B.
  bool moe_ggml_gpu_router = false;

  // Qwen4Exp device pipeline validation switch. Enables stream-ordered reuse
  // of the HC/MoE GGML workspaces after the producer/consumer chain has been
  // queued on the shared backend. Keep false for conservative A/B runs.
  bool ggml_async_graph_reuse = false;

  // GPU for the ggml MoE graph. -1 = auto (most free VRAM), which is usually
  // the WRONG card on this box -- see dense_dispatch.hpp's own accessor.
  int moe_ggml_device = -1;

  // Run Qwen4Exp hyper-connections as ggml GPU graphs (see
  // dense_dispatch.hpp's hc_ggml() for the measurement). Uses moe_ggml_device.
  bool hc_ggml = false;

  // Fuses Qwen4Exp GDN's four shared-input dense_proj calls into one ggml
  // graph. See dense_dispatch.hpp's gdn_proj_ggml() for the measurement.
  bool gdn_proj_ggml = false;
  // Runs the complete Qwen4Exp GDN attention sub-block on one GGML CUDA
  // graph, including recurrence, gated output norm and ssm_out. Requires
  // gdn_proj_ggml and qwen4exp_gdn_conv_gpu; falls back per layer.
  bool gdn_full_ggml = false;
  // Routed-expert matmuls via dequant->BF16 + cuBLAS for S>1. Real speed
  // win, real precision change -- see dense_dispatch.hpp's own comment.
  bool moe_expert_bf16_gemm = false;

  // Route S>1 dense projections through dequantise-to-BF16 + cuBLAS instead
  // of the fused GEMV-shaped kernels. DEFAULT ON as of 2026-09-04: measured
  // prefill 9.49 -> 11.16 tok/s on qwen4exp (dense_gpu 57.10 -> 12.00s, the
  // GEMM itself 45.96 -> 0.97s), output byte-identical. dense_matmul_host is
  // family-agnostic so GLM-5.3 / DeepSeek-V4 share it -- verified on GLM-5.3
  // before the flip. Set false to restore the fused kernels unchanged. See
  // kquant_dequant.hpp's dense_bf16_prefill_gemm_flag() for the numbers.
  bool dense_bf16_prefill_gemm = false;

  // Use a dedicated non-blocking copy stream for host-staged dense decode
  // projections.  The default remains off because older model paths rely on
  // the legacy stream's implicit ordering; Qwen4Exp enables it after the
  // event-ordered path is validated on that model.
  bool dense_async_copies = false;

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

  // predict_gpu1_rank_start/end + predict_gpu1_cap_per_layer: "Idea 1"
  // prediction-based GPU1 dual-role-slice loading (2026-08-16). See
  // Gpu1PredictPromoteConfig's own comment (dense_dispatch.hpp) for the
  // full design -- deliberately separate from ram_vram_promote_top_k
  // above, targets weaker ranks (4+) but ONLY the isolated dual-role
  // slice, never the shared vault pool. Off by default (rank_start=-1).
  int predict_gpu1_rank_start = -1;
  int predict_gpu1_rank_end = -1;
  int predict_gpu1_cap_per_layer = 1;

  // "Direction 1" (2026-08-16, follow-up to the reverted predict_gpu1_*
  // speculative mechanism above): route every Nth REACTIVE (proven-hot,
  // real hit_count threshold crossed -- not a prediction) promotion to
  // GPU1's dual-role slice instead of the general pool. See
  // RAMSafetyManager::reactive_promote_gpu1_every_n's own comment
  // (ram_manager.hpp) for the full rationale. 0 (default) = off.
  int reactive_promote_gpu1_every_n = 0;

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
  double ram_vram_promote_adaptive_threshold =
      0.30; // 30%, provisional --
            // re-tune once more than one real capture exists (single-run data
            // so far: k0=44.2% k1=43.0% k2=30.5% k3=25.3% k4=22.8% k5=16.7%,
            // GLM-... see CLAUDE.md; NOT yet validated across a fresh-vs-repeat
            // prompt spread the way the fetch-worthy metric was).
  int ram_vram_promote_min_samples =
      30; // per-rank floor before that
          // rank's own accuracy is trusted enough to act on.

  // ram_vram_promote_layer_gate (2026-08-26): an ADDITIONAL veto layered
  // on top of whichever mechanism above already selected a promotion
  // candidate (static top_k, adaptive rank-walk, or ATTN-PILOT's own
  // attn_pilot_vram_promote_top_k) -- gated on THAT SPECIFIC LAYER's own
  // live-measured prediction accuracy, not a per-rank figure pooled
  // across all layers the way the adaptive mode above is. Motivated by a
  // real [PILOT-BY-LAYER] finding (see CLAUDE.md): all three pilots show
  // a genuine, large (33-44 percentage-point) accuracy spread BY LAYER,
  // not just by rank -- early layers (roughly 3-20) predict reliably
  // worse than mid/late layers (roughly 40-77) for the SAME rank, on the
  // SAME model. A flat global cutoff can't see this; this can, using the
  // exact same live per-layer counters [PILOT-BY-LAYER] already reports
  // (PerfCounters::n_pilot_recall_*_by_layer / n_pilot_attn_recall_*_by_
  // layer -- see dense_dispatch.hpp's layer_promote_allowed for the
  // read). Each pilot is judged against its OWN per-layer accuracy, not
  // a shared one -- DEPTH-1 and ATTN-PILOT can (and do) disagree on
  // which layers they're each reliable at. Self-calibrating and per-
  // session, like ram_vram_promote_adaptive above -- no static per-model
  // profile to keep in sync with a changing .zeromoe_usage file.
  // Deliberately only ever NARROWS what the top_k/adaptive layer above
  // already allowed, never widens it past that ceiling -- "do less for
  // unreliable layers", not "do more for reliable ones", the safer of
  // the two directions given every prior session's promotion-volume
  // regression was caused by too MUCH speculative promotion, never too
  // little. Off by default; real A/B pending.
  bool ram_vram_promote_layer_gate = false;
  double ram_vram_promote_layer_gate_threshold =
      0.65; // roughly the
            // dividing line between the two real per-layer accuracy clusters
            // observed (low band up to ~69%, high band from ~80%) -- see
            // CLAUDE.md's [PILOT-BY-LAYER] entry for the real numbers this
            // was picked from.
  int ram_vram_promote_layer_gate_min_samples =
      30; // per-layer floor,
          // same role/value as ram_vram_promote_min_samples above but scoped
          // per layer instead of per rank -- below this many live checks for
          // a layer, the gate doesn't veto yet (falls back to "allow",
          // matching pre-gate behavior) since there's no real signal yet.

  // ram_vram_promote_layer_gate_top_fraction (2026-09-13): user request --
  // "promote to the most confident layers only". The flat `threshold`
  // above is NOT that: it's a pass/fail cutoff against an absolute
  // number, so if most tracked layers happen to clear 0.65 it allows
  // nearly all of them (observed: a real A/B run showed ~no difference
  // vs. the gate off, consistent with this). This field switches
  // layer_promote_allowed (dense_dispatch.hpp) into a RANK-based mode
  // instead: tracked layers (>= min_samples live checks) are sorted by
  // their OWN live accuracy and only the top `top_fraction` of them
  // pass, regardless of the absolute number. 0.0 (default) = disabled,
  // falls back to the flat `threshold` behavior above unchanged -- so
  // every existing config (including every non-DeepSeek model) that
  // never sets this field keeps its exact current behavior. Only takes
  // effect when ram_vram_promote_layer_gate is also true. Same per-
  // pilot-type independence as the flat mode (DEPTH-1 and ATTN-PILOT
  // ranked separately, never pooled).
  double ram_vram_promote_layer_gate_top_fraction = 0.0;

  // ram_vram_promote_dynamic_layer (2026-08-26): a THIRD mode, mutually
  // exclusive with static top_k and adaptive above (same "setting more
  // than one disables all, with a warning" validation, extended to
  // three-way) -- and a genuine graduation of ram_vram_promote_layer_gate
  // above, direct response to real user feedback that the plain veto
  // ("do less for bad layers") didn't move throughput either way and
  // should instead scale SMOOTHLY ("do less for bad layers, MORE for
  // good ones") rather than just clamping everything to one flat
  // ceiling. Per LAYER (judged against that pilot's OWN live per-layer
  // accuracy, same counters/read as layer_promote_allowed -- see
  // dense_dispatch.hpp's dynamic_layer_top_k), picks a width from FIVE
  // explicit bands -- 2026-08-26 follow-up, widened from the original
  // 3-tier (min/base/max) design to match real per-layer data directly:
  // a real run's [PILOT-BY-LAYER] table showed 15 of 75 tracked layers
  // (~20%, mostly the 54-66 band plus a few scattered ones) genuinely
  // clearing 90% DEPTH-1 accuracy -- a real, non-trivial population, and
  // meaningfully NARROWER than the ~60-layer ">=80%" population the
  // original max_k=4 test (which regressed) widened indiscriminately.
  // The bet: reserving the widest promotion for ONLY that narrow top
  // band, while the broad 80-90% band gets a smaller, already-validated
  // bump (k=3, confirmed a real win), might capture extra value from the
  // very best layers without reproducing the churn a blanket ">=80%->4"
  // caused.
  //   accuracy >= threshold_4 (0.90) -> k_tier4 (4)
  //   accuracy >= threshold_3 (0.80) -> k_tier3 (3)  [same value/threshold
  //                                                    as the validated
  //                                                    max_k=3 test]
  //   accuracy >= threshold_2 (0.70) -> k_tier2 (2)  [= the settled flat
  //                                                    baseline]
  //   accuracy >= threshold_1 (0.50) -> k_tier1 (1)
  //   accuracy <  threshold_1        -> k_floor (0)
  //   not yet dynamic_min_samples of live data -> cold_start_k (2) --
  //     deliberately a SEPARATE field from k_tier2, even though they
  //     default to the same value, so cold-start behavior doesn't
  //     silently change if k_tier2 is ever re-tuned independently.
  // Off by default; real A/B pending (a prior, coarser veto-only test at
  // ram_vram_promote_layer_gate above was real but couldn't rescue a
  // known top_k=3 regression, root-caused to that gate seeing only the
  // per-LAYER axis while that regression was driven by a per-RANK effect
  // -- see CLAUDE.md for the full writeup; this dynamic mode has the
  // exact same per-layer-only blind spot in principle, so watch the SAME
  // risk when raising any k_tier* past an already-validated value).
  bool ram_vram_promote_dynamic_layer = false;
  double ram_vram_promote_dynamic_threshold_4 = 0.90;
  double ram_vram_promote_dynamic_threshold_3 = 0.80;
  double ram_vram_promote_dynamic_threshold_2 = 0.70;
  double ram_vram_promote_dynamic_threshold_1 = 0.50;
  int ram_vram_promote_dynamic_k_tier4 = 4;
  int ram_vram_promote_dynamic_k_tier3 = 3;
  int ram_vram_promote_dynamic_k_tier2 = 2;
  int ram_vram_promote_dynamic_k_tier1 = 1;
  int ram_vram_promote_dynamic_k_floor = 0;
  int ram_vram_promote_dynamic_cold_start_k = 2;
  int ram_vram_promote_dynamic_min_samples = 30;

  // Redistributes RAMSafetyManager's fixed total RAM-tier budget
  // (uniform slots_per_layer * n_layers) across layers proportional to
  // each layer's real historical need instead of splitting it evenly.
  // Motivated by direct measurement on this project's own accumulated
  // .zeromoe_usage file (75 layers, 672 selections/layer after
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
  bool enable_thinking = true;

  // GLM-5.3-Flash (GLM5NEXT) only. When thinking is enabled, this selects
  // the leading "<|system|>Reasoning Effort: ..." turn. When thinking is
  // disabled, the prompt uses the template's explicit empty <think></think>
  // direct-answer marker and this setting is ignored. Other model families
  // ignore this setting.
  std::string reasoning_effort = "max";

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

  bool load(const std::string &path);
  bool validate();
  static void write_sample(const std::string &path);
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

inline std::string trim(const std::string &s) {
  size_t a = s.find_first_not_of(" \t\r\n");
  size_t b = s.find_last_not_of(" \t\r\n");
  return (a == std::string::npos) ? "" : s.substr(a, b - a + 1);
}

inline std::string strip_comment(const std::string &line) {
  bool in_string = false;
  for (size_t i = 0; i < line.size(); ++i) {
    if (line[i] == '"')
      in_string = !in_string;
    if (line[i] == '#' && !in_string)
      return line.substr(0, i);
  }
  return line;
}

struct Value {
  std::string str;
  double num = 0;
  bool boolean = false;
  std::vector<int> int_array; // only populated when type == IntArray
  enum Type { String, Number, Bool, IntArray } type = String;
};

struct Table {
  std::unordered_map<std::string, Value> values;

  bool has(const std::string &key) const { return values.count(key) > 0; }

  // Every get_*() above silently falls back to `def` on a type mismatch,
  // by design (same "def" spirit noted on get_int_array) -- but that
  // makes a present-but-wrong-type key (e.g. `prefetch_pilot_depth =
  // true` where an integer was expected) INDISTINGUISHABLE from the key
  // being absent entirely, so a real typo silently vanishes with no
  // warning anywhere. This lets a call site tell the two apart and warn
  // specifically on "present but wrong type", without changing any
  // existing get_*()'s own default-on-mismatch behavior (many call sites
  // rely on that leniency and shouldn't need to change).
  std::optional<Value::Type> get_type(const std::string &key) const {
    auto it = values.find(key);
    return (it != values.end()) ? std::optional<Value::Type>(it->second.type)
                                : std::nullopt;
  }

  std::string get_str(const std::string &key,
                      const std::string &def = "") const {
    auto it = values.find(key);
    return (it != values.end()) ? it->second.str : def;
  }
  int get_int(const std::string &key, int def = 0) const {
    auto it = values.find(key);
    return (it != values.end() && it->second.type == Value::Number)
               ? (int)it->second.num
               : def;
  }
  double get_double(const std::string &key, double def = 0.0) const {
    auto it = values.find(key);
    return (it != values.end() && it->second.type == Value::Number)
               ? it->second.num
               : def;
  }
  float get_float(const std::string &key, float def = 0.0f) const {
    return (float)get_double(key, def);
  }
  bool get_bool(const std::string &key, bool def = false) const {
    auto it = values.find(key);
    return (it != values.end() && it->second.type == Value::Bool)
               ? it->second.boolean
               : def;
  }
  // e.g. `cpu_affinity_layers = [4, 5, 7]`. Missing key or wrong type
  // (not a bracketed array) returns an empty vector, same "def" spirit
  // as the other get_*'s -- callers treat empty as "no layers affected".
  std::vector<int> get_int_array(const std::string &key) const {
    auto it = values.find(key);
    return (it != values.end() && it->second.type == Value::IntArray)
               ? it->second.int_array
               : std::vector<int>{};
  }
};

struct Document {
  Table root;                    // Top-level key = value pairs
  std::vector<Table> gpu_tables; // [[gpus]] array-of-tables entries
};

inline Value parse_value(const std::string &raw) {
  Value v;
  std::string s = trim(raw);
  if (s.empty())
    return v;

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
      if (t.empty())
        continue;
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

inline Document parse(const std::string &content) {
  Document doc;
  Table *current = &doc.root;
  bool in_gpu_array = false;

  std::istringstream stream(content);
  std::string line;
  while (std::getline(stream, line)) {
    line = trim(strip_comment(line));
    if (line.empty())
      continue;

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
      current = &doc.root; // flatten non-gpu sections into root
      continue;
    }

    // key = value
    size_t eq = line.find('=');
    if (eq == std::string::npos)
      continue;
    std::string key = trim(line.substr(0, eq));
    std::string val_str = trim(line.substr(eq + 1));
    current->values[key] = parse_value(val_str);
  }
  return doc;
}

inline Document parse_file(const std::string &path) {
  std::ifstream f(path);
  if (!f.is_open()) {
    std::cerr << "[CONFIG] Cannot open: " << path << "\n";
    return {};
  }
  std::string content((std::istreambuf_iterator<char>(f)),
                      std::istreambuf_iterator<char>());
  return parse(content);
}

} // namespace toml

// ============================================================
// EngineConfig implementation
// ============================================================
inline bool EngineConfig::load(const std::string &path) {
  auto doc = toml::parse_file(path);
  std::string settings_error;
  if (!zeromoe::graphsettings::collect_config(doc, graph_settings, settings_error)) {
    std::cerr << "[CONFIG] " << settings_error;
    return false;
  }

  model_path = doc.root.get_str("model_path");
  model_format = doc.root.get_str("model_format", model_format);
  exl3_backend = doc.root.get_str("exl3_backend", exl3_backend);
  exl3_ggml_bridge_model_path =
      doc.root.get_str("exl3_ggml_bridge_model_path",
                      exl3_ggml_bridge_model_path);
  exl3_ggml_bridge_layer =
      doc.root.get_int("exl3_ggml_bridge_layer", exl3_ggml_bridge_layer);
  exl3_ggml_bridge_layers =
      doc.root.get_int_array("exl3_ggml_bridge_layers");
  exl3_ggml_graph_capture =
      doc.root.get_bool("exl3_ggml_graph_capture", exl3_ggml_graph_capture);
  exl3_ggml_routed_graph_capture = doc.root.get_bool(
      "exl3_ggml_routed_graph_capture", exl3_ggml_routed_graph_capture);
  exllamav3_python = doc.root.get_str("exllamav3_python", exllamav3_python);
  exllamav3_runner_path = doc.root.get_str("exllamav3_runner_path", exllamav3_runner_path);
  exllamav3_cuda_schedule = doc.root.get_str("exllamav3_cuda_schedule", exllamav3_cuda_schedule);
  model_load_only = doc.root.get_bool("model_load_only", model_load_only);
  metadata_model_path = doc.root.get_str("metadata_model_path", metadata_model_path);
  prompt = doc.root.get_str("prompt", prompt);
  exl3_gpu = doc.root.get_bool("exl3_gpu", exl3_gpu);
  exl3_gpu_hc = doc.root.get_bool("exl3_gpu_hc", exl3_gpu_hc);
  exl3_gpu_device = doc.root.get_int("exl3_gpu_device", exl3_gpu_device);
  exl3_gpu_cache_gb = doc.root.get_double("exl3_gpu_cache_gb", exl3_gpu_cache_gb);
  exl3_gpu_expert_layer_begin =
      doc.root.get_int("exl3_gpu_expert_layer_begin", exl3_gpu_expert_layer_begin);
  exl3_gpu_expert_split_layer =
      doc.root.get_int("exl3_gpu_expert_split_layer", exl3_gpu_expert_split_layer);
  exl3_gpu_split_device = doc.root.get_int("exl3_gpu_split_device", exl3_gpu_split_device);
  exl3_gpu_split_cache_gb =
      doc.root.get_double("exl3_gpu_split_cache_gb", exl3_gpu_split_cache_gb);
  exl3_cpu_expert_cache_gb =
      doc.root.get_double("exl3_cpu_expert_cache_gb", exl3_cpu_expert_cache_gb);
  exl3_cpu_expert_int8_vnni =
      doc.root.get_bool("exl3_cpu_expert_int8_vnni", exl3_cpu_expert_int8_vnni);
  dense_model_path = doc.root.get_str("dense_model_path");
  mtp_model_path = doc.root.get_str("mtp_model_path");
  context_size = doc.root.get_int("context_size", context_size);
  max_new_tokens = doc.root.get_int("max_new_tokens", max_new_tokens);
  temperature = doc.root.get_float("temperature", temperature);
  top_p = doc.root.get_float("top_p", top_p);
  drop_pages = doc.root.get_bool("drop_pages", drop_pages);
  cpu_only = doc.root.get_bool("cpu_only", cpu_only);
  cpu_affinity_layers = doc.root.get_int_array("cpu_affinity_layers");
  qwen4exp_graph_expert_vram_budget_gb = doc.root.get_double(
      "qwen4exp_graph_expert_vram_budget_gb",
      qwen4exp_graph_expert_vram_budget_gb);
  qwen4exp_option_b_layer = (int)doc.root.get_int(
      "qwen4exp_option_b_layer", qwen4exp_option_b_layer);
  qwen4exp_option_b_top_n = (int)doc.root.get_int(
      "qwen4exp_option_b_top_n", qwen4exp_option_b_top_n);
  qwen4exp_option_b_min_coverage = doc.root.get_float(
      "qwen4exp_option_b_min_coverage", qwen4exp_option_b_min_coverage);
  qwen4exp_option_b_right_size_top_n = doc.root.get_bool(
      "qwen4exp_option_b_right_size_top_n",
      qwen4exp_option_b_right_size_top_n);
  qwen4exp_option_b_global_top_k = doc.root.get_int(
      "qwen4exp_option_b_global_top_k", qwen4exp_option_b_global_top_k);
  qwen4exp_option_b_global_floor_per_layer = doc.root.get_int(
      "qwen4exp_option_b_global_floor_per_layer",
      qwen4exp_option_b_global_floor_per_layer);
  if (qwen4exp_option_b_global_top_k < 0)
    qwen4exp_option_b_global_top_k = 0;
  if (qwen4exp_option_b_global_floor_per_layer < 0)
    qwen4exp_option_b_global_floor_per_layer = 0;
  qwen4exp_option_b_selected_layers = doc.root.get_int_array(
      "qwen4exp_option_b_selected_layers");
  qwen4exp_option_b_max_layers = (int)doc.root.get_int(
      "qwen4exp_option_b_max_layers", qwen4exp_option_b_max_layers);
  qwen4exp_option_b_fill_vram = doc.root.get_bool(
      "qwen4exp_option_b_fill_vram", qwen4exp_option_b_fill_vram);
  qwen4exp_option_b_disable_legacy_cache = doc.root.get_bool(
      "qwen4exp_option_b_disable_legacy_cache",
      qwen4exp_option_b_disable_legacy_cache);
  qwen4exp_option_b_concentrated_only = doc.root.get_bool(
      "qwen4exp_option_b_concentrated_only",
      qwen4exp_option_b_concentrated_only);
  qwen4exp_option_b_stage_expert = (int)doc.root.get_int(
      "qwen4exp_option_b_stage_expert", qwen4exp_option_b_stage_expert);
  qwen4exp_option_b_stage_count = (int)doc.root.get_int(
      "qwen4exp_option_b_stage_count", qwen4exp_option_b_stage_count);
  qwen4exp_full_graph =
      doc.root.get_bool("qwen4exp_full_graph", qwen4exp_full_graph);
  qwen4exp_full_graph_decode = doc.root.get_bool(
      "qwen4exp_full_graph_decode", qwen4exp_full_graph_decode);
  qwen4exp_prefill_state_handoff = doc.root.get_bool(
      "qwen4exp_prefill_state_handoff", qwen4exp_prefill_state_handoff);
  if (qwen4exp_option_b_layer < -1) {
    LOG_WARN("[CONFIG] qwen4exp_option_b_layer=%d is invalid -- disabling\n",
             qwen4exp_option_b_layer);
    qwen4exp_option_b_layer = -1;
  }
  if (qwen4exp_option_b_top_n < 0) {
    LOG_WARN("[CONFIG] qwen4exp_option_b_top_n=%d is invalid -- disabling\n",
             qwen4exp_option_b_top_n);
    qwen4exp_option_b_top_n = 0;
  }
  if (qwen4exp_option_b_max_layers < 0) {
    LOG_WARN("[CONFIG] qwen4exp_option_b_max_layers=%d is invalid -- "
             "disabling the layer cap\n",
             qwen4exp_option_b_max_layers);
    qwen4exp_option_b_max_layers = 0;
  }
  if (qwen4exp_option_b_min_coverage < 0.0f ||
      qwen4exp_option_b_min_coverage > 1.0f) {
    LOG_WARN("[CONFIG] qwen4exp_option_b_min_coverage=%.3f is invalid -- "
             "using 0.80\n",
             qwen4exp_option_b_min_coverage);
    qwen4exp_option_b_min_coverage = 0.80f;
  }
  if (qwen4exp_option_b_stage_expert < -1) {
    LOG_WARN("[CONFIG] qwen4exp_option_b_stage_expert=%d is invalid -- "
             "disabling single-slot staging proof\n",
             qwen4exp_option_b_stage_expert);
    qwen4exp_option_b_stage_expert = -1;
  }
  if (qwen4exp_option_b_stage_count < 1) {
    LOG_WARN("[CONFIG] qwen4exp_option_b_stage_count=%d is invalid -- "
             "using one staged expert\n",
             qwen4exp_option_b_stage_count);
    qwen4exp_option_b_stage_count = 1;
  }
  cpu_offload_max_experts =
      doc.root.get_int("cpu_offload_max_experts", cpu_offload_max_experts);
  if (cpu_offload_max_experts < 0 || cpu_offload_max_experts > 8) {
    LOG_WARN(
        "[CONFIG] cpu_offload_max_experts must be 0-8 (got %d) -- clamping\n",
        cpu_offload_max_experts);
    cpu_offload_max_experts = std::max(0, std::min(8, cpu_offload_max_experts));
  }
  prefill_dispatch_lookahead_tokens = doc.root.get_int(
      "prefill_dispatch_lookahead_tokens", prefill_dispatch_lookahead_tokens);
  if (prefill_dispatch_lookahead_tokens < 0) {
    LOG_WARN("[CONFIG] prefill_dispatch_lookahead_tokens must be >= 0 (got %d) "
             "-- clamping to 0\n",
             prefill_dispatch_lookahead_tokens);
    prefill_dispatch_lookahead_tokens = 0;
  }
  prefill_hint_order_by_earliest_token =
      doc.root.get_bool("prefill_hint_order_by_earliest_token",
                        prefill_hint_order_by_earliest_token);
  pilot_widen_all_tokens =
      doc.root.get_bool("pilot_widen_all_tokens", pilot_widen_all_tokens);
  prefetch_widen_all_tokens =
      doc.root.get_bool("prefetch_widen_all_tokens", prefetch_widen_all_tokens);
  prefill_window_cap =
      doc.root.get_bool("prefill_window_cap", prefill_window_cap);
  prefill_batch_tokens =
      doc.root.get_int("prefill_batch_tokens", prefill_batch_tokens);
  prefill_window_extra_gb =
      doc.root.get_double("prefill_window_extra_gb", prefill_window_extra_gb);
  prefill_retention_weight =
      doc.root.get_int("prefill_retention_weight", prefill_retention_weight);
  prefill_retention_top_n = doc.root.get_int("prefill_retention_top_n", prefill_retention_top_n);
  prefill_retention_min_demand =
      doc.root.get_int("prefill_retention_min_demand", prefill_retention_min_demand);
  prefill_retention_flat_cut =
      doc.root.get_double("prefill_retention_flat_cut", prefill_retention_flat_cut);
  prefill_retention_peaky_cut =
      doc.root.get_double("prefill_retention_peaky_cut", prefill_retention_peaky_cut);
  prefill_retention_decay =
      doc.root.get_bool("prefill_retention_decay", prefill_retention_decay);
  ram_grace_period_tokens =
      doc.root.get_int("ram_grace_period_tokens", ram_grace_period_tokens);
  ram_grace_period_bonus =
      doc.root.get_int("ram_grace_period_bonus", ram_grace_period_bonus);
  if (ram_grace_period_tokens < 0) {
    LOG_WARN("[CONFIG] ram_grace_period_tokens must be >= 0 (got %d) -- "
             "clamping to 0 (feature off)\n", ram_grace_period_tokens);
    ram_grace_period_tokens = 0;
  }
  if (ram_grace_period_bonus < 0) {
    LOG_WARN("[CONFIG] ram_grace_period_bonus must be >= 0 (got %d) -- "
             "clamping to 0\n", ram_grace_period_bonus);
    ram_grace_period_bonus = 0;
  }
  if (prefill_retention_weight < 0) {
    LOG_WARN("[CONFIG] prefill_retention_weight must be >= 0 (got %d) -- clamping to 0 "
             "(feature off)\n", prefill_retention_weight);
    prefill_retention_weight = 0;
  }
  if (prefill_window_extra_gb < 0.0) {
    LOG_WARN("[CONFIG] prefill_window_extra_gb must be >= 0 (got %.2f) -- "
             "clamping to 0 (window\n"
             "         effectively disabled: no extra slots can be granted)\n",
             prefill_window_extra_gb);
    prefill_window_extra_gb = 0.0;
  }
  if (prefetch_widen_all_tokens && !pilot_widen_all_tokens) {
    LOG_WARN("[CONFIG] prefetch_widen_all_tokens=true but "
             "pilot_widen_all_tokens=false -- the "
             "widened prediction is never computed, so there is nothing to "
             "push as a real hint. "
             "This is a silent no-op, not an error -- set "
             "pilot_widen_all_tokens=true too if real "
             "hint-push is intended.\n");
  }
  prefetch_pilot_k = doc.root.get_int("prefetch_pilot_k", prefetch_pilot_k);
  prefetch_pilot_k_per_layer =
      doc.root.get_int_array("prefetch_pilot_k_per_layer");
  prefetch_pilot_k_adaptive = doc.root.get_bool("prefetch_pilot_k_adaptive",
                                                 prefetch_pilot_k_adaptive);
  prefetch_pilot_k_adaptive_threshold =
      doc.root.get_double("prefetch_pilot_k_adaptive_threshold",
                          prefetch_pilot_k_adaptive_threshold);
  prefetch_pilot_k_adaptive_min_samples =
      doc.root.get_int("prefetch_pilot_k_adaptive_min_samples",
                       prefetch_pilot_k_adaptive_min_samples);
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
  if (auto t = doc.root.get_type("prefetch_pilot_depth");
      t && *t != toml::Value::Number) {
    LOG_WARN("[CONFIG] prefetch_pilot_depth is set but is not a number (e.g. a "
             "boolean like "
             "`true` was found) -- this key takes an integer, 1 or 2 (e.g. "
             "prefetch_pilot_depth = 2), not a toggle -- ignoring, falling "
             "back to 1\n");
  }
  prefetch_pilot_depth =
      doc.root.get_int("prefetch_pilot_depth", prefetch_pilot_depth);
  if (prefetch_pilot_depth != 1 && prefetch_pilot_depth != 2) {
    LOG_WARN("[CONFIG] prefetch_pilot_depth=%d is invalid (must be 1 or 2) -- "
             "falling back to 1\n",
             prefetch_pilot_depth);
    prefetch_pilot_depth = 1;
  }
  prefetch_pilot_k_depth2 =
      doc.root.get_int("prefetch_pilot_k_depth2", prefetch_pilot_k_depth2);
  prefetch_attn_pilot =
      doc.root.get_bool("prefetch_attn_pilot", prefetch_attn_pilot);
  prefetch_attn_pilot_k =
      doc.root.get_int("prefetch_attn_pilot_k", prefetch_attn_pilot_k);
  attn_pilot_vram_promote_top_k = doc.root.get_int(
      "attn_pilot_vram_promote_top_k", attn_pilot_vram_promote_top_k);
  attn_pilot_vram_promote_dynamic = doc.root.get_bool(
      "attn_pilot_vram_promote_dynamic", attn_pilot_vram_promote_dynamic);
  direct_io = doc.root.get_bool("direct_io", direct_io);
  prefill_warm_via_prefetcher = doc.root.get_bool("prefill_warm_via_prefetcher",
                                                  prefill_warm_via_prefetcher);
  prefill_opportunistic_promote = doc.root.get_bool(
      "prefill_opportunistic_promote", prefill_opportunistic_promote);
  decode_async_prewarm =
      doc.root.get_bool("decode_async_prewarm", decode_async_prewarm);
  decode_async_prewarm_max_batch = doc.root.get_int(
      "decode_async_prewarm_max_batch", decode_async_prewarm_max_batch);
  decode_async_prewarm_nonblocking = doc.root.get_bool(
      "decode_async_prewarm_nonblocking", decode_async_prewarm_nonblocking);
  if (decode_async_prewarm_max_batch < 1) {
    LOG_WARN("[CONFIG] decode_async_prewarm_max_batch must be >= 1 (got %d) -- "
             "clamping to 1\n",
             decode_async_prewarm_max_batch);
    decode_async_prewarm_max_batch = 1;
  }
  attn_qabsorb_gpu = doc.root.get_bool("attn_qabsorb_gpu", attn_qabsorb_gpu);
  attn_qabsorb_gpu_max_vram_gb = doc.root.get_double(
      "attn_qabsorb_gpu_max_vram_gb", attn_qabsorb_gpu_max_vram_gb);
  if (attn_qabsorb_gpu_max_vram_gb < 0.0) {
    LOG_WARN("[CONFIG] attn_qabsorb_gpu_max_vram_gb must be >= 0 (got %.2f) -- "
             "clamping to 0\n",
             attn_qabsorb_gpu_max_vram_gb);
    attn_qabsorb_gpu_max_vram_gb = 0.0;
  }
  attn_score_gpu = doc.root.get_bool("attn_score_gpu", attn_score_gpu);
  glm5next_kda_gpu = doc.root.get_bool("glm5next_kda_gpu", glm5next_kda_gpu);
  glm5next_router_gpu = doc.root.get_bool("glm5next_router_gpu", glm5next_router_gpu);
  glm5next_moe_prefill_gpu = doc.root.get_bool(
      "glm5next_moe_prefill_gpu", glm5next_moe_prefill_gpu);
  glm5next_dense_prefill_fusion = doc.root.get_bool(
      "glm5next_dense_prefill_fusion", glm5next_dense_prefill_fusion);
  qwen4exp_attn_gpu = doc.root.get_bool("qwen4exp_attn_gpu", qwen4exp_attn_gpu);
  qwen4exp_gdn_gpu = doc.root.get_bool("qwen4exp_gdn_gpu", qwen4exp_gdn_gpu);
  qwen4exp_gdn_conv_gpu = doc.root.get_bool("qwen4exp_gdn_conv_gpu", qwen4exp_gdn_conv_gpu);
  qwen4exp_router_pilot =
      doc.root.get_bool("qwen4exp_router_pilot", qwen4exp_router_pilot);
  qwen4exp_router_pilot_k = (int)doc.root.get_int(
      "qwen4exp_router_pilot_k", qwen4exp_router_pilot_k);
  if (qwen4exp_router_pilot_k < 1) {
    LOG_WARN("[CONFIG] qwen4exp_router_pilot_k=%d is invalid -- clamping to 1\n",
             qwen4exp_router_pilot_k);
    qwen4exp_router_pilot_k = 1;
  } else if (qwen4exp_router_pilot_k > 16) {
    LOG_WARN("[CONFIG] qwen4exp_router_pilot_k=%d exceeds the bounded Qwen "
             "pilot capacity -- clamping to 16\n",
             qwen4exp_router_pilot_k);
    qwen4exp_router_pilot_k = 16;
  }
  qwen4exp_router_pilot_prefetch = doc.root.get_bool(
      "qwen4exp_router_pilot_prefetch", qwen4exp_router_pilot_prefetch);
  qwen4exp_router_pilot_prefetch_k = (int)doc.root.get_int(
      "qwen4exp_router_pilot_prefetch_k",
      qwen4exp_router_pilot_prefetch_k);
  if (qwen4exp_router_pilot_prefetch_k < 0) {
    LOG_WARN("[CONFIG] qwen4exp_router_pilot_prefetch_k=%d is invalid -- "
             "clamping to 0 (use prediction K)\n",
             qwen4exp_router_pilot_prefetch_k);
    qwen4exp_router_pilot_prefetch_k = 0;
  } else if (qwen4exp_router_pilot_prefetch_k > 16) {
    LOG_WARN("[CONFIG] qwen4exp_router_pilot_prefetch_k=%d exceeds the bounded "
             "Qwen pilot capacity -- clamping to 16\n",
             qwen4exp_router_pilot_prefetch_k);
    qwen4exp_router_pilot_prefetch_k = 16;
  }
  qwen4exp_router_pilot_prefetch_exclude_layers =
      doc.root.get_int_array("qwen4exp_router_pilot_prefetch_exclude_layers");
  qwen4exp_router_pilot_predict_layers =
      doc.root.get_int_array("qwen4exp_router_pilot_predict_layers");
  qwen4exp_router_pilot_skip_fully_cached = doc.root.get_bool(
      "qwen4exp_router_pilot_skip_fully_cached",
      qwen4exp_router_pilot_skip_fully_cached);
  qwen4exp_router_pilot_skip_prefetch_excluded_layers = doc.root.get_bool(
      "qwen4exp_router_pilot_skip_prefetch_excluded_layers",
      qwen4exp_router_pilot_skip_prefetch_excluded_layers);
  qwen4exp_decode_ggml_moe =
      doc.root.get_bool("qwen4exp_decode_ggml_moe", qwen4exp_decode_ggml_moe);
  qwen4exp_decode_scheduler = doc.root.get_bool(
      "qwen4exp_decode_scheduler", qwen4exp_decode_scheduler);
  qwen4exp_decode_device_hc = doc.root.get_bool(
      "qwen4exp_decode_device_hc", qwen4exp_decode_device_hc);
  qwen4exp_decode_batch_stage = doc.root.get_bool(
      "qwen4exp_decode_batch_stage", qwen4exp_decode_batch_stage);
  qwen4exp_decode_custom_hc_q8 = doc.root.get_bool(
      "qwen4exp_decode_custom_hc_q8", qwen4exp_decode_custom_hc_q8);
  qwen4exp_hc_async_readback = doc.root.get_bool(
      "qwen4exp_hc_async_readback", qwen4exp_hc_async_readback);
  qwen4exp_decode_gpu_router = doc.root.get_bool(
      "qwen4exp_decode_gpu_router", qwen4exp_decode_gpu_router);
  qwen4exp_cpu_ggml_moe =
      doc.root.get_bool("qwen4exp_cpu_ggml_moe", qwen4exp_cpu_ggml_moe);
  qwen4exp_cpu_strided_grouped = doc.root.get_bool(
      "qwen4exp_cpu_strided_grouped", qwen4exp_cpu_strided_grouped);
  qwen4exp_cpu_fused_grouped = doc.root.get_bool(
      "qwen4exp_cpu_fused_grouped", qwen4exp_cpu_fused_grouped);
  qwen4exp_cpu_gpu_overlap = doc.root.get_bool(
      "qwen4exp_cpu_gpu_overlap", qwen4exp_cpu_gpu_overlap);
  qwen4exp_cpu_gpu_overlap_max_layers = (int)doc.root.get_int(
      "qwen4exp_cpu_gpu_overlap_max_layers",
      qwen4exp_cpu_gpu_overlap_max_layers);
  if (qwen4exp_cpu_gpu_overlap_max_layers < 0) {
    LOG_WARN("[CONFIG] qwen4exp_cpu_gpu_overlap_max_layers=%d is invalid -- "
             "clamping to 0 (disabled)\n",
             qwen4exp_cpu_gpu_overlap_max_layers);
    qwen4exp_cpu_gpu_overlap_max_layers = 0;
  }
  qwen4exp_cpu_shared_device = doc.root.get_bool(
      "qwen4exp_cpu_shared_device", qwen4exp_cpu_shared_device);
  qwen4exp_cpu_offload =
      doc.root.get_bool("qwen4exp_cpu_offload", qwen4exp_cpu_offload);
  qwen4exp_cpu_offload_layers =
      doc.root.get_int_array("qwen4exp_cpu_offload_layers");
  qwen4exp_cpu_offload_force_cold_count = (int)doc.root.get_int(
      "qwen4exp_cpu_offload_force_cold_count",
      qwen4exp_cpu_offload_force_cold_count);
  if (qwen4exp_cpu_offload_force_cold_count < 0) {
    LOG_WARN("[CONFIG] qwen4exp_cpu_offload_force_cold_count=%d is invalid; "
             "clamping to 0\n",
             qwen4exp_cpu_offload_force_cold_count);
    qwen4exp_cpu_offload_force_cold_count = 0;
  }
  if (qwen4exp_cpu_offload_force_cold_count > 16)
    qwen4exp_cpu_offload_force_cold_count = 16;
  qwen4exp_cpu_offload_force_expert = (int)doc.root.get_int(
      "qwen4exp_cpu_offload_force_expert",
      qwen4exp_cpu_offload_force_expert);
  qwen4exp_cpu_offload_event_fence = doc.root.get_bool(
      "qwen4exp_cpu_offload_event_fence", qwen4exp_cpu_offload_event_fence);
  qwen4exp_gpu_stage_enabled = doc.root.get_bool(
      "qwen4exp_gpu_stage_enabled", qwen4exp_gpu_stage_enabled);
  qwen4exp_gpu_stage_layers =
      doc.root.get_int_array("qwen4exp_gpu_stage_layers");
  qwen4exp_track_router_usage = doc.root.get_bool(
      "qwen4exp_track_router_usage", qwen4exp_track_router_usage);
  qwen4exp_cpu_offload_ring_slots = (int)doc.root.get_int(
      "qwen4exp_cpu_offload_ring_slots", qwen4exp_cpu_offload_ring_slots);
  if (qwen4exp_cpu_offload_ring_slots < 1)
    qwen4exp_cpu_offload_ring_slots = 1;
  if (qwen4exp_cpu_offload_ring_slots > 4)
    qwen4exp_cpu_offload_ring_slots = 4;
  prompt_cache = doc.root.get_bool("prompt_cache", prompt_cache);
  mtp_enabled = doc.root.get_bool("mtp_enabled", mtp_enabled);
  mtp_graph = doc.root.get_bool("mtp_graph", mtp_graph);
  mtp_draft_tokens = (int)doc.root.get_int("mtp_draft_tokens", mtp_draft_tokens);
  if (mtp_draft_tokens < 1) mtp_draft_tokens = 1;
  if (mtp_draft_tokens > 8) {
    LOG_WARN("[CONFIG] mtp_draft_tokens=%d is too large for the bounded GLM "
             "bridge -- clamping to 8\n", mtp_draft_tokens);
    mtp_draft_tokens = 8;
  }
  moe_gpu_accum = doc.root.get_bool("moe_gpu_accum", moe_gpu_accum);
  dense_bf16_kquant = doc.root.get_bool("dense_bf16_kquant", dense_bf16_kquant);
  expert_gpu_activation = doc.root.get_bool("expert_gpu_activation", expert_gpu_activation);
  vram_min_free_mb = (long)doc.root.get_int("vram_min_free_mb", (int64_t)vram_min_free_mb);
  mem_guard = doc.root.get_bool("mem_guard", mem_guard);
  mem_guard_warn_mb = (long)doc.root.get_int("mem_guard_warn_mb", (int64_t)mem_guard_warn_mb);
  mem_guard_stop_mb = (long)doc.root.get_int("mem_guard_stop_mb", (int64_t)mem_guard_stop_mb);
  mem_guard_poll_ms = (int)doc.root.get_int("mem_guard_poll_ms", mem_guard_poll_ms);
  attn_score_gpu_max_vram_gb = doc.root.get_double("attn_score_gpu_max_vram_gb",
                                                   attn_score_gpu_max_vram_gb);
  if (attn_score_gpu_max_vram_gb < 0.0) {
    LOG_WARN("[CONFIG] attn_score_gpu_max_vram_gb must be >= 0 (got %.2f) -- "
             "clamping to 0\n",
             attn_score_gpu_max_vram_gb);
    attn_score_gpu_max_vram_gb = 0.0;
  }
  attn_score_gpu_max_context = doc.root.get_int("attn_score_gpu_max_context",
                                                attn_score_gpu_max_context);
  if (attn_score_gpu_max_context < 0) {
    LOG_WARN("[CONFIG] attn_score_gpu_max_context must be >= 0 (got %d) -- "
             "clamping to 0 (no cap)\n",
             attn_score_gpu_max_context);
    attn_score_gpu_max_context = 0;
  }
  dsv4_score_gpu = doc.root.get_bool("dsv4_score_gpu", dsv4_score_gpu);
  dsv4_score_gpu_max_vram_gb = doc.root.get_double("dsv4_score_gpu_max_vram_gb",
                                                    dsv4_score_gpu_max_vram_gb);
  if (dsv4_score_gpu_max_vram_gb < 0.0) {
    LOG_WARN("[CONFIG] dsv4_score_gpu_max_vram_gb must be >= 0 (got %.2f) -- "
             "clamping to 0\n",
             dsv4_score_gpu_max_vram_gb);
    dsv4_score_gpu_max_vram_gb = 0.0;
  }
  unpacked_expert_format =
      doc.root.get_bool("unpacked_expert_format", unpacked_expert_format);
  io_uring = doc.root.get_bool("io_uring", io_uring);
  io_uring_workers = doc.root.get_int("io_uring_workers", io_uring_workers);
  disk_batch_padding =
      doc.root.get_bool("disk_batch_padding", disk_batch_padding);
  disk_batch_padding_target =
      doc.root.get_int("disk_batch_padding_target", disk_batch_padding_target);
  disk_batch_padding_evict_dup = doc.root.get_bool(
      "disk_batch_padding_evict_dup", disk_batch_padding_evict_dup);
  prefetch_worker_threads =
      doc.root.get_int("prefetch_worker_threads", prefetch_worker_threads);
  host_ram_gb = doc.root.get_double("host_ram_gb", host_ram_gb);
  ram_cache_total_experts = doc.root.get_int(
      "ram_cache_total_experts", ram_cache_total_experts);
  ram_preload_all_experts = doc.root.get_bool(
      "ram_preload_all_experts", ram_preload_all_experts);
  ram_preload_all_experts_explicit =
      doc.root.has("ram_preload_all_experts");
  mirror_model_path = doc.root.get_str("mirror_model_path", mirror_model_path);
  mirror_coverage_file =
      doc.root.get_str("mirror_coverage_file", mirror_coverage_file);
  ram_pinned_budget_gb =
      doc.root.get_double("ram_pinned_budget_gb", ram_pinned_budget_gb);
  ram_pinned_budget_gb_explicit = doc.root.has("ram_pinned_budget_gb");
  ram_pin_eager_preregister =
      doc.root.get_bool("ram_pin_eager_preregister", ram_pin_eager_preregister);
  ram_preload_seed_hit_count = doc.root.get_bool(
      "ram_preload_seed_hit_count", ram_preload_seed_hit_count);
  verbose_moe_log = doc.root.get_bool("verbose_moe_log", verbose_moe_log);
  promote_hit_threshold =
      doc.root.get_int("promote_hit_threshold", promote_hit_threshold);
  pin_threshold = doc.root.get_int("pin_threshold", pin_threshold);
  pin_file = doc.root.get_str("pin_file", pin_file);
  prefill_pin_file = doc.root.get_str("prefill_pin_file", prefill_pin_file);
  freeze_usage_stats =
      doc.root.get_bool("freeze_usage_stats", freeze_usage_stats);
  embedding_device = doc.root.get_str("embedding_device", embedding_device);
  pin_topk_per_layer =
      doc.root.get_int("pin_topk_per_layer", pin_topk_per_layer);
  pin_strategy = doc.root.get_str("pin_strategy", pin_strategy);
  pin_global_floor_per_layer = doc.root.get_int("pin_global_floor_per_layer",
                                                pin_global_floor_per_layer);
  vram_preload_prefix_layers = doc.root.get_int(
      "vram_preload_prefix_layers", vram_preload_prefix_layers);
  vram_cache_full_layer_prefix = doc.root.get_int(
      "vram_cache_full_layer_prefix", vram_cache_full_layer_prefix);
  pin_core_gpu_top_n = doc.root.get_int("pin_core_gpu_top_n", pin_core_gpu_top_n);
  pin_vram_lru_fraction =
      doc.root.get_double("pin_vram_lru_fraction", pin_vram_lru_fraction);
  pin_ram_fraction = doc.root.get_double("pin_ram_fraction", pin_ram_fraction);
  pin_ram_fraction_explicit = doc.root.has("pin_ram_fraction");
  pin_global_tiers = doc.root.get_bool("pin_global_tiers", pin_global_tiers);
  vram_evict_demote_to_ram =
      doc.root.get_bool("vram_evict_demote_to_ram", vram_evict_demote_to_ram);
  dsv4_outproj_gpu = doc.root.get_bool("dsv4_outproj_gpu", dsv4_outproj_gpu);
  cpu_dot_ggml_avx2 = doc.root.get_bool("cpu_dot_ggml_avx2", cpu_dot_ggml_avx2);
  cpu_dot_ggml_iq1m = doc.root.get_bool("cpu_dot_ggml_iq1m", cpu_dot_ggml_iq1m);
  moe_prefill_batched = doc.root.get_bool("moe_prefill_batched", moe_prefill_batched);
  moe_decode_batched = doc.root.get_bool("moe_decode_batched", moe_decode_batched);
  moe_prefill_stream = doc.root.get_bool("moe_prefill_stream", moe_prefill_stream);
  moe_prefill_stream_chunk = (int)doc.root.get_int("moe_prefill_stream_chunk", moe_prefill_stream_chunk);
  moe_ggml_mul_mat_id = doc.root.get_bool("moe_ggml_mul_mat_id", moe_ggml_mul_mat_id);
  moe_ggml_shared_expert = doc.root.get_bool("moe_ggml_shared_expert", moe_ggml_shared_expert);
  moe_ggml_planar_copies =
      doc.root.get_bool("moe_ggml_planar_copies", moe_ggml_planar_copies);
  moe_ggml_gpu_router =
      doc.root.get_bool("moe_ggml_gpu_router", moe_ggml_gpu_router);
  ggml_async_graph_reuse =
      doc.root.get_bool("ggml_async_graph_reuse", ggml_async_graph_reuse);
  moe_ggml_device = (int)doc.root.get_int("moe_ggml_device", moe_ggml_device);
  hc_ggml = doc.root.get_bool("hc_ggml", hc_ggml);
gdn_proj_ggml = doc.root.get_bool("gdn_proj_ggml", gdn_proj_ggml);
  gdn_full_ggml = doc.root.get_bool("gdn_full_ggml", gdn_full_ggml);
  moe_expert_bf16_gemm = doc.root.get_bool("moe_expert_bf16_gemm", moe_expert_bf16_gemm);
  dense_bf16_prefill_gemm = doc.root.get_bool("dense_bf16_prefill_gemm", dense_bf16_prefill_gemm);
  dense_async_copies = doc.root.get_bool("dense_async_copies", dense_async_copies);
  ram_vram_promote_top_k =
      doc.root.get_int("ram_vram_promote_top_k", ram_vram_promote_top_k);
  ram_vram_promote_adaptive =
      doc.root.get_bool("ram_vram_promote_adaptive", ram_vram_promote_adaptive);
  ram_vram_promote_adaptive_threshold =
      doc.root.get_double("ram_vram_promote_adaptive_threshold",
                          ram_vram_promote_adaptive_threshold);
  ram_vram_promote_min_samples = doc.root.get_int(
      "ram_vram_promote_min_samples", ram_vram_promote_min_samples);
  ram_vram_promote_layer_gate = doc.root.get_bool("ram_vram_promote_layer_gate",
                                                  ram_vram_promote_layer_gate);
  ram_vram_promote_layer_gate_threshold =
      doc.root.get_double("ram_vram_promote_layer_gate_threshold",
                          ram_vram_promote_layer_gate_threshold);
  ram_vram_promote_layer_gate_min_samples =
      doc.root.get_int("ram_vram_promote_layer_gate_min_samples",
                       ram_vram_promote_layer_gate_min_samples);
  ram_vram_promote_layer_gate_top_fraction =
      doc.root.get_double("ram_vram_promote_layer_gate_top_fraction",
                          ram_vram_promote_layer_gate_top_fraction);
  ram_vram_promote_dynamic_layer = doc.root.get_bool(
      "ram_vram_promote_dynamic_layer", ram_vram_promote_dynamic_layer);
  ram_vram_promote_dynamic_threshold_4 =
      doc.root.get_double("ram_vram_promote_dynamic_threshold_4",
                          ram_vram_promote_dynamic_threshold_4);
  ram_vram_promote_dynamic_threshold_3 =
      doc.root.get_double("ram_vram_promote_dynamic_threshold_3",
                          ram_vram_promote_dynamic_threshold_3);
  ram_vram_promote_dynamic_threshold_2 =
      doc.root.get_double("ram_vram_promote_dynamic_threshold_2",
                          ram_vram_promote_dynamic_threshold_2);
  ram_vram_promote_dynamic_threshold_1 =
      doc.root.get_double("ram_vram_promote_dynamic_threshold_1",
                          ram_vram_promote_dynamic_threshold_1);
  ram_vram_promote_dynamic_k_tier4 = doc.root.get_int(
      "ram_vram_promote_dynamic_k_tier4", ram_vram_promote_dynamic_k_tier4);
  ram_vram_promote_dynamic_k_tier3 = doc.root.get_int(
      "ram_vram_promote_dynamic_k_tier3", ram_vram_promote_dynamic_k_tier3);
  ram_vram_promote_dynamic_k_tier2 = doc.root.get_int(
      "ram_vram_promote_dynamic_k_tier2", ram_vram_promote_dynamic_k_tier2);
  ram_vram_promote_dynamic_k_tier1 = doc.root.get_int(
      "ram_vram_promote_dynamic_k_tier1", ram_vram_promote_dynamic_k_tier1);
  ram_vram_promote_dynamic_k_floor = doc.root.get_int(
      "ram_vram_promote_dynamic_k_floor", ram_vram_promote_dynamic_k_floor);
  ram_vram_promote_dynamic_cold_start_k =
      doc.root.get_int("ram_vram_promote_dynamic_cold_start_k",
                       ram_vram_promote_dynamic_cold_start_k);
  ram_vram_promote_dynamic_min_samples =
      doc.root.get_int("ram_vram_promote_dynamic_min_samples",
                       ram_vram_promote_dynamic_min_samples);
  predict_gpu1_rank_start =
      doc.root.get_int("predict_gpu1_rank_start", predict_gpu1_rank_start);
  predict_gpu1_rank_end =
      doc.root.get_int("predict_gpu1_rank_end", predict_gpu1_rank_end);
  predict_gpu1_cap_per_layer = doc.root.get_int("predict_gpu1_cap_per_layer",
                                                predict_gpu1_cap_per_layer);
  reactive_promote_gpu1_every_n = doc.root.get_int(
      "reactive_promote_gpu1_every_n", reactive_promote_gpu1_every_n);
  ram_cache_layer_adaptive =
      doc.root.get_bool("ram_cache_layer_adaptive", ram_cache_layer_adaptive);
  expert_fetch_threads =
      doc.root.get_int("expert_fetch_threads", expert_fetch_threads);
  omp_max_threads = doc.root.get_int("omp_max_threads", omp_max_threads);
  attn_score_threads =
      doc.root.get_int("attn_score_threads", attn_score_threads);
  // NOTE: these four were previously missing from parsing entirely --
  // every configs/*.toml's `pinned_vram_capacity`/`move_on_promotion`/
  // `active_layer_capacity`/`use_active_layer_area` lines were silently
  // no-ops, only "working" because each one happened to match its
  // compiled default (1024/true/256/true). Any run that tried to
  // actually CHANGE one of these via TOML got the default instead, with
  // no warning.
  pinned_vram_capacity =
      doc.root.get_int("pinned_vram_capacity", pinned_vram_capacity);
  move_on_promotion = doc.root.get_bool("move_on_promotion", move_on_promotion);
  ram_copy_reclaim_threshold = doc.root.get_double("ram_copy_reclaim_threshold",
                                                   ram_copy_reclaim_threshold);
  active_layer_capacity =
      doc.root.get_int("active_layer_capacity", active_layer_capacity);
  use_active_layer_area =
      doc.root.get_bool("use_active_layer_area", use_active_layer_area);
  active_layer_promote_hit_threshold = doc.root.get_int(
      "active_layer_promote_hit_threshold", active_layer_promote_hit_threshold);
  decode_promote_all_misses =
      doc.root.get_bool("decode_promote_all_misses", decode_promote_all_misses);
  speculative_promote_to_active_layer =
      doc.root.get_bool("speculative_promote_to_active_layer",
                        speculative_promote_to_active_layer);
  speculative_active_layer_max_k = doc.root.get_int(
      "speculative_active_layer_max_k", speculative_active_layer_max_k);
  speculative_active_layer_promote_on_confirm =
      doc.root.get_bool("speculative_active_layer_promote_on_confirm",
                        speculative_active_layer_promote_on_confirm);
  active_layer_routing_policy = doc.root.get_str("active_layer_routing_policy",
                                                 active_layer_routing_policy);
  active_layer_upload_worker_threads = doc.root.get_int(
      "active_layer_upload_worker_threads", active_layer_upload_worker_threads);
  if (active_layer_upload_worker_threads < 0) {
    LOG_WARN("[CONFIG] active_layer_upload_worker_threads must be >= 0 (got "
             "%d) -- clamping to 0\n",
             active_layer_upload_worker_threads);
    active_layer_upload_worker_threads = 0;
  }
  raw_prompt = doc.root.get_bool("raw_prompt", raw_prompt);
  system_prompt = doc.root.get_str("system_prompt", system_prompt);
  enable_thinking = doc.root.get_bool("enable_thinking", enable_thinking);
  reasoning_effort = doc.root.get_str("reasoning_effort", reasoning_effort);
  log_path = doc.root.get_str("log_path", log_path);
  log_level = doc.root.get_str("log_level", log_level);
  trace_filter = doc.root.get_str("trace_filter", trace_filter);
  trace_abort_on_nan =
      doc.root.get_bool("trace_abort_on_nan", trace_abort_on_nan);

  for (auto &gt : doc.gpu_tables) {
    GpuConfig gc;
    gc.device_id = gt.get_int("device_id", -1);
    gc.name = gt.get_str("name");
    gc.role = gt.get_str("role", "dense+moe");
    gc.model_path = gt.get_str("model_path");
    gc.vram_gb = gt.get_double("vram_gb");
    gc.dense_gb = gt.get_double("dense_gb");
    gc.kv_cache_gb = gt.get_double("kv_cache_gb");
    gc.kv_bits = gt.get_int("kv_bits", 4);
    gc.moe_cache_gb = gt.get_double("moe_cache_gb");
    gc.moe_cache_reserve_gb = gt.get_double("moe_cache_reserve_gb");
    gc.active_layer_weight = gt.get_int("active_layer_weight", 0);
    gc.active_layer_capacity_slots =
        gt.get_int("active_layer_capacity_slots", 0);
    gpus.push_back(gc);
  }

  if (model_path.empty()) {
    std::cerr << "[CONFIG] Error: model_path is required.\n";
    return false;
  }

  std::cerr << "[CONFIG] Loaded from " << path << " (" << gpus.size()
            << " GPU entries)\n";
  return true;
}

inline bool EngineConfig::validate() {
  if (!zeromoe::memory::valid_embedding_device(embedding_device)) {
    std::cerr << "[CONFIG] embedding_device must be auto, cpu or gpu\n";
    return false;
  }
  for (const auto &gc : gpus) {
    if (!std::isfinite(gc.moe_cache_reserve_gb) || gc.moe_cache_reserve_gb < 0.0 ||
        gc.moe_cache_reserve_gb >= double(SIZE_MAX) / 1e9) {
      std::cerr << "[CONFIG] moe_cache_reserve_gb must be finite, non-negative and representable in bytes\n";
      return false;
    }
  }
  if (exllamav3_cuda_schedule != "auto" && exllamav3_cuda_schedule != "spin" &&
      exllamav3_cuda_schedule != "yield" && exllamav3_cuda_schedule != "blocking") {
    std::cerr << "[CONFIG] exllamav3_cuda_schedule must be auto, spin, yield or blocking\n";
    return false;
  }
  if (exl3_ggml_bridge_layer < -1 ||
      (exl3_ggml_bridge_layer >= 0 && !exl3_ggml_bridge_layers.empty())) {
    std::cerr << "[CONFIG] Use either exl3_ggml_bridge_layer or "
                 "exl3_ggml_bridge_layers, not both.\n";
    return false;
  }
  for (size_t index = 0; index < exl3_ggml_bridge_layers.size(); ++index) {
    if (exl3_ggml_bridge_layers[index] < 0 ||
        std::find(exl3_ggml_bridge_layers.begin(),
                  exl3_ggml_bridge_layers.begin() + index,
                  exl3_ggml_bridge_layers[index]) !=
            exl3_ggml_bridge_layers.begin() + index) {
      std::cerr << "[CONFIG] exl3_ggml_bridge_layers must contain unique "
                   "non-negative layer indices.\n";
      return false;
    }
  }
  if ((!exl3_ggml_bridge_model_path.empty() || exl3_ggml_bridge_layer >= 0 ||
       !exl3_ggml_bridge_layers.empty()) &&
      exl3_ggml_bridge_model_path.empty()) {
    std::cerr << "[CONFIG] EXL3 GGML bridge layers require "
                 "exl3_ggml_bridge_model_path.\n";
    return false;
  }
  if (exl3_ggml_graph_capture && exl3_ggml_bridge_model_path.empty()) {
    std::cerr << "[CONFIG] exl3_ggml_graph_capture requires "
                 "exl3_ggml_bridge_model_path.\n";
    return false;
  }
  if (exl3_ggml_routed_graph_capture &&
      (!exl3_ggml_graph_capture ||
       exl3_ggml_bridge_model_path.empty())) {
    std::cerr << "[CONFIG] exl3_ggml_routed_graph_capture requires "
                 "exl3_ggml_graph_capture and exl3_ggml_bridge_model_path.\n";
    return false;
  }
  if (!exl3_ggml_bridge_model_path.empty() &&
      (!std::isfinite(exl3_gpu_cache_gb) || exl3_gpu_cache_gb <= 0.0 ||
       exl3_gpu_cache_gb > 1024.0)) {
    std::cerr << "[CONFIG] EXL3 GGML bridge requires exl3_gpu_cache_gb in (0, 1024].\n";
    return false;
  }
  if (!exl3_ggml_bridge_model_path.empty() &&
      (!std::isfinite(exl3_cpu_expert_cache_gb) ||
       exl3_cpu_expert_cache_gb < 0.0 ||
       exl3_cpu_expert_cache_gb > 1024.0)) {
    std::cerr << "[CONFIG] EXL3 GGML bridge requires exl3_cpu_expert_cache_gb in [0, 1024].\n";
    return false;
  }
  if (exl3_backend != "native" && exl3_backend != "exllamav3") {
    std::cerr << "[CONFIG] exl3_backend must be 'native' or 'exllamav3'\n";
    return false;
  }
  if (exl3_backend == "exllamav3" &&
      (model_format != "exl3" || model_load_only || exllamav3_python.empty() ||
       exllamav3_runner_path.empty())) {
    std::cerr << "[CONFIG] ExLlamaV3 requires EXL3 inference mode, exllamav3_python "
                 "and exllamav3_runner_path\n";
    return false;
  }
  if (model_format != "gguf" && model_format != "exl3") {
    std::cerr << "[CONFIG] model_format must be 'gguf' or 'exl3', got: "
              << model_format << "\n";
    return false;
  }
  if (model_format == "exl3")
    return true;

  int hw_count = 0;
  const cudaError_t device_count_status = cudaGetDeviceCount(&hw_count);
  if (device_count_status != cudaSuccess) {
    std::cerr << "[CONFIG] cudaGetDeviceCount failed: "
              << cudaGetErrorString(device_count_status)
              << " (count=" << hw_count << ")\n";
    // Keep the existing validation behavior, but make a transient driver
    // failure distinguishable from a configuration that names a nonexistent
    // ordinal.  This matters for parity runs: a no-device run must never be
    // mistaken for a CPU-only model result.
    hw_count = 0;
  }

  std::vector<GpuConfig> valid_gpus;
  for (auto &gc : gpus) {
    if (gc.device_id < 0 || gc.device_id >= hw_count) {
      std::cerr << "[CONFIG] GPU " << gc.device_id
                << " not present, skipping.\n";
      continue;
    }
    if (gc.vram_gb <= 0) {
      cudaDeviceProp prop;
      cudaGetDeviceProperties(&prop, gc.device_id);
      gc.vram_gb = prop.totalGlobalMem / 1e9;
      if (gc.name.empty())
        gc.name = prop.name;
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
    const std::string &r = gpus[i].role;
    if (r == "dense" || r == "dense+moe") {
      if (dense_gpu_idx < 0)
        dense_gpu_idx = (int)i;
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
    if (std::find(moe_gpu_idxs.begin(), moe_gpu_idxs.end(), 0) ==
        moe_gpu_idxs.end())
      moe_gpu_idxs.push_back(0);
  }

  if (dense_model_path.empty())
    dense_model_path = model_path;
  for (auto &gc : gpus) {
    if (gc.model_path.empty())
      gc.model_path = model_path;
  }

  // Auto-compute memory layout if not specified
  for (auto &gc : gpus) {
    if (gc.role == "dense" || gc.role == "dense+moe") {
      double available = gc.vram_gb - 1.0;
      if (gc.dense_gb <= 0)
        gc.dense_gb = std::min(8.0, available * 0.5);
      if (gc.kv_cache_gb <= 0)
        gc.kv_cache_gb = std::min(5.0, available * 0.3);
      if (gc.role == "dense+moe" && gc.moe_cache_gb <= 0 &&
          !qwen4exp_option_b_disable_legacy_cache) {
        gc.moe_cache_gb = available - gc.dense_gb - gc.kv_cache_gb;
        if (gc.moe_cache_gb < 0)
          gc.moe_cache_gb = 0;
      }
    }
    if (gc.role == "moe") {
      if (gc.moe_cache_gb <= 0 &&
          !qwen4exp_option_b_disable_legacy_cache)
        gc.moe_cache_gb = gc.vram_gb - 1.0;
    }
  }

  if (pin_strategy != "per_layer_topk" && pin_strategy != "global_topn") {
    std::cerr << "[CONFIG] Unknown pin_strategy '" << pin_strategy
              << "' (expected \"per_layer_topk\" or \"global_topn\") -- "
              << "falling back to \"per_layer_topk\".\n";
    pin_strategy = "per_layer_topk";
  }
  if (vram_preload_prefix_layers < 0) {
    std::cerr << "[CONFIG] vram_preload_prefix_layers cannot be negative -- "
                 "using 0 (disabled).\n";
    vram_preload_prefix_layers = 0;
  }
  if (gdn_full_ggml && !gdn_proj_ggml) {
    std::cerr
        << "[CONFIG] gdn_full_ggml=true requires gdn_proj_ggml=true (see "
           "gdn_full_ggml's own comment) -- with gdn_proj_ggml=false, "
           "every GDN layer falls back to the per-layer CPU/dense_proj path "
           "at runtime (hyper_connections.cpp's device_gdn_input now checks "
           "gdn_proj_ggml() too, so this falls back safely rather than "
           "corrupting decode as it silently did before that fix) -- but "
           "you get none of gdn_full_ggml's intended speedup. Set "
           "gdn_proj_ggml=true as well to actually use it.\n";
  }
  if (vram_cache_full_layer_prefix < 0) {
    std::cerr << "[CONFIG] vram_cache_full_layer_prefix cannot be negative -- "
                 "using 0 (disabled).\n";
    vram_cache_full_layer_prefix = 0;
  }

  {
    int promote_modes_set = (ram_vram_promote_top_k > 0 ? 1 : 0) +
                            (ram_vram_promote_adaptive ? 1 : 0) +
                            (ram_vram_promote_dynamic_layer ? 1 : 0);
    if (promote_modes_set > 1) {
      std::cerr
          << "[CONFIG] ram_vram_promote_top_k / ram_vram_promote_adaptive / "
          << "ram_vram_promote_dynamic_layer are mutually exclusive but more "
             "than "
          << "one was set -- disabling ALL rather than guessing which one you "
          << "meant. Set exactly one.\n";
      ram_vram_promote_top_k = 0;
      ram_vram_promote_adaptive = false;
      ram_vram_promote_dynamic_layer = false;
    }
  }

  if (predict_gpu1_rank_start >= 0) {
    if (predict_gpu1_rank_end < predict_gpu1_rank_start ||
        predict_gpu1_rank_end > 7 || predict_gpu1_rank_start > 7) {
      std::cerr << "[CONFIG] predict_gpu1_rank_start/end ("
                << predict_gpu1_rank_start << ".." << predict_gpu1_rank_end
                << ") invalid -- both must be in [0,7] "
                << "(pilot_pred_ids only ever covers 8 ranks) with start <= "
                   "end. Disabling.\n";
      predict_gpu1_rank_start = -1;
      predict_gpu1_rank_end = -1;
    } else if (predict_gpu1_cap_per_layer < 1) {
      std::cerr << "[CONFIG] predict_gpu1_cap_per_layer="
                << predict_gpu1_cap_per_layer
                << " must be >= 1 -- forcing 1.\n";
      predict_gpu1_cap_per_layer = 1;
    }
  }

  if (reactive_promote_gpu1_every_n < 0) {
    std::cerr << "[CONFIG] reactive_promote_gpu1_every_n="
              << reactive_promote_gpu1_every_n
              << " must be >= 0 -- forcing 0 (off).\n";
    reactive_promote_gpu1_every_n = 0;
  }

  if (disk_batch_padding && !io_uring) {
    std::cerr << "[CONFIG] disk_batch_padding=true requires io_uring=true "
                 "(padding rides the "
              << "decode thread's own io_uring ring, "
                 "RAMSafetyManager::decode_uring_ring) -- "
              << "disabling disk_batch_padding for this run.\n";
    disk_batch_padding = false;
  }
  disk_batch_padding_target =
      std::max(1, std::min(21, disk_batch_padding_target));

  return true;
}

inline void EngineConfig::print() const {
  LOG_STREAM_INFO() << "[CONFIG] ========== Engine Configuration ==========\n";
  LOG_STREAM_INFO() << "[CONFIG] Qwen prefill device-state handoff: "
                    << (qwen4exp_prefill_state_handoff ? "ON" : "off")
                    << " (retains prefill workspace when enabled)\n";
  LOG_STREAM_INFO() << "[CONFIG] Model: " << model_path << "\n";
  LOG_STREAM_INFO() << "[CONFIG] Model format: " << model_format
                    << " | load-only: " << (model_load_only ? "yes" : "no")
                    << "\n";
  if (!exl3_ggml_bridge_model_path.empty()) {
    LOG_STREAM_INFO() << "[CONFIG] EXL3 GGML bridge model: "
                      << exl3_ggml_bridge_model_path << " | layer="
                      << exl3_ggml_bridge_layer << " | layers=";
    for (size_t index = 0; index < exl3_ggml_bridge_layers.size(); ++index)
      LOG_STREAM_INFO() << (index == 0 ? "" : ",")
                        << exl3_ggml_bridge_layers[index];
    LOG_STREAM_INFO() << " | segmented CUDA graph capture: "
                      << (exl3_ggml_graph_capture ? "ON" : "off")
                      << " | routed FFN graph cache: "
                      << (exl3_ggml_routed_graph_capture ? "ON" : "off")
                      << "\n";
  }
  if (!dense_model_path.empty() && dense_model_path != model_path)
    LOG_STREAM_INFO() << "[CONFIG] Dense model: " << dense_model_path << "\n";
  if (!mtp_model_path.empty())
    LOG_STREAM_INFO() << "[CONFIG] MTP model (shared draft-block checkpoint): "
                      << mtp_model_path << "\n";
  LOG_STREAM_INFO() << "[CONFIG] Context: " << context_size
                    << " tokens | Max gen: " << max_new_tokens << "\n";
  LOG_STREAM_INFO() << "[CONFIG] Temperature: " << temperature
                    << " | Top-p: " << top_p << "\n";
  LOG_STREAM_INFO() << "[CONFIG] Drop pages: " << (drop_pages ? "yes" : "no")
                    << "\n";
  LOG_STREAM_INFO() << "[CONFIG] Direct I/O (O_DIRECT expert reads): "
                    << (direct_io ? "requested" : "off")
                    << (direct_io
                            ? " (falls back to buffered per-shard if "
                              "unsupported/unaligned -- see [DIRECT-IO] log)"
                            : "")
                    << "\n";
  if (unpacked_expert_format)
    LOG_STREAM_INFO() << "[CONFIG] Unpacked expert format: ON (single combined "
                         "DMA per expert; "
                         "startup sanity check will FATAL if the model isn't "
                         "actually tight/aligned)\n";
  LOG_STREAM_INFO() << "[CONFIG] Prefill warm-up via prefetcher: "
                    << (prefill_warm_via_prefetcher
                            ? "ON"
                            : "off (legacy synchronous)")
                    << "\n";
  if (prefill_warm_via_prefetcher)
    LOG_STREAM_INFO() << "[CONFIG] Prefill opportunistic VRAM promotion: "
                      << (prefill_opportunistic_promote
                              ? "ON"
                              : "off (hints land in RAM only until Phase 3's "
                                "own synchronous dispatch)")
                      << "\n";
  LOG_STREAM_INFO()
      << "[CONFIG] Decode async pre-warm (batched active-layer H2D for "
         "decode-time misses): "
      << (decode_async_prewarm ? "ON"
                               : "off (legacy sequential blocking upload)")
      << (decode_async_prewarm
              ? (", max_batch=" +
                 std::to_string(decode_async_prewarm_max_batch))
              : std::string())
      << (decode_async_prewarm
              ? (std::string(", nonblocking=") +
                 (decode_async_prewarm_nonblocking
                      ? "ON (poll via find_active_expert, no "
                        "cudaStreamSynchronize)"
                      : "off (blocking sync_active_layer_batch, legacy)"))
              : std::string())
      << "\n";
  LOG_STREAM_INFO()
      << "[CONFIG] MLA qabsorb BF16 batched GPU path: "
      << (attn_qabsorb_gpu ? "ON" : "off (CPU per-head matmul() loop)")
      << (attn_qabsorb_gpu
              ? (", max_vram=" + std::to_string(attn_qabsorb_gpu_max_vram_gb) +
                 "GB")
              : std::string())
      << " (no effect on DeepSeek-V4 -- dsv4_attention_gguf has no "
         "absorption/WUK at all)\n";
  LOG_STREAM_INFO()
      << "[CONFIG] MLA score GPU path (3-GEMM+softmax, decode + prefill): "
      << (attn_score_gpu ? "ON" : "off (CPU AVX-512 per-head loop)")
      << (attn_score_gpu ? (", max_vram=" +
                            std::to_string(attn_score_gpu_max_vram_gb) + "GB")
                         : std::string())
      << " (DeepSeek-V4's windowed score is already cheap on CPU -- see "
         "attn_score_gpu's own comment)\n";
  LOG_STREAM_INFO() << "[CONFIG] GLM5NEXT grouped GPU prefill: "
                    << (glm5next_moe_prefill_gpu ? "ON" : "off")
                    << " (GLM5NEXT only; host-validated routing)\n";
  LOG_STREAM_INFO() << "[CONFIG] GLM5NEXT KDA dense projection fusion: "
                    << (glm5next_dense_prefill_fusion ? "ON" : "off")
                    << " (S>1 GGML CUDA graph, legacy fallback)\n";
  LOG_STREAM_INFO() << "[CONFIG] GLM NextN/MTP speculative decoding: "
                    << (mtp_enabled ? "ON" : "off")
                    << " (draft_tokens=" << mtp_draft_tokens
                    << ", model-gated validation)\n";
  LOG_STREAM_INFO()
      << "[CONFIG] DSV4 raw-SWA/HCA score GPU path (decode only, CSA/"
         "ratio==4 always CPU): "
      << (dsv4_score_gpu ? "ON" : "off (CPU AVX-512 per-head loop)")
      << (dsv4_score_gpu ? (", max_vram=" +
                            std::to_string(dsv4_score_gpu_max_vram_gb) + "GB")
                         : std::string())
      << " (UNVERIFIED ON REAL HARDWARE -- see dsv4_score_gpu's own "
         "comment)\n";
  LOG_STREAM_INFO() << "[CONFIG] io_uring (batched prefetcher reads): "
                    << (io_uring ? "requested" : "off")
                    << (io_uring ? " (falls back to sequential preads if "
                                   "unsupported -- see [IO-URING] log)"
                                 : "")
                    << "\n";
  if (io_uring) {
    LOG_STREAM_INFO()
        << "[CONFIG] io_uring workers (decode-thread ring's io-wq pool): "
        << io_uring_workers << "\n";
  }
  if (disk_batch_padding)
    LOG_STREAM_INFO()
        << "[CONFIG] Disk batch padding: ON (target "
        << disk_batch_padding_target
        << " experts/submission, speculative reads from route_freq"
        << (disk_batch_padding_evict_dup ? ", CAN evict also-VRAM-resident RAM "
                                           "slots once free slots run out"
                                         : ", free-slots-only (V1)")
        << ")\n";
  LOG_STREAM_INFO() << "[CONFIG] Prefetch worker threads: "
                    << prefetch_worker_threads << "\n";
  LOG_STREAM_INFO()
      << "[CONFIG] Prefetch pilot_k (hints/layer transition): default="
      << prefetch_pilot_k
      << (prefetch_pilot_k_per_layer.empty()
              ? ""
              : ", " + std::to_string(prefetch_pilot_k_per_layer.size()) +
                    " layer override value(s) set (see [PREFETCH-LAYER] "
                    "report)")
      << (prefetch_pilot_k_adaptive
              ? (", adaptive ON (per-layer/rank live accuracy, threshold " +
                 std::to_string(prefetch_pilot_k_adaptive_threshold * 100.0) +
                 "%, min_samples " +
                 std::to_string(prefetch_pilot_k_adaptive_min_samples) +
                 " -- narrows this ceiling per layer, never widens it)")
              : "")
      << "\n";
  LOG_STREAM_INFO()
      << "[CONFIG] Prefetch pilot depth: " << prefetch_pilot_depth
      << (prefetch_pilot_depth == 2
              ? (" (l+2 hints ON, top-" +
                 std::to_string(prefetch_pilot_k_depth2) +
                 " ranks pushed -- UNVERIFIED for wall-clock impact, see "
                 "'Pilot recall (DEPTH-2...' " +
                 "and 'Pilot recall by rank' in the shutdown report)")
              : " (l+2 measurement-only -- see 'Pilot recall (DEPTH-2...' in "
                "the shutdown report "
                "before enabling)")
      << "\n";
  LOG_STREAM_INFO() << "[CONFIG] ATTN-PILOT (predicts from the previous "
                       "layer's post-attention state): "
                    << (prefetch_attn_pilot
                            ? ("ON, top-" +
                               std::to_string(prefetch_attn_pilot_k) +
                               " ranks pushed -- UNVERIFIED for wall-clock "
                               "impact, see 'Pilot recall "
                               "(ATTN-PILOT...' in the shutdown report")
                            : "measurement-only -- see 'Pilot recall "
                              "(ATTN-PILOT...' in the shutdown report "
                              "before enabling")
                    << "\n";
  if (attn_pilot_vram_promote_top_k > 0) {
    LOG_STREAM_INFO()
        << "[CONFIG] ATTN-PILOT RAM->VRAM speculative promotion: top-"
        << attn_pilot_vram_promote_top_k
        << " (independent of prefetch_attn_pilot above)\n";
  }
  if (ram_vram_promote_dynamic_layer && attn_pilot_vram_promote_dynamic) {
    LOG_STREAM_INFO()
        << "[CONFIG] ATTN-PILOT RAM->VRAM speculative promotion: dynamic "
           "(shares ram_vram_promote_dynamic_* tiers below, judged against "
           "ATTN-PILOT's own per-layer accuracy)\n";
  }
  if (ram_grace_period_tokens > 0) {
    LOG_STREAM_INFO()
        << "[CONFIG] RAM-tier load grace period: " << ram_grace_period_tokens
        << " decode token(s), +" << ram_grace_period_bonus
        << " eviction_score bonus (additive bias, not a hard exemption -- "
           "see ram_grace_period_tokens's own comment)\n";
  }
  if (cpu_only)
    LOG_STREAM_INFO()
        << "[CONFIG] CPU-only mode: ON (all GPU dispatch disabled)\n";
  if (!cpu_affinity_layers.empty()) {
    LOG_STREAM_INFO() << "[CONFIG] CPU-affinity layers (never GPU-dispatched "
                         "or VRAM-uploaded): ";
    for (size_t i = 0; i < cpu_affinity_layers.size(); ++i)
      LOG_STREAM_INFO() << (i > 0 ? ", " : "") << cpu_affinity_layers[i];
    LOG_STREAM_INFO() << "\n";
  }
  if (cpu_offload_max_experts > 0) {
    LOG_STREAM_INFO()
        << "[CONFIG] CPU-offload cap (decode-time real misses computed on CPU "
           "instead of VRAM-uploaded): "
        << cpu_offload_max_experts
        << " expert(s)/layer/token max -- see '[CPU-OFFLOAD]' shutdown report "
           "line\n";
  }
  if (prefill_dispatch_lookahead_tokens > 0) {
    LOG_STREAM_INFO() << "[CONFIG] Prefill dispatch lookahead (additive "
                         "RAM-warm widening of each "
                         "token's own get_experts_batch call, prefill-only): "
                      << prefill_dispatch_lookahead_tokens
                      << " token(s) ahead -- see '[PREFILL-LOOKAHEAD]' "
                         "shutdown report line\n";
  }
  if (prefill_hint_order_by_earliest_token) {
    LOG_STREAM_INFO() << "[CONFIG] Prefill Phase 2 hint push order: by "
                         "earliest-needing-token "
                         "(targets the least-lead-time tokens, not just the "
                         "most-shared experts)\n";
  }
  if (prefill_retention_weight > 0) {
    LOG_STREAM_INFO() << "[CONFIG] Prefill->decode retention handoff: weight "
                      << prefill_retention_weight
                      << " per prefill selection added to RAM eviction_score "
                         "(FREQ_BONUS_TICKS=64 = one real cache hit, for scale)\n";
  }
  if (prefill_window_cap) {
    LOG_STREAM_INFO() << "[CONFIG] Prefill sliding-window RAM capacity: ON "
                         "(active prefill layer grows "
                         "to its own real Phase-1 demand; hard extra budget "
                      << prefill_window_extra_gb
                      << " GB -- P3 trim not built, this budget is what "
                         "bounds memory) -- see the '[PREFILL-WINDOW]' "
                         "shutdown report line\n";
  }
  if (pilot_widen_all_tokens) {
    LOG_STREAM_INFO()
        << "[CONFIG] Pilot widen-all-tokens: ON"
        << (prefetch_widen_all_tokens
                ? " + real hint push ON (disk-prefetch only, no VRAM promotion)"
                : " (measurement only -- prefetch_widen_all_tokens=false)")
        << " -- see '[PILOT-WIDE]' shutdown report line\n";
  }
  LOG_STREAM_INFO() << "[CONFIG] Host RAM (DDR) budget: "
                    << (host_ram_gb > 0
                            ? std::to_string(host_ram_gb) + " GB (explicit)"
                            : "auto-detect")
                    << "\n";
  LOG_STREAM_INFO() << "[CONFIG] Host RAM expert-slot limit: "
                    << (ram_cache_total_experts > 0
                            ? std::to_string(ram_cache_total_experts) +
                                  " total MoE experts (safe byte cap still applies)"
                            : "automatic from host_ram_gb")
                    << "\n";
  LOG_STREAM_INFO() << "[CONFIG] Promote H2D threshold: "
                    << promote_hit_threshold << " hits\n";
  LOG_STREAM_INFO() << "[CONFIG] Pin threshold: " << pin_threshold
                    << (pin_threshold > 0 ? " hits" : " (pinning disabled)")
                    << "\n";
  if (!prefill_pin_file.empty())
    LOG_STREAM_INFO() << "[CONFIG] Prefill pin file: " << prefill_pin_file
                      << " (preferred for prefill-time preload ranking; falls "
                         "back to pin_file per-layer "
                         "where this has no data)\n";
  LOG_STREAM_INFO()
      << "[CONFIG] Pin file: "
      << (pin_file.empty() ? "(preload disabled)" : pin_file)
      << " | strategy=" << pin_strategy
      << (pin_strategy == "global_topn"
              ? (" (floor " + std::to_string(pin_global_floor_per_layer) +
                 "/layer, remainder ranked globally)")
              : (" (top-" +
                 (pin_topk_per_layer > 0 ? std::to_string(pin_topk_per_layer)
                                         : std::string("auto")) +
                 "/layer)"))
      << ", next " << (pin_vram_lru_fraction * 100.0)
      << "% of VRAM tier capacity (unpinned)"
      << ", next " << (pin_ram_fraction * 100.0) << "% of RAM tier capacity\n";
  if (freeze_usage_stats)
    LOG_STREAM_INFO() << "[CONFIG] Usage-file writes: FROZEN "
                         "(pin_file/prefill_pin_file read at startup "
                         "as configured, but NOT overwritten at shutdown -- "
                         "for controlled A/B testing "
                         "against a fixed ranking)\n";
  LOG_STREAM_INFO()
      << "[CONFIG] VRAM evict -> RAM demote: "
      << (vram_evict_demote_to_ram
              ? "ON (D2H-salvage evicted experts into the RAM tier)"
              : "off")
      << "\n";
  LOG_STREAM_INFO() << "[CONFIG] DeepSeek-V4 outproj GPU dispatch: "
                    << (dsv4_outproj_gpu
                            ? "ON (cuBLAS GEMV, weight VRAM-resident once per "
                              "layer, ~5.8GB total)"
                            : "off")
                    << "\n";
  LOG_STREAM_INFO()
      << "[CONFIG] CPU-fallback dot (IQ2_XXS/IQ3_XXS): "
      << (cpu_dot_ggml_avx2
              ? "AVX2 ggml-port (EXPERIMENTAL -- ~1.7-1.9x faster per isolated "
                "bench, "
                "IQ3_XXS has an unresolved accuracy caveat, see CLAUDE.md)"
              : "existing FMA path")
      << "\n";
  LOG_STREAM_INFO()
      << "[CONFIG] CPU-fallback dot (Qwen IQ1_M): "
      << (cpu_dot_ggml_iq1m
              ? "ggml Q8_K path (current IQ1_M kernel is AVX2-width; no AVX-512/VNNI IQ1_M specialization)"
              : "row dequant + FP32 path")
      << "\n";
  LOG_STREAM_INFO()
      << "[CONFIG] Qwen CPU strided grouped projections: "
      << (qwen4exp_cpu_strided_grouped
              ? "ON (DDR arena, shared Q8_K activation + fused IQ1_M gate/up)"
              : "off (persistent GGML strided graph)")
      << "\n";
  LOG_STREAM_INFO()
      << "[CONFIG] Qwen CPU one-team fused projections: "
      << (qwen4exp_cpu_fused_grouped ? "ON (IQ1_M/IQ2_XXS + IQ4_NL)" : "off")
      << "\n";
  LOG_STREAM_INFO()
      << "[CONFIG] RAM->VRAM speculative promotion: "
      << (ram_vram_promote_top_k > 0
              ? ("static top-" + std::to_string(ram_vram_promote_top_k))
              : (ram_vram_promote_adaptive
                     ? ("adaptive (threshold " +
                        std::to_string(ram_vram_promote_adaptive_threshold *
                                       100.0) +
                        "%, min_samples " +
                        std::to_string(ram_vram_promote_min_samples) + "/rank)")
                     : (ram_vram_promote_dynamic_layer
                            ? ("dynamic-per-layer (k=" +
                               std::to_string(
                                   ram_vram_promote_dynamic_k_floor) +
                               " below " +
                               std::to_string(
                                   ram_vram_promote_dynamic_threshold_1 *
                                   100.0) +
                               "%, " +
                               std::to_string(
                                   ram_vram_promote_dynamic_k_tier1) +
                               " below " +
                               std::to_string(
                                   ram_vram_promote_dynamic_threshold_2 *
                                   100.0) +
                               "%, " +
                               std::to_string(
                                   ram_vram_promote_dynamic_k_tier2) +
                               " below " +
                               std::to_string(
                                   ram_vram_promote_dynamic_threshold_3 *
                                   100.0) +
                               "%, " +
                               std::to_string(
                                   ram_vram_promote_dynamic_k_tier3) +
                               " below " +
                               std::to_string(
                                   ram_vram_promote_dynamic_threshold_4 *
                                   100.0) +
                               "%, " +
                               std::to_string(
                                   ram_vram_promote_dynamic_k_tier4) +
                               " at/above; cold-start=" +
                               std::to_string(
                                   ram_vram_promote_dynamic_cold_start_k) +
                               ", min_samples " +
                               std::to_string(
                                   ram_vram_promote_dynamic_min_samples) +
                               "/layer)")
                            : std::string("off"))))
      << "\n";
  LOG_STREAM_INFO()
      << "[CONFIG] RAM->VRAM per-layer accuracy gate: "
      << (ram_vram_promote_layer_gate
              ? ((ram_vram_promote_layer_gate_top_fraction > 0.0
                      ? ("ON (rank-based, top " +
                         std::to_string(
                             ram_vram_promote_layer_gate_top_fraction * 100.0) +
                         "% most-confident layers")
                      : ("ON (threshold " +
                         std::to_string(
                             ram_vram_promote_layer_gate_threshold * 100.0) +
                         "%")) +
                 ", min_samples " +
                 std::to_string(ram_vram_promote_layer_gate_min_samples) +
                 "/layer)")
              : std::string("off"))
      << "\n";
  LOG_STREAM_INFO() << "[CONFIG] Reactive->GPU1 routing (Direction 1): "
                    << (reactive_promote_gpu1_every_n > 0
                            ? ("ON, every " +
                               std::to_string(reactive_promote_gpu1_every_n) +
                               "th qualifying reactive promotion (real "
                               "hit_count-proven demand only)")
                            : std::string("off"))
                    << "\n";
  LOG_STREAM_INFO()
      << "[CONFIG] Predict->GPU1 promotion (Idea 1): "
      << (predict_gpu1_rank_start >= 0
              ? ("ON, rank " + std::to_string(predict_gpu1_rank_start) + ".." +
                 std::to_string(predict_gpu1_rank_end) + ", cap " +
                 std::to_string(predict_gpu1_cap_per_layer) +
                 "/layer transition " +
                 "(dual-role slice only, never the shared vault pool)")
              : std::string("off"))
      << "\n";
  LOG_STREAM_INFO()
      << "[CONFIG] Speculative promote to active layer: "
      << (speculative_promote_to_active_layer
              ? ("ON, max_k=" + std::to_string(speculative_active_layer_max_k) +
                 ", promote_on_confirm=" +
                 (speculative_active_layer_promote_on_confirm ? "ON" : "off") +
                 " (pilot's l+1 promotion targets the active-layer pool, not "
                 "the main VRAM cache)")
              : "off (default -- speculative promotion targets the main VRAM "
                "cache)")
      << "\n";
  {
    int n_al_gpus = 0;
    std::string al_detail;
    for (auto &gc : gpus) {
      if (gc.active_layer_weight > 0) {
        ++n_al_gpus;
        al_detail +=
            " gpu" + std::to_string(gc.device_id) +
            "(w=" + std::to_string(gc.active_layer_weight) +
            ",slots=" + std::to_string(gc.active_layer_capacity_slots) + ")";
      }
    }
    LOG_STREAM_INFO()
        << "[CONFIG] Active-layer pool GPUs: "
        << (n_al_gpus >= 2
                ? ("multi-GPU (" + std::to_string(n_al_gpus) +
                   " participating, policy=" + active_layer_routing_policy +
                   "):" + al_detail)
                : std::string(
                      "single-GPU (legacy, vault_gpus[0]/core_gpu, sized by "
                      "active_layer_capacity -- set active_layer_weight on 2+ "
                      "[[gpus]] entries to opt into multi-GPU routing)"))
        << "\n";
  }
  LOG_STREAM_INFO()
      << "[CONFIG] Active-layer upload worker pool: "
      << (active_layer_upload_worker_threads > 0
              ? (std::to_string(active_layer_upload_worker_threads) +
                 " worker(s) -- unpinned-source active-layer uploads offloaded "
                 "off the "
                 "decode thread")
              : std::string(
                    "off (default -- all active-layer uploads issued inline)"))
      << "\n";
  LOG_STREAM_INFO()
      << "[CONFIG] RAM cache layer-adaptive: "
      << (ram_cache_layer_adaptive
              ? "ON (per-layer capacity by historical need, not uniform)"
              : "off")
      << "\n";
  LOG_STREAM_INFO() << "[CONFIG] Expert fetch threads: " << expert_fetch_threads
                    << "\n";
  LOG_STREAM_INFO() << "[CONFIG] OMP max threads (process-wide default): "
                    << (omp_max_threads > 0 ? std::to_string(omp_max_threads)
                                            : std::string("auto"))
                    << "\n";
  LOG_STREAM_INFO()
      << "[CONFIG] Attn score threads (dedicated, attention_layer.cpp's causal "
         "score loop only): "
      << (attn_score_threads > 0
              ? std::to_string(attn_score_threads)
              : std::string(
                    "no override (uses the process-wide default above)"))
      << "\n";
  LOG_STREAM_INFO() << "[CONFIG] Pinned VRAM capacity: " << pinned_vram_capacity
                    << " | move_on_promotion: "
                    << (move_on_promotion ? "MOVE" : "COPY")
                    << (!move_on_promotion && ram_copy_reclaim_threshold > 0.0
                            ? (" (ram_copy_reclaim_threshold=" +
                               std::to_string(ram_copy_reclaim_threshold) + ")")
                            : std::string(""))
                    << "\n";
  LOG_STREAM_INFO()
      << "[CONFIG] Active layer area: "
      << (use_active_layer_area
              ? std::to_string(active_layer_capacity) + " slots, promote at " +
                    std::to_string(active_layer_promote_hit_threshold) +
                    " hit(s)"
              : "disabled")
      << "\n";
  if (raw_prompt)
    LOG_STREAM_INFO()
        << "[CONFIG] Raw prompt mode: ON -- REPL will NOT apply the chat "
           "template, stdin content is tokenized verbatim\n";
  LOG_STREAM_INFO() << "[CONFIG] Verbose MoE cache logging: "
                    << (verbose_moe_log ? "on" : "off") << "\n";
  LOG_STREAM_INFO() << "[CONFIG] Log level: " << log_level << " | Log path: "
                    << (log_path.empty() ? "(stderr)" : log_path) << "\n";
  LOG_STREAM_INFO() << "[CONFIG] Trace filter: "
                    << (trace_filter.empty() ? "(all)" : trace_filter)
                    << " | Trace abort-on-nan: "
                    << (trace_abort_on_nan ? "on" : "off") << "\n";
  for (size_t i = 0; i < gpus.size(); ++i) {
    const auto &g = gpus[i];
    LOG_STREAM_INFO() << "[CONFIG] GPU " << g.device_id << " (" << g.name
                      << ") — " << g.role << " — " << g.vram_gb
                      << " GB total\n";
    if (g.dense_gb > 0)
      LOG_STREAM_INFO() << "[CONFIG]   Dense: " << g.dense_gb << " GB\n";
    if (g.kv_cache_gb > 0)
      LOG_STREAM_INFO() << "[CONFIG]   KV Cache: " << g.kv_cache_gb << " GB ("
                        << g.kv_bits << "-bit)\n";
    if (g.moe_cache_gb > 0)
      LOG_STREAM_INFO() << "[CONFIG]   MoE Cache: " << g.moe_cache_gb
                        << " GB\n";
  }
  LOG_STREAM_INFO()
      << "[CONFIG] =============================================\n";
}

inline void EngineConfig::write_sample(const std::string &path) {
  std::ofstream f(path);
  f << R"(# ZeroMoE Engine Configuration
# See configs/ for example profiles (single_gpu, dual_gpu, triple_gpu)

model_path = "/models/GLM-5.2-UD-IQ2_M/"
model_format = "gguf"       # "gguf" or "exl3"
model_load_only = false      # EXL3 metadata/payload load smoke; no inference
# dense_model_path = "/models/GLM-5.2-Q4_K/"  # optional: separate dense weights
# mtp_model_path = "/models/mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf"  # optional: a
#   standalone "shared" MTP/NextN checkpoint (Qwen4Exp convention) -- omits
#   token_embd.weight/output.weight, which are reused from model_path.

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

# Qwen full-graph prefill/decode may hand recurrent state directly to CUDA
# decode instead of exporting it to host RAM. This retains the prefill
# workspace during decode; prompt caching and MTP keep host snapshots.
# qwen4exp_full_graph = true
# qwen4exp_full_graph_decode = true
qwen4exp_prefill_state_handoff = false

# qwen4exp_graph_expert_vram_budget_gb: full-graph decode only. Generalizes
# cpu_affinity_layers/device_expert_layers' whole-layer choice into a real
# per-expert split: spends this many GB of VRAM on the globally hottest
# routed experts (ranked from pin_file's usage histogram), across any layer
# not already claimed by cpu_affinity_layers or device_expert_layers. 0.0
# (default) = off, identical to today's behavior.
# qwen4exp_graph_expert_vram_budget_gb = 2.0

# qwen4exp_cpu_gpu_overlap: optional Qwen-only overlap of the CPU routed
# expert graph with the GPU shared expert. The implementation submits a
# shared-only graph and a second host-routed-add graph per CPU layer, so it is
# only useful for a small CPU-affinity set. The default limit of 16 disables
# that overlap automatically for larger CPU splits; set to 0 to remove the
# limit for an explicit experiment.
# qwen4exp_cpu_gpu_overlap = false
# qwen4exp_cpu_gpu_overlap_max_layers = 16
# qwen4exp_cpu_shared_device: after a Qwen CPU-affinity routed graph finishes,
# hand its result directly to the device shared-expert graph. This avoids the
# host-facing shared-expert path's extra device-to-host/device transfers, but
# does not overlap the CPU graph with a shared-only GPU graph. It is Qwen-only
# and disabled by default until an end-to-end A/B confirms the hardware win.
# qwen4exp_cpu_shared_device = false

# cpu_offload_max_experts: decode-time cap on how many of a single token/
# layer's GENUINE disk-cold misses (not RAM- nor VRAM-resident) skip the
# VRAM upload and compute directly on CPU instead, once their bytes land in
# the RAM tier via the normal SSD->DDR read (that read itself still happens
# either way -- only the DDR->VRAM DMA is skipped). 0 (default) = off. Real
# GLM-5.2 data (see CLAUDE.md "CPU-offload feasibility") showed decode-time
# misses per layer/token distributed 74.4%/21.9%/3.3%/0.4% at 0/1/2/3
# simultaneous misses (never 4+ on that run) -- cap=1 covers 96.3% of real
# miss events, cap=2 covers 99.6%. Valid range 0-8 (8 = CPU-only for cold
# misses specifically, not the same as cpu_only). See CLAUDE.md.
cpu_offload_max_experts = 0

# direct_io: read routed-expert slabs with O_DIRECT instead of buffered
# reads (Linux only). Requires the model to have been repacked with
# tools/gguf_tool pack (general.alignment >= 4096) -- a plain/un-repacked
# GGUF logs a warning and falls back to buffered reads automatically,
# never a hard failure. See tools/README.md for the repack workflow.
direct_io      = false

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
# decode_async_prewarm_nonblocking: removes decode_async_prewarm's own last
# blocking cudaStreamSynchronize -- lets find_active_expert's already-proven
# poll-before-trust check (0.0% "not ready" across dozens of real runs on
# the speculative-fill path) resolve completion instead. Protection for the
# RAM-tier source slot moves from ram_pin/ram_unpin to pending_copy_event
# (self-healing, same mechanism main-cache COPY-mode promotions already
# use). Only takes effect when decode_async_prewarm=true. Default off
# (unchanged blocking behavior). UNVERIFIED ON REAL HARDWARE.
decode_async_prewarm_nonblocking = false

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
# glm5next_moe_prefill_gpu: GLM5NEXT-only grouped GPU expert prefill. This
# is intentionally separate from moe_prefill_batched because GLM's mixed
# expert tensor geometry needs its own family-gated validation. The host
# still owns cache residency decisions; grouped expert FFNs run on CUDA.
glm5next_moe_prefill_gpu = false
# glm5next_dense_prefill_fusion: GLM5NEXT-only fusion of the eight KDA dense
# projections into one GGML CUDA graph for S>1. Any unsupported shape or
# allocation falls back to the existing dense_proj calls.
glm5next_dense_prefill_fusion = false
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

# dsv4_score_gpu: cuBLAS GPU path for DeepSeek-V4's raw-SWA + HCA attention
# score phase (decode only), replacing the CPU AVX-512 per-head loop --
# scoped to ratio==0 (raw-SWA) and ratio==128 (HCA) layers only. CSA
# (ratio==4) layers always stay on CPU regardless of this flag -- their
# Lightning-Indexer-selected blocks are a small, data-dependent gather that
# would need a second softmax merged back into this one, deliberately out of
# scope. Unlike attn_qabsorb_gpu/attn_score_gpu's DeepSeek-V4 caveat ("the
# sliding window already makes this cheap"), HCA's compressed-block count
# grows with total context (capacity = context_size/128) the same unbounded
# way GLM-5.2's MLA score does, and shares one joint softmax with the raw
# window regardless -- so this is worth having once a checkpoint actually
# uses HCA layers. UNVERIFIED ON REAL HARDWARE (this Mac has no CUDA
# toolchain) -- see tests/bench_dsv4_score_cpu_vs_gpu.cu for the bench to
# run before enabling this, and the "DSV4 raw-SWA/HCA score GPU path"
# shutdown report line for real call counts / CPU-fallback rate once you do.
dsv4_score_gpu = false
# dsv4_score_gpu_max_vram_gb: hard ceiling on the two persistent per-layer
# GPU mirrors (KVc ring + HCA compressed-block cache), checked independently
# per mirror -- on top of a real cudaMemGetInfo headroom check. A layer
# whose mirror doesn't fit under this budget just stays on the CPU score
# path (see n_dsv4_score_gpu_fallback_cpu in the shutdown report), never a
# hard failure.
dsv4_score_gpu_max_vram_gb = 1.0

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

# prefetch_pilot_k_adaptive (2026-09-13): data-driven per-layer fetch width
# instead of a flat/guessed number -- for each layer, walks ranks 0,1,2...
# using the live [PILOT-BY-LAYER-RANK] accuracy for THAT layer, stopping at
# the first rank whose own accuracy drops below the threshold below. Only
# ever narrows prefetch_pilot_k/prefetch_pilot_k_per_layer's ceiling for
# that layer, never widens past it. Off by default -- every model/config
# that doesn't set this keeps its current fetch-width behavior exactly.
prefetch_pilot_k_adaptive = false
prefetch_pilot_k_adaptive_threshold = 0.50
prefetch_pilot_k_adaptive_min_samples = 30

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

# prefetch_attn_pilot / prefetch_attn_pilot_k: ATTN-PILOT real hint wiring
# (2026-08-25) -- predicts a layer from the PREVIOUS layer's post-attention/
# pre-FFN state (not DEPTH-1's post-FFN state, not DEPTH-2's naive 2-layer
# reuse). Measurement (recall counters, "Pilot recall (ATTN-PILOT..." report)
# always runs regardless of this setting. Real GLM-5.2 data: 75.8% overall
# recall for the SAME target layer DEPTH-1 predicts (84.8%), or 75.8% when
# read as reaching 2 layers past the last known-complete layer the way
# DEPTH-2 does (72.1%) -- see CLAUDE.md for the full table. UNVERIFIED ON
# REAL HARDWARE for wall-clock decode impact -- only the recall numbers and
# the real-hardware bug-fixed measurement pipeline are verified so far.
prefetch_attn_pilot = false
prefetch_attn_pilot_k = 8

# attn_pilot_vram_promote_top_k: SEPARATE from prefetch_attn_pilot_k above
# -- of ATTN-PILOT's own already-RAM-resident top-ranked picks, how many
# get speculatively promoted straight to VRAM (same proven COPY-only
# mechanism DEPTH-1's own ram_vram_promote_top_k already uses, a SEPARATE
# cutoff not shared with it). 0 (default) = off. Watch cache-hit rate
# alongside decode throughput when raising -- over-aggressive speculative
# promotion has caused real regressions before (see CLAUDE.md).
attn_pilot_vram_promote_top_k = 0
# Opts ATTN-PILOT into the shared ram_vram_promote_dynamic_* policy below
# instead of the fixed cutoff above -- only takes effect when
# ram_vram_promote_dynamic_layer is ALSO true. Off by default.
attn_pilot_vram_promote_dynamic = false

# Host RAM (DDR) budget for the MoE expert LRU cache, in GB.
# 0 = auto-detect (88% of currently-available system memory).
host_ram_gb    = 0
# Optional total number of MoE experts resident in the host-RAM tier across
# all MoE layers. 0 = automatic byte-aware capacity; host_ram_gb remains the
# hard safety limit.
ram_cache_total_experts = 0
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
pin_file = ".zeromoe_usage"
# prefill_pin_file: separate usage file tracking ONLY prefill-phase routing
# selections, not decode's. Decode contributes far more routing volume per
# request than prefill does, so in the combined pin_file histogram a
# decode-favored expert can outrank one prefill genuinely needs every run --
# this file, when set, is preferred for prefill's own preload ranking
# (falling back to pin_file per-layer where this file has no data yet).
# Empty (default) = not used, falls back entirely to pin_file.
# prefill_pin_file = ".zeromoe_usage_prefill"
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
# pin_core_gpu_top_n: only used when pin_strategy="global_topn". Of the
# experts already selected for pinning, the top N by real global count are
# placed specifically on the CoreEngine/dual-role GPU's own small expert-
# cache slice instead of the default (unrestricted, which in practice
# always lands on the much bigger primary vault GPU). Doesn't add a new
# VRAM reservation or change WHICH experts get pinned -- only WHERE some of
# them physically land; falls back to the old placement if the CoreEngine
# GPU's own slice is too small to fit all N. 0 = disabled (old behavior).
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

# ram_vram_promote_layer_gate (2026-08-26): an ADDITIONAL veto on top of
# whichever mode above (static/adaptive, plus attn_pilot_vram_promote_top_k
# below) already picked a promotion candidate -- vetoes it if THIS LAYER's
# own live-measured prediction accuracy (same data the [PILOT-BY-LAYER] run
# log line reports) is below threshold. Real per-layer data shows a genuine
# 33-44pp accuracy spread by layer (early layers much less predictable than
# mid/late ones, see CLAUDE.md) that a flat top_k/adaptive cutoff can't see.
# Self-calibrating per session, like the adaptive mode above. Only ever
# NARROWS what the layer above already allowed, never widens past it. Off
# by default; real A/B pending.
ram_vram_promote_layer_gate = false
ram_vram_promote_layer_gate_threshold = 0.65
ram_vram_promote_layer_gate_min_samples = 30
# ram_vram_promote_layer_gate_top_fraction (2026-09-13): rank-based mode --
# "most confident layers only" instead of the flat threshold above. 0.0
# (default) = disabled, uses the flat threshold unchanged. e.g. 0.3 = only
# the top 30% of tracked layers by their OWN live accuracy pass the gate,
# regardless of the absolute number. Every existing config that doesn't
# set this keeps its current flat-threshold behavior exactly.
ram_vram_promote_layer_gate_top_fraction = 0.0

# ram_vram_promote_dynamic_layer (2026-08-26): a graduation of the veto-only
# gate above -- mutually exclusive with ram_vram_promote_top_k/adaptive
# (setting more than one of the three disables all, with a warning). Per
# layer, picks a width from FIVE bands based on that layer's OWN live
# accuracy (same counters the [PILOT-BY-LAYER] report reads):
#   accuracy >= threshold_4 (0.90) -> k_tier4 (4)
#   accuracy >= threshold_3 (0.80) -> k_tier3 (3)
#   accuracy >= threshold_2 (0.70) -> k_tier2 (2, the settled flat baseline)
#   accuracy >= threshold_1 (0.50) -> k_tier1 (1)
#   accuracy <  threshold_1        -> k_floor (0)
#   not enough live data yet (< min_samples checks) -> cold_start_k (2)
# attn_pilot_vram_promote_dynamic (above) opts ATTN-PILOT into the SAME
# tiers, judged against its own per-layer accuracy. Off by default; real
# A/B pending.
ram_vram_promote_dynamic_layer = false
ram_vram_promote_dynamic_threshold_4 = 0.90
ram_vram_promote_dynamic_threshold_3 = 0.80
ram_vram_promote_dynamic_threshold_2 = 0.70
ram_vram_promote_dynamic_threshold_1 = 0.50
ram_vram_promote_dynamic_k_tier4 = 4
ram_vram_promote_dynamic_k_tier3 = 3
ram_vram_promote_dynamic_k_tier2 = 2
ram_vram_promote_dynamic_k_tier1 = 1
ram_vram_promote_dynamic_k_floor = 0
ram_vram_promote_dynamic_cold_start_k = 2
ram_vram_promote_dynamic_min_samples = 30

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

# 2026-08-26, default off, UNVERIFIED ON REAL HARDWARE: redirect DEPTH-1/
# ATTN-PILOT's speculative one-layer-ahead RAM->VRAM promotion into this
# active-layer pool instead of the main VRAM LRU cache -- see
# EngineConfig::speculative_promote_to_active_layer's own comment
# (engine_config.hpp) for the full motivation and design. Requires
# use_active_layer_area=true above (a no-op otherwise -- there's no active
# layer pool to redirect into). Check the "Pilot speculative active-layer
# fill" shutdown report line for the real used/wasted rate.
speculative_promote_to_active_layer = false

# Separate, narrower rank cutoff for the redirect above, independent of
# ram_vram_promote_top_k/dynamic_layer's own width -- see dense_dispatch.
# hpp's speculative_active_layer_max_k_value's own comment for the real A/B
# that motivated this (real cache-hit gain, but net-negative decode
# throughput at the full dynamic-tier width, since the added DMA volume
# competes for the same bandwidth-bound pipe decode_async_prewarm also
# uses). Only has any effect when speculative_promote_to_active_layer=true.
speculative_active_layer_max_k = 2
# Default false. See EngineConfig::speculative_active_layer_promote_on_
# confirm's own comment for the real data motivating this: a speculative
# fill's first real touch confirms the pilot's guess (~94-95% correct)
# but is currently skipped for promotion, making promotion to the
# persistent main cache effectively impossible for this path -- real
# measured cost, 30-39% more promotion-ATTEMPT volume than V1's own
# persistent main-cache promotion for the same real underlying need.
# When true, count that first confirming touch -- known, real risk of
# reopening the main-cache eviction damage this whole feature was built
# to avoid, just at a much lower wrong-guess rate. UNVERIFIED ON REAL
# HARDWARE as of this writing -- needs a real A/B before trusting it.
speculative_active_layer_promote_on_confirm = false

# Multi-GPU active-layer pool routing (2026-08-28). Opt-in and additive --
# with no [[gpus]] entry setting active_layer_weight, this is completely
# inert and behavior matches the legacy single-GPU pool exactly. To enable,
# set active_layer_weight (relative routing weight, any positive int scale
# -- only ratios between participating GPUs matter) and
# active_layer_capacity_slots (this GPU's own pool size, same unit as the
# legacy active_layer_capacity above) on 2+ [[gpus]] entries below, e.g.:
#   [[gpus]]
#   device_id = 0
#   ...
#   active_layer_weight = 4
#   active_layer_capacity_slots = 224
#   [[gpus]]
#   device_id = 1
#   ...
#   active_layer_weight = 1
#   active_layer_capacity_slots = 32
# active_layer_routing_policy selects which pluggable policy (dense_
# dispatch.hpp's route_active_layer_gpu) decides, per candidate, which
# participating GPU to target -- "tiered" (only policy implemented so far)
# reserves the lowest-priority tail of each layer transition's candidates
# for the weakest weight-tier, round-robining the higher-priority majority
# evenly across GPUs of equal/near-equal weight. UNVERIFIED ON REAL
# HARDWARE as of this writing.
active_layer_routing_policy = "tiered"

# active_layer_upload_worker_threads: 0 = off (default, all active-layer
# uploads issued inline on the decode thread, unchanged prior behavior).
# When > 0, that many background worker threads take over the register+
# copy+event sequence for UNPINNED-source uploads specifically (a pinned
# source's inline call is already near-free and is left untouched). Real,
# isolated microbenchmark (tests/bench_register_parallel.cu) confirmed
# genuine concurrency across simultaneous cudaHostRegister/cudaMemcpyAsync
# calls on this hardware before this feature was built -- start small (2-4).
# UNVERIFIED ON REAL HARDWARE as of this writing.
active_layer_upload_worker_threads = 0

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
