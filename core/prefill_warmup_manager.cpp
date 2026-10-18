#include "active_layer_upload_workers.hpp"
#include "async_prefetcher.hpp"
#include "cpu_expert_executor.hpp"
#include "gpu_expert_executor.hpp"
#include "dense_dispatch.hpp"
#include "engine.hpp"
#include "expert_activation.hpp"
#include "ggml_gdn_proj_backend.hpp"
#include "ggml_glm5next_graph.hpp"
#include "ggml_hc_backend.hpp"
#include "ggml_moe_backend.hpp"
#include "ggml_qwen4exp_graph.hpp"
#include "kernels_cpu.hpp"
#include "kquant_dequant.hpp"
#include "logger.hpp"
#include "mtp_speculative.hpp"
#include "moe_pipeline.hpp"
#include "moe_router.hpp"
#include "perf_timer.hpp"
#include "prefill_expert_stream.hpp"
#include "ram_manager.hpp"
#include "scratch_buffers.hpp"
#include "tensor_trace.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <immintrin.h>
#include <limits>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

extern void mla_attention_absorbed_gguf(StreamingEngine &engine, int layer_idx,
                                        const float *Q_roped, int S,
                                        int pos_base, float *out);
extern void dense_attention_gguf(StreamingEngine &engine, int layer_idx,
                                 const float *hidden_states, int S,
                                 int pos_base, float *out);
extern bool qwen4exp_hc_device_live();
extern bool qwen4exp_hc_post_combine_device(StreamingEngine &engine,
                                            int layer_idx, int sub_block, int S,
                                            int D,
                                            const void *device_sub_block_out,
                                            const void *producer_event);
extern bool qwen4exp_hc_post_combine_device_raw(StreamingEngine &engine,
                                                int layer_idx, int sub_block,
                                                int S, int D,
                                                const void *device_data);
extern bool qwen4exp_hc_prepare_host_layer_input(StreamingEngine &engine,
                                                 int layer_idx, int S);
extern bool qwen4exp_hc_download_streams(StreamingEngine &engine, int S);
extern bool qwen4exp_hc_commit_host_layer_input(StreamingEngine &engine,
                                                int layer_idx, int S);
extern bool qwen4exp_copy_gdn_device_output_to_host(const void *device_output,
                                                    float *host, int S);

#include "moe_pipeline_internal.hpp"
#include "prefill_warmup_manager.hpp"

void PrefillWarmupManager::prepare(MoePipelineContext &context) {
  auto &engine = context.engine;
  auto &ram_mgr = context.ram_mgr;
  auto &async_io_queue = context.async_io_queue;
  const int layer_idx = context.layer_idx;
  const float *hidden_states = context.hidden_states;
  const int S = context.S;
  float *output = context.output;
  auto &scratch = context.scratch;
  const int *token_ids = context.token_ids;
  auto &c = context.c;
  const int D = context.D;
  const int topk = context.topk;
  const int n_experts = context.n_experts;
  const int moe_inter = context.moe_inter;
  const float clamp_exp = context.clamp_exp;
  const float clamp_shexp = context.clamp_shexp;
  auto &gpu_router_handled = context.gpu_router_handled;
  auto &gpu_route_device = context.gpu_route_device;
  auto &gpu_route_ids_device = context.gpu_route_ids_device;
  auto &gpu_route_weights_device = context.gpu_route_weights_device;
  auto &device_router = context.device_router;
  auto &gpu_route_ids_ks = context.gpu_route_ids_ks;
  auto &gpu_route_weights_ks = context.gpu_route_weights_ks;
  auto &batched_handled = context.batched_handled;
  // --- Prefill sliding-window RAM capacity, P2 (2026-08-30) ------------
  // EngineConfig::prefill_window_cap. Placed HERE, immediately after
  // Phase 1 and BEFORE Phase 2 pushes this layer's prefetch hints, for
  // the whole reason the feature exists: Phase 2's hints are what
  // overrun the RAM tier's per-layer cap and make the prefetcher evict
  // its own not-yet-consumed work (82.2% measured untouched-waste), so
  // the cap has to already be raised by the time those hints go out.
  //
  // The demand number is EXACT, not a prediction: Phase 1 above just ran
  // the real batched router over every token in the batch, so
  // expert_freq>0 is precisely the set of experts this layer will
  // actually dispatch -- no speculation, no accuracy question, unlike
  // every cross-layer pilot mechanism in this file.
  //
  // Prefill-only by construction (S > 1). Decode keeps the uniform cap:
  // there, every layer genuinely IS live simultaneously (each token
  // traverses all of them), which is exactly the assumption
  // calculate_safe_cache_capacity's divide-by-n_layers already encodes
  // correctly.
  if (!gpu_route_device && S > 1 && ram_mgr.prefill_window_enabled) {
    int unique_needed = 0;
    for (int e = 0; e < n_experts; ++e)
      if (scratch.expert_freq[e] > 0)
        ++unique_needed;
    ram_mgr.set_prefill_window_cap(engine, layer_idx, unique_needed);
  }

  // MEASUREMENT ONLY, prefill-only -- see PerfCounters::
  // n_moe_bucket_unique_experts's own comment for the full motivation
  // (the llama.cpp -ub/mul_mat_id batching investigation). scratch.
  // expert_freq is already fully populated for this layer's whole
  // S-token batch by the Phase 1 loop just above -- expert_freq[e] IS
  // the real bucket size a batched-per-expert dispatch would see for
  // expert e this layer, and sum(expert_freq)==S*topk is today's real
  // per-(token,k) dispatch count. Read-only tally, zero effect on
  // dispatch/caching. Decode (S==1) is excluded on purpose: every
  // bucket there is trivially size <=1 by construction (one token),
  // so there's nothing to measure.
  if (!gpu_route_device && S > 1) {
    int unique_experts = 0;
    int max_bucket = 0;
    for (int e = 0; e < n_experts; ++e) {
      int f = scratch.expert_freq[e];
      if (f > 0) {
        ++unique_experts;
        if (f > max_bucket)
          max_bucket = f;
      }
    }
    int total_dispatches = S * topk;
    perf().n_moe_bucket_unique_experts.fetch_add((uint64_t)unique_experts,
                                                 std::memory_order_relaxed);
    perf().n_moe_bucket_total_dispatches.fetch_add((uint64_t)total_dispatches,
                                                   std::memory_order_relaxed);
    perf().n_moe_bucket_layers_sampled.fetch_add(1, std::memory_order_relaxed);
    double reduction = unique_experts > 0
                           ? (double)total_dispatches / (double)unique_experts
                           : 0.0;
    LOG_INFO("[MOE-BUCKET] layer=%d tokens=%d topk=%d total_dispatches=%d "
             "unique_experts=%d "
             "max_bucket=%d reduction=%.2fx\n",
             layer_idx, S, topk, total_dispatches, unique_experts, max_bucket,
             reduction);
  }

  // --- Phase 2: cold-start VRAM warm-up (prefill only, S > 1) ---
  // A brand-new turn's prefill sees every prompt token in one shot, so
  // unlike steady-state decode (one token, one uncertain guess at a time)
  // we already have a full-batch demand signal *before* running a single
  // expert. Mirror how an OS warms an empty page cache on first touch:
  // load into any still-EMPTY VRAM slot immediately, prioritized by
  // in-batch frequency, since filling empty space costs nothing (no
  // eviction). Once the cache is full, fall back to the steady-state
  // hotness-gated promotion in RAMSafetyManager::get_expert() — same as
  // an OS reverting to LRU swap-out once the cache is warm.
  if (!gpu_route_device && S > 1 && ram_mgr.vram_cache &&
      !is_cpu_affinity_layer(layer_idx)) {
    PERF_START(prefill_warm);
    std::vector<int> candidates;
    candidates.reserve(n_experts);
    for (int e = 0; e < n_experts; ++e)
      if (scratch.expert_freq[e] > 0)
        candidates.push_back(e);
    // EngineConfig::prefill_hint_order_by_earliest_token -- see that
    // field's own comment for the motivation. Same candidate SET
    // either way (every expert with expert_freq>0); only the PUSH
    // ORDER changes.
    const bool dsv41_grouped =
        engine.family_ops.family == ModelFamily::DeepSeek_V41 &&
        moe_prefill_batched();
    // Grouped dispatch walks expert IDs, so issue its read-ahead in that
    // same order. Candidates are already ascending here.
    if (!dsv41_grouped && engine.prefill_hint_order_by_earliest_token) {
      std::sort(candidates.begin(), candidates.end(), [&](int a, int b) {
        return scratch.first_needed_by_token[a] <
               scratch.first_needed_by_token[b];
      });
    } else if (!dsv41_grouped) {
      std::sort(candidates.begin(), candidates.end(), [&](int a, int b) {
        return scratch.expert_freq[a] > scratch.expert_freq[b];
      });
    }

    // prefill_warm_via_prefetcher=true needs a GPU-type-capability check
    // WITHOUT ever calling ram_mgr.get_expert() -- that synchronous
    // RAM/disk fetch is exactly the blocking cost this mode exists to
    // avoid. expert_layer_geometry is a pure tensor-table lookup (same
    // expert-0-representative convention Phase 3's own shadow-check
    // uses below), computed once per layer rather than per candidate.
    bool hint_mode_gpu_capable = true;
    if (engine.prefill_warm_via_prefetcher) {
      size_t p2_gb = 0, p2_ub = 0, p2_db = 0;
      const GGUFTensor *p2_gM = nullptr, *p2_uM = nullptr, *p2_dM = nullptr;
      hint_mode_gpu_capable =
          engine.expert_layer_geometry(layer_idx, p2_gb, p2_ub, p2_db, p2_gM,
                                       p2_uM, p2_dM) &&
          gpu_type_supported(p2_gM->type) && gpu_type_supported(p2_uM->type) &&
          gpu_type_supported(p2_dM->type);
    }

    // Active-layer BATCH pre-warm (async issue + one shared sync) --
    // a SEPARATE pool from the main vram_cache pre-warm loop below.
    // GLM's dominant cold-miss dispatch path is upload_expert_active
    // (find_active_expert/upload_expert_active), not the main cache,
    // so THIS pool is what actually targets active_up's real cost
    // (2026-08-12 prefill investigation, see CLAUDE.md: active_up was
    // ~29% of prefill's own wall time, the single largest bucket).
    // No resource-competition risk with the main-cache loop below
    // (unlike prefill_opportunistic_promote's reverted mistake) --
    // these are two independent VRAM pools with no shared gate.
    // Scoped to THIS layer only, entirely inside Phase 2 -- by the
    // time Phase 3 begins a few lines below, sync_active_layer_batch
    // has already confirmed every issued copy landed (load-bearing
    // ordering, not just tidy -- see upload_expert_active_async's own
    // correctness contract in gpu_topology.hpp). Anything not covered
    // here (not yet RAM-resident, active-layer area unavailable/full,
    // or GPU-incapable) falls through to Phase 3's existing reactive
    // path unchanged -- never a correctness issue, only a missed-
    // overlap opportunity for that specific expert.
    // The mmap stream acquires the current layer's exact expert set directly
    // in Phase 3b, so active-layer VRAM prewarm would duplicate those H2D
    // transfers and is guaranteed to be reclaimed without a dispatch hit.
    // Keep the prefetch-hint loop below active: its RAM reads warm the OS
    // page cache that mmap staging depends on, as a real A/B demonstrated.
    const bool prefill_stream_bypasses_active_prewarm =
        moe_prefill_stream() && engine.model_file.mmap_active();
    if (!prefill_stream_bypasses_active_prewarm && !dsv41_grouped &&
        !moe_pipeline_detail::qwen4exp_decode_ggml_path(engine) &&
        ram_mgr.vram_cache->active_layer_capacity > 0) {
      std::vector<ExpertLocation *> active_prewarm_batch;
      std::vector<MoEExpertBlock *> active_prewarm_pinned_blocks;
      const size_t active_capacity =
          (size_t)ram_mgr.vram_cache->active_layer_capacity;
      active_prewarm_batch.reserve(
          std::min(candidates.size(), active_capacity));
      active_prewarm_pinned_blocks.reserve(
          std::min(candidates.size(), active_capacity));
      for (int expert_id : candidates) {
        // Phase 2 is a best-effort warmup, not a second unbounded dispatch
        // queue. Once every active-layer slot has an issued copy, stop
        // asking the topology for more victims. The old loop attempted all
        // 39-72 unique experts in a prefill layer with a 24-slot pool,
        // generating a warning for every impossible upload and needlessly
        // contending on cache_mutex. Phase 3 still handles the remaining
        // experts through its normal synchronous path.
        if (active_prewarm_batch.size() >= active_capacity)
          break;
        // Side-effect-free peek (active_layer_has), NOT
        // find_active_expert -- see that function's own comment
        // for why calling the real, hit-count-bumping/promotion-
        // triggering lookup speculatively here would corrupt the
        // exact state this batch is simultaneously choosing
        // victims from.
        if (ram_mgr.vram_cache->active_layer_has(layer_idx, expert_id))
          continue;
        if (ram_mgr.vram_cache->find_expert(layer_idx, expert_id))
          continue; // main-cache hit already covers it
        // Non-blocking: only pre-warm what's ALREADY RAM-resident
        // (from phase_b's own prefetch coverage) -- never trigger
        // a fresh disk read here, that would defeat the point.
        const uint8_t *ready_slab = nullptr;
        // The H2D below is asynchronous. Keep the exact RAM slot pinned from
        // this atomic lookup until sync_active_layer_batch confirms that the
        // DMA has finished; a plain find-then-async-copy leaves an eviction /
        // slab-regrow window and can upload another expert's bytes.
        MoEExpertBlock *ready = ram_mgr.find_ram_slot_if_ready_held(
            engine.layers_experts[layer_idx],
            (size_t)ram_mgr.cap_for_layer(layer_idx), layer_idx, expert_id,
            &ready_slab, nullptr);
        if (!ready) {
          perf().n_active_layer_prewarm_not_ram_ready.fetch_add(
              1, std::memory_order_relaxed);
          continue;
        }
        bool ready_gpu_capable =
            gpu_type_supported(ready->gate_proj.meta.type) &&
            gpu_type_supported(ready->up_proj.meta.type) &&
            gpu_type_supported(ready->down_proj.meta.type);
        if (!ready_gpu_capable) {
          ram_mgr.ram_unpin(*ready);
          continue;
        }
        // 2026-08-27: real coverage measurement -- see
        // n_active_layer_upload_src_pinned's own comment
        // (perf_timer.hpp).
        if (ready->slab_pinned)
          perf().n_active_layer_upload_src_pinned.fetch_add(
              1, std::memory_order_relaxed);
        else {
          perf().n_active_layer_upload_src_unpinned.fetch_add(
              1, std::memory_order_relaxed);
          perf().mark_unpinned_touch(layer_idx, expert_id);
        }
        // Real total, not slab_capacity -- see decode_async_prewarm's own
        // fix comment for the full incident.
        ExpertLocation *v = ram_mgr.vram_cache->upload_expert_active_async(
            layer_idx, expert_id, ready_slab,
            ready->gate_proj.meta.size_bytes + ready->up_proj.meta.size_bytes +
                ready->down_proj.meta.size_bytes);
        // 2026-08-30 diagnostic -- see MoEExpertBlock::consumed_by_
        // vram_upload (engine.hpp). Phase 2's own active-layer
        // pre-warm is the single largest suspected contributor to
        // the ambiguous 86.6% "Unified RAM waste" figure: it reads
        // this RAM slot's bytes straight into the active-layer VRAM
        // pool, and Phase 3's dispatch then resolves out of THAT
        // pool via find_active_expert -- so the RAM slot is
        // genuinely consumed but its own hit_count is never bumped,
        // making it indistinguishable at eviction time from a slot
        // nobody ever read.
        if (v) {
          active_prewarm_batch.push_back(v);
          active_prewarm_pinned_blocks.push_back(ready);
          ram_mgr.mark_consumed_by_vram_upload(*ready);
        } else {
          // No destination slot was claimed, so the source hold is no longer
          // needed. Do not leave an otherwise healthy RAM slot unevictable.
          ram_mgr.ram_unpin(*ready);
        }
      }
      if (!active_prewarm_batch.empty()) {
        ram_mgr.vram_cache->sync_active_layer_batch(active_prewarm_batch);
      }
      for (MoEExpertBlock *ready : active_prewarm_pinned_blocks)
        ram_mgr.ram_unpin(*ready);
    }

    for (int expert_id : candidates) {
      if (ram_mgr.vram_cache->find_expert(layer_idx, expert_id))
        continue; // already resident

      if (engine.prefill_warm_via_prefetcher) {
        // Async path (config prefill_warm_via_prefetcher=true): push
        // a hint to the SAME queue/worker-thread pipeline decode's
        // own PILOT prediction already uses, instead of loading
        // synchronously right here. Trade-off documented on the
        // config field itself -- this legitimately might not finish
        // in time for THIS layer's own Phase 3 dispatch a moment
        // later, falling through to the ordinary (already-correct)
        // miss path for this call; never a correctness issue.
        //
        // Deliberately NOT gated on vram_cache->has_free_slot() the
        // way the legacy synchronous path below still is (2026-08-
        // 12 fix). A pushed hint only ever populates the RAM tier
        // -- active_prefetch (async_prefetcher.hpp) never touches
        // VRAM from a prefetch-worker thread (see claim_or_wait's
        // own CRITICAL rule) -- so it doesn't need a free VRAM
        // slot at all, only RAM-tier capacity (host_ram_gb,
        // currently 120GB, far less contended than the ~1400-slot
        // VRAM cache that saturates partway through nearly every
        // prefill run). Gating the hint-push on VRAM capacity it
        // doesn't need was needlessly conservative: real per-
        // request [PREFILL] instrumentation showed phase_b (real,
        // decode-thread-BLOCKING disk reads inside Phase 3's own
        // miss handling) costing ~26% of prefill's own wall time
        // -- exactly the cost this gate creates, by refusing to
        // even START a background RAM fetch for any layer whose
        // candidates come up once VRAM happens to be full,
        // forcing that layer's real Phase-3 miss to pay for a
        // synchronous disk read from scratch instead of finding
        // it already RAM-resident. No VRAM-competition risk here
        // (unlike prefill_opportunistic_promote, which DID
        // compete for VRAM capacity and was reverted for exactly
        // that reason) -- this only ever adds queue pressure on
        // the prefetcher's own worker threads, not VRAM slots.
        // UNVERIFIED on real hardware -- watch "Prefetch pickup"
        // (avg ms/backlog) for a WORSE queue backlog than before
        // (this could push MORE total hints than previously,
        // since the loop no longer breaks early), which would
        // mean this trades one bottleneck for another rather than
        // removing it. See CLAUDE.md.
        if (!hint_mode_gpu_capable)
          continue;
        async_io_queue.push_hint(layer_idx, expert_id);
        perf().n_prefill_hint_pushed.fetch_add(1, std::memory_order_relaxed);
        // Track for cross-layer opportunistic VRAM promotion (see
        // LayerScratch::pending_prefill_promotions's own comment,
        // and the follow-up pass above). Silently drops the newest
        // hint if the small pending list is already full rather
        // than growing unbounded -- Phase 3's synchronous dispatch
        // is always the correctness backstop either way, so a
        // dropped tracking entry only forgoes the OPPORTUNISTIC
        // early promotion for that one expert, never correctness.
        if (!engine.prefill_opportunistic_promote) {
          // Not tracked at all -- see EngineConfig::prefill_
          // opportunistic_promote's own comment for the A/B this
          // gate exists to enable.
        } else if (scratch.pending_prefill_count <
                   LayerScratch::PENDING_PREFILL_CAP) {
          scratch.pending_prefill_promotions[scratch.pending_prefill_count++] =
              {layer_idx, expert_id, 0};
        } else {
          perf().n_prefill_pending_cap_dropped.fetch_add(
              1, std::memory_order_relaxed);
        }
        continue;
      }

      // ---- Legacy synchronous path (default, unchanged behavior) ----
      // has_free_slot() gate lives HERE now, not at the top of the
      // loop -- this path DOES promote directly into VRAM
      // (upload_expert_async below), so it genuinely needs a free
      // slot, unlike the hint-push path above. See that branch's own
      // comment for the full reasoning.
      if (!ram_mgr.vram_cache->has_free_slot())
        break; // cache warm now — stop preloading
      // This path queues an asynchronous main-cache H2D directly from the
      // RAM slab. Request an atomic caller hold from get_expert so a
      // prefetch worker cannot evict/regrow the source between the lookup
      // and upload_expert_async(). The promotion bookkeeping below transfers
      // protection to pending_copy_event once the DMA has been recorded.
      MoEExpertBlock *blk = ram_mgr.get_expert(engine, layer_idx, expert_id,
                                               /*pin_for_caller=*/true);
      if (!blk || !blk->slab_buffer) {
        if (blk)
          ram_mgr.ram_unpin(*blk);
        continue;
      }

      // Same all-three-projections-must-be-GPU-supported gate as
      // RAMSafetyManager::get_expert's steady-state promotion — a
      // Dynamic-quant model where down uses an unimplemented format
      // (e.g. IQ3_XXS) can never dispatch to GPU regardless of VRAM
      // residency, so warming those into VRAM only wastes a free slot
      // that a genuinely GPU-capable expert could use later.
      bool gpu_capable = gpu_type_supported(blk->gate_proj.meta.type) &&
                         gpu_type_supported(blk->up_proj.meta.type) &&
                         gpu_type_supported(blk->down_proj.meta.type);
      if (!gpu_capable) {
        ram_mgr.ram_unpin(*blk);
        continue;
      }

      LOG_DEBUG("[MOE-VRAM]  WARM    layer=%d expert=%d freq=%d/%d tokens "
                "(cold-start prefill, free slot)\n",
                layer_idx, expert_id, scratch.expert_freq[expert_id], S);
      // FIX (real, measured "[TIER-OVERLAP]" leak -- see CLAUDE.md's
      // "third, independent tier-overlap source" entry): this call
      // used to bypass get_expert's own MOVE-vs-COPY promotion
      // bookkeeping entirely, since it's a direct external
      // upload_expert_async call, not a hit routed through
      // get_expert's internal Step-1 branch. apply_promotion_
      // bookkeeping mirrors that same logic for this call site.
      perf().n_prefill_warm_attempted.fetch_add(1, std::memory_order_relaxed);
      // Real total, not slab_capacity -- see decode_async_prewarm's own
      // fix comment for the full incident.
      ExpertLocation *promoted = ram_mgr.vram_cache->upload_expert_async(
          layer_idx, expert_id, blk->slab_buffer,
          blk->gate_proj.meta.size_bytes + blk->up_proj.meta.size_bytes +
              blk->down_proj.meta.size_bytes,
          -1, blk->slab_pinned);
      if (promoted)
        perf().n_prefill_warm_succeeded.fetch_add(1, std::memory_order_relaxed);
      else
        perf().n_prefill_warm_failed.fetch_add(1, std::memory_order_relaxed);
      ram_mgr.apply_promotion_bookkeeping(*blk, layer_idx, expert_id, promoted);
      ram_mgr.ram_unpin(*blk);
    }

    // Cross-layer follow-up for LayerScratch::pending_prefill_promotions
    // (see that field's own comment) -- runs AFTER the candidate loop
    // above, not before. ORDERING FIX (2026-08-12): running this FIRST
    // (the original placement) let it consume vram_cache->has_free_slot()
    // capacity before THIS layer's own hint-pushing loop got to decide
    // how much of that shared budget it wanted -- confirmed via a real
    // A/B to starve LATER layers of hints entirely (n_prefill_hint_pushed
    // nearly halved, 420 vs ~930), a genuine regression (see CLAUDE.md).
    // Running it AFTER means the candidate loop above always gets first
    // claim on whatever's free; this pass only ever mops up whatever's
    // left, which can never come at another layer's expense. Still
    // fires every Phase-2 entry regardless of THIS layer's own
    // prefill_warm_via_prefetcher status, since a pending entry can
    // reference an EARLIER layer. Cheap no-op when the list is empty.
    if (engine.prefill_opportunistic_promote &&
        scratch.pending_prefill_count > 0) {
      constexpr int PENDING_PREFILL_MAX_AGE =
          12; // layers; starting guess, unverified on hardware
      int write = 0;
      for (int i = 0; i < scratch.pending_prefill_count; ++i) {
        auto &pend = scratch.pending_prefill_promotions[i];
        ++pend.age_layers;
        bool resolved = false;

        if (ram_mgr.vram_cache->find_expert(pend.layer_idx, pend.expert_id)) {
          perf().n_prefill_pending_already_resident.fetch_add(
              1, std::memory_order_relaxed);
          resolved = true; // something else already promoted it (e.g. a
                           // reactive hit) -- fine
        } else if (!ram_mgr.vram_cache->has_free_slot()) {
          perf().n_prefill_pending_no_free_slot.fetch_add(
              1, std::memory_order_relaxed);
          resolved = true; // cache is warm now -- opportunistic promotion is
                           // moot for the rest of this run
        } else {
          const uint8_t *ready_slab = nullptr;
          MoEExpertBlock *ready = ram_mgr.find_ram_slot_if_ready_held(
              engine.layers_experts[pend.layer_idx],
              (size_t)ram_mgr.cap_for_layer(pend.layer_idx), pend.layer_idx,
              pend.expert_id, &ready_slab, nullptr);
          if (!ready) {
            perf().n_prefill_pending_still_loading.fetch_add(
                1, std::memory_order_relaxed);
            resolved = (pend.age_layers >= PENDING_PREFILL_MAX_AGE);
            if (resolved)
              perf().n_prefill_pending_aged_out.fetch_add(
                  1, std::memory_order_relaxed);
          } else {
            size_t p2_gb = 0, p2_ub = 0, p2_db = 0;
            const GGUFTensor *p2_gM = nullptr, *p2_uM = nullptr,
                             *p2_dM = nullptr;
            bool ready_gpu_capable =
                engine.expert_layer_geometry(pend.layer_idx, p2_gb, p2_ub,
                                             p2_db, p2_gM, p2_uM, p2_dM) &&
                gpu_type_supported(p2_gM->type) &&
                gpu_type_supported(p2_uM->type) &&
                gpu_type_supported(p2_dM->type);
            resolved = true; // decided one way or the other below, either way
                             // don't re-check it again
            if (ready_gpu_capable && ready_slab) {
              // Real total, not slab_capacity -- see decode_async_prewarm's
              // own fix comment for the full incident.
              ExpertLocation *promoted =
                  ram_mgr.vram_cache->upload_expert_async(
                      pend.layer_idx, pend.expert_id, ready_slab,
                      ready->gate_proj.meta.size_bytes +
                          ready->up_proj.meta.size_bytes +
                          ready->down_proj.meta.size_bytes,
                      -1, ready->slab_pinned);
              // 2026-08-30 diagnostic -- see MoEExpertBlock::
              // consumed_by_vram_upload (engine.hpp).
              if (promoted)
                ram_mgr.mark_consumed_by_vram_upload(*ready);
              if (promoted)
                perf().n_prefill_pending_promoted.fetch_add(
                    1, std::memory_order_relaxed);
              ram_mgr.apply_promotion_bookkeeping(*ready, pend.layer_idx,
                                                  pend.expert_id, promoted);
            }
            // The async main-cache upload has either transferred its source
            // protection to pending_copy_event (apply_promotion_bookkeeping)
            // or did not issue a copy. In both cases this lookup hold is no
            // longer needed once the bookkeeping transition is complete.
            ram_mgr.ram_unpin(*ready);
          }
        }

        if (!resolved)
          scratch.pending_prefill_promotions[write++] =
              pend; // keep, still legitimately pending
      }
      scratch.pending_prefill_count = write;
    }

    PERF_END(prefill_warm, t_prefill_warm_time);
  }

}

void PrefillWarmupManager::finish(MoePipelineContext &context) {
  auto &engine = context.engine;
  auto &ram_mgr = context.ram_mgr;
  auto &async_io_queue = context.async_io_queue;
  const int layer_idx = context.layer_idx;
  const float *hidden_states = context.hidden_states;
  const int S = context.S;
  float *output = context.output;
  auto &scratch = context.scratch;
  const int *token_ids = context.token_ids;
  auto &c = context.c;
  const int D = context.D;
  const int topk = context.topk;
  const int n_experts = context.n_experts;
  const int moe_inter = context.moe_inter;
  const float clamp_exp = context.clamp_exp;
  const float clamp_shexp = context.clamp_shexp;
  auto &gpu_router_handled = context.gpu_router_handled;
  auto &gpu_route_device = context.gpu_route_device;
  auto &gpu_route_ids_device = context.gpu_route_ids_device;
  auto &gpu_route_weights_device = context.gpu_route_weights_device;
  auto &device_router = context.device_router;
  auto &gpu_route_ids_ks = context.gpu_route_ids_ks;
  auto &gpu_route_weights_ks = context.gpu_route_weights_ks;
  auto &batched_handled = context.batched_handled;
  // Three-tier caching: Cleanup active layer area after processing this layer
  // This reclaims temporary workspace used for current-layer experts
  if (ram_mgr.vram_cache && ram_mgr.vram_cache->active_layer_pool) {
    PERF_SCOPE(t_layer_cleanup_total);
    // FIX ("[TIER-OVERLAP]" leak, see CLAUDE.md): the rare end-of-layer
    // promotion path (a slot that crossed active_layer_promote_threshold
    // but couldn't promote inline via find_active_expert, e.g. because
    // it was still held mid-compute at that moment) has the same
    // durable-main-cache-now-exists fact to report as the inline hit
    // path does -- but unlike that call site, there's no single
    // MoEExpertBlock* in scope here (this is a sweep over up to
    // active_layer_capacity slots, not one dispatch's gpu_ctx), so this
    // goes through RAMSafetyManager::finish_sync_promotion's lookup
    // overload (a real RAM-tier lookup by (layer, expert_id)) rather
    // than gpu_ctx->exp directly -- see that function's own comment in
    // ram_manager.hpp for why this reclaims the slot immediately rather
    // than just marking it (a real run showed marking-only barely moves
    // [TIER-OVERLAP] within a short test).
    ram_mgr.vram_cache->cleanup_active_layer_area(
        layer_idx,
        [&](int l, int eid) { ram_mgr.finish_sync_promotion(engine, l, eid); });
  }

  // prefill->decode retention handoff (2026-08-31). Placed HERE, at the real
  // end of this layer's prefill pass, and NOT next to the Phase-1 window
  // block where it was first (wrongly) wired.
  //
  // The bug that placement had, worth recording because it made the whole
  // feature nearly inert: expert_freq is final after Phase 1, so stamping
  // right after Phase 1 *looks* correct -- but at that instant most of this
  // layer's experts are NOT RESIDENT YET. They arrive during Phase 2's
  // prefetch hints and Phase 3's own dispatch, i.e. AFTER that point, so
  // they were never stamped and kept prefill_demand=0 -- precisely the
  // newly-loaded experts that most need retention. Stamping after Phase 3
  // instead catches every expert this layer actually brought in.
  //
  // Note this is genuinely "who survived", not "who was needed": a slot
  // evicted mid-pass is gone and cannot be stamped. That is the intended
  // semantics -- eviction_score can only reorder what is still resident, so
  // the stamp describes exactly the population it can influence.
  if (S > 1) {
    ram_mgr.stamp_prefill_demand(engine, layer_idx, scratch.expert_freq);
  }

}
