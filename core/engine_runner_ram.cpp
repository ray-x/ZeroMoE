// EngineRunner phases: init_ram_tier.
// Split from the former run_engine() in src/main.cpp; bodies are unchanged except where
// a former local became an EngineRunner member of the same name.
#include "engine_runner.hpp"
#include "graph_config.hpp"
#include "execution_policy.hpp"

#include <limits>

std::expected<void, std::string> EngineRunner::validate_expert_placement() {
  const auto &c = engine.model_file.architecture_config;
  if (c.n_experts <= 0 || cache_layers <= 0 || !ram_mgr.vram_cache)
    return {};

  uint64_t total_slots = 0;
  uint64_t ram_slots = 0;
  uint64_t total_bytes = 0;
  uint64_t ram_bytes = 0;
  int moe_layers = 0;
  auto add_bytes = [](uint64_t &dst, uint64_t count, uint64_t bytes) {
    if (count != 0 && bytes > (std::numeric_limits<uint64_t>::max() - dst) / count)
      dst = std::numeric_limits<uint64_t>::max();
    else
      dst += count * bytes;
  };
  for (int layer = 0; layer < cache_layers; ++layer) {
    const size_t bytes = engine.compute_expert_slab_bytes(layer);
    if (bytes == 0)
      continue;
    ++moe_layers;
    total_slots += (uint64_t)c.n_experts;
    const uint64_t layer_ram = (uint64_t)std::min(
        c.n_experts, std::max(0, ram_mgr.cap_for_layer(layer)));
    ram_slots += layer_ram;
    add_bytes(total_bytes, (uint64_t)c.n_experts, (uint64_t)bytes);
    add_bytes(ram_bytes, layer_ram, (uint64_t)bytes);
  }
  if (total_slots == 0)
    return {};

  uint64_t vram_slots = 0;
  uint64_t vram_bytes = 0;
  for (const auto &slot : ram_mgr.vram_cache->expert_vram_cache) {
    if (slot.pool_layer < 0 || slot.pool_layer >= cache_layers || slot.size == 0)
      continue;
    ++vram_slots;
    if (vram_bytes > std::numeric_limits<uint64_t>::max() - slot.size)
      vram_bytes = std::numeric_limits<uint64_t>::max();
    else
      vram_bytes += (uint64_t)slot.size;
  }
  // Uniform (non-compact) pools do not assign a fixed pool_layer until a
  // slot is populated. Their physical capacity is still exact at this point.
  if (vram_slots == 0 && ram_mgr.vram_cache->expert_vram_cache_cap > 0) {
    vram_slots = (uint64_t)ram_mgr.vram_cache->expert_vram_cache_cap;
    add_bytes(vram_bytes, vram_slots, (uint64_t)expert_slab_bytes);
  }

  const uint64_t max_unique_slots =
      std::min(total_slots, ram_slots + vram_slots);
  const char *placement =
      vram_slots >= total_slots
          ? "all-vram"
          : (vram_slots > 0 && ram_slots > 0
                 ? "mixed-vram-ddr"
                 : (ram_slots > 0 ? "ddr-only" : "storage-backed"));
  const double gib = 1024.0 * 1024.0 * 1024.0;
  engine.moe_placement = {
      .placement = vram_slots >= total_slots
                       ? zeromoe::graph::ExpertPlacementClass::AllVram
                       : (vram_slots > 0 && ram_slots > 0
                              ? zeromoe::graph::ExpertPlacementClass::MixedVramDdr
                              : (ram_slots > 0
                                     ? zeromoe::graph::ExpertPlacementClass::DdrOnly
                                     : zeromoe::graph::ExpertPlacementClass::StorageBacked)),
      .model_slots = total_slots,
      .ddr_slots = ram_slots,
      .vram_slots = vram_slots,
      .model_bytes = total_bytes,
      .ddr_bytes = ram_bytes,
      .vram_bytes = vram_bytes,
      .complete_if_disjoint = max_unique_slots == total_slots,
  };
  const FullGraphModel *graph_model = engine.family_ops.graph_model;
  const auto capabilities =
      graph_model ? graph_model->moe_execution_capabilities()
                  : zeromoe::graph::MoEExecutionCapabilities{};
  const auto prefill_policy = zeromoe::graph::resolve_execution_policy(
      engine.moe_placement, capabilities, ModelGraphPhase::Prefill);
  const auto decode_policy = zeromoe::graph::resolve_execution_policy(
      engine.moe_placement, capabilities, ModelGraphPhase::Decode);
  engine.moe_prefill_execution = prefill_policy.moe;
  engine.moe_decode_execution = decode_policy.moe;
  LOG_STREAM_INFO()
      << "[PLACEMENT-PREFLIGHT] class=" << placement
      << " model_experts=" << total_slots << " across " << moe_layers
      << " layer(s), model_bytes=" << (double)total_bytes / gib << " GiB; "
      << "VRAM=" << vram_slots << " slots/" << (double)vram_bytes / gib
      << " GiB; DDR=" << ram_slots << " slots/" << (double)ram_bytes / gib
      << " GiB; maximum disjoint coverage=" << max_unique_slots << "/"
      << total_slots << " ("
      << (100.0 * (double)max_unique_slots / (double)total_slots) << "%)\n";
  LOG_STREAM_INFO() << "[EXECUTION-POLICY] prefill="
                    << prefill_policy.reason << "; decode="
                    << decode_policy.reason << "\n";

  const bool mixed = vram_slots > 0 && ram_slots > 0 &&
                     vram_slots < total_slots;
  const bool early_ready_capable =
      graph_mode && graph_model && graph_model->supports_cpu_lane();
  if (mixed && graph_mode && !early_ready_capable) {
    LOG_STREAM_WARN()
        << "[PLACEMENT-WARN] mixed VRAM+DDR experts require concurrent hot-GPU "
           "and cold-CPU/offload execution, but graph model '"
        << (graph_model ? graph_model->name() : "none")
        << "' does not advertise a common early-ready CPU lane. Decode will "
           "serialize branches or use synchronous callbacks; performance may "
           "be far below the model-size target.\n";
  }
  if (max_unique_slots < total_slots) {
    LOG_STREAM_WARN()
        << "[PLACEMENT-WARN] DDR+VRAM can cover at most " << max_unique_slots
        << "/" << total_slots
        << " expert identities even with disjoint tiers; decode-time storage "
           "reads are unavoidable unless budgets or placement change.\n";
  } else if (mixed && !config.pin_global_tiers) {
    LOG_STREAM_WARN()
        << "[PLACEMENT-WARN] DDR+VRAM can cover the model only if their hot "
           "sets are substantially disjoint, but pin_global_tiers=false. "
           "Measure [TIER-OVERLAP]; duplicated identities can turn a nominally "
           "resident model into decode-time storage misses.\n";
  }
  if (ram_slots > 0 && !config.ram_preload_all_experts &&
      config.pin_ram_fraction <= 0.0) {
    LOG_STREAM_WARN()
        << "[PLACEMENT-WARN] the DDR tier has capacity for " << ram_slots
        << " experts, but ram_preload_all_experts=false and "
           "pin_ram_fraction=0; it starts empty and the first requests will "
           "populate it through demand reads.\n";
  }
  if (vram_slots > 0 &&
      (config.pinned_vram_capacity <= 0 ||
       (uint64_t)config.pinned_vram_capacity >= vram_slots)) {
    LOG_STREAM_WARN()
        << "[PLACEMENT-WARN] pinned_vram_capacity="
        << config.pinned_vram_capacity << " for " << vram_slots
        << " physical slots leaves no guaranteed reactive VRAM headroom. "
           "Reserve unpinned slots or use a separate bounded assist ring.\n";
  }
  return {};
}

std::expected<void, std::string> EngineRunner::init_ram_tier() {
  auto &cfg = engine.model_file.architecture_config;
  // prefetcher.start() moved to AFTER the [PIN] preload block below (see
  // that block's own closing comment for why) -- the worker threads it
  // spawns now claim/populate/publish directly into engine.layers_experts
  // via RAMSafetyManager::claim_or_wait, and [PIN]'s Tier-3 preload writes
  // into that exact same structure; starting the threads only once that
  // preload has fully finished removes any need to reason about ordering
  // between them.
  //
  // Decode thread's OWN io_uring ring for RAMSafetyManager::get_experts_batch's
  // Phase B (the synchronous miss path -- see ram_manager.hpp's
  // decode_uring_ring comment for why this is a separate scope from the
  // prefetcher's rings). Unaffected by the prefetcher.start() move -- [PIN]
  // preload never uses this ring, only engine.load_expert_slab's plain
  // synchronous pread.
  ram_mgr.init_decode_uring(config.io_uring, config.io_uring_workers);

  // extended to a graduated 3-tier scheme (was:
  // a single top-K-pinned VRAM tier only). Read the usage file written
  // by a PREVIOUS run's dump_usage() at shutdown -- this already records
  // EVERY expert ever routed to (StreamingEngine::route_freq is a full
  // n_layers*n_experts histogram, not just a top-K sample), so all the
  // ranking data below has been available all along; what's new is
  // USING more of it. Per layer, ranked by historical selection count:
  //   ranks [0, pin_topk_per_layer)                              -> VRAM,
  //   PINNED ranks [pin_topk_per_layer, +pin_vram_lru_per_layer)         ->
  //   VRAM, unpinned (reactive LRU) ranks [..+pin_vram_lru_per_layer,
  //   +pin_ram_per_layer)       -> host RAM tier
  // First-ever run (no usage file yet) is a silent no-op for all three
  // tiers -- both caches fill reactively exactly as before this feature
  // existed. One-time startup cost (real disk reads), not a per-token one.
  //
  // ram_mgr.slots_per_layer is needed to safely bound the RAM tier
  // preload (it's the real per-layer RAM cache cap get_expert/
  // get_experts_batch scan/evict over -- writing into a slot index at or
  // beyond it would be invisible to every lookup until some UNRELATED
  // later request happens to raise the cap, which never happens once
  // set). Normally only computed per-request (generate_text calls this
  // with the real prompt's max_ctx), so call it once here too with the
  // configured context_size as a representative estimate purely for
  // preload sizing -- idempotent, generate_text's own later call simply
  // recomputes it from the real request.
  if (ram_mgr.vram_cache)
    ram_mgr.calculate_safe_cache_capacity(engine, config.context_size);

  // ram_cache_layer_adaptive: reshapes the uniform slots_per_layer budget
  // calculate_safe_cache_capacity just computed into a per-layer
  // distribution driven by real historical need (see apply_layer_
  // adaptive_capacity's own comment) -- must run AFTER calculate_safe_
  // cache_capacity (needs its slots_per_layer as the total budget to
  // redistribute) and BEFORE the [PIN] block below (Tier 3's ram_cap
  // bound needs the final per-layer caps, via ram_mgr.cap_for_layer).
  // No-op if disabled, or if there's no usage history yet -- see that
  // function's own "first-ever run is identical to disabled" guarantee.
  if (config.ram_cache_layer_adaptive) {
    ram_mgr.apply_layer_adaptive_capacity(config.pin_file, cache_layers,
                                          cfg.n_experts);
  }

  // Prefill sliding-window RAM capacity (2026-08-30, P1+P2) -- must come
  // AFTER both capacity steps above, since the window is a third override
  // tier layered on top of whatever base cap they settled (uniform, or
  // adaptive per-layer). No-op when disabled: sizes the override vector
  // all-zero, which cap_for_layer falls straight through.
  ram_mgr.init_prefill_window(engine, config.prefill_window_cap,
                              config.prefill_window_extra_gb);

  // Optional graph/RAM handoff. A stock GGML graph cannot safely refer to
  // an LRU slot whose slab may be evicted, so only layers whose complete
  // expert set fits the already-calculated RAM capacity are exposed as
  // fixed-stride graph tensors. The remaining layers use the graph's
  // source-backed fallback and retain normal lazy cache behavior.
  if (ram_graph_cache) {
    const char *layer0_diag_env =
        zeromoe::graphcfg::env("ZEROMOE_QWEN4EXP_DIAG_LAYER0_RESIDUAL");
    const bool narrow_layer0_diag =
        layer0_diag_env && layer0_diag_env[0] == '1';
    std::vector<int> graph_ram_full_layers;
    std::vector<int> graph_ram_projection_compact_layers;
    const int n_experts = engine.model_file.architecture_config.n_experts;
    for (int layer = 0; layer < (int)engine.layers_experts.size(); ++layer) {
      if (engine.qwen_ram_vram_only_layer(layer))
        continue;
      const bool qwen_cpu_layer =
          qwen_cpu_ram_cache &&
          (is_cpu_affinity_layer(layer) ||
           engine.qwen4exp_cpu_offload_layer_enabled(layer));
      // A full-graph RAM cache is intentionally complete-model wide,
      // but the Qwen CPU-affinity path only needs fixed-stride arenas
      // for its CPU layers.  `graph_ram_cache` is also true for the
      // Qwen CPU mode, so testing it directly here used to allocate
      // graph arenas for every layer.  That was wasteful and made the
      // source ownership harder to audit; non-CPU Qwen layers already
      // use the ordinary RAM cache.  In either case the bytes are
      // populated by the preload block below, never by GGUF mmap at
      // decode time.
      const bool full_graph_ram_layer =
          graph_mode && graph_ram_cache && !narrow_layer0_diag;
      const bool narrow_layer0_ram_layer =
          narrow_layer0_diag && layer == 0 &&
          engine.qwen4exp_cpu_offload_layer_enabled(layer);
      if (!full_graph_ram_layer && !qwen_cpu_layer &&
          !narrow_layer0_ram_layer)
        continue;
      if (n_experts <= 0 || engine.compute_expert_slab_bytes(layer) == 0 ||
          (int)engine.layers_experts[(size_t)layer].size() < n_experts ||
          ram_mgr.cap_for_layer(layer) < n_experts)
        continue;
      graph_ram_full_layers.push_back(layer);
      // Qwen's full graph can consume separate compact gate/up/down views on
      // CPU. GPU layers keep their fused gate/up graph by gathering the two
      // planes directly into the final VRAM tensor, so the arena itself can
      // use the same no-duplicate layout for every Qwen graph layer. Keep an
      // explicit opt-out for A/B comparisons; it changes only the DDR arena
      // layout, not layer placement or graph topology.
      const char *compact_env = zeromoe::graphcfg::env(
          "ZEROMOE_QWEN4EXP_GRAPH_PROJECTION_COMPACT");
      const bool projection_compact_enabled =
          !(compact_env && compact_env[0] == '0');
      const char *host_node_env =
          zeromoe::graphcfg::env("ZEROMOE_QWEN4EXP_HOST_NODE");
      const bool host_node_enabled = host_node_env && host_node_env[0] == '1';
      // The legacy scheduler CPUOffload path still uses the single-stride
      // runner, so retain its interleaved arena. The opt-in host-node path
      // now supports independent projection strides and can use the compact
      // planes without creating a second copy.
      if (engine.family_ops.family == ModelFamily::Qwen4Exp &&
          projection_compact_enabled &&
          (!engine.qwen4exp_cpu_offload_layer_enabled(layer) ||
           host_node_enabled))
        graph_ram_projection_compact_layers.push_back(layer);
    }
    if (!engine.prepare_graph_ram_arenas(
            graph_ram_full_layers, graph_ram_projection_compact_layers)) {
      LOG_STREAM_WARN()
          << "[GRAPH-RAM] no complete layer arena was admitted; "
             "full graph will use its bounded source-backed path\n";
    }
  }

  // prefill->decode retention handoff (2026-08-31). 0 = off (default),
  // in which case eviction_score is byte-identical to its prior form.
  RAMSafetyManager::set_prefill_retention_weight(
      (uint64_t)std::max(0, config.prefill_retention_weight));
  RAMSafetyManager::set_prefill_retention_shape(
      std::max(0, config.prefill_retention_top_n),
      std::max(1, config.prefill_retention_min_demand),
      config.prefill_retention_flat_cut, config.prefill_retention_peaky_cut);

  // RAM-tier load grace period (2026-09-04). 0 tokens = off (default), in
  // which case eviction_score is byte-identical to its prior form.
  RAMSafetyManager::set_ram_grace_period(
      (uint64_t)std::max(0, config.ram_grace_period_tokens),
      (uint64_t)std::max(0, config.ram_grace_period_bonus));


  // vram_evict_demote_to_ram: wired here (after calculate_safe_cache_capacity
  // has set a real ram_mgr.slots_per_layer, and before anything -- including
  // the [PIN] preload block right below -- can trigger a real VRAM eviction)
  // so a demote never runs against an uninitialized RAM-tier cap. See
  // gpu_topology.hpp's on_vram_evict and ram_manager.hpp's demote_from_vram
  // for why this is safe to call from the decode thread only (upload_expert),
  // never the prefetcher thread (upload_expert_async, deliberately left
  // unwired). Captures engine/ram_mgr by reference -- both outlive the
  // GpuTopology instance this callback is attached to (same function scope,
  // torn down together at the end of run_engine).
  if (config.vram_evict_demote_to_ram && ram_mgr.vram_cache) {
    ram_mgr.vram_cache->on_vram_evict =
        [this](int layer, int expert_id, const uint8_t *device_ptr,
                            size_t bytes, int gpu_id) {
          ram_mgr.demote_from_vram(engine, layer, expert_id, device_ptr, bytes,
                                   gpu_id);
        };
  }

  return {};
}
