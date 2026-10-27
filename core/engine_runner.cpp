// EngineRunner phases: apply_engine_config, init_host_and_gpu.
// Split from the former run_engine() in src/main.cpp; bodies are unchanged except where
// a former local became an EngineRunner member of the same name.
#include "engine_runner.hpp"
#include "graph_config.hpp"
#include "graph_settings_global.hpp"
#include <cstdlib>

std::expected<void, std::string> EngineRunner::apply_graph_tuning() {
  const FullGraphModel *model = engine.family_ops.graph_model;
  const bool cpu_only = config.cpu_only || zeromoe::graphcfg::env("ZEROMOE_CPU_ONLY");
  const bool cuda_moe = !cpu_only && !gpu.topology.devices.empty() &&
                        engine.model_file.architecture_config.n_experts > 0;
  engine.cuda_moe_graph_policy_applies = cuda_moe;
  if (!model) {
    if (cuda_moe) {
      const auto arch_it = engine.model_file.kv_string.find("general.architecture");
      const std::string architecture =
          arch_it == engine.model_file.kv_string.end() ? "unknown"
                                                       : arch_it->second;
      return std::unexpected(
          "CUDA MoE architecture '" + architecture +
          "' has no registered FullGraphModel adapter; layered CUDA decode is not a supported onboarding path");
    }
    return {};
  }
  const GraphRuntimeTuning tuning = model->runtime_tuning(engine);
  if (tuning.requires_preloaded_ddr_experts &&
      !config.ram_preload_all_experts) {
    if (config.ram_preload_all_experts_explicit) {
      return std::unexpected(
          std::string(model->name()) +
          " graph decode requires ram_preload_all_experts=true; the explicit false value cannot seed its DDR expert arenas");
    }
    config.ram_preload_all_experts = true;
    engine.qwen_ram_preload_all_experts = true;
    LOG_INFO("[GRAPH-TUNING] %s enabled complete DDR expert preload because its decode graph requires host expert arenas\n",
             model->name());
  }
  if (tuning.default_ram_pin_fraction >= 0.0 &&
      !config.pin_ram_fraction_explicit) {
    config.pin_ram_fraction = tuning.default_ram_pin_fraction;
    LOG_INFO("[GRAPH-TUNING] %s RAM pin fraction defaulted to %.0f%%\n",
             model->name(), 100.0 * config.pin_ram_fraction);
  } else if (tuning.default_ram_pin_fraction > 0.0 &&
             config.pin_ram_fraction < 0.5) {
    LOG_WARN("[GRAPH-TUNING] %s explicitly leaves most DDR expert arenas pageable (pin_ram_fraction=%.2f); expert transfer throughput can be much lower\n",
             model->name(), config.pin_ram_fraction);
  }
  if (tuning.default_ram_pinned_budget_gb >= 0.0 &&
      !config.ram_pinned_budget_gb_explicit) {
    config.ram_pinned_budget_gb = tuning.default_ram_pinned_budget_gb;
    LOG_INFO("[GRAPH-TUNING] %s RAM pin budget defaulted to %.1f GiB\n",
             model->name(), config.ram_pinned_budget_gb);
  } else if (tuning.default_ram_pinned_budget_gb > 0.0 &&
             config.ram_pinned_budget_gb <= 0.0) {
    LOG_WARN("[GRAPH-TUNING] %s has no RAM pin budget; pin_ram_fraction cannot provide pinned expert arenas without ram_pinned_budget_gb\n",
             model->name());
  }
  if (model->graph_only() && model->device_expert_layers(engine) <= 0) {
    LOG_WARN("[GRAPH-TUNING] %s has no VRAM expert layers; all routed experts use DDR execution/transfer\n",
             model->name());
  }
  engine.ram_pinned_budget_bytes = static_cast<size_t>(
      config.ram_pinned_budget_gb * 1024.0 * 1024.0 * 1024.0);
  return {};
}

std::expected<void, std::string>
EngineRunner::resolve_cuda_moe_graph_policy() {
  const bool cpu_only = config.cpu_only || zeromoe::graphcfg::env("ZEROMOE_CPU_ONLY");
  const bool cuda_moe = !cpu_only && !gpu.topology.devices.empty() &&
                        engine.model_file.architecture_config.n_experts > 0;
  engine.cuda_moe_graph_policy_applies = cuda_moe;
  const FullGraphModel *model = engine.family_ops.graph_model;
  if (!model) {
    if (!cuda_moe)
      return {};
    const auto arch_it = engine.model_file.kv_string.find("general.architecture");
    const std::string architecture =
        arch_it == engine.model_file.kv_string.end() ? "unknown"
                                                     : arch_it->second;
    return std::unexpected(
        "CUDA MoE architecture '" + architecture +
        "' has no registered FullGraphModel adapter; layered CUDA decode is not a supported onboarding path");
  }
  if (!cuda_moe)
    return {};

  const GraphMemoryPlan plan = model->plan_decode_memory(engine);
  switch (plan.fit) {
  case GraphMemoryFit::Fits:
    engine.cuda_moe_graph_memory_plan = "fits";
    break;
  case GraphMemoryFit::Insufficient:
    engine.cuda_moe_graph_memory_plan = "insufficient";
    break;
  case GraphMemoryFit::Unsupported:
    engine.cuda_moe_graph_memory_plan = "unsupported";
    if (plan.reason.empty())
      return std::unexpected(std::string(model->name()) +
                             " planner rejected the model topology without a reason");
    return std::unexpected(std::string(model->name()) +
                           " does not support this model topology: " +
                           plan.reason);
  case GraphMemoryFit::Unknown:
    engine.cuda_moe_graph_memory_plan = "unknown";
    break;
  }
  if (plan.device_expert_layers >= 0) {
    engine.cuda_moe_graph_device_expert_layers = plan.device_expert_layers;
    engine.cuda_moe_graph_device_expert_trailing =
        plan.device_expert_trailing;
  }

  if (plan.fit == GraphMemoryFit::Insufficient) {
    if (plan.reason.empty() || plan.required_device_bytes == 0 ||
        plan.available_device_bytes == 0 ||
        plan.required_device_bytes <= plan.available_device_bytes) {
      return std::unexpected(
          std::string(model->name()) +
          " planner reported insufficient graph memory without a concrete required>available byte plan and reason; refusing layered fallback");
    }
    if (model->graph_only()) {
      return std::unexpected(
          std::string(model->name()) + " is graph-only, but its memory planner reports insufficient capacity: " +
          plan.reason);
    }
    engine.cuda_moe_layered_fallback_reason = plan.reason;
    LOG_WARN("[MOE-GRAPH-POLICY] family=%s policy=memory-fallback memory_plan=insufficient required_vram_bytes=%zu available_vram_bytes=%zu reason=%s\n",
             model->name(), plan.required_device_bytes,
             plan.available_device_bytes, plan.reason.c_str());
    return {};
  }

  if (plan.fit == GraphMemoryFit::Fits &&
      (plan.required_device_bytes == 0 || plan.available_device_bytes == 0 ||
       plan.required_device_bytes > plan.available_device_bytes)) {
    return std::unexpected(
        std::string(model->name()) +
        " planner reported graph memory fits without a valid required<=available byte plan");
  }

  // Unknown is fail-closed. A family may authorize layered execution only
  // when its graph memory planner explicitly reports insufficient capacity.
  engine.cuda_moe_full_graph_decode_required = true;
  if (!model->enabled(engine, ModelGraphPhase::Decode)) {
    return std::unexpected(
        std::string(model->name()) +
        " is a CUDA MoE family, but graph decode is disabled and no memory planner authorized layered fallback");
  }
  LOG_INFO("[MOE-GRAPH-POLICY] family=%s policy=required memory_plan=%s required_vram_bytes=%zu available_vram_bytes=%zu\n",
           model->name(), engine.cuda_moe_graph_memory_plan.c_str(),
           plan.required_device_bytes, plan.available_device_bytes);
  return {};
}

std::expected<void, std::string> EngineRunner::run() {
  if (auto result = apply_engine_config(); !result)
    return result;
  if (auto result = init_host_and_gpu(); !result)
    return result;
  if (auto result = load_model(); !result)
    return result;
  if (auto result = resolve_cuda_moe_graph_policy(); !result)
    return result;
  if (auto result = apply_graph_tuning(); !result)
    return result;
  if (auto result = resolve_cuda_moe_graph_policy(); !result)
    return result;
  if (auto result = plan_expert_cache(); !result)
    return result;
  if (auto result = setup_vram_expert_cache(); !result)
    return result;
  if (auto result = offload_dense_tensors(); !result)
    return result;
  if (auto result = init_ram_tier(); !result)
    return result;
  if (auto result = validate_expert_placement(); !result)
    return result;
  if (auto result = preload_experts(); !result)
    return result;
  if (auto result = start_workers(); !result)
    return result;
  auto serve_result = config.serve_api ? run_api() : run_repl();
  if (!serve_result)
    request_failure = serve_result.error();
  if (auto result = shutdown_and_report(); !result)
    return result;
  return {};
}

std::expected<void, std::string> EngineRunner::apply_engine_config() {
  if (!config.graph_settings.empty()) {
#if !defined(ZEROMOE_USE_GGML_CUDA)
    return std::unexpected(
        "graph_* settings require a CUDA build (this binary is CPU-only)");
#endif
    if (config.cpu_only || zeromoe::graphcfg::env("ZEROMOE_CPU_ONLY"))
      return std::unexpected("graph_* settings require CUDA graph execution, not CPU-only mode");
  }
  try {
    zeromoe::graphsettings::configure_settings(config.graph_settings);
  } catch (const std::exception &e) {
    return std::unexpected(e.what());
  }
  engine.drop_expert_pages = config.drop_pages;
  engine.unpacked_expert_format = config.unpacked_expert_format;
  engine.prefill_warm_via_prefetcher = config.prefill_warm_via_prefetcher;
  engine.qwen_ram_preload_all_experts = config.ram_preload_all_experts;
  engine.qwen4exp_graph_expert_vram_budget_gb =
      config.qwen4exp_graph_expert_vram_budget_gb;
  engine.qwen4exp_graph_pin_file = config.pin_file;
  engine.exl3_ggml_bridge_model_path = config.exl3_ggml_bridge_model_path;
  engine.exl3_ggml_bridge_layer = config.exl3_ggml_bridge_layer;
  engine.exl3_ggml_bridge_layers = config.exl3_ggml_bridge_layers;
  engine.exl3_ggml_graph_capture = config.exl3_ggml_graph_capture;
  engine.exl3_ggml_routed_graph_capture =
      config.exl3_ggml_routed_graph_capture;
  engine.qwen4exp_option_b_layer = config.qwen4exp_option_b_layer;
  engine.qwen4exp_option_b_top_n = config.qwen4exp_option_b_top_n;
  engine.qwen4exp_option_b_min_coverage =
      config.qwen4exp_option_b_min_coverage;
  engine.qwen4exp_option_b_right_size_top_n =
      config.qwen4exp_option_b_right_size_top_n;
  engine.qwen4exp_option_b_global_top_k =
      config.qwen4exp_option_b_global_top_k;
  engine.prefill_batch_tokens = config.prefill_batch_tokens;
  engine.qwen4exp_option_b_global_floor_per_layer =
      config.qwen4exp_option_b_global_floor_per_layer;
  engine.qwen4exp_option_b_max_layers = config.qwen4exp_option_b_max_layers;
  engine.qwen4exp_option_b_fill_vram = config.qwen4exp_option_b_fill_vram;
  engine.qwen4exp_option_b_concentrated_only =
      config.qwen4exp_option_b_concentrated_only;
  engine.qwen4exp_option_b_stage_expert =
      config.qwen4exp_option_b_stage_expert;
  engine.qwen4exp_option_b_stage_count =
      config.qwen4exp_option_b_stage_count;
  qwen4exp_full_graph_configure(config.qwen4exp_full_graph,
                                config.qwen4exp_full_graph_decode,
                                config.qwen4exp_prefill_state_handoff);
  ram_mgr.configured_total_experts =
      std::max(0, config.ram_cache_total_experts);
  // Must be set BEFORE the [PIN] preload block below -- that's where the
  // bulk of the RAM tier's slabs are first allocated, and each slab is
  // pinned exactly once at its own allocation (ensure_slab_capacity).
  // Setting it later would silently leave every preloaded slab pageable.
  engine.ram_pinned_budget_bytes =
      (size_t)(config.ram_pinned_budget_gb * 1024.0 * 1024.0 * 1024.0);
  if (config.ram_pinned_budget_gb > 0.0) {
    LOG_STREAM_INFO()
        << "[CONFIG] RAM slab persistent pin budget: "
        << config.ram_pinned_budget_gb
        << " GB of host_ram_gb=" << config.host_ram_gb
        << " -- slabs are cudaHostRegister'd once at "
           "allocation so upload_expert_active_async's DMA can overlap compute "
           "(see ram_pinned_budget_gb's own comment). Pinned pages are never "
           "reclaimable by the kernel; raise this deliberately, not "
           "wholesale.\n";
    LOG_STREAM_INFO()
        << "[CONFIG] RAM pin eager pre-registration (extends Tier 3's own "
           "ranking-informed preload further down its real candidate list when "
           "budget remains, single-threaded, before any request is timed -- "
           "see "
           "ram_pin_eager_preregister's own comment): "
        << (config.ram_pin_eager_preregister ? "ON" : "off") << "\n";
    LOG_STREAM_INFO()
        << "[CONFIG] RAM preload hit_count seeding (Tier 3 slots start "
           "'known warm' from their real pin_file count, clamped below "
           "promote_hit_threshold -- see ram_preload_seed_hit_count's own "
           "comment): "
        << (config.ram_preload_seed_hit_count ? "ON" : "off") << "\n";
  }
  engine.prefill_opportunistic_promote = config.prefill_opportunistic_promote;
  engine.decode_async_prewarm = config.decode_async_prewarm;
  engine.decode_async_prewarm_max_batch = config.decode_async_prewarm_max_batch;
  engine.decode_async_prewarm_nonblocking =
      config.decode_async_prewarm_nonblocking;
  engine.cpu_offload_max_experts = config.cpu_offload_max_experts;
  engine.prefill_retention_decay = config.prefill_retention_decay;
  engine.prefill_dispatch_lookahead_tokens =
      config.prefill_dispatch_lookahead_tokens;
  engine.prefill_hint_order_by_earliest_token =
      config.prefill_hint_order_by_earliest_token;
  engine.pilot_widen_all_tokens = config.pilot_widen_all_tokens;
  engine.prefetch_widen_all_tokens = config.prefetch_widen_all_tokens;
  engine.attn_qabsorb_gpu = config.attn_qabsorb_gpu;
  engine.attn_qabsorb_gpu_max_vram_gb = config.attn_qabsorb_gpu_max_vram_gb;
  engine.attn_score_gpu = config.attn_score_gpu;
  engine.qwen4exp_gdn_conv_gpu = config.qwen4exp_gdn_conv_gpu;
  engine.qwen4exp_router_pilot = config.qwen4exp_router_pilot;
  engine.qwen4exp_router_pilot_k = config.qwen4exp_router_pilot_k;
  engine.qwen4exp_router_pilot_prefetch = config.qwen4exp_router_pilot_prefetch;
  engine.qwen4exp_router_pilot_prefetch_k =
      config.qwen4exp_router_pilot_prefetch_k;
  engine.qwen4exp_router_pilot_prefetch_exclude_layers =
      config.qwen4exp_router_pilot_prefetch_exclude_layers;
  engine.qwen4exp_router_pilot_predict_layers =
      config.qwen4exp_router_pilot_predict_layers;
  engine.qwen4exp_router_pilot_skip_fully_cached =
      config.qwen4exp_router_pilot_skip_fully_cached;
  engine.qwen4exp_router_pilot_skip_prefetch_excluded_layers =
      config.qwen4exp_router_pilot_skip_prefetch_excluded_layers;
  engine.qwen4exp_decode_ggml_moe = config.qwen4exp_decode_ggml_moe;
  engine.qwen4exp_decode_scheduler = config.qwen4exp_decode_scheduler;
  engine.qwen4exp_decode_device_hc = config.qwen4exp_decode_device_hc;
  engine.qwen4exp_decode_batch_stage = config.qwen4exp_decode_batch_stage;
  engine.qwen4exp_decode_custom_hc_q8 = config.qwen4exp_decode_custom_hc_q8;
  engine.qwen4exp_decode_gpu_router = config.qwen4exp_decode_gpu_router;
  engine.qwen4exp_cpu_ggml_moe = config.qwen4exp_cpu_ggml_moe;
  engine.qwen4exp_cpu_strided_grouped =
      config.qwen4exp_cpu_strided_grouped;
  engine.qwen4exp_cpu_fused_grouped = config.qwen4exp_cpu_fused_grouped;
  engine.qwen4exp_cpu_gpu_overlap = config.qwen4exp_cpu_gpu_overlap;
  engine.qwen4exp_cpu_gpu_overlap_max_layers =
      config.qwen4exp_cpu_gpu_overlap_max_layers;
  engine.qwen4exp_cpu_shared_device = config.qwen4exp_cpu_shared_device;
  engine.qwen4exp_cpu_offload = config.qwen4exp_cpu_offload;
  engine.qwen4exp_cpu_offload_layers = config.qwen4exp_cpu_offload_layers;
  std::sort(engine.qwen4exp_cpu_offload_layers.begin(),
            engine.qwen4exp_cpu_offload_layers.end());
  engine.qwen4exp_cpu_offload_layers.erase(
      std::unique(engine.qwen4exp_cpu_offload_layers.begin(),
                  engine.qwen4exp_cpu_offload_layers.end()),
      engine.qwen4exp_cpu_offload_layers.end());
  engine.qwen4exp_cpu_offload_force_cold_count =
      config.qwen4exp_cpu_offload_force_cold_count;
  engine.qwen4exp_cpu_offload_force_expert =
      config.qwen4exp_cpu_offload_force_expert;
  engine.qwen4exp_cpu_offload_event_fence =
      config.qwen4exp_cpu_offload_event_fence;
  engine.qwen4exp_cpu_offload_ring_slots =
      config.qwen4exp_cpu_offload_ring_slots;
  engine.qwen4exp_gpu_stage_enabled = config.qwen4exp_gpu_stage_enabled;
  engine.qwen4exp_gpu_stage_layers = config.qwen4exp_gpu_stage_layers;
  std::sort(engine.qwen4exp_gpu_stage_layers.begin(),
            engine.qwen4exp_gpu_stage_layers.end());
  engine.qwen4exp_gpu_stage_layers.erase(
      std::unique(engine.qwen4exp_gpu_stage_layers.begin(),
                  engine.qwen4exp_gpu_stage_layers.end()),
      engine.qwen4exp_gpu_stage_layers.end());
  engine.qwen4exp_track_router_usage = config.qwen4exp_track_router_usage;
  {
    std::string mode = config.qwen4exp_expert_promotion;
    if (const char *env = std::getenv("ZEROMOE_QWEN4EXP_EXPERT_PROMOTION")) mode = env;
    engine.qwen4exp_expert_promotion = mode == "on" ? 2 : mode == "observe" ? 1 : 0;
  }
  engine.glm5next_kda_gpu = config.glm5next_kda_gpu;
  engine.glm5next_router_gpu = config.glm5next_router_gpu;
  engine.glm5next_moe_prefill_gpu = config.glm5next_moe_prefill_gpu;
  engine.glm5next_dense_prefill_fusion = config.glm5next_dense_prefill_fusion;
  engine.mtp_enabled = config.mtp_enabled;
  engine.mtp_graph_native = config.mtp_graph && config.mtp_enabled;
  engine.qwen4exp_wide_export = engine.mtp_graph_native;
  engine.mtp_draft_tokens = config.mtp_draft_tokens;
  engine.prompt_cache = config.prompt_cache;
  return {};
}

std::expected<void, std::string> EngineRunner::init_host_and_gpu() {
  static MemoryGuard g_mem_guard;
  if (config.mem_guard)
    g_mem_guard.start(config.mem_guard_warn_mb, config.mem_guard_stop_mb,
                      config.mem_guard_poll_ms);
  // Graph-owned expert arenas and the legacy RAM-tier slabs are allocated by
  // different subsystems.  Configure one process-wide admission ceiling so
  // a graph source allocation cannot silently sit outside host_ram_gb.  For
  // host_ram_gb=0, resolve the same 88%-of-currently-available policy used
  // by RAMSafetyManager; this is only an admission ceiling, while the
  // background watchdog still protects the live low-water mark.
  size_t host_budget_bytes = 0;
  if (config.host_ram_gb > 0.0) {
    const double bytes = config.host_ram_gb * 1.0e9;
    host_budget_bytes = bytes >= (double)SIZE_MAX ? SIZE_MAX : (size_t)bytes;
  } else {
    const long available_mb = MemoryGuard::read_available_mb();
    if (available_mb > 0)
      host_budget_bytes =
          (size_t)((double)available_mb * 1024.0 * 1024.0 * 0.88);
  }
  MemoryGuard::configure_host_budget(host_budget_bytes);
  LOG_STREAM_INFO()
      << "[MEM-GUARD] shared host admission budget: "
      << (host_budget_bytes > 0
              ? std::to_string((double)host_budget_bytes / 1.0e9) + " GB"
              : "unbounded (automatic budget unavailable)")
      << "\n";
  engine.attn_score_gpu_max_vram_gb = config.attn_score_gpu_max_vram_gb;
  engine.attn_score_gpu_max_context = config.attn_score_gpu_max_context;
  engine.dsv4_score_gpu = config.dsv4_score_gpu;
  engine.dsv4_score_gpu_max_vram_gb = config.dsv4_score_gpu_max_vram_gb;
  ram_mgr.configured_ram_gb = config.host_ram_gb;
  ram_mgr.promote_hit_threshold = (uint32_t)config.promote_hit_threshold;
  ram_mgr.reactive_promote_gpu1_every_n = config.reactive_promote_gpu1_every_n;
  ram_mgr.pin_threshold = (uint32_t)std::max(0, config.pin_threshold);
  ram_mgr.verbose_cache_log = config.verbose_moe_log;
  ram_mgr.expert_fetch_threads = std::max(1, config.expert_fetch_threads);
  ram_mgr.move_on_promotion = config.move_on_promotion;
  ram_mgr.ram_copy_reclaim_threshold = config.ram_copy_reclaim_threshold;
  ram_mgr.disk_batch_padding = config.disk_batch_padding;
  ram_mgr.disk_batch_padding_target = config.disk_batch_padding_target;
  ram_mgr.disk_batch_padding_evict_dup = config.disk_batch_padding_evict_dup;
  gpu.topology.active_layer_promote_threshold =
      std::max(1, config.active_layer_promote_hit_threshold);
  gpu.topology.decode_promote_all_misses = config.decode_promote_all_misses;
  gpu.topology.speculative_active_layer_fill =
      config.speculative_promote_to_active_layer;
  // The CUDA runtime reports all physical adapters, but the TOML's
  // [[gpus]] list is an explicit allocation allow-list. Without passing it
  // into the topology, a GPU omitted from a single-GPU profile was still
  // auto-assigned as an ExpertVault and received a large expert pool.
  gpu.topology.configured_device_ids.clear();
  for (const auto &gc : config.gpus)
    gpu.topology.configured_device_ids.push_back(gc.device_id);
  if (config.dense_gpu_idx >= 0) {
    gpu.device_id = config.gpus[config.dense_gpu_idx].device_id;
    // REAL BUG FIX: gpu.device_id above was being set correctly here,
    // then IMMEDIATELY overwritten by CudaManager::initialize() one
    // line into gpu.initialize() below, with whatever
    // GpuTopology::assign_roles()'s SM-architecture/VRAM auto-detect
    // heuristic picked -- assign_roles() took no parameters and had
    // no way to know a config preference even existed, so a user's
    // explicit role="dense" on the OLDER-architecture GPU was silently
    // discarded every time (confirmed on a real run: TOML said the
    // 3090 was dense/CoreEngine, but "[GPU] Hardware Profile" showed
    // the newer-arch 5060 Ti as CoreEngine instead). Plumbing the same
    // device_id through to forced_core_device_id lets assign_roles()
    // actually honor it.
    gpu.topology.forced_core_device_id =
        config.gpus[config.dense_gpu_idx].device_id;
  }
  if (!gpu.initialize()) {
    return std::unexpected("Failed to initialize GPU subsystem");
  }
  gpu.set_dense_async_copies(config.dense_async_copies);
  gpu.report_vram_usage("after gpu.initialize()");
  engine.gpu = &gpu; // give the forward path access to dense GPU dispatch
  // Captured separately (not read back via engine.gpu) so StreamingEngine's
  // destructor can free its own device-resident buffers (resident_layers'
  // device_ptr, d_Lc/d_Rc/d_Lc_q4/d_Rc_q4, d_hidden) correctly regardless
  // of whether `gpu` (declared before `engine` above) has already been
  // destroyed by the time engine's destructor runs.
  engine.gpu_device_id = gpu.get_core_device();
  return {};
}
