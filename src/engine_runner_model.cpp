// EngineRunner phases: load_model.
// Split from the former run_engine() in src/main.cpp; bodies are unchanged except where
// a former local became an EngineRunner member of the same name.
#include "engine_runner.hpp"

class GGUFGraphInitializer {
public:
  static bool build_model_graph(StreamingEngine &engine, int max_ctx);
};

std::expected<void, std::string> EngineRunner::load_model() {
  const std::string &model_path = config.model_path;
  int max_ctx = config.context_size;
#if defined(MOESTREAM_EXL3_CUDA) && defined(MOESTREAM_USE_GGML_CUDA)
  if (!engine.exl3_ggml_bridge_model_path.empty()) {
    const size_t cache_bytes = static_cast<size_t>(
        config.exl3_gpu_cache_gb * 1024.0 * 1024.0 * 1024.0);
    try {
    engine.exl3_gpu_executor_owner = std::make_shared<Exl3GpuExecutor>(
        engine.gpu_device_id, cache_bytes);
    engine.exl3_gpu_executor_owner->set_routed_graph_capture(
        config.exl3_ggml_routed_graph_capture);
    } catch (const std::exception &error) {
      return std::unexpected(std::string("Failed to initialize EXL3 GGML "
                                         "bridge cache: ") + error.what());
    }
    engine.exl3_gpu_executor = engine.exl3_gpu_executor_owner.get();
    if (config.exl3_cpu_expert_cache_gb > 0.0) {
      const size_t host_cache_bytes = static_cast<size_t>(
          config.exl3_cpu_expert_cache_gb * 1024.0 * 1024.0 * 1024.0);
      engine.exl3_cpu_matrix_cache =
          std::make_shared<Exl3CpuMatrixCache>(host_cache_bytes);
    }
    LOG_STREAM_INFO() << "[EXL3-GGML-BRIDGE] cache executor initialized on GPU "
                      << engine.gpu_device_id << " with "
                      << config.exl3_gpu_cache_gb << " GiB cache budget\n";
  }
#endif

  if (!engine.model_file.load_header(model_path)) {
    return std::unexpected(
        "Failed to parse GGUF file or map descriptor tree structure");
  }
  // Opt-in only (config.direct_io, default false) -- see GGUFFile::
  // enable_direct_io's own comment for the alignment check + per-shard
  // test-read verification this does before trusting O_DIRECT on any
  // shard. Called right after load_header (all shard fds + KV metadata
  // are populated by then) and before anything starts reading experts.
  if (config.direct_io)
    engine.model_file.enable_direct_io();
  // Opened AFTER enable_direct_io deliberately: open_mirror matches each
  // shard's O_DIRECT state against the primary's, which isn't decided until
  // the call above has run and verified it per shard.
  if (!config.mirror_model_path.empty())
    engine.model_file.open_mirror(config.mirror_model_path);

  // Opt-in, only reachable when the driver actually opened above (config.
  // cuda_cufile alone doesn't guarantee that -- see cufile_driver_guard).
  // EngineConfig::validate() already refused cuda_cufile=true without
  // direct_io=true, so direct_io has already run by this point whenever
  // this branch is live. engine.cuda_cufile_active (the flag execution_
  // layer.cpp's Phase 3 loop actually checks) is only set true if at
  // least one shard genuinely registered -- "requested" vs "actually
  // active", same distinction direct_io/io_uring already draw.
  if (config.cuda_cufile && cufile_driver_opened) {
    int n = engine.model_file.enable_cufile();
    engine.cuda_cufile_active = (n > 0);
  }

  // Optional separate dense GGUF: dense/preloaded tensors read from here,
  // routed experts still stream from model_path. Enables a
  // dense(IQ4_XS)+expert(IQ2_*) mix across two files.
  if (!config.dense_model_path.empty() &&
      config.dense_model_path != model_path) {
    if (!engine.dense_model_file.load_header(config.dense_model_path)) {
      return std::unexpected("Failed to parse dense GGUF (dense_model_path)");
    }
    engine.has_dense_model = true;
    if (config.direct_io)
      engine.dense_model_file.enable_direct_io();
  }

  // Optional standalone "shared" MTP/NextN checkpoint (Qwen4Exp convention
  // -- see EngineConfig::mtp_model_path's own comment). Loaded BEFORE
  // build_model_graph below so graph_initializer.cpp's own MTP validation
  // can see engine.has_mtp_model/mtp_model_file when deciding whether
  // Qwen4Exp MTP is actually available for this run.
  if (!config.mtp_model_path.empty()) {
    if (!engine.mtp_model_file.load_header(config.mtp_model_path)) {
      return std::unexpected("Failed to parse MTP GGUF (mtp_model_path)");
    }
    engine.has_mtp_model = true;
    if (config.direct_io)
      engine.mtp_model_file.enable_direct_io();
  }

  if (!tokenizer.load(engine.model_file)) {
    return std::unexpected(
        "Tokenizer vocabulary table completely missing from GGUF metadata");
  }

  if (!GGUFGraphInitializer::build_model_graph(engine, max_ctx)) {
    return std::unexpected(
        "Failure triggered during topological model graph initialization pass");
  }
  if (engine.family_ops.family == ModelFamily::DeepSeek_V4) {
    LOG_STREAM_WARN()
        << "[DEPRECATED] DeepSeek-V4 (GGUF arch \"deepseek4\") is no longer "
           "supported: its full-graph path was removed and only the slow "
           "layer-by-layer path remains, without maintenance. Use DeepSeek-V4.1 "
           "(arch \"deepseek41\").\n";
  }
  engine.qwen4exp_full_graph_required =
      engine.family_ops.family == ModelFamily::Qwen4Exp &&
      qwen4exp_full_graph_decode_enabled();
  if (engine.qwen4exp_full_graph_required &&
      !qwen4exp_full_graph_enabled()) {
    return std::unexpected(
        "Qwen4Exp full-graph decode is enabled but full-graph prefill is "
        "disabled; configure both graph phases or explicitly exempt this run");
  }
  if (engine.qwen4exp_full_graph_required && engine.mtp_enabled) {
    return std::unexpected(
        "Qwen4Exp MTP is not integrated with the full-graph-only trunk; "
        "disable MTP explicitly or use a graph-enabled MTP implementation");
  }
  if (engine.qwen4exp_full_graph_required) {
    LOG_INFO("[QWEN4EXP-GRAPH-POLICY] full-graph prefill/decode required; "
             "a graph build or replay failure aborts the request\n");
  }
  engine.init_route_freq();

  // Alignment diagnostic. Must come AFTER build_model_graph: it walks every
  // MoE layer, and architecture_config.n_layers is only known as of that
  // call (an earlier placement silently checked zero layers and printed
  // nothing at all). Direct-io/mirror setup has already run by here too, so
  // the report reflects the O_DIRECT state actually in force. Emits to
  // stderr as well as the log -- see the function's own comment for why.
  engine.warn_if_expert_tensors_unaligned();

  // Whole-shard mmap for the prefill expert-streaming path. Enabled here
  // rather than lazily so a failure is visible at startup and simply
  // leaves moe_prefill_stream inert (moe_forward_gguf checks
  // mmap_active()) instead of failing mid-prefill. Deliberately AFTER
  // enable_direct_io: the mapping uses its own non-O_DIRECT fd, and the
  // ordering makes that independence obvious at the call site.
  if (config.moe_prefill_stream) {
    if (!config.moe_prefill_batched && !config.glm5next_moe_prefill_gpu) {
      LOG_WARN("[CONFIG] moe_prefill_stream=true has no effect without "
               "moe_prefill_batched=true (it replaces how that path "
               "acquires weights) -- staying on the cached path.\n");
    } else if (!engine.model_file.enable_mmap()) {
      LOG_WARN("[CONFIG] moe_prefill_stream=true but the shard mmap could "
               "not be established -- staying on the cached path.\n");
    }
  }

  // cpu_affinity_layers (config): must be set before ANYTHING below can
  // dispatch an expert or preload VRAM -- expert_ffn_resolve_gpu_loc,
  // RAMSafetyManager's reactive-hit promotion, Phase 2's prefill warm-up,
  // and the [PIN] preload block right below all consult
  // is_cpu_affinity_layer() (dense_dispatch.hpp). architecture_config.
  // n_layers is only known as of build_model_graph above, so (like the
  // unpacked-format sanity check right below) this is the earliest point
  // it can be set.
  if (!config.cpu_affinity_layers.empty()) {
    set_cpu_affinity_layers(config.cpu_affinity_layers,
                            engine.model_file.architecture_config.n_layers);
  }
  // Same set-once-before-any-dispatch timing as cpu_affinity_layers above
  // -- execution_layer.cpp's pilot loop reads pilot_k_for_layer() on
  // every layer transition starting with the very first prefill token.
  set_prefetch_pilot_k(config.prefetch_pilot_k_per_layer,
                       config.prefetch_pilot_k);
  // prefetch_pilot_k_adaptive (2026-09-13) -- narrows the pilot_k_for_layer
  // ceiling above to each layer's own live most-confident ranks. See
  // PilotKAdaptiveConfig's own comment (dense_dispatch.hpp).
  set_pilot_k_adaptive_config(config.prefetch_pilot_k_adaptive,
                              config.prefetch_pilot_k_adaptive_threshold,
                              config.prefetch_pilot_k_adaptive_min_samples);
  // Same timing/reasoning as the pilot_k call directly above, for
  // EngineConfig::prefetch_pilot_depth / prefetch_pilot_k_depth2.
  set_prefetch_pilot_depth(config.prefetch_pilot_depth,
                           config.prefetch_pilot_k_depth2);
  set_attn_pilot_enabled(config.prefetch_attn_pilot,
                         config.prefetch_attn_pilot_k);
  set_attn_pilot_vram_promote_top_k(config.attn_pilot_vram_promote_top_k);
  set_attn_pilot_vram_promote_dynamic(config.attn_pilot_vram_promote_dynamic);
  // Same timing/reasoning again, for the pilot-driven RAM->VRAM
  // speculative promotion config (ram_vram_promote_top_k/adaptive) --
  // execution_layer.cpp's pilot loop reads this on the very same l+1
  // transition as pilot_k_for_layer above.
  set_ram_vram_promote_config(config.ram_vram_promote_top_k,
                              config.ram_vram_promote_adaptive,
                              config.ram_vram_promote_adaptive_threshold,
                              config.ram_vram_promote_min_samples);
  // Per-layer accuracy veto (2026-08-25) -- layered on top of whichever
  // mode set_ram_vram_promote_config/set_attn_pilot_vram_promote_top_k
  // just configured above, see LayerPromoteGateConfig's own comment
  // (dense_dispatch.hpp).
  set_layer_promote_gate_config(config.ram_vram_promote_layer_gate,
                                config.ram_vram_promote_layer_gate_threshold,
                                config.ram_vram_promote_layer_gate_min_samples,
                                config.ram_vram_promote_layer_gate_top_fraction);
  // Per-layer dynamic WIDTH (2026-08-26) -- graduates the veto above into
  // a smooth per-layer promotion width, see RamVramPromoteConfig's own
  // dynamic_* fields (dense_dispatch.hpp) and EngineConfig::ram_vram_
  // promote_dynamic_layer's own comment.
  set_ram_vram_promote_dynamic_config(
      config.ram_vram_promote_dynamic_layer,
      config.ram_vram_promote_dynamic_threshold_4,
      config.ram_vram_promote_dynamic_threshold_3,
      config.ram_vram_promote_dynamic_threshold_2,
      config.ram_vram_promote_dynamic_threshold_1,
      config.ram_vram_promote_dynamic_k_tier4,
      config.ram_vram_promote_dynamic_k_tier3,
      config.ram_vram_promote_dynamic_k_tier2,
      config.ram_vram_promote_dynamic_k_tier1,
      config.ram_vram_promote_dynamic_k_floor,
      config.ram_vram_promote_dynamic_cold_start_k,
      config.ram_vram_promote_dynamic_min_samples);
  set_gpu1_predict_promote_config(
      config.predict_gpu1_rank_start >= 0, config.predict_gpu1_rank_start,
      config.predict_gpu1_rank_end, config.predict_gpu1_cap_per_layer);
  // Redirect pilot promotion into the active-layer pool instead of the
  // main VRAM cache (2026-08-26) -- see EngineConfig::speculative_promote_
  // to_active_layer's own comment. The dense_dispatch.hpp-side flag
  // (read by execution_layer.cpp's call sites) and the GpuTopology-side
  // flag (gpu.topology.speculative_active_layer_fill, set above --
  // consulted by cleanup_active_layer_area) are deliberately two separate
  // settings from the SAME config value: the former decides which
  // function the pilot loop calls, the latter decides whether cleanup
  // protects a one-layer-ahead entry. Both must agree for the feature to
  // behave correctly, which is why both are set here from the same
  // config.speculative_promote_to_active_layer, not independently.
  set_speculative_promote_to_active_layer(
      config.speculative_promote_to_active_layer);
  set_speculative_active_layer_max_k(config.speculative_active_layer_max_k);
  set_speculative_active_layer_promote_on_confirm(
      config.speculative_active_layer_promote_on_confirm);

  // 2026-08-28: multi-GPU active-layer routing registry -- see
  // EngineConfig::active_layer_routing_policy's own comment for the
  // full design. Built from whichever [[gpus]] entries opted in via
  // active_layer_weight > 0; route_active_layer_gpu_impl/
  // active_layer_multi_gpu_enabled (dense_dispatch.hpp) both correctly
  // degrade to "no routing" when fewer than 2 entries qualify, so this
  // is safe to call unconditionally regardless of whether the feature
  // is actually configured.
  {
    std::vector<ActiveLayerRouteGpu> al_gpus;
    for (auto &gc : config.gpus) {
      if (gc.active_layer_weight > 0)
        al_gpus.push_back({gc.device_id, gc.active_layer_weight});
    }
    set_active_layer_routing_config(config.active_layer_routing_policy,
                                    al_gpus);
  }

  // unpacked_expert_format is an explicit performance contract (single
  // combined DMA per expert) -- validate it holds for the WHOLE model
  // now, before any real generation work (or any expert data read) ever
  // starts, rather than let it fail unpredictably mid-run or silently
  // degrade. architecture_config.n_layers/n_experts is only populated as
  // of build_model_graph above, so this is the earliest point the check
  // can run. See StreamingEngine::sanity_check_unpacked_format for what
  // gets validated and why a violation is fatal rather than a fallback.
  if (engine.unpacked_expert_format && !engine.sanity_check_unpacked_format()) {
    return std::unexpected(
        "unpacked_expert_format=true but the model failed the aligned-unpacked "
        "sanity check -- see the [FATAL] line(s) above for exactly which "
        "layer/expert failed and why. Refusing to start rather than silently "
        "fall back to the slow path this flag exists to avoid.");
  }
  // The other direction of the same mismatch: unpacked_expert_format is
  // OFF, but the model itself is genuinely in the per-expert layout --
  // a real, measured incident (see CLAUDE.md): this silently disables
  // BOTH io_uring Phase B batching and the VRAM-shadow bypass fast path
  // for the entire run with zero visible error, showing up only as 100%
  // "layer-not-packed" on the (easy-to-miss) Decode-thread io_uring
  // batches shutdown line and a disk= figure that exceeds its own
  // containing expert= wall-clock window. Warn-only, not a hard refusal
  // like the block above -- the run can still complete correctly here,
  // just slowly, so there's no contract violation to refuse over.
  if (!engine.unpacked_expert_format && engine.looks_like_unpacked_format()) {
    LOG_STREAM_WARN()
        << "[WARN] This model appears to be in the per-expert (\"unpacked\") "
           "tensor "
           "format (gguf_tool verify would report 'Expert format: per-expert' "
           "for "
           "it), but unpacked_expert_format is not set -- io_uring batching "
           "and the "
           "VRAM-shadow bypass fast path will BOTH be silently disabled for "
           "this "
           "entire run, forcing every dispatch through the slow, unbatched "
           "fallback "
           "path. Set unpacked_expert_format=true in your config to use the "
           "single-DMA fast path this format is built for. See CLAUDE.md.\n";
  }

  return {};
}
