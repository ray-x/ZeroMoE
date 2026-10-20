// EngineRunner phases: plan_expert_cache, setup_vram_expert_cache.
// Split from the former run_engine() in src/main.cpp; bodies are unchanged except where
// a former local became an EngineRunner member of the same name.
#include "engine_runner.hpp"
#include "ggml_bailingmoe3_graph.hpp"

// True when the GLM full graph (glm5next implementation) is the active execution path of this engine's family.
static bool glm_graph_in_use(const StreamingEngine &engine) {
  const ModelFamily f = engine.family_ops.family;
  if (f == ModelFamily::GLM5NEXT)
    return glm5next_full_graph_enabled();
  const FullGraphModel *m = engine.family_ops.graph_model;
  return f == ModelFamily::GLM_DSA && m &&
         (m->enabled(engine, ModelGraphPhase::Prefill) || m->enabled(engine, ModelGraphPhase::Decode));
}

std::expected<void, std::string> EngineRunner::plan_expert_cache() {
  // --- Multi-GPU memory planning (from optimize.md) ---
  auto &cfg = engine.model_file.architecture_config;

  // Allocate expert VRAM cache on vault GPU(s)
  // The optional GLM NextN/MTP block is a real streamed MoE layer too.
  // Use the cache topology populated by graph_initializer rather than only
  // the target trunk, so its per-layer geometry is sized consistently.
  cache_layers = static_cast<int>(engine.layers_experts.size());
  expert_layer_slab_bytes.assign((size_t)std::max(0, cache_layers),
                                 0);
  if (cfg.n_experts > 0 && cfg.moe_inter > 0) {
    // Real per-expert byte size, accounting for each projection's own quant
    // type (e.g. GLM-5.2 UD ships gate/up=Q2_K, down=Q3_K — a flat "2-bit
    // guess" undersizes the slab by ~45% and causes VRAM cache corruption /
    // OOB reads once an expert is promoted). GpuTopology uses ONE uniform
    // slot size for the entire VRAM pool across every layer, but "Dynamic"
    // GGUF quantization (e.g. Unsloth UD) can use a DIFFERENT quant type
    // per layer — so sizing from just the first MoE layer found can still
    // undersize the pool for a later, bigger-quant layer. Scan every MoE
    // layer and take the max.
    for (int l = cfg.first_dense; l < cache_layers; ++l) {
      const size_t raw_layer_bytes = engine.compute_expert_slab_bytes(l);
      // The compact pool is still laid out in byte-addressable VRAM,
      // but 4 KiB region padding keeps every following slot/page base
      // aligned for future direct GPU/file transfers. Explicit
      // full-layer prefix regions additionally need a stride that is a
      // whole number of quantization rows so GGML's MUL_MAT_ID can view
      // the resident slots without a repack. Ordinary compact regions
      // retain the smaller 4 KiB-only layout.
      constexpr size_t COMPACT_REGION_ALIGNMENT = 4096;
      size_t layer_slot_bytes = raw_layer_bytes;
      if (layer_slot_bytes > 0) {
        size_t slot_alignment = COMPACT_REGION_ALIGNMENT;
        // The GLM5NEXT full graph aliases resident slots as MUL_MAT_ID views for ANY layer with
        // hot experts, so every layer needs a row-aligned slot stride, not just the full-layer prefix.
        const bool glm_graph_rows = glm_graph_in_use(engine);
        const bool dsv41_graph_rows =
            engine.family_ops.family == ModelFamily::DeepSeek_V41 &&
            (moestream::graphcfg::flag_on("MOESTREAM_DEEPSEEK_V41_CACHE_MOE_GRAPH") ||
             moestream::graphcfg::flag_on("MOESTREAM_DEEPSEEK_V41_CACHE_MOE_DECODE"));
        if (config.vram_cache_full_layer_prefix > l || glm_graph_rows ||
            dsv41_graph_rows) {
          const size_t row_alignment =
              engine.expert_cache_layer_stride_alignment(l);
          if (row_alignment == SIZE_MAX) {
            LOG_STREAM_WARN()
                << "[GPU] Full-layer cache row alignment overflow at layer "
                << l << "; leaving the 4 KiB compact stride\n";
          } else if (row_alignment > 1) {
            // The direct-view path only needs row alignment. Do
            // not take the LCM with 4096 here: for IQ formats the
            // LCM can be tens of MiB per expert and would waste
            // most of the cache on alignment padding alone.
            slot_alignment = row_alignment;
          }
        }
        const size_t rem = layer_slot_bytes % slot_alignment;
        if (rem != 0 && layer_slot_bytes <= SIZE_MAX - (slot_alignment - rem)) {
          layer_slot_bytes += slot_alignment - rem;
        }
      }
      expert_layer_slab_bytes[(size_t)l] = layer_slot_bytes;
      expert_slab_bytes = std::max(expert_slab_bytes, raw_layer_bytes);
    }
    const size_t stride_alignment = engine.expert_cache_stride_alignment();
    if (expert_slab_bytes > 0 &&
        engine.family_ops.family == ModelFamily::GLM_DSA &&
        !glm_graph_in_use(engine)) {
      // GLM_DSA dispatches one layer/expert at a time. It never exposes
      // active-layer slots as a zero-copy GGML MUL_MAT_ID tensor view, so
      // the model-wide LCM of every quantized projection's row size is not
      // required here. On GLM-5.3 UD-IQ2_M that LCM was 216 MB: applying it
      // to 256 active-layer slots requested 55.4 GB on the 3090 and silently
      // removed its promotion pool. 4 KiB slot alignment is sufficient for
      // the layered CUDA execution path; keep row-compatible padding for the
      // graph families below.
      constexpr size_t layered_slot_alignment = 4096;
      const size_t rem = expert_slab_bytes % layered_slot_alignment;
      if (rem != 0 &&
          expert_slab_bytes <= SIZE_MAX - (layered_slot_alignment - rem)) {
        expert_slab_bytes += layered_slot_alignment - rem;
      }
      LOG_STREAM_INFO() << "[GPU] Expert staging/legacy stride: "
                        << expert_slab_bytes
                        << " bytes (GLM_DSA layered path, 4 KiB aligned; "
                        << "skipped graph row alignment " << stride_alignment
                        << ")\n";
    } else if (expert_slab_bytes > 0 && stride_alignment > 1 &&
               stride_alignment != SIZE_MAX) {
      // The direct resident-layer GGML view uses the cache's physical slot
      // stride as quantized tensor nb[2].  Graph families that expose these
      // slots as zero-copy MUL_MAT_ID views need whole-row alignment for every
      // routed projection, including mixed IQ formats.
      const size_t rem = expert_slab_bytes % stride_alignment;
      if (rem != 0)
        expert_slab_bytes += stride_alignment - rem;
      LOG_STREAM_INFO() << "[GPU] Expert staging/legacy stride: "
                        << expert_slab_bytes
                        << " bytes (row-compatible alignment "
                        << stride_alignment
                        << "; compact per-layer VRAM regions use their own 4 "
                           "KiB-padded sizes)\n";
    } else if (stride_alignment == SIZE_MAX) {
      LOG_STREAM_WARN() << "[GPU] Expert cache row alignment overflow; "
                        << "disabling zero-repack stride adjustment\n";
    }
  }
  // GLM NextN/MTP's own target-verification pass (MtpSpeculator::
  // target_forward, mtp_speculative.cpp) always calls layers_forward_gguf
  // directly and never the full graph -- that call site's own comment
  // already anticipates "a full-graph fast path already produced final
  // output" as a real (if previously unreachable) case and guards it
  // explicitly. The full graph itself still does not build the optional
  // NextN draft block, so MTP's draft/verification loop continues to run
  // entirely on the legacy path regardless of this flag; only the MAIN
  // generation loop's own prefill is affected.
  //
  // This used to unconditionally disable the full graph whenever MTP was
  // enabled. That was a real, warranted precaution at the time: the full
  // graph's prefill-to-decode handoff (KDA/GDN recurrent state, conv
  // history) had a silent state-transposition bug (ggml-cuda's
  // gated_delta_net stores state [k-fast,v-slow]; this codebase's own
  // legacy CPU delta step expects [k-slow,v-fast] -- see build_kda's own
  // comment, ggml_glm5next_graph.cpp) that corrupted every decode token
  // after a full-graph prefill. Running MTP -- which depends on an
  // accurate continuing state to draft against -- on top of that would
  // have made an already-hard-to-diagnose bug even harder to isolate. Now
  // that the transposition is fixed (ggml_cont(ggml_transpose(...)) before
  // the state is copied out), there is no longer a known correctness
  // reason to keep the two mutually exclusive.
  const FullGraphModel *graph_model = engine.family_ops.graph_model;
  // Graph mode owns startup-time resources when either phase is graph-backed.
  // In particular, GLM can intentionally keep batched prefill on the legacy
  // path while using the reusable full graph for single-token decode; still
  // prepare its shared-cache references and reusable host source here.
  graph_mode =
      graph_model != nullptr &&
      (graph_model->enabled(engine, ModelGraphPhase::Prefill) ||
       graph_model->enabled(engine, ModelGraphPhase::Decode));
  // Full-graph host-resident expert layers use fixed-stride DDR arenas when
  // ram_preload_all_experts is enabled. GLM also aliases its usage-ranked
  // resident experts in the existing VRAM cache; other graph families retain
  // their own source-ownership policy behind the explicit environment flag.
  graph_ram_cache =
      graph_mode &&
      (std::getenv("MOESTREAM_FULL_GRAPH_USE_RAM_CACHE") != nullptr ||
       (glm_graph_in_use(engine) && config.ram_preload_all_experts) ||
       (engine.family_ops.family == ModelFamily::Qwen4Exp &&
        config.ram_preload_all_experts));
  // DeepSeek-V4.1 and opt-in Ling graph nodes dispatch MoE through the
  // RAMSafetyManager/GpuTopology caches. They own no second expert arena, so
  // keep the ordinary usage-ranked preload enabled in graph mode.
  graph_uses_native_moe_cache =
      graph_mode &&
      (engine.family_ops.family == ModelFamily::DeepSeek_V41 ||
       (engine.family_ops.family == ModelFamily::BailingMoE3 &&
        bailingmoe3_native_graph_enabled()));
  // Native-cache graph callbacks use the existing resident dense weights as
  // well. Keep one native device pool for those calls; the graph borrows its
  // device_ptrs instead of making a second copy.
  graph_owns_dense =
      graph_mode && !graph_uses_native_moe_cache;
  glm_graph_aliases_vram_cache =
      graph_mode && glm_graph_in_use(engine);
  if (graph_mode && engine.family_ops.family == ModelFamily::Qwen4Exp &&
      qwen4exp_full_graph_decode_enabled()) {
    const int n_layers =
        std::max(0, engine.model_file.architecture_config.n_layers);
    const int n_experts =
        std::max(0, engine.model_file.architecture_config.n_experts);
    const int device_layers = std::min(
        std::max(0, qwen4exp_full_graph_device_expert_layers()), n_layers);
    if (device_layers > 0) {
      engine.qwen_ram_vram_only_layers.assign((size_t)n_layers, 0);
      const bool trailing = qwen4exp_full_graph_device_expert_trailing();
      const int first_layer = trailing ? n_layers - device_layers : 0;
      size_t vram_only_bytes = 0;
      int vram_only_layer_count = 0;
      for (int layer = first_layer; layer < first_layer + device_layers;
           ++layer) {
        if (is_cpu_affinity_layer(layer))
          continue;
        engine.qwen_ram_vram_only_layers[(size_t)layer] = 1;
        ++vram_only_layer_count;
        const size_t expert_bytes = engine.compute_expert_slab_bytes(layer);
        if (n_experts > 0 &&
            expert_bytes <= SIZE_MAX / (size_t)n_experts &&
            vram_only_bytes <=
                SIZE_MAX - expert_bytes * (size_t)n_experts)
          vram_only_bytes += expert_bytes * (size_t)n_experts;
      }
      LOG_STREAM_INFO()
          << "[MOE-OWNERSHIP] full-graph device block [" << first_layer
          << ", " << (first_layer + device_layers)
          << ") marks " << vram_only_layer_count
          << " non-CPU-affinity layer(s) as VRAM-only, approximately "
          << (double)vram_only_bytes / (1024.0 * 1024.0 * 1024.0)
          << " GiB; persistent DDR preload/graph-RAM exclusion applies "
             "when those tiers are enabled\n";
    }
  }
  // GLM5NEXT's device-resident expert layers (glm5next_full_graph's own
  // device_expert_layers) were previously invisible to this arena-request
  // loop: only Qwen4Exp populated qwen_ram_vram_only_layers, so GLM always
  // requested a complete host-RAM arena for every non-dense layer even when
  // MOESTREAM_GLM5NEXT_GRAPH_PRELOAD_LAYERS asked for some of them to live
  // in VRAM instead. That silently doubled down on host RAM (the graph's
  // per-layer arena reservation is on top of the RAM tier's own preload)
  // and could exhaust the MemoryGuard ledger's watermark headroom for the
  // model's last layer(s) even though real MemAvailable had room to spare --
  // see prepare_expert_source_only's "boot host admission refused" log.
  if (graph_mode && graph_ram_cache && config.ram_preload_all_experts &&
      glm_graph_in_use(engine)) {
    const int n_layers =
        std::max(0, engine.model_file.architecture_config.n_layers);
    const int n_experts =
        std::max(0, engine.model_file.architecture_config.n_experts);
    const int first_dense = engine.model_file.architecture_config.first_dense;
    const int device_layers = std::min(
        std::max(0, glm5next_full_graph_device_expert_layers()),
        std::max(0, n_layers - first_dense));
    if (device_layers > 0) {
      engine.qwen_ram_vram_only_layers.assign((size_t)n_layers, 0);
      size_t vram_only_bytes = 0;
      int vram_only_layer_count = 0;
      for (int layer = first_dense; layer < first_dense + device_layers;
           ++layer) {
        if (is_cpu_affinity_layer(layer))
          continue;
        engine.qwen_ram_vram_only_layers[(size_t)layer] = 1;
        ++vram_only_layer_count;
        const size_t expert_bytes = engine.compute_expert_slab_bytes(layer);
        if (n_experts > 0 &&
            expert_bytes <= SIZE_MAX / (size_t)n_experts &&
            vram_only_bytes <=
                SIZE_MAX - expert_bytes * (size_t)n_experts)
          vram_only_bytes += expert_bytes * (size_t)n_experts;
      }
      LOG_STREAM_INFO()
          << "[MOE-OWNERSHIP] GLM5NEXT full-graph device block ["
          << first_dense << ", " << (first_dense + device_layers)
          << ") excludes " << vram_only_layer_count
          << " non-CPU-affinity layer(s) from persistent DDR preload and"
             " graph-RAM arenas (VRAM-only), approximately "
          << (double)vram_only_bytes / (1024.0 * 1024.0 * 1024.0)
          << " GiB\n";
    }
  }
  // Qwen CPU-affinity layers use the same fixed-stride host-RAM arena as
  // the full graph, but only when the configured RAM preload really owns
  // every expert in those layers. This gives the CPU MUL_MAT_ID graph a
  // stable DDR source without enabling the full-graph cache or changing
  // any other model family's cache policy.
  qwen_cpu_ram_cache =
      !graph_mode && config.qwen4exp_cpu_ggml_moe &&
      engine.family_ops.family == ModelFamily::Qwen4Exp &&
      (!config.cpu_affinity_layers.empty() ||
       (config.qwen4exp_cpu_offload &&
        !config.qwen4exp_cpu_offload_layers.empty())) &&
      config.ram_preload_all_experts;
  ram_graph_cache = graph_ram_cache || qwen_cpu_ram_cache;
  if (engine.family_ops.family == ModelFamily::Qwen4Exp &&
      config.ram_preload_all_experts) {
    LOG_STREAM_INFO()
        << "[CONFIG] Qwen MoE source contract: use the already-preloaded "
           "DDR RAM tier directly; file-backed GGUF expert mmap fallback "
           "is disabled. The PLE table may still be mmap-backed, but MoE "
           "expert planes are not.\n";
  }
  if (engine.mtp_enabled && glm_graph_in_use(engine)) {
    LOG_STREAM_INFO()
        << "[MTP] GLM NextN/MTP is enabled alongside the GLM full graph: "
           "the main generation loop's prefill uses the full graph, "
           "MTP's own draft/verification pass continues to run entirely "
           "on the legacy path (see mtp_speculative.cpp's target_forward, "
           "which never calls the full graph).\n";
  }
  if (engine.family_ops.family == ModelFamily::DeepSeek_V41 &&
      moestream::graphcfg::flag_on("MOESTREAM_DEEPSEEK_V41_CACHE_MOE_GRAPH")) {
    // This path binds each complete pool as a strided MUL_MAT_ID tensor.
    // The global LRU (already supported by GpuTopology) lets one layer's
    // required set borrow free slots across the pool. Validate its uniform
    // geometry again in the graph builder before exposing any CUDA view.
    expert_layer_slab_bytes.clear();
    LOG_INFO("[DSV41-CACHE-GRAPH] uniform native VRAM pools, stride=%zu; "
             "graph-owned expert bytes=0\n", expert_slab_bytes);
  }
  return {};
}

std::expected<void, std::string> EngineRunner::setup_vram_expert_cache() {
  auto &cfg = engine.model_file.architecture_config;
  const FullGraphModel *graph_model = engine.family_ops.graph_model;
  if (!graph_mode) {
    gpu.setup_expert_cache(expert_slab_bytes, cfg.n_experts, config,
                           expert_layer_slab_bytes);
  } else {
    // 2026-09-09: two real, measured incidents, both from this same
    // decision point.
    //
    // Incident A (why this can never be "skip the cache entirely"): most
    // family full graphs only handle S>1 prefill. GLM5NEXT now also has a
    // reusable S=1 decode graph, but this legacy cache remains the safe
    // fallback if either graph is declined. The old code skipped the legacy
    // cache completely whenever MOESTREAM_FULL_GRAPH_USE_RAM_CACHE was unset,
    // which for
    // qwen4exp/deepseek_v4 (neither of which implement the RAM-arena
    // dedup that env var was designed to gate -- that's GLM5NEXT-only)
    // meant EVERY graph_mode run had zero decode-time GPU cache. Measured
    // on real hardware: decode collapsed to 0.13 tok/s, expert_calls=
    // 0GPU/8160CPU, "CPU fallback cause: 8160 upload-failed" -- there was
    // no cache slot to upload into at all.
    //
    // Incident B (why the cache size can't just be a flat 4GB either):
    // setting MOESTREAM_FULL_GRAPH_USE_RAM_CACHE (the old workaround for
    // Incident A) reinstated a flat 4GB fallback cache regardless of how
    // much VRAM a graph's own device-resident expert arena
    // (device_expert_layers) already committed. At device_expert_layers=
    // 20 on a 24GB card (16.11 GiB device expert arena + 2.92 GiB dense
    // graph arena), the extra flat 4GB starved the graph's own later
    // ggml_gallocr_reserve() call of its last ~1.8GB, which ggml does not
    // handle gracefully -- it aborted the whole process (SIGSEGV).
    //
    // Fix: size the expert cache against REAL current free VRAM on the
    // device the graph will use, minus this family's graph-owned device
    // expert-layer estimate (same expert_slab_bytes/n_experts this cache
    // itself is sized from) and a flat reserve for the
    // graph's dense arena + compute workspace (measured 2.92+1.80 GiB
    // above; kept generous since neither scales down with
    // device_expert_layers). Floored at 0 (never negative) -- a 0-sized
    // cache still means every decode step pays the CPU-fallback cost,
    // but at least never crashes the process fighting the graph for the
    // same VRAM.
    //
    // 2026-09-28: the original fix additionally clamped this to a flat
    // 4GB ceiling ("capped at the original 4GB ceiling, kept generous"),
    // reusing Incident B's pre-fix number as a known-safe value rather
    // than trusting the now-headroom-aware math on its own. That ceiling
    // is vestigial, not a safety requirement: Incident B's real cause was
    // an UNCONDITIONAL flat cache stacked on top of a large already-
    // committed device_expert_layers arena, and headroom_gb below already
    // subtracts that same commitment (plus a 6GB reserve) before this
    // point -- at device_expert_layers=20, the incident's own config,
    // headroom_gb is already small and no fixed ceiling is needed to keep
    // it safe. At low/zero device_expert_layers (e.g. the "dense split"
    // topology, GRAPH_PRELOAD_LAYERS=0), the flat ceiling was the ONLY
    // thing capping this cache -- confirmed on real hardware: it silently
    // discarded ~12 GiB of real, already-computed-safe headroom on a
    // 23.3 GiB-free card (only ~50% VRAM used), while the configured
    // moe_cache_gb (11.5 GiB) sat unreachable behind a stale 4GB cap.
    // Removed; headroom_gb's own subtraction is the real budget now.
    constexpr double kGraphOverheadReserveGb = 6.0;
    constexpr double kBytesPerGb = 1024.0 * 1024.0 * 1024.0;

    const bool glm_shared_expert_cache =
        glm_graph_in_use(engine);
    // GLM's graph aliases the existing cache, so it owns no additional
    // device-resident expert block to reserve ahead of cache sizing.
    const int dev_layers = !glm_shared_expert_cache && graph_model
                               ? graph_model->device_expert_layers()
                               : 0;

    double est_device_expert_gb = 0.0;
    if (dev_layers > 0 && expert_slab_bytes > 0)
      est_device_expert_gb = (double)expert_slab_bytes * (double)cfg.n_experts *
                             (double)dev_layers / kBytesPerGb;

    // Generalized GPU staging (ggml_qwen4exp_graph.cpp's build_gpu_stage_node):
    // a fixed, topk-sized (not n_experts-sized) compact gate+up+down
    // destination per staged host-resident decode layer, bump-allocated
    // from the SAME device_expert_buffer arena the block above estimates --
    // this is a rough pre-estimate only (the real accounting, which
    // correctly excludes layers already covered by dev_layers, lives in
    // device_expert_bytes()); folded in here so this early fallback-cache
    // sizing pass doesn't leave the legacy cache too large and get
    // squeezed out later by the real graph's own allocation, the same
    // class of bug Incident B (this file's own comment above) already
    // documents.
    double est_gpu_stage_gb = 0.0;
    if (engine.qwen4exp_gpu_stage_enabled && expert_slab_bytes > 0 &&
        cfg.topk > 0 && !engine.qwen4exp_gpu_stage_layers.empty())
      est_gpu_stage_gb = (double)expert_slab_bytes * (double)cfg.topk *
                         (double)engine.qwen4exp_gpu_stage_layers.size() /
                         kBytesPerGb;

    // Option B's static hot/cold split (plan_expert_vram_split() in
    // ggml_qwen4exp_graph.cpp) commits its own device-resident expert
    // arena later, using the SAME usage ranking/right-sizing this early
    // pass replicates below -- it was never subtracted from this cache's
    // headroom before. 2026-09-29: removing this file's old flat 4GB
    // fallback-cache ceiling (see Incident B's follow-up comment above)
    // exposed that gap: with device_expert_layers=0 the legacy cache grew
    // to its full configured moe_cache_gb (11.5 GiB) instead of being
    // capped, leaving only ~5.8 GiB free where Option B's committed hot
    // set needed ~12.6 GiB -- a mid-decode cudaMalloc OOM that corrupted
    // output (ggml-cuda's own allocator error prints straight to stdout,
    // bypassing LOG_ERROR, and leaked into the token stream). Reproduced
    // identically on both IQ1_M and Q2_0, so it's this budget gap, not a
    // quant-type issue.
    //
    // Mirrors plan_expert_vram_split()'s own per-layer ranking, coverage
    // check, and qwen4exp_option_b_right_size_top_n shrink (using the same
    // qwen4exp_graph_pin_file usage snapshot, already assigned above) so
    // this estimate matches what that function will actually commit,
    // rather than a flat top_n-for-every-selected-layer upper bound (which
    // overshoots badly: several selected layers never clear
    // qwen4exp_option_b_min_coverage and commit nothing). A first attempt
    // at this fix used that flat bound and it starved the legacy cache to
    // 0 GiB, which hit a separate, pre-existing "0 GiB legacy cache" bug
    // (a CUDA illegal-memory-access abort) -- an accurate estimate avoids
    // exercising that rare path at all. Concentrated-only auto-selection
    // still can't be estimated here (the candidate layer list itself comes
    // from this same usage pass) and contributes 0, same as before this
    // fix.
    double est_option_b_gb = 0.0;
    if (engine.family_ops.family == ModelFamily::Qwen4Exp &&
        config.qwen4exp_option_b_top_n > 0 &&
        (config.qwen4exp_option_b_layer >= 0 ||
         !config.qwen4exp_option_b_selected_layers.empty())) {
      std::vector<int> option_b_layers;
      if (config.qwen4exp_option_b_layer >= 0)
        option_b_layers.push_back(config.qwen4exp_option_b_layer);
      else
        option_b_layers = config.qwen4exp_option_b_selected_layers;
      auto [usage_entries, usage_maxc] =
          StreamingEngine::load_usage(engine.qwen4exp_graph_pin_file);
      (void)usage_maxc;
      if (!usage_entries.empty() && cfg.n_experts > 0) {
        size_t option_b_bytes = 0;
        // Global pool mode: estimate from the same selection the planner will
        // commit (option_b_global::select), not from a per-layer top_n.
        const bool est_global = config.qwen4exp_option_b_global_top_k > 0;
        if (est_global) {
          std::vector<int> live_layers;
          for (int layer : option_b_layers) {
            if (layer < 0 || layer >= cfg.n_layers ||
                is_cpu_affinity_layer(layer))
              continue;
            if (dev_layers > 0) {
              const bool in_baseline_block =
                  qwen4exp_full_graph_device_expert_trailing()
                      ? layer >= cfg.n_layers - dev_layers
                      : layer < dev_layers;
              if (in_baseline_block)
                continue;
            }
            live_layers.push_back(layer);
          }
          const auto sel = option_b_global::select(
              usage_entries, live_layers, cfg.n_layers, cfg.n_experts,
              config.qwen4exp_option_b_global_top_k,
              config.qwen4exp_option_b_global_floor_per_layer,
              config.qwen4exp_option_b_top_n);
          for (const auto &[layer, ids] : sel.hot) {
            const size_t expert_bytes = engine.compute_expert_slab_bytes(layer);
            if (expert_bytes > 0 && ids.size() <= SIZE_MAX / expert_bytes)
              option_b_bytes += expert_bytes * ids.size();
          }
        }
        for (int layer : est_global ? std::vector<int>{} : option_b_layers) {
          if (layer < 0 || layer >= cfg.n_layers ||
              is_cpu_affinity_layer(layer))
            continue;
          if (dev_layers > 0) {
            const bool in_baseline_block =
                qwen4exp_full_graph_device_expert_trailing()
                    ? layer >= cfg.n_layers - dev_layers
                    : layer < dev_layers;
            if (in_baseline_block)
              continue;
          }
          std::vector<uint64_t> counts(static_cast<size_t>(cfg.n_experts), 0);
          for (const auto &e : usage_entries) {
            if (e.layer == layer && e.expert >= 0 && e.expert < cfg.n_experts)
              counts[static_cast<size_t>(e.expert)] += e.count;
          }
          uint64_t total_count = 0;
          for (uint64_t count : counts)
            total_count += count;
          if (total_count == 0)
            continue;
          std::vector<int> ranked(static_cast<size_t>(cfg.n_experts));
          for (int expert = 0; expert < cfg.n_experts; ++expert)
            ranked[static_cast<size_t>(expert)] = expert;
          std::stable_sort(ranked.begin(), ranked.end(), [&](int a, int b) {
            if (counts[static_cast<size_t>(a)] != counts[static_cast<size_t>(b)])
              return counts[static_cast<size_t>(a)] > counts[static_cast<size_t>(b)];
            return a < b;
          });
          int hot_count = std::min(config.qwen4exp_option_b_top_n, cfg.n_experts);
          if (config.qwen4exp_option_b_right_size_top_n) {
            uint64_t cum = 0;
            for (int k = 0; k < hot_count; ++k) {
              cum += counts[static_cast<size_t>(ranked[static_cast<size_t>(k)])];
              if (static_cast<double>(cum) / static_cast<double>(total_count) >=
                  config.qwen4exp_option_b_min_coverage) {
                hot_count = k + 1;
                break;
              }
            }
          }
          uint64_t hot_route_count = 0;
          for (int k = 0; k < hot_count; ++k)
            hot_route_count +=
                counts[static_cast<size_t>(ranked[static_cast<size_t>(k)])];
          const double coverage = static_cast<double>(hot_route_count) /
                                  static_cast<double>(total_count);
          if (coverage < config.qwen4exp_option_b_min_coverage)
            continue;
          const size_t expert_bytes = engine.compute_expert_slab_bytes(layer);
          if (expert_bytes > 0 &&
              (size_t)hot_count <= SIZE_MAX / expert_bytes)
            option_b_bytes += expert_bytes * (size_t)hot_count;
        }
        est_option_b_gb = (double)option_b_bytes / kBytesPerGb;
      }
    }

    // The GLM graph aliases the existing VRAM cache for expert weights; its
    // dense graph and scheduler workspace are covered by the generic reserve.
    const double graph_overhead_reserve_gb = kGraphOverheadReserveGb;

    // Ground-truth free VRAM right now, same GPU setup_expert_cache is
    // about to target. Single-GPU-vault configs only (the graphs'
    // current real deployment target) -- core_free_vram_bytes() reads
    // the "core" device, which is also the vault/graph device in every
    // config these families ship with today.
    const double free_gb = (double)gpu.core_free_vram_bytes() / kBytesPerGb;
    const double headroom_gb = free_gb - est_device_expert_gb -
                               est_option_b_gb -
                               est_gpu_stage_gb - graph_overhead_reserve_gb;
    const double fallback_gb = std::max(headroom_gb, 0.0);

    // Design policy (2026-10-05): Qwen4Exp has no layered decode. With the
    // full-graph decode enabled the graph reads only its own device arena
    // (Option-B hot experts / resident layers) and computes the rest on the
    // CPU from the DDR arenas, so the legacy VRAM expert cache (pinned + LRU
    // tiers) and the active-layer area are never filled or consulted: a run
    // measured 0 / 4650 slots filled and 0 hits while reserving 6.4 GiB.
    // Disable them whenever the Qwen4Exp decode graph is requested, except
    // when its selected-layer EXL3 custom op explicitly borrows these slots.
    // The explicit qwen4exp_option_b_disable_legacy_cache key stays honoured
    // for runs that still use the older prefill-only graph.
    const bool qwen_full_graph_decode =
        engine.family_ops.family == ModelFamily::Qwen4Exp &&
        qwen4exp_full_graph_decode_enabled();
    const bool exl3_ggml_bridge_enabled =
        !engine.exl3_ggml_bridge_model_path.empty();
    const bool option_b_disable_legacy_cache =
        (qwen_full_graph_decode && !exl3_ggml_bridge_enabled) ||
        (config.qwen4exp_option_b_disable_legacy_cache &&
         config.qwen4exp_option_b_top_n > 0 &&
         engine.family_ops.family == ModelFamily::Qwen4Exp);
    EngineConfig fallback_config = config;
    if (option_b_disable_legacy_cache) {
      for (auto &gc : fallback_config.gpus) {
        if (gc.device_id == gpu.get_core_device())
          gc.moe_cache_gb = 0.0;
      }
      fallback_config.use_active_layer_area = false;
      LOG_INFO("[GPU] Qwen4Exp full graph: legacy VRAM expert cache and "
               "active-layer area disabled (no layered decode); experts "
               "come from the static graph arena or the DDR arenas\n");
    }
    // Allocation keys are decimal GB; the headroom calculation above is
    // GiB. Convert before comparing or the graph cap silently removes an
    // additional ~7% of the cache beyond its intended workspace reserve.
    const double fallback_config_gb = fallback_gb * kBytesPerGb / 1.0e9;
    for (auto &gc : fallback_config.gpus) {
      if (gc.moe_cache_gb > fallback_config_gb)
        gc.moe_cache_gb = fallback_config_gb;
    }
    if (glm_graph_in_use(engine)) {
      // The graph aliases pinned entries in the main cache; the temporary
      // active-layer pool is outside that address space and competes with
      // the shared MoE cache for VRAM.
      fallback_config.use_active_layer_area = false;
      LOG_STREAM_INFO()
          << "[GPU] Active-layer temporary area disabled for the GLM graph; "
             "hot experts alias the main VRAM cache\n";
    }
    LOG_STREAM_INFO()
        << "[GPU] VRAM expert cache capacity: capped at <=" << fallback_gb
        << " GiB/GPU ("
        << (glm_shared_expert_cache ? "shared with GLM graph; " : "")
        << "sized from free="
        << free_gb
        << " GiB - graph_expert_layers(" << dev_layers << ")~="
        << est_device_expert_gb
        << " GiB - gpu_stage_layers(" << engine.qwen4exp_gpu_stage_layers.size()
        << ")~=" << est_gpu_stage_gb << " GiB - option_b~=" << est_option_b_gb
        << " GiB - "
        << graph_overhead_reserve_gb << " GiB graph overhead reserve"
        << " -- see this call site's "
           "own comment for the two real incidents this guards against)\n";
    const bool option_b_stage_enabled =
        config.qwen4exp_option_b_stage_expert >= 0;
    if (fallback_gb <= 0.0 && !option_b_stage_enabled) {
      // This graph-only fallback is allowed to have no legacy cache.  Keep
      // the temporary active-layer pool off too; otherwise it would consume
      // another ~0.2 GB after the explicit zero-cache decision and could
      // still starve the resident graph allocator.  This is local to the
      // graph fallback config; normal model/cache policy is unchanged.
      fallback_config.use_active_layer_area = false;
      LOG_WARN(
          "[GPU] Expert cache capacity is 0 GiB -- graph allocations leave "
          "no VRAM headroom after the %.1f GiB overhead reserve; graph "
          "expert references will use DDR, and legacy decode may use CPU if "
          "the reusable graph declines. Lower graph residency if that "
          "fallback matters for this "
          "run.\n",
          graph_overhead_reserve_gb);
    } else if (fallback_gb <= 0.0 && option_b_stage_enabled) {
      // The Layer-13 proof needs one slot per concurrently staged expert.
      // These are small active-layer slabs, not the legacy cache, and fit in
      // the residual headroom even when the generic graph fallback
      // intentionally clamps that cache to zero.
      fallback_config.use_active_layer_area = true;
      fallback_config.active_layer_capacity = std::max(
          1, std::max(fallback_config.active_layer_capacity,
                      config.qwen4exp_option_b_stage_count));
      LOG_INFO("[GPU] Retaining %d active-layer slot(s) for the explicit "
               "Layer-13 staging proof despite zero legacy-cache fallback\n",
               fallback_config.active_layer_capacity);
    }
    gpu.setup_expert_cache(expert_slab_bytes, cfg.n_experts, fallback_config,
                           expert_layer_slab_bytes,
                           fallback_gb <= 0.0 ||
                               option_b_disable_legacy_cache);
  }
  return {};
}
