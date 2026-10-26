// Qwen4Exp full graph: Decode preparation: per-token input feeding and layer-13 staging.
//
// Split out of the former single-file ggml_qwen4exp_graph.cpp; shared
// declarations live in ggml_qwen4exp_graph_internal.hpp.

#include "ggml_qwen4exp_graph_internal.hpp"

#if defined(ZEROMOE_USE_GGML_CUDA)

using namespace qwen4exp_graph;

namespace qwen4exp_graph {

// Writes the standard positions/mask/decode_indices tensors of a decode graph
// for the token at pos_base.
bool feed_decode_common_inputs(FullGraph &g, int pos_base) {
  if (!g.positions || !g.attn_mask || !g.decode_indices)
    return false;
  static_cast<int32_t *>(g.positions->data)[0] = pos_base;
  auto *mask = static_cast<ggml_fp16_t *>(g.attn_mask->data);
  for (int k = 0; k < g.kv_capacity; ++k)
    mask[k] = ggml_fp32_to_fp16(k <= pos_base ? 0.0f : -INFINITY);
  auto *indices = static_cast<int32_t *>(g.decode_indices->data);
  // See build_graph()'s initial decode_indices fill for why this must be
  // h * kv_capacity + pos_base, not pos_base * Hkv + h.
  for (int h = 0; h < g.Hkv; ++h)
    indices[h] = h * g.kv_capacity + pos_base;
  return true;
}

} // namespace qwen4exp_graph

bool qwen4exp_full_graph_decode_prepare(StreamingEngine &engine, int pos_base,
                                        const int *token_ids) {
  const auto decode_prepare_started = std::chrono::steady_clock::now();
  if (!full_graph_decode_enabled_for_engine(engine) || pos_base < 0 ||
      (engine.family_ops.family != ModelFamily::Qwen4Exp &&
       engine.family_ops.family != ModelFamily::Qwen))
    return false;
  // Keep the same fail-closed rule as prefill.  A full-graph decode call
  // takes precedence over the layered token dispatcher, so allowing this
  // combination would make the configured CPUOffload layers disappear from
  // the actual execution path rather than run through Milestone B.
  if (engine.family_ops.family == ModelFamily::Qwen4Exp &&
      engine.qwen4exp_cpu_offload &&
      !engine.qwen4exp_cpu_offload_layers.empty() &&
      !env_qwen4exp_host_node_enabled()) {
    LOG_ERROR(
        "[GGML-QWEN-FULL-DECODE] refusing decode: CPUOffload layers require "
        "ZEROMOE_QWEN4EXP_HOST_NODE=1 under the full-graph-only Qwen policy\n");
    return false;
  }
  const auto &c = engine.model_file.architecture_config;
  if (engine.max_t <= 0 || pos_base >= engine.max_t)
    return false;

  // The decode graph seeds recurrent/KV state from the graph's validated
  // prefill writeback.  Do not silently combine it with the existing layered
  // GPU attention/GDN state owners; that would create two divergent state
  // stores.  A later capacity growth is allowed because it first downloads
  // the current decode graph state below.
  const bool continuing_decode =
      g_graph && g_graph->ready && g_graph->engine == &engine &&
      g_graph->decode_mode;
  const bool has_graph_prefill =
      g_graph && g_graph->ready && g_graph->engine == &engine &&
      !g_graph->decode_mode;
  // Qwen3.5/Qwen3.6 currently use the shared Qwen executor for prompt
  // prefill. Its K/V and recurrent state are authoritative; the first
  // full-graph decode copies them into graph-owned device tensors once.
  // Validate every layer before allocating the decode graph.
  bool has_qwen_host_prefill_state = false;
  if (engine.family_ops.family == ModelFamily::Qwen &&
      !continuing_decode && !has_graph_prefill) {
    if (pos_base <= 0) {
      LOG_ERROR("[GGML-QWEN-FULL-DECODE] host prefill position is invalid: %d\n",
                pos_base);
    } else if (engine.qwen4exp_recur_state.size() <
                   static_cast<size_t>(c.n_layers) ||
               engine.qwen4exp_conv_history.size() <
                   static_cast<size_t>(c.n_layers) ||
               engine.Kc.size() < static_cast<size_t>(c.n_layers) ||
               engine.Vc.size() < static_cast<size_t>(c.n_layers)) {
      LOG_ERROR("[GGML-QWEN-FULL-DECODE] host prefill cache vectors are short: "
                "layers=%d recurrent=%zu history=%zu K=%zu V=%zu\n",
                c.n_layers, engine.qwen4exp_recur_state.size(),
                engine.qwen4exp_conv_history.size(), engine.Kc.size(),
                engine.Vc.size());
    } else {
      const size_t state_size =
          static_cast<size_t>(c.ssm_state_size) * c.ssm_state_size *
          std::max(1, c.ssm_time_step_rank);
      const size_t conv_width =
          2ull * c.ssm_group_count * c.ssm_state_size + c.ssm_inner_size;
      const size_t history_size =
          static_cast<size_t>(std::max(0, c.ssm_conv_kernel - 1)) * conv_width;
      has_qwen_host_prefill_state = true;
      for (int layer = 0; layer < c.n_layers; ++layer) {
        if (qwen35moe_is_gdn_layer(c, layer)) {
          const size_t state_got = engine.qwen4exp_recur_state[layer].size();
          const size_t history_got = engine.qwen4exp_conv_history[layer].size();
          if (state_got != state_size || history_got != history_size) {
            LOG_ERROR("[GGML-QWEN-FULL-DECODE] GDN host state mismatch at "
                      "layer=%d state=%zu/%zu history=%zu/%zu\n",
                      layer, state_got, state_size, history_got, history_size);
            has_qwen_host_prefill_state = false;
            break;
          }
        } else if (!engine.Kc[layer] || !engine.Vc[layer]) {
          LOG_ERROR("[GGML-QWEN-FULL-DECODE] attention host KV missing at "
                    "layer=%d K=%d V=%d\n",
                    layer, engine.Kc[layer] != nullptr,
                    engine.Vc[layer] != nullptr);
          has_qwen_host_prefill_state = false;
          break;
        }
      }
      if (has_qwen_host_prefill_state)
        LOG_INFO("[GGML-QWEN-FULL-DECODE] validated host Qwen KV/GDN prefill "
                 "state for %d layers at position %d\n",
                 c.n_layers, pos_base);
    }
  }
  if (!continuing_decode && !has_graph_prefill &&
      !has_qwen_host_prefill_state) {
    LOG_ERROR("[GGML-QWEN-FULL-DECODE] refusing decode: state was not seeded "
              "by a successful full-graph prefill or validated Qwen host "
              "prefill state\n");
    return false;
  }
  if (!continuing_decode && !has_graph_prefill && has_qwen_host_prefill_state)
    LOG_INFO("[GGML-QWEN-FULL-DECODE] seeding device KV/GDN state from "
             "validated Qwen prefill caches at position %d\n", pos_base);

  // Decode full-graph mode is intentionally stricter than the ordinary graph:
  // host-resident expert planes must already be in complete DDR graph arenas,
  // while the configured device block is loaded directly into VRAM during
  // graph construction. Models with lazy host expert ownership stay on the
  // established layered path.
  for (int layer = 0; layer < c.n_layers; ++layer) {
    if (engine.qwen_ram_vram_only_layer(layer))
      continue;
    if (!engine.graph_ram_layer_ready(layer)) {
      LOG_ERROR("[GGML-QWEN-FULL-DECODE] refusing decode: layer=%d has no "
                "complete preloaded host DDR expert arena\n",
               layer);
      return false;
    }
  }

  const int required = pos_base + 1;
  const int dev = engine.gpu_device_id >= 0 ? engine.gpu_device_id : 0;

  // Whether this session should attempt Phase 2 (true per-token dynamic
  // GPU/CPU dispatch for the partial layer) instead of Phase 1's static,
  // session-level commit. Both share the same usage-ranked expert set
  // (plan_partial_layer/qwen4exp_partial_layer_gpu_ids) -- Phase 2 just
  // decides per-token instead of once at prefill.
  if (engine.family_ops.family != ModelFamily::Qwen)
    plan_partial_layer(engine); // Qwen full-fit uses every expert on device.
  if (g_graph && g_graph->ready && g_graph->engine == &engine &&
      g_graph->decode_mode && g_graph->kv_capacity >= required &&
      g_graph->decode_last_pos + 1 == pos_base) {
    if (has_ple_layer(c)) {
      FullGraph *targets[1] = {g_graph.get()};
      for (FullGraph *t : targets) {
        if (!t || t->ple_layer < 0)
          continue;
        std::vector<float> ple;
        if (!prepare_ple_embedding(*t, ple, 1, pos_base, token_ids) ||
            !t->ple_embedding ||
            ple.size() * sizeof(float) > ggml_nbytes(t->ple_embedding))
          return false;
        std::memcpy(t->ple_embedding->data, ple.data(),
                    ple.size() * sizeof(float));
      }
    }
    return true;
  }

  if (g_graph && g_graph->ready && g_graph->decode_mode &&
      g_graph->engine == &engine) {
    if (!sync_decode_state_to_host(*g_graph)) {
      LOG_ERROR("[GGML-QWEN-FULL-DECODE] refusing request: state download "
                "failed while growing KV capacity\n");
      return false;
    }
    if (g_graph_prefill_arena_owner &&
        g_graph_prefill_arena_owner->engine == &engine)
      g_graph_prefill_arena_owner->prefill_host_state_deferred = false;
    if (g_graph_prefill_state_owner &&
        g_graph_prefill_state_owner->engine == &engine)
      g_graph_prefill_state_owner->prefill_host_state_deferred = false;
  }
  // VRAM diagnostic (2026-09-15): two cheap cudaMemGetInfo reads bracketing
  // prefill's own g_graph teardown (before = prefill's buffers still
  // resident, after = immediately post-free, before decode allocates
  // anything of its own). Confirmed the ~0.33 GiB gap between PREFLIGHT's
  // startup free and this post-teardown free is generic CUDA context/
  // driver overhead (module/PTX loading, cuBLAS handle init) that persists
  // for the process lifetime -- NOT a leak in this file's own arena
  // teardown, which correctly frees essentially 100% of what it allocated
  // (confirmed: freed amount here matched the prefill arena's own tracked
  // size almost exactly). Kept as a permanent low-cost sanity check.
  FullGraph *prefill_arena_source = nullptr;
  FullGraph *prefill_state_source = nullptr;
  const bool decode_compact_arena_needed =
      engine.family_ops.family != ModelFamily::Qwen &&
      ((engine.qwen4exp_partial_layer >= 0 &&
        !engine.qwen4exp_partial_layer_gpu_ids.empty()) ||
       (engine.qwen4exp_gpu_stage_enabled &&
        !engine.qwen4exp_gpu_stage_layers.empty()));
  const bool borrow_prefill_arenas = env_decode_borrow_prefill_arenas() &&
                                     !decode_compact_arena_needed;
  if (env_decode_borrow_prefill_arenas() && decode_compact_arena_needed) {
    LOG_INFO("[GGML-QWEN-FULL-DECODE] arena borrowing skipped because this "
             "decode graph needs decode-only compact expert storage\n");
  }
  if (borrow_prefill_arenas) {
    if (has_graph_prefill && g_graph->prefill_host_state_deferred &&
        g_graph_prefill_arena_owner)
      g_graph_prefill_state_owner = std::move(g_graph);
    if (g_graph_prefill_arena_owner &&
        g_graph_prefill_arena_owner->ready &&
        g_graph_prefill_arena_owner->engine == &engine &&
        !g_graph_prefill_arena_owner->decode_mode) {
      prefill_arena_source = g_graph_prefill_arena_owner.get();
    } else if (has_graph_prefill && g_graph && g_graph->ready) {
      // The active prefill graph owns the dense/expert arenas. Retain it as
      // the owner while the decode graph is constructed and used as borrower.
      g_graph_prefill_arena_owner = std::move(g_graph);
      prefill_arena_source = g_graph_prefill_arena_owner.get();
    }
    if (!prefill_arena_source) {
      LOG_WARN("[GGML-QWEN-FULL-DECODE] arena borrowing requested without a "
               "live prefill arena owner; using the ordinary rebuild path\n");
    } else {
      LOG_INFO("[GGML-QWEN-FULL-DECODE] borrowing dense/expert arenas from "
               "the completed prefill graph\n");
    }
  }
  prefill_state_source = g_graph_prefill_state_owner
                             ? g_graph_prefill_state_owner.get()
                             : prefill_arena_source;
  // The prefill graph has finished executing.  When its recurrent/PLE output
  // tensors can be handed directly to decode, keep the scheduler alive: its
  // activation buffer owns those physical addresses.  Otherwise release the
  // completed scheduler workspace and use the established host-state seed
  // path.  The latter remains necessary for KV because prefill allocates only
  // S rows while decode reserves the full context capacity.
  const bool prefill_state_handoff =
      env_prefill_state_handoff_enabled() && prefill_state_source &&
      prefill_state_source->prefill_host_state_deferred &&
      prepare_prefill_state_handoff(*prefill_state_source);
  if (prefill_state_source && prefill_state_source->prefill_host_state_deferred &&
      !prefill_state_handoff &&
      !sync_prefill_recurrent_state_to_host(*prefill_state_source))
    return false;
  if (prefill_arena_source && prefill_arena_source->sched &&
      !prefill_state_handoff) {
    ggml_backend_sched_free(prefill_arena_source->sched);
    prefill_arena_source->sched = nullptr;
    LOG_INFO("[GGML-QWEN-FULL-DECODE] released completed prefill scheduler "
             "workspace; retained dense/expert arenas for decode\n");
  } else if (prefill_state_handoff) {
    LOG_INFO("[GGML-QWEN-FULL-DECODE] retained prefill scheduler workspace "
             "for direct recurrent-state handoff\n");
  }
  {
    size_t free_before = 0, total_before = 0;
    cudaMemGetInfo(&free_before, &total_before);
    g_graph.reset();
    size_t free_after = 0, total_after = 0;
    cudaMemGetInfo(&free_after, &total_after);
    LOG_INFO("[GGML-QWEN-FULL-VRAM-TRACE] pre-reset free=%.3f GiB, "
             "post-reset free=%.3f GiB (freed %.3f GiB%s)\n",
             free_before / (1024.0 * 1024.0 * 1024.0),
             free_after / (1024.0 * 1024.0 * 1024.0),
             (static_cast<double>(free_after) -
              static_cast<double>(free_before)) /
                 (1024.0 * 1024.0 * 1024.0),
             prefill_arena_source ? "; prefill arenas retained" : "");
  }

  // How many trailing device_expert_layers to give up on this attempt,
  // relative to the configured/env value -- only ever incremented by the
  // plain (Phase2Mode::Off) retry loop further down, after build_graph
  // itself reports (via g_qwen4exp_build_graph_retry_smaller_expert_layers)
  // that the scheduler's own admission floor, not device_expert_layers
  // itself, was the reason construction was declined.
  int device_expert_layers_shrink = 0;

  // Builds ONE decode FullGraph instance -- the ordinary single-instance
  // decode graph when mode==Off, or one leg of Phase 2's 3-instance split
  // otherwise. Identical to the pre-Phase-2 body of this function in every
  // other respect (same backends/arenas/scheduler setup, same PLE/state
  // seeding), factored out only so Phase 2 can call it up to three times.
  auto build_one = [&](FullGraph *state_source,
                       FullGraph *dense_source,
                       FullGraph *expert_source) -> std::unique_ptr<FullGraph> {
    const auto build_one_started = std::chrono::steady_clock::now();
    auto setup_stage_started = build_one_started;
    double cpu_setup_ms = 0.0;
    double arena_setup_ms = 0.0;
    double scheduler_setup_ms = 0.0;
    double ple_setup_ms = 0.0;
    double graph_build_ms = 0.0;
    double state_seed_ms = 0.0;
    auto g = std::make_unique<FullGraph>();
    g->graph_generation =
        g_full_graph_generation.fetch_add(1, std::memory_order_relaxed);
    if (g->graph_generation == 0)
      g->graph_generation =
          g_full_graph_generation.fetch_add(1, std::memory_order_relaxed);
    g->engine = &engine;
    g->device = dev;
    g->S = 1;
    g->D = c.hidden;
    g->HC = engine.family_ops.family == ModelFamily::Qwen ? 1 : c.hc_mult;
    g->n_layers = c.n_layers;
    g->H = c.n_heads;
    g->Hkv = std::max(1, c.n_head_kv);
    g->hd = c.head_dim;
    g->Hk_gdn = c.ssm_group_count;
    g->Hv_gdn = c.ssm_time_step_rank;
    g->gdn_hd = c.ssm_state_size;
    g->key_dim = g->Hk_gdn * g->gdn_hd;
    g->value_dim = c.ssm_inner_size;
    g->conv_dim = 2 * g->key_dim + g->value_dim;
    g->d_conv = c.ssm_conv_kernel;
    g->decode_mode = true;
    g->kv_capacity = env_decode_kv_capacity(engine.max_t, required);
    g->decode_last_pos = pos_base - 1;
    // device_expert_layers_shrink: only ever nonzero when the caller (the
    // plain Phase2Mode::Off build_one() call below) is retrying after
    // build_graph's own scheduler-admission-floor decline -- see
    // g_qwen4exp_build_graph_retry_smaller_expert_layers's own comment.
    // Phase 2's prefix/suffix build_one() calls never set it, so this is a
    // no-op for them.
    if (engine.family_ops.family == ModelFamily::Qwen) {
      // A full-fit Qwen planner commits the complete expert set to the graph.
      // Do not turn a graph allocation failure into an implicit partial-MoE
      // residency retry; that run must fail closed.
      g->device_expert_layers = std::clamp(
          engine.cuda_moe_graph_device_expert_layers, 0, c.n_layers);
      g->device_expert_trailing =
          engine.cuda_moe_graph_device_expert_trailing;
    } else {
      g->device_expert_layers =
          std::max(0, std::min(qwen4exp_full_graph_device_expert_layers(),
                               c.n_layers) -
                          device_expert_layers_shrink);
      g->device_expert_trailing = env_device_expert_trailing();
    }
    const char *partial_gate_up =
        zeromoe::graphcfg::env("ZEROMOE_QWEN4EXP_GRAPH_PRELOAD_PARTIAL_GATE_UP");
    // partial_gate_up's boundary math (layer==device_expert_layers, the
    // first CPU layer just past a LEADING block) has no trailing-mode
    // equivalent yet -- force it off rather than apply leading-only math
    // to a trailing block. See device_expert_trailing's own comment.
    g->device_expert_partial_gate_up = g->device_expert_layers > 0 &&
                                       !g->device_expert_trailing &&
                                       partial_gate_up &&
                                       partial_gate_up[0] == '1';
    g->state_source_graph = state_source;
    g->dense_source_graph = dense_source;
    g->expert_source_graph = expert_source;
    plan_expert_vram_split(*g); // must run before init_device_expert_arena
    g->cuda_raw = static_cast<ggml_backend_t>(ggml_shared_backend(dev));
    g->cuda = full_graph_cuda_scheduler_backend(*g);
    g->cpu = ggml_backend_cpu_init();
    if (!g->cuda_raw || !g->cuda || !g->cpu)
      return nullptr;
    verify_cuda_graphs_compiled_in(g->cuda_raw);
    g->cpu_threads = graph_cpu_threads();
    ggml_backend_cpu_set_n_threads(g->cpu, g->cpu_threads);
    ggml_threadpool_params cpu_tp_params =
        ggml_threadpool_params_default(g->cpu_threads);
    pin_host_node_threads(cpu_tp_params, g->cpu_threads);
    // 2026-09-15: tested ggml's own threadpool .poll (default 50 -- NOT
    // controlled by OMP_WAIT_POLICY/GOMP_SPINCOUNT, which only affect
    // libgomp's separate pool) as a candidate for the mixed CPU/GPU decode
    // regression via a temporary env override swept across 0/50/400/1000.
    // Ruled out via THROUGHPUT: all four values landed within noise of each
    // other (11.76-12.37 tok/s at device_expert_layers=20).
    // TEMPORARY re-test (2026-09-15, same day): now that the ELEVENTH ROUND
    // per-op gap instrumentation exists (ZEROMOE_QWEN4EXP_GRAPH_LAYER_
    // TIMING=1), re-checking whether .poll moves the GAP time specifically
    // (a real but small effect could have been lost in the earlier
    // throughput-only measurement's noise). Remove this override again if
    // inconclusive, per the original decision record's own discipline.
    if (const char *poll_env =
            zeromoe::graphcfg::env("ZEROMOE_QWEN4EXP_CPU_POLL_RETEST")) {
      char *end = nullptr;
      const long v = std::strtol(poll_env, &end, 10);
      if (end != poll_env && v >= 0 && v <= 100)
        cpu_tp_params.poll = static_cast<uint32_t>(v);
    }
    // ggml pins the creating thread to the pool's first CPU; keep the main
    // thread's own affinity unchanged (only the callback team should be pinned).
    cpu_set_t main_affinity;
    CPU_ZERO(&main_affinity);
    const bool have_main_affinity =
        sched_getaffinity(0, sizeof(main_affinity), &main_affinity) == 0;
    g->cpu_threadpool = ggml_threadpool_new(&cpu_tp_params);
    if (have_main_affinity && cpu_tp_params.strict_cpu)
      sched_setaffinity(0, sizeof(main_affinity), &main_affinity);
    if (g->cpu_threadpool)
      ggml_backend_cpu_set_threadpool(g->cpu, g->cpu_threadpool);
    cpu_setup_ms = std::chrono::duration<double, std::milli>(
                       std::chrono::steady_clock::now() - setup_stage_started)
                       .count();
    setup_stage_started = std::chrono::steady_clock::now();
    LOG_INFO("[GGML-QWEN-FULL-DECODE] CPU backend configured with %d "
             "threads%s\n",
             g->cpu_threads,
             g->cpu_threadpool ? " (persistent threadpool)" : "");

    // Every expert view aliases the already populated DDR graph arena, so
    // this mode has no graph-owned expert source admission or GGUF expert
    // read.  Phase-2 can nevertheless request a large compact split arena;
    // reject it before cudaMalloc when the measured free VRAM cannot cover
    // the estimate plus a scheduler margin.  The old allocator path has
    // historically turned such a probe into a process-fatal OOM.
    if (!init_device_dense_arena(*g) || !init_device_state_arena(*g))
      return nullptr;
    const bool compact_arena_planned =
        g->device_expert_layers > 0 || !g->split_gpu_expert_ids.empty() ||
        (!engine.qwen4exp_partial_layer_gpu_ids.empty() &&
         engine.qwen4exp_partial_layer >= 0);
    if (compact_arena_planned && !expert_source) {
      size_t free_bytes = 0, total_bytes = 0;
      const cudaError_t mem_status = cudaMemGetInfo(&free_bytes, &total_bytes);
      constexpr size_t kPhase2ArenaMargin = 768ull * 1024ull * 1024ull;
      const size_t expert_need = device_expert_bytes(*g);
      const bool risky_override =
          zeromoe::graphcfg::env("ZEROMOE_QWEN4EXP_PHASE2_ALLOW_RISKY_ALLOC") != nullptr;
      const bool insufficient =
          mem_status != cudaSuccess ||
          expert_need > SIZE_MAX - kPhase2ArenaMargin ||
          free_bytes < expert_need + kPhase2ArenaMargin;
      if (insufficient && !risky_override) {
        qwen4exp_memory_snapshot("decode.vram_admission.declined", g->device,
                                 "compact_expert_arena");
        LOG_WARN("[GGML-QWEN-FULL] declining decode graph before expert "
                 "arena allocation: free=%.3f GiB need=%.3f GiB margin="
                 "%.3f GiB status=%s\n",
                 mem_status == cudaSuccess
                     ? static_cast<double>(free_bytes) /
                           (1024.0 * 1024.0 * 1024.0)
                     : -1.0,
                 static_cast<double>(expert_need) /
                     (1024.0 * 1024.0 * 1024.0),
                 static_cast<double>(kPhase2ArenaMargin) /
                     (1024.0 * 1024.0 * 1024.0),
                 cudaGetErrorString(mem_status));
        return nullptr;
      }
    }
    if (!init_device_expert_arena(*g))
      return nullptr;
    arena_setup_ms = std::chrono::duration<double, std::milli>(
                         std::chrono::steady_clock::now() - setup_stage_started)
                         .count();
    setup_stage_started = std::chrono::steady_clock::now();
    ggml_backend_t sched_cuda = full_graph_cuda_scheduler_backend(*g);
    ggml_backend_t backends[] = {sched_cuda, g->cpu};
    ggml_backend_buffer_type_t sched_bufts[2] = {};
    ggml_backend_buffer_type_t *sched_bufts_ptr = nullptr;
    if (graph_cuda_host_buffers()) {
      sched_bufts[0] = ggml_backend_get_default_buffer_type(g->cuda);
      sched_bufts[1] = ggml_backend_dev_host_buffer_type(
          ggml_backend_get_device(g->cuda));
      if (sched_bufts[1]) {
        sched_bufts_ptr = sched_bufts;
        LOG_INFO("[GGML-QWEN-FULL-DECODE] scheduler CPU buffer=%s (CUDA "
                 "host buffer)\n",
                 ggml_backend_buft_name(sched_bufts[1]));
      }
    }
    // Match the prefill scheduler mode above; event/four-copy mode is opt-in.
    // op_offload MUST stay false -- see the prefill call site's own comment
    // (qwen4exp_full_graph_prepare, above) for the real S>=32 OOM/crash this
    // prevents. Decode is always S==1 so this flag is a no-op for it either
    // way; kept false here purely for consistency with the shared scheduler
    // contract, not because decode itself is at risk.
    g->sched =
        ggml_backend_sched_new(backends, sched_bufts_ptr, 2, kGraphSize,
                               graph_scheduler_parallel(), false);
    if (!g->sched)
      return nullptr;
    LOG_INFO(
        "[GGML-QWEN-FULL-DECODE] scheduler pipeline_parallel=%s copies=%d\n",
        graph_scheduler_parallel() ? "true" : "false",
        ggml_backend_sched_get_n_copies(g->sched));
    // See the prefill graph above: GRAPH_REPLAY_TIMING is deliberately
    // non-invasive and must not enable the scheduler eval callback.
    g->layer_timing_enabled = env_layer_timing_enabled();
    g->layer_timing_verbose_enabled = env_layer_timing_verbose();
    if (g->layer_timing_enabled || g->layer_timing_verbose_enabled) {
      g->layer_timing_cpu_ms.assign(static_cast<size_t>(g->n_layers), 0.0);
      ggml_backend_sched_set_eval_callback(g->sched, layer_timing_eval_callback,
                                           g.get());
    }

    scheduler_setup_ms = std::chrono::duration<double, std::milli>(
                             std::chrono::steady_clock::now() -
                             setup_stage_started)
                             .count();
    setup_stage_started = std::chrono::steady_clock::now();
    std::vector<float> ple;
    if (!prepare_ple_embedding(*g, ple, 1, pos_base, token_ids) &&
        has_ple_layer(c)) {
      LOG_ERROR("[GGML-QWEN-FULL-DECODE] graph preparation failed: PLE "
                "embedding preparation declined\n");
      return nullptr;
    }
    ple_setup_ms = std::chrono::duration<double, std::milli>(
                       std::chrono::steady_clock::now() - setup_stage_started)
                       .count();
    setup_stage_started = std::chrono::steady_clock::now();
    qwen4exp_memory_snapshot("decode.build_graph.before", g->device, "");
    if (!build_graph(*g, has_ple_layer(c) ? &ple : nullptr)) {
      qwen4exp_memory_snapshot("decode.build_graph.failed", g->device, "");
      // The plain (Phase2Mode::Off) caller retries at a smaller
      // device_expert_layers when this is set, so "layered decode remains
      // active" would be misleading here -- it only becomes true if that
      // retry loop itself gives up. Phase 2 callers never retry, so for them
      // the message is unconditionally accurate either way.
      if (g_qwen4exp_build_graph_retry_smaller_expert_layers) {
        LOG_WARN("[GGML-QWEN-FULL-DECODE] graph construction declined "
                 "retrying at a smaller "
                 "device_expert_layers before falling back to layered "
                 "decode\n");
      } else {
        LOG_ERROR("[GGML-QWEN-FULL-DECODE] graph construction failed "
                  "request cannot continue\n");
      }
      return nullptr;
    }
    qwen4exp_memory_snapshot("decode.build_graph.after", g->device, "");
    graph_build_ms = std::chrono::duration<double, std::milli>(
                         std::chrono::steady_clock::now() - setup_stage_started)
                         .count();
    setup_stage_started = std::chrono::steady_clock::now();

    // Seed the device state once from the state/cache that the existing
    // prefill or layered path produced. This is a one-time boundary
    // transfer; later decode submissions update these tensors in the graph
    // itself. Skipped when state_source is set: that physical memory was
    // already seeded when the source instance was built (see
    // new_device_state_tensor's replay path -- writing it again here would
    // just re-write the identical bytes through an aliased tensor handle).
    {
      const size_t state_n =
          static_cast<size_t>(g->gdn_hd) * g->gdn_hd * g->Hv_gdn;
      const size_t hist_n = static_cast<size_t>(g->d_conv - 1) * g->conv_dim;
      for (size_t i = 0; i < g->gdn_layers.size(); ++i) {
        const int l = g->gdn_layers[i];
        const bool hist_aliased =
            g->state_handoff_tensors.count(g->gdn_hist_storage[i]) != 0;
        const bool state_aliased =
            g->state_handoff_tensors.count(g->gdn_state_storage[i]) != 0;
        if (hist_aliased && state_aliased)
          continue;
        if (l < 0 ||
            l >= static_cast<int>(engine.qwen4exp_recur_state.size()) ||
            l >= static_cast<int>(engine.qwen4exp_conv_history.size()) ||
            engine.qwen4exp_recur_state[l].size() != state_n ||
            engine.qwen4exp_conv_history[l].size() != hist_n)
          return nullptr;
        // qwen4exp_recur_state is host-legacy layout; gdn_state_storage
        // feeds the decode self-loop's ggml op directly and needs native
        // layout. See gdn_state_transpose_host.
        if (!state_aliased) {
          std::vector<float> native_state(state_n);
          gdn_state_transpose_host(native_state.data(),
                                   engine.qwen4exp_recur_state[l].data(),
                                   g->gdn_hd, g->Hv_gdn);
          ggml_backend_tensor_set(g->gdn_state_storage[i], native_state.data(),
                                  0, state_n * sizeof(float));
        }
        if (!hist_aliased)
          ggml_backend_tensor_set(g->gdn_hist_storage[i],
                                  engine.qwen4exp_conv_history[l].data(), 0,
                                  hist_n * sizeof(float));
      }
      const size_t kv_row = static_cast<size_t>(g->Hkv) * g->hd;
      const size_t kv_bytes =
          static_cast<size_t>(pos_base) * kv_row * sizeof(float);
      for (size_t i = 0; i < g->kv_layers.size(); ++i) {
        const int l = g->kv_layers[i];
        if (i >= g->kv_state_k.size() || i >= g->kv_state_v.size() ||
            l < 0 || l >= static_cast<int>(engine.Kc.size()) ||
            l >= static_cast<int>(engine.Vc.size()) || !engine.Kc[l] ||
            !engine.Vc[l])
          return nullptr;
        if (kv_bytes > 0) {
          if (g->state_handoff_tensors.count(g->kv_state_k[i]) == 0)
            ggml_backend_tensor_set(g->kv_state_k[i], engine.Kc[l], 0,
                                    kv_bytes);
          if (g->state_handoff_tensors.count(g->kv_state_v[i]) == 0)
            ggml_backend_tensor_set(g->kv_state_v[i], engine.Vc[l], 0,
                                    kv_bytes);
        }
      }
      if (g->ple_layer >= 0 && g->ple_hist_storage) {
        const size_t ple_n = static_cast<size_t>(c.ple_conv_kernel - 1) *
                             c.ple_ngram_size * g->HC * g->D;
        if (g->ple_layer >=
                static_cast<int>(engine.qwen4exp_ple_conv_history.size()) ||
            engine.qwen4exp_ple_conv_history[g->ple_layer].size() != ple_n)
          return nullptr;
        if (g->state_handoff_tensors.count(g->ple_hist_storage) == 0)
          ggml_backend_tensor_set(
              g->ple_hist_storage,
              engine.qwen4exp_ple_conv_history[g->ple_layer].data(), 0,
              ple_n * sizeof(float));
      }
    }
    state_seed_ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - setup_stage_started)
                        .count();
    g->ready = true;
    const double total_setup_ms = std::chrono::duration<double, std::milli>(
                                      std::chrono::steady_clock::now() -
                                      build_one_started)
                                      .count();
    LOG_INFO("[GGML-QWEN-FULL-DECODE-SETUP] backend=%.1fms "
             "arenas=%.1fms scheduler=%.1fms ple=%.1fms graph=%.1fms "
             "state_seed=%.1fms total=%.1fms\n",
             cpu_setup_ms, arena_setup_ms,
             scheduler_setup_ms, ple_setup_ms, graph_build_ms, state_seed_ms,
             total_setup_ms);
    return g;
  };


  // device_expert_layers's arena-allocation stage already retries at a
  // smaller layer count when the ARENA itself doesn't fit
  // (init_device_expert_arena's own allow_layer_clamp loop) -- but the
  // scheduler's own admission-floor check further inside build_graph runs
  // AFTER that arena is already committed, and until now had no retry of
  // its own: a config that needed just one or two fewer device_expert_layers
  // to leave room for the scheduler's workspace fell all the way back to the
  // legacy per-layer decode path instead (a real, ~4x-plus throughput
  // cliff, not a graceful one-layer-at-a-time degradation). Real, reproduced
  // case: qwen4exp_option_b_selected_layers with 6+ split layers at
  // device_expert_layers=19 on a 24 GiB card, 2026-09-26.
  std::unique_ptr<FullGraph> g;
  if (prefill_arena_source) {
    // The borrowed arena fixes the prefill's resident layout, so the
    // ordinary retry loop cannot shrink device_expert_layers meaningfully.
    // If the borrower graph does not fit or fails to build, release the owner
    // before returning to the established layered path.
    // Borrow only what an owner really holds. The prefill owner has an expert
    // arena only for whole-layer residency; a per-expert split arena is
    // decode-only and comes from the decode arena owner (built at bootstrap).
    // With neither, the expert arena is allocated fresh below (same as the
    // recreate path) while the dense arena is still borrowed.
    FullGraph *expert_source = nullptr;
    if (prefill_arena_source->expert.buffer) {
      expert_source = prefill_arena_source;
    } else if (g_graph_decode_arena_owner && g_graph_decode_arena_owner->ready &&
               g_graph_decode_arena_owner->engine == &engine &&
               g_graph_decode_arena_owner->device == dev &&
               g_graph_decode_arena_owner->expert.buffer) {
      expert_source = g_graph_decode_arena_owner.get();
      LOG_INFO("[GGML-QWEN-FULL-DECODE] rebinding the per-expert split arena "
               "prepared at startup (no re-upload)\n");
    }
    g = build_one(prefill_state_handoff ? prefill_state_source : nullptr,
                  prefill_arena_source, expert_source);
    if (!g) {
      if (prefill_state_source && prefill_state_source->prefill_host_state_deferred &&
          !sync_prefill_recurrent_state_to_host(*prefill_state_source))
        return false;
      // Never fall to the layered path with half-reused arenas (that produced
      // garbled output on split configs): drop every owner and rebuild fresh.
      g_graph_prefill_state_owner.reset();
      g_graph_prefill_arena_owner.reset();
      g_graph_decode_arena_owner.reset();
      prefill_arena_source = nullptr;
      LOG_WARN("[GGML-QWEN-FULL-DECODE] borrowed-arena graph build failed; "
               "released retained arenas and rebuilding the decode graph "
               "with fresh arenas\n");
    }
  }
  if (!g) {
    constexpr int kMaxExpertLayerShrinkRetries = 8;
    for (int attempt = 0; attempt <= kMaxExpertLayerShrinkRetries; ++attempt) {
      g = build_one(nullptr, nullptr, nullptr);
      if (g)
        break;
      if (!g_qwen4exp_build_graph_retry_smaller_expert_layers)
        return false; // some other, non-retry-able construction failure
      ++device_expert_layers_shrink;
      LOG_WARN("[GGML-QWEN-FULL-DECODE] retrying with device_expert_layers "
               "shrunk by %d (scheduler admission floor needed more headroom "
               "than the arena-only check reserved)\n",
               device_expert_layers_shrink);
    }
    if (!g)
      return false;
  }
  if (engine.family_ops.family == ModelFamily::Qwen &&
      (g->device_expert_layers != c.n_layers ||
       !g->split_gpu_expert_ids.empty())) {
    LOG_ERROR("[QWEN35MOE-GRAPH] planner selected a full-fit run, but graph "
              "placement is incomplete (device_layers=%d/%d split_layers=%zu); "
              "refusing partial expert residency\n",
              g->device_expert_layers, c.n_layers,
              g->split_gpu_expert_ids.size());
    return false;
  }
  if (engine.family_ops.family == ModelFamily::Qwen) {
    LOG_INFO("[QWEN35MOE-GRAPH] decode ready: expert_layers_on_gpu=%d/%d "
             "experts_per_layer=%d expert_arena_bytes=%zu dense_arena_bytes=%zu "
             "kv_capacity=%d; no CPU expert or layered decode path\n",
             g->device_expert_layers, c.n_layers, c.n_experts,
             g->expert.capacity, g->dense.capacity, g->kv_capacity);
  } else {
    LOG_INFO("[GGML-QWEN-FULL-DECODE] enabled: one scheduler submission per "
             "token, KV capacity=%d, device GDN/KV/PLE state, DDR expert "
             "arenas=%d\n",
             g->kv_capacity, c.n_layers);
  }
  g_graph = std::move(g);
  LOG_INFO("[GGML-QWEN-FULL-DECODE-SETUP] prepare_total=%.1fms "
           "(function entry through ready graph)\n",
           std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - decode_prepare_started)
               .count());
  return true;
}

namespace qwen4exp_graph {

bool qwen4exp_stage_layer13_before_graph(FullGraph &g, int pos_base) {
  if (!g.stage_enabled)
    return true;
  if (!g.engine || !g.engine->gpu || g.stage_bindings.empty())
    return false;
  if (g.stage_layer < 0 ||
      static_cast<size_t>(g.stage_layer) >= g.engine->graph_ram_arenas.size())
    return false;

  // The proof deliberately requires a pinned DDR source.  Pin only the
  // target layer if the normal graph-arena registration A/B was not enabled;
  // this keeps the proof self-contained without pinning the other 47 arenas.
  auto &arena = g.engine->graph_ram_arenas[static_cast<size_t>(g.stage_layer)];
  if (!arena.pinned) {
    if (g.engine->ram_pinned_budget_bytes == 0 ||
        arena.allocation_base == nullptr || arena.allocation_bytes == 0)
      return false;
    const size_t previous = g.engine->ram_pinned_bytes_used.fetch_add(
        arena.bytes, std::memory_order_relaxed);
    if (previous > g.engine->ram_pinned_budget_bytes ||
        arena.bytes > g.engine->ram_pinned_budget_bytes - previous) {
      g.engine->ram_pinned_bytes_used.fetch_sub(arena.bytes,
                                                std::memory_order_relaxed);
      LOG_WARN("[OPTION-B-STAGE] Layer %d source pin budget exhausted\n",
               g.stage_layer);
      return false;
    }
    cudaSetDevice(g.device);
    const cudaError_t pin_err = cudaHostRegister(
        arena.allocation_base, arena.allocation_bytes, cudaHostRegisterPortable);
    if (pin_err != cudaSuccess) {
      g.engine->ram_pinned_bytes_used.fetch_sub(arena.bytes,
                                                std::memory_order_relaxed);
      cudaGetLastError();
      LOG_WARN("[OPTION-B-STAGE] Layer %d source pin failed: %s\n",
               g.stage_layer, cudaGetErrorString(pin_err));
      return false;
    }
    arena.pinned = true;
    LOG_INFO("[OPTION-B-STAGE] pinned Layer %d DDR arena: %.3f GiB\n",
             g.stage_layer,
             static_cast<double>(arena.bytes) /
                 (1024.0 * 1024.0 * 1024.0));
  }

  cudaSetDevice(g.device);
  if (!g.stage_wait_done_event) {
    if (cudaEventCreate(&g.stage_wait_start_event) != cudaSuccess ||
        cudaEventCreate(&g.stage_wait_done_event) != cudaSuccess ||
        cudaEventCreate(&g.stage_d2d_start_event) != cudaSuccess ||
        cudaEventCreate(&g.stage_d2d_end_event) != cudaSuccess ||
        cudaEventCreate(&g.stage_compute_start_event) != cudaSuccess ||
        cudaEventCreate(&g.stage_compute_end_event) != cudaSuccess)
      return false;
  }

  GpuTopology &topology = g.engine->gpu->topology;
  const auto host_issue_start = std::chrono::steady_clock::now();
  g.stage_slots.clear();
  for (auto &binding : g.stage_bindings) {
    ExpertLocation *slot = topology.upload_expert_active_projections_async(
        g.stage_layer, binding.expert, binding.src_gate, binding.gate_bytes,
        binding.src_up, binding.up_bytes, binding.src_down,
        binding.down_bytes, g.device);
    if (!slot || !slot->upload_event) {
      LOG_WARN("[OPTION-B-STAGE] failed to queue Layer %d expert %d upload "
               "(%zu requested)\n",
               g.stage_layer, binding.expert, g.stage_bindings.size());
      return false;
    }
    binding.slot = slot;
    g.stage_slots.push_back(slot);
  }
  g.stage_slot = g.stage_slots.front();

  cudaStream_t graph_stream =
      zeromoe_ggml_backend_cuda_get_stream(g.cuda_raw);
  if (!graph_stream) {
    LOG_WARN("[OPTION-B-STAGE] ggml CUDA stream bridge unavailable\n");
    return false;
  }
  if (cudaEventRecord(g.stage_wait_start_event, graph_stream) != cudaSuccess)
    return false;
  for (const auto *slot : g.stage_slots) {
    if (cudaStreamWaitEvent(graph_stream, slot->upload_event, 0) !=
        cudaSuccess)
      return false;
  }
  if (cudaEventRecord(g.stage_wait_done_event, graph_stream) != cudaSuccess ||
      cudaEventRecord(g.stage_d2d_start_event, graph_stream) != cudaSuccess)
    return false;
  for (const auto &binding : g.stage_bindings) {
    const auto *slot = binding.slot;
    if (cudaMemcpyAsync(binding.dst_gate, slot->device_ptr,
                        binding.gate_bytes, cudaMemcpyDeviceToDevice,
                        graph_stream) != cudaSuccess ||
        cudaMemcpyAsync(binding.dst_up,
                        slot->device_ptr + binding.gate_bytes,
                        binding.up_bytes, cudaMemcpyDeviceToDevice,
                        graph_stream) != cudaSuccess ||
        cudaMemcpyAsync(binding.dst_down,
                        slot->device_ptr + binding.gate_bytes + binding.up_bytes,
                        binding.down_bytes, cudaMemcpyDeviceToDevice,
                        graph_stream) != cudaSuccess)
      return false;
  }
  if (cudaEventRecord(g.stage_d2d_end_event, graph_stream) != cudaSuccess ||
      cudaEventRecord(g.stage_compute_start_event, graph_stream) !=
          cudaSuccess)
    return false;
  const double host_issue_us = std::chrono::duration<double, std::micro>(
                                   std::chrono::steady_clock::now() -
                                   host_issue_start)
                                   .count();
  LOG_INFO("[OPTION-B-STAGE] pos=%d queued layer=%d experts=%zu "
           "host_issue=%.3fus source=%s\n",
           pos_base, g.stage_layer, g.stage_slots.size(), host_issue_us,
           g.stage_slots.front()->upload_source_pinned ? "pinned" : "pageable");
  return true;
}

} // namespace qwen4exp_graph

#endif // ZEROMOE_USE_GGML_CUDA
