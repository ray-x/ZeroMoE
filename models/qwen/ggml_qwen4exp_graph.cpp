// Qwen4Exp full graph: Public lifecycle API: enabled/configure/bootstrap/prepare/shutdown, arena
// ownership between the prefill and decode graphs, global state.
//
// Split out of the former single-file ggml_qwen4exp_graph.cpp; shared
// declarations live in ggml_qwen4exp_graph_internal.hpp.

#include "ggml_qwen4exp_graph_internal.hpp"
#include "request_stats.hpp"

#if defined(ZEROMOE_USE_GGML_CUDA)

using namespace qwen4exp_graph;

namespace qwen4exp_graph {

// The CUDA host callback and the ordinary GGML CPU scheduler otherwise use
// independent execution contexts.  During the callback's CPU critical
// section, starting another CPU split causes the two pools to compete for
// the same cores and memory bandwidth.  The scheduler-side weak hook (added
// by patches/ggml-backend-host-node-gate.patch) waits on this counter.  It is
// deliberately process-global because a replay has one host callback stream
// per active MoE marker, and the guarded region is short.
std::atomic<unsigned> g_qwen4exp_host_node_cpu_active{0};

std::atomic<uint64_t> g_qwen4exp_host_node_cpu_gate_wait_ns{0};

std::atomic<unsigned> g_qwen4exp_ordinary_cpu_active{0};

std::atomic<uint64_t> g_qwen4exp_host_node_ordinary_cpu_wait_ns{0};

std::atomic<uint64_t> g_qwen4exp_cpu_split_enter_wait_ns{0};

std::mutex g_qwen4exp_cpu_execution_mutex;

// Set by either of two independent VRAM-exhaustion sites -- init_device_
// expert_arena's own allocation-declined path (real cudaMalloc failure, or a
// pre-check that predicts one) and build_graph's later scheduler-admission-
// floor check -- just before each declines specifically because there isn't
// enough VRAM for the CURRENT device_expert_layers value, with room left to
// shrink it further. Both call sites reach it as a plain global rather than
// a return value because both are deep inside functions the plain
// (Phase2Mode::Off) decode path's build_one()/build_graph() call chain
// already tears down and rebuilds FROM SCRATCH on any failure -- so "was
// THIS specific, retry-worth-it reason why construction just failed" only
// needs a cheap signal read right after build_one() returns nullptr, not a
// return-value plumbed through every intermediate call. Reset at the top of
// every build_graph() call so a stale true from an earlier, unrelated
// attempt can never leak into a later retry decision it doesn't apply to.
thread_local bool
    g_qwen4exp_build_graph_retry_smaller_expert_layers = false;

std::unique_ptr<FullGraph> g_graph;

std::atomic<uint64_t> g_full_graph_generation{1};

// Optional owner for the immutable dense/expert device arenas shared by a
// decode graph after full-graph prefill. The active decode graph borrows these
// buffers through the same source-graph bindings used by Phase 2 siblings.
// Keep the owner alive until the borrower has been destroyed.
std::unique_ptr<FullGraph> g_graph_prefill_arena_owner;

std::unique_ptr<FullGraph> g_graph_prefill_state_owner;

// Owner of the decode-only per-expert split arena (Option-B hot experts gathered
// into compact VRAM tensors). The prefill graph never allocates it (split tensors
// exist only for S == 1), so the prefill arena owner cannot lend it to decode. It
// is built once at startup (qwen4exp_full_graph_bootstrap) so the first request's
// decode graph rebinds the already-uploaded device addresses instead of re-uploading
// ~13 GiB (~2.7 s). Holds only the expert arena and its address map.
std::unique_ptr<FullGraph> g_graph_decode_arena_owner;



bool g_config_full_graph = false;

bool g_config_full_graph_decode = false;

bool g_config_prefill_state_handoff = false;

ggml_backend_t full_graph_cuda_scheduler_backend(FullGraph &g) {
  if (!g.engine || !g.engine->exl3_gpu_executor)
    return g.cuda_raw;
  if (!g.exl3_cuda_adapter) {
    g.exl3_cuda_adapter = exl3_ggml_cuda_backend_wrap(g.cuda_raw, false);
    if (!g.exl3_cuda_adapter &&
        !g.engine->exl3_ggml_bridge_model_path.empty()) {
      LOG_ERROR("[EXL3-GGML-BRIDGE] CUDA backend adapter creation failed; "
                "refusing to fall back to GGUF experts\n");
      g.exl3_bridge_failed = true;
    }
    if (g.exl3_cuda_adapter && g.engine->exl3_ggml_graph_capture &&
        !exl3_ggml_cuda_backend_set_graph_capture(
            g.exl3_cuda_adapter, true)) {
      LOG_ERROR("[EXL3-GGML-BRIDGE] failed to configure CUDA graph capture\n");
      ggml_backend_free(g.exl3_cuda_adapter);
      g.exl3_cuda_adapter = nullptr;
      if (!g.engine->exl3_ggml_bridge_model_path.empty())
        g.exl3_bridge_failed = true;
    } else if (g.exl3_cuda_adapter &&
               g.engine->exl3_ggml_graph_capture) {
      LOG_INFO("[EXL3-GGML-BRIDGE] segmented CUDA graph capture enabled\n");
    }
  }
  return g.exl3_cuda_adapter ? g.exl3_cuda_adapter : g.cuda_raw;
}

} // namespace qwen4exp_graph

bool qwen4exp_full_graph_enabled() { return env_enabled(); }

int qwen4exp_full_graph_device_expert_layers() {
  return env_enabled() ? env_device_expert_layers() : 0;
}

bool qwen4exp_full_graph_device_expert_trailing() {
  return env_enabled() && env_device_expert_trailing();
}

bool qwen4exp_full_graph_decode_enabled() {
  return env_enabled() && env_decode_enabled();
}

void qwen4exp_full_graph_configure(bool enabled, bool decode_enabled,
                                   bool prefill_state_handoff) {
  g_config_full_graph = enabled;
  g_config_full_graph_decode = decode_enabled;
  g_config_prefill_state_handoff = prefill_state_handoff;
}

bool qwen4exp_full_graph_sync_state_to_host(StreamingEngine &engine) {
  FullGraph *prefill_state = g_graph_prefill_state_owner
                                 ? g_graph_prefill_state_owner.get()
                                 : g_graph_prefill_arena_owner.get();
  const bool deferred_prefill =
      prefill_state && prefill_state->engine == &engine &&
      prefill_state->prefill_host_state_deferred;
  if (g_graph && g_graph->ready && g_graph->engine == &engine) {
    if (g_graph->decode_mode &&
        (g_graph->decode_host_state_dirty || deferred_prefill)) {
      if (!sync_decode_state_to_host(*g_graph))
        return false;
      if (deferred_prefill)
        prefill_state->prefill_host_state_deferred = false;
      g_graph.reset();
      LOG_INFO("[GGML-QWEN-FULL-STATE-HANDOFF] materialized decode state; "
               "remaining tokens use layered fallback\n");
      return true;
    }
    if (!g_graph->decode_mode && g_graph->prefill_host_state_deferred) {
      if (!sync_prefill_recurrent_state_to_host(*g_graph))
        return false;
      if (!g_graph_prefill_arena_owner)
        g_graph_prefill_arena_owner = std::move(g_graph);
      else
        g_graph_prefill_state_owner = std::move(g_graph);
      LOG_INFO("[GGML-QWEN-FULL-STATE-HANDOFF] materialized prefill state; "
               "remaining tokens use layered fallback\n");
      return true;
    }
  }
  if (deferred_prefill)
    return sync_prefill_recurrent_state_to_host(*prefill_state);
  return true;
}

void qwen4exp_full_graph_shutdown() {
  zeromoe::graphcfg::log_consulted("qwen4exp-full-graph");
  // Explicit, EARLY teardown (before the CUDA driver itself starts
  // shutting down) is why this function exists at all -- see g_graph's
  // own destroy()'s comment. The two Phase 2 suffix globals are exactly
  // as CUDA-backed as g_graph and must be reset here too, in this order
  // (suffix instances first): SuffixCpu's GDN/KV/PLE state tensors for
  // layers > split_layer are aliases into SuffixGpu's own
  // device_state_buffer (see FullGraph::state_source_graph), so
  // SuffixGpu must outlive SuffixCpu's teardown.
  g_graph.reset();
  g_graph_prefill_state_owner.reset();
  g_graph_prefill_arena_owner.reset();
  g_graph_decode_arena_owner.reset();
}

namespace qwen4exp_graph {

// True when the decode graph will build a per-expert hot/cold split (Option B):
// the same condition plan_expert_vram_split() uses to decide it has work to do.
bool engine_plans_decode_expert_split(const StreamingEngine &engine) {
  const bool option_b_auto = engine.qwen4exp_option_b_concentrated_only &&
                             engine.qwen4exp_option_b_layer < 0;
  return engine.qwen4exp_option_b_layer >= 0 || option_b_auto ||
         !engine.qwen4exp_option_b_layers.empty();
}

// Moves the per-expert split arena (and its address map) out of a decode graph
// into a standalone owner so it can outlive that graph. Returns null when the
// graph does not own such an arena. The graph must not free the arena afterwards.
std::unique_ptr<FullGraph> extract_decode_arena_owner(
    FullGraph &g, StreamingEngine &engine) {
  if (!g.ready || !g.decode_mode || !g.expert.buffer ||
      g.expert_source_graph || g.expert.capacity == 0)
    return nullptr;
  auto owner = std::make_unique<FullGraph>();
  owner->engine = &engine;
  owner->device = g.device;
  owner->ready = true;
  owner->expert.buffer = g.expert.buffer;
  owner->expert.base = g.expert.base;
  owner->expert.capacity = g.expert.capacity;
  owner->expert.used = g.expert.used;
  owner->expert_addrs = std::move(g.expert_addrs);
  owner->expert_soft = std::move(g.expert_soft);
  owner->expert_soft_of_layer = std::move(g.expert_soft_of_layer);
  owner->expert_buft = g.expert_buft;
  g.expert_soft.clear();
  g.expert_soft_of_layer.clear();
  owner->split_gpu_expert_ids = g.split_gpu_expert_ids;
  owner->device_expert_layers = g.device_expert_layers;
  owner->device_expert_trailing = g.device_expert_trailing;
  owner->device_expert_partial_gate_up = g.device_expert_partial_gate_up;
  g.expert.buffer = nullptr;
  g.expert.base = nullptr;
  g.expert.capacity = 0;
  g.expert.used = 0;
  return owner;
}

// The retained per-expert split arena (~13 GiB) makes the next decode graph
// cheap to build, but it competes with the prefill workspace for VRAM. Keep it
// resident for prompts whose workspace fits (and recover it from the live
// decode graph after a long prompt), release it before a prefill that needs the
// space. Called before the prefill graph is built.
void manage_decode_arena_for_prefill(StreamingEngine &engine, int S) {
  // The generation loop and the execution layer both prepare a prefill batch, so this can run again for a shape whose prefill
  // graph is already built. That decision was taken when the graph was built, and the graph's own workspace now makes the
  // free VRAM look small: re-deciding would release more of the arena (or all of it) for nothing.
  if (g_graph && g_graph->ready && !g_graph->decode_mode && g_graph->engine == &engine && g_graph->S == S)
    return;
  const bool live_owned =
      g_graph && g_graph->decode_mode && g_graph->expert.buffer &&
      !g_graph->expert_source_graph && g_graph->expert.capacity > 0;
  if (!g_graph_decode_arena_owner && !live_owned)
    return;
  size_t free_bytes = 0, total_bytes = 0;
  cudaMemGetInfo(&free_bytes, &total_bytes);
  // The live decode graph (state, workspace) is destroyed before the prefill
  // graph is built, so credit its small footprint.
  const size_t live_credit =
      (g_graph && g_graph->decode_mode) ? (static_cast<size_t>(512) << 20) : 0;
  const size_t need = estimate_prefill_workspace_bytes(S, g_qwen4exp_prefill_op_offload);
  const bool keep = free_bytes + live_credit >= need;
  if (keep) {
    if (!g_graph_decode_arena_owner && live_owned) {
      auto owner = extract_decode_arena_owner(*g_graph, engine);
      g_graph.reset(); // borrower-free: owns no arena any more
      if (owner) {
        g_graph_decode_arena_owner = std::move(owner);
        LOG_INFO("[GGML-QWEN-FULL] retaining the per-expert split arena for the "
                 "next decode (prompt S=%d fits: %.2f GiB free, ~%.2f GiB "
                 "needed)\n",
                 S, static_cast<double>(free_bytes) / (1024.0 * 1024.0 * 1024.0),
                 static_cast<double>(need) / (1024.0 * 1024.0 * 1024.0));
      }
    }
    return;
  }
  // The prefill needs the space. With a soft tier, free only as many soft segments as it takes (the next decode build
  // then re-uploads just those); otherwise destroy the arena.
  FullGraph *soft_owner = g_graph_decode_arena_owner.get();
  if (!soft_owner && live_owned && !g_graph->expert_soft.empty())
    soft_owner = g_graph.get();
  if (soft_owner && !soft_owner->expert_soft.empty() &&
      free_bytes + live_credit + expert_soft_bytes_allocated(*soft_owner) >= need) {
    if (!g_graph_decode_arena_owner && live_owned) {
      auto owner = extract_decode_arena_owner(*g_graph, engine);
      g_graph.reset(); // borrower-free: owns no arena any more
      if (owner)
        g_graph_decode_arena_owner = std::move(owner);
    } else if (g_graph && g_graph->decode_mode) {
      g_graph.reset(); // the decode borrower aliases the arena
    }
    if (g_graph_decode_arena_owner) {
      size_t freed = 0;
      for (int guard = 0; guard < 64; ++guard) {
        size_t now_free = 0, now_total = 0;
        cudaMemGetInfo(&now_free, &now_total);
        if (now_free >= need)
          break;
        const size_t got = release_expert_soft_segments(*g_graph_decode_arena_owner, need - now_free);
        if (got == 0)
          break;
        freed += got;
      }
      zeromoe::stats::note("soft_released_gib", static_cast<double>(freed) / (1024.0 * 1024.0 * 1024.0));
      LOG_INFO("[GGML-QWEN-FULL] released %.2f GiB of the soft expert tier before a prefill (S=%d needs ~%.2f GiB, "
               "%.2f GiB was free); the next decode re-uploads only that part\n",
               static_cast<double>(freed) / (1024.0 * 1024.0 * 1024.0), S,
               static_cast<double>(need) / (1024.0 * 1024.0 * 1024.0),
               static_cast<double>(free_bytes) / (1024.0 * 1024.0 * 1024.0));
      return;
    }
  }
  // Destroy the decode borrower first (it aliases the arena), then the arena itself.
  if (g_graph && g_graph->decode_mode)
    g_graph.reset();
  if (g_graph_decode_arena_owner)
    zeromoe::stats::note("arena_released_whole", 1.0);
  if (g_graph_decode_arena_owner) {
    g_graph_decode_arena_owner.reset();
    LOG_INFO("[GGML-QWEN-FULL] released the per-expert split arena before a "
             "long prefill (S=%d needs ~%.2f GiB, only %.2f GiB free); the "
             "next decode will rebuild it\n",
             S, static_cast<double>(need) / (1024.0 * 1024.0 * 1024.0),
             static_cast<double>(free_bytes) / (1024.0 * 1024.0 * 1024.0));
  }
}

int g_qwen4exp_suggested_batch = 0;
bool g_qwen4exp_prefill_op_offload = false;
const StreamingEngine *g_qwen4exp_cont_engine = nullptr;
int g_qwen4exp_cont_total = 0;

} // namespace qwen4exp_graph

int qwen4exp_full_graph_suggested_batch() { return qwen4exp_graph::g_qwen4exp_suggested_batch; }

bool qwen4exp_full_graph_prepare(StreamingEngine &engine, int S, int pos_base,
                                 const int *token_ids) {
  if (!env_enabled() || S <= 1 || pos_base < 0 ||
      engine.family_ops.family != ModelFamily::Qwen4Exp)
    return false;
  // A batch after the first continues the previous batch of the same prompt (recurrent state, dense KV rows, PLE window
  // are in the engine). The paths with their own per-layer machinery are not continued.
  if (pos_base > 0) {
    if (g_qwen4exp_cont_engine != &engine || g_qwen4exp_cont_total != pos_base ||
        (int64_t)pos_base + S > (int64_t)engine.max_t || engine.qwen4exp_partial_layer >= 0 ||
        !engine.qwen4exp_gpu_stage_layers.empty() || engine.qwen4exp_cpu_offload || engine.prompt_cache) {
      LOG_WARN("[GGML-QWEN-FULL] prefill batch at pos_base=%d S=%d is not continued: state_engine_match=%d state_total=%d "
               "max_t=%d partial_layer=%d stage_layers=%zu cpu_offload=%d prompt_cache=%d\n",
               pos_base, S, g_qwen4exp_cont_engine == &engine ? 1 : 0, g_qwen4exp_cont_total, engine.max_t,
               engine.qwen4exp_partial_layer, engine.qwen4exp_gpu_stage_layers.size(),
               engine.qwen4exp_cpu_offload ? 1 : 0, engine.prompt_cache ? 1 : 0);
      return false;
    }
  }
  // CPUOffload's prefill graph still uses the established scheduler CPU
  // branch. The host-node bridge is decode-only because its callback is a
  // single-token strided CPU contract; prefill must retain the ordinary
  // batched MUL_MAT_ID path to seed recurrent/KV state correctly. Decode is
  // allowed below only when every configured offload layer is replaced by a
  // CUDA-visible host marker.
  if (engine.qwen4exp_cpu_offload &&
      !engine.qwen4exp_cpu_offload_layers.empty() &&
      !env_qwen4exp_host_node_enabled()) {
    LOG_ERROR(
        "[GGML-QWEN-FULL] refusing prefill: CPUOffload layers require the "
        "cudaHostNode bridge under the full-graph-only Qwen policy\n");
    return false;
  }
  const auto &c = engine.model_file.architecture_config;
  plan_partial_layer(engine);
  const bool prefill_compact_arena_needed =
      (engine.qwen4exp_partial_layer >= 0 &&
       !engine.qwen4exp_partial_layer_gpu_ids.empty()) ||
      (engine.qwen4exp_gpu_stage_enabled &&
       !engine.qwen4exp_gpu_stage_layers.empty());
  const bool reuse_prefill_weight_arenas =
      env_decode_borrow_prefill_arenas() && !prefill_compact_arena_needed;
  if (engine.qwen_ram_preload_all_experts) {
    for (int layer = 0; layer < c.n_layers; ++layer) {
      if (engine.qwen_ram_vram_only_layer(layer))
        continue;
      if (!engine.graph_ram_layer_ready(layer)) {
        LOG_WARN("[GGML-QWEN-FULL] disabled: ram_preload_all_experts "
                 "requires a ready DDR arena for every host layer (missing "
                 "layer=%d); no GGUF mmap fallback\n",
                 layer);
        return false;
      }
    }
  }
  const int dev = engine.gpu_device_id >= 0 ? engine.gpu_device_id : 0;
  if (reuse_prefill_weight_arenas)
    manage_decode_arena_for_prefill(engine, S);
  if (g_graph && g_graph->ready && g_graph->engine == &engine &&
      g_graph->sched && !g_graph->decode_mode && g_graph->S == S && g_graph->pos_base == pos_base) {
    // PLE rows are request data, so refresh that input when the graph shape
    // is reused for a different prompt.  The model graph itself remains
    // persistent, exactly like llama.cpp's shape cache.
    if (has_ple_layer(c)) {
      std::vector<float> ple;
      if (!prepare_ple_embedding(*g_graph, ple, S, pos_base, token_ids) ||
          !g_graph->ple_embedding ||
          ple.size() * sizeof(float) > ggml_nbytes(g_graph->ple_embedding))
        return false;
      std::memcpy(g_graph->ple_embedding->data, ple.data(),
                  ple.size() * sizeof(float));
    }
    return true;
  }
  // Decode temporarily replaces g_graph with its S=1 graph, while the
  // completed prefill graph remains alive as the immutable weight-arena
  // owner.  If the next prompt has the same shape, promote that graph back
  // instead of rebuilding its thousands of nodes and re-uploading weights.
  // The decode borrower must be destroyed first because it aliases the
  // retained graph's CUDA arenas.
  if (reuse_prefill_weight_arenas && g_graph_prefill_state_owner &&
      g_graph_prefill_state_owner->ready &&
      g_graph_prefill_state_owner->sched &&
      g_graph_prefill_state_owner->engine == &engine &&
      g_graph_prefill_state_owner->S == S && g_graph_prefill_state_owner->pos_base == pos_base) {
    g_graph.reset();
    g_graph = std::move(g_graph_prefill_state_owner);
    LOG_INFO("[GGML-QWEN-FULL] reusing retained prefill state graph for "
             "matching prompt shape S=%d\n", S);
    if (has_ple_layer(c)) {
      std::vector<float> ple;
      if (!prepare_ple_embedding(*g_graph, ple, S, pos_base, token_ids) ||
          !g_graph->ple_embedding ||
          ple.size() * sizeof(float) > ggml_nbytes(g_graph->ple_embedding))
        return false;
      std::memcpy(g_graph->ple_embedding->data, ple.data(),
                  ple.size() * sizeof(float));
    }
    return true;
  }
  if (reuse_prefill_weight_arenas && g_graph_prefill_arena_owner &&
      g_graph_prefill_arena_owner->ready &&
      g_graph_prefill_arena_owner->sched &&
      g_graph_prefill_arena_owner->engine == &engine &&
      !g_graph_prefill_arena_owner->decode_mode &&
      g_graph_prefill_arena_owner->S == S && g_graph_prefill_arena_owner->pos_base == pos_base) {
    g_graph.reset();
    g_graph = std::move(g_graph_prefill_arena_owner);
    LOG_INFO("[GGML-QWEN-FULL] reusing retained prefill graph for matching "
             "prompt shape S=%d\n",
             S);
    if (has_ple_layer(c)) {
      std::vector<float> ple;
      if (!prepare_ple_embedding(*g_graph, ple, S, pos_base, token_ids) ||
          !g_graph->ple_embedding ||
          ple.size() * sizeof(float) > ggml_nbytes(g_graph->ple_embedding))
        return false;
      std::memcpy(g_graph->ple_embedding->data, ple.data(),
                  ple.size() * sizeof(float));
    }
    return true;
  }
  // Phase-2 suffix graphs borrow state, dense, and expert storage from the
  // previous prefix graph.  Destroy them before replacing that owner;
  // otherwise a persistent multi-prompt run can leave suffix instances with
  // dangling ggml/CUDA buffer views during the next rebuild.
  g_graph.reset();
  g_graph_prefill_state_owner.reset();
  auto g = std::make_unique<FullGraph>();
  g->engine = &engine;
  g->device = dev;
  g->S = S;
  g->pos_base = pos_base;
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
  if (engine.family_ops.family == ModelFamily::Qwen) {
    g->device_expert_layers = std::clamp(
        engine.cuda_moe_graph_device_expert_layers, 0, c.n_layers);
    g->device_expert_trailing =
        engine.cuda_moe_graph_device_expert_trailing;
  } else {
    g->device_expert_layers =
        std::min(qwen4exp_full_graph_device_expert_layers(), c.n_layers);
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
  plan_expert_vram_split(*g); // must run before init_device_expert_arena
  // Keep the immutable model-weight arenas across requests.  The old
  // prefill's scheduler owns a large activation workspace, so release that
  // scheduler once the old decode borrower is gone; keep the old graph
  // object only as the owner of its already-populated dense/expert buffers
  // and stable tensor-address maps.
  FullGraph *prefill_weight_source = nullptr;
  if (reuse_prefill_weight_arenas && g_graph_prefill_arena_owner &&
      g_graph_prefill_arena_owner->ready &&
      g_graph_prefill_arena_owner->engine == &engine &&
      g_graph_prefill_arena_owner->device == dev &&
      g_graph_prefill_arena_owner->dense.buffer &&
      (g_graph_prefill_arena_owner->expert.buffer ||
       g->device_expert_layers == 0) &&
      g_graph_prefill_arena_owner->device_expert_layers ==
          g->device_expert_layers &&
      g_graph_prefill_arena_owner->device_expert_trailing ==
          g->device_expert_trailing &&
      g_graph_prefill_arena_owner->device_expert_partial_gate_up ==
          g->device_expert_partial_gate_up &&
      g_graph_prefill_arena_owner->split_gpu_expert_ids ==
          g->split_gpu_expert_ids) {
    prefill_weight_source = g_graph_prefill_arena_owner.get();
    if (prefill_weight_source->sched) {
      ggml_backend_sched_free(prefill_weight_source->sched);
      prefill_weight_source->sched = nullptr;
    }
    g->dense_source_graph = prefill_weight_source;
    // A split config has no resident expert layers in the prefill graph, so the
    // owner has no expert arena to lend; borrow the dense arena only.
    if (prefill_weight_source->expert.buffer)
      g->expert_source_graph = prefill_weight_source;
    LOG_INFO("[GGML-QWEN-FULL] reusing retained dense%s weight arenas "
             "for new prompt shape S=%d\n",
             prefill_weight_source->expert.buffer ? "/expert" : "", S);
  } else if (g_graph_prefill_arena_owner) {
    // A different engine/device/residency layout cannot safely bind to these
    // addresses. g_graph (the borrower) was destroyed above, so the owner is
    // now safe to release before allocating a fresh set of arenas.
    g_graph_prefill_arena_owner.reset();
  }
  g->cuda_raw = static_cast<ggml_backend_t>(ggml_shared_backend(dev));
  g->cuda = full_graph_cuda_scheduler_backend(*g);
  g->cpu = ggml_backend_cpu_init();
  if (!g->cuda_raw || !g->cuda || !g->cpu)
    return false;
  verify_cuda_graphs_compiled_in(g->cuda_raw);
  g->cpu_threads = graph_cpu_threads();
  ggml_backend_cpu_set_n_threads(g->cpu, g->cpu_threads);
  ggml_threadpool_params cpu_tp_params =
      ggml_threadpool_params_default(g->cpu_threads);
  g->cpu_threadpool = ggml_threadpool_new(&cpu_tp_params);
  if (g->cpu_threadpool)
    ggml_backend_cpu_set_threadpool(g->cpu, g->cpu_threadpool);
  LOG_INFO("[GGML-QWEN-FULL] CPU backend configured with %d threads%s\n",
           g->cpu_threads,
           g->cpu_threadpool ? " (persistent threadpool)" : "");

  // Admit the graph-owned expert source set once, before any of its
  // individual host buffers are allocated.  Layers with a complete
  // preloaded graph-RAM arena are excluded: their expert planes are aliased
  // directly from the existing DDR cache below, so this graph does not make
  // a second copy.  Any remaining layers use one graph-owned DDR copy rather
  // than a decode-time GGUF mmap source.  The extra watermark covers the
  // GGML host context, prompt/state inputs, and allocator/I/O fluctuation;
  // no /proc/meminfo read is performed in the per-buffer allocation loop.
  const size_t expert_bytes = full_expert_source_bytes(engine);
  // 10 GiB, per explicit user instruction after the 2026-09-09 crash (was
  // 4 GiB -- the smallest, least-conservative watermark of any admission
  // check in either graph file; aligned to the same 10 GiB floor as the
  // rest).
  constexpr size_t kGraphHostWatermark = 10ull * 1024ull * 1024ull * 1024ull;
  if (expert_bytes > 0 &&
      !MemoryGuard::try_reserve_host_bytes(expert_bytes, kGraphHostWatermark)) {
    LOG_WARN("[GGML-QWEN-FULL] host watermark refused graph source=%.2f GiB + "
             "%.2f GiB reserve; "
             "MemAvailable=%ld MiB\n",
             static_cast<double>(expert_bytes) / (1024.0 * 1024.0 * 1024.0),
             static_cast<double>(kGraphHostWatermark) /
                 (1024.0 * 1024.0 * 1024.0),
             MemoryGuard::last_available_mb());
    return false;
  }
  g->host_reservation_bytes = expert_bytes;
  g->host_precharged_bytes = expert_bytes;
  LOG_INFO("[GGML-QWEN-FULL] host source admission: graph-owned=%.2f GiB + "
           "%.2f GiB reserve; "
           "complete preloaded DDR arenas are aliased directly\n",
           static_cast<double>(expert_bytes) / (1024.0 * 1024.0 * 1024.0),
           static_cast<double>(kGraphHostWatermark) /
               (1024.0 * 1024.0 * 1024.0));
  if (!init_device_dense_arena(*g) || !init_device_state_arena(*g))
    return false;
  // Reserve fixed dense weights before choosing the variable-size expert
  // arena. Otherwise an oversized all-GPU request can consume the remaining
  // VRAM with experts, appear to fit after clamping, and then fail here when
  // the dense arena is allocated.
  if (!init_device_expert_arena(*g))
    return false;
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
      LOG_INFO("[GGML-QWEN-FULL] scheduler CPU buffer=%s (CUDA host buffer)\n",
               ggml_backend_buft_name(sched_bufts[1]));
    }
  }
  // Match llama.cpp's pipeline_parallel=false scheduler mode by default.  The
  // graph's CPU/GPU transitions are dependency-serial; the event/four-copy
  // mode is retained as an explicit opt-in for comparison.
  //
  // op_offload MUST stay false here. When true,
  // ggml_backend_sched_backend_id_from_cur's host-weight-offload heuristic
  // (ggml-backend.cpp, ~line 1178) can override a host-resident MoE expert
  // weight's otherwise-correct CPU placement and push its consuming
  // MUL_MAT_ID op onto CUDA instead -- but ONLY for S>=32 (ggml's own
  // ggml_backend_offload_op prefill-scale gate), so it is invisible in any
  // S=1 decode test. Confirmed on real hardware: a 26-host-layer /
  // 22-device-layer full-graph prefill crashed with a genuine CUDA OOM
  // trying to cudaMalloc ~84-90 GiB (approximately the complete host-layer
  // expert footprint) the moment a prompt's prefill token count reached 32
  // (38-token prompt crashed; 22/24/30-token prompts on the same config did
  // not) -- exactly the failure mode bind_host_expert_scheduler_backend's
  // own comment already warns about, and exactly what op_offload=false was
  // chosen to prevent (see memory qwen4exp-mixed-gpu-decode-gap.md: "risks
  // the exact host-weight/strided-DDR corruption... for any future S>=32
  // codepath reusing this scheduler"). Device-resident expert layers are
  // unaffected either way -- their tensors are already explicitly bound to
  // the CUDA arena, independent of this flag.
  // Prefill batches (S >= 32) run the host-resident expert matmuls on the GPU: the scheduler copies the experts a batch
  // uses across (ggml patch 0007), as llama.cpp's default does (BeeLlama on this box, same model: 200-257 tok/s with it,
  // 30 without). Measured here: 1611 tokens 61 -> 305 tok/s. The reserve needs one pipeline copy (4.7 GiB at 1611
  // tokens; 44 GiB with the four copies decode uses), so prefill builds with copies=1. ZEROMOE_QWEN4EXP_PREFILL_OP_
  // OFFLOAD=0 restores the CPU experts. Off for the paths with their own per-layer expert machinery.
  const bool prefill_op_offload =
      zeromoe::graphcfg::atoi_or("ZEROMOE_QWEN4EXP_PREFILL_OP_OFFLOAD", 1) != 0 &&
      engine.qwen4exp_partial_layer < 0 && engine.qwen4exp_gpu_stage_layers.empty() && !engine.qwen4exp_cpu_offload;
  g_qwen4exp_prefill_op_offload = prefill_op_offload;
  g->sched =
      ggml_backend_sched_new(backends, sched_bufts_ptr, 2, kGraphSize,
                             // one copy with op-offload: the four pipeline copies multiplied the expert input buffers
                             // (measured reserve at 1611 tokens: 44 GiB with four copies, 4.7 GiB with one)
                             prefill_op_offload ? false : graph_scheduler_parallel(), prefill_op_offload);
  if (!g->sched)
    return false;
  LOG_INFO("[GGML-QWEN-FULL] scheduler pipeline_parallel=%s copies=%d\n",
           graph_scheduler_parallel() ? "true" : "false",
           ggml_backend_sched_get_n_copies(g->sched));
  // Replay timing must not install ggml's eval callback.  The callback path
  // decomposes CUDA graph work into per-node submissions and therefore
  // changes the execution being measured.  Keep the invasive per-node
  // timing behind its explicit switch; replay timing itself is collected
  // around the scheduler submit/synchronize boundary below.
  g->layer_timing_enabled = env_layer_timing_enabled();
  g->layer_timing_verbose_enabled = env_layer_timing_verbose();
  if (g->layer_timing_enabled || g->layer_timing_verbose_enabled) {
    g->layer_timing_cpu_ms.assign(static_cast<size_t>(g->n_layers), 0.0);
    ggml_backend_sched_set_eval_callback(g->sched, layer_timing_eval_callback,
                                         g.get());
  }

  // The model graph starts from the already expanded HC embedding.  PLE's
  // hash gather is prepared once here, outside the measured prefill window;
  // learned PLE projections and its causal convolution remain graph nodes.
  std::vector<float> ple;
  if (!prepare_ple_embedding(*g, ple, S, pos_base, token_ids) &&
      has_ple_layer(c)) {
    LOG_WARN("[GGML-QWEN-FULL] PLE embedding preparation declined\n");
    return false;
  }
  if (!build_graph(*g, has_ple_layer(c) ? &ple : nullptr)) {
    LOG_WARN("[GGML-QWEN-FULL] graph construction declined; using fallback\n");
    return false;
  }
  g_graph = std::move(g);
  return true;
}

bool qwen4exp_full_graph_bootstrap(StreamingEngine &engine) {
  zeromoe::graphcfg::log_effective("qwen4exp-full-graph");
  const char *bootstrap =
      zeromoe::graphcfg::env("ZEROMOE_QWEN4EXP_GRAPH_BOOTSTRAP");
  // Build and retain the prompt-independent dense/expert arenas during
  // startup. Graph construction must not hide a multi-second GGUF read and
  // VRAM upload inside the first prefill. Keep the old lazy behavior as an
  // explicit diagnostic override for profiles that include startup-to-first-
  // token in their timing boundary.
  if (bootstrap && std::strcmp(bootstrap, "0") == 0)
    return true;

  // The arenas are only reusable across the prefill/decode graph swap when
  // decode explicitly borrows them. In all other modes bootstrapping would
  // add startup work without avoiding a later device-weight upload.
  if (engine.family_ops.family != ModelFamily::Qwen4Exp ||
      !qwen4exp_full_graph_decode_enabled() ||
      !env_decode_borrow_prefill_arenas())
    return true;

  if (g_graph_prefill_arena_owner &&
      g_graph_prefill_arena_owner->ready &&
      g_graph_prefill_arena_owner->engine == &engine) {
    LOG_INFO("[GGML-QWEN-FULL-BOOTSTRAP] reusable weight arenas already "
             "prepared\n");
    return true;
  }

  // Device and dense weight arenas are independent of prompt length. Match
  // the configured common prefill shape (16 tokens for TG400) so that request
  // can reuse the graph too; other prompt shapes rebuild only graph/scheduler
  // state and borrow these same immutable arenas. No graph compute is issued
  // here, and PLE input is refreshed from the real prompt later.
  if (engine.max_t < 2) {
    LOG_WARN("[GGML-QWEN-FULL-BOOTSTRAP] context limit is too small for "
             "prefill graph warmup; keeping lazy graph preparation\n");
    return false;
  }
  int bootstrap_tokens = std::min(16, engine.max_t);
  if (const char *value =
          zeromoe::graphcfg::env("ZEROMOE_QWEN4EXP_GRAPH_BOOTSTRAP_TOKENS")) {
    char *end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (end == value || *end != '\0' || parsed < 2 ||
        parsed > engine.max_t) {
      LOG_WARN("[GGML-QWEN-FULL-BOOTSTRAP] invalid token count '%s' "
               "(expected 2..%d); keeping lazy graph preparation\n",
               value, engine.max_t);
      return false;
    }
    bootstrap_tokens = static_cast<int>(parsed);
  }
  std::vector<int> bootstrap_token_ids(
      static_cast<size_t>(bootstrap_tokens), 0);
  const auto started = std::chrono::steady_clock::now();
  if (!qwen4exp_full_graph_prepare(engine, bootstrap_tokens, 0,
                                   bootstrap_token_ids.data()) ||
      !g_graph || !g_graph->ready) {
    LOG_WARN("[GGML-QWEN-FULL-BOOTSTRAP] arena preparation failed; "
             "the first prompt will use the bounded lazy path\n");
    return false;
  }

  // Retain the completed graph as the immutable source owner. If a prompt
  // has another shape, qwen4exp_full_graph_prepare() frees this graph's
  // scheduler and aliases its dense/expert buffers into the request graph.
  g_graph_prefill_arena_owner = std::move(g_graph);
  const double elapsed_ms = std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - started)
                                .count();
  LOG_INFO("[GGML-QWEN-FULL-BOOTSTRAP] prepared reusable arenas in %.1fms "
           "(dense=%.2f GiB experts=%.2f GiB; bootstrap shape S=%d; "
           "one-time startup cost is reported separately from requests)\n",
           elapsed_ms,
           static_cast<double>(g_graph_prefill_arena_owner
                                   ->dense.used) /
               (1024.0 * 1024.0 * 1024.0),
           static_cast<double>(g_graph_prefill_arena_owner
                                   ->expert.used) /
               (1024.0 * 1024.0 * 1024.0),
           bootstrap_tokens);

  // Per-expert split configs (Option B): the decode-only split arena (~13 GiB of
  // hot experts) cannot be borrowed from the prefill graph, so without this the
  // first request pays the whole gather + upload (~2.7 s) inside decode setup.
  // Build the decode graph once here, keep only its expert arena (and address
  // map) in g_graph_decode_arena_owner, and let every later decode graph rebind
  // those addresses with no re-upload.
  if (engine_plans_decode_expert_split(engine) && !g_graph_decode_arena_owner) {
    const auto split_started = std::chrono::steady_clock::now();
    // decode_prepare expects the live prefill graph in g_graph and re-retains it
    // as the dense-arena owner itself.
    g_graph = std::move(g_graph_prefill_arena_owner);
    const bool built = qwen4exp_full_graph_decode_prepare(
        engine, bootstrap_tokens, bootstrap_token_ids.data());
    if (!built) {
      LOG_WARN("[GGML-QWEN-FULL-BOOTSTRAP] per-expert split arena preparation "
               "failed; the first decode will build it lazily\n");
    } else if (g_graph) {
      auto owner = extract_decode_arena_owner(*g_graph, engine);
      // Destroy the throwaway decode graph (the arena borrower) first, then
      // check how much VRAM the retained arena leaves for prefill workspaces.
      g_graph.reset();
      if (owner) {
        size_t free_bytes = 0, total_bytes = 0;
        cudaMemGetInfo(&free_bytes, &total_bytes);
        const double free_gib =
            static_cast<double>(free_bytes) / (1024.0 * 1024.0 * 1024.0);
        constexpr size_t kMinHeadroomBytes = static_cast<size_t>(1) << 30;
        if (free_bytes >= kMinHeadroomBytes) {
          const double arena_gib =
              static_cast<double>(owner->expert.capacity) /
              (1024.0 * 1024.0 * 1024.0);
          const size_t bound = owner->expert_addrs.size();
          g_graph_decode_arena_owner = std::move(owner);
          LOG_INFO("[GGML-QWEN-FULL-BOOTSTRAP] prepared the per-expert split "
                   "arena (%.2f GiB, %zu bound tensors) in %.1fms, %.2f GiB "
                   "VRAM left free; decode graph construction will rebind it "
                   "without re-uploading\n",
                   arena_gib, bound,
                   std::chrono::duration<double, std::milli>(
                       std::chrono::steady_clock::now() - split_started)
                       .count(),
                   free_gib);
        } else {
          LOG_WARN("[GGML-QWEN-FULL-BOOTSTRAP] not retaining the per-expert "
                   "split arena: only %.2f GiB of VRAM would remain free for "
                   "prefill\n",
                   free_gib);
        }
      }
    }
    g_graph.reset(); // no throwaway graph may survive bootstrap
  }
  return true;
}

#else

bool qwen4exp_full_graph_enabled() { return false; }
bool qwen4exp_full_graph_decode_enabled() { return false; }
void qwen4exp_full_graph_configure(bool, bool, bool) {}
bool qwen4exp_full_graph_sync_state_to_host(StreamingEngine &) { return true; }
int qwen4exp_full_graph_device_expert_layers() { return 0; }
bool qwen4exp_full_graph_device_expert_trailing() { return false; }
bool qwen4exp_full_graph_prepare(StreamingEngine &, int, int, const int *) {
  return false;
}
int qwen4exp_full_graph_suggested_batch() { return 0; }
bool qwen4exp_full_graph_decode_prepare(StreamingEngine &, int, const int *) {
  return false;
}
bool qwen4exp_full_graph_run(StreamingEngine &, float *, int, int,
                             const int *) {
  return false;
}
bool qwen4exp_full_graph_decode_run(StreamingEngine &, float *, int,
                                    const int *, float *, int) {
  return false;
}
void qwen4exp_full_graph_shutdown() {}

#endif
