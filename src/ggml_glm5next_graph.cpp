// GLM5Next full graph: Public lifecycle API: enabled/bootstrap/prepare/shutdown, decode-state
// sync.
//
// Split out of the former single-file ggml_glm5next_graph.cpp; shared
// declarations live in ggml_glm5next_graph_impl.hpp.

#include "ggml_glm5next_graph_impl.hpp"

#if defined(MOESTREAM_USE_GGML_CUDA)

using namespace glm5next_graph;

namespace glm5next_graph {

std::unique_ptr<FullGraph> g_graph;

// The large host expert source is independent of prompt length.  Keep it in
// a source-only container while the prompt-shaped GGML graph is rebuilt or
// before the first prompt arrives.
std::unique_ptr<FullGraph> g_glm5next_source;

GpuTopology *g_glm5next_vram_cache = nullptr;

StreamingEngine *g_glm5next_decode_disabled_engine = nullptr;

std::unique_ptr<OffloadState> g_offload;

std::mutex g_offload_mutex;

std::unique_ptr<StageRuntime> g_stage;

bool g_stage_failed = false;

// GPU1 hot weights live in buffers that outlive graph rebuilds (key: layer*3+plane).
std::map<int, ggml_backend_buffer_t> g_hot1_buffers;

} // namespace glm5next_graph

bool glm5next_full_graph_enabled() { return env_enabled(); }

void glm5next_full_graph_set_vram_cache(GpuTopology *topology) {
  g_glm5next_vram_cache = topology;
}

bool glm5next_full_graph_bootstrap(StreamingEngine &engine) {
  if (!env_enabled() || engine.family_ops.family != ModelFamily::GLM5NEXT)
    return false;
  moestream::graphcfg::log_effective("glm5next-full-graph");
  // Build and seed the offload cache at boot, not inside the first decode token.
  FullGraph candidate;
  candidate.engine = &engine;
  candidate.topology = g_glm5next_vram_cache;
  candidate.device = engine.gpu_device_id >= 0 ? engine.gpu_device_id : 0;
  candidate.first_dense = engine.model_file.architecture_config.first_dense;
  candidate.device_expert_layers =
      std::min(glm5next_full_graph_device_expert_layers(),
               std::max(0, engine.model_file.architecture_config.n_layers -
                               candidate.first_dense));
  if (configure_partial_experts(candidate))
    candidate.device_expert_layers = 0;
  if (offload_first_layer() >= 0 &&
      !get_offload(engine, candidate.partial_expert_residency
                               ? &candidate.partial_experts
                               : nullptr))
    return false;

  if (g_glm5next_source &&
      source_configuration_matches(*g_glm5next_source, candidate)) {
    LOG_INFO("[GGML-GLM5NEXT-FULL] boot expert source already resident; "
             "reusing it\n");
    return true;
  }
  g_glm5next_source.reset();
  auto source = std::make_unique<FullGraph>();
  source->engine = candidate.engine;
  source->topology = candidate.topology;
  source->device = candidate.device;
  source->first_dense = candidate.first_dense;
  source->device_expert_layers = candidate.device_expert_layers;
  source->partial_expert_residency = candidate.partial_expert_residency;
  source->vram_reference_experts = candidate.vram_reference_experts;
  source->dynamic_expert_cache = candidate.dynamic_expert_cache;
  source->partial_sources_compacted = candidate.partial_sources_compacted;
  source->partial_hot_bytes = candidate.partial_hot_bytes;
  source->partial_cold_bytes = candidate.partial_cold_bytes;
  source->partial_hot_records = candidate.partial_hot_records;
  source->vram_reference_bytes = candidate.vram_reference_bytes;
  source->partial_usage_file = std::move(candidate.partial_usage_file);
  source->partial_experts = std::move(candidate.partial_experts);
  source->vram_expert_holds = std::move(candidate.vram_expert_holds);
  if (!prepare_expert_source_only(*source, "GGML-GLM5NEXT-FULL"))
    return false;
  g_glm5next_source = std::move(source);
  return true;
}

int glm5next_full_graph_device_expert_layers() {
  return env_enabled() ? env_device_expert_layers() : 0;
}

void glm5next_full_graph_shutdown() {
  moestream::graphcfg::log_consulted("glm5next-full-graph");
  g_graph.reset();
  for (auto &kv : g_hot1_buffers)
    ggml_backend_buffer_free(kv.second);
  g_hot1_buffers.clear();
  g_glm5next_source.reset();
  g_glm5next_decode_disabled_engine = nullptr;
}

bool glm5next_full_graph_prepare(StreamingEngine &engine, int S, int pos_base,
                                 const int *) {
  if (!env_enabled() || S <= 1 || pos_base != 0 ||
      engine.family_ops.family != ModelFamily::GLM5NEXT)
    return false;
  const auto &c = engine.model_file.architecture_config;
  if (g_graph && g_graph->ready && g_graph->engine == &engine &&
      g_graph->S == S)
    return true;
  g_glm5next_decode_disabled_engine = nullptr;

  // Keep the large source arena alive while releasing only the old
  // prompt-shaped graph.  This is what makes boot preload useful for a later
  // prompt with a different token count.
  if (g_graph && g_graph->has_expert_source()) {
    if (!g_glm5next_source)
      g_glm5next_source = std::make_unique<FullGraph>();
    if (!g_graph->move_expert_source_to(*g_glm5next_source))
      return false;
  }
  g_graph.reset();
  auto g = std::make_unique<FullGraph>();
  g->engine = &engine;
  g->topology = g_glm5next_vram_cache;
  g->device = engine.gpu_device_id >= 0 ? engine.gpu_device_id : 0;
  g->S = S;
  g->D = c.hidden;
  g->HC = std::max(1, c.hc_mult);
  g->n_layers = c.n_layers;
  g->H = c.n_heads;
  g->hd = c.kda_head_dim;
  g->d_inner = g->H * g->hd;
  g->d_conv = c.ssm_conv_kernel;
  g->qk_nope = c.qk_nope;
  g->kv_lora = c.kv_lora;
  g->indexer_heads = c.glm5next_indexer_head_count;
  g->indexer_key_length = c.glm5next_indexer_key_length;
  g->indexer_top_k = c.glm5next_indexer_top_k;
  g->indexer_kpool = c.glm5next_indexer_kpool;
  g->indexer_select_tail =
      engine.model_file.get_config_int("attention.indexer.kpool_select_tail", 0) != 0;
  g->v_head = c.v_head;
  g->first_dense = c.first_dense;
  g->device_expert_layers = std::min(glm5next_full_graph_device_expert_layers(),
                                     std::max(0, c.n_layers - c.first_dense));
  if (configure_partial_experts(*g))
    g->device_expert_layers = 0;
  g->cuda = static_cast<ggml_backend_t>(ggml_shared_backend(g->device));
  g->cpu = ggml_backend_cpu_init();
  if (g->cpu) {
    const char *e = moestream::graphcfg::env("MOESTREAM_GLM5NEXT_CPU_THREADS");
    ggml_backend_cpu_set_n_threads(g->cpu, e ? std::max(1, std::atoi(e)) : 6);
  }
  if (!g->cuda || !g->cpu)
    return false;

  if (g_glm5next_source &&
      source_configuration_matches(*g_glm5next_source, *g)) {
    if (!g_glm5next_source->move_expert_source_to(*g))
      return false;
    LOG_INFO("[GGML-GLM5NEXT-FULL] reusing boot expert source: %.2f GiB DDR; "
             "no SSD reload for prompt S=%d\n",
             (double)g->expert_source_used / (1024.0 * 1024.0 * 1024.0), S);
  } else if (!prepare_expert_source_only(*g, "GGML-GLM5NEXT-FULL")) {
    return false;
  }
  if (!init_expert_arena(*g) || !init_dense_arena(*g))
    return false;
  // GPU1-hot experts compute in place on the second GPU (MOESTREAM_GLM5NEXT_PREFILL_GPU1=0 routes them through the cold
  // branch, copying each used expert to the main GPU).
  if (moestream::graphcfg::atoi_or("MOESTREAM_GLM5NEXT_PREFILL_GPU1", 1) != 0)
    for (const auto &pl : g->partial_experts)
      if (!pl.hot1.empty()) {
        g->gpu1 = static_cast<ggml_backend_t>(ggml_shared_backend(1));
        break;
      }
  ggml_backend_t backends[] = {g->cuda, g->gpu1 ? g->gpu1 : g->cpu, g->cpu};
  // Batched prefill: let the scheduler run the host-resident (cold) expert matmuls on the GPU by copying the weights
  // across, instead of on the CPU (MOESTREAM_GLM5NEXT_PREFILL_OP_OFFLOAD=1).
  const bool prefill_offload = moestream::graphcfg::flag_on("MOESTREAM_GLM5NEXT_PREFILL_OP_OFFLOAD");
  g->sched = ggml_backend_sched_new(backends, nullptr, g->gpu1 ? 3 : 2, kGraphSize, false, prefill_offload);
  if (!g->sched)
    return false;
  if (!build_graph(*g)) {
    LOG_WARN("[GGML-GLM5NEXT-FULL] graph construction declined; using legacy "
             "path\n");
    return false;
  }
  g_graph = std::move(g);
  LOG_INFO("[GGML-GLM5NEXT-FULL] graph prepared for S=%d\n", S);
  return true;
}

namespace glm5next_graph {

void transpose_glm5next_state(float *dst, const float *src, int hd,
                                      int heads) {
  const size_t plane = (size_t)hd * hd;
  for (int h = 0; h < heads; ++h)
    for (int row = 0; row < hd; ++row)
      for (int col = 0; col < hd; ++col)
        dst[(size_t)h * plane + (size_t)row * hd + col] =
            src[(size_t)h * plane + (size_t)col * hd + row];
}

bool sync_glm5next_decode_state_to_host(FullGraph &g) {
  if (!g.decode_mode || !g.ready || !g.engine || g.decode_last_pos < 0)
    return false;
  StreamingEngine &engine = *g.engine;
  const size_t state_n = (size_t)g.hd * g.hd * g.H;
  const size_t hist_n = (size_t)(g.d_conv - 1) * 3 * g.d_inner;
  std::vector<float> state(state_n);
  for (size_t i = 0; i < g.kda_layers.size(); ++i) {
    const int layer = g.kda_layers[i];
    if (layer < 0 || layer >= (int)engine.glm5next_recur_state.size() ||
        layer >= (int)engine.glm5next_conv_history.size() ||
        engine.glm5next_recur_state[layer].size() != state_n ||
        engine.glm5next_conv_history[layer].size() != hist_n ||
        i >= g.kda_state_storage.size() || i >= g.kda_hist_storage.size())
      return false;
    ggml_backend_tensor_get(g.kda_state_storage[i], state.data(), 0,
                            state_n * sizeof(float));
    transpose_glm5next_state(engine.glm5next_recur_state[layer].data(),
                             state.data(), g.hd, g.H);
    ggml_backend_tensor_get(g.kda_hist_storage[i],
                            engine.glm5next_conv_history[layer].data(), 0,
                            hist_n * sizeof(float));
  }

  const int valid_rows =
      std::min(g.decode_last_pos + 1, std::max(0, engine.max_t));
  const size_t kv_n = (size_t)valid_rows * g.kv_lora;
  std::vector<ggml_fp16_t> half_cache(kv_n);
  std::vector<float> cache(kv_n);
  for (size_t i = 0; i < g.dsa_layers.size(); ++i) {
    const int layer = g.dsa_layers[i];
    if (layer < 0 || layer >= (int)engine.Lc.size() || !engine.Lc[layer] ||
        i >= g.dsa_kv_storage.size())
      return false;
    if (kv_n == 0)
      continue;
    ggml_backend_tensor_get(g.dsa_kv_storage[i], half_cache.data(), 0,
                            kv_n * sizeof(ggml_fp16_t));
    for (size_t j = 0; j < kv_n; ++j)
      cache[j] = ggml_fp16_to_fp32(half_cache[j]);
    std::memcpy(engine.Lc[layer], cache.data(), kv_n * sizeof(float));
  }
  return true;
}

} // namespace glm5next_graph

#else

bool glm5next_full_graph_enabled() { return false; }
bool glm5next_full_graph_bootstrap(StreamingEngine &) { return false; }
void glm5next_full_graph_set_vram_cache(GpuTopology *) {}
int glm5next_full_graph_device_expert_layers() { return 0; }
bool glm5next_full_graph_prepare(StreamingEngine &, int, int, const int *) {
  return false;
}
bool glm5next_full_graph_run(StreamingEngine &, float *, int, int,
                             const int *) {
  return false;
}
bool glm5next_full_graph_decode_prepare(StreamingEngine &, int, const int *) {
  return false;
}
bool glm5next_full_graph_decode_run(StreamingEngine &, float *, int,
                                    const int *) {
  return false;
}
void glm5next_full_graph_shutdown() {}

#endif
