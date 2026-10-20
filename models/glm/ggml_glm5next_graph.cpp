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
  if (!env_enabled() || !glm5next_graph::glm_family(engine.family_ops.family))
    return false;
  moestream::graphcfg::log_effective("glm5next-full-graph");
  g_skip_indexer_weights = engine.family_ops.family == ModelFamily::GLM_DSA;
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
  if (moestream::graphcfg::atoi_or("MOESTREAM_GLM5NEXT_GPU1_HOT_PRELOAD", 1) != 0 && !preload_partial_hot1(candidate))
    LOG_WARN("[GGML-GLM5NEXT-GPU1-HOT] boot preload failed; hot experts will upload at the first decode build\n");

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

static bool glm5next_full_graph_prepare_once(StreamingEngine &engine, int S, int pos_base, const int *tokens);

// What a prefill chunk leaves for the next one. Kept apart from g_graph, which a declined attempt (or the retry without the
// prefetch ring) has already released.
static int g_suggested_batch = 0;
int glm5next_full_graph_suggested_batch() { return g_suggested_batch; }

static struct PrefillContinuation {
  const StreamingEngine *engine = nullptr;
  int total_tokens = 0;
  std::vector<Glm5nextIndexerSeed> seeds;
} g_continuation;

void glm5next_publish_continuation(StreamingEngine &engine, glm5next_graph::FullGraph &g) {
  g_continuation.engine = &engine;
  g_continuation.total_tokens = g.total_tokens;
  g_continuation.seeds = g.glm5next_indexer_seeds;
}

// The speculative prefetch ring takes VRAM the batched-prefill workspace may then not find: when a graph that had a ring
// is declined, it is built once more without it (a legacy prefill of the same prompt is ~100x slower).
bool glm5next_full_graph_prepare(StreamingEngine &engine, int S, int pos_base, const int *tokens) {
  if (pos_base == 0)
    prefetch_suppress_prompt(false);
  (void)prefetch_ring_was_built();
  if (glm5next_full_graph_prepare_once(engine, S, pos_base, tokens))
    return true;
  if (!prefetch_ring_was_built())
    return false;
  LOG_WARN("[GGML-GLM5NEXT-FULL] graph declined with the prefetch ring (%.2f GiB on the main GPU); retrying without it\n",
           prefetch_ring_main_gib());
  prefetch_suppress_prompt(true);
  prefetch_note_ring_failure(S);
  return glm5next_full_graph_prepare_once(engine, S, pos_base, tokens);
}

static bool glm5next_full_graph_prepare_once(StreamingEngine &engine, int S, int pos_base, const int *) {
  if (!env_enabled() || S <= 1 || pos_base < 0 ||
      !glm5next_graph::glm_family(engine.family_ops.family))
    return false;
  const auto &c = engine.model_file.architecture_config;
  // The latent cache (engine.Lc) holds max_t rows; a chunk that would write past it is left to the caller.
  if ((int64_t)pos_base + S > (int64_t)engine.max_t)
    return false;
  // A chunk after the first (pos_base > 0) continues the previous chunk's graph state: its KDA state and conv history, the
  // latent rows and the indexer seeds, all of which the previous run left in the engine / the old graph. Prefill attention
  // is dense, which equals the model's top-k selection only while the context fits in top_k.
  if (pos_base > 0) {
    const int kpool = c.glm5next_indexer_kpool, top_k = c.glm5next_indexer_top_k;
    if (moestream::graphcfg::atoi_or("MOESTREAM_GLM5NEXT_PREFILL_NO_CONTINUATION", 0) != 0)
      return false; // A/B reference: the rest of the prompt is stepped
    if (g_continuation.engine != &engine || g_continuation.total_tokens != pos_base)
      return false;
    if (kpool > 1 && pos_base % kpool != 0)
      return false;
    // Context beyond top_k: the chunk's attention selects pools per query like the decode graph (sparse_prefill below);
    // MOESTREAM_GLM5NEXT_PREFILL_DENSE_BEYOND_TOPK=1 keeps the dense attention instead (A/B only).
  }
  if (g_graph && g_graph->ready && g_graph->engine == &engine &&
      g_graph->S == S && g_graph->pos_base == pos_base && !g_graph->decode_mode)
    return true;
  std::vector<Glm5nextIndexerSeed> prev_seeds;
  if (pos_base > 0)
    prev_seeds = g_continuation.seeds;
  g_continuation = PrefillContinuation{};
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
  g->pos_base = pos_base;
  g->total_tokens = pos_base + S;
  g->prev_indexer_seeds = std::move(prev_seeds);
  {
    const int kpool = c.glm5next_indexer_kpool, top_k = c.glm5next_indexer_top_k;
    static const bool dense_beyond = moestream::graphcfg::atoi_or("MOESTREAM_GLM5NEXT_PREFILL_DENSE_BEYOND_TOPK", 0) != 0;
    static const bool force_sparse = moestream::graphcfg::atoi_or("MOESTREAM_GLM5NEXT_PREFILL_FORCE_SPARSE", 0) != 0; // test hook
    if (top_k > 0 && (pos_base + S > top_k || force_sparse) && !dense_beyond) {
      // The selection needs the pooled indexer keys of every earlier chunk (the seeds) and pool-aligned boundaries.
      if (kpool <= 1 || top_k % kpool != 0 || (pos_base > 0 && g->prev_indexer_seeds.empty()))
        return false;
      g->sparse_prefill = true;
    }
  }
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
  g->qk_rope = c.qk_rope;
  g->dsa_plain = engine.family_ops.family == ModelFamily::GLM_DSA;
  g_skip_indexer_weights = g->dsa_plain;
  g->kv_row = g->dsa_plain ? c.kv_lora + c.qk_rope : c.kv_lora;
  g->rope_base = engine.model_file.get_config_float("rope.freq_base", 10000.0f);
  g->first_dense = c.first_dense;
  g->device_expert_layers = std::min(glm5next_full_graph_device_expert_layers(),
                                     std::max(0, c.n_layers - c.first_dense));
  if (configure_partial_experts(*g))
    g->device_expert_layers = 0;
  g->cuda = static_cast<ggml_backend_t>(ggml_shared_backend(g->device));
  g->cpu = ggml_backend_cpu_init();
  if (g->cpu) {
    // Prefill runs the cold-expert matmuls on the CPU for a whole batch: it has its own thread count.
    const char *e = moestream::graphcfg::env("MOESTREAM_GLM5NEXT_PREFILL_CPU_THREADS");
    if (!e || !e[0])
      e = moestream::graphcfg::env("MOESTREAM_GLM5NEXT_CPU_THREADS");
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
  // Memory pre-check: the scheduler's workspace is requested from the allocator in one piece, and a request the device
  // cannot satisfy must never reach it. Device-0 consumption measured from free memory after the build: ~0.8 MiB per token (0.20 GiB left at 814
  // tokens, 0.004 at 1065 with the ring); the scheduler's total workspace is 1.35 GiB + 0.45 MiB/token, mostly host/GPU1; host side needs headroom above the compute floor. When the batch does not fit, the
  // largest one that does is recorded for the caller (suggested_prefill_batch) and this prepare is declined.
  {
    size_t free_b = 0, total_b = 0;
    cudaSetDevice(g->device);
    const bool have = cudaMemGetInfo(&free_b, &total_b) == cudaSuccess;
    constexpr double kGiB = 1024.0 * 1024.0 * 1024.0, kMiB = 1024.0 * 1024.0;
    const double fixed = 0.2 * kGiB, per_token = 0.85 * kMiB, margin = 0.15 * kGiB; // device 0, see the ring check
    const double avail_host_gb = std::max(0.0, (double)MemoryGuard::read_available_mb() / 1024.0);
    const double host_need_gb = 12.0 + (double)S / 1024.0; // 10 GiB compute floor + slack + ~1 MiB per token
    int fit = S;
    if (have && fixed + per_token * S + margin > (double)free_b)
      fit = (int)std::max(0.0, ((double)free_b - fixed - margin) / per_token);
    if (avail_host_gb > 0 && avail_host_gb < host_need_gb)
      fit = 0;
    if (fit < S) {
      fit = (fit / 8) * 8;
      g_suggested_batch = fit >= 64 ? fit : 0;
      LOG_WARN("[GGML-GLM5NEXT-FULL] prefill batch of %d tokens needs ~%.2f GiB of device workspace but %.2f GiB are free "
               "(host: %.1f GiB available, %.1f needed); declining before any allocation%s\n",
               S, (fixed + per_token * S) / kGiB, have ? (double)free_b / kGiB : -1.0, avail_host_gb, host_need_gb,
               g_suggested_batch > 0 ? ("; " + std::to_string(g_suggested_batch) + " tokens would fit").c_str() : "");
      return false;
    }
    g_suggested_batch = 0;
  }
  // GPU1-hot experts compute in place on the second GPU (MOESTREAM_GLM5NEXT_PREFILL_GPU1=0 routes them through the cold
  // branch, copying each used expert to the main GPU).
  if (moestream::graphcfg::atoi_or("MOESTREAM_GLM5NEXT_PREFILL_GPU1", 1) != 0)
    for (const auto &pl : g->partial_experts)
      if (!pl.hot1.empty()) {
        g->gpu1 = static_cast<ggml_backend_t>(ggml_shared_backend(second_gpu_device(g->device)));
        break;
      }
  ggml_backend_t backends[] = {g->cuda, g->gpu1 ? g->gpu1 : g->cpu, g->cpu};
  // Batched prefill: let the scheduler run the host-resident (cold) expert matmuls on the GPU by copying the weights
  // across, instead of on the CPU (MOESTREAM_GLM5NEXT_PREFILL_OP_OFFLOAD=1).
  const bool prefill_offload = moestream::graphcfg::flag_on("MOESTREAM_GLM5NEXT_PREFILL_OP_OFFLOAD");
  // The scheduler needs events (MOESTREAM_SCHED_EVENTS, set below): without them every weight-input copy first
  // synchronizes the whole destination stream, which serializes the copies of the two GPUs. parallel=true is not an
  // option: it multiplies the graph inputs by the number of copies (56 GB of workspace).
  if (moestream::graphcfg::atoi_or("MOESTREAM_GLM5NEXT_PREFILL_SCHED_EVENTS", 1) != 0)
    setenv("MOESTREAM_SCHED_EVENTS", "1", 1);
  const bool sched_parallel = moestream::graphcfg::atoi_or("MOESTREAM_GLM5NEXT_PREFILL_SCHED_PARALLEL", 0) != 0;
  g->sched = ggml_backend_sched_new(backends, nullptr, g->gpu1 ? 3 : 2, kGraphSize, sched_parallel, prefill_offload);
  if (!g->sched)
    return false;
  if (!build_graph(*g)) {
    LOG_WARN("[GGML-GLM5NEXT-FULL] graph construction declined; using legacy "
             "path\n");
    return false;
  }
  g_graph = std::move(g);
  {
    // Free memory per device once the workspace is allocated (calibrates the pre-check above).
    int ndev = 0;
    cudaGetDeviceCount(&ndev);
    for (int d = 0; d < ndev && d < 2; ++d) {
      size_t f = 0, t = 0;
      cudaSetDevice(d);
      if (cudaMemGetInfo(&f, &t) == cudaSuccess)
        LOG_INFO("[GGML-GLM5NEXT-FULL] after graph build: device %d free %.3f GiB\n", d,
                 (double)f / (1024.0 * 1024.0 * 1024.0));
    }
    cudaSetDevice(g_graph->device);
  }
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
  const size_t row_w = g.dsa_plain ? g.kv_row : g.kv_lora;
  const size_t kv_n = (size_t)valid_rows * row_w;
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
    if (g.dsa_plain) {
      if (layer >= (int)engine.Rc.size() || !engine.Rc[layer])
        return false;
      for (int t = 0; t < valid_rows; ++t) {
        std::memcpy(engine.Lc[layer] + (size_t)t * g.kv_lora, cache.data() + (size_t)t * row_w, g.kv_lora * sizeof(float));
        std::memcpy(engine.Rc[layer] + (size_t)t * g.qk_rope, cache.data() + (size_t)t * row_w + g.kv_lora,
                    g.qk_rope * sizeof(float));
      }
    } else {
      std::memcpy(engine.Lc[layer], cache.data(), kv_n * sizeof(float));
    }
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
