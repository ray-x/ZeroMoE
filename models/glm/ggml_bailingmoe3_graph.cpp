#include "ggml_bailingmoe3_graph.hpp"

#include "async_prefetcher.hpp"
#include "cache_backed_moe.hpp"
#include "dense_dispatch.hpp"
#include "engine.hpp"
#include "family_ops.hpp"
#include "graph_config.hpp"
#include "logger.hpp"
#include "memory_guard.hpp"
#include "moe_pipeline.hpp"
#include "ram_manager.hpp"
#include "scratch_buffers.hpp"

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <algorithm>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

extern void dense_ffn_gguf(StreamingEngine &engine, int layer_idx,
                           const float *hidden_states, int S, float *output,
                           LayerScratch &scratch);

namespace {

#if defined(ZEROMOE_USE_GGML_CUDA)

constexpr size_t kLingGraphContextBytes = 8ull << 20;
constexpr int kLingGraphNodes = 1024;

struct LingNativeGraph;

struct LingLayerSite {
  LingNativeGraph *owner = nullptr;
  int layer = -1;
};

struct LingNativeGraph {
  StreamingEngine *engine = nullptr;
  int D = 0;
  int S = 0;
  int pos_base = 0;
  std::vector<int> token_ids;
  std::vector<LingLayerSite> sites;
  ggml_backend_t cpu = nullptr;
  ggml_backend_sched_t sched = nullptr;
  ggml_context *ctx = nullptr;
  ggml_cgraph *graph = nullptr;
  ggml_tensor *input = nullptr;
  ggml_tensor *output = nullptr;
  size_t context_reservation = 0;

  ~LingNativeGraph() {
    if (sched)
      ggml_backend_sched_free(sched);
    if (ctx)
      ggml_free(ctx);
    if (cpu)
      ggml_backend_free(cpu);
    if (context_reservation)
      MemoryGuard::release_host_bytes(context_reservation);
  }
};

struct LingNativeRuntime {
  StreamingEngine *engine = nullptr;
  RAMSafetyManager *ram = nullptr;
  LockFreePrefetcher *prefetch = nullptr;
  LayerScratch *scratch = nullptr;
  zeromoe::graph::CacheBackedMoeRuntime moe;
};

LingNativeRuntime g_runtime;
std::unique_ptr<LingNativeGraph> g_graph;

void ling_layer_node(ggml_tensor *dst, const ggml_tensor *src, int ith, int,
                     void *userdata) {
  if (ith != 0)
    return;
  auto &site = *static_cast<LingLayerSite *>(userdata);
  LingNativeGraph &g = *site.owner;
  StreamingEngine &engine = *g.engine;
  if (g_runtime.engine != &engine || !g_runtime.ram || !g_runtime.prefetch ||
      !g_runtime.scratch)
    throw std::runtime_error("Ling native graph cache runtime is not bound");

  const size_t bytes = (size_t)g.S * (size_t)g.D * sizeof(float);
  if (!src || !src->data || !dst || !dst->data || ggml_nbytes(src) != bytes ||
      ggml_nbytes(dst) != bytes)
    throw std::runtime_error("Ling native graph layer tensor shape mismatch");

  // ggml_map_custom1 gives this layer its own output buffer. The native
  // FamilyOps hooks update the residual in place, so seed that buffer first.
  std::memcpy(dst->data, src->data, bytes);
  auto *hidden = static_cast<float *>(dst->data);
  LayerScratch &scratch = *g_runtime.scratch;
  if (engine.family_ops.layer_input_fn)
    engine.family_ops.layer_input_fn(engine, site.layer, g.S, g.pos_base,
                                     g.token_ids.data());

  engine.family_ops.hc_pre_fn(engine, site.layer, 0, hidden, g.S, g.D);
  engine.family_ops.attention_fn(engine, site.layer, hidden, g.S, g.pos_base,
                                 scratch.attn_out.data());
  engine.family_ops.hc_post_fn(engine, site.layer, 0, hidden,
                               scratch.attn_out.data(), g.S, g.D);

  engine.family_ops.hc_pre_fn(engine, site.layer, 1, hidden, g.S, g.D);
  bailingmoe3_native_graph_ffn(engine, site.layer, hidden, g.S,
                               scratch.moe_out.data(), g.token_ids.data());
  engine.family_ops.hc_post_fn(engine, site.layer, 1, hidden,
                               scratch.moe_out.data(), g.S, g.D);
}

std::unique_ptr<LingNativeGraph> build_graph(StreamingEngine &engine, int S,
                                             int pos_base,
                                             const int *token_ids) {
  if (S <= 0 || pos_base < 0 || !token_ids ||
      engine.family_ops.family != ModelFamily::BailingMoE3 ||
      !engine.family_ops.attention_fn)
    return nullptr;

  const auto &c = engine.model_file.architecture_config;
  auto g = std::make_unique<LingNativeGraph>();
  g->engine = &engine;
  g->D = c.hidden;
  g->S = S;
  g->pos_base = pos_base;
  g->token_ids.assign(token_ids, token_ids + S);
  g->sites.reserve((size_t)c.n_layers);

  if (!MemoryGuard::try_reserve_host_bytes(kLingGraphContextBytes))
    return nullptr;
  g->context_reservation = kLingGraphContextBytes;
  g->ctx = ggml_init({kLingGraphContextBytes, nullptr, true});
  if (!g->ctx)
    return nullptr;
  g->graph = ggml_new_graph_custom(g->ctx, kLingGraphNodes, false);
  g->cpu = ggml_backend_cpu_init();
  if (!g->graph || !g->cpu)
    return nullptr;
  ggml_backend_t backends[] = {g->cpu};
  g->sched = ggml_backend_sched_new(backends, nullptr, 1, kLingGraphNodes,
                                    false, false);
  if (!g->sched)
    return nullptr;

  g->input = ggml_new_tensor_2d(g->ctx, GGML_TYPE_F32, c.hidden, S);
  if (!g->input)
    return nullptr;
  ggml_set_name(g->input, "ling_native_graph_input");
  ggml_set_input(g->input);

  ggml_tensor *cur = g->input;
  for (int layer = 0; layer < c.n_layers; ++layer) {
    g->sites.push_back({g.get(), layer});
    cur = ggml_map_custom1(g->ctx, cur, ling_layer_node, 1, &g->sites.back());
    if (!cur)
      return nullptr;
    ggml_set_name(cur, ("ling_native_layer_" + std::to_string(layer)).c_str());
    ggml_backend_sched_set_tensor_backend(g->sched, cur, g->cpu);
  }
  g->output = cur;
  ggml_set_output(g->output);
  ggml_build_forward_expand(g->graph, g->output);
  if (!ggml_backend_sched_alloc_graph(g->sched, g->graph))
    return nullptr;

  LOG_INFO("[BAILINGMOE3-CACHE-GRAPH] built S=%d layers=%d nodes=%d splits=%d; "
           "graph-owned expert bytes=0\n",
           S, c.n_layers, ggml_graph_n_nodes(g->graph),
           ggml_backend_sched_get_n_splits(g->sched));
  return g;
}

#endif // ZEROMOE_USE_GGML_CUDA

} // namespace

bool bailingmoe3_native_graph_enabled() {
#if defined(ZEROMOE_USE_GGML_CUDA)
  return zeromoe::graphcfg::flag_on("ZEROMOE_LING_FULL_GRAPH") &&
         zeromoe::graphcfg::flag_on("ZEROMOE_LING_NATIVE_GRAPH");
#else
  return false;
#endif
}

bool bailingmoe3_hybrid_cuda_graph_enabled() {
#if defined(ZEROMOE_USE_GGML_CUDA)
  return bailingmoe3_native_graph_enabled() &&
         zeromoe::graphcfg::flag_on("ZEROMOE_LING_HYBRID_CUDA_GRAPH");
#else
  return false;
#endif
}

void bailingmoe3_native_graph_bind_runtime(StreamingEngine &engine,
                                           RAMSafetyManager &ram,
                                           LockFreePrefetcher &prefetch,
                                           LayerScratch &scratch) {
#if defined(ZEROMOE_USE_GGML_CUDA)
  g_runtime = {&engine, &ram, &prefetch, &scratch};
  g_runtime.moe.bind(engine, ram, prefetch, scratch);
#else
  (void)engine;
  (void)ram;
  (void)prefetch;
  (void)scratch;
#endif
}

void bailingmoe3_native_graph_ffn(StreamingEngine &engine, int layer_idx,
                                 const float *hidden_states, int S,
                                 float *output, const int *token_ids) {
#if defined(ZEROMOE_USE_GGML_CUDA)
  if (g_runtime.engine != &engine || !g_runtime.ram || !g_runtime.prefetch ||
      !g_runtime.scratch)
    throw std::runtime_error("Ling native MoE runtime is not bound");
  if (!hidden_states || !output || S <= 0 || layer_idx < 0 ||
      layer_idx >= engine.model_file.architecture_config.n_layers)
    throw std::runtime_error("Ling native MoE callback input is invalid");

  auto &scratch = *g_runtime.scratch;
  const auto &c = engine.model_file.architecture_config;
  const size_t elements = (size_t)S * (size_t)c.hidden;
  if (scratch.moe_out.size() < elements)
    throw std::runtime_error("Ling native MoE scratch is not sized for input");
  if (layer_idx < c.first_dense) {
    dense_ffn_gguf(engine, layer_idx, hidden_states, S, output, scratch);
  } else {
    g_runtime.moe.run(layer_idx, hidden_states, S, output, token_ids);
  }
#else
  (void)engine;
  (void)layer_idx;
  (void)hidden_states;
  (void)S;
  (void)output;
  (void)token_ids;
  throw std::runtime_error("Ling native MoE graph requires GGML CUDA");
#endif
}

bool bailingmoe3_native_graph_prepare(StreamingEngine &engine, int S,
                                      int pos_base, const int *token_ids) {
#if defined(ZEROMOE_USE_GGML_CUDA)
  if (!bailingmoe3_native_graph_enabled() || !token_ids || S <= 0 ||
      pos_base < 0 || pos_base + S > engine.max_t)
    return false;
  if (g_graph && g_graph->engine == &engine && g_graph->S == S) {
    g_graph->pos_base = pos_base;
    g_graph->token_ids.assign(token_ids, token_ids + S);
    return true;
  }
  g_graph.reset();
  g_graph = build_graph(engine, S, pos_base, token_ids);
  return g_graph != nullptr;
#else
  (void)engine;
  (void)S;
  (void)pos_base;
  (void)token_ids;
  return false;
#endif
}

bool bailingmoe3_native_graph_run(StreamingEngine &engine, float *hidden, int S,
                                  int pos_base, const int *token_ids) {
#if defined(ZEROMOE_USE_GGML_CUDA)
  if (!g_graph || g_graph->engine != &engine || !hidden || S != g_graph->S ||
      !bailingmoe3_native_graph_prepare(engine, S, pos_base, token_ids))
    return false;
  const size_t bytes = (size_t)S * (size_t)g_graph->D * sizeof(float);
  ggml_backend_tensor_set(g_graph->input, hidden, 0, bytes);
  const auto status =
      ggml_backend_sched_graph_compute(g_graph->sched, g_graph->graph);
  ggml_backend_sched_synchronize(g_graph->sched);
  if (status != GGML_STATUS_SUCCESS)
    throw std::runtime_error("Ling native graph compute failed");
  ggml_backend_tensor_get(g_graph->output, hidden, 0, bytes);
  return true;
#else
  (void)engine;
  (void)hidden;
  (void)S;
  (void)pos_base;
  (void)token_ids;
  return false;
#endif
}

void bailingmoe3_native_graph_shutdown() {
#if defined(ZEROMOE_USE_GGML_CUDA)
  g_graph.reset();
  g_runtime = {};
#endif
}
