// Qwen4Exp full graph: adaptive expert promotion (docs/design/expert-promotion.md).
//
// After each run (one token of plain decode, or the committed prefix of an MTP window) this reads
// the routed ids of every layer, lets ExpertPromotionPolicy plan promotions, copies the promoted
// experts' planes from the pinned host arena into the victims' device slots on a side stream, and
// rewrites the route tables so the victim is computed on the CPU from the next run on and the
// promoted expert on the GPU once its copy has landed (evict now, resident after the copy).

#include "ggml_qwen4exp_graph_internal.hpp"
#include "moe_partial_route.hpp"
#include "nvtx_range.hpp"
#include <algorithm>
#include <cstdio>

#if defined(ZEROMOE_USE_GGML_CUDA)

using namespace qwen4exp_graph;

namespace {

using zeromoe::graph::ExpertPromotionPolicy;
using zeromoe::graph::PromotionAction;
using zeromoe::graph::PromotionConfig;

bool promotion_init(FullGraph &g) {
  const auto &c = g.engine->model_file.architecture_config;
  if (!g.usage_track_ids || g.usage_track_ids->type != GGML_TYPE_I32 ||
      g.usage_track_ids->ne[0] != static_cast<int64_t>(c.topk) * g.n_layers ||
      g.usage_track_ids->ne[1] != g.S)
    return false;
  std::vector<std::vector<int>> slots(static_cast<size_t>(g.n_layers));
  size_t hot_layers = 0;
  for (const auto &[layer, ids] : g.split_gpu_expert_ids) {
    if (layer < 0 || layer >= g.n_layers) continue;
    const auto it = g.promo_layers.find(layer);
    if (it == g.promo_layers.end()) continue;
    const auto &pl = it->second;
    bool complete = pl.remap_gpu && pl.remap_cpu && pl.mask_gpu && pl.mask_cpu;
    for (const auto &plane : pl.plane)
      complete = complete && plane.dst && plane.dst->data && plane.src && plane.bytes > 0;
    if (!complete) continue;
    slots[static_cast<size_t>(layer)] = ids;
    ++hot_layers;
  }
  if (hot_layers == 0) return false;
  PromotionConfig cfg;
  cfg.enabled = g.engine->qwen4exp_expert_promotion >= 2;
  if (const char *v = std::getenv("ZEROMOE_PROMO_CAP")) cfg.cap = std::atoi(v);
  if (const char *v = std::getenv("ZEROMOE_PROMO_BURST")) cfg.burst_extra = std::atoi(v);
  if (const char *v = std::getenv("ZEROMOE_PROMO_MIN_MISSES")) cfg.min_misses = std::atoi(v);
  if (const char *v = std::getenv("ZEROMOE_PROMO_FREQ_WEIGHT")) cfg.freq_weight = std::strtoull(v, nullptr, 10);
  if (const char *v = std::getenv("ZEROMOE_PROMO_DECAY_EVERY")) cfg.decay_every = static_cast<uint32_t>(std::strtoul(v, nullptr, 10));
  g.promo = std::make_unique<ExpertPromotionPolicy>(cfg, c.n_experts, slots);
  if (cudaSetDevice(g.device) != cudaSuccess ||
      cudaStreamCreateWithFlags(&g.promo_stream, cudaStreamNonBlocking) != cudaSuccess ||
      cudaEventCreateWithFlags(&g.promo_event, cudaEventDisableTiming) != cudaSuccess) {
    g.promo.reset();
    return false;
  }
  LOG_INFO("[EXPERT-PROMOTION] %s: %zu layers with hot sets, cap=%d/token burst=+%d for %d tokens, "
           "min_misses=%d, eviction=last_used+%llu*hits\n",
           cfg.enabled ? "promoting" : "observing only (frozen set)", hot_layers, cfg.cap,
           cfg.burst_extra, cfg.burst_tokens, cfg.min_misses,
           static_cast<unsigned long long>(cfg.freq_weight));
  return true;
}

// Rewrite the four route tables of `layer` from the policy's residency (same encoding as
// build_expert_split_tables) and refresh the early-ready lane's cached and device copies.
void write_tables(FullGraph &g, int layer) {
  const int E = g.engine->model_file.architecture_config.n_experts;
  auto &pl = g.promo_layers.at(layer);
  std::vector<int32_t> gpu_index(static_cast<size_t>(E), -1), cpu_index(static_cast<size_t>(E), -1);
  for (int e = 0; e < E; ++e) {
    const int s = g.promo->slot_of(layer, e);
    if (s >= 0) gpu_index[static_cast<size_t>(e)] = s;
    else cpu_index[static_cast<size_t>(e)] = e;
  }
  const auto gm = moe_partial::ExpertBranchMap::build(gpu_index, 1, true, {}, moe_partial::SentinelPolicy::Auto);
  const auto cm = moe_partial::ExpertBranchMap::build(cpu_index, 1, true, {}, moe_partial::SentinelPolicy::Auto);
  std::memcpy(pl.remap_gpu->data, gm.route_ids().data(), gm.route_ids().size() * sizeof(int32_t));
  std::memcpy(pl.remap_cpu->data, cm.route_ids().data(), cm.route_ids().size() * sizeof(int32_t));
  std::memcpy(pl.mask_gpu->data, gm.route_mask().data(), gm.route_mask().size() * sizeof(float));
  std::memcpy(pl.mask_cpu->data, cm.route_mask().data(), cm.route_mask().size() * sizeof(float));
  const auto lane = g.early_ready.find(layer);
  if (lane == g.early_ready.end()) return;
  EarlyReadyLane &early = *lane->second;
  early.refresh_routes();
  if (early.resident_tables) {
    ggml_tensor *host[4] = {pl.remap_gpu, pl.remap_cpu, pl.mask_gpu, pl.mask_cpu};
    for (int i = 0; i < 4; ++i)
      if (early.gpu_tables[i])
        ggml_backend_tensor_set(early.gpu_tables[i], host[i]->data, 0, ggml_nbytes(host[i]));
  }
}

}  // namespace

namespace {

struct PredictStats {
  uint64_t layers_tokens = 0;      // (layer, token) pairs with a prediction
  uint64_t actual = 0, hit = 0;    // routed experts, and how many the prediction contained
  uint64_t actual_cold = 0, cold_hit = 0;   // routed experts that were not resident, and how many were predicted
  uint64_t predicted_cold = 0, predicted_cold_used = 0;   // predicted non-resident experts, and how many were routed
};
PredictStats g_pred;

FILE *promo_trace_file() {
  static FILE *f = [] {
    const char *path = std::getenv("ZEROMOE_PROMO_TRACE");
    return path && path[0] ? std::fopen(path, "w") : nullptr;
  }();
  return f;
}

}  // namespace

// Call after a run has completed (device idle) and `tokens` of its tokens are committed.
void qwen4exp_promotion_step(FullGraph &g, int tokens) {
  if (g.engine->qwen4exp_expert_promotion <= 0 || !g.decode_mode || tokens < 1) return;
  ZE_NVTX("promo.step");
  if (!g.promo && !promotion_init(g)) {
    static bool warned = false;
    if (!warned) { warned = true; LOG_WARN("[EXPERT-PROMOTION] unavailable for this graph; static hot set\n"); }
    g.engine->qwen4exp_expert_promotion = 0;
    return;
  }
  ExpertPromotionPolicy &policy = *g.promo;
  const auto &c = g.engine->model_file.architecture_config;
  std::vector<uint8_t> dirty(static_cast<size_t>(g.n_layers), 0);

  // 1. routed ids of the committed tokens, counted against the residency the run actually executed with
  ZE_NVTX("promo.ids");
  const size_t row = static_cast<size_t>(c.topk) * g.n_layers;
  g.promo_ids_host.resize(row * static_cast<size_t>(g.S));
  ggml_backend_tensor_get(g.usage_track_ids, g.promo_ids_host.data(), 0,
                          g.promo_ids_host.size() * sizeof(int32_t));
  FILE *trace = promo_trace_file();
  const int ntok = std::min(tokens, g.S);
  for (int t = 0; t < ntok; ++t)
    for (int l = 0; l < g.n_layers; ++l) {
      if (policy.slots(l) == 0) continue;
      const int32_t *ids = &g.promo_ids_host[static_cast<size_t>(t) * row + static_cast<size_t>(l) * c.topk];
      if (trace) {
        std::fprintf(trace, "%llu %d %d", static_cast<unsigned long long>(g.promo_steps + static_cast<uint64_t>(t)), t, l);
        for (int k = 0; k < c.topk; ++k)
          std::fprintf(trace, " %d:%d", ids[k], ids[k] >= 0 && ids[k] < c.n_experts && policy.resident(l, ids[k]) ? 1 : 0);
        std::fputc('\n', trace);
      }
      // next-layer prediction (early-ready lane): layer l's lane predicted layer l+1's routing for each of its tokens
      {
        const auto lane = g.early_ready.find(l);
        if (lane != g.early_ready.end() && lane->second->prediction && l + 1 < g.n_layers &&
            lane->second->prediction_layer == l + 1 && policy.slots(l + 1) > 0) {
          const int k = std::min<int>(c.topk, 64);
          std::vector<int32_t> pred(static_cast<size_t>(k) * g.S);
          if (ggml_nbytes(lane->second->prediction) >= pred.size() * sizeof(int32_t)) {
            ggml_backend_tensor_get(lane->second->prediction, pred.data(), 0, pred.size() * sizeof(int32_t));
            const int32_t *mine = &pred[static_cast<size_t>(t) * k];
            const int32_t *actual = &g.promo_ids_host[static_cast<size_t>(t) * row + static_cast<size_t>(l + 1) * c.topk];
            ++g_pred.layers_tokens;
            for (int i = 0; i < k; ++i) {
              const bool in_pred = std::find(mine, mine + k, actual[i]) != mine + k;
              const bool cold = actual[i] >= 0 && actual[i] < c.n_experts && !policy.resident(l + 1, actual[i]);
              ++g_pred.actual; g_pred.hit += in_pred;
              if (cold) { ++g_pred.actual_cold; g_pred.cold_hit += in_pred; }
            }
            for (int i = 0; i < k; ++i) {
              if (mine[i] < 0 || mine[i] >= c.n_experts || policy.resident(l + 1, mine[i])) continue;
              ++g_pred.predicted_cold;
              g_pred.predicted_cold_used += std::find(actual, actual + c.topk, mine[i]) != actual + c.topk;
            }
          }
        }
      }
      for (int k = 0; k < c.topk; ++k)
        if (ids[k] >= 0 && ids[k] < c.n_experts) policy.note_routed(l, ids[k]);
    }
  if (trace) std::fflush(trace);

  // 2. copies that have landed make their experts resident
  if (!g.promo_pending.empty()) {
    if (cudaEventQuery(g.promo_event) == cudaSuccess) {
      for (const auto &a : g.promo_pending) { policy.confirm(a); dirty[static_cast<size_t>(a.layer)] = 1; }
      g.promo_pending.clear();
    } else {
      ++g.promo_late;
    }
  }

  // 3. plan, unless the previous batch of copies is still in flight
  if (g.promo_pending.empty()) {
    auto actions = policy.plan(tokens);
    for (const auto &a : actions) {
      auto &pl = g.promo_layers.at(a.layer);
      for (const auto &plane : pl.plane)
        cudaMemcpyAsync(static_cast<uint8_t *>(plane.dst->data) + static_cast<size_t>(a.slot) * plane.bytes,
                        plane.src + static_cast<size_t>(a.expert_in) * plane.stride, plane.bytes,
                        cudaMemcpyHostToDevice, g.promo_stream);
      dirty[static_cast<size_t>(a.layer)] = 1;
    }
    if (!actions.empty()) {
      cudaEventRecord(g.promo_event, g.promo_stream);
      g.promo_pending = std::move(actions);
    }
  }

  // 4. route tables of every layer whose residency changed
  for (int l = 0; l < g.n_layers; ++l)
    if (dirty[static_cast<size_t>(l)]) write_tables(g, l);

  g.promo_steps += static_cast<uint64_t>(tokens);
  if (g.promo_steps / 100 != (g.promo_steps - static_cast<uint64_t>(tokens)) / 100) {
    const auto &st = policy.stats();
    const double hits = static_cast<double>(st.hits_static + st.hits_promoted);
    LOG_INFO("[EXPERT-PROMOTION] tokens=%llu routed=%llu miss=%.2f%% planned=%llu confirmed=%llu late=%llu | "
             "hits from promoted=%.1f%% wasted=%llu first_use=%.1f tokens\n",
             static_cast<unsigned long long>(g.promo_steps), static_cast<unsigned long long>(st.routed),
             st.routed ? 100.0 * static_cast<double>(st.misses) / static_cast<double>(st.routed) : 0.0,
             static_cast<unsigned long long>(st.planned), static_cast<unsigned long long>(st.confirmed),
             static_cast<unsigned long long>(g.promo_late),
             hits > 0 ? 100.0 * static_cast<double>(st.hits_promoted) / hits : 0.0,
             static_cast<unsigned long long>(st.wasted),
             st.first_use_n ? static_cast<double>(st.first_use_tokens) / static_cast<double>(st.first_use_n) : 0.0);
    if (g_pred.layers_tokens)
      LOG_INFO("[NEXT-LAYER-PREDICTION] samples=%llu recall@%d=%.1f%% | cold experts: recall=%.1f%% (%llu of %llu) "
               "precision=%.1f%% (%llu of %llu predicted cold were used)\n",
               static_cast<unsigned long long>(g_pred.layers_tokens), c.topk,
               100.0 * static_cast<double>(g_pred.hit) / static_cast<double>(g_pred.actual),
               g_pred.actual_cold ? 100.0 * static_cast<double>(g_pred.cold_hit) / static_cast<double>(g_pred.actual_cold) : 0.0,
               static_cast<unsigned long long>(g_pred.cold_hit), static_cast<unsigned long long>(g_pred.actual_cold),
               g_pred.predicted_cold ? 100.0 * static_cast<double>(g_pred.predicted_cold_used) / static_cast<double>(g_pred.predicted_cold) : 0.0,
               static_cast<unsigned long long>(g_pred.predicted_cold_used), static_cast<unsigned long long>(g_pred.predicted_cold));
  }
}

#endif  // ZEROMOE_USE_GGML_CUDA
