// Qwen4Exp full graph: adaptive expert promotion (docs/design/expert-promotion.md).
//
// After each run (one token of plain decode, or the committed prefix of an MTP window) this reads
// the routed ids of every layer, lets ExpertPromotionPolicy plan promotions, copies the promoted
// experts' planes from the pinned host arena into the victims' device slots on a side stream, and
// rewrites the route tables so the victim is computed on the CPU from the next run on and the
// promoted expert on the GPU once its copy has landed (evict now, resident after the copy).

#include "ggml_qwen4exp_graph_internal.hpp"
#include "moe_partial_route.hpp"

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

// Call after a run has completed (device idle) and `tokens` of its tokens are committed.
void qwen4exp_promotion_step(FullGraph &g, int tokens) {
  if (g.engine->qwen4exp_expert_promotion <= 0 || !g.decode_mode || tokens < 1) return;
  if (!g.promo && !promotion_init(g)) {
    static bool warned = false;
    if (!warned) { warned = true; LOG_WARN("[EXPERT-PROMOTION] unavailable for this graph; static hot set\n"); }
    g.engine->qwen4exp_expert_promotion = 0;
    return;
  }
  ExpertPromotionPolicy &policy = *g.promo;
  const auto &c = g.engine->model_file.architecture_config;
  std::vector<uint8_t> dirty(static_cast<size_t>(g.n_layers), 0);

  // 1. copies that have landed make their experts resident
  if (!g.promo_pending.empty()) {
    if (cudaEventQuery(g.promo_event) == cudaSuccess) {
      for (const auto &a : g.promo_pending) { policy.confirm(a); dirty[static_cast<size_t>(a.layer)] = 1; }
      g.promo_pending.clear();
    } else {
      ++g.promo_late;
    }
  }

  // 2. routed ids of the committed tokens
  const size_t row = static_cast<size_t>(c.topk) * g.n_layers;
  g.promo_ids_host.resize(row * static_cast<size_t>(g.S));
  ggml_backend_tensor_get(g.usage_track_ids, g.promo_ids_host.data(), 0,
                          g.promo_ids_host.size() * sizeof(int32_t));
  for (int t = 0; t < std::min(tokens, g.S); ++t)
    for (int l = 0; l < g.n_layers; ++l) {
      if (policy.slots(l) == 0) continue;
      for (int k = 0; k < c.topk; ++k) {
        const int32_t e = g.promo_ids_host[static_cast<size_t>(t) * row + static_cast<size_t>(l) * c.topk + k];
        if (e >= 0 && e < c.n_experts) policy.note_routed(l, e);
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
    LOG_INFO("[EXPERT-PROMOTION] tokens=%llu routed=%llu miss=%.2f%% planned=%llu confirmed=%llu late=%llu\n",
             static_cast<unsigned long long>(g.promo_steps), static_cast<unsigned long long>(st.routed),
             st.routed ? 100.0 * static_cast<double>(st.misses) / static_cast<double>(st.routed) : 0.0,
             static_cast<unsigned long long>(st.planned), static_cast<unsigned long long>(st.confirmed),
             static_cast<unsigned long long>(g.promo_late));
  }
}

#endif  // ZEROMOE_USE_GGML_CUDA
