// Qwen4Exp full graph: Planning: VRAM split between experts and dense state, partial-layer
// selection, expert source sizing.
//
// Split out of the former single-file ggml_qwen4exp_graph.cpp; shared
// declarations live in ggml_qwen4exp_graph_internal.hpp.

#include "ggml_qwen4exp_graph_internal.hpp"

#if defined(ZEROMOE_USE_GGML_CUDA)

using namespace qwen4exp_graph;

namespace qwen4exp_graph {

// Per-expert VRAM split planner (qwen4exp_graph_expert_vram_budget_gb).
// Generalizes cpu_affinity_layers/device_expert_layers' whole-layer choice:
// instead of an all-or-nothing residency decision per layer, spend a fixed
// VRAM budget on the globally hottest routed experts (by real usage, same
// ranking data pin_topk_per_layer/global_topn already consult), across any
// layer NOT already claimed by cpu_affinity_layers or device_expert_layers/
// device_expert_partial_gate_up -- those two mechanisms are checked FIRST
// and always win, so this planner never double-places a layer they already
// resolved. Must run before init_device_expert_arena() so the arena can
// size itself for the resulting split_gpu_expert_ids. v1 scope: decode only
// (g.S==1) -- see build_moe()'s own S==1 guard for why; a config with the
// budget set during prefill (S>1) simply plans nothing here, no error.
void plan_expert_vram_split(FullGraph &g) {
  g.split_gpu_expert_ids.clear();
  if (g.engine && g.engine->family_ops.family == ModelFamily::Qwen &&
      g.engine->cuda_moe_full_graph_decode_required)
    return; // The fit planner owns the complete Qwen expert placement.
  if (g.S != 1 && !g.decode_mode)
    return;

  // Option B is a deterministic usage-driven plan: a static hot set per
  // selected layer, with complementary experts left in the fixed-stride DDR
  // arena.  In concentrated-only mode the selected layers were chosen from
  // the usage histogram by the startup preload pass.  The graph repeats the
  // eligibility checks here because this is the final owner of the CUDA
  // arena layout and must never split a layer already covered by the complete
  // baseline GPU block.
  const int option_layer = g.engine->qwen4exp_option_b_layer;
  const int option_top_n = g.engine->qwen4exp_option_b_top_n;
  const bool option_b_auto =
      g.engine->qwen4exp_option_b_concentrated_only && option_layer < 0;
  if (option_top_n > 0 &&
      (option_layer >= 0 || option_b_auto ||
       !g.engine->qwen4exp_option_b_layers.empty())) {
    const auto &c = g.engine->model_file.architecture_config;

    const std::string path = g.engine->qwen4exp_graph_pin_file.empty()
                                 ? std::string(".zeromoe_usage")
                                 : g.engine->qwen4exp_graph_pin_file;
    auto [entries, maxc] = StreamingEngine::load_usage(path);
    (void)maxc;
    if (entries.empty()) {
      LOG_WARN("[OPTION-B] no usage data at %s; Layer %d remains CPU/DDR\n",
               path.c_str(), option_layer);
      return;
    }

    std::vector<int> option_layers = g.engine->qwen4exp_option_b_layers;
    if (option_layer >= 0)
      option_layers = {option_layer};
    else if (option_layers.empty() && option_b_auto) {
      option_layers.reserve(static_cast<size_t>(c.n_layers));
      for (int layer = 0; layer < c.n_layers; ++layer)
        option_layers.push_back(layer);
    }

    // Global pool mode: one ranking over every eligible layer's experts (the
    // startup pass in main.cpp runs the same helper, so the layer list and the
    // hot sets agree). Per-layer top_n, right-sizing and the coverage gate do
    // not apply; top_n is only a per-layer ceiling.
    const bool global_mode = g.engine->qwen4exp_option_b_global_top_k > 0;
    option_b_global::Result global_sel;
    if (global_mode) {
      std::vector<int> live_layers;
      for (int layer : option_layers) {
        if (layer < 0 || layer >= c.n_layers || c.n_experts <= 0)
          continue;
        if (is_cpu_affinity_layer(layer) || device_expert_block_covers(g, layer))
          continue;
        live_layers.push_back(layer);
      }
      global_sel = option_b_global::select(
          entries, live_layers, c.n_layers, c.n_experts,
          g.engine->qwen4exp_option_b_global_top_k,
          g.engine->qwen4exp_option_b_global_floor_per_layer, option_top_n);
      LOG_INFO("[OPTION-B] global pool: top-%d over %zu layer(s) -> %zu "
               "expert(s) in %zu layer(s), pooled coverage=%.2f%%\n",
               g.engine->qwen4exp_option_b_global_top_k, live_layers.size(),
               global_sel.total_hot, global_sel.hot.size(),
               global_sel.pool_coverage * 100.0);
    }

    int committed_layers = 0;
    for (int layer : option_layers) {
      if (layer < 0 || layer >= c.n_layers || c.n_experts <= 0) {
        LOG_WARN("[OPTION-B] invalid target layer=%d (layers=%d); skipping\n",
                 layer, c.n_layers);
        continue;
      }
      if (is_cpu_affinity_layer(layer) || device_expert_block_covers(g, layer))
        continue;

      // Usage files normally contain one row per expert, but aggregate rather
      // than trusting that format. Zero-count experts are included so a
      // top-N request really means N out of the model's complete expert
      // domain, not merely N of the experts seen in an earlier run.
      std::vector<uint64_t> counts(static_cast<size_t>(c.n_experts), 0);
      for (const auto &e : entries) {
        if (e.layer == layer && e.expert >= 0 && e.expert < c.n_experts)
          counts[static_cast<size_t>(e.expert)] += e.count;
      }
      uint64_t total_count = 0;
      for (uint64_t count : counts)
        total_count += count;
      if (total_count == 0)
        continue;

      std::vector<int> ranked(static_cast<size_t>(c.n_experts));
      for (int expert = 0; expert < c.n_experts; ++expert)
        ranked[static_cast<size_t>(expert)] = expert;
      std::stable_sort(ranked.begin(), ranked.end(), [&](int a, int b) {
        if (counts[static_cast<size_t>(a)] != counts[static_cast<size_t>(b)])
          return counts[static_cast<size_t>(a)] > counts[static_cast<size_t>(b)];
        return a < b;
      });

      int hot_count = std::min(option_top_n, c.n_experts);
      // top_n is a flat per-model ceiling, but layers differ widely in how
      // concentrated their real routing is -- a layer that already clears
      // min_coverage well before top_n is still charged the full top_n VRAM
      // cost below. When opted in, shrink hot_count to the smallest prefix
      // of THIS layer's own ranking that reaches min_coverage, so VRAM scales
      // with actual per-layer concentration instead of a one-size-fits-all
      // count. Off by default: existing configs keep their exact, previously
      // measured per-layer VRAM cost.
      if (g.engine->qwen4exp_option_b_right_size_top_n && total_count > 0) {
        const double target = g.engine->qwen4exp_option_b_min_coverage;
        uint64_t cum = 0;
        for (int k = 0; k < hot_count; ++k) {
          cum += counts[static_cast<size_t>(ranked[static_cast<size_t>(k)])];
          if (static_cast<double>(cum) / static_cast<double>(total_count) >=
              target) {
            hot_count = k + 1;
            break;
          }
        }
      }
      std::vector<int> hot_ids(ranked.begin(), ranked.begin() + hot_count);
      if (global_mode) {
        const auto it = global_sel.hot.find(layer);
        if (it == global_sel.hot.end()) {
          LOG_INFO("[OPTION-B] Layer %d remains CPU/DDR: no expert of this "
                   "layer made the global top-%d\n",
                   layer, g.engine->qwen4exp_option_b_global_top_k);
          continue;
        }
        hot_ids = it->second;
        hot_count = static_cast<int>(hot_ids.size());
      }
      uint64_t hot_route_count = 0;
      for (int expert : hot_ids)
        hot_route_count += counts[static_cast<size_t>(expert)];
      if (g.engine->qwen4exp_option_b_fill_vram && !global_mode) {
        // Start with the requested coverage floor, then let the arena
        // planner trim the least-used tail if the remaining VRAM cannot hold
        // every expert from every selected layer.
        hot_ids.insert(hot_ids.end(), ranked.begin() + hot_count,
                       ranked.end());
      }
      const double coverage = static_cast<double>(hot_route_count) /
                              static_cast<double>(total_count);
      if (!global_mode && coverage < g.engine->qwen4exp_option_b_min_coverage) {
        LOG_INFO("[OPTION-B] Layer %d remains CPU/DDR: top-%d coverage=%.2f%% "
                 "is below %.2f%%\n",
                 layer, hot_count, coverage * 100.0,
                 g.engine->qwen4exp_option_b_min_coverage * 100.0);
        continue;
      }

      std::vector<int> staged_experts;
      // The staging proof is intentionally still Layer-13-only. Automatic
      // concentrated-layer selection must not silently create active-layer
      // slots or forced routes for every selected layer.
      if (layer == option_layer) {
        int stage_seed = g.engine->qwen4exp_option_b_stage_expert;
        const int stage_count = std::min(
            std::max(1, g.engine->qwen4exp_option_b_stage_count), c.topk);
        if (stage_seed >= 0 && stage_seed < c.n_experts) {
          for (int candidate = stage_seed;
               candidate < c.n_experts &&
               static_cast<int>(staged_experts.size()) < stage_count;
               ++candidate) {
            if (std::find(hot_ids.begin(), hot_ids.end(), candidate) !=
                hot_ids.end())
              continue;
            staged_experts.push_back(candidate);
          }
          for (int candidate = 0;
               candidate < stage_seed &&
               static_cast<int>(staged_experts.size()) < stage_count;
               ++candidate) {
            if (std::find(hot_ids.begin(), hot_ids.end(), candidate) !=
                hot_ids.end())
              continue;
            staged_experts.push_back(candidate);
          }
        }
      }
      if (!staged_experts.empty())
        hot_ids.insert(hot_ids.end(), staged_experts.begin(),
                       staged_experts.end());
      if (!staged_experts.empty()) {
        g.stage_experts = staged_experts;
        g.stage_layer = layer;
        g.stage_expert = staged_experts.front();
      }

      const size_t placed_hot = hot_ids.size();
      const size_t per_expert = g.engine->compute_expert_slab_bytes(layer);
      size_t bytes = 0;
      if (per_expert != 0 && hot_ids.size() <= SIZE_MAX / per_expert)
        bytes = per_expert * hot_ids.size();
      g.split_gpu_expert_ids[layer] = std::move(hot_ids);
      ++committed_layers;
      LOG_INFO("[OPTION-B] Layer %d static hot/cold split: hot=%zu cold=%d "
               "staged=%d coverage=%.2f%% threshold=%.2f%% VRAM=%.3f GiB "
               "source=%s\n",
               layer, placed_hot,
               c.n_experts - static_cast<int>(placed_hot) -
                   static_cast<int>(staged_experts.size()),
               static_cast<int>(staged_experts.size()), coverage * 100.0,
               g.engine->qwen4exp_option_b_min_coverage * 100.0,
               static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0),
               path.c_str());
    }
    if (option_b_auto)
      LOG_INFO("[OPTION-B] concentrated-only graph plan committed %d CPU-side "
               "layer(s); baseline GPU block and flat layers were skipped\n",
               committed_layers);
    return;
  }

  const double budget_gb = g.engine->qwen4exp_graph_expert_vram_budget_gb;
  if (budget_gb <= 0.0)
    return; // feature off -- identical to today, zero planning cost

  const std::string path = g.engine->qwen4exp_graph_pin_file.empty()
                               ? std::string(".zeromoe_usage")
                               : g.engine->qwen4exp_graph_pin_file;
  auto [entries, maxc] = StreamingEngine::load_usage(path);
  (void)maxc;
  if (entries.empty()) {
    LOG_INFO("[GGML-QWEN-FULL-SPLIT] no usage data at %s -- per-expert VRAM "
             "split stays off this run.\n",
             path.c_str());
    return;
  }

  const auto &c = g.engine->model_file.architecture_config;
  const int n_layers = c.n_layers;
  const int n_experts = c.n_experts;

  // Bounds-check + drop any (layer,expert) already resolved by
  // cpu_affinity_layers/device_expert_layers/device_expert_partial_gate_up --
  // same stale-usage-file tolerance as main.cpp's sanitize_usage_entries.
  std::vector<StreamingEngine::UsageEntry> eligible;
  eligible.reserve(entries.size());
  for (const auto &e : entries) {
    if (e.layer < 0 || e.layer >= n_layers || e.expert < 0 ||
        e.expert >= n_experts)
      continue;
    if (is_cpu_affinity_layer(e.layer))
      continue;
    if (device_expert_block_covers(g, e.layer))
      continue;
    if (g.device_expert_partial_gate_up && e.layer == g.device_expert_layers)
      continue; // boundary layer's own asymmetric gate/up-only residency
    eligible.push_back(e);
  }
  if (eligible.empty()) {
    LOG_INFO("[GGML-QWEN-FULL-SPLIT] no eligible layers (all claimed by "
             "cpu_affinity_layers/device_expert_layers) -- per-expert VRAM "
             "split stays off this run.\n");
    return;
  }

  // Single global priority queue over all (layer, expert, count) triples --
  // spends the fixed budget on the model's genuinely hottest experts first,
  // rather than splitting it evenly per layer. Stable tie-break by
  // (layer, expert) for determinism, same reasoning as main.cpp's [PIN]
  // block's own stable_sort.
  std::stable_sort(eligible.begin(), eligible.end(),
                   [](const StreamingEngine::UsageEntry &a,
                      const StreamingEngine::UsageEntry &b) {
                     if (a.count != b.count)
                       return a.count > b.count;
                     if (a.layer != b.layer)
                       return a.layer < b.layer;
                     return a.expert < b.expert;
                   });

  size_t remaining_budget =
      static_cast<size_t>(budget_gb * 1024.0 * 1024.0 * 1024.0);
  std::unordered_map<int, std::unordered_set<int>> chosen;
  size_t experts_placed = 0;
  size_t bytes_spent = 0;
  for (const auto &e : eligible) {
    if (remaining_budget == 0)
      break;
    auto &layer_set = chosen[e.layer];
    if (layer_set.count(e.expert))
      continue; // usage file is expected unique per (layer,expert); don't trust that blindly
    const size_t cost = g.engine->compute_expert_slab_bytes(e.layer);
    if (cost == 0 || cost > remaining_budget)
      continue; // doesn't fit -- skip, keep scanning cheaper/later entries
    layer_set.insert(e.expert);
    remaining_budget -= cost;
    bytes_spent += cost;
    ++experts_placed;
  }

  for (auto &[layer, ids_set] : chosen) {
    if (ids_set.empty())
      continue;
    std::vector<int> &dst = g.split_gpu_expert_ids[layer];
    dst.assign(ids_set.begin(), ids_set.end());
    std::sort(dst.begin(), dst.end()); // deterministic local-index order
  }

  LOG_INFO("[GGML-QWEN-FULL-SPLIT] budget=%.3f GiB spent=%.3f GiB across %zu "
           "layer(s), %zu expert(s) total\n",
           budget_gb, bytes_spent / (1024.0 * 1024.0 * 1024.0),
           g.split_gpu_expert_ids.size(), experts_placed);
}

// Phase 1 partial-layer planner: ranks ONE specific layer's experts by real
// usage (same pin_file the startup [PIN] preload block and
// plan_expert_vram_split already read) and keeps the top N. Idempotent --
// only does real work once per process (engine.qwen4exp_partial_layer_planned
// guards it), since a layer's usage ranking doesn't change mid-process and
// this must survive the prefill->decode FullGraph object swap (see
// engine.hpp's own comment on why this state is engine-owned).
void plan_partial_layer(StreamingEngine &engine) {
  engine.qwen4exp_partial_layer = env_partial_layer();
  if (engine.qwen4exp_partial_layer < 0)
    return;
  engine.qwen4exp_partial_layer_top_n = env_partial_layer_top_n();
  engine.qwen4exp_partial_layer_min_coverage = env_partial_layer_min_coverage();
  if (engine.qwen4exp_partial_layer_planned)
    return;

  const std::string path = engine.qwen4exp_graph_pin_file.empty()
                               ? std::string(".zeromoe_usage")
                               : engine.qwen4exp_graph_pin_file;
  auto [entries, maxc] = StreamingEngine::load_usage(path);
  (void)maxc;

  std::vector<StreamingEngine::UsageEntry> layer_entries;
  for (const auto &e : entries)
    if (e.layer == engine.qwen4exp_partial_layer)
      layer_entries.push_back(e);
  std::stable_sort(layer_entries.begin(), layer_entries.end(),
                   [](const auto &a, const auto &b) {
                     return a.count > b.count;
                   });

  engine.qwen4exp_partial_layer_gpu_ids.clear();
  for (size_t i = 0; i < layer_entries.size() &&
                     static_cast<int>(i) < engine.qwen4exp_partial_layer_top_n;
       ++i)
    engine.qwen4exp_partial_layer_gpu_ids.push_back(layer_entries[i].expert);
  engine.qwen4exp_partial_layer_planned = true;

  LOG_INFO("[PARTIAL-LAYER] layer=%d top_n=%d ranked from %s: %zu known "
           "expert(s) for this layer, using top %zu for the GPU-resident "
           "compact set\n",
           engine.qwen4exp_partial_layer, engine.qwen4exp_partial_layer_top_n,
           path.c_str(), layer_entries.size(),
           engine.qwen4exp_partial_layer_gpu_ids.size());
}

size_t full_expert_source_bytes(const StreamingEngine &engine,
                                int layer_limit)  {
  const auto &c = engine.model_file.architecture_config;
  size_t total = 0;
  const int limit = layer_limit > 0 ? std::min(layer_limit, c.n_layers)
                                    : c.n_layers;
  for (int layer = 0; layer < limit; ++layer) {
    if (engine.qwen_ram_vram_only_layer(layer))
      continue;
    // When main.cpp prepared a complete graph-RAM arena, the three packed
    // expert planes for this layer are already resident in DDR.  The full
    // graph aliases those bytes below; do not pre-charge or allocate a second
    // graph-owned source set for the same layer.
    if (engine.graph_ram_layer_ready(layer))
      continue;
    for (const char *stem : {"ffn_gate_exps.weight", "ffn_up_exps.weight",
                             "ffn_down_exps.weight"}) {
      const GGUFTensor *m =
          model_tensor(engine, "blk." + std::to_string(layer) + "." + stem);
      if (!m || m->size_bytes == 0)
        return 0;
      if (static_cast<size_t>(m->size_bytes) >
          std::numeric_limits<size_t>::max() - total)
        return 0;
      total += static_cast<size_t>(m->size_bytes);
    }
  }
  return total;
}

// Resolve one packed Qwen expert projection to the complete, preloaded
// graph-RAM arena. This is deliberately Qwen-specific. The arena may use the
// legacy [gate|up|down] per-expert layout or the projection-compact layout
// (three separate expert-axis planes); in both cases this returns a strided
// view without making a graph-owned copy.
bool preloaded_ram_expert_source(FullGraph &g, const std::string &name,
                                 const GGUFTensor &meta, void *&ptr,
                                 size_t &bytes, size_t &expert_stride,
                                 size_t &arena_bytes) {
  const std::string prefix = "blk.";
  if (name.compare(0, prefix.size(), prefix) != 0)
    return false;

  char *end = nullptr;
  const long parsed_layer = std::strtol(name.c_str() + prefix.size(), &end, 10);
  if (end == name.c_str() + prefix.size() || parsed_layer < 0 || *end != '.')
    return false;
  const int layer = static_cast<int>(parsed_layer);
  const std::string suffix(end + 1);
  const auto *arena = g.engine->graph_ram_arena(layer);
  if (!arena)
    return false;
  size_t plane_offset = 0;
  size_t plane_bytes = 0;
  size_t plane_stride = 0;
  if (suffix == "ffn_gate_exps.weight") {
    plane_offset = arena->gate_offset;
    plane_bytes = arena->gate_bytes;
    plane_stride = arena->gate_stride;
  } else if (suffix == "ffn_up_exps.weight") {
    plane_offset = arena->projection_compact ? arena->up_offset
                                             : arena->gate_bytes;
    plane_bytes = arena->up_bytes;
    plane_stride = arena->up_stride;
  } else if (suffix == "ffn_down_exps.weight") {
    plane_offset = arena->projection_compact
                       ? arena->down_offset
                       : arena->gate_bytes + arena->up_bytes;
    plane_bytes = arena->down_bytes;
    plane_stride = arena->down_stride;
  } else {
    return false;
  }

  if (!g.engine->graph_ram_layer_ready(layer))
    return false;
  const int n_experts = g.engine->model_file.architecture_config.n_experts;
  const size_t n_exp = static_cast<size_t>(std::max(0, n_experts));
  if (!arena || !arena->base || n_experts <= 0 ||
      arena->n_experts != n_experts || plane_bytes == 0 ||
      plane_stride == 0 ||
      plane_offset > arena->bytes ||
      plane_bytes > arena->bytes - plane_offset ||
      n_exp > SIZE_MAX / plane_bytes ||
      meta.size_bytes != plane_bytes * n_exp || plane_stride < plane_bytes ||
      static_cast<size_t>(n_experts - 1) >
          (SIZE_MAX - plane_bytes) / plane_stride)
    return false;

  const size_t view_bytes =
      plane_stride * static_cast<size_t>(n_experts - 1) + plane_bytes;
  if (plane_offset > arena->bytes || view_bytes > arena->bytes - plane_offset)
    return false;

  ptr = arena->base + plane_offset;
  bytes = view_bytes;
  expert_stride = plane_stride;
  arena_bytes = arena->bytes;
  return true;
}

int device_expert_resident_layer_count(const FullGraph &g) {
  const auto &c = g.engine->model_file.architecture_config;
  const int limit = std::min(std::max(0, g.device_expert_layers), c.n_layers);
  const int block_start = g.device_expert_trailing ? c.n_layers - limit : 0;
  const int block_end = g.device_expert_trailing ? c.n_layers : limit;
  int count = 0;
  for (int layer = block_start; layer < block_end; ++layer)
    if (!is_cpu_affinity_layer(layer))
      ++count;
  return count;
}

size_t device_expert_bytes(const FullGraph &g) {
  const auto &engine = *g.engine;
  const auto &c = engine.model_file.architecture_config;
  const int limit = std::min(std::max(0, g.device_expert_layers), c.n_layers);
  // Block bounds mirror device_expert_block_covers() exactly: leading mode
  // walks [0,limit), trailing mode walks [n_layers-limit,n_layers). Kept as
  // an explicit loop (not a device_expert_block_covers() call per layer)
  // since this function also needs the leading-only boundary-layer index
  // (`limit`) below, unchanged from before trailing mode existed.
  const int block_start = g.device_expert_trailing ? c.n_layers - limit : 0;
  const int block_end = g.device_expert_trailing ? c.n_layers : limit;
  size_t total = 0;
  for (int layer = block_start; layer < block_end; ++layer) {
    // cpu_affinity_layers wins over the candidate device block in
    // graph_expert_projection_resident(). Do not reserve VRAM for these
    // layers: otherwise a sparse CPU mask inside a wide candidate block
    // inflates the arena and can trigger unnecessary layer clamping/OOM.
    if (is_cpu_affinity_layer(layer))
      continue;
    for (const char *stem : {"ffn_gate_exps.weight", "ffn_up_exps.weight",
                             "ffn_down_exps.weight"}) {
      const GGUFTensor *m =
          model_tensor(engine, "blk." + std::to_string(layer) + "." + stem);
      if (!m || m->size_bytes == 0 ||
          total > SIZE_MAX - static_cast<size_t>(m->size_bytes))
        return 0;
      total += static_cast<size_t>(m->size_bytes);
    }
  }
  // Boundary-layer (partial_gate_up) accounting is leading-only by
  // construction -- device_expert_partial_gate_up is forced off whenever
  // device_expert_trailing is set (see that setter), so limit < c.n_layers
  // here always refers to the leading block's own boundary.
  if (g.device_expert_partial_gate_up && limit < c.n_layers) {
    const int layer = limit;
    for (const char *stem : {"ffn_gate_exps.weight", "ffn_up_exps.weight"}) {
      const GGUFTensor *m =
          model_tensor(engine, "blk." + std::to_string(layer) + "." + stem);
      if (!m || m->size_bytes == 0 ||
          total > SIZE_MAX - static_cast<size_t>(m->size_bytes))
        return 0;
      total += static_cast<size_t>(m->size_bytes);
    }
  }
  // plan_expert_vram_split()'s per-layer split tensors: M hot experts'
  // combined gate+up+down bytes per split layer. These layers are disjoint
  // from the leading-N/boundary accounting above by construction (the
  // planner never places a layer device_expert_layers/
  // device_expert_partial_gate_up already claimed).
  for (const auto &[layer, ids] : g.split_gpu_expert_ids) {
    if (ids.empty())
      continue;
    const size_t per_expert = engine.compute_expert_slab_bytes(layer);
    if (per_expert == 0)
      continue;
    if (ids.size() > SIZE_MAX / per_expert)
      return 0;
    const size_t layer_bytes = per_expert * ids.size();
    if (total > SIZE_MAX - layer_bytes)
      return 0;
    total += layer_bytes;
  }
  // Phase 1/2 partial-layer compact GPU tensor (build_moe_partial_gpu):
  // qwen4exp_partial_layer_gpu_ids.size() experts' gate+up+down for
  // engine.qwen4exp_partial_layer, on every decode instance that could
  // possibly build it -- Phase 1's g_graph when it might commit, and
  // ALL THREE Phase 2 instances (only SuffixGpu actually builds the
  // tensor, but reserving on Prefix/SuffixCpu too is a small, bounded,
  // always-safe no-op there). Without this, g.expert.buffer is
  // never allocated at all when device_expert_layers<=0 and
  // split_gpu_expert_ids is empty (this feature's only two OTHER
  // consumers) -- build_split_gpu_tensor then silently fails via
  // bind_device_expert_gather's own `!g.expert.buffer` guard,
  // and the whole compact path falls back to ordinary dispatch for
  // that layer (found via real hardware testing, 2026-09-18: the
  // [PARTIAL-LAYER] "compact GPU tensor construction failed... RAM
  // arena not ready?" warning).
  if (g.decode_mode && !engine.qwen4exp_partial_layer_gpu_ids.empty() &&
      engine.qwen4exp_partial_layer >= 0) {
    const size_t per_expert =
        engine.compute_expert_slab_bytes(engine.qwen4exp_partial_layer);
    const size_t n = engine.qwen4exp_partial_layer_gpu_ids.size();
    if (per_expert != 0 && n <= SIZE_MAX / per_expert) {
      const size_t layer_bytes = per_expert * n;
      if (total <= SIZE_MAX - layer_bytes)
        total += layer_bytes;
    }
  }
  // Generalized GPU staging (build_gpu_stage_node): a fixed compact
  // gate+up+down destination per staged host-resident layer, sized for
  // `topk` slots (not n_experts) -- see that function's own comment for
  // why this is safe to size this small. A layer already covered by the
  // static device_expert_layers block doesn't need a separate staging
  // allocation (it's already fully device-resident); skip those to avoid
  // double-counting.
  if (g.decode_mode && engine.qwen4exp_gpu_stage_enabled) {
    for (int layer : engine.qwen4exp_gpu_stage_layers) {
      if (device_expert_block_covers(g, layer))
        continue;
      const size_t per_expert = engine.compute_expert_slab_bytes(layer);
      const size_t topk = static_cast<size_t>(std::max(0, c.topk));
      if (per_expert == 0 || topk == 0 || topk > SIZE_MAX / per_expert)
        continue;
      const size_t layer_bytes = per_expert * topk;
      if (total > SIZE_MAX - layer_bytes)
        return 0;
      total += layer_bytes;
    }
  }
  return total;
}

} // namespace qwen4exp_graph

#endif // ZEROMOE_USE_GGML_CUDA
