#pragma once

// Option-B "global pool" hot-set selection.
//
// The per-layer Option-B mode gives every selected layer the same number of
// VRAM-resident experts (qwen4exp_option_b_top_n), however concentrated that
// layer's routing is. The global mode instead ranks every (layer, expert) pair
// of the selected layers by its usage count and keeps the top K of the whole
// pool, so a layer with concentrated routing gets many hot experts and a flat
// layer few. Every layer routes the same number of tokens, so raw counts are
// directly comparable across layers and the pooled top-K maximizes the expected
// share of routed experts that land in VRAM for a fixed expert budget.
//
// Header-only and free of engine types so main.cpp (preload/gating pass) and
// the graph planner (arena layout) call the same code and cannot disagree.

#include <algorithm>
#include <cstdint>
#include <map>
#include <vector>

namespace option_b_global {

struct Result {
  // layer -> hot expert ids, ordered by that layer's own descending usage
  // (ties by ascending expert id), the same order the per-layer mode uses.
  std::map<int, std::vector<int>> hot;
  // Fraction of the selected layers' routed traffic covered by the hot sets,
  // per layer and over the whole pool (0 when a layer has no usage).
  std::map<int, double> layer_coverage;
  double pool_coverage = 0.0;
  size_t total_hot = 0;
};

// Entry needs .layer, .expert and .count (StreamingEngine::UsageEntry fits).
//   layers          candidate layers (duplicates and out-of-range ids ignored)
//   top_k           total experts to keep over all candidate layers
//   floor_per_layer every candidate layer with usage keeps at least this many
//                   of its own hottest experts (clamped so it never exceeds
//                   top_k / number of layers)
//   max_per_layer   ceiling on one layer's hot count (<= 0 or >= n_experts
//                   means no ceiling)
template <typename Entry>
Result select(const std::vector<Entry> &entries, const std::vector<int> &layers,
              int n_layers, int n_experts, int top_k, int floor_per_layer,
              int max_per_layer) {
  Result out;
  if (top_k <= 0 || n_experts <= 0 || n_layers <= 0)
    return out;

  std::vector<int> cand;
  for (int layer : layers) {
    if (layer < 0 || layer >= n_layers)
      continue;
    if (std::find(cand.begin(), cand.end(), layer) == cand.end())
      cand.push_back(layer);
  }
  std::sort(cand.begin(), cand.end());
  if (cand.empty())
    return out;

  const int cap =
      (max_per_layer > 0 && max_per_layer < n_experts) ? max_per_layer
                                                       : n_experts;

  std::map<int, std::vector<uint64_t>> counts;
  for (int layer : cand)
    counts[layer].assign(static_cast<size_t>(n_experts), 0);
  for (const auto &e : entries) {
    auto it = counts.find(e.layer);
    if (it == counts.end() || e.expert < 0 || e.expert >= n_experts)
      continue;
    it->second[static_cast<size_t>(e.expert)] += static_cast<uint64_t>(e.count);
  }

  struct Item {
    int layer;
    int expert;
    uint64_t count;
  };
  auto hotter = [](const Item &a, const Item &b) {
    if (a.count != b.count)
      return a.count > b.count;
    if (a.layer != b.layer)
      return a.layer < b.layer;
    return a.expert < b.expert;
  };

  std::map<int, uint64_t> layer_total;
  std::vector<Item> pool;
  std::map<int, std::vector<Item>> by_layer;
  for (int layer : cand) {
    uint64_t total = 0;
    auto &v = by_layer[layer];
    for (int ex = 0; ex < n_experts; ++ex) {
      const uint64_t c = counts[layer][static_cast<size_t>(ex)];
      total += c;
      v.push_back({layer, ex, c});
    }
    layer_total[layer] = total;
    std::stable_sort(v.begin(), v.end(), hotter);
  }

  // Layers with no usage at all cannot be ranked: they get nothing here and
  // stay CPU-side (same as the per-layer mode).
  std::vector<int> live;
  for (int layer : cand)
    if (layer_total[layer] > 0)
      live.push_back(layer);
  if (live.empty())
    return out;

  int floor_n = std::max(0, floor_per_layer);
  floor_n = std::min(floor_n, cap);
  floor_n = std::min(floor_n, top_k / static_cast<int>(live.size()));

  std::map<int, std::vector<char>> taken;
  std::map<int, int> taken_n;
  size_t placed = 0;
  for (int layer : live) {
    taken[layer].assign(static_cast<size_t>(n_experts), 0);
    for (int i = 0; i < floor_n; ++i) {
      taken[layer][static_cast<size_t>(by_layer[layer][static_cast<size_t>(i)]
                                           .expert)] = 1;
      ++taken_n[layer];
      ++placed;
    }
  }

  for (int layer : live)
    for (const auto &item : by_layer[layer])
      if (!taken[layer][static_cast<size_t>(item.expert)])
        pool.push_back(item);
  std::stable_sort(pool.begin(), pool.end(), hotter);

  for (const auto &item : pool) {
    if (placed >= static_cast<size_t>(top_k))
      break;
    if (taken_n[item.layer] >= cap)
      continue;
    taken[item.layer][static_cast<size_t>(item.expert)] = 1;
    ++taken_n[item.layer];
    ++placed;
  }

  uint64_t covered_all = 0, total_all = 0;
  for (int layer : live) {
    // Unselected layers still contribute traffic to the pool denominator
    // (e.g. top_k < live.size(), when the floor is clamped to zero).
    total_all += layer_total[layer];
    std::vector<int> ids;
    uint64_t covered = 0;
    for (const auto &item : by_layer[layer]) {
      if (taken[layer][static_cast<size_t>(item.expert)]) {
        ids.push_back(item.expert);
        covered += item.count;
      }
    }
    out.layer_coverage[layer] =
        static_cast<double>(covered) / static_cast<double>(layer_total[layer]);
    if (ids.empty())
      continue;
    covered_all += covered;
    out.total_hot += ids.size();
    out.hot[layer] = std::move(ids);
  }
  out.pool_coverage = total_all == 0 ? 0.0
                                     : static_cast<double>(covered_all) /
                                           static_cast<double>(total_all);
  return out;
}

} // namespace option_b_global
