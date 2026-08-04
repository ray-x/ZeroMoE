// route_usage.hpp — CUDA-free routing-frequency histogram.
//
// Tracks how often each (layer, expert) pair is selected by the router, and
// produces the pin-file format the engine's startup preload consumes:
//
//   <layer> <expert> <count>
//
// This is intentionally free of engine.hpp / CUDA headers so it can be used
// by both the full (CUDA) engine and the CPU-only binary.
#pragma once
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

class RouteUsage {
public:
  // Increment the counter for one routed expert selection.
  void bump(int layer, int expert) { counts_[key(layer, expert)]++; }

  // Increment by a pre-merged amount (used when merging with an on-disk file).
  void bump(int layer, int expert, uint64_t delta) {
    if (delta)
      counts_[key(layer, expert)] += delta;
  }

  // Load a file in the format "layer expert count" and merge it into memory.
  bool load(const std::string &path) {
    std::ifstream f(path);
    if (!f.is_open())
      return false;

    std::string line;
    while (std::getline(f, line)) {
      std::istringstream ss(line);
      int layer, expert;
      uint64_t count;
      if (ss >> layer >> expert >> count)
        bump(layer, expert, count);
    }
    return true;
  }

  // Write the current histogram to a file, sorted by (layer, count desc).
  bool dump(const std::string &path) const {
    std::ofstream f(path);
    if (!f.is_open()) {
      std::cerr << "[ROUTE-USAGE] Cannot write: " << path << "\n";
      return false;
    }

    // Group by layer so the output is deterministic and human-readable.
    std::unordered_map<int, std::vector<std::pair<int, uint64_t>>> by_layer;
    for (const auto &[k, count] : counts_) {
      auto [layer, expert] = split(k);
      by_layer[layer].push_back({expert, count});
    }

    std::vector<int> layers;
    for (const auto &p : by_layer)
      layers.push_back(p.first);
    std::sort(layers.begin(), layers.end());

    for (int layer : layers) {
      auto &vec = by_layer[layer];
      std::sort(vec.begin(), vec.end(), [](const auto &a, const auto &b) {
        return a.second > b.second;
      });
      for (const auto &[expert, count] : vec) {
        f << layer << " " << expert << " " << count << "\n";
      }
    }
    return true;
  }

  // Return the top-K experts for a layer, sorted by count descending.
  std::vector<int> topk_for_layer(int layer, int k) const {
    std::vector<std::pair<int, uint64_t>> vec;
    for (const auto &[hk, count] : counts_) {
      auto [l, e] = split(hk);
      if (l == layer)
        vec.push_back({e, count});
    }
    std::sort(vec.begin(), vec.end(),
              [](const auto &a, const auto &b) { return a.second > b.second; });
    if (k > (int)vec.size())
      k = (int)vec.size();
    std::vector<int> out;
    out.reserve(k);
    for (int i = 0; i < k; ++i)
      out.push_back(vec[i].first);
    return out;
  }

  // Experts with count >= threshold, returned as a (layer,expert) set.
  std::unordered_set<uint64_t> pin_set(int topk_per_layer,
                                       uint64_t threshold) const {
    std::unordered_set<uint64_t> out;
    std::unordered_map<int, std::vector<std::pair<int, uint64_t>>> by_layer;
    for (const auto &[k, count] : counts_) {
      if (count < threshold)
        continue;
      auto [layer, expert] = split(k);
      by_layer[layer].push_back({expert, count});
    }
    for (auto &[layer, vec] : by_layer) {
      std::sort(vec.begin(), vec.end(), [](const auto &a, const auto &b) {
        return a.second > b.second;
      });
      int take = std::min(topk_per_layer, (int)vec.size());
      for (int i = 0; i < take; ++i)
        out.insert(key(layer, vec[i].first));
    }
    return out;
  }

  size_t size() const { return counts_.size(); }
  bool empty() const { return counts_.empty(); }

  void clear() { counts_.clear(); }

  // Merge another histogram into this one.
  void merge(const RouteUsage &other) {
    for (const auto &[k, count] : other.counts_)
      counts_[k] += count;
  }

private:
  std::unordered_map<uint64_t, uint64_t> counts_;

  static uint64_t key(int layer, int expert) {
    return (static_cast<uint64_t>(layer) << 32) | static_cast<uint32_t>(expert);
  }
  static std::pair<int, int> split(uint64_t k) {
    return {static_cast<int>(k >> 32), static_cast<int>(k & 0xFFFFFFFFu)};
  }
};
