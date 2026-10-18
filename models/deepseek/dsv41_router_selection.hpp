#pragma once

#include <algorithm>
#include <cstddef>
#include <functional>
#include <limits>
#include <vector>

struct Dsv41RouterScore {
  int id = -1;
  float score = 0.0f;
};

// Resolve a numerically ambiguous top-k boundary using a deterministic expert
// ID order. Scores strictly outside `epsilon` keep their usual score order;
// only candidates within epsilon of the kth score use the ID tie-break.
// Returns true when this call used the boundary rule.
inline bool dsv41_select_topk_stable_boundary(
    std::vector<Dsv41RouterScore> &scores, int n_experts, int topk,
    float epsilon, std::vector<int> &selected_ids) {
  selected_ids.clear();
  if (topk <= 0 || n_experts < topk ||
      scores.size() < static_cast<size_t>(n_experts))
    return false;

  std::partial_sort(scores.begin(), scores.begin() + topk,
                    scores.begin() + n_experts,
                    [](const auto &a, const auto &b) {
                      return a.score > b.score;
                    });
  const float cutoff = scores[static_cast<size_t>(topk - 1)].score;
  float next = -std::numeric_limits<float>::infinity();
  for (int i = topk; i < n_experts; ++i)
    next = std::max(next, scores[static_cast<size_t>(i)].score);
  if (cutoff - next > epsilon) {
    for (int i = 0; i < topk; ++i)
      selected_ids.push_back(scores[static_cast<size_t>(i)].id);
    return false;
  }

  std::vector<Dsv41RouterScore> strict;
  std::vector<int> boundary;
  strict.reserve(static_cast<size_t>(topk));
  boundary.reserve(static_cast<size_t>(topk) + 4);
  for (int i = 0; i < n_experts; ++i) {
    const auto &candidate = scores[static_cast<size_t>(i)];
    if (candidate.score > cutoff + epsilon)
      strict.push_back(candidate);
    else if (candidate.score >= cutoff - epsilon)
      boundary.push_back(candidate.id);
  }
  std::sort(strict.begin(), strict.end(), [](const auto &a, const auto &b) {
    return a.score != b.score ? a.score > b.score : a.id > b.id;
  });
  std::sort(boundary.begin(), boundary.end(), std::greater<int>());
  for (const auto &candidate : strict)
    selected_ids.push_back(candidate.id);
  for (int id : boundary) {
    if (selected_ids.size() >= static_cast<size_t>(topk)) break;
    selected_ids.push_back(id);
  }

  if (selected_ids.size() != static_cast<size_t>(topk)) {
    selected_ids.clear();
    for (int i = 0; i < topk; ++i)
      selected_ids.push_back(scores[static_cast<size_t>(i)].id);
    return false;
  }
  return true;
}
