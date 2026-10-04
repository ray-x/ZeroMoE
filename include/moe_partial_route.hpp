#pragma once

// Route tables for a hot/cold-split MoE layer (some experts resident in VRAM = "hot", the rest served
// from host RAM = "cold").  Pure C++ (no ggml): every graph builder -- GLM-5.x, DeepSeek-V4, Qwen4Exp,
// and any future model -- obtains its per-branch route tables ONLY through these classes, so the
// "inactive route" behaviour cannot be forgotten or hard-coded differently per model:
//
//   * a route this branch does not own gets mmid_inactive_route_id() (-1 = free skip on the patched
//     ggml for single-token decode, otherwise a valid dummy id) -- see mmid_route_skip.hpp;
//   * a 0/1 mask always accompanies the ids and zeroes any dummy-computed route at the merge, so
//     both sentinel choices are numerically identical;
//   * prefill (n_tokens > 1) never uses the negative skip.
//
// A model supplies only *addressing*: for each expert, its index inside this branch's weight tensor
// (compact position, physical VRAM slot, or full-view expert id) or -1 when the branch does not hold it.

#include "mmid_route_skip.hpp"

#include <cstddef>
#include <cstdint>
#include <cmath>
#include <vector>

namespace moe_partial {

// The execution target is decided once for one layer invocation.  The route
// slot, rather than the expert ID, is the unit of ownership: a router may
// select the same expert more than once and every selected slot still carries
// its own probability contribution.
enum class RouteTarget : uint8_t {
  ResidentGpu,
  StagedGpu,
  Cpu,
};

struct RouteAssignment {
  int32_t expert_id = -1;
  int32_t original_route_slot = -1;
  float router_weight = 0.0f;
  RouteTarget target = RouteTarget::Cpu;
  int32_t weight_slot = -1;
  uint64_t slot_generation = 0;
};

// Immutable per-layer classification.  `resident_index[e]` and
// `staged_index[e]` contain branch-local weight slots, or -1 when that branch
// cannot execute expert e.  A route not resident or staged is assigned to CPU;
// this is the exact fallback for a DDR/SSD service that has not published a
// usable staged slot yet.  Construction rejects invalid IDs, non-finite
// router weights, and overlapping resident/staged ownership.
class RouteClassification {
public:
  static bool build(const std::vector<int32_t> &route_ids,
                    const std::vector<float> &router_weights,
                    const std::vector<int32_t> &resident_index,
                    const std::vector<int32_t> &staged_index,
                    uint64_t slot_generation, RouteClassification *out) {
    if (!out || route_ids.size() != router_weights.size() ||
        resident_index.size() != staged_index.size())
      return false;
    RouteClassification candidate;
    candidate.assignments_.reserve(route_ids.size());
    for (size_t slot = 0; slot < route_ids.size(); ++slot) {
      const int32_t expert = route_ids[slot];
      if (expert < 0 || static_cast<size_t>(expert) >= resident_index.size() ||
          !std::isfinite(router_weights[slot]))
        return false;
      const int32_t resident = resident_index[static_cast<size_t>(expert)];
      const int32_t staged = staged_index[static_cast<size_t>(expert)];
      if (resident >= 0 && staged >= 0)
        return false;
      RouteAssignment assignment;
      assignment.expert_id = expert;
      assignment.original_route_slot = static_cast<int32_t>(slot);
      assignment.router_weight = router_weights[slot];
      assignment.slot_generation = slot_generation;
      if (resident >= 0) {
        assignment.target = RouteTarget::ResidentGpu;
        assignment.weight_slot = resident;
      } else if (staged >= 0) {
        assignment.target = RouteTarget::StagedGpu;
        assignment.weight_slot = staged;
      }
      candidate.assignments_.push_back(assignment);
    }
    // The construction loop appends exactly one assignment for every input
    // slot. Keep this explicit so future target additions cannot silently
    // drop a route while refactoring the classifier.
    if (candidate.assignments_.size() != route_ids.size())
      return false;
    *out = std::move(candidate);
    return true;
  }

  const std::vector<RouteAssignment> &assignments() const { return assignments_; }
  size_t size() const { return assignments_.size(); }

  size_t count(RouteTarget target) const {
    size_t result = 0;
    for (const auto &assignment : assignments_)
      result += assignment.target == target;
    return result;
  }

  // True means every original route slot appears exactly once and no target
  // carries an invalid branch slot. This is the merge invariant checked by
  // graph code before it can submit branch work.
  bool exact_once() const {
    std::vector<uint8_t> seen(assignments_.size(), 0);
    for (const auto &assignment : assignments_) {
      if (assignment.original_route_slot < 0 ||
          static_cast<size_t>(assignment.original_route_slot) >= seen.size() ||
          seen[static_cast<size_t>(assignment.original_route_slot)] ||
          !std::isfinite(assignment.router_weight))
        return false;
      if (assignment.target != RouteTarget::Cpu && assignment.weight_slot < 0)
        return false;
      seen[static_cast<size_t>(assignment.original_route_slot)] = 1;
    }
    for (uint8_t value : seen)
      if (!value) return false;
    return true;
  }

private:
  std::vector<RouteAssignment> assignments_;
};

enum class SentinelPolicy {
  Auto,       // default: negative skip when supported for this shape, else dummy 0
  ForceDummy, // always the valid dummy id (A/B tests, unpatched-ggml simulation)
};

// One branch of a split layer.  Immutable after construction.
class ExpertBranchMap {
public:
  // index_of_expert[e]: index inside this branch's weight tensor, or < 0 when the branch cannot address e.
  // owned[e]:           whether this branch is responsible for e's contribution (mask = 1).  When empty it
  //                     defaults to index_of_expert[e] >= 0.
  static ExpertBranchMap build(const std::vector<int32_t> &index_of_expert,
                               int n_tokens, bool single_token_decode,
                               const std::vector<uint8_t> &owned = {},
                               SentinelPolicy policy = SentinelPolicy::Auto) {
    ExpertBranchMap m;
    m.n_experts_ = (int)index_of_expert.size();
    m.n_tokens_ = n_tokens;
    m.inactive_ = policy == SentinelPolicy::ForceDummy
                      ? 0
                      : mmid_inactive_route_id(single_token_decode &&
                                               n_tokens == 1);
    const size_t n = index_of_expert.size();
    m.ids_.assign(n * (size_t)n_tokens, m.inactive_);
    m.mask_.assign(n * (size_t)n_tokens, 0.0f);
    for (int t = 0; t < n_tokens; ++t) {
      for (size_t e = 0; e < n; ++e) {
        const size_t i = (size_t)t * n + e;
        if (index_of_expert[e] >= 0)
          m.ids_[i] = index_of_expert[e];
        const bool own = owned.empty() ? index_of_expert[e] >= 0
                                       : (e < owned.size() && owned[e] != 0);
        m.mask_[i] = own ? 1.0f : 0.0f;
      }
    }
    return m;
  }

  // [n_tokens * n_experts], token-major: ids()[t * n_experts + e]
  const std::vector<int32_t> &route_ids() const { return ids_; }
  const std::vector<float> &route_mask() const { return mask_; }
  int32_t inactive_id() const { return inactive_; }
  bool skips_inactive_routes() const { return inactive_ < 0; }
  int n_experts() const { return n_experts_; }
  int n_tokens() const { return n_tokens_; }

private:
  std::vector<int32_t> ids_;
  std::vector<float> mask_;
  int32_t inactive_ = 0;
  int n_experts_ = 0;
  int n_tokens_ = 0;
};

// Both branches of one layer.
class HotColdSplit {
public:
  // hot_index / cold_index: per-expert addressing as described above (-1 = not held by that branch).
  // exact_partition: require every expert to be held by exactly one branch (the static, non-dynamic
  // residency contract).  Returns false and leaves the object empty when violated.
  static bool build(const std::vector<int32_t> &hot_index,
                    const std::vector<int32_t> &cold_index, int n_tokens,
                    bool single_token_decode, bool exact_partition,
                    HotColdSplit *out,
                    SentinelPolicy policy = SentinelPolicy::Auto) {
    if (hot_index.size() != cold_index.size())
      return false;
    if (exact_partition)
      for (size_t e = 0; e < hot_index.size(); ++e)
        if ((hot_index[e] >= 0) == (cold_index[e] >= 0))
          return false;
    out->hot_ = ExpertBranchMap::build(hot_index, n_tokens,
                                       single_token_decode, {}, policy);
    out->cold_ = ExpertBranchMap::build(cold_index, n_tokens,
                                        single_token_decode, {}, policy);
    return true;
  }

  const ExpertBranchMap &hot() const { return hot_; }
  const ExpertBranchMap &cold() const { return cold_; }

private:
  ExpertBranchMap hot_, cold_;
};

} // namespace moe_partial
