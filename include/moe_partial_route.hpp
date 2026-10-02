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
#include <vector>

namespace moe_partial {

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
