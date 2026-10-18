#pragma once

// ggml adapter for moe_partial_route.hpp: turns an ExpertBranchMap's constant tables into the per-branch
// MUL_MAT_ID inputs, and merges the hot and cold branch outputs.  Graph builders call these instead of
// re-implementing the get_rows / mask / merge sequence, so every model inherits the same routing
// semantics (including the inactive-route default) from moe_partial_route.hpp.

#include "ggml.h"
#include "moe_partial_route.hpp"

namespace moe_partial {

struct BranchInputs {
  ggml_tensor *ids = nullptr;     // [topk, n_tokens] I32: branch-local index, or the inactive id
  ggml_tensor *weights = nullptr; // [1, topk, n_tokens] F32: router weights, zeroed for routes not owned
};

// route_ids_tensor / route_mask_tensor: constant graph inputs shaped [1, n_experts, n_tokens] (I32 / F32)
// holding ExpertBranchMap::route_ids() / route_mask() of one branch.
// ids: [topk, n_tokens] I32 router selection; weights: [1, topk, n_tokens] F32 router weights.
inline BranchInputs make_branch_inputs(ggml_context *ctx,
                                       ggml_tensor *route_ids_tensor,
                                       ggml_tensor *route_mask_tensor,
                                       ggml_tensor *ids, ggml_tensor *weights,
                                       int topk, int n_tokens) {
  BranchInputs r;
  if (!ctx || !route_ids_tensor || !route_mask_tensor || !ids || !weights)
    return r;
  ggml_tensor *mapped = ggml_get_rows(ctx, route_ids_tensor, ids);
  ggml_tensor *mask = ggml_get_rows(ctx, route_mask_tensor, ids);
  r.ids = ggml_reshape_2d(ctx, mapped, topk, n_tokens);
  r.weights =
      ggml_mul(ctx, weights, ggml_reshape_3d(ctx, mask, 1, topk, n_tokens));
  return r;
}

// Runs the hot and/or cold branch and sums their outputs.  `run_hot` / `run_cold` are callables taking
// the branch's BranchInputs and returning the branch's [D, n_tokens] output (nullptr on failure).
// Returns nullptr when a requested branch fails or when neither branch exists.
template <class HotFn, class ColdFn>
ggml_tensor *combine_hot_cold(ggml_context *ctx, bool has_hot, bool has_cold,
                              const BranchInputs &hot_in,
                              const BranchInputs &cold_in, HotFn &&run_hot,
                              ColdFn &&run_cold) {
  ggml_tensor *out = nullptr;
  if (has_hot) {
    out = run_hot(hot_in);
    if (!out)
      return nullptr;
  }
  if (has_cold) {
    ggml_tensor *cold = run_cold(cold_in);
    if (!cold)
      return nullptr;
    out = out ? ggml_add(ctx, out, cold) : cold;
  }
  return out;
}

} // namespace moe_partial
