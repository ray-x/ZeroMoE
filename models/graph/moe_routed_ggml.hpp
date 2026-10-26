#pragma once

// ggml building blocks for routed-MoE layers, shared by every model's graph
// builder: router -> top-k ids -> normalised weights, one expert FFN over a set
// of expert tensors, and the weighted sum over the top-k axis.
//
// The model-specific parts (which expert tensors a branch uses, residency maps,
// miss nodes, tracing) stay in the caller; this header only owns the op
// sequences that used to be copied into each builder. Node creation order is
// part of the contract: callers compare logits byte-for-byte across refactors.

#include "ggml.h"

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace moe_routed {

enum class Gating { Softmax, Sigmoid, SqrtSoftplus };

struct RouterSpec {
  Gating gating = Gating::Softmax;
  int n_experts = 0;
  int topk = 0;
  int n_tokens = 1;
  int n_groups = 0;
  int n_groups_used = 0;
  bool weights_norm = false;  // divide by the sum of the selected weights
  float weights_scale = 0.0f; // 0 or 1 = no scaling
};

// Router probabilities [n_experts, n_tokens] from the router matmul.
inline ggml_tensor *gate_probs(ggml_context *ctx, const RouterSpec &spec,
                               ggml_tensor *router, ggml_tensor *cur) {
  ggml_tensor *logits = ggml_mul_mat(ctx, router, cur);
  switch (spec.gating) {
  case Gating::Sigmoid:
    return ggml_sigmoid(ctx, logits);
  case Gating::SqrtSoftplus:
    return ggml_sqrt(ctx, ggml_softplus(ctx, logits));
  case Gating::Softmax:
  default:
    return ggml_soft_max(ctx, logits);
  }
}

// Top-k expert ids [topk, n_tokens] of probs (+ optional selection bias).
inline ggml_tensor *select_ids(ggml_context *ctx, const RouterSpec &spec,
                               ggml_tensor *probs, ggml_tensor *selection_bias) {
  ggml_tensor *selection =
      selection_bias ? ggml_add(ctx, probs, selection_bias) : probs;
  if (spec.n_groups > 1 && spec.n_groups_used > 0 &&
      spec.n_groups_used < spec.n_groups &&
      spec.n_experts % spec.n_groups == 0) {
    const int experts_per_group = spec.n_experts / spec.n_groups;
    ggml_tensor *groups = ggml_reshape_3d(
        ctx, selection, experts_per_group, spec.n_groups, spec.n_tokens);
    ggml_tensor *top2_ids = ggml_argsort_top_k(ctx, groups, 2);
    ggml_tensor *group_rows = ggml_reshape_4d(
        ctx, groups, 1, experts_per_group, spec.n_groups, spec.n_tokens);
    ggml_tensor *top2 = ggml_get_rows(ctx, group_rows, top2_ids);
    ggml_tensor *scores = ggml_reshape_3d(
        ctx, top2, top2->ne[1], top2->ne[2], top2->ne[3]);
    scores = ggml_sum_rows(ctx, scores);
    scores = ggml_reshape_2d(ctx, scores, spec.n_groups, spec.n_tokens);
    ggml_tensor *group_ids =
        ggml_argsort_top_k(ctx, scores, spec.n_groups_used);
    ggml_tensor *selected = ggml_get_rows(ctx, groups, group_ids);
    selection = ggml_set_rows(
        ctx, ggml_fill(ctx, groups, -INFINITY), selected, group_ids);
    selection =
        ggml_reshape_2d(ctx, selection, spec.n_experts, spec.n_tokens);
  }
  return ggml_argsort_top_k(ctx, selection, spec.topk);
}

// Router weights [1, topk, n_tokens] of the selected experts, normalised and
// scaled per the spec.
inline ggml_tensor *route_weights(ggml_context *ctx, const RouterSpec &spec,
                                  ggml_tensor *probs, ggml_tensor *ids) {
  ggml_tensor *probs3 =
      ggml_reshape_3d(ctx, probs, 1, spec.n_experts, spec.n_tokens);
  ggml_tensor *picked = ggml_get_rows(ctx, probs3, ids);
  ggml_tensor *w2 = ggml_reshape_2d(ctx, picked, spec.topk, spec.n_tokens);
  if (spec.weights_norm) {
    ggml_tensor *sum = ggml_sum_rows(ctx, w2);
    sum = ggml_clamp(ctx, sum, 6.103515625e-5f, INFINITY);
    w2 = ggml_div(ctx, w2, sum);
  }
  if (spec.weights_scale != 0.0f && spec.weights_scale != 1.0f)
    w2 = ggml_scale(ctx, w2, spec.weights_scale);
  return ggml_reshape_3d(ctx, w2, 1, spec.topk, spec.n_tokens);
}

// One routed branch's expert tensors. Either gate_up (gate rows then up rows
// per expert) or both gate and up must be set.
struct ExpertWeights {
  ggml_tensor *gate_up = nullptr;
  ggml_tensor *gate = nullptr;
  ggml_tensor *up = nullptr;
  ggml_tensor *down = nullptr;
  bool valid() const { return down && (gate_up || (gate && up)); }
};

struct FfnNodes {
  ggml_tensor *gate_up = nullptr;  // only on the merged path
  ggml_tensor *gate = nullptr;
  ggml_tensor *up = nullptr;
  ggml_tensor *act = nullptr;
  ggml_tensor *down = nullptr;  // [D, topk, n_tokens], not yet weighted
};

// gate/up -> SwiGLU -> down for the experts named by `ids` ([topk, n_tokens]).
// `input3` is [D, 1, n_tokens]. `clamp` > 1e-6 applies the model's SwiGLU
// clamp: `clamp_fused` clamps gate/up before the activation, otherwise the
// activation output is clamped.
inline FfnNodes expert_ffn(ggml_context *ctx, const ExpertWeights &w,
                           ggml_tensor *input3, ggml_tensor *ids,
                           float clamp = 0.0f, bool clamp_fused = false) {
  FfnNodes n;
  if (w.gate_up) {
    // One routed matmul for the adjacent gate+up planes, then zero-copy views.
    n.gate_up = ggml_mul_mat_id(ctx, w.gate_up, input3, ids);
    const int64_t half = n.gate_up->ne[0] / 2;
    n.gate = ggml_view_3d(ctx, n.gate_up, half, n.gate_up->ne[1],
                          n.gate_up->ne[2], n.gate_up->nb[1], n.gate_up->nb[2],
                          0);
    n.up = ggml_view_3d(ctx, n.gate_up, half, n.gate_up->ne[1],
                        n.gate_up->ne[2], n.gate_up->nb[1], n.gate_up->nb[2],
                        static_cast<size_t>(half) * n.gate_up->nb[0]);
  } else {
    n.up = ggml_mul_mat_id(ctx, w.up, input3, ids);
    n.gate = ggml_mul_mat_id(ctx, w.gate, input3, ids);
  }
  if (clamp > 1e-6f && clamp_fused) {
    ggml_tensor *up_c = ggml_clamp(ctx, n.up, -clamp, clamp);
    ggml_tensor *gate_c = ggml_clamp(ctx, n.gate, -INFINITY, clamp);
    n.act = ggml_mul(ctx, ggml_silu(ctx, gate_c), up_c);
  } else {
    n.act = ggml_swiglu_split(ctx, n.gate, n.up);
    if (clamp > 1e-6f)
      n.act = ggml_clamp(ctx, n.act, -clamp, clamp);
  }
  n.down = ggml_mul_mat_id(ctx, w.down, n.act, ids);
  return n;
}

// Sum over the top-k axis of `experts` ([D, topk, n_tokens]) -> [D, n_tokens].
inline ggml_tensor *sum_topk(ggml_context *ctx, ggml_tensor *experts, int64_t d,
                             int64_t n_tokens, int topk) {
  ggml_tensor *out =
      ggml_view_2d(ctx, experts, d, n_tokens, experts->nb[2], 0);
  // Keep the first expert as a VIEW when topk > 1. ggml-cuda recognizes
  // MUL(expert_outputs, router_weights) -> expert views -> ADD reduction as
  // one weighted-MoE reduction kernel. Materializing view 0 with CONT inserts
  // a node between the MUL and view/add chain, disables that fusion, and
  // launches one small broadcast-add kernel for each remaining top-k route.
  // This matches llama.cpp's build_moe_ffn aggregation order.
  if (topk <= 1)
    return ggml_cont(ctx, out);
  for (int k = 1; k < topk; ++k)
    out = ggml_add(ctx, out,
                   ggml_view_2d(ctx, experts, d, n_tokens, experts->nb[2],
                                static_cast<size_t>(k) * experts->nb[1]));
  return out;
}

}  // namespace moe_routed
