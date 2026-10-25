#pragma once

// Family-neutral routed-MoE graph block (R2.1 B/E).
//
// A family supplies the mathematics (router -> ids/weights, shared expert, SwiGLU
// clamp) and a placement (hot VRAM branch, cold branch). This block owns what is
// common to every mixed VRAM/DDR model: per-branch route inputs, the early-ready
// lane fork, the exact weighted merge, and the shared-expert add. Which executor
// runs the cold branch is decided by the ExecutionPolicy's MoEExecutionStrategy,
// never by the family.

#include "early_ready_graph_builder.hpp"
#include "execution_policy.hpp"
#include "moe_execution_strategy.hpp"
#include "moe_partial_route_ggml.hpp"
#include "moe_routed_ggml.hpp"

#include <functional>
#include <memory>

namespace zeromoe::graph {
class PhaseExpertCache;

// One expert branch: compact expert tensors plus the constant maps that translate
// the model's expert ids into branch-local ids / ownership mask.
struct MoEBranch {
  moe_routed::ExpertWeights weights;
  ggml_tensor *route_map = nullptr;  // [1, n_experts, n_tokens] I32
  ggml_tensor *route_mask = nullptr; // [1, n_experts, n_tokens] F32
  bool present = false;              // the branch owns at least one expert
};

struct MoEBlockGraph {
  PhaseExpertCache *phase_cache = nullptr;
  const StreamingEngine *engine = nullptr;
  ggml_context *ctx = nullptr;
  ggml_cgraph *graph = nullptr;
  ggml_backend_sched_t sched = nullptr;
  ggml_backend_t cuda = nullptr;
  ggml_backend_t cpu = nullptr;
  int device = -1;
  bool decode = false;
  int n_tokens = 0;
  const char *name_prefix = "moe";
  // Prefill only: keep the weighted per-branch sum on this backend so the
  // [D, topk, S] tensor never crosses GPUs through the host.
  bool pin_prefill_branch_sums = false;
};

struct MoEBlockLayer {
  int layer = -1;
  ggml_tensor *input = nullptr;         // [D, S] F32 normed hidden state
  ggml_tensor *route_ids = nullptr;     // [topk, S] I32
  ggml_tensor *route_weights = nullptr; // [1, topk, S] F32
  float swiglu_clamp = 0.0f;
  bool swiglu_clamp_fused = false;
};

using BranchRunFn = std::function<ggml_tensor *(
    const moe_routed::ExpertWeights &, ggml_tensor *ids, ggml_tensor *weights,
    ggml_backend_t pin_backend)>;

struct MoEBlockHooks {
  // Rewrites the cold branch inputs before execution (SSD miss nodes, staging).
  std::function<bool(moe_partial::BranchInputs &)> adjust_cold_inputs;
  // Replaces the plain cold-branch runner (prefetch / split variants).
  std::function<ggml_tensor *(const moe_partial::BranchInputs &,
                              const BranchRunFn &)> run_cold;
  // Shared-expert output, built on `input` (nullptr when the layer has none).
  std::function<ggml_tensor *()> build_shared;
  // Takes ownership of a freshly built lane; installs residency/assist state and
  // returns the stable pointer used for the rest of the block (nullptr = reject).
  std::function<EarlyReadyLane *(std::unique_ptr<EarlyReadyLane>)> adopt_lane;
  // Adds branches that live outside the hot/cold split (second-GPU hot set,
  // staged ring) to `out`.
  std::function<ggml_tensor *(ggml_tensor *out)> extra_branches;
  // Called with the lane before its CPU marker is expanded (post-stage nodes).
  std::function<void(EarlyReadyLane &)> before_marker;
};

struct MoEBlockResult {
  ggml_tensor *out = nullptr;      // [D, S]; nullptr on failure
  EarlyReadyLane *lane = nullptr;  // non-null when the early-ready path was built
};

inline ggml_tensor *build_routed_branch(const MoEBlockSpec &spec,
                                        const MoEBlockGraph &bg,
                                        const MoEBlockLayer &bl,
                                        ggml_tensor *input3,
                                        const moe_routed::ExpertWeights &w,
                                        ggml_tensor *ids, ggml_tensor *weights,
                                        ggml_backend_t pin_backend = nullptr) {
  if (!w.down || !ids || !weights || (!w.gate_up && (!w.gate || !w.up)))
    return nullptr;
  const moe_routed::FfnNodes ffn = moe_routed::expert_ffn(
      bg.ctx, w, input3, ids, bl.swiglu_clamp, bl.swiglu_clamp_fused);
  ggml_tensor *experts = ggml_mul(bg.ctx, ffn.down, weights);
  ggml_tensor *summed =
      moe_routed::sum_topk(bg.ctx, experts, spec.d_model, bg.n_tokens, spec.topk);
  if (pin_backend && bg.sched && bg.pin_prefill_branch_sums && !bg.decode) {
    ggml_backend_sched_set_tensor_backend(bg.sched, experts, pin_backend);
    ggml_backend_sched_set_tensor_backend(bg.sched, summed, pin_backend);
  }
  return summed;
}

// Builds hot + cold + shared. `lane_wanted` is the placement's request for
// overlap; the resolved ExecutionPolicyDecision decides whether it is honoured.
inline MoEBlockResult build_moe_block(const MoEBlockSpec &spec,
                                      const MoEBlockGraph &bg,
                                      const MoEBlockLayer &bl,
                                      const MoEBranch &hot, const MoEBranch &cold,
                                      bool lane_wanted,
                                      const ExecutionPolicyDecision &decision,
                                      const MoEBlockHooks &hooks = {}) {
  MoEBlockResult res;
  if (!bg.ctx || !bl.input || !bl.route_ids || !bl.route_weights)
    return res;
  ggml_context *ctx = bg.ctx;
  ggml_tensor *input3 =
      ggml_reshape_3d(ctx, bl.input, spec.d_model, 1, bg.n_tokens);
  const BranchRunFn run = [&](const moe_routed::ExpertWeights &w,
                              ggml_tensor *ids, ggml_tensor *weights,
                              ggml_backend_t pin) {
    return build_routed_branch(spec, bg, bl, input3, w, ids, weights, pin);
  };

  moe_partial::BranchInputs hot_in, cold_in;
  if (hot.present) {
    if (!hot.route_map || !hot.route_mask) return res;
    hot_in = moe_partial::make_branch_inputs(ctx, hot.route_map, hot.route_mask,
                                             bl.route_ids, bl.route_weights,
                                             spec.topk, bg.n_tokens);
  }
  if (cold.present) {
    if (!cold.route_map || !cold.route_mask) return res;
    cold_in = moe_partial::make_branch_inputs(
        ctx, cold.route_map, cold.route_mask, bl.route_ids, bl.route_weights,
        spec.topk, bg.n_tokens);
    if (hooks.adjust_cold_inputs && !hooks.adjust_cold_inputs(cold_in))
      return res;
  }

  const bool overlap_ok = decision.moe == MoEExecutionKind::EarlyReadyOffload;
  ggml_tensor *out = nullptr;
  ggml_tensor *shared = nullptr;
  if (cold.present && lane_wanted && overlap_ok && hooks.adopt_lane) {
    auto built = build_early_ready_graph_lane({
        .sched = bg.sched, .cuda = bg.cuda, .cpu = bg.cpu, .ctx = ctx,
        .graph = bg.graph, .device = bg.device, .layer = bl.layer,
        .d_model = spec.d_model, .topk = spec.topk, .decode = bg.decode,
        .n_tokens = bg.n_tokens, .swiglu_clamp = bl.swiglu_clamp,
        .swiglu_clamp_fused = bl.swiglu_clamp_fused,
        .name_prefix = bg.name_prefix, .input = bl.input,
        .route_ids = bl.route_ids, .route_weights = bl.route_weights,
        .cold_remap = cold.route_map, .cold_mask = cold.route_mask,
        .cold_weights = cold.weights});
    if (built) res.lane = hooks.adopt_lane(std::move(built));
  }

  if (res.lane) {
    if (hooks.before_marker) hooks.before_marker(*res.lane);
    // fork -> hot GPU work -> shared expert -> CPU lane marker -> merge. The
    // marker goes in after everything the GPU can do without the cold result.
    ggml_tensor *lane_input3 =
        ggml_reshape_3d(ctx, res.lane->gpu_input, spec.d_model, 1, bg.n_tokens);
    ggml_tensor *hot_out = nullptr;
    if (hot.present) {
      const BranchRunFn lane_run = [&](const moe_routed::ExpertWeights &w,
                                       ggml_tensor *ids, ggml_tensor *weights,
                                       ggml_backend_t pin) {
        return build_routed_branch(spec, bg, bl, lane_input3, w, ids, weights,
                                   pin);
      };
      hot_out = lane_run(hot.weights, hot_in.ids, hot_in.weights, nullptr);
      if (!hot_out) return res;
      ggml_build_forward_expand(bg.graph, hot_out);
    }
    if (hooks.build_shared) shared = hooks.build_shared();
    if (shared) ggml_build_forward_expand(bg.graph, shared);
    ggml_build_forward_expand(bg.graph, res.lane->cpu_marker);
    ggml_tensor *cold_out = moe_routed::sum_topk(
        ctx, res.lane->gpu_output, spec.d_model, bg.n_tokens, spec.topk);
    out = hot_out ? ggml_add(ctx, hot_out, cold_out) : cold_out;
  } else {
    out = moe_partial::combine_hot_cold(
        ctx, hot.present, cold.present, hot_in, cold_in,
        [&](const moe_partial::BranchInputs &in) {
          return run(hot.weights, in.ids, in.weights, nullptr);
        },
        [&](const moe_partial::BranchInputs &in) -> ggml_tensor * {
          if (hooks.run_cold) return hooks.run_cold(in, run);
          return run(cold.weights, in.ids, in.weights, nullptr);
        });
  }
  if (!out) return res;
  if (hooks.extra_branches) {
    out = hooks.extra_branches(out);
    if (!out) return res;
  }
  if (!shared && hooks.build_shared) shared = hooks.build_shared();
  if (shared) out = ggml_add(ctx, out, shared);
  res.out = out;
  return res;
}

} // namespace zeromoe::graph
