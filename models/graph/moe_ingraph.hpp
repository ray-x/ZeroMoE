#pragma once

// The glue every in-graph MoE layer repeats around build_moe_block: classify
// the placement, resolve the execution policy for the phase, remember which
// lane the block adopted so GPU branches can read its activation snapshot, and
// warn when a requested lane could not be built (R2.1, rollout step 2).
//
// Families keep what is genuinely theirs: where the hot/cold weight tensors
// and route maps come from, and the hooks (miss node, shared expert, extra GPU
// branches, staging). They hand those over as a plan; this header owns the
// order of operations so MiMo, Ling and GLM cannot drift apart.

#include "moe_block.hpp"
#include "phase_expert_cache.hpp"
#include "phase_route_partition.hpp"

#include "logger.hpp"

namespace zeromoe::graph {

// Which class the placement falls in. `cold_only` / `hot_only` are what a
// layer with a single side counts as for this family's tier (MiMo: StorageBacked
// / AllVram; GLM: its DDR-or-storage class for both).
inline ExpertPlacementClass classify_expert_placement(bool has_hot, bool has_cold,
                                                      ExpertPlacementClass cold_only,
                                                      ExpertPlacementClass hot_only) {
  if (has_hot && has_cold) return ExpertPlacementClass::MixedVramDdr;
  return has_cold ? cold_only : hot_only;
}

struct MoEIngraphPlan {
  MoEBlockSpec spec;
  MoEBlockGraph graph;
  MoEBlockLayer layer;
  MoEBranch hot, cold;
  ExpertPlacementClass placement = ExpertPlacementClass::AllVram;
  MoEExecutionCapabilities caps;
  bool lane_wanted = false;   // the block may fork an early-ready lane
  MoEBlockHooks hooks;
  const char *tag = "[MOE]";

  // Set by build_moe_ingraph before extra_branches runs; null without a lane.
  EarlyReadyLane *adopted = nullptr;
};

inline MoEBlockResult build_moe_ingraph(MoEIngraphPlan &plan) {
  // The extension is an additional GPU hot branch. Remove its identities from
  // the cold branch before the lane snapshots its remap/mask. No router values
  // or weights are changed; each authoritative route belongs to one branch.
  auto *phase = plan.graph.decode ? plan.graph.phase_cache : nullptr;
  if (phase && phase->layer(plan.layer.layer)) {
    const auto &hot = phase->layer(plan.layer.layer)->hot;
    if (!plan.cold.present || !plan.graph.engine)
      throw std::logic_error("phase cache needs a cold branch and model geometry");
    std::vector<float> mask(size_t(plan.spec.n_experts));
    std::vector<int32_t> remap(size_t(plan.spec.n_experts));
    ggml_backend_tensor_get(plan.cold.route_mask, mask.data(), 0, mask.size()*sizeof(float));
    ggml_backend_tensor_get(plan.cold.route_map, remap.data(), 0, remap.size()*sizeof(int32_t));
    zeromoe::memory::claim_phase_routes(hot.experts, remap, mask, mmid_inactive_route_id(true));
    ggml_backend_tensor_set(plan.cold.route_mask, mask.data(), 0, mask.size()*sizeof(float));
    ggml_backend_tensor_set(plan.cold.route_map, remap.data(), 0, remap.size()*sizeof(int32_t));
    const auto inner = plan.hooks.extra_branches;
    plan.hooks.extra_branches = [&plan, phase, inner](ggml_tensor *out) {
      if (inner) out = inner(out);
      auto *input = plan.adopted ? plan.adopted->gpu_input : plan.layer.input;
      return phase->build_extra(*plan.graph.engine, plan.spec, plan.graph, plan.layer, input, out);
    };
    plan.placement = ExpertPlacementClass::MixedVramDdr;
  }
  const ExecutionPolicyDecision decision = resolve_execution_policy(
      MoEPlacementSummary{plan.placement}, plan.caps,
      plan.graph.decode ? ModelGraphPhase::Decode : ModelGraphPhase::Prefill);

  if (plan.hooks.adopt_lane) {
    const auto inner = plan.hooks.adopt_lane;
    plan.hooks.adopt_lane = [&plan, inner](std::unique_ptr<EarlyReadyLane> lane) {
      plan.adopted = inner(std::move(lane));
      return plan.adopted;
    };
  }
  const MoEBlockResult r = build_moe_block(plan.spec, plan.graph, plan.layer, plan.hot,
                                           plan.cold, plan.lane_wanted, decision,
                                           plan.hooks);
  if (r.out && plan.graph.decode && plan.cold.present && plan.lane_wanted &&
      plan.caps.early_ready && !r.lane)
    LOG_WARN("%s layer %d: early-ready lane unavailable; cold experts run synchronously\n",
             plan.tag, plan.layer.layer);
  return r;
}

} // namespace zeromoe::graph
