#pragma once

// EarlyReadyLaneStrategy — the full-graph implementation of ColdMoEStrategy.
//
// A behavior-preserving adapter over the existing EarlyReadyLane: the lane's
// std::function members (begin_load / finish_load / expert_planes / before_compute) are exactly
// the ColdMoEAdapter surface, so begin() forwards them unchanged. The lane keeps running its
// own enqueue/run() orchestration (the patched scheduler bridge); this adapter only re-expresses
// its residency state as the strategy state machine.
//
// NOT wired into the graph builder yet — see docs/cold-moe-strategy-contract.md §4 step 1.
// Because nothing calls this adapter, behavior is unchanged by construction.

#include "cold_moe_strategy.hpp"
#include "early_ready_lane.hpp"

namespace moestream::graph {

class EarlyReadyLaneStrategy final : public ColdMoEStrategy {
 public:
  explicit EarlyReadyLaneStrategy(::EarlyReadyLane *lane) : lane_(lane) {}

  bool begin(const ColdMoEInvocation &inv,
             const ColdMoEAdapter &adapter) override {
    if (!lane_)
      return false;
    // The adapter IS the lane's residency callbacks; forward them unchanged.
    lane_->begin_load = adapter.begin_load;
    lane_->finish_load = adapter.finish_load;
    lane_->before_compute = adapter.before_compute;
    lane_->expert_planes = adapter.expert_planes;
    // Record the invocation so state()/finish()/release() can key off its generation
    // (§3.5/§3.8): a route classified at generation g must never bind to a slot reused at g+1.
    layer_ = inv.layer;
    generation_ = inv.generation;
    return true;
  }

  State state(const ColdMoEInvocation &inv) const override {
    if (!lane_)
      return State::Error;
    // Generation cross-check: the invocation must match the lane's current epoch. A stale
    // invocation (classified before the lane advanced) is Error, never Ready (§3.8).
    if (inv.generation != lane_->calls)
      return State::Error;
    return lane_->completed ? State::Ready : State::Pending;
  }

  bool finish(const ColdMoEInvocation &inv) override {
    // The lane's run() already drove residency to completion on the scheduler bridge; there is
    // nothing to wait on here. Fail (rather than report Ready) on a stale invocation so a stale
    // route never computes (§3.8 rule 2).
    if (!lane_ || inv.generation != lane_->calls)
      return false;
    return lane_->completed;
  }

  void release(const ColdMoEInvocation &inv) override {
    // The lane owns its buffers; nothing to free. A stale release (generation advanced past this
    // invocation) must be a no-op so it cannot free a live slot (§3.8 rule 3).
    if (lane_ && inv.generation == generation_)
      generation_ = kNoGeneration;
  }

 private:
  static constexpr uint64_t kNoGeneration = ~0ull;
  ::EarlyReadyLane *lane_ = nullptr;
  int layer_ = -1;
  uint64_t generation_ = kNoGeneration;
};

}  // namespace moestream::graph
