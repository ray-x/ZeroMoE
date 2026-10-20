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

  bool begin(const ColdMoEInvocation & /*inv*/,
             const ColdMoEAdapter &adapter) override {
    if (!lane_)
      return false;
    // The adapter IS the lane's residency callbacks; forward them unchanged.
    lane_->begin_load = adapter.begin_load;
    lane_->finish_load = adapter.finish_load;
    lane_->expert_planes = adapter.expert_planes;
    lane_->before_compute = nullptr;  // the split-load path uses begin_load/finish_load
    return true;
  }

  State state(const ColdMoEInvocation & /*inv*/) const override {
    if (!lane_)
      return State::Error;
    // The lane is Ready once its run() has completed for this token.
    return lane_->completed ? State::Ready : State::Pending;
  }

  bool finish(const ColdMoEInvocation & /*inv*/) override {
    // Residency completion is driven inside lane_->run() via finish_load. Until the wiring step
    // moves that drive here, the strategy's finish() is a no-op for the lane.
    return lane_ != nullptr;
  }

  void release(const ColdMoEInvocation & /*inv*/) override {
    // The lane owns and frees its buffers; nothing extra to release yet.
  }

 private:
  ::EarlyReadyLane *lane_ = nullptr;
};

}  // namespace moestream::graph
