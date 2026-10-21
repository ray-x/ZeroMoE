#pragma once

// EarlyReadyLaneStrategy — the full-graph implementation of ColdMoEStrategy.
//
// A behavior-preserving adapter over the existing EarlyReadyLane. install() forwards the
// family's addressing callbacks (ColdMoEAdapter) to the lane's std::function members
// (begin_load / finish_load / expert_planes / before_compute), which are exactly the
// ColdMoEAdapter surface. The lane keeps running its own enqueue/run() orchestration (the
// patched scheduler bridge); this adapter only re-expresses its residency state as the strategy
// state machine.
//
// Per-invocation begin()/state()/finish()/release() key off the invocation's (layer, generation)
// identity (§3.5/§3.8), held stable from begin() until release(). state() reports the lane's
// residency-readiness flag `residency_ready`, NOT its `completed` flag — `completed` marks run()
// finished (compute already done), whereas Ready means "resident and safe to compute" (§3.8). A
// stale invocation (generation advanced) or a wrong-layer invocation is Error, never Ready.
//
// state()/finish()/release() are not yet consulted per token by the decode path (the lane still
// drives residency through run()); this adapter makes the lifecycle correct as a representation.

#include "cold_moe_strategy.hpp"
#include "early_ready_lane.hpp"

namespace moestream::graph {

class EarlyReadyLaneStrategy final : public ColdMoEStrategy {
 public:
  explicit EarlyReadyLaneStrategy(::EarlyReadyLane *lane) : lane_(lane) {}

  bool install(const ColdMoEAdapter &adapter) override {
    if (!lane_)
      return false;
    // The adapter IS the lane's residency callbacks; forward them unchanged.
    lane_->begin_load = adapter.begin_load;
    lane_->finish_load = adapter.finish_load;
    lane_->before_compute = adapter.before_compute;
    lane_->expert_planes = adapter.expert_planes;
    return true;
  }

  bool begin(const ColdMoEInvocation &inv) override {
    if (!lane_)
      return false;
    // Record the invocation identity so state()/finish()/release() can key off it, and a route
    // classified at generation g never binds to a slot reused at g+1 (§3.5/§3.8).
    layer_ = inv.layer;
    generation_ = inv.generation;
    return true;
  }

  State state(const ColdMoEInvocation &inv) const override {
    if (!lane_)
      return State::Error;
    // Identity cross-check: the invocation must be the ACTIVE one (same layer, same generation).
    // A stale invocation (generation advanced past it) or a wrong-layer invocation is Error,
    // never Ready (§3.8).
    if (inv.layer != layer_ || inv.generation != generation_)
      return State::Error;
    // Residency readiness is the lane's `residency_ready` flag, set once the owned experts are
    // resident and before the CPU graph runs. It is deliberately NOT `completed` (compute done).
    return lane_->residency_ready ? State::Ready : State::Pending;
  }

  bool finish(const ColdMoEInvocation &inv) override {
    // The lane's run() already drove residency + compute on the scheduler bridge, so there is
    // nothing to wait on. Fail (rather than report Ready) on a stale or wrong-layer invocation
    // so a stale route never computes (§3.8 rule 2).
    if (!lane_ || inv.layer != layer_ || inv.generation != generation_)
      return false;
    return lane_->residency_ready;
  }

  void release(const ColdMoEInvocation &inv) override {
    // The lane owns its buffers; nothing to free. A stale release (generation advanced past this
    // invocation) must be a no-op so it cannot free a live slot (§3.8 rule 3). Idempotent.
    if (lane_ && inv.layer == layer_ && inv.generation == generation_)
      generation_ = kNoGeneration;
  }

 private:
  static constexpr uint64_t kNoGeneration = ~0ull;
  ::EarlyReadyLane *lane_ = nullptr;
  int layer_ = -1;
  uint64_t generation_ = kNoGeneration;
};

}  // namespace moestream::graph
