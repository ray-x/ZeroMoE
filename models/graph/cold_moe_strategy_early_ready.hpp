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
// identity (§3.5/§3.8), held stable from begin() until release() by a shared InvocationState.
// state() reports the lane's residency-readiness flag `residency_ready`, NOT its `completed` flag
// — `completed` marks run() finished (compute already done), whereas Ready means "resident and
// safe to compute" (§3.8). A stale invocation (generation advanced) or a wrong-layer invocation is
// Error, never Ready.
//
// state()/finish()/release() are driven per token by the scheduler bridge for the GLM lane
// (`moestream_ggml_sched_early_cpu`: begin before run(), state()/finish()/release() around it).
// This adapter is OBSERVATION-ONLY (`drives_residency()` is false): the lane's own run() starts
// and waits for residency, so finish() is a readiness query and must not be used as a barrier.

#include "cold_moe_strategy.hpp"
#include "early_ready_lane.hpp"
#include "invocation_state.hpp"

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

  // Observation-only: EarlyReadyLane::run() starts and waits for residency internally (the
  // scheduler bridge invokes the lane's own begin_load/finish_load), so this adapter cannot
  // start a load from finish(). Callers must consult state() after their own residency step;
  // finish() is a readiness query here, not a barrier.
  bool drives_residency() const override { return false; }

  bool begin(const ColdMoEInvocation &inv) override {
    if (!lane_)
      return false;
    // Record the invocation identity so state()/finish()/release() can key off it, and a route
    // classified at generation g never binds to a slot reused at g+1 (§3.5/§3.8).
    gate_.begin(inv.layer, inv.generation);
    return true;
  }

  State state(const ColdMoEInvocation &inv) const override {
    if (!lane_)
      return State::Error;
    return gate_.state(inv.layer, inv.generation, residency());
  }

  bool finish(const ColdMoEInvocation &inv) override {
    // The lane's run() already drove residency + compute on the scheduler bridge, so there is
    // nothing to wait on. Fail (rather than report Ready) on a stale or wrong-layer invocation
    // so a stale route never computes (§3.8 rule 2).
    if (!lane_)
      return false;
    return gate_.finish(inv.layer, inv.generation, residency());
  }

  void release(const ColdMoEInvocation &inv) override {
    // The lane owns its buffers; nothing to free. A stale release (generation advanced past this
    // invocation) must be a no-op so it cannot free a live slot (§3.8 rule 3). Idempotent.
    if (!lane_)
      return;
    gate_.release(inv.layer, inv.generation);
  }

 private:
  // Residency readiness is the lane's `residency_ready` flag (set once the owned experts are
  // resident, before the CPU graph runs). The lane has no recoverable load-error state: a failed
  // load aborts today, so it never reports Residency::Failed.
  InvocationState::Residency residency() const {
    return lane_->residency_ready ? InvocationState::Residency::Ready
                                  : InvocationState::Residency::InProgress;
  }

  ::EarlyReadyLane *lane_ = nullptr;
  InvocationState gate_;
};

}  // namespace moestream::graph
