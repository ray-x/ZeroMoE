#pragma once

// InvocationState — the pure, backend-free portion of a ColdMoEStrategy's per-invocation
// lifecycle (§3.8). It holds the active invocation identity (layer, generation) and derives
// Pending/Ready/Error from that identity plus the caller-supplied residency condition. Extracted
// from EarlyReadyLaneStrategy so the lifecycle can be unit-tested without ggml/CUDA, and shared by
// every strategy implementation so both paths obey one tested state machine.
//
// The single invariant it enforces: a route classified at generation g must never bind to a slot
// reused at g+1. Anything that names a different layer or generation is Error, never Ready.

#include "cold_moe_strategy.hpp"  // ColdMoEStrategy::State

#include <cstdint>

namespace zeromoe::graph {

class InvocationState {
 public:
  using State = ColdMoEStrategy::State;

  // The residency condition a strategy reports for the invocation, before identity is considered.
  // InProgress -> State::Pending; Ready -> State::Ready; Failed -> State::Error (a load error).
  // Ready means "resident and safe to compute", NOT "compute finished" — `completed` is a separate
  // lifecycle point and is deliberately absent from this type (§3.8).
  enum class Residency : uint8_t { InProgress, Ready, Failed };

  static constexpr uint64_t kNoGeneration = ~0ull;

  // Record the active invocation. Overwrites any prior active identity.
  void begin(int layer, uint64_t generation) {
    layer_ = layer;
    generation_ = generation;
  }

  // True when no invocation is active (nothing recorded, or it was released).
  bool inactive() const { return generation_ == kNoGeneration; }

  // Does (layer, generation) name the currently active invocation?
  bool matches(int layer, uint64_t generation) const {
    return generation_ != kNoGeneration && layer == layer_ && generation == generation_;
  }

  // The state machine: Error for a stale/wrong invocation or a failed load; otherwise Ready/Pending
  // from the residency condition. Ready is the only state in which compute may proceed (§3.8 rule 1).
  State state(int layer, uint64_t generation, Residency residency) const {
    if (!matches(layer, generation) || residency == Residency::Failed)
      return State::Error;
    return residency == Residency::Ready ? State::Ready : State::Pending;
  }

  // finish()'s shared check: false on a stale/wrong invocation or a failed load, otherwise true
  // exactly when residency is Ready. (Blocking on residency is the strategy's own concern.)
  bool finish(int layer, uint64_t generation, Residency residency) const {
    return matches(layer, generation) && residency == Residency::Ready;
  }

  // Idempotent release; a no-op once the generation has advanced, so a stale release after a slot
  // reuse cannot free a live buffer (§3.8 rule 3).
  void release(int layer, uint64_t generation) {
    if (matches(layer, generation))
      generation_ = kNoGeneration;
  }

  int layer() const { return layer_; }
  uint64_t generation() const { return generation_; }

 private:
  int layer_ = -1;
  uint64_t generation_ = kNoGeneration;
};

}  // namespace zeromoe::graph
