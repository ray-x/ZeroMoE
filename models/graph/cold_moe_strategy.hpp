#pragma once

// ColdMoEStrategy — the shared cold-MoE execution contract.
//
// This is the interface from docs/cold-moe-strategy-contract.md. It owns the ORCHESTRATION
// (state machine, residency, transfer, compute) for a layer whose routed experts may be cold.
// A model family supplies only addressing through ColdMoEAdapter; it never re-implements
// caching, staging, pilot, or the CPU lane.
//
// The interface is deliberately a superset of the callbacks EarlyReadyLane already exposes
// (begin_load / finish_load / expert_planes), so the full-graph lane becomes one implementation
// and the cache-backed runner (backends/ggml) becomes another.
//
// Pure C++ (no ggml / CUDA) so it can be reasoned about and unit-tested in isolation.

#include "moe_partial_route.hpp"  // RouteClassification: route-slot ownership + exact-once weights

#include <cstdint>
#include <functional>
#include <vector>

namespace moestream::graph {

// One token's routing for one MoE layer. Immutable for the lifetime of the invocation;
// produced once, consumed by every branch, released after the last consumer.
struct ColdMoEInvocation {
  uint64_t generation = 0;   // generation identity (§3.5): a route must never bind to a slot
                             // reused for a different expert between classify and compute.
  int layer = -1;
  moe_partial::RouteClassification routes;  // route-slot ownership + exact-once weights (§3.1/§3.2)
};

// The adapter a family supplies. Addressing only — owns no cache/staging/lane machinery.
//
// Exactly the union of EarlyReadyLane's residency callbacks, so the lane is one implementation:
// the split-load path supplies begin_load/finish_load (background residency, returns the still
// pending experts); the synchronous path supplies before_compute (block until resident); the
// GPU-assist path supplies expert_planes. A family supplies whichever it supports.
struct ColdMoEAdapter {
  // Plane pointers (gate/up/down) for `expert`; false when this family cannot address it.
  std::function<bool(int expert, const void *planes[3])> expert_planes;
  // Begin making `experts` resident (background); returns experts not yet ready in `pending`.
  std::function<bool(const int32_t *experts, int n, std::vector<int32_t> &pending)> begin_load;
  // Block until every expert handed to begin_load is ready.
  std::function<bool()> finish_load;
  // Block until `experts` are resident (synchronous, non-split path). Mutually exclusive with
  // begin_load/finish_load.
  std::function<bool(const int32_t *experts, int n)> before_compute;
};

// The shared backend. One instance per family (or shared across compatible families); it owns
// the state machine and the residency/transfer/compute resources.
class ColdMoEStrategy {
 public:
  enum class State : uint8_t { Pending, Ready, Error };

  // Classify + begin residency for the invocation. False only on an unrecoverable error.
  virtual bool begin(const ColdMoEInvocation &inv, const ColdMoEAdapter &adapter) = 0;

  // Pending until every owned route's expert is fully resident and valid; Ready when
  // computable; Error when unrecoverable (§3.3).
  virtual State state(const ColdMoEInvocation &inv) const = 0;

  // Block until Ready (or return false on Error). Re-entrant on the decode thread (§3.3).
  virtual bool finish(const ColdMoEInvocation &inv) = 0;

  // Free the invocation's held buffers after the last consumer (§3.4). The strategy's own
  // CUDA-backed state is released in shutdown(), after the consumers, per §3.7.
  virtual void release(const ColdMoEInvocation &inv) = 0;

  virtual ~ColdMoEStrategy() = default;
};

}  // namespace moestream::graph
