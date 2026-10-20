#pragma once

// CacheBackedStrategy — the cache-backed implementation of ColdMoEStrategy.
//
// The cache-backed path (DeepSeek-V4.1, Ling native) runs each MoE layer through one host
// callback that dispatches to GgmlMoeRunner (transfer + compute) over the native expert caches
// (RAMSafetyManager = DDR tier, GpuTopology = VRAM tier). That path is synchronous per layer,
// unlike the full-graph lane's fork + concurrent CPU graph.
//
// This adapter re-expresses that synchronous path as the strategy state machine. It is a SHAPE
// at this step: the class holds the existing pieces and documents the mapping, but the method
// bodies are filled in during the wiring step (docs/cold-moe-strategy-contract.md §4 step 1),
// because the graph callbacks that drive begin_layer / claim / hold are the missing context.
// Nothing calls this adapter yet, so behavior is unchanged by construction.
//
// Mapping (to be wired):
//   begin()   -> RAMSafetyManager::claim_or_wait (make cold experts resident) + GpuTopology::hold
//   state()   -> Ready once every owned expert is resident (cache hit), Pending while loading
//   finish()  -> the existing per-layer sync (GgmlMoeRunner::run_layer waits for its H2D/D2D)
//   release() -> GpuTopology::release hold after the last consumer

#include "cold_moe_strategy.hpp"

class RAMSafetyManager;
class GpuTopology;
namespace moe { class GgmlMoeRunner; }

namespace moestream::graph {

class CacheBackedStrategy final : public ColdMoEStrategy {
 public:
  CacheBackedStrategy(moe::GgmlMoeRunner *runner, RAMSafetyManager *ram,
                      GpuTopology *topology)
      : runner_(runner), ram_(ram), topology_(topology) {}

  bool begin(const ColdMoEInvocation & /*inv*/,
             const ColdMoEAdapter & /*adapter*/) override {
    // TODO(wiring): drive RAMSafetyManager::claim_or_wait + GpuTopology::hold from the
    // invocation's owned experts; store the adapter's begin_load/finish_load for finish().
    return runner_ && ram_ && topology_;
  }

  State state(const ColdMoEInvocation & /*inv*/) const override {
    // TODO(wiring): Ready once every owned expert is resident, Pending while any is loading.
    return State::Pending;
  }

  bool finish(const ColdMoEInvocation & /*inv*/) override {
    // TODO(wiring): block on residency + the layer's transfer/compute sync.
    return true;
  }

  void release(const ColdMoEInvocation & /*inv*/) override {
    // TODO(wiring): GpuTopology::release after the last consumer.
  }

 private:
  moe::GgmlMoeRunner *runner_ = nullptr;
  RAMSafetyManager *ram_ = nullptr;
  GpuTopology *topology_ = nullptr;
};

}  // namespace moestream::graph
