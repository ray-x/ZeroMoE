#pragma once

// CacheBackedStrategy — the cache-backed implementation of ColdMoEStrategy.
//
// The cache-backed path (DeepSeek-V4.1, Ling native) runs each MoE layer through one host
// callback that dispatches to GgmlMoeRunner (transfer + compute) over the native expert caches
// (RAMSafetyManager = DDR tier, GpuTopology = VRAM tier). It is synchronous per layer, unlike
// the full-graph lane's fork + concurrent CPU graph.
//
// This adapter re-expresses that synchronous path as the strategy state machine. It is a
// SPEC at this step: the class holds the existing pieces and documents the grounded mapping,
// but the method bodies are filled in during the wiring step (docs/cold-moe-strategy-contract.md
// §4 step 1). Nothing calls this adapter yet, so behavior is unchanged by construction.
//
// The residency is SPLIT across two phases in two files — this is the single most important
// thing to get right when wiring, because it means the adapter is not a self-contained class:
//
//   Phase A — RAM residency (make cold experts resident in DDR), driven in
//             core/execution_layer.cpp's `ram_needed_k` dispatch loop:
//               RamClaimResult r = ram->claim_or_wait(engine, layer, expert, who, ...);
//               // r.status: Hit (already resident) | Loading (someone else loading) |
//               //           Claimed (this thread claimed; must load then publish)
//               // Loading -> ram->wait_for_ram_slot(engine, layer, expert, ...) to block.
//             State: Hit -> Ready; Loading/Claimed -> Pending; load error -> Error (abort today).
//
//   Phase B — VRAM hold + transfer + compute, driven in
//             core/batched_dispatch_pipeline.cpp's run_ggml_moe_layer:
//               topo->hold_for_compute(layer, e, device_ptr, from_active_layer)   // hold VRAM slot
//               runner->begin_layer(plan)                                        // set up layer
//               runner->run_layer*(x, ids, w, out)                               // H2D/D2D + compute (syncs)
//               topo->release_hold(...) / runner->defer_vram_hold(...)           // release after compute
//
// Mapping (to be wired):
//   begin()   -> the Phase A claim loop over the invocation's owned experts (claim_or_wait +
//                wait_for_ram_slot), then the Phase B VRAM holds (hold_for_compute).
//   state()   -> Ready once every owned expert is a cache hit AND its VRAM slot is held;
//                Pending while any is Loading/Claimed or not yet held.
//   finish()  -> runner->run_layer* (the per-layer sync — it waits for its own H2D/D2D).
//   release() -> topo->release_hold(...) after the last consumer.
//
// Constructor gap: Phase A needs the StreamingEngine (for claim_or_wait's `engine` + the per-layer
// expert geometry) and the per-layer dispatch context (used/ids/weights/normed/output/device
// route), not just (runner, ram, topology). Those are threaded in at the wiring step; until then
// the three backends below are the only handle this adapter needs.

#include "cold_moe_strategy.hpp"

class RAMSafetyManager;
class GpuTopology;
class StreamingEngine;
namespace moe { class GgmlMoeRunner; }

namespace moestream::graph {

class CacheBackedStrategy final : public ColdMoEStrategy {
 public:
  CacheBackedStrategy(moe::GgmlMoeRunner *runner, RAMSafetyManager *ram,
                      GpuTopology *topology)
      : runner_(runner), ram_(ram), topology_(topology) {}

  bool begin(const ColdMoEInvocation & /*inv*/,
             const ColdMoEAdapter & /*adapter*/) override {
    // TODO(wiring): Phase A over inv.routes' owned experts — ram->claim_or_wait(engine, layer,
    // expert, who, ...); on Loading, ram->wait_for_ram_slot(...); on Claimed, load + publish.
    // Then Phase B — topo->hold_for_compute(layer, e, device_ptr, from_active_layer) for each
    // resident expert. Store the (layer, expert, device_ptr, from_active_layer) tuples for
    // release(). The adapter's begin_load/finish_load are the lane's callbacks and are unused
    // here (the cache-backed path claims directly through RAMSafetyManager).
    return runner_ && ram_ && topology_;
  }

  State state(const ColdMoEInvocation & /*inv*/) const override {
    // TODO(wiring): Ready once every owned expert is a cache hit (RamClaimStatus::Hit) and its
    // VRAM slot is held; Pending while any is Loading/Claimed; Error on a failed load.
    // (§3.8: consult before compute; Ready is the only compute-permitted state.)
    return State::Pending;
  }

  bool finish(const ColdMoEInvocation & /*inv*/) override {
    // TODO(wiring): runner->run_layer*(x, ids, w, out) — the per-layer transfer + compute. The
    // runner syncs its own H2D/D2D inside run_layer, so this is the Ready->done barrier.
    return true;
  }

  void release(const ColdMoEInvocation & /*inv*/) override {
    // TODO(wiring): topo->release_hold(layer, expert, device_ptr, from_active_layer) for every
    // tuple held in begin(); a stale release (generation advanced) is a no-op (§3.8 rule 3).
  }

 private:
  moe::GgmlMoeRunner *runner_ = nullptr;
  RAMSafetyManager *ram_ = nullptr;
  GpuTopology *topology_ = nullptr;
};

}  // namespace moestream::graph
