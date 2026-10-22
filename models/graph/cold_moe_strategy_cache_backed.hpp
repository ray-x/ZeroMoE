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
//                Pending while any is Loading/Claimed or not yet held; Error on a failed load.
//   finish()  -> runner->run_layer* (the per-layer sync — it waits for its own H2D/D2D).
//   release() -> topo->release_hold(...) after the last consumer.
//
// Constructor gap: Phase A needs the StreamingEngine (for claim_or_wait's `engine` + the per-layer
// expert geometry) and the per-layer dispatch context (used/ids/weights/normed/output/device
// route), not just (runner, ram, topology). Those are threaded in at the wiring step; until then
// the three backends below are the only handle this adapter needs.

#include "moe_execution_strategy.hpp"
#include "moe_pipeline.hpp"
#include "ram_manager.hpp"
#include "scratch_buffers.hpp"

#include <cmath>

namespace moestream::graph {

class CacheBackedStrategy final : public MoEExecutionStrategy {
 public:
  MoEExecutionKind kind() const override {
    return MoEExecutionKind::CacheBacked;
  }
  const char *name() const override { return "cache-backed"; }

  bool bind(const MoERuntimeResources &resources,
            std::string *reason = nullptr) override {
    resources_ = resources;
    if (!resources_.engine || !resources_.ram || !resources_.prefetch ||
        !resources_.scratch) {
      if (reason) *reason = "cache-backed runtime resources are incomplete";
      return false;
    }
    return true;
  }

  bool execute(const MoEBlockSpec &spec, const MoEExecutionRequest &request,
               std::string *reason = nullptr) override {
    if (!resources_.engine || !resources_.ram || !resources_.prefetch ||
        !resources_.scratch || !request.input || !request.output ||
        !request.token_ids || request.n_tokens <= 0 || request.layer < 0) {
      if (reason) *reason = "invalid cache-backed MoE request";
      return false;
    }
    const auto &c = resources_.engine->model_file.architecture_config;
    if (spec.d_model != c.hidden || spec.d_ff != c.moe_inter ||
        spec.n_experts != c.n_experts || spec.topk != c.topk) {
      if (reason) *reason = "MoE block specification does not match model";
      return false;
    }
    if (resources_.ram->vram_cache)
      resources_.ram->vram_cache->prefill_current_layer = request.layer;
    moe_forward_gguf(*resources_.engine, *resources_.ram,
                     *resources_.prefetch, request.layer, request.input,
                     request.n_tokens, request.output, *resources_.scratch,
                     request.token_ids);
    const size_t n = (size_t)request.n_tokens * (size_t)spec.d_model;
    for (size_t i = 0; i < n; ++i) {
      if (!std::isfinite(request.output[i])) {
        if (reason) *reason = "cache-backed MoE produced non-finite output";
        return false;
      }
    }
    return true;
  }

 private:
  MoERuntimeResources resources_;
};

}  // namespace moestream::graph
