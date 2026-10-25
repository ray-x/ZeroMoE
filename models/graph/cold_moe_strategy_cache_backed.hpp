#pragma once

// Synchronous MoEExecutionStrategy used through CacheBackedMoeRuntime by
// MiMo's fallback, Ling-native and DeepSeek-V4.1's prefill/decode callbacks.
// execute() calls the existing shared MoE pipeline once; that pipeline owns
// RAM claims, VRAM compute holds, transfers, kernels and release. This adapter
// does not duplicate eviction or implement ColdMoEStrategy's asynchronous
// begin/state/finish/release lifecycle. Extracting that lifecycle remains a
// separate migration; the adapter is live, not an unwired specification.

#include "moe_execution_strategy.hpp"
#include "moe_pipeline.hpp"
#include "ram_manager.hpp"
#include "scratch_buffers.hpp"

#include <cmath>

namespace zeromoe::graph {

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

}  // namespace zeromoe::graph
