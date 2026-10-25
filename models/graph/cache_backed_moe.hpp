#pragma once

#include "cold_moe_strategy_cache_backed.hpp"
#include "execution_policy.hpp"
#include <stdexcept>

namespace zeromoe::graph {

// A graph callback adapter to the live native MoE backend. Residency, cache
// holds, transfers and expert kernels stay in the existing shared pipeline.
struct CacheBackedMoeRuntime {
  ExecutionPolicy policy;
  StreamingEngine *engine = nullptr;
  uint64_t generation = 0;

  CacheBackedMoeRuntime() {
    // RAMSafetyManager/GpuTopology already own victim selection. This adapter
    // makes that existing implementation explicit in ExecutionPolicy without
    // introducing a second eviction decision.
    policy.cache = std::make_shared<DelegatingCachePolicy>(
        [](int, int, std::vector<ExpertRef> &) {});
    policy.moe = std::make_shared<CacheBackedStrategy>();
  }

  void bind(StreamingEngine &e, RAMSafetyManager &r, LockFreePrefetcher &p,
             LayerScratch &s) {
    engine = &e;
    std::string reason;
    if (!policy.moe || !policy.moe->bind({&e, &r, &p, &s}, &reason))
      throw std::runtime_error(reason);
  }

  void run(int layer, const float *input, int tokens, float *output,
            const int *ids) {
    if (!engine || !input || !output || !ids)
      throw std::runtime_error("cache-backed graph runtime is not bound");
    MoEExecutionRequest request;
    request.generation = ++generation;
    request.phase = tokens == 1 ? ModelGraphPhase::Decode
                                : ModelGraphPhase::Prefill;
    request.layer = layer;
    request.n_tokens = tokens;
    request.token_ids = ids;
    request.input = input;
    request.output = output;
    std::string reason;
    if (!policy.moe->execute(
            moe_block_spec(engine->model_file.architecture_config), request,
            &reason))
      throw std::runtime_error(reason);
  }
};
} // namespace zeromoe::graph
