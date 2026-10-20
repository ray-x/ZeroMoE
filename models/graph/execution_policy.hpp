#pragma once

// ExecutionPolicy — the independently selectable optimization policies (§4.7 of
// docs/graph-design-optimization-plan.md).
//
// Five narrow interfaces, all living under the "execution plan" layer and varying independently
// of a model's mathematics:
//
//   CachePolicy          admission / eviction (which experts stay in DDR/VRAM)
//   Predictor            future experts + confidence + target layer
//   PrefetchPolicy       spend bandwidth/memory on predictions
//   PromotionPolicy      move experts into faster storage
//   MoEExecutionStrategy resident/staged/CPU -> combined   (see cold_moe_strategy.hpp)
//
// Narrow = each interface owns exactly one decision. A model selects an ExecutionPolicy, so the
// same model can run a resident-GPU strategy on one machine and CPU/GPU offload on another, or
// swap the predictor, without touching its mathematical definition.
//
// Pure C++ (no ggml / CUDA). The existing mechanisms (ExpertCache / GpuTopology / the
// ridge+pilot tables / async_prefetcher) become implementations/adapters of these interfaces
// during the migration, exactly like EarlyReadyLaneStrategy does for MoEExecutionStrategy.

#include "cold_moe_strategy.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace moestream::graph {

// A (layer, expert) addressing key. Local to this header so the policy interfaces stay free of
// the storage headers that define the full ExpertKey.
struct ExpertRef {
  int layer = -1;
  int expert = -1;
};

// ---------------------------------------------------------------------------
// CachePolicy — admission / eviction.
// ---------------------------------------------------------------------------
// Owns the decision of which experts remain resident in a tier. The store (which it does NOT
// own) locates and loads weights; this interface only says "admit this access, evict these".
class CachePolicy {
 public:
  // For an access to (layer, expert), append the victims to evict (if any) to admit it. The
  // store applies the evictions and performs the load.
  virtual void admit(int layer, int expert, std::vector<ExpertRef> &evict_out) const = 0;
  virtual ~CachePolicy() = default;
};

// ---------------------------------------------------------------------------
// Predictor — future routed experts.
// ---------------------------------------------------------------------------
// Owns the prediction of which experts the NEXT layer will route to, from this layer's hidden
// state. Speculative by definition: real routing decides the required work, and demand loading
// must complete correctly when the prediction misses.
struct Prediction {
  int target_layer = -1;           // the layer being predicted
  std::vector<int32_t> experts;    // predicted expert ids, ranked by score
  std::vector<float> scores;       // confidence, aligned with experts
};

class Predictor {
 public:
  // Predict the next layer's experts from `hidden` (dim floats, this layer's activation).
  virtual bool predict(int layer, const float *hidden, int dim, Prediction &out) const = 0;
  virtual ~Predictor() = default;
};

// ---------------------------------------------------------------------------
// PrefetchPolicy — spend bandwidth/memory on predictions.
// ---------------------------------------------------------------------------
// Owns the decision of what to prefetch given predictions and a budget. It must never make a
// demand miss worse: prefetching is best-effort.
class PrefetchPolicy {
 public:
  // Given the predictions and a budget (max experts), append the (layer, expert) pairs to
  // prefetch, in the order they should be issued.
  virtual void plan(const std::vector<Prediction> &predictions, size_t budget_experts,
                    std::vector<ExpertRef> &out) const = 0;
  virtual ~PrefetchPolicy() = default;
};

// ---------------------------------------------------------------------------
// PromotionPolicy — move experts into faster storage.
// ---------------------------------------------------------------------------
// Owns the decision of which experts move to a faster tier (e.g. RAM -> VRAM). The store/mover
// does the transfer; this interface only says "promote this expert now".
class PromotionPolicy {
 public:
  // Given an expert's recent re-hit count, decide whether to promote it.
  virtual bool should_promote(int layer, int expert, uint32_t hits) const = 0;
  virtual ~PromotionPolicy() = default;
};

// ---------------------------------------------------------------------------
// ExecutionPolicy — the aggregate of all five policies.
// ---------------------------------------------------------------------------
// MoEExecutionStrategy is the fifth policy; it is ColdMoEStrategy today.
using MoEExecutionStrategy = ColdMoEStrategy;

struct ExecutionPolicy {
  std::unique_ptr<CachePolicy> cache;
  std::unique_ptr<Predictor> predictor;
  std::unique_ptr<PrefetchPolicy> prefetch;
  std::unique_ptr<PromotionPolicy> promotion;
  std::unique_ptr<MoEExecutionStrategy> moe;

  // Validate the combination. Some policies depend on capabilities the others supply; an
  // unsupported combination returns false and sets `reason`. Keep the structural rules here and
  // add capability checks as the adapters land (they are what expose the real dependencies).
  bool valid(std::string *reason = nullptr) const {
    auto fail = [&](const char *msg) {
      if (reason)
        *reason = msg;
      return false;
    };
    if (!cache)
      return fail("cache policy is required (the MoE executor reads from it)");
    if (!moe)
      return fail("MoE execution strategy is required");
    // Prefetching is speculative: you cannot prefetch without a predictor to feed it.
    if (prefetch && !predictor)
      return fail("prefetch policy requires a predictor");
    // Promotion moves experts between tiers that the cache policy manages.
    if (promotion && !cache)
      return fail("promotion policy requires a cache policy");
    return true;
  }
};

}  // namespace moestream::graph
