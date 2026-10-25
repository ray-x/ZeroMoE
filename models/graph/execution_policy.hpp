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

#include "moe_execution_strategy.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace zeromoe::graph {

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

class DelegatingCachePolicy final : public CachePolicy {
 public:
  using AdmitFn = std::function<void(int, int, std::vector<ExpertRef> &)>;
  explicit DelegatingCachePolicy(AdmitFn fn) : fn_(std::move(fn)) {}
  void admit(int layer, int expert,
             std::vector<ExpertRef> &evict_out) const override {
    if (fn_) fn_(layer, expert, evict_out);
  }
 private:
  AdmitFn fn_;
};

// ---------------------------------------------------------------------------
// Predictor — future routed experts (speculation with reconciliation).
// ---------------------------------------------------------------------------
// Owns the prediction of which experts the NEXT layer will route to, from this layer's hidden
// state. This is the expert-level counterpart of MTP token drafting: both speculate ahead of the
// authoritative answer, and both define exactly what happens when the prediction is wrong.
//
//   Predicts:   the experts likely to be needed, ranked by confidence.
//   Verifies:   the layer's real router — its top-k output is authoritative and always wins.
//   Reconciles: demand loading supplies whatever the router actually selected; predicted-but-unused
//               prefetch is wasted work reclaimed by CachePolicy's eviction (bounded, not an error).
//
// A predictor must never change the model's mathematical definition: a wrong prediction costs
// wasted prefetch, never a wrong result.
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

class DelegatingPredictor final : public Predictor {
 public:
  using PredictFn = std::function<bool(int, const float *, int, Prediction &)>;
  explicit DelegatingPredictor(PredictFn fn) : fn_(std::move(fn)) {}
  bool predict(int layer, const float *hidden, int dim,
               Prediction &out) const override {
    return fn_ && fn_(layer, hidden, dim, out);
  }
 private:
  PredictFn fn_;
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

class RankedPrefetchPolicy final : public PrefetchPolicy {
 public:
  void plan(const std::vector<Prediction> &predictions,
            size_t budget_experts,
            std::vector<ExpertRef> &out) const override {
    for (const Prediction &p : predictions) {
      for (int32_t expert : p.experts) {
        if (out.size() >= budget_experts) return;
        if (expert >= 0) out.push_back({p.target_layer, expert});
      }
    }
  }
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

class HitThresholdPromotionPolicy final : public PromotionPolicy {
 public:
  explicit HitThresholdPromotionPolicy(uint32_t threshold)
      : threshold_(threshold) {}
  bool should_promote(int, int, uint32_t hits) const override {
    return threshold_ > 0 && hits >= threshold_;
  }
 private:
  uint32_t threshold_ = 0;
};

// ---------------------------------------------------------------------------
// ExecutionPolicy — the aggregate of all five policies.
// ---------------------------------------------------------------------------
struct ExecutionPolicy {
  std::shared_ptr<CachePolicy> cache;
  std::shared_ptr<Predictor> predictor;
  std::shared_ptr<PrefetchPolicy> prefetch;
  std::shared_ptr<PromotionPolicy> promotion;
  std::shared_ptr<MoEExecutionStrategy> moe;

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

struct ExecutionPolicyDecision {
  MoEExecutionKind moe = MoEExecutionKind::CacheBacked;
  const char *reason = "safe cache-backed fallback";
};

inline ExecutionPolicyDecision resolve_execution_policy(
    const MoEPlacementSummary &placement,
    const MoEExecutionCapabilities &caps,
    ModelGraphPhase phase) {
  (void)phase; // phase-specific implementations can refine this decision.
  if (placement.placement == ExpertPlacementClass::AllVram &&
      caps.resident_gpu)
    return {MoEExecutionKind::ResidentGpu,
            "all expert identities fit in VRAM"};
  if (placement.placement == ExpertPlacementClass::MixedVramDdr &&
      caps.early_ready)
    return {MoEExecutionKind::EarlyReadyOffload,
            "mixed VRAM+DDR placement supports concurrent branches"};
  if (placement.placement == ExpertPlacementClass::StorageBacked &&
      caps.early_ready)
    return {MoEExecutionKind::EarlyReadyOffload,
            "storage misses overlap CPU work and asynchronous demand loads"};
  if (placement.placement == ExpertPlacementClass::DdrOnly &&
      caps.early_ready && caps.prefer_overlap)
    return {MoEExecutionKind::EarlyReadyOffload,
            "overlap lane forced for DDR-resident experts"};
  if (placement.placement == ExpertPlacementClass::DdrOnly &&
      caps.cpu_resident)
    return {MoEExecutionKind::CpuResident,
            "DDR-resident CPU execution avoids lane overhead"};
  return {MoEExecutionKind::CacheBacked,
          caps.cache_backed ? "cache-backed fallback selected"
                            : "no compatible optimized strategy"};
}

}  // namespace zeromoe::graph
