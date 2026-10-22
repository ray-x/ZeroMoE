#pragma once

// Runtime boundary between model MoE mathematics and expert execution.
// A family supplies MoEBlockSpec plus one MoEExecutionRequest per layer. The
// strategy owns residency, transfers, CPU/GPU partitioning, synchronization
// and final combination. This is deliberately independent of attention.

#include "family_ops.hpp"
#include "cold_moe_strategy.hpp"

#include <cstdint>
#include <string>

struct StreamingEngine;
class RAMSafetyManager;
class LockFreePrefetcher;
struct LayerScratch;

namespace moestream::graph {

enum class ExpertPlacementClass : uint8_t {
  AllVram,
  MixedVramDdr,
  DdrOnly,
  StorageBacked,
};

enum class MoEExecutionKind : uint8_t {
  ResidentGpu,
  EarlyReadyOffload,
  CacheBacked,
  CpuResident,
};

struct MoEBlockSpec {
  int d_model = 0;
  int d_ff = 0;
  int n_experts = 0;
  int topk = 0;
  GatingFunc gating = GatingFunc::Softmax;
  bool normalize_selected_weights = false;
  float weights_scale = 0.0f;
  float swiglu_clamp = 0.0f;
  int n_shared_experts = 0;
};

struct MoEPlacementSummary {
  ExpertPlacementClass placement = ExpertPlacementClass::StorageBacked;
  uint64_t model_slots = 0;
  uint64_t ddr_slots = 0;
  uint64_t vram_slots = 0;
  uint64_t model_bytes = 0;
  uint64_t ddr_bytes = 0;
  uint64_t vram_bytes = 0;
  bool complete_if_disjoint = false;
};

struct MoEExecutionCapabilities {
  bool resident_gpu = true;
  bool cache_backed = true;
  bool cpu_resident = true;
  bool early_ready = false;
  bool gpu_assist = false;
  // Operator override: run the overlap lane even for DDR-only layers.
  bool prefer_overlap = false;
};

struct MoERuntimeResources {
  StreamingEngine *engine = nullptr;
  RAMSafetyManager *ram = nullptr;
  LockFreePrefetcher *prefetch = nullptr;
  LayerScratch *scratch = nullptr;
};

struct MoEExecutionRequest {
  uint64_t generation = 0;
  ModelGraphPhase phase = ModelGraphPhase::Prefill;
  int layer = -1;
  int n_tokens = 0;
  const int *token_ids = nullptr;
  const float *input = nullptr;
  float *output = nullptr;

  // Optional authoritative router result. A strategy may compute routing from
  // input when these are null; when supplied it must preserve route slots and
  // apply each weight exactly once.
  const int32_t *route_ids = nullptr;
  const float *route_weights = nullptr;
  int route_count = 0;
};

class MoEExecutionStrategy {
 public:
  virtual ~MoEExecutionStrategy() = default;
  virtual MoEExecutionKind kind() const = 0;
  virtual const char *name() const = 0;
  virtual bool bind(const MoERuntimeResources &resources,
                    std::string *reason = nullptr) = 0;
  virtual bool execute(const MoEBlockSpec &spec,
                       const MoEExecutionRequest &request,
                       std::string *reason = nullptr) = 0;
  virtual ColdMoEStrategy *residency_strategy() { return nullptr; }
  virtual bool graph_integrated() const { return false; }
};

inline MoEBlockSpec moe_block_spec(const ArchitectureConfig &c) {
  return {.d_model = c.hidden,
          .d_ff = c.moe_inter,
          .n_experts = c.n_experts,
          .topk = c.topk,
          .gating = c.gating_func,
          .normalize_selected_weights = c.expert_weights_norm,
          .weights_scale = c.expert_weights_scale,
          .swiglu_clamp = 0.0f,
          .n_shared_experts = c.n_expert_shared};
}

} // namespace moestream::graph
