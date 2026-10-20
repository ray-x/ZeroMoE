#pragma once

#include "gguf_loader.hpp"

#include <cstddef>
#include <cstdint>

class GpuTopology;
class RAMSafetyManager;
class StreamingEngine;
struct ExpertLocation;
struct GpuExpertScratch;
struct MoEExpertBlock;

struct ExpertGpuContext {
  GpuTopology *topo = nullptr;
  const ExpertLocation *loc = nullptr;
  int layer_idx = -1;
  int expert_id = -1;
  const uint8_t *slab_buffer = nullptr;
  size_t slab_capacity = 0;
  MoEExpertBlock *exp = nullptr;
  bool force_cpu_offload = false;
};

struct PendingGpuExpert {
  int k = -1;
  MoEExpertBlock *exp = nullptr;
  const ExpertLocation *loc = nullptr;
  bool needs_release = false;
  bool loc_from_active_layer = false;
  int expert_id = -1;
  bool accumulated_on_device = false;
};

const ExpertLocation *expert_ffn_resolve_gpu_loc(
    const ExpertGpuContext *context, int S, const GGUFTensor &gate_meta,
    const GGUFTensor &up_meta, const GGUFTensor &down_meta,
    bool *needs_release, bool *location_from_active_layer,
    RAMSafetyManager &ram_mgr, StreamingEngine &engine);

bool expert_ffn_gpu_issue(GpuExpertScratch &scratch,
                          const ExpertLocation *location, const float *input,
                          const GGUFTensor &gate_meta,
                          const GGUFTensor &up_meta, int S, int D,
                          int moe_inter, GpuTopology *topology,
                          const float *shared_device_input = nullptr);

void expert_ffn_gpu_finish_stage1(
    GpuExpertScratch &scratch, const ExpertLocation *location,
    float *gate_buffer, float *up_buffer, const GGUFTensor &gate_meta,
    const GGUFTensor &down_meta, int layer_idx, int expert_id, int S, int D,
    int moe_inter, float swiglu_clamp, GpuTopology *topology);

void expert_ffn_gpu_finish_stage2(
    const ExpertLocation *location, float *output, int layer_idx, int expert_id,
    int S, int D, GpuTopology *topology, bool needs_release,
    bool location_from_active_layer, GpuExpertScratch &scratch);
