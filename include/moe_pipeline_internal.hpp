#pragma once

#include "engine.hpp"
#include "ggml_moe_backend.hpp"
#include "scratch_buffers.hpp"

#include <cstdint>
#include <vector>

class GgmlRouterRunner;
class LockFreePrefetcher;
class RAMSafetyManager;

struct QwenRamCpuMoeLayer {
  GgmlQwenCpuExpertRef meta;
  const void *gate_plane = nullptr;
  const void *up_plane = nullptr;
  const void *down_plane = nullptr;
  size_t expert_stride = 0;
};

struct MoePipelineContext {
  StreamingEngine &engine;
  RAMSafetyManager &ram_mgr;
  LockFreePrefetcher &async_io_queue;
  int layer_idx;
  const float *hidden_states;
  int S;
  float *output;
  LayerScratch &scratch;
  const int *token_ids;
  ArchitectureConfig &c;
  int D;
  int topk;
  int n_experts;
  int moe_inter;
  float clamp_exp;
  float clamp_shexp;

  bool gpu_router_handled = false;
  bool gpu_route_device = false;
  const void *gpu_route_ids_device = nullptr;
  const void *gpu_route_weights_device = nullptr;
  GgmlRouterRunner *device_router = nullptr;
  std::vector<int32_t> gpu_route_ids_ks;
  std::vector<float> gpu_route_weights_ks;
  bool batched_handled = false;
};

namespace moe_pipeline_internal {
void add_decode_qwen_cpu_graph_ms(double elapsed_ms);
void add_decode_qwen_shared_submit_ms(double elapsed_ms);
void add_decode_qwen_cpu_offload_wait_ms(double elapsed_ms);
void add_decode_qwen_cpu_offload_total_ms(double elapsed_ms);
void add_decode_qwen_cpu_offload_fence_wait_ms(double elapsed_ms);
void add_decode_qwen_cpu_offload_h2d_ms(double elapsed_ms);
void add_decode_qwen_cpu_offload_ring_reuse_wait_ms(double elapsed_ms);
void add_decode_qwen_cpu_offload_cold_slots(uint64_t count);
void add_decode_qwen_cpu_offload_worker_call();
void add_decode_qwen_cpu_offload_worker_success();
void add_decode_qwen_cpu_offload_worker_failure();
void add_decode_qwen_cpu_offload_fenced_call();
void add_decode_qwen_cpu_offload_fence_failure();
void qwen4exp_record_pilot_fetch_start(int layer_idx);
} // namespace moe_pipeline_internal
