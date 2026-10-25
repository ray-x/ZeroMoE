#pragma once

// Family-independent graph construction for an EarlyReadyLane. Placement
// code supplies hot/cold maps and expert tensors; this builder owns the fork,
// host route-table binding, CPU marker and uploaded cold-output dependency.

#include "early_ready_lane.hpp"

#include <memory>
#include <string>
#include <vector>

namespace zeromoe::graph {

struct EarlyReadyGraphBuildSpec {
  ggml_backend_sched_t sched = nullptr;
  ggml_backend_t cuda = nullptr;
  ggml_backend_t cpu = nullptr;
  ggml_context *ctx = nullptr;
  ggml_cgraph *graph = nullptr;
  int device = -1;
  int layer = -1;
  int d_model = 0;
  int topk = 0;
  bool decode = false;
  int n_tokens = 0;
  float swiglu_clamp = 0.0f;
  bool swiglu_clamp_fused = false;
  const char *name_prefix = "moe";
  ggml_tensor *input = nullptr;
  ggml_tensor *route_ids = nullptr;
  ggml_tensor *route_weights = nullptr;
  ggml_tensor *cold_remap = nullptr;
  ggml_tensor *cold_mask = nullptr;
  moe_routed::ExpertWeights cold_weights;
};

inline std::unique_ptr<EarlyReadyLane> build_early_ready_graph_lane(
    const EarlyReadyGraphBuildSpec &s) {
  if (!s.decode || s.n_tokens != 1 || !s.sched || !s.cuda || !s.cpu ||
      !s.ctx || !s.graph || s.device < 0 || s.layer < 0 || s.d_model <= 0 ||
      s.topk <= 0 || !s.input || !s.route_ids || !s.route_weights ||
      !s.cold_remap || !s.cold_mask || !s.cold_weights.valid() ||
      s.input->type != GGML_TYPE_F32)
    return nullptr;
  if (!ggml_backend_sched_early_cpu_supported ||
      ggml_backend_sched_early_cpu_supported() != 1) {
    LOG_WARN("[MOE-EARLY-READY] layer %d: linked ggml has no early-CPU "
             "scheduler hook\n", s.layer);
    return nullptr;
  }

  auto lane = std::make_unique<EarlyReadyLane>();
  lane->swiglu_clamp = s.swiglu_clamp;
  lane->swiglu_clamp_fused = s.swiglu_clamp_fused;
  const auto host_table = [&](ggml_tensor *src) -> ggml_tensor * {
    ggml_tensor *t = ggml_new_tensor_3d(s.ctx, src->type, src->ne[0],
                                         src->ne[1], src->ne[2]);
    if (!t) return nullptr;
    ggml_backend_buffer_t buf =
        ggml_backend_alloc_buffer(s.cpu, ggml_nbytes(t));
    if (!buf) return nullptr;
    lane->extra_buffers.push_back(buf);
    if (ggml_backend_tensor_alloc(buf, t, ggml_backend_buffer_get_base(buf)) !=
        GGML_STATUS_SUCCESS)
      return nullptr;
    std::vector<uint8_t> tmp(ggml_nbytes(t));
    ggml_backend_tensor_get(src, tmp.data(), 0, tmp.size());
    ggml_backend_tensor_set(t, tmp.data(), 0, tmp.size());
    return t;
  };
  ggml_tensor *remap = host_table(s.cold_remap);
  ggml_tensor *mask = host_table(s.cold_mask);
  if (!remap || !mask ||
      !lane->prepare(s.cpu, s.device, s.d_model, s.topk, s.cold_weights,
                     remap, mask))
    return nullptr;
  lane->trace_layer = s.layer;

  ggml_tensor *dst = ggml_dup_tensor(s.ctx, s.input);
  lane->device_buffer = ggml_backend_alloc_buffer(s.cuda, ggml_nbytes(dst));
  if (!lane->device_buffer ||
      ggml_backend_tensor_alloc(lane->device_buffer, dst,
                                ggml_backend_buffer_get_base(
                                    lane->device_buffer)) !=
          GGML_STATUS_SUCCESS)
    return nullptr;
  ggml_tensor *fork = ggml_cpy(s.ctx, s.input, dst);
  fork->src[2] = ggml_cont(s.ctx, s.route_ids);
  fork->src[3] = ggml_cont(s.ctx, s.route_weights);
  fork->src[4] = nullptr;
  fork->extra = lane.get();
  const std::string prefix = s.name_prefix ? s.name_prefix : "moe";
  ggml_set_name(fork, (prefix + "_early_ready_fork_l" +
                       std::to_string(s.layer)).c_str());
  ggml_backend_sched_set_tensor_backend(s.sched, dst, s.cuda);
  ggml_backend_sched_set_tensor_backend(s.sched, fork, s.cuda);
  lane->gpu_input = fork;
  ggml_build_forward_expand(s.graph, fork);

  ggml_tensor *cold_leaf = ggml_dup_tensor(s.ctx, lane->output);
  cold_leaf->buffer = lane->output->buffer;
  cold_leaf->data = lane->output->data;
  ggml_tensor *marker = ggml_map_custom1_inplace(
      s.ctx, cold_leaf, EarlyReadyLane::fallback, 1, lane.get());
  marker->extra = lane.get();
  ggml_set_name(marker, (prefix + "_early_ready_cold_l" +
                         std::to_string(s.layer)).c_str());
  ggml_backend_sched_set_tensor_backend(s.sched, cold_leaf, s.cpu);
  ggml_backend_sched_set_tensor_backend(s.sched, marker, s.cpu);
  if (!lane->async_output_copy)
    return nullptr;
  lane->cpu_marker = marker;
  if (!lane->prepare_gpu_output(s.cuda, s.ctx))
    return nullptr;
  ggml_set_name(lane->gpu_output,
                (prefix + "_early_ready_output_l" +
                 std::to_string(s.layer)).c_str());
  ggml_backend_sched_set_tensor_backend(s.sched, lane->gpu_output, s.cuda);
  return lane;
}

} // namespace zeromoe::graph
