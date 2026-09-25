#pragma once

#if defined(MOESTREAM_USE_GGML_CUDA)

#include "ggml_resource_guards.hpp"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

namespace moe {

class GgmlGraphAllocation {
 public:
  GgmlGraphAllocation() = default;
  GgmlGraphAllocation(GgmlContextGuard context, ggml_gallocr_t allocator,
                      ggml_cgraph *graph) noexcept;
  ~GgmlGraphAllocation();

  GgmlGraphAllocation(const GgmlGraphAllocation &) = delete;
  GgmlGraphAllocation &operator=(const GgmlGraphAllocation &) = delete;
  GgmlGraphAllocation(GgmlGraphAllocation &&other) noexcept;
  GgmlGraphAllocation &operator=(GgmlGraphAllocation &&other) noexcept;

  explicit operator bool() const noexcept;
  ggml_context *release_context() noexcept;
  ggml_gallocr_t release_allocator() noexcept;
  ggml_cgraph *graph() const noexcept { return graph_; }

 private:
  GgmlContextGuard context_;
  ggml_gallocr_t allocator_ = nullptr;
  ggml_cgraph *graph_ = nullptr;
};

GgmlGraphAllocation allocate_graph(GgmlContextGuard context,
                                   ggml_cgraph *graph,
                                   ggml_backend_t backend);

struct RouterGraphNodes {
  ggml_tensor *logits = nullptr;
  ggml_tensor *ids = nullptr;
  ggml_tensor *weights = nullptr;
};

RouterGraphNodes build_router_topology(ggml_context *ctx,
                                       ggml_tensor *router_weight,
                                       ggml_tensor *input,
                                       ggml_tensor *selection_bias,
                                       int n_experts, int topk, int n_tokens,
                                       bool normalize, float weight_scale);

struct SharedGraphNodes {
  ggml_tensor *gate = nullptr;
  ggml_tensor *up = nullptr;
  ggml_tensor *activation = nullptr;
  ggml_tensor *shared_raw = nullptr;
  ggml_tensor *shared = nullptr;
  ggml_tensor *combined = nullptr;
};

SharedGraphNodes build_shared_topology(
    ggml_context *ctx, ggml_tensor *gate_weight, ggml_tensor *up_weight,
    ggml_tensor *down_weight, ggml_tensor *gate_input_weight,
    ggml_tensor *input, ggml_tensor *routed, float clamp);

struct MoeGraphNodes {
  ggml_tensor *gate = nullptr;
  ggml_tensor *up = nullptr;
  ggml_tensor *activation = nullptr;
  ggml_tensor *down = nullptr;
  ggml_tensor *output = nullptr;
};

MoeGraphNodes build_moe_topology(
    ggml_context *ctx, ggml_tensor *gate_weight, ggml_tensor *up_weight,
    ggml_tensor *down_weight, ggml_tensor *expert_input, ggml_tensor *ids,
    ggml_tensor *weights, int d_model, int n_tokens, int topk, bool chunked);

}  // namespace moe

#endif
