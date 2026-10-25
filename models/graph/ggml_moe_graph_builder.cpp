#include "dense_dispatch.hpp"
#include "ggml_moe_graph_builder.hpp"

#if defined(ZEROMOE_USE_GGML_CUDA)

#include <cmath>
#include <utility>

namespace moe {

GgmlGraphAllocation::GgmlGraphAllocation(
    GgmlContextGuard context, ggml_gallocr_t allocator,
    ggml_cgraph *graph) noexcept
    : context_(std::move(context)), allocator_(allocator), graph_(graph) {}

GgmlGraphAllocation::~GgmlGraphAllocation() {
  if (allocator_) ggml_gallocr_free(allocator_);
}

GgmlGraphAllocation::GgmlGraphAllocation(
    GgmlGraphAllocation &&other) noexcept
    : context_(std::move(other.context_)),
      allocator_(std::exchange(other.allocator_, nullptr)),
      graph_(std::exchange(other.graph_, nullptr)) {}

GgmlGraphAllocation &GgmlGraphAllocation::operator=(
    GgmlGraphAllocation &&other) noexcept {
  if (this != &other) {
    if (allocator_) ggml_gallocr_free(allocator_);
    context_ = std::move(other.context_);
    allocator_ = std::exchange(other.allocator_, nullptr);
    graph_ = std::exchange(other.graph_, nullptr);
  }
  return *this;
}

GgmlGraphAllocation::operator bool() const noexcept {
  return context_ && allocator_ && graph_;
}

ggml_context *GgmlGraphAllocation::release_context() noexcept {
  return context_.release();
}

ggml_gallocr_t GgmlGraphAllocation::release_allocator() noexcept {
  return std::exchange(allocator_, nullptr);
}

GgmlGraphAllocation allocate_graph(GgmlContextGuard context,
                                   ggml_cgraph *graph,
                                   ggml_backend_t backend) {
  if (!context || !graph || !backend) return {};
  ggml_gallocr_t allocator = ggml_gallocr_new(
      ggml_backend_get_default_buffer_type(backend));
  if (!allocator || !ggml_gallocr_alloc_graph(allocator, graph)) {
    if (allocator) ggml_gallocr_free(allocator);
    return {};
  }
  return GgmlGraphAllocation(std::move(context), allocator, graph);
}

RouterGraphNodes build_router_topology(
    ggml_context *ctx, ggml_tensor *router_weight, ggml_tensor *input,
    ggml_tensor *selection_bias, int n_experts, int topk, int n_tokens,
    bool normalize, float weight_scale) {
  RouterGraphNodes result;
  if (!ctx || !router_weight || !input) return result;
  result.logits = ggml_mul_mat(ctx, router_weight, input);
  ggml_tensor *probs =
      result.logits ? ggml_soft_max(ctx, result.logits) : nullptr;
  ggml_tensor *selection =
      probs && selection_bias ? ggml_add(ctx, probs, selection_bias) : probs;
  result.ids =
      selection
          ? ggml_cont(ctx, ggml_argsort_top_k(ctx, selection, topk))
          : nullptr;
  ggml_tensor *probs3 =
      probs ? ggml_reshape_3d(ctx, probs, 1, n_experts, n_tokens) : nullptr;
  result.weights =
      probs3 && result.ids ? ggml_get_rows(ctx, probs3, result.ids) : nullptr;
  if (normalize && result.weights) {
    ggml_tensor *w2 =
        ggml_reshape_2d(ctx, result.weights, topk, n_tokens);
    ggml_tensor *sum = ggml_sum_rows(ctx, w2);
    sum = ggml_clamp(ctx, sum, 6.103515625e-5f, INFINITY);
    w2 = ggml_div(ctx, w2, sum);
    result.weights = ggml_reshape_3d(ctx, w2, 1, topk, n_tokens);
  }
  if (weight_scale != 0.0f && weight_scale != 1.0f && result.weights)
    result.weights = ggml_scale(ctx, result.weights, weight_scale);
  return result;
}

SharedGraphNodes build_shared_topology(
    ggml_context *ctx, ggml_tensor *gate_weight, ggml_tensor *up_weight,
    ggml_tensor *down_weight, ggml_tensor *gate_input_weight,
    ggml_tensor *input, ggml_tensor *routed, float clamp) {
  SharedGraphNodes result;
  if (!ctx || !gate_weight || !up_weight || !down_weight || !input)
    return result;
  result.gate = ggml_mul_mat(ctx, gate_weight, input);
  result.up = ggml_mul_mat(ctx, up_weight, input);
  if (clamp > 0.0f && swiglu_clamp_fused() && result.gate && result.up) {
    // ggml_swiglu_clamp semantics: clamp gate/up before the activation.
    result.activation =
        ggml_mul(ctx, ggml_silu(ctx, ggml_clamp(ctx, result.gate, -INFINITY, clamp)),
                 ggml_clamp(ctx, result.up, -clamp, clamp));
  } else {
    result.activation =
        result.gate && result.up
            ? ggml_mul(ctx, ggml_silu(ctx, result.gate), result.up)
            : nullptr;
    if (clamp > 0.0f && result.activation)
      result.activation = ggml_clamp(ctx, result.activation, -clamp, clamp);
  }
  result.shared_raw = result.activation
                          ? ggml_mul_mat(ctx, down_weight, result.activation)
                          : nullptr;
  result.shared = result.shared_raw;
  if (gate_input_weight && result.shared_raw) {
    ggml_tensor *scalar =
        ggml_sigmoid(ctx, ggml_mul_mat(ctx, gate_input_weight, input));
    result.shared = scalar ? ggml_mul(ctx, result.shared_raw, scalar) : nullptr;
  }
  result.combined =
      routed && result.shared ? ggml_add(ctx, routed, result.shared)
                              : result.shared;
  return result;
}

MoeGraphNodes build_moe_topology(
    ggml_context *ctx, ggml_tensor *gate_weight, ggml_tensor *up_weight,
    ggml_tensor *down_weight, ggml_tensor *expert_input, ggml_tensor *ids,
    ggml_tensor *weights, int d_model, int n_tokens, int topk, bool chunked) {
  MoeGraphNodes result;
  if (!ctx || !gate_weight || !up_weight || !down_weight || !expert_input ||
      !ids || !weights)
    return result;
  result.gate = ggml_mul_mat_id(ctx, gate_weight, expert_input, ids);
  result.up = ggml_mul_mat_id(ctx, up_weight, expert_input, ids);
  result.activation =
      result.gate && result.up
          ? ggml_swiglu_split(ctx, result.gate, result.up)
          : nullptr;
  result.down = result.activation
                    ? ggml_mul_mat_id(ctx, down_weight, result.activation, ids)
                    : nullptr;
  ggml_tensor *weighted =
      result.down ? ggml_mul(ctx, result.down, weights) : nullptr;
  if (!weighted) return result;
  if (chunked) {
    result.output =
        ggml_view_2d(ctx, weighted, d_model, n_tokens, weighted->nb[2], 0);
    return result;
  }
  for (int k = 0; k < topk; ++k) {
    ggml_tensor *view =
        ggml_view_2d(ctx, weighted, d_model, n_tokens, weighted->nb[2],
                     static_cast<size_t>(k) * weighted->nb[1]);
    result.output =
        k == 0 ? view : ggml_add(ctx, result.output, view);
  }
  if (topk == 1 && result.output)
    result.output = ggml_cont(ctx, result.output);
  return result;
}

}  // namespace moe

#endif
