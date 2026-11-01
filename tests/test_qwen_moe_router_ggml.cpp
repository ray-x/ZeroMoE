#include "moe_routed_ggml.hpp"

#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace {

void check(bool ok, const char *message) {
  if (!ok) {
    throw std::runtime_error(message);
  }
}

void router_and_selected_weight_test() {
  constexpr int d = 3;
  constexpr int experts = 4;
  constexpr int topk = 2;
  constexpr int tokens = 2;
  constexpr float scale = 1.5f;

  std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(
      ggml_init({4 << 20, nullptr, true}), ggml_free);
  check(ctx != nullptr, "failed to create ggml context");

  auto *router = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, d, experts);
  auto *hidden = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, d, tokens);
  check(router && hidden, "failed to allocate router inputs");

  // Four distinct router rows make each token's top-k order unambiguous.
  const std::vector<float> router_values = {
      1.0f, 0.0f, 0.0f,
      0.0f, 1.0f, 0.0f,
      0.0f, 0.0f, 1.0f,
      0.5f, 0.5f, 0.5f,
  };
  const std::vector<float> hidden_values = {
      0.1f, 2.0f, 1.0f,
      3.0f, 0.5f, -1.0f,
  };
  moe_routed::RouterSpec spec;
  spec.gating = moe_routed::Gating::Softmax;
  spec.n_experts = experts;
  spec.topk = topk;
  spec.n_tokens = tokens;
  spec.weights_norm = true;
  spec.weights_scale = scale;

  moe_routed::RouterGateNodes gate_nodes;
  auto *probs = moe_routed::gate_probs(ctx.get(), spec, router, hidden,
                                       &gate_nodes);
  auto *ids = moe_routed::select_ids(ctx.get(), spec, probs, nullptr);
  auto *weights = moe_routed::route_weights(ctx.get(), spec, probs, ids);
  check(probs && ids && weights && gate_nodes.logits && gate_nodes.probs,
        "router graph builder returned a null tensor");

  auto *graph = ggml_new_graph(ctx.get());
  check(graph != nullptr, "failed to create router graph");
  ggml_build_forward_expand(graph, gate_nodes.logits);
  ggml_build_forward_expand(graph, gate_nodes.probs);
  ggml_build_forward_expand(graph, ids);
  ggml_build_forward_expand(graph, weights);

  std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(
      ggml_backend_cpu_init(), ggml_backend_free);
  check(backend != nullptr, "failed to initialize ggml CPU backend");
  ggml_backend_cpu_set_n_threads(backend.get(), 1);
  ggml_backend_buffer_t buffer =
      ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get());
  check(buffer != nullptr, "failed to allocate router graph tensors");

  ggml_backend_tensor_set(router, router_values.data(), 0,
                          router_values.size() * sizeof(float));
  ggml_backend_tensor_set(hidden, hidden_values.data(), 0,
                          hidden_values.size() * sizeof(float));
  check(ggml_backend_graph_compute(backend.get(), graph) ==
            GGML_STATUS_SUCCESS,
        "router graph compute failed");

  std::vector<float> logits(static_cast<size_t>(experts) * tokens);
  std::vector<float> actual_probs(logits.size());
  std::vector<int32_t> actual_ids(static_cast<size_t>(topk) * tokens);
  std::vector<float> actual_weights(actual_ids.size());
  for (int t = 0; t < tokens; ++t) {
    for (int e = 0; e < experts; ++e) {
      const size_t offset = static_cast<size_t>(e) * gate_nodes.logits->nb[0] +
                            static_cast<size_t>(t) * gate_nodes.logits->nb[1];
      const size_t index = static_cast<size_t>(t) * experts + e;
      ggml_backend_tensor_get(gate_nodes.logits, &logits[index], offset,
                              sizeof(float));
      const size_t probs_offset =
          static_cast<size_t>(e) * gate_nodes.probs->nb[0] +
          static_cast<size_t>(t) * gate_nodes.probs->nb[1];
      ggml_backend_tensor_get(gate_nodes.probs, &actual_probs[index],
                              probs_offset, sizeof(float));
    }
    for (int k = 0; k < topk; ++k) {
      const size_t index = static_cast<size_t>(t) * topk + k;
      const size_t ids_offset = static_cast<size_t>(k) * ids->nb[0] +
                                static_cast<size_t>(t) * ids->nb[1];
      ggml_backend_tensor_get(ids, &actual_ids[index], ids_offset,
                              sizeof(int32_t));
      const size_t weights_offset =
          static_cast<size_t>(k) * weights->nb[1] +
          static_cast<size_t>(t) * weights->nb[2];
      ggml_backend_tensor_get(weights, &actual_weights[index], weights_offset,
                              sizeof(float));
    }
  }

  std::vector<float> expected_logits(logits.size());
  std::vector<float> expected_probs(logits.size());
  std::vector<int32_t> expected_ids(actual_ids.size());
  std::vector<float> expected_weights(actual_weights.size());
  for (int t = 0; t < tokens; ++t) {
    float max_logit = -INFINITY;
    for (int e = 0; e < experts; ++e) {
      float value = 0.0f;
      for (int j = 0; j < d; ++j) {
        value += router_values[static_cast<size_t>(e) * d + j] *
                 hidden_values[static_cast<size_t>(t) * d + j];
      }
      expected_logits[static_cast<size_t>(t) * experts + e] = value;
      max_logit = std::max(max_logit, value);
    }
    float denominator = 0.0f;
    for (int e = 0; e < experts; ++e) {
      const float value = std::exp(
          expected_logits[static_cast<size_t>(t) * experts + e] - max_logit);
      expected_probs[static_cast<size_t>(t) * experts + e] = value;
      denominator += value;
    }
    for (int e = 0; e < experts; ++e) {
      expected_probs[static_cast<size_t>(t) * experts + e] /= denominator;
    }

    std::vector<int32_t> order(experts);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int32_t a, int32_t b) {
      return expected_probs[static_cast<size_t>(t) * experts + a] >
             expected_probs[static_cast<size_t>(t) * experts + b];
    });
    float selected_sum = 0.0f;
    for (int k = 0; k < topk; ++k) {
      const int32_t id = order[k];
      expected_ids[static_cast<size_t>(t) * topk + k] = id;
      selected_sum += expected_probs[static_cast<size_t>(t) * experts + id];
    }
    for (int k = 0; k < topk; ++k) {
      const auto index = static_cast<size_t>(t) * topk + k;
      const auto id = expected_ids[index];
      expected_weights[index] = scale *
          expected_probs[static_cast<size_t>(t) * experts + id] / selected_sum;
    }
  }

  for (size_t i = 0; i < logits.size(); ++i) {
    check(std::abs(logits[i] - expected_logits[i]) < 1e-5f,
          "router logits differ from scalar reference");
    check(std::abs(actual_probs[i] - expected_probs[i]) < 1e-5f,
          "router probabilities differ from scalar reference");
  }
  if (actual_ids != expected_ids) {
    for (int t = 0; t < tokens; ++t) {
      std::fprintf(stderr, "token %d topk actual=%d,%d expected=%d,%d\\n", t,
                   actual_ids[static_cast<size_t>(t) * topk],
                   actual_ids[static_cast<size_t>(t) * topk + 1],
                   expected_ids[static_cast<size_t>(t) * topk],
                   expected_ids[static_cast<size_t>(t) * topk + 1]);
    }
  }
  check(actual_ids == expected_ids,
        "ordered top-k expert IDs differ from scalar reference");
  for (size_t i = 0; i < actual_weights.size(); ++i) {
    check(std::abs(actual_weights[i] - expected_weights[i]) < 1e-5f,
          "normalized selected weights differ from scalar reference");
  }

  ggml_backend_buffer_free(buffer);
  std::puts("Qwen shared router logits/probabilities/IDs/weights: PASS");
}

} // namespace

int main() {
  try {
    router_and_selected_weight_test();
    return 0;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "Qwen MoE router unit test: %s\n", e.what());
    return 1;
  }
}
