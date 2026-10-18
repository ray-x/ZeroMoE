#pragma once

// Builds the shared router description from a model's architecture config, so
// every family's graph builder derives it the same way.

#include "gguf_loader.hpp"
#include "moe_routed_ggml.hpp"

namespace moe_routed {

// `honor_gating_func`: families whose router is always a plain softmax
// (Qwen4Exp) pass false; GLM5NEXT / DeepSeek-V4 select sigmoid or sqrt-softplus
// through ArchitectureConfig::gating_func.
inline RouterSpec router_spec(const ArchitectureConfig &c, int n_tokens,
                              bool honor_gating_func = true) {
  RouterSpec s;
  s.n_experts = c.n_experts;
  s.topk = c.topk;
  s.n_tokens = n_tokens;
  s.weights_norm = c.expert_weights_norm;
  s.weights_scale = c.expert_weights_scale;
  if (honor_gating_func) {
    switch (c.gating_func) {
    case GatingFunc::Sigmoid:
      s.gating = Gating::Sigmoid;
      break;
    case GatingFunc::SqrtSoftplus:
      s.gating = Gating::SqrtSoftplus;
      break;
    case GatingFunc::Softmax:
    default:
      s.gating = Gating::Softmax;
      break;
    }
  }
  return s;
}

}  // namespace moe_routed
