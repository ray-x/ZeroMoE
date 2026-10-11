// family_spec.hpp — per-family policy object for the full-graph builders.
//
// FamilyOps (family_ops.hpp) holds function pointers resolved from the GGUF;
// FamilySpec is the class layer above it for the decisions the graph builders
// used to take from scattered booleans and family checks: how the router is
// described, whether the CPU lane may be used, and which graph entry points
// the family has. A builder asks `family_spec(family)` instead of testing
// ModelFamily or passing flags.
#pragma once

#include "family_ops.hpp"
#include "model_family.hpp"
#include "moe_routed_config.hpp"

class FamilySpec {
 public:
  virtual ~FamilySpec() = default;

  virtual ModelFamily family() const = 0;
  virtual const char *name() const = 0;

  // Router description for `n_tokens` tokens. Families whose router is a plain
  // softmax ignore ArchitectureConfig::gating_func.
  virtual moe_routed::RouterSpec router_spec(const ArchitectureConfig &c,
                                             int n_tokens) const {
    return moe_routed::router_spec(c, n_tokens, /*honor_gating_func=*/true);
  }

  // Whether a layer's cold experts may run on a private CPU lane
  // (EarlyReadyLane) concurrently with the layer's GPU work.
  virtual bool supports_cpu_lane() const { return false; }

  // Whole-model graph entry points (enabled / bootstrap / build / run / ...).
  virtual ModelGraphOps graph_ops() const { return model_graph_ops_for(family()); }
};

namespace family_spec_detail {

class Qwen4ExpSpec final : public FamilySpec {
 public:
  ModelFamily family() const override { return ModelFamily::Qwen4Exp; }
  const char *name() const override { return "qwen4exp"; }
  moe_routed::RouterSpec router_spec(const ArchitectureConfig &c,
                                     int n_tokens) const override {
    return moe_routed::router_spec(c, n_tokens, /*honor_gating_func=*/false);
  }
  bool supports_cpu_lane() const override { return true; }
};

class Glm5NextSpec final : public FamilySpec {
 public:
  ModelFamily family() const override { return ModelFamily::GLM5NEXT; }
  const char *name() const override { return "glm5next"; }
  bool supports_cpu_lane() const override { return true; }
};

class DeepseekV4Spec final : public FamilySpec {
 public:
  ModelFamily family() const override { return ModelFamily::DeepSeek_V4; }
  const char *name() const override { return "deepseek_v4"; }
};

class DefaultSpec final : public FamilySpec {
 public:
  ModelFamily family() const override { return ModelFamily::Unknown; }
  const char *name() const override { return "default"; }
};

}  // namespace family_spec_detail

inline const FamilySpec &family_spec(ModelFamily fam) {
  static const family_spec_detail::Qwen4ExpSpec qwen;
  static const family_spec_detail::Glm5NextSpec glm;
  static const family_spec_detail::DeepseekV4Spec dsv4;
  static const family_spec_detail::DefaultSpec other;
  switch (fam) {
  case ModelFamily::Qwen4Exp:
    return qwen;
  case ModelFamily::GLM5NEXT:
    return glm;
  case ModelFamily::DeepSeek_V4:
    return dsv4;
  default:
    return other;
  }
}
