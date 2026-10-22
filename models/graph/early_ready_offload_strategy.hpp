#pragma once

// Graph-integrated mixed VRAM/DDR executor. The patched scheduler drives the
// EarlyReadyLane; this class exposes the same object through both the common
// execution-policy interface and the cold-residency lifecycle.

#include "cold_moe_strategy_early_ready.hpp"
#include "moe_execution_strategy.hpp"

namespace moestream::graph {

class EarlyReadyOffloadStrategy final : public MoEExecutionStrategy,
                                         public ColdMoEStrategy {
 public:
  EarlyReadyOffloadStrategy(::EarlyReadyLane *lane,
                            const ColdMoEAdapter &adapter)
      : lane_(lane), lifecycle_(lane) {
    installed_ = lifecycle_.install(adapter);
  }

  MoEExecutionKind kind() const override {
    return MoEExecutionKind::EarlyReadyOffload;
  }
  const char *name() const override { return "early-ready-offload"; }
  bool graph_integrated() const override { return true; }
  ColdMoEStrategy *residency_strategy() override { return this; }

  bool bind(const MoERuntimeResources &, std::string *reason) override {
    if (!installed_ && reason)
      *reason = "early-ready adapter installation failed";
    return installed_;
  }
  bool execute(const MoEBlockSpec &, const MoEExecutionRequest &,
               std::string *reason) override {
    if (reason)
      *reason = "early-ready execution is scheduler-integrated";
    return false;
  }

  bool install(const ColdMoEAdapter &adapter) override {
    installed_ = lifecycle_.install(adapter);
    return installed_;
  }
  bool drives_residency() const override {
    return lifecycle_.drives_residency();
  }
  bool begin(const ColdMoEInvocation &inv) override {
    return installed_ && lifecycle_.begin(inv);
  }
  State state(const ColdMoEInvocation &inv) const override {
    return installed_ ? lifecycle_.state(inv) : State::Error;
  }
  bool finish(const ColdMoEInvocation &inv) override {
    return installed_ && lifecycle_.finish(inv);
  }
  void release(const ColdMoEInvocation &inv) override {
    if (installed_)
      lifecycle_.release(inv);
  }

 private:
  ::EarlyReadyLane *lane_ = nullptr;
  EarlyReadyLaneStrategy lifecycle_;
  bool installed_ = false;
};

} // namespace moestream::graph
