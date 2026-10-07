#include "qwen4exp_adaptive_policy.hpp"

#include <cassert>
#include <cmath>
#include <cstdio>

using namespace moestream::qwen4exp;

int main() {
  auto cpu = choose_adaptive_target(
      {0.1, 0.5, 0.2, 0.4, 0.5, 0.0, true, true, false});
  assert(cpu.target == AdaptiveTarget::Cpu);
  assert(cpu.cpu_ms == 0.6 && cpu.staged_gpu_ms == 1.1);

  auto gpu = choose_adaptive_target(
      {4.0, 3.0, 0.2, 0.4, 0.5, 0.0, true, true, false});
  assert(gpu.target == AdaptiveTarget::StagedGpu);
  assert(gpu.cpu_ms == 7.0);

  auto storage = choose_adaptive_target({10.0, 2.0, 0.1, 0.2, 0.4, 8.0,
                                          true, true, true});
  assert(storage.target == AdaptiveTarget::StagedGpu);
  assert(std::abs(storage.staged_gpu_ms - 8.7) < 1e-12);
  AdaptiveTiming dispatch_timing{10.0, 2.0, 0.1, 0.2, 0.4, 8.0,
                                 true, true, true};
  dispatch_timing.graph_dispatch_ms = 0.3;
  auto dispatch_cost = choose_adaptive_target(dispatch_timing);
  assert(std::abs(dispatch_cost.staged_gpu_ms - 9.0) < 1e-12);

  // Before the first staged sample, zero-initialized transfer and kernel
  // values are not evidence that the GPU path is free.
  auto unmeasured_stage = choose_adaptive_target(
      {0.1, 0.5, 0.0, 0.0, 0.0, 0.0, true, false, true});
  assert(unmeasured_stage.target == AdaptiveTarget::Cpu);
  assert(std::isinf(unmeasured_stage.staged_gpu_ms));

  auto pending = choose_adaptive_target({0.1, 0.1, 0.0, 0.0, 0.1, 0.0,
                                         true, false, true});
  assert(pending.target == AdaptiveTarget::Cpu);
  assert(std::isinf(pending.staged_gpu_ms));

  auto invalid = choose_adaptive_target(
      {NAN, 1.0, 0.0, 0.0, 0.0, 0.0, false, true, false});
  assert(invalid.target == AdaptiveTarget::StagedGpu);
  assert(std::isinf(invalid.cpu_ms));

  auto no_samples = choose_adaptive_target({});
  assert(no_samples.target == AdaptiveTarget::Cpu);
  assert(std::isinf(no_samples.cpu_ms) &&
         std::isinf(no_samples.staged_gpu_ms));
  std::puts("PASS: adaptive CPU/staged-GPU timing policy");
}
