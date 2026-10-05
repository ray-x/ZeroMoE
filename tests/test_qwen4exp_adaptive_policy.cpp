#include "qwen4exp_adaptive_policy.hpp"

#include <cassert>
#include <cmath>
#include <cstdio>

using namespace moestream::qwen4exp;

int main() {
  auto cpu = choose_adaptive_target({0.1, 0.5, 0.2, 0.4, 0.5, 0.0});
  assert(cpu.target == AdaptiveTarget::Cpu);
  assert(cpu.cpu_ms == 0.6 && cpu.staged_gpu_ms == 1.1);

  auto gpu = choose_adaptive_target({4.0, 3.0, 0.2, 0.4, 0.5, 0.0});
  assert(gpu.target == AdaptiveTarget::StagedGpu);
  assert(gpu.cpu_ms == 7.0);

  auto storage = choose_adaptive_target({10.0, 2.0, 0.1, 0.2, 0.4, 8.0,
                                          true, true, true});
  assert(storage.target == AdaptiveTarget::StagedGpu);
  assert(std::abs(storage.staged_gpu_ms - 8.7) < 1e-12);

  auto pending = choose_adaptive_target({0.1, 0.1, 0.0, 0.0, 0.1, 0.0,
                                         true, false, true});
  assert(pending.target == AdaptiveTarget::Cpu);
  assert(std::isinf(pending.staged_gpu_ms));

  auto invalid = choose_adaptive_target({NAN, 1.0, 0.0, 0.0, 0.0, 0.0});
  assert(invalid.target == AdaptiveTarget::StagedGpu);
  assert(std::isinf(invalid.cpu_ms));
  std::puts("PASS: adaptive CPU/staged-GPU timing policy");
}
