#pragma once

#include <cmath>
#include <limits>

namespace moestream::qwen4exp {

enum class AdaptiveTarget { Cpu, StagedGpu };

// One estimate for the cold route set of a single layer invocation. The
// caller supplies measurements from CPU execution, the upload stream, H2D,
// graph dispatch/GPU execution and the storage service. A target is invalid
// until its timings have been measured; zero-initialized estimates must never
// make an unmeasured path look free.
struct AdaptiveTiming {
  double cpu_queue_ms = 0.0;
  double cpu_compute_ms = 0.0;
  double transfer_queue_ms = 0.0;
  double h2d_ms = 0.0;
  double gpu_compute_ms = 0.0;
  double storage_ms = 0.0;
  bool cpu_ready = false;
  bool staged_ready = false;
  bool storage_required = false;
  // Host payload preparation plus graph launch/readback time not represented
  // by CUDA-event GPU compute or H2D intervals.
  double graph_dispatch_ms = 0.0;
};

struct AdaptiveDecision {
  AdaptiveTarget target = AdaptiveTarget::Cpu;
  double cpu_ms = std::numeric_limits<double>::infinity();
  double staged_gpu_ms = std::numeric_limits<double>::infinity();
};

inline bool nonnegative_finite(double value) {
  return std::isfinite(value) && value >= 0.0;
}

inline AdaptiveDecision choose_adaptive_target(const AdaptiveTiming &timing) {
  AdaptiveDecision decision;
  const bool cpu_valid = timing.cpu_ready &&
      nonnegative_finite(timing.cpu_queue_ms) &&
      nonnegative_finite(timing.cpu_compute_ms);
  const bool gpu_valid = timing.staged_ready &&
      nonnegative_finite(timing.transfer_queue_ms) &&
      nonnegative_finite(timing.h2d_ms) &&
      nonnegative_finite(timing.gpu_compute_ms) &&
      nonnegative_finite(timing.graph_dispatch_ms) &&
      (!timing.storage_required || nonnegative_finite(timing.storage_ms));
  if (cpu_valid)
    decision.cpu_ms = timing.cpu_queue_ms + timing.cpu_compute_ms;
  if (gpu_valid) {
    decision.staged_gpu_ms = timing.transfer_queue_ms + timing.h2d_ms +
        timing.graph_dispatch_ms + timing.gpu_compute_ms +
        (timing.storage_required ? timing.storage_ms : 0.0);
  }
  // CPU wins exact ties. It has no staging ownership to retain, which keeps
  // the fallback deterministic when the estimates are still warming up.
  if (decision.staged_gpu_ms < decision.cpu_ms)
    decision.target = AdaptiveTarget::StagedGpu;
  return decision;
}

}  // namespace moestream::qwen4exp
