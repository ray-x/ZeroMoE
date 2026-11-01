#pragma once

#include <algorithm>
#include <cstddef>

namespace zeromoe::hardware {

// Capacity/performance proxy for automatic CUDA device selection. VRAM is
// multiplied by a coarse compute proxy (SM count, clock, architecture) so a
// GPU that can keep more expert weights resident is preferred, while newer
// or faster devices still win when capacities are comparable. Explicit
// per-GPU device_id values always override this heuristic.
inline long double gpu_capacity_performance_score(
    size_t total_vram_bytes, int sm_major, int sm_minor,
    int multiprocessor_count, int clock_rate_khz) {
  const long double vram = static_cast<long double>(total_vram_bytes);
  const long double compute =
      static_cast<long double>(std::max(1, multiprocessor_count)) *
      static_cast<long double>(std::max(1, clock_rate_khz));
  const long double architecture = static_cast<long double>(
      std::max(1, sm_major * 10 + sm_minor));
  return vram * compute * architecture;
}

} // namespace zeromoe::hardware
