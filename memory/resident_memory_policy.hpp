#pragma once

// Shared startup decisions for persistent weights and expert caches. No CUDA,
// model-family checks or allocations: callers supply measured device memory.
#include <algorithm>
#include <cstddef>
#include <string_view>

namespace zeromoe::memory {

inline bool valid_embedding_device(std::string_view device) {
  return device == "auto" || device == "cpu" || device == "gpu";
}

inline bool embedding_on_gpu(std::string_view device, bool graph_requires_gpu) {
  return device == "gpu" || (device == "auto" && graph_requires_gpu);
}

struct ExpertCacheBudget {
  size_t bytes = 0;
  size_t limit_bytes = 0;
  bool fitted = false;
  bool clamped = false;
};

// budget_bytes is the caller's accounting base: total VRAM for a dedicated
// vault (reserve includes its context), or measured free VRAM after dense
// weights for a dual-role GPU (reserve covers future state/workspace).
// free_bytes is always measured, and prevents spending memory already in use.
// A fitted zero is a real zero, never the legacy "0 means auto" sentinel.
inline ExpertCacheBudget expert_cache_budget(size_t budget_bytes, size_t free_bytes,
                                             size_t requested_bytes, size_t reserve_bytes,
                                             bool fit, bool zero_disables = false) {
  const size_t remaining = budget_bytes > reserve_bytes ? budget_bytes - reserve_bytes : 0;
  const size_t limit = std::min(remaining, free_bytes);
  const size_t requested = fit ? remaining : (requested_bytes || zero_disables ? requested_bytes : remaining);
  return {std::min(requested, limit), limit, fit, requested > limit};
}

} // namespace zeromoe::memory
