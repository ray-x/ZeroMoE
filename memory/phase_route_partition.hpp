#pragma once

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace zeromoe::memory {

// Transfer complete expert identities from the cold branch to a phase-hot
// branch. This changes ownership only: router slots and weights stay intact.
// Validate before mutating so an invalid partition cannot leave partial maps.
inline void claim_phase_routes(const std::vector<int> &experts,
                               std::vector<int32_t> &cold_remap,
                               std::vector<float> &cold_mask, int32_t inactive) {
  if (cold_remap.size() != cold_mask.size())
    throw std::invalid_argument("phase route map/mask sizes disagree");
  std::vector<bool> seen(cold_mask.size(), false);
  for (int expert : experts) {
    if (expert < 0 || size_t(expert) >= cold_mask.size() || seen[size_t(expert)] ||
        cold_mask[size_t(expert)] != 1.0f)
      throw std::invalid_argument("phase expert is duplicate or not owned by the cold branch");
    seen[size_t(expert)] = true;
  }
  for (int expert : experts) {
    cold_mask[size_t(expert)] = 0.0f;
    cold_remap[size_t(expert)] = inactive;
  }
}

} // namespace zeromoe::memory
