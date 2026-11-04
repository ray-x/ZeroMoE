#pragma once

#include <algorithm>
#include <cstddef>
#include <limits>
#include <string>
#include <vector>

namespace zeromoe::memory {

struct VramSlotPoolCapacity {
  int gpu_id = -1;
  size_t slots = 0;
  size_t pinned = 0;
};

struct VramDynamicSlotReserve {
  int gpu_id = -1;
  size_t slots = 0;
};

struct VramUnpinnedSlotCount {
  int gpu_id = -1;
  size_t slots = 0;
};

struct VramDynamicSlotPlan {
  std::vector<VramDynamicSlotReserve> per_gpu;
  std::string error;
  explicit operator bool() const { return error.empty(); }
};

// Split a global dynamic-slot reserve over active GPU pools. Keep at least one
// unpinned slot on every participating GPU, then distribute the remainder in
// proportion to each pool's capacity remaining after its current pins.
inline VramDynamicSlotPlan plan_vram_dynamic_slots(
    size_t requested, std::vector<VramSlotPoolCapacity> pools) {
  VramDynamicSlotPlan plan;
  pools.erase(std::remove_if(pools.begin(), pools.end(),
                             [](const auto &p) { return p.slots == 0; }),
              pools.end());
  std::sort(pools.begin(), pools.end(),
            [](const auto &a, const auto &b) { return a.gpu_id < b.gpu_id; });
  if (pools.empty()) {
    if (requested) plan.error = "no active VRAM pools";
    return plan;
  }
  for (size_t i = 0; i < pools.size(); ++i) {
    if (pools[i].gpu_id < 0 || pools[i].pinned > pools[i].slots) {
      plan.error = "invalid GPU pool capacity or pinned slot count";
      return plan;
    }
    if (i && pools[i - 1].gpu_id == pools[i].gpu_id) {
      plan.error = "duplicate GPU pool id";
      return plan;
    }
  }
  if (requested == 0) {
    for (const auto &pool : pools) plan.per_gpu.push_back({pool.gpu_id, 0});
    return plan;
  }
  if (requested < pools.size()) {
    plan.error = "dynamic VRAM reserve must leave at least one slot per GPU";
    return plan;
  }

  size_t available_total = 0;
  size_t weight_total = 0;
  for (const auto &pool : pools) {
    const size_t available = pool.slots - pool.pinned;
    if (!available) {
      plan.error = "a GPU pool is fully pinned and has no dynamic slot";
      return plan;
    }
    if (available_total > std::numeric_limits<size_t>::max() - available) {
      plan.error = "GPU slot capacity sum overflow";
      return plan;
    }
    available_total += available;
    weight_total += available - 1;
    plan.per_gpu.push_back({pool.gpu_id, 1});
  }
  if (requested > available_total) {
    plan.error = "dynamic VRAM reserve exceeds available GPU slots";
    return plan;
  }

  const size_t remainder = requested - pools.size();
  struct Fraction { size_t index; long double value; };
  std::vector<Fraction> fractions;
  fractions.reserve(pools.size());
  size_t assigned = pools.size();
  for (size_t i = 0; i < pools.size(); ++i) {
    const size_t weight = pools[i].slots - pools[i].pinned - 1;
    const long double exact = weight_total
        ? static_cast<long double>(remainder) * weight / weight_total
        : 0.0L;
    const size_t whole = static_cast<size_t>(exact);
    plan.per_gpu[i].slots += whole;
    assigned += whole;
    fractions.push_back({i, exact - static_cast<long double>(whole)});
  }
  std::sort(fractions.begin(), fractions.end(), [&](const auto &a, const auto &b) {
    if (a.value != b.value) return a.value > b.value;
    return pools[a.index].gpu_id < pools[b.index].gpu_id;
  });
  for (size_t i = 0; assigned < requested; ++i) {
    if (i >= fractions.size()) i = 0;
    const size_t index = fractions[i].index;
    const size_t capacity = pools[index].slots - pools[index].pinned;
    if (plan.per_gpu[index].slots >= capacity) continue;
    ++plan.per_gpu[index].slots;
    ++assigned;
  }
  return plan;
}

// Preserve current topology order for static preloads, but stop before they
// consume each device's dynamic graph-staging reserve.
inline int choose_static_vram_upload_gpu(
    const std::vector<VramDynamicSlotReserve> &reserve,
    const std::vector<VramUnpinnedSlotCount> &unpinned) {
  for (const auto &gpu : reserve) {
    const auto found = std::find_if(unpinned.begin(), unpinned.end(),
        [&](const auto &count) { return count.gpu_id == gpu.gpu_id; });
    if (found != unpinned.end() && found->slots > gpu.slots)
      return gpu.gpu_id;
  }
  return -1;
}

inline bool is_pinned_ddr_dummy_candidate(int expected_expert,
                                          int block_expert,
                                          bool ownerless,
                                          bool pinned,
                                          bool has_slab) {
  return expected_expert >= 0 && block_expert == expected_expert &&
         ownerless && pinned && has_slab;
}

struct DeepSeekDdrDummyCandidate {
  int expert_id = -1;
  int block_expert_id = -1;
  bool ownerless = false;
  bool pinned = false;
  bool has_slab = false;
  bool absent_from_vram = false;
};

struct DeepSeekDummyStagePlan {
  std::vector<int> expert_ids;
  std::string error;
  explicit operator bool() const { return error.empty(); }
};

// Preflight the DeepSeek prefill fallback before it mutates the live cache.
// A GPU with no same-layer resident dummy needs both an evictable local slot
// and a pinned DDR expert that is not already present in another cache entry.
inline DeepSeekDummyStagePlan plan_deepseek_prefill_dummy_stage(
    bool cpu_cold_prefill, size_t same_layer_slots,
    size_t local_dynamic_slots,
    const std::vector<DeepSeekDdrDummyCandidate> &candidates) {
  DeepSeekDummyStagePlan plan;
  if (same_layer_slots) return plan;
  if (!cpu_cold_prefill) {
    plan.error = "no same-layer cache slot and CPU prefill staging is disabled";
    return plan;
  }
  if (!local_dynamic_slots) {
    plan.error = "no unpinned dynamic slot on this GPU";
    return plan;
  }
  for (const auto &candidate : candidates) {
    if (!candidate.absent_from_vram ||
        !is_pinned_ddr_dummy_candidate(candidate.expert_id,
                                       candidate.block_expert_id,
                                       candidate.ownerless,
                                       candidate.pinned,
                                       candidate.has_slab))
      continue;
    plan.expert_ids.push_back(candidate.expert_id);
  }
  if (plan.expert_ids.empty()) {
    plan.error = "no stageable pinned DDR dummy expert";
  }
  return plan;
}

} // namespace zeromoe::memory
