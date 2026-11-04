#include "core/vram_dynamic_slot_plan.hpp"
#include "memory/memory_guard.hpp"
#include "models/deepseek/deepseek_cache_policy.hpp"

#include <cstdio>
#include <cstdlib>
#include <vector>

using zeromoe::memory::VramDynamicSlotReserve;
using zeromoe::memory::VramSlotPoolCapacity;
using zeromoe::memory::VramUnpinnedSlotCount;
using zeromoe::memory::DeepSeekDdrDummyCandidate;
using zeromoe::memory::choose_static_vram_upload_gpu;
using zeromoe::memory::is_pinned_ddr_dummy_candidate;
using zeromoe::memory::plan_deepseek_prefill_dummy_stage;
using zeromoe::memory::plan_vram_dynamic_slots;

namespace {
void check(bool condition, const char *message) {
  if (!condition) {
    std::fprintf(stderr, "FAIL: %s\n", message);
    std::exit(1);
  }
}

size_t slots_for(const std::vector<VramUnpinnedSlotCount> &counts, int gpu) {
  for (const auto &count : counts)
    if (count.gpu_id == gpu) return count.slots;
  return 0;
}

void simulate_static_uploads(const std::vector<VramDynamicSlotReserve> &reserve,
                             std::vector<VramUnpinnedSlotCount> &unpinned,
                             int count) {
  for (int i = 0; i < count; ++i) {
    const int gpu = choose_static_vram_upload_gpu(reserve, unpinned);
    check(gpu >= 0, "static preload must leave the dynamic reserve untouched");
    bool found = false;
    for (auto &entry : unpinned) {
      if (entry.gpu_id == gpu && entry.slots > 0) {
        --entry.slots;
        found = true;
        break;
      }
    }
    check(found, "selected GPU must have an unpinned slot");
  }
}

void test_dual_gpu_preload_preserves_dynamic_slots() {
  const auto plan = plan_vram_dynamic_slots(
      32, {{0, 1409, 0}, {1, 1226, 0}});
  check((bool)plan, "dual-GPU reserve should be feasible");
  check(plan.per_gpu.size() == 2, "reserve must include both active GPUs");
  check(plan.per_gpu[0].gpu_id == 0 && plan.per_gpu[0].slots == 17,
        "3090 reserve should be 17 slots");
  check(plan.per_gpu[1].gpu_id == 1 && plan.per_gpu[1].slots == 15,
        "5060 Ti reserve should be 15 slots");

  std::vector<VramUnpinnedSlotCount> unpinned = {{0, 1409}, {1, 1226}};
  simulate_static_uploads(plan.per_gpu, unpinned, 1024); // global hot pin set
  simulate_static_uploads(plan.per_gpu, unpinned, 1579); // VRAM-LRU set
  check(slots_for(unpinned, 0) == 17,
        "3090 must retain dynamic slots for same-layer dummy staging");
  check(slots_for(unpinned, 1) == 15,
        "5060 Ti must retain its share of staging slots");
  check(choose_static_vram_upload_gpu(plan.per_gpu, unpinned) == -1,
        "static preload must stop exactly at the planned dynamic reserve");
}

void test_existing_core_pins_reduce_available_pool() {
  const auto plan = plan_vram_dynamic_slots(
      32, {{0, 1409, 1024}, {1, 1226, 0}});
  check((bool)plan, "reserve should fit after core-GPU pins");
  check(plan.per_gpu[0].slots == 8 && plan.per_gpu[1].slots == 24,
        "reserve split must reflect capacity remaining after existing pins");
}

void test_infeasible_dynamic_reserve_fails_closed() {
  check(!(bool)plan_vram_dynamic_slots(1, {{0, 8, 0}, {1, 8, 0}}),
        "each active GPU needs a dynamic slot");
  check(!(bool)plan_vram_dynamic_slots(17, {{0, 8, 0}, {1, 8, 0}}),
        "reserve cannot exceed free slots");
  check(!(bool)plan_vram_dynamic_slots(2, {{0, 8, 8}, {1, 8, 0}}),
        "fully pinned GPU cannot satisfy its dynamic reserve");
}

void test_dummy_candidate_requires_pinned_matching_ddr_source() {
  check(is_pinned_ddr_dummy_candidate(7, 7, true, true, true),
        "matching pinned DDR expert with a slab is usable");
  check(!is_pinned_ddr_dummy_candidate(7, 8, true, true, true),
        "mismatched block identity is rejected");
  check(!is_pinned_ddr_dummy_candidate(7, 7, false, true, true),
        "expert owned by a GPU is not a CPU spill candidate");
  check(!is_pinned_ddr_dummy_candidate(7, 7, true, false, true),
        "un-pinned DDR source is rejected");
  check(!is_pinned_ddr_dummy_candidate(7, 7, true, true, false),
        "missing slab is rejected");
}

void test_prefill_dummy_stage_uses_gpu_local_reserve() {
  const auto reserve = plan_vram_dynamic_slots(
      32, {{0, 1409, 0}, {1, 1226, 0}});
  check((bool)reserve, "dual-GPU staging plan should be valid");

  std::vector<VramUnpinnedSlotCount> unpinned = {{0, 1409}, {1, 1226}};
  simulate_static_uploads(reserve.per_gpu, unpinned, 2603);

  const std::vector<DeepSeekDdrDummyCandidate> candidates = {
      {7, 8, true, true, true, true},  // corrupt identity must be ignored
      {19, 19, true, true, true, true}};
  for (int gpu : {0, 1}) {
    const auto plan = plan_deepseek_prefill_dummy_stage(
        true, 0, slots_for(unpinned, gpu), candidates);
    check((bool)plan, "each GPU should be able to stage a local dummy");
    check(plan.expert_ids.size() == 1 && plan.expert_ids[0] == 19,
          "prefill must select the matching pinned DDR expert");
  }

  // Reproduce the earlier failure shape: DDR candidates exist, but the
  // target GPU has no unpinned local slot because all reserve landed elsewhere.
  const auto stranded = plan_deepseek_prefill_dummy_stage(
      true, 0, 0, candidates);
  check(!(bool)stranded &&
            stranded.error == "no unpinned dynamic slot on this GPU",
        "dry-run must reject candidates that cannot be staged on the target GPU");
}

void test_prefill_dummy_stage_rejects_missing_fallback_inputs() {
  const std::vector<DeepSeekDdrDummyCandidate> none;
  check(!(bool)plan_deepseek_prefill_dummy_stage(true, 0, 1, none),
        "a dynamic slot without a pinned DDR candidate is insufficient");
  check(!(bool)plan_deepseek_prefill_dummy_stage(false, 0, 1, {}),
        "decode or disabled CPU prefill cannot use the DDR staging path");
  check((bool)plan_deepseek_prefill_dummy_stage(false, 1, 0, {}),
        "an existing same-layer slot needs no staging fallback");
}

void test_four_point_five_gib_host_floor_is_configurable() {
  using zeromoe::graphsettings::valid_integer;
  check(valid_integer("mem_guard_floor_mb", -1),
        "unset host-floor override should remain valid");
  check(valid_integer("mem_guard_floor_mb", 4608),
        "4.5 GiB host-floor override should be valid");
  check(!valid_integer("mem_guard_floor_mb", 4607),
        "values below the 4.5 GiB minimum should be rejected");
  check(::MemoryGuard::effective_stop_limit_mb(4608) == 10240,
        "default guard floor should remain 10 GiB without an explicit override");
  check(::MemoryGuard::resolve_stop_limit_mb(10240, 4608) == 4608,
        "explicit 4.5 GiB graph floor should reach the live allocation guard");
}

void test_deepseek_cpu_offload_is_phase_specific() {
  using zeromoe::deepseek_graph::use_cpu_cold_graph;
  check(use_cpu_cold_graph(false, 959, true, false),
        "multi-token prefill retains the CPU DDR spill path");
  check(!use_cpu_cold_graph(true, 1, true, false),
        "decode with staging enabled must use the GPU cache path");
  check(use_cpu_cold_graph(true, 1, true, true),
        "explicit CPU-cold decode remains selectable");
  check(!use_cpu_cold_graph(false, 1, true, false),
        "single-token non-decode request is not batched prefill");
  check(!use_cpu_cold_graph(true, 64, true, false),
        "batched decode does not select the single-token CPU-cold path");
}
} // namespace

int main() {
  test_dual_gpu_preload_preserves_dynamic_slots();
  test_existing_core_pins_reduce_available_pool();
  test_infeasible_dynamic_reserve_fails_closed();
  test_dummy_candidate_requires_pinned_matching_ddr_source();
  test_prefill_dummy_stage_uses_gpu_local_reserve();
  test_prefill_dummy_stage_rejects_missing_fallback_inputs();
  test_four_point_five_gib_host_floor_is_configurable();
  test_deepseek_cpu_offload_is_phase_specific();
  std::puts("VRAM dynamic-slot dry-run: PASS");
  return 0;
}
