// Unit test for include/moe_partial_route.hpp (no ggml, no model, no GPU).
//   g++ -std=c++20 -Iinclude tests/test_moe_partial_route.cpp -o /tmp/test_moe_partial_route
//   /tmp/test_moe_partial_route capable    # patched ggml advertises the negative-id skip
//   /tmp/test_moe_partial_route stock      # stock ggml: capability function reports 0
//   /tmp/test_moe_partial_route optout     # capable ggml but MOESTREAM_MMID_NO_NEGATIVE_SKIP=1
//   /tmp/test_moe_partial_route missing    # capability symbol absent is covered by the weak link (compile check only)
// The capability result is cached once per process, so each mode runs in its own process.

#include "moe_partial_route.hpp"

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>

static int g_capability = 0;
extern "C" int ggml_cpu_mul_mat_id_negative_skip(void) { return g_capability; }

// The host-node route classifier is required to stay allocation-free. Count
// scalar/array allocations only while exact_once() is running, after all test
// fixtures have been prepared.
static bool g_track_allocations = false;
static size_t g_tracked_allocations = 0;

void *operator new(std::size_t size) {
  if (g_track_allocations)
    ++g_tracked_allocations;
  if (void *memory = std::malloc(size == 0 ? 1 : size))
    return memory;
  throw std::bad_alloc();
}

void *operator new[](std::size_t size) {
  if (g_track_allocations)
    ++g_tracked_allocations;
  if (void *memory = std::malloc(size == 0 ? 1 : size))
    return memory;
  throw std::bad_alloc();
}

void operator delete(void *memory) noexcept { std::free(memory); }
void operator delete[](void *memory) noexcept { std::free(memory); }
void operator delete(void *memory, std::size_t) noexcept { std::free(memory); }
void operator delete[](void *memory, std::size_t) noexcept { std::free(memory); }

static int g_failures = 0;
#define CHECK(cond, msg)                                                       \
  do {                                                                         \
    if (!(cond)) {                                                             \
      std::fprintf(stderr, "FAIL: %s  [%s:%d]\n", msg, __FILE__, __LINE__);    \
      ++g_failures;                                                            \
    }                                                                          \
  } while (0)

using moe_partial::ExpertBranchMap;
using moe_partial::HotColdSplit;
using moe_partial::RouteClassification;
using moe_partial::RouteTarget;
using moe_partial::SentinelPolicy;

int main(int argc, char **argv) {
  const char *mode = argc > 1 ? argv[1] : "capable";
  bool expect_skip = false;
  if (!std::strcmp(mode, "capable")) {
    g_capability = 1;
    expect_skip = true;
  } else if (!std::strcmp(mode, "stock")) {
    g_capability = 0;
  } else if (!std::strcmp(mode, "optout")) {
    g_capability = 1;
    setenv("MOESTREAM_MMID_NO_NEGATIVE_SKIP", "1", 1);
  } else {
    std::fprintf(stderr, "unknown mode %s\n", mode);
    return 2;
  }

  // 6 experts; hot holds {1,4} at compact positions 0,1; cold holds the other four at positions 0..3.
  const std::vector<int32_t> hot_index = {-1, 0, -1, -1, 1, -1};
  const std::vector<int32_t> cold_index = {0, -1, 1, 2, -1, 3};

  // ---- single-token decode -------------------------------------------------------------------
  HotColdSplit dec;
  CHECK(HotColdSplit::build(hot_index, cold_index, 1, true, true, &dec),
        "exact partition accepted");
  const int32_t want_inactive = expect_skip ? -1 : 0;
  CHECK(dec.hot().inactive_id() == want_inactive, "decode hot inactive id");
  CHECK(dec.cold().inactive_id() == want_inactive, "decode cold inactive id");
  CHECK(dec.hot().skips_inactive_routes() == expect_skip, "skip flag");
  const auto &h = dec.hot().route_ids();
  CHECK(h.size() == 6, "decode table size");
  CHECK(h[1] == 0 && h[4] == 1, "hot owned experts keep their branch index");
  CHECK(h[0] == want_inactive && h[2] == want_inactive && h[3] == want_inactive &&
            h[5] == want_inactive,
        "hot non-owned routes use the inactive id");
  const auto &c = dec.cold().route_ids();
  CHECK(c[0] == 0 && c[2] == 1 && c[3] == 2 && c[5] == 3, "cold indices");
  CHECK(c[1] == want_inactive && c[4] == want_inactive, "cold non-owned inactive");
  const auto &hm = dec.hot().route_mask();
  const auto &cm = dec.cold().route_mask();
  for (int e = 0; e < 6; ++e)
    CHECK(hm[e] + cm[e] == 1.0f, "every expert is owned by exactly one branch (mask)");
  CHECK(hm[1] == 1.0f && hm[0] == 0.0f, "hot mask");

  // ---- prefill (S > 1): never negative, tables repeated per token, token-major ----------------
  HotColdSplit pre;
  CHECK(HotColdSplit::build(hot_index, cold_index, 3, false, true, &pre), "prefill build");
  CHECK(pre.hot().inactive_id() == 0, "prefill uses the valid dummy id, never negative");
  CHECK(pre.hot().route_ids().size() == 18, "prefill table repeated per token");
  CHECK(pre.hot().route_ids()[6 + 1] == 0 && pre.hot().route_ids()[12 + 4] == 1,
        "token-major layout keeps per-token copies");
  HotColdSplit pre_claim;  // even if a caller claims decode with S>1, the class refuses the skip
  CHECK(HotColdSplit::build(hot_index, cold_index, 3, true, true, &pre_claim), "build");
  CHECK(pre_claim.hot().inactive_id() == 0, "single_token_decode with n_tokens>1 still safe");

  // ---- policy override ------------------------------------------------------------------------
  HotColdSplit forced;
  CHECK(HotColdSplit::build(hot_index, cold_index, 1, true, true, &forced,
                            SentinelPolicy::ForceDummy),
        "forced build");
  CHECK(forced.hot().inactive_id() == 0, "ForceDummy is always 0");

  // ---- partition validation -------------------------------------------------------------------
  HotColdSplit bad;
  CHECK(!HotColdSplit::build({0, 1, -1}, {-1, 0, 1}, 1, true, true, &bad),
        "expert held by both branches rejected");
  CHECK(!HotColdSplit::build({0, -1, -1}, {-1, -1, 0}, 1, true, true, &bad),
        "expert held by neither branch rejected");
  HotColdSplit dyn;  // dynamic residency: overlap allowed, no exactness requirement
  CHECK(HotColdSplit::build({0, 1, -1}, {0, 1, 2}, 1, true, false, &dyn),
        "non-exact (dynamic) partition accepted");
  CHECK(!HotColdSplit::build({0, 1}, {0, 1, 2}, 1, true, false, &dyn),
        "size mismatch rejected");

  // ---- explicit ownership (hot expert whose physical slot is missing) --------------------------
  ExpertBranchMap m = ExpertBranchMap::build({-1, 0}, 1, true, {1, 1});
  CHECK(m.route_mask()[0] == 1.0f && m.route_ids()[0] == want_inactive,
        "owned-but-unaddressable route: masked-in, inactive id");

  // ---- immutable per-layer route classification ----------------------------------------------
  // The classification is slot-based: duplicate expert IDs retain two
  // independent router contributions and both must be assigned exactly once.
  const std::vector<float> weights = {0.10f, 0.20f, 0.30f, 0.40f};
  RouteClassification classification;
  classification.reserve(4);
  CHECK(RouteClassification::build({0, 1, 2, 3}, weights,
                                   {0, 1, 2, 3}, {-1, -1, -1, -1}, 7,
                                   &classification),
        "all-hot classification builds");
  CHECK(classification.exact_once() && classification.size() == 4,
        "all-hot routes execute exactly once");
  CHECK(classification.count(RouteTarget::ResidentGpu) == 4 &&
            classification.count(RouteTarget::StagedGpu) == 0 &&
            classification.count(RouteTarget::Cpu) == 0,
        "all-hot target counts");
  std::vector<int32_t> projected_ids(4);
  std::vector<float> projected_weights(4);
  CHECK(classification.write_branch_routes(RouteTarget::Cpu, 0,
                                           projected_ids.data(),
                                           projected_weights.data(), 4),
        "all-hot CPU branch projection succeeds");
  for (float weight : projected_weights)
    CHECK(weight == 0.0f, "all-hot CPU branch has a fresh zero contribution");

  CHECK(RouteClassification::build({0, 1, 2, 3}, weights,
                                   {-1, -1, -1, -1}, {-1, -1, -1, -1}, 8,
                                   &classification),
        "all-cold classification builds");
  CHECK(classification.exact_once() && classification.count(RouteTarget::Cpu) == 4,
        "all-cold routes execute exactly once on CPU");

  CHECK(RouteClassification::build({0, 1, 2, 3}, weights,
                                   {0, -1, -1, 1}, {-1, 0, 1, -1}, 9,
                                   &classification),
        "mixed classification builds");
  CHECK(classification.exact_once() &&
            classification.count(RouteTarget::ResidentGpu) == 2 &&
            classification.count(RouteTarget::StagedGpu) == 2,
        "mixed target counts");
  for (const auto &assignment : classification.assignments())
    CHECK(assignment.slot_generation == 9, "generation is carried by every slot");

  CHECK(RouteClassification::build({1, 1, 2, 1}, weights,
                                   {-1, 4, 5, -1}, {-1, -1, -1, -1}, 10,
                                   &classification),
        "duplicate-route classification builds");
  CHECK(classification.exact_once() && classification.size() == 4 &&
            classification.assignments()[0].original_route_slot == 0 &&
            classification.assignments()[1].original_route_slot == 1 &&
            classification.assignments()[3].original_route_slot == 3 &&
            classification.count(RouteTarget::ResidentGpu) == 4,
        "duplicate expert slots are preserved independently");

  const std::vector<int32_t> dynamic_ids = {0, 1, 2, 1};
  const std::vector<float> dynamic_weights = {0.1f, 0.2f, 0.3f, 0.4f};
  const std::vector<int32_t> dynamic_resident_index = {0, -1, -1};
  const std::vector<int32_t> dynamic_staged_index = {-1, 5, -1};
  CHECK(RouteClassification::build(dynamic_ids.data(), dynamic_weights.data(),
                                   dynamic_ids.size(),
                                   dynamic_resident_index,
                                   dynamic_staged_index, 11,
                                   &classification),
        "resident/staged/CPU classification builds without changing route slots");
  for (RouteTarget target : {RouteTarget::ResidentGpu, RouteTarget::StagedGpu,
                             RouteTarget::Cpu}) {
    CHECK(classification.write_branch_routes(target, 0, projected_ids.data(),
                                             projected_weights.data(), 4),
          "branch projection writes into preallocated route buffers");
    for (size_t slot = 0; slot < dynamic_ids.size(); ++slot) {
      const bool owns = classification.assignments()[slot].target == target;
      CHECK(projected_ids[slot] == (owns ? dynamic_ids[slot] : 0),
            "branch projection preserves owned IDs and masks others");
      CHECK(projected_weights[slot] == (owns ? dynamic_weights[slot] : 0.0f),
            "original router weight is contributed by exactly one branch");
    }
  }
  CHECK(classification.write_branch_routes(RouteTarget::Cpu, 0,
                                           projected_ids.data(),
                                           projected_weights.data(), 4, true),
        "CPU projection can preserve IDs for its immutable remap table");
  for (size_t slot = 0; slot < dynamic_ids.size(); ++slot) {
    const bool owns =
        classification.assignments()[slot].target == RouteTarget::Cpu;
    CHECK(projected_ids[slot] == dynamic_ids[slot],
          "preserved IDs retain the CPU remap table's negative-skip path");
    CHECK(projected_weights[slot] ==
              (owns ? dynamic_weights[slot] : 0.0f),
          "preserved-ID projection still weights exactly one branch");
  }

  const auto *classification_storage = classification.assignments().data();
  int32_t callback_ids[4] = {};
  float callback_weights[4] = {};
  g_tracked_allocations = 0;
  g_track_allocations = true;
  const bool allocation_free_callback_path =
      RouteClassification::build(
          dynamic_ids.data(), dynamic_weights.data(), dynamic_ids.size(),
          dynamic_resident_index, dynamic_staged_index, 12, &classification) &&
      classification.exact_once() &&
      classification.write_branch_routes(RouteTarget::Cpu, 0, callback_ids,
                                          callback_weights, 4, true);
  g_track_allocations = false;
  CHECK(allocation_free_callback_path,
        "reserved route classification and CPU projection succeed");
  CHECK(g_tracked_allocations == 0,
        "reserved classify/validate/project performs no callback allocation");
  CHECK(classification.assignments().data() == classification_storage,
        "classification reuses its immutable assignment storage");

  CHECK(!RouteClassification::build({0, 1}, {0.5f, 0.5f},
                                    {0, 1}, {3, -1}, 11,
                                    &classification),
        "overlapping resident and staged ownership is rejected");
  CHECK(!RouteClassification::build({0, 4}, {0.5f, 0.5f},
                                    {0, 1}, {-1, -1}, 12,
                                    &classification),
        "out-of-range route ID is rejected");
  CHECK(classification.size() == 4 && classification.exact_once(),
        "rejected input leaves the prior immutable snapshot intact");

  if (g_failures == 0)
    std::printf("PASS (%s): inactive id=%d for decode, 0 for prefill\n", mode, want_inactive);
  return g_failures == 0 ? 0 : 1;
}
