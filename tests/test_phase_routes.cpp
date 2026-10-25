#include "phase_route_partition.hpp"
#include "placement_epoch.hpp"
#include <cmath>
#include <cstdio>

static int failures = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); ++failures; } } while (0)

int main() {
  {
    zeromoe::memory::PlacementEpoch epoch;
    CHECK(epoch.generation() == 0 && !epoch.borrow());
    epoch.publish();
    auto graph = epoch.borrow();
    CHECK(epoch.generation() == 1 && epoch.matches(graph));
    CHECK(!epoch.retire() && epoch.generation() == 1);
    graph.reset();
    CHECK(epoch.retire() && epoch.generation() == 2);
    epoch.publish();
    auto rebuilt = epoch.borrow();
    CHECK(*rebuilt == 3 && epoch.matches(rebuilt));
    rebuilt.reset();
    CHECK(epoch.retire() && epoch.generation() == 4);
    CHECK(epoch.retire() && epoch.generation() == 4); // idempotent prefill entry
  }
  using zeromoe::memory::claim_phase_routes;
  for (int inactive : {-1, 0}) {
    // Global identities and compact cold indices deliberately differ.
    std::vector<int32_t> remap = {inactive, 0, 1, 2};
    std::vector<float> cold = {0, 1, 1, 1};
    claim_phase_routes({2}, remap, cold, inactive);
    CHECK(remap[1] == 0 && remap[2] == inactive && remap[3] == 2);
    const std::vector<float> permanent = {1, 0, 0, 0}, phase = {0, 0, 1, 0};
    for (int e = 0; e < 4; ++e) CHECK(permanent[e] + cold[e] + phase[e] == 1.0f);
    // Original route order and unequal weights, including repeated route
    // slots, must give the same sum without renormalizing any branch.
    const int ids[] = {3, 0, 2, 2};
    const float weights[] = {0.1f, 0.2f, 0.3f, 0.4f};
    float expected = 0, actual = 0;
    for (int i = 0; i < 4; ++i) {
      const float value = float(ids[i] * 7 + 3);
      expected += weights[i] * value;
      actual += weights[i] * value * (permanent[ids[i]] + cold[ids[i]] + phase[ids[i]]);
    }
    CHECK(std::abs(actual - expected) < 1e-6f);
  }
  for (const auto &bad : std::vector<std::vector<int>>{{0}, {-1}, {4}, {1, 1}, {1, 4}}) {
    std::vector<int32_t> remap = {-1, 0, 1, 2};
    std::vector<float> cold = {0, 1, 1, 1};
    const auto before = cold;
    bool rejected = false;
    try { claim_phase_routes(bad, remap, cold, -1); }
    catch (const std::invalid_argument &) { rejected = true; }
    CHECK(rejected && cold == before && remap[1] == 0);
  }
  if (failures) return 1;
  std::puts("phase_route_test: OK");
}
