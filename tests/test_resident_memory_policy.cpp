#include "resident_memory_policy.hpp"
#include <cstdio>

using namespace zeromoe::memory;
static int failures = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); ++failures; } } while (0)

int main() {
  constexpr size_t GiB = size_t(1) << 30;
  CHECK(valid_embedding_device("auto") && valid_embedding_device("cpu") && valid_embedding_device("gpu"));
  CHECK(!valid_embedding_device("GPU") && !valid_embedding_device("typo"));
  CHECK(!embedding_on_gpu("auto", false));
  CHECK(embedding_on_gpu("auto", true));
  CHECK(!embedding_on_gpu("cpu", true));
  CHECK(embedding_on_gpu("gpu", false));

  // Dedicated vault: the configured reserve includes context memory.
  CHECK(expert_cache_budget(24*GiB, 23*GiB, 12*GiB, GiB, false).bytes == 12*GiB);
  CHECK(expert_cache_budget(24*GiB, 23*GiB, 12*GiB, GiB, true).bytes == 23*GiB);
  // Allocations outside this engine cannot be ignored by an auto-fit budget.
  const auto occupied = expert_cache_budget(24*GiB, 8*GiB, 12*GiB, GiB, true);
  CHECK(occupied.bytes == 8*GiB && occupied.clamped);
  // Dual-role GPU: budget is the actual free memory after dense weights.
  CHECK(expert_cache_budget(18*GiB, 18*GiB, 15*GiB, 2*GiB, true).bytes == 16*GiB);
  CHECK(expert_cache_budget(18*GiB, 18*GiB, 20*GiB, GiB, false).bytes == 17*GiB);
  // A reserve larger than VRAM used to yield 0, which the vault then treated
  // as the legacy auto sentinel and allocated almost all VRAM instead.
  CHECK(expert_cache_budget(8*GiB, 7*GiB, 4*GiB, 10*GiB, true).bytes == 0);
  CHECK(expert_cache_budget(8*GiB, 7*GiB, 0, GiB, false, true).bytes == 0);
  CHECK(expert_cache_budget(8*GiB, 7*GiB, 0, GiB, false, false).bytes == 7*GiB);
  CHECK(expert_cache_budget(0, 0, 0, GiB, true).bytes == 0);
  if (failures) return 1;
  std::puts("resident_memory_policy_test: OK");
}
