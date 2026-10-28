// CPU-only proof of how expert_promotion resolves (no CUDA, no ggml).
//
//   c++ -std=c++23 -O2 -Icore tests/test_expert_promotion_mode.cpp -o /tmp/zeromoe-expert-promotion-mode

#include "expert_promotion_mode.hpp"

#include <cstdio>

using zeromoe::graph::expert_promotion_mode_valid;
using zeromoe::graph::resolve_expert_promotion_mode;

namespace {
int failures = 0;

void expect_level(const char *mode, bool mtp, int want) {
  const int got = resolve_expert_promotion_mode(mode, mtp);
  if (got != want) {
    std::fprintf(stderr, "FAIL resolve(\"%s\", mtp=%d) = %d, want %d\n", mode, mtp, got, want);
    ++failures;
  }
}

void expect_valid(const char *mode, bool want) {
  if (expert_promotion_mode_valid(mode) != want) {
    std::fprintf(stderr, "FAIL valid(\"%s\") != %d\n", mode, want);
    ++failures;
  }
}
}  // namespace

int main() {
  // "auto" is the default: promotion follows MTP (the speculative path is where it pays most and is measured).
  expect_level("auto", true, 2);
  expect_level("auto", false, 0);
  // An explicit setting always wins over MTP in both directions.
  expect_level("on", false, 2);
  expect_level("on", true, 2);
  expect_level("off", true, 0);
  expect_level("off", false, 0);
  expect_level("observe", true, 1);
  expect_level("observe", false, 1);
  // Unknown values (only reachable through the environment override) mean off, never on.
  expect_level("bogus", true, 0);
  expect_level("", true, 0);

  expect_valid("auto", true);
  expect_valid("on", true);
  expect_valid("observe", true);
  expect_valid("off", true);
  expect_valid("", false);
  expect_valid("true", false);
  expect_valid("ON", false);

  std::printf("expert promotion mode: %s\n", failures ? "FAIL" : "PASS");
  return failures ? 1 : 0;
}
