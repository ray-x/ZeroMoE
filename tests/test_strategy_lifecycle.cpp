// Unit test for models/graph/invocation_state.hpp — the shared per-invocation lifecycle
// (§3.8). Pure C++ (no ggml, no model, no GPU); tests the state machine that both
// EarlyReadyLaneStrategy and CacheBackedStrategy delegate to.
//
//   g++ -std=c++20 -I models/graph -I backends/ggml tests/test_strategy_lifecycle.cpp
//       -o /tmp/test_strategy_lifecycle
//   /tmp/test_strategy_lifecycle
//
// Coverage (the reviewer's P2-3 regression list):
//   * stale generation          -> Error, never Ready
//   * wrong layer               -> Error, never Ready
//   * residency InProgress/Ready-> Pending/Ready  (NOT `completed`)
//   * residency Failed          -> Error (load error)
//   * release is idempotent and a no-op once the generation advances
//   * `completed` is deliberately absent — Ready is derived from residency only

#include "invocation_state.hpp"

#include <cstdio>

static int g_failures = 0;
#define CHECK(cond, msg)                                                       \
  do {                                                                         \
    if (!(cond)) {                                                             \
      std::fprintf(stderr, "FAIL: %s  [%s:%d]\n", msg, __FILE__, __LINE__);    \
      ++g_failures;                                                            \
    }                                                                          \
  } while (0)

using moestream::graph::InvocationState;
using State = InvocationState::State;
using Residency = InvocationState::Residency;

int main() {
  InvocationState g;

  // ---- no active invocation: anything is Error -----------------------------------------------
  CHECK(g.inactive(), "fresh gate has no active invocation");
  CHECK(g.state(5, 3, Residency::InProgress) == State::Error, "no begin -> Error");
  CHECK(g.state(5, 3, Residency::Ready) == State::Error, "no begin -> Error (even Ready)");

  // ---- begin + matching identity -------------------------------------------------------------
  g.begin(5, 3);
  CHECK(!g.inactive(), "begin() records an active invocation");
  CHECK(g.matches(5, 3), "matches the recorded (layer, generation)");
  CHECK(!g.matches(5, 4), "does not match a different generation");
  CHECK(!g.matches(6, 3), "does not match a different layer");
  CHECK(g.state(5, 3, Residency::InProgress) == State::Pending, "residency in progress -> Pending");
  CHECK(g.state(5, 3, Residency::Ready) == State::Ready, "residency ready -> Ready");
  CHECK(g.state(5, 3, Residency::Failed) == State::Error, "residency failed -> Error (load error)");

  // ---- stale generation -> Error, never Ready ------------------------------------------------
  CHECK(g.state(5, 4, Residency::Ready) == State::Error, "stale generation -> Error, not Ready");

  // ---- wrong layer -> Error, never Ready -----------------------------------------------------
  CHECK(g.state(6, 3, Residency::Ready) == State::Error, "wrong layer -> Error, not Ready");

  // ---- finish(): Ready only when identity matches AND residency is Ready ----------------------
  CHECK(g.finish(5, 3, Residency::Ready) == true, "finish Ready");
  CHECK(g.finish(5, 3, Residency::InProgress) == false, "finish blocks while Pending");
  CHECK(g.finish(5, 3, Residency::Failed) == false, "finish fails on load error");
  CHECK(g.finish(5, 4, Residency::Ready) == false, "finish fails on stale generation");
  CHECK(g.finish(6, 3, Residency::Ready) == false, "finish fails on wrong layer");

  // ---- release: idempotent, and a no-op once the generation advances -------------------------
  g.release(5, 3);
  CHECK(g.inactive(), "release clears the active invocation");
  g.release(5, 3);  // idempotent: already released
  CHECK(g.inactive(), "release is idempotent");

  g.begin(5, 3);
  g.begin(5, 4);  // generation advanced: the lane moved on
  g.release(5, 3);  // stale release must not clear the gen-4 identity
  CHECK(!g.inactive(), "stale release is a no-op");
  CHECK(g.matches(5, 4), "gen-4 identity survives a stale gen-3 release");
  g.release(5, 4);
  CHECK(g.inactive(), "matching release clears gen-4");

  // ---- identity is stable until release: layer/generation accessors --------------------------
  g.begin(7, 11);
  CHECK(g.layer() == 7 && g.generation() == 11, "accessors reflect the active identity");
  g.release(7, 11);

  if (g_failures == 0)
    std::printf("PASS: invocation lifecycle (stale/wrong-layer/load-error/release)\n");
  return g_failures == 0 ? 0 : 1;
}
