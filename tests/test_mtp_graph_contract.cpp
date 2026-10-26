#include "mtp_graph_contract.hpp"

#include <cstdio>
#include <limits>
#include <vector>

int main() {
  using namespace zeromoe::graph;
  int failures = 0;
  const auto check = [&](bool ok, const char *name) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", name); ++failures; }
  };
  MtpGraphCursor cursor({2, 6, 4, 40});
  std::vector<float> embedding(2, 1), hidden(6, 2), next(6), logits(4);
  MtpGraphRequest r{0, embedding, hidden, next, logits};
  check(cursor.validate(r).empty(), "distinct embedding/hidden widths");
  auto short_request = r;
  short_request.hidden = std::span(hidden).first(2);
  check(!cursor.validate(short_request).empty(), "reject collapsed hidden");
  auto overlap = r;
  overlap.logits = std::span(next).first(4);
  check(!cursor.validate(overlap).empty(), "reject overlapping outputs");
  check(cursor.next_position() == 0 && !cursor.poisoned(),
        "validation does not advance or poison");
  r.position = 1;
  check(!cursor.validate(r).empty(), "reject gaps");
  r.position = -1;
  check(!cursor.validate(r).empty(), "reject negative positions");
  r.position = 0;
  embedding[0] = std::numeric_limits<float>::infinity();
  check(!cursor.validate(r).empty(), "reject non-finite input");
  embedding[0] = 1;
  cursor.poison();
  check(!cursor.validate(r).empty(), "reject poisoned session");
  cursor.reset();
  for (int p = 0; p < 40; ++p) {
    r.position = p;
    check(cursor.validate(r).empty(), "sequential replay");
    cursor.commit();
  }
  r.position = 40;
  check(!cursor.validate(r).empty(), "reject context overflow");
  cursor.reset();
  r.position = 0;
  check(cursor.validate(r).empty(), "reset restores position zero");
  MtpGraphCursor invalid({0, 6, 4, 40});
  check(!invalid.validate(r).empty(), "reject invalid graph shape");
  if (!failures) std::puts("MTP graph contract: PASS");
  return failures ? 1 : 0;
}
