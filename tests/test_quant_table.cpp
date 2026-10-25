// Every row of memory/quant_table.hpp against the linked ggml's own type traits.
#include "quant_table.hpp"
#include "ggml.h"
#include <cstdio>
#include <initializer_list>
#include <cstring>
#include <strings.h>

static int failures = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); ++failures; } } while (0)

int main() {
  using namespace zeromoe::quant;
  for (const QuantInfo &q : kQuantTable) {
    const ggml_type t = static_cast<ggml_type>(q.type);
    const char *gname = ggml_type_name(t);
    if (!gname || std::strcmp(gname, "NONE") == 0) {
      std::fprintf(stderr, "type %u (%s) is not known to the linked ggml\n", q.type, q.name);
      ++failures;
      continue;
    }
    if (strcasecmp(gname, q.name) != 0)
      std::fprintf(stderr, "note: type %u is %s in ggml, %s here\n", q.type, gname, q.name);
    CHECK(ggml_blck_size(t) == (int64_t)q.block);
    CHECK(ggml_type_size(t) == q.bytes);
    for (uint64_t ne0 : {uint64_t(q.block), uint64_t(q.block) * 7, uint64_t(q.block) * 64}) {
      CHECK(row_bytes(q.type, ne0) == ggml_row_size(t, (int64_t)ne0));
      CHECK(tensor_bytes(q.type, ne0 * 3) == ggml_row_size(t, (int64_t)ne0) * 3);
    }
  }
  CHECK(row_bytes(4242, 256) == 0);        // unknown: offload store semantics
  CHECK(tensor_bytes(4242, 256) == 256);   // unknown: GGUF loader fallback
  CHECK(takes_mmq(39) && takes_mmq(16) && takes_mmq(14));    // MXFP4, IQ2_XXS, Q6_K
  CHECK(!takes_mmq(29) && !takes_mmq(0) && !takes_mmq(1) && !takes_mmq(30)); // IQ1_M, F32, F16, BF16
  CHECK(!takes_mmq(4242));
  if (failures) { std::fprintf(stderr, "%d failure(s)\n", failures); return 1; }
  std::printf("quant_table_test: OK (%zu types)\n", sizeof(kQuantTable) / sizeof(kQuantTable[0]));
  return 0;
}
