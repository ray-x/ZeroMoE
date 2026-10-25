// Compare two LLAMA_NODE_DUMP files (CPU vs GPU runs of ling_llama_logits): per-node relative L2, first offenders.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <map>
#include <string>
#include <vector>
struct Rec { std::string name, op; std::vector<float> v; };
static bool rd(FILE *f, Rec &r) {
  uint32_t n;
  if (fread(&n, 4, 1, f) != 1) return false;
  r.name.resize(n); fread(r.name.data(), 1, n, f);
  fread(&n, 4, 1, f); r.op.resize(n); fread(r.op.data(), 1, n, f);
  fread(&n, 4, 1, f); r.v.resize(n); return fread(r.v.data(), 4, n, f) == n;
}
int main(int argc, char **argv) {
  if (argc < 3) return 2;
  FILE *a = fopen(argv[1], "rb"), *b = fopen(argv[2], "rb");
  if (!a || !b) return 1;
  Rec x, y; int i = 0, shown = 0, big = 0; std::map<std::string, std::pair<int,int>> perop;
  while (rd(a, x) && rd(b, y)) {
    if (x.name != y.name || x.op != y.op || x.v.size() != y.v.size()) { printf("graph mismatch at %d: %s/%s vs %s/%s\n", i, x.name.c_str(), x.op.c_str(), y.name.c_str(), y.op.c_str()); break; }
    double d = 0, n = 0;
    for (size_t k = 0; k < x.v.size(); ++k) { double e = double(x.v[k]) - y.v[k]; d += e * e; n += double(x.v[k]) * x.v[k]; }
    const double rel = n > 0 ? std::sqrt(d / n) : (d == 0 ? 0 : 9);
    auto &p = perop[x.op]; p.second++; if (rel > 1e-2) p.first++;
    if (rel > 1e-3 && shown < 40) { printf("#%d %-28s %-14s n=%zu rel=%.3e\n", i, x.name.c_str(), x.op.c_str(), x.v.size(), rel); ++shown; }
    if (rel > 0.3 && big < 12) { printf("BIG #%d %-28s %-14s n=%zu rel=%.3e\n", i, x.name.c_str(), x.op.c_str(), x.v.size(), rel); ++big; }
    ++i;
  }
  printf("%d nodes\nop: nodes with rel>1e-2 / total\n", i);
  for (auto &[op, p] : perop) if (p.first) printf("  %-16s %d / %d\n", op.c_str(), p.first, p.second);
}
