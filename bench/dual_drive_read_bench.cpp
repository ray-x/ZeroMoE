// Aggregate expert read throughput when the store is spread over two drives (':'-separated paths), random experts, N threads, O_DIRECT.
// run: dual_drive_read_bench <pathA[:pathB]> [samples=3000] [threads=1,4,8]   (experts are read from whichever drive holds them)
#include "offload_store.hpp"
#include <algorithm>
#include <chrono>
#include <random>
#include <thread>
using namespace moestream::offload;
using clk = std::chrono::steady_clock;
int main(int argc, char **argv) {
  if (argc < 2) return 2;
  OffloadStore st; std::string err;
  if (!st.open_auto(argv[1], &err)) { fprintf(stderr, "open: %s\n", err.c_str()); return 1; }
  const int samples = argc > 2 ? atoi(argv[2]) : 3000;
  std::vector<int> ths; { std::string t = argc > 3 ? argv[3] : "1,4,8"; size_t p = 0; while (p < t.size()) { size_t q = t.find(',', p); if (q == std::string::npos) q = t.size(); ths.push_back(atoi(t.substr(p, q - p).c_str())); p = q + 1; } }
  auto keys = st.keys();
  std::mt19937 rng(3);
  std::vector<ExpertKey> picks(static_cast<size_t>(samples));
  size_t bytes = 0;
  for (auto &k : picks) { k = keys[rng() % keys.size()]; uint64_t pb[3]; st.plane_bytes(k.layer, k.expert, pb); bytes += pb[0] + pb[1] + pb[2]; }
  printf("%zu experts, direct=%d\n%-8s %10s %10s %12s\n", keys.size(), st.direct() ? 1 : 0, "threads", "experts/s", "MB/s", "mean ms/exp");
  for (int th : ths) for (int rep = 0; rep < 2; ++rep) {
    std::vector<std::thread> ts; std::vector<double> lat(static_cast<size_t>(th), 0);
    const auto t0 = clk::now();
    for (int t = 0; t < th; ++t) ts.emplace_back([&, t] {
      void *b[3]; for (auto &x : b) if (posix_memalign(&x, 4096, 8 << 20)) std::abort();
      double acc = 0; int n = 0;
      for (size_t i = static_cast<size_t>(t); i < picks.size(); i += static_cast<size_t>(th), ++n) {
        const auto a = clk::now();
        if (st.read_expert(picks[i].layer, picks[i].expert, b) != ReadStatus::Ok) std::abort();
        acc += std::chrono::duration<double>(clk::now() - a).count();
      }
      lat[static_cast<size_t>(t)] = acc / std::max(n, 1);
      for (auto *x : b) std::free(x);
    });
    for (auto &x : ts) x.join();
    const double dt = std::chrono::duration<double>(clk::now() - t0).count();
    double ml = 0; for (double v : lat) ml += v; ml /= th;
    printf("%-8d %10.0f %10.0f %12.2f\n", th, samples / dt, bytes / dt / 1e6, ml * 1e3);
  }
}
