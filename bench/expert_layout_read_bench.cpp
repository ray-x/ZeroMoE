// Expert read cost by GGUF layout, O_DIRECT, random experts, N threads:
//   packed   : the standard blk.L.ffn_{gate,up,down}_exps.weight tensors, 3 aligned reads per expert (gate, up, down planes)
//   embedded : offload.blk.* tight per-expert blocks written by `gguf_tool pack --expert-interleave`, ONE preadv per expert
// build: c++ -std=c++20 -O2 -pthread -Iinclude -o expert_layout_read_bench bench/expert_layout_read_bench.cpp
// run:   expert_layout_read_bench <model dir with shards> [samples=3000] [threads=1,4,8]
#include "offload_store.hpp"
#include <algorithm>
#include <chrono>
#include <map>
#include <random>
#include <thread>
using namespace zeromoe::offload;
using clk = std::chrono::steady_clock;
static double secs(clk::time_point a, clk::time_point b) { return std::chrono::duration<double>(b - a).count(); }
int main(int argc, char **argv) {
  if (argc < 2) { fprintf(stderr, "usage: %s <model dir> [samples] [threads,...]\n", argv[0]); return 2; }
  const std::string dir = argv[1];
  const int samples = argc > 2 ? atoi(argv[2]) : 3000;
  std::vector<int> thread_list;
  { std::string t = argc > 3 ? argv[3] : "1,4,8"; size_t p = 0; while (p < t.size()) { size_t q = t.find(',', p); if (q == std::string::npos) q = t.size(); thread_list.push_back(atoi(t.substr(p, q - p).c_str())); p = q + 1; } }
  std::vector<std::string> files;
  for (const auto &e : std::filesystem::directory_iterator(dir)) if (e.path().extension() == ".gguf") files.push_back(e.path().string());
  std::sort(files.begin(), files.end());
  OffloadStore emb;
  std::string err;
  if (!emb.open_embedded(files, &err)) { fprintf(stderr, "open_embedded: %s\n", err.c_str()); return 1; }
  // packed layout: (file, offset, per-expert bytes) of every projection of every layer
  struct Proj { int fd; uint64_t base, per; };
  std::map<int, std::array<Proj, 3>> packed;
  int n_exp = 0;
  for (const std::string &f : files) {
    GgufHeader h;
    if (!read_gguf_header(f, h, &err)) { fprintf(stderr, "%s\n", err.c_str()); return 1; }
    const int fd = ::open(f.c_str(), O_RDONLY | O_DIRECT);
    for (const auto &t : h.tensors) {
      int L = 0;
      char kind[8] = {0};
      if (std::sscanf(t.name.c_str(), "blk.%d.ffn_%7[a-z]_exps.weight", &L, kind) != 2) continue;
      const int k = !strcmp(kind, "gate") ? 0 : !strcmp(kind, "up") ? 1 : !strcmp(kind, "down") ? 2 : -1;
      if (k < 0 || t.dims.size() < 3) continue;
      n_exp = static_cast<int>(t.dims[2]);
      packed[L][static_cast<size_t>(k)] = {fd, h.data_start + t.offset, ggml_row_bytes(t.type, t.dims[0]) * t.dims[1]};
    }
  }
  std::vector<int> layers;
  for (auto &[L, p] : packed) if (emb.contains(L, 0)) layers.push_back(L);
  printf("model %s: %zu shards, %zu layers, %d experts/layer, embedded blocks %zu, O_DIRECT embedded=%d\n", dir.c_str(), files.size(), layers.size(),
         n_exp, emb.size(), emb.direct() ? 1 : 0);
  std::mt19937 rng(7);
  std::vector<std::pair<int, int>> picks(static_cast<size_t>(samples));
  for (auto &p : picks) p = {layers[rng() % layers.size()], static_cast<int>(rng() % static_cast<unsigned>(n_exp))};
  size_t bytes_total = 0;
  for (auto &p : picks) { uint64_t pb[3]; emb.plane_bytes(p.first, p.second, pb); bytes_total += pb[0] + pb[1] + pb[2]; }
  printf("%-8s %-9s %10s %10s %12s\n", "threads", "layout", "experts/s", "MB/s", "mean ms/exp");
  for (int th : thread_list) {
    for (int rep = 0; rep < 4; ++rep) {
      const bool use_emb = rep & 1;
      std::vector<double> lat(static_cast<size_t>(th), 0);
      std::vector<std::thread> ts;
      const auto t0 = clk::now();
      for (int t = 0; t < th; ++t) ts.emplace_back([&, t] {
        std::vector<void *> bufs(3);
        for (auto &b : bufs) { if (posix_memalign(&b, 4096, 8 << 20)) std::abort(); }
        double acc = 0; int cnt = 0;
        for (size_t i = static_cast<size_t>(t); i < picks.size(); i += static_cast<size_t>(th), ++cnt) {
          const auto [L, E] = picks[i];
          const auto a = clk::now();
          if (use_emb) {
            void *dst[3] = {bufs[0], bufs[1], bufs[2]};
            if (emb.read_expert(L, E, dst) != ReadStatus::Ok) std::abort();
          } else {
            for (size_t k = 0; k < 3; ++k) {
              const Proj &p = packed[L][k];
              if (pread(p.fd, bufs[k], p.per, static_cast<off_t>(p.base + static_cast<uint64_t>(E) * p.per)) != static_cast<ssize_t>(p.per)) std::abort();
            }
          }
          acc += secs(a, clk::now());
        }
        lat[static_cast<size_t>(t)] = acc / std::max(cnt, 1);
        for (auto *b : bufs) std::free(b);
      });
      for (auto &x : ts) x.join();
      const double dt = secs(t0, clk::now());
      double ml = 0; for (double v : lat) ml += v; ml /= th;
      printf("%-8d %-9s %10.0f %10.0f %12.2f\n", th, use_emb ? "embedded" : "packed", samples / dt, bytes_total / dt / 1e6, ml * 1e3);
    }
  }
  return 0;
}
