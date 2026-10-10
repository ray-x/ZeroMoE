// Expert-transfer microbenchmark: H2D bandwidth, 3 DMA vs 1 DMA per expert, and the full SSD->pinned->VRAM path
// with the original GGUF layout (3 O_DIRECT reads + 3 DMAs) vs the repacked sidecar (1 read + 1 DMA).
// build: nvcc -O2 -std=c++17 -o qwen_expert_dma_bench qwen_expert_dma_bench.cu -lpthread
// run:   qwen_expert_dma_bench <src.gguf> <sidecar.bin> <index.tsv> [samples=1500]
// index.tsv line: layer base stride E raw n0 n1 n2 o0 o1 o2   (n = bytes/expert per part, o = part tensor offset in src)
#include <cuda_runtime.h>
#include <fcntl.h>
#include <unistd.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <sstream>
#include <thread>
#include <vector>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)
using clk = std::chrono::steady_clock;
static double now() { return std::chrono::duration<double>(clk::now().time_since_epoch()).count(); }
static size_t up4k(size_t n) { return (n + 4095) / 4096 * 4096; }
struct Lay { int layer; long base, stride, E, raw, n[3], o[3]; };

static void h2d_curve() {
  printf("\n[A] H2D bandwidth, one DMA of the given size (async, 200 DMAs/stream-sync)\n");
  printf("%-10s %14s %14s %16s %16s\n", "size", "pinned GB/s", "pageable GB/s", "pinned us/DMA", "sync-each us");
  void *dev; CK(cudaMalloc(&dev, 64u << 20));
  uint8_t *pin; CK(cudaMallocHost((void **)&pin, 64u << 20)); memset(pin, 1, 64u << 20);
  std::vector<uint8_t> page(64u << 20, 2);
  cudaStream_t s; CK(cudaStreamCreate(&s));
  for (size_t kb : {64, 256, 460, 1024, 1536, 2048, 4096, 16384}) {
    size_t n = kb * 1024; int iters = 200;
    auto run = [&](const uint8_t *src, bool sync_each) {
      for (int w = 0; w < 5; ++w) CK(cudaMemcpyAsync(dev, src, n, cudaMemcpyHostToDevice, s));
      CK(cudaStreamSynchronize(s));
      double t0 = now();
      for (int i = 0; i < iters; ++i) {
        CK(cudaMemcpyAsync(dev, src, n, cudaMemcpyHostToDevice, s));
        if (sync_each) CK(cudaStreamSynchronize(s));
      }
      CK(cudaStreamSynchronize(s));
      return (now() - t0) / iters;
    };
    double tp = run(pin, false), tg = run(page.data(), false), ts = run(pin, true);
    printf("%-10zu %14.2f %14.2f %16.1f %16.1f\n", kb, n / tp / 1e9, n / tg / 1e9, tp * 1e6, ts * 1e6);
  }
  cudaFree(dev); cudaFreeHost(pin);
}

static void dma_variants(const Lay &L) {
  const int N = 512; const size_t exp_bytes = L.stride;
  printf("\n[B] per-expert transfer, layer %d: parts %.2f/%.2f/%.2f MB, %d experts per measurement\n", L.layer,
         L.n[0] / 1e6, L.n[1] / 1e6, L.n[2] / 1e6, N);
  printf("%-34s %12s %10s %14s %14s\n", "variant", "experts/s", "GB/s", "pipelined us", "3-exp batch us");
  uint8_t *pin; CK(cudaMallocHost((void **)&pin, exp_bytes * N)); memset(pin, 3, exp_bytes * N);
  uint8_t *bounce; CK(cudaMallocHost((void **)&bounce, exp_bytes));
  std::vector<uint8_t> arena(exp_bytes * N, 4);
  uint8_t *dev; CK(cudaMalloc(&dev, exp_bytes * 8));
  cudaStream_t s; CK(cudaStreamCreate(&s));
  const long row[3] = {std::max<long>(L.n[0] / 640, 1), std::max<long>(L.n[1] / 640, 1), std::max<long>(L.n[2] / 2560, 1)};
  auto go = [&](const char *name, auto &&one) {
    for (int i = 0; i < 16; ++i) one(i, i & 7);
    CK(cudaStreamSynchronize(s));
    double t0 = now();
    for (int i = 0; i < N; ++i) one(i, i & 7);
    CK(cudaStreamSynchronize(s));
    double dt = (now() - t0) / N;
    double t1 = now(); int B = 0;
    for (int b = 0; b + 3 <= N; b += 3, ++B) { for (int k = 0; k < 3; ++k) one(b + k, k); CK(cudaStreamSynchronize(s)); }
    double db = (now() - t1) / B;
    printf("%-34s %12.0f %10.2f %14.1f %14.1f\n", name, 1 / dt, L.raw / dt / 1e9, dt * 1e6, db * 1e6);
  };
  go("3 DMA from packed pinned", [&](int i, int slot) {
    uint8_t *h = pin + (size_t)i * exp_bytes; uint8_t *d = dev + (size_t)slot * exp_bytes; size_t o = 0;
    for (int p = 0; p < 3; ++p) { CK(cudaMemcpyAsync(d + o, h + o, L.n[p], cudaMemcpyHostToDevice, s)); o += L.n[p]; }
  });
  go("1 DMA from packed pinned", [&](int i, int slot) {
    CK(cudaMemcpyAsync(dev + (size_t)slot * exp_bytes, pin + (size_t)i * exp_bytes, L.raw, cudaMemcpyHostToDevice, s));
  });
  go("1 DMA from packed pageable", [&](int i, int slot) {
    CK(cudaMemcpyAsync(dev + (size_t)slot * exp_bytes, arena.data() + (size_t)i * exp_bytes, L.raw, cudaMemcpyHostToDevice, s));
  });
  // Engine-like: row-by-row CPU copy of each plane pageable arena -> pinned bounce, then one DMA per plane.
  // The bounce is single-buffered, so each expert waits for the previous DMAs before it is rewritten.
  go("engine-like: row memcpy + 3 DMA", [&](int i, int slot) {
    CK(cudaStreamSynchronize(s));
    const uint8_t *src = arena.data() + (size_t)i * exp_bytes; size_t o = 0;
    for (int p = 0; p < 3; ++p) {
      for (long r = 0; r + row[p] <= L.n[p]; r += row[p]) memcpy(bounce + o + r, src + o + r, row[p]);
      CK(cudaMemcpyAsync(dev + (size_t)slot * exp_bytes + o, bounce + o, L.n[p], cudaMemcpyHostToDevice, s));
      o += L.n[p];
    }
  });
  cudaFree(dev); cudaFreeHost(pin); cudaFreeHost(bounce);
}

static void full_path(const char *src, const char *side, const std::vector<Lay> &lay, int samples) {
  printf("\n[C] disk(O_DIRECT) -> pinned -> VRAM, %d random experts, %zu layers; each thread: read, DMA, sync\n",
         samples, lay.size());
  printf("%-8s %-10s %10s %10s %14s\n", "threads", "layout", "experts/s", "MB/s", "mean ms/exp");
  int fs = open(src, O_RDONLY | O_DIRECT), fr = open(side, O_RDONLY | O_DIRECT);
  if (fs < 0 || fr < 0) { perror("open O_DIRECT"); exit(1); }
  std::mt19937 rng(1);
  std::vector<std::pair<int, int>> picks(samples);
  for (auto &p : picks) p = {(int)(rng() % lay.size()), (int)(rng() % lay[0].E)};
  size_t useful = 0; for (auto &p : picks) useful += lay[p.first].raw;
  size_t maxb = 0; for (auto &l : lay) maxb = std::max(maxb, up4k(l.stride) + 3 * 8192);
  for (int th : {1, 2, 4, 8}) {
    for (int rep = 0; rep < 4; ++rep) {
      bool repack = rep & 1;
      std::vector<std::thread> ts; std::vector<double> lat(th, 0);
      double t0 = now();
      for (int t = 0; t < th; ++t) ts.emplace_back([&, t] {
        uint8_t *pin; CK(cudaMallocHost((void **)&pin, maxb * 3)); uint8_t *dev; CK(cudaMalloc(&dev, maxb * 3));
        cudaStream_t s; CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
        double acc = 0; int cnt = 0;
        for (size_t i = t; i < picks.size(); i += th, ++cnt) {
          const Lay &L = lay[picks[i].first]; long e = picks[i].second; double a0 = now();
          if (repack) {
            ssize_t r = pread(fr, pin, L.stride, L.base + e * L.stride);
            if (r != (ssize_t)L.stride) { perror("pread"); exit(1); }
            CK(cudaMemcpyAsync(dev, pin, L.raw, cudaMemcpyHostToDevice, s));
          } else {
            size_t o = 0;
            for (int p = 0; p < 3; ++p) {
              long a = L.o[p] + e * L.n[p], lo = a / 4096 * 4096; size_t ln = up4k(a + L.n[p] - lo);
              ssize_t r = pread(fs, pin + p * maxb, ln, lo);
              if (r < (ssize_t)(a + L.n[p] - lo)) { perror("pread"); exit(1); }
              CK(cudaMemcpyAsync(dev + o, pin + p * maxb + (a - lo), L.n[p], cudaMemcpyHostToDevice, s));
              o += L.n[p];
            }
          }
          CK(cudaStreamSynchronize(s)); acc += now() - a0;
        }
        lat[t] = acc / std::max(cnt, 1);
        cudaFree(dev); cudaFreeHost(pin); cudaStreamDestroy(s);
      });
      for (auto &x : ts) x.join();
      double dt = now() - t0, ml = 0; for (double v : lat) ml += v; ml /= th;
      printf("%-8d %-10s %10.0f %10.0f %14.2f\n", th, repack ? "repack" : "orig", samples / dt, useful / dt / 1e6, ml * 1e3);
    }
  }
}

int main(int argc, char **argv) {
  if (argc < 4) { fprintf(stderr, "usage: %s src.gguf sidecar.bin index.tsv [samples]\n", argv[0]); return 1; }
  std::vector<Lay> lay; std::ifstream f(argv[3]); std::string line;
  while (std::getline(f, line)) {
    std::istringstream is(line); Lay l;
    if (is >> l.layer >> l.base >> l.stride >> l.E >> l.raw >> l.n[0] >> l.n[1] >> l.n[2] >> l.o[0] >> l.o[1] >> l.o[2]) lay.push_back(l);
  }
  int dev = 0; CK(cudaGetDevice(&dev)); cudaDeviceProp pr; CK(cudaGetDeviceProperties(&pr, dev));
  printf("device %d: %s | %zu layers\n", dev, pr.name, lay.size());
  h2d_curve();
  dma_variants(lay[0]);
  full_path(argv[1], argv[2], lay, argc > 4 ? atoi(argv[4]) : 1500);
  return 0;
}
