// Microbenchmark 1: does pinned DMA of expert-sized slots into the GPU interfere with the CPU cold-expert kernel?
// Microbenchmark 2: host-memory footprint of cudaHostAlloc vs cudaHostRegister on a sparse virtual region.
//   pinned_dma_vs_cpu_bench interf <cpu_threads> <pool_gib> <mode>   mode = cpu | dma | both | bounce | pageable
//   pinned_dma_vs_cpu_bench mem <mode> <n_experts> [thp]             mode = alloc | reg_slot | reg_whole | none
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <cuda_runtime.h>
#include <sys/mman.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>
static double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)

static double mem_used_gb() {
  FILE *f = fopen("/proc/meminfo", "r"); char l[256]; double tot = 0, av = 0;
  while (fgets(l, sizeof l, f)) { long v; if (sscanf(l, "MemTotal: %ld", &v) == 1) tot = v; if (sscanf(l, "MemAvailable: %ld", &v) == 1) av = v; }
  fclose(f); return (tot - av) / 1048576.0;
}
static double rss_gb() {
  FILE *f = fopen("/proc/self/status", "r"); char l[256]; double r = 0;
  while (fgets(l, sizeof l, f)) { long v; if (sscanf(l, "VmRSS: %ld", &v) == 1) r = v; }
  fclose(f); return r / 1048576.0;
}

// GLM-like expert: D=4096, F=1536, IQ2_XXS, top-8 all cold.
static const int D = 4096, F = 1536, NE = 64, TOPK = 8;
struct CpuKernel {
  ggml_backend_t be; ggml_context *ctx; ggml_cgraph *gr; ggml_backend_buffer_t buf; ggml_tensor *ids; double bytes_per_expert;
  void init(int nth) {
    be = ggml_backend_cpu_init(); ggml_backend_cpu_set_n_threads(be, nth);
    ggml_init_params ip = {ggml_tensor_overhead() * 64 + ggml_graph_overhead_custom(64, false) + 4096, nullptr, true};
    ctx = ggml_init(ip); ggml_type t = GGML_TYPE_IQ2_XXS;
    ggml_tensor *Wg = ggml_new_tensor_3d(ctx, t, D, F, NE), *Wu = ggml_new_tensor_3d(ctx, t, D, F, NE), *Wd = ggml_new_tensor_3d(ctx, t, F, D, NE);
    ggml_tensor *x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, 1, 1); ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, TOPK, 1);
    ggml_tensor *g = ggml_mul_mat_id(ctx, Wg, x, ids), *u = ggml_mul_mat_id(ctx, Wu, x, ids);
    ggml_tensor *d = ggml_mul_mat_id(ctx, Wd, ggml_swiglu_split(ctx, g, u), ids);
    gr = ggml_new_graph_custom(ctx, 64, false); ggml_build_forward_expand(gr, d);
    buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    ggml_tensor *W[3] = {Wg, Wu, Wd};
    for (int i = 0; i < 3; i++) {
      std::vector<float> row(W[i]->ne[0]); std::vector<uint8_t> tmp(ggml_row_size(t, W[i]->ne[0])); std::vector<float> im(W[i]->ne[0], 1.f);
      for (int k = 0; k < (int)row.size(); k++) row[k] = 0.01f * ((k * 7919) % 13 - 6);
      ggml_quantize_chunk(t, row.data(), tmp.data(), 0, 1, W[i]->ne[0], im.data());
      for (size_t r = 0; r < ggml_nrows(W[i]); r++) ggml_backend_tensor_set(W[i], tmp.data(), r * tmp.size(), tmp.size());
    }
    std::vector<float> xv(D); for (int k = 0; k < D; k++) xv[k] = 0.02f * ((k * 31) % 11 - 5);
    ggml_backend_tensor_set(x, xv.data(), 0, D * 4);
    bytes_per_expert = (double)(ggml_nbytes(Wg) + ggml_nbytes(Wu) + ggml_nbytes(Wd)) / NE;
  }
  double run_us(int it) {
    int32_t iv[TOPK]; for (int i = 0; i < TOPK; i++) iv[i] = (it * 17 + i * 7 + 3) % NE;
    ggml_backend_tensor_set(ids, iv, 0, sizeof iv);
    double t0 = now_s(); ggml_backend_graph_compute(be, gr); return (now_s() - t0) * 1e6;
  }
};

static int interf(int nth, double pool_gib, std::string mode) {
  CpuKernel k; k.init(nth);
  const size_t slot = (size_t)k.bytes_per_expert;          // one whole expert (gate+up+down)
  const size_t nslots = (size_t)(pool_gib * 1073741824.0) / slot;
  uint8_t *pool = nullptr;
  if (mode == "pageable" || mode == "bounce") { pool = (uint8_t *)aligned_alloc(4096, nslots * slot); }
  else CK(cudaHostAlloc((void **)&pool, nslots * slot, cudaHostAllocPortable));
  memset(pool, 0x5a, nslots * slot);
  const int K = 3;                                          // slots per "layer" copy, like staging K
  uint8_t *dev; CK(cudaMalloc(&dev, slot * K * 2));
  uint8_t *bounce = nullptr; CK(cudaHostAlloc((void **)&bounce, slot * K, cudaHostAllocPortable));
  cudaStream_t st; CK(cudaStreamCreate(&st));
  std::atomic<bool> stop{false}; std::atomic<double> dma_bytes{0}; std::atomic<double> dma_t{0};
  auto dma = [&]() {
    std::mt19937 rng(1); double bytes = 0, t0 = now_s();
    while (!stop.load()) {
      for (int j = 0; j < K; j++) {
        const uint8_t *src = pool + (rng() % nslots) * slot;
        if (mode == "bounce") { memcpy(bounce + j * slot, src, slot); CK(cudaMemcpyAsync(dev + j * slot, bounce + j * slot, slot, cudaMemcpyHostToDevice, st)); }
        else CK(cudaMemcpyAsync(dev + j * slot, src, slot, cudaMemcpyHostToDevice, st));
        bytes += slot;
      }
      CK(cudaStreamSynchronize(st));
    }
    dma_bytes = bytes; dma_t = now_s() - t0;
  };
  std::thread th;
  const bool do_dma = mode != "cpu";
  const bool do_cpu = mode != "dma";
  if (do_dma) th = std::thread(dma);
  std::vector<double> us;
  const double tend = now_s() + 8.0;
  if (do_cpu) { for (int w = 0; w < 20; w++) k.run_us(w); for (int it = 0; now_s() < tend; it++) us.push_back(k.run_us(it + 100)); }
  else std::this_thread::sleep_for(std::chrono::seconds(8));
  stop = true; if (th.joinable()) th.join();
  double mean = 0; for (double v : us) mean += v; if (!us.empty()) mean /= us.size();
  std::sort(us.begin(), us.end());
  printf("mode=%-8s threads=%d slot=%.2fMB K=%d | cpu layer(8 experts) mean=%.0f us p50=%.0f p95=%.0f n=%zu | dma %.2f GB/s\n", mode.c_str(), nth, slot / 1e6, K, mean,
         us.empty() ? 0 : us[us.size() / 2], us.empty() ? 0 : us[us.size() * 95 / 100], us.size(), dma_t > 0 ? dma_bytes / dma_t / 1e9 : 0.0);
  return 0;
}

static int memtest(std::string mode, int n_experts, bool thp) {
  // Sparse virtual arena (like VirtualArena): slot index = layer*288+expert, 3 planes of ~1.6 MB, plane stride rounded up to 512 B only.
  const size_t plane = 1638912, slot = 3 * plane;            // not 4 KiB aligned on purpose (as in the GGUF-derived planes)
  const size_t nvirt = 42ull * 288;                          // 12096 slots ~ 59 GB virtual
  const size_t total = nvirt * slot;
  double u0 = mem_used_gb();
  uint8_t *base = nullptr;
  if (mode == "alloc") { CK(cudaHostAlloc((void **)&base, (size_t)n_experts * slot, cudaHostAllocPortable)); }
  else {
    base = (uint8_t *)mmap(nullptr, total, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (thp) madvise(base, total, MADV_HUGEPAGE); else madvise(base, total, MADV_NOHUGEPAGE);
    if (mode == "reg_whole") { CK(cudaHostRegister(base, total, cudaHostRegisterPortable)); }
  }
  std::mt19937 rng(7); std::vector<size_t> pick;
  if (mode == "alloc") for (int i = 0; i < n_experts; i++) pick.push_back(i);
  else { std::vector<size_t> all(nvirt); for (size_t i = 0; i < nvirt; i++) all[i] = i; std::shuffle(all.begin(), all.end(), rng); pick.assign(all.begin(), all.begin() + n_experts); }
  for (size_t s : pick) {
    uint8_t *p = (mode == "alloc" ? base + s * slot : base + s * slot);
    memset(p, 0x11, slot);
    if (mode == "reg_slot") for (int k = 0; k < 3; k++) CK(cudaHostRegister(p + k * plane, plane, cudaHostRegisterPortable));
  }
  double u1 = mem_used_gb();
  printf("mode=%-9s thp=%d experts=%d data=%.2f GB | system used delta=%.2f GB, process RSS=%.2f GB\n", mode.c_str(), (int)thp, n_experts, (double)n_experts * slot / 1e9, u1 - u0, rss_gb());
  return 0;
}

int main(int argc, char **argv) {
  if (argc >= 5 && !strcmp(argv[1], "interf")) return interf(atoi(argv[2]), atof(argv[3]), argv[4]);
  if (argc >= 4 && !strcmp(argv[1], "mem")) return memtest(argv[2], atoi(argv[3]), argc > 4 && !strcmp(argv[4], "thp"));
  fprintf(stderr, "usage: interf <threads> <pool_gib> <cpu|dma|both|bounce|pageable> | mem <alloc|reg_slot|reg_whole|none> <n_experts> [thp]\n");
  return 2;
}
