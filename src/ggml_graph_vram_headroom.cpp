// Free-VRAM floor check shared by the GLM5Next and DeepSeek-V4.1 full-graph builders
// (it used to live in ggml_deepseek_v4_graph.cpp, which was retired with DeepSeek-V4 graph support).

#include "glm5next_graph_internal.hpp"

#if defined(MOESTREAM_USE_GGML_CUDA)

namespace glm5next_graph {

// Real, measured incident (2026-09-09, qwen4exp side, same ggml scheduler
// code path this file also uses): ggml_backend_sched_alloc_graph's internal
// ggml_gallocr_reserve_n_impl does not fail gracefully on a real cudaMalloc
// OOM -- it logs "cudaMalloc failed: out of memory" and then segfaults the
// whole process instead of returning a checkable failure. That happened when
// main.cpp's fallback VRAM cache reservation (a fixed 4GB, unaware of
// device_expert_layers' own VRAM use) left the scheduler's own reserve()
// short by ~1.8GB. main.cpp now sizes that fallback cache against real free
// VRAM and this family's own device_expert_layers estimate, but this check
// stays as a second, independent line of defense: refuse to even attempt the
// allocator call when real free VRAM on the graph's device is below a
// generous floor, and decline to the legacy path instead of risking the same
// crash again.
bool graph_alloc_vram_headroom_ok(const FullGraph &g, const char *log_tag) {
  size_t free_b = 0, total_b = 0;
  int prev_device = -1;
  cudaGetDevice(&prev_device);
  cudaSetDevice(g.device);
  const cudaError_t err = cudaMemGetInfo(&free_b, &total_b);
  if (prev_device >= 0)
    cudaSetDevice(prev_device);
  size_t kMinGraphAllocHeadroomBytes =
      2ull * 1024ull * 1024ull * 1024ull; // 2 GiB
  if (const char *e = moestream::graphcfg::env("MOESTREAM_GLM5NEXT_ALLOC_FLOOR_MIB"))
    kMinGraphAllocHeadroomBytes = (size_t)std::max(0, std::atoi(e)) << 20;
  if (err != cudaSuccess || free_b < kMinGraphAllocHeadroomBytes) {
    LOG_ERROR("%s refusing graph allocator reserve: device %d free=%.2f GiB "
              "< %.2f GiB safety floor (expert arena + dense arena already "
              "committed %.2f GiB) -- falling back to the legacy path\n",
              log_tag, g.device,
              err == cudaSuccess
                  ? static_cast<double>(free_b) / (1024.0 * 1024.0 * 1024.0)
                  : -1.0,
              static_cast<double>(kMinGraphAllocHeadroomBytes) /
                  (1024.0 * 1024.0 * 1024.0),
              static_cast<double>(g.expert.capacity + g.dense.capacity) /
                  (1024.0 * 1024.0 * 1024.0));
    return false;
  }
  return true;
}

}  // namespace glm5next_graph

#endif
