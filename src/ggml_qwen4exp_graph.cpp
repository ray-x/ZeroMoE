// Real multi-layer Qwen4Exp GGML graph.
//
// This is intentionally a separate bridge rather than a rewrite of the
// established execution path.  Llama.cpp's important property is that the
// C++ caller constructs one dependency graph for the entire trunk and then
// gives it to ggml_backend_sched_graph_compute_async().  The scheduler may
// split that graph internally around host-backed expert weights, but the
// caller does not return to C++ after every layer.  This file reproduces that
// shape for MoEStream's model representation.

#include "ggml_qwen4exp_graph.hpp"

#include "direct_io.hpp"
#include "engine.hpp"
#include "ggml_moe_backend.hpp"
#include "kquant_dequant.hpp"
#include "logger.hpp"
#include "memory_guard.hpp"
#include "ple_hash.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#if defined(MOESTREAM_USE_GGML_CUDA)

#include <cuda_runtime.h>

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

namespace {

constexpr int kGraphSize = 50000;
constexpr size_t kTensorContextBytes = 128ull * 1024ull * 1024ull;

struct HostAllocation {
  void *ptr = nullptr;
  size_t bytes = 0;
  bool pinned = false;
  size_t reservation_bytes = 0;
};

struct FullGraph {
  StreamingEngine *engine = nullptr;
  int device = -1;
  int S = 0;
  int D = 0;
  int HC = 0;
  int n_layers = 0;
  int H = 0;
  int Hkv = 0;
  int hd = 0;
  int Hk_gdn = 0;
  int Hv_gdn = 0;
  int gdn_hd = 0;
  int key_dim = 0;
  int value_dim = 0;
  int conv_dim = 0;
  int d_conv = 0;

  ggml_backend_t cuda = nullptr;
  ggml_backend_t cpu = nullptr;
  ggml_backend_sched_t sched = nullptr;
  ggml_backend_buffer_t device_expert_buffer = nullptr;
  void *device_expert_base = nullptr;
  size_t device_expert_capacity = 0;
  size_t device_expert_used = 0;
  ggml_backend_buffer_t device_dense_buffer = nullptr;
  void *device_dense_base = nullptr;
  size_t device_dense_capacity = 0;
  size_t device_dense_used = 0;
  int device_dense_tensors = 0;
  int device_expert_layers = 0;
  bool device_expert_partial_gate_up = false;
  ggml_context *ctx = nullptr;
  ggml_cgraph *graph = nullptr;

  ggml_tensor *input = nullptr;         // [D, HC, S], CPU input
  ggml_tensor *output = nullptr;        // [D, S], final HC output
  ggml_tensor *final_streams = nullptr; // [D, HC, S], for state/debug writeback
  ggml_tensor *positions = nullptr;     // [S], I32
  ggml_tensor *attn_mask = nullptr;     // [S, S], F16 (flash-attn contract)
  ggml_tensor *ple_embedding = nullptr;
  ggml_tensor *ple_hist_out = nullptr;

  std::vector<ggml_tensor *> kv_k;
  std::vector<ggml_tensor *> kv_v;
  std::vector<int> kv_layers;
  std::vector<ggml_tensor *> gdn_state_out;
  std::vector<ggml_tensor *> gdn_hist_out;
  std::vector<int> gdn_layers;
  std::vector<float *> gdn_state_inputs;
  std::vector<float *> gdn_hist_inputs;
  int ple_layer = -1;
  float *ple_hist_input = nullptr;

  std::vector<HostAllocation> owned;
  std::vector<ggml_backend_buffer_t> host_buffers;
  std::vector<void *> host_inputs;

  size_t graph_bytes = 0;
  size_t host_weight_bytes = 0;
  // Bytes/views borrowed from StreamingEngine's already-populated graph-RAM
  // arenas.  These are not owned by this graph and must not be counted as a
  // second host-side expert copy.
  size_t host_ram_alias_bytes = 0;
  size_t host_ram_alias_tensors = 0;
  size_t host_pinned_bytes = 0;
  size_t host_pin_failures = 0;
  size_t host_reservation_bytes = 0;
  // ggml_init(no_alloc=true) still allocates a host-side context arena. Keep
  // that arena in the process-wide admission ledger as well, otherwise graph
  // construction can exceed host_ram_gb between explicit host-buffer
  // allocations.
  size_t context_reservation_bytes = 0;
  // The source admission is made once before graph construction, while this
  // graph materializes that source as many individual host allocations.  This
  // remainder prevents alloc_host() from charging those same bytes twice.
  size_t host_precharged_bytes = 0;
  bool ready = false;

  ~FullGraph() { destroy(); }

  void destroy() {
    if (sched) {
      // qwen4exp_full_graph_run synchronizes before returning.  Avoid calling
      // into the borrowed CUDA backend during process teardown: the shared
      // backend can already be in CUDA runtime shutdown at this point.
      ggml_backend_sched_free(sched);
      sched = nullptr;
    }
    if (cpu) {
      ggml_backend_free(cpu);
      cpu = nullptr;
    }
    if (device_expert_buffer) {
      ggml_backend_buffer_free(device_expert_buffer);
      device_expert_buffer = nullptr;
    }
    device_expert_base = nullptr;
    device_expert_capacity = 0;
    device_expert_used = 0;
    if (device_dense_buffer) {
      ggml_backend_buffer_free(device_dense_buffer);
      device_dense_buffer = nullptr;
    }
    device_dense_base = nullptr;
    device_dense_capacity = 0;
    device_dense_used = 0;
    device_dense_tensors = 0;
    device_expert_layers = 0;
    device_expert_partial_gate_up = false;
    for (ggml_backend_buffer_t b : host_buffers) {
      if (b)
        ggml_backend_buffer_free(b);
    }
    host_buffers.clear();
    if (ctx) {
      ggml_free(ctx);
      ctx = nullptr;
    }
    for (HostAllocation &a : owned) {
      if (a.pinned)
        cudaHostUnregister(a.ptr);
      if (a.ptr)
        std::free(a.ptr);
      MemoryGuard::release_host_bytes(a.reservation_bytes);
      a.ptr = nullptr;
      a.pinned = false;
    }
    owned.clear();
    host_inputs.clear();
    cuda = nullptr;
    cpu = nullptr;
    graph = nullptr;
    input = output = final_streams = positions = attn_mask = ple_embedding =
        nullptr;
    ple_hist_out = nullptr;
    kv_k.clear();
    kv_v.clear();
    kv_layers.clear();
    gdn_state_out.clear();
    gdn_hist_out.clear();
    gdn_layers.clear();
    gdn_state_inputs.clear();
    gdn_hist_inputs.clear();
    ple_layer = -1;
    ple_hist_input = nullptr;
    ready = false;
    graph_bytes = 0;
    host_weight_bytes = 0;
    host_ram_alias_bytes = 0;
    host_ram_alias_tensors = 0;
    host_pinned_bytes = 0;
    host_pin_failures = 0;
    MemoryGuard::release_host_bytes(host_reservation_bytes);
    host_reservation_bytes = 0;
    MemoryGuard::release_host_bytes(context_reservation_bytes);
    context_reservation_bytes = 0;
    host_precharged_bytes = 0;
    engine = nullptr;
    S = 0;
  }
};

std::unique_ptr<FullGraph> g_graph;

bool env_enabled() {
  static const bool enabled = [] {
    const char *v = std::getenv("MOESTREAM_QWEN4EXP_FULL_GRAPH");
    return v && v[0] == '1';
  }();
  return enabled;
}

int env_device_expert_layers() {
  const char *v = std::getenv("MOESTREAM_QWEN4EXP_GRAPH_PRELOAD_LAYERS");
  if (!v || !v[0])
    return 0;
  char *end = nullptr;
  long n = std::strtol(v, &end, 10);
  if (end == v || *end != '\0' || n <= 0)
    return 0;
  return static_cast<int>(std::min<long>(n, 48));
}

bool has_ple_layer(const ArchitectureConfig &c) {
  return !c.ple_layers.empty() && c.ple_n_heads > 0 &&
         c.ple_n_embd_per_layer > 0;
}

bool type_ok(GGUFType type) {
  return static_cast<uint32_t>(type) < static_cast<uint32_t>(GGML_TYPE_COUNT);
}

ggml_tensor *new_tensor(ggml_context *ctx, ggml_type type,
                        const std::vector<int64_t> &shape) {
  switch (shape.size()) {
  case 1:
    return ggml_new_tensor_1d(ctx, type, shape[0]);
  case 2:
    return ggml_new_tensor_2d(ctx, type, shape[0], shape[1]);
  case 3:
    return ggml_new_tensor_3d(ctx, type, shape[0], shape[1], shape[2]);
  case 4:
    return ggml_new_tensor_4d(ctx, type, shape[0], shape[1], shape[2],
                              shape[3]);
  default:
    return nullptr;
  }
}

ggml_backend_buffer_t bind_host(FullGraph &g, ggml_tensor *t, void *ptr,
                                size_t bytes, ggml_backend_buffer_usage usage) {
  if (!t || !ptr || bytes < ggml_nbytes(t))
    return nullptr;
  ggml_backend_buffer_t b = ggml_backend_cpu_buffer_from_ptr(ptr, bytes);
  if (!b)
    return nullptr;
  ggml_backend_buffer_set_usage(b, usage);
  if (ggml_backend_tensor_alloc(b, t, ptr) != GGML_STATUS_SUCCESS) {
    ggml_backend_buffer_free(b);
    return nullptr;
  }
  g.host_buffers.push_back(b);
  return b;
}

void *alloc_host(FullGraph &g, size_t bytes) {
  void *p = nullptr;
  if (bytes == 0)
    return nullptr;
  size_t reservation = 0;
  if (g.host_precharged_bytes >= bytes) {
    g.host_precharged_bytes -= bytes;
  } else {
    reservation = bytes - g.host_precharged_bytes;
    g.host_precharged_bytes = 0;
    if (!MemoryGuard::try_reserve_host_bytes(reservation)) {
      LOG_WARN("[GGML-QWEN-FULL-MEM] host graph allocation refused: %.2f MB "
               "(reserved=%.2f GB, budget=%.2f GB)\n",
               bytes / 1e6, MemoryGuard::reserved_host_bytes() / 1e9,
               MemoryGuard::host_budget_bytes() / 1e9);
      return nullptr;
    }
  }
  const int err = posix_memalign(&p, 4096, bytes);
  if (err != 0 || !p) {
    MemoryGuard::release_host_bytes(reservation);
    return nullptr;
  }
  g.owned.push_back({p, bytes, false, reservation});
  return p;
}

ggml_tensor *host_input(FullGraph &g, ggml_context *ctx, ggml_type type,
                        const std::vector<int64_t> &shape, size_t bytes,
                        const char *name) {
  ggml_tensor *t = new_tensor(ctx, type, shape);
  if (!t)
    return nullptr;
  void *p = alloc_host(g, std::max(bytes, ggml_nbytes(t)));
  if (!p || !bind_host(g, t, p, std::max(bytes, ggml_nbytes(t)),
                       GGML_BACKEND_BUFFER_USAGE_COMPUTE))
    return nullptr;
  ggml_set_input(t);
  ggml_set_name(t, name);
  g.host_inputs.push_back(p);
  return t;
}

const QuantizedTensor *resident(const StreamingEngine &engine,
                                const std::string &name) {
  auto it = engine.resident_layers.find(name);
  if (it == engine.resident_layers.end() || !it->second.host_ptr)
    return nullptr;
  return &it->second;
}

bool graph_dense_weight_name(const std::string &name);

ggml_tensor *resident_tensor(
    FullGraph &g, ggml_context *ctx, const std::string &name,
    ggml_backend_buffer_usage usage = GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
  const QuantizedTensor *q = resident(*g.engine, name);
  if (!q || !type_ok(q->meta.type))
    return nullptr;
  std::vector<int64_t> shape;
  shape.reserve(q->meta.shape.size());
  for (uint64_t d : q->meta.shape)
    shape.push_back(static_cast<int64_t>(d));
  ggml_tensor *t = new_tensor(ctx, static_cast<ggml_type>(q->meta.type), shape);
  if (!t || q->meta.size_bytes < ggml_nbytes(t))
    return nullptr;
  if (g.device_dense_buffer && usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
      graph_dense_weight_name(name)) {
    const ggml_backend_buffer_type_t buft =
        ggml_backend_buffer_get_type(g.device_dense_buffer);
    const size_t alloc_size = ggml_backend_buft_get_alloc_size(buft, t);
    const size_t alignment =
        std::max<size_t>(256, ggml_backend_buft_get_alignment(buft));
    const size_t aligned =
        (g.device_dense_used + alignment - 1) & ~(alignment - 1);
    if (aligned > g.device_dense_capacity ||
        alloc_size > g.device_dense_capacity - aligned)
      return nullptr;
    void *addr = static_cast<uint8_t *>(g.device_dense_base) + aligned;
    if (ggml_backend_tensor_alloc(g.device_dense_buffer, t, addr) !=
        GGML_STATUS_SUCCESS)
      return nullptr;
    const cudaError_t err = cudaMemcpy(addr, q->host_ptr, q->meta.size_bytes,
                                       cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
      LOG_ERROR("[GGML-QWEN-FULL] dense device upload failed for %s: %s\n",
                name.c_str(), cudaGetErrorString(err));
      return nullptr;
    }
    g.device_dense_used = aligned + alloc_size;
    ++g.device_dense_tensors;
  } else if (!bind_host(g, t, q->host_ptr, q->meta.size_bytes, usage)) {
    return nullptr;
  }
  ggml_set_name(t, name.c_str());
  return t;
}

bool graph_dense_weight_name(const std::string &name) {
  // Keep this list aligned with the names consumed by build_graph(). Routed
  // *_exps tensors have their own residency policy; indexer/NextN tensors are
  // not part of this v1 graph, and absorbed-MLA tensors are unused here.
  if (name.find("_exps.weight") != std::string::npos ||
      name.find("indexer") != std::string::npos ||
      name.find("nextn") != std::string::npos ||
      name.find("attn_k_b.weight") != std::string::npos ||
      name.find("attn_v_b.weight") != std::string::npos ||
      name.find("attn_output_a.weight") != std::string::npos)
    return false;
  return name.find("attn") != std::string::npos ||
         name.find("hc_") != std::string::npos ||
         name.find("ssm_") != std::string::npos ||
         name.find("ple_") != std::string::npos ||
         name.find("norm") != std::string::npos ||
         name.find("gate_inp") != std::string::npos ||
         name.find("shexp") != std::string::npos ||
         name.find("ffn_gate.weight") != std::string::npos ||
         name.find("ffn_up.weight") != std::string::npos ||
         name.find("ffn_down.weight") != std::string::npos ||
         name.find("exp_probs_b.bias") != std::string::npos ||
         name == "output_hc_norm.weight" || name == "output_hc_down.weight" ||
         name == "output_hc_up.weight";
}

bool init_device_dense_arena(FullGraph &g) {
  size_t bytes = 0;
  int tensors = 0;
  for (const auto &[name, q] : g.engine->resident_layers) {
    if (!q.host_ptr || !graph_dense_weight_name(name))
      continue;
    constexpr size_t kTensorSlack = 4096;
    if (bytes > SIZE_MAX - q.meta.size_bytes - kTensorSlack)
      return false;
    bytes += q.meta.size_bytes + kTensorSlack;
    ++tensors;
  }
  if (bytes == 0)
    return false;
  constexpr size_t kArenaSlack = 1ull << 20;
  if (bytes > SIZE_MAX - kArenaSlack)
    return false;
  const size_t capacity = bytes + kArenaSlack;
  const ggml_backend_buffer_type_t buft =
      ggml_backend_get_default_buffer_type(g.cuda);
  g.device_dense_buffer = ggml_backend_buft_alloc_buffer(buft, capacity);
  if (!g.device_dense_buffer) {
    LOG_WARN(
        "[GGML-QWEN-FULL] dense device arena allocation declined: %.2f GiB\n",
        static_cast<double>(capacity) / (1024.0 * 1024.0 * 1024.0));
    return false;
  }
  ggml_backend_buffer_set_usage(g.device_dense_buffer,
                                GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
  g.device_dense_base = ggml_backend_buffer_get_base(g.device_dense_buffer);
  g.device_dense_capacity = ggml_backend_buffer_get_size(g.device_dense_buffer);
  g.device_dense_used = 0;
  g.device_dense_tensors = 0;
  LOG_INFO("[GGML-QWEN-FULL] dense device arena=%.2f GiB for %d graph weight "
           "tensors\n",
           static_cast<double>(g.device_dense_capacity) /
               (1024.0 * 1024.0 * 1024.0),
           tensors);
  return true;
}

bool graph_expert_projection_resident(const FullGraph &g, int layer,
                                      const char *stem) {
  if (layer >= 0 && layer < g.device_expert_layers)
    return true;
  return g.device_expert_partial_gate_up && layer == g.device_expert_layers &&
         std::strcmp(stem, "ffn_down_exps.weight") != 0;
}

const GGUFTensor *model_tensor(const StreamingEngine &engine,
                               const std::string &name) {
  auto it = engine.model_file.tensors.find(name);
  return it == engine.model_file.tensors.end() ? nullptr : &it->second;
}

size_t full_expert_source_bytes(const StreamingEngine &engine) {
  const auto &c = engine.model_file.architecture_config;
  size_t total = 0;
  for (int layer = 0; layer < c.n_layers; ++layer) {
    // When main.cpp prepared a complete graph-RAM arena, the three packed
    // expert planes for this layer are already resident in DDR.  The full
    // graph aliases those bytes below; do not pre-charge or allocate a second
    // graph-owned source set for the same layer.
    if (engine.graph_ram_layer_ready(layer))
      continue;
    for (const char *stem : {"ffn_gate_exps.weight", "ffn_up_exps.weight",
                             "ffn_down_exps.weight"}) {
      const GGUFTensor *m =
          model_tensor(engine, "blk." + std::to_string(layer) + "." + stem);
      if (!m || m->size_bytes == 0)
        return 0;
      if (static_cast<size_t>(m->size_bytes) >
          std::numeric_limits<size_t>::max() - total)
        return 0;
      total += static_cast<size_t>(m->size_bytes);
    }
  }
  return total;
}

// Resolve one packed Qwen expert projection to the complete, preloaded
// graph-RAM arena.  This is deliberately Qwen-specific: the arena layout is
// [gate|up|down] per expert with a fixed expert stride, whereas other model
// families have different graph/cache ownership rules.
bool preloaded_ram_expert_source(FullGraph &g, const std::string &name,
                                 const GGUFTensor &meta, void *&ptr,
                                 size_t &bytes, size_t &expert_stride,
                                 size_t &arena_bytes) {
  const std::string prefix = "blk.";
  if (name.compare(0, prefix.size(), prefix) != 0)
    return false;

  char *end = nullptr;
  const long parsed_layer = std::strtol(name.c_str() + prefix.size(), &end, 10);
  if (end == name.c_str() + prefix.size() || parsed_layer < 0 || *end != '.')
    return false;
  const int layer = static_cast<int>(parsed_layer);
  const std::string suffix(end + 1);
  const auto *arena = g.engine->graph_ram_arena(layer);
  if (!arena)
    return false;
  size_t plane_offset = 0;
  size_t plane_bytes = 0;
  if (suffix == "ffn_gate_exps.weight") {
    plane_bytes = arena->gate_bytes;
  } else if (suffix == "ffn_up_exps.weight") {
    plane_offset = arena->gate_bytes;
    plane_bytes = arena->up_bytes;
  } else if (suffix == "ffn_down_exps.weight") {
    plane_offset = arena->gate_bytes + arena->up_bytes;
    plane_bytes = arena->down_bytes;
  } else {
    return false;
  }

  if (!g.engine->graph_ram_layer_ready(layer))
    return false;
  const int n_experts = g.engine->model_file.architecture_config.n_experts;
  const size_t n_exp = static_cast<size_t>(std::max(0, n_experts));
  if (!arena || !arena->base || n_experts <= 0 ||
      arena->n_experts != n_experts || plane_bytes == 0 ||
      plane_offset > arena->bytes ||
      plane_bytes > arena->bytes - plane_offset ||
      n_exp > SIZE_MAX / plane_bytes ||
      meta.size_bytes != plane_bytes * n_exp || arena->stride < plane_bytes ||
      static_cast<size_t>(n_experts - 1) >
          (SIZE_MAX - plane_bytes) / arena->stride)
    return false;

  const size_t view_bytes =
      arena->stride * static_cast<size_t>(n_experts - 1) + plane_bytes;
  if (plane_offset > arena->bytes || view_bytes > arena->bytes - plane_offset)
    return false;

  ptr = arena->base + plane_offset;
  bytes = view_bytes;
  expert_stride = arena->stride;
  arena_bytes = arena->bytes;
  return true;
}

size_t device_expert_bytes(const FullGraph &g) {
  const auto &engine = *g.engine;
  const auto &c = engine.model_file.architecture_config;
  const int limit = std::min(std::max(0, g.device_expert_layers), c.n_layers);
  size_t total = 0;
  for (int layer = 0; layer < limit; ++layer) {
    for (const char *stem : {"ffn_gate_exps.weight", "ffn_up_exps.weight",
                             "ffn_down_exps.weight"}) {
      const GGUFTensor *m =
          model_tensor(engine, "blk." + std::to_string(layer) + "." + stem);
      if (!m || m->size_bytes == 0 ||
          total > SIZE_MAX - static_cast<size_t>(m->size_bytes))
        return 0;
      total += static_cast<size_t>(m->size_bytes);
    }
  }
  if (g.device_expert_partial_gate_up && limit < c.n_layers) {
    const int layer = limit;
    for (const char *stem : {"ffn_gate_exps.weight", "ffn_up_exps.weight"}) {
      const GGUFTensor *m =
          model_tensor(engine, "blk." + std::to_string(layer) + "." + stem);
      if (!m || m->size_bytes == 0 ||
          total > SIZE_MAX - static_cast<size_t>(m->size_bytes))
        return 0;
      total += static_cast<size_t>(m->size_bytes);
    }
  }
  return total;
}

bool init_device_expert_arena(FullGraph &g) {
  if (g.device_expert_layers <= 0)
    return true;
  const size_t bytes = device_expert_bytes(g);
  if (bytes == 0)
    return false;
  constexpr size_t kAlignment = 256;
  const size_t tensor_slack =
      static_cast<size_t>(g.device_expert_layers) * 3 * kAlignment;
  constexpr size_t kArenaSlack = 1ull << 20;
  if (bytes > SIZE_MAX - tensor_slack - kArenaSlack)
    return false;
  const size_t capacity = bytes + tensor_slack + kArenaSlack;
  const ggml_backend_buffer_type_t buft =
      ggml_backend_get_default_buffer_type(g.cuda);
  g.device_expert_buffer = ggml_backend_buft_alloc_buffer(buft, capacity);
  if (!g.device_expert_buffer) {
    LOG_WARN(
        "[GGML-QWEN-FULL] device expert arena allocation declined: %.2f GiB\n",
        static_cast<double>(capacity) / (1024.0 * 1024.0 * 1024.0));
    return false;
  }
  ggml_backend_buffer_set_usage(g.device_expert_buffer,
                                GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
  g.device_expert_base = ggml_backend_buffer_get_base(g.device_expert_buffer);
  g.device_expert_capacity =
      ggml_backend_buffer_get_size(g.device_expert_buffer);
  g.device_expert_used = 0;
  LOG_INFO(
      "[GGML-QWEN-FULL] device expert arena=%.2f GiB for layers [0,%d)%s\n",
      static_cast<double>(g.device_expert_capacity) /
          (1024.0 * 1024.0 * 1024.0),
      g.device_expert_layers,
      g.device_expert_partial_gate_up ? " + next-layer gate/up" : "");
  return true;
}

bool bind_device_expert(FullGraph &g, ggml_tensor *t, const void *host,
                        size_t bytes) {
  if (!g.device_expert_buffer || !t || !host || bytes < ggml_nbytes(t))
    return false;
  const ggml_backend_buffer_type_t buft =
      ggml_backend_buffer_get_type(g.device_expert_buffer);
  const size_t alloc_size = ggml_backend_buft_get_alloc_size(buft, t);
  const size_t alignment =
      std::max<size_t>(256, ggml_backend_buft_get_alignment(buft));
  const size_t aligned =
      (g.device_expert_used + alignment - 1) & ~(alignment - 1);
  if (aligned > g.device_expert_capacity ||
      alloc_size > g.device_expert_capacity - aligned)
    return false;
  void *addr = static_cast<uint8_t *>(g.device_expert_base) + aligned;
  if (ggml_backend_tensor_alloc(g.device_expert_buffer, t, addr) !=
      GGML_STATUS_SUCCESS)
    return false;
  const cudaError_t err = cudaMemcpy(addr, host, bytes, cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    LOG_ERROR("[GGML-QWEN-FULL] device expert upload failed: %s\n",
              cudaGetErrorString(err));
    return false;
  }
  g.device_expert_used = aligned + alloc_size;
  return true;
}

ggml_tensor *load_full_weight(FullGraph &g, ggml_context *ctx,
                              const std::string &name,
                              bool device_resident = false) {
  const GGUFTensor *m = model_tensor(*g.engine, name);
  if (!m || !type_ok(m->type) || m->shape.size() < 2)
    return nullptr;
  std::vector<int64_t> shape;
  for (uint64_t d : m->shape)
    shape.push_back(static_cast<int64_t>(d));
  ggml_tensor *t = new_tensor(ctx, static_cast<ggml_type>(m->type), shape);
  if (!t)
    return nullptr;
  const size_t bytes = static_cast<size_t>(m->size_bytes);
  const size_t required = std::max(bytes, ggml_nbytes(t));

  // A Qwen full graph must consume the same complete DDR expert planes that
  // the RAM tier already loaded.  In that configuration the source is an
  // interleaved fixed-stride arena, so adjust only the expert-axis stride;
  // no selected weights are copied and no GGUF mapping is consulted.
  if (!device_resident) {
    void *ram_ptr = nullptr;
    size_t ram_bytes = 0;
    size_t ram_stride = 0;
    size_t ram_arena_bytes = 0;
    if (preloaded_ram_expert_source(g, name, *m, ram_ptr, ram_bytes, ram_stride,
                                    ram_arena_bytes)) {
      if (shape.size() < 3 ||
          t->ne[2] != g.engine->model_file.architecture_config.n_experts ||
          ram_stride < t->nb[2])
        return nullptr;
      t->nb[2] = ram_stride;
      t->nb[3] = ram_stride * static_cast<size_t>(t->ne[2]);
      if (!bind_host(g, t, ram_ptr, ram_bytes,
                     GGML_BACKEND_BUFFER_USAGE_WEIGHTS))
        return nullptr;
      // The three projection views overlap the same one-layer arena. Count
      // the backing arena once, on the gate view, rather than summing the
      // three strided view spans and reporting a fictitious 3x allocation.
      if (name.find("ffn_gate_exps.weight") != std::string::npos)
        g.host_ram_alias_bytes += ram_arena_bytes;
      ++g.host_ram_alias_tensors;
      ggml_set_name(t, name.c_str());
      return t;
    }

    // A graph-RAM arena is an explicit ownership decision: once it exists,
    // this Qwen layer is supposed to use the already-preloaded DDR bytes.
    // Do not quietly turn an incomplete arena into a second graph-owned
    // startup read from the GGUF file.  Rejecting the optional graph keeps
    // the normal RAM/cache dispatcher as the correctness fallback and makes
    // the source policy visible instead of hiding an SSD read behind this
    // path.  This guard is deliberately Qwen-full-graph-only; other model
    // families have their own graph source ownership rules.
    int graph_ram_layer = -1;
    if (name.compare(0, 4, "blk.") == 0) {
      char *end = nullptr;
      const long parsed = std::strtol(name.c_str() + 4, &end, 10);
      if (end != name.c_str() + 4 && end && *end == '.')
        graph_ram_layer = static_cast<int>(parsed);
    }
    if (graph_ram_layer >= 0 && g.engine->qwen_ram_preload_all_experts) {
      LOG_WARN("[GGML-QWEN-FULL] refusing file-backed expert source for "
               "layer=%d: ram_preload_all_experts requires the preloaded "
               "DDR arena\n",
               graph_ram_layer);
      return nullptr;
    }
    if (graph_ram_layer >= 0 &&
        g.engine->graph_ram_layer_allocated(graph_ram_layer) &&
        !g.engine->graph_ram_layer_ready(graph_ram_layer)) {
      LOG_WARN("[GGML-QWEN-FULL] refusing file-backed expert source for "
               "layer=%d: preloaded DDR arena is allocated but not ready\n",
               graph_ram_layer);
      return nullptr;
    }
  }

  void *p = alloc_host(g, required);
  if (!p)
    return nullptr;
  // Fallback for graph configurations without a complete preloaded RAM
  // arena. This is a startup copy into graph-owned DDR via pread; it is not
  // the decode-time CPU source used by the DDR-affinity path above.
  const int fd = g.engine->model_file.fd_for_tensor(*m);
  const ssize_t got = direct_io::pread_direct_aware(
      fd, g.engine->model_file.direct_io_active_for(*m),
      g.engine->model_file.direct_io_alignment, p, bytes, m->offset);
  if (got < 0 || static_cast<size_t>(got) != bytes) {
    LOG_ERROR("[GGML-QWEN-FULL] short read for %s: %zd/%zu\n", name.c_str(),
              got, bytes);
    return nullptr;
  }
  if (device_resident) {
    if (!bind_device_expert(g, t, p, bytes))
      return nullptr;
  } else if (!bind_host(g, t, p, required, GGML_BACKEND_BUFFER_USAGE_WEIGHTS)) {
    return nullptr;
  } else {
    // GGML's CUDA scheduler uses set_tensor_async for host-backed expert
    // weights. Register each graph-owned source once, after it has been
    // faulted in by the read above, so those copies can use DMA directly.
    // Registration is outside the measured prefill window and is best effort:
    // this machine has a finite aggregate CUDA host-registration ceiling.
    const size_t owned_index = g.owned.size() - 1;
    const cudaError_t pin_err =
        cudaHostRegister(p, required, cudaHostRegisterPortable);
    if (pin_err == cudaSuccess) {
      g.owned[owned_index].pinned = true;
      g.host_pinned_bytes += required;
    } else {
      ++g.host_pin_failures;
      cudaGetLastError();
      if (g.host_pin_failures == 1) {
        LOG_WARN(
            "[GGML-QWEN-FULL] host expert pinning declined at %.2f GiB: %s; "
            "remaining sources stay pageable\n",
            static_cast<double>(g.host_pinned_bytes) /
                (1024.0 * 1024.0 * 1024.0),
            cudaGetErrorString(pin_err));
      }
    }
  }
  g.host_weight_bytes += bytes;
  ggml_set_name(t, name.c_str());
  return t;
}

ggml_tensor *expert_weight(FullGraph &g, ggml_context *ctx, int layer,
                           const char *stem) {
  return load_full_weight(g, ctx, "blk." + std::to_string(layer) + "." + stem,
                          graph_expert_projection_resident(g, layer, stem));
}

ggml_tensor *dense_weight(FullGraph &g, ggml_context *ctx, int layer,
                          const char *stem) {
  return resident_tensor(g, ctx, "blk." + std::to_string(layer) + "." + stem);
}

ggml_tensor *hc_mix(FullGraph &g, ggml_context *ctx, ggml_tensor *x,
                    ggml_tensor *norm, ggml_tensor *down, ggml_tensor *up,
                    ggml_tensor *inject, ggml_tensor **inject_out) {
  ggml_tensor *xn =
      ggml_rms_norm(ctx, x, g.engine->model_file.architecture_config.eps);
  xn = ggml_reshape_2d(ctx, xn, g.D * g.HC, g.S);
  xn = ggml_mul(ctx, xn, norm);
  ggml_tensor *lo = ggml_mul_mat(ctx, down, xn);
  lo = ggml_silu(ctx, ggml_scale(ctx, lo, 1.0f / static_cast<float>(g.HC)));
  ggml_tensor *gate = ggml_sigmoid(ctx, ggml_mul_mat(ctx, up, lo));
  ggml_tensor *gated = ggml_mul(ctx, xn, gate);
  ggml_tensor *g3 = ggml_reshape_3d(ctx, gated, g.D, g.HC, g.S);

  ggml_tensor *mixed =
      ggml_cont(ctx, ggml_view_2d(ctx, g3, g.D, g.S,
                                  ggml_row_size(g3->type, g.D) * g.HC, 0));
  for (int h = 1; h < g.HC; ++h) {
    ggml_tensor *s =
        ggml_view_2d(ctx, g3, g.D, g.S, ggml_row_size(g3->type, g.D) * g.HC,
                     ggml_row_size(g3->type, g.D) * h);
    mixed = ggml_add(ctx, mixed, s);
  }
  mixed = ggml_scale(ctx, mixed, 1.0f / static_cast<float>(g.HC));
  if (inject_out)
    *inject_out = inject ? ggml_mul_mat(ctx, inject, xn) : nullptr;
  return mixed;
}

ggml_tensor *hc_combine(FullGraph &g, ggml_context *ctx, ggml_tensor *residual,
                        ggml_tensor *block, ggml_tensor *inject) {
  ggml_tensor *w = ggml_scale(
      ctx, ggml_sigmoid(ctx, ggml_scale(ctx, inject, 1.0f / g.HC)), 2.0f);
  w = ggml_reshape_3d(ctx, w, 1, g.HC, g.S);
  ggml_tensor *b = ggml_reshape_3d(ctx, block, g.D, 1, g.S);
  b = ggml_repeat_4d(ctx, b, g.D, g.HC, g.S, 1);
  return ggml_add(ctx, residual, ggml_mul(ctx, b, w));
}

ggml_tensor *build_moe(FullGraph &g, ggml_context *ctx, ggml_tensor *cur,
                       int layer, ggml_tensor *router,
                       ggml_tensor *selection_bias, ggml_tensor *gate_exps,
                       ggml_tensor *up_exps, ggml_tensor *down_exps) {
  const auto &c = g.engine->model_file.architecture_config;
  ggml_tensor *logits = ggml_mul_mat(ctx, router, cur);
  ggml_tensor *probs = ggml_soft_max(ctx, logits);
  ggml_tensor *selection =
      selection_bias ? ggml_add(ctx, probs, selection_bias) : probs;
  ggml_tensor *ids = ggml_argsort_top_k(ctx, selection, c.topk);
  ggml_tensor *probs3 = ggml_reshape_3d(ctx, probs, 1, c.n_experts, g.S);
  ggml_tensor *weights = ggml_get_rows(ctx, probs3, ids);
  ggml_tensor *weights2 = ggml_reshape_2d(ctx, weights, c.topk, g.S);
  if (c.expert_weights_norm) {
    ggml_tensor *sum = ggml_sum_rows(ctx, weights2);
    sum = ggml_clamp(ctx, sum, 6.103515625e-5f, INFINITY);
    weights2 = ggml_div(ctx, weights2, sum);
  }
  if (c.expert_weights_scale != 0.0f && c.expert_weights_scale != 1.0f)
    weights2 = ggml_scale(ctx, weights2, c.expert_weights_scale);
  weights = ggml_reshape_3d(ctx, weights2, 1, c.topk, g.S);
  ggml_build_forward_expand(g.graph, weights);

  ggml_tensor *input3 = ggml_reshape_3d(ctx, cur, g.D, 1, g.S);
  ggml_tensor *up = ggml_mul_mat_id(ctx, up_exps, input3, ids);
  ggml_tensor *gate = ggml_mul_mat_id(ctx, gate_exps, input3, ids);
  ggml_tensor *act = ggml_swiglu_split(ctx, gate, up);
  ggml_tensor *experts = ggml_mul_mat_id(ctx, down_exps, act, ids);
  experts = ggml_mul(ctx, experts, weights);

  ggml_tensor *out = ggml_view_2d(ctx, experts, g.D, g.S, experts->nb[2], 0);
  out = ggml_cont(ctx, out);
  for (int k = 1; k < c.topk; ++k) {
    ggml_tensor *e = ggml_view_2d(ctx, experts, g.D, g.S, experts->nb[2],
                                  static_cast<size_t>(k) * experts->nb[1]);
    out = ggml_add(ctx, out, e);
  }
  (void)layer;
  return out;
}

ggml_tensor *build_shared(FullGraph &g, ggml_context *ctx, ggml_tensor *cur,
                          int layer, ggml_tensor *gate, ggml_tensor *up,
                          ggml_tensor *down, ggml_tensor *scalar_gate) {
  if (!gate || !up || !down)
    return nullptr;
  ggml_tensor *gate_out = ggml_mul_mat(ctx, gate, cur);
  ggml_tensor *up_out = ggml_mul_mat(ctx, up, cur);
  ggml_tensor *act = ggml_swiglu_split(ctx, gate_out, up_out);
  ggml_tensor *out = ggml_mul_mat(ctx, down, act);
  if (scalar_gate) {
    ggml_tensor *sg = ggml_sigmoid(ctx, ggml_mul_mat(ctx, scalar_gate, cur));
    out = ggml_mul(ctx, out, sg);
  }
  (void)layer;
  return out;
}

ggml_tensor *build_gdn(FullGraph &g, ggml_context *ctx, ggml_tensor *cur,
                       int layer, ggml_tensor *w_qkv, ggml_tensor *w_gate,
                       ggml_tensor *w_alpha, ggml_tensor *w_beta,
                       ggml_tensor *w_dt, ggml_tensor *w_a, ggml_tensor *w_conv,
                       ggml_tensor *w_norm, ggml_tensor *w_out,
                       ggml_tensor *hist_in, ggml_tensor *state_in,
                       ggml_tensor **state_out, ggml_tensor **hist_out) {
  ggml_tensor *qkv = ggml_mul_mat(ctx, w_qkv, cur);
  ggml_tensor *z = ggml_mul_mat(ctx, w_gate, cur);
  ggml_tensor *alpha = ggml_mul_mat(ctx, w_alpha, cur);
  ggml_tensor *beta_raw = ggml_mul_mat(ctx, w_beta, cur);
  ggml_tensor *glog =
      ggml_mul(ctx, ggml_softplus(ctx, ggml_add(ctx, alpha, w_dt)), w_a);
  ggml_tensor *beta = ggml_sigmoid(ctx, beta_raw);

  ggml_tensor *conv_in = ggml_concat(ctx, hist_in, ggml_transpose(ctx, qkv), 0);
  ggml_tensor *conv = ggml_silu(ctx, ggml_ssm_conv(ctx, conv_in, w_conv));
  const size_t row = ggml_row_size(GGML_TYPE_F32, g.conv_dim);
  ggml_tensor *q =
      ggml_view_4d(ctx, conv, g.gdn_hd, g.Hk_gdn, g.S, 1,
                   ggml_row_size(GGML_TYPE_F32, g.gdn_hd), row, row * g.S, 0);
  ggml_tensor *k =
      ggml_view_4d(ctx, conv, g.gdn_hd, g.Hk_gdn, g.S, 1,
                   ggml_row_size(GGML_TYPE_F32, g.gdn_hd), row, row * g.S,
                   ggml_row_size(GGML_TYPE_F32, g.key_dim));
  ggml_tensor *v =
      ggml_view_4d(ctx, conv, g.gdn_hd, g.Hv_gdn, g.S, 1,
                   ggml_row_size(GGML_TYPE_F32, g.gdn_hd), row, row * g.S,
                   ggml_row_size(GGML_TYPE_F32, 2 * g.key_dim));
  q = ggml_l2_norm(ctx, q, g.engine->model_file.architecture_config.eps);
  k = ggml_l2_norm(ctx, k, g.engine->model_file.architecture_config.eps);
  glog = ggml_reshape_4d(ctx, glog, 1, g.Hv_gdn, g.S, 1);
  beta = ggml_reshape_4d(ctx, beta, 1, g.Hv_gdn, g.S, 1);
  ggml_tensor *gdn =
      ggml_gated_delta_net(ctx, q, k, v, glog, beta, state_in, 1);

  const size_t hd_row = ggml_row_size(GGML_TYPE_F32, g.gdn_hd);
  const size_t value_row = ggml_row_size(GGML_TYPE_F32, g.value_dim);
  const size_t state_row = ggml_row_size(GGML_TYPE_F32, g.gdn_hd * g.gdn_hd);
  ggml_tensor *core = ggml_view_4d(ctx, gdn, g.gdn_hd, g.Hv_gdn, g.S, 1, hd_row,
                                   value_row, value_row * g.S, 0);
  // ggml-cuda's gated_delta_net kernel stores the per-head state TRANSPOSED
  // relative to the [k][v] math convention this codebase's own legacy CPU
  // qwen4exp_gdn_delta_step uses -- confirmed directly in
  // ggml/src/ggml-cuda/gated_delta_net.cu's own comment ("state is stored
  // transposed: M[col][i] = S[i][col]", where col indexes v and i indexes
  // k), i.e. the raw view below has ne[0]=k (contiguous), ne[1]=v. Legacy's
  // own flat layout (qwen4exp_recur_state[l], engine.hpp) is st[k*hd+v] --
  // v contiguous, k slower -- the opposite axis order. Left unfixed, the
  // flat ggml_backend_tensor_get+memcpy in qwen4exp_full_graph_run silently
  // hands decode a transposed state matrix: same byte count, same total
  // size (gdn_hd==gdn_hd), no crash, just wrong values from the very first
  // decode token onward. Transpose+materialize here so the exported
  // tensor's own byte layout already matches what the flat copy-out assumes.
  ggml_tensor *state_raw =
      ggml_view_4d(ctx, gdn, g.gdn_hd, g.gdn_hd, g.Hv_gdn, 1, hd_row, state_row,
                   state_row * g.Hv_gdn,
                   static_cast<size_t>(g.value_dim) * g.S * sizeof(float));
  *state_out = ggml_cont(ctx, ggml_transpose(ctx, state_raw));
  ggml_tensor *norm = ggml_mul(
      ctx,
      ggml_rms_norm(ctx, core, g.engine->model_file.architecture_config.eps),
      w_norm);
  ggml_tensor *z4 = ggml_reshape_4d(ctx, z, g.gdn_hd, g.Hv_gdn, g.S, 1);
  ggml_tensor *gated = ggml_mul(ctx, norm, ggml_sigmoid(ctx, z4));
  ggml_tensor *out =
      ggml_mul_mat(ctx, w_out, ggml_reshape_2d(ctx, gated, g.value_dim, g.S));
  *hist_out =
      ggml_cont(ctx, ggml_view_3d(ctx, conv_in, g.d_conv - 1, g.conv_dim, 1,
                                  conv_in->nb[1], conv_in->nb[2],
                                  ggml_row_size(conv_in->type, g.S)));
  (void)layer;
  return out;
}

ggml_tensor *build_full_attention(FullGraph &g, ggml_context *ctx,
                                  ggml_tensor *cur, int layer, ggml_tensor *w_q,
                                  ggml_tensor *w_k, ggml_tensor *w_v,
                                  ggml_tensor *w_qn, ggml_tensor *w_kn,
                                  ggml_tensor *w_o, ggml_tensor **k_cache,
                                  ggml_tensor **v_cache) {
  ggml_tensor *qfull = ggml_mul_mat(ctx, w_q, cur);
  ggml_tensor *q =
      ggml_view_3d(ctx, qfull, g.hd, g.H, g.S, sizeof(float) * g.hd * 2,
                   sizeof(float) * g.hd * 2 * g.H, 0);
  q = ggml_mul(
      ctx, ggml_rms_norm(ctx, q, g.engine->model_file.architecture_config.eps),
      w_qn);
  ggml_tensor *k =
      ggml_reshape_3d(ctx, ggml_mul_mat(ctx, w_k, cur), g.hd, g.Hkv, g.S);
  k = ggml_mul(
      ctx, ggml_rms_norm(ctx, k, g.engine->model_file.architecture_config.eps),
      w_kn);
  ggml_tensor *v =
      ggml_reshape_3d(ctx, ggml_mul_mat(ctx, w_v, cur), g.hd, g.Hkv, g.S);
  const int n_rot = g.engine->model_file.architecture_config.rope_n_rot > 0
                        ? g.engine->model_file.architecture_config.rope_n_rot
                        : g.hd;
  q = ggml_rope(ctx, q, g.positions, n_rot, GGML_ROPE_TYPE_NORMAL);
  k = ggml_rope(ctx, k, g.positions, n_rot, GGML_ROPE_TYPE_NORMAL);
  // The host dense KV cache stores the same normalized and rotated K/V values
  // that Bee's graph writes to its cache.  Keep these F32 tensors before the
  // layout-only permute/cast used by flash attention.
  if (k_cache)
    *k_cache = k;
  if (v_cache)
    *v_cache = v;
  q = ggml_permute(ctx, q, 0, 2, 1, 3);
  k = ggml_permute(ctx, k, 0, 2, 1, 3);
  v = ggml_permute(ctx, v, 0, 2, 1, 3);
  // The CUDA MMA flash-attention implementation intentionally consumes Q as
  // F32 and converts/loads K and V as F16.  Casting Q to F16 selects the MMA
  // path and then trips its Q-type assertion during graph execution.
  if (k->type == GGML_TYPE_F32)
    k = ggml_cast(ctx, k, GGML_TYPE_F16);
  if (v->type == GGML_TYPE_F32)
    v = ggml_cast(ctx, v, GGML_TYPE_F16);
  ggml_tensor *attn = ggml_flash_attn_ext(
      ctx, q, k, v, g.attn_mask, 1.0f / std::sqrt(static_cast<float>(g.hd)),
      0.0f, 0.0f);
  ggml_tensor *gate =
      ggml_view_3d(ctx, qfull, g.hd, g.H, g.S, sizeof(float) * g.hd * 2,
                   sizeof(float) * g.hd * 2 * g.H, sizeof(float) * g.hd);
  gate = ggml_reshape_3d(ctx, ggml_cont_2d(ctx, gate, g.H * g.hd, g.S), g.hd,
                         g.H, g.S);
  attn = ggml_mul(ctx, attn, ggml_sigmoid(ctx, gate));
  attn = ggml_reshape_2d(ctx, attn, g.H * g.hd, g.S);
  (void)layer;
  return ggml_mul_mat(ctx, w_o, attn);
}

bool prepare_ple_embedding(FullGraph &g, std::vector<float> &embedding, int S,
                           int pos_base, const int *token_ids) {
  const auto &c = g.engine->model_file.architecture_config;
  if (!has_ple_layer(c) || !token_ids || pos_base != 0)
    return !has_ple_layer(c);
  const GGUFTensor *table =
      model_tensor(*g.engine, "per_layer_token_embd.weight");
  if (!table || table->shape.size() != 2)
    return false;
  const int row_dim = static_cast<int>(table->shape[0]);
  const int rb = row_bytes(table->type, row_dim);
  if (rb <= 0 || row_dim != c.ple_n_embd_per_layer)
    return false;

  std::vector<int32_t> rows(static_cast<size_t>(S) * c.ple_n_heads);
  const int64_t eos = c.ple_eos_token_id;
  for (int i = 0; i < S; ++i) {
    int64_t ctx[8] = {};
    if (c.ple_ngram_size > 8)
      return false;
    ctx[0] = token_ids[i];
    bool cut = false;
    for (int s = 1; s < c.ple_ngram_size; ++s) {
      int j = i - s;
      ctx[s] = j >= 0 ? token_ids[j] : eos;
      if (cut || ctx[s] == eos) {
        ctx[s] = eos;
        cut = true;
      }
    }
    for (int n = 2; n <= c.ple_ngram_size; ++n) {
      int base = (n - 2) * c.ple_heads_per_ngram;
      for (int h = 0; h < c.ple_heads_per_ngram; ++h) {
        int hi = base + h;
        rows[static_cast<size_t>(i) * c.ple_n_heads + hi] =
            static_cast<int32_t>(
                ple_ngram_row(ctx, n, c.ple_layer_multipliers.data(),
                              c.ple_head_vocab_sizes[static_cast<size_t>(hi)],
                              c.ple_head_offsets[static_cast<size_t>(hi)]));
      }
    }
  }

  std::unordered_map<int32_t, std::vector<float>> cache;
  auto *raw = static_cast<uint8_t *>(alloc_host(g, static_cast<size_t>(rb)));
  if (!raw)
    return false;
  const int fd = g.engine->model_file.fd_for_tensor(*table);
  for (int32_t r : rows) {
    if (cache.count(r))
      continue;
    const ssize_t got = direct_io::pread_direct_aware(
        fd, g.engine->model_file.direct_io_active_for(*table),
        g.engine->model_file.direct_io_alignment, raw, static_cast<size_t>(rb),
        table->offset + static_cast<uint64_t>(r) * static_cast<size_t>(rb));
    if (got != static_cast<ssize_t>(rb))
      return false;
    std::vector<float> row(static_cast<size_t>(row_dim));
    if (!dequant_row(table->type, raw, row.data(), row_dim))
      return false;
    cache.emplace(r, std::move(row));
  }
  embedding.resize(static_cast<size_t>(S) * c.ple_n_heads * row_dim);
  for (int i = 0; i < S; ++i) {
    for (int h = 0; h < c.ple_n_heads; ++h) {
      const auto &row =
          cache.at(rows[static_cast<size_t>(i) * c.ple_n_heads + h]);
      std::memcpy(embedding.data() +
                      (static_cast<size_t>(i) * c.ple_n_heads + h) * row_dim,
                  row.data(), static_cast<size_t>(row_dim) * sizeof(float));
    }
  }
  return true;
}

bool build_graph(FullGraph &g, const std::vector<float> *ple_embedding) {
  const auto &c = g.engine->model_file.architecture_config;
  ggml_init_params ip = {};
  ip.mem_size = kTensorContextBytes;
  ip.no_alloc = true;
  if (!MemoryGuard::try_reserve_host_bytes(kTensorContextBytes)) {
    LOG_WARN("[GGML-QWEN-FULL-MEM] host graph context refused: %.2f MiB "
             "(reserved=%.2f GiB, budget=%.2f GB)\n",
             (double)kTensorContextBytes / (1024.0 * 1024.0),
             (double)MemoryGuard::reserved_host_bytes() / 1.0e9,
             (double)MemoryGuard::host_budget_bytes() / 1.0e9);
    return false;
  }
  g.context_reservation_bytes = kTensorContextBytes;
  g.ctx = ggml_init(ip);
  if (!g.ctx) {
    MemoryGuard::release_host_bytes(g.context_reservation_bytes);
    g.context_reservation_bytes = 0;
    return false;
  }
  g.graph = ggml_new_graph_custom(g.ctx, kGraphSize, false);
  if (!g.graph)
    return false;

  g.input = host_input(g, g.ctx, GGML_TYPE_F32, {g.D, g.HC, g.S},
                       static_cast<size_t>(g.D) * g.HC * g.S * sizeof(float),
                       "qwen_full_input");
  g.positions = host_input(g, g.ctx, GGML_TYPE_I32, {g.S},
                           static_cast<size_t>(g.S) * sizeof(int32_t),
                           "qwen_full_positions");
  g.attn_mask = host_input(g, g.ctx, GGML_TYPE_F16, {g.S, g.S},
                           static_cast<size_t>(g.S) * g.S * sizeof(ggml_fp16_t),
                           "qwen_full_causal_mask");
  if (!g.input || !g.positions || !g.attn_mask)
    return false;
  auto *pos = static_cast<int32_t *>(g.host_inputs[1]);
  auto *mask = static_cast<ggml_fp16_t *>(g.host_inputs[2]);
  for (int i = 0; i < g.S; ++i)
    pos[i] = i;
  for (int q = 0; q < g.S; ++q)
    for (int k = 0; k < g.S; ++k)
      mask[static_cast<size_t>(q) * g.S + k] =
          ggml_fp32_to_fp16(k <= q ? 0.0f : -INFINITY);

  ggml_tensor *res = g.input;
  for (int l = 0; l < g.n_layers; ++l) {
    const std::string p = "blk." + std::to_string(l) + ".";
    if (qwen4exp_is_ple_layer(c, l)) {
      if (!ple_embedding)
        return false;
      g.ple_embedding =
          host_input(g, g.ctx, GGML_TYPE_F32,
                     {c.ple_n_embd_per_layer * c.ple_n_heads, g.S},
                     static_cast<size_t>(c.ple_n_embd_per_layer) *
                         c.ple_n_heads * g.S * sizeof(float),
                     "qwen_full_ple_embedding");
      if (!g.ple_embedding)
        return false;
      std::memcpy(g.host_inputs.back(), ple_embedding->data(),
                  ple_embedding->size() * sizeof(float));
      g.ple_layer = l;
      // PLE graph is constructed below with the same equations as Bee's
      // build_ple.  The input embedding is host gathered, while its learned
      // projections and the rest of the operation stay in this graph.
      ggml_tensor *ple_key = resident_tensor(g, g.ctx, p + "ple_key.weight");
      ggml_tensor *ple_value =
          resident_tensor(g, g.ctx, p + "ple_value.weight");
      ggml_tensor *ple_nk =
          resident_tensor(g, g.ctx, p + "ple_norm_key.weight");
      ggml_tensor *ple_nq =
          resident_tensor(g, g.ctx, p + "ple_norm_query.weight");
      ggml_tensor *ple_nc =
          resident_tensor(g, g.ctx, p + "ple_norm_conv.weight");
      ggml_tensor *ple_conv =
          resident_tensor(g, g.ctx, p + "ple_conv1d.weight");
      if (!ple_key || !ple_value || !ple_nk || !ple_nq || !ple_nc || !ple_conv)
        return false;
      ggml_tensor *pk = ggml_mul_mat(g.ctx, ple_key, g.ple_embedding);
      ggml_tensor *pv = ggml_mul_mat(g.ctx, ple_value, g.ple_embedding);
      auto group_norm = [&](ggml_tensor *x, ggml_tensor *w) {
        x = ggml_reshape_3d(g.ctx, x, g.D, g.HC, g.S);
        x = ggml_rms_norm(g.ctx, x, c.eps);
        x = ggml_reshape_2d(g.ctx, x, g.D * g.HC, g.S);
        x = ggml_mul(g.ctx, x, w);
        return ggml_reshape_3d(g.ctx, x, g.D, g.HC, g.S);
      };
      pk = group_norm(pk, ple_nk);
      ggml_tensor *pq = group_norm(res, ple_nq);
      ggml_tensor *score =
          ggml_scale(g.ctx, ggml_sum_rows(g.ctx, ggml_mul(g.ctx, pk, pq)),
                     1.0f / std::sqrt(static_cast<float>(g.D)));
      ggml_tensor *gate = ggml_sigmoid(
          g.ctx,
          ggml_mul(g.ctx, ggml_sgn(g.ctx, score),
                   ggml_sqrt(g.ctx, ggml_clamp(g.ctx, ggml_abs(g.ctx, score),
                                               1e-6f, INFINITY))));
      ggml_tensor *v3 = ggml_repeat_4d(
          g.ctx, ggml_reshape_3d(g.ctx, pv, g.D, 1, g.S), g.D, g.HC, g.S, 1);
      ggml_tensor *gated = ggml_mul(g.ctx, v3, gate);
      ggml_tensor *normed =
          group_norm(ggml_reshape_2d(g.ctx, gated, g.D * g.HC, g.S), ple_nc);
      normed = ggml_reshape_2d(g.ctx, normed, g.D * g.HC, g.S);

      const int hist = (c.ple_conv_kernel - 1) * c.ple_ngram_size;
      ggml_tensor *ph =
          host_input(g, g.ctx, GGML_TYPE_F32, {hist, g.D * g.HC, 1},
                     static_cast<size_t>(hist) * g.D * g.HC * sizeof(float),
                     "qwen_full_ple_history");
      if (!ph)
        return false;
      g.ple_hist_input = static_cast<float *>(ph->data);
      if (l < static_cast<int>(g.engine->qwen4exp_ple_conv_history.size())) {
        auto &h = g.engine->qwen4exp_ple_conv_history[l];
        if (h.size() == static_cast<size_t>(hist) * g.D * g.HC)
          std::memcpy(g.host_inputs.back(), h.data(), h.size() * sizeof(float));
      }
      ggml_tensor *padded =
          ggml_concat(g.ctx, ph, ggml_transpose(g.ctx, normed), 0);
      ggml_tensor *conv_out = nullptr;
      for (int k = 0; k < c.ple_conv_kernel; ++k) {
        const int start = hist - (c.ple_conv_kernel - 1 - k) * c.ple_ngram_size;
        ggml_tensor *shifted = ggml_cont(
            g.ctx,
            ggml_transpose(g.ctx,
                           ggml_view_3d(g.ctx, padded, g.S, g.D * g.HC, 1,
                                        padded->nb[1], padded->nb[2],
                                        ggml_row_size(padded->type, start))));
        ggml_tensor *wk = ggml_cont(
            g.ctx, ggml_view_2d(g.ctx, ple_conv, 1, g.D * g.HC, ple_conv->nb[1],
                                static_cast<size_t>(k) * ple_conv->nb[0]));
        wk = ggml_reshape_1d(g.ctx, wk, g.D * g.HC);
        if (wk->type != GGML_TYPE_F32)
          wk = ggml_cast(g.ctx, wk, GGML_TYPE_F32);
        ggml_tensor *term = ggml_mul(g.ctx, shifted, wk);
        conv_out = conv_out ? ggml_add(g.ctx, conv_out, term) : term;
      }
      conv_out =
          ggml_reshape_3d(g.ctx, ggml_silu(g.ctx, conv_out), g.D, g.HC, g.S);
      res = ggml_add(g.ctx, res, ggml_add(g.ctx, gated, conv_out));
      g.ple_hist_out = ggml_cont(
          g.ctx, ggml_view_3d(g.ctx, padded, hist, g.D * g.HC, 1, padded->nb[1],
                              padded->nb[2], ggml_row_size(padded->type, g.S)));
    }

    ggml_tensor *inject = nullptr;
    ggml_tensor *hn = resident_tensor(g, g.ctx, p + "hc_attn_norm.weight");
    ggml_tensor *hdn = resident_tensor(g, g.ctx, p + "hc_attn_down.weight");
    ggml_tensor *hup = resident_tensor(g, g.ctx, p + "hc_attn_up.weight");
    ggml_tensor *hinj = resident_tensor(g, g.ctx, p + "hc_attn_inject.weight");
    if (!hn || !hdn || !hup || !hinj)
      return false;
    ggml_tensor *attn_in = hc_mix(g, g.ctx, res, hn, hdn, hup, hinj, &inject);
    ggml_tensor *attn_out = nullptr;
    if (qwen4exp_is_gdn_layer(c, l)) {
      ggml_tensor *qkv = resident_tensor(g, g.ctx, p + "attn_qkv.weight");
      ggml_tensor *zg = resident_tensor(g, g.ctx, p + "attn_gate.weight");
      ggml_tensor *al = resident_tensor(g, g.ctx, p + "ssm_alpha.weight");
      ggml_tensor *be = resident_tensor(g, g.ctx, p + "ssm_beta.weight");
      ggml_tensor *dt = resident_tensor(g, g.ctx, p + "ssm_dt.bias");
      ggml_tensor *aa = resident_tensor(g, g.ctx, p + "ssm_a");
      ggml_tensor *cv = resident_tensor(g, g.ctx, p + "ssm_conv1d.weight");
      ggml_tensor *nm = resident_tensor(g, g.ctx, p + "ssm_norm.weight");
      ggml_tensor *oo = resident_tensor(g, g.ctx, p + "ssm_out.weight");
      if (!qkv || !zg || !al || !be || !dt || !aa || !cv || !nm || !oo)
        return false;
      const int hist = g.d_conv - 1;
      ggml_tensor *hi =
          host_input(g, g.ctx, GGML_TYPE_F32, {hist, g.conv_dim, 1},
                     static_cast<size_t>(hist) * g.conv_dim * sizeof(float),
                     "qwen_full_gdn_history");
      ggml_tensor *si = host_input(
          g, g.ctx, GGML_TYPE_F32, {g.gdn_hd, g.gdn_hd, g.Hv_gdn, 1},
          static_cast<size_t>(g.gdn_hd) * g.gdn_hd * g.Hv_gdn * sizeof(float),
          "qwen_full_gdn_state");
      if (!hi || !si)
        return false;
      g.gdn_hist_inputs.push_back(static_cast<float *>(hi->data));
      g.gdn_state_inputs.push_back(static_cast<float *>(si->data));
      if (l < static_cast<int>(g.engine->qwen4exp_conv_history.size()) &&
          g.engine->qwen4exp_conv_history[l].size() ==
              static_cast<size_t>(hist) * g.conv_dim)
        std::memcpy(g.host_inputs[g.host_inputs.size() - 2],
                    g.engine->qwen4exp_conv_history[l].data(),
                    static_cast<size_t>(hist) * g.conv_dim * sizeof(float));
      if (l < static_cast<int>(g.engine->qwen4exp_recur_state.size()) &&
          g.engine->qwen4exp_recur_state[l].size() ==
              static_cast<size_t>(g.gdn_hd) * g.gdn_hd * g.Hv_gdn)
        std::memcpy(g.host_inputs.back(),
                    g.engine->qwen4exp_recur_state[l].data(),
                    static_cast<size_t>(g.gdn_hd) * g.gdn_hd * g.Hv_gdn *
                        sizeof(float));
      ggml_tensor *so = nullptr, *ho = nullptr;
      attn_out = build_gdn(g, g.ctx, attn_in, l, qkv, zg, al, be, dt, aa, cv,
                           nm, oo, hi, si, &so, &ho);
      g.gdn_state_out.push_back(so);
      g.gdn_hist_out.push_back(ho);
      g.gdn_layers.push_back(l);
    } else {
      ggml_tensor *wq = dense_weight(g, g.ctx, l, "attn_q.weight");
      ggml_tensor *wk = dense_weight(g, g.ctx, l, "attn_k.weight");
      ggml_tensor *wv = dense_weight(g, g.ctx, l, "attn_v.weight");
      ggml_tensor *wqn = dense_weight(g, g.ctx, l, "attn_q_norm.weight");
      ggml_tensor *wkn = dense_weight(g, g.ctx, l, "attn_k_norm.weight");
      ggml_tensor *wo = dense_weight(g, g.ctx, l, "attn_output.weight");
      if (!wq || !wk || !wv || !wqn || !wkn || !wo)
        return false;
      ggml_tensor *k_cache = nullptr;
      ggml_tensor *v_cache = nullptr;
      attn_out = build_full_attention(g, g.ctx, attn_in, l, wq, wk, wv, wqn,
                                      wkn, wo, &k_cache, &v_cache);
      if (!k_cache || !v_cache)
        return false;
      g.kv_k.push_back(k_cache);
      g.kv_v.push_back(v_cache);
      g.kv_layers.push_back(l);
    }
    res = hc_combine(g, g.ctx, res, attn_out, inject);

    inject = nullptr;
    hn = resident_tensor(g, g.ctx, p + "hc_ffn_norm.weight");
    hdn = resident_tensor(g, g.ctx, p + "hc_ffn_down.weight");
    hup = resident_tensor(g, g.ctx, p + "hc_ffn_up.weight");
    hinj = resident_tensor(g, g.ctx, p + "hc_ffn_inject.weight");
    if (!hn || !hdn || !hup || !hinj)
      return false;
    ggml_tensor *ffn_in = hc_mix(g, g.ctx, res, hn, hdn, hup, hinj, &inject);
    ggml_tensor *router = resident_tensor(g, g.ctx, p + "ffn_gate_inp.weight");
    ggml_tensor *bias = resident_tensor(g, g.ctx, p + "exp_probs_b.bias");
    ggml_tensor *ge = expert_weight(g, g.ctx, l, "ffn_gate_exps.weight");
    ggml_tensor *ue = expert_weight(g, g.ctx, l, "ffn_up_exps.weight");
    ggml_tensor *de = expert_weight(g, g.ctx, l, "ffn_down_exps.weight");
    if (!router || !ge || !ue || !de)
      return false;
    ggml_tensor *moe = build_moe(g, g.ctx, ffn_in, l, router, bias, ge, ue, de);
    ggml_tensor *sg =
        resident_tensor(g, g.ctx, p + "ffn_gate_inp_shexp.weight");
    ggml_tensor *sge = resident_tensor(g, g.ctx, p + "ffn_gate_shexp.weight");
    ggml_tensor *sup = resident_tensor(g, g.ctx, p + "ffn_up_shexp.weight");
    ggml_tensor *sdo = resident_tensor(g, g.ctx, p + "ffn_down_shexp.weight");
    if (sge && sup && sdo) {
      ggml_tensor *shared =
          build_shared(g, g.ctx, ffn_in, l, sge, sup, sdo, sg);
      moe = ggml_add(g.ctx, moe, shared);
    }
    res = hc_combine(g, g.ctx, res, moe, inject);
  }

  ggml_tensor *on = resident_tensor(g, g.ctx, "output_hc_norm.weight");
  ggml_tensor *od = resident_tensor(g, g.ctx, "output_hc_down.weight");
  ggml_tensor *ou = resident_tensor(g, g.ctx, "output_hc_up.weight");
  if (!on || !od || !ou)
    return false;
  g.final_streams = res;
  g.output = hc_mix(g, g.ctx, res, on, od, ou, nullptr, nullptr);
  ggml_set_output(g.final_streams);
  ggml_build_forward_expand(g.graph, g.final_streams);
  ggml_set_output(g.output);
  ggml_build_forward_expand(g.graph, g.output);
  for (ggml_tensor *t : g.kv_k) {
    ggml_set_output(t);
    ggml_build_forward_expand(g.graph, t);
  }
  for (ggml_tensor *t : g.kv_v) {
    ggml_set_output(t);
    ggml_build_forward_expand(g.graph, t);
  }
  for (ggml_tensor *t : g.gdn_state_out) {
    ggml_set_output(t);
    ggml_build_forward_expand(g.graph, t);
  }
  for (ggml_tensor *t : g.gdn_hist_out) {
    ggml_set_output(t);
    ggml_build_forward_expand(g.graph, t);
  }
  if (g.ple_hist_out) {
    ggml_set_output(g.ple_hist_out);
    ggml_build_forward_expand(g.graph, g.ple_hist_out);
  }
  // Real, measured incident (2026-09-09): ggml_backend_sched_alloc_graph's
  // internal ggml_gallocr_reserve_n_impl does not fail gracefully on a real
  // cudaMalloc OOM -- it logged "cudaMalloc failed: out of memory" and then
  // segfaulted the whole process (exit 139) instead of returning a checkable
  // failure, at the exact point where main.cpp's own fallback VRAM cache
  // reservation (a fixed 4GB, unaware of device_expert_layers' own VRAM use)
  // left the scheduler's own reserve() short by ~1.8GB. main.cpp now sizes
  // that fallback cache against real free VRAM and this family's own
  // device_expert_layers estimate, but this check stays as a second,
  // independent line of defense: refuse to even attempt the allocator call
  // when real free VRAM on the graph's device is below a generous floor,
  // and decline to the legacy path instead of risking the same crash again.
  {
    size_t free_b = 0, total_b = 0;
    int prev_device = -1;
    cudaGetDevice(&prev_device);
    cudaSetDevice(g.device);
    const cudaError_t err = cudaMemGetInfo(&free_b, &total_b);
    if (prev_device >= 0)
      cudaSetDevice(prev_device);
    constexpr size_t kMinGraphAllocHeadroomBytes =
        2ull * 1024ull * 1024ull * 1024ull; // 2 GiB
    if (err != cudaSuccess || free_b < kMinGraphAllocHeadroomBytes) {
      LOG_ERROR(
          "[GGML-QWEN-FULL] refusing graph allocator reserve: device %d "
          "free=%.2f GiB "
          "< %.2f GiB safety floor (device expert arena + dense arena already "
          "committed %.2f GiB) -- falling back to the legacy path\n",
          g.device,
          err == cudaSuccess
              ? static_cast<double>(free_b) / (1024.0 * 1024.0 * 1024.0)
              : -1.0,
          static_cast<double>(kMinGraphAllocHeadroomBytes) /
              (1024.0 * 1024.0 * 1024.0),
          static_cast<double>(g.device_expert_capacity +
                              g.device_dense_capacity) /
              (1024.0 * 1024.0 * 1024.0));
      return false;
    }
  }
  if (!ggml_backend_sched_alloc_graph(g.sched, g.graph))
    return false;
  g.graph_bytes = ggml_backend_sched_get_buffer_size(g.sched, g.cuda);

  // The graph is only useful for performance if the scheduler keeps most of
  // the dependency chain on CUDA.  A high split count can otherwise hide a
  // CPU fallback for an op or a view/copy boundary, which recreates the
  // per-layer host orchestration that this graph is meant to remove.
  int cuda_nodes = 0;
  int cpu_nodes = 0;
  int other_nodes = 0;
  int backend_transitions = 0;
  ggml_backend_t previous_backend = nullptr;
  std::unordered_map<std::string, int> non_cuda_ops;
  for (int i = 0; i < ggml_graph_n_nodes(g.graph); ++i) {
    ggml_tensor *node = ggml_graph_node(g.graph, i);
    ggml_backend_t backend =
        ggml_backend_sched_get_tensor_backend(g.sched, node);
    if (backend == g.cuda) {
      ++cuda_nodes;
    } else if (backend == g.cpu) {
      ++cpu_nodes;
      ++non_cuda_ops[ggml_op_name(node->op)];
    } else {
      ++other_nodes;
      ++non_cuda_ops[backend ? ggml_backend_name(backend) : "NULL"];
    }
    if (previous_backend && backend != previous_backend)
      ++backend_transitions;
    previous_backend = backend;
  }
  LOG_INFO("[GGML-QWEN-FULL] scheduler nodes cuda=%d cpu=%d other=%d "
           "transitions=%d\n",
           cuda_nodes, cpu_nodes, other_nodes, backend_transitions);
  if (!non_cuda_ops.empty()) {
    std::string detail;
    for (const auto &entry : non_cuda_ops) {
      if (!detail.empty())
        detail += ", ";
      detail += entry.first + "=" + std::to_string(entry.second);
    }
    LOG_INFO("[GGML-QWEN-FULL] non-CUDA node ops: %s\n", detail.c_str());
  }
  LOG_INFO("[GGML-QWEN-FULL] graph built layers=%d nodes=%d splits=%d "
           "workspace=%.2f GiB\n",
           g.n_layers, ggml_graph_n_nodes(g.graph),
           ggml_backend_sched_get_n_splits(g.sched),
           static_cast<double>(g.graph_bytes) / (1024.0 * 1024.0 * 1024.0));
  LOG_INFO(
      "[GGML-QWEN-FULL] dense device arena used=%.2f/%.2f GiB; %d tensors\n",
      static_cast<double>(g.device_dense_used) / (1024.0 * 1024.0 * 1024.0),
      static_cast<double>(g.device_dense_capacity) / (1024.0 * 1024.0 * 1024.0),
      g.device_dense_tensors);
  if (g.device_expert_layers > 0) {
    const int complete_records = g.device_expert_layers * c.n_experts;
    LOG_INFO(
        "[GGML-QWEN-FULL] device expert arena used=%.2f/%.2f GiB; "
        "%d complete layer(s), %d expert records, all 3 projections resident\n",
        static_cast<double>(g.device_expert_used) / (1024.0 * 1024.0 * 1024.0),
        static_cast<double>(g.device_expert_capacity) /
            (1024.0 * 1024.0 * 1024.0),
        g.device_expert_layers, complete_records);
  }
  LOG_INFO(
      "[GGML-QWEN-FULL] host expert source=%.2f GiB graph-owned DDR copy; "
      "%.2f GiB aliased from preloaded DDR arena (%zu projection views)\n",
      static_cast<double>(g.host_weight_bytes) / (1024.0 * 1024.0 * 1024.0),
      static_cast<double>(g.host_ram_alias_bytes) / (1024.0 * 1024.0 * 1024.0),
      g.host_ram_alias_tensors);
  if (g.host_weight_bytes > 0) {
    LOG_INFO("[GGML-QWEN-FULL] host expert source pinned=%.2f GiB; "
             "pin failures=%zu\n",
             static_cast<double>(g.host_pinned_bytes) /
                 (1024.0 * 1024.0 * 1024.0),
             g.host_pin_failures);
  }
  LOG_INFO("[GGML-QWEN-FULL-MEM] host ledger=%.2f/%.2f GB; graph source="
           "%.2f GiB + context=%.2f MiB + owned-extra=%.2f MiB; device "
           "workspace=%.2f GiB (GPU, outside host_ram_gb)\n",
           (double)MemoryGuard::reserved_host_bytes() / 1.0e9,
           (double)MemoryGuard::host_budget_bytes() / 1.0e9,
           (double)g.host_weight_bytes / (1024.0 * 1024.0 * 1024.0),
           (double)g.context_reservation_bytes / (1024.0 * 1024.0),
           (double)(g.host_weight_bytes > g.host_reservation_bytes
                        ? g.host_weight_bytes - g.host_reservation_bytes
                        : 0) /
               (1024.0 * 1024.0),
           (double)g.graph_bytes / (1024.0 * 1024.0 * 1024.0));
  g.ready = true;
  return true;
}

} // namespace

bool qwen4exp_full_graph_enabled() { return env_enabled(); }

int qwen4exp_full_graph_device_expert_layers() {
  return env_enabled() ? env_device_expert_layers() : 0;
}

void qwen4exp_full_graph_shutdown() { g_graph.reset(); }

bool qwen4exp_full_graph_prepare(StreamingEngine &engine, int S, int pos_base,
                                 const int *token_ids) {
  if (!env_enabled() || S <= 1 || pos_base != 0 ||
      engine.family_ops.family != ModelFamily::Qwen4Exp)
    return false;
  const auto &c = engine.model_file.architecture_config;
  if (engine.qwen_ram_preload_all_experts) {
    for (int layer = 0; layer < c.n_layers; ++layer) {
      if (!engine.graph_ram_layer_ready(layer)) {
        LOG_WARN("[GGML-QWEN-FULL] disabled: ram_preload_all_experts "
                 "requires a ready DDR arena for every layer (missing "
                 "layer=%d); no GGUF mmap fallback\n",
                 layer);
        return false;
      }
    }
  }
  const int dev = engine.gpu_device_id >= 0 ? engine.gpu_device_id : 0;
  if (g_graph && g_graph->ready && g_graph->engine == &engine &&
      g_graph->S == S) {
    // PLE rows are request data, so refresh that input when the graph shape
    // is reused for a different prompt.  The model graph itself remains
    // persistent, exactly like Bee's shape cache.
    if (has_ple_layer(c)) {
      std::vector<float> ple;
      if (!prepare_ple_embedding(*g_graph, ple, S, pos_base, token_ids) ||
          !g_graph->ple_embedding ||
          ple.size() * sizeof(float) > ggml_nbytes(g_graph->ple_embedding))
        return false;
      std::memcpy(g_graph->ple_embedding->data, ple.data(),
                  ple.size() * sizeof(float));
    }
    return true;
  }
  g_graph.reset();
  auto g = std::make_unique<FullGraph>();
  g->engine = &engine;
  g->device = dev;
  g->S = S;
  g->D = c.hidden;
  g->HC = c.hc_mult;
  g->n_layers = c.n_layers;
  g->H = c.n_heads;
  g->Hkv = std::max(1, c.n_head_kv);
  g->hd = c.head_dim;
  g->Hk_gdn = c.ssm_group_count;
  g->Hv_gdn = c.ssm_time_step_rank;
  g->gdn_hd = c.ssm_state_size;
  g->key_dim = g->Hk_gdn * g->gdn_hd;
  g->value_dim = c.ssm_inner_size;
  g->conv_dim = 2 * g->key_dim + g->value_dim;
  g->d_conv = c.ssm_conv_kernel;
  g->device_expert_layers =
      std::min(qwen4exp_full_graph_device_expert_layers(), c.n_layers);
  const char *partial_gate_up =
      std::getenv("MOESTREAM_QWEN4EXP_GRAPH_PRELOAD_PARTIAL_GATE_UP");
  g->device_expert_partial_gate_up = g->device_expert_layers > 0 &&
                                     partial_gate_up &&
                                     partial_gate_up[0] == '1';
  g->cuda = static_cast<ggml_backend_t>(ggml_shared_backend(dev));
  g->cpu = ggml_backend_cpu_init();
  if (!g->cuda || !g->cpu)
    return false;

  // Admit the graph-owned expert source set once, before any of its
  // individual host buffers are allocated.  Layers with a complete
  // preloaded graph-RAM arena are excluded: their expert planes are aliased
  // directly from the existing DDR cache below, so this graph does not make
  // a second copy.  Any remaining layers use one graph-owned DDR copy rather
  // than a decode-time GGUF mmap source.  The extra watermark covers the
  // GGML host context, prompt/state inputs, and allocator/I/O fluctuation;
  // no /proc/meminfo read is performed in the per-buffer allocation loop.
  const size_t expert_bytes = full_expert_source_bytes(engine);
  // 10 GiB, per explicit user instruction after the 2026-09-09 crash (was
  // 4 GiB -- the smallest, least-conservative watermark of any admission
  // check in either graph file; aligned to the same 10 GiB floor as the
  // rest).
  constexpr size_t kGraphHostWatermark = 10ull * 1024ull * 1024ull * 1024ull;
  if (expert_bytes > 0 &&
      !MemoryGuard::try_reserve_host_bytes(expert_bytes, kGraphHostWatermark)) {
    LOG_WARN("[GGML-QWEN-FULL] host watermark refused graph source=%.2f GiB + "
             "%.2f GiB reserve; "
             "MemAvailable=%ld MiB\n",
             static_cast<double>(expert_bytes) / (1024.0 * 1024.0 * 1024.0),
             static_cast<double>(kGraphHostWatermark) /
                 (1024.0 * 1024.0 * 1024.0),
             MemoryGuard::last_available_mb());
    return false;
  }
  g->host_reservation_bytes = expert_bytes;
  g->host_precharged_bytes = expert_bytes;
  LOG_INFO("[GGML-QWEN-FULL] host source admission: graph-owned=%.2f GiB + "
           "%.2f GiB reserve; "
           "complete preloaded DDR arenas are aliased directly\n",
           static_cast<double>(expert_bytes) / (1024.0 * 1024.0 * 1024.0),
           static_cast<double>(kGraphHostWatermark) /
               (1024.0 * 1024.0 * 1024.0));
  if (!init_device_expert_arena(*g))
    return false;
  if (!init_device_dense_arena(*g))
    return false;
  ggml_backend_t backends[] = {g->cuda, g->cpu};
  g->sched =
      ggml_backend_sched_new(backends, nullptr, 2, kGraphSize, false, true);
  if (!g->sched)
    return false;

  // The model graph starts from the already expanded HC embedding.  PLE's
  // hash gather is prepared once here, outside the measured prefill window;
  // learned PLE projections and its causal convolution remain graph nodes.
  std::vector<float> ple;
  if (!prepare_ple_embedding(*g, ple, S, pos_base, token_ids) &&
      has_ple_layer(c)) {
    LOG_WARN("[GGML-QWEN-FULL] PLE embedding preparation declined\n");
    return false;
  }
  if (!build_graph(*g, has_ple_layer(c) ? &ple : nullptr)) {
    LOG_WARN("[GGML-QWEN-FULL] graph construction declined; using fallback\n");
    return false;
  }
  g_graph = std::move(g);
  return true;
}

bool qwen4exp_full_graph_run(StreamingEngine &engine, float *hidden_states,
                             int S, int pos_base, const int *token_ids) {
  // Prefill-only while recurrent and dense KV state writeback is validated.
  if (!env_enabled() || !hidden_states || S <= 1 || pos_base != 0 || !g_graph ||
      !g_graph->ready || g_graph->engine != &engine || g_graph->S != S)
    return false;
  FullGraph &g = *g_graph;
  auto fail = [&](const char *why) {
    LOG_WARN("[GGML-QWEN-FULL] execution declined: %s\n", why);
    return false;
  };
  LOG_INFO(
      "[GGML-QWEN-FULL] execute candidate S=%d gdn=%zu kv=%zu ple_layer=%d\n",
      S, g.gdn_layers.size(), g.kv_layers.size(), g.ple_layer);

  // A fresh position-zero graph starts a new recurrent/KV stream.  This is
  // the same reset boundary used by the established Qwen attention path.
  for (size_t i = 0; i < engine.qwen4exp_conv_history.size(); ++i)
    std::fill(engine.qwen4exp_conv_history[i].begin(),
              engine.qwen4exp_conv_history[i].end(), 0.0f);
  for (size_t i = 0; i < engine.qwen4exp_recur_state.size(); ++i)
    std::fill(engine.qwen4exp_recur_state[i].begin(),
              engine.qwen4exp_recur_state[i].end(), 0.0f);
  std::fill(engine.qwen4exp_ple_tok_hist.begin(),
            engine.qwen4exp_ple_tok_hist.end(),
            static_cast<int32_t>(
                engine.model_file.architecture_config.ple_eos_token_id));
  engine.qwen4exp_ple_next_pos = 0;

  for (size_t i = 0; i < g.gdn_layers.size(); ++i) {
    const int l = g.gdn_layers[i];
    if (i >= g.gdn_state_inputs.size() || i >= g.gdn_hist_inputs.size())
      return fail("GDN input bookkeeping mismatch");
    const size_t state_n = static_cast<size_t>(g.gdn_hd) * g.gdn_hd * g.Hv_gdn;
    const size_t hist_n = static_cast<size_t>(g.d_conv - 1) * g.conv_dim;
    if (l < 0 || l >= static_cast<int>(engine.qwen4exp_recur_state.size()) ||
        l >= static_cast<int>(engine.qwen4exp_conv_history.size()) ||
        engine.qwen4exp_recur_state[l].size() != state_n ||
        engine.qwen4exp_conv_history[l].size() != hist_n)
      return fail("GDN state/history size mismatch");
    std::memcpy(g.gdn_state_inputs[i], engine.qwen4exp_recur_state[l].data(),
                state_n * sizeof(float));
    std::memcpy(g.gdn_hist_inputs[i], engine.qwen4exp_conv_history[l].data(),
                hist_n * sizeof(float));
  }
  if (g.ple_layer >= 0 && g.ple_hist_input) {
    const auto &c = engine.model_file.architecture_config;
    // The engine keeps PLE history in the conv layout [HC*D, hist].  The
    // PLE embedding uses [n_heads, embd_per_layer], but those are not the
    // storage dimensions of the causal-convolution state.
    const size_t hist_n = static_cast<size_t>(c.ple_conv_kernel - 1) *
                          c.ple_ngram_size * g.HC * g.D;
    if (g.ple_layer >=
            static_cast<int>(engine.qwen4exp_ple_conv_history.size()) ||
        engine.qwen4exp_ple_conv_history[g.ple_layer].size() != hist_n)
      return fail("PLE history size mismatch");
    std::memcpy(g.ple_hist_input,
                engine.qwen4exp_ple_conv_history[g.ple_layer].data(),
                hist_n * sizeof(float));
  } else if (has_ple_layer(engine.model_file.architecture_config)) {
    return fail("PLE graph input is missing");
  }
  if (g.host_inputs.empty() ||
      engine.hc_stream_buf.size() < static_cast<size_t>(g.D) * g.HC * S)
    return fail("expanded HC input is missing or too small");
  std::memcpy(g.host_inputs[0], engine.hc_stream_buf.data(),
              static_cast<size_t>(g.D) * g.HC * S * sizeof(float));

  // Real-hardware incident (2026-09-09, GLM5NEXT side): a 480-token prompt
  // crashed the whole machine (hard reboot required, no surviving kernel/
  // OOM log -- this box has no persistent journald storage) with the last
  // surviving application-log line printed just before this exact kind of
  // compute call. Every OTHER host allocation on this path (arena prep,
  // expert source, graph context) goes through MemoryGuard::try_reserve_
  // host_bytes first; this compute call did not. Re-check REAL, fresh
  // MemAvailable (not the background watchdog's own possibly-500ms-stale
  // allocation_blocked() flag) immediately before committing to it -- a
  // large gap can open between prepare()'s own admission checks and this
  // moment (preload alone can take minutes). Not a confirmed fix for
  // whatever the real mechanism was (no kernel evidence survived to
  // confirm it), just a cheap, additional, defense-in-depth backstop,
  // applied identically to qwen4exp's own graph for the same reason.
  {
    const long avail_mb = MemoryGuard::read_available_mb();
    constexpr long kMinComputeHeadroomMb =
        10240; // 10 GiB, per explicit user instruction after the 2026-09-09
               // crash
    if (avail_mb >= 0 && avail_mb < kMinComputeHeadroomMb) {
      LOG_ERROR("[GGML-QWEN-FULL] refusing graph compute: MemAvailable=%ld MiB "
                "< %ld MiB safety floor -- falling back to the legacy path\n",
                avail_mb, kMinComputeHeadroomMb);
      return false;
    }
  }

  const auto t_compute_start = std::chrono::steady_clock::now();
  ggml_status st = ggml_backend_sched_graph_compute_async(g.sched, g.graph);
  const auto t_submit_end = std::chrono::steady_clock::now();
  if (st != GGML_STATUS_SUCCESS) {
    LOG_ERROR("[GGML-QWEN-FULL] graph compute failed: status=%d\n", (int)st);
    return false;
  }
  ggml_backend_sched_synchronize(g.sched);
  const auto t_sync_end = std::chrono::steady_clock::now();
  ggml_backend_tensor_get(g.output, hidden_states, 0,
                          static_cast<size_t>(g.D) * S * sizeof(float));
  const auto t_output_end = std::chrono::steady_clock::now();

  for (size_t i = 0; i < g.gdn_layers.size(); ++i) {
    const int l = g.gdn_layers[i];
    const size_t state_n = static_cast<size_t>(g.gdn_hd) * g.gdn_hd * g.Hv_gdn;
    const size_t hist_n = static_cast<size_t>(g.d_conv - 1) * g.conv_dim;
    ggml_backend_tensor_get(g.gdn_state_out[i],
                            engine.qwen4exp_recur_state[l].data(), 0,
                            state_n * sizeof(float));
    ggml_backend_tensor_get(g.gdn_hist_out[i],
                            engine.qwen4exp_conv_history[l].data(), 0,
                            hist_n * sizeof(float));
  }
  for (size_t i = 0; i < g.kv_layers.size(); ++i) {
    const int l = g.kv_layers[i];
    if (l < 0 || l >= static_cast<int>(engine.Kc.size()) ||
        l >= static_cast<int>(engine.Vc.size()) || !engine.Kc[l] ||
        !engine.Vc[l])
      return fail("dense KV cache is not allocated");
    const size_t kv_n = static_cast<size_t>(g.S) * g.Hkv * g.hd;
    std::vector<float> k(kv_n), v(kv_n);
    ggml_backend_tensor_get(g.kv_k[i], k.data(), 0, kv_n * sizeof(float));
    ggml_backend_tensor_get(g.kv_v[i], v.data(), 0, kv_n * sizeof(float));
    const size_t row = static_cast<size_t>(g.Hkv) * g.hd;
    for (int s = 0; s < g.S && s + pos_base < engine.max_t; ++s) {
      std::memcpy(engine.Kc[l] + static_cast<size_t>(s + pos_base) * row,
                  k.data() + static_cast<size_t>(s) * row, row * sizeof(float));
      std::memcpy(engine.Vc[l] + static_cast<size_t>(s + pos_base) * row,
                  v.data() + static_cast<size_t>(s) * row, row * sizeof(float));
    }
  }
  if (g.ple_layer >= 0 && g.ple_hist_out) {
    const auto &c = engine.model_file.architecture_config;
    const size_t hist_n = static_cast<size_t>(c.ple_conv_kernel - 1) *
                          c.ple_ngram_size * g.HC * g.D;
    auto &dst = engine.qwen4exp_ple_conv_history[g.ple_layer];
    ggml_backend_tensor_get(g.ple_hist_out, dst.data(), 0,
                            hist_n * sizeof(float));
    engine.qwen4exp_ple_tok_hist.clear();
    const int keep = std::max(0, c.ple_ngram_size - 1);
    const int start = std::max(0, S - keep);
    for (int i = start; i < S; ++i)
      engine.qwen4exp_ple_tok_hist.push_back(token_ids[i]);
    engine.qwen4exp_ple_next_pos = S;
  }
  const auto t_state_end = std::chrono::steady_clock::now();
  const auto ms = [](const auto &a, const auto &b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
  };
  LOG_INFO(
      "[GGML-QWEN-FULL-TIME] submit=%.3f ms sync=%.3f ms output_get=%.3f ms "
      "state_cache_get=%.3f ms total=%.3f ms\n",
      ms(t_compute_start, t_submit_end), ms(t_submit_end, t_sync_end),
      ms(t_sync_end, t_output_end), ms(t_output_end, t_state_end),
      ms(t_compute_start, t_state_end));
  engine.qwen4exp_full_graph_output_ready = true;
  return true;
}

#else

bool qwen4exp_full_graph_enabled() { return false; }
int qwen4exp_full_graph_device_expert_layers() { return 0; }
bool qwen4exp_full_graph_prepare(StreamingEngine &, int, int, const int *) {
  return false;
}
bool qwen4exp_full_graph_run(StreamingEngine &, float *, int, int,
                             const int *) {
  return false;
}
void qwen4exp_full_graph_shutdown() {}

#endif
