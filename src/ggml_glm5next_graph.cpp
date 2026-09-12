// GLM-5.3-Flash (glm5next) full prefill graph.
//
// The legacy engine executes one layer at a time and returns the hidden state
// to C++ between layers.  This bridge keeps the complete GLM trunk in one
// GGML graph.  Host backed expert tensors remain valid GGML weight nodes, so
// the CUDA scheduler owns the required copies and dependency edges.

#include "ggml_glm5next_graph.hpp"
#include "ggml_deepseek_v4_graph.hpp"

#include "dense_dispatch.hpp"
#include "direct_io.hpp"
#include "engine.hpp"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "ggml_moe_backend.hpp"
#include "kquant_dequant.hpp"
#include "logger.hpp"
#include "memory_guard.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#if defined(MOESTREAM_USE_GGML_CUDA)

#include <cuda_runtime.h>

namespace {

constexpr int kGraphSize = 60000;
constexpr size_t kTensorContextBytes = 192ull * 1024ull * 1024ull;

struct HostAllocation {
  void *ptr = nullptr;
  size_t bytes = 0;
  bool pinned = false;
  size_t reservation_bytes = 0;
};

// A graph can keep historically hot experts in a compact CUDA tensor while
// presenting the remaining experts as a compact host tensor.  Device maps
// translate the model's original expert IDs, so routing stays inside GGML.
struct PartialExpertLayer {
  std::vector<int> hot;
  std::vector<int> cold;
  ggml_tensor *hot_gate_up = nullptr;
  ggml_tensor *hot_gate = nullptr;
  ggml_tensor *hot_up = nullptr;
  ggml_tensor *hot_down = nullptr;
  ggml_tensor *cold_map = nullptr;
  ggml_tensor *hot_map = nullptr;
  ggml_tensor *cold_mask = nullptr;
  ggml_tensor *hot_mask = nullptr;
};

struct FullGraph {
  StreamingEngine *engine = nullptr;
  int device = -1;
  int S = 0;
  int D = 0;
  int HC = 0;
  int n_layers = 0;
  int H = 0;
  int hd = 0;
  int d_inner = 0;
  int d_conv = 0;
  int qk_nope = 0;
  int qk_rope = 0;
  int kv_lora = 0;
  int v_head = 0;
  int o_group_count = 0;
  int o_lora_rank = 0;
  int sliding_window = 0;
  int first_dense = 0;
  int device_expert_layers = 0;
  bool partial_expert_residency = false;
  bool partial_sources_compacted = false;
  size_t partial_hot_bytes = 0;
  size_t partial_cold_bytes = 0;
  size_t partial_hot_records = 0;
  std::string partial_usage_file;
  std::vector<PartialExpertLayer> partial_experts;

  ggml_backend_t cuda = nullptr;
  ggml_backend_t cpu = nullptr;
  ggml_backend_sched_t sched = nullptr;
  ggml_backend_buffer_t dense_buffer = nullptr;
  void *dense_base = nullptr;
  size_t dense_capacity = 0;
  size_t dense_used = 0;
  int dense_tensors = 0;
  ggml_backend_buffer_t expert_buffer = nullptr;
  void *expert_base = nullptr;
  size_t expert_capacity = 0;
  size_t expert_used = 0;
  ggml_context *ctx = nullptr;
  ggml_cgraph *graph = nullptr;

  ggml_tensor *input = nullptr;  // [D, HC, S], host input
  ggml_tensor *output = nullptr; // [D, S], after HC mean; LM head norms
  ggml_tensor *final_streams = nullptr;
  ggml_tensor *attn_mask = nullptr; // [S, S, 1, 1], F16 causal mask
  ggml_tensor *positions = nullptr; // [S], I32 token positions
  ggml_tensor *positions_mod4 = nullptr;
  ggml_tensor *positions_mod128 = nullptr;
  ggml_tensor *block_positions4 = nullptr;
  ggml_tensor *block_positions128 = nullptr;
  ggml_tensor *csa_zero_values = nullptr;
  ggml_tensor *csa_zero_scores = nullptr;

  struct Dsv4LayerTensors {
    int layer = -1;
    int ratio = 0;
    ggml_tensor *raw_kv = nullptr;             // [qh, S], rotated K==V
    ggml_tensor *comp_kv = nullptr;            // [width, S]
    ggml_tensor *comp_score = nullptr;         // [width, S], incl. APE
    ggml_tensor *indexer_comp_kv = nullptr;    // CSA only
    ggml_tensor *indexer_comp_score = nullptr; // CSA only
    ggml_tensor *keys = nullptr;               // [qh, floor(S/ratio)]
    ggml_tensor *indexer_keys = nullptr;       // CSA only
    ggml_tensor *mask = nullptr;               // [SWA + compressed + sink, S]
  };
  std::vector<Dsv4LayerTensors> dsv4_layers;
  std::vector<ggml_tensor *> dsv4_layer_outputs;
  std::vector<std::pair<std::string, ggml_tensor *>> dsv4_stage_outputs;

  std::vector<ggml_tensor *> kda_state_out;
  std::vector<ggml_tensor *> kda_hist_out;
  std::vector<int> kda_layers;
  std::vector<float *> kda_state_inputs;
  std::vector<float *> kda_hist_inputs;
  std::vector<ggml_tensor *> dsa_kv_out;
  std::vector<int> dsa_layers;

  std::vector<HostAllocation> owned;
  std::vector<ggml_backend_buffer_t> host_buffers;
  std::vector<void *> host_inputs;

  size_t host_weight_bytes = 0;
  size_t host_pinned_bytes = 0;
  size_t host_pin_failures = 0;
  // Routed projections are kept in one graph-owned DDR arena.  Keeping the
  // 126 large projections as individual malloc/cudaHostRegister allocations
  // made graph preparation spend minutes in allocator/page registration and
  // made the host RSS curve needlessly jagged.  The GGML descriptors still
  // point at each projection's own slice, so MUL_MAT_ID sees the same layout.
  void *expert_source_base = nullptr;
  size_t expert_source_capacity = 0;
  size_t expert_source_used = 0;
  size_t expert_source_tensors = 0;
  std::unordered_map<std::string, void *> expert_source_views;
  std::unordered_map<std::string, size_t> expert_source_view_bytes;
  // Partial compaction is in-place, but hot uploads still need the original
  // full-source addresses after the cold views have been rewritten.
  std::unordered_map<std::string, void *> expert_source_full_views;
  std::unordered_map<std::string, size_t> expert_source_full_view_bytes;
  // Expert source slices are registered individually rather than registering
  // the complete 67--78 GiB arena.  The latter can exceed this host's CUDA
  // host-registration ceiling and would leave the whole graph on pageable
  // memory.  Each entry here is a non-owning range inside expert_source_base;
  // destroy() unregisters it before the arena is freed.
  std::vector<HostAllocation> expert_source_pinned_ranges;
  size_t host_reservation_bytes = 0;
  // ggml_init(no_alloc=true) still owns a host-side context arena.  Keep it
  // in the same admission ledger as the graph's explicit host buffers; the
  // old watchdog only saw the latter and could therefore under-report the
  // graph's host peak during construction.
  size_t context_reservation_bytes = 0;
  size_t graph_workspace_bytes = 0;
  bool ready = false;

  bool has_expert_source() const {
    return expert_source_base != nullptr && expert_source_capacity != 0;
  }

  // Transfer only the source-side state to a graph with a different prompt
  // shape.  The graph context/scheduler and its prompt-sized tensors must be
  // rebuilt, while the large DDR expert arena must remain alive.  Partial
  // residency's tensor pointers belong to the old GGML context, so preserve
  // only its layer/expert lists and clear those pointers before the old graph
  // is destroyed.
  void move_expert_source_to(FullGraph &dst) {
    dst.expert_source_base = expert_source_base;
    dst.expert_source_capacity = expert_source_capacity;
    dst.expert_source_used = expert_source_used;
    dst.expert_source_tensors = expert_source_tensors;
    dst.expert_source_views = std::move(expert_source_views);
    dst.expert_source_view_bytes = std::move(expert_source_view_bytes);
    dst.expert_source_full_views = std::move(expert_source_full_views);
    dst.expert_source_full_view_bytes =
        std::move(expert_source_full_view_bytes);
    dst.expert_source_pinned_ranges = std::move(expert_source_pinned_ranges);
    dst.host_reservation_bytes = host_reservation_bytes;

    dst.partial_expert_residency = partial_expert_residency;
    dst.partial_sources_compacted = partial_sources_compacted;
    dst.partial_hot_bytes = partial_hot_bytes;
    dst.partial_cold_bytes = partial_cold_bytes;
    dst.partial_hot_records = partial_hot_records;
    dst.partial_usage_file = std::move(partial_usage_file);
    dst.partial_experts = std::move(partial_experts);
    for (PartialExpertLayer &p : dst.partial_experts) {
      p.hot_gate_up = nullptr;
      p.hot_gate = nullptr;
      p.hot_up = nullptr;
      p.hot_down = nullptr;
      p.cold_map = nullptr;
      p.hot_map = nullptr;
      p.cold_mask = nullptr;
      p.hot_mask = nullptr;
    }

    expert_source_base = nullptr;
    expert_source_capacity = 0;
    expert_source_used = 0;
    expert_source_tensors = 0;
    expert_source_pinned_ranges.clear();
    host_reservation_bytes = 0;
    partial_expert_residency = false;
    partial_sources_compacted = false;
    partial_hot_bytes = 0;
    partial_cold_bytes = 0;
    partial_hot_records = 0;
    partial_usage_file.clear();
    partial_experts.clear();
  }

  ~FullGraph() { destroy(); }

  void destroy() {
    if (sched) {
      ggml_backend_sched_free(sched);
      sched = nullptr;
    }
    if (cpu) {
      ggml_backend_free(cpu);
      cpu = nullptr;
    }
    if (expert_buffer) {
      ggml_backend_buffer_free(expert_buffer);
      expert_buffer = nullptr;
    }
    if (dense_buffer) {
      ggml_backend_buffer_free(dense_buffer);
      dense_buffer = nullptr;
    }
    partial_expert_residency = false;
    partial_sources_compacted = false;
    partial_hot_bytes = 0;
    partial_cold_bytes = 0;
    partial_hot_records = 0;
    partial_usage_file.clear();
    partial_experts.clear();
    for (ggml_backend_buffer_t b : host_buffers)
      if (b)
        ggml_backend_buffer_free(b);
    host_buffers.clear();
    if (ctx) {
      ggml_free(ctx);
      ctx = nullptr;
    }
    for (HostAllocation &a : owned) {
      if (a.pinned)
        cudaHostUnregister(a.ptr);
      std::free(a.ptr);
      MemoryGuard::release_host_bytes(a.reservation_bytes);
    }
    owned.clear();
    host_inputs.clear();
    expert_source_base = nullptr;
    expert_source_capacity = 0;
    expert_source_used = 0;
    expert_source_tensors = 0;
    expert_source_views.clear();
    expert_source_view_bytes.clear();
    expert_source_full_views.clear();
    expert_source_full_view_bytes.clear();
    for (const HostAllocation &a : expert_source_pinned_ranges)
      if (a.pinned)
        cudaHostUnregister(a.ptr);
    expert_source_pinned_ranges.clear();
    cuda = nullptr;
    graph = nullptr;
    input = output = final_streams = attn_mask = positions = nullptr;
    positions_mod4 = positions_mod128 = block_positions4 = block_positions128 =
        nullptr;
    csa_zero_values = csa_zero_scores = nullptr;
    kda_state_out.clear();
    kda_hist_out.clear();
    kda_layers.clear();
    kda_state_inputs.clear();
    kda_hist_inputs.clear();
    dsa_kv_out.clear();
    dsa_layers.clear();
    dsv4_layers.clear();
    dsv4_stage_outputs.clear();
    MemoryGuard::release_host_bytes(host_reservation_bytes);
    host_reservation_bytes = 0;
    MemoryGuard::release_host_bytes(context_reservation_bytes);
    context_reservation_bytes = 0;
    engine = nullptr;
    ready = false;
  }
};

std::unique_ptr<FullGraph> g_graph;
// The large host expert source is independent of prompt length.  Keep it in
// a source-only container while the prompt-shaped GGML graph is rebuilt or
// before the first prompt arrives.
std::unique_ptr<FullGraph> g_glm5next_source;

bool env_enabled() {
  static const bool enabled = [] {
    const char *v = std::getenv("MOESTREAM_GLM5NEXT_FULL_GRAPH");
    return v && v[0] == '1';
  }();
  return enabled;
}

int env_device_expert_layers() {
  const char *v = std::getenv("MOESTREAM_GLM5NEXT_GRAPH_PRELOAD_LAYERS");
  if (!v || !v[0])
    return 0;
  char *end = nullptr;
  long n = std::strtol(v, &end, 10);
  if (end == v || *end != '\0' || n <= 0)
    return 0;
  return static_cast<int>(std::min<long>(n, 45));
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

const GGUFTensor *model_tensor(const StreamingEngine &engine,
                               const std::string &name) {
  auto it = engine.model_file.tensors.find(name);
  return it == engine.model_file.tensors.end() ? nullptr : &it->second;
}

const QuantizedTensor *resident(const StreamingEngine &engine,
                                const std::string &name) {
  auto it = engine.resident_layers.find(name);
  if (it == engine.resident_layers.end() || !it->second.host_ptr)
    return nullptr;
  return &it->second;
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
  if (bytes == 0 || !MemoryGuard::try_reserve_host_bytes(bytes)) {
    if (bytes != 0)
      LOG_WARN(
          "[GGML-GLM5NEXT-FULL-MEM] host graph allocation refused: %.2f MB "
          "(reserved=%.2f GB, budget=%.2f GB)\n",
          bytes / 1e6, MemoryGuard::reserved_host_bytes() / 1e9,
          MemoryGuard::host_budget_bytes() / 1e9);
    return nullptr;
  }
  const int err = posix_memalign(&p, 4096, bytes);
  if (err != 0 || !p) {
    MemoryGuard::release_host_bytes(bytes);
    return nullptr;
  }
  g.owned.push_back({p, bytes, false, bytes});
  return p;
}

void *alloc_expert_source(FullGraph &g, size_t bytes) {
  if (!g.expert_source_base || bytes == 0)
    return nullptr;
  constexpr size_t kAlignment = 4096;
  const size_t aligned =
      (g.expert_source_used + kAlignment - 1) & ~(kAlignment - 1);
  if (aligned > g.expert_source_capacity ||
      bytes > g.expert_source_capacity - aligned)
    return nullptr;
  void *p = static_cast<uint8_t *>(g.expert_source_base) + aligned;
  g.expert_source_used = aligned + bytes;
  ++g.expert_source_tensors;
  return p;
}

ggml_tensor *host_input(FullGraph &g, ggml_context *ctx, ggml_type type,
                        const std::vector<int64_t> &shape, size_t bytes,
                        const char *name) {
  ggml_tensor *t = new_tensor(ctx, type, shape);
  if (!t)
    return nullptr;
  const size_t required = std::max(bytes, ggml_nbytes(t));
  void *p = alloc_host(g, required);
  if (!p || !bind_host(g, t, p, required, GGML_BACKEND_BUFFER_USAGE_COMPUTE))
    return nullptr;
  ggml_set_input(t);
  ggml_set_name(t, name);
  g.host_inputs.push_back(p);
  return t;
}

bool graph_dense_weight_name(const std::string &name) {
  if (name.find("_exps.weight") != std::string::npos ||
      name.find("indexer") != std::string::npos ||
      name.find("nextn") != std::string::npos)
    return false;
  return name.find("attn") != std::string::npos ||
         name.find("hc_") != std::string::npos ||
         name.find("ssm_") != std::string::npos ||
         name.find("norm") != std::string::npos ||
         name.find("gate_inp") != std::string::npos ||
         name.find("shexp") != std::string::npos ||
         name.find("ffn_gate.weight") != std::string::npos ||
         name.find("ffn_up.weight") != std::string::npos ||
         name.find("ffn_down.weight") != std::string::npos ||
         name.find("exp_probs_b.bias") != std::string::npos;
}

bool arena_alloc(ggml_backend_buffer_t buffer, void *base, size_t capacity,
                 size_t &used, ggml_tensor *t, const void *src,
                 size_t src_bytes) {
  if (!buffer || !base || !t || !src || src_bytes < ggml_nbytes(t))
    return false;
  const auto buft = ggml_backend_buffer_get_type(buffer);
  const size_t alloc_size = ggml_backend_buft_get_alloc_size(buft, t);
  const size_t align =
      std::max<size_t>(256, ggml_backend_buft_get_alignment(buft));
  const size_t off = (used + align - 1) & ~(align - 1);
  if (off > capacity || alloc_size > capacity - off)
    return false;
  void *dst = static_cast<uint8_t *>(base) + off;
  if (ggml_backend_tensor_alloc(buffer, t, dst) != GGML_STATUS_SUCCESS)
    return false;
  if (cudaMemcpy(dst, src, src_bytes, cudaMemcpyHostToDevice) != cudaSuccess)
    return false;
  used = off + alloc_size;
  return true;
}

// The V4 graph builder is kept beside the shared arena helpers, while the
// existing GLM builder's tensor/FFN helpers are defined below it.
ggml_tensor *resident_tensor(FullGraph &g, ggml_context *ctx,
                             const std::string &name);
ggml_tensor *build_ffn(FullGraph &g, ggml_context *ctx, ggml_tensor *cur,
                       int layer, ggml_tensor *forced_ids);
ggml_tensor *hc_pre(FullGraph &g, ggml_context *ctx, ggml_tensor *x,
                    ggml_tensor *fn, ggml_tensor *scale, ggml_tensor *base,
                    ggml_tensor **post, ggml_tensor **comb);
ggml_tensor *hc_post(FullGraph &g, ggml_context *ctx, ggml_tensor *x,
                     ggml_tensor *residual, ggml_tensor *post,
                     ggml_tensor *comb);

// ---------------------------------------------------------------------------
// DeepSeek-V4 full prefill graph.
//
// V4 cannot reuse GLM5NEXT's KDA/absorbed-MLA builder: its real Flash file
// has raw SWA attention plus ratio-4 CSA and ratio-128 HCA layers.  The
// builder below keeps the same graph-owned dense/expert arenas and HC/MoE
// nodes, but expresses V4's compressed keys and attention sinks as GGML
// nodes.  For the normal Flash prompt sizes (well below CSA top-k * 4), all
// completed compressed blocks are visible, so the indexer only needs to be
// carried forward for the subsequent decode path; no host top-k loop is
// inserted into the prefill graph.
// ---------------------------------------------------------------------------

bool dsv4_env_enabled() {
  static const bool enabled = [] {
    const char *v = std::getenv("MOESTREAM_DEEPSEEK_V4_FULL_GRAPH");
    return v && v[0] == '1';
  }();
  return enabled;
}

bool dsv4_graph_diagnostic_enabled() {
  const char *v = std::getenv("MOESTREAM_DEEPSEEK_V4_GRAPH_DIAGNOSTIC");
  return v && v[0] == '1';
}

int dsv4_env_device_expert_layers() {
  const char *v = std::getenv("MOESTREAM_DEEPSEEK_V4_GRAPH_PRELOAD_LAYERS");
  if (!v || !v[0])
    return 0;
  char *end = nullptr;
  long n = std::strtol(v, &end, 10);
  if (end == v || *end != '\0' || n <= 0)
    return 0;
  return static_cast<int>(std::min<long>(n, 43));
}

ggml_tensor *dsv4_host_i32(FullGraph &g, ggml_context *ctx, int n,
                           const char *name, int32_t **out) {
  ggml_tensor *t =
      host_input(g, ctx, GGML_TYPE_I32, {n}, (size_t)n * sizeof(int32_t), name);
  if (t && out)
    *out = static_cast<int32_t *>(t->data);
  return t;
}

ggml_tensor *dsv4_host_i32_2d(FullGraph &g, ggml_context *ctx, int ne0, int ne1,
                              const char *name, int32_t **out) {
  ggml_tensor *t = host_input(g, ctx, GGML_TYPE_I32, {ne0, ne1},
                              (size_t)ne0 * ne1 * sizeof(int32_t), name);
  if (t && out)
    *out = static_cast<int32_t *>(t->data);
  return t;
}

ggml_tensor *dsv4_host_f32(FullGraph &g, ggml_context *ctx,
                           const std::vector<int64_t> &shape, const char *name,
                           float **out) {
  size_t n = 1;
  for (int64_t d : shape)
    n *= (size_t)d;
  ggml_tensor *t =
      host_input(g, ctx, GGML_TYPE_F32, shape, n * sizeof(float), name);
  if (t && out)
    *out = static_cast<float *>(t->data);
  return t;
}

struct Dsv4RopeParams {
  float theta = 10000.0f;
  float freq_scale = 1.0f;
  float ext_factor = 0.0f;
  float attn_factor = 1.0f;
  int n_ctx_orig = 0;
};

Dsv4RopeParams dsv4_rope_params(const FullGraph &g, int ratio) {
  const auto &c = g.engine->model_file.architecture_config;
  Dsv4RopeParams p;
  p.theta = ratio == 0 ? g.engine->model_file.get_config_float("rope.freq_base",
                                                               10000.0f)
                       : c.dsv4_compress_rope_freq_base;
  if (ratio == 0)
    return p;
  p.freq_scale = c.rope_freq_scale;
  p.ext_factor = c.rope_ext_factor;
  p.n_ctx_orig = c.rope_n_ctx_orig;
  // ggml's YaRN kernel multiplies the supplied attention factor by
  // (1+0.1*log(1/freq_scale)) when ext_factor is enabled.  DeepSeek-V4's
  // dsv4_mscale intentionally divides by that term instead, so pre-adjust
  // the argument to make the GGML operation produce the V4 value.
  if (p.ext_factor != 0.0f && p.freq_scale > 0.0f) {
    const float f = 1.0f + 0.1f * std::log(1.0f / p.freq_scale);
    p.attn_factor = c.rope_attn_factor / (f * f);
  } else {
    p.attn_factor = c.rope_attn_factor;
  }
  return p;
}

ggml_tensor *dsv4_rope_tail(FullGraph &g, ggml_context *ctx, ggml_tensor *x,
                            int nope, int rope, ggml_tensor *positions,
                            int ratio, bool inverse) {
  if (rope <= 0)
    return x;
  const Dsv4RopeParams p = dsv4_rope_params(g, ratio);
  ggml_tensor *head =
      ggml_view_3d(ctx, x, nope, x->ne[1], x->ne[2], x->nb[1], x->nb[2], 0);
  ggml_tensor *tail = ggml_view_3d(ctx, x, rope, x->ne[1], x->ne[2], x->nb[1],
                                   x->nb[2], (size_t)nope * x->nb[0]);
  ggml_tensor *rot =
      inverse ? ggml_rope_ext_back(ctx, tail, positions, nullptr, rope,
                                   GGML_ROPE_TYPE_NORMAL, p.n_ctx_orig, p.theta,
                                   p.freq_scale, p.ext_factor, p.attn_factor,
                                   32.0f, 1.0f)
              : ggml_rope_ext(ctx, tail, positions, nullptr, rope,
                              GGML_ROPE_TYPE_NORMAL, p.n_ctx_orig, p.theta,
                              p.freq_scale, p.ext_factor, p.attn_factor, 32.0f,
                              1.0f);
  return ggml_concat(ctx, head, rot, 0);
}

ggml_tensor *dsv4_compressed_keys(FullGraph &g, ggml_context *ctx,
                                  ggml_tensor *comp_kv, ggml_tensor *comp_score,
                                  ggml_tensor *comp_ape, ggml_tensor *comp_norm,
                                  int ratio, int key_dim, bool csa,
                                  ggml_tensor *mod_positions,
                                  ggml_tensor *block_positions) {
  const int blocks = g.S / ratio;
  if (blocks <= 0)
    return nullptr;
  const int width = csa ? 2 * key_dim : key_dim;
  const int n_read = blocks * ratio;

  // Match Llama.cpp's state compressor.  The IDs are block-major: for HCA
  // they select every prompt row once; for CSA they select the previous
  // four-row window for every block followed by the current four-row window.
  // This makes the following softmax operate over positions (the first
  // dimension after permute), rather than over key channels.
  const int n_indices = csa ? 2 * n_read : n_read;
  int32_t *index_data = nullptr;
  ggml_tensor *state_read_idxs = dsv4_host_i32(g, ctx, n_indices,
                                               csa ? "dsv4_graph_csa_read_idxs"
                                                   : "dsv4_graph_hca_read_idxs",
                                               &index_data);
  if (!state_read_idxs)
    return nullptr;
  if (!csa) {
    for (int i = 0; i < n_read; ++i)
      index_data[i] = i;
  } else {
    for (int b = 0; b < blocks; ++b) {
      for (int i = 0; i < ratio; ++i) {
        const int j = b * ratio + i;
        // The appended row at index S is the zero/-inf history used for the
        // first CSA block, exactly as dsv4_append_zero_row() in Bee.
        index_data[j] = b == 0 ? g.S : (b - 1) * ratio + i;
        index_data[n_read + j] = j;
      }
    }
  }

  ggml_tensor *kv_source = comp_kv;
  ggml_tensor *score_source = comp_score;
  if (csa) {
    // Use a graph-derived row so this works for both the normal qh-wide CSA
    // compressor and the narrower indexer compressor.  It avoids a separate
    // fixed-width host zero buffer and is the same value-wise operation as
    // Bee's dsv4_append_zero_row().
    ggml_tensor *kv_zero = ggml_reshape_2d(
        ctx, ggml_scale(ctx, ggml_view_1d(ctx, comp_kv, width, 0), 0.0f), width,
        1);
    ggml_tensor *score_zero = ggml_reshape_2d(
        ctx,
        ggml_scale_bias(ctx, ggml_view_1d(ctx, comp_score, width, 0), 0.0f,
                        -INFINITY),
        width, 1);
    kv_source = ggml_concat(ctx, comp_kv, kv_zero, 1);
    score_source = ggml_concat(ctx, comp_score, score_zero, 1);
  }

  ggml_tensor *kv_rows = ggml_get_rows(ctx, kv_source, state_read_idxs);
  ggml_tensor *score_rows = ggml_get_rows(ctx, score_source, state_read_idxs);
  ggml_tensor *values = nullptr;
  ggml_tensor *scores = nullptr;
  if (!csa) {
    ggml_tensor *kv = ggml_reshape_3d(ctx, kv_rows, key_dim, ratio, blocks);
    ggml_tensor *score =
        ggml_reshape_3d(ctx, score_rows, key_dim, ratio, blocks);
    values = ggml_cont(ctx, ggml_permute(ctx, kv, 1, 0, 2, 3));
    scores = ggml_cont(ctx, ggml_permute(ctx, score, 1, 0, 2, 3));
  } else {
    // Each row has [previous/current] channels.  Split the gathered rows
    // before concatenating along the position dimension, as in Bee's
    // build_overlap_compressed_kv_from_state().
    ggml_tensor *kv_prev = ggml_cont(
        ctx, ggml_view_2d(ctx, kv_rows, key_dim, n_read, kv_rows->nb[1], 0));
    ggml_tensor *score_prev =
        ggml_cont(ctx, ggml_view_2d(ctx, score_rows, key_dim, n_read,
                                    score_rows->nb[1], 0));
    const size_t second_half =
        (size_t)n_read * kv_rows->nb[1] + ggml_row_size(kv_rows->type, key_dim);
    const size_t second_score_half = (size_t)n_read * score_rows->nb[1] +
                                     ggml_row_size(score_rows->type, key_dim);
    ggml_tensor *kv_cur =
        ggml_cont(ctx, ggml_view_2d(ctx, kv_rows, key_dim, n_read,
                                    kv_rows->nb[1], second_half));
    ggml_tensor *score_cur =
        ggml_cont(ctx, ggml_view_2d(ctx, score_rows, key_dim, n_read,
                                    score_rows->nb[1], second_score_half));
    ggml_tensor *kv =
        ggml_concat(ctx, ggml_reshape_3d(ctx, kv_prev, key_dim, ratio, blocks),
                    ggml_reshape_3d(ctx, kv_cur, key_dim, ratio, blocks), 1);
    ggml_tensor *score = ggml_concat(
        ctx, ggml_reshape_3d(ctx, score_prev, key_dim, ratio, blocks),
        ggml_reshape_3d(ctx, score_cur, key_dim, ratio, blocks), 1);
    values = ggml_cont(ctx, ggml_permute(ctx, kv, 1, 0, 2, 3));
    scores = ggml_cont(ctx, ggml_permute(ctx, score, 1, 0, 2, 3));
  }

  ggml_tensor *weights = ggml_soft_max(ctx, scores);
  ggml_tensor *comp = ggml_mul(ctx, values, weights);
  comp = ggml_sum_rows(ctx, comp);
  comp = ggml_cont(ctx, ggml_permute(ctx, comp, 1, 0, 2, 3));
  comp = ggml_mul(
      ctx,
      ggml_rms_norm(ctx, comp, g.engine->model_file.architecture_config.eps),
      comp_norm);
  comp = dsv4_rope_tail(g, ctx, comp, key_dim - g.qk_rope, g.qk_rope,
                        block_positions, ratio, false);
  (void)mod_positions;
  (void)comp_ape;
  return ggml_cont_3d(ctx, comp, key_dim, 1, blocks);
}

ggml_tensor *build_dsv4_attention(FullGraph &g, ggml_context *ctx,
                                  ggml_tensor *cur, int layer) {
  const auto &c = g.engine->model_file.architecture_config;
  const std::string p = "blk." + std::to_string(layer) + ".";
  const int qh = g.qk_nope + g.qk_rope;
  const int ratio =
      c.dsv4_compress_ratios.empty()
          ? 0
          : static_cast<int>(c.dsv4_compress_ratios[(size_t)layer]);
  const bool csa = ratio == 4;
  FullGraph::Dsv4LayerTensors lt;
  lt.layer = layer;
  lt.ratio = ratio;

  ggml_tensor *wqa = resident_tensor(g, ctx, p + "attn_q_a.weight");
  ggml_tensor *wqan = resident_tensor(g, ctx, p + "attn_q_a_norm.weight");
  ggml_tensor *wqb = resident_tensor(g, ctx, p + "attn_q_b.weight");
  ggml_tensor *wkv = resident_tensor(g, ctx, p + "attn_kv.weight");
  ggml_tensor *wkvn = resident_tensor(g, ctx, p + "attn_kv_a_norm.weight");
  ggml_tensor *woa = resident_tensor(g, ctx, p + "attn_output_a.weight");
  ggml_tensor *wob = resident_tensor(g, ctx, p + "attn_output_b.weight");
  if (!wqa || !wqan || !wqb || !wkv || !wkvn || !woa || !wob)
    return nullptr;

  ggml_tensor *q_lora = ggml_mul_mat(ctx, wqa, cur);
  q_lora = ggml_mul(ctx, ggml_rms_norm(ctx, q_lora, c.eps), wqan);
  ggml_tensor *q =
      ggml_reshape_3d(ctx, ggml_mul_mat(ctx, wqb, q_lora), qh, g.H, g.S);
  q = ggml_rms_norm(ctx, q, c.eps); // V4's weightless per-head Q norm
  q = dsv4_rope_tail(g, ctx, q, g.qk_nope, g.qk_rope, g.positions, ratio,
                     false);

  ggml_tensor *kv =
      ggml_reshape_3d(ctx, ggml_mul_mat(ctx, wkv, cur), qh, 1, g.S);
  kv = ggml_mul(ctx, ggml_rms_norm(ctx, kv, c.eps), wkvn);
  kv = dsv4_rope_tail(g, ctx, kv, g.qk_nope, g.qk_rope, g.positions, ratio,
                      false);
  lt.raw_kv = kv;

  ggml_tensor *compressed = nullptr;
  int ncompressed = 0;
  if (ratio != 0) {
    ggml_tensor *ckv = resident_tensor(g, ctx, p + "attn_compressor_kv.weight");
    ggml_tensor *cgate =
        resident_tensor(g, ctx, p + "attn_compressor_gate.weight");
    ggml_tensor *cape =
        resident_tensor(g, ctx, p + "attn_compressor_ape.weight");
    ggml_tensor *cnorm =
        resident_tensor(g, ctx, p + "attn_compressor_norm.weight");
    if (!ckv || !cgate || !cape || !cnorm)
      return nullptr;
    const int width = csa ? 2 * qh : qh;
    lt.comp_kv = ggml_mul_mat(ctx, ckv, cur);
    ggml_tensor *gate = ggml_mul_mat(ctx, cgate, cur);
    ggml_tensor *ape_rows =
        ggml_get_rows(ctx, cape, csa ? g.positions_mod4 : g.positions_mod128);
    lt.comp_score = ggml_add(ctx, gate, ape_rows);
    compressed = dsv4_compressed_keys(
        g, ctx, lt.comp_kv, lt.comp_score, cape, cnorm, ratio, qh, csa,
        csa ? g.positions_mod4 : g.positions_mod128,
        csa ? g.block_positions4 : g.block_positions128);
    ncompressed = g.S / ratio;
    lt.keys = compressed;

    if (csa) {
      ggml_tensor *ikv =
          resident_tensor(g, ctx, p + "indexer_compressor_kv.weight");
      ggml_tensor *igate =
          resident_tensor(g, ctx, p + "indexer_compressor_gate.weight");
      ggml_tensor *iape =
          resident_tensor(g, ctx, p + "indexer_compressor_ape.weight");
      ggml_tensor *inorm =
          resident_tensor(g, ctx, p + "indexer_compressor_norm.weight");
      if (!ikv || !igate || !iape || !inorm)
        return nullptr;
      const int idim = c.dsv4_indexer_key_length;
      lt.indexer_comp_kv = ggml_mul_mat(ctx, ikv, cur);
      lt.indexer_comp_score =
          ggml_add(ctx, ggml_mul_mat(ctx, igate, cur),
                   ggml_get_rows(ctx, iape, g.positions_mod4));
      lt.indexer_keys = dsv4_compressed_keys(
          g, ctx, lt.indexer_comp_kv, lt.indexer_comp_score, iape, inorm, 4,
          idim, true, g.positions_mod4, g.block_positions4);
    }
  }

  ggml_tensor *seq = kv;
  if (compressed) {
    seq = ggml_concat(ctx, seq,
                      ggml_reshape_3d(ctx, compressed, qh, 1, ncompressed), 2);
  }
  const int key_count = g.S + ncompressed;

  // Use GGML's native attention-sink input.  Appending a synthetic K/V
  // dimension would change V4's qh=512 head width to 513, which disables the
  // CUDA FATTN kernel on RTX 3090 and sends every attention layer to the CPU.
  ggml_tensor *sink_tensor = resident_tensor(g, ctx, p + "attn_sinks.weight");
  if (!sink_tensor || sink_tensor->type != GGML_TYPE_F32 ||
      sink_tensor->ne[0] != (int64_t)g.H)
    return nullptr;
  // FATTN's CUDA GQA path requires K->ne[1] to be a multiple of 256 for
  // qh=512.  Pad with zero K/V rows and mask those rows out.  The padding is
  // a graph-only workspace shape choice; it never enters the cache state.
  const int padded_key_count = ((key_count + 255) / 256) * 256;
  ggml_tensor *seq_padded = seq;
  if (padded_key_count > key_count) {
    float *pad_data = nullptr;
    ggml_tensor *pad =
        dsv4_host_f32(g, ctx, {qh, 1, padded_key_count - key_count},
                      "dsv4_graph_attention_padding", &pad_data);
    if (!pad)
      return nullptr;
    std::fill(pad_data, pad_data + (size_t)qh * (padded_key_count - key_count),
              0.0f);
    seq_padded = ggml_concat(ctx, seq, pad, 2);
  }
  ggml_tensor *real =
      ggml_repeat_4d(ctx, seq_padded, qh, g.H, padded_key_count, 1);

  float *mask_data = nullptr;
  ggml_tensor *mask =
      dsv4_host_f32(g, ctx, {padded_key_count, g.S, 1, 1},
                    "dsv4_graph_attention_mask_f32", &mask_data);
  if (!mask)
    return nullptr;
  const int window = g.sliding_window > 0 ? g.sliding_window : g.S;
  for (int qpos = 0; qpos < g.S; ++qpos) {
    for (int kpos = 0; kpos < padded_key_count; ++kpos) {
      bool visible = false;
      if (kpos < key_count && kpos < g.S)
        visible = kpos <= qpos && kpos >= std::max(0, qpos - window + 1);
      else if (kpos < key_count)
        visible = (kpos - g.S) < (qpos + 1) / std::max(1, ratio);
      mask_data[(size_t)qpos * padded_key_count + kpos] =
          visible ? 0.0f : -INFINITY;
    }
  }
  // FATTN's CUDA path expects an F16 additive mask.  Keep the host source as
  // F32 while constructing it, then cast once as part of the graph.
  ggml_tensor *mask_f16 = ggml_cast(ctx, mask, GGML_TYPE_F16);
  lt.mask = mask_f16;

  ggml_tensor *q_attn =
      ggml_permute(ctx, ggml_reshape_4d(ctx, q, qh, g.H, g.S, 1), 0, 2, 1, 3);
  ggml_tensor *k_attn = ggml_permute(ctx, real, 0, 2, 1, 3);
  ggml_tensor *v_attn = ggml_permute(ctx, real, 0, 2, 1, 3);
  if (k_attn->type == GGML_TYPE_F32)
    k_attn = ggml_cast(ctx, k_attn, GGML_TYPE_F16);
  if (v_attn->type == GGML_TYPE_F32)
    v_attn = ggml_cast(ctx, v_attn, GGML_TYPE_F16);
  ggml_tensor *attn =
      ggml_flash_attn_ext(ctx, q_attn, k_attn, v_attn, mask_f16,
                          1.0f / std::sqrt((float)qh), 0.0f, 0.0f);
  ggml_flash_attn_ext_add_sinks(attn, sink_tensor);
  ggml_tensor *value = ggml_permute(ctx, attn, 0, 2, 1, 3);
  ggml_tensor *context =
      ggml_cont(ctx, ggml_view_3d(ctx, value, qh, g.H, g.S, value->nb[1],
                                  value->nb[2], 0));
  context = dsv4_rope_tail(g, ctx, context, g.qk_nope, g.qk_rope, g.positions,
                           ratio, true);

  const int groups = std::max(1, c.o_group_count);
  const int rank = c.o_lora_rank;
  const int group_in = (qh * g.H) / groups;
  if (rank <= 0 || groups <= 0 || group_in * groups != qh * g.H ||
      woa->ne[0] != group_in || woa->ne[1] != (int64_t)(groups * rank))
    return nullptr;
  // Keep the grouped output projection as one batched GGML matmul.  This is
  // Llama.cpp's layout: [group_in, groups, tokens] is permuted to
  // [group_in, tokens, groups], projected by [group_in, rank, groups], then
  // restored and made contiguous for wo_b.  The previous loop emitted one
  // matmul and concat per group, creating eight extra graph boundaries.
  ggml_tensor *woa3 = ggml_reshape_3d(ctx, woa, group_in, rank, groups);
  // context is already Bee's [head_dim, n_head, tokens] tensor.  Reshape it
  // directly so the group and token strides are derived from that layout;
  // flattening to 2D first loses the explicit head axis around rope_back.
  ggml_tensor *grouped_input =
      ggml_reshape_3d(ctx, context, group_in, groups, g.S);
  grouped_input = ggml_permute(ctx, grouped_input, 0, 2, 1, 3);
  ggml_tensor *grouped = ggml_mul_mat(ctx, woa3, grouped_input);
  grouped = ggml_permute(ctx, grouped, 0, 2, 1, 3);
  grouped = ggml_cont_2d(ctx, grouped, rank * groups, g.S);
  g.dsv4_layers.push_back(lt);
  return ggml_mul_mat(ctx, wob, grouped);
}

ggml_tensor *dsv4_hash_ids(FullGraph &g, ggml_context *ctx, int layer,
                           const int *token_ids) {
  const auto &c = g.engine->model_file.architecture_config;
  const std::string name =
      "blk." + std::to_string(layer) + ".ffn_gate_tid2eid.weight";
  const QuantizedTensor *table = resident(*g.engine, name);
  if (!table || table->meta.type != GGUFType::I32 ||
      table->meta.shape.size() < 2 ||
      table->meta.shape[0] != (uint64_t)c.topk || !token_ids)
    return nullptr;
  const int64_t n_vocab = (int64_t)table->meta.shape[1];
  const int32_t *src = reinterpret_cast<const int32_t *>(table->host_ptr);
  int32_t *dst = nullptr;
  // ggml_get_rows preserves the ID tensor's shape after its row dimension.
  // Keep IDs as [topk,S], matching the normal argsort_top_k output.  A flat
  // [topk*S] tensor is numerically equivalent for a host loop but changes the
  // routed expert tensor layout and can make MUL_MAT_ID select the wrong
  // token grouping on the CUDA backend.
  ggml_tensor *ids =
      dsv4_host_i32_2d(g, ctx, c.topk, g.S, "dsv4_graph_hash_ids", &dst);
  if (!ids)
    return nullptr;
  for (int s = 0; s < g.S; ++s) {
    const int tok = token_ids[s];
    if (tok < 0 || (int64_t)tok >= n_vocab)
      return nullptr;
    for (int k = 0; k < c.topk; ++k) {
      const int id = src[(size_t)tok * c.topk + k];
      if (id < 0 || id >= c.n_experts)
        return nullptr;
      dst[(size_t)s * c.topk + k] = id;
    }
  }
  return ids;
}

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
  constexpr size_t kMinGraphAllocHeadroomBytes =
      2ull * 1024ull * 1024ull * 1024ull; // 2 GiB
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
              static_cast<double>(g.expert_capacity + g.dense_capacity) /
                  (1024.0 * 1024.0 * 1024.0));
    return false;
  }
  return true;
}

bool build_dsv4_graph(FullGraph &g, const int *token_ids) {
  const auto &c = g.engine->model_file.architecture_config;
  ggml_init_params ip = {};
  ip.mem_size = kTensorContextBytes;
  ip.no_alloc = true;
  if (!MemoryGuard::try_reserve_host_bytes(kTensorContextBytes)) {
    LOG_WARN("[GGML-DSV4-FULL-MEM] host graph context refused: %.2f MiB "
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

  g.input =
      host_input(g, g.ctx, GGML_TYPE_F32, {g.D, g.HC, g.S},
                 (size_t)g.D * g.HC * g.S * sizeof(float), "dsv4_full_input");
  if (!g.input)
    return false;
  int32_t *pos_data = nullptr;
  g.positions = dsv4_host_i32(g, g.ctx, g.S, "dsv4_full_positions", &pos_data);
  if (!g.positions)
    return false;
  for (int s = 0; s < g.S; ++s)
    pos_data[s] = s;

  bool need4 = false, need128 = false, need_csa = false;
  for (int l = 0; l < g.n_layers; ++l) {
    const int ratio = c.dsv4_compress_ratios.empty()
                          ? 0
                          : (int)c.dsv4_compress_ratios[(size_t)l];
    need4 |= ratio == 4;
    need128 |= ratio == 128;
    need_csa |= ratio == 4;
  }
  auto make_mod = [&](int ratio, const char *name, ggml_tensor **out_t) {
    int32_t *data = nullptr;
    *out_t = dsv4_host_i32(g, g.ctx, g.S, name, &data);
    if (!*out_t)
      return false;
    for (int s = 0; s < g.S; ++s)
      data[s] = s % ratio;
    return true;
  };
  if (need4 && !make_mod(4, "dsv4_full_positions_mod4", &g.positions_mod4))
    return false;
  if (need128 &&
      !make_mod(128, "dsv4_full_positions_mod128", &g.positions_mod128))
    return false;
  auto make_blocks = [&](int ratio, const char *name, ggml_tensor **out_t) {
    const int blocks = g.S / ratio;
    if (blocks <= 0)
      return true;
    int32_t *data = nullptr;
    *out_t = dsv4_host_i32(g, g.ctx, blocks, name, &data);
    if (!*out_t)
      return false;
    for (int b = 0; b < blocks; ++b)
      data[b] = b * ratio;
    return true;
  };
  if (need4 &&
      !make_blocks(4, "dsv4_full_block_positions4", &g.block_positions4))
    return false;
  if (need128 &&
      !make_blocks(128, "dsv4_full_block_positions128", &g.block_positions128))
    return false;
  if (need_csa) {
    float *zv = nullptr;
    float *zs = nullptr;
    g.csa_zero_values = dsv4_host_f32(g, g.ctx, {g.qk_nope + g.qk_rope, 4},
                                      "dsv4_csa_zero_values", &zv);
    g.csa_zero_scores = dsv4_host_f32(g, g.ctx, {g.qk_nope + g.qk_rope, 4},
                                      "dsv4_csa_zero_scores", &zs);
    if (!g.csa_zero_values || !g.csa_zero_scores)
      return false;
    const size_t zero_n = (size_t)(g.qk_nope + g.qk_rope) * 4;
    std::fill(zv, zv + zero_n, 0.0f);
    std::fill(zs, zs + zero_n, -INFINITY);
  }

  ggml_tensor *res = g.input;
  for (int l = 0; l < g.n_layers; ++l) {
    if (MemoryGuard::allocation_blocked())
      return false;
    const std::string p = "blk." + std::to_string(l) + ".";
    ggml_tensor *post = nullptr;
    ggml_tensor *comb = nullptr;
    ggml_tensor *afn = resident_tensor(g, g.ctx, p + "hc_attn_fn.weight");
    ggml_tensor *abase = resident_tensor(g, g.ctx, p + "hc_attn_base.weight");
    ggml_tensor *ascale = resident_tensor(g, g.ctx, p + "hc_attn_scale.weight");
    if (!afn || !abase || !ascale)
      return false;
    ggml_tensor *residual = res;
    ggml_tensor *cur = hc_pre(g, g.ctx, res, afn, ascale, abase, &post, &comb);
    ggml_tensor *an = resident_tensor(g, g.ctx, p + "attn_norm.weight");
    if (!an)
      return false;
    cur = ggml_mul(g.ctx, ggml_rms_norm(g.ctx, cur, c.eps), an);
    ggml_tensor *attn = build_dsv4_attention(g, g.ctx, cur, l);
    if (!attn || g.dsv4_layers.empty() || g.dsv4_layers.back().layer != l)
      return false;
    if (dsv4_graph_diagnostic_enabled())
      g.dsv4_stage_outputs.emplace_back(
          "layer=" + std::to_string(l) + " attention", attn);
    res = hc_post(g, g.ctx, attn, residual, post, comb);

    ggml_tensor *ffn_fn = resident_tensor(g, g.ctx, p + "hc_ffn_fn.weight");
    ggml_tensor *ffn_base = resident_tensor(g, g.ctx, p + "hc_ffn_base.weight");
    ggml_tensor *ffn_scale =
        resident_tensor(g, g.ctx, p + "hc_ffn_scale.weight");
    ggml_tensor *fn = resident_tensor(g, g.ctx, p + "ffn_norm.weight");
    if (!ffn_fn || !ffn_base || !ffn_scale || !fn)
      return false;
    post = comb = nullptr;
    cur = hc_pre(g, g.ctx, res, ffn_fn, ffn_scale, ffn_base, &post, &comb);
    cur = ggml_mul(g.ctx, ggml_rms_norm(g.ctx, cur, c.eps), fn);
    ggml_tensor *forced_ids = l < c.hash_layer_count
                                  ? dsv4_hash_ids(g, g.ctx, l, token_ids)
                                  : nullptr;
    if (l < c.hash_layer_count && !forced_ids)
      return false;
    ggml_tensor *ffn = build_ffn(g, g.ctx, cur, l, forced_ids);
    if (!ffn)
      return false;
    if (dsv4_graph_diagnostic_enabled())
      g.dsv4_stage_outputs.emplace_back("layer=" + std::to_string(l) + " ffn",
                                        ffn);
    res = hc_post(g, g.ctx, ffn, res, post, comb);
    if (dsv4_graph_diagnostic_enabled()) {
      g.dsv4_stage_outputs.emplace_back(
          "layer=" + std::to_string(l) + " residual", res);
      g.dsv4_layer_outputs.push_back(res);
    }
  }

  // DeepSeek-V4's model-level HC head is the learned gated collapse.  The
  // surrounding generation loop still applies output_norm after this graph,
  // exactly as it does after the legacy hc_collapse_output path.
  ggml_tensor *flat = ggml_reshape_2d(g.ctx, res, g.D * g.HC, g.S);
  ggml_tensor *normed = ggml_rms_norm(g.ctx, flat, c.eps);
  ggml_tensor *head_fn = resident_tensor(g, g.ctx, "output_hc_fn.weight");
  ggml_tensor *head_base = resident_tensor(g, g.ctx, "output_hc_base.weight");
  ggml_tensor *head_scale = resident_tensor(g, g.ctx, "output_hc_scale.weight");
  if (!head_fn || !head_base || !head_scale)
    return false;
  ggml_tensor *head_logits = ggml_mul_mat(g.ctx, head_fn, normed);
  ggml_tensor *scale = ggml_view_1d(g.ctx, head_scale, 1, 0);
  ggml_tensor *base = ggml_view_1d(g.ctx, head_base, g.HC, 0);
  ggml_tensor *weights = ggml_scale_bias(
      g.ctx,
      ggml_sigmoid(g.ctx,
                   ggml_add(g.ctx, ggml_mul(g.ctx, head_logits, scale), base)),
      1.0f, c.hc_eps);
  ggml_tensor *out = nullptr;
  for (int h = 0; h < g.HC; ++h) {
    ggml_tensor *stream =
        ggml_view_2d(g.ctx, res, g.D, g.S, res->nb[2], (size_t)h * res->nb[1]);
    ggml_tensor *weight = ggml_view_2d(g.ctx, weights, 1, g.S, weights->nb[1],
                                       (size_t)h * weights->nb[0]);
    ggml_tensor *term = ggml_mul(g.ctx, stream, weight);
    out = out ? ggml_add(g.ctx, out, term) : term;
  }
  g.final_streams = res;
  g.output = out;
  ggml_set_output(g.final_streams);
  ggml_build_forward_expand(g.graph, g.final_streams);
  ggml_set_output(g.output);
  ggml_build_forward_expand(g.graph, g.output);
  for (ggml_tensor *layer_output : g.dsv4_layer_outputs) {
    ggml_set_output(layer_output);
    ggml_build_forward_expand(g.graph, layer_output);
  }
  for (const auto &lt : g.dsv4_layers) {
    ggml_set_output(lt.raw_kv);
    ggml_build_forward_expand(g.graph, lt.raw_kv);
    if (lt.comp_kv) {
      ggml_set_output(lt.comp_kv);
      ggml_build_forward_expand(g.graph, lt.comp_kv);
      ggml_set_output(lt.comp_score);
      ggml_build_forward_expand(g.graph, lt.comp_score);
      if (lt.keys) {
        ggml_set_output(lt.keys);
        ggml_build_forward_expand(g.graph, lt.keys);
      }
    }
    if (lt.indexer_comp_kv) {
      ggml_set_output(lt.indexer_comp_kv);
      ggml_build_forward_expand(g.graph, lt.indexer_comp_kv);
      ggml_set_output(lt.indexer_comp_score);
      ggml_build_forward_expand(g.graph, lt.indexer_comp_score);
      if (lt.indexer_keys) {
        ggml_set_output(lt.indexer_keys);
        ggml_build_forward_expand(g.graph, lt.indexer_keys);
      }
    }
  }
  if (!graph_alloc_vram_headroom_ok(g, "[GGML-DSV4-FULL]"))
    return false;
  if (!ggml_backend_sched_alloc_graph(g.sched, g.graph))
    return false;
  g.graph_workspace_bytes = ggml_backend_sched_get_buffer_size(g.sched, g.cuda);
  int cuda_nodes = 0, cpu_nodes = 0, transitions = 0;
  ggml_backend_t previous = nullptr;
  std::unordered_map<std::string, int> non_cuda;
  for (int i = 0; i < ggml_graph_n_nodes(g.graph); ++i) {
    ggml_tensor *node = ggml_graph_node(g.graph, i);
    ggml_backend_t b = ggml_backend_sched_get_tensor_backend(g.sched, node);
    if (b == g.cuda)
      ++cuda_nodes;
    else if (b == g.cpu) {
      ++cpu_nodes;
      ++non_cuda[ggml_op_name(node->op)];
    }
    if (previous && previous != b)
      ++transitions;
    previous = b;
  }
  LOG_INFO("[GGML-DSV4-FULL] scheduler nodes cuda=%d cpu=%d transitions=%d\n",
           cuda_nodes, cpu_nodes, transitions);
  if (!non_cuda.empty()) {
    std::string detail;
    for (const auto &[name, count] : non_cuda) {
      if (!detail.empty())
        detail += ", ";
      detail += name + "=" + std::to_string(count);
    }
    LOG_INFO("[GGML-DSV4-FULL] non-CUDA nodes: %s\n", detail.c_str());
  }
  LOG_INFO("[GGML-DSV4-FULL] graph built layers=%d nodes=%d splits=%d "
           "workspace=%.2f GiB compressed=%zu\n",
           g.n_layers, ggml_graph_n_nodes(g.graph),
           ggml_backend_sched_get_n_splits(g.sched),
           (double)g.graph_workspace_bytes / (1024.0 * 1024.0 * 1024.0),
           std::count_if(g.dsv4_layers.begin(), g.dsv4_layers.end(),
                         [](const auto &x) { return x.ratio != 0; }));
  LOG_INFO(
      "[GGML-DSV4-FULL-MEM] host ledger=%.2f/%.2f GB; graph context=%.2f MiB; "
      "device workspace=%.2f GiB (GPU, outside host_ram_gb)\n",
      (double)MemoryGuard::reserved_host_bytes() / 1.0e9,
      (double)MemoryGuard::host_budget_bytes() / 1.0e9,
      (double)g.context_reservation_bytes / (1024.0 * 1024.0),
      (double)g.graph_workspace_bytes / (1024.0 * 1024.0 * 1024.0));
  g.ready = true;
  return true;
}

std::unique_ptr<FullGraph> g_dsv4_graph;

ggml_tensor *resident_tensor(FullGraph &g, ggml_context *ctx,
                             const std::string &name) {
  const QuantizedTensor *q = resident(*g.engine, name);
  if (!q || !type_ok(q->meta.type))
    return nullptr;
  std::vector<int64_t> shape;
  for (uint64_t d : q->meta.shape)
    shape.push_back((int64_t)d);
  ggml_tensor *t = new_tensor(ctx, (ggml_type)q->meta.type, shape);
  if (!t || q->meta.size_bytes < ggml_nbytes(t))
    return nullptr;
  if (graph_dense_weight_name(name)) {
    if (!arena_alloc(g.dense_buffer, g.dense_base, g.dense_capacity,
                     g.dense_used, t, q->host_ptr, q->meta.size_bytes))
      return nullptr;
    ++g.dense_tensors;
  } else if (!bind_host(g, t, q->host_ptr, q->meta.size_bytes,
                        GGML_BACKEND_BUFFER_USAGE_WEIGHTS)) {
    return nullptr;
  }
  ggml_set_name(t, name.c_str());
  return t;
}

ggml_tensor *load_full_weight(FullGraph &g, ggml_context *ctx,
                              const std::string &name, bool device_expert,
                              bool routed_expert = false) {
  const GGUFTensor *m = model_tensor(*g.engine, name);
  if (!m || !type_ok(m->type) || m->shape.empty())
    return nullptr;
  std::vector<int64_t> shape;
  for (uint64_t d : m->shape)
    shape.push_back((int64_t)d);
  ggml_tensor *t = new_tensor(ctx, (ggml_type)m->type, shape);
  if (!t)
    return nullptr;
  const size_t bytes = (size_t)m->size_bytes;
  const size_t required = std::max(bytes, ggml_nbytes(t));
  // Device-resident graph layers use a short-lived aligned read buffer:
  // keeping another DDR copy after the cudaMemcpy would defeat the purpose of
  // the explicit VRAM preload.  All other routed projections use the single
  // graph source arena, while ordinary dense weights retain their existing
  // ownership/lifetime behavior.
  void *p = nullptr;
  bool transient_device_source = false;
  bool transient_host_reserved = false;
  if (device_expert) {
    if (!MemoryGuard::try_reserve_host_bytes(required))
      return nullptr;
    transient_host_reserved = true;
    if (posix_memalign(&p, 4096, required) != 0)
      p = nullptr;
    transient_device_source = true;
  } else if (routed_expert) {
    auto it = g.expert_source_views.find(name);
    p = it == g.expert_source_views.end() ? nullptr : it->second;
  } else {
    p = alloc_host(g, required);
  }
  if (!p) {
    if (transient_host_reserved)
      MemoryGuard::release_host_bytes(required);
    return nullptr;
  }
  // Non-device routed projections are read in one parallel I/O phase before
  // graph construction (see preload_expert_source).  Device-preloaded
  // projections and ordinary static tensors still use the local read path.
  if (device_expert || !routed_expert) {
    const int fd = g.engine->model_file.fd_for_tensor(*m);
    const ssize_t got = direct_io::pread_direct_aware(
        fd, g.engine->model_file.direct_io_active_for(*m),
        g.engine->model_file.direct_io_alignment, p, bytes, m->offset);
    if (got != (ssize_t)bytes) {
      LOG_ERROR("[GGML-GLM5NEXT-FULL] short read for %s: %zd/%zu\n",
                name.c_str(), got, bytes);
      if (transient_device_source)
        std::free(p);
      if (transient_host_reserved)
        MemoryGuard::release_host_bytes(required);
      return nullptr;
    }
  }
  bool ok = false;
  if (device_expert) {
    ok = arena_alloc(g.expert_buffer, g.expert_base, g.expert_capacity,
                     g.expert_used, t, p, bytes);
    if (transient_device_source)
      std::free(p);
    if (transient_host_reserved)
      MemoryGuard::release_host_bytes(required);
  } else if (graph_dense_weight_name(name)) {
    ok = arena_alloc(g.dense_buffer, g.dense_base, g.dense_capacity,
                     g.dense_used, t, p, bytes);
    if (ok)
      ++g.dense_tensors;
  } else {
    ok = bind_host(g, t, p, required, GGML_BACKEND_BUFFER_USAGE_WEIGHTS) !=
         nullptr;
    // Routed projections live in the single pageable DDR source arena.  Do
    // not register each slice separately: cudaHostRegister requires the
    // range to be page aligned and registering 126 slices would both defeat
    // the one-arena design and consume the host pin budget.  Static dense
    // weights keep the existing pinned path.
    if (ok && !routed_expert) {
      const size_t idx = g.owned.size() - 1;
      const cudaError_t err =
          cudaHostRegister(p, required, cudaHostRegisterPortable);
      if (err == cudaSuccess) {
        g.owned[idx].pinned = true;
        g.host_pinned_bytes += required;
      } else {
        ++g.host_pin_failures;
        cudaGetLastError();
      }
    }
  }
  if (!ok)
    return nullptr;
  g.host_weight_bytes += bytes;
  ggml_set_name(t, name.c_str());
  return t;
}

struct PackedGateUpMeta {
  const GGUFTensor *gate = nullptr;
  const GGUFTensor *up = nullptr;
  std::string name;
  size_t bytes = 0;
};

bool packed_gate_up_meta(const StreamingEngine &engine, int layer,
                         PackedGateUpMeta *out) {
  if (!out)
    return false;
  const std::string prefix = "blk." + std::to_string(layer) + ".";
  const std::string gate_name = prefix + "ffn_gate_exps.weight";
  const std::string up_name = prefix + "ffn_up_exps.weight";
  const GGUFTensor *gate = model_tensor(engine, gate_name);
  const GGUFTensor *up = model_tensor(engine, up_name);
  if (!gate || !up || !type_ok(gate->type) || gate->type != up->type ||
      gate->shape.size() != 3 || up->shape != gate->shape ||
      gate->size_bytes == 0 || up->size_bytes == 0 ||
      gate->size_bytes > SIZE_MAX - up->size_bytes)
    return false;

  // Expert tensors are laid out as contiguous expert slices along ne[2].  A
  // native ffn_gate_up_exps tensor interleaves the two projections inside
  // each expert slice: gate[e] followed by up[e].  The source loader below
  // performs that interleave; simply appending the complete gate tensor to
  // the complete up tensor would make expert e read rows belonging to other
  // experts.  Require the packed row layout to be exactly representable by
  // GGML as a sanity check before exposing it to MUL_MAT_ID.
  ggml_init_params ip = {};
  ip.mem_size = 16 * 1024;
  ip.no_alloc = true;
  ggml_context *ctx = ggml_init(ip);
  if (!ctx)
    return false;
  ggml_tensor *packed =
      ggml_new_tensor_3d(ctx, (ggml_type)gate->type, (int64_t)gate->shape[0],
                         (int64_t)gate->shape[1] * 2, (int64_t)gate->shape[2]);
  const size_t expected = packed ? ggml_nbytes(packed) : 0;
  ggml_free(ctx);
  if (!packed || expected != gate->size_bytes + up->size_bytes)
    return false;

  out->gate = gate;
  out->up = up;
  out->name = prefix + "ffn_gate_up_exps.weight";
  out->bytes = expected;
  return true;
}

double full_graph_resident_gb(const char *name) {
  const char *v = std::getenv(name);
  if (!v || !v[0])
    return 0.0;
  char *end = nullptr;
  const double gb = std::strtod(v, &end);
  return end != v && *end == '\0' && gb > 0.0 ? gb : 0.0;
}

size_t graph_host_live_bytes(const StreamingEngine &engine) {
  size_t total = 0;
  const auto add = [&total](size_t bytes) {
    if (bytes > SIZE_MAX - total) {
      total = SIZE_MAX;
      return;
    }
    total += bytes;
  };
  for (const auto &[name, tensor] : engine.resident_layers)
    if (tensor.host_ptr)
      add((size_t)tensor.meta.size_bytes);
  // Pointer-based KV caches do not retain their allocation size.  Their
  // configured sizes are cheap to reconstruct from the architecture and are
  // included explicitly so graph source admission also charges the host KV
  // state already allocated by graph_initializer.
  const auto &c = engine.model_file.architecture_config;
  if (!engine.Lc.empty() && c.kv_lora > 0 && c.n_layers >= 0)
    add((size_t)(c.n_layers + 1) * (size_t)engine.max_t * (size_t)c.kv_lora *
        sizeof(float));
  if (!engine.Rc.empty() && c.qk_rope > 0 && c.n_layers >= 0)
    add((size_t)(c.n_layers + 1) * (size_t)engine.max_t * (size_t)c.qk_rope *
        sizeof(float));
  if (!engine.Kc.empty() && c.n_head_kv > 0 && c.head_dim > 0)
    add((size_t)c.n_layers * (size_t)engine.max_t * (size_t)c.n_head_kv *
        (size_t)c.head_dim * sizeof(float) * 2);
  if (!engine.KVc.empty() && engine.sliding_window > 0 &&
      c.qk_nope + c.qk_rope > 0)
    add((size_t)c.n_layers * (size_t)engine.sliding_window *
        (size_t)(c.qk_nope + c.qk_rope) * sizeof(float));
  for (const auto &v : engine.glm5next_conv_history)
    add(v.size() * sizeof(float));
  for (const auto &v : engine.glm5next_recur_state)
    add(v.size() * sizeof(float));
  for (const auto &state : engine.dsv4_compressed) {
    add(state.keys.size() * sizeof(float));
    add(state.state_kv.size() * sizeof(float));
    add(state.state_score.size() * sizeof(float));
    add(state.indexer_keys.size() * sizeof(float));
    add(state.indexer_state_kv.size() * sizeof(float));
    add(state.indexer_state_score.size() * sizeof(float));
  }
  add(engine.hc_stream_buf.size() * sizeof(float));
  return total;
}

std::string partial_usage_path(const FullGraph &g) {
  const char *env = std::getenv("MOESTREAM_FULL_GRAPH_USAGE_FILE");
  if (env && env[0])
    return env;
  return g.engine->family_ops.family == ModelFamily::DeepSeek_V4
             ? ".moestream_ds4_usage_prefill"
             : ".moestream_usage_prefill_glm53flash";
}

PartialExpertLayer *partial_layer(FullGraph &g, int layer) {
  if (!g.partial_expert_residency || layer < 0 ||
      layer >= (int)g.partial_experts.size())
    return nullptr;
  PartialExpertLayer &p = g.partial_experts[(size_t)layer];
  return p.hot.empty() && p.cold.empty() ? nullptr : &p;
}

const PartialExpertLayer *partial_layer(const FullGraph &g, int layer) {
  if (!g.partial_expert_residency || layer < 0 ||
      layer >= (int)g.partial_experts.size())
    return nullptr;
  const PartialExpertLayer &p = g.partial_experts[(size_t)layer];
  return p.hot.empty() && p.cold.empty() ? nullptr : &p;
}

bool configure_partial_experts(FullGraph &g) {
  const double requested_gb =
      full_graph_resident_gb("MOESTREAM_FULL_GRAPH_RESIDENT_GB");
  if (requested_gb <= 0.0)
    return false;
  const auto &c = g.engine->model_file.architecture_config;
  auto [entries, max_count] =
      StreamingEngine::load_usage(partial_usage_path(g));
  (void)max_count;
  if (entries.empty()) {
    LOG_WARN("[GGML-FULL] partial residency requested but usage file %s is "
             "empty or missing; "
             "keeping all expert weights streamed\n",
             partial_usage_path(g).c_str());
    return false;
  }

  std::vector<std::vector<StreamingEngine::UsageEntry>> by_layer(
      (size_t)std::max(0, c.n_layers));
  for (const auto &e : entries) {
    if (e.layer < c.first_dense || e.layer >= c.n_layers || e.expert < 0 ||
        e.expert >= c.n_experts || e.count == 0)
      continue;
    by_layer[(size_t)e.layer].push_back(e);
  }
  for (auto &v : by_layer) {
    std::sort(v.begin(), v.end(), [](const auto &a, const auto &b) {
      if (a.count != b.count)
        return a.count > b.count;
      return a.expert < b.expert;
    });
    v.erase(std::unique(v.begin(), v.end(),
                        [](const auto &a, const auto &b) {
                          return a.expert == b.expert;
                        }),
            v.end());
  }

  // All GLM5NEXT/DSV4 expert records in a layer are contiguous and have a
  // uniform byte size.  Use the actual GGUF sizes rather than assuming every
  // model has the same per-expert footprint.
  std::vector<size_t> record_bytes((size_t)std::max(0, c.n_layers), 0);
  for (int l = c.first_dense; l < c.n_layers; ++l) {
    PackedGateUpMeta packed;
    const GGUFTensor *down = model_tensor(
        *g.engine, "blk." + std::to_string(l) + ".ffn_down_exps.weight");
    size_t gate_up_bytes = 0;
    if (packed_gate_up_meta(*g.engine, l, &packed)) {
      gate_up_bytes = packed.bytes;
    } else {
      const GGUFTensor *gate = model_tensor(
          *g.engine, "blk." + std::to_string(l) + ".ffn_gate_exps.weight");
      const GGUFTensor *up = model_tensor(
          *g.engine, "blk." + std::to_string(l) + ".ffn_up_exps.weight");
      if (gate && up && gate->size_bytes <= SIZE_MAX - up->size_bytes)
        gate_up_bytes = (size_t)gate->size_bytes + (size_t)up->size_bytes;
    }
    if (!down || c.n_experts <= 0 || gate_up_bytes == 0 ||
        gate_up_bytes % (size_t)c.n_experts != 0 ||
        down->size_bytes % (size_t)c.n_experts != 0 ||
        gate_up_bytes / (size_t)c.n_experts >
            SIZE_MAX - (size_t)down->size_bytes / (size_t)c.n_experts)
      continue;
    record_bytes[(size_t)l] = gate_up_bytes / (size_t)c.n_experts +
                              (size_t)down->size_bytes / (size_t)c.n_experts;
  }

  const size_t budget = requested_gb >= (double)SIZE_MAX
                            ? SIZE_MAX
                            : (size_t)(requested_gb * 1024.0 * 1024.0 * 1024.0);
  g.partial_experts.assign((size_t)std::max(0, c.n_layers), {});
  size_t used = 0;
  size_t hot_records = 0;
  size_t max_rank = 0;
  for (int l = c.first_dense; l < c.n_layers; ++l)
    max_rank = std::max(max_rank, by_layer[(size_t)l].size());

  // Select the same rank from each layer before taking the next rank.  This
  // gives every layer a useful hot set and matches the layer-by-layer graph
  // execution order better than consuming the whole budget in a few layers.
  for (size_t rank = 0; rank < max_rank; ++rank) {
    for (int l = c.first_dense; l < c.n_layers; ++l) {
      if (rank >= by_layer[(size_t)l].size() || record_bytes[(size_t)l] == 0)
        continue;
      const size_t bytes = record_bytes[(size_t)l];
      if (bytes > budget - std::min(used, budget))
        continue;
      g.partial_experts[(size_t)l].hot.push_back(
          by_layer[(size_t)l][rank].expert);
      used += bytes;
      ++hot_records;
    }
  }
  if (hot_records == 0) {
    g.partial_experts.clear();
    LOG_WARN(
        "[GGML-FULL] partial residency budget %.2f GiB was smaller than one "
        "expert record or usage had no compatible layers\n",
        requested_gb);
    return false;
  }

  size_t cold_bytes = 0;
  for (int l = c.first_dense; l < c.n_layers; ++l) {
    PartialExpertLayer &p = g.partial_experts[(size_t)l];
    if (!p.hot.empty()) {
      std::sort(p.hot.begin(), p.hot.end());
      p.hot.erase(std::unique(p.hot.begin(), p.hot.end()), p.hot.end());
    }
    if (record_bytes[(size_t)l] == 0)
      continue;
    std::vector<uint8_t> is_hot((size_t)c.n_experts, 0);
    for (int e : p.hot)
      is_hot[(size_t)e] = 1;
    for (int e = 0; e < c.n_experts; ++e)
      if (!is_hot[(size_t)e])
        p.cold.push_back(e);
    cold_bytes += record_bytes[(size_t)l] * p.cold.size();
  }
  g.partial_expert_residency = true;
  g.partial_hot_bytes = used;
  g.partial_cold_bytes = cold_bytes;
  g.partial_hot_records = hot_records;
  g.partial_usage_file = partial_usage_path(g);
  LOG_INFO("[GGML-FULL] partial resident set: %.2f GiB hot, %.2f GiB cold, "
           "%zu hot expert records across %d layers, usage=%s\n",
           (double)used / (1024.0 * 1024.0 * 1024.0),
           (double)cold_bytes / (1024.0 * 1024.0 * 1024.0), hot_records,
           c.n_layers - c.first_dense, g.partial_usage_file.c_str());
  return true;
}

bool read_tensor_direct(const StreamingEngine &engine, const GGUFTensor &meta,
                        void *dst) {
  const int fd = engine.model_file.fd_for_tensor(meta);
  const bool is_direct = engine.model_file.direct_io_active_for(meta);
  const ssize_t got = direct_io::pread_direct_aware(
      fd, is_direct, engine.model_file.direct_io_alignment, dst,
      (size_t)meta.size_bytes, meta.offset);
  const bool ok = got == (ssize_t)meta.size_bytes;
#if defined(__linux__)
  // When a GGUF is not 4096-aligned, the fallback pread path populates the
  // kernel page cache in addition to the graph-owned DDR source.  Releasing
  // those file pages after each completed tensor keeps a 78 GiB source load
  // from briefly becoming a second 78 GiB resident copy.  O_DIRECT already
  // bypasses this cache, so the advice is only needed for the default path.
  if (ok && !is_direct)
    (void)posix_fadvise(fd, (off_t)meta.offset, (off_t)meta.size_bytes,
                        POSIX_FADV_DONTNEED);
#endif
  return ok;
}

bool read_pack_gate_up(const StreamingEngine &engine, const GGUFTensor &gate,
                       const GGUFTensor &up, void *dst) {
  if (!dst || gate.shape.size() != 3 || up.shape != gate.shape ||
      gate.shape[2] == 0 || gate.size_bytes % gate.shape[2] != 0 ||
      up.size_bytes % up.shape[2] != 0)
    return false;
  const size_t gate_bytes = (size_t)gate.size_bytes;
  const size_t up_bytes = (size_t)up.size_bytes;
  if (gate_bytes > SIZE_MAX - up_bytes)
    return false;
  const size_t scratch_bytes = gate_bytes + up_bytes;
  if (!MemoryGuard::try_reserve_host_bytes(scratch_bytes))
    return false;
  void *scratch = nullptr;
  if (posix_memalign(&scratch, 4096, scratch_bytes) != 0 || !scratch) {
    MemoryGuard::release_host_bytes(scratch_bytes);
    return false;
  }
  const bool ok_gate = read_tensor_direct(engine, gate, scratch);
  const bool ok_up =
      ok_gate && read_tensor_direct(
                     engine, up, static_cast<uint8_t *>(scratch) + gate_bytes);
  if (!ok_up) {
    std::free(scratch);
    MemoryGuard::release_host_bytes(scratch_bytes);
    return false;
  }

  const size_t n_experts = (size_t)gate.shape[2];
  const size_t gate_expert_bytes = gate_bytes / n_experts;
  const size_t up_expert_bytes = up_bytes / n_experts;
  uint8_t *out = static_cast<uint8_t *>(dst);
  const uint8_t *gate_src = static_cast<const uint8_t *>(scratch);
  const uint8_t *up_src = gate_src + gate_bytes;
  for (size_t e = 0; e < n_experts; ++e) {
    std::memcpy(out, gate_src + e * gate_expert_bytes, gate_expert_bytes);
    out += gate_expert_bytes;
    std::memcpy(out, up_src + e * up_expert_bytes, up_expert_bytes);
    out += up_expert_bytes;
  }
  std::free(scratch);
  MemoryGuard::release_host_bytes(scratch_bytes);
  return true;
}

ggml_tensor *load_gate_up_weight(FullGraph &g, ggml_context *ctx, int layer,
                                 bool device_expert) {
  PackedGateUpMeta meta;
  if (!packed_gate_up_meta(*g.engine, layer, &meta))
    return nullptr;

  if (const PartialExpertLayer *p = partial_layer(g, layer)) {
    if (p->cold.empty())
      return nullptr;
    auto ptr_it = g.expert_source_views.find(meta.name);
    auto bytes_it = g.expert_source_view_bytes.find(meta.name);
    if (ptr_it == g.expert_source_views.end() ||
        bytes_it == g.expert_source_view_bytes.end())
      return nullptr;
    ggml_tensor *cold = ggml_new_tensor_3d(
        ctx, (ggml_type)meta.gate->type, (int64_t)meta.gate->shape[0],
        (int64_t)meta.gate->shape[1] * 2, (int64_t)p->cold.size());
    if (!cold || bytes_it->second < ggml_nbytes(cold) ||
        !bind_host(g, cold, ptr_it->second, bytes_it->second,
                   GGML_BACKEND_BUFFER_USAGE_WEIGHTS))
      return nullptr;
    g.host_weight_bytes += bytes_it->second;
    ggml_set_name(cold, (meta.name + ".cold").c_str());
    return cold;
  }

  ggml_tensor *t = ggml_new_tensor_3d(
      ctx, (ggml_type)meta.gate->type, (int64_t)meta.gate->shape[0],
      (int64_t)meta.gate->shape[1] * 2, (int64_t)meta.gate->shape[2]);
  if (!t)
    return nullptr;

  void *p = nullptr;
  bool transient = false;
  bool transient_reserved = false;
  if (device_expert) {
    if (!MemoryGuard::try_reserve_host_bytes(meta.bytes))
      return nullptr;
    transient_reserved = true;
    if (posix_memalign(&p, 4096, meta.bytes) != 0)
      p = nullptr;
    transient = true;
  } else {
    auto it = g.expert_source_views.find(meta.name);
    p = it == g.expert_source_views.end() ? nullptr : it->second;
  }
  if (!p) {
    if (transient_reserved)
      MemoryGuard::release_host_bytes(meta.bytes);
    return nullptr;
  }

  if (device_expert) {
    if (!read_pack_gate_up(*g.engine, *meta.gate, *meta.up, p)) {
      LOG_ERROR("[GGML-GLM5NEXT-FULL] short read for packed %s\n",
                meta.name.c_str());
      if (transient)
        std::free(p);
      if (transient_reserved)
        MemoryGuard::release_host_bytes(meta.bytes);
      return nullptr;
    }
  }

  bool ok = false;
  if (device_expert) {
    ok = arena_alloc(g.expert_buffer, g.expert_base, g.expert_capacity,
                     g.expert_used, t, p, meta.bytes);
    if (transient)
      std::free(p);
    if (transient_reserved)
      MemoryGuard::release_host_bytes(meta.bytes);
  } else {
    ok = bind_host(g, t, p, meta.bytes, GGML_BACKEND_BUFFER_USAGE_WEIGHTS) !=
         nullptr;
  }
  if (!ok)
    return nullptr;
  g.host_weight_bytes += meta.bytes;
  ggml_set_name(t, meta.name.c_str());
  return t;
}

ggml_tensor *load_cold_weight(FullGraph &g, ggml_context *ctx, int layer,
                              const char *stem) {
  const PartialExpertLayer *p = partial_layer(g, layer);
  if (!p || p->cold.empty())
    return nullptr;
  const std::string name = "blk." + std::to_string(layer) + "." + stem;
  const GGUFTensor *m = model_tensor(*g.engine, name);
  auto ptr_it = g.expert_source_views.find(name);
  auto bytes_it = g.expert_source_view_bytes.find(name);
  if (!m || ptr_it == g.expert_source_views.end() ||
      bytes_it == g.expert_source_view_bytes.end() || m->shape.size() != 3)
    return nullptr;
  std::vector<int64_t> shape;
  for (uint64_t d : m->shape)
    shape.push_back((int64_t)d);
  shape[2] = (int64_t)p->cold.size();
  ggml_tensor *t = new_tensor(ctx, (ggml_type)m->type, shape);
  if (!t || bytes_it->second < ggml_nbytes(t) ||
      !bind_host(g, t, ptr_it->second, bytes_it->second,
                 GGML_BACKEND_BUFFER_USAGE_WEIGHTS))
    return nullptr;
  g.host_weight_bytes += bytes_it->second;
  ggml_set_name(t, name.c_str());
  return t;
}

bool arena_copy_selected(FullGraph &g, ggml_tensor *t, const void *full_src,
                         size_t full_bytes, const std::vector<int> &selected) {
  const auto &c = g.engine->model_file.architecture_config;
  if (!t || !full_src || c.n_experts <= 0 ||
      full_bytes % (size_t)c.n_experts != 0 || selected.empty())
    return false;
  const size_t record = full_bytes / (size_t)c.n_experts;
  if (record > SIZE_MAX / selected.size())
    return false;
  const size_t bytes = record * selected.size();
  if (bytes != ggml_nbytes(t))
    return false;
  if (!MemoryGuard::try_reserve_host_bytes(bytes))
    return false;
  void *scratch = nullptr;
  if (posix_memalign(&scratch, 4096, bytes) != 0 || !scratch) {
    MemoryGuard::release_host_bytes(bytes);
    return false;
  }
  uint8_t *dst = static_cast<uint8_t *>(scratch);
  const uint8_t *src = static_cast<const uint8_t *>(full_src);
  for (size_t i = 0; i < selected.size(); ++i)
    std::memcpy(dst + i * record, src + (size_t)selected[i] * record, record);
  const bool ok = arena_alloc(g.expert_buffer, g.expert_base, g.expert_capacity,
                              g.expert_used, t, scratch, bytes);
  std::free(scratch);
  MemoryGuard::release_host_bytes(bytes);
  return ok;
}

ggml_tensor *make_hot_weight(FullGraph &g, ggml_context *ctx, int layer,
                             const char *stem) {
  const PartialExpertLayer *p = partial_layer(g, layer);
  if (!p || p->hot.empty())
    return nullptr;
  const std::string name = "blk." + std::to_string(layer) + "." + stem;
  const GGUFTensor *m = model_tensor(*g.engine, name);
  auto ptr_it = g.expert_source_full_views.find(name);
  auto bytes_it = g.expert_source_full_view_bytes.find(name);
  if (!m || ptr_it == g.expert_source_full_views.end() ||
      bytes_it == g.expert_source_full_view_bytes.end() || m->shape.size() != 3)
    return nullptr;
  std::vector<int64_t> shape;
  for (uint64_t d : m->shape)
    shape.push_back((int64_t)d);
  shape[2] = (int64_t)p->hot.size();
  ggml_tensor *t = new_tensor(ctx, (ggml_type)m->type, shape);
  if (!t ||
      !arena_copy_selected(g, t, ptr_it->second, bytes_it->second, p->hot))
    return nullptr;
  ggml_set_name(t, (name + ".hot").c_str());
  return t;
}

ggml_tensor *make_hot_gate_up(FullGraph &g, ggml_context *ctx, int layer) {
  const PartialExpertLayer *p = partial_layer(g, layer);
  if (!p || p->hot.empty())
    return nullptr;
  PackedGateUpMeta meta;
  if (!packed_gate_up_meta(*g.engine, layer, &meta))
    return nullptr;
  auto ptr_it = g.expert_source_full_views.find(meta.name);
  auto bytes_it = g.expert_source_full_view_bytes.find(meta.name);
  if (ptr_it == g.expert_source_full_views.end() ||
      bytes_it == g.expert_source_full_view_bytes.end())
    return nullptr;
  ggml_tensor *t = ggml_new_tensor_3d(
      ctx, (ggml_type)meta.gate->type, (int64_t)meta.gate->shape[0],
      (int64_t)meta.gate->shape[1] * 2, (int64_t)p->hot.size());
  if (!t ||
      !arena_copy_selected(g, t, ptr_it->second, bytes_it->second, p->hot))
    return nullptr;
  ggml_set_name(t, (meta.name + ".hot").c_str());
  return t;
}

bool prepare_partial_hot(FullGraph &g, ggml_context *ctx, int layer) {
  PartialExpertLayer *p = partial_layer(g, layer);
  if (!p || p->hot.empty())
    return true;
  if (p->hot_down || p->hot_gate_up || p->hot_gate)
    return true;
  PackedGateUpMeta packed;
  if (packed_gate_up_meta(*g.engine, layer, &packed)) {
    p->hot_gate_up = make_hot_gate_up(g, ctx, layer);
    p->hot_down = make_hot_weight(g, ctx, layer, "ffn_down_exps.weight");
    return p->hot_gate_up && p->hot_down;
  }
  p->hot_gate = make_hot_weight(g, ctx, layer, "ffn_gate_exps.weight");
  p->hot_up = make_hot_weight(g, ctx, layer, "ffn_up_exps.weight");
  p->hot_down = make_hot_weight(g, ctx, layer, "ffn_down_exps.weight");
  return p->hot_gate && p->hot_up && p->hot_down;
}

ggml_tensor *partial_map(FullGraph &g, ggml_context *ctx, int layer, bool hot,
                         bool mask) {
  PartialExpertLayer *p = partial_layer(g, layer);
  if (!p)
    return nullptr;
  ggml_tensor *&slot = mask ? (hot ? p->hot_mask : p->cold_mask)
                            : (hot ? p->hot_map : p->cold_map);
  if (slot)
    return slot;
  const auto &c = g.engine->model_file.architecture_config;
  const size_t n = (size_t)std::max(0, c.n_experts);
  const ggml_type type = mask ? GGML_TYPE_F32 : GGML_TYPE_I32;
  // GET_ROWS requires src0->ne[2] == ids->ne[1].  Repeat the small map for
  // every prompt token so the map is a regular GGML input to the CUDA graph.
  ggml_tensor *t = ggml_new_tensor_3d(ctx, type, 1, (int64_t)n, g.S);
  if (!t)
    return nullptr;
  std::vector<uint8_t> is_hot(n, 0);
  for (int e : p->hot)
    if (e >= 0 && (size_t)e < n)
      is_hot[(size_t)e] = 1;
  std::vector<float> mask_values;
  std::vector<int32_t> map_values;
  const void *data = nullptr;
  if (mask) {
    mask_values.assign(n * (size_t)g.S, 0.0f);
    for (int s = 0; s < g.S; ++s)
      for (int e = 0; e < c.n_experts; ++e)
        mask_values[(size_t)s * n + (size_t)e] =
            is_hot[(size_t)e] == (hot ? 1 : 0) ? 1.0f : 0.0f;
    data = mask_values.data();
  } else {
    map_values.assign(n * (size_t)g.S, 0);
    const auto &ids = hot ? p->hot : p->cold;
    for (size_t i = 0; i < ids.size(); ++i)
      for (int s = 0; s < g.S; ++s)
        map_values[(size_t)s * n + (size_t)ids[i]] = (int32_t)i;
    data = map_values.data();
  }
  if (!arena_alloc(g.dense_buffer, g.dense_base, g.dense_capacity, g.dense_used,
                   t, data, ggml_nbytes(t)))
    return nullptr;
  ggml_set_name(t, (std::string("full_graph_") + (hot ? "hot_" : "cold_") +
                    (mask ? "mask_" : "map_") + std::to_string(layer))
                       .c_str());
  slot = t;
  return slot;
}

ggml_tensor *static_full_weight(FullGraph &g, ggml_context *ctx,
                                const std::string &name) {
  // DSA's attn_k_b and attn_v_b are intentionally absent from the legacy
  // resident manifest because that path builds CPU WUK/WUV caches.  In this
  // graph they are ordinary static device weights.
  return load_full_weight(g, ctx, name, false);
}

ggml_tensor *graph_ram_weight(FullGraph &g, ggml_context *ctx, int layer,
                              const char *stem) {
  const auto fail = [&](const char *why) -> ggml_tensor * {
    LOG_WARN(
        "[GGML-GLM5NEXT-FULL] graph RAM weight failed layer=%d stem=%s: %s\n",
        layer, stem ? stem : "<null>", why);
    return nullptr;
  };
  if (!g.engine)
    return fail("no engine");
  if (!g.engine->graph_ram_layer_ready(layer))
    return fail("arena is not ready");
  const auto *arena = g.engine->graph_ram_arena(layer);
  if (!arena || !arena->base || arena->n_experts <= 0)
    return fail("missing arena");

  const std::string name = "blk." + std::to_string(layer) + "." + stem;
  const GGUFTensor *meta = model_tensor(*g.engine, name);
  if (!meta || !type_ok(meta->type) || meta->shape.size() != 3 ||
      meta->size_bytes == 0)
    return fail("missing or unsupported GGUF tensor metadata");

  size_t offset = 0;
  if (std::strcmp(stem, "ffn_up_exps.weight") == 0) {
    offset = arena->gate_bytes;
  } else if (std::strcmp(stem, "ffn_down_exps.weight") == 0) {
    if (arena->gate_bytes > SIZE_MAX - arena->up_bytes)
      return fail("projection offset overflow");
    offset = arena->gate_bytes + arena->up_bytes;
  } else if (std::strcmp(stem, "ffn_gate_exps.weight") != 0) {
    return fail("unknown projection stem");
  }
  // meta->size_bytes is the GGUF tensor's FULL size across every expert in
  // the layer (shape [.., .., n_experts]) -- the arena's stride budget is
  // for ONE expert's slice. Comparing the whole-tensor size against a
  // single-expert budget always fails (a 288-expert tensor is ~288x the
  // per-expert slice) -- this is exactly what made every graph_ram-ready
  // MoE layer decline at build_ffn, real-hardware-confirmed via the
  // "projection does not fit in expert stride" warning firing for all
  // three projections at the very first routed-expert layer. Divide down
  // to the per-expert size (matching how prepare_graph_ram_arenas derived
  // arena->gate_bytes/up_bytes/down_bytes in the first place) before
  // checking it against the remaining per-expert stride.
  if ((size_t)arena->n_experts <= 0 ||
      meta->size_bytes % (size_t)arena->n_experts != 0)
    return fail("tensor size not evenly divisible by expert count");
  const size_t per_expert_bytes = meta->size_bytes / (size_t)arena->n_experts;
  if (offset > arena->stride || per_expert_bytes > arena->stride - offset)
    return fail("projection does not fit in expert stride");

  std::vector<int64_t> shape;
  shape.reserve(meta->shape.size());
  for (uint64_t dim : meta->shape)
    shape.push_back((int64_t)dim);
  ggml_tensor *t = new_tensor(ctx, (ggml_type)meta->type, shape);
  if (!t)
    return fail("GGML tensor creation failed");
  if (ggml_nbytes(t) != meta->size_bytes)
    return fail("GGML tensor size differs from GGUF tensor size");

  // The same host arena is exposed through three GGML buffer handles. Each
  // tensor starts at its projection offset, while nb[2] skips one complete
  // [gate|up|down] expert slice. This is the external-weight layout already
  // used by the regular MoE GGML backend, with ownership retained by the
  // StreamingEngine rather than by the graph.
  ggml_backend_buffer_t buffer =
      ggml_backend_cpu_buffer_from_ptr(arena->base, arena->bytes);
  if (!buffer)
    return fail("CPU external buffer creation failed");
  ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
  t->nb[2] = arena->stride;
  t->nb[3] = t->nb[2] * (size_t)t->ne[2];
  t->data = arena->base + offset;
  t->buffer = buffer;
  if (ggml_backend_buffer_init_tensor(buffer, t) != GGML_STATUS_SUCCESS) {
    ggml_backend_buffer_free(buffer);
    return fail("CPU external buffer tensor initialization failed");
  }
  g.host_buffers.push_back(buffer);
  g.host_weight_bytes += meta->size_bytes;
  ggml_set_name(t, (name + ".graph_ram").c_str());
  return t;
}

ggml_tensor *expert_weight(FullGraph &g, ggml_context *ctx, int layer,
                           const char *stem) {
  if (partial_layer(g, layer))
    return load_cold_weight(g, ctx, layer, stem);
  if (g.engine->graph_ram_layer_ready(layer)) {
    if (ggml_tensor *t = graph_ram_weight(g, ctx, layer, stem))
      return t;
  }
  const bool on_device =
      layer >= g.first_dense && layer < g.first_dense + g.device_expert_layers;
  return load_full_weight(g, ctx, "blk." + std::to_string(layer) + "." + stem,
                          on_device, true);
}

ggml_tensor *expert_gate_up_weight(FullGraph &g, ggml_context *ctx, int layer) {
  if (partial_layer(g, layer))
    return load_gate_up_weight(g, ctx, layer, false);
  // The shared RAM arena stores gate and up as separate planes inside each
  // expert slice. Force the separate projection views so expert e's gate and
  // up bytes are addressed with the same fixed stride.
  if (g.engine->graph_ram_layer_ready(layer))
    return nullptr;
  const bool on_device =
      layer >= g.first_dense && layer < g.first_dense + g.device_expert_layers;
  return load_gate_up_weight(g, ctx, layer, on_device);
}

size_t expert_source_bytes(const StreamingEngine &engine,
                           int device_expert_layers) {
  const auto &c = engine.model_file.architecture_config;
  size_t total = 0;
  for (int l = c.first_dense; l < c.n_layers; ++l) {
    if (l < c.first_dense + device_expert_layers)
      continue;
    if (engine.graph_ram_layer_ready(l))
      continue;
    for (const char *stem : {"ffn_gate_exps.weight", "ffn_up_exps.weight",
                             "ffn_down_exps.weight"}) {
      const GGUFTensor *m =
          model_tensor(engine, "blk." + std::to_string(l) + "." + stem);
      if (!m ||
          total > std::numeric_limits<size_t>::max() - (size_t)m->size_bytes)
        return 0;
      total += (size_t)m->size_bytes;
    }
  }
  return total;
}

bool compact_partial_source(FullGraph &g) {
  if (!g.partial_expert_residency || g.partial_sources_compacted)
    return true;
  const auto &c = g.engine->model_file.architecture_config;
  g.expert_source_full_views = g.expert_source_views;
  g.expert_source_full_view_bytes = g.expert_source_view_bytes;

  std::unordered_map<std::string, void *> cold_views;
  std::unordered_map<std::string, size_t> cold_view_bytes;
  size_t compact_used = 0;
  size_t compact_tensors = 0;
  const size_t alignment = 4096;

  const auto compact_one = [&](const std::string &name,
                               const PartialExpertLayer &p) -> bool {
    auto src_it = g.expert_source_full_views.find(name);
    auto size_it = g.expert_source_full_view_bytes.find(name);
    if (src_it == g.expert_source_full_views.end() ||
        size_it == g.expert_source_full_view_bytes.end())
      return false;
    if (c.n_experts <= 0 || size_it->second % (size_t)c.n_experts != 0)
      return false;
    const size_t record = size_it->second / (size_t)c.n_experts;
    const size_t bytes = record * p.cold.size();
    if (bytes == 0)
      return true;
    const size_t off = (compact_used + alignment - 1) & ~(alignment - 1);
    if (off > g.expert_source_capacity ||
        bytes > g.expert_source_capacity - off)
      return false;
    if (!MemoryGuard::try_reserve_host_bytes(bytes))
      return false;
    void *scratch = nullptr;
    if (posix_memalign(&scratch, alignment, bytes) != 0 || !scratch) {
      MemoryGuard::release_host_bytes(bytes);
      return false;
    }
    const uint8_t *src = static_cast<const uint8_t *>(src_it->second);
    uint8_t *tmp = static_cast<uint8_t *>(scratch);
    for (size_t i = 0; i < p.cold.size(); ++i) {
      std::memcpy(tmp + i * record, src + (size_t)p.cold[i] * record, record);
    }
    std::memmove(static_cast<uint8_t *>(g.expert_source_base) + off, tmp,
                 bytes);
    std::free(scratch);
    MemoryGuard::release_host_bytes(bytes);
    cold_views.emplace(name,
                       static_cast<uint8_t *>(g.expert_source_base) + off);
    cold_view_bytes.emplace(name, bytes);
    compact_used = off + bytes;
    ++compact_tensors;
    return true;
  };

  for (int l = c.first_dense; l < c.n_layers; ++l) {
    const PartialExpertLayer &p = g.partial_experts[(size_t)l];
    const std::string prefix = "blk." + std::to_string(l) + ".";
    PackedGateUpMeta packed;
    if (packed_gate_up_meta(*g.engine, l, &packed)) {
      if (!compact_one(prefix + "ffn_gate_up_exps.weight", p) ||
          !compact_one(prefix + "ffn_down_exps.weight", p))
        return false;
    } else {
      for (const char *stem : {"ffn_gate_exps.weight", "ffn_up_exps.weight",
                               "ffn_down_exps.weight"}) {
        if (!compact_one(prefix + stem, p))
          return false;
      }
    }
  }
  g.expert_source_views.swap(cold_views);
  g.expert_source_view_bytes.swap(cold_view_bytes);
  g.expert_source_used = compact_used;
  g.expert_source_tensors = compact_tensors;
  g.partial_sources_compacted = true;
  LOG_INFO("[GGML-FULL] compacted cold source: %.2f GiB, %zu tensors; "
           "full source remains one in-place DDR allocation\n",
           (double)compact_used / (1024.0 * 1024.0 * 1024.0), compact_tensors);
  return true;
}

bool preload_expert_source(FullGraph &g) {
  const auto &c = g.engine->model_file.architecture_config;
  struct Job {
    const GGUFTensor *meta = nullptr;
    void *dst = nullptr;
    std::string name;
    const GGUFTensor *packed_meta = nullptr;
  };
  std::vector<Job> jobs;
  jobs.reserve((size_t)std::max(0, c.n_layers - c.first_dense) * 3);

  const int device_end = c.first_dense + g.device_expert_layers;
  for (int l = c.first_dense; l < c.n_layers; ++l) {
    if (l < device_end)
      continue;
    if (g.engine->graph_ram_layer_ready(l))
      continue;
    PackedGateUpMeta packed;
    if (packed_gate_up_meta(*g.engine, l, &packed)) {
      void *dst = alloc_expert_source(g, packed.bytes);
      if (!dst) {
        LOG_ERROR("[GGML-GLM5NEXT-FULL] DDR source arena exhausted at %s\n",
                  packed.name.c_str());
        return false;
      }
      g.expert_source_views.emplace(packed.name, dst);
      g.expert_source_view_bytes.emplace(packed.name, packed.bytes);
      jobs.push_back({packed.gate, dst, packed.name, packed.up});
    } else {
      for (const char *stem : {"ffn_gate_exps.weight", "ffn_up_exps.weight"}) {
        const std::string name = "blk." + std::to_string(l) + "." + stem;
        const GGUFTensor *m = model_tensor(*g.engine, name);
        if (!m || !type_ok(m->type) || m->size_bytes == 0) {
          LOG_ERROR("[GGML-GLM5NEXT-FULL] missing routed tensor %s\n",
                    name.c_str());
          return false;
        }
        void *dst = alloc_expert_source(g, (size_t)m->size_bytes);
        if (!dst) {
          LOG_ERROR("[GGML-GLM5NEXT-FULL] DDR source arena exhausted at %s\n",
                    name.c_str());
          return false;
        }
        g.expert_source_views.emplace(name, dst);
        g.expert_source_view_bytes.emplace(name, (size_t)m->size_bytes);
        jobs.push_back({m, dst, name});
      }
    }

    const std::string down_name =
        "blk." + std::to_string(l) + ".ffn_down_exps.weight";
    const GGUFTensor *down = model_tensor(*g.engine, down_name);
    if (!down || !type_ok(down->type) || down->size_bytes == 0) {
      LOG_ERROR("[GGML-GLM5NEXT-FULL] missing routed tensor %s\n",
                down_name.c_str());
      return false;
    }
    void *down_dst = alloc_expert_source(g, (size_t)down->size_bytes);
    if (!down_dst) {
      LOG_ERROR("[GGML-GLM5NEXT-FULL] DDR source arena exhausted at %s\n",
                down_name.c_str());
      return false;
    }
    g.expert_source_views.emplace(down_name, down_dst);
    g.expert_source_view_bytes.emplace(down_name, (size_t)down->size_bytes);
    jobs.push_back({down, down_dst, down_name});
  }
  if (jobs.empty())
    return true;

  const auto start = std::chrono::steady_clock::now();
  std::atomic<size_t> next{0};
  std::atomic<bool> failed{false};
  std::string first_error;
  std::mutex error_mutex;
  const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
  // Packed gate/up tensors need a temporary gate+up buffer while they are
  // interleaved into the graph arena.  Four readers made that temporary peak
  // large enough to compete with the 78 GiB source arena on a 128 GiB host.
  // This is boot-time I/O, so cap the default at two; an explicit override is
  // available for a machine with more headroom.
  unsigned read_threads = 2;
  if (const char *env = std::getenv("MOESTREAM_FULL_GRAPH_READ_THREADS")) {
    char *end = nullptr;
    const long n = std::strtol(env, &end, 10);
    if (end != env && *end == '\0' && n > 0)
      read_threads = (unsigned)std::min<long>(n, 16);
  }
  const unsigned n_threads =
      std::min<unsigned>(read_threads, std::min<unsigned>(hw, jobs.size()));
  auto worker = [&]() {
    for (;;) {
      const size_t i = next.fetch_add(1, std::memory_order_relaxed);
      if (i >= jobs.size())
        return;
      if (failed.load(std::memory_order_relaxed))
        return;
      // This is a one-time source-load phase, not a per-allocation hot path.
      // Poll between small batches so a page-cache-heavy unaligned GGUF can
      // fail closed before the host reaches swap pressure.
      if ((i & 15u) == 0 && MemoryGuard::allocation_blocked()) {
        if (!failed.exchange(true, std::memory_order_relaxed)) {
          std::lock_guard<std::mutex> lock(error_mutex);
          first_error = "host memory watermark during source preload";
        }
        return;
      }
      const Job &job = jobs[i];
      const bool ok = job.packed_meta
                          ? read_pack_gate_up(*g.engine, *job.meta,
                                              *job.packed_meta, job.dst)
                          : read_tensor_direct(*g.engine, *job.meta, job.dst);
      if (!ok) {
        if (!failed.exchange(true, std::memory_order_relaxed)) {
          std::lock_guard<std::mutex> lock(error_mutex);
          first_error = job.name;
        }
        return;
      }
    }
  };
  std::vector<std::thread> workers;
  workers.reserve(n_threads);
  for (unsigned i = 0; i < n_threads; ++i)
    workers.emplace_back(worker);
  for (auto &worker_thread : workers)
    worker_thread.join();
  if (failed.load(std::memory_order_relaxed)) {
    LOG_ERROR("[GGML-GLM5NEXT-FULL] parallel DDR source read failed: %s\n",
              first_error.c_str());
    return false;
  }
  const double seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
          .count();
  LOG_INFO("[GGML-GLM5NEXT-FULL] parallel DDR source read: %.2f GiB in %.3f s "
           "(%.2f GiB/s, %zu tensors, %u readers)\n",
           (double)g.expert_source_used / (1024.0 * 1024.0 * 1024.0), seconds,
           seconds > 0.0 ? (double)g.expert_source_used /
                               (seconds * 1024.0 * 1024.0 * 1024.0)
                         : 0.0,
           jobs.size(), n_threads);

  if (!compact_partial_source(g)) {
    LOG_ERROR("[GGML-FULL] partial expert source compaction failed\n");
    return false;
  }

  // A pageable source makes cudaMemcpyAsync pay the driver's staging cost at
  // every graph split.  That was the dominant measured GLM graph cost:
  // 8.50 s spent submitting 65.1 GiB of copies while the scheduler itself
  // needed only 1.30 s.  Register a bounded set of the already-loaded source
  // slices once, preserving the graph's single arena and avoiding a second
  // full-size host allocation.  The default cap is deliberately below the
  // 10--30 GiB aggregate registration failure range observed on this box;
  // callers can lower or raise it with MOESTREAM_FULL_GRAPH_PIN_GB, while a
  // registration failure stops the pass and leaves the remaining slices
  // pageable rather than retrying until the machine is under pressure.
  size_t pin_budget = g.engine->ram_pinned_budget_bytes;
  const char *pin_env = std::getenv("MOESTREAM_FULL_GRAPH_PIN_GB");
  if (pin_env && pin_env[0]) {
    char *end = nullptr;
    const double gb = std::strtod(pin_env, &end);
    if (end != pin_env && *end == '\0' && gb >= 0.0) {
      const double bytes = gb * 1024.0 * 1024.0 * 1024.0;
      pin_budget = bytes >= (double)SIZE_MAX ? SIZE_MAX : (size_t)bytes;
    }
  } else if (pin_budget > 24ull * 1024ull * 1024ull * 1024ull) {
    pin_budget = 24ull * 1024ull * 1024ull * 1024ull;
  }

  size_t pinned = 0;
  size_t attempts = 0;
  size_t failures = 0;
  if (pin_budget > 0) {
    const auto try_pin = [&](const std::string &name, void *ptr, size_t bytes) {
      if (!ptr || bytes == 0 || pinned > pin_budget ||
          bytes > pin_budget - pinned)
        return false;
      ++attempts;
      const cudaError_t err =
          cudaHostRegister(ptr, bytes, cudaHostRegisterPortable);
      if (err != cudaSuccess) {
        ++failures;
        cudaGetLastError();
        LOG_WARN("[GGML-GLM5NEXT-FULL] source pin stopped at %.2f GiB: "
                 "cudaHostRegister(%s, %.2f GiB) failed: %s\n",
                 (double)pinned / (1024.0 * 1024.0 * 1024.0), name.c_str(),
                 (double)bytes / (1024.0 * 1024.0 * 1024.0),
                 cudaGetErrorString(err));
        return false;
      }
      g.expert_source_pinned_ranges.push_back({ptr, bytes, true});
      pinned += bytes;
      return true;
    };

    bool stop = false;
    for (int l = c.first_dense; l < c.n_layers && !stop; ++l) {
      if (l < device_end)
        continue;
      PackedGateUpMeta packed;
      const std::string packed_name =
          "blk." + std::to_string(l) + ".ffn_gate_up_exps.weight";
      const std::string down_name =
          "blk." + std::to_string(l) + ".ffn_down_exps.weight";
      auto packed_it = g.expert_source_views.find(packed_name);
      auto down_it = g.expert_source_views.find(down_name);
      if (packed_gate_up_meta(*g.engine, l, &packed) &&
          packed_it != g.expert_source_views.end() &&
          down_it != g.expert_source_views.end()) {
        // Keep both projections of a layer together when the budget allows;
        // this preserves the normal gate/up and down copy behavior instead of
        // converting half the layer to pageable-source timing.
        const size_t packed_bytes = g.expert_source_view_bytes[packed_name];
        const size_t down_bytes = g.expert_source_view_bytes[down_name];
        const size_t layer_bytes = packed_bytes + down_bytes;
        if (pinned > pin_budget || layer_bytes > pin_budget - pinned)
          break;
        if (!try_pin(packed_name, packed_it->second, packed_bytes)) {
          stop = true;
          break;
        }
        if (!try_pin(down_name, down_it->second, down_bytes)) {
          stop = true;
          break;
        }
      } else {
        // Older/unusual files retain the separate three-projection source
        // layout.  Pin complete per-layer triplets under the same cap.
        size_t layer_bytes = 0;
        std::vector<std::pair<std::string, size_t>> parts;
        for (const char *stem : {"ffn_gate_exps.weight", "ffn_up_exps.weight",
                                 "ffn_down_exps.weight"}) {
          const std::string name = "blk." + std::to_string(l) + "." + stem;
          const GGUFTensor *m = model_tensor(*g.engine, name);
          auto it = g.expert_source_views.find(name);
          if (!m || it == g.expert_source_views.end()) {
            parts.clear();
            break;
          }
          const size_t view_bytes = g.expert_source_view_bytes[name];
          if (layer_bytes > SIZE_MAX - view_bytes) {
            parts.clear();
            break;
          }
          layer_bytes += view_bytes;
          parts.emplace_back(name, view_bytes);
        }
        if (parts.empty() || pinned > pin_budget ||
            layer_bytes > pin_budget - pinned)
          break;
        for (const auto &[name, bytes] : parts) {
          if (!try_pin(name, g.expert_source_views[name], bytes)) {
            stop = true;
            break;
          }
        }
      }
    }
  }
  g.host_pinned_bytes += pinned;
  g.host_pin_failures += failures;
  LOG_INFO("[GGML-GLM5NEXT-FULL] source pin: %.2f/%.2f GiB, %zu ranges, "
           "%zu attempts, %zu failures (bounded, one-time)\n",
           (double)pinned / (1024.0 * 1024.0 * 1024.0),
           (double)pin_budget / (1024.0 * 1024.0 * 1024.0),
           g.expert_source_pinned_ranges.size(), attempts, failures);
  return true;
}

size_t device_expert_bytes(const FullGraph &g) {
  const auto &c = g.engine->model_file.architecture_config;
  size_t total = 0;
  if (g.partial_expert_residency) {
    for (int l = c.first_dense; l < c.n_layers; ++l) {
      const PartialExpertLayer &p = g.partial_experts[(size_t)l];
      if (p.hot.empty())
        continue;
      PackedGateUpMeta packed;
      const std::string prefix = "blk." + std::to_string(l) + ".";
      if (packed_gate_up_meta(*g.engine, l, &packed)) {
        const GGUFTensor *down =
            model_tensor(*g.engine, prefix + "ffn_down_exps.weight");
        if (!down || c.n_experts <= 0 ||
            packed.bytes % (size_t)c.n_experts != 0 ||
            down->size_bytes % (size_t)c.n_experts != 0)
          return 0;
        const size_t per_gate_up = packed.bytes / (size_t)c.n_experts;
        const size_t per_down = (size_t)down->size_bytes / (size_t)c.n_experts;
        if (per_gate_up > SIZE_MAX / p.hot.size() ||
            total > SIZE_MAX - per_gate_up * p.hot.size())
          return 0;
        total += per_gate_up * p.hot.size();
        if (per_down > SIZE_MAX / p.hot.size() ||
            total > SIZE_MAX - per_down * p.hot.size())
          return 0;
        total += per_down * p.hot.size();
      } else {
        for (const char *stem : {"ffn_gate_exps.weight", "ffn_up_exps.weight",
                                 "ffn_down_exps.weight"}) {
          const GGUFTensor *m = model_tensor(*g.engine, prefix + stem);
          if (!m || c.n_experts <= 0 ||
              m->size_bytes % (size_t)c.n_experts != 0)
            return 0;
          const size_t per = (size_t)m->size_bytes / (size_t)c.n_experts;
          if (per > SIZE_MAX / p.hot.size() ||
              total > SIZE_MAX - per * p.hot.size())
            return 0;
          total += per * p.hot.size();
        }
      }
    }
    return total;
  }
  const int end = std::min(c.n_layers, c.first_dense + g.device_expert_layers);
  for (int l = c.first_dense; l < end; ++l) {
    for (const char *stem : {"ffn_gate_exps.weight", "ffn_up_exps.weight",
                             "ffn_down_exps.weight"}) {
      const GGUFTensor *m =
          model_tensor(*g.engine, "blk." + std::to_string(l) + "." + stem);
      if (!m ||
          total > std::numeric_limits<size_t>::max() - (size_t)m->size_bytes)
        return 0;
      total += (size_t)m->size_bytes;
    }
  }
  return total;
}

size_t dense_arena_bytes(const FullGraph &g) {
  size_t total = 0;
  for (const auto &[name, q] : g.engine->resident_layers) {
    if (!q.host_ptr || !graph_dense_weight_name(name))
      continue;
    if (total > std::numeric_limits<size_t>::max() - q.meta.size_bytes - 4096)
      return 0;
    total += q.meta.size_bytes + 4096;
  }
  const auto &c = g.engine->model_file.architecture_config;
  for (int l = 0; l < c.n_layers; ++l) {
    if (glm5next_is_kda_layer(c, l))
      continue;
    for (const char *stem : {"attn_k_b.weight", "attn_v_b.weight"}) {
      const std::string name = "blk." + std::to_string(l) + "." + stem;
      // These tensors are normally in resident_layers already (the legacy
      // path keeps them host-side to build WUK/WUV).  Only add them here when
      // a manifest variant omitted them and static_full_weight() will read
      // them directly during graph construction.
      if (resident(*g.engine, name))
        continue;
      const GGUFTensor *m = model_tensor(*g.engine, name);
      if (m) {
        if (total > std::numeric_limits<size_t>::max() - m->size_bytes - 4096)
          return 0;
        total += (size_t)m->size_bytes + 4096;
      }
    }
  }
  if (g.partial_expert_residency) {
    // Two I32 ID maps and two F32 masks per MoE layer.  They are tiny, but
    // account for them explicitly so a nearly-full CUDA dense arena does not
    // rely on its generic slack by accident.
    const size_t per_layer = (size_t)c.n_experts * (size_t)g.S *
                             (2 * sizeof(int32_t) + 2 * sizeof(float));
    if ((size_t)std::max(0, c.n_layers - c.first_dense) >
        (SIZE_MAX - total) / (per_layer + 4096))
      return 0;
    total +=
        (size_t)std::max(0, c.n_layers - c.first_dense) * (per_layer + 4096);
  }
  return total + (1ull << 20);
}

bool init_dense_arena(FullGraph &g) {
  const size_t bytes = dense_arena_bytes(g);
  if (bytes == 0)
    return false;
  const auto buft = ggml_backend_get_default_buffer_type(g.cuda);
  g.dense_buffer = ggml_backend_buft_alloc_buffer(buft, bytes);
  if (!g.dense_buffer) {
    LOG_WARN("[GGML-GLM5NEXT-FULL] dense arena allocation declined: %.2f GiB\n",
             (double)bytes / (1024.0 * 1024.0 * 1024.0));
    return false;
  }
  ggml_backend_buffer_set_usage(g.dense_buffer,
                                GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
  g.dense_base = ggml_backend_buffer_get_base(g.dense_buffer);
  g.dense_capacity = ggml_backend_buffer_get_size(g.dense_buffer);
  return true;
}

bool init_expert_arena(FullGraph &g) {
  if (!g.partial_expert_residency && g.device_expert_layers <= 0)
    return true;
  const size_t bytes = device_expert_bytes(g);
  if (bytes == 0)
    return false;
  const size_t capacity =
      bytes + (size_t)std::max(1, g.n_layers) * 3 * 256 + (1ull << 20);
  const auto buft = ggml_backend_get_default_buffer_type(g.cuda);
  g.expert_buffer = ggml_backend_buft_alloc_buffer(buft, capacity);
  if (!g.expert_buffer) {
    LOG_WARN(
        "[GGML-GLM5NEXT-FULL] expert arena allocation declined: %.2f GiB\n",
        (double)capacity / (1024.0 * 1024.0 * 1024.0));
    return false;
  }
  ggml_backend_buffer_set_usage(g.expert_buffer,
                                GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
  g.expert_base = ggml_backend_buffer_get_base(g.expert_buffer);
  g.expert_capacity = ggml_backend_buffer_get_size(g.expert_buffer);
  LOG_INFO("[GGML-GLM5NEXT-FULL] device expert arena=%.2f GiB%s\n",
           (double)g.expert_capacity / (1024.0 * 1024.0 * 1024.0),
           g.partial_expert_residency ? " for partial hot expert set"
                                      : " for prefix MoE layers");
  return true;
}

bool init_expert_source(FullGraph &g, size_t source_bytes) {
  // Leave one page of alignment slack per projection.  The reservation is
  // still based on the exact GGUF bytes; this small fixed overhead is covered
  // by the graph's existing watermark and is reclaimed with the graph.
  constexpr size_t kSlack = 1ull << 20;
  if (source_bytes == 0)
    return true;
  if (source_bytes > SIZE_MAX - kSlack)
    return false;
  const size_t capacity = source_bytes + kSlack;
  void *p = nullptr;
  if (posix_memalign(&p, 4096, capacity) != 0 || !p) {
    LOG_WARN("[GGML-GLM5NEXT-FULL] DDR expert source allocation declined: %.2f "
             "GiB\n",
             (double)capacity / (1024.0 * 1024.0 * 1024.0));
    return false;
  }
  // The source reservation is admitted by prepare() before this allocation;
  // do not charge the same arena a second time as an ordinary owned buffer.
  g.owned.push_back({p, capacity, false, 0});
  g.expert_source_base = p;
  g.expert_source_capacity = capacity;
  g.expert_source_used = 0;
  g.expert_source_tensors = 0;
  LOG_INFO("[GGML-GLM5NEXT-FULL] DDR expert source arena=%.2f GiB (one "
           "allocation)\n",
           (double)capacity / (1024.0 * 1024.0 * 1024.0));
  return true;
}

bool source_configuration_matches(const FullGraph &source,
                                  const FullGraph &candidate) {
  return source.has_expert_source() && source.engine == candidate.engine &&
         source.device_expert_layers == candidate.device_expert_layers &&
         source.partial_expert_residency ==
             candidate.partial_expert_residency &&
         source.partial_usage_file == candidate.partial_usage_file;
}

bool prepare_expert_source_only(FullGraph &g, const char *tag) {
  const size_t source = expert_source_bytes(*g.engine, g.device_expert_layers);
  constexpr size_t kSourceSlack = 1ull << 20;
  // 10 GiB, per explicit user instruction after the 2026-09-09 crash (was
  // 8 GiB) -- align every host-memory watermark in this file to the same
  // floor rather than several different ad hoc numbers.
  constexpr size_t kWatermark = 10ull * 1024ull * 1024ull * 1024ull;
  const size_t host_live = graph_host_live_bytes(*g.engine);
  if (source > SIZE_MAX - kSourceSlack ||
      (source > 0 && !MemoryGuard::try_reserve_host_bytes(source + kSourceSlack,
                                                          kWatermark))) {
    LOG_WARN(
        "[%s] boot host admission refused source=%.2f GiB existing=%.2f GiB "
        "reserve=%.2f GiB budget=%.2f GB MemAvailable=%ld MiB\n",
        tag, (double)source / (1024.0 * 1024.0 * 1024.0),
        (double)host_live / (1024.0 * 1024.0 * 1024.0),
        (double)kWatermark / (1024.0 * 1024.0 * 1024.0),
        (double)MemoryGuard::host_budget_bytes() / 1.0e9,
        MemoryGuard::last_available_mb());
    return false;
  }
  g.host_reservation_bytes = source + (source ? kSourceSlack : 0);
  LOG_INFO("[%s] boot host admission: source=%.2f GiB existing=%.2f GiB "
           "reserve=%.2f GiB budget=%.2f GB\n",
           tag, (double)source / (1024.0 * 1024.0 * 1024.0),
           (double)host_live / (1024.0 * 1024.0 * 1024.0),
           (double)kWatermark / (1024.0 * 1024.0 * 1024.0),
           (double)MemoryGuard::host_budget_bytes() / 1.0e9);
  if (!init_expert_source(g, source) || !preload_expert_source(g)) {
    LOG_WARN("[%s] boot expert source preload failed; graph will use the "
             "legacy path\n",
             tag);
    return false;
  }
  LOG_INFO("[%s] boot expert source ready: %.2f GiB DDR, %zu tensors; "
           "prompt graph remains lazy\n",
           tag, (double)g.expert_source_used / (1024.0 * 1024.0 * 1024.0),
           g.expert_source_tensors);
  return true;
}

ggml_tensor *hc_affine(ggml_context *ctx, ggml_tensor *x, ggml_tensor *scale,
                       ggml_tensor *base) {
  return ggml_add(ctx, ggml_mul(ctx, x, scale), base);
}

ggml_tensor *hc_mean(FullGraph &g, ggml_context *ctx, ggml_tensor *x) {
  ggml_tensor *out = ggml_view_2d(ctx, x, g.D, g.S, x->nb[2], 0);
  for (int h = 1; h < g.HC; ++h)
    out = ggml_add(
        ctx, out,
        ggml_view_2d(ctx, x, g.D, g.S, x->nb[2], (size_t)h * x->nb[1]));
  return ggml_scale(ctx, out, 1.0f / (float)g.HC);
}

ggml_tensor *hc_pre(FullGraph &g, ggml_context *ctx, ggml_tensor *x,
                    ggml_tensor *fn, ggml_tensor *scale, ggml_tensor *base,
                    ggml_tensor **post, ggml_tensor **comb) {
  const auto &c = g.engine->model_file.architecture_config;
  const int mix_dim = (2 + g.HC) * g.HC;
  ggml_tensor *flat = ggml_reshape_2d(ctx, x, g.D * g.HC, g.S);
  ggml_tensor *flat_norm = ggml_rms_norm(ctx, flat, c.eps);
  ggml_tensor *mixes = ggml_mul_mat(ctx, fn, flat_norm);

  ggml_tensor *scale_pre = ggml_view_1d(ctx, scale, 1, 0);
  ggml_tensor *scale_post = ggml_view_1d(ctx, scale, 1, sizeof(float));
  ggml_tensor *scale_comb = ggml_view_1d(ctx, scale, 1, 2 * sizeof(float));
  ggml_tensor *base_pre = ggml_view_1d(ctx, base, g.HC, 0);
  ggml_tensor *base_post = ggml_view_1d(ctx, base, g.HC, g.HC * sizeof(float));
  ggml_tensor *base_comb =
      ggml_view_1d(ctx, base, g.HC * g.HC, 2 * g.HC * sizeof(float));

  ggml_tensor *pre = ggml_view_2d(ctx, mixes, g.HC, g.S, mixes->nb[1], 0);
  pre = ggml_scale_bias(
      ctx, ggml_sigmoid(ctx, hc_affine(ctx, pre, scale_pre, base_pre)), 1.0f,
      c.hc_eps);
  *post = ggml_view_2d(ctx, mixes, g.HC, g.S, mixes->nb[1],
                       (size_t)g.HC * mixes->nb[0]);
  *post = ggml_scale(
      ctx, ggml_sigmoid(ctx, hc_affine(ctx, *post, scale_post, base_post)),
      2.0f);
  *comb = ggml_view_2d(ctx, mixes, g.HC * g.HC, g.S, mixes->nb[1],
                       (size_t)2 * g.HC * mixes->nb[0]);
  *comb = hc_affine(ctx, *comb, scale_comb, base_comb);
  *comb = ggml_reshape_3d(ctx, *comb, g.HC, g.HC, g.S);

  // Exact DeepSeek-V4/GLM reference order: row softmax, epsilon, one column
  // normalization, then row/column Sinkhorn iterations.
  ggml_tensor *eps = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
  eps = ggml_fill(ctx, eps, c.hc_eps);
  *comb = ggml_add(ctx, ggml_soft_max(ctx, *comb), eps);
  auto norm_cols = [&]() {
    ggml_tensor *p = ggml_cont(ctx, ggml_permute(ctx, *comb, 1, 0, 2, 3));
    ggml_tensor *sum = ggml_add(ctx, ggml_sum_rows(ctx, p), eps);
    sum = ggml_permute(ctx, sum, 1, 0, 2, 3);
    *comb = ggml_div(ctx, *comb, sum);
  };
  auto norm_rows = [&]() {
    ggml_tensor *sum = ggml_add(ctx, ggml_sum_rows(ctx, *comb), eps);
    *comb = ggml_div(ctx, *comb, sum);
  };
  norm_cols();
  for (int i = 1; i < c.hc_sinkhorn_iters; ++i) {
    norm_rows();
    norm_cols();
  }

  ggml_tensor *result = nullptr;
  for (int h = 0; h < g.HC; ++h) {
    ggml_tensor *xh =
        ggml_view_2d(ctx, x, g.D, g.S, x->nb[2], (size_t)h * x->nb[1]);
    ggml_tensor *ph =
        ggml_view_2d(ctx, pre, 1, g.S, pre->nb[1], (size_t)h * pre->nb[0]);
    ggml_tensor *term = ggml_mul(ctx, xh, ph);
    result = result ? ggml_add(ctx, result, term) : term;
  }
  return result;
}

ggml_tensor *hc_post(FullGraph &g, ggml_context *ctx, ggml_tensor *x,
                     ggml_tensor *residual, ggml_tensor *post,
                     ggml_tensor *comb) {
  ggml_tensor *out = nullptr;
  for (int dst = 0; dst < g.HC; ++dst) {
    ggml_tensor *pd =
        ggml_view_2d(ctx, post, 1, g.S, post->nb[1], (size_t)dst * post->nb[0]);
    ggml_tensor *cur = ggml_mul(ctx, x, pd);
    for (int src = 0; src < g.HC; ++src) {
      ggml_tensor *rs = ggml_view_2d(ctx, residual, g.D, g.S, residual->nb[2],
                                     (size_t)src * residual->nb[1]);
      ggml_tensor *cw =
          ggml_view_2d(ctx, comb, 1, g.S, comb->nb[2],
                       (size_t)dst * comb->nb[0] + (size_t)src * comb->nb[1]);
      cur = ggml_add(ctx, cur, ggml_mul(ctx, rs, cw));
    }
    ggml_tensor *stream = ggml_reshape_3d(ctx, cur, g.D, 1, g.S);
    out = out ? ggml_concat(ctx, out, stream, 1) : stream;
  }
  return out;
}

ggml_tensor *build_kda(FullGraph &g, ggml_context *ctx, ggml_tensor *cur,
                       int layer, ggml_tensor *wq, ggml_tensor *wk,
                       ggml_tensor *wv, ggml_tensor *cq, ggml_tensor *ck,
                       ggml_tensor *cv, ggml_tensor *fa, ggml_tensor *fb,
                       ggml_tensor *ga, ggml_tensor *gb, ggml_tensor *beta_w,
                       ggml_tensor *a, ggml_tensor *dt, ggml_tensor *norm_w,
                       ggml_tensor *out_w, ggml_tensor *hist_in,
                       ggml_tensor *state_in, ggml_tensor **state_out,
                       ggml_tensor **hist_out) {
  const auto &c = g.engine->model_file.architecture_config;
  ggml_tensor *q = ggml_mul_mat(ctx, wq, cur);
  ggml_tensor *k = ggml_mul_mat(ctx, wk, cur);
  ggml_tensor *v = ggml_mul_mat(ctx, wv, cur);
  ggml_tensor *qkv = ggml_concat(ctx, ggml_concat(ctx, q, k, 0), v, 0);
  qkv = ggml_reshape_3d(ctx, qkv, 3 * g.d_inner, g.S, 1);

  ggml_tensor *conv_w = ggml_concat(
      ctx,
      ggml_concat(ctx, ggml_reshape_2d(ctx, cq, g.d_conv, g.d_inner),
                  ggml_reshape_2d(ctx, ck, g.d_conv, g.d_inner), 1),
      ggml_reshape_2d(ctx, cv, g.d_conv, g.d_inner), 1);
  ggml_tensor *conv_in = ggml_concat(ctx, hist_in, ggml_transpose(ctx, qkv), 0);
  ggml_tensor *conv = ggml_silu(ctx, ggml_ssm_conv(ctx, conv_in, conv_w));
  const size_t row = ggml_row_size(conv->type, 3 * g.d_inner);
  const size_t head_row = ggml_row_size(conv->type, g.hd);
  q = ggml_view_4d(ctx, conv, g.hd, g.H, g.S, 1, head_row, row, row * g.S, 0);
  k = ggml_view_4d(ctx, conv, g.hd, g.H, g.S, 1, head_row, row, row * g.S,
                   ggml_row_size(conv->type, g.d_inner));
  v = ggml_view_4d(ctx, conv, g.hd, g.H, g.S, 1, head_row, row, row * g.S,
                   ggml_row_size(conv->type, 2 * g.d_inner));
  q = ggml_l2_norm(ctx, q, 1e-6f);
  k = ggml_l2_norm(ctx, k, 1e-6f);

  ggml_tensor *decay = ggml_mul_mat(ctx, fb, ggml_mul_mat(ctx, fa, cur));
  decay = ggml_add(ctx, decay, dt);
  decay = ggml_reshape_3d(ctx, decay, g.hd, g.H, g.S);
  decay = ggml_mul(ctx, decay, ggml_reshape_3d(ctx, a, 1, g.H, 1));
  decay = ggml_scale(ctx, ggml_sigmoid(ctx, ggml_scale(ctx, decay, -1.0f)),
                     c.kda_gate_lower_bound);
  decay = ggml_reshape_4d(ctx, decay, g.hd, g.H, g.S, 1);

  ggml_tensor *beta = ggml_mul_mat(ctx, beta_w, cur);
  beta = ggml_sigmoid(ctx, ggml_reshape_4d(ctx, beta, 1, g.H, g.S, 1));
  ggml_tensor *scan =
      ggml_gated_delta_net(ctx, q, k, v, decay, beta, state_in, 1);

  const size_t value_row = ggml_row_size(scan->type, g.d_inner);
  const size_t state_row = ggml_row_size(scan->type, g.hd * g.hd);
  // ggml-cuda's gated_delta_net kernel stores the per-head state TRANSPOSED
  // relative to the [k][v] math convention this codebase's own legacy CPU
  // qwen4exp_gdn_delta_step (and, by the identical pattern, the KDA
  // recurrence) use -- confirmed directly in
  // ggml/src/ggml-cuda/gated_delta_net.cu's own comment ("state is stored
  // transposed: M[col][i] = S[i][col]", where col indexes v and i indexes
  // k), i.e. the raw view below has ne[0]=k (contiguous), ne[1]=v. Legacy's
  // own flat layout (glm5next_recur_state[l], engine.hpp) is st[k*hd+v] --
  // v contiguous, k slower -- the opposite axis order. Left unfixed, the
  // flat ggml_backend_tensor_get+memcpy in glm5next_full_graph_run silently
  // hands decode a transposed state matrix: same byte count, same total
  // size (hd==hd), no crash, just wrong values from the very first decode
  // token onward. Transpose+materialize here so the exported tensor's own
  // byte layout already matches what the flat copy-out assumes.
  ggml_tensor *state_raw =
      ggml_view_4d(ctx, scan, g.hd, g.hd, g.H, 1, head_row, state_row,
                   state_row * g.H, (size_t)g.d_inner * g.S * sizeof(float));
  *state_out = ggml_cont(ctx, ggml_transpose(ctx, state_raw));
  ggml_tensor *core = ggml_view_4d(ctx, scan, g.hd, g.H, g.S, 1, head_row,
                                   value_row, value_row * g.S, 0);
  ggml_tensor *norm = ggml_mul(ctx, ggml_rms_norm(ctx, core, c.eps), norm_w);
  ggml_tensor *gate = ggml_mul_mat(ctx, gb, ggml_mul_mat(ctx, ga, cur));
  gate = ggml_reshape_3d(ctx, gate, g.hd, g.H, g.S);
  ggml_tensor *gated = ggml_mul(ctx, norm, ggml_sigmoid(ctx, gate));
  ggml_tensor *out =
      ggml_mul_mat(ctx, out_w, ggml_reshape_2d(ctx, gated, g.d_inner, g.S));
  *hist_out =
      ggml_cont(ctx, ggml_view_3d(ctx, conv_in, g.d_conv - 1, 3 * g.d_inner, 1,
                                  conv_in->nb[1], conv_in->nb[2],
                                  ggml_row_size(conv_in->type, g.S)));
  (void)layer;
  return out;
}

ggml_tensor *build_dsa(FullGraph &g, ggml_context *ctx, ggml_tensor *cur,
                       int layer, ggml_tensor *wqa, ggml_tensor *wqan,
                       ggml_tensor *wqb, ggml_tensor *wkva, ggml_tensor *wkvan,
                       ggml_tensor *wkb, ggml_tensor *wvb, ggml_tensor *wo,
                       ggml_tensor **kv_out) {
  const auto &c = g.engine->model_file.architecture_config;
  ggml_tensor *qr = ggml_rms_norm(ctx, ggml_mul_mat(ctx, wqa, cur), c.eps);
  qr = ggml_mul(ctx, qr, wqan);
  ggml_tensor *q = ggml_mul_mat(ctx, wqb, qr);
  q = ggml_reshape_3d(ctx, q, g.qk_nope, g.H, g.S);
  ggml_tensor *kv = ggml_mul_mat(ctx, wkva, cur);
  kv = ggml_mul(ctx, ggml_rms_norm(ctx, kv, c.eps), wkvan);
  *kv_out = kv;

  q = ggml_permute(ctx, q, 0, 2, 1, 3);
  q = ggml_mul_mat(ctx, wkb, q);
  q = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));
  const int64_t kv_tokens = ((int64_t)g.S + 255) & ~((int64_t)255);
  ggml_tensor *latent = ggml_reshape_3d(ctx, kv, g.kv_lora, 1, g.S);
  if (kv_tokens > g.S) {
    // ggml-cuda's 512-wide FATTN kernels require the KV sequence length to
    // be a multiple of FATTN_KQ_STRIDE (256).  Bee's normal KV-cache path
    // already supplies a padded cache.  This standalone no-cache graph must
    // provide the same layout explicitly; zero padding is masked out below.
    latent = ggml_pad(ctx, latent, 0, 0, (int)(kv_tokens - g.S), 0);
  }

  // Keep this layout sequence identical to llama.cpp's build_attn_mha().
  // In particular, the FATTN CUDA dispatch expects the absorbed MQA key and
  // value to be permuted after q has been split into streams, and it expects
  // F32 latent K/V to be explicitly converted to F16.  The previous manual
  // path omitted those details, causing every DSA FLASH_ATTN_EXT node to
  // fall back to the CPU scheduler backend.
  const bool v_trans = latent->nb[1] > latent->nb[2];
  const int64_t n_stream = latent->ne[3];
  q = ggml_view_4d(ctx, q, q->ne[0], q->ne[1], q->ne[2] / n_stream, n_stream,
                   q->nb[1], q->nb[2], q->nb[3] / n_stream, 0);
  q = ggml_permute(ctx, q, 0, 2, 1, 3);
  ggml_tensor *k = ggml_permute(ctx, latent, 0, 2, 1, 3);
  ggml_tensor *v = ggml_permute(ctx, latent, 0, 2, 1, 3);
  if (v_trans)
    v = ggml_transpose(ctx, v);
  if (k->type == GGML_TYPE_F32)
    k = ggml_cast(ctx, k, GGML_TYPE_F16);
  if (v->type == GGML_TYPE_F32)
    v = ggml_cast(ctx, v, GGML_TYPE_F16);

  ggml_tensor *attn = ggml_flash_attn_ext(
      ctx, q, k, v, g.attn_mask,
      1.0f / std::sqrt((float)(g.qk_nope > 0 ? g.qk_nope : 1)), 0.0f, 0.0f);
  if (g.dsa_layers.empty()) {
    LOG_INFO("[GGML-GLM5NEXT-FULL] DSA FATTN layer=%d support_cuda=%d "
             "q=(%lld,%lld,%lld,%lld)/%s k=(%lld,%lld,%lld,%lld)/%s "
             "v=(%lld,%lld,%lld,%lld)/%s mask=(%lld,%lld,%lld,%lld)/%s\n",
             layer, ggml_backend_supports_op(g.cuda, attn) ? 1 : 0,
             (long long)q->ne[0], (long long)q->ne[1], (long long)q->ne[2],
             (long long)q->ne[3], ggml_type_name(q->type), (long long)k->ne[0],
             (long long)k->ne[1], (long long)k->ne[2], (long long)k->ne[3],
             ggml_type_name(k->type), (long long)v->ne[0], (long long)v->ne[1],
             (long long)v->ne[2], (long long)v->ne[3], ggml_type_name(v->type),
             (long long)g.attn_mask->ne[0], (long long)g.attn_mask->ne[1],
             (long long)g.attn_mask->ne[2], (long long)g.attn_mask->ne[3],
             ggml_type_name(g.attn_mask->type));
  }
  // Same v_mla path as build_attn_mha(): FATTN [kv_lora,S,H] is interpreted
  // as [kv_lora,H,S] for WUV, then made contiguous before the 2D projection.
  ggml_tensor *value = ggml_permute(ctx, attn, 0, 2, 1, 3);
  value = ggml_mul_mat(ctx, wvb, value);
  value = ggml_permute(ctx, value, 0, 2, 1, 3);
  value = ggml_cont(ctx, value);
  value = ggml_reshape_2d(ctx, value, g.v_head * g.H, g.S);
  (void)layer;
  return ggml_mul_mat(ctx, wo, value);
}

ggml_tensor *build_ffn(FullGraph &g, ggml_context *ctx, ggml_tensor *cur,
                       int layer, ggml_tensor *forced_ids = nullptr) {
  const std::string p = "blk." + std::to_string(layer) + ".";
  const auto &c = g.engine->model_file.architecture_config;
  if (layer < c.first_dense) {
    ggml_tensor *gate = resident_tensor(g, ctx, p + "ffn_gate.weight");
    ggml_tensor *up = resident_tensor(g, ctx, p + "ffn_up.weight");
    ggml_tensor *down = resident_tensor(g, ctx, p + "ffn_down.weight");
    if (!gate || !up || !down)
      return nullptr;
    ggml_tensor *a = ggml_mul_mat(ctx, gate, cur);
    ggml_tensor *b = ggml_mul_mat(ctx, up, cur);
    ggml_tensor *act = ggml_swiglu_split(ctx, a, b);
    return ggml_mul_mat(ctx, down, act);
  }

  ggml_tensor *router = resident_tensor(g, ctx, p + "ffn_gate_inp.weight");
  ggml_tensor *bias = resident_tensor(g, ctx, p + "exp_probs_b.bias");
  ggml_tensor *gate_up = expert_gate_up_weight(g, ctx, layer);
  ggml_tensor *ge = nullptr;
  ggml_tensor *ue = nullptr;
  if (!gate_up) {
    // Keep the separate projection path for older or unusual GGUFs whose
    // gate/up tensors cannot be represented as one contiguous quantized
    // tensor.  GLM-5.3 and DeepSeek-V4 use compatible layouts and take the
    // fused path above.
    ge = expert_weight(g, ctx, layer, "ffn_gate_exps.weight");
    ue = expert_weight(g, ctx, layer, "ffn_up_exps.weight");
  }
  ggml_tensor *de = expert_weight(g, ctx, layer, "ffn_down_exps.weight");
  const PartialExpertLayer *partial_weights = partial_layer(g, layer);
  if (!router)
    return nullptr;
  if (!partial_weights && (!gate_up && (!ge || !ue)) ||
      (!partial_weights && !de))
    return nullptr;
  if (partial_weights && !partial_weights->cold.empty() &&
      ((!gate_up && (!ge || !ue)) || !de))
    return nullptr;

  ggml_tensor *logits = ggml_mul_mat(ctx, router, cur);
  ggml_tensor *probs = nullptr;
  switch (c.gating_func) {
  case GatingFunc::Sigmoid:
    probs = ggml_sigmoid(ctx, logits);
    break;
  case GatingFunc::SqrtSoftplus:
    probs = ggml_sqrt(ctx, ggml_softplus(ctx, logits));
    break;
  case GatingFunc::Softmax:
  default:
    probs = ggml_soft_max(ctx, logits);
    break;
  }
  ggml_tensor *selection = bias ? ggml_add(ctx, probs, bias) : probs;
  // DeepSeek-V4's first hash-routed layers use a token-id -> expert table
  // for selection, while the learned router still supplies the unbiased
  // weights.  The table is prepared as a host input by the V4 graph builder;
  // every other layer keeps GGML's normal biased top-k node.
  ggml_tensor *ids =
      forced_ids ? forced_ids : ggml_argsort_top_k(ctx, selection, c.topk);
  ggml_tensor *weights =
      ggml_get_rows(ctx, ggml_reshape_3d(ctx, probs, 1, c.n_experts, g.S), ids);
  ggml_tensor *weights2 = ggml_reshape_2d(ctx, weights, c.topk, g.S);
  if (c.expert_weights_norm) {
    ggml_tensor *sum = ggml_clamp(ctx, ggml_sum_rows(ctx, weights2),
                                  6.103515625e-5f, INFINITY);
    weights2 = ggml_div(ctx, weights2, sum);
  }
  if (c.expert_weights_scale != 0.0f && c.expert_weights_scale != 1.0f)
    weights2 = ggml_scale(ctx, weights2, c.expert_weights_scale);
  weights = ggml_reshape_3d(ctx, weights2, 1, c.topk, g.S);
  ggml_build_forward_expand(g.graph, weights);

  float clamp = 0.0f;
  if (layer >= 0 && layer < (int)c.swiglu_clamp_exp.size())
    clamp = c.swiglu_clamp_exp[(size_t)layer];
  ggml_tensor *input3 = ggml_reshape_3d(ctx, cur, g.D, 1, g.S);
  if (g.partial_expert_residency && !prepare_partial_hot(g, ctx, layer))
    return nullptr;

  const auto run_branch = [&](ggml_tensor *branch_gate_up,
                              ggml_tensor *branch_gate, ggml_tensor *branch_up,
                              ggml_tensor *branch_down, ggml_tensor *branch_ids,
                              ggml_tensor *branch_weights) -> ggml_tensor * {
    if (!branch_down || !branch_ids || !branch_weights ||
        (!branch_gate_up && (!branch_gate || !branch_up)))
      return nullptr;
    ggml_tensor *up = nullptr;
    ggml_tensor *gate = nullptr;
    if (branch_gate_up) {
      // Match Llama.cpp's merged gate_up graph: one routed projection produces
      // [2*n_ff, top_k, S], then zero-copy views feed SwiGLU.
      ggml_tensor *gate_up_out =
          ggml_mul_mat_id(ctx, branch_gate_up, input3, branch_ids);
      const int64_t n_ff = gate_up_out->ne[0] / 2;
      gate = ggml_view_3d(ctx, gate_up_out, n_ff, gate_up_out->ne[1],
                          gate_up_out->ne[2], gate_up_out->nb[1],
                          gate_up_out->nb[2], 0);
      up = ggml_view_3d(ctx, gate_up_out, n_ff, gate_up_out->ne[1],
                        gate_up_out->ne[2], gate_up_out->nb[1],
                        gate_up_out->nb[2], (size_t)n_ff * gate_up_out->nb[0]);
    } else {
      up = ggml_mul_mat_id(ctx, branch_up, input3, branch_ids);
      gate = ggml_mul_mat_id(ctx, branch_gate, input3, branch_ids);
    }
    ggml_tensor *act = nullptr;
    if (clamp > 1e-6f && swiglu_clamp_fused()) {
      ggml_tensor *up_clamped = ggml_clamp(ctx, up, -clamp, clamp);
      ggml_tensor *gate_clamped = ggml_clamp(ctx, gate, -INFINITY, clamp);
      act = ggml_mul(ctx, ggml_silu(ctx, gate_clamped), up_clamped);
    } else {
      act = ggml_swiglu_split(ctx, gate, up);
      if (clamp > 1e-6f)
        act = ggml_clamp(ctx, act, -clamp, clamp);
    }
    ggml_tensor *experts = ggml_mul_mat_id(ctx, branch_down, act, branch_ids);
    experts = ggml_mul(ctx, experts, branch_weights);
    ggml_tensor *branch =
        ggml_cont(ctx, ggml_view_2d(ctx, experts, g.D, g.S, experts->nb[2], 0));
    for (int k = 1; k < c.topk; ++k)
      branch = ggml_add(ctx, branch,
                        ggml_view_2d(ctx, experts, g.D, g.S, experts->nb[2],
                                     (size_t)k * experts->nb[1]));
    return branch;
  };

  ggml_tensor *out = nullptr;
  if (PartialExpertLayer *partial = partial_layer(g, layer)) {
    const bool has_hot = !partial->hot.empty();
    const bool has_cold = !partial->cold.empty();
    ggml_tensor *hot_ids = nullptr;
    ggml_tensor *cold_ids = nullptr;
    ggml_tensor *hot_weights = nullptr;
    ggml_tensor *cold_weights = nullptr;
    if (has_hot) {
      ggml_tensor *hot_map = partial_map(g, ctx, layer, true, false);
      ggml_tensor *hot_mask = partial_map(g, ctx, layer, true, true);
      if (!hot_map || !hot_mask)
        return nullptr;
      ggml_tensor *mapped = ggml_get_rows(ctx, hot_map, ids);
      ggml_tensor *mask = ggml_get_rows(ctx, hot_mask, ids);
      hot_ids = ggml_reshape_2d(ctx, mapped, c.topk, g.S);
      hot_weights =
          ggml_mul(ctx, weights, ggml_reshape_3d(ctx, mask, 1, c.topk, g.S));
    }
    if (has_cold) {
      ggml_tensor *cold_map = partial_map(g, ctx, layer, false, false);
      ggml_tensor *cold_mask = partial_map(g, ctx, layer, false, true);
      if (!cold_map || !cold_mask)
        return nullptr;
      ggml_tensor *mapped = ggml_get_rows(ctx, cold_map, ids);
      ggml_tensor *mask = ggml_get_rows(ctx, cold_mask, ids);
      cold_ids = ggml_reshape_2d(ctx, mapped, c.topk, g.S);
      cold_weights =
          ggml_mul(ctx, weights, ggml_reshape_3d(ctx, mask, 1, c.topk, g.S));
    }
    if (has_hot) {
      ggml_tensor *hot =
          run_branch(partial->hot_gate_up, partial->hot_gate, partial->hot_up,
                     partial->hot_down, hot_ids, hot_weights);
      if (!hot)
        return nullptr;
      out = hot;
    }
    if (has_cold) {
      ggml_tensor *cold =
          run_branch(gate_up, ge, ue, de, cold_ids, cold_weights);
      if (!cold)
        return nullptr;
      out = out ? ggml_add(ctx, out, cold) : cold;
    }
  } else {
    out = run_branch(gate_up, ge, ue, de, ids, weights);
    if (!out)
      return nullptr;
  }

  ggml_tensor *sg = resident_tensor(g, ctx, p + "ffn_gate_shexp.weight");
  ggml_tensor *su = resident_tensor(g, ctx, p + "ffn_up_shexp.weight");
  ggml_tensor *sd = resident_tensor(g, ctx, p + "ffn_down_shexp.weight");
  if (sg && su && sd) {
    ggml_tensor *gate_shared = ggml_mul_mat(ctx, sg, cur);
    ggml_tensor *up_shared = ggml_mul_mat(ctx, su, cur);
    const float clamp_shared =
        (layer >= 0 && layer < (int)c.swiglu_clamp_shexp.size())
            ? c.swiglu_clamp_shexp[(size_t)layer]
            : 0.0f;
    ggml_tensor *a = nullptr;
    if (clamp_shared > 1e-6f && swiglu_clamp_fused()) {
      // GLM5NEXT/reference ggml_swiglu_clamp semantics.
      up_shared = ggml_clamp(ctx, up_shared, -clamp_shared, clamp_shared);
      gate_shared = ggml_clamp(ctx, gate_shared, -INFINITY, clamp_shared);
      a = ggml_mul(ctx, ggml_silu(ctx, gate_shared), up_shared);
    } else {
      a = ggml_swiglu_split(ctx, gate_shared, up_shared);
      // DeepSeek-V4's established path clamps the completed SwiGLU product
      // when the fused GLM5NEXT flag is not active.
      if (clamp_shared > 1e-6f)
        a = ggml_clamp(ctx, a, -clamp_shared, clamp_shared);
    }
    out = ggml_add(ctx, out, ggml_mul_mat(ctx, sd, a));
  }
  return out;
}

bool build_graph(FullGraph &g) {
  const auto &c = g.engine->model_file.architecture_config;
  ggml_init_params ip = {};
  ip.mem_size = kTensorContextBytes;
  ip.no_alloc = true;
  if (!MemoryGuard::try_reserve_host_bytes(kTensorContextBytes)) {
    LOG_WARN("[GGML-GLM5NEXT-FULL-MEM] host graph context refused: %.2f MiB "
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
                       (size_t)g.D * g.HC * g.S * sizeof(float),
                       "glm5next_full_input");
  const int64_t kv_tokens = ((int64_t)g.S + 255) & ~((int64_t)255);
  g.attn_mask = host_input(g, g.ctx, GGML_TYPE_F16, {kv_tokens, g.S, 1, 1},
                           (size_t)kv_tokens * g.S * sizeof(ggml_fp16_t),
                           "glm5next_full_causal_mask");
  if (!g.input || !g.attn_mask)
    return false;
  auto *mask = static_cast<ggml_fp16_t *>(g.host_inputs[1]);
  for (int q = 0; q < g.S; ++q)
    for (int k = 0; k < kv_tokens; ++k)
      mask[(size_t)q * kv_tokens + k] =
          ggml_fp32_to_fp16(k < g.S && k <= q ? 0.0f : -INFINITY);

  ggml_tensor *res = g.input;
  for (int l = 0; l < g.n_layers; ++l) {
    // The graph owns a complete DDR source arena, so this is a coarse
    // per-layer admission point rather than a check on every tensor/alloc.
    // If the background guard sees the host watermark crossed, decline the
    // experimental graph and let the caller use the bounded legacy path.
    if (MemoryGuard::allocation_blocked()) {
      LOG_WARN("[GGML-GLM5NEXT-FULL] graph build stopped at layer %d/%d: "
               "host memory watermark reached (MemAvailable=%ld MiB)\n",
               l, g.n_layers, MemoryGuard::last_available_mb());
      return false;
    }
    const std::string p = "blk." + std::to_string(l) + ".";
    ggml_tensor *post = nullptr;
    ggml_tensor *comb = nullptr;
    ggml_tensor *afn = resident_tensor(g, g.ctx, p + "hc_attn_fn.weight");
    ggml_tensor *abase = resident_tensor(g, g.ctx, p + "hc_attn_base.weight");
    ggml_tensor *ascale = resident_tensor(g, g.ctx, p + "hc_attn_scale.weight");
    if (!afn || !abase || !ascale)
      return false;
    ggml_tensor *residual = res;
    ggml_tensor *cur = hc_pre(g, g.ctx, res, afn, ascale, abase, &post, &comb);
    ggml_tensor *an = resident_tensor(g, g.ctx, p + "attn_norm.weight");
    if (!an)
      return false;
    cur = ggml_mul(g.ctx, ggml_rms_norm(g.ctx, cur, c.eps), an);

    ggml_tensor *attn = nullptr;
    if (glm5next_is_kda_layer(c, l)) {
      ggml_tensor *wq = resident_tensor(g, g.ctx, p + "attn_q.weight");
      ggml_tensor *wk = resident_tensor(g, g.ctx, p + "attn_k.weight");
      ggml_tensor *wv = resident_tensor(g, g.ctx, p + "attn_v.weight");
      ggml_tensor *cq = resident_tensor(g, g.ctx, p + "ssm_conv1d_q.weight");
      ggml_tensor *ck = resident_tensor(g, g.ctx, p + "ssm_conv1d_k.weight");
      ggml_tensor *cv = resident_tensor(g, g.ctx, p + "ssm_conv1d_v.weight");
      ggml_tensor *fa = resident_tensor(g, g.ctx, p + "ssm_f_a.weight");
      ggml_tensor *fb = resident_tensor(g, g.ctx, p + "ssm_f_b.weight");
      ggml_tensor *ga = resident_tensor(g, g.ctx, p + "ssm_g_a.weight");
      ggml_tensor *gb = resident_tensor(g, g.ctx, p + "ssm_g_b.weight");
      ggml_tensor *be = resident_tensor(g, g.ctx, p + "ssm_beta.weight");
      ggml_tensor *aa = resident_tensor(g, g.ctx, p + "ssm_a");
      ggml_tensor *dt = resident_tensor(g, g.ctx, p + "ssm_dt.bias");
      ggml_tensor *nm = resident_tensor(g, g.ctx, p + "ssm_norm.weight");
      ggml_tensor *ow = resident_tensor(g, g.ctx, p + "attn_output.weight");
      if (!wq || !wk || !wv || !cq || !ck || !cv || !fa || !fb || !ga || !gb ||
          !be || !aa || !dt || !nm || !ow)
        return false;
      const int hist = g.d_conv - 1;
      ggml_tensor *hi = host_input(
          g, g.ctx, GGML_TYPE_F32, {hist, 3 * g.d_inner, 1},
          (size_t)hist * 3 * g.d_inner * sizeof(float), "glm5next_kda_history");
      ggml_tensor *si = host_input(
          g, g.ctx, GGML_TYPE_F32, {g.hd, g.hd, g.H, 1},
          (size_t)g.hd * g.hd * g.H * sizeof(float), "glm5next_kda_state");
      if (!hi || !si)
        return false;
      g.kda_hist_inputs.push_back((float *)hi->data);
      g.kda_state_inputs.push_back((float *)si->data);
      ggml_tensor *so = nullptr;
      ggml_tensor *ho = nullptr;
      attn = build_kda(g, g.ctx, cur, l, wq, wk, wv, cq, ck, cv, fa, fb, ga, gb,
                       be, aa, dt, nm, ow, hi, si, &so, &ho);
      if (!attn || !so || !ho)
        return false;
      g.kda_state_out.push_back(so);
      g.kda_hist_out.push_back(ho);
      g.kda_layers.push_back(l);
    } else {
      ggml_tensor *wqa = resident_tensor(g, g.ctx, p + "attn_q_a.weight");
      ggml_tensor *wqan = resident_tensor(g, g.ctx, p + "attn_q_a_norm.weight");
      ggml_tensor *wqb = resident_tensor(g, g.ctx, p + "attn_q_b.weight");
      ggml_tensor *wkva = resident_tensor(g, g.ctx, p + "attn_kv_a_mqa.weight");
      ggml_tensor *wkvan =
          resident_tensor(g, g.ctx, p + "attn_kv_a_norm.weight");
      ggml_tensor *wkb = static_full_weight(g, g.ctx, p + "attn_k_b.weight");
      ggml_tensor *wvb = static_full_weight(g, g.ctx, p + "attn_v_b.weight");
      ggml_tensor *wo = resident_tensor(g, g.ctx, p + "attn_output.weight");
      if (!wqa || !wqan || !wqb || !wkva || !wkvan || !wkb || !wvb || !wo)
        return false;
      ggml_tensor *kv = nullptr;
      attn = build_dsa(g, g.ctx, cur, l, wqa, wqan, wqb, wkva, wkvan, wkb, wvb,
                       wo, &kv);
      if (!attn || !kv)
        return false;
      g.dsa_kv_out.push_back(kv);
      g.dsa_layers.push_back(l);
    }
    res = hc_post(g, g.ctx, attn, residual, post, comb);

    residual = res;
    post = comb = nullptr;
    ggml_tensor *ffn_fn = resident_tensor(g, g.ctx, p + "hc_ffn_fn.weight");
    ggml_tensor *ffn_base = resident_tensor(g, g.ctx, p + "hc_ffn_base.weight");
    ggml_tensor *ffn_scale =
        resident_tensor(g, g.ctx, p + "hc_ffn_scale.weight");
    if (!ffn_fn || !ffn_base || !ffn_scale)
      return false;
    cur = hc_pre(g, g.ctx, res, ffn_fn, ffn_scale, ffn_base, &post, &comb);
    ggml_tensor *fn = resident_tensor(g, g.ctx, p + "ffn_norm.weight");
    if (!fn)
      return false;
    cur = ggml_mul(g.ctx, ggml_rms_norm(g.ctx, cur, c.eps), fn);
    ggml_build_forward_expand(g.graph, residual);
    ggml_build_forward_expand(g.graph, post);
    ggml_build_forward_expand(g.graph, comb);
    ggml_tensor *ffn = build_ffn(g, g.ctx, cur, l);
    if (!ffn) {
      LOG_WARN("[GGML-GLM5NEXT-FULL] build_ffn failed at layer=%d "
               "graph_ram_ready=%d\n",
               l, g.engine->graph_ram_layer_ready(l) ? 1 : 0);
      return false;
    }
    res = hc_post(g, g.ctx, ffn, residual, post, comb);
    if (l >= c.first_dense && (l == c.first_dense || l + 1 == g.n_layers ||
                               ((l - c.first_dense + 1) % 4) == 0)) {
      LOG_INFO("[GGML-GLM5NEXT-FULL] graph source progress layer=%d/%d "
               "expert_source=%.2f GiB MemAvailable=%ld MiB\n",
               l + 1, g.n_layers,
               (double)g.expert_source_used / (1024.0 * 1024.0 * 1024.0),
               MemoryGuard::last_available_mb());
    }
  }

  g.final_streams = res;
  // The surrounding generation loop applies output_norm.weight immediately
  // before the LM-head projection.  Return the same pre-norm activation as
  // the legacy layer path; applying output_norm here would normalize twice.
  g.output = hc_mean(g, g.ctx, res);

  ggml_set_output(g.final_streams);
  ggml_build_forward_expand(g.graph, g.final_streams);
  ggml_set_output(g.output);
  ggml_build_forward_expand(g.graph, g.output);
  for (ggml_tensor *t : g.kda_state_out) {
    ggml_set_output(t);
    ggml_build_forward_expand(g.graph, t);
  }
  for (ggml_tensor *t : g.kda_hist_out) {
    ggml_set_output(t);
    ggml_build_forward_expand(g.graph, t);
  }
  for (ggml_tensor *t : g.dsa_kv_out) {
    ggml_set_output(t);
    ggml_build_forward_expand(g.graph, t);
  }
  if (!graph_alloc_vram_headroom_ok(g, "[GGML-GLM5NEXT-FULL]"))
    return false;
  if (!ggml_backend_sched_alloc_graph(g.sched, g.graph))
    return false;
  g.graph_workspace_bytes = ggml_backend_sched_get_buffer_size(g.sched, g.cuda);

  int cuda_nodes = 0, cpu_nodes = 0, other_nodes = 0, transitions = 0;
  ggml_backend_t previous = nullptr;
  std::unordered_map<std::string, int> non_cuda;
  for (int i = 0; i < ggml_graph_n_nodes(g.graph); ++i) {
    ggml_tensor *node = ggml_graph_node(g.graph, i);
    ggml_backend_t b = ggml_backend_sched_get_tensor_backend(g.sched, node);
    if (b == g.cuda)
      ++cuda_nodes;
    else if (b == g.cpu) {
      ++cpu_nodes;
      ++non_cuda[ggml_op_name(node->op)];
    } else {
      ++other_nodes;
      ++non_cuda[b ? ggml_backend_name(b) : "NULL"];
    }
    if (previous && previous != b)
      ++transitions;
    previous = b;
    if (node->op == GGML_OP_FLASH_ATTN_EXT) {
      const ggml_tensor *q = node->src[0];
      const ggml_tensor *k = node->src[1];
      const ggml_tensor *v = node->src[2];
      const ggml_tensor *m = node->src[3];
      LOG_INFO("[GGML-GLM5NEXT-FULL] FATTN node=%d backend=%s support_cuda=%d "
               "q=(%lld,%lld,%lld,%lld)/%s k=(%lld,%lld,%lld,%lld)/%s "
               "v=(%lld,%lld,%lld,%lld)/%s mask=(%lld,%lld,%lld,%lld)/%s nb="
               "%zu,%zu,%zu,%zu/%zu,%zu,%zu,%zu\n",
               i, b ? ggml_backend_name(b) : "NULL",
               ggml_backend_supports_op(g.cuda, node) ? 1 : 0,
               (long long)q->ne[0], (long long)q->ne[1], (long long)q->ne[2],
               (long long)q->ne[3], ggml_type_name(q->type),
               (long long)k->ne[0], (long long)k->ne[1], (long long)k->ne[2],
               (long long)k->ne[3], ggml_type_name(k->type),
               (long long)v->ne[0], (long long)v->ne[1], (long long)v->ne[2],
               (long long)v->ne[3], ggml_type_name(v->type),
               (long long)m->ne[0], (long long)m->ne[1], (long long)m->ne[2],
               (long long)m->ne[3], ggml_type_name(m->type), q->nb[0], q->nb[1],
               q->nb[2], q->nb[3], k->nb[0], k->nb[1], k->nb[2], k->nb[3]);
    }
  }
  LOG_INFO("[GGML-GLM5NEXT-FULL] scheduler nodes cuda=%d cpu=%d other=%d "
           "transitions=%d\n",
           cuda_nodes, cpu_nodes, other_nodes, transitions);
  if (!non_cuda.empty()) {
    std::string detail;
    for (const auto &[name, count] : non_cuda) {
      if (!detail.empty())
        detail += ", ";
      detail += name + "=" + std::to_string(count);
    }
    LOG_INFO("[GGML-GLM5NEXT-FULL] non-CUDA nodes: %s\n", detail.c_str());
  }
  LOG_INFO("[GGML-GLM5NEXT-FULL] graph built layers=%d nodes=%d splits=%d "
           "workspace=%.2f GiB "
           "KDA=%zu DSA=%zu\n",
           g.n_layers, ggml_graph_n_nodes(g.graph),
           ggml_backend_sched_get_n_splits(g.sched),
           (double)g.graph_workspace_bytes / (1024.0 * 1024.0 * 1024.0),
           g.kda_layers.size(), g.dsa_layers.size());
  LOG_INFO(
      "[GGML-GLM5NEXT-FULL] dense arena=%.2f/%.2f GiB tensors=%d; "
      "expert arena used=%.2f/%.2f GiB; host source=%.2f GiB pinned=%.2f GiB "
      "pin_failures=%zu\n",
      (double)g.dense_used / (1024.0 * 1024.0 * 1024.0),
      (double)g.dense_capacity / (1024.0 * 1024.0 * 1024.0), g.dense_tensors,
      (double)g.expert_used / (1024.0 * 1024.0 * 1024.0),
      (double)g.expert_capacity / (1024.0 * 1024.0 * 1024.0),
      (double)g.host_weight_bytes / (1024.0 * 1024.0 * 1024.0),
      (double)g.host_pinned_bytes / (1024.0 * 1024.0 * 1024.0),
      g.host_pin_failures);
  LOG_INFO("[GGML-GLM5NEXT-FULL-MEM] host ledger=%.2f/%.2f GB; graph source="
           "%.2f GiB + context=%.2f MiB + owned-extra=%.2f MiB; device "
           "workspace=%.2f GiB (GPU, outside host_ram_gb)\n",
           (double)MemoryGuard::reserved_host_bytes() / 1.0e9,
           (double)MemoryGuard::host_budget_bytes() / 1.0e9,
           (double)g.expert_source_capacity / (1024.0 * 1024.0 * 1024.0),
           (double)g.context_reservation_bytes / (1024.0 * 1024.0),
           (double)(g.host_weight_bytes > g.expert_source_used
                        ? g.host_weight_bytes - g.expert_source_used
                        : 0) /
               (1024.0 * 1024.0),
           (double)g.graph_workspace_bytes / (1024.0 * 1024.0 * 1024.0));
  g.ready = true;
  return true;
}

} // namespace

bool glm5next_full_graph_enabled() { return env_enabled(); }

bool glm5next_full_graph_bootstrap(StreamingEngine &engine) {
  if (!env_enabled() || engine.family_ops.family != ModelFamily::GLM5NEXT)
    return false;

  FullGraph candidate;
  candidate.engine = &engine;
  candidate.device = engine.gpu_device_id >= 0 ? engine.gpu_device_id : 0;
  candidate.first_dense = engine.model_file.architecture_config.first_dense;
  candidate.device_expert_layers =
      std::min(glm5next_full_graph_device_expert_layers(),
               std::max(0, engine.model_file.architecture_config.n_layers -
                               candidate.first_dense));
  if (configure_partial_experts(candidate))
    candidate.device_expert_layers = 0;

  if (g_glm5next_source &&
      source_configuration_matches(*g_glm5next_source, candidate)) {
    LOG_INFO("[GGML-GLM5NEXT-FULL] boot expert source already resident; "
             "reusing it\n");
    return true;
  }
  g_glm5next_source.reset();
  auto source = std::make_unique<FullGraph>();
  source->engine = candidate.engine;
  source->device = candidate.device;
  source->first_dense = candidate.first_dense;
  source->device_expert_layers = candidate.device_expert_layers;
  source->partial_expert_residency = candidate.partial_expert_residency;
  source->partial_sources_compacted = candidate.partial_sources_compacted;
  source->partial_hot_bytes = candidate.partial_hot_bytes;
  source->partial_cold_bytes = candidate.partial_cold_bytes;
  source->partial_hot_records = candidate.partial_hot_records;
  source->partial_usage_file = std::move(candidate.partial_usage_file);
  source->partial_experts = std::move(candidate.partial_experts);
  if (!prepare_expert_source_only(*source, "GGML-GLM5NEXT-FULL"))
    return false;
  g_glm5next_source = std::move(source);
  return true;
}

int glm5next_full_graph_device_expert_layers() {
  return env_enabled() ? env_device_expert_layers() : 0;
}

void glm5next_full_graph_shutdown() {
  g_graph.reset();
  g_glm5next_source.reset();
}

bool glm5next_full_graph_prepare(StreamingEngine &engine, int S, int pos_base,
                                 const int *) {
  if (!env_enabled() || S <= 1 || pos_base != 0 ||
      engine.family_ops.family != ModelFamily::GLM5NEXT)
    return false;
  const auto &c = engine.model_file.architecture_config;
  if (g_graph && g_graph->ready && g_graph->engine == &engine &&
      g_graph->S == S)
    return true;

  // Keep the large source arena alive while releasing only the old
  // prompt-shaped graph.  This is what makes boot preload useful for a later
  // prompt with a different token count.
  if (g_graph && g_graph->has_expert_source()) {
    if (!g_glm5next_source)
      g_glm5next_source = std::make_unique<FullGraph>();
    g_graph->move_expert_source_to(*g_glm5next_source);
  }
  g_graph.reset();
  auto g = std::make_unique<FullGraph>();
  g->engine = &engine;
  g->device = engine.gpu_device_id >= 0 ? engine.gpu_device_id : 0;
  g->S = S;
  g->D = c.hidden;
  g->HC = std::max(1, c.hc_mult);
  g->n_layers = c.n_layers;
  g->H = c.n_heads;
  g->hd = c.kda_head_dim;
  g->d_inner = g->H * g->hd;
  g->d_conv = c.ssm_conv_kernel;
  g->qk_nope = c.qk_nope;
  g->kv_lora = c.kv_lora;
  g->v_head = c.v_head;
  g->first_dense = c.first_dense;
  g->device_expert_layers = std::min(glm5next_full_graph_device_expert_layers(),
                                     std::max(0, c.n_layers - c.first_dense));
  if (configure_partial_experts(*g))
    g->device_expert_layers = 0;
  g->cuda = static_cast<ggml_backend_t>(ggml_shared_backend(g->device));
  g->cpu = ggml_backend_cpu_init();
  if (!g->cuda || !g->cpu)
    return false;

  if (g_glm5next_source &&
      source_configuration_matches(*g_glm5next_source, *g)) {
    g_glm5next_source->move_expert_source_to(*g);
    LOG_INFO("[GGML-GLM5NEXT-FULL] reusing boot expert source: %.2f GiB DDR; "
             "no SSD reload for prompt S=%d\n",
             (double)g->expert_source_used / (1024.0 * 1024.0 * 1024.0), S);
  } else if (!prepare_expert_source_only(*g, "GGML-GLM5NEXT-FULL")) {
    return false;
  }
  if (!init_expert_arena(*g) || !init_dense_arena(*g))
    return false;
  ggml_backend_t backends[] = {g->cuda, g->cpu};
  g->sched =
      ggml_backend_sched_new(backends, nullptr, 2, kGraphSize, false, true);
  if (!g->sched)
    return false;
  if (!build_graph(*g)) {
    LOG_WARN("[GGML-GLM5NEXT-FULL] graph construction declined; using legacy "
             "path\n");
    return false;
  }
  g_graph = std::move(g);
  LOG_INFO("[GGML-GLM5NEXT-FULL] graph prepared for S=%d\n", S);
  return true;
}

bool glm5next_full_graph_run(StreamingEngine &engine, float *hidden_states,
                             int S, int pos_base, const int *) {
  if (!env_enabled() || !hidden_states || S <= 1 || pos_base != 0 || !g_graph ||
      !g_graph->ready || g_graph->engine != &engine || g_graph->S != S)
    return false;
  FullGraph &g = *g_graph;
  const auto &c = engine.model_file.architecture_config;
  const size_t state_n = (size_t)g.hd * g.hd * g.H;
  const size_t hist_n = (size_t)(g.d_conv - 1) * 3 * g.d_inner;

  for (size_t i = 0; i < g.kda_layers.size(); ++i) {
    const int l = g.kda_layers[i];
    if (l < 0 || l >= (int)engine.glm5next_recur_state.size() ||
        l >= (int)engine.glm5next_conv_history.size() ||
        engine.glm5next_recur_state[l].size() != state_n ||
        engine.glm5next_conv_history[l].size() != hist_n)
      return false;
    std::fill(g.kda_state_inputs[i], g.kda_state_inputs[i] + state_n, 0.0f);
    std::fill(g.kda_hist_inputs[i], g.kda_hist_inputs[i] + hist_n, 0.0f);
  }
  if (engine.hc_stream_buf.size() < (size_t)g.D * g.HC * S ||
      g.host_inputs.empty())
    return false;
  std::memcpy(g.host_inputs[0], engine.hc_stream_buf.data(),
              (size_t)g.D * g.HC * S * sizeof(float));

  // Real-hardware incident (2026-09-09): a 480-token prompt crashed the
  // whole machine (hard reboot required, no surviving kernel/OOM log --
  // this box has no persistent journald storage) with the last surviving
  // application-log line being "graph prepared for S=...", i.e. somewhere
  // in the gap between prepare() returning and this compute call. Every
  // OTHER host allocation on this path (arena prep, expert source, graph
  // context) goes through MemoryGuard::try_reserve_host_bytes first; this
  // compute call did not, and it is the last real memory-relevant, cheaply
  // avoidable step before the model correctness this session verified
  // could otherwise be undone by a repeat crash. Re-check REAL, fresh
  // MemAvailable (not the background watchdog's own possibly-500ms-stale
  // allocation_blocked() flag) immediately before committing to it -- a
  // large gap can open between the admission checks earlier in prepare()
  // (preload alone took minutes) and this moment. Not a fix for whatever
  // the real mechanism was (unconfirmed -- no kernel evidence survived),
  // just a cheap, additional, defense-in-depth backstop.
  {
    const long avail_mb = MemoryGuard::read_available_mb();
    constexpr long kMinComputeHeadroomMb =
        10240; // 10 GiB, per explicit user instruction after the 2026-09-09
               // crash
    if (avail_mb >= 0 && avail_mb < kMinComputeHeadroomMb) {
      LOG_ERROR(
          "[GGML-GLM5NEXT-FULL] refusing graph compute: MemAvailable=%ld MiB "
          "< %ld MiB safety floor -- falling back to the legacy path rather "
          "than risk the host-memory incident this check was added for "
          "(2026-09-09, see this call site's own comment)\n",
          avail_mb, kMinComputeHeadroomMb);
      return false;
    }
  }

  const auto start = std::chrono::steady_clock::now();
  if (ggml_backend_sched_graph_compute_async(g.sched, g.graph) !=
      GGML_STATUS_SUCCESS)
    return false;
  ggml_backend_sched_synchronize(g.sched);
  const auto synced = std::chrono::steady_clock::now();
  ggml_backend_tensor_get(g.output, hidden_states, 0,
                          (size_t)g.D * S * sizeof(float));

  for (size_t i = 0; i < g.kda_layers.size(); ++i) {
    const int l = g.kda_layers[i];
    ggml_backend_tensor_get(g.kda_state_out[i],
                            engine.glm5next_recur_state[l].data(), 0,
                            state_n * sizeof(float));
    ggml_backend_tensor_get(g.kda_hist_out[i],
                            engine.glm5next_conv_history[l].data(), 0,
                            hist_n * sizeof(float));
  }
  const size_t kv_n = (size_t)g.S * g.kv_lora;
  std::vector<float> kv(kv_n);
  for (size_t i = 0; i < g.dsa_layers.size(); ++i) {
    const int l = g.dsa_layers[i];
    if (l < 0 || l >= (int)engine.Lc.size() || !engine.Lc[l])
      return false;
    ggml_backend_tensor_get(g.dsa_kv_out[i], kv.data(), 0,
                            kv_n * sizeof(float));
    std::memcpy(engine.Lc[l], kv.data(), kv_n * sizeof(float));
  }
  const auto done = std::chrono::steady_clock::now();
  const auto ms = [](auto a, auto b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
  };
  LOG_INFO("[GGML-GLM5NEXT-FULL-TIME] sync=%.3f ms state_get=%.3f ms "
           "total=%.3f ms\n",
           ms(start, synced), ms(synced, done), ms(start, done));
  engine.glm5next_full_graph_output_ready = true;
  (void)c;
  return true;
}

bool deepseek_v4_full_graph_enabled() { return dsv4_env_enabled(); }

int deepseek_v4_full_graph_device_expert_layers() {
  return dsv4_env_enabled() ? dsv4_env_device_expert_layers() : 0;
}

void deepseek_v4_full_graph_shutdown() { g_dsv4_graph.reset(); }

bool deepseek_v4_full_graph_prepare(StreamingEngine &engine, int S,
                                    int pos_base, const int *token_ids) {
  if (!dsv4_env_enabled() || S <= 1 || pos_base != 0 ||
      engine.family_ops.family != ModelFamily::DeepSeek_V4)
    return false;
  const auto &c = engine.model_file.architecture_config;
  // The graph deliberately attends every completed CSA block.  That is an
  // exact match while the visible block count stays below the model's
  // Lightning Indexer top-k; beyond that point the legacy path remains the
  // correctness fallback until dynamic in-graph top-k selection is added.
  if (c.dsv4_indexer_top_k > 0 && S > c.dsv4_indexer_top_k * 4) {
    LOG_WARN(
        "[GGML-DSV4-FULL] prompt S=%d exceeds CSA dense-selection limit %d; "
        "using legacy indexed attention\n",
        S, c.dsv4_indexer_top_k * 4);
    return false;
  }
  if (g_dsv4_graph && g_dsv4_graph->ready && g_dsv4_graph->engine == &engine &&
      g_dsv4_graph->S == S)
    return true;

  g_dsv4_graph.reset();
  auto g = std::make_unique<FullGraph>();
  g->engine = &engine;
  g->device = engine.gpu_device_id >= 0 ? engine.gpu_device_id : 0;
  g->S = S;
  g->D = c.hidden;
  g->HC = std::max(1, c.hc_mult);
  g->n_layers = c.n_layers;
  g->H = c.n_heads;
  g->qk_nope = c.qk_nope;
  g->qk_rope = c.qk_rope;
  g->o_group_count = c.o_group_count;
  g->o_lora_rank = c.o_lora_rank;
  g->sliding_window = c.sliding_window;
  g->first_dense = c.first_dense;
  g->device_expert_layers =
      std::min(deepseek_v4_full_graph_device_expert_layers(),
               std::max(0, c.n_layers - c.first_dense));
  if (configure_partial_experts(*g))
    g->device_expert_layers = 0;
  g->cuda = static_cast<ggml_backend_t>(ggml_shared_backend(g->device));
  g->cpu = ggml_backend_cpu_init();
  if (!g->cuda || !g->cpu)
    return false;

  const size_t source = expert_source_bytes(engine, g->device_expert_layers);
  const size_t host_live = graph_host_live_bytes(engine);
  constexpr size_t kSourceSlack = 1ull << 20;
  // 10 GiB, per explicit user instruction after the 2026-09-09 crash.
  constexpr size_t kWatermark = 10ull * 1024ull * 1024ull * 1024ull;
  if (source == 0 || source > SIZE_MAX - kSourceSlack ||
      !MemoryGuard::try_reserve_host_bytes(source + kSourceSlack, kWatermark)) {
    LOG_WARN(
        "[GGML-DSV4-FULL] host watermark refused source=%.2f GiB + existing "
        "host=%.2f GiB + reserve=%.2f GiB; budget=%.2f GB MemAvailable=%ld "
        "MiB\n",
        (double)source / (1024.0 * 1024.0 * 1024.0),
        (double)host_live / (1024.0 * 1024.0 * 1024.0),
        (double)kWatermark / (1024.0 * 1024.0 * 1024.0),
        (double)MemoryGuard::host_budget_bytes() / 1.0e9,
        MemoryGuard::last_available_mb());
    return false;
  }
  g->host_reservation_bytes = source + kSourceSlack;
  LOG_INFO("[GGML-DSV4-FULL] host admission: source=%.2f GiB existing=%.2f GiB "
           "reserve=%.2f GiB budget=%.2f GB\n",
           (double)source / (1024.0 * 1024.0 * 1024.0),
           (double)host_live / (1024.0 * 1024.0 * 1024.0),
           (double)kWatermark / (1024.0 * 1024.0 * 1024.0),
           (double)MemoryGuard::host_budget_bytes() / 1.0e9);
  if (!init_expert_source(*g, source) || !preload_expert_source(*g) ||
      !init_expert_arena(*g) || !init_dense_arena(*g))
    return false;
  ggml_backend_t backends[] = {g->cuda, g->cpu};
  g->sched =
      ggml_backend_sched_new(backends, nullptr, 2, kGraphSize, false, true);
  if (!g->sched || !build_dsv4_graph(*g, token_ids)) {
    LOG_WARN(
        "[GGML-DSV4-FULL] graph construction declined; using legacy path\n");
    return false;
  }
  g_dsv4_graph = std::move(g);
  LOG_INFO("[GGML-DSV4-FULL] graph prepared for S=%d\n", S);
  return true;
}

bool deepseek_v4_full_graph_run(StreamingEngine &engine, float *hidden_states,
                                int S, int pos_base, const int *) {
  if (!dsv4_env_enabled() || !hidden_states || S <= 1 || pos_base != 0 ||
      !g_dsv4_graph || !g_dsv4_graph->ready ||
      g_dsv4_graph->engine != &engine || g_dsv4_graph->S != S)
    return false;
  FullGraph &g = *g_dsv4_graph;
  const auto &c = engine.model_file.architecture_config;
  const int window = g.sliding_window > 0 ? g.sliding_window : engine.max_t;

  // The graph starts a fresh MoEStream sequence.  Clear the state first so a
  // failed/short graph run cannot leave the following legacy decode mixed
  // with stale compressed rows from a prior request.
  for (const auto &lt : g.dsv4_layers) {
    if (lt.layer < 0 || lt.layer >= (int)engine.KVc.size() ||
        !engine.KVc[lt.layer] || lt.layer >= (int)engine.dsv4_compressed.size())
      return false;
    const int ratio = lt.ratio;
    if (ratio != 0 && engine.dsv4_compressed[lt.layer].ratio != ratio)
      return false;
    if (ratio != 0)
      engine.dsv4_compressed[lt.layer].reset();
  }
  if (g.host_inputs.empty() ||
      engine.hc_stream_buf.size() < (size_t)g.D * g.HC * S)
    return false;
  std::memcpy(g.host_inputs[0], engine.hc_stream_buf.data(),
              (size_t)g.D * g.HC * S * sizeof(float));

  // Same defense-in-depth backstop as glm5next_full_graph_run's own compute
  // call (see that call site's own comment for the real-hardware incident
  // that motivated it) -- DSV4's full graph hasn't been run on real hardware
  // yet at all, so there is no evidence either way here, but the gap this
  // closes (a fresh, synchronous MemAvailable check right before the one
  // uncounted compute step in this whole admission chain) applies equally.
  {
    const long avail_mb = MemoryGuard::read_available_mb();
    constexpr long kMinComputeHeadroomMb =
        10240; // 10 GiB, per explicit user instruction after the 2026-09-09
               // crash
    if (avail_mb >= 0 && avail_mb < kMinComputeHeadroomMb) {
      LOG_ERROR("[GGML-DSV4-FULL] refusing graph compute: MemAvailable=%ld MiB "
                "< %ld MiB safety floor -- falling back to the legacy path\n",
                avail_mb, kMinComputeHeadroomMb);
      return false;
    }
  }

  const auto start = std::chrono::steady_clock::now();
  if (ggml_backend_sched_graph_compute_async(g.sched, g.graph) !=
      GGML_STATUS_SUCCESS)
    return false;
  ggml_backend_sched_synchronize(g.sched);
  const auto synced = std::chrono::steady_clock::now();
  ggml_backend_tensor_get(g.output, hidden_states, 0,
                          (size_t)g.D * S * sizeof(float));

  if (dsv4_graph_diagnostic_enabled()) {
    for (const auto &stage : g.dsv4_stage_outputs) {
      // Attention/FFN stages are [D,S] tensors, while residual stages are
      // [D,HC,S].  Reading every stage into the residual-sized buffer used
      // to over-read the former and triggered GGML's bounds assertion.
      const size_t bytes = ggml_nbytes(stage.second);
      if (bytes == 0 || bytes % sizeof(float) != 0) {
        LOG_ERROR("[GGML-DSV4-DIAG] %s has unsupported byte size=%zu\n",
                  stage.first.c_str(), bytes);
        return false;
      }
      std::vector<float> layer_values(bytes / sizeof(float));
      ggml_backend_tensor_get(stage.second, layer_values.data(), 0, bytes);
      size_t finite = 0;
      float min_value = INFINITY;
      float max_value = -INFINITY;
      for (float value : layer_values) {
        if (std::isfinite(value)) {
          ++finite;
          min_value = std::min(min_value, value);
          max_value = std::max(max_value, value);
        }
      }
      LOG_INFO("[GGML-DSV4-DIAG] %s shape=(%lld,%lld,%lld,%lld) finite=%zu/%zu "
               "min=%g max=%g\n",
               stage.first.c_str(), (long long)stage.second->ne[0],
               (long long)stage.second->ne[1], (long long)stage.second->ne[2],
               (long long)stage.second->ne[3], finite, layer_values.size(),
               min_value, max_value);
      if (finite != layer_values.size())
        break;
    }
  }

  for (const auto &lt : g.dsv4_layers) {
    const int l = lt.layer;
    const size_t raw_n = (size_t)(g.qk_nope + g.qk_rope) * S;
    std::vector<float> raw(raw_n);
    ggml_backend_tensor_get(lt.raw_kv, raw.data(), 0, raw_n * sizeof(float));
    for (int s = 0; s < S; ++s) {
      const int slot = s % std::max(1, window);
      std::memcpy(engine.KVc[l] + (size_t)slot * (g.qk_nope + g.qk_rope),
                  raw.data() + (size_t)s * (g.qk_nope + g.qk_rope),
                  (size_t)(g.qk_nope + g.qk_rope) * sizeof(float));
    }
    if (lt.ratio == 0)
      continue;
    auto &state = engine.dsv4_compressed[l];
    const int width =
        lt.ratio == 4 ? 2 * (g.qk_nope + g.qk_rope) : (g.qk_nope + g.qk_rope);
    std::vector<float> comp((size_t)width * S);
    std::vector<float> score((size_t)width * S);
    ggml_backend_tensor_get(lt.comp_kv, comp.data(), 0,
                            comp.size() * sizeof(float));
    ggml_backend_tensor_get(lt.comp_score, score.data(), 0,
                            score.size() * sizeof(float));
    for (int s = 0; s < S; ++s) {
      const int slot = s % state.state_size;
      std::memcpy(state.state_kv.data() + (size_t)slot * width,
                  comp.data() + (size_t)s * width,
                  (size_t)width * sizeof(float));
      std::memcpy(state.state_score.data() + (size_t)slot * width,
                  score.data() + (size_t)s * width,
                  (size_t)width * sizeof(float));
    }
    const int blocks = S / lt.ratio;
    if (blocks > 0 && lt.keys) {
      std::vector<float> keys((size_t)(g.qk_nope + g.qk_rope) * blocks);
      ggml_backend_tensor_get(lt.keys, keys.data(), 0,
                              keys.size() * sizeof(float));
      std::memcpy(state.keys.data(), keys.data(),
                  std::min(state.keys.size(), keys.size()) * sizeof(float));
    }
    if (lt.indexer_comp_kv) {
      const int idim = c.dsv4_indexer_key_length;
      const int iwidth = 2 * idim;
      std::vector<float> ikv((size_t)iwidth * S);
      std::vector<float> iscore((size_t)iwidth * S);
      ggml_backend_tensor_get(lt.indexer_comp_kv, ikv.data(), 0,
                              ikv.size() * sizeof(float));
      ggml_backend_tensor_get(lt.indexer_comp_score, iscore.data(), 0,
                              iscore.size() * sizeof(float));
      for (int s = 0; s < S; ++s) {
        const int slot = s % state.state_size;
        std::memcpy(state.indexer_state_kv.data() + (size_t)slot * iwidth,
                    ikv.data() + (size_t)s * iwidth,
                    (size_t)iwidth * sizeof(float));
        std::memcpy(state.indexer_state_score.data() + (size_t)slot * iwidth,
                    iscore.data() + (size_t)s * iwidth,
                    (size_t)iwidth * sizeof(float));
      }
      if (blocks > 0 && lt.indexer_keys) {
        std::vector<float> keys((size_t)idim * blocks);
        ggml_backend_tensor_get(lt.indexer_keys, keys.data(), 0,
                                keys.size() * sizeof(float));
        std::memcpy(state.indexer_keys.data(), keys.data(),
                    std::min(state.indexer_keys.size(), keys.size()) *
                        sizeof(float));
      }
    }
  }
  const auto done = std::chrono::steady_clock::now();
  const auto ms = [](auto a, auto b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
  };
  LOG_INFO(
      "[GGML-DSV4-FULL-TIME] sync=%.3f ms state_get=%.3f ms total=%.3f ms\n",
      ms(start, synced), ms(synced, done), ms(start, done));
  engine.deepseek_v4_full_graph_output_ready = true;
  return true;
}

#else

bool glm5next_full_graph_enabled() { return false; }
bool glm5next_full_graph_bootstrap(StreamingEngine &) { return false; }
int glm5next_full_graph_device_expert_layers() { return 0; }
bool glm5next_full_graph_prepare(StreamingEngine &, int, int, const int *) {
  return false;
}
bool glm5next_full_graph_run(StreamingEngine &, float *, int, int,
                             const int *) {
  return false;
}
void glm5next_full_graph_shutdown() {}
bool deepseek_v4_full_graph_enabled() { return false; }
int deepseek_v4_full_graph_device_expert_layers() { return 0; }
bool deepseek_v4_full_graph_prepare(StreamingEngine &, int, int, const int *) {
  return false;
}
bool deepseek_v4_full_graph_run(StreamingEngine &, float *, int, int,
                                const int *) {
  return false;
}
void deepseek_v4_full_graph_shutdown() {}

#endif
