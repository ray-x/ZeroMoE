// Qwen4Exp full graph: Memory placement: host allocations, dense/state/expert arenas, resident
// tensor lookup, expert source preparation.
//
// Split out of the former single-file ggml_qwen4exp_graph.cpp; shared
// declarations live in ggml_glm5next_graph_impl.hpp.

#include "ggml_glm5next_graph_impl.hpp"

#if defined(MOESTREAM_USE_GGML_CUDA)

using namespace glm5next_graph;

namespace glm5next_graph {

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
      name.find("nextn") != std::string::npos)
    return false;
  if (name.find("indexer.") != std::string::npos ||
      name.find("indexer_compressor_") != std::string::npos)
    return true;
  return name == "output.weight" || name.find("attn") != std::string::npos ||
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

size_t device_expert_bytes(const FullGraph &g) {
  if (g.vram_reference_experts)
    return 0;
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
    if (name == "output.weight" && !g.decode_mode)
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

bool init_decode_state_arena(FullGraph &g) {
  if (!g.decode_mode || !g.cuda || g.kv_capacity <= 0 || g.kv_lora <= 0)
    return false;
  const auto &c = g.engine->model_file.architecture_config;
  size_t bytes = 1ull << 20; // alignment/slack for backend tensor allocations
  auto add_tensor = [&bytes](size_t tensor_bytes) {
    constexpr size_t kAlignment = 4096;
    if (tensor_bytes > SIZE_MAX - bytes - kAlignment)
      return false;
    bytes += tensor_bytes + kAlignment;
    return true;
  };
  for (int l = 0; l < g.n_layers; ++l) {
    if (glm5next_is_kda_layer(c, l)) {
      const size_t state_n = (size_t)g.hd * g.hd * g.H;
      const size_t hist_n = (size_t)(g.d_conv - 1) * 3 * g.d_inner;
      if (state_n > SIZE_MAX / sizeof(float) ||
          hist_n > SIZE_MAX / sizeof(float) ||
          !add_tensor(state_n * sizeof(float)) ||
          !add_tensor(hist_n * sizeof(float)))
        return false;
    } else {
      if ((size_t)g.kv_capacity >
          SIZE_MAX / (size_t)g.kv_lora / sizeof(ggml_fp16_t) ||
          !add_tensor((size_t)g.kv_capacity * g.kv_lora *
                      sizeof(ggml_fp16_t)))
        return false;
    }
  }
  if (g.sparse_decode) {
    if (g.indexer_key_length <= 0 || g.indexer_kpool <= 1 ||
        g.kv_capacity % g.indexer_kpool != 0)
      return false;
    const size_t capacity = (size_t)g.kv_capacity;
    const size_t key_length = (size_t)g.indexer_key_length;
    const size_t pool_count = capacity / (size_t)g.indexer_kpool;
    if (key_length > SIZE_MAX / 2 / sizeof(float) / capacity ||
        key_length > SIZE_MAX / sizeof(float) / pool_count)
      return false;
    for (int l = 0; l < g.n_layers; ++l) {
      if (glm5next_is_kda_layer(c, l))
        continue;
      const std::string prefix = "blk." + std::to_string(l) + ".";
      if (!model_tensor(*g.engine, prefix + "indexer.proj.weight"))
        continue;
      if (!add_tensor(2 * key_length * capacity * sizeof(float)) ||
          !add_tensor(key_length * pool_count * sizeof(float)))
        return false;
    }
  }
  const auto buft = ggml_backend_get_default_buffer_type(g.cuda);
  g.device_state_buffer = ggml_backend_buft_alloc_buffer(buft, bytes);
  if (!g.device_state_buffer) {
    LOG_WARN("[GGML-GLM5NEXT-FULL-DECODE] device state arena declined: "
             "%.2f GiB (KV capacity=%d)\n",
             (double)bytes / (1024.0 * 1024.0 * 1024.0), g.kv_capacity);
    return false;
  }
  ggml_backend_buffer_set_usage(g.device_state_buffer,
                                GGML_BACKEND_BUFFER_USAGE_COMPUTE);
  g.device_state_base = ggml_backend_buffer_get_base(g.device_state_buffer);
  g.device_state_capacity = ggml_backend_buffer_get_size(g.device_state_buffer);
  g.device_state_used = 0;
  LOG_INFO("[GGML-GLM5NEXT-FULL-DECODE] device state arena=%.2f GiB, "
           "KV capacity=%d\n",
           (double)g.device_state_capacity / (1024.0 * 1024.0 * 1024.0),
           g.kv_capacity);
  return true;
}

ggml_tensor *new_decode_state_tensor(FullGraph &g, ggml_context *ctx,
                                     ggml_type type,
                                     const std::vector<int64_t> &shape,
                                     const char *name) {
  ggml_tensor *t = new_tensor(ctx, type, shape);
  if (!t || !g.device_state_buffer)
    return nullptr;
  const ggml_backend_buffer_type_t buft =
      ggml_backend_buffer_get_type(g.device_state_buffer);
  const size_t alloc_size = ggml_backend_buft_get_alloc_size(buft, t);
  const size_t alignment =
      std::max<size_t>(256, ggml_backend_buft_get_alignment(buft));
  const size_t aligned =
      (g.device_state_used + alignment - 1) & ~(alignment - 1);
  if (aligned > g.device_state_capacity ||
      alloc_size > g.device_state_capacity - aligned)
    return nullptr;
  void *addr = static_cast<uint8_t *>(g.device_state_base) + aligned;
  if (ggml_backend_tensor_alloc(g.device_state_buffer, t, addr) !=
      GGML_STATUS_SUCCESS)
    return nullptr;
  g.device_state_used = aligned + alloc_size;
  ggml_set_name(t, name);
  return t;
}

bool init_expert_arena(FullGraph &g) {
  if (!g.partial_expert_residency && g.device_expert_layers <= 0)
    return true;
  if (g.vram_reference_experts) {
    const bool has_hot_refs = std::any_of(
        g.partial_experts.begin(), g.partial_experts.end(),
        [](const PartialExpertLayer &p) { return !p.hot.empty(); });
    if (!has_hot_refs)
      return true;
    // GGML needs a CUDA buffer handle to classify these immutable source
    // tensors, but their data pointers belong to GpuTopology's cache pool.
    // This tiny owned allocation is metadata/handle backing only; no expert
    // weights are copied into it.
    constexpr size_t kReferenceBufferBytes = 1ull << 20;
    const auto buft = ggml_backend_get_default_buffer_type(g.cuda);
    g.expert_buffer = ggml_backend_buft_alloc_buffer(
        buft, kReferenceBufferBytes);
    if (!g.expert_buffer) {
      LOG_WARN("[GGML-GLM5NEXT-FULL] could not create CUDA buffer handle for "
               "VRAM expert references\n");
      return false;
    }
    ggml_backend_buffer_set_usage(g.expert_buffer,
                                  GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    g.expert_base = ggml_backend_buffer_get_base(g.expert_buffer);
    g.expert_capacity = ggml_backend_buffer_get_size(g.expert_buffer);
      LOG_INFO("[GGML-GLM5NEXT-FULL] VRAM expert weights are aliased from "
               "resident cache slots; graph-owned weight bytes=0, descriptor "
             "buffer=%.2f MiB\n",
             (double)g.expert_capacity / (1024.0 * 1024.0));
    return true;
  }
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
  if (!source.has_expert_source() || source.engine != candidate.engine ||
      source.topology != candidate.topology ||
      source.vram_reference_experts != candidate.vram_reference_experts ||
      source.device_expert_layers != candidate.device_expert_layers ||
      source.partial_expert_residency != candidate.partial_expert_residency ||
      source.dynamic_expert_cache != candidate.dynamic_expert_cache ||
      source.partial_usage_file != candidate.partial_usage_file ||
      source.partial_experts.size() != candidate.partial_experts.size())
    return false;
  for (size_t i = 0; i < source.partial_experts.size(); ++i) {
    const PartialExpertLayer &a = source.partial_experts[i];
    const PartialExpertLayer &b = candidate.partial_experts[i];
    if (a.hot != b.hot || a.cold != b.cold ||
        a.vram_slot_by_expert != b.vram_slot_by_expert ||
        a.vram_layer_base != b.vram_layer_base ||
        a.vram_layer_slots != b.vram_layer_slots ||
        a.vram_slot_stride != b.vram_slot_stride)
      return false;
  }
  if (source.vram_reference_bytes != candidate.vram_reference_bytes)
    return false;
  return true;
}

bool prepare_expert_source_only(FullGraph &g, const char *tag) {
  const size_t source = expert_source_bytes(g);
  constexpr size_t kSourceSlack = 1ull << 20;
  // 10 GiB, per explicit user instruction after the 2026-09-09 crash (was
  // 8 GiB) -- align every host-memory watermark in this file to the same
  // floor rather than several different ad hoc numbers.
  constexpr size_t kWatermark = 10ull * 1024ull * 1024ull * 1024ull;
  const size_t host_live = graph_host_live_bytes(*g.engine);
  const long pre_call_available_mb = MemoryGuard::read_available_mb();
  const size_t pre_call_reserved = MemoryGuard::reserved_host_bytes();
  LOG_INFO("[%s] admission attempt: source=%.2f GiB existing(host_live)=%.2f "
           "GiB ledger_reserved=%.2f GiB MemAvailable=%ld MiB (sampled "
           "immediately before the try_reserve_host_bytes call)\n",
           tag, (double)source / (1024.0 * 1024.0 * 1024.0),
           (double)host_live / (1024.0 * 1024.0 * 1024.0),
           (double)pre_call_reserved / (1024.0 * 1024.0 * 1024.0),
           pre_call_available_mb);
  if (g.dynamic_expert_cache) {
    // The per-token GLM cache needs an immutable source for every expert, but
    // that source need not be another eagerly populated 78 GiB anonymous
    // arena. Keep GGUF shard pages file-backed and reclaimable; the cold CPU
    // branch and the pinned H2D staging worker fault only the ranges they use.
    if (source == 0 || source > SIZE_MAX - kSourceSlack ||
        !MemoryGuard::try_reserve_host_bytes(1, kWatermark)) {
      LOG_WARN(
          "[%s] mapped-source admission refused source=%.2f GiB existing="
          "%.2f GiB reserve=%.2f GiB budget=%.2f GB MemAvailable=%ld MiB "
          "(live, not watchdog-stale) stop_mb=%ld\n",
          tag, (double)source / (1024.0 * 1024.0 * 1024.0),
          (double)host_live / (1024.0 * 1024.0 * 1024.0),
          (double)kWatermark / (1024.0 * 1024.0 * 1024.0),
          (double)MemoryGuard::host_budget_bytes() / 1.0e9,
          MemoryGuard::read_available_mb(), MemoryGuard::stop_limit_mb());
      return false;
    }
    MemoryGuard::release_host_bytes(1);
    g.expert_source_mmap = true;
    g.expert_source_capacity = source; // logical bytes, not one contiguous span
    if (!preload_expert_source(g)) {
      LOG_WARN("[%s] could not establish mapped expert views; graph will "
               "use the legacy path\n",
               tag);
      return false;
    }
    LOG_INFO("[%s] mapped-source admission: %.2f GiB logical source, "
             "existing=%.2f GiB reserve=%.2f GiB budget=%.2f GB; no full "
             "anonymous source copy\n",
             tag, (double)source / (1024.0 * 1024.0 * 1024.0),
             (double)host_live / (1024.0 * 1024.0 * 1024.0),
             (double)kWatermark / (1024.0 * 1024.0 * 1024.0),
             (double)MemoryGuard::host_budget_bytes() / 1.0e9);
    return true;
  }
  if (source > SIZE_MAX - kSourceSlack ||
      (source > 0 && !MemoryGuard::try_reserve_host_bytes(source + kSourceSlack,
                                                          kWatermark))) {
    LOG_WARN(
        "[%s] boot host admission refused source=%.2f GiB existing(host_live)="
        "%.2f GiB ledger_reserved=%.2f GiB reserve=%.2f GiB budget=%.2f GB "
        "MemAvailable=%ld MiB (live, not watchdog-stale) stop_mb=%ld "
        "blocked=%d\n",
        tag, (double)source / (1024.0 * 1024.0 * 1024.0),
        (double)host_live / (1024.0 * 1024.0 * 1024.0),
        (double)MemoryGuard::reserved_host_bytes() / (1024.0 * 1024.0 * 1024.0),
        (double)kWatermark / (1024.0 * 1024.0 * 1024.0),
        (double)MemoryGuard::host_budget_bytes() / 1.0e9,
        MemoryGuard::read_available_mb(), MemoryGuard::stop_limit_mb(),
        (int)MemoryGuard::allocation_blocked());
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

} // namespace glm5next_graph

#endif // MOESTREAM_USE_GGML_CUDA
