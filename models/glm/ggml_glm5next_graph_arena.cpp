// GLM5Next full graph: Memory placement: host allocations, dense/state/expert arenas, resident
// tensor lookup, expert source preparation.
//
// Split out of the former single-file ggml_glm5next_graph.cpp; shared
// declarations live in ggml_glm5next_graph_impl.hpp.

#include "ggml_glm5next_graph_impl.hpp"

#if defined(MOESTREAM_USE_GGML_CUDA)

using namespace glm5next_graph;

namespace glm5next_graph {

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

bool g_skip_indexer_weights = false;

bool graph_dense_weight_name(const std::string &name) {
  if (g_skip_indexer_weights && name.find("indexer") != std::string::npos)
    return false;
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

// Binds a dense-weight descriptor to the shared owner's address for `name`,
// or appends and uploads it once.  The owner's bump pointer only ever holds
// immutable weights; mutable route maps use g.scratch instead.
bool dense_weight_bind(FullGraph &g, ggml_tensor *t,
                       const std::string &name, const void *src,
                       size_t src_bytes) {
  DenseArenaOwner *owner = g.dense_owner;
  if (!owner || !owner->arena.buffer || !owner->arena.base || !t)
    return false;

  auto it = owner->addrs.find(name);
  if (it != owner->addrs.end()) {
    if (it->second.bytes != ggml_nbytes(t)) {
      LOG_ERROR("[GGML-GLM5NEXT-FULL] dense reuse shape mismatch name=%s "
                "cached=%zu requested=%zu\n",
                name.c_str(), it->second.bytes, ggml_nbytes(t));
      return false;
    }
    if (ggml_backend_tensor_alloc(owner->arena.buffer, t, it->second.addr) !=
        GGML_STATUS_SUCCESS) {
      LOG_ERROR("[GGML-GLM5NEXT-FULL] dense reuse bind failed name=%s\n",
                name.c_str());
      return false;
    }
    ++owner->cache_hits;
    ++g.dense_cache_hits;
    g.dense.used = owner->arena.used;
    ++g.dense.tensors;
    return true;
  }

  if (!src || src_bytes < ggml_nbytes(t))
    return false;

  const auto buft = ggml_backend_buffer_get_type(owner->arena.buffer);
  const size_t alloc_size = ggml_backend_buft_get_alloc_size(buft, t);
  const size_t align =
      std::max<size_t>(256, ggml_backend_buft_get_alignment(buft));
  const size_t off = (owner->arena.used + align - 1) & ~(align - 1);
  if (off > owner->arena.capacity ||
      alloc_size > owner->arena.capacity - off) {
    LOG_ERROR("[GGML-GLM5NEXT-FULL] dense owner capacity exhausted name=%s "
              "used=%zu capacity=%zu need=%zu\n",
              name.c_str(), owner->arena.used, owner->arena.capacity,
              alloc_size);
    return false;
  }
  if (src_bytes > alloc_size) {
    LOG_ERROR("[GGML-GLM5NEXT-FULL] dense upload exceeds tensor allocation "
              "name=%s src=%zu alloc=%zu\n",
              name.c_str(), src_bytes, alloc_size);
    return false;
  }
  void *addr = static_cast<uint8_t *>(owner->arena.base) + off;
  if (ggml_backend_tensor_alloc(owner->arena.buffer, t, addr) !=
      GGML_STATUS_SUCCESS) {
    LOG_ERROR("[GGML-GLM5NEXT-FULL] dense arena bind failed name=%s\n",
              name.c_str());
    return false;
  }
  const auto upload_start = std::chrono::steady_clock::now();
  const cudaError_t err =
      cudaMemcpy(addr, src, src_bytes, cudaMemcpyHostToDevice);
  const double upload_ms = std::chrono::duration<double, std::milli>(
                               std::chrono::steady_clock::now() - upload_start)
                               .count();
  g.dense_upload_ms += upload_ms;
  owner->upload_ms += upload_ms;
  ++g.dense_uploads;
  ++owner->uploads;
  if (err != cudaSuccess) {
    LOG_ERROR("[GGML-GLM5NEXT-FULL] dense upload failed name=%s: %s\n",
              name.c_str(), cudaGetErrorString(err));
    return false;
  }
  owner->arena.used = off + alloc_size;
  ++owner->arena.tensors;
  owner->addrs.emplace(
      name, DenseArenaOwner::Entry{addr, ggml_nbytes(t)});
  g.dense.used = owner->arena.used;
  ++g.dense.tensors;
  return true;
}

ggml_tensor *resident_tensor(FullGraph &g, ggml_context *ctx,
                             const std::string &name) {
  const bool cacheable = ctx == g.ctx;
  if (cacheable) {
    auto it = g.resident_cache.find(name);
    if (it != g.resident_cache.end())
      return it->second;
  }
  ggml_tensor *r = resident_tensor_uncached(g, ctx, name);
  if (cacheable && r)
    g.resident_cache.emplace(name, r);
  return r;
}

ggml_tensor *resident_tensor_uncached(FullGraph &g, ggml_context *ctx,
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
  if (g.native_moe_cache && q->device_ptr) {
    // The Ling hybrid graph uses the engine's existing dense pool. The small
    // marker buffer supplies GGML's CUDA backend identity; the actual weight
    // address remains owned by CudaManager.
    t->data = q->device_ptr;
    t->buffer = g.dense.buffer;
    ++g.dense.tensors;
  } else if (g.native_moe_cache) {
    if (!bind_host(g, t, q->host_ptr, q->meta.size_bytes,
                   GGML_BACKEND_BUFFER_USAGE_WEIGHTS))
      return nullptr;
  } else if (graph_dense_weight_name(name)) {
    if (!dense_weight_bind(g, t, name, q->host_ptr, q->meta.size_bytes))
      return nullptr;
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
    // DeepSeek binds device-resident dense weights from the engine's pool.
    if (g.engine->family_ops.family == ModelFamily::DeepSeek_V41 && q.device_ptr)
      continue;
    // The owner serves both phases, so reserve the union: output.weight is
    // decode-only in the old phase-sized arena, but a prefill graph may
    // materialize it too and must never append past the allocation.
    if (total > std::numeric_limits<size_t>::max() - q.meta.size_bytes - 4096)
      return 0;
    total += q.meta.size_bytes + 4096;
  }
  const auto &c = g.engine->model_file.architecture_config;
  for (int l = 0; l < c.n_layers; ++l) {
    if (hybrid_is_kda_layer(c, l))
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
  return total + (1ull << 20);
}

// Per-graph mutable tensors that used to share the dense-weight bump arena:
// hot/cold + GPU1-hot route maps and masks, plus pilot-stage hot masks.
// Keeping them in a separate arena means a rebuild refreshes them without
// either overwriting immutable weights or appending a new set forever.
size_t dense_scratch_bytes(const FullGraph &g) {
  const auto &c = g.engine->model_file.architecture_config;
  if (c.n_experts <= 0 || g.S <= 0)
    return 1ull << 20;
  size_t total = 1ull << 20;
  const size_t map_bytes = (size_t)c.n_experts * (size_t)g.S * sizeof(int32_t);
  auto add = [&total](size_t count, size_t unit) {
    if (unit != 0 && count > (SIZE_MAX - total) / unit)
      return false;
    total += count * unit;
    return true;
  };
  const size_t moe_layers =
      (size_t)std::max(0, c.n_layers - c.first_dense);
  if (g.partial_expert_residency || g.vram_reference_experts) {
    // hot_map/hot_mask/cold_map/cold_mask
    if (!add(moe_layers * 4, map_bytes + 4096))
      return 0;
    // hot1_map/hot1_mask (conservatively for every MoE layer)
    if (!add(moe_layers * 2, map_bytes + 4096))
      return 0;
  }
  // stage_hot_mask: one [n_experts] F32 tensor per active pilot layer.
  if (!add((size_t)c.n_layers, (size_t)c.n_experts * sizeof(float) + 4096))
    return 0;
  return total;
}

static bool init_scratch_arena(FullGraph &g) {
  const size_t bytes = dense_scratch_bytes(g);
  if (bytes == 0)
    return false;
  const auto buft = ggml_backend_get_default_buffer_type(g.cuda);
  g.scratch.buffer = ggml_backend_buft_alloc_buffer(buft, bytes);
  if (!g.scratch.buffer) {
    LOG_WARN("[GGML-GLM5NEXT-FULL] scratch arena allocation declined: %.2f "
             "MiB\n",
             (double)bytes / (1024.0 * 1024.0));
    return false;
  }
  // Keep the historical WEIGHTS hint: ggml's scheduler assigns an op from a
  // source buffer's usage, and the route maps must stay on GPU0 with the cold
  // branch.  COMPUTE here moved ~2,230 nodes to the second GPU and raised the
  // decode split count from 273 to 477.  These tensors are refreshed between
  // graph executions, not written concurrently with compute.
  ggml_backend_buffer_set_usage(g.scratch.buffer,
                                GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
  g.scratch.base = ggml_backend_buffer_get_base(g.scratch.buffer);
  g.scratch.capacity = ggml_backend_buffer_get_size(g.scratch.buffer);
  g.scratch.used = 0;
  g.scratch.tensors = 0;
  return true;
}

bool init_dense_arena(FullGraph &g) {
  if (g.native_moe_cache) {
    constexpr size_t kMarkerBytes = 1ull << 20;
    size_t free_bytes = 0, total_bytes = 0;
    if (cudaSetDevice(g.device) != cudaSuccess ||
        cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess ||
        free_bytes < kMarkerBytes + (512ull << 20)) {
      LOG_WARN("[GGML-LING-HYBRID] insufficient checked VRAM for CUDA graph "
               "marker (free=%.2f GiB); declining hybrid graph\n",
               (double)free_bytes / (1024.0 * 1024.0 * 1024.0));
      return false;
    }
    auto buft = ggml_backend_get_default_buffer_type(g.cuda);
    g.dense.buffer = ggml_backend_buft_alloc_buffer(buft, kMarkerBytes);
    if (!g.dense.buffer)
      return false;
    ggml_backend_buffer_set_usage(g.dense.buffer,
                                  GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    g.dense.base = ggml_backend_buffer_get_base(g.dense.buffer);
    g.dense.capacity = ggml_backend_buffer_get_size(g.dense.buffer);
    LOG_INFO("[GGML-LING-HYBRID] using native dense pool; CUDA marker "
             "allocated=%.3f MiB\n",
             (double)g.dense.capacity / (1024.0 * 1024.0));
    return init_scratch_arena(g);
  }

  // The old graph has already been released by the caller before this point,
  // so an owner from a different engine/device cannot have live borrowers.
  if (g_glm5next_dense_owner &&
      (g_glm5next_dense_owner->engine != g.engine ||
       g_glm5next_dense_owner->device != g.device)) {
    LOG_INFO("[GGML-GLM5NEXT-FULL] releasing dense owner for a different "
             "engine/device\n");
    g_glm5next_dense_owner.reset();
  }
  if (!g_glm5next_dense_owner) {
    const size_t bytes = dense_arena_bytes(g);
    if (bytes == 0)
      return false;
    size_t free_bytes = 0, total_bytes = 0;
    cudaSetDevice(g.device);
    const bool have_free =
        cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess;
    constexpr size_t kAdmissionSlack = 256ull << 20;
    if (have_free && free_bytes < bytes + kAdmissionSlack) {
      LOG_WARN("[GGML-GLM5NEXT-FULL] dense owner declined before allocation: "
               "need %.2f GiB + %.2f GiB slack, free %.2f GiB\n",
               (double)bytes / (1024.0 * 1024.0 * 1024.0),
               (double)kAdmissionSlack / (1024.0 * 1024.0 * 1024.0),
               (double)free_bytes / (1024.0 * 1024.0 * 1024.0));
      return false;
    }
    auto owner = std::make_unique<DenseArenaOwner>();
    owner->engine = g.engine;
    owner->device = g.device;
    const auto buft = ggml_backend_get_default_buffer_type(g.cuda);
    owner->arena.buffer = ggml_backend_buft_alloc_buffer(buft, bytes);
    if (!owner->arena.buffer) {
      LOG_WARN("[GGML-GLM5NEXT-FULL] dense owner allocation declined: %.2f "
               "GiB\n",
               (double)bytes / (1024.0 * 1024.0 * 1024.0));
      return false;
    }
    ggml_backend_buffer_set_usage(owner->arena.buffer,
                                  GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    owner->arena.base = ggml_backend_buffer_get_base(owner->arena.buffer);
    owner->arena.capacity = ggml_backend_buffer_get_size(owner->arena.buffer);
    owner->arena.used = 0;
    owner->arena.tensors = 0;
    LOG_INFO("[GGML-GLM5NEXT-FULL] dense owner arena=%.2f GiB (union, includes "
             "output.weight); free before=%.2f GiB\n",
             (double)owner->arena.capacity / (1024.0 * 1024.0 * 1024.0),
             have_free ? (double)free_bytes / (1024.0 * 1024.0 * 1024.0) : -1.0);
    g_glm5next_dense_owner = std::move(owner);
  }

  DenseArenaOwner *owner = g_glm5next_dense_owner.get();
  g.dense_owner = owner;
  g.dense.buffer = owner->arena.buffer;
  g.dense.base = owner->arena.base;
  g.dense.capacity = owner->arena.capacity;
  g.dense.used = owner->arena.used;
  g.dense.tensors = 0;
  g.dense_upload_ms = 0.0;
  g.dense_uploads = 0;
  g.dense_cache_hits = 0;
  if (!init_scratch_arena(g))
    return false;
  LOG_INFO("[GGML-GLM5NEXT-FULL] dense owner bound: arena=%.2f/%.2f GiB "
           "weights=%zu, scratch=%.2f MiB\n",
           (double)g.dense.used / (1024.0 * 1024.0 * 1024.0),
           (double)g.dense.capacity / (1024.0 * 1024.0 * 1024.0),
           owner->addrs.size(),
           (double)g.scratch.capacity / (1024.0 * 1024.0));
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
    if (hybrid_is_kda_layer(c, l)) {
      const size_t state_n = (size_t)g.hd * g.hd * g.H;
      const size_t hist_n = (size_t)(g.d_conv - 1) * 3 * g.d_inner;
      if (state_n > SIZE_MAX / sizeof(float) ||
          hist_n > SIZE_MAX / sizeof(float) ||
          !add_tensor(state_n * sizeof(float)) ||
          !add_tensor(hist_n * sizeof(float)))
        return false;
    } else {
      if ((size_t)g.kv_capacity >
          SIZE_MAX / (size_t)g.kv_cache_width / sizeof(ggml_fp16_t) ||
          !add_tensor((size_t)g.kv_capacity * g.kv_cache_width *
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
      if (hybrid_is_kda_layer(c, l))
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
  g.state.buffer = ggml_backend_buft_alloc_buffer(buft, bytes);
  if (!g.state.buffer) {
    LOG_WARN("[GGML-GLM5NEXT-FULL-DECODE] device state arena declined: "
             "%.2f GiB (KV capacity=%d)\n",
             (double)bytes / (1024.0 * 1024.0 * 1024.0), g.kv_capacity);
    return false;
  }
  ggml_backend_buffer_set_usage(g.state.buffer,
                                GGML_BACKEND_BUFFER_USAGE_COMPUTE);
  g.state.base = ggml_backend_buffer_get_base(g.state.buffer);
  g.state.capacity = ggml_backend_buffer_get_size(g.state.buffer);
  g.state.used = 0;
  LOG_INFO("[GGML-GLM5NEXT-FULL-DECODE] device state arena=%.2f GiB, "
           "KV capacity=%d\n",
           (double)g.state.capacity / (1024.0 * 1024.0 * 1024.0),
           g.kv_capacity);
  return true;
}

ggml_tensor *new_decode_state_tensor(FullGraph &g, ggml_context *ctx,
                                     ggml_type type,
                                     const std::vector<int64_t> &shape,
                                     const char *name) {
  ggml_tensor *t = new_tensor(ctx, type, shape);
  if (!t || !g.state.buffer)
    return nullptr;
  const ggml_backend_buffer_type_t buft =
      ggml_backend_buffer_get_type(g.state.buffer);
  const size_t alloc_size = ggml_backend_buft_get_alloc_size(buft, t);
  const size_t alignment =
      std::max<size_t>(256, ggml_backend_buft_get_alignment(buft));
  const size_t aligned =
      (g.state.used + alignment - 1) & ~(alignment - 1);
  if (aligned > g.state.capacity ||
      alloc_size > g.state.capacity - aligned)
    return nullptr;
  void *addr = static_cast<uint8_t *>(g.state.base) + aligned;
  if (ggml_backend_tensor_alloc(g.state.buffer, t, addr) !=
      GGML_STATUS_SUCCESS)
    return nullptr;
  g.state.used = aligned + alloc_size;
  ggml_set_name(t, name);
  return t;
}

bool init_expert_arena(FullGraph &g) {
  if (!g.partial_expert_residency && g.device_expert_layers <= 0)
    return true;
  if (g.vram_reference_experts) {
    const bool ling = g.engine &&
                      g.engine->family_ops.family == ModelFamily::BailingMoE3;
    const bool has_primary_refs = std::any_of(
        g.partial_experts.begin(), g.partial_experts.end(),
        [](const PartialExpertLayer &p) { return !p.hot.empty(); });
    const bool has_secondary_refs =
        ling && std::any_of(g.partial_experts.begin(), g.partial_experts.end(),
                            [](const PartialExpertLayer &p) {
                              return !p.hot1.empty();
                            });
    if (!has_primary_refs && !has_secondary_refs)
      return true;

    // GGML requires a CUDA buffer handle to classify an aliased expert tensor.
    // These tiny descriptors carry no expert weights; data points directly to
    // GpuTopology's resident cache slots. Check free VRAM on each device first.
    constexpr size_t kReferenceBufferBytes = 1ull << 20;
    constexpr size_t kFreeMargin = 512ull << 20;
    const auto allocate_descriptor = [&](int device, ggml_backend_t backend,
                                         ggml_backend_buffer_t &buffer) {
      if (!backend)
        return false;
      int previous_device = -1;
      if (cudaGetDevice(&previous_device) != cudaSuccess ||
          cudaSetDevice(device) != cudaSuccess)
        return false;
      size_t free_bytes = 0, total_bytes = 0;
      const bool have_memory =
          cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess;
      (void)total_bytes;
      if (!have_memory || free_bytes < kReferenceBufferBytes + kFreeMargin) {
        LOG_WARN("[GGML-LING-CACHE] descriptor declined on GPU%d: free="
                 "%.2f GiB, require %.2f GiB including margin\n",
                 device, (double)free_bytes / (1024.0 * 1024.0 * 1024.0),
                 (double)(kReferenceBufferBytes + kFreeMargin) /
                     (1024.0 * 1024.0 * 1024.0));
        if (previous_device != device)
          cudaSetDevice(previous_device);
        return false;
      }
      buffer = ggml_backend_buft_alloc_buffer(
          ggml_backend_get_default_buffer_type(backend),
          kReferenceBufferBytes);
      if (previous_device != device)
        cudaSetDevice(previous_device);
      if (!buffer)
        return false;
      ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
      return true;
    };

    if (has_primary_refs) {
      if (!allocate_descriptor(g.device, g.cuda, g.expert.buffer))
        return false;
      g.expert.base = ggml_backend_buffer_get_base(g.expert.buffer);
      g.expert.capacity = ggml_backend_buffer_get_size(g.expert.buffer);
    }
    if (has_secondary_refs) {
      const int gpu1_id = second_gpu_device(g.device);
      auto gpu1 = static_cast<ggml_backend_t>(ggml_shared_backend(gpu1_id));
      if (!allocate_descriptor(gpu1_id, gpu1, g.expert1_buffer))
        return false;
    }
    LOG_INFO("[GGML-%s-CACHE] resident MoE weights are aliased; "
             "graph-owned MoE weight bytes=0, descriptors GPU%d=%.2f MiB "
             "GPU%d=%.2f MiB\n",
             ling ? "LING" : "GLM5NEXT", g.device,
             g.expert.buffer
                 ? (double)ggml_backend_buffer_get_size(g.expert.buffer) /
                       (1024.0 * 1024.0)
                 : 0.0,
             second_gpu_device(g.device),
             g.expert1_buffer
                 ? (double)ggml_backend_buffer_get_size(g.expert1_buffer) /
                       (1024.0 * 1024.0)
                 : 0.0);
    return true;
  }
  const size_t bytes = device_expert_bytes(g);
  if (bytes == 0)
    return false;
  const size_t capacity =
      bytes + (size_t)std::max(1, g.n_layers) * 3 * 256 + (1ull << 20);
  const auto buft = ggml_backend_get_default_buffer_type(g.cuda);
  g.expert.buffer = ggml_backend_buft_alloc_buffer(buft, capacity);
  if (!g.expert.buffer) {
    LOG_WARN(
        "[GGML-GLM5NEXT-FULL] expert arena allocation declined: %.2f GiB\n",
        (double)capacity / (1024.0 * 1024.0 * 1024.0));
    return false;
  }
  ggml_backend_buffer_set_usage(g.expert.buffer,
                                GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
  g.expert.base = ggml_backend_buffer_get_base(g.expert.buffer);
  g.expert.capacity = ggml_backend_buffer_get_size(g.expert.buffer);
  LOG_INFO("[GGML-GLM5NEXT-FULL] device expert arena=%.2f GiB%s\n",
           (double)g.expert.capacity / (1024.0 * 1024.0 * 1024.0),
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
