// GLM5Next full graph: Expert and dense weight loading: direct reads, gate/up packing, source
// compaction and preloading.
//
// Split out of the former single-file ggml_glm5next_graph.cpp; shared
// declarations live in ggml_glm5next_graph_impl.hpp.

#include "ggml_glm5next_graph_impl.hpp"

#if defined(MOESTREAM_USE_GGML_CUDA)

using namespace glm5next_graph;

namespace glm5next_graph {

ggml_tensor *load_full_weight(FullGraph &g, ggml_context *ctx,
                              const std::string &name, bool device_expert,
                              bool routed_expert)  {
  const GGUFTensor *m = model_tensor(*g.engine, name);
  if (!m || !type_ok(m->type) || m->shape.empty())
    return nullptr;
  std::vector<int64_t> shape;
  for (uint64_t d : m->shape)
    shape.push_back((int64_t)d);
  ggml_tensor *t = new_tensor(ctx, (ggml_type)m->type, shape);
  if (!t)
    return nullptr;
  const bool dense_weight =
      !device_expert && !routed_expert && graph_dense_weight_name(name);
  if (dense_weight &&
      dense_weight_bind(g, t, name, nullptr, ggml_nbytes(t))) {
    // The shared dense owner already holds this weight: bind a new descriptor
    // without reading the file or uploading it again.
    ggml_set_name(t, name.c_str());
    return t;
  }
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
    ok = arena_alloc(g.expert.buffer, g.expert.base, g.expert.capacity,
                     g.expert.used, t, p, bytes);
    if (transient_device_source)
      std::free(p);
    if (transient_host_reserved)
      MemoryGuard::release_host_bytes(required);
  } else if (dense_weight) {
    ok = dense_weight_bind(g, t, name, p, bytes);
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
  const char *v = moestream::graphcfg::env(name);
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
  const char *env = moestream::graphcfg::env("MOESTREAM_FULL_GRAPH_USAGE_FILE");
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

// Single-snapshot replacement for engine->graph_ram_layer_ready(layer) for
// layers that have a PartialExpertLayer. See PartialExpertLayer::
// ram_ready_cached for why: partial_map(), expert_weight() and
// expert_gate_up_weight() must all see the same answer for one layer within
// one build_graph() call, even though the underlying state is live and
// mutated by background threads. Falls back to the live check for layers
// with no partial residency, where no second tensor's shape depends on the
// same decision.
bool partial_layer_ram_ready(FullGraph &g, int layer) {
  PartialExpertLayer *p = partial_layer(g, layer);
  if (!p)
    return g.engine->graph_ram_layer_ready(layer);
  if (p->ram_ready_cached < 0)
    p->ram_ready_cached = g.engine->graph_ram_layer_ready(layer) ? 1 : 0;
  return p->ram_ready_cached != 0;
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

bool read_tensor_range_direct(const StreamingEngine &engine,
                              const GGUFTensor &meta, size_t relative_offset,
                              void *dst, size_t bytes) {
  if (!dst || relative_offset > (size_t)meta.size_bytes ||
      bytes > (size_t)meta.size_bytes - relative_offset ||
      meta.offset > UINT64_MAX - relative_offset)
    return false;
  const int fd = engine.model_file.fd_for_tensor(meta);
  const bool is_direct = engine.model_file.direct_io_active_for(meta);
  const uint64_t offset = meta.offset + relative_offset;
  const ssize_t got = direct_io::pread_direct_aware(
      fd, is_direct, engine.model_file.direct_io_alignment, dst, bytes, offset);
  const bool ok = got == (ssize_t)bytes;
#if defined(__linux__)
  if (ok && !is_direct)
    (void)posix_fadvise(fd, (off_t)offset, (off_t)bytes,
                        POSIX_FADV_DONTNEED);
#endif
  return ok;
}

bool read_tensor_selected_experts(const StreamingEngine &engine,
                                  const GGUFTensor &meta, int n_experts,
                                  const std::vector<int> &selected,
                                  void *dst) {
  if (selected.empty())
    return true;
  if (!dst || n_experts <= 0 || meta.shape.size() != 3 ||
      meta.shape[2] != (uint64_t)n_experts ||
      meta.size_bytes % (size_t)n_experts != 0)
    return false;
  const size_t record_bytes = (size_t)meta.size_bytes / (size_t)n_experts;
  constexpr size_t kMaxBatchExperts = 8;
  size_t i = 0;
  while (i < selected.size()) {
    if (selected[i] < 0 || selected[i] >= n_experts)
      return false;
    size_t run_end = i + 1;
    while (run_end < selected.size() &&
           selected[run_end] == selected[run_end - 1] + 1)
      ++run_end;
    while (i < run_end) {
      const size_t count = std::min(kMaxBatchExperts, run_end - i);
      const size_t file_offset = (size_t)selected[i] * record_bytes;
      const size_t bytes = count * record_bytes;
      auto *out = static_cast<uint8_t *>(dst) + i * record_bytes;
      if (!read_tensor_range_direct(engine, meta, file_offset, out, bytes))
        return false;
      i += count;
    }
  }
  return true;
}

bool read_pack_gate_up_selected(const StreamingEngine &engine,
                               const GGUFTensor &gate, const GGUFTensor &up,
                               int n_experts,
                               const std::vector<int> &selected, void *dst) {
  if (selected.empty())
    return true;
  if (!dst || gate.shape.size() != 3 || up.shape != gate.shape ||
      n_experts <= 0 || gate.shape[2] != (uint64_t)n_experts ||
      gate.size_bytes % (size_t)n_experts != 0 ||
      up.size_bytes % (size_t)n_experts != 0)
    return false;

  const size_t gate_record = (size_t)gate.size_bytes / (size_t)n_experts;
  const size_t up_record = (size_t)up.size_bytes / (size_t)n_experts;
  constexpr size_t kMaxBatchExperts = 8;
  const size_t max_batch = std::min(kMaxBatchExperts, selected.size());
  if (gate_record > SIZE_MAX / max_batch ||
      up_record > SIZE_MAX / max_batch)
    return false;
  const size_t gate_scratch_bytes = gate_record * max_batch;
  const size_t up_scratch_bytes = up_record * max_batch;
  if (gate_scratch_bytes > SIZE_MAX - up_scratch_bytes)
    return false;
  const size_t scratch_bytes = gate_scratch_bytes + up_scratch_bytes;
  if (!MemoryGuard::try_reserve_host_bytes(scratch_bytes))
    return false;
  void *scratch = nullptr;
  if (posix_memalign(&scratch, 4096, scratch_bytes) != 0 || !scratch) {
    MemoryGuard::release_host_bytes(scratch_bytes);
    return false;
  }

  auto *gate_scratch = static_cast<uint8_t *>(scratch);
  auto *up_scratch = gate_scratch + gate_scratch_bytes;
  auto *out = static_cast<uint8_t *>(dst);
  size_t i = 0;
  bool ok = true;
  while (i < selected.size() && ok) {
    if (selected[i] < 0 || selected[i] >= n_experts) {
      ok = false;
      break;
    }
    size_t run_end = i + 1;
    while (run_end < selected.size() &&
           selected[run_end] == selected[run_end - 1] + 1)
      ++run_end;
    while (i < run_end && ok) {
      const size_t count = std::min(kMaxBatchExperts, run_end - i);
      const size_t file_offset = (size_t)selected[i] * gate_record;
      ok = read_tensor_range_direct(engine, gate, file_offset, gate_scratch,
                                    count * gate_record) &&
           read_tensor_range_direct(engine, up,
                                    (size_t)selected[i] * up_record,
                                    up_scratch, count * up_record);
      if (!ok)
        break;
      for (size_t j = 0; j < count; ++j) {
        uint8_t *record = out + (i + j) * (gate_record + up_record);
        std::memcpy(record, gate_scratch + j * gate_record, gate_record);
        std::memcpy(record + gate_record, up_scratch + j * up_record,
                    up_record);
      }
      i += count;
    }
  }
  std::free(scratch);
  MemoryGuard::release_host_bytes(scratch_bytes);
  return ok;
}

ggml_tensor *load_gate_up_weight(FullGraph &g, ggml_context *ctx, int layer,
                                 bool device_expert) {
  PackedGateUpMeta meta;
  if (!packed_gate_up_meta(*g.engine, layer, &meta))
    return nullptr;

  if (const PartialExpertLayer *p = partial_layer(g, layer)) {
    if (p->cold.empty())
      return nullptr;
    if (g.dynamic_expert_cache && g.expert_source_mmap)
      return nullptr;
    auto ptr_it = g.expert_source_views.find(meta.name);
    auto bytes_it = g.expert_source_view_bytes.find(meta.name);
    if (ptr_it == g.expert_source_views.end() ||
        bytes_it == g.expert_source_view_bytes.end())
      return nullptr;
    const int64_t cold_experts =
        g.dynamic_expert_cache ? (int64_t)meta.gate->shape[2]
                               : (int64_t)p->cold.size();
    ggml_tensor *cold = ggml_new_tensor_3d(
        ctx, (ggml_type)meta.gate->type, (int64_t)meta.gate->shape[0],
        (int64_t)meta.gate->shape[1] * 2, cold_experts);
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
    ok = arena_alloc(g.expert.buffer, g.expert.base, g.expert.capacity,
                     g.expert.used, t, p, meta.bytes);
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
  shape[2] = g.dynamic_expert_cache ? (int64_t)m->shape[2]
                                    : (int64_t)p->cold.size();
  ggml_tensor *t = new_tensor(ctx, (ggml_type)m->type, shape);
  if (!t || bytes_it->second < ggml_nbytes(t) ||
      !bind_host(g, t, ptr_it->second, bytes_it->second,
                 GGML_BACKEND_BUFFER_USAGE_WEIGHTS))
    return nullptr;
  g.host_weight_bytes += bytes_it->second;
  ggml_set_name(t, name.c_str());
  return t;
}

template <typename Loader>
bool arena_load_selected(FullGraph &g, ggml_tensor *t, Loader &&load) {
  if (!t)
    return false;
  const size_t bytes = ggml_nbytes(t);
  if (!MemoryGuard::try_reserve_host_bytes(bytes))
    return false;
  void *scratch = nullptr;
  if (posix_memalign(&scratch, 4096, bytes) != 0 || !scratch) {
    MemoryGuard::release_host_bytes(bytes);
    return false;
  }
  const bool ok = load(scratch) &&
                  arena_alloc(g.expert.buffer, g.expert.base,
                              g.expert.capacity, g.expert.used, t, scratch,
                              bytes);
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
  if (!m || m->shape.size() != 3)
    return nullptr;
  std::vector<int64_t> shape;
  for (uint64_t d : m->shape)
    shape.push_back((int64_t)d);
  shape[2] = (int64_t)p->hot.size();
  ggml_tensor *t = new_tensor(ctx, (ggml_type)m->type, shape);
  if (!t || !arena_load_selected(g, t, [&](void *dst) {
        return read_tensor_selected_experts(
            *g.engine, *m, g.engine->model_file.architecture_config.n_experts,
            p->hot, dst);
      }))
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
  ggml_tensor *t = ggml_new_tensor_3d(
      ctx, (ggml_type)meta.gate->type, (int64_t)meta.gate->shape[0],
      (int64_t)meta.gate->shape[1] * 2, (int64_t)p->hot.size());
  if (!t || !arena_load_selected(g, t, [&](void *dst) {
        return read_pack_gate_up_selected(
            *g.engine, *meta.gate, *meta.up,
            g.engine->model_file.architecture_config.n_experts, p->hot, dst);
      }))
    return nullptr;
  ggml_set_name(t, (meta.name + ".hot").c_str());
  return t;
}

ggml_tensor *expert_weight(FullGraph &g, ggml_context *ctx, int layer,
                           const char *stem) {
  return move_expert_to_gpu1(g, ctx, layer,
                             expert_weight_impl(g, ctx, layer, stem));
}

ggml_tensor *expert_gate_up_weight(FullGraph &g, ggml_context *ctx, int layer) {
  return move_expert_to_gpu1(g, ctx, layer,
                             expert_gate_up_weight_impl(g, ctx, layer));
}

size_t expert_source_bytes(const StreamingEngine &engine,
                           int device_expert_layers) {
  const auto &c = engine.model_file.architecture_config;
  size_t total = 0;
  for (int l = c.first_dense; l < c.n_layers; ++l) {
    if (l < c.first_dense + device_expert_layers)
      continue;
    if (offload_layer(l) || engine.graph_ram_layer_ready(l))
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

size_t expert_source_bytes(const FullGraph &g) {
  if (!g.partial_expert_residency)
    return expert_source_bytes(*g.engine, g.device_expert_layers);

  // Ling borrows hot weights from GpuTopology and cold weights from the
  // existing DDR/OffloadStore owner. Never construct a second graph-owned MoE
  // source arena; if neither external owner can serve a route, decline graph
  // construction instead of copying expert weights into it.
  if (g.vram_reference_experts && g.engine &&
      g.engine->family_ops.family == ModelFamily::BailingMoE3)
    return 0;

  // Neurall-style dynamic caching keeps the original expert records in the
  // host source arena while hot copies come and go in VRAM. This avoids the
  // old eviction path's device-to-host readback and keeps every cold fallback
  // address stable across cache updates.
  if (g.dynamic_expert_cache)
    return expert_source_bytes(*g.engine, g.device_expert_layers);

  const auto &c = g.engine->model_file.architecture_config;
  size_t total = 0;
  for (int l = c.first_dense; l < c.n_layers; ++l) {
    if (l < c.first_dense + g.device_expert_layers || offload_layer(l) ||
        g.engine->graph_ram_layer_ready(l))
      continue;
    if (c.n_experts <= 0 || l >= (int)g.partial_experts.size())
      return 0;
    const size_t cold = g.partial_experts[(size_t)l].cold.size();
    if (cold == 0)
      continue;
    for (const char *stem : {"ffn_gate_exps.weight", "ffn_up_exps.weight",
                             "ffn_down_exps.weight"}) {
      const GGUFTensor *m =
          model_tensor(*g.engine, "blk." + std::to_string(l) + "." + stem);
      if (!m || m->size_bytes % (size_t)c.n_experts != 0)
        return 0;
      const size_t record = (size_t)m->size_bytes / (size_t)c.n_experts;
      if (record > SIZE_MAX / cold ||
          total > SIZE_MAX - record * cold)
        return 0;
      total += record * cold;
    }
  }
  return total;
}

bool compact_partial_source(FullGraph &g) {
  if (!g.partial_expert_residency || g.partial_sources_compacted)
    return true;
  if (g.dynamic_expert_cache) {
    // Dynamic mode deliberately retains full-expert source views. The cold
    // graph maps routed IDs to their original expert indices, so no in-place
    // cold-only compaction is appropriate here.
    g.partial_sources_compacted = true;
    return true;
  }
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

bool preload_mmap_expert_source(FullGraph &g) {
  if (!g.dynamic_expert_cache ||
      (!g.engine->model_file.mmap_active() &&
       !g.engine->model_file.enable_mmap()))
    return false;

  const auto &c = g.engine->model_file.architecture_config;
  const int device_end = c.first_dense + g.device_expert_layers;
  size_t mapped_bytes = 0;
  const auto add_view = [&](const GGUFTensor *meta) -> bool {
    if (!meta || !type_ok(meta->type) || meta->size_bytes == 0 ||
        meta->size_bytes > SIZE_MAX ||
        (size_t)meta->size_bytes > SIZE_MAX - mapped_bytes)
      return false;
    const uint8_t *mapped = g.engine->model_file.mmap_bytes(
        *meta, meta->offset, (size_t)meta->size_bytes);
    if (!mapped)
      return false;
    const bool inserted = g.expert_source_views
                              .emplace(meta->name,
                                       const_cast<uint8_t *>(mapped))
                              .second;
    if (!inserted ||
        !g.expert_source_view_bytes
             .emplace(meta->name, (size_t)meta->size_bytes)
             .second)
      return false;
    mapped_bytes += (size_t)meta->size_bytes;
    ++g.expert_source_tensors;
    return true;
  };

  for (int layer = c.first_dense; layer < c.n_layers; ++layer) {
    if (layer < device_end || g.engine->graph_ram_layer_ready(layer))
      continue;
    PackedGateUpMeta packed;
    if (packed_gate_up_meta(*g.engine, layer, &packed)) {
      // Keep the file's native separate gate/up planes.  Interleaving them
      // into a temporary packed tensor would recreate the full-size host
      // allocation this mmap path is specifically avoiding.
      if (!add_view(packed.gate) || !add_view(packed.up))
        return false;
    } else {
      const std::string prefix = "blk." + std::to_string(layer) + ".";
      if (!add_view(model_tensor(*g.engine, prefix + "ffn_gate_exps.weight")) ||
          !add_view(model_tensor(*g.engine, prefix + "ffn_up_exps.weight")))
        return false;
    }
    const std::string down_name =
        "blk." + std::to_string(layer) + ".ffn_down_exps.weight";
    if (!add_view(model_tensor(*g.engine, down_name)))
      return false;
  }

  if (mapped_bytes == 0 || mapped_bytes != g.expert_source_capacity) {
    LOG_WARN("[GGML-GLM5NEXT-FULL] mapped source size mismatch: mapped=%.2f "
             "GiB expected=%.2f GiB\n",
             (double)mapped_bytes / (1024.0 * 1024.0 * 1024.0),
             (double)g.expert_source_capacity / (1024.0 * 1024.0 * 1024.0));
    return false;
  }
  g.expert_source_used = mapped_bytes;
  LOG_INFO("[GGML-GLM5NEXT-FULL] dynamic expert source mapped read-only: "
           "%.2f GiB across %zu tensors; pages remain OS-reclaimable\n",
           (double)mapped_bytes / (1024.0 * 1024.0 * 1024.0),
           g.expert_source_tensors);
  return true;
}

bool preload_expert_source(FullGraph &g) {
  if (g.expert_source_mmap)
    return preload_mmap_expert_source(g);

  const auto &c = g.engine->model_file.architecture_config;
  struct Job {
    const GGUFTensor *meta = nullptr;
    void *dst = nullptr;
    std::string name;
    const GGUFTensor *packed_meta = nullptr;
    const std::vector<int> *selected_experts = nullptr;
  };
  std::vector<Job> jobs;
  jobs.reserve((size_t)std::max(0, c.n_layers - c.first_dense) * 3);

  const int device_end = c.first_dense + g.device_expert_layers;
  for (int l = c.first_dense; l < c.n_layers; ++l) {
    if (l < device_end)
      continue;
    if (offload_layer(l) || g.engine->graph_ram_layer_ready(l))
      continue;
    const std::vector<int> *selected_experts = nullptr;
    if (g.partial_expert_residency && !g.dynamic_expert_cache) {
      if (l >= (int)g.partial_experts.size())
        return false;
      selected_experts = &g.partial_experts[(size_t)l].cold;
    }
    PackedGateUpMeta packed;
    if (packed_gate_up_meta(*g.engine, l, &packed)) {
      size_t bytes = packed.bytes;
      if (selected_experts) {
        if (c.n_experts <= 0 || packed.bytes % (size_t)c.n_experts != 0)
          return false;
        bytes = packed.bytes / (size_t)c.n_experts *
                selected_experts->size();
      }
      if (bytes > 0) {
        void *dst = alloc_expert_source(g, bytes);
        if (!dst) {
          LOG_ERROR("[GGML-GLM5NEXT-FULL] DDR source arena exhausted at %s\n",
                    packed.name.c_str());
          return false;
        }
        g.expert_source_views.emplace(packed.name, dst);
        g.expert_source_view_bytes.emplace(packed.name, bytes);
        jobs.push_back(
            {packed.gate, dst, packed.name, packed.up, selected_experts});
      }
    } else {
      for (const char *stem : {"ffn_gate_exps.weight", "ffn_up_exps.weight"}) {
        const std::string name = "blk." + std::to_string(l) + "." + stem;
        const GGUFTensor *m = model_tensor(*g.engine, name);
        if (!m || !type_ok(m->type) || m->size_bytes == 0) {
          LOG_ERROR("[GGML-GLM5NEXT-FULL] missing routed tensor %s\n",
                  name.c_str());
          return false;
        }
        size_t bytes = (size_t)m->size_bytes;
        if (selected_experts) {
          if (c.n_experts <= 0 || m->size_bytes % (size_t)c.n_experts != 0)
            return false;
          bytes = (size_t)m->size_bytes / (size_t)c.n_experts *
                  selected_experts->size();
        }
        if (bytes > 0) {
          void *dst = alloc_expert_source(g, bytes);
          if (!dst) {
            LOG_ERROR(
                "[GGML-GLM5NEXT-FULL] DDR source arena exhausted at %s\n",
                name.c_str());
            return false;
          }
          g.expert_source_views.emplace(name, dst);
          g.expert_source_view_bytes.emplace(name, bytes);
          jobs.push_back({m, dst, name, nullptr, selected_experts});
        }
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
    size_t down_bytes = (size_t)down->size_bytes;
    if (selected_experts) {
      if (c.n_experts <= 0 || down->size_bytes % (size_t)c.n_experts != 0)
        return false;
      down_bytes = (size_t)down->size_bytes / (size_t)c.n_experts *
                   selected_experts->size();
    }
    if (down_bytes > 0) {
      void *down_dst = alloc_expert_source(g, down_bytes);
      if (!down_dst) {
        LOG_ERROR("[GGML-GLM5NEXT-FULL] DDR source arena exhausted at %s\n",
                  down_name.c_str());
        return false;
      }
      g.expert_source_views.emplace(down_name, down_dst);
      g.expert_source_view_bytes.emplace(down_name, down_bytes);
      jobs.push_back({down, down_dst, down_name, nullptr, selected_experts});
    }
  }
  if (jobs.empty())
  {
    if (g.partial_expert_residency)
      g.partial_sources_compacted = true;
    return true;
  }

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
  if (const char *env = moestream::graphcfg::env("MOESTREAM_FULL_GRAPH_READ_THREADS")) {
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
      const bool ok = job.selected_experts
                          ? (job.packed_meta
                                 ? read_pack_gate_up_selected(
                                       *g.engine, *job.meta, *job.packed_meta,
                                       c.n_experts, *job.selected_experts,
                                       job.dst)
                                 : read_tensor_selected_experts(
                                       *g.engine, *job.meta, c.n_experts,
                                       *job.selected_experts, job.dst))
                          : (job.packed_meta
                                 ? read_pack_gate_up(*g.engine, *job.meta,
                                                     *job.packed_meta, job.dst)
                                 : read_tensor_direct(*g.engine, *job.meta,
                                                      job.dst));
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

  // Static partial residency is read directly into the compact cold layout.
  // Dynamic mode intentionally retains every original record as the stable
  // host backing store for asynchronous H2D cache fills.
  if (g.partial_expert_residency && !g.dynamic_expert_cache)
    g.partial_sources_compacted = true;
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
  const char *pin_env = moestream::graphcfg::env("MOESTREAM_FULL_GRAPH_PIN_GB");
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

} // namespace glm5next_graph

#endif // MOESTREAM_USE_GGML_CUDA
