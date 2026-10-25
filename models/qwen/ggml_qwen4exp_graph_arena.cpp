// Qwen4Exp full graph: Memory placement: host allocations, device dense/state/expert arenas,
// resident-tensor lookup.
//
// Split out of the former single-file ggml_qwen4exp_graph.cpp; shared
// declarations live in ggml_qwen4exp_graph_internal.hpp.

#include "ggml_qwen4exp_graph_internal.hpp"

#if defined(ZEROMOE_USE_GGML_CUDA)

using namespace qwen4exp_graph;

namespace qwen4exp_graph {

// A preallocated host buffer and a scheduler backend assignment are separate
// pieces of state in ggml_backend_sched.  The former keeps gallocr from
// allocating storage for the tensor; the latter keeps the operation that
// consumes the tensor on the CPU backend.  This matters for full-graph
// Qwen4Exp builds: without the explicit assignment, a host-backed expert
// weight can be reconsidered during scheduler splitting and its CPU-layer
// MoE subtree can be represented as a CUDA allocation request.  At 22 GPU /
// 26 CPU layers that request is approximately the full 26-layer expert
// footprint (~88 GiB), instead of a small activation workspace.
void bind_host_expert_scheduler_backend(FullGraph &g, ggml_tensor *t) {
  static int zeromoe_bind_diag = -1;
  if (zeromoe_bind_diag < 0) {
    const char *e = zeromoe::graphcfg::env("ZEROMOE_BIND_HOST_DIAG");
    zeromoe_bind_diag = (e && e[0] == '1') ? 1 : 0;
  }
  if (zeromoe_bind_diag) {
    LOG_WARN(
        "[ZEROMOE-BIND-DIAG] name=%s sched=%p cpu=%p t=%p t->buffer=%p "
        "is_host=%d\n",
        t ? ggml_get_name(t) : "(null)", (void *)g.sched, (void *)g.cpu,
        (void *)t, t ? (void *)t->buffer : nullptr,
        (t && t->buffer) ? (int)ggml_backend_buffer_is_host(t->buffer) : -1);
  }
  if (!g.sched || !g.cpu || !t || !t->buffer ||
      !ggml_backend_buffer_is_host(t->buffer)) {
    if (zeromoe_bind_diag)
      LOG_WARN("[ZEROMOE-BIND-DIAG] SKIPPED pin for name=%s\n",
               t ? ggml_get_name(t) : "(null)");
    return;
  }
  ggml_backend_sched_set_tensor_backend(g.sched, t, g.cpu);
}

ggml_tensor *resident_tensor(
    FullGraph &g, ggml_context *ctx, const std::string &name,
    ggml_backend_buffer_usage usage)  {
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
  if (g.dense.buffer && usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
      graph_dense_weight_name(name)) {
    // Prefix/suffix Phase 2 graphs are mutually exclusive at runtime and
    // only read these weights.  Reuse the prefix-owned buffer rather than
    // allocating and uploading a full dense arena for every graph instance.
    // A suffix may request a name the prefix never materialized; in that
    // case append it to the owner's pre-sized arena and record the address
    // for the remaining siblings.
    FullGraph &owner = g.dense_source_graph
                           ? *g.dense_source_graph
                           : g;
    auto existing = owner.dense_addrs.find(name);
    void *addr = nullptr;
    if (existing != owner.dense_addrs.end()) {
      addr = existing->second;
      const ggml_status bind_status =
          ggml_backend_tensor_alloc(owner.dense.buffer, t, addr);
      if (bind_status != GGML_STATUS_SUCCESS) {
        LOG_ERROR("[GGML-QWEN-FULL-ARENA-BIND-FAIL] arena=dense name=%s "
                  "reason=existing-address status=%d bytes=%zu\n",
                  name.c_str(), static_cast<int>(bind_status),
                  ggml_nbytes(t));
        return nullptr;
      }
    } else {
      const ggml_backend_buffer_type_t buft =
          ggml_backend_buffer_get_type(owner.dense.buffer);
      const size_t alloc_size = ggml_backend_buft_get_alloc_size(buft, t);
      const size_t alignment =
          std::max<size_t>(256, ggml_backend_buft_get_alignment(buft));
      const size_t aligned =
          (owner.dense.used + alignment - 1) & ~(alignment - 1);
      if (aligned > owner.dense.capacity ||
          alloc_size > owner.dense.capacity - aligned) {
        LOG_ERROR("[GGML-QWEN-FULL-ARENA-BIND-FAIL] arena=dense name=%s "
                  "reason=capacity used=%zu capacity=%zu need=%zu\n",
                  name.c_str(), owner.dense.used,
                  owner.dense.capacity, alloc_size);
        return nullptr;
      }
      addr = static_cast<uint8_t *>(owner.dense.base) + aligned;
      const ggml_status bind_status =
          ggml_backend_tensor_alloc(owner.dense.buffer, t, addr);
      if (bind_status != GGML_STATUS_SUCCESS) {
        LOG_ERROR("[GGML-QWEN-FULL-ARENA-BIND-FAIL] arena=dense name=%s "
                  "reason=new-address status=%d bytes=%zu\n",
                  name.c_str(), static_cast<int>(bind_status),
                  ggml_nbytes(t));
        return nullptr;
      }
      const auto t_memcpy_start = std::chrono::steady_clock::now();
      const cudaError_t err = cudaMemcpy(
          addr, q->host_ptr, q->meta.size_bytes, cudaMemcpyHostToDevice);
      owner.dense_upload_ms_total += std::chrono::duration<double, std::milli>(
                                        std::chrono::steady_clock::now() -
                                        t_memcpy_start)
                                        .count();
      if (err != cudaSuccess) {
        LOG_ERROR("[GGML-QWEN-FULL] dense device upload failed for %s: %s\n",
                  name.c_str(), cudaGetErrorString(err));
        return nullptr;
      }
      owner.dense.used = aligned + alloc_size;
      owner.dense_addrs.emplace(name, addr);
    }
    ++g.dense.tensors;
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
         name == "output_hc_up.weight" || name == "output.weight" ||
         name == "output_norm.weight";
}

bool init_device_dense_arena(FullGraph &g) {
  if (g.dense_source_graph) {
    FullGraph &source = *g.dense_source_graph;
    if (!source.dense.buffer || !source.dense.base ||
        source.dense.capacity == 0)
      return false;
    g.dense.buffer = source.dense.buffer;
    g.dense.base = source.dense.base;
    g.dense.capacity = source.dense.capacity;
    g.dense.used = 0;
    g.dense.tensors = 0;
    LOG_INFO("[GGML-QWEN-FULL] dense device arena borrowed from source graph: "
             "%.2f GiB\n",
             static_cast<double>(g.dense.capacity) /
                 (1024.0 * 1024.0 * 1024.0));
    return true;
  }
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
  g.dense.buffer = ggml_backend_buft_alloc_buffer(buft, capacity);
  if (!g.dense.buffer) {
    LOG_WARN(
        "[GGML-QWEN-FULL] dense device arena allocation declined: %.2f GiB\n",
        static_cast<double>(capacity) / (1024.0 * 1024.0 * 1024.0));
    return false;
  }
  ggml_backend_buffer_set_usage(g.dense.buffer,
                                GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
  g.dense.base = ggml_backend_buffer_get_base(g.dense.buffer);
  g.dense.capacity = ggml_backend_buffer_get_size(g.dense.buffer);
  g.dense.used = 0;
  g.dense.tensors = 0;
  LOG_INFO("[GGML-QWEN-FULL] dense device arena=%.2f GiB for %d graph weight "
           "tensors\n",
           static_cast<double>(g.dense.capacity) /
               (1024.0 * 1024.0 * 1024.0),
           tensors);
  return true;
}

bool init_device_state_arena(FullGraph &g) {
  g.prefill_device_state_inputs =
      !g.decode_mode && prefill_device_state_inputs_enabled(*g.engine);
  if ((!g.decode_mode || g.kv_capacity <= 0) &&
      !g.prefill_device_state_inputs)
    return true;
  const auto &c = g.engine->model_file.architecture_config;
  size_t bytes = 0;
  auto add = [&](size_t n) {
    constexpr size_t kSlack = 4096;
    if (n > SIZE_MAX - kSlack || bytes > SIZE_MAX - n - kSlack)
      return false;
    bytes += n + kSlack;
    return true;
  };
  for (int l = 0; l < g.n_layers; ++l) {
    if (!qwen4exp_is_gdn_layer(c, l) && g.decode_mode) {
      const size_t kv = static_cast<size_t>(g.kv_capacity) * g.Hkv * g.hd *
                        sizeof(float);
      if (!add(kv) || !add(kv))
        return false;
    } else if (qwen4exp_is_gdn_layer(c, l)) {
      const size_t state = static_cast<size_t>(g.gdn_hd) * g.gdn_hd *
                           g.Hv_gdn * sizeof(float);
      const size_t hist = static_cast<size_t>(g.d_conv - 1) * g.conv_dim *
                          sizeof(float);
      if (!add(state) || !add(hist))
        return false;
    }
    if (g.decode_mode && qwen4exp_is_ple_layer(c, l)) {
      const size_t ple_hist = static_cast<size_t>(c.ple_conv_kernel - 1) *
                              c.ple_ngram_size * g.D * g.HC * sizeof(float);
      if (!add(ple_hist))
        return false;
    }
  }
  constexpr size_t kArenaSlack = 1ull << 20;
  if (bytes > SIZE_MAX - kArenaSlack)
    return false;
  const size_t capacity = bytes + kArenaSlack;
  const ggml_backend_buffer_type_t buft =
      ggml_backend_get_default_buffer_type(g.cuda);
  g.state.buffer = ggml_backend_buft_alloc_buffer(buft, capacity);
  if (!g.state.buffer) {
    LOG_WARN("[GGML-QWEN-FULL-DECODE] device state arena declined: %.2f GiB\n",
             static_cast<double>(capacity) / (1024.0 * 1024.0 * 1024.0));
    return false;
  }
  ggml_backend_buffer_set_usage(g.state.buffer,
                                GGML_BACKEND_BUFFER_USAGE_COMPUTE);
  g.state.base = ggml_backend_buffer_get_base(g.state.buffer);
  g.state.capacity = ggml_backend_buffer_get_size(g.state.buffer);
  g.state.used = 0;
  if (g.prefill_device_state_inputs)
    ggml_backend_buffer_clear(g.state.buffer, 0);
  LOG_INFO("[GGML-QWEN-FULL-DECODE] device state arena=%.2f GiB, KV capacity=%d\n",
           static_cast<double>(g.state.capacity) /
               (1024.0 * 1024.0 * 1024.0),
           g.kv_capacity);
  if (g.prefill_device_state_inputs)
    LOG_INFO("[GGML-QWEN-FULL] immutable zero GDN inputs on CUDA: "
             "no per-prompt host clearing or H2D state upload\n");
  return true;
}

ggml_tensor *new_device_state_tensor(FullGraph &g, ggml_type type,
                                     const std::vector<int64_t> &shape,
                                     const char *name) {
  ggml_tensor *t = new_tensor(g.ctx, type, shape);
  if (!t)
    return nullptr;
  // Phase 2 SuffixCpu (or a decode graph following full-graph prefill): bind
  // to the SAME physical state a source instance already owns for this exact
  // call position, instead of allocating a fresh, independent buffer.  The
  // prefill handoff table deliberately contains null entries for KV tensors
  // whose decode capacity is larger than the S-token prefill allocation; such
  // entries fall through to the normal decode-state arena allocation below.
  if (g.state_source_graph) {
    FullGraph &src = *g.state_source_graph;
    if (src.state_handoff_ready) {
      if (g.state_handoff_cursor >= src.state_handoff_addrs.size() ||
          g.state_handoff_cursor >= src.state_handoff_buffers.size())
        return nullptr;
      const size_t cursor = g.state_handoff_cursor++;
      ggml_backend_buffer_t buffer = src.state_handoff_buffers[cursor];
      void *addr = src.state_handoff_addrs[cursor];
      if (buffer && addr) {
        if (ggml_backend_tensor_alloc(buffer, t, addr) !=
            GGML_STATUS_SUCCESS)
          return nullptr;
        g.state_handoff_tensors.insert(t);
        ggml_set_name(t, name);
        return t;
      }
      // Null source entries are intentionally allocated in this graph's
      // private state arena (currently only expanded decode KV tensors).
    } else {
      if (!src.state.buffer ||
          g.state_replay_cursor >= src.state_addrs.size())
        return nullptr;
      void *addr = src.state_addrs[g.state_replay_cursor++];
      if (ggml_backend_tensor_alloc(src.state.buffer, t, addr) !=
          GGML_STATUS_SUCCESS)
        return nullptr;
      g.state_handoff_tensors.insert(t);
      ggml_set_name(t, name);
      return t;
    }
  }
  if (!g.state.buffer)
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
  // Record for a potential Phase 2 sibling built right after this one
  // (harmless no-op cost otherwise -- one pointer per call, freed with
  // the rest of this instance).
  g.state_addrs.push_back(addr);
  ggml_set_name(t, name);
  return t;
}

// True for a layer inside the device_expert_layers-sized GPU-resident
// block, wherever that block sits (leading, the historical default, or
// trailing, matching llama.cpp's `-ncmoe` convention -- see
// FullGraph::device_expert_trailing's own comment). Single source of
// truth for the block's bounds so device_expert_bytes() and
// plan_expert_vram_split()'s eligibility filter can't drift from
// graph_expert_projection_resident()'s own notion of which layers are
// covered.
bool device_expert_block_covers(const FullGraph &g, int layer) {
  if (g.device_expert_layers <= 0)
    return false;
  const int n_layers = g.engine->model_file.architecture_config.n_layers;
  if (g.device_expert_trailing)
    return layer >= n_layers - g.device_expert_layers;
  return layer < g.device_expert_layers;
}

bool graph_expert_projection_resident(const FullGraph &g, int layer,
                                      const char *stem) {
  // cpu_affinity_layers (config): force this layer's routed-expert weights
  // to stay CPU-resident regardless of device_expert_layers, mirroring the
  // legacy dispatch path's own is_cpu_affinity_layer() contract ("never
  // pin or pre-warm VRAM for these layers"). Checked first so it overrides
  // the leading-N device_expert_layers count below even if this layer would
  // otherwise have qualified for VRAM residency.
  if (is_cpu_affinity_layer(layer))
    return false;
  if (layer >= 0 && device_expert_block_covers(g, layer))
    return true;
  // device_expert_partial_gate_up's boundary-layer optimization is only
  // meaningful for the leading convention (the boundary is the first CPU
  // layer just after the leading GPU block, layer==device_expert_layers);
  // it is forced off in trailing mode at the config-set call site instead
  // of generalizing this boundary math, so this line never fires there.
  return g.device_expert_partial_gate_up && layer == g.device_expert_layers &&
         std::strcmp(stem, "ffn_down_exps.weight") != 0;
}

bool init_device_expert_arena(FullGraph &g) {
  if (g.expert_source_graph) {
    FullGraph &source = *g.expert_source_graph;
    if (!source.expert.buffer || !source.expert.base ||
        source.expert.capacity == 0)
      return false;
    // The owner may have reduced a whole-layer request to fit the actual
    // device budget.  Reuse its final placement/layout, not this borrower's
    // original request, so every graph binds the same physical records.
    g.device_expert_layers = source.device_expert_layers;
    g.device_expert_partial_gate_up = source.device_expert_partial_gate_up;
    g.split_gpu_expert_ids = source.split_gpu_expert_ids;
    g.expert.buffer = source.expert.buffer;
    g.expert.base = source.expert.base;
    g.expert.capacity = source.expert.capacity;
    g.expert.used = 0;
    LOG_INFO("[GGML-QWEN-FULL] device expert arena borrowed from source graph: "
             "%.2f GiB\n",
             static_cast<double>(g.expert.capacity) /
                 (1024.0 * 1024.0 * 1024.0));
    return true;
  }
  const bool split_active = !g.split_gpu_expert_ids.empty();
  // See device_expert_bytes's own comment: the Phase 1/2 partial-layer
  // compact tensor needs this same arena, on every decode instance that
  // could possibly build it, even when neither of the other two features
  // that normally drive this arena's sizing is active.
  const bool partial_active =
      g.decode_mode && g.engine &&
      !g.engine->qwen4exp_partial_layer_gpu_ids.empty() &&
      g.engine->qwen4exp_partial_layer >= 0;
  // device_expert_bytes() already sizes in qwen4exp_gpu_stage_layers'
  // compact per-layer tensors (see its own comment), but this early-return
  // guard never checked for them -- with 0 preloaded layers and no Option-B
  // split, it returned before device_expert_buffer was ever allocated, so
  // EVERY gpu-stage layer's make_dst() call failed on a null buffer
  // ("compact device destination allocation failed") regardless of how
  // small the real requirement was. Same class of gap partial_active above
  // already guards against for the Phase 1/2 tensor.
  const bool gpu_stage_active =
      g.decode_mode && g.engine && g.engine->qwen4exp_gpu_stage_enabled &&
      !g.engine->qwen4exp_gpu_stage_layers.empty();
  if (g.device_expert_layers <= 0 && !split_active && !partial_active &&
      !gpu_stage_active)
    return true;
  const ggml_backend_buffer_type_t buft =
      ggml_backend_get_default_buffer_type(g.cuda);
  const int requested_layers = g.device_expert_layers;
  // A request for all 48 routed layers is valid as a placement preference but
  // cannot fit on the 24-GiB card.  Clamp only whole-layer residency requests
  // to the largest arena that actually allocates; the graph then keeps the
  // remaining layers on the DDR/CPU backend instead of abandoning the entire
  // full graph. Per-expert split allocations (and the partial-layer compact
  // tensor, same rationale) remain strict until their own budget is made
  // independently retryable.
  const bool allow_layer_clamp =
      requested_layers > 0 && !split_active && !partial_active;

  auto sync_option_b_hot_masks = [&]() {
    if (!g.engine || !g.engine->qwen4exp_option_b_fill_vram)
      return;
    const int n_experts = g.engine->model_file.architecture_config.n_experts;
    for (const auto &[layer, ids] : g.split_gpu_expert_ids) {
      std::vector<uint8_t> mask(static_cast<size_t>(std::max(0, n_experts)),
                                0);
      for (int expert : ids) {
        if (expert >= 0 && expert < n_experts)
          mask[static_cast<size_t>(expert)] = 1;
      }
      g.engine->qwen4exp_option_b_hot_masks[layer] = std::move(mask);
      if (layer == g.engine->qwen4exp_option_b_layer)
        g.engine->qwen4exp_option_b_hot_mask =
            g.engine->qwen4exp_option_b_hot_masks[layer];
    }
  };

  if (split_active && g.engine && g.engine->qwen4exp_option_b_fill_vram &&
      g.engine->qwen4exp_option_b_global_top_k <= 0) {
    const auto &c = g.engine->model_file.architecture_config;
    const int minimum = std::min(g.engine->qwen4exp_option_b_top_n,
                                 c.n_experts);
    size_t free_bytes = 0, total_bytes = 0;
    int previous_device = -1;
    cudaGetDevice(&previous_device);
    cudaSetDevice(g.device);
    const cudaError_t meminfo = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (previous_device >= 0)
      cudaSetDevice(previous_device);
    constexpr size_t kGraphFreeFloor = 1ull * 1024ull * 1024ull * 1024ull;
    constexpr size_t kArenaSlack = 1ull << 20;
    const size_t tensor_slack = g.split_gpu_expert_ids.size() * 3 * 256;
    const size_t reserve = kGraphFreeFloor + tensor_slack + kArenaSlack;
    const size_t max_data_bytes =
        meminfo == cudaSuccess && free_bytes > reserve
            ? free_bytes - reserve
            : 0;
    std::unordered_map<int, std::vector<int>> removed_experts;
    size_t bytes = device_expert_bytes(g);
    while (minimum > 0 && bytes > max_data_bytes) {
      int trim_layer = -1;
      size_t trim_per_expert = 0;
      for (const auto &[layer, ids] : g.split_gpu_expert_ids) {
        if (static_cast<int>(ids.size()) <= minimum)
          continue;
        const size_t per_expert = g.engine->compute_expert_slab_bytes(layer);
        if (per_expert > trim_per_expert) {
          trim_layer = layer;
          trim_per_expert = per_expert;
        }
      }
      if (trim_layer >= 0 && trim_per_expert != 0) {
        auto &ids = g.split_gpu_expert_ids[trim_layer];
        removed_experts[trim_layer].push_back(ids.back());
        ids.pop_back(); // ranked least-used expert in this layer
        bytes = device_expert_bytes(g);
        continue;
      }

      // If every selected layer is at the requested 50% floor and the
      // aggregate still does not fit, remove the lowest-coverage layer. This
      // preserves the user's layer-ranking rule instead of silently falling
      // back to the legacy dispatcher for the entire graph.
      int drop_layer = -1;
      double drop_coverage = std::numeric_limits<double>::infinity();
      for (const auto &[layer, ids] : g.split_gpu_expert_ids) {
        (void)ids;
        const auto it = g.engine->qwen4exp_option_b_layer_coverage.find(layer);
        const double coverage =
            it == g.engine->qwen4exp_option_b_layer_coverage.end()
                ? 0.0
                : it->second;
        if (coverage < drop_coverage ||
            (coverage == drop_coverage && layer > drop_layer)) {
          drop_layer = layer;
          drop_coverage = coverage;
        }
      }
      if (drop_layer < 0)
        break;
      LOG_WARN("[OPTION-B] VRAM capacity dropped layer %d at its %d-expert "
               "floor (coverage=%.3f)\n",
               drop_layer, minimum, drop_coverage * 100.0);
      g.split_gpu_expert_ids.erase(drop_layer);
      removed_experts.erase(drop_layer);
      g.engine->qwen4exp_option_b_hot_masks.erase(drop_layer);
      g.engine->qwen4exp_option_b_layer_coverage.erase(drop_layer);
      g.engine->qwen4exp_option_b_layers.erase(
          std::remove(g.engine->qwen4exp_option_b_layers.begin(),
                      g.engine->qwen4exp_option_b_layers.end(), drop_layer),
          g.engine->qwen4exp_option_b_layers.end());
      bytes = device_expert_bytes(g);
    }

    // After dropping whole layers, put back as many removed tail experts as
    // the remaining capacity allows. This makes the final placement as full
    // as possible rather than stopping at an unnecessarily conservative 50%.
    bool added = true;
    while (added) {
      added = false;
      for (auto &[layer, removed] : removed_experts) {
        auto it = g.split_gpu_expert_ids.find(layer);
        if (it == g.split_gpu_expert_ids.end() || removed.empty())
          continue;
        const size_t per_expert = g.engine->compute_expert_slab_bytes(layer);
        if (per_expert == 0 || bytes + per_expert > max_data_bytes)
          continue;
        it->second.push_back(removed.back());
        removed.pop_back();
        bytes = device_expert_bytes(g);
        added = true;
      }
    }
    sync_option_b_hot_masks();
    size_t final_experts = 0;
    for (const auto &[layer, ids] : g.split_gpu_expert_ids)
      final_experts += ids.size();
    LOG_INFO("[OPTION-B] VRAM fill: %.2f/%.2f GiB expert data, %zu expert "
             "records across %zu split layer(s), floor=%.2f GiB\n",
             static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0),
             static_cast<double>(max_data_bytes) /
                 (1024.0 * 1024.0 * 1024.0),
             final_experts, g.split_gpu_expert_ids.size(),
             static_cast<double>(kGraphFreeFloor) /
                 (1024.0 * 1024.0 * 1024.0));
  }

  for (;;) {
    size_t bytes = device_expert_bytes(g);
    if (bytes == 0) {
      if (g.device_expert_layers <= 0 && !split_active && !partial_active) {
        g.split_gpu_expert_ids.clear();
        return true;
      }
      if (allow_layer_clamp && g.device_expert_layers > 1) {
        --g.device_expert_layers;
        continue;
      }
      if (allow_layer_clamp && g.device_expert_layers == 1) {
        g.device_expert_layers = 0;
        g.device_expert_partial_gate_up = false;
        LOG_WARN("[GGML-QWEN-FULL] no whole expert layer fits in the "
                 "device arena; keeping all routed experts in DDR/CPU\n");
        return true;
      }
      return false;
    }
    constexpr size_t kAlignment = 256;
    size_t tensor_slack =
        static_cast<size_t>(device_expert_resident_layer_count(g)) * 3 *
        kAlignment;
    // Each split layer allocates 3 more small tensors (gate/up/down) into the
    // same bump arena -- same per-tensor alignment slack accounting as above.
    tensor_slack += g.split_gpu_expert_ids.size() * 3 * kAlignment;
    // The partial-layer compact tensor is also 3 tensors (gate/up/down).
    if (partial_active)
      tensor_slack += 3 * kAlignment;
    constexpr size_t kArenaSlack = 1ull << 20;
    if (bytes > SIZE_MAX - tensor_slack - kArenaSlack) {
      if (allow_layer_clamp && g.device_expert_layers > 1) {
        --g.device_expert_layers;
        continue;
      }
      return false;
    }
    size_t capacity = bytes + tensor_slack + kArenaSlack;
    // Soft tier (ZEROMOE_QWEN_SOFT_PIN_GIB > 0, decode split arenas only): the split tensors of the last layers get
    // their own ~1 GiB buffers, so a prefill that needs VRAM can free some of them and the next decode build re-uploads
    // just those instead of the whole arena. The hard buffer shrinks by the same payload; total VRAM is unchanged.
    for (auto &seg : g.expert_soft)
      if (seg.buffer)
        ggml_backend_buffer_free(seg.buffer);
    g.expert_soft.clear();
    g.expert_soft_of_layer.clear();
    g.expert_buft = buft;
    if (split_active && g.decode_mode && g.engine) {
      // Default 6 GiB (prompts up to ~2,300 tokens at K=14,000 re-upload only what the prefill displaced); 0 turns the tier
      // off. Capped at half of the split payload so a small arena keeps a real hard part.
      const double soft_gib = zeromoe::graphcfg::atof_or("ZEROMOE_QWEN_SOFT_PIN_GIB", 6.0);
      const size_t soft_target =
          soft_gib > 0.0 ? std::min(static_cast<size_t>(soft_gib * 1024.0 * 1024.0 * 1024.0), bytes / 2) : 0;
      constexpr size_t kSoftSegmentBytes = 1ull << 30;
      if (soft_target > 0) {
        std::vector<int> layers;
        for (const auto &[layer, ids] : g.split_gpu_expert_ids)
          if (!ids.empty())
            layers.push_back(layer);
        std::sort(layers.rbegin(), layers.rend());
        size_t soft_payload = 0, current = 0;
        FullGraph::ExpertSoftSegment seg;
        for (int layer : layers) {
          if (soft_payload >= soft_target)
            break;
          const size_t per_expert = g.engine->compute_expert_slab_bytes(layer);
          const size_t layer_payload =
              per_expert * g.split_gpu_expert_ids[layer].size() + 3 * kAlignment;
          seg.layers.push_back(layer);
          current += layer_payload;
          soft_payload += layer_payload;
          if (current >= kSoftSegmentBytes) {
            seg.capacity = current + kArenaSlack;
            g.expert_soft.push_back(std::move(seg));
            seg = FullGraph::ExpertSoftSegment{};
            current = 0;
          }
        }
        if (!seg.layers.empty()) {
          seg.capacity = current + kArenaSlack;
          g.expert_soft.push_back(std::move(seg));
        }
        bool soft_ok = soft_payload < capacity;
        for (size_t i = 0; soft_ok && i < g.expert_soft.size(); ++i) {
          auto &sg = g.expert_soft[i];
          sg.buffer = ggml_backend_buft_alloc_buffer(buft, sg.capacity);
          if (!sg.buffer) {
            soft_ok = false;
            break;
          }
          ggml_backend_buffer_set_usage(sg.buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
          sg.base = ggml_backend_buffer_get_base(sg.buffer);
          sg.used = 0;
        }
        if (!soft_ok) {
          LOG_WARN("[GGML-QWEN-FULL] soft expert tier of %.2f GiB could not be allocated; "
                   "using one arena\n", static_cast<double>(soft_payload) / (1024.0 * 1024.0 * 1024.0));
          for (auto &sg : g.expert_soft)
            if (sg.buffer)
              ggml_backend_buffer_free(sg.buffer);
          g.expert_soft.clear();
        } else {
          for (size_t i = 0; i < g.expert_soft.size(); ++i)
            for (int layer : g.expert_soft[i].layers)
              g.expert_soft_of_layer[layer] = static_cast<int>(i);
          capacity -= soft_payload;
          LOG_INFO("[GGML-QWEN-FULL] soft expert tier: %zu segment(s), %.2f GiB (layers %d down to %d); hard arena %.2f GiB\n",
                   g.expert_soft.size(), static_cast<double>(soft_payload) / (1024.0 * 1024.0 * 1024.0),
                   g.expert_soft.front().layers.front(),
                   g.expert_soft.back().layers.back(),
                   static_cast<double>(capacity) / (1024.0 * 1024.0 * 1024.0));
        }
      }
    }
    g.expert.buffer = ggml_backend_buft_alloc_buffer(buft, capacity);
    if (g.expert.buffer)
      {
        // build_graph() applies the same one-GiB free-VRAM floor before the
        // scheduler reserves its workspace.  Apply it here too, while the
        // whole-layer request is still retryable; otherwise a 24-layer arena
        // can allocate successfully and only fail later during graph
        // construction, returning the entire request to the legacy path.
        size_t free_bytes = 0, total_bytes = 0;
        const cudaError_t meminfo =
            cudaMemGetInfo(&free_bytes, &total_bytes);
        constexpr size_t kGraphFreeFloor = 1ull * 1024ull * 1024ull * 1024ull;
        if (meminfo == cudaSuccess && free_bytes < kGraphFreeFloor &&
            allow_layer_clamp && g.device_expert_layers > 1) {
          ggml_backend_buffer_free(g.expert.buffer);
          g.expert.buffer = nullptr;
          g.expert.base = nullptr;
          g.expert.capacity = 0;
          g.expert.used = 0;
          --g.device_expert_layers;
          continue;
        }
        break;
      }
    if (allow_layer_clamp && g.device_expert_layers > 1) {
      --g.device_expert_layers;
      continue;
    }
    if (allow_layer_clamp && g.device_expert_layers == 1) {
      g.device_expert_layers = 0;
      g.device_expert_partial_gate_up = false;
      LOG_WARN("[GGML-QWEN-FULL] device expert arena could not fit even "
               "one whole layer; keeping all routed experts in DDR/CPU\n");
      return true;
    }
    LOG_WARN(
        "[GGML-QWEN-FULL] device expert arena allocation declined: %.2f GiB\n",
        static_cast<double>(capacity) / (1024.0 * 1024.0 * 1024.0));
    // allow_layer_clamp is deliberately false whenever split_active (see its
    // own comment) -- shrinking device_expert_layers in place here, mid-loop,
    // while split_gpu_expert_ids stays fixed from an earlier
    // plan_expert_vram_split() call, is NOT safe (the trailing-window
    // boundary moves, so device_expert_block_covers()'s exclusion set for
    // the split layers would be stale). But the CALLER's retry loop around
    // build_one() already rebuilds the entire FullGraph from scratch on each
    // attempt -- including a fresh plan_expert_vram_split() -- so it hits
    // none of that staleness. Signal it the same way build_graph's own later
    // check does, so a config that only fails HERE (real cudaMalloc OOM
    // during arena allocation, not just build_graph's scheduler-admission
    // floor) still gets the graceful shrink-and-retry instead of collapsing
    // straight to legacy dispatch. Real, reproduced case: qwen4exp_option_b_
    // selected_layers=7 layers at device_expert_layers=18, 2026-09-26.
    if (g.device_expert_layers > 0)
      g_qwen4exp_build_graph_retry_smaller_expert_layers = true;
    return false;
  }
  if (g.device_expert_layers != requested_layers)
    LOG_WARN("[GGML-QWEN-FULL] requested %d device expert layers, but only "
             "%d whole layers fit; remaining layers stay in DDR/CPU\n",
             requested_layers, g.device_expert_layers);
  ggml_backend_buffer_set_usage(g.expert.buffer,
                                GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
  g.expert.base = ggml_backend_buffer_get_base(g.expert.buffer);
  g.expert.capacity =
      ggml_backend_buffer_get_size(g.expert.buffer);
  g.expert.used = 0;
  {
    const int n_layers = g.engine->model_file.architecture_config.n_layers;
    const int block_start =
        g.device_expert_trailing ? n_layers - g.device_expert_layers : 0;
    const int block_end =
        g.device_expert_trailing ? n_layers : g.device_expert_layers;
    LOG_INFO(
        "[GGML-QWEN-FULL] device expert arena=%.2f GiB; %d resident layer(s) "
        "in candidate range [%d,%d)%s%s split_layers=%zu\n",
        static_cast<double>(g.expert.capacity) /
            (1024.0 * 1024.0 * 1024.0),
        device_expert_resident_layer_count(g), block_start, block_end,
        g.device_expert_partial_gate_up ? " + next-layer gate/up" : "",
        split_active ? " + per-expert split" : "",
        g.split_gpu_expert_ids.size());
  }
  return true;
}

std::string expert_ptr_text(const void *ptr) {
  return std::to_string(reinterpret_cast<uintptr_t>(ptr));
}

// Phase-2 siblings can borrow an expert buffer through another borrower
// (for example, a staged child borrows from Prefix, which itself can borrow
// the retained decode arena). Always allocate and look up addresses in the
// graph that owns the buffer and its bump cursor/map. Using only the direct
// source graph lets a borrower with expert.used==0 allocate a new
// payload tensor over weights already stored near the start of the shared
// arena.
FullGraph *expert_arena_owner(FullGraph &g) {
  FullGraph *owner = &g;
  for (int depth = 0; owner->expert_source_graph; ++depth) {
    if (depth >= 16 || owner->expert_source_graph == owner) {
      LOG_ERROR("[GGML-QWEN-FULL-ARENA-BIND-FAIL] cyclic/deep expert arena "
                "owner chain graph=%p\n",
                static_cast<void *>(&g));
      return nullptr;
    }
    owner = owner->expert_source_graph;
  }
  return owner;
}

std::string device_expert_single_key(const void *host, size_t bytes,
                                    const std::string &source_tag) {
  // VRAM-only model layers are staged from GGUF into a temporary host buffer
  // for their first H2D copy. That staging address changes on every graph
  // build, while the model tensor and its destination arena slot do not.
  // Prefer the stable model tensor name whenever the caller has one so a
  // later graph can find and reuse the committed VRAM slot before rereading
  // the tensor from disk. Keep pointer identity for anonymous callers.
  if (!source_tag.empty())
    return "named:" + source_tag + ":" + std::to_string(bytes);
  return "single:" + expert_ptr_text(host) + ":" +
         std::to_string(bytes);
}

void *alloc_registered_host(FullGraph &g, size_t bytes) {
  void *ptr = alloc_host(g, bytes);
  if (!ptr || bytes == 0)
    return nullptr;
  const cudaError_t err =
      cudaHostRegister(ptr, bytes, cudaHostRegisterPortable);
  if (err != cudaSuccess) {
    cudaGetLastError();
    LOG_WARN("[QWEN4EXP-HOST-NODE] pinned host registration declined for "
             "%zu bytes: %s\n",
             bytes, cudaGetErrorString(err));
    return nullptr;
  }
  g.owned.back().pinned = true;
  g.host_pinned_bytes += bytes;
  return ptr;
}

} // namespace qwen4exp_graph

#endif // ZEROMOE_USE_GGML_CUDA
