// GLM5Next full graph: Partial-layer expert placement: VRAM reference experts, hot1 (GPU1)
// assignment, dynamic upload worker, route tap.
//
// Split out of the former single-file ggml_glm5next_graph.cpp; shared
// declarations live in ggml_glm5next_graph_impl.hpp.

#include <future>
#include "ggml_glm5next_graph_impl.hpp"

#if defined(MOESTREAM_USE_GGML_CUDA)

using namespace glm5next_graph;

namespace glm5next_graph {

bool configure_vram_reference_experts(FullGraph &g) {
  if (!g.topology || !g.engine ||
      !glm5next_graph::glm_family(g.engine->family_ops.family))
    return false;
  const auto &c = g.engine->model_file.architecture_config;
  if (c.n_experts <= 0 || c.n_layers <= c.first_dense) {
    LOG_WARN("[GGML-GLM5NEXT-CACHE] VRAM reference mode needs the compact, "
             "layer-partitioned expert cache geometry\n");
    return false;
  }
  if (!g.topology->expert_compact_layout)
    LOG_WARN("[GGML-GLM5NEXT-CACHE] cache is not layer-partitioned; using "
             "DDR references only for the graph hot/cold split\n");

  g.partial_experts.assign((size_t)c.n_layers, {});
  size_t aliased_bytes = 0;
  size_t hot_records = 0;
  std::lock_guard<std::recursive_mutex> lock(g.topology->cache_mutex);
  for (int layer = c.first_dense; layer < c.n_layers; ++layer) {
    const size_t raw_stride = g.engine->compute_expert_slab_bytes(layer);
    if (raw_stride == 0)
      continue;
    // Cache slots are padded (4 KiB, or row alignment inside the full-layer
    // prefix), so the physical slot stride is the slot's own size, not the
    // raw slab size.
    size_t stride = raw_stride;
    for (const ExpertLocation &loc : g.topology->expert_vram_cache) {
      if (g.topology->expert_compact_layout && loc.pool_layer == layer &&
          loc.gpu_id == g.device && loc.device_ptr && loc.size >= raw_stride) {
        stride = loc.size;
        break;
      }
    }
    PartialExpertLayer &p = g.partial_experts[(size_t)layer];
    p.vram_slot_by_expert.assign((size_t)c.n_experts, -1);
    p.vram_slot_stride = stride;

    size_t first_offset = SIZE_MAX;
    size_t slot_count = 0;
    for (const ExpertLocation &loc : g.topology->expert_vram_cache) {
      if (g.topology->expert_compact_layout && loc.pool_layer == layer &&
          loc.gpu_id == g.device &&
          loc.device_ptr && loc.size == stride) {
        first_offset = std::min(first_offset, loc.offset);
        ++slot_count;
      }
    }
    if (slot_count > 0) {
      bool region_valid =
          first_offset != SIZE_MAX && slot_count <= SIZE_MAX / stride &&
          first_offset <= SIZE_MAX - slot_count * stride;
      if (region_valid) {
        p.vram_layer_base = nullptr;
        p.vram_layer_slots = slot_count;
        size_t expected_offset = first_offset;
        for (const ExpertLocation &loc : g.topology->expert_vram_cache) {
          if (!g.topology->expert_compact_layout || loc.pool_layer != layer ||
              loc.gpu_id != g.device || !loc.device_ptr || loc.size != stride)
            continue;
          if (loc.offset < first_offset ||
              (loc.offset - first_offset) % stride != 0 ||
              loc.offset >= first_offset + slot_count * stride ||
              loc.offset > SIZE_MAX - stride) {
            region_valid = false;
            break;
          }
          if (!p.vram_layer_base || loc.offset == first_offset) {
            const size_t delta = loc.offset - first_offset;
            const uintptr_t address =
                reinterpret_cast<uintptr_t>(loc.device_ptr);
            if (address < delta) {
              region_valid = false;
              break;
            }
            p.vram_layer_base = reinterpret_cast<uint8_t *>(address - delta);
          }
          expected_offset = std::max(expected_offset, loc.offset + stride);
        }
        region_valid = region_valid && p.vram_layer_base &&
                       expected_offset == first_offset + slot_count * stride;
      }
      if (!region_valid) {
        LOG_WARN("[GGML-GLM5NEXT-CACHE] invalid VRAM cache geometry for "
                 "layer=%d; graph will leave its experts in the cold DDR "
                 "path\n", layer);
        p.vram_layer_base = nullptr;
        p.vram_layer_slots = 0;
        slot_count = 0;
      }
    }

    for (const ExpertLocation &loc : g.topology->expert_vram_cache) {
      if (!g.topology->expert_compact_layout || loc.pool_layer != layer ||
          loc.gpu_id != g.device ||
          loc.layer_idx != layer || loc.expert_id < 0 ||
          loc.expert_id >= c.n_experts || !loc.upload_ready ||
          !loc.device_ptr || loc.size != stride || !p.vram_layer_base)
        continue;
      const size_t delta = loc.offset - first_offset;
      if (delta % stride != 0 || delta / stride >= slot_count) {
        LOG_WARN("[GGML-GLM5NEXT-CACHE] invalid pinned slot map for "
                 "layer=%d expert=%d; leaving it on the cold DDR path\n",
                 layer, loc.expert_id);
        continue;
      }
      const int expert = loc.expert_id;
      if (p.vram_slot_by_expert[(size_t)expert] >= 0)
        continue;
      p.vram_slot_by_expert[(size_t)expert] = (int32_t)(delta / stride);
      p.hot.push_back(expert);
      g.vram_expert_holds.push_back({layer, expert, loc.device_ptr});
      ++hot_records;
      aliased_bytes += stride;
    }

    if (moestream::graphcfg::env("MOESTREAM_GLM5NEXT_DEBUG_PARTIAL") && layer < 8) {
      size_t n_pool = 0, n_ready = 0, n_ptr = 0, n_size = 0, n_gpu = 0;
      for (const ExpertLocation &l2 : g.topology->expert_vram_cache) {
        if (l2.pool_layer != layer) continue;
        ++n_pool; n_ready += l2.upload_ready; n_ptr += l2.device_ptr != nullptr;
        n_size += l2.size == stride; n_gpu += l2.gpu_id == g.device;
      }
      LOG_WARN("[ALIAS-DEBUG] layer=%d compact=%d stride=%zu slot_count=%zu "
               "base=%d pool=%zu ready=%zu ptr=%zu size_ok=%zu gpu_ok=%zu hot=%zu\n",
               layer, (int)g.topology->expert_compact_layout, stride, slot_count,
               p.vram_layer_base != nullptr, n_pool, n_ready, n_ptr, n_size, n_gpu,
               p.hot.size());
    }
    std::vector<uint8_t> is_hot((size_t)c.n_experts, 0);
    for (int expert : p.hot)
      is_hot[(size_t)expert] = 1;
    for (int expert = 0; expert < c.n_experts; ++expert)
      if (!is_hot[(size_t)expert])
        p.cold.push_back(expert);
    p.slot_last_use.assign(p.hot.size(), 0);
    p.slot_in_flight.assign(p.hot.size(), 0);
    p.miss_count.assign((size_t)c.n_experts, 0);
    p.heat.assign((size_t)c.n_experts, 0.0f);
    p.heat_at.assign((size_t)c.n_experts, 0);
    p.expert_in_flight.assign((size_t)c.n_experts, 0);
  }

  g.partial_expert_residency = true;
  g.vram_reference_experts = true;
  g.dynamic_expert_cache = false;
  g.partial_sources_compacted = false;
  g.partial_hot_bytes = 0; // aliased memory is not a graph allocation
  g.partial_cold_bytes = 0;
  g.partial_hot_records = hot_records;
  g.vram_reference_bytes = aliased_bytes;
  g.partial_usage_file = "<resident-vram-cache>";
  LOG_INFO("[GGML-GLM5NEXT-CACHE] expert storage aliases: %zu resident VRAM "
           "records (%.2f GiB referenced, not copied); DDR cold experts "
           "use existing graph-RAM slabs where available\n",
           hot_records,
           (double)aliased_bytes / (1024.0 * 1024.0 * 1024.0));
  return true;
}

bool configure_partial_experts_base(FullGraph &g) {
  if (configure_vram_reference_experts(g))
    return true;
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
    p.slot_last_use.assign(p.hot.size(), 0);
    p.slot_in_flight.assign(p.hot.size(), 0);
    p.miss_count.assign((size_t)c.n_experts, 0);
    p.heat.assign((size_t)c.n_experts, 0.0f);
    p.heat_at.assign((size_t)c.n_experts, 0);
    p.expert_in_flight.assign((size_t)c.n_experts, 0);
    for (size_t slot = 0; slot < p.hot.size(); ++slot) {
      const int expert = p.hot[slot];
      const auto usage = std::find_if(
          by_layer[(size_t)l].begin(), by_layer[(size_t)l].end(),
          [expert](const StreamingEngine::UsageEntry &entry) {
            return entry.expert == expert;
          });
      if (usage != by_layer[(size_t)l].end()) {
        const size_t rank = (size_t)std::distance(
            by_layer[(size_t)l].begin(), usage);
        const uint64_t initial =
            by_layer[(size_t)l].size() - std::min(
                rank, by_layer[(size_t)l].size());
        p.slot_last_use[slot] = (int)std::min<uint64_t>(initial, INT_MAX);
        g.dynamic_clock = std::max<uint64_t>(
            g.dynamic_clock, by_layer[(size_t)l].size() + 1);
      }
    }
    cold_bytes += record_bytes[(size_t)l] * p.cold.size();
  }
  g.partial_expert_residency = true;
  g.dynamic_expert_cache = dynamic_cache_enabled();
  g.partial_hot_bytes = used;
  g.partial_cold_bytes = cold_bytes;
  g.partial_hot_records = hot_records;
  g.partial_usage_file = partial_usage_path(g);
  LOG_INFO("[GGML-FULL] partial resident set: %.2f GiB hot, %.2f GiB cold, "
           "%zu hot expert records across %d layers, usage=%s%s\n",
           (double)used / (1024.0 * 1024.0 * 1024.0),
           (double)cold_bytes / (1024.0 * 1024.0 * 1024.0), hot_records,
           c.n_layers - c.first_dense, g.partial_usage_file.c_str(),
           g.dynamic_expert_cache ? ", dynamic LRU enabled" : "");
  return true;
}

// Same hot/cold split for every caller (boot candidate, prompt graph, decode graph) so their layouts compare equal.
bool configure_partial_experts(FullGraph &g) {
  if (!configure_partial_experts_base(g))
    return false;
  if (g.partial_expert_residency && ggml_shared_backend(second_gpu_device(g.device))) {
    const char *gh = moestream::graphcfg::env("MOESTREAM_GLM5NEXT_GPU1_HOT_GIB");
    if (gh && gh[0])
      assign_gpu1_hot(g, std::atof(gh));
  }
  return true;
}

ggml_tensor *make_vram_ref_weight(FullGraph &g, ggml_context *ctx, int layer,
                                  const char *stem) {
  const PartialExpertLayer *p = partial_layer(g, layer);
  if (!p || p->hot.empty() || !p->vram_layer_base ||
      p->vram_layer_slots == 0 || p->vram_slot_stride == 0 ||
      !g.expert.buffer)
    return nullptr;
  const std::string name = "blk." + std::to_string(layer) + "." + stem;
  const GGUFTensor *meta = model_tensor(*g.engine, name);
  const size_t n_experts =
      (size_t)g.engine->model_file.architecture_config.n_experts;
  if (!meta || meta->shape.size() != 3 || n_experts == 0 ||
      meta->size_bytes % n_experts != 0)
    return nullptr;
  size_t offset = 0;
  if (std::strcmp(stem, "ffn_up_exps.weight") == 0) {
    const GGUFTensor *gate = model_tensor(
        *g.engine, "blk." + std::to_string(layer) + ".ffn_gate_exps.weight");
    if (!gate || gate->size_bytes % n_experts != 0)
      return nullptr;
    offset = (size_t)gate->size_bytes / n_experts;
  } else if (std::strcmp(stem, "ffn_down_exps.weight") == 0) {
    const GGUFTensor *gate = model_tensor(
        *g.engine, "blk." + std::to_string(layer) + ".ffn_gate_exps.weight");
    const GGUFTensor *up = model_tensor(
        *g.engine, "blk." + std::to_string(layer) + ".ffn_up_exps.weight");
    if (!gate || !up || gate->size_bytes % n_experts != 0 ||
        up->size_bytes % n_experts != 0)
      return nullptr;
    const size_t gate_bytes = (size_t)gate->size_bytes / n_experts;
    const size_t up_bytes = (size_t)up->size_bytes / n_experts;
    if (gate_bytes > SIZE_MAX - up_bytes)
      return nullptr;
    offset = gate_bytes + up_bytes;
  } else if (std::strcmp(stem, "ffn_gate_exps.weight") != 0) {
    return nullptr;
  }
  const size_t per_expert_bytes = (size_t)meta->size_bytes / n_experts;
  if (offset > p->vram_slot_stride ||
      per_expert_bytes > p->vram_slot_stride - offset ||
      p->vram_layer_slots > SIZE_MAX / p->vram_slot_stride)
    return nullptr;
  std::vector<int64_t> shape;
  for (uint64_t dim : meta->shape)
    shape.push_back((int64_t)dim);
  shape[2] = (int64_t)p->vram_layer_slots;
  ggml_tensor *t = new_tensor(ctx, (ggml_type)meta->type, shape);
  if (!t)
    return nullptr;
  // ggml-cuda mmvq derives the per-expert offset as nb[2]/type_size in block
  // units, so a stride that is not a whole number of rows silently misaddresses
  // every nonzero slot (NaN decode output). Refuse the alias instead.
  // Only nb[2] must be a whole number of rows; the offset just positions the
  // base pointer inside the slot (down starts after gate+up, whose row size
  // differs), so it only needs the block struct's 2-byte alignment.
  if (p->vram_slot_stride % ggml_row_size(t->type, t->ne[0]) != 0 ||
      offset % 2 != 0) {
    static bool warned = false;
    if (!warned) {
      warned = true;
      LOG_STREAM_WARN() << "[GLM5NEXT-GRAPH] (stride=" << p->vram_slot_stride
                        << " offset=" << offset << " row_size="
                        << ggml_row_size(t->type, t->ne[0]) << " type_size="
                        << ggml_type_size(t->type) << " type="
                        << ggml_type_name(t->type) << ")\n";
      LOG_STREAM_WARN() << "[GLM5NEXT-GRAPH] VRAM slot stride "
                        << p->vram_slot_stride << " is not row-aligned for "
                        << name << "; set vram_cache_full_layer_prefix to "
                        << "cover all MoE layers. Not aliasing resident slots.\n";
    }
    return nullptr;
  }
  t->nb[2] = p->vram_slot_stride;
  t->nb[3] = t->nb[2] * (size_t)t->ne[2];
  t->data = p->vram_layer_base + offset;
  // The buffer type tags this tensor for the graph CUDA backend. The actual
  // data pointer remains owned by GpuTopology's expert cache pool.
  t->buffer = g.expert.buffer;
  ggml_set_name(t, (name + ".vram_cache_ref").c_str());
  return t;
}

bool prepare_partial_hot(FullGraph &g, ggml_context *ctx, int layer) {
  PartialExpertLayer *p = partial_layer(g, layer);
  if (!p || p->hot.empty())
    return true;
  if (p->hot_down || p->hot_gate_up || p->hot_gate)
    return true;
  if (g.vram_reference_experts) {
    p->hot_gate = make_vram_ref_weight(g, ctx, layer,
                                       "ffn_gate_exps.weight");
    p->hot_up = make_vram_ref_weight(g, ctx, layer,
                                     "ffn_up_exps.weight");
    p->hot_down = make_vram_ref_weight(g, ctx, layer,
                                       "ffn_down_exps.weight");
    return p->hot_gate && p->hot_up && p->hot_down;
  }
  PackedGateUpMeta packed;
  // The mmap-backed dynamic source intentionally retains gate and up as
  // separate read-only file views.  Keep the hot slots in the same layout so
  // asynchronous promotion can copy each projection directly into its slot;
  // a packed hot tensor cannot be paired with separate cold source tensors.
  if (g.dynamic_expert_cache && g.expert_source_mmap) {
    p->hot_gate = make_hot_weight(g, ctx, layer, "ffn_gate_exps.weight");
    p->hot_up = make_hot_weight(g, ctx, layer, "ffn_up_exps.weight");
    p->hot_down = make_hot_weight(g, ctx, layer, "ffn_down_exps.weight");
    return p->hot_gate && p->hot_up && p->hot_down;
  }
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

static bool layer_experts_take_mmq(FullGraph &g, int layer);

ggml_tensor *partial_map_hot1(FullGraph &g, ggml_context *ctx, int layer,
                              bool mask) {
  PartialExpertLayer *p = partial_layer(g, layer);
  if (!p)
    return nullptr;
  ggml_tensor *&slot = mask ? p->hot1_mask : p->hot1_map;
  if (slot)
    return slot;
  const size_t n = (size_t)std::max(0, g.engine->model_file.architecture_config.n_experts);
  ggml_tensor *t = ggml_new_tensor_3d(ctx, mask ? GGML_TYPE_F32 : GGML_TYPE_I32, 1,
                                      (int64_t)n, g.S);
  if (!t)
    return nullptr;
  std::vector<int32_t> index_of_expert(n, -1);
  std::vector<uint8_t> owned(n, 0);
  for (size_t i = 0; i < p->hot1.size(); ++i) {
    const int e = p->hot1[i];
    if (e >= 0 && (size_t)e < n) {
      index_of_expert[(size_t)e] = (int32_t)i;
      owned[(size_t)e] = 1;
    }
  }
  const moe_partial::ExpertBranchMap branch_map = moe_partial::ExpertBranchMap::build(
      index_of_expert, g.S, g.decode_mode && g.S == 1, owned,
      g.S > 1 && layer_experts_take_mmq(g, layer) ? moe_partial::SentinelPolicy::AutoBatched
                                                  : moe_partial::SentinelPolicy::Auto);
  const void *data = mask ? (const void *)branch_map.route_mask().data()
                          : (const void *)branch_map.route_ids().data();
  if (!arena_alloc(g.dense.buffer, g.dense.base, g.dense.capacity, g.dense.used,
                   t, data, ggml_nbytes(t)))
    return nullptr;
  ggml_set_name(t, (std::string("full_graph_hot1_") + (mask ? "mask_" : "map_") +
                    std::to_string(layer)).c_str());
  slot = t;
  return t;
}

ggml_tensor *hot1_weight(FullGraph &g, ggml_context *ctx, int layer,
                         const char *stem, int plane) {
  PartialExpertLayer *p = partial_layer(g, layer);
  const GGUFTensor *m = model_tensor(*g.engine, "blk." + std::to_string(layer) + "." + stem);
  const size_t n_experts = (size_t)g.engine->model_file.architecture_config.n_experts;
  if (!p || p->hot1.empty() || !m || !type_ok(m->type) || m->shape.size() != 3 ||
      n_experts == 0 || m->size_bytes % n_experts != 0)
    return nullptr;
  const size_t per_expert = (size_t)m->size_bytes / n_experts;
  std::vector<int64_t> shape;
  for (uint64_t d : m->shape)
    shape.push_back((int64_t)d);
  shape[2] = (int64_t)p->hot1.size();
  ggml_tensor *t = new_tensor(ctx, (ggml_type)m->type, shape);
  if (!t || ggml_nbytes(t) != per_expert * p->hot1.size())
    return nullptr;
  const int key = layer * 3 + plane;
  ggml_backend_buffer_t buf = nullptr;
  auto it = g_hot1_buffers.find(key);
  const bool fresh = it == g_hot1_buffers.end();
  if (fresh) {
    // MOESTREAM_GLM5NEXT_GPU1_HOT_ON_GPU0=1 is a verification aid: same branch, buffers on the main GPU, so its
    // logits can be compared against the GPU1 placement with identical kernels.
    static const bool on_gpu0 = moestream::graphcfg::env("MOESTREAM_GLM5NEXT_GPU1_HOT_ON_GPU0") != nullptr;
    buf = ggml_backend_buft_alloc_buffer(
        ggml_backend_get_default_buffer_type(on_gpu0 ? g.cuda : g.gpu1), ggml_nbytes(t) + 4096);
    if (!buf)
      return nullptr;
    ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    g_hot1_buffers[key] = buf;
  } else {
    buf = it->second;
  }
  if (ggml_backend_tensor_alloc(buf, t, ggml_backend_buffer_get_base(buf)) != GGML_STATUS_SUCCESS)
    return nullptr;
  if (fresh) {
    // Read the hot experts 16 at a time (the file is on NVMe: one blocking pread per expert ran at ~0.7 GB/s and made
    // the first decode token take ~20 s), then upload them in order.
    const int fd = g.engine->model_file.fd_for_tensor(*m);
    // Experts already in the DDR pool (seeded for batched prefill) are copied straight from it: no SSD read.
    std::vector<size_t> from_file;
    OffloadState *off = g_offload.get();
    for (size_t j = 0; j < p->hot1.size(); ++j) {
      const int e = p->hot1[j];
      if (off && off->arena && layer >= off->first && off->arena->plane_bytes(layer, plane) == per_expert &&
          off->residency->ready({layer, e})) {
        ggml_backend_tensor_set(t, off->arena->ptr(layer, e, plane), j * per_expert, per_expert);
      } else {
        from_file.push_back(j);
      }
    }
    constexpr size_t kBatch = 16;
    std::vector<std::vector<char>> stage(kBatch, std::vector<char>(per_expert));
    for (size_t base = 0; base < from_file.size(); base += kBatch) {
      const size_t n = std::min(kBatch, from_file.size() - base);
      std::vector<std::future<ssize_t>> reads;
      for (size_t j = 0; j < n; ++j)
        reads.push_back(std::async(std::launch::async, [&, j] {
          return ::pread(fd, stage[j].data(), per_expert,
                         (off_t)(m->offset + (uint64_t)p->hot1[from_file[base + j]] * per_expert));
        }));
      for (size_t j = 0; j < n; ++j) {
        if (reads[j].get() != (ssize_t)per_expert) {
          LOG_ERROR("[GGML-GLM5NEXT-GPU1-HOT] short read layer %d expert %d\n", layer, p->hot1[from_file[base + j]]);
          for (size_t r = j + 1; r < n; ++r) reads[r].wait();
          return nullptr;
        }
      }
      for (size_t j = 0; j < n; ++j) {
        ggml_backend_tensor_set(t, stage[j].data(), from_file[base + j] * per_expert, per_expert);
        ::posix_fadvise(fd, (off_t)(m->offset + (uint64_t)p->hot1[from_file[base + j]] * per_expert), (off_t)per_expert,
                        POSIX_FADV_DONTNEED);
      }
    }
    g.gpu1_bytes += ggml_nbytes(t);
  }
  ggml_set_name(t, (std::string("blk.") + std::to_string(layer) + "." + stem + ".gpu1_hot").c_str());
  return t;
}

bool prepare_partial_hot1(FullGraph &g, ggml_context *ctx, int layer) {
  PartialExpertLayer *p = partial_layer(g, layer);
  if (!p || p->hot1.empty() || !(g.decode_mode && g.gpu1) || p->hot1_down)
    return true;
  p->hot1_gate = hot1_weight(g, ctx, layer, "ffn_gate_exps.weight", 0);
  p->hot1_up = hot1_weight(g, ctx, layer, "ffn_up_exps.weight", 1);
  p->hot1_down = hot1_weight(g, ctx, layer, "ffn_down_exps.weight", 2);
  return p->hot1_gate && p->hot1_up && p->hot1_down;
}

// Upload every layer's hot-set planes into the persistent GPU1 buffers at boot, so the first decode graph build finds
// them resident instead of reading them inside the first token.
bool preload_partial_hot1(FullGraph &g) {
  if (!g.partial_expert_residency)
    return true;
  auto *backend = static_cast<ggml_backend_t>(ggml_shared_backend(second_gpu_device(g.device)));
  if (!backend)
    return true;
  const auto t0 = std::chrono::steady_clock::now();
  size_t layers = 0;
  for (size_t layer = 0; layer < g.partial_experts.size(); ++layer) {
    if (g.partial_experts[layer].hot1.empty())
      continue;
    ggml_init_params ip = {};
    ip.mem_size = ggml_tensor_overhead() * 8;
    ip.no_alloc = true;
    ggml_context *ctx = ggml_init(ip);
    if (!ctx)
      return false;
    ggml_backend_t saved = g.gpu1;
    g.gpu1 = backend;
    const bool ok = hot1_weight(g, ctx, (int)layer, "ffn_gate_exps.weight", 0) &&
                    hot1_weight(g, ctx, (int)layer, "ffn_up_exps.weight", 1) &&
                    hot1_weight(g, ctx, (int)layer, "ffn_down_exps.weight", 2);
    g.gpu1 = saved;
    ggml_free(ctx);
    if (!ok)
      return false;
    ++layers;
  }
  if (layers)
    LOG_INFO("[GGML-GLM5NEXT-GPU1-HOT] preloaded %zu layers (%.2f GiB) to device %d at boot in %.1f s\n", layers,
             (double)g.gpu1_bytes / (1024.0 * 1024.0 * 1024.0), second_gpu_device(g.device),
             std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
  return true;
}

// Move the next-hottest cold experts (by usage) into the GPU1 hot set until `gib` of expert bytes is reached.
void assign_gpu1_hot(FullGraph &g, double gib) {
  if (gib <= 0.0 || g.partial_experts.empty())
    return;
  const auto &c = g.engine->model_file.architecture_config;
  const char *uenv = moestream::graphcfg::env("MOESTREAM_GLM5NEXT_GPU1_HOT_USAGE");
  const std::string usage_path = uenv && uenv[0]
                                     ? std::string(uenv)
                                     : offload_env("MOESTREAM_GLM5NEXT_OFFLOAD_USAGE");
  auto [entries, max_count] = StreamingEngine::load_usage(usage_path);
  (void)max_count;
  std::vector<std::vector<uint8_t>> is_cold((size_t)c.n_layers);
  std::vector<size_t> bytes((size_t)c.n_layers, 0);
  for (int l = c.first_dense; l < c.n_layers; ++l) {
    PartialExpertLayer &p = g.partial_experts[(size_t)l];
    is_cold[(size_t)l].assign((size_t)c.n_experts, 0);
    for (int e : p.cold)
      if (e >= 0 && e < c.n_experts)
        is_cold[(size_t)l][(size_t)e] = 1;
    for (const char *stem : {"ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"}) {
      const GGUFTensor *m = model_tensor(*g.engine, "blk." + std::to_string(l) + "." + stem);
      if (m && c.n_experts > 0)
        bytes[(size_t)l] += (size_t)m->size_bytes / (size_t)c.n_experts;
    }
  }
  // MOESTREAM_GLM5NEXT_GPU1_MIN_LAYER=L: the second GPU only holds hot experts of layers >= L. Pair it with a primary-GPU
  // ranking whose counts are zeroed for those layers (and pin_global_floor_per_layer = 0), so each layer's hot set lives on one
  // GPU and its trunk needs no cross-GPU hop. Measured +3% on held-out prompts (docs/ssd-offload/13-...).
  const int gpu1_min_layer = moestream::graphcfg::atoi_or("MOESTREAM_GLM5NEXT_GPU1_MIN_LAYER", 0);
  std::vector<StreamingEngine::UsageEntry> cand;
  for (const auto &e : entries)
    if (e.layer >= std::max(c.first_dense, gpu1_min_layer) && e.layer < c.n_layers && e.expert >= 0 &&
        e.expert < c.n_experts && e.count > 0 && is_cold[(size_t)e.layer][(size_t)e.expert])
      cand.push_back(e);
  std::sort(cand.begin(), cand.end(), [](const auto &a, const auto &b) {
    if (a.count != b.count)
      return a.count > b.count;
    if (a.layer != b.layer)
      return a.layer < b.layer;
    return a.expert < b.expert;
  });
  const size_t budget = (size_t)(gib * 1024.0 * 1024.0 * 1024.0);
  size_t used = 0, taken = 0;
  for (const auto &e : cand) {
    if (used + bytes[(size_t)e.layer] > budget)
      continue;
    PartialExpertLayer &p = g.partial_experts[(size_t)e.layer];
    p.hot1.push_back(e.expert);
    used += bytes[(size_t)e.layer];
    ++taken;
    is_cold[(size_t)e.layer][(size_t)e.expert] = 0;
  }
  for (int l = c.first_dense; l < c.n_layers; ++l) {
    PartialExpertLayer &p = g.partial_experts[(size_t)l];
    if (p.hot1.empty())
      continue;
    std::sort(p.hot1.begin(), p.hot1.end());
    p.cold.erase(std::remove_if(p.cold.begin(), p.cold.end(),
                                [&](int e) { return !is_cold[(size_t)l][(size_t)e]; }),
                 p.cold.end());
  }
  if (moestream::graphcfg::env("MOESTREAM_GLM5NEXT_OFFLOAD_DEBUG")) {
    for (int l : {3, 4}) {
      std::string a, b;
      for (int e : g.partial_experts[(size_t)l].hot1) a += std::to_string(e) + ",";
      for (int e : g.partial_experts[(size_t)l].hot) b += std::to_string(e) + ",";
      LOG_INFO("[HOT1-DEBUG] layer %d hot1=%s\n[HOT1-DEBUG] layer %d hot=%s cold=%zu\n", l, a.c_str(), l, b.c_str(),
               g.partial_experts[(size_t)l].cold.size());
    }
  }
  LOG_INFO("[GGML-GLM5NEXT-GPU1-HOT] %zu experts (%.2f GiB) assigned to GPU1\n", taken,
           (double)used / (1024.0 * 1024.0 * 1024.0));
}

// Batched (S > 1) negative-id skip is guaranteed only for the quantized expert types that take
// ggml-cuda's MMQ path (patches/ggml/0006). IQ1_M and float experts use other batched paths, so
// those layers keep the valid-dummy-id route for prefill.
static bool layer_experts_take_mmq(FullGraph &g, int layer) {
  for (const char *stem :
       {"ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"}) {
    const GGUFTensor *t = model_tensor(
        *g.engine, "blk." + std::to_string(layer) + "." + stem);
    if (!t)
      return false;
    switch (t->type) {
    case GGUFType::Q4_0: case GGUFType::Q4_1: case GGUFType::Q5_0:
    case GGUFType::Q5_1: case GGUFType::Q8_0: case GGUFType::Q2_K:
    case GGUFType::Q3_K: case GGUFType::Q4_K: case GGUFType::Q5_K:
    case GGUFType::Q6_K: case GGUFType::IQ2_XXS: case GGUFType::IQ2_XS:
    case GGUFType::IQ2_S: case GGUFType::IQ3_XXS: case GGUFType::IQ3_S:
    case GGUFType::IQ4_NL: case GGUFType::IQ4_XS: case GGUFType::IQ1_S:
      break;
    default:
      return false;
    }
  }
  return true;
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

  // How each expert is addressed inside THIS branch's weight tensor (-1 = the
  // branch does not hold it).  The sentinel for routes it does not own, the
  // 0/1 ownership mask and the per-token repetition all come from
  // moe_partial::ExpertBranchMap, so every model inherits the same default
  // (negative-id skip for single-token decode on a capable ggml, valid dummy id
  // otherwise) -- see include/moe_partial_route.hpp.
  std::vector<int32_t> index_of_expert(n, -1);
  if (g.dynamic_expert_cache && !hot) {
    for (size_t expert = 0; expert < n; ++expert)
      if (!is_hot[expert])
        index_of_expert[expert] = (int32_t)expert;
  } else if (g.vram_reference_experts) {
    if (hot) {
      for (int expert : p->hot) {
        if (expert < 0 || (size_t)expert >= n ||
            (size_t)expert >= p->vram_slot_by_expert.size())
          continue;
        const int32_t physical_slot = p->vram_slot_by_expert[(size_t)expert];
        if (physical_slot < 0)
          return nullptr;
        index_of_expert[(size_t)expert] = physical_slot;
      }
    } else {
      int32_t cold_slot = 0;
      const bool full_ddr_view =
          offload_layer(layer) || partial_layer_ram_ready(g, layer);
      for (int expert : p->cold) {
        if (expert < 0 || (size_t)expert >= n)
          return nullptr;
        index_of_expert[(size_t)expert] = full_ddr_view ? expert : cold_slot++;
      }
    }
  } else {
    const auto &ids = hot ? p->hot : p->cold;
    for (size_t i = 0; i < ids.size(); ++i) {
      if (ids[i] < 0 || (size_t)ids[i] >= n)
        continue;
      index_of_expert[(size_t)ids[i]] = (int32_t)i;
    }
  }
  std::vector<uint8_t> owned(n, 0);
  for (size_t e = 0; e < n; ++e)
    owned[e] = (is_hot[e] != 0) == hot ? 1 : 0;
  if (!hot && !p->hot1.empty()) {
    // hot1 experts left `cold`; decode serves them from GPU1, every other build keeps them here.
    const bool on_gpu1 = g.gpu1 != nullptr;
    for (int e : p->hot1) {
      if (e < 0 || (size_t)e >= n)
        continue;
      if (on_gpu1) {
        owned[(size_t)e] = 0;
      } else {
        index_of_expert[(size_t)e] = e; // full-arena identity view
        owned[(size_t)e] = 1;
      }
    }
  }

  // MOESTREAM_GLM5NEXT_PARTIAL_ZERO_SENTINEL=1 forces the valid dummy id (A/B tests).
  static const bool force_dummy =
      moestream::graphcfg::env("MOESTREAM_GLM5NEXT_PARTIAL_ZERO_SENTINEL") != nullptr;
  // Batched MMQ negative-id skipping regressed Ling's short-prompt prefill
  // (13.2 s vs 1.4 s for the same two-token input). Keep batched Ling routes
  // on the valid dummy id; the single-token decode path still uses MMVQ's
  // negative-id skip through SentinelPolicy::Auto.
  const bool allow_batched_negative_skip =
      g.engine->family_ops.family != ModelFamily::BailingMoE3 &&
      layer_experts_take_mmq(g, layer);
  const moe_partial::ExpertBranchMap branch_map =
      moe_partial::ExpertBranchMap::build(
          index_of_expert, g.S, g.decode_mode && g.S == 1, owned,
          force_dummy ? moe_partial::SentinelPolicy::ForceDummy
          : allow_batched_negative_skip
              ? moe_partial::SentinelPolicy::AutoBatched
              : moe_partial::SentinelPolicy::Auto);
  static bool logged_inactive = false;
  if (!logged_inactive) {
    logged_inactive = true;
    LOG_INFO("[GGML-GLM5NEXT-FULL] partial-layer inactive route id=%d "
             "(mmid negative-skip %s%s)\n",
             (int)branch_map.inactive_id(),
             mmid_negative_skip_available() ? "available" : "unavailable",
             force_dummy ? ", forced dummy by env" : "");
  }
  const void *data = mask ? (const void *)branch_map.route_mask().data()
                          : (const void *)branch_map.route_ids().data();
  if (!arena_alloc(g.dense.buffer, g.dense.base, g.dense.capacity, g.dense.used,
                   t, data, ggml_nbytes(t)))
    return nullptr;
  ggml_set_name(t, (std::string("full_graph_") + (hot ? "hot_" : "cold_") +
                    (mask ? "mask_" : "map_") + std::to_string(layer))
                       .c_str());
  slot = t;
  return slot;
}

bool upload_partial_maps(FullGraph &g, PartialExpertLayer &p) {
  const auto &c = g.engine->model_file.architecture_config;
  const int layer = (int)(&p - g.partial_experts.data());
  if (c.n_experts <= 0 || g.S <= 0 || !p.hot_map || !p.hot_mask ||
      !p.cold_map || !p.cold_mask)
    return false;
  const size_t n = (size_t)c.n_experts;
  const size_t count = n * (size_t)g.S;
  std::vector<int32_t> hot_slots(n, -1), cold_slots(n, -1);
  for (size_t slot = 0; slot < p.hot.size(); ++slot) {
    const int expert = p.hot[slot];
    if (expert == -1 && g.dynamic_expert_cache)
      continue;
    if (expert < 0 || expert >= c.n_experts)
      return false;
    hot_slots[(size_t)expert] =
        g.vram_reference_experts &&
                (size_t)expert < p.vram_slot_by_expert.size()
            ? p.vram_slot_by_expert[(size_t)expert]
            : (int32_t)slot;
  }
  if (!g.dynamic_expert_cache) {
    for (size_t slot = 0; slot < p.cold.size(); ++slot) {
      const int expert = p.cold[slot];
      if (expert < 0 || expert >= c.n_experts ||
          hot_slots[(size_t)expert] >= 0)
        return false;
      cold_slots[(size_t)expert] =
          g.vram_reference_experts && partial_layer_ram_ready(g, layer)
              ? expert
              : (int32_t)slot;
    }
  }
  // Per-expert addressing for each branch: an expert is held by the hot branch
  // when it has a physical slot, otherwise by the cold branch.  The sentinel and
  // mask semantics come from moe_partial::HotColdSplit (include/moe_partial_route.hpp).
  std::vector<int32_t> hot_index(n, -1), cold_index(n, -1);
  for (size_t expert = 0; expert < n; ++expert) {
    const int32_t hs = hot_slots[expert];
    const int32_t cs = g.dynamic_expert_cache ? (int32_t)expert
                                             : cold_slots[expert];
    if (!g.dynamic_expert_cache && ((hs >= 0) == (cs >= 0)))
      return false;
    if (hs >= 0)
      hot_index[expert] = hs;
    else
      cold_index[expert] = cs;
  }
  moe_partial::HotColdSplit split;
  if (!moe_partial::HotColdSplit::build(hot_index, cold_index, g.S,
                                        g.S == 1, /*exact_partition=*/true,
                                        &split))
    return false;
  const std::vector<int32_t> &hot_map = split.hot().route_ids();
  const std::vector<int32_t> &cold_map = split.cold().route_ids();
  const std::vector<float> &hot_mask = split.hot().route_mask();
  const std::vector<float> &cold_mask = split.cold().route_mask();

  const auto set = [](ggml_tensor *tensor, const void *data, size_t bytes) {
    return tensor && tensor->data && ggml_nbytes(tensor) == bytes &&
           cudaMemcpy(tensor->data, data, bytes, cudaMemcpyHostToDevice) ==
               cudaSuccess;
  };
  if (!set(p.hot_map, hot_map.data(), count * sizeof(int32_t)) ||
      !set(p.hot_mask, hot_mask.data(), count * sizeof(float)) ||
      !set(p.cold_map, cold_map.data(), count * sizeof(int32_t)) ||
      !set(p.cold_mask, cold_mask.data(), count * sizeof(float))) {
    const cudaError_t err = cudaGetLastError();
    LOG_WARN("[GGML-GLM5NEXT-CACHE] route-map upload failed: %s\n",
             cudaGetErrorString(err));
    return false;
  }
  return true;
}

bool ensure_dynamic_upload_worker(FullGraph &g) {
  if (g.dynamic_upload)
    return true;
  size_t staging_capacity = 0;
  const auto add_layer = [&](const PartialExpertLayer &p) {
    size_t bytes = 0;
    const auto add_pair = [&](const ggml_tensor *hot,
                              const ggml_tensor *source) -> bool {
      if (!hot || !source || !hot->data || !source->data ||
          hot->nb[2] == 0 || hot->nb[2] != source->nb[2] ||
          hot->nb[2] > SIZE_MAX - bytes)
        return false;
      bytes += hot->nb[2];
      return true;
    };
    bool valid = false;
    if (p.hot_gate_up || p.cold_gate_up) {
      valid = add_pair(p.hot_gate_up, p.cold_gate_up) &&
              add_pair(p.hot_down, p.cold_down);
    } else {
      valid = add_pair(p.hot_gate, p.cold_gate) &&
              add_pair(p.hot_up, p.cold_up) &&
              add_pair(p.hot_down, p.cold_down);
    }
    if (valid)
      staging_capacity = std::max(staging_capacity, bytes);
  };
  for (int layer = 0; layer < (int)g.partial_experts.size(); ++layer)
    if (!g.partial_experts[(size_t)layer].hot.empty())
      add_layer(g.partial_experts[(size_t)layer]);
  if (staging_capacity == 0 ||
      !MemoryGuard::try_reserve_host_bytes(staging_capacity)) {
    LOG_WARN("[GGML-GLM5NEXT-CACHE] async upload staging admission refused "
             "(%.2f MiB)\n",
             (double)staging_capacity / (1024.0 * 1024.0));
    return false;
  }

  auto state = std::make_unique<DynamicUploadState>();
  state->host_reservation_bytes = staging_capacity;
  if (posix_memalign(&state->staging, 4096, staging_capacity) != 0 ||
      !state->staging) {
    MemoryGuard::release_host_bytes(state->host_reservation_bytes);
    state->host_reservation_bytes = 0;
    return false;
  }
  state->staging_capacity = staging_capacity;
  if (cudaSetDevice(g.device) != cudaSuccess ||
      cudaHostRegister(state->staging, state->staging_capacity,
                       cudaHostRegisterPortable) != cudaSuccess) {
    const cudaError_t err = cudaGetLastError();
    LOG_WARN("[GGML-GLM5NEXT-CACHE] pinned upload staging unavailable: %s\n",
             cudaGetErrorString(err));
    std::free(state->staging);
    MemoryGuard::release_host_bytes(state->host_reservation_bytes);
    return false;
  }
  if (cudaStreamCreateWithFlags(&state->stream, cudaStreamNonBlocking) !=
      cudaSuccess) {
    const cudaError_t err = cudaGetLastError();
    LOG_WARN("[GGML-GLM5NEXT-CACHE] upload stream creation failed: %s\n",
             cudaGetErrorString(err));
    cudaHostUnregister(state->staging);
    std::free(state->staging);
    MemoryGuard::release_host_bytes(state->host_reservation_bytes);
    return false;
  }

  DynamicUploadState *raw = state.get();
  const int device = g.device;
  try {
    state->worker = std::thread([raw, device]() {
      if (cudaSetDevice(device) != cudaSuccess) {
        std::lock_guard<std::mutex> lock(raw->mutex);
        raw->stopping = true;
        return;
      }
      for (;;) {
        DynamicUploadJob job;
        {
          std::unique_lock<std::mutex> lock(raw->mutex);
          raw->cv.wait(lock, [&]() {
            return raw->stopping || !raw->queue.empty();
          });
          if (raw->stopping)
            return;
          job = std::move(raw->queue.front());
          raw->queue.pop_front();
          raw->active = true;
        }

        const auto upload_start = std::chrono::steady_clock::now();
        bool success = true;
        size_t upload_bytes = 0;
        for (const DynamicUploadCopy &copy : job.copies) {
          if (!copy.source || !copy.destination || copy.bytes == 0 ||
              copy.staging_offset > raw->staging_capacity ||
              copy.bytes > raw->staging_capacity - copy.staging_offset ||
              copy.bytes > SIZE_MAX - upload_bytes) {
            success = false;
            break;
          }
          upload_bytes += copy.bytes;
          auto *stage = static_cast<uint8_t *>(raw->staging) +
                        copy.staging_offset;
          std::memcpy(stage, copy.source, copy.bytes);
          if (cudaMemcpyAsync(copy.destination, stage, copy.bytes,
                              cudaMemcpyHostToDevice, raw->stream) !=
              cudaSuccess) {
            success = false;
            break;
          }
        }
        // Some copies may already be queued when a later enqueue fails. The
        // staging buffer must not be reused until every submitted copy has
        // stopped reading from it, even on this error path.
        if (cudaStreamSynchronize(raw->stream) != cudaSuccess)
          success = false;
        if (!success) {
          const cudaError_t err = cudaGetLastError();
          LOG_WARN("[GGML-GLM5NEXT-CACHE] async expert upload failed "
                   "layer=%d expert=%d: %s\n",
                   job.layer, job.incoming_expert, cudaGetErrorString(err));
        }
        const double elapsed_ms = std::chrono::duration<double, std::milli>(
                                      std::chrono::steady_clock::now() -
                                      upload_start)
                                      .count();
        {
          std::lock_guard<std::mutex> lock(raw->mutex);
          raw->completed.push_back({job.layer, job.hot_slot,
                                    job.incoming_expert, success,
                                    upload_bytes, elapsed_ms});
          raw->active = false;
          if (!success)
            raw->stopping = true;
        }
      }
    });
  } catch (...) {
    cudaStreamDestroy(state->stream);
    cudaHostUnregister(state->staging);
    std::free(state->staging);
    MemoryGuard::release_host_bytes(state->host_reservation_bytes);
    return false;
  }
  LOG_INFO("[GGML-GLM5NEXT-CACHE] async H2D worker ready: staging=%.2f MiB "
           "device=%d\n",
           (double)staging_capacity / (1024.0 * 1024.0), g.device);
  g.dynamic_upload = std::move(state);
  return true;
}

bool make_dynamic_upload_job(PartialExpertLayer &p, int layer, int hot_slot,
                             int incoming_expert, int evicted_expert,
                             DynamicUploadJob *job) {
  if (!job || hot_slot < 0 || (size_t)hot_slot >= p.hot.size() ||
      incoming_expert < 0)
    return false;
  job->layer = layer;
  job->hot_slot = hot_slot;
  job->incoming_expert = incoming_expert;
  job->evicted_expert = evicted_expert;
  size_t offset = 0;
  const auto add_pair = [&](ggml_tensor *hot, ggml_tensor *source) -> bool {
    if (!hot || !source || !hot->data || !source->data ||
        hot->nb[2] == 0 || hot->nb[2] != source->nb[2] ||
        incoming_expert >= source->ne[2] || hot_slot >= hot->ne[2] ||
        (size_t)incoming_expert > SIZE_MAX / source->nb[2] ||
        (size_t)hot_slot > SIZE_MAX / hot->nb[2] ||
        hot->nb[2] > SIZE_MAX - offset)
      return false;
    const size_t bytes = hot->nb[2];
    const auto *src = static_cast<const uint8_t *>(source->data) +
                      (size_t)incoming_expert * source->nb[2];
    auto *dst = static_cast<uint8_t *>(hot->data) +
                (size_t)hot_slot * hot->nb[2];
    job->copies.push_back({src, dst, bytes, offset});
    offset += bytes;
    return true;
  };
  if (p.hot_gate_up || p.cold_gate_up) {
    return add_pair(p.hot_gate_up, p.cold_gate_up) &&
           add_pair(p.hot_down, p.cold_down);
  }
  return add_pair(p.hot_gate, p.cold_gate) &&
         add_pair(p.hot_up, p.cold_up) &&
         add_pair(p.hot_down, p.cold_down);
}

bool publish_dynamic_uploads(FullGraph &g) {
  if (!g.dynamic_upload)
    return true;
  std::vector<DynamicUploadResult> completed;
  {
    std::lock_guard<std::mutex> lock(g.dynamic_upload->mutex);
    completed.swap(g.dynamic_upload->completed);
  }
  for (const DynamicUploadResult &result : completed) {
    if (result.layer < 0 ||
        (size_t)result.layer >= g.partial_experts.size())
      continue;
    PartialExpertLayer &p = g.partial_experts[(size_t)result.layer];
    if (result.hot_slot < 0 || (size_t)result.hot_slot >= p.hot.size() ||
        (size_t)result.hot_slot >= p.slot_in_flight.size() ||
        result.incoming_expert < 0 ||
        (size_t)result.incoming_expert >= p.expert_in_flight.size()) {
      g.ready = false;
      return false;
    }
    p.slot_in_flight[(size_t)result.hot_slot] = 0;
    p.expert_in_flight[(size_t)result.incoming_expert] = 0;
    if (!result.success) {
      // The slot was invalidated before the worker started. Keep it empty and
      // let the unchanged full host source serve both the evicted and incoming
      // routes; it can be reused by a later upload.
      continue;
    }
    if (result.elapsed_ms > 0.0 && std::isfinite(result.elapsed_ms)) {
      g.dynamic_upload_ms_per_job =
          g.dynamic_upload_ms_per_job > 0.0
              ? 0.8 * g.dynamic_upload_ms_per_job + 0.2 * result.elapsed_ms
              : result.elapsed_ms;
    }
    if (p.hot[(size_t)result.hot_slot] != -1) {
      LOG_ERROR("[GGML-GLM5NEXT-CACHE] completed upload targeted a live slot "
                "layer=%d slot=%d\n",
                result.layer, result.hot_slot);
      g.ready = false;
      return false;
    }
    const auto cold = std::find(p.cold.begin(), p.cold.end(),
                                result.incoming_expert);
    if (cold == p.cold.end()) {
      LOG_ERROR("[GGML-GLM5NEXT-CACHE] completed expert missing from cold "
                "set layer=%d expert=%d\n",
                result.layer, result.incoming_expert);
      g.ready = false;
      return false;
    }
    p.hot[(size_t)result.hot_slot] = result.incoming_expert;
    p.cold.erase(cold);
    p.slot_last_use[(size_t)result.hot_slot] =
        (int)std::min<uint64_t>(g.dynamic_clock, INT_MAX);
    p.miss_count[(size_t)result.incoming_expert] = 0;
    if (!upload_partial_maps(g, p)) {
      g.ready = false;
      return false;
    }
    ++g.dynamic_promotions;
  }
  return true;
}

RouteTap &route_tap() {
  static RouteTap t = [] {
    RouteTap r;
    if (const char *e = moestream::graphcfg::env("MOESTREAM_GLM5NEXT_ROUTE_TAP"))
      r.path = e;
    return r;
  }();
  return t;
}

void route_tap_flush(RouteTap &t) {
  std::FILE *f = std::fopen(t.path.c_str(), "w");
  if (!f)
    return;
  for (size_t l = 0; l < t.counts.size(); ++l)
    for (size_t e = 0; e < t.counts[l].size(); ++e)
      if (t.counts[l][e])
        std::fprintf(f, "%zu %zu %u\n", l, e, t.counts[l][e]);
  std::fclose(f);
}

// MOESTREAM_GLM5NEXT_DYNAMIC_HEAT=theta (> 0): promote a cold expert into VRAM only when its decayed re-reference score (+1 per route,
// halved every DYNAMIC_HEAT_HALFLIFE tokens, default 30) is >= theta, instead of the cumulative "missed twice" rule. Admission by reuse:
// an upload costs about one CPU compute, so it only pays for experts that come back.
float dynamic_heat_threshold() {
  static const float v = [] {
    const char *e = std::getenv("MOESTREAM_GLM5NEXT_DYNAMIC_HEAT");
    return e && e[0] ? (float)std::atof(e) : 0.0f;
  }();
  return v;
}
float dynamic_heat_halflife() {
  static const float v = [] {
    const char *e = std::getenv("MOESTREAM_GLM5NEXT_DYNAMIC_HEAT_HALFLIFE");
    const float h = e && e[0] ? (float)std::atof(e) : 30.0f;
    return h > 0.5f ? h : 30.0f;
  }();
  return v;
}
float decayed_heat(const PartialExpertLayer &p, int expert, uint64_t clock) {
  const float h = p.heat[(size_t)expert];
  if (h <= 0.0f)
    return 0.0f;
  const float dt = (float)(clock - std::min<uint64_t>(clock, p.heat_at[(size_t)expert]));
  return h * std::exp2(-dt / dynamic_heat_halflife());
}

void update_partial_routes(FullGraph &g, int pos_base, double compute_ms) {
  if (!g.partial_expert_residency || g.moe_route_ids.empty())
    return;
  const auto &c = g.engine->model_file.architecture_config;
  const uint64_t clock = ++g.dynamic_clock;
  RouteTap &tap = route_tap();
  if (!tap.path.empty() && !tap.loaded) {
    tap.loaded = true;
    tap.counts.assign((size_t)c.n_layers,
                      std::vector<uint32_t>((size_t)c.n_experts, 0));
    if (std::FILE *f = std::fopen(tap.path.c_str(), "r")) {
      unsigned l, e, n;
      while (std::fscanf(f, "%u %u %u", &l, &e, &n) == 3)
        if (l < tap.counts.size() && e < tap.counts[l].size())
          tap.counts[l][e] += n;
      std::fclose(f);
    }
  }
  for (const auto &[layer, route_tensor] : g.moe_route_ids) {
    if (!route_tensor || layer < 0 ||
        (size_t)layer >= g.partial_experts.size())
      continue;
    std::vector<int32_t> route_ids((size_t)c.topk);
    ggml_backend_tensor_get(route_tensor, route_ids.data(), 0,
                            route_ids.size() * sizeof(int32_t));
    PartialExpertLayer &p = g.partial_experts[(size_t)layer];
    for (int32_t expert : route_ids) {
      if (expert < 0 || expert >= c.n_experts)
        continue;
      if (!tap.path.empty() && (size_t)layer < tap.counts.size())
        ++tap.counts[(size_t)layer][(size_t)expert];
      ++p.route_count;
      if ((size_t)expert < p.expert_in_flight.size() &&
          p.expert_in_flight[(size_t)expert]) {
        if (p.miss_count[(size_t)expert] < UINT32_MAX)
          ++p.miss_count[(size_t)expert];
        if ((size_t)expert < p.heat.size()) {
          p.heat[(size_t)expert] = decayed_heat(p, expert, clock) + 1.0f;
          p.heat_at[(size_t)expert] = clock;
        }
        continue;
      }
      const auto hot = std::find(p.hot.begin(), p.hot.end(), expert);
      if (hot != p.hot.end()) {
        ++p.hot_route_count;
        const size_t slot = (size_t)(hot - p.hot.begin());
        if (slot < p.slot_last_use.size())
          p.slot_last_use[slot] = (int)std::min<uint64_t>(clock, INT_MAX);
        if ((size_t)expert < p.miss_count.size())
          p.miss_count[(size_t)expert] = 0;
      } else if ((size_t)expert < p.miss_count.size() &&
                 p.miss_count[(size_t)expert] < UINT32_MAX) {
        ++p.miss_count[(size_t)expert];
      }
    }
  }

  if (!tap.path.empty() && ++tap.tokens % 25 == 0)
    route_tap_flush(tap);
  if (tap.tokens > 0 && tap.tokens % 100 == 0 && !tap.path.empty()) {
    std::string line;
    for (int l = c.first_dense; l < c.n_layers; ++l) {
      const PartialExpertLayer &p = g.partial_experts[(size_t)l];
      if (!p.hot.empty() && !p.cold.empty() && p.route_count) {
        char b[64];
        std::snprintf(b, sizeof(b), " L%d=%.1f%%", l,
                      100.0 * (double)p.hot_route_count / (double)p.route_count);
        line += b;
      }
    }
    if (!line.empty())
      LOG_WARN("[GLM5NEXT-PARTIAL-HIT] tokens=%d per-layer hot-route hit:%s\n",
               tap.tokens, line.c_str());
  }

  if (g.dynamic_expert_cache) {
    if (compute_ms > 0.0 && std::isfinite(compute_ms)) {
      g.dynamic_compute_ms_ema =
          g.dynamic_compute_ms_ema > 0.0
              ? 0.8 * g.dynamic_compute_ms_ema + 0.2 * compute_ms
              : compute_ms;
    }
    // Publish only at this post-compute boundary. The just-finished token's
    // route statistics then reflect the map it actually executed with.
    if (!publish_dynamic_uploads(g))
      return;
    if (g.dynamic_upload) {
      std::lock_guard<std::mutex> lock(g.dynamic_upload->mutex);
      if (g.dynamic_upload->stopping)
        g.dynamic_upload_disabled = true;
    }
    int budget = dynamic_upload_budget_override();
    if (budget < 0 && !g.dynamic_upload_disabled) {
      if (!ensure_dynamic_upload_worker(g)) {
        g.dynamic_upload_disabled = true;
      } else {
        size_t active_layers = 0;
        for (int layer = c.first_dense; layer < c.n_layers; ++layer) {
          const PartialExpertLayer &p = g.partial_experts[(size_t)layer];
          active_layers += !p.hot.empty() && !p.cold.empty();
        }
        if (g.dynamic_upload_ms_per_job <= 0.0) {
          // Neurall-style bootstrap: enqueue one initial candidate per layer,
          // then derive the steady-state budget from observed copy time.
          budget = (int)std::min<size_t>(active_layers, 128);
        } else {
          size_t pending = 0;
          {
            std::lock_guard<std::mutex> lock(g.dynamic_upload->mutex);
            pending = g.dynamic_upload->queue.size() +
                      g.dynamic_upload->completed.size() +
                      (g.dynamic_upload->active ? 1 : 0);
          }
          const double affordable =
              dynamic_upload_fraction() * g.dynamic_compute_ms_ema /
              g.dynamic_upload_ms_per_job;
          const size_t total_budget =
              std::min<size_t>(active_layers,
                               (size_t)std::max(0.0, std::floor(affordable)));
          budget = total_budget > pending
                       ? (int)std::min<size_t>(total_budget - pending, 128)
                       : 0;
        }
      }
    }
    if (budget < 0)
      budget = 0;
    if (budget > 0 && !g.dynamic_upload_disabled) {
      if (!g.dynamic_upload && !ensure_dynamic_upload_worker(g)) {
        g.dynamic_upload_disabled = true;
      } else {
      int scheduled = 0;
      DynamicUploadState &state = *g.dynamic_upload;
      size_t queued = 0;
      {
        std::lock_guard<std::mutex> lock(state.mutex);
        queued = state.queue.size() + (state.active ? 1 : 0);
      }
      const size_t queue_limit = (size_t)std::max(2, budget * 2);
      if (pos_base % 64 == 0) {
        // Why nothing gets promoted: tally the per-layer gates once in a while.
        int no_hot = 0, no_cold = 0, bad_vec = 0, no_cand = 0, ok = 0;
        for (int layer = c.first_dense; layer < c.n_layers; ++layer) {
          const PartialExpertLayer &p = g.partial_experts[(size_t)layer];
          if (p.hot.empty()) { ++no_hot; continue; }
          if (p.cold.empty()) { ++no_cold; continue; }
          if (p.slot_last_use.size() != p.hot.size() || p.slot_in_flight.size() != p.hot.size() ||
              p.miss_count.size() != (size_t)c.n_experts || p.expert_in_flight.size() != (size_t)c.n_experts) { ++bad_vec; continue; }
          bool cand = false;
          for (int expert : p.cold)
            if (expert >= 0 && expert < c.n_experts && p.miss_count[(size_t)expert] > 1) { cand = true; break; }
          cand ? ++ok : ++no_cand;
        }
        LOG_INFO("[GGML-GLM5NEXT-CACHE] dynamic gates pos=%d budget=%d queued=%zu: layers without hot=%d, without cold=%d, bad vectors=%d, "
                 "no candidate=%d, candidate available=%d\n", pos_base, budget, queued, no_hot, no_cold, bad_vec, no_cand, ok);
      }
      for (int layer = c.first_dense;
           layer < c.n_layers && scheduled < budget &&
           queued < queue_limit;
           ++layer) {
        PartialExpertLayer &p = g.partial_experts[(size_t)layer];
        if (p.hot.empty() || p.cold.empty() ||
            p.slot_last_use.size() != p.hot.size() ||
            p.slot_in_flight.size() != p.hot.size() ||
            p.miss_count.size() != (size_t)c.n_experts ||
            p.expert_in_flight.size() != (size_t)c.n_experts)
          continue;
        int candidate_index = -1;
        uint32_t candidate_hits = 1; // Ignore one-off routes.
        float best_heat = 0.0f;
        for (size_t i = 0; i < p.cold.size(); ++i) {
          const int expert = p.cold[i];
          if (expert < 0 || expert >= c.n_experts ||
              p.expert_in_flight[(size_t)expert])
            continue;
          if (dynamic_heat_threshold() > 0.0f && p.heat.size() == (size_t)c.n_experts) {
            const float h = decayed_heat(p, expert, g.dynamic_clock);
            if (h >= dynamic_heat_threshold() && h > best_heat) {
              best_heat = h;
              candidate_index = (int)i;
            }
            continue;
          }
          const uint32_t hits = p.miss_count[(size_t)expert];
          if (hits > candidate_hits) {
            candidate_hits = hits;
            candidate_index = (int)i;
          }
        }
        if (candidate_index < 0)
          continue;

        int hot_slot = -1;
        for (size_t i = 0; i < p.hot.size(); ++i) {
          if (p.hot[i] < 0 && !p.slot_in_flight[i]) {
            hot_slot = (int)i;
            break;
          }
        }
        if (hot_slot < 0) {
          int oldest = INT_MAX;
          for (size_t i = 0; i < p.hot.size(); ++i) {
            if (p.slot_in_flight[i] || p.hot[i] < 0)
              continue;
            if (p.slot_last_use[i] < oldest) {
              oldest = p.slot_last_use[i];
              hot_slot = (int)i;
            }
          }
        }
        if (hot_slot < 0)
          continue;

        const int incoming = p.cold[(size_t)candidate_index];
        const int evicted = p.hot[(size_t)hot_slot];
        DynamicUploadJob job;
        if (!make_dynamic_upload_job(p, layer, hot_slot, incoming, evicted,
                                     &job)) {
          LOG_WARN("[GGML-GLM5NEXT-CACHE] unable to form full-source upload "
                   "layer=%d incoming=%d\n",
                   layer, incoming);
          continue;
        }
        if (evicted >= 0 &&
            std::find(p.cold.begin(), p.cold.end(), evicted) == p.cold.end())
          p.cold.push_back(evicted);
        p.hot[(size_t)hot_slot] = -1;
        p.slot_in_flight[(size_t)hot_slot] = 1;
        p.expert_in_flight[(size_t)incoming] = 1;
        if (!upload_partial_maps(g, p)) {
          g.ready = false;
          return;
        }
        {
          std::lock_guard<std::mutex> lock(state.mutex);
          if (state.stopping || state.queue.size() >= queue_limit) {
            p.hot[(size_t)hot_slot] = evicted;
            p.slot_in_flight[(size_t)hot_slot] = 0;
            p.expert_in_flight[(size_t)incoming] = 0;
            if (evicted >= 0)
              p.cold.erase(std::remove(p.cold.begin(), p.cold.end(), evicted),
                           p.cold.end());
            if (!upload_partial_maps(g, p))
              g.ready = false;
            continue;
          }
          state.queue.push_back(std::move(job));
          state.cv.notify_one();
        }
        ++scheduled;
        ++queued;
      }
      if (scheduled > 0)
        LOG_INFO("[GGML-GLM5NEXT-CACHE] pos=%d queued=%d/%d async expert "
                 "uploads\n",
                 pos_base, scheduled, budget);
      }
    }
  }

  if ((pos_base + 1) % 32 == 0) {
    uint64_t routes = 0;
    uint64_t hot_routes = 0;
    for (int layer = c.first_dense; layer < c.n_layers; ++layer) {
      const PartialExpertLayer &p = g.partial_experts[(size_t)layer];
      routes += p.route_count;
      hot_routes += p.hot_route_count;
    }
    LOG_INFO("[GGML-GLM5NEXT-CACHE] pos=%d route hit=%.1f%% (%llu/%llu) "
             "dynamic_promotions=%llu h2d_ema=%.3f ms/job "
             "compute_ema=%.3f ms/token\n",
             pos_base,
             routes ? 100.0 * (double)hot_routes / routes : 0.0,
             (unsigned long long)hot_routes,
             (unsigned long long)routes,
             (unsigned long long)g.dynamic_promotions,
             g.dynamic_upload_ms_per_job, g.dynamic_compute_ms_ema);
  }
}

bool hold_graph_vram_experts(FullGraph &g) {
  if (!g.vram_reference_experts || g.vram_expert_holds.empty())
    return true;
  if (!g.topology)
    return false;
  size_t held = 0;
  for (const FullGraph::VramExpertHold &ref : g.vram_expert_holds) {
    if (!g.topology->hold_for_compute(ref.layer, ref.expert,
                                      ref.device_ptr, false)) {
      while (held > 0) {
        const auto &prior = g.vram_expert_holds[--held];
        g.topology->release_hold(prior.layer, prior.expert,
                                 prior.device_ptr, false);
      }
      LOG_ERROR("[GGML-GLM5NEXT-CACHE] unable to hold aliased VRAM expert "
                "layer=%d expert=%d for graph compute\n",
                ref.layer, ref.expert);
      return false;
    }
    ++held;
  }
  return true;
}

void release_graph_vram_experts(FullGraph &g) {
  if (!g.vram_reference_experts || !g.topology)
    return;
  for (auto it = g.vram_expert_holds.rbegin();
       it != g.vram_expert_holds.rend(); ++it)
    g.topology->release_hold(it->layer, it->expert, it->device_ptr, false);
}

ggml_tensor *static_full_weight(FullGraph &g, ggml_context *ctx,
                                const std::string &name) {
  // DSA's attn_k_b and attn_v_b are intentionally absent from the legacy
  // resident manifest because that path builds CPU WUK/WUV caches.  In this
  // graph they are ordinary static weights. Ling's native-cache graph keeps
  // these tensors in resident_layers on the host for the existing MLA path;
  // bind that source directly instead of trying to copy it into the graph's
  // 1 MiB dense marker arena.
  if (g.native_moe_cache) {
    if (ggml_tensor *t = resident_tensor(g, ctx, name))
      return t;
    LOG_WARN("[GGML-LING-HYBRID] static weight is not resident: %s; "
             "trying the legacy graph-owned loader\n",
             name.c_str());
  }
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
  g.host_reference_bytes += meta->size_bytes;
  ggml_set_name(t, (name + ".graph_ram").c_str());
  return t;
}

} // namespace glm5next_graph

#endif // MOESTREAM_USE_GGML_CUDA
