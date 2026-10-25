#pragma once

// Aliased-cache expert placement (R2.1 B): a family-independent description of
// where every routed expert lives when the graph borrows the engine's existing
// tiers instead of owning copies.
//
//   hot   : experts resident in GpuTopology's VRAM cache. The graph addresses the
//           cache slots in place (no copy); an optional second GPU contributes a
//           second hot set from its own cache.
//   cold  : every other expert, served from the DDR tier / SSD by the CPU lane.
//
// Nothing here depends on a family graph or on FullGraph. A family supplies the
// engine, the topology and the ggml context; this header produces the hot/cold
// sets, alias weight tensors, route maps and the compute-hold list.

#include "engine.hpp"
#include "gpu_topology.hpp"
#include "graph_core.hpp"
#include "logger.hpp"
#include "moe_partial_route.hpp"
#include "quant_table.hpp"

#include "ggml-backend.h"
#include "ggml.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <mutex>
#include <cstdio>
#include <unistd.h>
#include <fcntl.h>
#include <string>
#include <vector>

namespace zeromoe::graph {

// One GPU's hot set inside a layer: the cache region it aliases plus the
// expert -> physical-slot table.
struct AliasedHotSet {
  std::vector<int> experts;            // expert ids in cache-slot discovery order
  std::vector<int32_t> slot_by_expert; // n_experts entries, -1 = not held here
  uint8_t *base = nullptr;             // device address of this layer's slot 0
  size_t slots = 0;
  size_t stride = 0;
  bool empty() const { return experts.empty(); }
};

struct AliasedLayerPlacement {
  AliasedHotSet primary;   // the trunk GPU's cache
  AliasedHotSet secondary; // the second GPU's cache (optional)
  std::vector<int> cold;   // experts neither cache holds
};

struct AliasedVramHold {
  int layer = -1;
  int expert = -1;
  const uint8_t *device_ptr = nullptr;
};

struct AliasedPlacementOptions {
  int device = 0;
  // Second GPU whose cache is aliased too; < 0 disables it.
  int secondary_device = -1;
  // False disables aliasing of the trunk GPU's cache (A/B aid).
  bool alias_primary = true;
};

class AliasedExpertPlacement {
 public:
  bool configured() const { return configured_; }
  const AliasedLayerPlacement *layer(int l) const {
    return l >= 0 && (size_t)l < layers_.size() ? &layers_[(size_t)l] : nullptr;
  }
  const std::vector<AliasedVramHold> &holds() const { return holds_; }
  size_t primary_records() const { return primary_records_; }
  size_t secondary_records() const { return secondary_records_; }
  size_t aliased_bytes() const { return aliased_bytes_; }
  bool compact_layout() const { return compact_layout_; }

  // Discovers the per-layer cache regions. Returns false when there is nothing
  // to alias (no topology, no experts). Layers whose region is not contiguous
  // are left on the cold path, never half-aliased.
  bool configure(StreamingEngine &engine, GpuTopology &topology,
                 const AliasedPlacementOptions &opt) {
    const auto &c = engine.model_file.architecture_config;
    if (c.n_experts <= 0 || c.n_layers <= c.first_dense)
      return false;
    layers_.assign((size_t)c.n_layers, {});
    holds_.clear();
    primary_records_ = secondary_records_ = aliased_bytes_ = 0;
    std::unique_lock<std::recursive_mutex> lock(topology.cache_mutex);
    compact_layout_ = topology.expert_compact_layout;
    const bool have_secondary = opt.secondary_device >= 0;

    for (int layer = c.first_dense; layer < c.n_layers; ++layer) {
      AliasedLayerPlacement &p = layers_[(size_t)layer];
      p.primary.slot_by_expert.assign((size_t)c.n_experts, -1);
      p.secondary.slot_by_expert.assign((size_t)c.n_experts, -1);
      const size_t raw_stride = engine.compute_expert_slab_bytes(layer);
      const Region first = opt.alias_primary
                               ? find_region(topology, layer, opt.device, raw_stride)
                               : Region{};
      const Region second = have_secondary
                                ? find_region(topology, layer, opt.secondary_device,
                                              raw_stride)
                                : Region{};
      std::vector<uint8_t> cached((size_t)c.n_experts, 0);
      add_region(c.n_experts, layer, first, p.primary, cached, primary_records_);
      add_region(c.n_experts, layer, second, p.secondary, cached,
                 secondary_records_);
      if ((!first.locations.empty() && !first.valid) ||
          (!second.locations.empty() && !second.valid))
        LOG_WARN("[ALIASED-PLACEMENT] invalid compact cache region at layer=%d; "
                 "unaliased experts stay on the DDR path\n", layer);
      for (int e = 0; e < c.n_experts; ++e)
        if (!cached[(size_t)e])
          p.cold.push_back(e);
    }
    configured_ = true;
    return true;
  }


  // Integrity check: compares the head and tail of every plane of up to
  // `per_layer` aliased experts per layer against the same bytes read from the
  // model file. Catches wrong plane offsets, strides and stale cache slots
  // before a graph computes on them. Returns the number of mismatching probes.
  size_t verify_against_model(const StreamingEngine &engine, int per_layer = 2,
                              size_t probe = 4096) const {
    const auto &c = engine.model_file.architecture_config;
    size_t bad = 0, checked = 0;
    static const char *stems[3] = {"ffn_gate_exps.weight", "ffn_up_exps.weight",
                                   "ffn_down_exps.weight"};
    for (int layer = c.first_dense; layer < c.n_layers; ++layer) {
      const AliasedLayerPlacement *lp = this->layer(layer);
      if (!lp || lp->primary.empty()) continue;
      size_t plane_off = 0;
      for (int pl = 0; pl < 3; ++pl) {
        const GGUFTensor *t = model_tensor(
            engine, "blk." + std::to_string(layer) + "." + stems[pl]);
        if (!t || c.n_experts <= 0) return bad + 1;
        const size_t per_expert = (size_t)t->size_bytes / (size_t)c.n_experts;
        const std::string &path = engine.model_file.shard_paths.at((size_t)t->shard_idx);
        const int fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) return bad + 1;
        const size_t n = std::min<size_t>(probe, per_expert);
        std::vector<uint8_t> want(n), got(n);
        int done = 0;
        for (int e : lp->primary.experts) {
          if (done++ >= per_layer) break;
          const int32_t slot = lp->primary.slot_by_expert[(size_t)e];
          for (int where = 0; where < 2; ++where) { // head, tail of the plane
            const size_t in_plane = where ? per_expert - n : 0;
            const off_t file_off = (off_t)(t->offset + (uint64_t)e * per_expert + in_plane);
            if (::pread(fd, want.data(), n, file_off) != (ssize_t)n) { ++bad; continue; }
            const uint8_t *src = lp->primary.base + (size_t)slot * lp->primary.stride + plane_off + in_plane;
            if (cudaMemcpy(got.data(), src, n, cudaMemcpyDeviceToHost) != cudaSuccess) {
              cudaGetLastError(); ++bad; continue;
            }
            ++checked;
            if (std::memcmp(want.data(), got.data(), n) != 0) {
              if (bad < 6)
                LOG_ERROR("[ALIASED-VERIFY] layer=%d expert=%d slot=%d plane=%d %s differs from the model file\n",
                          layer, e, slot, pl, where ? "tail" : "head");
              ++bad;
            }
          }
        }
        ::close(fd);
        plane_off += per_expert;
      }
    }
    size_t empty_slot0 = 0, partial_regions = 0, regions = 0;
    for (int layer = c.first_dense; layer < c.n_layers; ++layer) {
      const AliasedLayerPlacement *lp = this->layer(layer);
      if (!lp || lp->primary.slots == 0) continue;
      ++regions;
      bool has0 = false;
      for (int e : lp->primary.experts) if (lp->primary.slot_by_expert[(size_t)e] == 0) has0 = true;
      if (!has0) ++empty_slot0;
      if (lp->primary.experts.size() != lp->primary.slots) ++partial_regions;
    }
    LOG_INFO("[ALIASED-VERIFY] %zu probes compared, %zu mismatched; %zu/%zu layer regions have an empty slot 0, %zu are partially filled\n",
             checked, bad, empty_slot0, regions, partial_regions);
    return bad;
  }

  // Pins every aliased slot for the lifetime of a graph run.
  bool hold_all(GpuTopology &topology) const {
    size_t held = 0;
    for (const AliasedVramHold &ref : holds_) {
      if (!topology.hold_for_compute(ref.layer, ref.expert, ref.device_ptr,
                                     false)) {
        while (held > 0) {
          const AliasedVramHold &prior = holds_[--held];
          topology.release_hold(prior.layer, prior.expert, prior.device_ptr,
                                false);
        }
        LOG_ERROR("[ALIASED-PLACEMENT] unable to hold aliased VRAM expert "
                  "layer=%d expert=%d for graph compute\n",
                  ref.layer, ref.expert);
        return false;
      }
      ++held;
    }
    return true;
  }

  void release_all(GpuTopology &topology) const {
    for (auto it = holds_.rbegin(); it != holds_.rend(); ++it)
      topology.release_hold(it->layer, it->expert, it->device_ptr, false);
  }

 private:
  struct Region {
    uint8_t *base = nullptr;
    size_t first_offset = 0;
    size_t slots = 0;
    size_t stride = 0;
    std::vector<const ExpertLocation *> locations;
    bool valid = false;
  };

  static Region find_region(GpuTopology &topology, int layer, int gpu_id,
                            size_t raw_stride) {
    Region region;
    if (!topology.expert_compact_layout || raw_stride == 0)
      return region;
    for (const ExpertLocation &loc : topology.expert_vram_cache)
      if (loc.pool_layer == layer && loc.gpu_id == gpu_id)
        region.locations.push_back(&loc);
    if (region.locations.empty())
      return region;
    std::sort(region.locations.begin(), region.locations.end(),
              [](const ExpertLocation *a, const ExpertLocation *b) {
                return a->offset < b->offset;
              });
    region.stride = region.locations.front()->size;
    if (region.stride < raw_stride || region.stride == 0)
      return region;
    for (const ExpertLocation *loc : region.locations)
      if (!loc->device_ptr || loc->size != region.stride)
        return region;
    region.first_offset = region.locations.front()->offset;
    region.slots = region.locations.size();
    if (region.slots > SIZE_MAX / region.stride ||
        region.first_offset > SIZE_MAX - region.slots * region.stride ||
        region.slots > (size_t)INT32_MAX)
      return region;
    const uintptr_t first =
        reinterpret_cast<uintptr_t>(region.locations.front()->device_ptr);
    if (first < region.first_offset)
      return region;
    const uintptr_t base = first - region.first_offset;
    for (size_t slot = 0; slot < region.slots; ++slot) {
      const size_t offset = region.first_offset + slot * region.stride;
      if (base > std::numeric_limits<uintptr_t>::max() - offset ||
          region.locations[slot]->offset != offset ||
          reinterpret_cast<uintptr_t>(region.locations[slot]->device_ptr) !=
              base + offset)
        return region;
    }
    // The graph tensor has a layer-local slot index (0..slots), so its data
    // pointer starts at this layer's first cache slot, not the pool base.
    region.base = reinterpret_cast<uint8_t *>(base + region.first_offset);
    region.valid = true;
    return region;
  }

  void add_region(int n_experts, int layer, const Region &region,
                  AliasedHotSet &set, std::vector<uint8_t> &cached,
                  size_t &records) {
    if (!region.valid)
      return;
    set.base = region.base;
    set.slots = region.slots;
    set.stride = region.stride;
    for (const ExpertLocation *loc : region.locations) {
      if (loc->layer_idx != layer || loc->expert_id < 0 ||
          loc->expert_id >= n_experts || !loc->upload_ready)
        continue;
      const size_t slot = (loc->offset - region.first_offset) / region.stride;
      const size_t expert = (size_t)loc->expert_id;
      if (slot >= region.slots || cached[expert])
        continue;
      cached[expert] = 1;
      set.slot_by_expert[expert] = (int32_t)slot;
      set.experts.push_back((int)expert);
      ++records;
      holds_.push_back({layer, (int)expert, loc->device_ptr});
      if (aliased_bytes_ <= SIZE_MAX - region.stride)
        aliased_bytes_ += region.stride;
    }
  }

  std::vector<AliasedLayerPlacement> layers_;
  std::vector<AliasedVramHold> holds_;
  size_t primary_records_ = 0, secondary_records_ = 0, aliased_bytes_ = 0;
  bool compact_layout_ = false;
  bool configured_ = false;
};

// Tensor view of one expert plane (gate/up/down) over an aliased cache region.
// `descriptor` only tags the tensor with the owning CUDA backend; it holds no
// weights. Returns nullptr when the geometry cannot be addressed in place.
inline ggml_tensor *make_alias_weight(const StreamingEngine &engine,
                                      ggml_context *ctx, int layer,
                                      const char *stem, const AliasedHotSet &set,
                                      ggml_backend_buffer_t descriptor,
                                      int gpu_id, bool secondary) {
  if (set.empty() || !set.base || set.slots == 0 || set.stride == 0 ||
      !descriptor)
    return nullptr;
  const std::string base = "blk." + std::to_string(layer) + ".";
  const GGUFTensor *meta = model_tensor(engine, base + stem);
  const size_t n_experts =
      (size_t)engine.model_file.architecture_config.n_experts;
  if (!meta || meta->shape.size() != 3 || n_experts == 0 ||
      meta->size_bytes % n_experts != 0)
    return nullptr;
  size_t offset = 0;
  if (std::strcmp(stem, "ffn_up_exps.weight") == 0) {
    const GGUFTensor *gate = model_tensor(engine, base + "ffn_gate_exps.weight");
    if (!gate || gate->size_bytes % n_experts != 0)
      return nullptr;
    offset = (size_t)gate->size_bytes / n_experts;
  } else if (std::strcmp(stem, "ffn_down_exps.weight") == 0) {
    const GGUFTensor *gate = model_tensor(engine, base + "ffn_gate_exps.weight");
    const GGUFTensor *up = model_tensor(engine, base + "ffn_up_exps.weight");
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
  if (offset > set.stride || per_expert_bytes > set.stride - offset ||
      set.slots > SIZE_MAX / set.stride)
    return nullptr;
  std::vector<int64_t> shape;
  for (uint64_t dim : meta->shape)
    shape.push_back((int64_t)dim);
  shape[2] = (int64_t)set.slots;
  ggml_tensor *t = new_tensor(ctx, (ggml_type)meta->type, shape);
  if (!t)
    return nullptr;
  if (set.stride % ggml_row_size(t->type, t->ne[0]) != 0 || offset % 2 != 0) {
    static bool warned = false;
    if (!warned) {
      warned = true;
      LOG_STREAM_WARN() << "[ALIASED-PLACEMENT] VRAM alias row-stride check "
                        << "failed (GPU=" << gpu_id << " stride=" << set.stride
                        << " offset=" << offset << " row_size="
                        << ggml_row_size(t->type, t->ne[0]) << " type="
                        << ggml_type_name(t->type) << ")\n";
    }
    return nullptr;
  }
  t->nb[2] = set.stride;
  t->nb[3] = t->nb[2] * (size_t)t->ne[2];
  t->data = set.base + offset;
  t->buffer = descriptor;
  ggml_set_name(t, (base + stem + (secondary ? ".gpu1_vram_cache_ref"
                                             : ".vram_cache_ref")).c_str());
  return t;
}


// True when every expert projection of `layer` runs ggml's MMQ MUL_MAT_ID path. Batched
// (S > 1) graphs may then use the patched negative-id skip for routes a branch does not own.
// Repeating a valid placeholder id instead overruns ggml's per-expert row ranges, because
// MUL_MAT_ID assumes a token selects each expert at most once.
inline bool layer_experts_take_mmq(const StreamingEngine &engine, int layer) {
  for (const char *stem : {"ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"}) {
    const GGUFTensor *t = model_tensor(engine, "blk." + std::to_string(layer) + "." + stem);
    if (!t) return false;
    if (!zeromoe::quant::takes_mmq(static_cast<uint32_t>(t->type))) return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
// Route maps. The maps translate the model's expert ids to branch-local ids.
// ---------------------------------------------------------------------------
struct AliasedMapRequest {
  int n_experts = 0;
  int n_tokens = 1;
  bool decode = false;
  // Cold branch addressing: true = identity ids into the full DDR/offload
  // arena, false = compacted (cold-only) tensor.
  bool full_ddr_view = false;
  // True when the second GPU's hot set is served by its own branch this build;
  // false folds those experts back into the cold branch (identity-indexed).
  bool secondary_active = false;
  moe_partial::SentinelPolicy policy = moe_partial::SentinelPolicy::Auto;
};


// Trunk GPU's hot branch. Returns false when a hot expert has no cache slot.
inline bool build_primary_map(const AliasedLayerPlacement &p,
                              const AliasedMapRequest &r,
                              moe_partial::ExpertBranchMap *out) {
  const size_t n = (size_t)std::max(0, r.n_experts);
  std::vector<int32_t> index_of_expert(n, -1);
  std::vector<uint8_t> owned(n, 0);
  for (int e : p.primary.experts) {
    if (e < 0 || (size_t)e >= n) continue;
    const int32_t slot = (size_t)e < p.primary.slot_by_expert.size()
                             ? p.primary.slot_by_expert[(size_t)e] : -1;
    if (slot < 0) return false;
    index_of_expert[(size_t)e] = slot;
    owned[(size_t)e] = 1;
  }
  *out = moe_partial::ExpertBranchMap::build(
      index_of_expert, r.n_tokens, r.decode && r.n_tokens == 1, owned, r.policy);
  return true;
}

// Cold (CPU / DDR) branch.
inline bool build_cold_map(const AliasedLayerPlacement &p,
                           const AliasedMapRequest &r,
                           moe_partial::ExpertBranchMap *out) {
  const size_t n = (size_t)std::max(0, r.n_experts);
  std::vector<int32_t> index_of_expert(n, -1);
  std::vector<uint8_t> owned(n, 0);
  int32_t cold_slot = 0;
  for (int e : p.cold) {
    if (e < 0 || (size_t)e >= n) return false;
    index_of_expert[(size_t)e] = r.full_ddr_view ? e : cold_slot++;
    owned[(size_t)e] = 1;
  }
  if (!p.secondary.empty() && !r.secondary_active) {
    for (int e : p.secondary.experts) {
      if (e < 0 || (size_t)e >= n) continue;
      index_of_expert[(size_t)e] = e; // full-arena identity view
      owned[(size_t)e] = 1;
    }
  }
  *out = moe_partial::ExpertBranchMap::build(
      index_of_expert, r.n_tokens, r.decode && r.n_tokens == 1, owned, r.policy);
  return true;
}

// Second GPU's hot branch.
inline bool build_secondary_map(const AliasedLayerPlacement &p,
                                const AliasedMapRequest &r,
                                moe_partial::ExpertBranchMap *out) {
  const size_t n = (size_t)std::max(0, r.n_experts);
  std::vector<int32_t> index_of_expert(n, -1);
  std::vector<uint8_t> owned(n, 0);
  for (int e : p.secondary.experts) {
    if (e < 0 || (size_t)e >= n) continue;
    const int32_t slot = (size_t)e < p.secondary.slot_by_expert.size()
                             ? p.secondary.slot_by_expert[(size_t)e] : -1;
    if (slot < 0) continue;
    index_of_expert[(size_t)e] = slot;
    owned[(size_t)e] = 1;
  }
  *out = moe_partial::ExpertBranchMap::build(
      index_of_expert, r.n_tokens, r.decode && r.n_tokens == 1, owned, r.policy);
  return true;
}

} // namespace zeromoe::graph
