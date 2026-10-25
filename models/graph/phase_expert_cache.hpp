#pragma once

// Optional, separately releasable decode-hot experts. The permanent topology
// cache never moves. Graph owners must retire borrowers before release(); KV
// and recurrent state belong to the model/engine, not this allocation.
#include "aliased_expert_placement.hpp"
#include "expert_store.hpp"
#include "moe_block.hpp"
#include "graph_settings_global.hpp"
#include "placement_epoch.hpp"
#include <array>
#include <chrono>
#include <functional>
#include <memory>
#include <stdexcept>

namespace zeromoe::graph {

struct PhaseCacheOptions { size_t bytes = 0, reserve_bytes = 0; };
inline PhaseCacheOptions phase_cache_options() {
  const auto &t = zeromoe::graphsettings::settings_for(zeromoe::graphsettings::Family::Other).tun;
  constexpr long max_mib = 1024 * 1024;
  return {size_t(std::clamp(t.phase_cache_mib.value, 0L, max_mib)) << 20,
          size_t(std::clamp(t.phase_cache_reserve_mib.value, 0L, max_mib)) << 20};
}

class PhaseExpertCache {
 public:
  struct Layer {
    AliasedHotSet hot;
    std::vector<int32_t> ids;
    std::vector<float> mask;
    ggml_backend_buffer_t ids_buffer = nullptr, mask_buffer = nullptr;
  };
  ~PhaseExpertCache() { release(); if (descriptor_) ggml_backend_buffer_free(descriptor_); }
  PhaseExpertCache() = default;
  PhaseExpertCache(const PhaseExpertCache &) = delete;
  PhaseExpertCache &operator=(const PhaseExpertCache &) = delete;

  uint64_t generation() const { return epoch_.generation(); }
  void bind(GraphCore &graph) const {
    graph.placement_generation = generation(); graph.placement_lease = epoch_.borrow();
  }
  bool active() const { return pool_ != nullptr; }
  size_t bytes() const { return bytes_; }
  const Layer *layer(int l) const {
    return active() && l >= 0 && size_t(l) < layers_.size() && !layers_[size_t(l)].hot.empty()
             ? &layers_[size_t(l)] : nullptr;
  }

  // Never calls ensure(), reads a model file, or evicts DDR entries. A reader
  // lease protects all source planes until the synchronous H2D copies finish.
  bool prepare_decode(const StreamingEngine &e, zeromoe::offload::ExpertStore &store,
                      ggml_backend_t backend, int device, size_t requested_bytes,
                      size_t reserve_bytes, const std::string &usage,
                      const std::function<bool(int, int)> &permanent_hot) {
    if (active() || requested_bytes == 0) return true;
    DeviceGuard guard(device);
    const auto start = std::chrono::steady_clock::now();
    size_t free = 0, total = 0;
    if (cudaMemGetInfo(&free, &total) != cudaSuccess) return false;
    const size_t available = free > reserve_bytes ? free - reserve_bytes : 0;
    const size_t limit = std::min(requested_bytes, available);
    if (limit <= 512) {
      LOG_WARN("[PHASE-CACHE] decode extension unavailable: free=%zu reserve=%zu\n", free, reserve_bytes);
      return false;
    }
    auto lease = store.reader_lock();
    const auto &c = e.model_file.architecture_config;
    if (layers_.empty()) {
      layers_.resize(size_t(c.n_layers));
      size_t used = 512;
      const auto ranked = zeromoe::offload::rank_hottest_first(
          usage, store.first_layer(), store.n_layers(), c.n_experts);
      for (const auto &key : ranked) {
        // Retain a refill guarantee while prefill uses the rest of the DDR
        // cache: these sources must not be evicted when the GPU copy is gone.
        if (permanent_hot(key.layer, key.expert) || !store.cache().pinned(key) ||
            !store.residency().ready(key)) continue;
        const size_t raw = e.compute_expert_slab_bytes(key.layer);
        const size_t alignment = e.expert_cache_layer_stride_alignment(key.layer);
        if (!alignment || alignment == SIZE_MAX || raw > SIZE_MAX - alignment) continue;
        const size_t stride = (raw + alignment - 1) / alignment * alignment;
        if (!stride || stride > limit - used || 256 > limit - used - stride) continue;
        auto &l = layers_[size_t(key.layer)];
        l.hot.stride = stride;
        l.hot.experts.push_back(key.expert);
        used += stride + 256; // includes a conservative per-layer alignment allowance
      }
    }
    size_t payload = 0, count = 0;
    for (size_t l = 0; l < layers_.size(); ++l) {
      auto &p = layers_[l];
      if (p.hot.empty()) continue;
      for (int expert : p.hot.experts) {
        if (permanent_hot(int(l), expert) || !store.residency().ready({int(l), expert})) {
          LOG_WARN("[PHASE-CACHE] DDR refill unavailable layer=%zu expert=%d; no demand load issued\n", l, expert);
          return false;
        }
      }
      payload = (payload + 255) & ~size_t(255);
      payload += p.hot.experts.size() * p.hot.stride;
      count += p.hot.experts.size();
    }
    if (!count || payload > limit - 512) return false;
    void *allocation = nullptr;
    if (cudaMalloc(&allocation, payload + 512) != cudaSuccess) {
      cudaGetLastError();
      LOG_WARN("[PHASE-CACHE] decode extension allocation failed: %zu bytes\n", payload + 512);
      return false;
    }
    pool_ = static_cast<uint8_t *>(allocation); bytes_ = payload + 512; device_ = device;
    const auto rollback = [&]() {
      if (!release()) throw std::runtime_error("phase cache rollback failed; refusing stale GPU bindings");
      return false;
    };
    if (!descriptor_) {
      descriptor_ = ggml_backend_alloc_buffer(backend, 4096);
      if (!descriptor_) return rollback();
      ggml_backend_buffer_set_usage(descriptor_, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    }
    size_t offset = 0;
    for (size_t layer = 0; layer < layers_.size(); ++layer) {
      auto &p = layers_[layer];
      if (p.hot.empty()) continue;
      offset = (offset + 255) & ~size_t(255);
      p.hot.base = pool_ + offset;
      p.hot.slots = p.hot.experts.size();
      p.hot.slot_by_expert.assign(size_t(c.n_experts), -1);
      p.mask.assign(size_t(c.n_experts), 0.0f);
      for (size_t i = 0; i < p.hot.experts.size(); ++i) {
        const int expert = p.hot.experts[i];
        p.hot.slot_by_expert[size_t(expert)] = int32_t(i); p.mask[size_t(expert)] = 1.0f;
        size_t plane_offset = 0;
        for (int k = 0; k < 3; ++k) {
          const size_t n = store.arena().plane_bytes(int(layer), k);
          if (n > p.hot.stride - plane_offset ||
              cudaMemcpy(p.hot.base + i * p.hot.stride + plane_offset,
                         store.arena().ptr(int(layer), expert, k), n,
                         cudaMemcpyHostToDevice) != cudaSuccess) {
            return rollback();
          }
          plane_offset += n;
        }
      }
      const auto map = moe_partial::ExpertBranchMap::build(p.hot.slot_by_expert, 1, true);
      p.ids = map.route_ids(); p.mask = map.route_mask();
      p.ids_buffer = ggml_backend_alloc_buffer(backend, p.ids.size()*sizeof(int32_t));
      p.mask_buffer = ggml_backend_alloc_buffer(backend, p.mask.size()*sizeof(float));
      if (!p.ids_buffer || !p.mask_buffer) return rollback();
      for (auto *b : {p.ids_buffer, p.mask_buffer})
        ggml_backend_buffer_set_usage(b, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
      if (cudaMemcpy(ggml_backend_buffer_get_base(p.ids_buffer), p.ids.data(), p.ids.size()*sizeof(int32_t), cudaMemcpyHostToDevice) != cudaSuccess ||
          cudaMemcpy(ggml_backend_buffer_get_base(p.mask_buffer), p.mask.data(), p.mask.size()*sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
        return rollback();
      }
      offset += p.hot.slots * p.hot.stride;
    }
    epoch_.publish();
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now()-start).count();
    LOG_INFO("[PHASE-CACHE] decode generation=%llu device=%d experts=%zu bytes=%zu refill_ms=%.3f SSD_loads=0\n",
             (unsigned long long)generation(), device, count, bytes_, ms);
    return true;
  }

  // Retire every graph borrowing these weights before calling. Plans/identities
  // survive, but CUDA pointers and graph input buffers do not.
  bool release() {
    if (!pool_) return true;
    if (epoch_.borrowed()) {
      LOG_ERROR("[PHASE-CACHE] refusing release: a graph still borrows generation=%llu\n",
                (unsigned long long)generation());
      return false;
    }
    DeviceGuard guard(device_);
    const auto start = std::chrono::steady_clock::now();
    if (cudaDeviceSynchronize() != cudaSuccess || cudaFree(pool_) != cudaSuccess) return false;
    const size_t released = bytes_;
    pool_ = nullptr; bytes_ = 0;
    for (auto &l : layers_) {
      if (l.ids_buffer) ggml_backend_buffer_free(l.ids_buffer);
      if (l.mask_buffer) ggml_backend_buffer_free(l.mask_buffer);
      l.ids_buffer = l.mask_buffer = nullptr; l.hot.base = nullptr;
    }
    epoch_.retire();
    LOG_INFO("[PHASE-CACHE] prefill generation=%llu released_bytes=%zu release_ms=%.3f\n",
             (unsigned long long)generation(), released,
             std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now()-start).count());
    return true;
  }

  ggml_tensor *build_extra(const StreamingEngine &e, const MoEBlockSpec &spec,
                           const MoEBlockGraph &bg, const MoEBlockLayer &bl,
                           ggml_tensor *input, ggml_tensor *out) const {
    const auto *p = layer(bl.layer);
    if (!p) return out;
    const auto map = [&](ggml_type type, void *data, ggml_backend_buffer_t buffer) {
      auto *t = ggml_new_tensor_3d(bg.ctx, type, 1, spec.n_experts, 1);
      t->data = data; t->buffer = buffer; return t;
    };
    auto *ids = map(GGML_TYPE_I32, ggml_backend_buffer_get_base(p->ids_buffer), p->ids_buffer);
    auto *mask = map(GGML_TYPE_F32, ggml_backend_buffer_get_base(p->mask_buffer), p->mask_buffer);
    auto in = moe_partial::make_branch_inputs(bg.ctx, ids, mask, bl.route_ids, bl.route_weights, spec.topk, 1);
    moe_routed::ExpertWeights w;
    w.gate = make_alias_weight(e, bg.ctx, bl.layer, "ffn_gate_exps.weight", p->hot, descriptor_, bg.device, false);
    w.up = make_alias_weight(e, bg.ctx, bl.layer, "ffn_up_exps.weight", p->hot, descriptor_, bg.device, false);
    w.down = make_alias_weight(e, bg.ctx, bl.layer, "ffn_down_exps.weight", p->hot, descriptor_, bg.device, false);
    if (!w.valid()) throw std::runtime_error("phase cache expert layout cannot be represented");
    auto *x = ggml_reshape_3d(bg.ctx, input, spec.d_model, 1, 1);
    auto *extra = build_routed_branch(spec, bg, bl, x, w, in.ids, in.weights, bg.cuda);
    if (!extra) throw std::runtime_error("phase cache branch construction failed");
    return out ? ggml_add(bg.ctx, out, extra) : extra;
  }

 private:
  struct DeviceGuard {
    int previous = -1;
    explicit DeviceGuard(int device) { cudaGetDevice(&previous); cudaSetDevice(device); }
    ~DeviceGuard() { if (previous >= 0) cudaSetDevice(previous); }
  };
  uint8_t *pool_ = nullptr;
  size_t bytes_ = 0;
  int device_ = -1;
  zeromoe::memory::PlacementEpoch epoch_;
  ggml_backend_buffer_t descriptor_ = nullptr;
  std::vector<Layer> layers_;
};

} // namespace zeromoe::graph
