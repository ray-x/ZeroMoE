#pragma once

// Runtime owner of the cold (DDR/SSD) expert tier for one process (R2.1 B).
//
// Wraps an ExpertStore built by offload_expert_provider.hpp and adds what every
// graph that uses it needs: arena-backed plane tensors, the "miss node" that
// makes routed experts resident before the CPU/GPU branch reads them, and the
// shared adoption of an EarlyReadyLane (residency callbacks + GPU assist). No
// family graph, no FullGraph, no environment lookups.

#include "early_ready_offload_strategy.hpp"
#include "offload_expert_provider.hpp"
#include "offload_residency_adapter.hpp"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <deque>
#include <memory>
#include <mutex>
#include <unordered_map>

namespace zeromoe::graph {

class OffloadExpertRuntime;

// One miss node. `not_cold` marks experts a VRAM branch serves; they are never
// loaded into DDR.
struct OffloadMissSite {
  OffloadExpertRuntime *runtime = nullptr;
  int layer = 0;
  bool prefill = false;
  int n_tokens = 1;
  std::vector<uint8_t> not_cold;
};

struct GpuAssistConfig {
  int slots = 0;   // 0 disables GPU assist
  int device = -1;
  int slots2 = 0;  // optional second assist device
  int device2 = -1;
  std::function<bool(int)> layer_ok; // layers whose experts the ring must fit
};

class OffloadExpertRuntime {
 public:
  int first_layer() const { return first_; }
  int n_layers() const { return n_layers_; }
  int n_experts() const { return n_experts_; }
  zeromoe::offload::ExpertStore &store() { return *store_; }
  zeromoe::offload::VirtualArena &arena() { return *arena_; }
  bool covers(int layer) const {
    return layer >= first_ && layer < first_ + n_layers_;
  }

  // Builds the store from `profile` and seeds DDR from `usage_file` (optional).
  bool create(const StreamingEngine &engine, const OffloadProfile &profile,
              const std::string &usage_file, const SeedExclusions *exclusions,
              std::string *err) {
    const auto &c = engine.model_file.architecture_config;
    zeromoe::offload::ExpertStoreConfig sc;
    if (!make_store_config(engine, profile, &sc, err))
      return false;
    store_ = zeromoe::offload::ExpertStore::create(sc, err);
    if (!store_)
      return false;
    arena_ = &store_->arena();
    first_ = profile.first_layer;
    n_layers_ = c.n_layers - first_;
    n_experts_ = c.n_experts;
    tag_ = profile.log_tag;
    if (!usage_file.empty())
      seed_store(*store_, usage_file, first_, n_layers_, n_experts_, exclusions,
                 tag_);
    LOG_INFO("%s layers %d..%d, capacity %zu experts (%.1f GiB), pinned %.2f\n",
             tag_, first_, c.n_layers - 1, store_->capacity(), sc.ddr_gib,
             sc.pinned_fraction);
    return true;
  }

  ggml_tensor *plane(const StreamingEngine &engine, ggml_context *ctx, int layer,
                     const char *stem,
                     const std::function<bool(GGUFType)> &type_supported,
                     std::vector<ggml_backend_buffer_t> *buffers) {
    return make_plane_tensor(engine, *arena_, n_experts_, ctx, layer, stem,
                             type_supported, buffers, tag_);
  }

  // [topk, S] I32 route ids -> same ids, after making the routed experts
  // resident. Negative ids (inactive routes) are skipped.
  ggml_tensor *miss_node(ggml_context *ctx, int layer, ggml_tensor *ids,
                         bool prefill, int n_tokens,
                         std::vector<uint8_t> not_cold) {
    if (!ids || !covers(layer))
      return nullptr;
    std::lock_guard<std::mutex> lock(mutex_);
    sites_.push_back({this, layer, prefill, n_tokens, std::move(not_cold)});
    ggml_tensor *out = ggml_map_custom1(ctx, ids, &OffloadExpertRuntime::miss_cb,
                                        1, &sites_.back());
    if (out)
      ggml_set_name(out, ("offload_ids_" + std::to_string(layer)).c_str());
    return out;
  }

  // Installs the residency callbacks (and GPU assist, if configured) on a lane
  // built by build_early_ready_graph_lane, hands ownership to `owner`, and
  // returns the stable lane pointer. nullptr rejects the lane.
  EarlyReadyLane *adopt_lane(int layer, const moe_routed::ExpertWeights &cold_w,
                             std::unique_ptr<EarlyReadyLane> lane,
                             const GpuAssistConfig &assist, bool split_load,
                             std::unordered_map<int, std::unique_ptr<EarlyReadyLane>> &owner) {
    if (!lane || !covers(layer))
      return nullptr;
    ColdMoEAdapter adapter =
        make_offload_residency_adapter(*store_, layer, split_load, tag_);
    if (assist.slots > 0 && assist.layer_ok && assist.layer_ok(layer)) {
      auto shared = ring(0, assist.device, assist.slots, assist.layer_ok);
      std::shared_ptr<GpuAssistShared> shared2;
      if (assist.device2 >= 0 && assist.device2 != assist.device &&
          assist.slots2 > 0)
        shared2 = ring(1, assist.device2, assist.slots2, assist.layer_ok);
      if (shared && lane->prepare_gpu_assist(cold_w, shared, 0)) {
        if (shared2 && !lane->prepare_gpu_assist(cold_w, shared2, 1)) {
          lane->assists[1].graph = nullptr;
          LOG_WARN("%s layer=%d second GPU assist unavailable\n", tag_, layer);
        }
        adapter.expert_planes = [this, layer](int expert, const void *planes[3]) {
          if (!store_->residency().ready({layer, expert}) ||
              !store_->is_pinned(layer, expert))
            return false; // an unpinned slot would make the async copy blocking
          for (int k = 0; k < 3; ++k)
            planes[k] = arena_->ptr(layer, expert, k);
          return planes[0] && planes[1] && planes[2];
        };
      } else {
        lane->assist.graph = nullptr;
        LOG_WARN("%s layer=%d GPU assist unavailable\n", tag_, layer);
      }
    }
    EarlyReadyLane *raw = lane.get();
    raw->strategy = std::make_unique<EarlyReadyOffloadStrategy>(raw, adapter);
    owner.emplace(layer, std::move(lane));
    return raw;
  }

 private:
  std::shared_ptr<GpuAssistShared> ring(int idx, int device, int slots,
                                        const std::function<bool(int)> &ok) {
    if (auto have = rings_[idx].lock())
      return have;
    auto s = make_assist_ring(*arena_, first_, n_layers_, device, slots, ok);
    if (s)
      rings_[idx] = s;
    return s;
  }

  static void miss_cb(ggml_tensor *dst, const ggml_tensor *a, int ith, int,
                      void *ud) {
    if (ith != 0)
      return;
    auto *site = static_cast<OffloadMissSite *>(ud);
    auto &store = *site->runtime->store_;
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<int32_t> ids;
    ids.reserve((size_t)ggml_nelements(a));
    for (int64_t i1 = 0; i1 < a->ne[1]; ++i1)
      for (int64_t i0 = 0; i0 < a->ne[0]; ++i0)
        ids.push_back(*(const int32_t *)((const char *)a->data + i1 * a->nb[1] +
                                         i0 * a->nb[0]));
    zeromoe::offload::ExpertResidency::Ensure r;
    if (site->prefill)
      r = store.ensure_transient(site->layer, ids.data(), (int)ids.size());
    else
      r = store.ensure(site->layer, ids.data(), (int)ids.size(),
                       zeromoe::offload::ExpertCache::Kind::Demand);
    auto &ct = store.counters();
    ct.calls.fetch_add(1, std::memory_order_relaxed);
    ct.routes.fetch_add(ids.size(), std::memory_order_relaxed);
    ct.cold_routes.fetch_add(
        (uint64_t)std::count_if(ids.begin(), ids.end(),
                                [](int32_t v) { return v >= 0; }),
        std::memory_order_relaxed);
    ct.misses.fetch_add((uint64_t)r.misses, std::memory_order_relaxed);
    ct.loads.fetch_add((uint64_t)(site->prefill ? 0 : r.loads),
                       std::memory_order_relaxed);
    ct.transient_loads.fetch_add((uint64_t)(site->prefill ? r.loads : 0),
                                 std::memory_order_relaxed);
    ct.evictions.fetch_add((uint64_t)r.evictions, std::memory_order_relaxed);
    ct.load_us.fetch_add((uint64_t)(r.load_ms * 1000.0),
                         std::memory_order_relaxed);
    if (r.errors > 0) {
      ct.errors.fetch_add((uint64_t)r.errors, std::memory_order_relaxed);
      LOG_ERROR("%s layer %d: %d expert load error(s); aborting rather than "
                "computing with missing experts\n",
                site->runtime->tag_, site->layer, r.errors);
      store.report("fatal");
      std::abort();
    }
    std::memcpy(dst->data, ids.data(), ids.size() * sizeof(int32_t));
    ct.cb_us.fetch_add(
        (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - t0).count(),
        std::memory_order_relaxed);
  }

  std::unique_ptr<zeromoe::offload::ExpertStore> store_;
  zeromoe::offload::VirtualArena *arena_ = nullptr;
  int first_ = 0, n_layers_ = 0, n_experts_ = 0;
  const char *tag_ = "[OFFLOAD]";
  std::deque<OffloadMissSite> sites_;
  std::mutex mutex_;
  std::weak_ptr<GpuAssistShared> rings_[2];
};

} // namespace zeromoe::graph
