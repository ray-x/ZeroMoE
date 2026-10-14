#pragma once

// Model-agnostic DDR tier of the expert hierarchy: the ExpertCache policy, the
// virtual arena the cache populates, the SSD store / GGUF fallback readers and
// the residency logic that ties them together, behind one object.
//
// What it does NOT contain: anything that touches ggml or CUDA. Graph code
// (miss nodes, pilot staging) gets the tensor pointers from arena() and the
// locking discipline from ensure()/reader_lock(); pinning for DMA is injected
// as callbacks so this header stays buildable without the CUDA toolkit.
//
// Locking: arena_mu_ is exclusive for cache bookkeeping, evictions and loads
// (miss nodes) and shared for readers of arena slots (the pilot-staging worker).
// With async_loads the lock covers only the bookkeeping; the reads run outside
// it (ExpertResidency::begin/finish), so a background prefetch and a demand
// miss overlap instead of serializing on I/O.

#include "expert_cache.hpp"
#include "logger.hpp"
#include "offload_store.hpp"

#include <array>
#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <shared_mutex>
#include <string>
#include <vector>

namespace moestream::offload {

struct ExpertStoreConfig {
  int first_layer = 0;
  int n_layers = 0;
  int n_experts = 0;
  // Per layer (index = layer - first_layer): bytes of one expert's gate/up/down
  // planes. Every plane size must be a multiple of 4096 (O_DIRECT).
  std::vector<std::array<uint64_t, 3>> plane_bytes;
  std::string store_path;                 // optional expert-contiguous store
  std::vector<std::string> model_shards;  // fallback source, required
  double ddr_gib = 40.0;
  ExpertCache::Policy policy = ExpertCache::Policy::Lru;
  double pinned_fraction = 0.5;
  double probation_fraction = 0.25;
  bool byte_capacity = false;  // weigh entries by their real per-layer size
  bool huge_pages = false;
  bool async_loads = false;
  const char *log_tag = "[OFFLOAD]";
  // Register never-evicted experts for DMA. Both must be set to enable it.
  std::function<bool(void *, size_t)> pin_fn;
  std::function<void(void *)> unpin_fn;
};

class ExpertStore {
 public:
  struct Counters {
    std::atomic<uint64_t> calls{0}, routes{0}, misses{0}, loads{0}, evictions{0},
        errors{0}, load_us{0}, transient_loads{0}, cb_us{0}, cold_routes{0};
    std::atomic<uint64_t> last_report_calls{0};
  };

  static std::unique_ptr<ExpertStore> create(const ExpertStoreConfig &cfg,
                                             std::string *err) {
    std::unique_ptr<ExpertStore> s(new ExpertStore());
    s->cfg_ = cfg;
    if (!s->init(err))
      return nullptr;
    return s;
  }

  ~ExpertStore() {
    if (pin_slots_ && arena_)
      for (int l = cfg_.first_layer; l < cfg_.first_layer + cfg_.n_layers; ++l)
        for (int e = 0; e < cfg_.n_experts; ++e)
          unpin_expert(l, e);
  }

  // Seeds the cache from a hottest-first ranking and reads the seeded experts.
  // Returns the number of experts that failed to load.
  size_t seed(const std::vector<ExpertKey> &ranked) {
    cache_->seed(ranked);
    return residency_->populate_resident();
  }

  ExpertResidency::Ensure ensure(int layer, const int32_t *ids, int n,
                                 ExpertCache::Kind kind) {
    if (!cfg_.async_loads) {
      std::unique_lock<std::shared_mutex> lock(arena_mu_);
      return residency_->ensure(layer, ids, n, kind);
    }
    ExpertResidency::Plan p;
    {
      std::unique_lock<std::shared_mutex> lock(arena_mu_);
      p = residency_->begin(layer, ids, n, kind);
    }
    return residency_->finish(
        p, [this] { return std::unique_lock<std::shared_mutex>(arena_mu_); });
  }

  // The two halves of ensure() for a caller that wants to overlap the reads with
  // other work: begin_ensure() does the bookkeeping, finish_ensure() reads and
  // publishes (call it WITHOUT holding the arena lock; it may run on any thread).
  ExpertResidency::Plan begin_ensure(int layer, const int32_t *ids, int n,
                                     ExpertCache::Kind kind) {
    std::unique_lock<std::shared_mutex> lock(arena_mu_);
    return residency_->begin(layer, ids, n, kind);
  }
  ExpertResidency::Ensure finish_ensure(ExpertResidency::Plan &p) {
    return residency_->finish(
        p, [this] { return std::unique_lock<std::shared_mutex>(arena_mu_); });
  }

  // Prefill: every expert is touched once, so it bypasses the cache.
  ExpertResidency::Ensure ensure_transient(int layer, const int32_t *ids, int n) {
    std::unique_lock<std::shared_mutex> lock(arena_mu_);
    return residency_->ensure_transient(layer, ids, n);
  }

  // Prefill read-ahead of one layer (see ExpertResidency::prefetch_transient). No arena lock: prefill has no readers
  // and the reads only touch slots of `layer`.
  int prefetch_transient(int layer) { return residency_->prefetch_transient(layer); }

  // Shared access to arena slots (a worker reading experts another thread may
  // evict). Hold it for the whole read.
  std::shared_lock<std::shared_mutex> reader_lock() {
    return std::shared_lock<std::shared_mutex>(arena_mu_);
  }

  const ExpertStoreConfig &config() const { return cfg_; }
  int first_layer() const { return cfg_.first_layer; }
  int n_layers() const { return cfg_.n_layers; }
  int n_experts() const { return cfg_.n_experts; }
  size_t capacity() const { return capacity_; }
  bool async_loads() const { return cfg_.async_loads; }
  ExpertCache &cache() { return *cache_; }
  VirtualArena &arena() { return *arena_; }
  ExpertResidency &residency() { return *residency_; }
  OffloadStore &store() { return store_; }
  bool store_open() const { return store_open_; }
  Counters &counters() { return counters_; }

  bool is_pinned(int layer, int expert) const {
    return pin_slots_ && pinned_[idx(layer, expert)] != 0;
  }
  bool pinning() const { return pin_slots_; }
  uint64_t pin_calls() const { return pin_calls_.load(); }
  uint64_t pin_failures() const { return pin_fail_.load(); }

  void report(const char *why) {
    const Counters &o = counters_;
    LOG_INFO("%s %s: calls=%llu routes=%llu misses=%llu "
             "loads=%llu transient=%llu evictions=%llu errors=%llu load=%.1f ms "
             "callback=%.1f ms cold_routes=%llu\n",
             cfg_.log_tag, why, (unsigned long long)o.calls.load(),
             (unsigned long long)o.routes.load(),
             (unsigned long long)o.misses.load(),
             (unsigned long long)o.loads.load(),
             (unsigned long long)o.transient_loads.load(),
             (unsigned long long)o.evictions.load(),
             (unsigned long long)o.errors.load(), o.load_us.load() / 1000.0,
             o.cb_us.load() / 1000.0, (unsigned long long)o.cold_routes.load());
    LOG_INFO("%s %s sources: store direct=%llu store buffered=%llu model buffered=%llu\n", cfg_.log_tag, why,
             (unsigned long long)store_.direct_reads(), (unsigned long long)store_.buffered_reads(),
             (unsigned long long)model_.reads.load());
    if (residency_ && cfg_.async_loads)
      LOG_INFO("%s %s async: demand lookups that waited on an in-flight read=%llu wait=%.1f ms\n",
               cfg_.log_tag, why, (unsigned long long)residency_->waited_experts(),
               residency_->wait_ms());
    if (cache_ && cache_->config().policy == ExpertCache::Policy::SegmentedLru) {
      const auto &st = cache_->stats();
      LOG_INFO("%s %s segmented: prefetch_inserts=%llu promotions=%llu "
               "prefetch_evicted_unused=%llu probation=%zu/%zu\n",
               cfg_.log_tag, why, (unsigned long long)st.prefetch_inserts,
               (unsigned long long)st.promotions,
               (unsigned long long)st.prefetch_evicted_unused,
               cache_->probation_used(), cache_->used());
    }
  }

 private:
  ExpertStore() = default;

  size_t idx(int layer, int expert) const {
    return (size_t)(layer - cfg_.first_layer) * (size_t)cfg_.n_experts + (size_t)expert;
  }

  bool init(std::string *err) {
    const auto fail = [&](const std::string &m) {
      if (err)
        *err = m;
      return false;
    };
    if (cfg_.n_layers <= 0 || cfg_.n_experts <= 0 ||
        (int)cfg_.plane_bytes.size() != cfg_.n_layers)
      return fail("bad expert store geometry");
    uint64_t expert_bytes_sum = 0;
    for (const auto &pb : cfg_.plane_bytes)
      expert_bytes_sum += pb[0] + pb[1] + pb[2];

    arena_ = std::make_unique<VirtualArena>(cfg_.first_layer, cfg_.n_layers,
                                            cfg_.n_experts, cfg_.plane_bytes,
                                            /*no_huge_pages=*/!cfg_.huge_pages);
    if (!arena_->ok())
      return fail("virtual arena mmap failed");

    std::string e;
    std::vector<ExpertSource *> sources;
    if (!cfg_.store_path.empty()) {
      if (store_.open_auto(cfg_.store_path, &e)) {
        store_open_ = true;
        sources.push_back(&store_);
      } else {
        LOG_WARN("%s store %s unusable (%s); model only\n", cfg_.log_tag,
                 cfg_.store_path.c_str(), e.c_str());
      }
    }
    if (cfg_.model_shards.empty())
      return fail("no model shards given");
    if (!model_.open(cfg_.model_shards, &e))
      return fail("model reader: " + e);
    sources.push_back(&model_);

    ExpertCache::Config cc;
    cc.first_layer = cfg_.first_layer;
    cc.n_layers = cfg_.n_layers;
    cc.n_experts = cfg_.n_experts;
    capacity_ = (size_t)(cfg_.ddr_gib * 1024.0 * 1024.0 * 1024.0 /
                         ((double)expert_bytes_sum / (double)cfg_.n_layers));
    cc.capacity = capacity_;
    cc.policy = cfg_.policy;
    cc.pinned_fraction = cfg_.pinned_fraction;
    cc.probation_fraction = cfg_.probation_fraction;
    if (cfg_.byte_capacity) {
      // Weigh every entry by its real per-layer size instead of the average
      // (layers can differ in quant type).
      cc.layer_bytes.clear();
      for (const auto &pb : cfg_.plane_bytes)
        cc.layer_bytes.push_back(pb[0] + pb[1] + pb[2]);
      capacity_ = (size_t)(cfg_.ddr_gib * 1024.0 * 1024.0 * 1024.0);
      cc.capacity = capacity_;
    }
    cache_ = std::make_unique<ExpertCache>(cc);
    residency_ = std::make_unique<ExpertResidency>(*cache_, *arena_, sources);

    if (cfg_.pin_fn && cfg_.unpin_fn) {
      pin_slots_ = true;
      pinned_.assign((size_t)cfg_.n_layers * (size_t)cfg_.n_experts, 0);
      // Register only the cache's never-evicted experts: they are registered
      // once at load and never unregistered, so there is no
      // register/unregister churn (~0.4 ms per expert) on the LRU half. Those
      // experts keep the bounce path.
      arena_->after_populate = [this](int l, int x) {
        if (cache_ && cache_->pinned({l, x}))
          pin_expert(l, x);
      };
      arena_->before_evict = [this](int l, int x) { unpin_expert(l, x); };
      // Prefill-transient experts are registered by the read-ahead thread (off the critical path) so their host-to-device
      // copy skips the driver's pageable bounce buffer; unregistered on evict. Opt-in (MOESTREAM_OFFLOAD_PIN_TRANSIENT=1):
      // measured 87.5 tok/s with vs 99.0 without on the 805-token prompt (registration cost outweighs the faster copy).
      if (const char *e = std::getenv("MOESTREAM_OFFLOAD_PIN_TRANSIENT"); e && e[0] == '1')
        arena_->after_transient = [this](int l, int x) { pin_expert(l, x); };
    }
    return true;
  }

  void pin_expert(int layer, int expert) {
    uint8_t &f = pinned_[idx(layer, expert)];
    if (f)
      return;
    int ok = 0;
    for (int k = 0; k < 3; ++k)
      if (cfg_.pin_fn(arena_->ptr(layer, expert, k), (size_t)arena_->plane_bytes(layer, k)))
        ++ok;
      else
        break;
    if (ok == 3) {
      f = 1;
      pin_calls_.fetch_add(1, std::memory_order_relaxed);
    } else {
      pin_fail_.fetch_add(1, std::memory_order_relaxed);
      for (int k = 0; k < ok; ++k)
        cfg_.unpin_fn(arena_->ptr(layer, expert, k));  // partially registered: undo
    }
  }
  void unpin_expert(int layer, int expert) {
    uint8_t &f = pinned_[idx(layer, expert)];
    if (!f)
      return;
    for (int k = 0; k < 3; ++k)
      cfg_.unpin_fn(arena_->ptr(layer, expert, k));
    f = 0;
  }

  ExpertStoreConfig cfg_;
  size_t capacity_ = 0;
  // Declaration order is destruction order reversed: residency_ goes first and
  // the arena outlives everything that points into it.
  std::unique_ptr<ExpertCache> cache_;
  std::unique_ptr<VirtualArena> arena_;
  OffloadStore store_;
  bool store_open_ = false;
  ModelReader model_;
  std::unique_ptr<ExpertResidency> residency_;
  std::shared_mutex arena_mu_;
  bool pin_slots_ = false;
  std::vector<uint8_t> pinned_;  // [(layer - first) * n_experts + expert]
  std::atomic<uint64_t> pin_calls_{0}, pin_fail_{0};
  Counters counters_;
};

}  // namespace moestream::offload
