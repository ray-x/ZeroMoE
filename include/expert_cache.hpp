#pragma once
// Residency bookkeeping for a dynamic DDR expert cache (pure C++, no OS dependencies).
//
// A cache decides WHICH (layer, expert) slabs are resident in host memory; it does not own memory. The loader and the
// arena that actually hold the bytes live in offload_store.hpp. Splitting the policy from the I/O keeps this part
// testable against the offline simulator (bench/expert_cache_sim.py): the semantics here are deliberately identical to it.
//
// Semantics (the contract the tests pin down)
//   * Universe: layers [first_layer, first_layer + n_layers) x experts [0, n_experts). Keys outside it are not tracked.
//   * Policy::Static: the seeded set is the resident set; access() reports hit/miss and never changes state.
//   * Policy::Lru: a hit moves the entry to most-recent; a miss inserts it, first evicting the least recent NON-PINNED
//     entry when the cache already holds `capacity` entries. If nothing can be evicted (everything pinned) the miss is
//     not cached and counted as a bypass.
//   * seed(hottest_first): at most `capacity` entries become resident; the hottest floor(capacity * pinned_fraction)
//     are pinned (never evicted); among the rest, hotter means more recent.
//   * A residency bitmap per layer ((n_experts + 63) / 64 words, bit e set <=> expert e resident) is kept in sync so a
//     graph node can test a layer's routed ids without calling into the cache.
// Not thread-safe: use from the decode thread, or serialize externally.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

namespace moestream::offload {

struct ExpertKey {
  int32_t layer = -1;
  int32_t expert = -1;
  friend bool operator==(const ExpertKey &, const ExpertKey &) = default;
  friend auto operator<=>(const ExpertKey &, const ExpertKey &) = default;
};

class ExpertCache {
public:
  enum class Policy { Static, Lru };
  struct Config {
    int first_layer = 0;
    int n_layers = 0;
    int n_experts = 0;
    size_t capacity = 0;
    Policy policy = Policy::Lru;
    double pinned_fraction = 0.0;
  };
  struct Access {
    bool hit = false;
    bool inserted = false; // a miss that became resident
    bool evicted = false;
    ExpertKey victim{};
  };
  struct Stats {
    uint64_t hits = 0, misses = 0, evictions = 0, bypass = 0;
  };

  explicit ExpertCache(const Config &c) : cfg_(c) {
    universe_ = static_cast<size_t>(std::max(c.n_layers, 0)) * static_cast<size_t>(std::max(c.n_experts, 0));
    words_ = static_cast<size_t>((std::max(c.n_experts, 0) + 63) / 64);
    bits_.assign(static_cast<size_t>(std::max(c.n_layers, 0)) * words_, 0);
    prev_.assign(universe_, kNil);
    next_.assign(universe_, kNil);
    flags_.assign(universe_, 0);
  }

  const Config &config() const { return cfg_; }
  size_t capacity() const { return cfg_.capacity; }
  size_t size() const { return resident_; }
  const Stats &stats() const { return stats_; }
  void reset_stats() { stats_ = {}; }

  bool in_range(ExpertKey k) const {
    return k.layer >= cfg_.first_layer && k.layer < cfg_.first_layer + cfg_.n_layers && k.expert >= 0 &&
           k.expert < cfg_.n_experts;
  }
  bool resident(ExpertKey k) const { return in_range(k) && (flags_[idx(k)] & kResident); }
  bool pinned(ExpertKey k) const { return in_range(k) && (flags_[idx(k)] & kPinned); }

  // Residency bitmap of one layer (words_per_layer() words), or nullptr for an untracked layer.
  const uint64_t *layer_bitmap(int layer) const {
    if (layer < cfg_.first_layer || layer >= cfg_.first_layer + cfg_.n_layers) return nullptr;
    return bits_.data() + static_cast<size_t>(layer - cfg_.first_layer) * words_;
  }
  size_t words_per_layer() const { return words_; }
  // True when every non-negative id in ids[0..n) is resident (negative ids are skipped routes).
  bool all_resident(int layer, const int32_t *ids, int n) const {
    const uint64_t *b = layer_bitmap(layer);
    if (!b) return false;
    for (int i = 0; i < n; ++i) {
      int32_t e = ids[i];
      if (e < 0) continue;
      if (e >= cfg_.n_experts || !((b[e >> 6] >> (e & 63)) & 1ull)) return false;
    }
    return true;
  }

  // Make the first min(capacity, n) distinct in-range entries of `hottest_first` resident. Call on an empty cache.
  void seed(const std::vector<ExpertKey> &hottest_first) {
    std::vector<ExpertKey> take;
    take.reserve(std::min(hottest_first.size(), cfg_.capacity));
    for (const ExpertKey &k : hottest_first) {
      if (take.size() >= cfg_.capacity) break;
      if (!in_range(k) || resident(k)) continue;
      take.push_back(k);
      flags_[idx(k)] |= kResident; // provisional mark so duplicates are skipped; finalized below
    }
    for (const ExpertKey &k : take) flags_[idx(k)] &= static_cast<uint8_t>(~kResident);
    size_t pins = cfg_.policy == Policy::Lru
                      ? std::min(take.size(), static_cast<size_t>(static_cast<double>(cfg_.capacity) * cfg_.pinned_fraction))
                      : 0;
    for (size_t i = 0; i < take.size(); ++i) mark_resident(take[i]);
    for (size_t i = 0; i < pins; ++i) flags_[idx(take[i])] |= kPinned;
    if (cfg_.policy == Policy::Lru)
      for (size_t i = take.size(); i > pins; --i) push_mru(idx(take[i - 1])); // hottest of the unpinned ends up most recent
  }

  Access access(ExpertKey k) {
    Access r;
    if (!in_range(k)) return r;
    const size_t i = idx(k);
    if (flags_[i] & kResident) {
      ++stats_.hits;
      if (cfg_.policy == Policy::Lru && !(flags_[i] & kPinned)) {
        unlink(i);
        push_mru(i);
      }
      r.hit = true;
      return r;
    }
    ++stats_.misses;
    if (cfg_.policy == Policy::Static) return r;
    if (resident_ >= cfg_.capacity) {
      if (head_ == kNil) { // nothing evictable (cache empty of unpinned entries, or capacity 0)
        ++stats_.bypass;
        return r;
      }
      const size_t v = head_;
      unlink(v);
      r.victim = key_of(v);
      mark_evicted(v);
      r.evicted = true;
      ++stats_.evictions;
    }
    mark_resident(k);
    push_mru(i);
    r.inserted = true;
    return r;
  }

  // Drop an entry (e.g. its load failed). No-op if not resident.
  void erase(ExpertKey k) {
    if (!resident(k)) return;
    const size_t i = idx(k);
    if (!(flags_[i] & kPinned)) unlink(i);
    mark_evicted(i);
  }

  void clear() {
    std::fill(bits_.begin(), bits_.end(), 0ull);
    std::fill(prev_.begin(), prev_.end(), kNil);
    std::fill(next_.begin(), next_.end(), kNil);
    std::fill(flags_.begin(), flags_.end(), static_cast<uint8_t>(0));
    head_ = tail_ = kNil;
    resident_ = 0;
  }

  // Resident keys, least recent first for the LRU list, then pinned ones (diagnostics/tests).
  std::vector<ExpertKey> lru_order() const {
    std::vector<ExpertKey> out;
    for (size_t i = head_; i != kNil; i = next_[i]) out.push_back(key_of(i));
    return out;
  }

private:
  static constexpr size_t kNil = static_cast<size_t>(-1);
  static constexpr uint8_t kResident = 1, kPinned = 2;

  size_t idx(ExpertKey k) const {
    return static_cast<size_t>(k.layer - cfg_.first_layer) * static_cast<size_t>(cfg_.n_experts) + static_cast<size_t>(k.expert);
  }
  ExpertKey key_of(size_t i) const {
    return {static_cast<int32_t>(i / static_cast<size_t>(cfg_.n_experts)) + cfg_.first_layer,
            static_cast<int32_t>(i % static_cast<size_t>(cfg_.n_experts))};
  }
  void set_bit(ExpertKey k, bool on) {
    uint64_t &w = bits_[static_cast<size_t>(k.layer - cfg_.first_layer) * words_ + static_cast<size_t>(k.expert >> 6)];
    const uint64_t m = 1ull << (k.expert & 63);
    w = on ? (w | m) : (w & ~m);
  }
  void mark_resident(ExpertKey k) {
    flags_[idx(k)] |= kResident;
    set_bit(k, true);
    ++resident_;
  }
  void mark_evicted(size_t i) {
    flags_[i] = 0;
    set_bit(key_of(i), false);
    --resident_;
  }
  void push_mru(size_t i) {
    prev_[i] = tail_;
    next_[i] = kNil;
    if (tail_ != kNil) next_[tail_] = i; else head_ = i;
    tail_ = i;
  }
  void unlink(size_t i) {
    const size_t p = prev_[i], n = next_[i];
    if (p != kNil) next_[p] = n; else head_ = n;
    if (n != kNil) prev_[n] = p; else tail_ = p;
    prev_[i] = next_[i] = kNil;
  }

  Config cfg_;
  size_t universe_ = 0, words_ = 0, resident_ = 0;
  size_t head_ = kNil, tail_ = kNil; // head_ = least recent
  std::vector<uint64_t> bits_;
  std::vector<size_t> prev_, next_;
  std::vector<uint8_t> flags_;
  Stats stats_;
};

// Hottest-first ranking of the whole universe from a usage file ("layer expert count" per line, counts summed if a
// key repeats). Ties are broken by (layer, expert) ascending, exactly like the simulator's sorted(key=(-count, key)).
inline std::vector<ExpertKey> rank_hottest_first(const std::string &usage_file, int first_layer, int n_layers, int n_experts) {
  std::vector<uint64_t> count(static_cast<size_t>(n_layers) * static_cast<size_t>(n_experts), 0);
  std::ifstream in(usage_file);
  int l = 0, e = 0;
  uint64_t c = 0;
  while (in >> l >> e >> c) {
    if (l < first_layer || l >= first_layer + n_layers || e < 0 || e >= n_experts) continue;
    count[static_cast<size_t>(l - first_layer) * static_cast<size_t>(n_experts) + static_cast<size_t>(e)] += c;
  }
  std::vector<ExpertKey> keys;
  keys.reserve(count.size());
  for (int L = 0; L < n_layers; ++L)
    for (int E = 0; E < n_experts; ++E) keys.push_back({first_layer + L, E});
  std::stable_sort(keys.begin(), keys.end(), [&](const ExpertKey &a, const ExpertKey &b) {
    const uint64_t ca = count[static_cast<size_t>(a.layer - first_layer) * static_cast<size_t>(n_experts) + static_cast<size_t>(a.expert)];
    const uint64_t cb = count[static_cast<size_t>(b.layer - first_layer) * static_cast<size_t>(n_experts) + static_cast<size_t>(b.expert)];
    return ca != cb ? ca > cb : a < b;
  });
  return keys;
}

} // namespace moestream::offload
