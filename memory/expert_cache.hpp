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
//   * Policy::SegmentedLru: two LRU lists besides the pinned set. Experts that were actually used (a demand miss, or a
//     prefetched expert that was then used) live in the PROTECTED list; experts that were only predicted (a Kind::Prefetch
//     miss) live in the PROBATION list. A demand hit on a probationary entry promotes it to protected-MRU; a prefetch
//     access to a resident entry changes nothing. Eviction takes the probation LRU end while probation holds more than
//     probation_fraction * capacity (or protected is empty), otherwise the protected LRU end, so a burst of wrong
//     predictions can only displace other predictions, not the experts that were really reused.
//   * Capacity is counted in entries by default. With layer_bytes (one weight per layer) `capacity` is a byte budget and
//     every entry weighs layer_bytes[layer - first_layer]; one access may then evict several entries (Access::victims).
//   * seed(hottest_first): entries become resident while the weight fits `capacity`; the hottest entries whose weight
//     fits floor(capacity * pinned_fraction) are pinned (never evicted); among the rest, hotter means more recent.
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
  enum class Policy { Static, Lru, SegmentedLru };
  enum class Kind { Demand, Prefetch };
  struct Config {
    int first_layer = 0;
    int n_layers = 0;
    int n_experts = 0;
    size_t capacity = 0; // entries, or bytes when layer_bytes is set
    Policy policy = Policy::Lru;
    double pinned_fraction = 0.0;
    double probation_fraction = 0.25;   // SegmentedLru only: the probation list's share of capacity
    std::vector<uint64_t> layer_bytes;  // optional per-layer entry weight (size n_layers); empty = every entry weighs 1
  };
  struct Access {
    bool hit = false;
    bool inserted = false; // a miss that became resident
    bool evicted = false;
    bool promoted = false; // a demand hit that moved a probationary entry to protected
    ExpertKey victim{};    // first victim (compatibility); all victims are in `victims`
    std::vector<ExpertKey> victims;
  };
  struct Stats {
    uint64_t hits = 0, misses = 0, evictions = 0, bypass = 0;
    uint64_t promotions = 0;          // probationary -> protected on a demand hit (prefetches that were used)
    uint64_t prefetch_inserts = 0;    // Kind::Prefetch misses that became resident
    uint64_t prefetch_evicted_unused = 0; // probationary entries evicted before any demand hit
  };

  explicit ExpertCache(const Config &c) : cfg_(c) {
    universe_ = static_cast<size_t>(std::max(c.n_layers, 0)) * static_cast<size_t>(std::max(c.n_experts, 0));
    words_ = static_cast<size_t>((std::max(c.n_experts, 0) + 63) / 64);
    bits_.assign(static_cast<size_t>(std::max(c.n_layers, 0)) * words_, 0);
    prev_.assign(universe_, kNil);
    next_.assign(universe_, kNil);
    flags_.assign(universe_, 0);
    if (!cfg_.layer_bytes.empty() && cfg_.layer_bytes.size() != static_cast<size_t>(std::max(c.n_layers, 0)))
      cfg_.layer_bytes.clear(); // wrong shape: fall back to entry counting rather than mis-weighing
  }

  const Config &config() const { return cfg_; }
  size_t capacity() const { return cfg_.capacity; }
  size_t size() const { return resident_; }
  size_t used() const { return used_; }                 // weight of all resident entries (pinned included)
  size_t probation_used() const { return prob_used_; }  // weight held by the probation list
  const Stats &stats() const { return stats_; }
  void reset_stats() { stats_ = {}; }

  bool in_range(ExpertKey k) const {
    return k.layer >= cfg_.first_layer && k.layer < cfg_.first_layer + cfg_.n_layers && k.expert >= 0 &&
           k.expert < cfg_.n_experts;
  }
  bool resident(ExpertKey k) const { return in_range(k) && (flags_[idx(k)] & kResident); }
  bool pinned(ExpertKey k) const { return in_range(k) && (flags_[idx(k)] & kPinned); }
  bool probationary(ExpertKey k) const { return in_range(k) && (flags_[idx(k)] & kProb); }

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

  // Make the leading distinct in-range entries of `hottest_first` resident while their weight fits capacity. Call on an
  // empty cache.
  void seed(const std::vector<ExpertKey> &hottest_first) {
    std::vector<ExpertKey> take;
    size_t weight = 0;
    for (const ExpertKey &k : hottest_first) {
      if (!in_range(k) || (flags_[idx(k)] & kTaken)) continue;
      const size_t w = weight_of(k.layer);
      if (weight + w > cfg_.capacity) break;
      take.push_back(k);
      weight += w;
      flags_[idx(k)] |= kTaken; // provisional mark so duplicates are skipped; cleared below
    }
    for (const ExpertKey &k : take) flags_[idx(k)] &= static_cast<uint8_t>(~kTaken);
    size_t pins = 0;
    if (cfg_.policy != Policy::Static) {
      const size_t pin_budget = static_cast<size_t>(static_cast<double>(cfg_.capacity) * cfg_.pinned_fraction);
      size_t pw = 0;
      while (pins < take.size() && pw + weight_of(take[pins].layer) <= pin_budget) pw += weight_of(take[pins++].layer);
    }
    for (size_t i = 0; i < take.size(); ++i) mark_resident(take[i]);
    for (size_t i = 0; i < pins; ++i) flags_[idx(take[i])] |= kPinned;
    if (cfg_.policy != Policy::Static)
      for (size_t i = take.size(); i > pins; --i) push_mru(prot_, idx(take[i - 1])); // hottest of the unpinned ends up most recent
  }

  Access access(ExpertKey k, Kind kind = Kind::Demand) {
    Access r;
    if (!in_range(k)) return r;
    const size_t i = idx(k);
    const bool segmented = cfg_.policy == Policy::SegmentedLru;
    if (flags_[i] & kResident) {
      ++stats_.hits;
      if (cfg_.policy != Policy::Static && !(flags_[i] & kPinned)) {
        if (segmented && (flags_[i] & kProb)) {
          if (kind == Kind::Demand) {
            unlink(prob_, i);
            flags_[i] &= static_cast<uint8_t>(~kProb);
            prob_used_ -= weight_of(k.layer);
            push_mru(prot_, i);
            r.promoted = true;
            ++stats_.promotions;
          }
        } else if (kind == Kind::Demand || !segmented) {
          unlink(prot_, i);
          push_mru(prot_, i);
        }
      }
      r.hit = true;
      return r;
    }
    ++stats_.misses;
    if (cfg_.policy == Policy::Static) return r;
    const size_t w = weight_of(k.layer);
    if (w > cfg_.capacity) { ++stats_.bypass; return r; }
    while (used_ + w > cfg_.capacity) {
      const size_t v = pick_victim();
      if (v == kNil) { // nothing evictable (everything pinned, or capacity 0)
        ++stats_.bypass;
        return r;
      }
      const ExpertKey vk = key_of(v);
      if (flags_[v] & kProb) {
        unlink(prob_, v);
        prob_used_ -= weight_of(vk.layer);
        ++stats_.prefetch_evicted_unused;
      } else {
        unlink(prot_, v);
      }
      mark_evicted(v);
      if (!r.evicted) r.victim = vk;
      r.evicted = true;
      r.victims.push_back(vk);
      ++stats_.evictions;
    }
    mark_resident(k);
    if (segmented && kind == Kind::Prefetch) {
      flags_[i] |= kProb;
      prob_used_ += w;
      push_mru(prob_, i);
      ++stats_.prefetch_inserts;
    } else {
      push_mru(prot_, i);
    }
    r.inserted = true;
    return r;
  }

  // Drop an entry (e.g. its load failed). No-op if not resident.
  void erase(ExpertKey k) {
    if (!resident(k)) return;
    const size_t i = idx(k);
    if (flags_[i] & kProb) {
      unlink(prob_, i);
      prob_used_ -= weight_of(k.layer);
    } else if (!(flags_[i] & kPinned)) {
      unlink(prot_, i);
    }
    mark_evicted(i);
  }

  void clear() {
    std::fill(bits_.begin(), bits_.end(), 0ull);
    std::fill(prev_.begin(), prev_.end(), kNil);
    std::fill(next_.begin(), next_.end(), kNil);
    std::fill(flags_.begin(), flags_.end(), static_cast<uint8_t>(0));
    prot_ = prob_ = List{};
    resident_ = used_ = prob_used_ = 0;
  }

  // Resident unpinned keys of the protected list, least recent first (diagnostics/tests).
  std::vector<ExpertKey> lru_order() const {
    std::vector<ExpertKey> out;
    for (size_t i = prot_.head; i != kNil; i = next_[i]) out.push_back(key_of(i));
    return out;
  }
  // Probation list, least recent first (diagnostics/tests).
  std::vector<ExpertKey> probation_order() const {
    std::vector<ExpertKey> out;
    for (size_t i = prob_.head; i != kNil; i = next_[i]) out.push_back(key_of(i));
    return out;
  }

private:
  static constexpr size_t kNil = static_cast<size_t>(-1);
  static constexpr uint8_t kResident = 1, kPinned = 2, kProb = 4, kTaken = 8;
  struct List { size_t head = kNil, tail = kNil; }; // head = least recent

  size_t weight_of(int layer) const {
    return cfg_.layer_bytes.empty() ? 1 : static_cast<size_t>(cfg_.layer_bytes[static_cast<size_t>(layer - cfg_.first_layer)]);
  }
  size_t idx(ExpertKey k) const {
    return static_cast<size_t>(k.layer - cfg_.first_layer) * static_cast<size_t>(cfg_.n_experts) + static_cast<size_t>(k.expert);
  }
  ExpertKey key_of(size_t i) const {
    return {static_cast<int32_t>(i / static_cast<size_t>(cfg_.n_experts)) + cfg_.first_layer,
            static_cast<int32_t>(i % static_cast<size_t>(cfg_.n_experts))};
  }
  // Victim choice: probation first while it is over its quota (or protected is empty), else the protected LRU end.
  size_t pick_victim() const {
    if (cfg_.policy == Policy::SegmentedLru && prob_.head != kNil) {
      const size_t quota = static_cast<size_t>(static_cast<double>(cfg_.capacity) * cfg_.probation_fraction);
      if (prob_used_ > quota || prot_.head == kNil) return prob_.head;
    }
    return prot_.head;
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
    used_ += weight_of(k.layer);
  }
  void mark_evicted(size_t i) {
    const ExpertKey k = key_of(i);
    flags_[i] = 0;
    set_bit(k, false);
    --resident_;
    used_ -= weight_of(k.layer);
  }
  void push_mru(List &l, size_t i) {
    prev_[i] = l.tail;
    next_[i] = kNil;
    if (l.tail != kNil) next_[l.tail] = i; else l.head = i;
    l.tail = i;
  }
  void unlink(List &l, size_t i) {
    const size_t p = prev_[i], n = next_[i];
    if (p != kNil) next_[p] = n; else l.head = n;
    if (n != kNil) prev_[n] = p; else l.tail = p;
    prev_[i] = next_[i] = kNil;
  }

  Config cfg_;
  size_t universe_ = 0, words_ = 0, resident_ = 0, used_ = 0, prob_used_ = 0;
  List prot_, prob_;
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
