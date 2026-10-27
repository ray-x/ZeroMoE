#pragma once

// Promotion policy for a static per-layer VRAM expert set (no CUDA, no ggml).
//
// Every layer owns K_l device slots, each holding one routed expert. Routing
// that lands on an expert outside the set is a miss (the CPU computes it). This
// class counts misses and, once per step, plans which missed experts replace
// which resident ones. The caller performs the copies and the table updates;
// the protocol is the one measured in docs/design/expert-promotion.md:
//
//   plan()     the victim stops being resident immediately (the CPU computes it
//              while the copy runs); the slot is reserved for the incoming expert.
//   confirm()  called once the copy has landed: the incoming expert is resident.
//
// A reserved slot is never chosen as a victim and nothing is routed to it, so
// no graph ever reads a half-written slot.
//
// Policy (see the study): candidates are experts with >= min_misses misses;
// the best `cap` of them across all layers are promoted each step, with a
// larger budget for the first `burst_steps` steps of a request; the victim is
// the lowest `last_used + freq_weight * hits` slot of that layer, ZeroMoE's
// existing tiered-cache eviction score. enabled=false freezes the set.

#include <algorithm>
#include <cstdint>
#include <vector>

namespace zeromoe::graph {

struct PromotionConfig {
  bool enabled = true;
  int min_misses = 2;          // misses before an expert is a candidate
  int cap = 12;                // promotions per committed token (engine sweep: 6 -> 12 gives +11% on a high-miss prompt)
  int burst_extra = 20;        // extra promotions per step while bursting
  int burst_tokens = 40;       // tokens at the start of a request that burst
  // eviction score = last_used + freq_weight * hits. With real evictions the old constants (64, halve
  // every 4096 hits = every ~9 tokens) made the score pure recency; 4096 with no practical decay keeps
  // frequently used experts through topic changes (+3 points of CPU evals removed, 4 random orders).
  uint64_t freq_weight = 4096;
  uint32_t decay_every = 16'000'000;  // halve all hit counts after this many hits (~never in a session)
};

struct PromotionAction {
  int layer = -1;
  int expert_in = -1;   // becomes resident in `slot` once confirmed
  int expert_out = -1;  // stops being resident when the action is planned
  int slot = -1;
  uint16_t misses = 0;  // evidence behind the promotion
};

struct PromotionStats {
  uint64_t routed = 0, misses = 0, planned = 0, confirmed = 0;
};

class ExpertPromotionPolicy {
 public:
  // slot_expert[l][s] = expert held by slot s of layer l at startup.
  ExpertPromotionPolicy(const PromotionConfig &cfg, int n_experts,
                        const std::vector<std::vector<int>> &slot_expert)
      : cfg_(cfg), n_experts_(n_experts) {
    const size_t L = slot_expert.size();
    slots_.resize(L);
    slot_of_.assign(L, std::vector<int>(size_t(n_experts), -1));
    misses_.assign(L, std::vector<uint16_t>(size_t(n_experts), 0));
    incoming_.assign(L, std::vector<uint8_t>(size_t(n_experts), 0));
    for (size_t l = 0; l < L; ++l) {
      slots_[l].resize(slot_expert[l].size());
      for (size_t s = 0; s < slot_expert[l].size(); ++s) {
        const int e = slot_expert[l][s];
        slots_[l][s].expert = e;
        if (e >= 0 && e < n_experts) slot_of_[l][size_t(e)] = int(s);
      }
    }
  }

  // A new request starts: restart the burst and age the miss evidence.
  void begin_request() {
    step_ = 0;  // tokens committed in this request
    for (auto &row : misses_)
      for (auto &m : row) m >>= 1;
  }

  // One routed expert of one token in `layer` (hit or miss).
  void note_routed(int layer, int expert) {
    ++stats_.routed;
    ++clock_;
    const int s = slot_of_[size_t(layer)][size_t(expert)];
    if (s >= 0) {
      Slot &slot = slots_[size_t(layer)][size_t(s)];
      slot.last_used = clock_;
      ++slot.hits;
      if (++hits_since_decay_ >= cfg_.decay_every) decay();
    } else {
      ++stats_.misses;
      uint16_t &m = misses_[size_t(layer)][size_t(expert)];
      if (m < UINT16_MAX) ++m;
    }
  }

  // End of a step that committed `tokens` tokens (1 for plain decode, ~3 for an MTP window): choose
  // promotions. Victims stop being resident now.
  std::vector<PromotionAction> plan(int tokens = 1) {
    std::vector<PromotionAction> out;
    if (!cfg_.enabled) return out;
    const int budget = cfg_.cap * tokens + (step_ < uint64_t(cfg_.burst_tokens) ? cfg_.burst_extra : 0);
    step_ += uint64_t(tokens);
    struct Cand { uint16_t misses; int layer, expert; };
    std::vector<Cand> pool;
    for (size_t l = 0; l < slots_.size(); ++l)
      for (int e = 0; e < n_experts_; ++e) {
        const uint16_t m = misses_[l][size_t(e)];
        if (m >= uint16_t(cfg_.min_misses) && slot_of_[l][size_t(e)] < 0 &&
            !incoming_[l][size_t(e)])
          pool.push_back({m, int(l), e});
      }
    std::stable_sort(pool.begin(), pool.end(),
                     [](const Cand &a, const Cand &b) { return a.misses > b.misses; });
    for (const Cand &c : pool) {
      if (int(out.size()) >= budget) break;
      auto &layer_slots = slots_[size_t(c.layer)];
      int victim = -1;
      uint64_t best = UINT64_MAX;
      for (size_t s = 0; s < layer_slots.size(); ++s) {
        const Slot &slot = layer_slots[s];
        if (slot.pending) continue;
        const uint64_t score = slot.last_used + cfg_.freq_weight * slot.hits;
        if (score < best) { best = score; victim = int(s); }
      }
      if (victim < 0) continue;
      Slot &slot = layer_slots[size_t(victim)];
      PromotionAction a;
      a.layer = c.layer; a.expert_in = c.expert; a.expert_out = slot.expert;
      a.slot = victim; a.misses = c.misses;
      if (slot.expert >= 0) slot_of_[size_t(c.layer)][size_t(slot.expert)] = -1;
      slot.pending = true;
      slot.incoming = c.expert;
      incoming_[size_t(c.layer)][size_t(c.expert)] = 1;
      misses_[size_t(c.layer)][size_t(c.expert)] = 0;
      ++stats_.planned;
      out.push_back(a);
    }
    return out;
  }

  // The copy for `a` has landed: the incoming expert is now resident.
  void confirm(const PromotionAction &a) {
    Slot &slot = slots_[size_t(a.layer)][size_t(a.slot)];
    slot.expert = a.expert_in;
    slot.pending = false;
    slot.incoming = -1;
    slot.last_used = clock_;
    slot.hits = a.misses;
    slot_of_[size_t(a.layer)][size_t(a.expert_in)] = a.slot;
    incoming_[size_t(a.layer)][size_t(a.expert_in)] = 0;
    ++stats_.confirmed;
  }

  bool resident(int layer, int expert) const {
    return slot_of_[size_t(layer)][size_t(expert)] >= 0;
  }
  int slot_of(int layer, int expert) const { return slot_of_[size_t(layer)][size_t(expert)]; }
  int slots(int layer) const { return int(slots_[size_t(layer)].size()); }
  int expert_in_slot(int layer, int slot) const { return slots_[size_t(layer)][size_t(slot)].expert; }
  bool slot_pending(int layer, int slot) const { return slots_[size_t(layer)][size_t(slot)].pending; }
  uint64_t step() const { return step_; }
  const PromotionStats &stats() const { return stats_; }
  const PromotionConfig &config() const { return cfg_; }

 private:
  struct Slot {
    int expert = -1;
    uint64_t last_used = 0;
    uint32_t hits = 0;
    bool pending = false;
    int incoming = -1;
  };
  void decay() {
    hits_since_decay_ = 0;
    for (auto &layer : slots_)
      for (auto &s : layer) s.hits >>= 1;
  }

  PromotionConfig cfg_;
  int n_experts_;
  std::vector<std::vector<Slot>> slots_;
  std::vector<std::vector<int>> slot_of_;         // layer -> expert -> slot or -1
  std::vector<std::vector<uint16_t>> misses_;     // layer -> expert -> misses since promoted/aged
  std::vector<std::vector<uint8_t>> incoming_;    // expert has a copy in flight
  uint64_t clock_ = 0, hits_since_decay_ = 0, step_ = 0;
  PromotionStats stats_;
};

}  // namespace zeromoe::graph
