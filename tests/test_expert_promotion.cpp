// CPU-only proof of the expert promotion policy (no CUDA, no ggml).
//
//   c++ -std=c++23 -O2 -Imodels/graph tests/test_expert_promotion.cpp -o /tmp/zeromoe-expert-promotion

#include "expert_promotion.hpp"

#include <cstdio>
#include <random>
#include <set>

using namespace zeromoe::graph;

namespace {

constexpr int L = 4, E = 64, K = 16;

std::vector<std::vector<int>> initial_slots() {
  std::vector<std::vector<int>> s(L);
  for (int l = 0; l < L; ++l)
    for (int i = 0; i < K; ++i) s[size_t(l)].push_back(i);  // experts 0..K-1 resident
  return s;
}

// Every invariant that must hold between any two calls.
bool invariants_hold(const ExpertPromotionPolicy &p) {
  for (int l = 0; l < L; ++l) {
    std::set<int> seen;
    int resident = 0, pending = 0;
    for (int s = 0; s < p.slots(l); ++s) {
      if (p.slot_pending(l, s)) { ++pending; continue; }
      const int e = p.expert_in_slot(l, s);
      if (e < 0 || !seen.insert(e).second) return false;     // duplicate expert in two slots
      if (p.slot_of(l, e) != s) return false;                // mapping disagrees with slot
      ++resident;
    }
    int counted = 0;
    for (int e = 0; e < E; ++e) counted += p.resident(l, e);
    if (counted != resident) return false;                   // resident() must equal the slots
    if (resident + pending != p.slots(l)) return false;      // slot count is conserved
  }
  return true;
}

}  // namespace

int main() {
  int failures = 0;
  const auto check = [&](bool ok, const char *name) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", name); ++failures; }
  };

  // 1) Random routing, random confirm delays: invariants hold and the budget is respected.
  {
    PromotionConfig cfg;
    ExpertPromotionPolicy p(cfg, E, initial_slots());
    std::mt19937 rng(7);
    std::vector<PromotionAction> inflight;
    bool ok = true, budget_ok = true;
    for (int step = 0; step < 400; ++step) {
      for (int t = 0; t < 3; ++t)
        for (int l = 0; l < L; ++l)
          for (int k = 0; k < 10; ++k) p.note_routed(l, int(rng() % E));
      const auto actions = p.plan();
      const int budget = cfg.cap + (step < cfg.burst_tokens ? cfg.burst_extra : 0);
      budget_ok = budget_ok && int(actions.size()) <= budget;
      for (const auto &a : actions) {
        ok = ok && !p.resident(a.layer, a.expert_in) && p.slot_pending(a.layer, a.slot);
        ok = ok && !p.resident(a.layer, a.expert_out);  // victim stops being resident at plan time
        inflight.push_back(a);
      }
      ok = ok && invariants_hold(p);
      // confirm a random subset (copies land at different times)
      for (size_t i = 0; i < inflight.size();) {
        if (rng() % 2) { p.confirm(inflight[i]); inflight.erase(inflight.begin() + long(i)); }
        else ++i;
      }
      ok = ok && invariants_hold(p);
    }
    check(ok, "invariants hold through plan/confirm with delayed copies");
    check(budget_ok, "promotions per step never exceed cap (+ burst)");
  }

  // 2) A hidden topic of experts: misses fall by more than half after the policy adapts.
  {
    PromotionConfig cfg;
    cfg.burst_extra = 0;
    ExpertPromotionPolicy p(cfg, E, initial_slots());
    std::mt19937 rng(3);
    uint64_t early_miss = 0, early_tot = 0, late_miss = 0, late_tot = 0;
    for (int step = 0; step < 300; ++step) {
      for (int l = 0; l < L; ++l)
        for (int k = 0; k < 10; ++k) {
          const int e = 40 + int(rng() % 8);  // topic: experts 40..47, none resident at start
          const bool hit = p.resident(l, e);
          p.note_routed(l, e);
          (step < 2 ? early_tot : late_tot) += 1;
          (step < 2 ? early_miss : late_miss) += !hit;
        }
      for (const auto &a : p.plan()) p.confirm(a);
    }
    check(early_miss > early_tot / 2, "topic starts mostly cold");
    check(late_miss * 4 < late_tot, "topic becomes resident (late miss rate < 25%)");
  }

  // 3) Victim choice: never-used slots go before used ones; reserved slots are not reused.
  {
    PromotionConfig cfg;
    cfg.cap = 1; cfg.burst_extra = 0; cfg.min_misses = 1;
    ExpertPromotionPolicy p(cfg, E, initial_slots());
    for (int e = 0; e < K - 1; ++e) p.note_routed(0, e);  // use all but expert K-1 in layer 0
    p.note_routed(0, 50);                                  // one miss
    auto a = p.plan();
    check(a.size() == 1 && a[0].expert_out == K - 1, "the never-used slot is the victim");
    for (int i = 0; i < 3; ++i) p.note_routed(0, 51);
    auto b = p.plan();                                     // first copy still in flight
    check(b.size() == 1 && b[0].slot != a[0].slot, "a reserved slot is not chosen again");
    p.confirm(a[0]);
    check(p.resident(0, 50) && p.slot_of(0, 50) == a[0].slot, "confirm makes the incoming expert resident");
  }

  // 4) Burst: more promotions in the first steps of a request than later; begin_request re-arms it.
  {
    PromotionConfig cfg;
    cfg.min_misses = 1; cfg.cap = 2; cfg.burst_extra = 10; cfg.burst_tokens = 3;
    ExpertPromotionPolicy p(cfg, E, initial_slots());
    auto count_step = [&]() {
      for (int l = 0; l < L; ++l)
        for (int e = 20; e < 40; ++e) p.note_routed(l, e);  // many distinct misses
      const auto acts = p.plan();
      for (const auto &a : acts) p.confirm(a);
      return int(acts.size());
    };
    const int first = count_step();
    count_step(); count_step();
    const int later = count_step();
    check(first > later && later <= cfg.cap, "burst budget applies to the first steps only");
    p.begin_request();
    check(count_step() > cfg.cap, "begin_request re-arms the burst");
  }

  // 4b) Effectiveness counters: a promotion that is used counts as a promoted hit, one evicted unused is wasted.
  {
    PromotionConfig cfg;
    cfg.cap = 1; cfg.burst_extra = 0; cfg.min_misses = 1;
    ExpertPromotionPolicy p(cfg, E, initial_slots());
    p.note_routed(0, 50);
    auto a = p.plan();
    p.confirm(a[0]);
    p.note_routed(0, 50);  // used once after the copy landed
    check(p.stats().hits_promoted == 1 && p.stats().hits_static == 0, "promoted hit counted separately");
    p.note_routed(0, 3);
    check(p.stats().hits_static == 1, "static hit counted");
    p.note_routed(0, 51);  // another miss: evicts something; make the new promotion get evicted unused
    auto b = p.plan();
    p.confirm(b[0]);
    for (int e = 0; e < K; ++e) p.note_routed(0, e);  // touch all static residents so the new one is the victim
    p.note_routed(0, 52);
    auto c = p.plan();
    check(!c.empty() && c[0].expert_out == 51 && p.stats().wasted == 1, "unused promotion evicted counts as wasted");
  }

  // 5) Frozen: enabled=false plans nothing and the set never changes.
  {
    PromotionConfig cfg; cfg.enabled = false;
    ExpertPromotionPolicy p(cfg, E, initial_slots());
    for (int i = 0; i < 100; ++i) for (int l = 0; l < L; ++l) p.note_routed(l, 50);
    check(p.plan().empty() && !p.resident(0, 50) && p.resident(0, 0), "frozen mode never promotes");
  }

  std::printf("expert promotion policy: %s\n", failures ? "FAIL" : "PASS");
  return failures ? 1 : 0;
}
