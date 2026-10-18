// pilot_confirmed_set.hpp — measurement-only instrumentation for a specific
// hypothesized eviction race, kept in its own tiny header (no engine.hpp /
// cuda_manager.hpp / etc.) so it can be included from BOTH the pilot loop's
// writer (execution_layer.cpp, via dense_dispatch.hpp) and the eviction
// readers (gpu_topology.hpp's evict_lru, ram_manager.hpp's RAM-tier
// eviction) without a circular include: gpu_topology.hpp cannot include
// dense_dispatch.hpp directly, since dense_dispatch.hpp pulls in
// cuda_manager.hpp, which holds a `GpuTopology topology;` BY-VALUE member --
// requiring GpuTopology's complete definition, which isn't done yet if
// gpu_topology.hpp is the one doing the including.
//
// The race this measures: the pilot's top-k loop (execution_layer.cpp)
// checks residency one layer ahead and skips pushing a fetch hint for any
// expert it finds ALREADY resident (RAM or VRAM-unpinned) -- see
// n_pilot_pred_already_ram/vram_lru, perf_timer.hpp. That decision is made
// once, at prediction time. Nothing stops an ORDINARY LRU eviction
// (GpuTopology::evict_lru, or RAMSafetyManager's own RAM-tier eviction) from
// picking that exact expert as a victim moments later, before decode's real
// dispatch ever reaches it -- eviction_score() (pure recency+frequency) has
// no concept of "the pilot just confirmed this is about to be needed." If
// that race is real, the evicted expert forces a genuine cache miss (and a
// synchronous disk re-read) despite having been correctly identified as
// safe/needed just before.
#pragma once
#include <mutex>
#include <algorithm>

// Tracks only the MOST RECENT pilot transition's "confirmed resident,
// skip-fetch" picks (typically <= pilot_k, default 8) -- NOT a rolling
// window across many transitions, deliberately: the race in question is
// specifically "confirmed resident THIS transition, evicted before the very
// next layer's dispatch" -- a wider window would blur this specific race
// with ordinary unrelated churn. Replaced wholesale (not merged) each time
// publish_pilot_confirmed_resident() runs, so a stale entry from several
// transitions ago can never still match.
struct PilotConfirmedEntry { int layer_idx; int expert_id; };
inline constexpr int PILOT_CONFIRMED_CAP = 24;  // >= 3 contributors x max realistic pilot_k (8): DEPTH-1 + DEPTH-2 + ATTN-PILOT; generous, not a hard model limit
struct PilotConfirmedSet {
    std::mutex m;
    PilotConfirmedEntry entries[PILOT_CONFIRMED_CAP];
    int count = 0;
};
inline PilotConfirmedSet& pilot_confirmed_resident_set() {
    static PilotConfirmedSet s;
    return s;
}
// Called once per layer transition from execution_layer.cpp's pilot loop,
// replacing the entire tracked set with this transition's already-resident
// (skip-fetch) picks. `n` capped to PILOT_CONFIRMED_CAP defensively --
// pilot_k is config-bounded but this avoids any risk of an out-of-range
// write if that bound is ever raised without updating this cap too.
inline void publish_pilot_confirmed_resident(const PilotConfirmedEntry* picks, int n) {
    auto& s = pilot_confirmed_resident_set();
    std::lock_guard<std::mutex> lock(s.m);
    s.count = std::min(n, PILOT_CONFIRMED_CAP);
    for (int i = 0; i < s.count; ++i) s.entries[i] = picks[i];
}
// Called from an eviction victim-selection site (gpu_topology.hpp's
// evict_lru, ram_manager.hpp's RAM-tier eviction) right when a real,
// non-empty victim is chosen -- true if that exact (layer, expert) was in
// the MOST RECENT pilot transition's confirmed-resident set.
inline bool was_pilot_confirmed_resident(int layer_idx, int expert_id) {
    auto& s = pilot_confirmed_resident_set();
    std::lock_guard<std::mutex> lock(s.m);
    for (int i = 0; i < s.count; ++i) {
        if (s.entries[i].layer_idx == layer_idx && s.entries[i].expert_id == expert_id) return true;
    }
    return false;
}
