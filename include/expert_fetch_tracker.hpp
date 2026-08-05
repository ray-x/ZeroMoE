// expert_fetch_tracker.hpp — measurement-only instrumentation answering a
// precise question: when a real dispatch resolves an expert via an
// INDEPENDENT cache hit (RAM-tier LRU, or the VRAM shadow-fast-path in
// execution_layer.cpp) -- i.e. a path that bypasses LockFreePrefetcher::
// consume_staged entirely -- had that exact (layer, expert) ALSO been
// recently touched by the async prefetcher? And if so, in what state:
// still being read (in flight), successfully fetched and STILL sitting
// ready in the staging pool unconsumed (the cleanest possible "pure
// redundancy" signal -- the fetch fully succeeded and was right there, a
// completely unrelated cache hit just got there first), or already
// evicted from staging before this hit even happened.
//
// This exists because prior investigation (see CLAUDE.md's prefetch
// history, and [PREFETCH-LAYER]'s hint-outcome line) established that only
// ~1-2% of successfully prefetched experts are ever consumed via
// consume_staged, despite ~75-84% prediction accuracy -- but nothing
// directly measured WHY the other ~98% went unused. This closes that gap:
// a flat, run-long, per-(layer,expert) record of the prefetcher's most
// recent activity, checked at the exact moment an independent cache hit
// would otherwise leave that question unanswered.
//
// Kept in its own tiny header (no engine.hpp/cuda_manager.hpp) so it can
// be included from async_prefetcher.hpp (the writer), AND from
// ram_manager.hpp / execution_layer.cpp (the readers, at the RAM-tier and
// VRAM shadow-fast-path hit sites) without risking the same circular-
// include class of problem pilot_confirmed_set.hpp was split out to avoid.
#pragma once
#include "perf_timer.hpp"  // perf() counters this tracker's readers record into -- lightweight, no engine.hpp/cuda_manager.hpp, safe to include here
#include <vector>
#include <atomic>
#include <chrono>
#include <cstdint>

enum class ExpertFetchState : uint8_t {
    None = 0,          // never fetched (or tracker not yet initialized)
    InFlight = 1,      // a worker thread is currently reading this expert's bytes
    StagedReady = 2,   // fetch completed successfully, sitting ready in staging, not yet consumed
    EvictedUnused = 3, // was StagedReady, but acquire_staging_slot recycled the slot before any miss consumed it
    Consumed = 4,      // was StagedReady, and a genuine cache miss consumed it via consume_staged (the intended-use case)
};

// Monotonic counter incremented exactly once per REAL forward pass
// (layers_forward_gguf, execution_layer.cpp -- once for prefill's single
// S>1 call, once per decode step's S=1 call), via advance_token_step().
// Exists to test a specific hypothesis about the "Cache-hit-beat-fetch"
// measurement below: are those events a SAME-TOKEN race (a fetch issued
// during THIS token's own layer-transition, beaten by something within
// that same narrow window -- step_delta==0), or an EARLIER token's
// abandoned fetch record being rediscovered much later by a completely
// unrelated token's own genuine cache hit (step_delta>0)? Wall-clock
// elapsed time alone (ExpertFetchTracker's existing timestamp) already
// hinted at the latter (tens of seconds, not milliseconds) -- this makes
// it a direct, unambiguous measurement instead of an inference from
// timing alone.
inline std::atomic<uint64_t>& current_token_step() {
    static std::atomic<uint64_t> v{0};
    return v;
}
inline void advance_token_step() {
    current_token_step().fetch_add(1, std::memory_order_relaxed);
}

// steady_clock microseconds since an arbitrary fixed epoch -- exposed as a
// free function (not just ExpertFetchTracker's private helper) so a caller
// with its OWN event to timestamp (e.g. RAMSafetyManager::get_experts_
// batch's Phase A, capturing the exact moment its own consume_staged
// check ran) can later compare that timestamp against a tracker mark's
// raw timestamp (ExpertFetchTracker::query_raw_mark_time below) on the
// SAME clock, for a precise before/after test -- see
// trace_phase_gap_if_staged_after_check's own comment for what this
// enables that elapsed-since-now (query()) alone can't.
inline uint64_t steady_now_us() {
    return (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

// Flat array indexed by (layer * n_experts + expert_id) -- bounded and small
// (a few tens of thousands of entries for any real model), so this is a
// plain vector of atomics, not a hash map: no locking needed, each cell is
// independently atomic, and there is no dynamic key space to manage.
// Mirrors the same flat-array idiom already used elsewhere in this project
// (dense_dispatch.hpp's cpu_affinity_layer_flags, async_prefetcher.hpp's
// layer_stats).
class ExpertFetchTracker {
    std::vector<std::atomic<uint64_t>> last_event_us_;
    std::vector<std::atomic<uint64_t>> last_event_step_;  // current_token_step() value at the moment of the last mark()
    std::vector<std::atomic<uint8_t>> state_;
    int n_experts_ = 0;

    static uint64_t now_us() { return steady_now_us(); }
    int idx(int layer, int expert) const { return layer * n_experts_ + expert; }

public:
    // Called once at startup (LockFreePrefetcher::start(), which already
    // knows n_layers/n_experts from the loaded model) -- never resized
    // after. A tracker with ready()==false (init() never called, e.g. a
    // test harness that doesn't wire this up) makes mark()/query() safe
    // no-ops rather than out-of-bounds accesses.
    void init(int n_layers, int n_experts) {
        n_experts_ = n_experts;
        last_event_us_ = std::vector<std::atomic<uint64_t>>((size_t)n_layers * (size_t)n_experts);
        last_event_step_ = std::vector<std::atomic<uint64_t>>((size_t)n_layers * (size_t)n_experts);
        state_ = std::vector<std::atomic<uint8_t>>((size_t)n_layers * (size_t)n_experts);
        for (auto& s : state_) s.store(0, std::memory_order_relaxed);
        for (auto& t : last_event_us_) t.store(0, std::memory_order_relaxed);
        for (auto& s : last_event_step_) s.store(0, std::memory_order_relaxed);
    }
    bool ready() const { return !state_.empty(); }

    void mark(int layer, int expert, ExpertFetchState st) {
        if (!ready()) return;
        int i = idx(layer, expert);
        if (i < 0 || i >= (int)state_.size()) return;
        state_[(size_t)i].store((uint8_t)st, std::memory_order_relaxed);
        last_event_us_[(size_t)i].store(now_us(), std::memory_order_relaxed);
        last_event_step_[(size_t)i].store(current_token_step().load(std::memory_order_relaxed), std::memory_order_relaxed);
    }

    // Returns the most recently recorded state for (layer, expert), how
    // long ago (microseconds) that state was last set, and how many
    // token-steps ago (out_step_delta -- see current_token_step's own
    // comment: 0 means the mark happened during THIS SAME forward pass,
    // consistent with a same-token race; >0 means it happened during an
    // EARLIER token's processing, consistent with a stale record being
    // rediscovered later). state==None (with both outputs 0) means no
    // prefetch activity has ever been recorded for this exact
    // (layer, expert) -- a genuinely independent cache hit with no
    // prefetch involvement at all.
    ExpertFetchState query(int layer, int expert, uint64_t& out_elapsed_us, uint64_t& out_step_delta) const {
        out_elapsed_us = 0;
        out_step_delta = 0;
        if (!ready()) return ExpertFetchState::None;
        int i = idx(layer, expert);
        if (i < 0 || i >= (int)state_.size()) return ExpertFetchState::None;
        uint8_t st = state_[(size_t)i].load(std::memory_order_relaxed);
        uint64_t at = last_event_us_[(size_t)i].load(std::memory_order_relaxed);
        uint64_t marked_step = last_event_step_[(size_t)i].load(std::memory_order_relaxed);
        uint64_t now_step = current_token_step().load(std::memory_order_relaxed);
        if (at != 0) out_elapsed_us = now_us() - at;
        if ((ExpertFetchState)st != ExpertFetchState::None && now_step >= marked_step) {
            out_step_delta = now_step - marked_step;
        }
        return (ExpertFetchState)st;
    }

    // Same lookup as query(), but returns the RAW mark timestamp
    // (steady_now_us() at the moment mark() last ran) instead of elapsed-
    // since-now -- for a caller holding its OWN earlier timestamp on the
    // same clock, needing a precise before/after comparison rather than
    // "how long ago relative to right now" (see
    // trace_phase_gap_if_staged_after_check below for the motivating use).
    // out_mark_at_us == 0 alongside a None return means never marked.
    ExpertFetchState query_raw_mark_time(int layer, int expert, uint64_t& out_mark_at_us) const {
        out_mark_at_us = 0;
        if (!ready()) return ExpertFetchState::None;
        int i = idx(layer, expert);
        if (i < 0 || i >= (int)state_.size()) return ExpertFetchState::None;
        uint8_t st = state_[(size_t)i].load(std::memory_order_relaxed);
        out_mark_at_us = last_event_us_[(size_t)i].load(std::memory_order_relaxed);
        return (ExpertFetchState)st;
    }
};

inline ExpertFetchTracker& expert_fetch_tracker() {
    static ExpertFetchTracker t;
    return t;
}

// Convenience combined check+record for an INDEPENDENT cache-hit site (one
// that resolves an expert WITHOUT ever calling consume_staged). Queries the
// tracker for (layer, expert) and, if it finds prefetch activity, records
// the matching n_{vram,ram}_hit_had_{inflight,staged,evicted}_fetch counter
// (perf_timer.hpp). Centralizes this instead of duplicating the same
// query+dispatch logic at all three real hit sites: execution_layer.cpp's
// VRAM shadow-fast-path, and ram_manager.hpp's get_expert Step 1 /
// get_experts_batch Phase A (both RAM-tier).
inline void record_hit_if_recently_fetched(int layer, int expert, bool is_vram_tier) {
    uint64_t elapsed_us = 0, step_delta = 0;
    ExpertFetchState st = expert_fetch_tracker().query(layer, expert, elapsed_us, step_delta);
    // Hypothesis test (see current_token_step's own comment): step_delta==0
    // means the fetch was marked during THIS SAME token's own forward pass
    // (consistent with a genuine same-transition race); step_delta>0 means
    // it was marked during an EARLIER token's processing (consistent with
    // a stale, abandoned record being rediscovered much later by an
    // unrelated hit). Tracked only for the two buckets that matter for
    // this question -- staged (the cleanest redundancy signal) and evicted
    // (double waste) -- in-flight is rare and doesn't bear on this
    // hypothesis either way.
    bool same_step = (step_delta == 0);
    switch (st) {
        case ExpertFetchState::InFlight:
            if (is_vram_tier) perf().n_vram_hit_had_inflight_fetch.fetch_add(1, std::memory_order_relaxed);
            else perf().n_ram_hit_had_inflight_fetch.fetch_add(1, std::memory_order_relaxed);
            break;
        case ExpertFetchState::StagedReady:
            if (is_vram_tier) {
                perf().n_vram_hit_had_staged_fetch.fetch_add(1, std::memory_order_relaxed);
                PerfCounters::atomic_add(perf().t_vram_hit_staged_fetch_elapsed_us_sum, (double)elapsed_us);
                PerfCounters::atomic_add(perf().t_vram_staged_fetch_step_delta_sum, (double)step_delta);
                if (same_step) perf().n_vram_staged_fetch_same_step.fetch_add(1, std::memory_order_relaxed);
            } else {
                perf().n_ram_hit_had_staged_fetch.fetch_add(1, std::memory_order_relaxed);
                PerfCounters::atomic_add(perf().t_ram_hit_staged_fetch_elapsed_us_sum, (double)elapsed_us);
                PerfCounters::atomic_add(perf().t_ram_staged_fetch_step_delta_sum, (double)step_delta);
                if (same_step) perf().n_ram_staged_fetch_same_step.fetch_add(1, std::memory_order_relaxed);
            }
            break;
        case ExpertFetchState::EvictedUnused:
            if (is_vram_tier) {
                perf().n_vram_hit_had_evicted_fetch.fetch_add(1, std::memory_order_relaxed);
                PerfCounters::atomic_add(perf().t_vram_evicted_fetch_step_delta_sum, (double)step_delta);
                if (same_step) perf().n_vram_evicted_fetch_same_step.fetch_add(1, std::memory_order_relaxed);
            } else {
                perf().n_ram_hit_had_evicted_fetch.fetch_add(1, std::memory_order_relaxed);
                PerfCounters::atomic_add(perf().t_ram_evicted_fetch_step_delta_sum, (double)step_delta);
                if (same_step) perf().n_ram_evicted_fetch_same_step.fetch_add(1, std::memory_order_relaxed);
            }
            break;
        case ExpertFetchState::Consumed:
        case ExpertFetchState::None:
        default:
            break;  // no prefetch involvement worth counting here
    }
}

// REMOVED: trace_ram_load_if_recently_fetched and
// trace_phase_gap_if_staged_after_check used to live here -- both measured
// races against LockFreePrefetcher's old separate staging pool (a genuine
// RAM-tier disk read happening for an expert the prefetcher ALSO touched;
// whether a Phase-B read raced a staging completion). Both mechanisms they
// measured are now structurally impossible: prefetch and the decode
// thread's own miss-handling claim/populate/publish through the SAME
// locked slot table (RAMSafetyManager::claim_or_wait, ram_manager.hpp), so
// there is no longer a separate staging pool or a Phase-A/Phase-B ordering
// gap for either function to detect. See ram_manager.hpp's claim_or_wait
// and perf_timer.hpp's "Unified RAM claim"/"Unified RAM wait" report lines
// for the replacement instrumentation.
