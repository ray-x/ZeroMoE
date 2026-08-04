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

// Traces a GENUINE disk-read RAM-tier population -- any real cache-MISS
// load site (a synchronous pread/io_uring read that actually happens, NOT
// a cache hit) -- against the SAME expert's prefetch-tracker record, if
// any. Complements record_hit_if_recently_fetched above: that function
// fires on a HIT that bypasses consume_staged entirely; this one fires at
// a genuine MISS-driven disk read, directly answering "is this specific
// disk read duplicating work the prefetcher also did (or is still doing)
// for the exact same bytes, and which code path triggered this load."
//
// `trigger` is a short, human-readable label identifying the call site
// (e.g. "single_miss_pread", "batch_miss_io_uring", "batch_miss_pread",
// "disk_batch_padding") -- included directly in the LOG_DEBUG trace line
// so grepping for [RAM-LOAD-VS-PREFETCH] shows exactly which path caused
// each traced load, not just that one happened. No-op (returns
// immediately, nothing logged or counted) when the tracker has no record
// at all for this (layer, expert) -- the common case, a "pure" miss with
// no prefetch involvement, not interesting for this specific trace.
inline void trace_ram_load_if_recently_fetched(int layer, int expert, const char* trigger) {
    uint64_t elapsed_us = 0, step_delta = 0;
    ExpertFetchState st = expert_fetch_tracker().query(layer, expert, elapsed_us, step_delta);
    if (st == ExpertFetchState::None) return;
    const char* state_name =
        st == ExpertFetchState::InFlight ? "IN_FLIGHT(prefetch worker reading the SAME bytes RIGHT NOW)" :
        st == ExpertFetchState::StagedReady ? "STAGED_READY(a completed, unconsumed fetch exists for this exact expert)" :
        st == ExpertFetchState::EvictedUnused ? "EVICTED_UNUSED(was fetched, never consumed, already recycled)" :
        "CONSUMED(was fetched and genuinely used earlier, now needed again)";
    LOG_DEBUG("[RAM-LOAD-VS-PREFETCH] layer=%d expert=%d trigger=%s prefetch_state=%s "
              "elapsed_since_mark=%.1fms steps_back=%llu -- this RAM-tier disk load is happening for an "
              "expert the prefetcher ALSO touched; steps_back>0 means that touch was during an EARLIER "
              "token, not concurrent with this exact load\n",
              layer, expert, trigger, state_name, elapsed_us / 1000.0, (unsigned long long)step_delta);
    switch (st) {
        case ExpertFetchState::InFlight:
            perf().n_ram_load_had_prefetch_inflight.fetch_add(1, std::memory_order_relaxed);
            break;
        case ExpertFetchState::StagedReady:
            perf().n_ram_load_had_prefetch_staged.fetch_add(1, std::memory_order_relaxed);
            break;
        case ExpertFetchState::EvictedUnused:
            perf().n_ram_load_had_prefetch_evicted.fetch_add(1, std::memory_order_relaxed);
            break;
        case ExpertFetchState::Consumed:
            perf().n_ram_load_had_prefetch_consumed.fetch_add(1, std::memory_order_relaxed);
            break;
        default:
            break;
    }
}

// Precise test for the Phase-A/Phase-B pipeline-ordering-gap hypothesis
// (RAMSafetyManager::get_experts_batch): Phase A checks consume_staged for
// every pending expert FIRST, across the whole batch, and only afterward
// does Phase B actually perform the real disk reads for whatever Phase A
// couldn't serve from staging. A real run's RAM-load-vs-prefetch trace
// found 86% of Phase-B loads that intersected with prefetch activity were
// in StagedReady state -- i.e. consume_staged had JUST failed, yet a
// ready copy existed moments later. Two explanations are possible: (a)
// the prefetcher's own worker thread finished staging this exact expert
// DURING the Phase A -> Phase B gap, arriving too late for Phase A's
// already-locked-in decision (the expected, benign mechanism -- not a
// bug, just a real ordering cost of checking staging once per batch
// instead of once per expert right before its own read), or (b) the data
// was ALREADY staged when Phase A's own check ran, and consume_staged
// still failed for some OTHER reason (a genuine anomaly worth separate
// investigation -- NOT the ordering-gap mechanism).
//
// `phase_a_check_at_us` is the caller's OWN timestamp (steady_now_us(),
// captured at the exact moment ITS Phase A consume_staged call for this
// expert returned false) -- compared here against the tracker's raw mark
// timestamp for the SAME expert to settle which explanation applies,
// definitively, per event: mark_at_us > phase_a_check_at_us proves (a);
// mark_at_us <= phase_a_check_at_us proves (b).
inline void trace_phase_gap_if_staged_after_check(int layer, int expert, uint64_t phase_a_check_at_us) {
    if (phase_a_check_at_us == 0) return;  // caller has no valid timestamp to compare against
    uint64_t mark_at_us = 0;
    ExpertFetchState st = expert_fetch_tracker().query_raw_mark_time(layer, expert, mark_at_us);
    if (st != ExpertFetchState::StagedReady || mark_at_us == 0) return;  // only meaningful for this exact state
    bool staged_after_check = mark_at_us > phase_a_check_at_us;
    double gap_ms = (double)(staged_after_check ? (mark_at_us - phase_a_check_at_us)
                                                 : (phase_a_check_at_us - mark_at_us)) / 1000.0;
    LOG_DEBUG("[PHASE-GAP] layer=%d expert=%d staged_after_phase_a_check=%s gap=%.2fms -- %s\n",
              layer, expert, staged_after_check ? "YES" : "NO", gap_ms,
              staged_after_check
                  ? "CONFIRMS the ordering-gap hypothesis: the prefetcher finished staging this exact "
                    "expert AFTER Phase A's own consume_staged check already failed for it"
                  : "ANOMALY: this expert was already staged BEFORE Phase A's own check ran, yet that "
                    "check still failed -- NOT the ordering-gap mechanism, worth separate investigation");
    if (staged_after_check) {
        perf().n_phase_gap_staged_after_check.fetch_add(1, std::memory_order_relaxed);
        PerfCounters::atomic_add(perf().t_phase_gap_ms_sum, gap_ms);
    } else {
        perf().n_phase_gap_staged_before_check_anomaly.fetch_add(1, std::memory_order_relaxed);
    }
}
