// tensor_trace.hpp — generalizes this project's existing per-file VERBOSE
// tensor-debugging convention (check_finite_attn/check_finite_hc/check_finite/
// check_hidden_finite, the [HC-TRACE]/[MLA-TRACE]/[SILU-TRACE-GPU]/etc. lines
// scattered across attention_layer.cpp/hyper_connections.cpp/
// execution_layer.cpp/generation_loop.hpp) with the two capabilities
// llama.cpp's examples/eval-callback provides that this project's ad hoc
// per-file helpers never had:
//   - filtering by tag (see so only the signal you're chasing, not the whole
//     VERBOSE firehose)
//   - abort-on-nan (stop at the FIRST bad tensor instead of scrolling through
//     pages of downstream NaN fallout)
//
// This keeps the existing per-file trace tags while centralizing the shared
// filter/abort policy and the common finite-buffer check.
//
// Enabling: requires log_level = "VERBOSE" (or --log-level VERBOSE). Two
// optional knobs on top, both wired in main.cpp right after Logger::init():
//   trace_filter = "HC,MLA"   (or --trace-filter "HC,MLA") -- comma-separated
//                              tag prefixes; empty (default) = trace everything.
//   trace_abort_on_nan = true (or --trace-abort-on-nan)    -- std::abort() at
//                              the first NaN/Inf instead of continuing.
// See CLAUDE.md section 5 for the full list of tags and how to read the
// printed `[<TAG>-TRACE l=<layer>] <label> |row|=<norm> vals=[...]` lines.
//
// Adding a new checkpoint at a call site:
//   if (trace_gate("MY-TAG")) trace_row("MY-TAG", layer_idx, label, data, n);
// (row_width > 0 traces only the last `row_width` elements, e.g. one token's
// row in an [S,D] buffer; omitted/<=0 traces the whole buffer.) For a NaN/Inf
// checker, call trace_abort_if_nan("MY-TAG", true) right after logging the
// ERROR line, before returning.
#pragma once
#include "logger.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

class TraceFilter {
public:
    static TraceFilter& instance() {
        static TraceFilter inst;
        return inst;
    }

    // `pattern_csv` is a comma-separated list of regex patterns matched
    // (anchored at the start, mirroring llama.cpp's own eval-callback
    // convention) against a call site's short tag ("HC", "MLA",
    // "EXPERT-GPU", "SILU-TRACE-GPU", ...). Empty => match everything,
    // identical to today's behavior (every tag traces once VERBOSE is on) --
    // so nothing changes for existing users unless they opt in.
    void configure(const std::string& pattern_csv, bool abort_on_nan) {
        patterns_.clear();
        abort_on_nan_ = abort_on_nan;
        std::stringstream ss(pattern_csv);
        std::string tok;
        while (std::getline(ss, tok, ',')) {
            if (tok.empty()) continue;
            try {
                patterns_.emplace_back("^" + tok, std::regex::optimize);
            } catch (const std::regex_error&) {
                std::fprintf(stderr, "[TRACE] Invalid trace_filter pattern '%s' -- ignored\n", tok.c_str());
            }
        }
    }

    bool matches(const char* tag) const {
        if (patterns_.empty()) return true;
        for (const auto& p : patterns_) {
            if (std::regex_search(tag, p)) return true;
        }
        return false;
    }

    bool abort_on_nan() const { return abort_on_nan_; }

private:
    TraceFilter() = default;
    std::vector<std::regex> patterns_;
    bool abort_on_nan_ = false;
};

// True once for both this call's VERBOSE threshold AND the tag filter --
// call sites should gate their whole trace block on this (matching the
// existing "gate BEFORE paying for the computation, not just before
// printing" discipline already established in this codebase) rather than
// Logger::would_log(VERBOSE) alone.
inline bool trace_gate(const char* tag) {
    return Logger::instance().would_log(LogLevel::VERBOSE) && TraceFilter::instance().matches(tag);
}

// Unified VERBOSE trace line for a float buffer: L2 norm of the last
// `row_width` elements (a token row) -- or, when row_width<=0, of the whole
// buffer -- plus up to 8 raw sample values from that same slice, so actual
// numbers are visible for side-by-side comparison against a llama.cpp
// eval-callback capture, not just a magnitude. Caller must have already
// checked trace_gate(tag); this function doesn't re-check it so callers that
// need to gate an expensive upstream computation (e.g. a GPU->host download)
// on the same condition don't pay for the check twice.
inline void trace_row(const char* tag, int layer_idx, const std::string& label,
                       const float* data, size_t n, int row_width = 0) {
    if (n == 0) return;
    size_t width = row_width > 0 ? (size_t)row_width : n;
    width = std::min(width, n);
    const float* row = data + (n - width);
    double sq = 0.0;
    for (size_t i = 0; i < width; ++i) sq += (double)row[i] * row[i];
    size_t nshow = std::min<size_t>(8, width);
    std::ostringstream vals;
    vals << "[";
    for (size_t i = 0; i < nshow; ++i) {
        if (i) vals << ", ";
        vals << row[i];
    }
    vals << (nshow < width ? ", ...]" : "]");
    LOG_STREAM_VERBOSE() << "[" << tag << "-TRACE l=" << layer_idx << "] " << label
                         << " |row|=" << std::sqrt(sq) << " vals=" << vals.str() << "\n";
}

// Call immediately after logging a NaN/Inf ERROR line, passing whether that
// check actually found something bad. Hard-aborts the process when
// trace_abort_on_nan is configured, mirroring llama.cpp examples/
// eval-callback's --abort-on-nan -- lets a debugging session stop at the
// FIRST bad tensor instead of scrolling through pages of downstream NaN
// fallout to find the original source.
inline void trace_abort_if_nan(const char* tag, bool bad) {
    if (bad && TraceFilter::instance().abort_on_nan()) {
        LOG_ERROR("[TRACE] Aborting: NaN/Inf detected at tag=%s (trace_abort_on_nan=true)\n", tag);
        std::abort();
    }
}

inline bool check_finite(const char* label, int layer_idx, int expert_id,
                         const float* data, size_t n) {
    int nan_count = 0;
    int inf_count = 0;
    for (size_t i = 0; i < n; ++i) {
        if (std::isnan(data[i])) ++nan_count;
        else if (std::isinf(data[i])) ++inf_count;
    }
    if (nan_count || inf_count) {
        LOG_ERROR("[EXPERT-GPU] %s layer=%d expert=%d: %d/%zu NaN, %d/%zu Inf.\n",
                  label, layer_idx, expert_id, nan_count, n, inf_count, n);
        trace_abort_if_nan("EXPERT-GPU", true);
        return false;
    }
    if (trace_gate("EXPERT-GPU")) {
        trace_row("EXPERT-GPU", layer_idx,
                  std::string(label) + " expert=" + std::to_string(expert_id),
                  data, n);
    }
    return true;
}
