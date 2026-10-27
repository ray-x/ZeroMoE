// tensor_trace.hpp — emits registered tensor traces at the normal verbose
// logging level. Use --verbosity 4 to include all trace tags in the selected
// log file or stderr, then filter the captured file with standard text tools.
// This keeps trace selection out of the runtime configuration surface.
#pragma once
#include "logger.hpp"
#include <algorithm>
#include <cmath>
#include <sstream>
#include <string>

// Call sites gate the whole trace block on this (matching the existing
// "gate BEFORE paying for the computation, not just before printing"
// discipline) rather than Logger::would_log(VERBOSE) alone.
inline bool trace_gate(const char* tag) {
    (void)tag;
    return Logger::instance().would_log(LogLevel::VERBOSE);
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
        return false;
    }
    if (trace_gate("EXPERT-GPU")) {
        trace_row("EXPERT-GPU", layer_idx,
                  std::string(label) + " expert=" + std::to_string(expert_id),
                  data, n);
    }
    return true;
}
