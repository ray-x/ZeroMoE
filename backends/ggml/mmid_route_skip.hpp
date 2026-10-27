#pragma once

// One place that decides how a hot/cold-split MoE layer marks a route a branch does not own.
//
// A route can be made free ("skipped") by giving GGML_OP_MUL_MAT_ID a NEGATIVE expert id: the CPU op
// zeroes the output and reads no weights, and the CUDA MMVQ decode kernels do the same. That is only
// guaranteed by the ZeroMoE ggml fork (zeromoe_mmid_negative_skip) and only for single-token
// decode; prefill's mmq/mmf/sorted CUDA paths and any unpatched ggml need a VALID dummy id (0) plus a
// 0/1 mask, at the price of every branch computing all top-k routes.
//
// Every graph builder (GLM5Next, Qwen4Exp, ...) must obtain its "inactive route" id from
// mmid_inactive_route_id() instead of hard-coding -1 or 0.

#include <cstdint>

#include "logger.hpp"

// The ZeroMoE ggml build always applies patches/ggml, so the capability is a
// build invariant rather than a runtime option. The declaration stays weak so a
// stock ggml still links for diagnostics; that build gets dummy routes and the
// warning below instead of a user-facing environment switch.
extern "C" int ggml_cpu_mul_mat_id_negative_skip(void) __attribute__((weak));

// True when the linked ggml advertises negative-id skipping.
inline bool mmid_negative_skip_available() {
  static const bool available = [] {
    const bool linked = &ggml_cpu_mul_mat_id_negative_skip != nullptr &&
                        ggml_cpu_mul_mat_id_negative_skip() != 0;
    if (!linked)
      LOG_WARN("[MMID] linked ggml does not advertise negative-id skip; using "
               "valid dummy routes (rebuild with the bundled ggml patches)\n");
    return linked;
  }();
  return available;
}

// Batched (S > 1) CUDA MMQ MUL_MAT_ID skips negative ids too, from patches/ggml/0006.  Without it a
// batched hot/cold prefill must use a valid dummy id, and ggml's batched id compaction then breaks:
// it assumes a token never repeats an expert, so several inactive slots sharing dummy id 0 make the
// per-expert row ranges longer than the rows actually written (wild MMQ output writes).
extern "C" int ggml_cuda_mul_mat_id_negative_skip_batched(void) __attribute__((weak));

inline bool mmid_negative_skip_batched_available() {
  static const bool available = [] {
    const bool linked = &ggml_cuda_mul_mat_id_negative_skip_batched != nullptr &&
                        ggml_cuda_mul_mat_id_negative_skip_batched() != 0;
    if (!linked)
      LOG_WARN("[MMID] linked ggml does not advertise batched negative-id skip; "
               "batched prefill uses valid dummy routes (rebuild with the "
               "bundled ggml patches)\n");
    return linked;
  }();
  return available;
}

// Expert id to use for a route this branch does not own.
//   single_token_decode: true only for S == 1 graphs (the shapes the skip kernels cover).
//   allow_batched:       the caller has verified the layer's expert types take the patched MMQ path.
// Returns -1 (skip: no compute, no weight reads) or 0 (valid dummy slot, result masked to zero).
inline int32_t mmid_inactive_route_id(bool single_token_decode, bool allow_batched = false) {
  if (single_token_decode)
    return mmid_negative_skip_available() ? -1 : 0;
  return (allow_batched && mmid_negative_skip_available() &&
          mmid_negative_skip_batched_available())
             ? -1
             : 0;
}
