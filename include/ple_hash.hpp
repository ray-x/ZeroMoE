// ple_hash.hpp — the pure n-gram hash at the heart of Qwen4Exp's PLE
// (per-layer token embedding), extracted so it is unit-testable in tests/
// test_components.cpp without engine dependencies.
//
// Reference: llama.cpp branch qwen4exp/qwen3.8-flash-next
// (src/models/qwen4exp.cpp's llm_graph_input_ple::set_input):
//   mixed_n = (t[p]*m[0]) ^ (t[p-1]*m[1]) ^ ... ^ (t[p-n+1]*m[n-1])
//   row     = mixed_n % head_vocab + head_offset
// uint64 math throughout -- the multipliers are 45-bit constants that would
// silently round through float32 (which is exactly why the reference
// converter reads them as raw int64, bypassing the float cast).
#pragma once
#include <cstdint>

// ctx[0..n) = the token window (ctx[0] = current token, ctx[s] = s positions
// back, EOS-padded/cut by the caller); n in [2, ngram_size]. Returns the
// gather-row index for this n-gram order's head.
inline int64_t ple_ngram_row(const int64_t *ctx, int n,
                             const int64_t *multipliers, int64_t head_vocab,
                             int64_t head_offset) {
  uint64_t mixed = (uint64_t)ctx[0] * (uint64_t)multipliers[0];
  for (int j = 1; j < n; ++j)
    mixed ^= (uint64_t)ctx[j] * (uint64_t)multipliers[j];
  return (int64_t)(mixed % (uint64_t)head_vocab) + head_offset;
}
