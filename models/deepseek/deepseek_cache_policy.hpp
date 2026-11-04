#pragma once

namespace zeromoe::deepseek_graph {

// DeepSeek can keep the batched CPU cold-expert path for prompt prefill while
// staging decode misses from resident DDR into the reserved VRAM cache slots.
// Decode staging keeps the cached MoE work on the full graph/GPU path and
// avoids serial per-expert CPU FFNs for every generated token.
inline bool use_cpu_cold_graph(bool decode_mode, int token_count,
                              bool cpu_cold_prefill_enabled,
                              bool cpu_cold_decode_enabled) {
  if (decode_mode)
    return token_count == 1 && cpu_cold_decode_enabled;
  return token_count > 1 && cpu_cold_prefill_enabled;
}

} // namespace zeromoe::deepseek_graph
