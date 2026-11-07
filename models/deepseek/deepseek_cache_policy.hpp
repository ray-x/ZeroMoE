#pragma once

#include <cstddef>
#include <optional>
#include <string_view>

namespace zeromoe::deepseek_graph {

enum class DecodeMissStrategy : int {
  H2D = 0,
  Cpu = 1,
  Split = 2,
};

inline std::optional<DecodeMissStrategy>
parse_decode_miss_strategy(std::string_view value) {
  if (value == "h2d") return DecodeMissStrategy::H2D;
  if (value == "cpu") return DecodeMissStrategy::Cpu;
  if (value == "split") return DecodeMissStrategy::Split;
  return std::nullopt;
}

inline bool decode_miss_uses_cpu(DecodeMissStrategy strategy,
                                 size_t ddr_miss_ordinal) {
  switch (strategy) {
  case DecodeMissStrategy::H2D: return false;
  case DecodeMissStrategy::Cpu: return true;
  case DecodeMissStrategy::Split: return (ddr_miss_ordinal & 1U) == 0;
  }
  return false;
}

// Async cache validation applies only to routes the graph expects to consume
// from VRAM. A CPU-selected DDR miss deliberately has no GPU slot.
inline bool requires_vram_route_validation(bool route_active, bool cpu_routed) {
  return route_active && !cpu_routed;
}

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

// Prefill CPU spills write into the prompt-sized scratch MoE buffer to avoid
// allocating another D x S output vector. Decode uses the site's persistent
// D x 1 output vector. The graph result supplies the authoritative element
// count for the prefill scratch buffer.
inline size_t cpu_merge_element_count(bool prefill, size_t graph_elements,
                                      size_t decode_output_elements) noexcept {
  return prefill ? graph_elements : decode_output_elements;
}

} // namespace zeromoe::deepseek_graph
