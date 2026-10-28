#pragma once

// expert_promotion: adaptive promotion of missed experts into a graph's static VRAM set
// (docs/design/expert-promotion.md). The TOML key is model-neutral; the implementation is currently the Qwen4Exp graph's.
// The key is deliberately not called graph_expert_promotion: root keys starting with graph_ belong to the typed graph-settings
// table (models/graph/graph_settings.hpp) and are rejected as unknown settings unless registered there.
//
//   "auto" (default)  on when graph-native MTP is on, off otherwise
//   "on"              swap missed experts in (level 2)
//   "observe"         count misses, never swap (level 1; frozen set, for A/B and validation)
//   "off"             static set (level 0)
//
// An explicit value always wins over MTP. Level numbers are what the engine stores.

#include <string>

namespace zeromoe::graph {

inline bool expert_promotion_mode_valid(const std::string &mode) {
  return mode == "auto" || mode == "on" || mode == "observe" || mode == "off";
}

// Unknown strings (reachable only through the environment override, the config parser rejects them) resolve to off.
inline int resolve_expert_promotion_mode(const std::string &mode, bool mtp_graph_native) {
  if (mode == "auto") return mtp_graph_native ? 2 : 0;
  if (mode == "on") return 2;
  if (mode == "observe") return 1;
  return 0;
}

}  // namespace zeromoe::graph
