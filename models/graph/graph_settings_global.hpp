#pragma once

// Process-wide binding of graph_settings.hpp to the real environment. Lookups
// go through graphcfg::env so the consulted-knob registry still sees them.
// Resolved once per family and logged once as a [SETTINGS] table.

#include "graph_config.hpp"
#include "graph_settings.hpp"

#include <mutex>

namespace zeromoe::graphsettings {

inline const ResolvedSettings &settings_for(Family fam) {
  static std::mutex mu;
  static ResolvedSettings cache[4];
  static bool done[4] = {false, false, false, false};
  std::lock_guard<std::mutex> lock(mu);
  const size_t i = static_cast<size_t>(fam);
  if (!done[i]) {
    std::string err;
    cache[i] = resolve_settings(
        fam, [](const char *n) { return zeromoe::graphcfg::env(n); }, &err);
    done[i] = true;
    static const char *const names[] = {"other", "mimo2", "ling", "glm-graph"};
    LOG_INFO("[SETTINGS] family=%s defaults_hash=%016llx\n%s", names[i],
             static_cast<unsigned long long>(defaults_hash()),
             format_settings(cache[i]).c_str());
    if (!err.empty())
      LOG_ERROR("[SETTINGS] invalid family overrides:\n%s", err.c_str());
    for (const std::string &f : settings_findings(cache[i]))
      LOG_WARN("[PREFLIGHT] %s: %s\n", names[i], f.c_str());
    if (cache[i].diag.preflight_strict.value && !settings_findings(cache[i]).empty()) {
      LOG_ERROR("[PREFLIGHT] strict mode: refusing to run with the findings above\n");
      std::abort();
    }
  }
  return cache[i];
}

// A preflight finding from a run-time check (graph_health.hpp): a warning, or an abort under
// ZEROMOE_PREFLIGHT_STRICT=1. Empty findings are ignored.
inline void report_preflight(const char *tag, Family fam, const std::string &finding) {
  if (finding.empty()) return;
  LOG_WARN("[PREFLIGHT] %s: %s\n", tag, finding.c_str());
  if (settings_for(fam).diag.preflight_strict.value) {
    LOG_ERROR("[PREFLIGHT] strict mode: refusing to run with the finding above\n");
    std::abort();
  }
}

} // namespace zeromoe::graphsettings
