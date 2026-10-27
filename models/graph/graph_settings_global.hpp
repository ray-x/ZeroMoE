#pragma once

// Process-wide binding of graph_settings.hpp to the real environment. Lookups
// go through graphcfg::env so the consulted-knob registry still sees them.
// Resolved once per family and logged once as a [SETTINGS] table.

#include "graph_config.hpp"
#include "graph_settings.hpp"

#include <mutex>
#include <array>

namespace zeromoe::graphsettings {

struct SettingsBinding {
  std::mutex mutex;
  ConfigValues config;
  CliValues cli;
  std::array<ResolvedSettings, kFamilyCount> cache;
  std::array<bool, kFamilyCount> done{};
};

inline SettingsBinding &settings_binding() {
  static SettingsBinding binding;
  return binding;
}

inline void configure_settings(const ConfigValues &config,
                               const CliValues &cli = {}) {
  auto &b = settings_binding();
  std::lock_guard lock(b.mutex);
  if (std::any_of(b.done.begin(), b.done.end(), [](bool done) { return done; }))
    throw std::logic_error("graph settings must be configured before graph initialization");
  b.config = config;
  b.cli = cli;
}

inline const ResolvedSettings &settings_for(Family fam) {
  auto &b = settings_binding();
  auto &cache = b.cache;
  auto &done = b.done;
  std::lock_guard<std::mutex> lock(b.mutex);
  const size_t i = static_cast<size_t>(fam);
  if (!done[i]) {
    std::string err;
    cache[i] = resolve_settings(
        fam, [](const char *n) { return zeromoe::graphcfg::env(n); }, &err,
        b.config, b.cli);
    if (!err.empty()) throw std::invalid_argument("invalid graph settings:\n" + err);
    done[i] = true;
    static const char *const names[] = {"other", "mimo2", "ling", "glm", "qwen"};
    LOG_INFO("[SETTINGS] family=%s defaults_hash=%016llx\n%s", names[i],
             static_cast<unsigned long long>(defaults_hash()),
             format_settings(cache[i]).c_str());
    for (const auto &warning : cache[i].warnings)
      LOG_WARN("[SETTINGS] %s\n", warning.c_str());
    for (const std::string &f : settings_findings(cache[i]))
      LOG_WARN("[PREFLIGHT] %s: %s\n", names[i], f.c_str());
    if (cache[i].diag.preflight_strict.value && !settings_findings(cache[i]).empty()) {
      LOG_ERROR("[PREFLIGHT] strict mode: refusing to run with the findings above\n");
      std::abort();
    }
  }
  return cache[i];
}

// The concrete model family this process loaded. The GLM graph implementation
// is shared by GLM-5.x, GLM-DSA and Ling, so resolving through this value keeps
// Ling's ZEROMOE_LING_* aliases and family defaults out of GLM runs and vice
// versa. Set once from the detected model family before graph initialization;
// the cache is keyed by family, so a late change after the first resolution
// would silently mix rows.
namespace detail {
struct ActiveFamilyState {
  Family family = Family::Other;
  bool set = false;
};
inline ActiveFamilyState &active_family_state() {
  static ActiveFamilyState state;
  return state;
}
} // namespace detail

inline void set_active_family(Family family) {
  auto &state = detail::active_family_state();
  if (state.set && state.family != family)
    throw std::logic_error("graph settings family already selected");
  state.family = family;
  state.set = true;
}

inline Family active_family() { return detail::active_family_state().family; }

inline const ResolvedSettings &active_settings() {
  return settings_for(active_family());
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
