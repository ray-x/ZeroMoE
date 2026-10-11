#pragma once

// Single entry point for every environment knob the full-graph code reads.
//
// Two layers:
//   * env(name) is a drop-in for std::getenv. It returns exactly what getenv
//     returns (nothing is cached, so in-process setenv in tests still works)
//     and records the first lookup of each name, so a run can print which
//     knobs the graph code consulted and what they resolved to.
//   * flag_on / flag_unless_off / integer / real / path are the typed readers
//     for new code. They encode the three idioms the graph files used by hand
//     (`v && v[0]=='1'`, `!(v && v[0]=='0')`, strict strtol) in one place.
//
// log_effective() prints one [GRAPH-CONFIG] block: every MOESTREAM_* variable
// set in the process environment, then every knob consulted so far that was
// left unset. A flag that is on in every compared run can no longer be
// credited with another flag's gain without the log showing it.

#include "logger.hpp"

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <unordered_set>

#include <unistd.h>  // declares environ (_GNU_SOURCE)

namespace moestream::graphcfg {

struct Entry {
  bool set = false;
  std::string value;
  unsigned long long reads = 0;
};

class Registry {
 public:
  static Registry &get() {
    static Registry r;
    return r;
  }
  // Records a lookup. `first_value` is null when the variable is unset.
  void note(const char *name, const char *first_value) {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = entries_.find(name);
    if (it == entries_.end()) {
      Entry e;
      e.set = first_value != nullptr;
      if (first_value)
        e.value = first_value;
      e.reads = 1;
      entries_.emplace(name, std::move(e));
    } else {
      ++it->second.reads;
    }
  }
  std::map<std::string, Entry> snapshot() const {
    std::lock_guard<std::mutex> lock(mu_);
    return entries_;
  }

 private:
  mutable std::mutex mu_;
  std::map<std::string, Entry> entries_;
};

// Drop-in for std::getenv. The per-thread pointer set keeps repeated lookups
// of the same literal off the registry mutex; the first lookup of each name
// per thread takes it once.
inline const char *env(const char *name) {
  const char *v = std::getenv(name);
  thread_local std::unordered_set<const void *> seen;
  if (seen.insert(name).second)
    Registry::get().note(name, v);
  return v;
}

inline bool present(const char *name) { return env(name) != nullptr; }

// Set and starting with '1'.
inline bool flag_on(const char *name) {
  const char *v = env(name);
  return v && v[0] == '1';
}

// On unless explicitly "0".
inline bool flag_unless_off(const char *name) {
  const char *v = env(name);
  return !(v && v[0] == '0');
}

// Whole-string strtol; unset, empty or malformed returns `dflt`.
inline long integer(const char *name, long dflt) {
  const char *v = env(name);
  if (!v || !v[0])
    return dflt;
  char *end = nullptr;
  errno = 0;
  const long n = std::strtol(v, &end, 10);
  return (end == v || *end != '\0' || errno != 0) ? dflt : n;
}

inline double real(const char *name, double dflt) {
  const char *v = env(name);
  if (!v || !v[0])
    return dflt;
  char *end = nullptr;
  const double d = std::strtod(v, &end);
  return (end == v || *end != '\0') ? dflt : d;
}

// Non-empty value, or nullptr.
inline const char *path(const char *name) {
  const char *v = env(name);
  return v && v[0] ? v : nullptr;
}

// Prints the effective configuration once per `tag`. `consulted_only=false`
// also lists every MOESTREAM_* variable the graph code never read (a typo or
// a knob owned by another module, or one the graph reads later).
inline void log_effective(const char *tag, bool include_unread = true) {
  static std::mutex once_mu;
  static std::unordered_set<std::string> printed;
  {
    std::lock_guard<std::mutex> lock(once_mu);
    if (!printed.insert(tag).second)
      return;
  }
  const auto consulted = Registry::get().snapshot();
  std::map<std::string, std::string> in_env;
  for (char **e = environ; e && *e; ++e) {
    if (std::strncmp(*e, "MOESTREAM_", 10) != 0)
      continue;
    const char *eq = std::strchr(*e, '=');
    if (!eq)
      continue;
    in_env[std::string(*e, static_cast<size_t>(eq - *e))] = eq + 1;
  }
  LOG_INFO("[GRAPH-CONFIG] %s: %zu MOESTREAM_* variable(s) set, %zu knob(s) "
           "consulted so far\n",
           tag, in_env.size(), consulted.size());
  for (const auto &kv : in_env) {
    const bool read = consulted.count(kv.first) != 0;
    if (!read && !include_unread)
      continue;
    LOG_INFO("[GRAPH-CONFIG]   %s=%s%s\n", kv.first.c_str(), kv.second.c_str(),
             read ? "" : "   (not read yet)");
  }
  size_t unset = 0;
  for (const auto &kv : consulted)
    if (!kv.second.set)
      ++unset;
  LOG_INFO("[GRAPH-CONFIG]   %zu consulted knob(s) left at their default\n",
           unset);
}

// Full table of every knob the graph code consulted, with the value it saw.
// Printed at shutdown, when lazily-read diagnostics have been reached.
inline void log_consulted(const char *tag) {
  const auto consulted = Registry::get().snapshot();
  LOG_INFO("[GRAPH-CONFIG] %s: %zu knob(s) consulted over the run\n", tag,
           consulted.size());
  for (const auto &kv : consulted) {
    if (kv.second.set)
      LOG_INFO("[GRAPH-CONFIG]   %s=%s\n", kv.first.c_str(),
               kv.second.value.c_str());
    else
      LOG_INFO("[GRAPH-CONFIG]   %s (default)\n", kv.first.c_str());
  }
}

}  // namespace moestream::graphcfg
