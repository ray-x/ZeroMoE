#pragma once

// Opt-in thread naming/placement for decode diagnostics.  This helper is
// deliberately inert unless ZEROMOE_THREAD_PROFILE=1 is set: changing
// affinity or scheduling policy globally would make an A/B benchmark invalid
// and can make a production process harder to recover.

#include "logger.hpp"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

struct ThreadProfile {
  std::string name;
  std::vector<int> cpu_ids;
  bool use_rt = false;
  int rt_priority = 80;
  int nice_value = 0;
  // Negative nice values require CAP_SYS_NICE on normal Linux systems.  Keep
  // them off unless the benchmark explicitly asks for them.
  bool apply_nice = false;
};

inline bool zeromoe_thread_profile_enabled() {
  const char *value = std::getenv("ZEROMOE_THREAD_PROFILE");
  return value && (value[0] == '1' || value[0] == 'y' || value[0] == 'Y' ||
                   value[0] == 't' || value[0] == 'T');
}

inline bool zeromoe_thread_profile_rt_enabled() {
  const char *value = std::getenv("ZEROMOE_THREAD_PROFILE_RT");
  return value && value[0] == '1';
}

inline bool zeromoe_thread_profile_nice_enabled() {
  const char *value = std::getenv("ZEROMOE_THREAD_PROFILE_APPLY_NICE");
  return value && value[0] == '1';
}

#if defined(__linux__)
namespace zeromoe_thread_profile_detail {

inline bool read_int_file(const std::string &path, int &value) {
  std::ifstream input(path);
  return static_cast<bool>(input >> value);
}

inline std::vector<int> allowed_cpus() {
  std::vector<int> result;
  cpu_set_t mask;
  CPU_ZERO(&mask);
  if (sched_getaffinity(0, sizeof(mask), &mask) != 0)
    return result;
  for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
    if (CPU_ISSET(cpu, &mask))
      result.push_back(cpu);
  }
  return result;
}

inline std::vector<int> physical_core_cpus(int physical_core_index) {
  const std::vector<int> allowed = allowed_cpus();
  if (physical_core_index < 0 || allowed.empty())
    return {};

  // Group the CPUs visible to this process by package/core.  This handles
  // both common layouts (0,6 and 0,1 sibling numbering) without assuming a
  // fixed SMT stride.
  std::vector<std::pair<int, int>> keys;
  std::vector<std::vector<int>> groups;
  for (int cpu : allowed) {
    int package = 0;
    int core = cpu;
    const std::string base = "/sys/devices/system/cpu/cpu" +
                             std::to_string(cpu) + "/topology/";
    read_int_file(base + "physical_package_id", package);
    read_int_file(base + "core_id", core);
    const std::pair<int, int> key{package, core};
    auto it = std::find(keys.begin(), keys.end(), key);
    if (it == keys.end()) {
      keys.push_back(key);
      groups.push_back({cpu});
    } else {
      groups[static_cast<size_t>(it - keys.begin())].push_back(cpu);
    }
  }
  if (physical_core_index >= static_cast<int>(groups.size()))
    return {};
  auto &group = groups[static_cast<size_t>(physical_core_index)];
  std::sort(group.begin(), group.end());
  return group;
}

inline bool set_nice_for_current_thread(int value) {
  const pid_t tid = static_cast<pid_t>(::syscall(SYS_gettid));
  return setpriority(PRIO_PROCESS, tid, value) == 0;
}

} // namespace zeromoe_thread_profile_detail
#endif

inline std::vector<int>
zeromoe_thread_profile_physical_core(int physical_core_index) {
#if defined(__linux__)
  return zeromoe_thread_profile_detail::physical_core_cpus(
      physical_core_index);
#else
  (void)physical_core_index;
  return {};
#endif
}

inline void configure_current_thread(const ThreadProfile &profile) {
  if (!zeromoe_thread_profile_enabled())
    return;

#if defined(__linux__)
  const pthread_t thread = pthread_self();
  const std::string name = profile.name.substr(0, 15);
  if (!name.empty()) {
    const int rc = pthread_setname_np(thread, name.c_str());
    if (rc != 0)
      LOG_WARN("[THREAD-PROFILE] %s: setname failed: %s\n", profile.name.c_str(),
               std::strerror(rc));
  }

  if (!profile.cpu_ids.empty()) {
    cpu_set_t mask;
    CPU_ZERO(&mask);
    for (int cpu : profile.cpu_ids) {
      if (cpu >= 0 && cpu < CPU_SETSIZE)
        CPU_SET(cpu, &mask);
    }
    const int rc = pthread_setaffinity_np(thread, sizeof(mask), &mask);
    if (rc != 0) {
      LOG_WARN("[THREAD-PROFILE] %s: affinity failed for requested CPUs: %s\n",
               profile.name.c_str(), std::strerror(rc));
    }
  }

  bool rt_applied = false;
  if (profile.use_rt && zeromoe_thread_profile_rt_enabled()) {
    sched_param param{};
    param.sched_priority = std::clamp(profile.rt_priority, 1, 99);
    const int rc = pthread_setschedparam(thread, SCHED_FIFO, &param);
    if (rc == 0) {
      rt_applied = true;
      LOG_INFO("[THREAD-PROFILE] %s: affinity set, SCHED_FIFO priority %d\n",
               profile.name.c_str(), param.sched_priority);
    } else {
      LOG_WARN("[THREAD-PROFILE] %s: SCHED_FIFO failed: %s\n",
               profile.name.c_str(), std::strerror(rc));
    }
  }

  // Nice is only attempted when explicitly requested, or as the fallback
  // after an explicitly requested RT policy failed.  A failed negative nice
  // request is reported, never treated as a silent success.
  if (!rt_applied &&
      (profile.apply_nice || zeromoe_thread_profile_nice_enabled())) {
    if (zeromoe_thread_profile_detail::set_nice_for_current_thread(
            profile.nice_value)) {
      LOG_INFO("[THREAD-PROFILE] %s: nice=%d\n", profile.name.c_str(),
               profile.nice_value);
    } else {
      LOG_WARN("[THREAD-PROFILE] %s: nice=%d failed: %s\n",
               profile.name.c_str(), profile.nice_value,
               std::strerror(errno));
    }
  }

  LOG_INFO("[THREAD-PROFILE] %s: configured affinity_count=%zu rt=%s nice=%s\n",
           profile.name.c_str(), profile.cpu_ids.size(),
           rt_applied ? "SCHED_FIFO" : "off",
           (!rt_applied &&
            (profile.apply_nice || zeromoe_thread_profile_nice_enabled()))
               ? "requested"
               : "off");
#else
  (void)profile;
#endif
}
