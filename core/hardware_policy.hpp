#pragma once

#include <algorithm>
#include <sched.h>
#include <thread>

namespace zeromoe::hardware {

// Count CPUs available to this process, respecting container/cgroup affinity.
// Fall back to the standard library when affinity is unavailable.
inline unsigned available_logical_cpu_count() {
  cpu_set_t allowed;
  CPU_ZERO(&allowed);
  if (sched_getaffinity(0, sizeof(allowed), &allowed) == 0) {
    const int count = CPU_COUNT(&allowed);
    if (count > 0)
      return static_cast<unsigned>(count);
  }
  return std::max(1u, std::thread::hardware_concurrency());
}

// A hardware-relative default that leaves roughly one fifth of the process'
// available logical CPUs for other work. Round up so it never falls below
// 80%; the minimum of one also covers single-CPU containers.
inline int automatic_cpu_thread_count(unsigned logical_cpus =
                                          available_logical_cpu_count()) {
  logical_cpus = std::max(1u, logical_cpus);
  return static_cast<int>((logical_cpus * 4u + 4u) / 5u);
}

} // namespace zeromoe::hardware
