#pragma once

// Run-time preflight checks that need counters or measurements, the other half of the checks in
// graph_settings.hpp (docs/design/model-family-framework.md s4.3). Pure: callers pass numbers and
// decide what to do with the findings (warn, or abort under graph_preflight_strict=true).

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace zeromoe::graph {

// Idle VRAM above this after the decode graph is built means a bigger expert cache fits.
inline constexpr double kIdleVramWarnGiB = 4.0;

struct ExpertCoverage {
  uint64_t total = 0, vram = 0, ddr = 0, duplicated = 0, covered = 0;
  uint64_t missing() const { return total - covered; }
};

// Call at a quiescent startup boundary, after seeding has finished. DDR means
// valid, ready data, not capacity, a claim, or a read that is still in flight.
// The same expert in two tiers counts once, and another layer is a new identity.
template <class InVram, class ReadyInDdr>
inline ExpertCoverage measure_expert_coverage(int first_layer, int n_layers, int n_experts,
                                               InVram in_vram, ReadyInDdr ready_in_ddr) {
  ExpertCoverage c;
  for (int l = first_layer; l < first_layer + n_layers; ++l) {
    for (int e = 0; e < n_experts; ++e) {
      const bool gpu = in_vram(l, e), ram = ready_in_ddr(l, e);
      ++c.total;
      c.vram += gpu;
      c.ddr += ram;
      c.duplicated += gpu && ram;
      c.covered += gpu || ram;
    }
  }
  return c;
}

inline std::string coverage_summary(const ExpertCoverage &c) {
  char buf[240];
  std::snprintf(buf, sizeof buf, "unique=%llu/%llu VRAM=%llu DDR-ready=%llu duplicated=%llu missing=%llu",
                (unsigned long long)c.covered, (unsigned long long)c.total,
                (unsigned long long)c.vram, (unsigned long long)c.ddr,
                (unsigned long long)c.duplicated, (unsigned long long)c.missing());
  return buf;
}

inline std::string coverage_finding(const ExpertCoverage &c) {
  if (c.missing() == 0) return {};
  char buf[320];
  std::snprintf(buf, sizeof buf,
                "expert coverage %.1f%% (%s): uncovered experts require a demand load during decode",
                100.0 * double(c.covered) / double(c.total), coverage_summary(c).c_str());
  return buf;
}

// Store loads that happened after the first measured decode window (the warm-up window is the baseline).
inline std::string decode_miss_finding(uint64_t loads_now, uint64_t loads_baseline, int steps) {
  if (loads_now <= loads_baseline) return {};
  char buf[160];
  std::snprintf(buf, sizeof buf, "%llu expert load(s) during decode (%d steps): the DDR tier does not cover the working set",
                (unsigned long long)(loads_now - loads_baseline), steps);
  return buf;
}

inline std::string idle_vram_finding(int device, double free_gib, double threshold_gib = kIdleVramWarnGiB) {
  if (free_gib <= threshold_gib) return {};
  char buf[160];
  std::snprintf(buf, sizeof buf, "GPU %d has %.1f GiB VRAM free after the graph is built: a larger moe_cache_gb fits",
                device, free_gib);
  return buf;
}

} // namespace zeromoe::graph
