// include/startup_preflight.hpp
//
// Real, cheap sanity checks run once at startup, BEFORE any RAM-tier
// allocation, [PIN] preload, or GPU VRAM pool allocation begins -- direct
// response to a real incident: a second zeromoe instance was launched
// (a mixed-precision smoke test) while an existing, independent zeromoe
// benchmark was already running, and the box crashed hard enough to force
// a reboot. Both processes assumed they each owned the FULL configured
// host_ram_gb/VRAM budget; running two together drove the box into swap
// thrashing (this box's own zram swap is already confirmed pathological
// for this workload -- see CLAUDE.md's "zram swap is pathological"
// entry: ~1.0x compression on already-quantized weights, i.e. real CPU
// cost for zero real capacity gained) hard enough to make the whole
// system unresponsive.
//
// Three checks, all fail-closed (refuse to start rather than silently
// proceed and risk repeating the incident):
//   1. Another live process literally named "zeromoe" -- the single
//      most direct, surgical signal for the exact incident above.
//   2. Real host RAM headroom (/proc/meminfo's MemAvailable, the kernel's
//      own "how much could actually be allocated right now without
//      swapping" estimate -- NOT MemFree, which undercounts reclaimable
//      cache) against this run's configured host_ram_gb plus a real
//      safety margin. Failing this BEFORE the [PIN] preload pass even
//      starts is the whole point -- once that pass begins pushing
//      gigabytes into a too-small remaining budget, the damage is
//      already happening.
//   3. Real free VRAM per configured GPU (cudaMemGetInfo) -- catches a
//      GPU already substantially claimed by another process (this one or
//      an unrelated CUDA workload) before this run's own pools try to
//      allocate into it and hit a real, deep-into-startup CUDA OOM.
//
// CUDA-only (this project's own main binary is always a CUDA build --
// see CLAUDE.md: "Will not build on Apple Silicon due to hard NVCC...
// dependencies" -- the CPU-only variant is the separate cpu_main.cpp,
// which does not use this header).
#pragma once

#include <cuda_runtime.h>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <dirent.h>
#include <fcntl.h>
#include <fstream>
#include <string>
#include <sys/file.h>
#include <unistd.h>
#include <vector>

namespace preflight {

// Real, current MemAvailable in decimal GB, matching host_ram_gb allocations
// (config.host_ram_gb * 1e9). Returns -1.0 if unreadable (a
// genuinely broken /proc -- this project only ever runs on Linux per its
// own hard AVX-512/CUDA dependencies, so "non-Linux" isn't a real case
// here, but a missing/unmounted /proc is still worth not hard-failing on).
inline double mem_available_gb() {
  std::ifstream f("/proc/meminfo");
  if (!f)
    return -1.0;
  std::string line;
  while (std::getline(f, line)) {
    if (line.rfind("MemAvailable:", 0) == 0) {
      long kb = 0;
      std::sscanf(line.c_str(), "MemAvailable: %ld kB", &kb);
      return static_cast<double>(kb) * 1024.0 / 1.0e9;
    }
  }
  return -1.0;
}

inline double default_host_margin_gb(double required_host_ram_gb) {
  // Preserve at least 20 GiB outside the configured process budget. The
  // earlier 10 GiB floor was insufficient when the DeepSeek run page-locked
  // almost all of its DDR expert cache and built the long-prompt graph.
  constexpr double floor_gb = 20.0 * 1024.0 * 1024.0 * 1024.0 / 1.0e9;
  const double proportional_gb = required_host_ram_gb * 0.05;
  return proportional_gb > floor_gb ? proportional_gb : floor_gb;
}

// Scans /proc/<pid>/comm for any OTHER live process literally named
// "zeromoe" (this binary's own comm name -- confirmed via /proc, not
// argv[0], since argv[0] can be an arbitrary invocation path while comm
// is the kernel's own 15-byte task name, stable regardless of how the
// binary was invoked). own_pid is excluded so this process is never
// self-flagged. A read failing for an individual /proc/<pid> entry (the
// process exited between readdir() and the open, a normal race) is
// silently skipped, not treated as a match.
inline std::vector<int> find_other_zeromoe_pids(pid_t own_pid) {
  std::vector<int> found;
  DIR *d = opendir("/proc");
  if (!d)
    return found;
  struct dirent *entry;
  while ((entry = readdir(d)) != nullptr) {
    if (entry->d_type != DT_DIR)
      continue;
    char *end = nullptr;
    long pid = std::strtol(entry->d_name, &end, 10);
    if (end == entry->d_name || *end != '\0' || pid <= 0 || pid == own_pid)
      continue;
    std::ifstream f("/proc/" + std::string(entry->d_name) + "/comm");
    std::string comm;
    if (f && std::getline(f, comm) && comm == "zeromoe")
      found.push_back((int)pid);
  }
  closedir(d);
  return found;
}

// The PID scan below is useful for diagnostics, but it is not an atomic
// admission decision: two instances can both scan an empty /proc and then
// allocate at the same time. Hold one small process-lifetime flock so the
// multi-session freeze incident cannot recur through that TOCTOU window.
// --force-start remains an explicit operator escape hatch for deliberately
// disjoint workloads.
inline int host_admission_lock_fd = -1;

inline bool acquire_host_admission_lock() {
  if (host_admission_lock_fd >= 0) return true;
  constexpr const char *kLockPath = "/tmp/zeromoe-host-admission.lock";
  const int fd = open(kLockPath, O_CREAT | O_RDWR | O_CLOEXEC, 0600);
  if (fd < 0) {
    fprintf(stderr,
            "[PREFLIGHT] FATAL: cannot open %s for the process-wide host "
            "admission lock: %s\n",
            kLockPath, std::strerror(errno));
    return false;
  }
  if (flock(fd, LOCK_EX | LOCK_NB) != 0) {
    fprintf(stderr,
            "[PREFLIGHT] FATAL: another ZeroMoE process currently holds "
            "the host admission lock; refusing concurrent startup. Stop it "
            "first, or pass --force-start only when the workloads are known "
            "to be safe and disjoint (%s).\n",
            std::strerror(errno));
    close(fd);
    return false;
  }
  host_admission_lock_fd = fd;
  return true;
}

inline void release_host_admission_lock() {
  if (host_admission_lock_fd < 0) return;
  flock(host_admission_lock_fd, LOCK_UN);
  close(host_admission_lock_fd);
  host_admission_lock_fd = -1;
}

// Runs all three checks and prints a clear FATAL for each one that fails
// (never silently proceeds). Returns true iff it is safe to continue.
// `required_host_ram_gb` is this run's own resolved config.host_ram_gb
// process-wide ceiling (MoE + resident/KV/state + graph + MTP);
// margin_gb defaults to a real, previously-necessary headroom (max(20 GiB,
// 5% of the budget), expressed in decimal GB -- see CLAUDE.md's "host_ram_gb swap-thrashing
// regression" entry: exactly-the-budget was already once a real,
// measured, reboot-surviving regression on this box, not a hypothetical).
// gpu_device_ids is this run's own resolved config.gpus[*].device_id
// list. `force` (wired to a new --force-start CLI flag, main.cpp-local,
// not a TOML config field) lets an operator proceed anyway when they are
// certain it's safe -- e.g. two zeromoe instances deliberately using
// disjoint GPU sets -- but always after seeing every real number that
// would otherwise have blocked the start.
inline bool run(double required_host_ram_gb,
                 const std::vector<int> &gpu_device_ids, bool force,
                 double margin_gb = -1.0) {
  // Preserve 20 GiB for the kernel, drivers, graph scratch, and desktop
  // services. The previous 10 GiB floor did not protect the host during the
  // full-DDR-page-pinned DeepSeek graph run.
  if (margin_gb < 0.0)
    margin_gb = default_host_margin_gb(required_host_ram_gb);

  bool ok = true;

  pid_t self = getpid();
  auto others = find_other_zeromoe_pids(self);
  if (!others.empty()) {
    fprintf(stderr,
            "[PREFLIGHT] FATAL: %zu other zeromoe process(es) already "
            "running (pid",
            others.size());
    for (int p : others)
      fprintf(stderr, " %d", p);
    fprintf(stderr,
            ") -- each instance assumes it owns the FULL configured "
            "host_ram_gb/VRAM budget; running two together is a real, "
            "previously-confirmed way to drive this box into swap "
            "thrashing or a hard crash requiring a reboot. Stop the "
            "other instance first, or pass --force-start if you are "
            "certain this is safe (e.g. deliberately disjoint GPU "
            "sets).\n");
    ok = false;
  }

  bool admission_lock_held = false;
  if (ok && !force) {
    admission_lock_held = acquire_host_admission_lock();
    if (!admission_lock_held) ok = false;
  }

  double avail = mem_available_gb();
  if (avail >= 0.0) {
    double need = required_host_ram_gb + margin_gb;
    if (avail < need) {
      fprintf(stderr,
              "[PREFLIGHT] FATAL: only %.1f GB of real, reclaimable RAM "
              "available right now (/proc/meminfo MemAvailable), but this "
              "run is configured for host_ram_gb=%.1f GB (+%.1f GB safety "
              "margin = %.1f GB needed). Starting anyway would push the "
              "model allocations (RAM tier, graph, and MTP) straight into swap -- "
              "this box's own zram swap is already confirmed pathological "
              "for this workload (real CPU cost, ~1.0x compression on "
              "already-quantized expert weights -- zero real capacity "
              "gained). Free memory (check for another zeromoe or "
              "otherwise heavy process), or lower host_ram_gb, and "
              "retry -- or pass --force-start if you are certain this is "
              "safe.\n",
              avail, required_host_ram_gb, margin_gb, need);
      ok = false;
    } else {
      fprintf(stderr,
              "[PREFLIGHT] OK: %.1f GB RAM available, %.1f GB needed "
              "(host_ram_gb=%.1f + %.1f GB margin).\n",
              avail, need, required_host_ram_gb, margin_gb);
    }
  } else {
    fprintf(stderr, "[PREFLIGHT] NOTE: could not read /proc/meminfo -- "
                    "skipping the real-RAM-available check.\n");
  }

  for (int dev : gpu_device_ids) {
    size_t free_b = 0, total_b = 0;
    cudaError_t prev_err = cudaGetLastError(); // clear any stale sticky error first
    (void)prev_err;
    cudaError_t set_err = cudaSetDevice(dev);
    cudaError_t info_err = (set_err == cudaSuccess)
                                ? cudaMemGetInfo(&free_b, &total_b)
                                : set_err;
    if (info_err != cudaSuccess) {
      fprintf(stderr,
              "[PREFLIGHT] NOTE: cudaMemGetInfo failed for GPU %d (%s) -- "
              "skipping the VRAM check for this device.\n",
              dev, cudaGetErrorString(info_err));
      continue;
    }
    double free_gb = free_b / 1073741824.0;
    double total_gb = total_b / 1073741824.0;
    // Not trying to reproduce this engine's own precise per-GPU VRAM
    // budget math here (dense_gb/moe_cache_gb/active_layer_capacity/WUK
    // caches all interact -- see engine_config.hpp) -- a real, absolute
    // "this GPU is basically already fully claimed by someone else" floor
    // is a robust, config-independent signal on its own.
    const double kMinFreeGb = 2.0;
    if (free_gb < kMinFreeGb) {
      fprintf(stderr,
              "[PREFLIGHT] FATAL: GPU %d has only %.2f/%.2f GB VRAM free "
              "-- looks already substantially claimed by another process "
              "(this box's own GPUs should be near-empty before this "
              "engine's own pools allocate). Check `nvidia-smi`, free the "
              "GPU, or pass --force-start if you are certain this is "
              "safe.\n",
              dev, free_gb, total_gb);
      ok = false;
    } else {
      fprintf(stderr, "[PREFLIGHT] OK: GPU %d has %.2f/%.2f GB VRAM free.\n",
              dev, free_gb, total_gb);
    }
  }

  if (!ok && force) {
    fprintf(stderr,
            "[PREFLIGHT] --force-start given: proceeding DESPITE the "
            "failure(s) above. This is exactly the condition that "
            "previously caused a real crash on this box -- only do this "
            "if you are certain it's safe.\n");
    return true;
  }
  if (!ok && admission_lock_held) release_host_admission_lock();
  return ok;
}

} // namespace preflight
