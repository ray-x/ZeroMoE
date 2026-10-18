// Tier-1 microbenchmark for QwenCpuOffloadRing's wait latency.
//
// Isolates the effect of MOESTREAM_QWEN4EXP_RING_SPIN_US on how quickly the
// decode thread notices a completed CPU-offload slot, independent of decode-
// loop noise (model load, GPU dense/attention work, prompt variance). Links
// the REAL models/qwen/qwen_cpu_offload_ring.cpp -- no GGML/model dependency -- and
// stands a plain std::thread in for GgmlQwenCpuMoeRunner's worker.
//
// MOESTREAM_QWEN4EXP_RING_SPIN_US is cached on first read inside
// qwen_cpu_offload_ring.cpp, so one process observes exactly one value.
// Sweep it externally (see bench/run_ring_spin_sweep.sh), one process launch
// per value.
//
// detect_latency = time from "worker finished its simulated compute" to
// "decode thread's wait_fence_enqueued() returned" -- includes the ring's
// own enqueue cost (two cudaEventRecord, one cudaMemcpyAsync, one
// cudaStreamWaitEvent) plus whatever spin/futex-wake cost this benchmark is
// isolating. ring_wait_us is the SAME event measured by the ring's own
// wait_ms out-param, i.e. exactly what production logs as
// qwen_cpu_offload_fence_wait -- reported alongside detect_us as a
// cross-check that this bench's external timing agrees with the real
// telemetry path. total_wait additionally includes the simulated compute
// time and is reported only for context.
//
// Two real bugs an earlier version of this file had, found while
// validating it (2026-09-27) -- both fixed here, described for anyone
// re-deriving this benchmark independently:
//   1. signal_ns was stamped by the worker thread AFTER calling
//      completion_callback(), not before. When the spin budget comfortably
//      exceeded the compute time, the main thread could observe the state
//      flip and return from wait_fence_enqueued() before the worker
//      executed its own next line -- a real race, producing spurious
//      near-zero detect_latency samples and a large drop in kept-sample
//      count. Fixed by stamping signal_ns before the call, which makes
//      wait_end >= signal a hardware guarantee instead of a race.
//   2. std::this_thread::sleep_for() below ~1ms is noisy on Linux (timer
//      granularity/scheduler wakeup jitter), so it did not reliably control
//      when the signal fired at short compute times. Fixed with a
//      calibrated busy-wait against the same now_ns() clock the main
//      thread times against.

#include "qwen_cpu_offload_ring.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <numeric>
#include <thread>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;

uint64_t now_ns() {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          Clock::now().time_since_epoch())
          .count());
}

int parse_positive(const char *value, int fallback) {
  if (!value || !value[0])
    return fallback;
  char *end = nullptr;
  const long parsed = std::strtol(value, &end, 10);
  if (end == value || *end != '\0' || parsed <= 0)
    return fallback;
  return static_cast<int>(parsed);
}

void summarize(std::vector<double> &v, const char *name) {
  if (v.empty()) {
    std::printf("[RING-SPIN-BENCH] %-32s n=0 (no samples)\n", name);
    return;
  }
  std::sort(v.begin(), v.end());
  const double sum = std::accumulate(v.begin(), v.end(), 0.0);
  const double mean = sum / static_cast<double>(v.size());
  const double p50 = v[v.size() / 2];
  const double p90 = v[static_cast<size_t>(v.size() * 0.9)];
  const double maxv = v.back();
  std::printf(
      "[RING-SPIN-BENCH] %-32s n=%zu mean=%.2fus p50=%.2fus p90=%.2fus "
      "max=%.2fus\n",
      name, v.size(), mean, p50, p90, maxv);
}

} // namespace

int main(int argc, char **argv) {
  const int iterations = argc > 1 ? parse_positive(argv[1], 200) : 200;
  const int d_model = argc > 2 ? parse_positive(argv[2], 2560) : 2560;
  const int slots = argc > 3 ? parse_positive(argv[3], 2) : 2;
  const int cpu_compute_us = argc > 4 ? parse_positive(argv[4], 1000) : 1000;

  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
    std::fprintf(stderr, "[RING-SPIN-BENCH] no CUDA device available\n");
    return 2;
  }
  if (cudaSetDevice(0) != cudaSuccess) {
    std::fprintf(stderr, "[RING-SPIN-BENCH] cudaSetDevice failed\n");
    return 1;
  }

  auto &ring = QwenCpuOffloadRing::instance();
  if (!ring.ensure(0, d_model, slots)) {
    std::fprintf(stderr, "[RING-SPIN-BENCH] ring.ensure() failed\n");
    return 1;
  }

  cudaStream_t compute_stream = nullptr;
  if (cudaStreamCreateWithFlags(&compute_stream, cudaStreamNonBlocking) !=
      cudaSuccess) {
    std::fprintf(stderr, "[RING-SPIN-BENCH] stream create failed\n");
    return 1;
  }

  const char *spin_env = std::getenv("MOESTREAM_QWEN4EXP_RING_SPIN_US");
  std::printf("[RING-SPIN-BENCH] iterations=%d d_model=%d slots=%d "
              "cpu_compute_us=%d MOESTREAM_QWEN4EXP_RING_SPIN_US=%s\n",
              iterations, d_model, slots, cpu_compute_us,
              spin_env ? spin_env : "(unset)");

  std::vector<double> detect_us;
  std::vector<double> ring_wait_us;
  std::vector<double> total_wait_us;
  detect_us.reserve(static_cast<size_t>(iterations));
  ring_wait_us.reserve(static_cast<size_t>(iterations));
  total_wait_us.reserve(static_cast<size_t>(iterations));
  int failures = 0;

  for (int i = 0; i < iterations; ++i) {
    int slot = -1;
    double reuse_wait_ms = 0.0;
    if (!ring.acquire(slot, reuse_wait_ms)) {
      ++failures;
      continue;
    }
    if (!ring.prepare_slot(slot, compute_stream)) {
      ring.abandon(slot);
      ++failures;
      continue;
    }

    std::atomic<uint64_t> signal_ns{0};
    void *user = ring.completion_user(slot);
    std::thread worker([cpu_compute_us, user, &signal_ns] {
      // Calibrated busy-wait, not sleep_for(): sleep_for() below ~1ms is
      // jittery on Linux (timer granularity/scheduler wakeup latency),
      // which made an earlier version of this bench unreliable at short
      // compute times. now_ns() is the same clock the main thread times
      // against, so this gives a precise, comparable delay.
      const uint64_t deadline =
          now_ns() + static_cast<uint64_t>(cpu_compute_us) * 1000ULL;
      while (now_ns() < deadline) {
      }
      // Stamped BEFORE calling completion_callback, not after it returns:
      // an earlier version stamped signal_ns after the call, which raced
      // against the main thread's own spin-detection when the spin budget
      // comfortably exceeded the compute time -- the main thread could
      // observe the state flip and return from wait_fence_enqueued()
      // before this thread executed its own next instruction. Stamping
      // first makes wait_end >= signal a hardware guarantee, not a race,
      // and makes detect_us measure exactly the ring's own enqueue+wake
      // cost (event records, memcpy, stream wait, then notify) with
      // nothing else mixed in.
      signal_ns.store(now_ns(), std::memory_order_release);
      QwenCpuOffloadRing::completion_callback(user, true);
    });

    const uint64_t wait_start = now_ns();
    double wait_ms = 0.0;
    bool cpu_success = false;
    const bool fenced = ring.wait_fence_enqueued(slot, wait_ms, cpu_success);
    const uint64_t wait_end = now_ns();
    worker.join();

    if (!fenced || !cpu_success) {
      ++failures;
    } else {
      const uint64_t signal = signal_ns.load(std::memory_order_acquire);
      if (signal > 0 && wait_end >= signal)
        detect_us.push_back(static_cast<double>(wait_end - signal) /
                             1000.0);
      // The ring's own self-measured wait time -- the exact quantity
      // production logs as qwen_cpu_offload_fence_wait. Comparing this
      // against detect_us (this bench's external measurement of the same
      // event) cross-checks the bench against the real telemetry path.
      ring_wait_us.push_back(wait_ms * 1000.0);
      total_wait_us.push_back(static_cast<double>(wait_end - wait_start) /
                               1000.0);
    }

    ring.mark_consumed(slot, compute_stream);
  }
  cudaStreamSynchronize(compute_stream);
  cudaStreamDestroy(compute_stream);

  summarize(detect_us, "detect_latency(signal->return)");
  summarize(ring_wait_us, "ring_self_wait_ms(fence_wait equiv.)");
  summarize(total_wait_us, "total_wait(acquire->return)");
  std::printf(
      "[RING-SPIN-BENCH] failures=%d h2d_total_ms=%.3f h2d_copies=%llu "
      "fence_failures=%llu reuse_waits=%llu\n",
      failures, ring.h2d_total_ms(),
      static_cast<unsigned long long>(ring.h2d_copies()),
      static_cast<unsigned long long>(ring.fence_failures()),
      static_cast<unsigned long long>(ring.reuse_waits()));
  return failures == 0 ? 0 : 1;
}
