#include "qwen_cpu_offload_ring.hpp"
#include "graph_config.hpp"

#include "logger.hpp"
#include "memory_guard.hpp"

#include <algorithm>
#include <chrono>
#include <cuda_runtime.h>
#include <cstdlib>
#include <limits>
#include <thread>

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

QwenCpuOffloadRing &QwenCpuOffloadRing::instance() {
  static QwenCpuOffloadRing ring;
  return ring;
}

namespace {

int ring_spin_budget_us() {
  static const int budget = [] {
    const char *value = zeromoe::graphcfg::env("ZEROMOE_QWEN4EXP_RING_SPIN_US");
    if (!value || !value[0])
      return 0;
    char *end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (end == value || *end != '\0' || parsed < 0)
      return 0;
    return static_cast<int>(std::min<long>(parsed, 1000));
  }();
  return budget;
}

inline void ring_pause() {
#if defined(__x86_64__) || defined(__i386__)
  _mm_pause();
#else
  std::this_thread::yield();
#endif
}

} // namespace

QwenCpuOffloadRing::~QwenCpuOffloadRing() { destroy(); }

bool QwenCpuOffloadRing::ensure(int device, int d_model, int requested_slots) {
  const int slots = std::max(1, std::min(requested_slots, kMaxSlots));
  if (device < 0 || d_model <= 0)
    return false;
  if (copy_stream_ && device_ == device && d_model_ == d_model &&
      slot_count_ == slots)
    return true;

  destroy();
  cudaSetDevice(device);
  bytes_ = static_cast<size_t>(d_model) * sizeof(float);
  const size_t host_bytes = bytes_ * static_cast<size_t>(slots);
  if (host_bytes == 0 ||
      !MemoryGuard::try_reserve_host_bytes(host_bytes)) {
    LOG_WARN("[CPU-OFFLOAD-RING] refusing %.3f MiB of pinned output slots\n",
             static_cast<double>(host_bytes) / (1024.0 * 1024.0));
    return false;
  }
  host_reserved_bytes_ = host_bytes;

  cudaStream_t copy = nullptr;
  if (cudaStreamCreateWithFlags(&copy, cudaStreamNonBlocking) != cudaSuccess) {
    MemoryGuard::release_host_bytes(host_reserved_bytes_);
    host_reserved_bytes_ = 0;
    return false;
  }
  copy_stream_ = reinterpret_cast<void *>(copy);
  device_ = device;
  d_model_ = d_model;
  slot_count_ = slots;

  for (int i = 0; i < slots; ++i) {
    Slot &slot = slots_[static_cast<size_t>(i)];
    slot.owner = this;
    slot.h2d_stream = copy_stream_;
    cudaEvent_t h2d_start = nullptr;
    cudaEvent_t h2d_done = nullptr;
    cudaEvent_t consumed = nullptr;
    if (cudaMallocHost(reinterpret_cast<void **>(&slot.host), bytes_) !=
            cudaSuccess ||
        cudaMalloc(&slot.device, bytes_) != cudaSuccess ||
        cudaEventCreate(&h2d_start) != cudaSuccess ||
        cudaEventCreate(&h2d_done) != cudaSuccess ||
        cudaEventCreateWithFlags(&consumed, cudaEventDisableTiming) !=
            cudaSuccess) {
      if (h2d_start) cudaEventDestroy(h2d_start);
      if (h2d_done) cudaEventDestroy(h2d_done);
      if (consumed) cudaEventDestroy(consumed);
      destroy();
      return false;
    }
    slot.h2d_start = reinterpret_cast<void *>(h2d_start);
    slot.h2d_done = reinterpret_cast<void *>(h2d_done);
    slot.consumed = reinterpret_cast<void *>(consumed);
    // A slot that has never been consumed is immediately available.  The
    // event is still recorded so a future diagnostic query never observes an
    // uninitialized CUDA event.
    cudaEventRecord(reinterpret_cast<cudaEvent_t>(slot.consumed), copy);
    slot.callback_state.store(0, std::memory_order_relaxed);
    slot.in_use = false;
    slot.consumed_recorded = false;
    slot.h2d_collected = false;
  }

  next_slot_ = 0;
  h2d_total_ms_ = 0.0;
  h2d_copies_ = 0;
  fence_failures_ = 0;
  reuse_waits_ = 0;
  LOG_INFO("[CPU-OFFLOAD-RING] ready: GPU=%d slots=%d output=%.1f KiB\n",
           device_, slot_count_, static_cast<double>(bytes_) / 1024.0);
  return true;
}

bool QwenCpuOffloadRing::acquire(int &slot_index, double &reuse_wait_ms) {
  slot_index = -1;
  reuse_wait_ms = 0.0;
  if (!copy_stream_ || slot_count_ <= 0)
    return false;

  cudaSetDevice(device_);
  Slot &slot = slots_[static_cast<size_t>(next_slot_)];
  if (slot.in_use && !slot.consumed_recorded)
    return false;
  if (slot.in_use && slot.consumed_recorded) {
    const auto start = std::chrono::steady_clock::now();
    cudaError_t ready = cudaEventQuery(reinterpret_cast<cudaEvent_t>(slot.consumed));
    if (ready == cudaErrorNotReady) {
      ++reuse_waits_;
      if (cudaEventSynchronize(reinterpret_cast<cudaEvent_t>(slot.consumed)) !=
          cudaSuccess)
        return false;
    } else if (ready != cudaSuccess) {
      cudaGetLastError();
      return false;
    }
    reuse_wait_ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - start)
                        .count();
    collect_slot_h2d(slot);
  }

  slot.in_use = true;
  slot.consumed_recorded = false;
  slot.h2d_collected = false;
  slot.compute_stream = nullptr;
  slot.callback_state.store(0, std::memory_order_release);
  slot_index = next_slot_;
  next_slot_ = (next_slot_ + 1) % slot_count_;
  return true;
}

bool QwenCpuOffloadRing::prepare_slot(int slot_index, void *compute_stream) {
  if (slot_index < 0 || slot_index >= slot_count_ || !compute_stream)
    return false;
  Slot &slot = slots_[static_cast<size_t>(slot_index)];
  if (!slot.in_use || slot.callback_state.load(std::memory_order_acquire) != 0)
    return false;
  slot.compute_stream = compute_stream;
  return true;
}

void *QwenCpuOffloadRing::completion_user(int slot_index) {
  if (slot_index < 0 || slot_index >= slot_count_)
    return nullptr;
  return &slots_[static_cast<size_t>(slot_index)];
}

void QwenCpuOffloadRing::abandon(int slot_index) {
  if (slot_index < 0 || slot_index >= slot_count_)
    return;
  Slot &slot = slots_[static_cast<size_t>(slot_index)];
  slot.in_use = false;
  slot.consumed_recorded = false;
  slot.callback_state.store(-1, std::memory_order_release);
  slot.callback_state.notify_all();
}

void QwenCpuOffloadRing::completion_callback(void *user, bool success) {
  if (!user)
    return;
  auto *slot = static_cast<Slot *>(user);
  if (!slot->owner || !slot->owner->enqueue_completion(*slot, success))
    slot->callback_state.store(-1, std::memory_order_release);
  slot->callback_state.notify_all();
}

bool QwenCpuOffloadRing::enqueue_completion(Slot &slot, bool success) {
  if (!success || !slot.compute_stream || !slot.device || !slot.host ||
      !slot.h2d_stream || !slot.h2d_start || !slot.h2d_done)
    return false;
  cudaSetDevice(device_);
  std::atomic_thread_fence(std::memory_order_release);
  const cudaStream_t copy = reinterpret_cast<cudaStream_t>(slot.h2d_stream);
  const cudaStream_t compute =
      reinterpret_cast<cudaStream_t>(slot.compute_stream);
  const cudaError_t start = cudaEventRecord(
      reinterpret_cast<cudaEvent_t>(slot.h2d_start), copy);
  const cudaError_t copy_status =
      start == cudaSuccess
          ? cudaMemcpyAsync(slot.device, slot.host, bytes_,
                            cudaMemcpyHostToDevice, copy)
          : start;
  const cudaError_t done =
      copy_status == cudaSuccess
          ? cudaEventRecord(reinterpret_cast<cudaEvent_t>(slot.h2d_done), copy)
          : copy_status;
  const cudaError_t wait =
      done == cudaSuccess
          ? cudaStreamWaitEvent(compute,
                                reinterpret_cast<cudaEvent_t>(slot.h2d_done),
                                0)
          : done;
  if (wait != cudaSuccess) {
    ++fence_failures_;
    LOG_WARN("[CPU-OFFLOAD-RING] H2D/fence enqueue failed: %s\n",
             cudaGetErrorString(wait));
    return false;
  }
  slot.callback_state.store(1, std::memory_order_release);
  return true;
}

bool QwenCpuOffloadRing::wait_fence_enqueued(int slot_index, double &wait_ms,
                                             bool &cpu_success) {
  wait_ms = 0.0;
  cpu_success = false;
  if (slot_index < 0 || slot_index >= slot_count_)
    return false;
  Slot &slot = slots_[static_cast<size_t>(slot_index)];
  const auto start = std::chrono::steady_clock::now();
  int state = slot.callback_state.load(std::memory_order_acquire);
  const int spin_us = ring_spin_budget_us();
  const auto spin_deadline =
      start + std::chrono::microseconds(spin_us);
  while (state == 0) {
    if (spin_us > 0) {
      while (state == 0 && std::chrono::steady_clock::now() < spin_deadline) {
        ring_pause();
        state = slot.callback_state.load(std::memory_order_acquire);
      }
      if (state != 0)
        break;
    }
    slot.callback_state.wait(0, std::memory_order_acquire);
    state = slot.callback_state.load(std::memory_order_acquire);
  }
  wait_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start)
                .count();
  cpu_success = state == 1;
  return cpu_success;
}

float *QwenCpuOffloadRing::host_output(int slot_index) {
  if (slot_index < 0 || slot_index >= slot_count_)
    return nullptr;
  return slots_[static_cast<size_t>(slot_index)].host;
}

const void *QwenCpuOffloadRing::device_output(int slot_index) const {
  if (slot_index < 0 || slot_index >= slot_count_)
    return nullptr;
  return slots_[static_cast<size_t>(slot_index)].device;
}

bool QwenCpuOffloadRing::mark_consumed(int slot_index, void *compute_stream) {
  if (slot_index < 0 || slot_index >= slot_count_ || !compute_stream)
    return false;
  Slot &slot = slots_[static_cast<size_t>(slot_index)];
  cudaSetDevice(device_);
  const cudaError_t e = cudaEventRecord(
      reinterpret_cast<cudaEvent_t>(slot.consumed),
      reinterpret_cast<cudaStream_t>(compute_stream));
  if (e != cudaSuccess)
    return false;
  slot.consumed_recorded = true;
  return true;
}

void QwenCpuOffloadRing::collect_slot_h2d(Slot &slot) {
  if (slot.h2d_collected || !slot.h2d_start || !slot.h2d_done)
    return;
  float ms = 0.0f;
  if (cudaEventElapsedTime(&ms,
                           reinterpret_cast<cudaEvent_t>(slot.h2d_start),
                           reinterpret_cast<cudaEvent_t>(slot.h2d_done)) ==
      cudaSuccess) {
    h2d_total_ms_ += static_cast<double>(ms);
    ++h2d_copies_;
    slot.h2d_collected = true;
  }
}

void QwenCpuOffloadRing::destroy() {
  if (device_ >= 0)
    cudaSetDevice(device_);
  for (int i = 0; i < slot_count_; ++i) {
    Slot &slot = slots_[static_cast<size_t>(i)];
    if (slot.in_use && slot.consumed_recorded && slot.consumed)
      cudaEventSynchronize(reinterpret_cast<cudaEvent_t>(slot.consumed));
    collect_slot_h2d(slot);
    if (slot.h2d_start)
      cudaEventDestroy(reinterpret_cast<cudaEvent_t>(slot.h2d_start));
    if (slot.h2d_done)
      cudaEventDestroy(reinterpret_cast<cudaEvent_t>(slot.h2d_done));
    if (slot.consumed)
      cudaEventDestroy(reinterpret_cast<cudaEvent_t>(slot.consumed));
    if (slot.device)
      cudaFree(slot.device);
    if (slot.host)
      cudaFreeHost(slot.host);
    slot.owner = nullptr;
    slot.host = nullptr;
    slot.device = nullptr;
    slot.compute_stream = nullptr;
    slot.h2d_stream = nullptr;
    slot.h2d_start = nullptr;
    slot.h2d_done = nullptr;
    slot.consumed = nullptr;
    slot.callback_state.store(0, std::memory_order_relaxed);
    slot.in_use = false;
    slot.consumed_recorded = false;
    slot.h2d_collected = false;
  }
  if (copy_stream_)
    cudaStreamDestroy(reinterpret_cast<cudaStream_t>(copy_stream_));
  copy_stream_ = nullptr;
  if (host_reserved_bytes_) {
    MemoryGuard::release_host_bytes(host_reserved_bytes_);
    host_reserved_bytes_ = 0;
  }
  device_ = -1;
  d_model_ = 0;
  slot_count_ = 0;
  next_slot_ = 0;
  bytes_ = 0;
}
