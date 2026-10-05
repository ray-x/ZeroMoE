#pragma once

// Host-side storage boundary for partial-layer execution.  This service never
// calls CUDA and never waits for storage from a CUDA host callback.  A callback
// submits an immutable request and returns; a worker publishes either a ready
// or error completion tagged with the request generation.  The completion
// graph (or the existing host scheduler) consumes only a matching generation.

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <optional>
#include <thread>
#include <utility>
#include <vector>

namespace moestream::qwen4exp {

enum class StorageState : uint8_t { Free, Pending, Ready, Error };

struct StorageRequest {
  uint64_t model_generation = 0;
  uint64_t request_generation = 0;
  int layer = -1;
  std::vector<int32_t> experts;
};

struct StorageCompletion {
  uint64_t model_generation = 0;
  uint64_t request_generation = 0;
  int layer = -1;
  StorageState state = StorageState::Error;
  int error_code = 0;
  // Host-side scheduling telemetry. These values are advisory and are
  // intentionally published with the completion so a later completion graph
  // can choose CPU versus staged-GPU without reading mutable service state.
  double queue_ms = 0.0;
  double service_ms = 0.0;
  std::vector<int32_t> experts;
};

struct StorageStats {
  uint64_t submitted = 0;
  uint64_t rejected = 0;
  uint64_t ready = 0;
  uint64_t errors = 0;
  double queue_ms_total = 0.0;
  double service_ms_total = 0.0;
};

struct StorageHandle {
  uint32_t slot = UINT32_MAX;
  uint64_t model_generation = 0;
  uint64_t request_generation = 0;

  bool valid() const { return slot != UINT32_MAX; }
};

// The loader owns the actual DDR/SSD operation.  It must not call CUDA APIs;
// it may fill the completion payload or return false with an error code.  The
// request and its expert list remain immutable for the duration of the call.
using StorageLoader = std::function<bool(const StorageRequest &,
                                         StorageCompletion &)>;

class StorageService {
 public:
  explicit StorageService(size_t slots, size_t queue_capacity,
                          StorageLoader loader)
      : slots_(slots), queue_capacity_(queue_capacity), loader_(std::move(loader)) {
    if (slots_ == 0) slots_ = 1;
    if (queue_capacity_ == 0) queue_capacity_ = slots_;
    records_.resize(slots_);
    worker_ = std::thread([this] { run(); });
  }

  StorageService(const StorageService &) = delete;
  StorageService &operator=(const StorageService &) = delete;

  ~StorageService() { shutdown(); }

  std::optional<StorageHandle> submit(StorageRequest request) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (stopping_ || queue_.size() >= queue_capacity_) {
      ++rejected_;
      return std::nullopt;
    }
    for (uint32_t slot = 0; slot < records_.size(); ++slot) {
      auto &record = records_[slot];
      if (record.state != StorageState::Free)
        continue;
      if (request.model_generation == 0 || request.request_generation == 0 ||
          request.layer < 0 || request.experts.empty())
        { ++rejected_; return std::nullopt; }
      record.request = std::move(request);
      record.completion = {};
      record.completion.model_generation = record.request.model_generation;
      record.completion.request_generation = record.request.request_generation;
      record.completion.layer = record.request.layer;
      record.enqueued_at = std::chrono::steady_clock::now();
      record.state = StorageState::Pending;
      queue_.push_back(slot);
      ++submitted_;
      wake_.notify_one();
      return StorageHandle{slot, record.request.model_generation,
                           record.request.request_generation};
    }
    return std::nullopt;
  }

  std::optional<StorageCompletion> poll(StorageHandle handle) const {
    if (!handle.valid() || handle.slot >= records_.size())
      return std::nullopt;
    std::lock_guard<std::mutex> lock(mutex_);
    const auto &record = records_[handle.slot];
    const StorageState state = record.state;
    if (state != StorageState::Ready && state != StorageState::Error)
      return std::nullopt;
    if (record.completion.model_generation != handle.model_generation ||
        record.completion.request_generation != handle.request_generation)
      return std::nullopt;
    return record.completion;
  }

  std::optional<StorageCompletion> wait(StorageHandle handle) const {
    if (!handle.valid() || handle.slot >= records_.size())
      return std::nullopt;
    std::unique_lock<std::mutex> lock(mutex_);
    done_.wait(lock, [&] {
      const auto state = records_[handle.slot].state;
      return stopping_ || state == StorageState::Ready || state == StorageState::Error;
    });
    const auto &record = records_[handle.slot];
    if (record.completion.model_generation != handle.model_generation ||
        record.completion.request_generation != handle.request_generation)
      return std::nullopt;
    return record.completion;
  }

  bool release(StorageHandle handle) {
    if (!handle.valid() || handle.slot >= records_.size())
      return false;
    std::lock_guard<std::mutex> lock(mutex_);
    auto &record = records_[handle.slot];
    if (record.completion.model_generation != handle.model_generation ||
        record.completion.request_generation != handle.request_generation)
      return false;
    const auto state = record.state;
    if (state != StorageState::Ready && state != StorageState::Error)
      return false;
    record.request = {};
    record.completion = {};
    record.state = StorageState::Free;
    return true;
  }

  StorageStats stats() const {
    StorageStats result;
    result.submitted = submitted_.load(std::memory_order_relaxed);
    result.rejected = rejected_.load(std::memory_order_relaxed);
    result.ready = ready_.load(std::memory_order_relaxed);
    result.errors = errors_.load(std::memory_order_relaxed);
    result.queue_ms_total = queue_ms_total_.load(std::memory_order_relaxed);
    result.service_ms_total = service_ms_total_.load(std::memory_order_relaxed);
    return result;
  }

  void shutdown() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (stopping_) return;
      stopping_ = true;
    }
    wake_.notify_all();
    done_.notify_all();
    if (worker_.joinable()) worker_.join();
  }

 private:
  struct Record {
    StorageState state = StorageState::Free;
    StorageRequest request;
    StorageCompletion completion;
    std::chrono::steady_clock::time_point enqueued_at{};
  };

  void run() {
    for (;;) {
      uint32_t slot = UINT32_MAX;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        wake_.wait(lock, [&] { return stopping_ || !queue_.empty(); });
        if (queue_.empty() && stopping_) return;
        slot = queue_.front();
        queue_.pop_front();
      }
      StorageRequest request;
      std::chrono::steady_clock::time_point enqueued_at;
      {
        std::lock_guard<std::mutex> lock(mutex_);
        request = records_[slot].request;
        enqueued_at = records_[slot].enqueued_at;
      }
      StorageCompletion completion;
      completion.model_generation = request.model_generation;
      completion.request_generation = request.request_generation;
      completion.layer = request.layer;
      completion.experts = request.experts;
      const auto service_start = std::chrono::steady_clock::now();
      completion.queue_ms = std::chrono::duration<double, std::milli>(
          service_start - enqueued_at).count();
      bool ok = false;
      try {
        ok = loader_ && loader_(request, completion);
      } catch (...) {
        completion.error_code = -2;
        ok = false;
      }
      completion.service_ms = std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - service_start).count();
      completion.state = ok ? StorageState::Ready : StorageState::Error;
      if (!ok && completion.error_code == 0) completion.error_code = -1;
      const double queue_ms = completion.queue_ms;
      const double service_ms = completion.service_ms;
      {
        std::lock_guard<std::mutex> lock(mutex_);
        records_[slot].completion = std::move(completion);
        records_[slot].state = ok ? StorageState::Ready : StorageState::Error;
      }
      queue_ms_total_.fetch_add(queue_ms, std::memory_order_relaxed);
      service_ms_total_.fetch_add(service_ms, std::memory_order_relaxed);
      (ok ? ready_ : errors_).fetch_add(1, std::memory_order_relaxed);
      done_.notify_all();
    }
  }

  mutable std::mutex mutex_;
  mutable std::condition_variable done_;
  std::condition_variable wake_;
  std::vector<Record> records_;
  std::deque<uint32_t> queue_;
  size_t slots_ = 0;
  size_t queue_capacity_ = 0;
  StorageLoader loader_;
  std::thread worker_;
  bool stopping_ = false;
  std::atomic<uint64_t> submitted_{0}, rejected_{0}, ready_{0}, errors_{0};
  std::atomic<double> queue_ms_total_{0.0}, service_ms_total_{0.0};
};

}  // namespace moestream::qwen4exp
