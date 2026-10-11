#pragma once

#include "graph_config.hpp"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "memory_guard.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <condition_variable>
#include <chrono>
#include <deque>
#include <functional>
#include <map>
#include <mutex>
#include <tuple>
#include <thread>
#include <vector>

extern "C" cudaError_t moestream_qwen_stage_begin(unsigned long long *, cudaStream_t);
extern "C" cudaError_t moestream_qwen_stage_publish(int *, unsigned long long *, int,
                                                    int, unsigned long long, cudaStream_t);
extern "C" cudaError_t moestream_qwen_stage_map(const int *, const unsigned long long *,
    const int *, unsigned long long *, int *, int *, float *, int, int, cudaStream_t);
extern "C" cudaError_t moestream_qwen_stage_delay(unsigned int, cudaStream_t);

// Shared host worker service for all staged layers. A single queue avoids one
// sleeping/waking thread per layer; each submitted closure still owns the
// target pool's stable bank and shape-specific CUDA tensors.
class Qwen4ExpStageWorkService {
 public:
  struct Stats {
    uint64_t submitted = 0;
    uint64_t rejected = 0;
    uint64_t completed = 0;
    double queue_ms_total = 0.0;
    double run_ms_total = 0.0;
  };

  explicit Qwen4ExpStageWorkService(size_t workers = 1) {
    workers = std::max<size_t>(1, workers);
    for (size_t i = 0; i < workers; ++i)
      threads_.emplace_back([this] { run(); });
  }
  Qwen4ExpStageWorkService(const Qwen4ExpStageWorkService &) = delete;
  Qwen4ExpStageWorkService &operator=(const Qwen4ExpStageWorkService &) = delete;
  ~Qwen4ExpStageWorkService() { shutdown(); }

  bool submit(std::function<void()> job) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (stopping_) {
      rejected_.fetch_add(1, std::memory_order_relaxed);
      return false;
    }
    jobs_.push_back({std::move(job), std::chrono::steady_clock::now()});
    submitted_.fetch_add(1, std::memory_order_relaxed);
    wake_.notify_one();
    return true;
  }

  Stats stats() const {
    return {submitted_.load(std::memory_order_relaxed),
            rejected_.load(std::memory_order_relaxed),
            completed_.load(std::memory_order_relaxed),
            queue_ms_total_.load(std::memory_order_relaxed),
            run_ms_total_.load(std::memory_order_relaxed)};
  }

  void shutdown() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (stopping_) return;
      stopping_ = true;
    }
    wake_.notify_all();
    for (auto &thread : threads_)
      if (thread.joinable()) thread.join();
    threads_.clear();
  }

 private:
  void run() {
    for (;;) {
      Job job;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        wake_.wait(lock, [this] { return stopping_ || !jobs_.empty(); });
        if (jobs_.empty() && stopping_) return;
        job = std::move(jobs_.front());
        jobs_.pop_front();
      }
      const auto start = std::chrono::steady_clock::now();
      queue_ms_total_.fetch_add(std::chrono::duration<double, std::milli>(
          start - job.enqueued_at).count(), std::memory_order_relaxed);
      job.fn();
      run_ms_total_.fetch_add(std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - start).count(),
          std::memory_order_relaxed);
      completed_.fetch_add(1, std::memory_order_relaxed);
    }
  }

  struct Job {
    std::function<void()> fn;
    std::chrono::steady_clock::time_point enqueued_at;
  };
  std::mutex mutex_;
  std::condition_variable wake_;
  std::deque<Job> jobs_;
  std::vector<std::thread> threads_;
  bool stopping_ = false;
  std::atomic<uint64_t> submitted_{0}, rejected_{0}, completed_{0};
  std::atomic<double> queue_ms_total_{0.0}, run_ms_total_{0.0};
};

// One shared worker service per compatible expert shape. The service is only
// an execution queue: GPU weight arenas and epoch banks remain per layer so a
// later layer can never overwrite a prior layer's in-flight consumer.
class Qwen4ExpStageShapeClassPool {
 public:
  struct Key {
    int topk = 0;
    std::array<int64_t, 6> shape{};
    std::array<uint32_t, 3> type{};

    bool operator<(const Key &other) const {
      return std::tie(topk, shape, type) <
             std::tie(other.topk, other.shape, other.type);
    }
  };

  Qwen4ExpStageWorkService *service_for(const Key &key) {
    auto it = services_.find(key);
    if (it != services_.end()) return it->second.get();
    auto service = std::make_unique<Qwen4ExpStageWorkService>(1);
    auto *result = service.get();
    services_.emplace(key, std::move(service));
    return result;
  }

  void shutdown() {
    for (auto &entry : services_) entry.second->shutdown();
  }

 private:
  std::map<Key, std::unique_ptr<Qwen4ExpStageWorkService>> services_;
};

// Each bank is held until both its DMA and the token that could consume it
// finish. Predictions may be late: an unpublished epoch always stays on CPU.
struct Qwen4ExpStagePool {
  struct Bank {
    std::atomic<bool> busy{false}, dma_queued{false}, prediction_sampled{true};
    unsigned long long epoch = 0;
    const int32_t *predictions = nullptr;
    const unsigned long long *prediction_epoch = nullptr;
    cudaEvent_t prediction_ready = nullptr, copied = nullptr;
    uint8_t *bounce = nullptr;
    std::chrono::steady_clock::time_point copy_enqueued_at{};
  } banks[2];
  ggml_tensor *source[3] = {}, *weights[3] = {};
  ggml_backend_t backend = nullptr;
  ggml_context *reference_ctx = nullptr;
  ggml_cgraph *reference_graph = nullptr;
  ggml_backend_buffer_t reference_buffer = nullptr, verify_buffer = nullptr;
  ggml_tensor *reference_weights[3] = {}, *reference_input = nullptr;
  ggml_tensor *reference_ids = nullptr, *reference_probs = nullptr;
  ggml_tensor *reference_output = nullptr, *verify_output = nullptr;
  Qwen4ExpStagePool *reference_owner = nullptr;
  bool verify_enabled = false;
  uint64_t verified = 0;
  double verify_max_delta = 0.0;
  double verify_expected_max_abs = 0.0;
  double verify_actual_max_abs = 0.0;
  ggml_backend_buffer_t buffers[3] = {}, metadata = nullptr;
  unsigned long long *epoch = nullptr, *slot_epochs = nullptr;
  int32_t *slot_ids = nullptr;
  std::vector<uint8_t> cold_mask;
  size_t plane_bytes[3] = {}, plane_offsets[3] = {}, bytes = 0, reservation = 0;
  int device = -1, topk = 0, slots = 0;
  unsigned int delay_us = 0;
  unsigned int worker_delay_us = 0;
  // Cap on predicted cold experts staged per layer (0 = no cap).
  unsigned int stage_max = 0;
  cudaStream_t copy_stream = nullptr;
  Qwen4ExpStageWorkService *service = nullptr;
  std::mutex mutex;
  std::condition_variable samples_finished;
  std::deque<int> jobs;
  unsigned long long completed_epoch = 0;
  std::atomic<uint64_t> requests{0}, dropped{0}, predicted{0}, uploaded{0}, late{0};
  std::atomic<uint64_t> dma_completed{0};
  std::atomic<double> dma_ms_total{0.0};

  struct Timing {
    uint64_t requests = 0, dropped = 0, predicted = 0, uploaded = 0,
             late = 0, dma_completed = 0;
    double dma_ms_total = 0.0;
  };

  Timing timing() const {
    return {requests.load(std::memory_order_relaxed),
            dropped.load(std::memory_order_relaxed),
            predicted.load(std::memory_order_relaxed),
            uploaded.load(std::memory_order_relaxed),
            late.load(std::memory_order_relaxed),
            dma_completed.load(std::memory_order_relaxed),
            dma_ms_total.load(std::memory_order_relaxed)};
  }

  ~Qwen4ExpStagePool() {
    if (device >= 0) cudaSetDevice(device);
    if (copy_stream) {
      cudaStreamSynchronize(copy_stream);
      cudaStreamDestroy(copy_stream);
    }
    for (auto &bank : banks) {
      if (bank.copied) cudaEventDestroy(bank.copied);
      if (bank.bounce) cudaFreeHost(bank.bounce);
    }
    for (auto buffer : buffers) if (buffer) ggml_backend_buffer_free(buffer);
    if (reference_buffer) ggml_backend_buffer_free(reference_buffer);
    if (reference_ctx) ggml_free(reference_ctx);
    if (verify_buffer) ggml_backend_buffer_free(verify_buffer);
    if (metadata) ggml_backend_buffer_free(metadata);
    if (reservation) MemoryGuard::release_host_bytes(reservation);
  }

  bool prepare(ggml_backend_t backend, ggml_context *ctx, int dev, int k,
               ggml_tensor *gate, ggml_tensor *up, ggml_tensor *down,
               const std::vector<uint8_t> &mask,
               Qwen4ExpStageWorkService *shared_service = nullptr) {
    device = dev; topk = k; slots = 2 * k; cold_mask = mask; this->backend = backend;
    service = shared_service;
    if (!service) return false;
    const char *verify = moestream::graphcfg::env("MOESTREAM_QWEN_EARLY_READY_STAGE_VERIFY");
    verify_enabled = verify && verify[0] == '1';
    const char *delay = moestream::graphcfg::env("MOESTREAM_QWEN_EARLY_READY_STAGE_DELAY_US");
    if (delay) delay_us = std::min<unsigned long>(1000000, std::strtoul(delay, nullptr, 10));
    const char *worker_delay = moestream::graphcfg::env("MOESTREAM_QWEN_EARLY_READY_STAGE_WORKER_DELAY_US");
    const char *stage_cap = moestream::graphcfg::env("MOESTREAM_QWEN_EARLY_READY_STAGE_MAX");
    if (stage_cap) stage_max = std::min<unsigned long>(32, std::strtoul(stage_cap, nullptr, 10));
    if (worker_delay)
      worker_delay_us = std::min<unsigned long>(1000000, std::strtoul(worker_delay, nullptr, 10));
    if (k <= 0 || k > 32) return false;
    source[0] = gate; source[1] = up; source[2] = down;
    for (int p = 0; p < 3; ++p) {
      plane_offsets[p] = bytes;
      plane_bytes[p] = ggml_row_size(source[p]->type, source[p]->ne[0]) * source[p]->ne[1];
      bytes += plane_bytes[p] * topk;
    }
    if (!MemoryGuard::try_reserve_host_bytes(2 * bytes + 8 * 1024 * 1024)) return false;
    reservation = 2 * bytes + 8 * 1024 * 1024;
    cudaSetDevice(device);
    for (int p = 0; p < 3; ++p) {
      weights[p] = ggml_new_tensor_3d(ctx, source[p]->type,
                                     source[p]->ne[0], source[p]->ne[1], slots);
      buffers[p] = ggml_backend_alloc_buffer(backend,
          ggml_backend_buft_get_alloc_size(ggml_backend_get_default_buffer_type(backend), weights[p]));
      if (!buffers[p] || ggml_backend_tensor_alloc(buffers[p], weights[p],
            ggml_backend_buffer_get_base(buffers[p])) != GGML_STATUS_SUCCESS) return false;
      ggml_backend_buffer_set_usage(buffers[p], GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
      ggml_backend_buffer_clear(buffers[p], 0);
    }
    const size_t id_offset = 64;
    const size_t ready_offset = (id_offset + slots * sizeof(int32_t) + 63) & ~size_t(63);
    metadata = ggml_backend_alloc_buffer(backend, ready_offset + slots * sizeof(unsigned long long));
    if (!metadata) return false;
    ggml_backend_buffer_clear(metadata, 0);
    auto *base = static_cast<uint8_t *>(ggml_backend_buffer_get_base(metadata));
    epoch = reinterpret_cast<unsigned long long *>(base);
    slot_ids = reinterpret_cast<int32_t *>(base + id_offset);
    slot_epochs = reinterpret_cast<unsigned long long *>(base + ready_offset);
    if (cudaStreamCreateWithFlags(&copy_stream, cudaStreamNonBlocking) != cudaSuccess) return false;
    for (auto &bank : banks) {
      if (cudaMallocHost(reinterpret_cast<void **>(&bank.bounce), bytes) != cudaSuccess ||
          cudaEventCreateWithFlags(&bank.copied, cudaEventDisableTiming) != cudaSuccess)
        return false;
    }
    return true;
  }

  bool prepare_verification(ggml_tensor *output, Qwen4ExpStagePool *shared) {
    verify_output = output;
    verify_buffer = ggml_backend_alloc_buffer(backend, ggml_nbytes(output));
    if (!verify_buffer || ggml_backend_tensor_alloc(verify_buffer, output,
            ggml_backend_buffer_get_base(verify_buffer)) != GGML_STATUS_SUCCESS) return false;
    ggml_set_output(output);
    // Verification runs sequentially after replay. Share the independently
    // loaded reference weights/workspace across layers, retaining only each
    // layer's actual output snapshot. Forty-two reference pools would consume
    // unnecessary VRAM and can prevent the model graph from being allocated.
    reference_owner = shared ? shared : this;
    if (reference_owner != this) {
      if (!reference_owner->reference_buffer || reference_owner->backend != backend ||
          reference_owner->slots != slots || reference_owner->topk != topk) return false;
      for (int p = 0; p < 3; ++p)
        if (reference_owner->source[p]->type != source[p]->type ||
            reference_owner->source[p]->ne[0] != source[p]->ne[0] ||
            reference_owner->source[p]->ne[1] != source[p]->ne[1]) return false;
      return true;
    }
    reference_ctx = ggml_init({64 * ggml_tensor_overhead() +
                               ggml_graph_overhead_custom(64, false), nullptr, true});
    if (!reference_ctx) return false;
    for (int p = 0; p < 3; ++p)
      reference_weights[p] = ggml_new_tensor_3d(reference_ctx, source[p]->type,
          source[p]->ne[0], source[p]->ne[1], slots);
    reference_input = ggml_new_tensor_3d(reference_ctx, GGML_TYPE_F32, source[0]->ne[0], 1, 1);
    reference_ids = ggml_new_tensor_2d(reference_ctx, GGML_TYPE_I32, topk, 1);
    reference_probs = ggml_new_tensor_3d(reference_ctx, GGML_TYPE_F32, 1, topk, 1);
    auto *up = ggml_mul_mat_id(reference_ctx, reference_weights[1], reference_input, reference_ids);
    auto *gate = ggml_mul_mat_id(reference_ctx, reference_weights[0], reference_input, reference_ids);
    auto *act = ggml_swiglu_split(reference_ctx, gate, up);
    auto *down = ggml_mul_mat_id(reference_ctx, reference_weights[2], act, reference_ids);
    reference_output = ggml_mul(reference_ctx, down, reference_probs);
    reference_graph = ggml_new_graph_custom(reference_ctx, 64, false);
    ggml_build_forward_expand(reference_graph, reference_output);
    reference_buffer = ggml_backend_alloc_ctx_tensors(reference_ctx, backend);
    return reference_buffer != nullptr;
  }

  void verify(const float *input, const float *probs, const int32_t *actual,
              const int32_t *selected) {
    if (!verify_enabled) return;
    GGML_ASSERT(reference_owner);
    auto &reference = *reference_owner;
    std::vector<int32_t> ids(topk, -1);
    cudaStream_t stream = nullptr;
    GGML_ASSERT(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
    for (int k = 0; k < topk; ++k) {
      if (selected[k] < 0) continue;
      ids[k] = k;
      for (int p = 0; p < 3; ++p) {
        auto *dst = static_cast<uint8_t *>(reference.reference_weights[p]->data) +
            k * reference.reference_weights[p]->nb[2];
        const auto *src = static_cast<const uint8_t *>(source[p]->data) + actual[k] * source[p]->nb[2];
        const size_t row = ggml_row_size(source[p]->type, source[p]->ne[0]);
        GGML_ASSERT(cudaMemcpy2DAsync(dst, reference.reference_weights[p]->nb[1], src, source[p]->nb[1],
                        row, source[p]->ne[1], cudaMemcpyHostToDevice, stream) == cudaSuccess);
      }
    }
    GGML_ASSERT(cudaStreamSynchronize(stream) == cudaSuccess);
    cudaStreamDestroy(stream);
    ggml_backend_tensor_set(reference.reference_input, input, 0, ggml_nbytes(reference.reference_input));
    ggml_backend_tensor_set(reference.reference_ids, ids.data(), 0, ggml_nbytes(reference.reference_ids));
    ggml_backend_tensor_set(reference.reference_probs, probs, 0, ggml_nbytes(reference.reference_probs));
    GGML_ASSERT(ggml_backend_graph_compute(backend, reference.reference_graph) == GGML_STATUS_SUCCESS);
    std::vector<float> expected(ggml_nelements(reference.reference_output)), got(expected.size());
    ggml_backend_tensor_get(reference.reference_output, expected.data(), 0, ggml_nbytes(reference.reference_output));
    ggml_backend_tensor_get(verify_output, got.data(), 0, ggml_nbytes(verify_output));
    for (size_t i = 0; i < expected.size(); ++i) {
      GGML_ASSERT(std::isfinite(expected[i]) && std::isfinite(got[i]));
      verify_expected_max_abs = std::max(verify_expected_max_abs,
                                         std::abs(double(expected[i])));
      verify_actual_max_abs = std::max(verify_actual_max_abs,
                                       std::abs(double(got[i])));
      const double delta = std::abs(double(expected[i]) - got[i]);
      verify_max_delta = std::max(verify_max_delta, delta);
      GGML_ASSERT(delta <= 1e-5 * (1.0 + std::abs(expected[i])));
    }
    ++verified;
  }

  void token_complete(unsigned long long tag) {
    // The producer's pinned prediction mailbox is overwritten next replay.
    // Finish only its host reads here, after the model GPU synchronization;
    // DMA remains asynchronous and bank ownership still covers late copies.
    {
      std::unique_lock<std::mutex> lock(mutex);
      samples_finished.wait(lock, [this] {
        for (const auto &bank : banks)
          if (bank.busy.load(std::memory_order_acquire) &&
              !bank.prediction_sampled.load(std::memory_order_acquire)) return false;
        return true;
      });
    }
    completed_epoch = tag;
    for (auto &bank : banks) {
      if (bank.busy.load(std::memory_order_acquire) &&
          bank.dma_queued.load(std::memory_order_acquire) && bank.epoch <= tag &&
          cudaEventQuery(bank.copied) == cudaSuccess) {
        account_dma_completion(bank);
        bank.busy.store(false, std::memory_order_release);
      }
    }
  }

  // The publication bitmap is a device-side readiness check, but it does not
  // by itself create a dependency from this pool's copy stream to the graph
  // stream that consumes the staged weights.  Once the mapper has selected a
  // slot for this epoch, the worker has already queued the bank's completion
  // event after all three plane copies and publication stores.  Joining that
  // event here prevents the consumer GEMM from racing the H2D DMA.
  bool wait_for_epoch(unsigned long long tag, cudaStream_t stream) {
    if (!stream) return false;
    auto &bank = banks[tag & 1];
    if (!bank.busy.load(std::memory_order_acquire) || bank.epoch != tag)
      return false;
    return cudaStreamWaitEvent(stream, bank.copied, 0) == cudaSuccess;
  }

  void account_dma_completion(Bank &bank) {
    if (!bank.dma_queued.exchange(false, std::memory_order_acq_rel)) return;
    dma_ms_total.fetch_add(std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - bank.copy_enqueued_at).count(),
        std::memory_order_relaxed);
    dma_completed.fetch_add(1, std::memory_order_relaxed);
  }

  void request(unsigned long long tag, const int32_t *predictions,
               const unsigned long long *prediction_epoch, cudaEvent_t ready) {
    const int index = tag & 1;
    auto &bank = banks[index];
    if (bank.busy.load(std::memory_order_acquire) &&
        bank.dma_queued.load(std::memory_order_acquire) && bank.epoch <= completed_epoch &&
        cudaEventQuery(bank.copied) == cudaSuccess) {
      account_dma_completion(bank);
      bank.busy.store(false, std::memory_order_release);
    }
    bool idle = false;
    if (!bank.busy.compare_exchange_strong(idle, true, std::memory_order_acq_rel)) {
      ++dropped;
      return;
    }
    bank.dma_queued.store(false, std::memory_order_relaxed);
    bank.prediction_sampled.store(false, std::memory_order_relaxed);
    bank.epoch = tag;
    bank.predictions = predictions;
    bank.prediction_epoch = prediction_epoch;
    bank.prediction_ready = ready;
    {
      std::lock_guard<std::mutex> lock(mutex);
      jobs.push_back(index);
    }
    ++requests;
    if (!service || !service->submit([this] { work(); })) {
      ++dropped;
      std::lock_guard<std::mutex> lock(mutex);
      if (!jobs.empty()) jobs.pop_back();
      bank.busy.store(false, std::memory_order_release);
    }
  }

  void work() {
    cudaSetDevice(device);
    int index;
    {
      std::lock_guard<std::mutex> lock(mutex);
      if (jobs.empty()) return;
      index = jobs.front(); jobs.pop_front();
    }
      auto &bank = banks[index];
      GGML_ASSERT(cudaEventSynchronize(bank.prediction_ready) == cudaSuccess);
      if (*bank.prediction_epoch != bank.epoch) {
        ++late;
        finish_prediction_read(bank);
        bank.busy.store(false, std::memory_order_release);
        return;
      }
      std::vector<int> candidates;
      for (int k = 0; k < topk; ++k) {
        const int e = bank.predictions[k];
        if (e >= 0 && e < int(cold_mask.size()) && cold_mask[e] &&
            std::find(candidates.begin(), candidates.end(), e) == candidates.end())
          candidates.push_back(e);
      }
      if (stage_max && candidates.size() > stage_max) candidates.resize(stage_max);
      finish_prediction_read(bank);
      predicted += candidates.size();
      // Debug-only late-publication test without occupying GPU resources.
      // Prediction reads are finished before sleeping, so the mailbox lifetime
      // guard can make progress while DMA and bank ownership remain pending.
      if (worker_delay_us && !candidates.empty())
        std::this_thread::sleep_for(std::chrono::microseconds(worker_delay_us));
      if (delay_us && !candidates.empty())
        GGML_ASSERT(moestream_qwen_stage_delay(delay_us, copy_stream) == cudaSuccess);
      for (size_t slot = 0; slot < candidates.size(); ++slot) {
        const int e = candidates[slot];
        const int gpu_slot = index * topk + slot;
        for (int p = 0; p < 3; ++p) {
          auto *bounce = bank.bounce + plane_offsets[p] + slot * plane_bytes[p];
          const auto *src = static_cast<const uint8_t *>(source[p]->data) + e * source[p]->nb[2];
          const size_t row_bytes = ggml_row_size(source[p]->type, source[p]->ne[0]);
          for (int64_t row = 0; row < source[p]->ne[1]; ++row)
            std::memcpy(bounce + row * row_bytes, src + row * source[p]->nb[1], row_bytes);
          auto *dst = static_cast<uint8_t *>(weights[p]->data) + gpu_slot * weights[p]->nb[2];
          GGML_ASSERT(cudaMemcpyAsync(dst, bounce, plane_bytes[p], cudaMemcpyHostToDevice,
                                     copy_stream) == cudaSuccess);
        }
        GGML_ASSERT(moestream_qwen_stage_publish(slot_ids, slot_epochs, gpu_slot, e,
                                                 bank.epoch, copy_stream) == cudaSuccess);
        ++uploaded;
      }
      GGML_ASSERT(cudaEventRecord(bank.copied, copy_stream) == cudaSuccess);
      bank.copy_enqueued_at = std::chrono::steady_clock::now();
      bank.dma_queued.store(true, std::memory_order_release);
  }

  void finish_prediction_read(Bank &bank) {
    {
      std::lock_guard<std::mutex> lock(mutex);
      bank.prediction_sampled.store(true, std::memory_order_release);
    }
    samples_finished.notify_one();
  }
};
