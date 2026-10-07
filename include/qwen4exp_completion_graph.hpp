#pragma once

// Completion-side CUDA graph for a prepared expert payload. StorageService
// owns the host I/O and generation checks; this object owns only the stable
// host/device payload addresses and the captured H2D copy. The consumer graph
// joins `ready_event()` on its own stream before reading device_payload().

#include "qwen4exp_storage_service.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <chrono>
#include <vector>

namespace moestream::qwen4exp {

enum class CompletionGraphState : uint8_t { Unprepared, Ready, Pending, Error };

class CompletionGraph {
 public:
  CompletionGraph() = default;
  CompletionGraph(const CompletionGraph &) = delete;
  CompletionGraph &operator=(const CompletionGraph &) = delete;

  ~CompletionGraph() { reset(); }

  bool prepare(size_t capacity_bytes) {
    reset();
    last_error_ = cudaSuccess;
    last_error_operation_ = nullptr;
    if (capacity_bytes == 0)
      return false;
    capacity_bytes_ = capacity_bytes;
    if (!check(cudaStreamCreateWithFlags(&upload_stream_, cudaStreamNonBlocking),
               "cudaStreamCreateWithFlags") ||
        !check(cudaMallocHost(reinterpret_cast<void **>(&host_payload_),
                              capacity_bytes_),
               "cudaMallocHost") ||
        !check(cudaMalloc(&device_payload_, capacity_bytes_), "cudaMalloc") ||
        !check(cudaEventCreateWithFlags(&ready_event_, cudaEventDisableTiming),
               "cudaEventCreateWithFlags(ready)") ||
        !check(cudaEventCreate(&upload_start_event_),
               "cudaEventCreate(upload start)") ||
        !check(cudaEventCreate(&upload_done_event_),
               "cudaEventCreate(upload done)"))
      return false;

    std::memset(host_payload_, 0, capacity_bytes_);
    if (!check(cudaStreamBeginCapture(upload_stream_,
                                      cudaStreamCaptureModeThreadLocal),
               "cudaStreamBeginCapture")) {
      reset();
      return false;
    }
    const bool captured =
        cudaMemcpyAsync(device_payload_, host_payload_, capacity_bytes_,
                        cudaMemcpyHostToDevice, upload_stream_) == cudaSuccess &&
        cudaEventRecordWithFlags(ready_event_, upload_stream_,
                                 cudaEventRecordExternal) == cudaSuccess;
    cudaError_t end_status = cudaSuccess;
    if (captured)
      end_status = cudaStreamEndCapture(upload_stream_, &graph_);
    else {
      cudaGraph_t discarded = nullptr;
      end_status = cudaStreamEndCapture(upload_stream_, &discarded);
      if (discarded)
        cudaGraphDestroy(discarded);
    }
    if (!captured || end_status != cudaSuccess || !graph_ ||
        !check(cudaGraphInstantiate(&graph_exec_, graph_, 0),
               "cudaGraphInstantiate")) {
      if (end_status != cudaSuccess)
        fail(end_status, "cudaStreamEndCapture");
      reset();
      return false;
    }
    state_ = CompletionGraphState::Ready;
    return true;
  }

  bool submit(const StorageCompletion &completion,
              uint64_t expected_model_generation,
              uint64_t expected_request_generation) {
    if (state_ == CompletionGraphState::Unprepared || !upload_stream_ ||
        state_ == CompletionGraphState::Pending || !graph_exec_ ||
        completion.state != StorageState::Ready ||
        completion.model_generation != expected_model_generation ||
        completion.request_generation != expected_request_generation ||
        !completion.payload || completion.payload->bytes.size() > capacity_bytes_)
      return false;
    const auto payload_prepare_start = std::chrono::steady_clock::now();
    std::memset(host_payload_, 0, capacity_bytes_);
    if (!completion.payload->bytes.empty())
      std::memcpy(host_payload_, completion.payload->bytes.data(),
                  completion.payload->bytes.size());
    payload_bytes_ = completion.payload->bytes.size();
    payload_offsets_ = completion.payload->offsets;
    model_generation_ = completion.model_generation;
    request_generation_ = completion.request_generation;
    layer_ = completion.layer;
    payload_prepare_ms_ = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() -
                              payload_prepare_start)
                              .count();
    submit_started_ = std::chrono::steady_clock::now();
    upload_wait_ms_ = 0.0;
    upload_transfer_ms_ = 0.0;
    if (!check(cudaEventRecord(upload_start_event_, upload_stream_),
               "cudaEventRecord(upload start)")) {
      state_ = CompletionGraphState::Error;
      return false;
    }
    const cudaError_t launch = cudaGraphLaunch(graph_exec_, upload_stream_);
    if (launch != cudaSuccess) {
      state_ = CompletionGraphState::Error;
      fail(launch, "cudaGraphLaunch");
      return false;
    }
    if (!check(cudaEventRecord(upload_done_event_, upload_stream_),
               "cudaEventRecord(upload done)")) {
      // The graph may already be in flight. Drain the dedicated stream on this
      // error path before allowing any caller to reuse the fixed payload
      // addresses.
      cudaStreamSynchronize(upload_stream_);
      state_ = CompletionGraphState::Error;
      return false;
    }
    state_ = CompletionGraphState::Pending;
    return true;
  }

  bool wait_on(cudaStream_t consumer_stream) {
    if (state_ != CompletionGraphState::Pending || !consumer_stream)
      return false;
    return check(cudaStreamWaitEvent(consumer_stream, ready_event_, 0),
                 "cudaStreamWaitEvent");
  }

  bool synchronize() {
    if (state_ != CompletionGraphState::Pending)
      return false;
    const cudaError_t status = cudaEventSynchronize(upload_done_event_);
    if (status != cudaSuccess) {
      fail(status, "cudaEventSynchronize");
      state_ = CompletionGraphState::Error;
      return false;
    }
    float transfer_ms = 0.0f;
    const cudaError_t timing = cudaEventElapsedTime(
        &transfer_ms, upload_start_event_, upload_done_event_);
    upload_transfer_ms_ =
        timing == cudaSuccess
            ? std::max(0.0, static_cast<double>(transfer_ms))
            : -1.0;
    upload_wait_ms_ = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - submit_started_)
                          .count();
    state_ = CompletionGraphState::Ready;
    return true;
  }

  CompletionGraphState state() const { return state_; }
  void *device_payload() const { return device_payload_; }
  size_t payload_bytes() const { return payload_bytes_; }
  // Wall time from submit() until the caller synchronizes. This can include
  // unrelated consumer-graph execution if synchronize() is intentionally
  // delayed, so adaptive decisions must use upload_transfer_ms() for H2D.
  double upload_wait_ms() const { return upload_wait_ms_; }
  double payload_prepare_ms() const { return payload_prepare_ms_; }
  // CUDA-event interval around the captured H2D copy. Unlike upload_wait_ms(),
  // this excludes host time spent waiting for the completion graph after the
  // consumer graph has already run.
  double upload_transfer_ms() const { return upload_transfer_ms_; }
  size_t capacity_bytes() const { return capacity_bytes_; }
  const std::vector<size_t> &payload_offsets() const { return payload_offsets_; }
  uint64_t model_generation() const { return model_generation_; }
  uint64_t request_generation() const { return request_generation_; }
  int layer() const { return layer_; }
  cudaEvent_t ready_event() const { return ready_event_; }
  cudaGraph_t captured_graph() const { return graph_; }
  cudaStream_t upload_stream() const { return upload_stream_; }
  cudaError_t last_error() const { return last_error_; }
  const char *last_error_operation() const { return last_error_operation_; }

  void reset() {
    if (graph_exec_) cudaGraphExecDestroy(graph_exec_);
    if (graph_) cudaGraphDestroy(graph_);
    if (upload_done_event_) cudaEventDestroy(upload_done_event_);
    if (upload_start_event_) cudaEventDestroy(upload_start_event_);
    if (ready_event_) cudaEventDestroy(ready_event_);
    if (device_payload_) cudaFree(device_payload_);
    if (host_payload_) cudaFreeHost(host_payload_);
    if (upload_stream_) cudaStreamDestroy(upload_stream_);
    graph_exec_ = nullptr;
    graph_ = nullptr;
    upload_done_event_ = nullptr;
    upload_start_event_ = nullptr;
    ready_event_ = nullptr;
    device_payload_ = nullptr;
    host_payload_ = nullptr;
    upload_stream_ = nullptr;
    capacity_bytes_ = 0;
    payload_bytes_ = 0;
    upload_wait_ms_ = 0.0;
    upload_transfer_ms_ = 0.0;
    payload_prepare_ms_ = 0.0;
    submit_started_ = {};
    payload_offsets_.clear();
    model_generation_ = 0;
    request_generation_ = 0;
    layer_ = -1;
    state_ = CompletionGraphState::Unprepared;
  }

 private:
  bool check(cudaError_t status, const char *operation) {
    if (status == cudaSuccess)
      return true;
    return fail(status, operation);
  }

  bool fail(cudaError_t status, const char *operation) {
    last_error_ = status;
    last_error_operation_ = operation;
    return false;
  }

  cudaStream_t upload_stream_ = nullptr;
  cudaEvent_t ready_event_ = nullptr;
  cudaGraph_t graph_ = nullptr;
  cudaGraphExec_t graph_exec_ = nullptr;
  cudaEvent_t upload_start_event_ = nullptr;
  cudaEvent_t upload_done_event_ = nullptr;
  uint8_t *host_payload_ = nullptr;
  void *device_payload_ = nullptr;
  size_t capacity_bytes_ = 0;
  size_t payload_bytes_ = 0;
  double upload_wait_ms_ = 0.0;
  double upload_transfer_ms_ = 0.0;
  double payload_prepare_ms_ = 0.0;
  std::chrono::steady_clock::time_point submit_started_{};
  std::vector<size_t> payload_offsets_;
  uint64_t model_generation_ = 0;
  uint64_t request_generation_ = 0;
  int layer_ = -1;
  CompletionGraphState state_ = CompletionGraphState::Unprepared;
  cudaError_t last_error_ = cudaSuccess;
  const char *last_error_operation_ = nullptr;
};

}  // namespace moestream::qwen4exp
