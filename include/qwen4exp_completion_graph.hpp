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
    if (capacity_bytes == 0)
      return false;
    capacity_bytes_ = capacity_bytes;
    if (cudaStreamCreateWithFlags(&upload_stream_, cudaStreamNonBlocking) !=
            cudaSuccess ||
        cudaMallocHost(reinterpret_cast<void **>(&host_payload_),
                       capacity_bytes_) != cudaSuccess ||
        cudaMalloc(&device_payload_, capacity_bytes_) != cudaSuccess ||
        cudaEventCreateWithFlags(&ready_event_, cudaEventDisableTiming) !=
            cudaSuccess)
      return false;

    std::memset(host_payload_, 0, capacity_bytes_);
    if (cudaStreamBeginCapture(upload_stream_,
                               cudaStreamCaptureModeThreadLocal) !=
        cudaSuccess) {
      reset();
      return false;
    }
    const bool captured =
        cudaMemcpyAsync(device_payload_, host_payload_, capacity_bytes_,
                        cudaMemcpyHostToDevice, upload_stream_) == cudaSuccess &&
        cudaEventRecord(ready_event_, upload_stream_) == cudaSuccess;
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
        cudaGraphInstantiate(&graph_exec_, graph_, 0) != cudaSuccess) {
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
        !graph_exec_ || completion.state != StorageState::Ready ||
        completion.model_generation != expected_model_generation ||
        completion.request_generation != expected_request_generation ||
        !completion.payload || completion.payload->bytes.size() > capacity_bytes_)
      return false;
    std::memset(host_payload_, 0, capacity_bytes_);
    if (!completion.payload->bytes.empty())
      std::memcpy(host_payload_, completion.payload->bytes.data(),
                  completion.payload->bytes.size());
    payload_bytes_ = completion.payload->bytes.size();
    model_generation_ = completion.model_generation;
    request_generation_ = completion.request_generation;
    layer_ = completion.layer;
    state_ = cudaGraphLaunch(graph_exec_, upload_stream_) == cudaSuccess
                 ? CompletionGraphState::Pending
                 : CompletionGraphState::Error;
    return state_ == CompletionGraphState::Pending;
  }

  bool wait_on(cudaStream_t consumer_stream) const {
    return state_ == CompletionGraphState::Pending && consumer_stream &&
           cudaStreamWaitEvent(consumer_stream, ready_event_, 0) ==
               cudaSuccess;
  }

  bool synchronize() {
    if (state_ != CompletionGraphState::Pending)
      return false;
    const cudaError_t status = cudaEventSynchronize(ready_event_);
    if (status != cudaSuccess) {
      state_ = CompletionGraphState::Error;
      return false;
    }
    state_ = CompletionGraphState::Ready;
    return true;
  }

  CompletionGraphState state() const { return state_; }
  void *device_payload() const { return device_payload_; }
  size_t payload_bytes() const { return payload_bytes_; }
  size_t capacity_bytes() const { return capacity_bytes_; }
  uint64_t model_generation() const { return model_generation_; }
  uint64_t request_generation() const { return request_generation_; }
  int layer() const { return layer_; }
  cudaEvent_t ready_event() const { return ready_event_; }

  void reset() {
    if (graph_exec_) cudaGraphExecDestroy(graph_exec_);
    if (graph_) cudaGraphDestroy(graph_);
    if (ready_event_) cudaEventDestroy(ready_event_);
    if (device_payload_) cudaFree(device_payload_);
    if (host_payload_) cudaFreeHost(host_payload_);
    if (upload_stream_) cudaStreamDestroy(upload_stream_);
    graph_exec_ = nullptr;
    graph_ = nullptr;
    ready_event_ = nullptr;
    device_payload_ = nullptr;
    host_payload_ = nullptr;
    upload_stream_ = nullptr;
    capacity_bytes_ = 0;
    payload_bytes_ = 0;
    model_generation_ = 0;
    request_generation_ = 0;
    layer_ = -1;
    state_ = CompletionGraphState::Unprepared;
  }

 private:
  cudaStream_t upload_stream_ = nullptr;
  cudaEvent_t ready_event_ = nullptr;
  cudaGraph_t graph_ = nullptr;
  cudaGraphExec_t graph_exec_ = nullptr;
  uint8_t *host_payload_ = nullptr;
  void *device_payload_ = nullptr;
  size_t capacity_bytes_ = 0;
  size_t payload_bytes_ = 0;
  uint64_t model_generation_ = 0;
  uint64_t request_generation_ = 0;
  int layer_ = -1;
  CompletionGraphState state_ = CompletionGraphState::Unprepared;
};

}  // namespace moestream::qwen4exp
