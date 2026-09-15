#pragma once

#include <cstddef>
#include <utility>

#if defined(MOESTREAM_USE_GGML)
#include "ggml.h"
#include "ggml-backend.h"
#endif

#if defined(MOESTREAM_USE_GGML_CUDA)
#include <cuda_runtime.h>
#endif

namespace moe {

#if defined(MOESTREAM_USE_GGML)

class GgmlContextGuard {
 public:
  GgmlContextGuard() = default;
  explicit GgmlContextGuard(ggml_context *ctx) noexcept : ctx_(ctx) {}
  ~GgmlContextGuard() { reset(); }

  GgmlContextGuard(const GgmlContextGuard &) = delete;
  GgmlContextGuard &operator=(const GgmlContextGuard &) = delete;
  GgmlContextGuard(GgmlContextGuard &&other) noexcept
      : ctx_(std::exchange(other.ctx_, nullptr)) {}
  GgmlContextGuard &operator=(GgmlContextGuard &&other) noexcept {
    if (this != &other) {
      reset();
      ctx_ = std::exchange(other.ctx_, nullptr);
    }
    return *this;
  }

  ggml_context *get() const noexcept { return ctx_; }
  explicit operator bool() const noexcept { return ctx_ != nullptr; }
  ggml_context *release() noexcept { return std::exchange(ctx_, nullptr); }
  void reset(ggml_context *ctx = nullptr) noexcept {
    if (ctx_) ggml_free(ctx_);
    ctx_ = ctx;
  }

 private:
  ggml_context *ctx_ = nullptr;
};

class GgmlBackendBufferGuard {
 public:
  GgmlBackendBufferGuard() = default;
  explicit GgmlBackendBufferGuard(ggml_backend_buffer_t buffer) noexcept
      : buffer_(buffer) {}
  ~GgmlBackendBufferGuard() { reset(); }

  GgmlBackendBufferGuard(const GgmlBackendBufferGuard &) = delete;
  GgmlBackendBufferGuard &operator=(const GgmlBackendBufferGuard &) = delete;
  GgmlBackendBufferGuard(GgmlBackendBufferGuard &&other) noexcept
      : buffer_(std::exchange(other.buffer_, nullptr)) {}
  GgmlBackendBufferGuard &operator=(GgmlBackendBufferGuard &&other) noexcept {
    if (this != &other) {
      reset();
      buffer_ = std::exchange(other.buffer_, nullptr);
    }
    return *this;
  }

  ggml_backend_buffer_t get() const noexcept { return buffer_; }
  explicit operator bool() const noexcept { return buffer_ != nullptr; }
  ggml_backend_buffer_t release() noexcept {
    return std::exchange(buffer_, nullptr);
  }
  void reset(ggml_backend_buffer_t buffer = nullptr) noexcept {
    if (buffer_) ggml_backend_buffer_free(buffer_);
    buffer_ = buffer;
  }

 private:
  ggml_backend_buffer_t buffer_ = nullptr;
};

class GgmlBackendEventGuard {
 public:
  GgmlBackendEventGuard() = default;
  explicit GgmlBackendEventGuard(ggml_backend_event_t event) noexcept
      : event_(event) {}
  ~GgmlBackendEventGuard() { reset(); }

  GgmlBackendEventGuard(const GgmlBackendEventGuard &) = delete;
  GgmlBackendEventGuard &operator=(const GgmlBackendEventGuard &) = delete;
  GgmlBackendEventGuard(GgmlBackendEventGuard &&other) noexcept
      : event_(std::exchange(other.event_, nullptr)) {}
  GgmlBackendEventGuard &operator=(GgmlBackendEventGuard &&other) noexcept {
    if (this != &other) {
      reset();
      event_ = std::exchange(other.event_, nullptr);
    }
    return *this;
  }

  ggml_backend_event_t get() const noexcept { return event_; }
  explicit operator bool() const noexcept { return event_ != nullptr; }
  ggml_backend_event_t release() noexcept {
    return std::exchange(event_, nullptr);
  }
  void reset(ggml_backend_event_t event = nullptr) noexcept {
    if (event_) ggml_backend_event_free(event_);
    event_ = event;
  }

 private:
  ggml_backend_event_t event_ = nullptr;
};

#endif

#if defined(MOESTREAM_USE_GGML_CUDA)

class CudaEventGuard {
 public:
  CudaEventGuard() = default;
  explicit CudaEventGuard(cudaEvent_t event) noexcept : event_(event) {}
  ~CudaEventGuard() { reset(); }

  CudaEventGuard(const CudaEventGuard &) = delete;
  CudaEventGuard &operator=(const CudaEventGuard &) = delete;
  CudaEventGuard(CudaEventGuard &&other) noexcept
      : event_(std::exchange(other.event_, nullptr)) {}
  CudaEventGuard &operator=(CudaEventGuard &&other) noexcept {
    if (this != &other) {
      reset();
      event_ = std::exchange(other.event_, nullptr);
    }
    return *this;
  }

  cudaEvent_t get() const noexcept { return event_; }
  explicit operator bool() const noexcept { return event_ != nullptr; }
  cudaEvent_t release() noexcept { return std::exchange(event_, nullptr); }
  void reset(cudaEvent_t event = nullptr) noexcept {
    if (event_) cudaEventDestroy(event_);
    event_ = event;
  }

 private:
  cudaEvent_t event_ = nullptr;
};

class CudaStreamGuard {
 public:
  CudaStreamGuard() = default;
  explicit CudaStreamGuard(cudaStream_t stream) noexcept : stream_(stream) {}
  ~CudaStreamGuard() { reset(); }

  CudaStreamGuard(const CudaStreamGuard &) = delete;
  CudaStreamGuard &operator=(const CudaStreamGuard &) = delete;
  CudaStreamGuard(CudaStreamGuard &&other) noexcept
      : stream_(std::exchange(other.stream_, nullptr)) {}
  CudaStreamGuard &operator=(CudaStreamGuard &&other) noexcept {
    if (this != &other) {
      reset();
      stream_ = std::exchange(other.stream_, nullptr);
    }
    return *this;
  }

  cudaStream_t get() const noexcept { return stream_; }
  explicit operator bool() const noexcept { return stream_ != nullptr; }
  cudaStream_t release() noexcept { return std::exchange(stream_, nullptr); }
  void reset(cudaStream_t stream = nullptr) noexcept {
    if (stream_) cudaStreamDestroy(stream_);
    stream_ = stream;
  }

 private:
  cudaStream_t stream_ = nullptr;
};

class PinnedHostAllocation {
 public:
  PinnedHostAllocation() = default;
  explicit PinnedHostAllocation(void *data) noexcept : data_(data) {}
  ~PinnedHostAllocation() { reset(); }

  PinnedHostAllocation(const PinnedHostAllocation &) = delete;
  PinnedHostAllocation &operator=(const PinnedHostAllocation &) = delete;
  PinnedHostAllocation(PinnedHostAllocation &&other) noexcept
      : data_(std::exchange(other.data_, nullptr)) {}
  PinnedHostAllocation &operator=(PinnedHostAllocation &&other) noexcept {
    if (this != &other) {
      reset();
      data_ = std::exchange(other.data_, nullptr);
    }
    return *this;
  }

  void *get() const noexcept { return data_; }
  explicit operator bool() const noexcept { return data_ != nullptr; }
  void *release() noexcept { return std::exchange(data_, nullptr); }
  void reset(void *data = nullptr) noexcept {
    if (data_) cudaFreeHost(data_);
    data_ = data;
  }

 private:
  void *data_ = nullptr;
};

#endif

}  // namespace moe
