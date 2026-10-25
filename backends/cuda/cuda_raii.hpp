#pragma once
// RAII owners for the CUDA resources that were released by hand in destructors:
// device memory, pinned host memory, events and streams.
//
// Rules that make these safe to adopt in existing code:
//  * Move-only. A moved-from object is empty and harmless to destroy.
//  * Every object remembers the device it was created on and makes that device
//    current only while it releases (then restores the caller's device), which is
//    what the hand-written destructors did with cudaSetDevice(device).
//  * Release never throws, never asserts and ignores the CUDA status. Static or
//    late destructors can run after the CUDA runtime has been torn down, where
//    cudaFree/cudaEventDestroy report an error; a destructor must not abort then.
//  * Allocation failures are returned as cudaError_t and leave the object EMPTY
//    (never half-updated), so a failed grow cannot be mistaken for success.
//  * Sizes are element counts; a count whose byte size overflows is rejected.
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>

namespace zeromoe::cuda {

// Makes `device` current for the lifetime of the scope; a no-op if it already is
// or if device < 0. Restores the previous device on exit.
class DeviceScope {
 public:
  explicit DeviceScope(int device) noexcept {
    if (device < 0 || cudaGetDevice(&prev_) != cudaSuccess)
      return;
    if (prev_ != device && cudaSetDevice(device) == cudaSuccess)
      active_ = true;
  }
  ~DeviceScope() {
    if (active_)
      cudaSetDevice(prev_);
  }
  DeviceScope(const DeviceScope &) = delete;
  DeviceScope &operator=(const DeviceScope &) = delete;

 private:
  int prev_ = -1;
  bool active_ = false;
};

namespace detail {
inline int current_device() noexcept {
  int d = -1;
  return cudaGetDevice(&d) == cudaSuccess ? d : -1;
}
template <typename T>
inline bool bytes_for(size_t count, size_t &bytes) noexcept {
  if (count > std::numeric_limits<size_t>::max() / sizeof(T))
    return false;
  bytes = count * sizeof(T);
  return true;
}
}  // namespace detail

// Device memory. Contents are never preserved across allocate()/reserve().
template <typename T>
class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  ~DeviceBuffer() { release(); }
  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;
  DeviceBuffer(DeviceBuffer &&o) noexcept { swap(o); }
  DeviceBuffer &operator=(DeviceBuffer &&o) noexcept {
    if (this != &o) {
      release();
      swap(o);
    }
    return *this;
  }

  // Exactly `count` elements on `device` (< 0: the current device). The old
  // allocation is released first, so peak usage never holds both. On failure the
  // buffer is empty and the CUDA error is returned.
  cudaError_t allocate(size_t count, int device = -1) noexcept {
    release();
    if (count == 0)
      return cudaSuccess;
    size_t bytes = 0;
    if (!detail::bytes_for<T>(count, bytes))
      return cudaErrorInvalidValue;
    if (device < 0)
      device = detail::current_device();
    DeviceScope scope(device);
    void *p = nullptr;
    const cudaError_t err = cudaMalloc(&p, bytes);
    if (err != cudaSuccess) {
      cudaGetLastError();  // clear the sticky error we just caused
      return err;
    }
    ptr_ = static_cast<T *>(p);
    capacity_ = count;
    device_ = device;
    return cudaSuccess;
  }

  // Grow-only: keeps the current allocation when it already holds `count`.
  cudaError_t reserve(size_t count, int device = -1) noexcept {
    if (count <= capacity_ && (device < 0 || device == device_))
      return cudaSuccess;
    return allocate(count, device);
  }

  void release() noexcept {
    if (ptr_) {
      DeviceScope scope(device_);
      cudaFree(ptr_);  // status ignored on purpose, see the header comment
    }
    ptr_ = nullptr;
    capacity_ = 0;
    device_ = -1;
  }

  void swap(DeviceBuffer &o) noexcept {
    std::swap(ptr_, o.ptr_);
    std::swap(capacity_, o.capacity_);
    std::swap(device_, o.device_);
  }

  T *get() const noexcept { return ptr_; }
  T *data() const noexcept { return ptr_; }
  size_t capacity() const noexcept { return capacity_; }
  size_t bytes() const noexcept { return capacity_ * sizeof(T); }
  int device() const noexcept { return device_; }
  bool empty() const noexcept { return ptr_ == nullptr; }
  explicit operator bool() const noexcept { return ptr_ != nullptr; }

 private:
  T *ptr_ = nullptr;
  size_t capacity_ = 0;
  int device_ = -1;
};

// Page-locked host memory (cudaMallocHost). Same contract as DeviceBuffer.
template <typename T>
class PinnedBuffer {
 public:
  PinnedBuffer() = default;
  ~PinnedBuffer() { release(); }
  PinnedBuffer(const PinnedBuffer &) = delete;
  PinnedBuffer &operator=(const PinnedBuffer &) = delete;
  PinnedBuffer(PinnedBuffer &&o) noexcept { swap(o); }
  PinnedBuffer &operator=(PinnedBuffer &&o) noexcept {
    if (this != &o) {
      release();
      swap(o);
    }
    return *this;
  }

  cudaError_t allocate(size_t count, int device = -1) noexcept {
    release();
    if (count == 0)
      return cudaSuccess;
    size_t bytes = 0;
    if (!detail::bytes_for<T>(count, bytes))
      return cudaErrorInvalidValue;
    if (device < 0)
      device = detail::current_device();
    DeviceScope scope(device);
    void *p = nullptr;
    const cudaError_t err = cudaMallocHost(&p, bytes);
    if (err != cudaSuccess) {
      cudaGetLastError();
      return err;
    }
    ptr_ = static_cast<T *>(p);
    capacity_ = count;
    device_ = device;
    return cudaSuccess;
  }

  cudaError_t reserve(size_t count, int device = -1) noexcept {
    if (count <= capacity_ && (device < 0 || device == device_))
      return cudaSuccess;
    return allocate(count, device);
  }

  void release() noexcept {
    if (ptr_) {
      DeviceScope scope(device_);
      cudaFreeHost(ptr_);
    }
    ptr_ = nullptr;
    capacity_ = 0;
    device_ = -1;
  }

  void swap(PinnedBuffer &o) noexcept {
    std::swap(ptr_, o.ptr_);
    std::swap(capacity_, o.capacity_);
    std::swap(device_, o.device_);
  }

  T *get() const noexcept { return ptr_; }
  T *data() const noexcept { return ptr_; }
  T &operator[](size_t i) const noexcept { return ptr_[i]; }
  size_t capacity() const noexcept { return capacity_; }
  size_t bytes() const noexcept { return capacity_ * sizeof(T); }
  int device() const noexcept { return device_; }
  bool empty() const noexcept { return ptr_ == nullptr; }
  explicit operator bool() const noexcept { return ptr_ != nullptr; }

 private:
  T *ptr_ = nullptr;
  size_t capacity_ = 0;
  int device_ = -1;
};

// Implicitly converts to the raw handle so existing cudaEventRecord(ev, s)
// call sites compile unchanged; creation goes through create().
class CudaEvent {
 public:
  CudaEvent() = default;
  ~CudaEvent() { release(); }
  CudaEvent(const CudaEvent &) = delete;
  CudaEvent &operator=(const CudaEvent &) = delete;
  CudaEvent(CudaEvent &&o) noexcept { swap(o); }
  CudaEvent &operator=(CudaEvent &&o) noexcept {
    if (this != &o) {
      release();
      swap(o);
    }
    return *this;
  }

  cudaError_t create(unsigned flags = cudaEventDisableTiming,
                     int device = -1) noexcept {
    release();
    if (device < 0)
      device = detail::current_device();
    DeviceScope scope(device);
    cudaEvent_t e = nullptr;
    const cudaError_t err = cudaEventCreateWithFlags(&e, flags);
    if (err != cudaSuccess) {
      cudaGetLastError();
      return err;
    }
    ev_ = e;
    device_ = device;
    return cudaSuccess;
  }

  void release() noexcept {
    if (ev_) {
      DeviceScope scope(device_);
      cudaEventDestroy(ev_);
    }
    ev_ = nullptr;
    device_ = -1;
  }

  void swap(CudaEvent &o) noexcept {
    std::swap(ev_, o.ev_);
    std::swap(device_, o.device_);
  }

  cudaEvent_t get() const noexcept { return ev_; }
  operator cudaEvent_t() const noexcept { return ev_; }
  int device() const noexcept { return device_; }

 private:
  cudaEvent_t ev_ = nullptr;
  int device_ = -1;
};

class CudaStream {
 public:
  CudaStream() = default;
  ~CudaStream() { release(); }
  CudaStream(const CudaStream &) = delete;
  CudaStream &operator=(const CudaStream &) = delete;
  CudaStream(CudaStream &&o) noexcept { swap(o); }
  CudaStream &operator=(CudaStream &&o) noexcept {
    if (this != &o) {
      release();
      swap(o);
    }
    return *this;
  }

  cudaError_t create(unsigned flags = cudaStreamNonBlocking,
                     int device = -1) noexcept {
    release();
    if (device < 0)
      device = detail::current_device();
    DeviceScope scope(device);
    cudaStream_t s = nullptr;
    const cudaError_t err = cudaStreamCreateWithFlags(&s, flags);
    if (err != cudaSuccess) {
      cudaGetLastError();
      return err;
    }
    st_ = s;
    device_ = device;
    return cudaSuccess;
  }

  void release() noexcept {
    if (st_) {
      DeviceScope scope(device_);
      cudaStreamDestroy(st_);
    }
    st_ = nullptr;
    device_ = -1;
  }

  void swap(CudaStream &o) noexcept {
    std::swap(st_, o.st_);
    std::swap(device_, o.device_);
  }

  cudaStream_t get() const noexcept { return st_; }
  operator cudaStream_t() const noexcept { return st_; }
  int device() const noexcept { return device_; }

 private:
  cudaStream_t st_ = nullptr;
  int device_ = -1;
};

}  // namespace zeromoe::cuda
