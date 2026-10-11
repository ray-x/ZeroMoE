// Minimal, non-owning stand-in for the slice of ATen/c10 that ExLlamaV3's
// exllamav3_ext EXL3 launchers use (exl3_gemm, exl3_mgemm, exl3_moe_coop and
// their autotune/device-context helpers). It lets those upstream sources
// compile UNMODIFIED without libtorch. Tensors are views over caller-owned
// device pointers; nothing here allocates device memory except at::empty.
#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace c10 {

template <class T> using optional = std::optional<T>;

struct Error : std::runtime_error {
  using std::runtime_error::runtime_error;
};

struct Device {
  int idx = 0;
  int index() const { return idx; }
  bool operator==(const Device &o) const { return idx == o.idx; }
};

} // namespace c10

namespace at {

enum ScalarType { kByte, kInt, kShort, kLong, kHalf, kFloat, kBFloat16 };
enum DeviceType { kCPU, kCUDA };

inline size_t scalar_size(ScalarType t) {
  switch (t) {
  case kByte: return 1;
  case kShort: case kHalf: case kBFloat16: return 2;
  case kInt: case kFloat: return 4;
  case kLong: return 8;
  }
  return 1;
}

struct Owned {
  void *ptr = nullptr;
  ~Owned() { if (ptr) cudaFree(ptr); }
};

class Tensor {
public:
  Tensor() = default;
  // Wrap an existing device pointer. Strides are contiguous row-major unless
  // given explicitly.
  Tensor(void *ptr, std::vector<int64_t> sizes, ScalarType type, int device,
         std::vector<int64_t> strides = {})
      : ptr_(ptr), sizes_(std::move(sizes)), strides_(std::move(strides)),
        type_(type), device_(device) {
    if (strides_.empty()) {
      strides_.assign(sizes_.size(), 1);
      for (int i = static_cast<int>(sizes_.size()) - 2; i >= 0; --i)
        strides_[i] = strides_[i + 1] * sizes_[i + 1];
    }
  }

  void *data_ptr() const { return ptr_; }
  template <class T> T *data_ptr() const { return static_cast<T *>(ptr_); }
  int64_t dim() const { return static_cast<int64_t>(sizes_.size()); }
  int64_t size(int64_t i) const { return sizes_[wrap(i)]; }
  int64_t stride(int64_t i) const { return strides_[wrap(i)]; }
  const std::vector<int64_t> &sizes() const { return sizes_; }
  int64_t numel() const {
    int64_t n = 1;
    for (int64_t s : sizes_) n *= s;
    return n;
  }
  ScalarType dtype() const { return type_; }
  ScalarType scalar_type() const { return type_; }
  c10::Device device() const { return c10::Device{device_}; }
  bool is_contiguous() const {
    int64_t expect = 1;
    for (int i = static_cast<int>(sizes_.size()) - 1; i >= 0; --i) {
      if (sizes_[i] != 1 && strides_[i] != expect) return false;
      expect *= sizes_[i];
    }
    return true;
  }
  bool defined() const { return ptr_ != nullptr; }
  void adopt(std::shared_ptr<Owned> o) { owner_ = std::move(o); }

private:
  size_t wrap(int64_t i) const {
    if (i < 0) i += static_cast<int64_t>(sizes_.size());
    if (i < 0 || i >= static_cast<int64_t>(sizes_.size()))
      throw c10::Error("exl3 stub: tensor dimension out of range");
    return static_cast<size_t>(i);
  }
  void *ptr_ = nullptr;
  std::vector<int64_t> sizes_, strides_;
  ScalarType type_ = kByte;
  int device_ = 0;
  std::shared_ptr<Owned> owner_;
};

struct TensorOptions {
  ScalarType type = kByte;
  int dev = 0;
  TensorOptions dtype(ScalarType t) const { auto o = *this; o.type = t; return o; }
  TensorOptions device(DeviceType, int d) const { auto o = *this; o.dev = d; return o; }
};

inline Tensor empty(std::vector<int64_t> sizes, const TensorOptions &o) {
  int64_t n = 1;
  for (int64_t d : sizes) n *= d;
  auto owner = std::make_shared<Owned>();
  int prev = 0;
  cudaGetDevice(&prev);
  cudaSetDevice(o.dev);
  const cudaError_t e = cudaMalloc(&owner->ptr, static_cast<size_t>(n) * scalar_size(o.type));
  cudaSetDevice(prev);
  if (e != cudaSuccess) throw c10::Error("exl3 stub: at::empty cudaMalloc failed");
  Tensor t(owner->ptr, std::move(sizes), o.type, o.dev);
  t.adopt(std::move(owner));
  return t;
}

} // namespace at

namespace moestream_exl3 {
template <class... Args> std::string cat(const Args &...args) {
  std::ostringstream os;
  (os << ... << args);
  return os.str();
}
} // namespace moestream_exl3

#define TORCH_CHECK(cond, ...)                                                \
  if (!(cond)) {                                                              \
    throw c10::Error("exl3: " + moestream_exl3::cat(__VA_ARGS__));            \
  }
#define TORCH_WARN(...)                                                       \
  std::fprintf(stderr, "[exl3 warn] %s\n",                                    \
               moestream_exl3::cat(__VA_ARGS__).c_str())

// Per-thread stream the upstream launchers enqueue on. MoEStream's custom node
// sets this to the ggml backend's CUDA stream before each call.
namespace moestream_exl3 {
cudaStream_t current_stream();
void set_current_stream(cudaStream_t stream);
} // namespace moestream_exl3
