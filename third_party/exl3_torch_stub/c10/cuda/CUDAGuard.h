#pragma once
#include "../../exl3_stub.hpp"
namespace c10 { namespace cuda {
struct CUDAGuard {
  int prev = 0;
  explicit CUDAGuard(int device) { cudaGetDevice(&prev); cudaSetDevice(device); }
  explicit CUDAGuard(Device d) : CUDAGuard(d.index()) {}
  ~CUDAGuard() { cudaSetDevice(prev); }
  CUDAGuard(const CUDAGuard &) = delete;
};
}}
namespace at { namespace cuda {
struct OptionalCUDAGuard {
  int prev = 0;
  explicit OptionalCUDAGuard(c10::Device d) { cudaGetDevice(&prev); cudaSetDevice(d.index()); }
  ~OptionalCUDAGuard() { cudaSetDevice(prev); }
  OptionalCUDAGuard(const OptionalCUDAGuard &) = delete;
};
}}
