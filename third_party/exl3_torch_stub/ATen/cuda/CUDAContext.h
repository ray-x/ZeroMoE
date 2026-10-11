#pragma once
#include "../../exl3_stub.hpp"
namespace at { namespace cuda {
struct CUDAStreamView { cudaStream_t s; cudaStream_t stream() const { return s; } };
inline CUDAStreamView getCurrentCUDAStream() { return {moestream_exl3::current_stream()}; }
}}
