// Small EXL3 helper kernels used next to the matmul path: hyper-connection
// mixing, SwiGLU, weighted accumulation. The EXL3 matmuls themselves live in
// src/exl3_upstream_ops.cu (upstream ExLlamaV3 kernels).
#include "exl3_cuda.hpp"

#include <climits>
#include <cmath>

namespace {

constexpr int kKeyTilesPerSplit = 32;

__global__ void hc_silu(float *low, int dimensions, int streams) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= dimensions) return;
  const float value = low[index] / static_cast<float>(streams);
  low[index] = value / (1.0f + expf(-value));
}

__global__ void hc_mix(const float *normed, const float *gate, float *hidden,
                        int rows, int width, int streams,
                        bool divide_each_stream) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= rows * width) return;
  const int row = index / width;
  const int column = index % width;
  float sum = 0.0f;
  for (int stream_index = 0; stream_index < streams; ++stream_index) {
    const size_t offset =
        (static_cast<size_t>(row) * streams + stream_index) * width + column;
    const float gate_value = gate[offset];
    if (!isfinite(gate_value)) {
      hidden[index] = __int_as_float(0x7fffffff);
      return;
    }
    float value = __fmul_rn(normed[offset], 1.0f / (1.0f + expf(-gate_value)));
    if (divide_each_stream) value /= static_cast<float>(streams);
    sum = __fadd_rn(sum, value);
  }
  hidden[index] = divide_each_stream ? sum : sum / static_cast<float>(streams);
}

__global__ void swiglu(float *gate, const float *up, int dimensions) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < dimensions)
    gate[index] = gate[index] / (1.0f + expf(-gate[index])) * up[index];
}

__global__ void accumulate(float *output, const float *expert, float weight,
                            int dimensions) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < dimensions) output[index] += weight * expert[index];
}

__global__ void scale_device_weight(float *output, const float *weight,
                                     int dimensions) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < dimensions)
    output[index] *= weight[0];
}

} // namespace

int exl3_cuda_key_splits(int input_features) {
  return (input_features / 16 + kKeyTilesPerSplit - 1) / kKeyTilesPerSplit;
}

cudaError_t launch_exl3_hc_silu(float *low, int dimensions, int streams,
                                cudaStream_t stream) {
  if (!low || dimensions <= 0 || streams <= 0) return cudaErrorInvalidValue;
  hc_silu<<<(static_cast<size_t>(dimensions) + 255) / 256, 256, 0, stream>>>(low, dimensions, streams);
  return cudaGetLastError();
}

cudaError_t launch_exl3_hc_mix(const float *normed, const float *gate,
                               float *hidden, int rows, int width, int streams,
                               bool divide_each_stream, cudaStream_t stream) {
  if (!normed || !gate || !hidden || rows <= 0 || width <= 0 || streams <= 0)
    return cudaErrorInvalidValue;
  const size_t dimensions = static_cast<size_t>(rows) * width;
  if (dimensions > INT_MAX) return cudaErrorInvalidValue;
  hc_mix<<<(dimensions + 255) / 256, 256, 0, stream>>>(
      normed, gate, hidden, rows, width, streams, divide_each_stream);
  return cudaGetLastError();
}

cudaError_t launch_exl3_swiglu(float *gate, const float *up, int dimensions,
                               cudaStream_t stream) {
  if (!gate || !up || dimensions <= 0) return cudaErrorInvalidValue;
  swiglu<<<(dimensions + 255) / 256, 256, 0, stream>>>(gate, up, dimensions);
  return cudaGetLastError();
}

cudaError_t launch_exl3_accumulate(float *output, const float *expert,
                                   float weight, int dimensions,
                                   cudaStream_t stream) {
  if (!output || !expert || dimensions <= 0) return cudaErrorInvalidValue;
  accumulate<<<(dimensions + 255) / 256, 256, 0, stream>>>(output, expert, weight, dimensions);
  return cudaGetLastError();
}

cudaError_t launch_exl3_scale_device_weight(float *output, const float *weight,
                                            int dimensions,
                                            cudaStream_t stream) {
  if (!output || !weight || dimensions <= 0)
    return cudaErrorInvalidValue;
  scale_device_weight<<<(dimensions + 255) / 256, 256, 0, stream>>>(
      output, weight, dimensions);
  return cudaGetLastError();
}

#ifndef MOESTREAM_EXL3_UPSTREAM
// Built without MOESTREAM_EXLLAMAV3_SRC: the EXL3 matmuls are unavailable.
cudaError_t launch_exl3_mul1_matmul(const Exl3CudaMatrixView &, const float *,
                                    float *, int, float *, cudaStream_t) {
  return cudaErrorNotSupported;
}
cudaError_t launch_exl3_mul1_matmul_accumulate(const Exl3CudaMatrixView &,
                                               const float *, float *, float,
                                               int, float *, cudaStream_t) {
  return cudaErrorNotSupported;
}
cudaError_t launch_exl3_mul1_matmul_accumulate_device_weight(
    const Exl3CudaMatrixView &, const float *, float *, const float *, int,
    float *, cudaStream_t) {
  return cudaErrorNotSupported;
}
cudaError_t launch_exl3_mul1_gate_up(const Exl3CudaMatrixView &,
                                     const Exl3CudaMatrixView &, const float *,
                                     float *, int, float *, float *,
                                     cudaStream_t) {
  return cudaErrorNotSupported;
}
#endif
