#pragma once

#include <cuda_runtime.h>
#include <cstdint>

struct Exl3CudaMatrixView {
  const uint16_t *trellis = nullptr;
  const uint16_t *suh = nullptr;
  const uint16_t *svh = nullptr;
  const uint16_t *bias = nullptr;
  int input_features = 0;
  int output_features = 0;
  int bits = 0;
};

int exl3_cuda_key_splits(int input_features);

cudaError_t launch_exl3_hc_silu(float *low, int dimensions, int streams,
                                cudaStream_t stream = nullptr);
cudaError_t launch_exl3_hc_mix(const float *normed, const float *gate,
                               float *hidden, int rows, int width, int streams,
                               bool divide_each_stream,
                               cudaStream_t stream = nullptr);

cudaError_t launch_exl3_swiglu(float *gate, const float *up, int dimensions,
                               cudaStream_t stream = nullptr);
cudaError_t launch_exl3_accumulate(float *output, const float *expert,
                                   float weight, int dimensions,
                                   cudaStream_t stream = nullptr);
cudaError_t launch_exl3_scale_device_weight(float *output,
                                            const float *weight,
                                            int dimensions,
                                            cudaStream_t stream = nullptr);

cudaError_t launch_exl3_mul1_matmul(const Exl3CudaMatrixView &matrix,
                                   const float *input, float *output, int rows,
                                   float *partials,
                                   cudaStream_t stream = nullptr);

cudaError_t launch_exl3_mul1_matmul_accumulate(
    const Exl3CudaMatrixView &matrix, const float *input, float *accumulator,
    float weight, int rows, float *partials, cudaStream_t stream = nullptr);

cudaError_t launch_exl3_mul1_matmul_accumulate_device_weight(
    const Exl3CudaMatrixView &matrix, const float *input, float *accumulator,
    const float *weight, int rows, float *partials,
    cudaStream_t stream = nullptr);

cudaError_t launch_exl3_mul1_gate_up(
    const Exl3CudaMatrixView &gate, const Exl3CudaMatrixView &up,
    const float *input, float *activation, int rows, float *gate_partials,
    float *up_partials, cudaStream_t stream = nullptr);

cudaError_t launch_exl3_routed_ffn(
    const uint8_t *payload, size_t expert_stride, size_t gate_offset,
    size_t up_offset, size_t down_offset, const int32_t *route_ids,
    const float *route_weights, int topk, int expert_count, const float *input,
    float *output, int rows, int hidden, int intermediate, float *activation,
    float *partials, int gate_bits, int up_bits, int down_bits,
    cudaStream_t stream = nullptr);

cudaError_t launch_exl3_pack_route_table(uint8_t *table,
                                         const int32_t *route_ids,
                                         const float *route_weights, int topk,
                                         int rows,
                                         const int32_t *expert_remap = nullptr,
                                         cudaStream_t stream = nullptr);

cudaError_t launch_exl3_mul1_reconstruct(const Exl3CudaMatrixView &matrix,
                                        float *weight,
                                        cudaStream_t stream = nullptr);

bool exl3_cuda_matmul_packed(float *output, const float *input,
                            const void *packed, int rows, int input_features,
                            int output_features);

// Upstream ExLlamaV3 backed implementations (src/exl3_upstream_ops.cu, built
// when MOESTREAM_EXLLAMAV3_SRC is set). Return cudaErrorNotSupported when the
// matrix shape is outside upstream's contract so callers fall back.
#ifdef MOESTREAM_EXL3_UPSTREAM
__attribute__((weak)) bool exl3_upstream_active();
__attribute__((weak)) cudaError_t exl3_upstream_matmul(const Exl3CudaMatrixView &matrix,
                                 const float *input, float *output,
                                 float *accumulator, float weight,
                                 const float *device_weight, int rows,
                                 cudaStream_t stream);
__attribute__((weak)) cudaError_t exl3_upstream_gate_up(const Exl3CudaMatrixView &gate,
                                  const Exl3CudaMatrixView &up,
                                  const float *input, float *activation,
                                  int rows, float *up_scratch,
                                  cudaStream_t stream);
// Whole routed FFN (rows <= 8, rows * topk <= 256) in two upstream launches.
// gate/up/down are arrays of rows * topk matrix-view pointers. Output is
// overwritten. Returns cudaErrorNotSupported if shapes/bit widths do not fit.
__attribute__((weak)) cudaError_t exl3_upstream_routed_ffn(
    const Exl3CudaMatrixView *const *gate, const Exl3CudaMatrixView *const *up,
    const Exl3CudaMatrixView *const *down, const float *host_weights,
    const float *device_weights, const float *input, float *output, int rows,
    int topk, int hidden, int intermediate, cudaStream_t stream);
#endif
