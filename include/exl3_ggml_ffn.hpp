#pragma once

#if defined(MOESTREAM_USE_GGML_CUDA)

#include "exl3_format.hpp"
#include "exl3_ggml_backend.hpp"
#include "exl3_gpu.hpp"

#include "ggml.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

struct Exl3GgmlRoutedPayloadLayout {
  size_t expert_stride = 0;
  size_t projection_offsets[3] = {};
  int expert_count = 0;
  int hidden = 0;
  int intermediate = 0;
  int bits[3] = {};
};

bool exl3_ggml_build_routed_payload(
    const std::vector<std::vector<uint8_t>> &matrices,
    std::vector<uint8_t> &payload, Exl3GgmlRoutedPayloadLayout &layout,
    std::string &error);

struct Exl3GgmlMatmulState {
  Exl3GgmlCustomNode custom;
  cudaStream_t stream = nullptr;
  float *partials = nullptr;
  Exl3PackedMatrixHeader header{};
  int rows = 0;

  ~Exl3GgmlMatmulState();
  bool initialize(const Exl3PackedMatrixHeader &matrix_header, int matrix_rows,
                  int device);
  static void compute(ggml_backend_t cuda_backend, ggml_tensor *node,
                      void *user_data);
};

ggml_tensor *exl3_ggml_map_matmul(ggml_context *ctx, ggml_tensor *input,
                                  ggml_tensor *opaque,
                                  Exl3GgmlMatmulState *state);

struct Exl3GgmlFfnState {
  Exl3GgmlFfnState() = default;
  Exl3GgmlCustomNode custom;
  cudaStream_t stream = nullptr;
  float *activation = nullptr;
  float *partials = nullptr;
  size_t offsets[3] = {};
  Exl3PackedMatrixHeader headers[3] = {};
  int rows = 0;

  ~Exl3GgmlFfnState();
  Exl3GgmlFfnState(const Exl3GgmlFfnState &) = delete;
  Exl3GgmlFfnState &operator=(const Exl3GgmlFfnState &) = delete;

  bool initialize(const Exl3PackedMatrixHeader (&matrix_headers)[3],
                  const size_t (&matrix_offsets)[3], int matrix_rows,
                  int device);

  static void compute(ggml_backend_t cuda_backend, ggml_tensor *node,
                      void *user_data);

 private:
  void enqueue(const ggml_tensor *input, const ggml_tensor *opaque,
               ggml_tensor *output);
};

ggml_tensor *exl3_ggml_map_fused_ffn(ggml_context *ctx, ggml_tensor *input,
                                     ggml_tensor *opaque,
                                     Exl3GgmlFfnState *state);

struct Exl3GgmlRoutedFfnState {
  Exl3GgmlCustomNode custom;
  cudaStream_t stream = nullptr;
  float *activation = nullptr;
  float *partials = nullptr;
  std::vector<size_t> offsets;
  std::vector<Exl3PackedMatrixHeader> headers;
  int rows = 0;
  size_t expert_stride = 0;
  size_t projection_offsets[3] = {};
  int expert_count = 0;
  int topk = 0;
  int hidden = 0;
  int intermediate = 0;
  int bits[3] = {};

  ~Exl3GgmlRoutedFfnState();
  Exl3GgmlRoutedFfnState() = default;
  Exl3GgmlRoutedFfnState(const Exl3GgmlRoutedFfnState &) = delete;
  Exl3GgmlRoutedFfnState &operator=(const Exl3GgmlRoutedFfnState &) = delete;

  bool initialize(const std::vector<Exl3PackedMatrixHeader> &matrix_headers,
                  const std::vector<size_t> &matrix_offsets, int matrix_rows,
                  int device);


  static void compute(ggml_backend_t cuda_backend, ggml_tensor *node,
                      void *user_data);

 private:
  void enqueue(const ggml_tensor *input, const ggml_tensor *opaque,
               const ggml_tensor *route_weights, ggml_tensor *output);
};

ggml_tensor *exl3_ggml_map_routed_ffn(ggml_context *ctx, ggml_tensor *input,
                                      ggml_tensor *opaque,
                                      ggml_tensor *route_weights,
                                      Exl3GgmlRoutedFfnState *state);

struct Exl3GgmlCachedRoutedFfnState {
  using ExternalExpertResolver = std::function<Exl3GpuExternalExpert(
      const std::string &, int, int, int,
      const std::function<std::vector<uint8_t>(const std::string &)> &)>;

  Exl3GgmlCustomNode custom;
  Exl3GpuExecutor *gpu_executor = nullptr;
  std::vector<std::string> expert_prefixes_by_id;
  std::function<std::vector<uint8_t>(const std::string &)> matrix_loader;
  ExternalExpertResolver external_expert_resolver;
  std::vector<Exl3GpuExternalExpert> external_experts;
  void *route_host_storage = nullptr;
  int32_t *route_ids_host = nullptr;
  float *route_weights_host = nullptr;
  bool routes_from_host = false;
  bool route_tensors_on_device = false;
  bool verify_cpu_reference = false;
  int model_layer = -1;
  size_t external_matrix_slot_capacity = 0;
  uint64_t route_readback_calls = 0;
  size_t route_readback_bytes = 0;
  double route_readback_ms = 0.0;
  double expert_resolution_ms = 0.0;
  double ffn_enqueue_ms = 0.0;
  double row_batch_wait_ms = 0.0;
  int rows = 0;
  int topk = 0;
  int hidden = 0;
  int intermediate = 0;

  ~Exl3GgmlCachedRoutedFfnState();

  bool initialize(
      Exl3GpuExecutor *executor,
      std::vector<std::string> matrix_expert_prefixes_by_id,
      int matrix_rows, int matrix_topk, int matrix_hidden,
      int matrix_intermediate);
  void set_matrix_loader(
      std::function<std::vector<uint8_t>(const std::string &)> loader);
  void set_external_expert_resolver(ExternalExpertResolver resolver);
  bool set_host_routes(const int32_t *route_ids, const float *route_weights);
  static void compute(ggml_backend_t cuda_backend, ggml_tensor *node,
                      void *user_data);
};

ggml_tensor *exl3_ggml_map_cached_routed_ffn(
    ggml_context *ctx, ggml_tensor *input, ggml_tensor *route_table,
    Exl3GgmlCachedRoutedFfnState *state);
ggml_tensor *exl3_ggml_map_cached_routed_ffn_host_routes(
    ggml_context *ctx, ggml_tensor *input, ggml_tensor *route_ids,
    ggml_tensor *route_weights, Exl3GgmlCachedRoutedFfnState *state);
ggml_tensor *exl3_ggml_map_cached_routed_ffn_device_routes(
    ggml_context *ctx, ggml_tensor *input, ggml_tensor *route_ids,
    ggml_tensor *route_weights, Exl3GgmlCachedRoutedFfnState *state);

struct Exl3GgmlCachedMatmulState {
  Exl3GgmlCustomNode custom;
  Exl3GpuExecutor *gpu_executor = nullptr;
  std::string key;
  int rows = 0;
  int input_features = 0;
  int output_features = 0;

  bool initialize(Exl3GpuExecutor *executor, int matrix_rows,
                  int matrix_input_features, int matrix_output_features);
  static void compute(ggml_backend_t cuda_backend, ggml_tensor *node,
                      void *user_data);
};

ggml_tensor *exl3_ggml_map_cached_matmul(
    ggml_context *ctx, ggml_tensor *input,
    Exl3GgmlCachedMatmulState *state);

#endif
