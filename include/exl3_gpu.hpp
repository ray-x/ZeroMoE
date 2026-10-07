#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <cuda_runtime_api.h>

struct Exl3GpuStats {
  uint64_t matmuls = 0;
  uint64_t cache_hits = 0;
  uint64_t uploads = 0;
  uint64_t evictions = 0;
  size_t resident_bytes = 0;
  double matmul_seconds = 0.0;
  uint64_t standalone_matmuls = 0;
  double standalone_matmul_seconds = 0.0;
  double routed_ffn_seconds = 0.0;
  double upload_seconds = 0.0;
  uint64_t routed_ffns = 0;
  uint64_t buffer_reuses = 0;
  uint64_t hc_mixes = 0;
};

struct Exl3GpuF32Weight {
  std::string key;
  const float *values = nullptr;
  int input_features = 0;
  int output_features = 0;
};

class Exl3GpuExecutor {
public:
  Exl3GpuExecutor(int device, size_t cache_budget);
  ~Exl3GpuExecutor();
  Exl3GpuExecutor(const Exl3GpuExecutor &) = delete;
  Exl3GpuExecutor &operator=(const Exl3GpuExecutor &) = delete;

  int device() const;
  bool matmul_cached(const std::string &key, const float *input, float *output,
                      int rows, int input_features, int output_features);
  void prepare_matrix_cache(const std::string &key, uint32_t type,
                            const uint8_t *weights, size_t bytes,
                            int input_features, int output_features,
                            bool persistent = true);
  bool matmul_device_cached(const std::string &key,
                            const float *device_input,
                            float *device_output, int rows,
                            int input_features, int output_features,
                            cudaStream_t stream);
  void matmul(const std::string &key, uint32_t type, const uint8_t *weights,
               size_t bytes, const float *input, float *output, int rows,
               int input_features, int output_features, bool persistent = true);
  Exl3GpuStats stats() const;
  void hc_mix(const Exl3GpuF32Weight &down, const Exl3GpuF32Weight &up,
               const Exl3GpuF32Weight *inject, const float *normed,
               float *hidden, float *injection, int rows, int width,
               int streams, bool divide_each_stream = true);
  void routed_ffn(const std::vector<std::string> &expert_prefixes,
                   const float *routing_weights, const float *input, float *output,
                   int hidden, int intermediate,
                   const std::function<std::vector<uint8_t>(const std::string &)> &load);
  void routed_ffn_device(
      const std::vector<std::string> &expert_prefixes_by_id,
      const int32_t *route_ids, const float *routing_weights,
      const float *device_input, float *device_output, int rows, int topk,
      int hidden, int intermediate, cudaStream_t stream,
      bool require_cache_hit,
      const std::function<std::vector<uint8_t>(const std::string &)> &load);
  void prepare_routed_cache(
      const std::vector<std::string> &expert_prefixes_by_id,
      const int32_t *route_ids, int count, int hidden, int intermediate,
      const std::function<std::vector<uint8_t>(const std::string &)> &load);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
