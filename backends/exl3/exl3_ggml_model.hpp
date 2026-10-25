#pragma once

#if defined(ZEROMOE_USE_GGML_CUDA)

#include "exl3_ggml_ffn.hpp"
#include "exl3_model_loader.hpp"

#include "ggml.h"
#include "ggml-backend.h"

#include <memory>
#include <functional>
#include <string>
#include <vector>

class Exl3GpuExecutor;
class GpuTopology;
struct StreamingEngine;
struct CPUEngine;

struct Exl3GgmlBridgeLayer {
  std::unique_ptr<Exl3GgmlCachedRoutedFfnState> ffn_state;
  Exl3GpuExecutor *gpu_executor = nullptr;
  int layer = -1;
  int device = -1;
  int rows = 0;
  int experts = 0;
  int topk = 0;

  Exl3GgmlBridgeLayer() = default;
  ~Exl3GgmlBridgeLayer();
  Exl3GgmlBridgeLayer(const Exl3GgmlBridgeLayer &) = delete;
  Exl3GgmlBridgeLayer &operator=(const Exl3GgmlBridgeLayer &) = delete;

  bool initialize(const std::string &model_path, ggml_context *ctx,
                  ggml_backend_t cuda_backend, Exl3GpuExecutor *executor,
                  int model_layer, int n_experts, int model_topk,
                  int model_rows, int cuda_device,
                  bool load_cache_misses_from_safetensors = false,
                  std::shared_ptr<Exl3CpuMatrixCache> cpu_matrix_cache = {},
                  GpuTopology *gpu_topology = nullptr,
                  StreamingEngine *streaming_engine = nullptr);
  bool set_layer(int model_layer);

  ggml_tensor *build(ggml_context *ctx, ggml_tensor *input,
                     ggml_tensor *route_ids, ggml_tensor *route_weights,
                     ggml_backend_sched_t scheduler,
                     ggml_backend_t cuda_backend);
};

class Exl3GgmlRoutedFfnGraph {
public:
  Exl3GgmlRoutedFfnGraph() = default;
  ~Exl3GgmlRoutedFfnGraph();
  Exl3GgmlRoutedFfnGraph(const Exl3GgmlRoutedFfnGraph &) = delete;
  Exl3GgmlRoutedFfnGraph &operator=(const Exl3GgmlRoutedFfnGraph &) = delete;

  bool initialize(const std::string &model_path, Exl3GpuExecutor *executor,
                  int n_experts, int topk, int rows, int hidden,
                  int intermediate, int device);
  bool run(
      int layer, const int32_t *route_ids, const float *route_weights,
      const float *input, float *output,
      const std::function<std::vector<uint8_t>(const std::string &)> &load);

private:
  void reset();

  ggml_backend_t cuda_raw = nullptr;
  ggml_backend_t cuda_adapter = nullptr;
  ggml_backend_t cpu_backend = nullptr;
  ggml_backend_sched_t scheduler = nullptr;
  ggml_context *context = nullptr;
  ggml_cgraph *graph = nullptr;
  ggml_tensor *input = nullptr;
  ggml_tensor *route_ids = nullptr;
  ggml_tensor *route_weights = nullptr;
  ggml_tensor *output = nullptr;
  std::unique_ptr<Exl3GgmlBridgeLayer> bridge;
  int experts = 0;
  int topk = 0;
  int rows = 0;
  int hidden = 0;
  int intermediate = 0;
};

class Exl3GgmlDenseMatmulGraph {
public:
  Exl3GgmlDenseMatmulGraph() = default;
  ~Exl3GgmlDenseMatmulGraph();
  Exl3GgmlDenseMatmulGraph(const Exl3GgmlDenseMatmulGraph &) = delete;
  Exl3GgmlDenseMatmulGraph &operator=(const Exl3GgmlDenseMatmulGraph &) = delete;

  bool initialize(Exl3GpuExecutor *executor, int rows, int input_features,
                  int output_features);
  bool run(const std::string &key, uint32_t type, const uint8_t *weights,
           size_t weight_bytes, const float *input, float *output);

private:
  void reset();

  ggml_backend_t cuda_raw = nullptr;
  ggml_backend_t cuda_adapter = nullptr;
  ggml_backend_t cpu_backend = nullptr;
  ggml_backend_sched_t scheduler = nullptr;
  ggml_context *context = nullptr;
  ggml_cgraph *graph = nullptr;
  ggml_tensor *input = nullptr;
  ggml_tensor *output = nullptr;
  Exl3GgmlCachedMatmulState state;
  Exl3GpuExecutor *executor = nullptr;
  int rows = 0;
  int input_features = 0;
  int output_features = 0;
};

class Exl3GgmlCpuRoutedFfnGraph {
public:
  using RowDispatch = bool (*)(void *context, int layer, const int *experts,
                               const float *weights, int count,
                               const float *input, float *output);

  Exl3GgmlCpuRoutedFfnGraph() = default;
  ~Exl3GgmlCpuRoutedFfnGraph();
  Exl3GgmlCpuRoutedFfnGraph(const Exl3GgmlCpuRoutedFfnGraph &) = delete;
  Exl3GgmlCpuRoutedFfnGraph &operator=(
      const Exl3GgmlCpuRoutedFfnGraph &) = delete;

  bool initialize(RowDispatch dispatch, void *dispatch_context,
                  int layer_count, int rows, int topk, int hidden);
  bool run(int layer, const int32_t *route_ids,
           const float *route_weights, const float *input, float *output);

private:
  static void compute(ggml_tensor *node, const ggml_tensor *input,
                      const ggml_tensor *route_ids,
                      const ggml_tensor *route_weights, int ith, int nth,
                      void *user_data);
  void reset();

  RowDispatch dispatch = nullptr;
  void *dispatch_context = nullptr;
  ggml_backend_t cpu_backend = nullptr;
  ggml_backend_sched_t scheduler = nullptr;
  ggml_context *context = nullptr;
  ggml_cgraph *graph = nullptr;
  ggml_tensor *input = nullptr;
  ggml_tensor *route_ids = nullptr;
  ggml_tensor *route_weights = nullptr;
  ggml_tensor *output = nullptr;
  int rows = 0;
  int topk = 0;
  int hidden = 0;
  int layer_count = 0;
  int layer = -1;
};

#endif
