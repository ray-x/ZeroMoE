#include "exl3_cpu.hpp"
#include "exl3_cuda.hpp"
#include "exl3_format.hpp"
#include "exl3_ggml_backend.hpp"
#include "exl3_model_loader.hpp"
#include "safetensors_adapter.hpp"

#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "ggml.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void check(cudaError_t status, const char *operation) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
}

struct RealExl3State {
  cudaStream_t stream = nullptr;
  float *partials = nullptr;
  Exl3PackedMatrixHeader header{};
  bool has_route_weight = false;

  ~RealExl3State() {
    if (partials)
      cudaFree(partials);
  }

  void run(const ggml_tensor *input, const ggml_tensor *opaque,
           const ggml_tensor *routing, ggml_tensor *output) {
    const auto *base = static_cast<const uint8_t *>(opaque->data);
    const Exl3CudaMatrixView matrix{
        reinterpret_cast<const uint16_t *>(base + header.trellis_offset),
        reinterpret_cast<const uint16_t *>(base + header.suh_offset),
        reinterpret_cast<const uint16_t *>(base + header.svh_offset),
        (header.flags & kExl3PackedHasBias)
            ? reinterpret_cast<const uint16_t *>(base + header.bias_offset)
            : nullptr,
        static_cast<int>(header.input_features),
        static_cast<int>(header.output_features),
        static_cast<int>(header.bits)};
    check(launch_exl3_mul1_matmul(
              matrix, static_cast<const float *>(input->data),
              static_cast<float *>(output->data),
              static_cast<int>(input->ne[1]), partials, stream),
          "launch real EXL3 GGML custom node");
    if (has_route_weight) {
      if (!routing || routing->type != GGML_TYPE_F32 || routing->ne[0] != 1)
        throw std::runtime_error("missing EXL3 route-weight tensor");
      check(launch_exl3_scale_device_weight(
                static_cast<float *>(output->data),
                static_cast<const float *>(routing->data),
                static_cast<int>(header.output_features), stream),
            "apply device route weight");
    }
  }

  static void compute(ggml_backend_t cuda_backend, ggml_tensor *node,
                      void *user_data) {
    auto *state = static_cast<RealExl3State *>(user_data);
    state->stream = exl3_ggml_cuda_stream(cuda_backend);
    state->run(node->src[0], node->src[1], node->src[2], node);
  }
};

template <typename Function>
double measure_enqueue(Function &&function, cudaStream_t stream, int warmup,
                       int iterations) {
  for (int index = 0; index < warmup; ++index)
    function();
  check(cudaStreamSynchronize(stream), "synchronize warmup");
  const auto start = std::chrono::steady_clock::now();
  for (int index = 0; index < iterations; ++index)
    function();
  check(cudaStreamSynchronize(stream), "synchronize enqueue benchmark");
  return std::chrono::duration<double, std::micro>(
             std::chrono::steady_clock::now() - start)
      .count() /
         iterations;
}

template <typename Function>
double measure_gpu(Function &&function, cudaStream_t stream, int warmup,
                   int iterations) {
  for (int index = 0; index < warmup; ++index)
    function();
  check(cudaStreamSynchronize(stream), "synchronize GPU warmup");

  cudaEvent_t begin = nullptr;
  cudaEvent_t end = nullptr;
  check(cudaEventCreate(&begin), "create GPU begin event");
  check(cudaEventCreate(&end), "create GPU end event");
  check(cudaEventRecord(begin, stream), "record GPU begin event");
  for (int index = 0; index < iterations; ++index)
    function();
  check(cudaEventRecord(end, stream), "record GPU end event");
  check(cudaEventSynchronize(end), "synchronize GPU end event");
  float milliseconds = 0.0f;
  check(cudaEventElapsedTime(&milliseconds, begin, end),
        "measure GPU events");
  cudaEventDestroy(begin);
  cudaEventDestroy(end);
  return static_cast<double>(milliseconds) * 1000.0 / iterations;
}

const Exl3LinearInfo *select_matrix(const Exl3Contract &contract) {
  const Exl3LinearInfo *selected = nullptr;
  for (const auto &linear : contract.linears) {
    if (!linear.valid || linear.codebook != Exl3Codebook::MUL1)
      continue;
    if (linear.key.find(".mlp.experts.") != std::string::npos &&
        linear.key.ends_with(".gate_proj"))
      return &linear;
    if (!selected || linear.in_features * linear.out_features <
                         selected->in_features * selected->out_features)
      selected = &linear;
  }
  return selected;
}

} // namespace

int main(int argc, char **argv) {
  try {
    const char *model_env = std::getenv("ZEROMOE_EXL3_MODEL");
    const std::string model_path =
        argc > 1 ? argv[1] : (model_env ? model_env : "");
    if (model_path.empty())
      throw std::runtime_error("pass <model_dir> or set ZEROMOE_EXL3_MODEL");
    const int iterations = argc > 2 ? std::atoi(argv[2]) : 100;
    const int device = argc > 3 ? std::atoi(argv[3]) : 0;
    bool capture = false;
    bool custom3 = false;
    for (int index = 4; index < argc; ++index) {
      capture = capture || std::string(argv[index]) == "--capture";
      custom3 = custom3 || std::string(argv[index]) == "--custom3";
    }
    if (iterations < 1 || iterations > 10000 || device < 0)
      throw std::runtime_error("usage: exl3_ggml_cuda_real_bench [model_dir] "
                               "[iterations] [cuda_device]");

    SafetensorsModelInfo model;
    if (!SafetensorsAdapter::parse_directory(model_path, model))
      throw std::runtime_error("cannot parse EXL3 safetensors directory: " +
                               model_path);
    const Exl3Contract contract = SafetensorsAdapter::map_exl3_contract(model);
    if (!contract.complete)
      throw std::runtime_error("EXL3 tensor groups are incomplete");
    const Exl3LinearInfo *linear = select_matrix(contract);
    if (!linear)
      throw std::runtime_error("no dispatchable MUL1 EXL3 matrix found");

    std::vector<uint8_t> packed;
    std::string error;
    if (!load_exl3_mul1_matrix(model, *linear, packed, error))
      throw std::runtime_error(error);
    Exl3PackedMatrixHeader header{};
    if (packed.size() < sizeof(header))
      throw std::runtime_error("packed EXL3 matrix is truncated");
    std::memcpy(&header, packed.data(), sizeof(header));
    if (!exl3_valid_packed_header(header, packed.size()))
      throw std::runtime_error("packed EXL3 matrix header is invalid");

    const int input_features = static_cast<int>(header.input_features);
    const int output_features = static_cast<int>(header.output_features);
    ggml_backend_t cuda_backend = ggml_backend_cuda_init(device);
    if (!cuda_backend)
      throw std::runtime_error("cannot initialize CUDA backend");
    ggml_backend_t adapter = exl3_ggml_cuda_backend_wrap(cuda_backend, true);
    if (!adapter)
      throw std::runtime_error("cannot initialize EXL3 GGML adapter");
    if (!exl3_ggml_cuda_backend_set_graph_capture(adapter, capture))
      throw std::runtime_error("cannot configure EXL3 GGML graph capture");

    auto *ctx = ggml_init({ggml_tensor_overhead() * 16 +
                               ggml_graph_overhead_custom(16, false) + 4096,
                           nullptr, true});
    if (!ctx)
      throw std::runtime_error("cannot initialize GGML context");
    auto *input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, input_features, 1);
    auto *opaque = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, packed.size());
    auto *routing = custom3 ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1)
                            : nullptr;
    auto state = std::make_unique<RealExl3State>();
    state->header = header;
    state->has_route_weight = custom3;
    Exl3GgmlCustomNode custom;
    custom.compute = RealExl3State::compute;
    custom.user_data = state.get();
    auto *node = custom3
                     ? exl3_ggml_map_custom3(ctx, input, opaque, routing,
                                             &custom)
                     : exl3_ggml_map_custom2(ctx, input, opaque, &custom);
    if (!node)
      throw std::runtime_error("cannot create EXL3 GGML custom node");
    node->ne[0] = output_features;
    node->ne[1] = 1;
    node->nb[1] = output_features * sizeof(float);
    auto *graph = ggml_new_graph_custom(ctx, 16, false);
    ggml_build_forward_expand(graph, node);
    auto *buffer = ggml_backend_alloc_ctx_tensors(ctx, cuda_backend);
    if (!buffer)
      throw std::runtime_error("cannot allocate CUDA GGML tensors");

    std::vector<float> input_host(input_features);
    for (int index = 0; index < input_features; ++index)
      input_host[index] = 0.1f * std::sin(index * 0.071f);
    ggml_backend_tensor_set(input, input_host.data(), 0,
                            input_host.size() * sizeof(float));
    ggml_backend_tensor_set(opaque, packed.data(), 0, packed.size());
    const float route_weight = 0.5f;
    if (routing)
      ggml_backend_tensor_set(routing, &route_weight, 0, sizeof(route_weight));
    check(cudaMalloc(&state->partials,
                     static_cast<size_t>(output_features) *
                         exl3_cuda_key_splits(input_features) * sizeof(float)),
          "allocate EXL3 partials");
    const cudaStream_t stream = exl3_ggml_cuda_stream(cuda_backend);
    const auto graph_run = [&] {
      if (ggml_backend_graph_compute(adapter, graph) != GGML_STATUS_SUCCESS)
        throw std::runtime_error("EXL3 GGML adapter graph failed");
    };

    std::vector<float> reference(output_features);
    if (!exl3_mul1_matmul_packed(packed.data(), input_host.data(),
                                 reference.data(), 1, input_features,
                                 output_features))
      throw std::runtime_error("CPU EXL3 parity matmul failed");
    if (custom3)
      for (float &value : reference)
        value *= route_weight;
    graph_run();
    if (capture)
      graph_run();
    std::vector<float> output(output_features);
    ggml_backend_tensor_get(node, output.data(), 0,
                            output.size() * sizeof(float));
    float max_abs_error = 0.0f;
    float max_reference = 0.0f;
    for (size_t index = 0; index < output.size(); ++index) {
      if (!std::isfinite(output[index]))
        throw std::runtime_error("real EXL3 adapter output is non-finite");
      max_abs_error = std::max(max_abs_error,
                               std::abs(output[index] - reference[index]));
      max_reference = std::max(max_reference, std::abs(reference[index]));
    }
    const double enqueue_us =
        measure_enqueue(graph_run, stream, 10, iterations);
    const double gpu_us = measure_gpu(graph_run, stream, 10, iterations);
    std::printf(
        "EXL3 GGML CUDA real matrix key=%s in=%d out=%d bits=%u "
        "packed_bytes=%zu enqueue_us=%.2f gpu_us=%.2f tok_s=%.2f "
        "capture=%d custom3=%d max_abs_error=%g max_reference=%g\n",
        linear->key.c_str(), input_features, output_features, header.bits,
        packed.size(), enqueue_us, gpu_us, 1.0e6 / gpu_us, capture ? 1 : 0,
        custom3 ? 1 : 0, max_abs_error, max_reference);

    ggml_backend_buffer_free(buffer);
    ggml_backend_free(adapter);
    ggml_free(ctx);
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "exl3_ggml_cuda_real_bench: %s\n", error.what());
    return 1;
  }
}
