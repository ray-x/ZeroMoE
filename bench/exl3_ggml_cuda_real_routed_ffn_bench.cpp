#include "exl3_cpu.hpp"
#include "exl3_format.hpp"
#include "exl3_ggml_backend.hpp"
#include "exl3_ggml_ffn.hpp"
#include "exl3_model_loader.hpp"
#include "safetensors_adapter.hpp"

#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "ggml.h"

#include <cuda_runtime.h>

#include <algorithm>
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

size_t align64(size_t value) { return (value + 63) & ~size_t(63); }

const Exl3LinearInfo *find_linear(const Exl3Contract &contract, int expert,
                                  const char *projection) {
  const std::string suffix = ".mlp.experts." + std::to_string(expert) + "." +
                             projection;
  for (const auto &linear : contract.linears) {
    if (linear.valid && linear.codebook == Exl3Codebook::MUL1 &&
        linear.key.ends_with(suffix))
      return &linear;
  }
  return nullptr;
}

template <typename Function>
double measure_gpu(Function &&function, cudaStream_t stream, int warmup,
                   int iterations) {
  for (int index = 0; index < warmup; ++index)
    function();
  check(cudaStreamSynchronize(stream), "synchronize routed FFN warmup");
  cudaEvent_t begin = nullptr;
  cudaEvent_t end = nullptr;
  check(cudaEventCreate(&begin), "create routed FFN begin event");
  check(cudaEventCreate(&end), "create routed FFN end event");
  check(cudaEventRecord(begin, stream), "record routed FFN begin event");
  for (int index = 0; index < iterations; ++index)
    function();
  check(cudaEventRecord(end, stream), "record routed FFN end event");
  check(cudaEventSynchronize(end), "synchronize routed FFN end event");
  float milliseconds = 0.0f;
  check(cudaEventElapsedTime(&milliseconds, begin, end),
        "measure routed FFN GPU events");
  cudaEventDestroy(begin);
  cudaEventDestroy(end);
  return static_cast<double>(milliseconds) * 1000.0 / iterations;
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
    const bool capture = argc > 4 && std::string(argv[4]) == "--capture";
    constexpr int expert_count = 2;
    const float route_values[expert_count] = {0.6f, 0.4f};
    if (iterations < 1 || iterations > 10000 || device < 0)
      throw std::runtime_error("usage: exl3_ggml_cuda_real_routed_ffn_bench "
                               "[model_dir] [iterations] [cuda_device] "
                               "[--capture]");

    SafetensorsModelInfo model;
    if (!SafetensorsAdapter::parse_directory(model_path, model))
      throw std::runtime_error("cannot parse EXL3 safetensors directory: " +
                               model_path);
    const Exl3Contract contract = SafetensorsAdapter::map_exl3_contract(model);
    if (!contract.complete)
      throw std::runtime_error("EXL3 tensor groups are incomplete");

    std::vector<uint8_t> payload;
    std::vector<size_t> offsets;
    std::vector<Exl3PackedMatrixHeader> headers;
    offsets.reserve(expert_count * 3);
    headers.reserve(expert_count * 3);
    for (int expert = 0; expert < expert_count; ++expert) {
      for (const char *projection : {"gate_proj", "up_proj", "down_proj"}) {
        const Exl3LinearInfo *linear =
            find_linear(contract, expert, projection);
        if (!linear)
          throw std::runtime_error("missing routed EXL3 projection");
        std::vector<uint8_t> matrix;
        std::string error;
        if (!load_exl3_mul1_matrix(model, *linear, matrix, error))
          throw std::runtime_error(error);
        offsets.push_back(payload.size());
        payload.resize(align64(payload.size() + matrix.size()));
        std::memcpy(payload.data() + offsets.back(), matrix.data(),
                    matrix.size());
        Exl3PackedMatrixHeader header{};
        std::memcpy(&header, matrix.data(), sizeof(header));
        headers.push_back(header);
      }
    }
    const int hidden = static_cast<int>(headers[0].input_features);
    const int intermediate = static_cast<int>(headers[0].output_features);
    for (int expert = 0; expert < expert_count; ++expert) {
      if (headers[expert * 3].input_features != hidden ||
          headers[expert * 3].output_features != intermediate ||
          headers[expert * 3 + 1].input_features != hidden ||
          headers[expert * 3 + 1].output_features != intermediate ||
          headers[expert * 3 + 2].input_features != intermediate ||
          headers[expert * 3 + 2].output_features != hidden)
        throw std::runtime_error("routed EXL3 FFN dimensions do not match");
    }

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
    auto *input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, 1);
    auto *opaque = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, payload.size());
    auto *route = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, expert_count, 1);
    auto state = std::make_unique<Exl3GgmlRoutedFfnState>();
    if (!state->initialize(headers, offsets, 1, device))
      throw std::runtime_error("cannot initialize routed EXL3 GGML FFN state");
    auto *node = exl3_ggml_map_routed_ffn(ctx, input, opaque, route,
                                          state.get());
    if (!node)
      throw std::runtime_error("cannot create routed EXL3 GGML FFN node");
    auto *graph = ggml_new_graph_custom(ctx, 16, false);
    ggml_build_forward_expand(graph, node);
    auto *buffer = ggml_backend_alloc_ctx_tensors(ctx, cuda_backend);
    if (!buffer)
      throw std::runtime_error("cannot allocate CUDA GGML tensors");

    std::vector<float> input_host(hidden);
    for (int index = 0; index < hidden; ++index)
      input_host[index] = 0.1f * std::sin(index * 0.071f);
    ggml_backend_tensor_set(input, input_host.data(), 0,
                            input_host.size() * sizeof(float));
    ggml_backend_tensor_set(opaque, payload.data(), 0, payload.size());
    ggml_backend_tensor_set(route, route_values, 0, sizeof(route_values));

    std::vector<float> reference(hidden, 0.0f);
    for (int expert = 0; expert < expert_count; ++expert) {
      std::vector<float> gate(intermediate), up(intermediate), down_input(intermediate),
          output(hidden);
      if (!exl3_mul1_matmul_packed(
              payload.data() + offsets[expert * 3], input_host.data(),
              gate.data(), 1, hidden, intermediate) ||
          !exl3_mul1_matmul_packed(
              payload.data() + offsets[expert * 3 + 1], input_host.data(),
              up.data(), 1, hidden, intermediate))
        throw std::runtime_error("routed EXL3 CPU gate/up failed");
      for (int index = 0; index < intermediate; ++index)
        down_input[index] = gate[index] / (1.0f + std::exp(-gate[index])) *
                            up[index];
      if (!exl3_mul1_matmul_packed(
              payload.data() + offsets[expert * 3 + 2], down_input.data(),
              output.data(), 1, intermediate, hidden))
        throw std::runtime_error("routed EXL3 CPU down failed");
      for (int index = 0; index < hidden; ++index)
        reference[index] += route_values[expert] * output[index];
    }

    const auto graph_run = [&] {
      if (ggml_backend_graph_compute(adapter, graph) != GGML_STATUS_SUCCESS)
        throw std::runtime_error("routed EXL3 GGML adapter graph failed");
    };
    graph_run();
    if (capture)
      graph_run();
    std::vector<float> output(hidden);
    ggml_backend_tensor_get(node, output.data(), 0,
                            output.size() * sizeof(float));
    float max_abs_error = 0.0f;
    float max_reference = 0.0f;
    for (size_t index = 0; index < output.size(); ++index) {
      if (!std::isfinite(output[index]))
        throw std::runtime_error("routed EXL3 output is non-finite");
      max_abs_error = std::max(max_abs_error,
                               std::abs(output[index] - reference[index]));
      max_reference = std::max(max_reference, std::abs(reference[index]));
    }
    const cudaStream_t stream = exl3_ggml_cuda_stream(cuda_backend);
    const double gpu_us = measure_gpu(graph_run, stream, 10, iterations);
    std::printf(
        "EXL3 GGML CUDA real routed FFN experts=%d hidden=%d intermediate=%d "
        "capture=%d packed_bytes=%zu gpu_us=%.2f tok_s=%.2f "
        "max_abs_error=%g max_reference=%g\n",
        expert_count, hidden, intermediate, capture ? 1 : 0, payload.size(),
        gpu_us, 1.0e6 / gpu_us, max_abs_error, max_reference);

    ggml_backend_buffer_free(buffer);
    ggml_backend_free(adapter);
    ggml_free(ctx);
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "exl3_ggml_cuda_real_routed_ffn_bench: %s\n",
                 error.what());
    return 1;
  }
}
