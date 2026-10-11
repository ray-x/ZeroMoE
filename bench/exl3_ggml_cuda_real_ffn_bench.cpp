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

size_t align64(size_t value) { return (value + 63) & ~size_t(63); }

const Exl3LinearInfo *find_linear(const Exl3Contract &contract,
                                  const char *projection) {
  const std::string suffix = std::string(".mlp.experts.0.") + projection;
  for (const auto &linear : contract.linears) {
    if (linear.valid && linear.codebook == Exl3Codebook::MUL1 &&
        linear.key.ends_with(suffix))
      return &linear;
  }
  return nullptr;
}

struct FfnPayload {
  std::vector<uint8_t> bytes;
  size_t offsets[3] = {};
  Exl3PackedMatrixHeader headers[3] = {};
};

template <typename Function>
double measure_gpu(Function &&function, cudaStream_t stream, int warmup,
                   int iterations) {
  for (int index = 0; index < warmup; ++index)
    function();
  check(cudaStreamSynchronize(stream), "synchronize FFN warmup");
  cudaEvent_t begin = nullptr;
  cudaEvent_t end = nullptr;
  check(cudaEventCreate(&begin), "create FFN begin event");
  check(cudaEventCreate(&end), "create FFN end event");
  check(cudaEventRecord(begin, stream), "record FFN begin event");
  for (int index = 0; index < iterations; ++index)
    function();
  check(cudaEventRecord(end, stream), "record FFN end event");
  check(cudaEventSynchronize(end), "synchronize FFN end event");
  float milliseconds = 0.0f;
  check(cudaEventElapsedTime(&milliseconds, begin, end),
        "measure FFN GPU events");
  cudaEventDestroy(begin);
  cudaEventDestroy(end);
  return static_cast<double>(milliseconds) * 1000.0 / iterations;
}

} // namespace

int main(int argc, char **argv) {
  try {
    const std::string model_path =
        argc > 1 ? argv[1] : "/data/sdb/models/qwen3.8-flash-next/exl3_3bpw";
    const int iterations = argc > 2 ? std::atoi(argv[2]) : 100;
    const int device = argc > 3 ? std::atoi(argv[3]) : 0;
    const bool capture = argc > 4 && std::string(argv[4]) == "--capture";
    if (iterations < 1 || iterations > 10000 || device < 0)
      throw std::runtime_error("usage: exl3_ggml_cuda_real_ffn_bench "
                               "[model_dir] [iterations] [cuda_device] "
                               "[--capture]");

    SafetensorsModelInfo model;
    if (!SafetensorsAdapter::parse_directory(model_path, model))
      throw std::runtime_error("cannot parse EXL3 safetensors directory: " +
                               model_path);
    const Exl3Contract contract = SafetensorsAdapter::map_exl3_contract(model);
    if (!contract.complete)
      throw std::runtime_error("EXL3 tensor groups are incomplete");

    const Exl3LinearInfo *linears[3] = {
        find_linear(contract, "gate_proj"), find_linear(contract, "up_proj"),
        find_linear(contract, "down_proj")};
    if (!linears[0] || !linears[1] || !linears[2])
      throw std::runtime_error("missing real expert gate/up/down matrices");

    FfnPayload payload;
    for (int index = 0; index < 3; ++index) {
      std::vector<uint8_t> matrix;
      std::string error;
      if (!load_exl3_mul1_matrix(model, *linears[index], matrix, error))
        throw std::runtime_error(error);
      payload.offsets[index] = payload.bytes.size();
      payload.bytes.resize(align64(payload.bytes.size() + matrix.size()));
      std::memcpy(payload.bytes.data() + payload.offsets[index], matrix.data(),
                  matrix.size());
      std::memcpy(&payload.headers[index], matrix.data(),
                  sizeof(Exl3PackedMatrixHeader));
    }
    const int hidden = static_cast<int>(payload.headers[0].input_features);
    const int intermediate = static_cast<int>(payload.headers[0].output_features);
    if (payload.headers[1].input_features != hidden ||
        payload.headers[1].output_features != intermediate ||
        payload.headers[2].input_features != intermediate ||
        payload.headers[2].output_features != hidden)
      throw std::runtime_error("real expert FFN dimensions do not match");

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
    auto *opaque = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, payload.bytes.size());
    auto state = std::make_unique<Exl3GgmlFfnState>();
    if (!state->initialize(payload.headers, payload.offsets, 1, device))
      throw std::runtime_error("cannot initialize real EXL3 GGML FFN state");
    auto *node = exl3_ggml_map_fused_ffn(ctx, input, opaque, state.get());
    if (!node)
      throw std::runtime_error("cannot create real EXL3 FFN custom node");
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
    ggml_backend_tensor_set(opaque, payload.bytes.data(), 0,
                            payload.bytes.size());
    const auto graph_run = [&] {
      if (ggml_backend_graph_compute(adapter, graph) != GGML_STATUS_SUCCESS)
        throw std::runtime_error("real EXL3 FFN adapter graph failed");
    };

    std::vector<float> gate(intermediate), up(intermediate), reference(hidden);
    std::vector<float> down_input(intermediate);
    if (!exl3_mul1_matmul_packed(
            payload.bytes.data() + payload.offsets[0], input_host.data(),
            gate.data(), 1, hidden, intermediate) ||
        !exl3_mul1_matmul_packed(
            payload.bytes.data() + payload.offsets[1], input_host.data(),
            up.data(), 1, hidden, intermediate))
      throw std::runtime_error("real EXL3 FFN CPU gate/up failed");
    for (int index = 0; index < intermediate; ++index)
      down_input[index] = gate[index] / (1.0f + std::exp(-gate[index])) *
                          up[index];
    if (!exl3_mul1_matmul_packed(
            payload.bytes.data() + payload.offsets[2], down_input.data(),
            reference.data(), 1, intermediate, hidden))
      throw std::runtime_error("real EXL3 FFN CPU down failed");

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
        throw std::runtime_error("real EXL3 FFN output is non-finite");
      max_abs_error = std::max(max_abs_error,
                               std::abs(output[index] - reference[index]));
      max_reference = std::max(max_reference, std::abs(reference[index]));
    }
    const cudaStream_t stream = exl3_ggml_cuda_stream(cuda_backend);
    const double gpu_us = measure_gpu(graph_run, stream, 10, iterations);
    std::printf(
        "EXL3 GGML CUDA real FFN layer0 expert0 hidden=%d intermediate=%d "
        "capture=%d packed_bytes=%zu gpu_us=%.2f tok_s=%.2f "
        "max_abs_error=%g max_reference=%g\n",
        hidden, intermediate, capture ? 1 : 0, payload.bytes.size(), gpu_us,
        1.0e6 / gpu_us, max_abs_error, max_reference);

    ggml_backend_buffer_free(buffer);
    ggml_backend_free(adapter);
    ggml_free(ctx);
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "exl3_ggml_cuda_real_ffn_bench: %s\n", error.what());
    return 1;
  }
}
