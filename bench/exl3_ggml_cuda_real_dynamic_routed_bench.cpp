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

constexpr int kExperts = 2;
constexpr int kTopK = 2;

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
  for (const auto &linear : contract.linears)
    if (linear.valid && linear.codebook == Exl3Codebook::MUL1 &&
        linear.key.ends_with(suffix))
      return &linear;
  return nullptr;
}

struct RouteTable {
  int32_t ids[kTopK] = {};
  float weights[kTopK] = {};
};

template <typename Function>
double measure_gpu(Function &&function, cudaStream_t stream, int warmup,
                   int iterations) {
  for (int index = 0; index < warmup; ++index)
    function();
  check(cudaStreamSynchronize(stream), "synchronize dynamic route warmup");
  cudaEvent_t begin = nullptr;
  cudaEvent_t end = nullptr;
  check(cudaEventCreate(&begin), "create dynamic route begin event");
  check(cudaEventCreate(&end), "create dynamic route end event");
  check(cudaEventRecord(begin, stream), "record dynamic route begin event");
  for (int index = 0; index < iterations; ++index)
    function();
  check(cudaEventRecord(end, stream), "record dynamic route end event");
  check(cudaEventSynchronize(end), "synchronize dynamic route end event");
  float milliseconds = 0.0f;
  check(cudaEventElapsedTime(&milliseconds, begin, end),
        "measure dynamic route GPU events");
  cudaEventDestroy(begin);
  cudaEventDestroy(end);
  return static_cast<double>(milliseconds) * 1000.0 / iterations;
}

} // namespace

int main(int argc, char **argv) {
  try {
    const std::string model_path =
        argc > 1 ? argv[1] : "/data/sdb/models/qwen3.8-flash-next/exl3_3bpw";
    const int iterations = argc > 2 ? std::atoi(argv[2]) : 20;
    const int device = argc > 3 ? std::atoi(argv[3]) : 0;
    const bool capture = argc > 4 && std::string(argv[4]) == "--capture";
    if (iterations < 1 || iterations > 10000 || device < 0)
      throw std::runtime_error("usage: exl3_ggml_cuda_real_dynamic_routed_bench "
                               "[model_dir] [iterations] [cuda_device] "
                               "[--capture]");

    SafetensorsModelInfo model;
    if (!SafetensorsAdapter::parse_directory(model_path, model))
      throw std::runtime_error("cannot parse EXL3 safetensors directory: " +
                               model_path);
    const Exl3Contract contract = SafetensorsAdapter::map_exl3_contract(model);
    if (!contract.complete)
      throw std::runtime_error("EXL3 tensor groups are incomplete");

    std::vector<std::vector<uint8_t>> matrices(kExperts * 3);
    size_t slot_size = 0;
    std::vector<Exl3PackedMatrixHeader> headers(kExperts * 3);
    for (int expert = 0; expert < kExperts; ++expert) {
      for (int projection = 0; projection < 3; ++projection) {
        const char *name[] = {"gate_proj", "up_proj", "down_proj"};
        const auto *linear = find_linear(contract, expert, name[projection]);
        if (!linear)
          throw std::runtime_error("missing dynamic routed EXL3 projection");
        std::string error;
        if (!load_exl3_mul1_matrix(model, *linear,
                                   matrices[expert * 3 + projection], error))
          throw std::runtime_error(error);
        std::memcpy(&headers[expert * 3 + projection],
                    matrices[expert * 3 + projection].data(),
                    sizeof(Exl3PackedMatrixHeader));
        slot_size = std::max(slot_size,
                             align64(matrices[expert * 3 + projection].size()));
      }
    }
    const size_t projection_offsets[3] = {0, slot_size, 2 * slot_size};
    const size_t expert_stride = 3 * slot_size;
    std::vector<uint8_t> payload(kExperts * expert_stride);
    for (int expert = 0; expert < kExperts; ++expert)
      for (int projection = 0; projection < 3; ++projection)
        std::memcpy(payload.data() + expert * expert_stride +
                        projection_offsets[projection],
                    matrices[expert * 3 + projection].data(),
                    matrices[expert * 3 + projection].size());

    const int hidden = static_cast<int>(headers[0].input_features);
    const int intermediate = static_cast<int>(headers[0].output_features);
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
    auto *route_table = ggml_new_tensor_1d(
        ctx, GGML_TYPE_I8, kTopK * (sizeof(int32_t) + sizeof(float)));
    auto *route_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, kTopK);
    auto *route_weights = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kTopK);
    auto route_pack = std::make_unique<Exl3GgmlRoutePackState>();
    if (!route_pack->initialize(kTopK))
      throw std::runtime_error("cannot initialize EXL3 GGML route pack state");
    route_table = exl3_ggml_map_route_table(
        ctx, route_table, route_ids, route_weights, route_pack.get());
    if (!route_table)
      throw std::runtime_error("cannot create EXL3 GGML route table node");
    auto state = std::make_unique<Exl3GgmlRoutedFfnState>();
    const int route_bits[3] = {static_cast<int>(headers[0].bits),
                               static_cast<int>(headers[1].bits),
                               static_cast<int>(headers[2].bits)};
    if (!state->initialize_dynamic(expert_stride, projection_offsets, kExperts,
                                   kTopK, hidden, intermediate, route_bits, 1,
                                   device))
      throw std::runtime_error("cannot initialize dynamic routed EXL3 state");
    auto *node = exl3_ggml_map_dynamic_routed_ffn(ctx, input, opaque,
                                                   route_table,
                                                   state.get());
    if (!node)
      throw std::runtime_error("cannot create dynamic routed EXL3 node");
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

    const auto cpu_reference = [&](const RouteTable &table) {
      std::vector<float> result(hidden, 0.0f);
      for (int route_index = 0; route_index < kTopK; ++route_index) {
        const int expert = table.ids[route_index];
        if (expert < 0 || expert >= kExperts)
          throw std::runtime_error("invalid CPU route ID");
        std::vector<float> gate(intermediate), up(intermediate),
            down_input(intermediate), output(hidden);
        if (!exl3_mul1_matmul_packed(
                matrices[expert * 3].data(), input_host.data(), gate.data(), 1,
                hidden, intermediate) ||
            !exl3_mul1_matmul_packed(matrices[expert * 3 + 1].data(),
                                     input_host.data(), up.data(), 1, hidden,
                                     intermediate))
          throw std::runtime_error("dynamic route CPU gate/up failed");
        for (int index = 0; index < intermediate; ++index)
          down_input[index] = gate[index] / (1.0f + std::exp(-gate[index])) *
                              up[index];
        if (!exl3_mul1_matmul_packed(
                matrices[expert * 3 + 2].data(), down_input.data(),
                output.data(), 1, intermediate, hidden))
          throw std::runtime_error("dynamic route CPU down failed");
        for (int index = 0; index < hidden; ++index)
          result[index] += table.weights[route_index] * output[index];
      }
      return result;
    };

    const auto graph_run = [&] {
      if (ggml_backend_graph_compute(adapter, graph) != GGML_STATUS_SUCCESS)
        throw std::runtime_error("dynamic routed EXL3 graph failed");
    };
    const auto run_case = [&](const RouteTable &table) {
      ggml_backend_tensor_set(route_ids, table.ids, 0, sizeof(table.ids));
      ggml_backend_tensor_set(route_weights, table.weights, 0,
                              sizeof(table.weights));
      graph_run();
      if (capture)
        graph_run();
      std::vector<float> output(hidden);
      ggml_backend_tensor_get(node, output.data(), 0,
                              output.size() * sizeof(float));
      const auto reference = cpu_reference(table);
      float max_error = 0.0f;
      for (size_t index = 0; index < output.size(); ++index)
        max_error = std::max(max_error,
                             std::abs(output[index] - reference[index]));
      return max_error;
    };

    const RouteTable first{{0, 1}, {0.6f, 0.4f}};
    const RouteTable second{{1, 0}, {0.25f, 0.75f}};
    const float first_error = run_case(first);
    const float second_error = run_case(second);
    const cudaStream_t stream = exl3_ggml_cuda_stream(cuda_backend);
    const double gpu_us = measure_gpu([&] { graph_run(); }, stream, 5,
                                      iterations);
    std::printf(
        "EXL3 GGML CUDA dynamic routed experts=%d topk=%d hidden=%d "
        "capture=%d payload_bytes=%zu gpu_us=%.2f tok_s=%.2f "
        "first_route_error=%g second_route_error=%g\n",
        kExperts, kTopK, hidden, capture ? 1 : 0, payload.size(), gpu_us,
        1.0e6 / gpu_us, first_error, second_error);

    ggml_backend_buffer_free(buffer);
    ggml_backend_free(adapter);
    ggml_free(ctx);
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "exl3_ggml_cuda_real_dynamic_routed_bench: %s\n",
                 error.what());
    return 1;
  }
}
