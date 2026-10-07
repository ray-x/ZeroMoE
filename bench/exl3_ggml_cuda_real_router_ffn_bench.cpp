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
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kTopK = 2;

void check(cudaError_t status, const char *operation) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
}

size_t align64(size_t value) { return (value + 63) & ~size_t(63); }

const SafetensorsTensorInfo *find_router(const SafetensorsModelInfo &model) {
  for (const auto &[name, tensor] : model.tensors) {
    if (name.find("mtp.") == std::string::npos &&
        name.find("layers.0.mlp.gate.weight") != std::string::npos &&
        name.find("gate_proj") == std::string::npos)
      return &tensor;
  }
  return nullptr;
}

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

std::vector<float> decode_router(const SafetensorsTensorInfo &info,
                                 const std::vector<uint8_t> &bytes) {
  const size_t elements = static_cast<size_t>(info.shape[0]) * info.shape[1];
  std::vector<float> result(elements);
  if (info.dtype == "F32") {
    std::memcpy(result.data(), bytes.data(), elements * sizeof(float));
  } else if (info.dtype == "F16") {
    const auto *source = reinterpret_cast<const ggml_fp16_t *>(bytes.data());
    for (size_t index = 0; index < elements; ++index)
      result[index] = ggml_fp16_to_fp32(source[index]);
  } else if (info.dtype == "BF16") {
    const auto *source = reinterpret_cast<const ggml_bf16_t *>(bytes.data());
    for (size_t index = 0; index < elements; ++index)
      result[index] = ggml_bf16_to_fp32(source[index]);
  } else {
    throw std::runtime_error("unsupported router dtype: " + info.dtype);
  }
  return result;
}

std::vector<int> cpu_topk(const std::vector<float> &logits, int topk,
                          std::vector<float> &weights) {
  std::vector<int> order(logits.size());
  std::iota(order.begin(), order.end(), 0);
  std::partial_sort(order.begin(), order.begin() + topk, order.end(),
                    [&](int left, int right) {
                      return logits[left] > logits[right];
                    });
  float maximum = -INFINITY;
  for (int index = 0; index < topk; ++index)
    maximum = std::max(maximum, logits[order[index]]);
  weights.resize(topk);
  float sum = 0.0f;
  for (int index = 0; index < topk; ++index) {
    weights[index] = std::exp(logits[order[index]] - maximum);
    sum += weights[index];
  }
  for (float &weight : weights)
    weight /= sum;
  order.resize(topk);
  return order;
}

template <typename Function>
double measure_gpu(Function &&function, cudaStream_t stream, int warmup,
                   int iterations) {
  for (int index = 0; index < warmup; ++index)
    function();
  check(cudaStreamSynchronize(stream), "synchronize router FFN warmup");
  cudaEvent_t begin = nullptr;
  cudaEvent_t end = nullptr;
  check(cudaEventCreate(&begin), "create router FFN begin event");
  check(cudaEventCreate(&end), "create router FFN end event");
  check(cudaEventRecord(begin, stream), "record router FFN begin event");
  for (int index = 0; index < iterations; ++index)
    function();
  check(cudaEventRecord(end, stream), "record router FFN end event");
  check(cudaEventSynchronize(end), "synchronize router FFN end event");
  float milliseconds = 0.0f;
  check(cudaEventElapsedTime(&milliseconds, begin, end),
        "measure router FFN GPU events");
  cudaEventDestroy(begin);
  cudaEventDestroy(end);
  return static_cast<double>(milliseconds) * 1000.0 / iterations;
}

} // namespace

int main(int argc, char **argv) {
  try {
    const std::string model_path =
        argc > 1 ? argv[1] : "/data/sdb/models/qwen3.8-flash-next/exl3_3bpw";
    const int iterations = argc > 2 ? std::atoi(argv[2]) : 10;
    const int device = argc > 3 ? std::atoi(argv[3]) : 0;
    const bool capture = argc > 4 && std::string(argv[4]) == "--capture";

    SafetensorsModelInfo model;
    if (!SafetensorsAdapter::parse_directory(model_path, model))
      throw std::runtime_error("cannot parse EXL3 model directory");
    const auto *router_info = find_router(model);
    if (!router_info || router_info->shape.size() != 2)
      throw std::runtime_error("layer-0 router tensor was not found");
    std::vector<uint8_t> router_bytes;
    std::string error;
    if (!read_exl3_tensor(model, router_info->name, router_bytes, error))
      throw std::runtime_error(error);
    const int experts = static_cast<int>(router_info->shape[0]);
    const int hidden = static_cast<int>(router_info->shape[1]);
    const auto router_rows = decode_router(*router_info, router_bytes);
    std::vector<float> input_host(hidden);
    for (int index = 0; index < hidden; ++index)
      input_host[index] = 0.1f * std::sin(index * 0.071f);
    std::vector<float> logits(experts, 0.0f);
    for (int expert = 0; expert < experts; ++expert)
      for (int index = 0; index < hidden; ++index)
        logits[expert] += router_rows[static_cast<size_t>(expert) * hidden + index] *
                          input_host[index];
    std::vector<float> cpu_weights;
    const std::vector<int> selected = cpu_topk(logits, kTopK, cpu_weights);

    const Exl3Contract contract = SafetensorsAdapter::map_exl3_contract(model);
    std::vector<std::vector<uint8_t>> matrices(kTopK * 3);
    for (int slot = 0; slot < kTopK; ++slot) {
      for (int projection = 0; projection < 3; ++projection) {
        const char *name[] = {"gate_proj", "up_proj", "down_proj"};
        const auto *linear = find_linear(contract, selected[slot], name[projection]);
        if (!linear)
          throw std::runtime_error("selected router expert is missing EXL3 data");
        if (!load_exl3_mul1_matrix(model, *linear,
                                   matrices[slot * 3 + projection], error))
          throw std::runtime_error(error);
      }
    }
    Exl3GgmlRoutedPayloadLayout payload_layout;
    std::vector<uint8_t> payload;
    if (!exl3_ggml_build_routed_payload(matrices, payload, payload_layout,
                                        error))
      throw std::runtime_error(error);
    const int intermediate = payload_layout.intermediate;
    const size_t projection_offsets[3] = {
        payload_layout.projection_offsets[0],
        payload_layout.projection_offsets[1],
        payload_layout.projection_offsets[2]};
    const size_t expert_stride = payload_layout.expert_stride;
    std::vector<float> reference(hidden, 0.0f);
    for (int slot = 0; slot < kTopK; ++slot) {
      std::vector<float> gate(intermediate), up(intermediate),
          down_input(intermediate), expert_output(hidden);
      if (!exl3_mul1_matmul_packed(matrices[slot * 3].data(), input_host.data(),
                                   gate.data(), 1, hidden, intermediate) ||
          !exl3_mul1_matmul_packed(matrices[slot * 3 + 1].data(),
                                   input_host.data(), up.data(), 1, hidden,
                                   intermediate))
        throw std::runtime_error("router FFN CPU gate/up failed");
      for (int index = 0; index < intermediate; ++index)
        down_input[index] = gate[index] / (1.0f + std::exp(-gate[index])) *
                            up[index];
      if (!exl3_mul1_matmul_packed(
              matrices[slot * 3 + 2].data(), down_input.data(),
              expert_output.data(), 1, intermediate, hidden))
        throw std::runtime_error("router FFN CPU down failed");
      for (int index = 0; index < hidden; ++index)
        reference[index] += cpu_weights[slot] * expert_output[index];
    }

    ggml_backend_t cuda_backend = ggml_backend_cuda_init(device);
    if (!cuda_backend)
      throw std::runtime_error("cannot initialize CUDA backend");
    ggml_backend_t adapter = exl3_ggml_cuda_backend_wrap(cuda_backend, true);
    if (!adapter || !exl3_ggml_cuda_backend_set_graph_capture(adapter, capture))
      throw std::runtime_error("cannot initialize EXL3 GGML adapter");
    auto *ctx = ggml_init({ggml_tensor_overhead() * 32 +
                               ggml_graph_overhead_custom(32, false) + 4096,
                           nullptr, true});
    auto *input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, 1);
    auto *router = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, experts);
    auto *opaque = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, payload.size());
    auto *route_template = ggml_new_tensor_1d(
        ctx, GGML_TYPE_I8, kTopK * (sizeof(int32_t) + sizeof(float)));
    auto *logits_node = ggml_mul_mat(ctx, router, input);
    auto *probs = ggml_soft_max(ctx, logits_node);
    auto *ids = ggml_argsort_top_k(ctx, probs, kTopK);
    auto *probs3 = ggml_reshape_3d(ctx, probs, 1, experts, 1);
    auto *weights_raw = ggml_get_rows(ctx, probs3, ids);
    auto *weights2 = ggml_reshape_2d(ctx, weights_raw, kTopK, 1);
    auto *weight_sum = ggml_sum_rows(ctx, weights2);
    weights2 = ggml_div(ctx, weights2, weight_sum);
    auto *weights = ggml_reshape_3d(ctx, weights2, 1, kTopK, 1);
    std::vector<int32_t> remap(experts, -1);
    for (int slot = 0; slot < kTopK; ++slot)
      remap[selected[slot]] = slot;
    auto route_pack = std::make_unique<Exl3GgmlRoutePackState>();
    if (!route_pack->initialize_remap(kTopK, remap, device))
      throw std::runtime_error("cannot initialize router remap");
    auto *route_table = exl3_ggml_map_route_table(
        ctx, route_template, ids, weights, route_pack.get());
    auto state = std::make_unique<Exl3GgmlRoutedFfnState>();
    const int bits[3] = {payload_layout.bits[0], payload_layout.bits[1],
                         payload_layout.bits[2]};
    if (!state->initialize_dynamic(expert_stride, projection_offsets, kTopK,
                                   kTopK, hidden, intermediate, bits, 1,
                                   device))
      throw std::runtime_error("cannot initialize router FFN state");
    auto *output = exl3_ggml_map_dynamic_routed_ffn(ctx, input, opaque,
                                                     route_table, state.get());
    if (!ctx || !input || !router || !opaque || !route_table || !output ||
        !logits_node || !probs || !ids || !probs3 || !weights)
      throw std::runtime_error("cannot build router FFN graph");
    auto *graph = ggml_new_graph_custom(ctx, 32, false);
    ggml_build_forward_expand(graph, output);
    auto *buffer = ggml_backend_alloc_ctx_tensors(ctx, cuda_backend);
    if (!buffer)
      throw std::runtime_error("cannot allocate router FFN graph tensors");
    std::vector<float> router_transposed(static_cast<size_t>(hidden) * experts);
    for (int expert = 0; expert < experts; ++expert)
      for (int index = 0; index < hidden; ++index)
        router_transposed[static_cast<size_t>(expert) * hidden + index] =
            router_rows[static_cast<size_t>(expert) * hidden + index];
    ggml_backend_tensor_set(input, input_host.data(), 0,
                            input_host.size() * sizeof(float));
    ggml_backend_tensor_set(router, router_transposed.data(), 0,
                            router_transposed.size() * sizeof(float));
    ggml_backend_tensor_set(opaque, payload.data(), 0, payload.size());
    const auto graph_run = [&] {
      if (ggml_backend_graph_compute(adapter, graph) != GGML_STATUS_SUCCESS)
        throw std::runtime_error("router FFN graph execution failed");
    };
    graph_run();
    if (capture)
      graph_run();
    std::vector<int32_t> graph_ids(kTopK);
    std::vector<float> graph_weights(kTopK);
    std::vector<float> graph_logits(experts);
    ggml_backend_tensor_get(ids, graph_ids.data(), 0,
                            graph_ids.size() * sizeof(int32_t));
    ggml_backend_tensor_get(weights, graph_weights.data(), 0,
                            graph_weights.size() * sizeof(float));
    ggml_backend_tensor_get(logits_node, graph_logits.data(), 0,
                            graph_logits.size() * sizeof(float));
    std::vector<float> output_host(hidden);
    ggml_backend_tensor_get(output, output_host.data(), 0,
                            output_host.size() * sizeof(float));
    float max_error = 0.0f;
    for (size_t index = 0; index < output_host.size(); ++index) {
      const float value = output_host[index];
      if (!std::isfinite(value))
        throw std::runtime_error("router FFN output is non-finite");
      max_error = std::max(max_error, std::abs(value - reference[index]));
    }
    const cudaStream_t stream = exl3_ggml_cuda_stream(cuda_backend);
    const double gpu_us = measure_gpu(graph_run, stream, 3, iterations);
    std::printf(
        "EXL3 GGML CUDA real router FFN router=%s dtype=%s experts=%d "
        "selected=%d,%d graph_ids=%d,%d capture=%d gpu_us=%.2f tok_s=%.2f "
        "graph_weights=%g,%g cpu_weights=%g,%g max_error=%g\n",
        router_info->name.c_str(), router_info->dtype.c_str(), experts,
        selected[0], selected[1], graph_ids[0], graph_ids[1], capture ? 1 : 0,
        gpu_us, 1.0e6 / gpu_us, graph_weights[0], graph_weights[1],
        cpu_weights[0], cpu_weights[1], max_error);
    std::printf("router_logits cpu=%g,%g graph=%g,%g selected=%d,%d\n",
                logits[selected[0]], logits[selected[1]],
                graph_logits[selected[0]], graph_logits[selected[1]],
                graph_ids[0], graph_ids[1]);
    ggml_backend_buffer_free(buffer);
    ggml_backend_free(adapter);
    ggml_free(ctx);
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "exl3_ggml_cuda_real_router_ffn_bench: %s\n",
                 error.what());
    return 1;
  }
}
