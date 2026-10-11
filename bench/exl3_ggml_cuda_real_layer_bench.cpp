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

constexpr int kSelectedExperts = 2;

void check(cudaError_t status, const char *operation) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
}

const SafetensorsTensorInfo *find_tensor(const SafetensorsModelInfo &model,
                                         int layer, const char *suffix) {
  const std::string needle = ".layers." + std::to_string(layer) + "." + suffix;
  for (const auto &[name, info] : model.tensors)
    if (name.find("mtp.") == std::string::npos && name.find(needle) != std::string::npos)
      return &info;
  return nullptr;
}

const Exl3LinearInfo *find_expert(const Exl3Contract &contract, int layer,
                                  int expert, const char *projection) {
  const std::string suffix = ".layers." + std::to_string(layer) +
                             ".mlp.experts." + std::to_string(expert) + "." +
                             projection;
  for (const auto &linear : contract.linears)
    if (linear.valid && linear.codebook == Exl3Codebook::MUL1 &&
        linear.key.find(suffix) != std::string::npos)
      return &linear;
  return nullptr;
}

std::vector<float> load_dense(const SafetensorsModelInfo &model,
                              const SafetensorsTensorInfo &info) {
  std::vector<uint8_t> bytes;
  std::string error;
  if (!read_exl3_tensor(model, info.name, bytes, error))
    throw std::runtime_error(error);
  if (info.shape.empty())
    throw std::runtime_error("dense tensor has no shape: " + info.name);
  size_t elements = 1;
  for (int64_t dimension : info.shape)
    elements *= static_cast<size_t>(dimension);
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
    throw std::runtime_error("unsupported dense tensor dtype: " + info.dtype);
  }
  return result;
}

ggml_tensor *new_dense_weight(ggml_context *ctx, const SafetensorsModelInfo &model,
                              const SafetensorsTensorInfo &info) {
  if (info.shape.size() != 2)
    return nullptr;
  auto values = load_dense(model, info);
  auto *tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, info.shape[1], info.shape[0]);
  if (!tensor)
    return nullptr;
  tensor->extra = new std::vector<float>(std::move(values));
  return tensor;
}

void set_owned_tensor(ggml_backend_t backend, ggml_tensor *tensor) {
  auto *values = static_cast<std::vector<float> *>(tensor->extra);
  ggml_backend_tensor_set(tensor, values->data(), 0,
                          values->size() * sizeof(float));
  delete values;
  tensor->extra = nullptr;
}

template <typename Function>
double measure_gpu(Function &&function, cudaStream_t stream, int warmup,
                   int iterations) {
  for (int index = 0; index < warmup; ++index)
    function();
  check(cudaStreamSynchronize(stream), "synchronize layer warmup");
  cudaEvent_t begin = nullptr;
  cudaEvent_t end = nullptr;
  check(cudaEventCreate(&begin), "create layer begin event");
  check(cudaEventCreate(&end), "create layer end event");
  check(cudaEventRecord(begin, stream), "record layer begin event");
  for (int index = 0; index < iterations; ++index)
    function();
  check(cudaEventRecord(end, stream), "record layer end event");
  check(cudaEventSynchronize(end), "synchronize layer end event");
  float milliseconds = 0.0f;
  check(cudaEventElapsedTime(&milliseconds, begin, end),
        "measure layer GPU events");
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
    const Exl3Contract contract = SafetensorsAdapter::map_exl3_contract(model);
    int layer = -1;
    const char *dense_names[] = {"self_attn.q_proj.weight",
                                 "self_attn.k_proj.weight",
                                 "self_attn.v_proj.weight",
                                 "self_attn.o_proj.weight",
                                 "self_attn.q_norm.weight",
                                 "self_attn.k_norm.weight"};
    for (int candidate = 0; candidate < 48 && layer < 0; ++candidate) {
      bool complete = true;
      for (const char *name : dense_names)
        complete = complete && find_tensor(model, candidate, name);
      if (complete)
        layer = candidate;
    }
    if (layer < 0) {
      int printed = 0;
      for (const auto &[name, info] : model.tensors) {
        if (name.find("layers.") == std::string::npos ||
            name.ends_with(".trellis") || name.ends_with(".suh") ||
            name.ends_with(".svh"))
          continue;
        std::fprintf(stderr, "[EXL3-INSPECT] %s dtype=%s\n", name.c_str(),
                     info.dtype.c_str());
        if (++printed == 80)
          break;
      }
      throw std::runtime_error("no complete full-attention layer found");
    }
    const auto *q_info = find_tensor(model, layer, "self_attn.q_proj.weight");
    const auto *k_info = find_tensor(model, layer, "self_attn.k_proj.weight");
    const auto *v_info = find_tensor(model, layer, "self_attn.v_proj.weight");
    const auto *o_info = find_tensor(model, layer, "self_attn.o_proj.weight");
    const auto *qn_info = find_tensor(model, layer, "self_attn.q_norm.weight");
    const auto *kn_info = find_tensor(model, layer, "self_attn.k_norm.weight");
    const int hidden = static_cast<int>(q_info->shape[1]);
    const int q_out = static_cast<int>(q_info->shape[0]);
    const int k_out = static_cast<int>(k_info->shape[0]);
    const int head_dim = static_cast<int>(qn_info->shape[0]);
    const int heads = q_out / (2 * head_dim);
    const int kv_heads = k_out / head_dim;
    if (heads <= 0 || kv_heads <= 0 || heads % kv_heads != 0)
      throw std::runtime_error("invalid full-attention head geometry");

    const Exl3LinearInfo *expert_linears[kSelectedExperts][3] = {};
    std::vector<std::vector<uint8_t>> matrices(kSelectedExperts * 3);
    for (int expert = 0; expert < kSelectedExperts; ++expert) {
      for (int projection = 0; projection < 3; ++projection) {
        const char *name[] = {"gate_proj", "up_proj", "down_proj"};
        expert_linears[expert][projection] =
            find_expert(contract, layer, expert, name[projection]);
        if (!expert_linears[expert][projection])
          throw std::runtime_error("missing full-layer EXL3 expert");
        std::string error;
        if (!load_exl3_mul1_matrix(model, *expert_linears[expert][projection],
                                   matrices[expert * 3 + projection], error))
          throw std::runtime_error(error);
      }
    }
    Exl3GgmlRoutedPayloadLayout payload_layout;
    std::vector<uint8_t> payload;
    std::string payload_error;
    if (!exl3_ggml_build_routed_payload(matrices, payload, payload_layout,
                                        payload_error))
      throw std::runtime_error(payload_error);

    ggml_backend_t cuda_backend = ggml_backend_cuda_init(device);
    if (!cuda_backend)
      throw std::runtime_error("cannot initialize CUDA backend");
    ggml_backend_t adapter = exl3_ggml_cuda_backend_wrap(cuda_backend, true);
    if (!adapter || !exl3_ggml_cuda_backend_set_graph_capture(adapter, capture))
      throw std::runtime_error("cannot initialize EXL3 GGML adapter");
    auto *ctx = ggml_init({ggml_tensor_overhead() * 64 +
                               ggml_graph_overhead_custom(64, false) + 4096,
                           nullptr, true});
    auto *input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, 1);
    auto *q_weight = new_dense_weight(ctx, model, *q_info);
    auto *k_weight = new_dense_weight(ctx, model, *k_info);
    auto *v_weight = new_dense_weight(ctx, model, *v_info);
    auto *o_weight = new_dense_weight(ctx, model, *o_info);
    auto *q_norm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, head_dim);
    auto *k_norm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, head_dim);
    auto qn_values = load_dense(model, *qn_info);
    auto kn_values = load_dense(model, *kn_info);
    ggml_backend_t backend = cuda_backend;
    auto *opaque = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, payload.size());
    auto *route_template = ggml_new_tensor_1d(
        ctx, GGML_TYPE_I8, kSelectedExperts * (sizeof(int32_t) + sizeof(float)));
    auto *route_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, kSelectedExperts);
    auto *route_weights = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kSelectedExperts);
    auto route_pack = std::make_unique<Exl3GgmlRoutePackState>();
    if (!route_pack->initialize(kSelectedExperts))
      throw std::runtime_error("cannot initialize layer route pack");
    auto *route_table = exl3_ggml_map_route_table(
        ctx, route_template, route_ids, route_weights, route_pack.get());
    auto state = std::make_unique<Exl3GgmlRoutedFfnState>();
    if (!state->initialize_dynamic(
            payload_layout.expert_stride, payload_layout.projection_offsets,
            kSelectedExperts, kSelectedExperts, payload_layout.hidden,
            payload_layout.intermediate, payload_layout.bits, 1, device))
      throw std::runtime_error("cannot initialize layer EXL3 FFN state");
    auto *ffn = exl3_ggml_map_dynamic_routed_ffn(ctx, input, opaque,
                                                 route_table, state.get());
    auto *positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    auto *mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 1, 1);
    auto *k_storage = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, head_dim, kv_heads);
    auto *v_storage = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, head_dim, kv_heads);
    auto *indices = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, kv_heads);
    if (!input || !q_weight || !k_weight || !v_weight || !o_weight || !q_norm ||
        !k_norm || !route_table || !ffn || !positions || !mask || !k_storage ||
        !v_storage || !indices)
      throw std::runtime_error("cannot allocate full-layer graph tensors");
    ggml_tensor *qfull = ggml_mul_mat(ctx, q_weight, input);
    ggml_tensor *q = ggml_view_3d(ctx, qfull, head_dim, heads, 1,
                                  sizeof(float) * head_dim * 2,
                                  sizeof(float) * head_dim * 2 * heads, 0);
    q = ggml_mul(ctx, ggml_rms_norm(ctx, q, 1e-6f), q_norm);
    ggml_tensor *k = ggml_reshape_3d(
        ctx, ggml_mul_mat(ctx, k_weight, input), head_dim, kv_heads, 1);
    k = ggml_mul(ctx, ggml_rms_norm(ctx, k, 1e-6f), k_norm);
    ggml_tensor *v = ggml_reshape_3d(
        ctx, ggml_mul_mat(ctx, v_weight, input), head_dim, kv_heads, 1);
    q = ggml_rope_ext(ctx, q, positions, nullptr, head_dim,
                      GGML_ROPE_TYPE_NEOX, 4096, 10000000.0f, 1.0f, 1.0f,
                      1.0f, 32.0f, 1.0f);
    k = ggml_rope_ext(ctx, k, positions, nullptr, head_dim,
                      GGML_ROPE_TYPE_NEOX, 4096, 10000000.0f, 1.0f, 1.0f,
                      1.0f, 32.0f, 1.0f);
    ggml_tensor *kset = ggml_set_rows(ctx, k_storage,
                                      ggml_reshape_2d(ctx, k, head_dim, kv_heads),
                                      indices);
    ggml_tensor *vset = ggml_set_rows(ctx, v_storage,
                                      ggml_reshape_2d(ctx, v, head_dim, kv_heads),
                                      indices);
    ggml_tensor *q_attn = ggml_permute(ctx, q, 0, 2, 1, 3);
    ggml_tensor *k_attn = ggml_cast(ctx, ggml_reshape_4d(ctx, kset, head_dim, 1,
                                                         kv_heads, 1),
                                    GGML_TYPE_F16);
    ggml_tensor *v_attn = ggml_cast(ctx, ggml_reshape_4d(ctx, vset, head_dim, 1,
                                                         kv_heads, 1),
                                    GGML_TYPE_F16);
    ggml_tensor *attn = ggml_flash_attn_ext(
        ctx, q_attn, k_attn, v_attn, mask,
        1.0f / std::sqrt(static_cast<float>(head_dim)), 0.0f, 0.0f);
    ggml_tensor *gate = ggml_view_3d(ctx, qfull, head_dim, heads, 1,
                                     sizeof(float) * head_dim * 2,
                                     sizeof(float) * head_dim * 2 * heads,
                                     sizeof(float) * head_dim);
    gate = ggml_reshape_3d(ctx, ggml_cont_2d(ctx, gate, heads * head_dim, 1),
                           head_dim, heads, 1);
    attn = ggml_mul(ctx, attn, ggml_sigmoid(ctx, gate));
    attn = ggml_reshape_2d(ctx, attn, heads * head_dim, 1);
    ggml_tensor *attn_out = ggml_mul_mat(ctx, o_weight, attn);
    // The bridge benchmark uses the attention output as the FFN input; the
    // production graph will insert Qwen HC/residual nodes at this boundary.
    ffn->src[0] = attn_out;
    auto *graph = ggml_new_graph_custom(ctx, 64, false);
    ggml_build_forward_expand(graph, ffn);
    auto *buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer)
      throw std::runtime_error("cannot allocate full-layer graph");
    std::vector<float> input_host(hidden);
    for (int index = 0; index < hidden; ++index)
      input_host[index] = 0.1f * std::sin(index * 0.071f);
    std::vector<int32_t> route_ids_host = {0, 1};
    std::vector<float> route_weights_host = {0.5f, 0.5f};
    ggml_backend_tensor_set(input, input_host.data(), 0,
                            input_host.size() * sizeof(float));
    ggml_backend_tensor_set(opaque, payload.data(), 0, payload.size());
    ggml_backend_tensor_set(route_ids, route_ids_host.data(), 0,
                            route_ids_host.size() * sizeof(int32_t));
    ggml_backend_tensor_set(route_weights, route_weights_host.data(), 0,
                            route_weights_host.size() * sizeof(float));
    ggml_backend_tensor_set(q_norm, qn_values.data(), 0,
                            qn_values.size() * sizeof(float));
    ggml_backend_tensor_set(k_norm, kn_values.data(), 0,
                            kn_values.size() * sizeof(float));
    std::vector<int32_t> zero_indices(kv_heads, 0);
    float zero_mask = 0.0f;
    ggml_backend_tensor_set(positions, &zero_indices[0], 0, sizeof(int32_t));
    ggml_backend_tensor_set(indices, zero_indices.data(), 0,
                            zero_indices.size() * sizeof(int32_t));
    ggml_backend_tensor_set(mask, &zero_mask, 0, sizeof(ggml_fp16_t));
    set_owned_tensor(backend, q_weight);
    set_owned_tensor(backend, k_weight);
    set_owned_tensor(backend, v_weight);
    set_owned_tensor(backend, o_weight);
    const auto graph_run = [&] {
      if (ggml_backend_graph_compute(adapter, graph) != GGML_STATUS_SUCCESS)
        throw std::runtime_error("full-layer graph execution failed");
    };
    graph_run();
    if (capture)
      graph_run();
    std::vector<float> output(hidden);
    ggml_backend_tensor_get(ffn, output.data(), 0,
                            output.size() * sizeof(float));
    for (float value : output)
      if (!std::isfinite(value))
        throw std::runtime_error("full-layer graph output is non-finite");
    const cudaStream_t stream = exl3_ggml_cuda_stream(cuda_backend);
    const double gpu_us = measure_gpu(graph_run, stream, 2, iterations);
    std::printf(
        "EXL3 GGML CUDA real layer layer=%d hidden=%d heads=%d kv_heads=%d "
        "capture=%d gpu_us=%.2f tok_s=%.2f finite=1\n",
        layer, hidden, heads, kv_heads, capture ? 1 : 0, gpu_us,
        1.0e6 / gpu_us);
    ggml_backend_buffer_free(buffer);
    ggml_backend_free(adapter);
    ggml_free(ctx);
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "exl3_ggml_cuda_real_layer_bench: %s\n", error.what());
    return 1;
  }
}
