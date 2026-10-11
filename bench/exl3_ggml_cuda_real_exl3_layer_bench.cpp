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

constexpr int kSelectedExperts = 2;

void check(cudaError_t status, const char *operation) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
}

const Exl3LinearInfo *find_attention(const Exl3Contract &contract, int layer,
                                     const char *projection) {
  const std::string suffix = ".layers." + std::to_string(layer) +
                             ".self_attn." + projection;
  for (const auto &linear : contract.linears)
    if (linear.valid && linear.codebook == Exl3Codebook::MUL1 &&
        linear.key.ends_with(suffix))
      return &linear;
  return nullptr;
}

const Exl3LinearInfo *find_expert(const Exl3Contract &contract, int layer,
                                  int expert, const char *projection) {
  const std::string suffix = ".layers." + std::to_string(layer) +
                             ".mlp.experts." + std::to_string(expert) + "." +
                             projection;
  for (const auto &linear : contract.linears)
    if (linear.valid && linear.codebook == Exl3Codebook::MUL1 &&
        linear.key.ends_with(suffix))
      return &linear;
  return nullptr;
}

const SafetensorsTensorInfo *find_raw(const SafetensorsModelInfo &model,
                                      int layer, const char *suffix) {
  const std::string needle = ".layers." + std::to_string(layer) + "." + suffix;
  for (const auto &[name, info] : model.tensors)
    if (name.find("mtp.") == std::string::npos && name.find(needle) != std::string::npos)
      return &info;
  return nullptr;
}

std::vector<float> read_dense(const SafetensorsModelInfo &model,
                              const SafetensorsTensorInfo &info) {
  std::vector<uint8_t> bytes;
  std::string error;
  if (!read_exl3_tensor(model, info.name, bytes, error))
    throw std::runtime_error(error);
  size_t elements = 1;
  for (int64_t dimension : info.shape)
    elements *= static_cast<size_t>(dimension);
  std::vector<float> values(elements);
  if (info.dtype == "F32")
    std::memcpy(values.data(), bytes.data(), elements * sizeof(float));
  else if (info.dtype == "F16") {
    const auto *source = reinterpret_cast<const ggml_fp16_t *>(bytes.data());
    for (size_t index = 0; index < elements; ++index)
      values[index] = ggml_fp16_to_fp32(source[index]);
  } else if (info.dtype == "BF16") {
    const auto *source = reinterpret_cast<const ggml_bf16_t *>(bytes.data());
    for (size_t index = 0; index < elements; ++index)
      values[index] = ggml_bf16_to_fp32(source[index]);
  } else
    throw std::runtime_error("unsupported tensor dtype: " + info.dtype);
  return values;
}

ggml_tensor *new_owned_vector(ggml_context *ctx,
                              const std::vector<float> &values) {
  auto *tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, values.size());
  if (tensor)
    tensor->extra = new std::vector<float>(values);
  return tensor;
}

ggml_tensor *new_owned_matrix(ggml_context *ctx,
                              const SafetensorsModelInfo &model,
                              const SafetensorsTensorInfo &info) {
  if (info.shape.size() != 2)
    return nullptr;
  auto values = read_dense(model, info);
  auto *tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, info.shape[1],
                                    info.shape[0]);
  if (tensor)
    tensor->extra = new std::vector<float>(std::move(values));
  return tensor;
}

void upload_owned_vector(ggml_tensor *tensor) {
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
  check(cudaStreamSynchronize(stream), "synchronize EXL3 layer warmup");
  cudaEvent_t begin = nullptr, end = nullptr;
  check(cudaEventCreate(&begin), "create EXL3 layer begin event");
  check(cudaEventCreate(&end), "create EXL3 layer end event");
  check(cudaEventRecord(begin, stream), "record EXL3 layer begin event");
  for (int index = 0; index < iterations; ++index)
    function();
  check(cudaEventRecord(end, stream), "record EXL3 layer end event");
  check(cudaEventSynchronize(end), "synchronize EXL3 layer end event");
  float milliseconds = 0.0f;
  check(cudaEventElapsedTime(&milliseconds, begin, end),
        "measure EXL3 layer events");
  cudaEventDestroy(begin);
  cudaEventDestroy(end);
  return static_cast<double>(milliseconds) * 1000.0 / iterations;
}

} // namespace

int main(int argc, char **argv) {
  try {
    const std::string model_path =
        argc > 1 ? argv[1] : "/data/sdb/models/qwen3.8-flash-next/exl3_3bpw";
    const int iterations = argc > 2 ? std::atoi(argv[2]) : 5;
    const int device = argc > 3 ? std::atoi(argv[3]) : 0;
    const bool capture = argc > 4 && std::string(argv[4]) == "--capture";
    const int rows = argc > 5 ? std::atoi(argv[5]) : 1;
    if (rows <= 0)
      throw std::runtime_error("rows must be positive");
    SafetensorsModelInfo model;
    if (!SafetensorsAdapter::parse_directory(model_path, model))
      throw std::runtime_error("cannot parse EXL3 model directory");
    const Exl3Contract contract = SafetensorsAdapter::map_exl3_contract(model);
    int layer = -1;
    const Exl3LinearInfo *attention[4] = {};
    const SafetensorsTensorInfo *q_norm_info = nullptr;
    const SafetensorsTensorInfo *k_norm_info = nullptr;
    for (int candidate = 0; candidate < 48 && layer < 0; ++candidate) {
      const Exl3LinearInfo *q = find_attention(contract, candidate, "q_proj");
      const Exl3LinearInfo *k = find_attention(contract, candidate, "k_proj");
      const Exl3LinearInfo *v = find_attention(contract, candidate, "v_proj");
      const Exl3LinearInfo *o = find_attention(contract, candidate, "o_proj");
      auto *qn = find_raw(model, candidate, "self_attn.q_norm.weight");
      auto *kn = find_raw(model, candidate, "self_attn.k_norm.weight");
      if (q && k && v && o && qn && kn) {
        layer = candidate;
        attention[0] = q;
        attention[1] = k;
        attention[2] = v;
        attention[3] = o;
        q_norm_info = qn;
        k_norm_info = kn;
      }
    }
    if (layer < 0)
      throw std::runtime_error("no EXL3 full-attention layer found");
    const auto *attn_hc_norm_info =
        find_raw(model, layer, "attn_hyper_connection.hc_norm.weight");
    const auto *attn_hc_down_info = find_raw(
        model, layer, "attn_hyper_connection.input_mix_weight_down.weight");
    const auto *attn_hc_up_info = find_raw(
        model, layer, "attn_hyper_connection.input_mix_weight_up.weight");
    const auto *attn_hc_inject_info = find_raw(
        model, layer, "attn_hyper_connection.block_inject_weight.weight");
    const auto *ffn_hc_norm_info =
        find_raw(model, layer, "mlp_hyper_connection.hc_norm.weight");
    const auto *ffn_hc_down_info = find_raw(
        model, layer, "mlp_hyper_connection.input_mix_weight_down.weight");
    const auto *ffn_hc_up_info = find_raw(
        model, layer, "mlp_hyper_connection.input_mix_weight_up.weight");
    const auto *ffn_hc_inject_info = find_raw(
        model, layer, "mlp_hyper_connection.block_inject_weight.weight");
    if (!attn_hc_norm_info || !attn_hc_down_info || !attn_hc_up_info ||
        !attn_hc_inject_info || !ffn_hc_norm_info || !ffn_hc_down_info ||
        !ffn_hc_up_info || !ffn_hc_inject_info)
      throw std::runtime_error("Qwen HC tensors are incomplete");

    std::vector<std::vector<uint8_t>> attention_bytes(4);
    Exl3PackedMatrixHeader attention_headers[4] = {};
    for (int index = 0; index < 4; ++index) {
      std::string error;
      if (!load_exl3_mul1_matrix(model, *attention[index],
                                 attention_bytes[index], error))
        throw std::runtime_error(error);
      std::memcpy(&attention_headers[index], attention_bytes[index].data(),
                  sizeof(Exl3PackedMatrixHeader));
    }
    const int hidden = attention_headers[0].input_features;
    const int q_out = attention_headers[0].output_features;
    const int k_out = attention_headers[1].output_features;
    const int head_dim = static_cast<int>(q_norm_info->shape[0]);
    const int heads = q_out / (2 * head_dim);
    const int kv_heads = k_out / head_dim;
    if (heads <= 0 || kv_heads <= 0 || heads % kv_heads != 0)
      throw std::runtime_error("invalid EXL3 full-attention geometry");
    const int hc = static_cast<int>(attn_hc_norm_info->shape[0]) / hidden;
    if (hc <= 0 || attn_hc_down_info->shape.size() != 2 ||
        ffn_hc_down_info->shape.size() != 2)
      throw std::runtime_error("invalid Qwen HC geometry");
    const int intermediate = contract.intermediate > 0 ? contract.intermediate : 640;

    std::vector<std::vector<uint8_t>> expert_matrices(kSelectedExperts * 3);
    const int expert_ids[kSelectedExperts] = {0, 1};
    for (int slot = 0; slot < kSelectedExperts; ++slot) {
      for (int projection = 0; projection < 3; ++projection) {
        const char *name[] = {"gate_proj", "up_proj", "down_proj"};
        const auto *linear = find_expert(contract, layer, expert_ids[slot],
                                         name[projection]);
        if (!linear)
          throw std::runtime_error("missing EXL3 attention-layer expert");
        std::string error;
        if (!load_exl3_mul1_matrix(model, *linear,
                                   expert_matrices[slot * 3 + projection], error))
          throw std::runtime_error(error);
      }
    }
    Exl3GgmlRoutedPayloadLayout payload_layout;
    std::vector<uint8_t> payload;
    std::string payload_error;
    if (!exl3_ggml_build_routed_payload(expert_matrices, payload,
                                        payload_layout, payload_error))
      throw std::runtime_error(payload_error);

    ggml_backend_t cuda_backend = ggml_backend_cuda_init(device);
    if (!cuda_backend)
      throw std::runtime_error("cannot initialize CUDA backend");
    ggml_backend_t adapter = exl3_ggml_cuda_backend_wrap(cuda_backend, true);
    if (!adapter || !exl3_ggml_cuda_backend_set_graph_capture(adapter, capture))
      throw std::runtime_error("cannot initialize EXL3 layer adapter");
    auto *ctx = ggml_init({ggml_tensor_overhead() * 256 +
                               ggml_graph_overhead_custom(128, false) + 16384,
                           nullptr, true});
    auto *input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hidden, hc, rows);
    auto *input_placeholder = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, rows);
    ggml_tensor *opaque[4] = {};
    Exl3GgmlMatmulState *matmul_state[4] = {
        new Exl3GgmlMatmulState, new Exl3GgmlMatmulState,
        new Exl3GgmlMatmulState, new Exl3GgmlMatmulState};
    ggml_tensor *matmul_nodes[4] = {};
    for (int index = 0; index < 4; ++index) {
      opaque[index] = ggml_new_tensor_1d(ctx, GGML_TYPE_I8,
                                         attention_bytes[index].size());
      if (!matmul_state[index]->initialize(attention_headers[index], rows, device))
        throw std::runtime_error("cannot initialize EXL3 attention matmul");
      matmul_nodes[index] = exl3_ggml_map_matmul(
          ctx, input_placeholder, opaque[index], matmul_state[index]);
    }
    auto q_norm = new_owned_vector(ctx, read_dense(model, *q_norm_info));
    auto k_norm = new_owned_vector(ctx, read_dense(model, *k_norm_info));
    auto attn_hc_norm =
        new_owned_vector(ctx, read_dense(model, *attn_hc_norm_info));
    auto attn_hc_down = new_owned_matrix(ctx, model, *attn_hc_down_info);
    auto attn_hc_up = new_owned_matrix(ctx, model, *attn_hc_up_info);
    auto attn_hc_inject = new_owned_matrix(ctx, model, *attn_hc_inject_info);
    auto ffn_hc_norm =
        new_owned_vector(ctx, read_dense(model, *ffn_hc_norm_info));
    auto ffn_hc_down = new_owned_matrix(ctx, model, *ffn_hc_down_info);
    auto ffn_hc_up = new_owned_matrix(ctx, model, *ffn_hc_up_info);
    auto ffn_hc_inject = new_owned_matrix(ctx, model, *ffn_hc_inject_info);
    auto build_hc_mix = [&](ggml_tensor *stream, ggml_tensor *norm,
                            ggml_tensor *down, ggml_tensor *up,
                            ggml_tensor *inject, ggml_tensor **inject_out) {
      auto *normalized = ggml_mul(
          ctx, ggml_reshape_2d(ctx,
                               ggml_rms_norm(ctx, stream, 1e-6f), hidden * hc,
                               rows),
          norm);
      auto *low = ggml_silu(ctx, ggml_scale(
                                 ctx, ggml_mul_mat(ctx, down, normalized),
                                 1.0f / static_cast<float>(hc)));
      auto *gate = ggml_sigmoid(ctx, ggml_mul_mat(ctx, up, low));
      auto *gated = ggml_reshape_2d(ctx, ggml_mul(ctx, normalized, gate),
                                    hidden, hc * rows);
      ggml_tensor *mixed = ggml_view_2d(
          ctx, gated, hidden, rows,
          static_cast<size_t>(hidden) * hc * sizeof(float), 0);
      for (int stream_index = 1; stream_index < hc; ++stream_index)
        mixed = ggml_add(ctx, mixed,
                         ggml_view_2d(ctx, gated, hidden, rows,
                                      static_cast<size_t>(hidden) * hc *
                                          sizeof(float),
                                      static_cast<size_t>(stream_index) *
                                          hidden * sizeof(float)));
      mixed = ggml_scale(ctx, mixed, 1.0f / static_cast<float>(hc));
      if (inject_out)
        *inject_out = ggml_mul_mat(ctx, inject, normalized);
      return mixed;
    };
    ggml_tensor *attn_inject_value = nullptr;
    auto *attn_input = build_hc_mix(input, attn_hc_norm, attn_hc_down,
                                    attn_hc_up, attn_hc_inject,
                                    &attn_inject_value);
    for (auto *node : matmul_nodes)
      node->src[0] = attn_input;
    auto *route_template = ggml_new_tensor_1d(
        ctx, GGML_TYPE_I8,
        rows * kSelectedExperts * (sizeof(int32_t) + sizeof(float)));
    auto *expert_opaque = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, payload.size());
    auto *route_ids =
        ggml_new_tensor_1d(ctx, GGML_TYPE_I32, kSelectedExperts * rows);
    auto *route_weights =
        ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kSelectedExperts * rows);
    auto route_pack = std::make_unique<Exl3GgmlRoutePackState>();
    route_pack->initialize(kSelectedExperts, rows);
    auto *route_table = exl3_ggml_map_route_table(
        ctx, route_template, route_ids, route_weights, route_pack.get());
    auto state = std::make_unique<Exl3GgmlRoutedFfnState>();
    if (!state->initialize_dynamic(
            payload_layout.expert_stride, payload_layout.projection_offsets,
            kSelectedExperts, kSelectedExperts, payload_layout.hidden,
            payload_layout.intermediate, payload_layout.bits, rows, device))
      throw std::runtime_error("cannot initialize EXL3 layer FFN");
    auto *ffn = exl3_ggml_map_dynamic_routed_ffn(
        ctx, input_placeholder, expert_opaque,
                                                 route_table, state.get());
    auto *positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, rows);
    auto *mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, rows, rows);
    auto *k_storage = rows == 1
                          ? ggml_new_tensor_2d(ctx, GGML_TYPE_F32, head_dim,
                                               kv_heads)
                          : nullptr;
    auto *v_storage = rows == 1
                          ? ggml_new_tensor_2d(ctx, GGML_TYPE_F32, head_dim,
                                               kv_heads)
                          : nullptr;
    auto *indices = rows == 1 ? ggml_new_tensor_1d(ctx, GGML_TYPE_I32, kv_heads)
                              : nullptr;
    auto *qfull = matmul_nodes[0];
    auto *q = ggml_view_3d(ctx, qfull, head_dim, heads, rows,
                           sizeof(float) * head_dim * 2,
                           sizeof(float) * head_dim * 2 * heads, 0);
    q = ggml_mul(ctx, ggml_rms_norm(ctx, q, 1e-6f), q_norm);
    auto *k = ggml_reshape_3d(ctx, matmul_nodes[1], head_dim, kv_heads, rows);
    k = ggml_mul(ctx, ggml_rms_norm(ctx, k, 1e-6f), k_norm);
    auto *v = ggml_reshape_3d(ctx, matmul_nodes[2], head_dim, kv_heads, rows);
    q = ggml_rope_ext(ctx, q, positions, nullptr, head_dim,
                      GGML_ROPE_TYPE_NEOX, 4096, 10000000.0f, 1.0f, 1.0f,
                      1.0f, 32.0f, 1.0f);
    k = ggml_rope_ext(ctx, k, positions, nullptr, head_dim,
                      GGML_ROPE_TYPE_NEOX, 4096, 10000000.0f, 1.0f, 1.0f,
                      1.0f, 32.0f, 1.0f);
    auto *kset = rows == 1
                     ? ggml_set_rows(ctx, k_storage,
                                     ggml_reshape_2d(ctx, k, head_dim, kv_heads),
                                     indices)
                     : k;
    auto *vset = rows == 1
                     ? ggml_set_rows(ctx, v_storage,
                                     ggml_reshape_2d(ctx, v, head_dim, kv_heads),
                                     indices)
                     : v;
    auto *q_attn = ggml_permute(ctx, q, 0, 2, 1, 3);
    auto *k_attn = ggml_cast(
        ctx, rows == 1 ? ggml_reshape_4d(ctx, kset, head_dim, 1, kv_heads, 1)
                       : ggml_reshape_4d(ctx, kset, head_dim, rows, kv_heads, 1),
        GGML_TYPE_F16);
    auto *v_attn = ggml_cast(
        ctx, rows == 1 ? ggml_reshape_4d(ctx, vset, head_dim, 1, kv_heads, 1)
                       : ggml_reshape_4d(ctx, vset, head_dim, rows, kv_heads, 1),
        GGML_TYPE_F16);
    auto *attn = ggml_flash_attn_ext(
        ctx, q_attn, k_attn, v_attn, mask,
        1.0f / std::sqrt(static_cast<float>(head_dim)), 0.0f, 0.0f);
    auto *gate = ggml_view_3d(ctx, qfull, head_dim, heads, rows,
                              sizeof(float) * head_dim * 2,
                              sizeof(float) * head_dim * 2 * heads,
                              sizeof(float) * head_dim);
    gate = ggml_reshape_3d(ctx,
                           ggml_cont_2d(ctx, gate, heads * head_dim, rows),
                           head_dim, heads, rows);
    attn = ggml_mul(ctx, attn, ggml_sigmoid(ctx, gate));
    attn = ggml_reshape_2d(ctx, attn, heads * head_dim, rows);
    auto *o_node = exl3_ggml_map_matmul(ctx, attn, opaque[3], matmul_state[3]);
    auto *attn_weight = ggml_reshape_3d(
        ctx, ggml_scale(ctx, ggml_sigmoid(ctx, ggml_scale(
                                      ctx, attn_inject_value,
                                      1.0f / static_cast<float>(hc))),
                                  2.0f),
        1, hc, rows);
    auto *attn_block = ggml_repeat_4d(
        ctx, ggml_reshape_3d(ctx, o_node, hidden, 1, rows), hidden, hc, rows,
        1);
    auto *post_attention =
        ggml_add(ctx, input, ggml_mul(ctx, attn_block, attn_weight));
    ggml_tensor *ffn_inject_value = nullptr;
    auto *ffn_input = build_hc_mix(post_attention, ffn_hc_norm, ffn_hc_down,
                                   ffn_hc_up, ffn_hc_inject,
                                   &ffn_inject_value);
    ffn->src[0] = ffn_input;
    auto *ffn_weight = ggml_reshape_3d(
        ctx, ggml_scale(ctx, ggml_sigmoid(ctx, ggml_scale(
                                      ctx, ffn_inject_value,
                                      1.0f / static_cast<float>(hc))),
                                  2.0f),
        1, hc, rows);
    auto *ffn_block = ggml_repeat_4d(
        ctx, ggml_reshape_3d(ctx, ffn, hidden, 1, rows), hidden, hc, rows, 1);
    auto *final = ggml_add(ctx, post_attention,
                           ggml_mul(ctx, ffn_block, ffn_weight));
    if (!input || !o_node || !ffn || !route_table)
      throw std::runtime_error("cannot build EXL3 attention-layer graph");
    auto *graph = ggml_new_graph_custom(ctx, 256, false);
    ggml_build_forward_expand(graph, final);
    auto *buffer = ggml_backend_alloc_ctx_tensors(ctx, cuda_backend);
    if (!buffer)
      throw std::runtime_error("cannot allocate EXL3 attention-layer graph");
    std::vector<float> input_host(static_cast<size_t>(hidden) * hc * rows);
    for (int row = 0; row < rows; ++row)
      for (int stream_index = 0; stream_index < hc; ++stream_index)
        for (int index = 0; index < hidden; ++index)
          input_host[(static_cast<size_t>(row) * hc + stream_index) * hidden +
                    index] = 0.1f * std::sin((index + row) * 0.071f);
    ggml_backend_tensor_set(input, input_host.data(), 0,
                            input_host.size() * sizeof(float));
    for (int index = 0; index < 4; ++index)
      ggml_backend_tensor_set(opaque[index], attention_bytes[index].data(), 0,
                              attention_bytes[index].size());
    ggml_backend_tensor_set(expert_opaque, payload.data(), 0, payload.size());
    std::vector<int32_t> route_ids_host(static_cast<size_t>(rows) *
                                        kSelectedExperts);
    std::vector<float> route_weights_host(route_ids_host.size(), 0.5f);
    for (int row = 0; row < rows; ++row) {
      route_ids_host[static_cast<size_t>(row) * kSelectedExperts] = 0;
      route_ids_host[static_cast<size_t>(row) * kSelectedExperts + 1] = 1;
    }
    ggml_backend_tensor_set(route_ids, route_ids_host.data(), 0,
                            route_ids_host.size() * sizeof(int32_t));
    ggml_backend_tensor_set(route_weights, route_weights_host.data(), 0,
                            route_weights_host.size() * sizeof(float));
    upload_owned_vector(q_norm);
    upload_owned_vector(k_norm);
    upload_owned_vector(attn_hc_norm);
    upload_owned_vector(attn_hc_down);
    upload_owned_vector(attn_hc_up);
    upload_owned_vector(attn_hc_inject);
    upload_owned_vector(ffn_hc_norm);
    upload_owned_vector(ffn_hc_down);
    upload_owned_vector(ffn_hc_up);
    upload_owned_vector(ffn_hc_inject);
    std::vector<int32_t> position_values(rows);
    std::iota(position_values.begin(), position_values.end(), 0);
    ggml_backend_tensor_set(positions, position_values.data(), 0,
                            position_values.size() * sizeof(int32_t));
    if (indices) {
      std::vector<int32_t> cache_indices(kv_heads);
      for (int head = 0; head < kv_heads; ++head)
        cache_indices[head] = head * rows;
      ggml_backend_tensor_set(indices, cache_indices.data(), 0,
                              cache_indices.size() * sizeof(int32_t));
    }
    std::vector<ggml_fp16_t> mask_values(static_cast<size_t>(rows) * rows);
    for (int query = 0; query < rows; ++query)
      for (int key = 0; key < rows; ++key)
        mask_values[static_cast<size_t>(query) * rows + key] =
            ggml_fp32_to_fp16(key <= query ? 0.0f : -INFINITY);
    ggml_backend_tensor_set(mask, mask_values.data(), 0,
                            mask_values.size() * sizeof(ggml_fp16_t));
    const auto graph_run = [&] {
      if (ggml_backend_graph_compute(adapter, graph) != GGML_STATUS_SUCCESS)
        throw std::runtime_error("EXL3 attention-layer graph failed");
    };
    graph_run();
    if (capture)
      graph_run();
    std::vector<float> output(hidden);
    output.resize(static_cast<size_t>(hidden) * hc * rows);
    ggml_backend_tensor_get(final, output.data(), 0,
                            output.size() * sizeof(float));
    for (float value : output)
      if (!std::isfinite(value))
        throw std::runtime_error("EXL3 attention-layer output is non-finite");
    const cudaStream_t stream = exl3_ggml_cuda_stream(cuda_backend);
    const double gpu_us = measure_gpu(graph_run, stream, 2, iterations);
    std::printf(
        "EXL3 GGML CUDA real EXL3 layer layer=%d hidden=%d heads=%d "
        "kv_heads=%d capture=%d gpu_us=%.2f tok_s=%.2f finite=1\n",
        layer, hidden, heads, kv_heads, capture ? 1 : 0, gpu_us,
        1.0e6 / gpu_us);
    ggml_backend_buffer_free(buffer);
    ggml_backend_free(adapter);
    ggml_free(ctx);
    for (auto *state_ptr : matmul_state)
      delete state_ptr;
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "exl3_ggml_cuda_real_exl3_layer_bench: %s\n",
                 error.what());
    return 1;
  }
}
