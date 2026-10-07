#include "exl3_ggml_model.hpp"

#if defined(MOESTREAM_USE_GGML_CUDA)

#include <cuda_runtime.h>
#include "ggml-cuda.h"
#include "ggml-cpu.h"
#include "exl3_gpu.hpp"
#include "ggml_moe_backend.hpp"

#include <memory>
#include <algorithm>
#include <climits>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

namespace {

struct Exl3ModelIndex {
  SafetensorsModelInfo model;
  Exl3Contract contract;
};

std::shared_ptr<const Exl3ModelIndex> model_index(const std::string &path) {
  static std::mutex mutex;
  static std::unordered_map<std::string, std::shared_ptr<const Exl3ModelIndex>>
      indexes;
  std::lock_guard<std::mutex> lock(mutex);
  auto found = indexes.find(path);
  if (found != indexes.end())
    return found->second;
  auto index = std::make_shared<Exl3ModelIndex>();
  if (!SafetensorsAdapter::parse_directory(path, index->model))
    return {};
  index->contract = SafetensorsAdapter::map_exl3_contract(index->model);
  if (!index->contract.complete)
    return {};
  indexes.emplace(path, index);
  return index;
}

} // namespace

Exl3GgmlBridgeLayer::~Exl3GgmlBridgeLayer() {
  ffn_state.reset();
  route_pack.reset();
}

bool Exl3GgmlBridgeLayer::initialize(
    const std::string &model_path, ggml_context *ctx,
    ggml_backend_t cuda_backend, Exl3GpuExecutor *executor, int model_layer,
    int n_experts, int model_topk, int model_rows, int cuda_device,
    bool load_cache_misses_from_safetensors) {
  if (model_path.empty() || !ctx || !cuda_backend || !executor || model_layer < 0 ||
      n_experts <= 0 || model_topk <= 0 || model_topk > n_experts ||
      model_rows <= 0 || cuda_device < 0)
    return false;
  const auto index = model_index(model_path);
  if (!index)
    return false;
  const Exl3Contract &contract = index->contract;
  if (contract.n_experts != n_experts)
    return false;

  const auto find_mul1 = [&](const std::string &key) -> const Exl3LinearInfo * {
    const auto found = std::find_if(
        contract.linears.begin(), contract.linears.end(),
        [&](const Exl3LinearInfo &linear) {
          return linear.valid && linear.codebook == Exl3Codebook::MUL1 &&
                 linear.key == key;
        });
    return found == contract.linears.end() ? nullptr : &*found;
  };
  const std::string layer_prefix =
      "model.language_model.layers." + std::to_string(model_layer) +
      ".mlp.experts.";
  const std::string first_prefix = layer_prefix + "0.";
  const Exl3LinearInfo *first_gate = find_mul1(first_prefix + "gate_proj");
  const Exl3LinearInfo *first_up = find_mul1(first_prefix + "up_proj");
  const Exl3LinearInfo *first_down = find_mul1(first_prefix + "down_proj");
  if (!first_gate || !first_up || !first_down ||
      first_gate->in_features <= 0 || first_gate->out_features <= 0 ||
      first_gate->in_features > INT_MAX || first_gate->out_features > INT_MAX)
    return false;
  const int hidden = static_cast<int>(first_gate->in_features);
  const int intermediate = static_cast<int>(first_gate->out_features);
  if (hidden % 128 || intermediate % 128 ||
      first_up->in_features != hidden || first_up->out_features != intermediate ||
      first_down->in_features != intermediate || first_down->out_features != hidden)
    return false;

  std::vector<std::string> expert_prefixes(static_cast<size_t>(n_experts));
  const char *projections[] = {"gate_proj", "up_proj", "down_proj"};
  for (int expert = 0; expert < n_experts; ++expert) {
    expert_prefixes[static_cast<size_t>(expert)] =
        layer_prefix + std::to_string(expert) + ".";
    const int expected_inputs[] = {hidden, hidden, intermediate};
    const int expected_outputs[] = {intermediate, intermediate, hidden};
    for (int projection = 0; projection < 3; ++projection) {
      const Exl3LinearInfo *linear = find_mul1(
          expert_prefixes[static_cast<size_t>(expert)] +
          projections[projection]);
      if (!linear || linear->in_features != expected_inputs[projection] ||
          linear->out_features != expected_outputs[projection])
        return false;
    }
  }

  route_template = ggml_new_tensor_1d(
      ctx, GGML_TYPE_I8,
      static_cast<int64_t>(model_topk) * model_rows *
          (sizeof(int32_t) + sizeof(float)));
  if (!route_template)
    return false;

  ffn_state = std::make_unique<Exl3GgmlCachedRoutedFfnState>();
  if (!ffn_state->initialize(executor, std::move(expert_prefixes), model_rows,
                             model_topk, hidden, intermediate))
    return false;
  if (load_cache_misses_from_safetensors) {
    ffn_state->set_matrix_loader(
        [index](const std::string &key) -> std::vector<uint8_t> {
          const auto found = std::find_if(
              index->contract.linears.begin(), index->contract.linears.end(),
              [&](const Exl3LinearInfo &linear) {
                return linear.valid && linear.codebook == Exl3Codebook::MUL1 &&
                       linear.key == key;
              });
          if (found == index->contract.linears.end())
            throw std::runtime_error("Missing EXL3 safetensors matrix: " + key);
          std::vector<uint8_t> packed;
          std::string error;
          if (!load_exl3_mul1_matrix(index->model, *found, packed, error))
            throw std::runtime_error(error);
          return packed;
        });
  }
  route_pack = std::make_unique<Exl3GgmlRoutePackState>();
  if (!route_pack->initialize(model_topk, model_rows))
    return false;
  gpu_executor = executor;
  layer = model_layer;
  device = cuda_device;
  rows = model_rows;
  experts = n_experts;
  topk = model_topk;
  return true;
}

bool Exl3GgmlBridgeLayer::set_layer(int model_layer) {
  if (model_layer < 0 || !ffn_state ||
      static_cast<int>(ffn_state->expert_prefixes_by_id.size()) != experts)
    return false;
  for (int expert = 0; expert < experts; ++expert) {
    ffn_state->expert_prefixes_by_id[static_cast<size_t>(expert)] =
        "model.language_model.layers." + std::to_string(model_layer) +
        ".mlp.experts." + std::to_string(expert) + ".";
  }
  layer = model_layer;
  return true;
}

ggml_tensor *Exl3GgmlBridgeLayer::build(ggml_context *ctx, ggml_tensor *input,
                                         ggml_tensor *route_ids,
                                         ggml_tensor *route_weights,
                                         ggml_backend_sched_t scheduler,
                                         ggml_backend_t cuda_backend) {
  if (!ctx || !input || !route_ids || !route_weights || !ffn_state ||
      !route_pack || !route_template || !scheduler || !cuda_backend ||
      input->type != GGML_TYPE_F32 || input->ne[0] != ffn_state->hidden ||
      input->ne[1] != rows || route_ids->type != GGML_TYPE_I32 ||
      route_ids->ne[0] != topk || route_ids->ne[1] != rows ||
      route_weights->type != GGML_TYPE_F32 || route_weights->ne[0] != topk ||
      route_weights->ne[1] != rows)
    return nullptr;
  ggml_backend_sched_set_tensor_backend(scheduler, route_template,
                                        cuda_backend);
  ggml_tensor *ids = ggml_cont(ctx, route_ids);
  ggml_tensor *weights = ggml_cont(ctx, route_weights);
  ggml_tensor *table = exl3_ggml_map_route_table(
      ctx, route_template, ids, weights, route_pack.get());
  if (!table)
    return nullptr;
  ggml_backend_sched_set_tensor_backend(scheduler, table, cuda_backend);
  ggml_tensor *output = exl3_ggml_map_cached_routed_ffn(
      ctx, input, table, ffn_state.get());
  if (output)
    ggml_backend_sched_set_tensor_backend(scheduler, output, cuda_backend);
  return output;
}

Exl3GgmlRoutedFfnGraph::~Exl3GgmlRoutedFfnGraph() { reset(); }

void Exl3GgmlRoutedFfnGraph::reset() {
  if (scheduler) {
    ggml_backend_sched_synchronize(scheduler);
    ggml_backend_sched_free(scheduler);
    scheduler = nullptr;
  }
  if (cpu_backend) {
    ggml_backend_free(cpu_backend);
    cpu_backend = nullptr;
  }
  bridge.reset();
  if (context) {
    ggml_free(context);
    context = nullptr;
  }
  if (cuda_adapter) {
    ggml_backend_free(cuda_adapter);
    cuda_adapter = nullptr;
  }
  graph = nullptr;
  input = nullptr;
  route_ids = nullptr;
  route_weights = nullptr;
  output = nullptr;
  cuda_raw = nullptr;
  experts = topk = rows = hidden = intermediate = 0;
}

bool Exl3GgmlRoutedFfnGraph::initialize(
    const std::string &model_path, Exl3GpuExecutor *executor, int n_experts,
    int model_topk, int model_rows, int model_hidden,
    int model_intermediate, int device) {
  reset();
  if (model_path.empty() || !executor || n_experts <= 0 || model_topk <= 0 ||
      model_topk > n_experts || model_rows <= 0 || model_hidden <= 0 ||
      model_intermediate <= 0 || device < 0)
    return false;
  cuda_raw = static_cast<ggml_backend_t>(ggml_shared_backend(device));
  if (!cuda_raw)
    return false;
  cuda_adapter = exl3_ggml_cuda_backend_wrap(cuda_raw, false);
  if (!cuda_adapter)
    return false;
  cpu_backend = ggml_backend_cpu_init();
  if (!cpu_backend)
    return false;
  ggml_backend_t backends[] = {cuda_adapter, cpu_backend};
  scheduler = ggml_backend_sched_new(backends, nullptr, 2, 256, false, false);
  if (!scheduler)
    return false;

  ggml_init_params params{};
  params.mem_size = 2 * 1024 * 1024;
  params.no_alloc = true;
  context = ggml_init(params);
  if (!context)
    return false;
  input = ggml_new_tensor_2d(context, GGML_TYPE_F32, model_hidden,
                             model_rows);
  route_ids = ggml_new_tensor_2d(context, GGML_TYPE_I32, model_topk,
                                 model_rows);
  route_weights = ggml_new_tensor_2d(context, GGML_TYPE_F32, model_topk,
                                     model_rows);
  graph = ggml_new_graph(context);
  bridge = std::make_unique<Exl3GgmlBridgeLayer>();
  if (!input || !route_ids || !route_weights || !graph ||
      !bridge->initialize(model_path, context, cuda_adapter, executor, 0,
                          n_experts, model_topk, model_rows, device) ||
      bridge->ffn_state->hidden != model_hidden ||
      bridge->ffn_state->intermediate != model_intermediate)
    return false;
  ggml_backend_sched_set_tensor_backend(scheduler, input, cuda_adapter);
  ggml_backend_sched_set_tensor_backend(scheduler, route_ids, cuda_adapter);
  ggml_backend_sched_set_tensor_backend(scheduler, route_weights,
                                        cuda_adapter);
  output = exl3_ggml_map_cached_routed_ffn_host_routes(
      context, input, route_ids, route_weights, bridge->ffn_state.get());
  if (!output)
    return false;
  ggml_backend_sched_set_tensor_backend(scheduler, output, cuda_adapter);
  ggml_build_forward_expand(graph, output);
  if (!ggml_backend_sched_alloc_graph(scheduler, graph))
    return false;
  experts = n_experts;
  topk = model_topk;
  rows = model_rows;
  hidden = model_hidden;
  intermediate = model_intermediate;
  return true;
}

bool Exl3GgmlRoutedFfnGraph::run(
    int model_layer, const int32_t *host_route_ids,
    const float *host_route_weights, const float *host_input,
    float *host_output,
    const std::function<std::vector<uint8_t>(const std::string &)> &load) {
  if (!scheduler || !bridge || !bridge->ffn_state || !input || !route_ids ||
      !route_weights || !output || !host_route_ids || !host_route_weights ||
      !host_input || !host_output || !load || !bridge->set_layer(model_layer))
    return false;
  const size_t route_count = static_cast<size_t>(rows) * topk;
  if (!bridge->ffn_state->set_host_routes(host_route_ids,
                                          host_route_weights))
    return false;
  bridge->ffn_state->set_matrix_loader(load);

  const size_t input_bytes =
      static_cast<size_t>(rows) * hidden * sizeof(float);
  const size_t ids_bytes = route_count * sizeof(int32_t);
  const size_t weights_bytes = route_count * sizeof(float);
  ggml_backend_tensor_set(input, host_input, 0, input_bytes);
  ggml_backend_tensor_set(route_ids, host_route_ids, 0, ids_bytes);
  ggml_backend_tensor_set(route_weights, host_route_weights, 0, weights_bytes);
  if (ggml_backend_sched_graph_compute_async(scheduler, graph) !=
      GGML_STATUS_SUCCESS) {
    ggml_backend_sched_synchronize(scheduler);
    return false;
  }
  ggml_backend_sched_synchronize(scheduler);
  ggml_backend_tensor_get(output, host_output, 0, input_bytes);
  return true;
}

Exl3GgmlDenseMatmulGraph::~Exl3GgmlDenseMatmulGraph() { reset(); }

void Exl3GgmlDenseMatmulGraph::reset() {
  if (scheduler) {
    ggml_backend_sched_synchronize(scheduler);
    ggml_backend_sched_free(scheduler);
    scheduler = nullptr;
  }
  if (cpu_backend) {
    ggml_backend_free(cpu_backend);
    cpu_backend = nullptr;
  }
  if (context) {
    ggml_free(context);
    context = nullptr;
  }
  if (cuda_adapter) {
    ggml_backend_free(cuda_adapter);
    cuda_adapter = nullptr;
  }
  cuda_raw = nullptr;
  graph = nullptr;
  input = nullptr;
  output = nullptr;
  executor = nullptr;
  rows = input_features = output_features = 0;
  state = {};
}

bool Exl3GgmlDenseMatmulGraph::initialize(
    Exl3GpuExecutor *matrix_executor, int matrix_rows,
    int matrix_input_features, int matrix_output_features) {
  reset();
  if (!matrix_executor || matrix_rows <= 0 || matrix_input_features <= 0 ||
      matrix_output_features <= 0)
    return false;
  executor = matrix_executor;
  rows = matrix_rows;
  input_features = matrix_input_features;
  output_features = matrix_output_features;
  cuda_raw = static_cast<ggml_backend_t>(
      ggml_shared_backend(matrix_executor->device()));
  if (!cuda_raw)
    return false;
  cuda_adapter = exl3_ggml_cuda_backend_wrap(cuda_raw, false);
  if (!cuda_adapter)
    return false;
  cpu_backend = ggml_backend_cpu_init();
  if (!cpu_backend)
    return false;
  ggml_backend_t backends[] = {cuda_adapter, cpu_backend};
  scheduler = ggml_backend_sched_new(backends, nullptr, 2, 128, false, false);
  if (!scheduler)
    return false;
  ggml_init_params params{};
  params.mem_size = 1024 * 1024;
  params.no_alloc = true;
  context = ggml_init(params);
  if (!context)
    return false;
  input = ggml_new_tensor_2d(context, GGML_TYPE_F32, input_features, rows);
  graph = ggml_new_graph(context);
  if (!input || !graph ||
      !state.initialize(matrix_executor, rows, input_features,
                        output_features))
    return false;
  ggml_backend_sched_set_tensor_backend(scheduler, input, cuda_adapter);
  output = exl3_ggml_map_cached_matmul(context, input, &state);
  if (!output)
    return false;
  ggml_backend_sched_set_tensor_backend(scheduler, output, cuda_adapter);
  ggml_build_forward_expand(graph, output);
  return ggml_backend_sched_alloc_graph(scheduler, graph);
}

bool Exl3GgmlDenseMatmulGraph::run(const std::string &key, uint32_t type,
                                  const uint8_t *weights, size_t weight_bytes,
                                  const float *host_input,
                                  float *host_output) {
  if (!executor || !scheduler || !input || !output || key.empty() ||
      !weights || weight_bytes == 0 || !host_input || !host_output)
    return false;
  state.key = key;
  try {
    executor->prepare_matrix_cache(key, type, weights, weight_bytes,
                                   input_features, output_features, true);
  } catch (const std::runtime_error &) {
    return false;
  }
  const size_t input_bytes =
      static_cast<size_t>(rows) * input_features * sizeof(float);
  const size_t output_bytes =
      static_cast<size_t>(rows) * output_features * sizeof(float);
  ggml_backend_tensor_set(input, host_input, 0, input_bytes);
  if (ggml_backend_sched_graph_compute_async(scheduler, graph) !=
      GGML_STATUS_SUCCESS) {
    ggml_backend_sched_synchronize(scheduler);
    return false;
  }
  ggml_backend_sched_synchronize(scheduler);
  ggml_backend_tensor_get(output, host_output, 0, output_bytes);
  return true;
}

Exl3GgmlCpuRoutedFfnGraph::~Exl3GgmlCpuRoutedFfnGraph() { reset(); }

void Exl3GgmlCpuRoutedFfnGraph::reset() {
  if (scheduler) {
    ggml_backend_sched_synchronize(scheduler);
    ggml_backend_sched_free(scheduler);
    scheduler = nullptr;
  }
  if (cpu_backend) {
    ggml_backend_free(cpu_backend);
    cpu_backend = nullptr;
  }
  if (context) {
    ggml_free(context);
    context = nullptr;
  }
  graph = nullptr;
  input = route_ids = route_weights = output = nullptr;
  dispatch = nullptr;
  dispatch_context = nullptr;
  rows = topk = hidden = 0;
  layer_count = 0;
  layer = -1;
}

bool Exl3GgmlCpuRoutedFfnGraph::initialize(
    RowDispatch row_dispatch, void *row_dispatch_context, int model_layer_count,
    int model_rows, int model_topk, int model_hidden) {
  reset();
  if (!row_dispatch || !row_dispatch_context || model_layer_count <= 0 ||
      model_rows <= 0 || model_topk <= 0 || model_hidden <= 0)
    return false;
  dispatch = row_dispatch;
  dispatch_context = row_dispatch_context;
  layer_count = model_layer_count;
  rows = model_rows;
  topk = model_topk;
  hidden = model_hidden;
  cpu_backend = ggml_backend_cpu_init();
  if (!cpu_backend)
    return false;
  ggml_backend_t backends[] = {cpu_backend};
  scheduler = ggml_backend_sched_new(backends, nullptr, 1, 128, false, false);
  if (!scheduler)
    return false;
  ggml_init_params params{};
  params.mem_size = 1024 * 1024;
  params.no_alloc = true;
  context = ggml_init(params);
  if (!context)
    return false;
  input = ggml_new_tensor_2d(context, GGML_TYPE_F32, hidden, rows);
  route_ids = ggml_new_tensor_2d(context, GGML_TYPE_I32, topk, rows);
  route_weights = ggml_new_tensor_2d(context, GGML_TYPE_F32, topk, rows);
  graph = ggml_new_graph(context);
  if (!input || !route_ids || !route_weights || !graph)
    return false;
  ggml_backend_sched_set_tensor_backend(scheduler, input, cpu_backend);
  ggml_backend_sched_set_tensor_backend(scheduler, route_ids, cpu_backend);
  ggml_backend_sched_set_tensor_backend(scheduler, route_weights, cpu_backend);
  output = ggml_map_custom3(context, input, route_ids, route_weights, compute,
                            1, this);
  if (!output)
    return false;
  output->ne[0] = hidden;
  output->ne[1] = rows;
  output->nb[1] = static_cast<size_t>(hidden) * sizeof(float);
  ggml_backend_sched_set_tensor_backend(scheduler, output, cpu_backend);
  ggml_build_forward_expand(graph, output);
  return ggml_backend_sched_alloc_graph(scheduler, graph);
}

void Exl3GgmlCpuRoutedFfnGraph::compute(
    ggml_tensor *node, const ggml_tensor *input_tensor,
    const ggml_tensor *route_ids_tensor,
    const ggml_tensor *route_weights_tensor, int ith, int,
    void *user_data) {
  if (ith != 0)
    return;
  auto *state = static_cast<Exl3GgmlCpuRoutedFfnGraph *>(user_data);
  if (!state || !state->dispatch || !state->dispatch_context || !node || !input_tensor ||
      !route_ids_tensor || !route_weights_tensor || !node->data ||
      !input_tensor->data || !route_ids_tensor->data ||
      !route_weights_tensor->data)
    throw std::runtime_error("invalid EXL3 CPU GGML routed FFN node");
  const auto *ids = static_cast<const int32_t *>(route_ids_tensor->data);
  const auto *weights = static_cast<const float *>(route_weights_tensor->data);
  const auto *source = static_cast<const float *>(input_tensor->data);
  auto *destination = static_cast<float *>(node->data);
  for (int row = 0; row < state->rows; ++row) {
    if (!state->dispatch(
            state->dispatch_context, state->layer,
            ids + static_cast<size_t>(row) * state->topk,
            weights + static_cast<size_t>(row) * state->topk, state->topk,
            source + static_cast<size_t>(row) * state->hidden,
            destination + static_cast<size_t>(row) * state->hidden))
      throw std::runtime_error("EXL3 CPU GGML routed FFN dispatch failed");
  }
}

bool Exl3GgmlCpuRoutedFfnGraph::run(
    int layer, const int32_t *host_route_ids, const float *host_route_weights,
    const float *host_input, float *host_output) {
  if (!dispatch || !dispatch_context || !scheduler || !input || !route_ids || !route_weights ||
      !output || !host_route_ids || !host_route_weights || !host_input ||
      !host_output || layer < 0 || layer >= layer_count)
    return false;
  this->layer = layer;
  const size_t input_bytes = static_cast<size_t>(rows) * hidden * sizeof(float);
  const size_t route_count = static_cast<size_t>(rows) * topk;
  ggml_backend_tensor_set(input, host_input, 0, input_bytes);
  ggml_backend_tensor_set(route_ids, host_route_ids, 0,
                          route_count * sizeof(int32_t));
  ggml_backend_tensor_set(route_weights, host_route_weights, 0,
                          route_count * sizeof(float));
  if (ggml_backend_sched_graph_compute_async(scheduler, graph) !=
      GGML_STATUS_SUCCESS) {
    ggml_backend_sched_synchronize(scheduler);
    return false;
  }
  ggml_backend_sched_synchronize(scheduler);
  ggml_backend_tensor_get(output, host_output, 0, input_bytes);
  return true;
}

#endif
