#include "exl3_cpu.hpp"
#include "exl3_format.hpp"
#include "exl3_ggml_backend.hpp"
#include "exl3_ggml_ffn.hpp"
#include "exl3_gpu.hpp"
#include "exl3_model_loader.hpp"
#include "safetensors_adapter.hpp"

#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

void check(cudaError_t status, const char *operation) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
}

struct ExternalExpertStorage {
  std::vector<uint8_t *> matrices;

  ~ExternalExpertStorage() {
    for (uint8_t *matrix : matrices)
      if (matrix)
        cudaFree(matrix);
  }
};

const SafetensorsTensorInfo *find_router(const SafetensorsModelInfo &model) {
  for (const auto &[name, tensor] : model.tensors)
    if (name.find("mtp.") == std::string::npos &&
        name.find("layers.0.mlp.gate.weight") != std::string::npos &&
        name.find("gate_proj") == std::string::npos)
      return &tensor;
  return nullptr;
}

std::vector<float> decode_router(const SafetensorsTensorInfo &info,
                                 const std::vector<uint8_t> &bytes) {
  const size_t count = static_cast<size_t>(info.shape[0] * info.shape[1]);
  std::vector<float> values(count);
  if (info.dtype == "F32") {
    std::memcpy(values.data(), bytes.data(), count * sizeof(float));
  } else if (info.dtype == "F16") {
    const auto *source = reinterpret_cast<const ggml_fp16_t *>(bytes.data());
    for (size_t index = 0; index < count; ++index)
      values[index] = ggml_fp16_to_fp32(source[index]);
  } else if (info.dtype == "BF16") {
    const auto *source = reinterpret_cast<const ggml_bf16_t *>(bytes.data());
    for (size_t index = 0; index < count; ++index)
      values[index] = ggml_bf16_to_fp32(source[index]);
  } else {
    throw std::runtime_error("unsupported router dtype: " + info.dtype);
  }
  return values;
}

std::vector<int32_t> select_topk(const float *logits, int experts, int topk,
                                 float *weights) {
  std::vector<int32_t> ids(static_cast<size_t>(experts));
  for (int expert = 0; expert < experts; ++expert)
    ids[static_cast<size_t>(expert)] = expert;
  std::partial_sort(ids.begin(), ids.begin() + topk, ids.end(),
                    [&](int left, int right) {
                      return logits[left] > logits[right];
                    });
  float maximum = -INFINITY;
  for (int route = 0; route < topk; ++route)
    maximum = std::max(maximum, logits[ids[static_cast<size_t>(route)]]);
  float sum = 0.0f;
  for (int route = 0; route < topk; ++route) {
    weights[route] =
        std::exp(logits[ids[static_cast<size_t>(route)]] - maximum);
    sum += weights[route];
  }
  for (int route = 0; route < topk; ++route)
    weights[route] /= sum;
  ids.resize(static_cast<size_t>(topk));
  return ids;
}

const Exl3LinearInfo &find_linear(
    const std::unordered_map<std::string, const Exl3LinearInfo *> &linears,
    const std::string &key) {
  const auto found = linears.find(key);
  if (found == linears.end())
    throw std::runtime_error("missing EXL3 matrix: " + key);
  return *found->second;
}

} // namespace

int main(int argc, char **argv) {
  try {
    const char *model_env = std::getenv("ZEROMOE_EXL3_MODEL");
    const std::string model_path =
        argc > 1 ? argv[1] : (model_env ? model_env : "");
    if (model_path.empty())
      throw std::runtime_error("pass <model_dir> or set ZEROMOE_EXL3_MODEL");
    const int iterations = argc > 2 ? std::max(1, std::atoi(argv[2])) : 8;
    const int device = argc > 3 ? std::atoi(argv[3]) : 0;
    const int rows = argc > 4 ? std::max(1, std::atoi(argv[4])) : 2;
    const bool external = argc > 5 && std::string(argv[5]) == "--external";
    const int topk = argc > 6 ? std::atoi(argv[6]) : 2;
    const bool graph_capture = argc > 7 && std::string(argv[7]) == "--graph";
    const bool repeat_route_set =
        std::getenv("ZEROMOE_EXL3_GRAPH_REPEAT_ROUTE") != nullptr;
    int previous_device = 0;
    const cudaError_t current_device_status = cudaGetDevice(&previous_device);
    if (current_device_status != cudaSuccess)
      throw std::runtime_error(std::string("cannot get current CUDA device: ") +
                               cudaGetErrorString(current_device_status));
    int device_count = 0;
    const cudaError_t device_count_status = cudaGetDeviceCount(&device_count);
    if (device_count_status != cudaSuccess)
      throw std::runtime_error(std::string("cannot enumerate CUDA devices: ") +
                               cudaGetErrorString(device_count_status));
    if (device < 0 || device >= device_count)
      throw std::runtime_error("requested CUDA device is unavailable");
    const cudaError_t set_device_status = cudaSetDevice(device);
    if (set_device_status != cudaSuccess)
      throw std::runtime_error(std::string("cannot select CUDA device: ") +
                               cudaGetErrorString(set_device_status));

    SafetensorsModelInfo model;
    if (!SafetensorsAdapter::parse_directory(model_path, model))
      throw std::runtime_error("cannot parse EXL3 model directory");
    const Exl3Contract contract = SafetensorsAdapter::map_exl3_contract(model);
    const auto *router_info = find_router(model);
    if (!contract.complete || !router_info || router_info->shape.size() != 2)
      throw std::runtime_error("incomplete Qwen EXL3 model contract");
    std::vector<uint8_t> router_bytes;
    std::string error;
    if (!read_exl3_tensor(model, router_info->name, router_bytes, error))
      throw std::runtime_error(error);
    const int experts = static_cast<int>(router_info->shape[0]);
    const int hidden = static_cast<int>(router_info->shape[1]);
    if (topk <= 0 || topk > experts)
      throw std::runtime_error("topk must be in [1, expert_count]");
    const std::vector<float> router = decode_router(*router_info, router_bytes);
    std::unordered_map<std::string, const Exl3LinearInfo *> linears;
    for (const auto &linear : contract.linears)
      if (linear.valid && linear.codebook == Exl3Codebook::MUL1)
        linears.emplace(linear.key, &linear);
    const std::string first_expert_prefix =
        "model.language_model.layers.0.mlp.experts.0.";
    const auto &first_gate =
        find_linear(linears, first_expert_prefix + "gate_proj");
    const auto &first_up =
        find_linear(linears, first_expert_prefix + "up_proj");
    const auto &first_down =
        find_linear(linears, first_expert_prefix + "down_proj");
    const int intermediate = static_cast<int>(first_gate.out_features);
    if (first_gate.in_features != hidden ||
        first_up.in_features != hidden || first_up.out_features != intermediate ||
        first_down.in_features != intermediate ||
        first_down.out_features != hidden)
      throw std::runtime_error("layer-0 EXL3 expert dimensions do not match router");

    std::vector<float> input(static_cast<size_t>(rows) * hidden);
    std::vector<int32_t> route_ids(static_cast<size_t>(rows) * topk);
    std::vector<float> route_weights(static_cast<size_t>(rows) * topk);
    std::vector<float> output(static_cast<size_t>(rows) * hidden);
    for (int row = 0; row < rows; ++row) {
      float *input_row = input.data() + static_cast<size_t>(row) * hidden;
      for (int index = 0; index < hidden; ++index)
        input_row[index] = 0.1f * std::sin((index + row * 29) * 0.071f);
      std::vector<float> logits(static_cast<size_t>(experts), 0.0f);
      for (int expert = 0; expert < experts; ++expert)
        for (int index = 0; index < hidden; ++index)
          logits[static_cast<size_t>(expert)] +=
              router[static_cast<size_t>(expert) * hidden + index] *
              input_row[index];
      float *weights_row = route_weights.data() +
                           static_cast<size_t>(row) * topk;
      const auto selected =
          select_topk(logits.data(), experts, topk, weights_row);
      std::copy(selected.begin(), selected.end(),
                route_ids.begin() + static_cast<size_t>(row) * topk);
    }

    const auto load_safetensors = [&](const std::string &key) {
      std::vector<uint8_t> packed;
      std::string load_error;
      if (!load_exl3_mul1_matrix(model, find_linear(linears, key), packed,
                                 load_error))
        throw std::runtime_error(load_error);
      return packed;
    };
    constexpr size_t ddr_cache_budget = 512ull * 1024 * 1024;
    Exl3CpuMatrixCache ddr_cache(ddr_cache_budget);
    const auto load = [&](const std::string &key) {
      auto matrix = ddr_cache.get(key);
      if (!matrix) {
        Exl3Mul1MatrixData loaded;
        std::string load_error;
        if (!load_exl3_mul1_components(model, find_linear(linears, key),
                                      loaded, load_error))
          throw std::runtime_error(load_error);
        matrix = ddr_cache.insert(key, std::move(loaded));
      }
      return exl3_mul1_pack_matrix(matrix->view());
    };
    const auto compute_reference = [&](const std::vector<int32_t> &ids,
                                       const std::vector<float> &weights) {
      std::vector<float> result(static_cast<size_t>(rows) * hidden, 0.0f);
      for (int row = 0; row < rows; ++row) {
        const float *input_row =
            input.data() + static_cast<size_t>(row) * hidden;
        for (int slot = 0; slot < topk; ++slot) {
          const size_t route = static_cast<size_t>(row) * topk + slot;
          const std::string prefix =
              "model.language_model.layers.0.mlp.experts." +
              std::to_string(ids[route]) + ".";
          const std::vector<uint8_t> gate =
              load_safetensors(prefix + "gate_proj");
          const std::vector<uint8_t> up = load_safetensors(prefix + "up_proj");
          const std::vector<uint8_t> down =
              load_safetensors(prefix + "down_proj");
          std::vector<float> gate_value(intermediate), up_value(intermediate);
          std::vector<float> down_input(intermediate), expert_output(hidden);
          if (!exl3_mul1_matmul_packed(gate.data(), input_row,
                                      gate_value.data(), 1, hidden,
                                      intermediate)) {
            Exl3PackedMatrixHeader header{};
            std::memcpy(&header, gate.data(), sizeof(header));
            std::fprintf(stderr,
                         "gate reference failed expert=%d expected=%dx%d packed=%ux%u bits=%u size=%llu\n",
                         ids[route], hidden, intermediate, header.input_features,
                         header.output_features, header.bits,
                         static_cast<unsigned long long>(header.total_size));
            throw std::runtime_error("CPU EXL3 gate reference failed");
          }
          if (!exl3_mul1_matmul_packed(up.data(), input_row, up_value.data(),
                                       1, hidden, intermediate)) {
            Exl3PackedMatrixHeader header{};
            std::memcpy(&header, up.data(), sizeof(header));
            std::fprintf(stderr,
                         "up reference failed expert=%d expected=%dx%d packed=%ux%u bits=%u size=%llu\n",
                         ids[route], hidden, intermediate, header.input_features,
                         header.output_features, header.bits,
                         static_cast<unsigned long long>(header.total_size));
            throw std::runtime_error("CPU EXL3 up reference failed");
          }
          for (int index = 0; index < intermediate; ++index)
            down_input[index] = gate_value[index] /
                                (1.0f + std::exp(-gate_value[index])) *
                                up_value[index];
          if (!exl3_mul1_matmul_packed(down.data(), down_input.data(),
                                       expert_output.data(), 1, intermediate,
                                       hidden))
            throw std::runtime_error("CPU EXL3 reference down failed");
          const float route_weight = weights[route];
          for (int index = 0; index < hidden; ++index)
            result[static_cast<size_t>(row) * hidden + index] +=
                route_weight * expert_output[index];
        }
      }
      return result;
    };
    const std::vector<float> reference =
        compute_reference(route_ids, route_weights);
    std::vector<int32_t> alternate_route_ids = route_ids;
    if (!repeat_route_set)
      for (int32_t &expert : alternate_route_ids)
        expert = (expert + 1) % experts;
    const std::vector<float> alternate_reference =
        compute_reference(alternate_route_ids, route_weights);

    std::vector<Exl3GpuExternalExpert> external_experts(
        static_cast<size_t>(experts));
    std::vector<std::shared_ptr<ExternalExpertStorage>> external_storage(
        static_cast<size_t>(experts));
    if (external) {
      std::unordered_map<int32_t, bool> selected_experts;
      for (int32_t expert : route_ids)
        selected_experts.emplace(expert, true);
      for (int32_t expert : alternate_route_ids)
        selected_experts.emplace(expert, true);
      const char *projection_names[] = {"gate_proj", "up_proj", "down_proj"};
      for (const auto &[expert_id, unused] : selected_experts) {
        (void)unused;
        const std::string prefix =
            "model.language_model.layers.0.mlp.experts." +
            std::to_string(expert_id) + ".";
        auto storage = std::make_shared<ExternalExpertStorage>();
        Exl3GpuExternalExpert view{};
        Exl3CudaMatrixView *views[] = {&view.gate, &view.up, &view.down};
        for (int projection = 0; projection < 3; ++projection) {
          const std::string key = prefix + projection_names[projection];
          const auto packed = load_safetensors(key);
          Exl3PackedMatrixHeader header{};
          if (packed.size() < sizeof(header))
            throw std::runtime_error("truncated external EXL3 matrix");
          std::memcpy(&header, packed.data(), sizeof(header));
          if (!exl3_valid_packed_header(header, packed.size()))
            throw std::runtime_error("invalid external EXL3 matrix header");
          uint8_t *device_packed = nullptr;
          check(cudaMalloc(&device_packed, packed.size()),
                "allocate external EXL3 matrix");
          storage->matrices.push_back(device_packed);
          check(cudaMemcpy(device_packed, packed.data(), packed.size(),
                           cudaMemcpyHostToDevice),
                "upload external EXL3 matrix");
          *views[projection] = {
              reinterpret_cast<const uint16_t *>(device_packed +
                                                 header.trellis_offset),
              reinterpret_cast<const uint16_t *>(device_packed +
                                                 header.suh_offset),
              reinterpret_cast<const uint16_t *>(device_packed +
                                                 header.svh_offset),
              (header.flags & kExl3PackedHasBias)
                  ? reinterpret_cast<const uint16_t *>(device_packed +
                                                       header.bias_offset)
                  : nullptr,
              static_cast<int>(header.input_features),
              static_cast<int>(header.output_features),
              static_cast<int>(header.bits)};
        }
        view.hold = storage;
        external_experts[static_cast<size_t>(expert_id)] = view;
        external_storage[static_cast<size_t>(expert_id)] = std::move(storage);
      }
    }

    Exl3GpuExecutor executor(device, 8ull * 1024 * 1024 * 1024);
    executor.set_routed_graph_capture(graph_capture);
    std::vector<std::string> prefixes(static_cast<size_t>(experts));
    for (int expert = 0; expert < experts; ++expert)
      prefixes[static_cast<size_t>(expert)] =
          "model.language_model.layers.0.mlp.experts." +
          std::to_string(expert) + ".";

    ggml_backend_t cuda_backend = ggml_backend_cuda_init(device);
    if (!cuda_backend)
      throw std::runtime_error("cannot initialize GGML CUDA backend");
    ggml_backend_t adapter = exl3_ggml_cuda_backend_wrap(cuda_backend, false);
    ggml_backend_t cpu_backend = ggml_backend_cpu_init();
    if (!adapter || !cpu_backend)
      throw std::runtime_error("cannot create custom-op graph backends");
    ggml_backend_t backends[] = {adapter, cpu_backend};
    ggml_backend_sched_t scheduler =
        ggml_backend_sched_new(backends, nullptr, 2, 256, false, false);
    ggml_init_params params{};
    params.mem_size = 2 * 1024 * 1024;
    params.no_alloc = true;
    ggml_context *ctx = ggml_init(params);
    if (!scheduler || !ctx)
      throw std::runtime_error("cannot allocate custom-op graph context");
    ggml_tensor *input_tensor =
        ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, rows);
    ggml_tensor *ids_tensor =
        ggml_new_tensor_2d(ctx, GGML_TYPE_I32, topk, rows);
    ggml_tensor *weights_tensor =
        ggml_new_tensor_2d(ctx, GGML_TYPE_F32, topk, rows);
    auto state = std::make_unique<Exl3GgmlCachedRoutedFfnState>();
    if (!input_tensor || !ids_tensor || !weights_tensor ||
        !state->initialize(&executor, prefixes, rows, topk, hidden,
                           intermediate))
      throw std::runtime_error("cannot initialize cache graph state");
    state->set_matrix_loader(load);
    if (external) {
      state->set_external_expert_resolver(
          [&external_experts](
              const std::string &, int expert_id, int, int,
              const std::function<std::vector<uint8_t>(const std::string &)> &)
              -> Exl3GpuExternalExpert {
            if (expert_id < 0 ||
                static_cast<size_t>(expert_id) >= external_experts.size() ||
                !external_experts[static_cast<size_t>(expert_id)].gate.trellis)
              throw std::runtime_error("external route expert was not staged");
            return external_experts[static_cast<size_t>(expert_id)];
          });
    }
    ggml_backend_sched_set_tensor_backend(scheduler, input_tensor, adapter);
    ggml_backend_sched_set_tensor_backend(scheduler, ids_tensor, adapter);
    ggml_backend_sched_set_tensor_backend(scheduler, weights_tensor, adapter);
    ggml_tensor *output_tensor = exl3_ggml_map_cached_routed_ffn_device_routes(
        ctx, input_tensor, ids_tensor, weights_tensor, state.get());
    if (!output_tensor)
      throw std::runtime_error("cannot create cache-backed GGML node");
    ggml_backend_sched_set_tensor_backend(scheduler, output_tensor, adapter);
    ggml_cgraph *graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output_tensor);
    if (!ggml_backend_sched_alloc_graph(scheduler, graph))
      throw std::runtime_error("cannot allocate cache-backed GGML graph");

    const auto run = [&](const std::vector<int32_t> &ids,
                         const std::vector<float> &weights,
                         std::vector<float> &result) {
      ggml_backend_tensor_set(input_tensor, input.data(), 0,
                              input.size() * sizeof(float));
      ggml_backend_tensor_set(ids_tensor, ids.data(), 0,
                              ids.size() * sizeof(int32_t));
      ggml_backend_tensor_set(weights_tensor, weights.data(), 0,
                              weights.size() * sizeof(float));
      if (ggml_backend_sched_graph_compute_async(scheduler, graph) !=
          GGML_STATUS_SUCCESS)
        throw std::runtime_error("cache-backed GGML graph execution failed");
      ggml_backend_sched_synchronize(scheduler);
      ggml_backend_tensor_get(output_tensor, result.data(), 0,
                              result.size() * sizeof(float));
    };
    const bool verify_external_parity =
        external && !std::getenv("ZEROMOE_EXL3_SKIP_INTERNAL_PARITY");
    const bool cpu_reference_output =
        std::getenv("ZEROMOE_EXL3_GGML_CPU_REFERENCE_OUTPUT") != nullptr;
    state->verify_cpu_reference = verify_external_parity || cpu_reference_output;
    state->model_layer = 0;
    run(route_ids, route_weights, output);
    run(alternate_route_ids, route_weights, output);
    state->verify_cpu_reference = cpu_reference_output;
    state->route_readback_calls = 0;
    state->route_readback_bytes = 0;
    state->route_readback_ms = 0.0;
    state->expert_resolution_ms = 0.0;
    state->ffn_enqueue_ms = 0.0;
    state->row_batch_wait_ms = 0.0;
    const cudaStream_t stream = exl3_ggml_cuda_stream(cuda_backend);
    if (!stream)
      throw std::runtime_error("cannot get cache graph CUDA stream");
    cudaEvent_t gpu_begin = nullptr;
    cudaEvent_t gpu_end = nullptr;
    check(cudaEventCreate(&gpu_begin), "create cache graph begin event");
    check(cudaEventCreate(&gpu_end), "create cache graph end event");
    check(cudaEventRecord(gpu_begin, stream), "record cache graph begin event");
    const auto begin = std::chrono::steady_clock::now();
    for (int iteration = 0; iteration < iterations; ++iteration) {
      const bool alternate = iteration % 2 != 0;
      run(alternate ? alternate_route_ids : route_ids, route_weights, output);
    }
    const auto end = std::chrono::steady_clock::now();
    check(cudaEventRecord(gpu_end, stream), "record cache graph end event");
    check(cudaEventSynchronize(gpu_end), "synchronize cache graph end event");
    float gpu_elapsed_ms = 0.0f;
    check(cudaEventElapsedTime(&gpu_elapsed_ms, gpu_begin, gpu_end),
          "measure cache graph GPU time");
    cudaEventDestroy(gpu_begin);
    cudaEventDestroy(gpu_end);
    const double elapsed_us =
        std::chrono::duration<double, std::micro>(
            end - begin)
            .count() /
        iterations;
    const double gpu_us =
        static_cast<double>(gpu_elapsed_ms) * 1000.0 / iterations;
    float max_error = 0.0f;
    const std::vector<float> &expected =
        iterations % 2 == 0 ? alternate_reference : reference;
    for (size_t index = 0; index < output.size(); ++index) {
      if (!std::isfinite(output[index]))
        throw std::runtime_error("cache-backed output is non-finite");
      max_error = std::max(max_error, std::abs(output[index] - expected[index]));
    }
    const Exl3GpuStats stats = executor.stats();
    std::unordered_map<int32_t, bool> active_experts;
    for (int32_t expert : route_ids)
      active_experts.emplace(expert, true);
    for (int32_t expert : alternate_route_ids)
      active_experts.emplace(expert, true);
    if ((!external && stats.uploads != active_experts.size() * 3) ||
        (external && stats.uploads != 0))
      throw std::runtime_error(
          "unexpected EXL3 projection uploads in routed-cache mode");
    const Exl3CpuMatrixCacheStats ddr_stats = ddr_cache.stats();
    const uint64_t expected_ddr_misses =
        external && !verify_external_parity ? 0 : active_experts.size() * 3;
    if (ddr_stats.misses != expected_ddr_misses ||
        ddr_stats.resident_bytes > ddr_cache_budget)
      throw std::runtime_error(
          "unexpected DDR EXL3 cache activity in routed-cache mode");
    std::printf("EXL3 GGML %s FFN rows=%d topk=%d route_sets=2 avg_us=%.2f "
                "gpu_us=%.2f max_abs_error=%g routed_ffns=%llu "
                "uploads=%llu hits=%llu DDR_misses=%llu DDR_hits=%llu "
                "graph_captures=%llu graph_replays=%llu graph_fallbacks=%llu "
                "resident_GiB=%.3f\n",
                external ? "external-route" : "cached-route", rows, topk,
                elapsed_us, gpu_us, max_error,
                static_cast<unsigned long long>(stats.routed_ffns),
                static_cast<unsigned long long>(stats.uploads),
                static_cast<unsigned long long>(stats.cache_hits),
                static_cast<unsigned long long>(ddr_stats.misses),
                static_cast<unsigned long long>(ddr_stats.hits),
                static_cast<unsigned long long>(stats.routed_graph_captures),
                static_cast<unsigned long long>(stats.routed_graph_replays),
                static_cast<unsigned long long>(
                    stats.routed_graph_eager_fallbacks),
                static_cast<double>(stats.resident_bytes) /
                    (1024.0 * 1024.0 * 1024.0));
    ggml_backend_sched_free(scheduler);
    ggml_backend_free(cpu_backend);
    ggml_backend_free(adapter);
    ggml_backend_free(cuda_backend);
    ggml_free(ctx);
    return max_error <= 1e-3f ? 0 : 2;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "exl3_ggml_cuda_cache_graph_bench: %s\n",
                 error.what());
    return 1;
  }
}
