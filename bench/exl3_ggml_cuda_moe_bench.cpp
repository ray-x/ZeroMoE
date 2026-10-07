#include "exl3_cuda.hpp"
#include "exl3_format.hpp"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-cuda.h"
#include "ggml-impl.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

extern "C" cudaStream_t moestream_ggml_backend_cuda_get_stream(
    ggml_backend_t backend);

namespace {

constexpr int kHidden = 2560;
constexpr int kIntermediate = 640;
constexpr int kBits = 4;

void check(cudaError_t status, const char *operation) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

size_t align64(size_t value) { return (value + 63) & ~size_t(63); }

std::vector<uint8_t> make_matrix(int input, int output, uint32_t seed) {
  const size_t trellis = static_cast<size_t>(input) * output / 8 * kBits;
  const size_t suh = static_cast<size_t>(input) * sizeof(uint16_t);
  const size_t svh = static_cast<size_t>(output) * sizeof(uint16_t);
  Exl3PackedMatrixHeader header{};
  header.magic = kExl3PackedMagic;
  header.version = 1;
  header.input_features = input;
  header.output_features = output;
  header.bits = kBits;
  header.trellis_offset = align64(sizeof(header));
  header.suh_offset = align64(header.trellis_offset + trellis);
  header.svh_offset = align64(header.suh_offset + suh);
  header.total_size = align64(header.svh_offset + svh);
  std::vector<uint8_t> packed(header.total_size);
  std::memcpy(packed.data(), &header, sizeof(header));
  std::mt19937 rng(seed);
  for (size_t index = 0; index < trellis; ++index)
    packed[header.trellis_offset + index] = static_cast<uint8_t>(rng());
  for (int index = 0; index < input; ++index)
    reinterpret_cast<uint16_t *>(packed.data() + header.suh_offset)[index] = 0x3c00;
  for (int index = 0; index < output; ++index)
    reinterpret_cast<uint16_t *>(packed.data() + header.svh_offset)[index] = 0x3c00;
  return packed;
}

struct LayerPayload {
  std::vector<uint8_t> bytes;
  size_t offsets[3] = {};
  Exl3PackedMatrixHeader headers[3] = {};
};

LayerPayload make_layer_payload(int layer, int experts) {
  LayerPayload payload;
  const int inputs[3] = {kHidden, kHidden, kIntermediate};
  const int outputs[3] = {kIntermediate, kIntermediate, kHidden};
  for (int expert = 0; expert < experts; ++expert) {
    for (int projection = 0; projection < 3; ++projection) {
      const auto matrix = make_matrix(inputs[projection], outputs[projection],
                                      2026100600u + layer * 100u + expert * 3u + projection);
      const size_t base = payload.bytes.size();
      payload.bytes.resize(align64(base + matrix.size()));
      std::memcpy(payload.bytes.data() + base, matrix.data(), matrix.size());
      if (expert == 0) {
        payload.offsets[projection] = base;
        std::memcpy(&payload.headers[projection], matrix.data(),
                    sizeof(Exl3PackedMatrixHeader));
      }
    }
  }
  return payload;
}

struct MoeState {
  cudaStream_t stream = nullptr;
  int experts = 0;
  bool capture = true;
  size_t offsets[3] = {};
  Exl3PackedMatrixHeader headers[3] = {};
  float *activation = nullptr;
  float *partials = nullptr;
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t executable = nullptr;

  ~MoeState() {
    if (executable) cudaGraphExecDestroy(executable);
    if (graph) cudaGraphDestroy(graph);
    if (partials) cudaFree(partials);
    if (activation) cudaFree(activation);
  }

  Exl3CudaMatrixView view(const uint8_t *base, int projection) const {
    const auto &header = headers[projection];
    return {
        reinterpret_cast<const uint16_t *>(base + offsets[projection] + header.trellis_offset),
        reinterpret_cast<const uint16_t *>(base + offsets[projection] + header.suh_offset),
        reinterpret_cast<const uint16_t *>(base + offsets[projection] + header.svh_offset),
        nullptr, static_cast<int>(header.input_features),
        static_cast<int>(header.output_features), static_cast<int>(header.bits)};
  }

  void enqueue(const ggml_tensor *input, const ggml_tensor *opaque,
               ggml_tensor *output) {
    const auto *base = static_cast<const uint8_t *>(opaque->data);
    const size_t gate_partials = static_cast<size_t>(kIntermediate) *
                                 exl3_cuda_key_splits(kHidden);
    const size_t down_partials = static_cast<size_t>(kHidden) *
                                 exl3_cuda_key_splits(kIntermediate);
    check(cudaMemsetAsync(output->data, 0, kHidden * sizeof(float), stream),
          "clear MoE output");
    for (int expert = 0; expert < experts; ++expert) {
      check(launch_exl3_mul1_gate_up(
                view(base, 0), view(base, 1), static_cast<const float *>(input->data),
                activation, 1, partials, partials + gate_partials, stream),
            "launch MoE gate/up");
      check(launch_exl3_mul1_matmul_accumulate(
                view(base, 2), activation, static_cast<float *>(output->data),
                1.0f / static_cast<float>(experts), 1, partials, stream),
            "launch MoE down");
    }
    (void)down_partials;
  }

  void run(const ggml_tensor *input, const ggml_tensor *opaque,
           ggml_tensor *output) {
    if (!capture) {
      enqueue(input, opaque, output);
      return;
    }
    if (!executable) {
      check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
            "begin MoE graph capture");
      enqueue(input, opaque, output);
      check(cudaStreamEndCapture(stream, &graph), "end MoE graph capture");
      check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0),
            "instantiate MoE graph");
    }
    check(cudaGraphLaunch(executable, stream), "launch MoE graph");
  }
};

struct Adapter {
  ggml_backend backend{};
  ggml_backend_t inner = nullptr;
};

const char *adapter_name(ggml_backend_t) { return "EXL3-CUDA-MOE-ADAPTER"; }

void adapter_free(ggml_backend_t backend) {
  auto *adapter = static_cast<Adapter *>(backend->context);
  ggml_backend_free(adapter->inner);
  delete adapter;
}

void adapter_synchronize(ggml_backend_t backend) {
  auto *adapter = static_cast<Adapter *>(backend->context);
  ggml_backend_synchronize(adapter->inner);
}

enum ggml_status adapter_graph_compute(ggml_backend_t, ggml_cgraph *graph) {
  for (int index = 0; index < graph->n_nodes; ++index) {
    ggml_tensor *node = graph->nodes[index];
    if (node->op != GGML_OP_MAP_CUSTOM2 || !node->extra)
      return GGML_STATUS_FAILED;
    auto *state = static_cast<MoeState *>(node->extra);
    state->run(node->src[0], node->src[1], node);
  }
  return GGML_STATUS_SUCCESS;
}

ggml_backend_t make_adapter(ggml_backend_t inner) {
  auto *adapter = new Adapter;
  adapter->inner = inner;
  adapter->backend.guid = ggml_backend_guid(inner);
  adapter->backend.device = ggml_backend_get_device(inner);
  adapter->backend.context = adapter;
  adapter->backend.iface = {
      adapter_name, adapter_free, nullptr, nullptr, nullptr, nullptr, nullptr,
      adapter_synchronize, nullptr, nullptr, nullptr, nullptr,
      adapter_graph_compute, nullptr, nullptr, nullptr};
  return &adapter->backend;
}

void custom_placeholder(ggml_tensor *, const ggml_tensor *, const ggml_tensor *,
                        int, int, void *) {}

template <typename Function>
double measure_enqueue(Function &&function, cudaStream_t stream, int warmup,
                       int iterations) {
  for (int index = 0; index < warmup; ++index) function();
  check(cudaStreamSynchronize(stream), "synchronize MoE warmup");
  const auto start = std::chrono::steady_clock::now();
  for (int index = 0; index < iterations; ++index) function();
  check(cudaStreamSynchronize(stream), "synchronize MoE benchmark");
  return std::chrono::duration<double, std::micro>(
             std::chrono::steady_clock::now() - start)
             .count() /
         iterations;
}

template <typename Function>
double measure_gpu(Function &&function, cudaStream_t stream, int warmup,
                   int iterations) {
  for (int index = 0; index < warmup; ++index) function();
  check(cudaStreamSynchronize(stream), "synchronize MoE GPU warmup");
  cudaEvent_t begin = nullptr;
  cudaEvent_t end = nullptr;
  check(cudaEventCreate(&begin), "create MoE begin event");
  check(cudaEventCreate(&end), "create MoE end event");
  check(cudaEventRecord(begin, stream), "record MoE begin event");
  for (int index = 0; index < iterations; ++index) function();
  check(cudaEventRecord(end, stream), "record MoE end event");
  check(cudaEventSynchronize(end), "synchronize MoE end event");
  float milliseconds = 0.0f;
  check(cudaEventElapsedTime(&milliseconds, begin, end),
        "measure MoE GPU events");
  cudaEventDestroy(begin);
  cudaEventDestroy(end);
  return static_cast<double>(milliseconds) * 1000.0 / iterations;
}

} // namespace

int main(int argc, char **argv) {
  try {
    const int layers = argc > 1 ? std::atoi(argv[1]) : 48;
    const int experts = argc > 2 ? std::atoi(argv[2]) : 10;
    const int iterations = argc > 3 ? std::atoi(argv[3]) : 100;
    const bool capture = argc <= 4 || std::string(argv[4]) != "--eager";
    if (layers < 1 || layers > 48 || experts < 1 || experts > 16 ||
        iterations < 1 || iterations > 10000)
      throw std::runtime_error("arguments must be layers 1..48, experts 1..16, iterations 1..10000");
    ggml_backend_t cuda_backend = ggml_backend_cuda_init(0);
    if (!cuda_backend) throw std::runtime_error("cannot initialize CUDA backend");
    ggml_backend_t adapter = make_adapter(cuda_backend);
    const size_t context_bytes = ggml_tensor_overhead() * (4 * layers + 8) +
                                 ggml_graph_overhead_custom(4 * layers + 8, false) +
                                 4096;
    auto *ctx = ggml_init({context_bytes, nullptr, true});
    if (!ctx) throw std::runtime_error("cannot initialize GGML context");
    auto *input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kHidden, 1);
    if (!input) throw std::runtime_error("cannot create input tensor");
    std::vector<std::unique_ptr<MoeState>> states;
    std::vector<LayerPayload> payloads;
    std::vector<ggml_tensor *> opaque;
    states.reserve(layers);
    payloads.reserve(layers);
    opaque.reserve(layers);
    auto *current = input;
    for (int layer = 0; layer < layers; ++layer) {
      payloads.push_back(make_layer_payload(layer, experts));
      auto *weight = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, payloads.back().bytes.size());
      auto state = std::make_unique<MoeState>();
      state->experts = experts;
      state->capture = capture;
      for (int projection = 0; projection < 3; ++projection) {
        state->offsets[projection] = payloads.back().offsets[projection];
        state->headers[projection] = payloads.back().headers[projection];
      }
      auto *node = ggml_map_custom2(ctx, current, weight, custom_placeholder, 1, nullptr);
      node->extra = state.get();
      current = node;
      opaque.push_back(weight);
      states.push_back(std::move(state));
    }
    auto *graph = ggml_new_graph_custom(ctx, 4 * layers + 8, false);
    ggml_build_forward_expand(graph, current);
    auto *buffer = ggml_backend_alloc_ctx_tensors(ctx, cuda_backend);
    if (!buffer) throw std::runtime_error("cannot allocate CUDA GGML tensors");
    std::vector<float> input_host(kHidden);
    for (int index = 0; index < kHidden; ++index)
      input_host[index] = 0.1f * std::sin(index * 0.071f);
    ggml_backend_tensor_set(input, input_host.data(), 0,
                            input_host.size() * sizeof(float));
    cudaStream_t stream = moestream_ggml_backend_cuda_get_stream(cuda_backend);
    for (int layer = 0; layer < layers; ++layer) {
      ggml_backend_tensor_set(opaque[layer], payloads[layer].bytes.data(), 0,
                              payloads[layer].bytes.size());
      states[layer]->stream = stream;
      check(cudaMalloc(&states[layer]->activation, kIntermediate * sizeof(float)),
            "allocate MoE activation");
      const size_t gate_partials = static_cast<size_t>(kIntermediate) *
                                   exl3_cuda_key_splits(kHidden);
      const size_t down_partials = static_cast<size_t>(kHidden) *
                                   exl3_cuda_key_splits(kIntermediate);
      check(cudaMalloc(&states[layer]->partials,
                       std::max(2 * gate_partials, down_partials) * sizeof(float)),
            "allocate MoE partials");
    }
    const auto graph_run = [&] {
      if (ggml_backend_graph_compute(adapter, graph) != GGML_STATUS_SUCCESS)
        throw std::runtime_error("MoE adapter graph failed");
    };
    graph_run();
    std::vector<float> output(kHidden);
    ggml_backend_tensor_get(current, output.data(), 0, output.size() * sizeof(float));
    size_t non_finite = 0;
    for (float value : output)
      non_finite += !std::isfinite(value);
    const double enqueue_us =
        measure_enqueue(graph_run, stream, 3, iterations);
    const double gpu_us = measure_gpu(graph_run, stream, 3, iterations);
    std::printf(
        "EXL3 GGML CUDA MoE bridge layers=%d experts=%d capture=%d "
        "enqueue_us=%.2f gpu_us=%.2f tok_s=%.2f non_finite=%zu\n",
        layers, experts, capture ? 1 : 0, enqueue_us, gpu_us,
        1.0e6 / gpu_us, non_finite);
    ggml_backend_buffer_free(buffer);
    ggml_backend_free(adapter);
    ggml_free(ctx);
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "exl3_ggml_cuda_moe_bench: %s\n", error.what());
    return 1;
  }
}
