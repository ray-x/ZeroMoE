#include "exl3_cuda.hpp"
#include "exl3_format.hpp"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-cuda.h"
#include "ggml-impl.h"

#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" cudaStream_t zeromoe_ggml_backend_cuda_get_stream(
    ggml_backend_t backend);

namespace {

constexpr int kHidden = 2560;
constexpr int kIntermediate = 640;
constexpr int kBits = 4;

void check(cudaError_t status, const char *operation) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

size_t packed_size(int input, int output) {
  const size_t trellis = static_cast<size_t>(input) * output / 8 * kBits;
  const size_t suh = static_cast<size_t>(input) * sizeof(uint16_t);
  const size_t svh = static_cast<size_t>(output) * sizeof(uint16_t);
  const auto align64 = [](size_t value) { return (value + 63) & ~size_t(63); };
  const size_t trellis_offset = align64(sizeof(Exl3PackedMatrixHeader));
  const size_t suh_offset = align64(trellis_offset + trellis);
  const size_t svh_offset = align64(suh_offset + suh);
  return align64(svh_offset + svh);
}

std::vector<uint8_t> make_packed_matrix(int input, int output, uint32_t seed) {
  const size_t trellis = static_cast<size_t>(input) * output / 8 * kBits;
  const size_t suh = static_cast<size_t>(input) * sizeof(uint16_t);
  const size_t svh = static_cast<size_t>(output) * sizeof(uint16_t);
  const auto align64 = [](size_t value) { return (value + 63) & ~size_t(63); };
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

struct FusedState {
  cudaStream_t stream = nullptr;
  float *activation = nullptr;
  float *partials = nullptr;
  size_t offsets[3] = {};
  Exl3PackedMatrixHeader headers[3] = {};
  cudaGraph_t cuda_graph = nullptr;
  cudaGraphExec_t cuda_graph_exec = nullptr;
  bool capture = true;

  ~FusedState() {
    if (cuda_graph_exec) cudaGraphExecDestroy(cuda_graph_exec);
    if (cuda_graph) cudaGraphDestroy(cuda_graph);
    if (partials) cudaFree(partials);
    if (activation) cudaFree(activation);
  }

  Exl3CudaMatrixView view(const uint8_t *base, int index) const {
    const auto &header = headers[index];
    return {
        reinterpret_cast<const uint16_t *>(base + offsets[index] + header.trellis_offset),
        reinterpret_cast<const uint16_t *>(base + offsets[index] + header.suh_offset),
        reinterpret_cast<const uint16_t *>(base + offsets[index] + header.svh_offset),
        nullptr,
        static_cast<int>(header.input_features),
        static_cast<int>(header.output_features),
        static_cast<int>(header.bits)};
  }

  void enqueue(const ggml_tensor *input, const ggml_tensor *opaque,
               ggml_tensor *output) {
    const auto *base = static_cast<const uint8_t *>(opaque->data);
    check(cudaMemsetAsync(output->data, 0, kHidden * sizeof(float), stream),
          "clear fused output");
    const size_t gate_partials = static_cast<size_t>(kIntermediate) *
                                 exl3_cuda_key_splits(kHidden);
    check(launch_exl3_mul1_gate_up(
              view(base, 0), view(base, 1), static_cast<const float *>(input->data),
              activation, 1, partials, partials + gate_partials, stream),
          "launch fused gate/up");
    check(launch_exl3_mul1_matmul_accumulate(
              view(base, 2), activation, static_cast<float *>(output->data),
              1.0f, 1, partials, stream),
          "launch fused down");
  }

  void run(const ggml_tensor *input, const ggml_tensor *opaque, ggml_tensor *output) {
    if (!capture) {
      enqueue(input, opaque, output);
      return;
    }
    if (!cuda_graph_exec) {
      check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
            "begin fused FFN graph capture");
      enqueue(input, opaque, output);
      check(cudaStreamEndCapture(stream, &cuda_graph), "end fused FFN graph capture");
      check(cudaGraphInstantiate(&cuda_graph_exec, cuda_graph, nullptr, nullptr, 0),
            "instantiate fused FFN graph");
    }
    check(cudaGraphLaunch(cuda_graph_exec, stream), "launch fused FFN graph");
  }
};

struct Adapter {
  ggml_backend backend{};
  ggml_backend_t inner = nullptr;
};

const char *adapter_name(ggml_backend_t) { return "EXL3-CUDA-FFN-ADAPTER"; }

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
    static_cast<FusedState *>(node->extra)->run(node->src[0], node->src[1], node);
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
  check(cudaStreamSynchronize(stream), "synchronize fused warmup");
  const auto start = std::chrono::steady_clock::now();
  for (int index = 0; index < iterations; ++index) function();
  check(cudaStreamSynchronize(stream), "synchronize fused benchmark");
  return std::chrono::duration<double, std::micro>(
             std::chrono::steady_clock::now() - start)
             .count() /
         iterations;
}

template <typename Function>
double measure_gpu(Function &&function, cudaStream_t stream, int warmup,
                   int iterations) {
  for (int index = 0; index < warmup; ++index) function();
  check(cudaStreamSynchronize(stream), "synchronize fused GPU warmup");
  cudaEvent_t begin = nullptr;
  cudaEvent_t end = nullptr;
  check(cudaEventCreate(&begin), "create fused begin event");
  check(cudaEventCreate(&end), "create fused end event");
  check(cudaEventRecord(begin, stream), "record fused begin event");
  for (int index = 0; index < iterations; ++index) function();
  check(cudaEventRecord(end, stream), "record fused end event");
  check(cudaEventSynchronize(end), "synchronize fused end event");
  float milliseconds = 0.0f;
  check(cudaEventElapsedTime(&milliseconds, begin, end),
        "measure fused GPU events");
  cudaEventDestroy(begin);
  cudaEventDestroy(end);
  return static_cast<double>(milliseconds) * 1000.0 / iterations;
}

} // namespace

int main(int argc, char **argv) {
  try {
    const int iterations = argc > 1 ? std::atoi(argv[1]) : 1000;
    const bool capture = argc <= 2 || std::string(argv[2]) != "--eager";
    if (iterations < 1 || iterations > 10000)
      throw std::runtime_error("iterations must be in [1, 10000]");
    const auto gate = make_packed_matrix(kHidden, kIntermediate, 2026100601);
    const auto up = make_packed_matrix(kHidden, kIntermediate, 2026100602);
    const auto down = make_packed_matrix(kIntermediate, kHidden, 2026100603);
    const auto align64 = [](size_t value) { return (value + 63) & ~size_t(63); };
    const size_t up_offset = align64(gate.size());
    const size_t down_offset = align64(up_offset + up.size());
    std::vector<uint8_t> packed(down_offset + down.size());
    std::memcpy(packed.data(), gate.data(), gate.size());
    std::memcpy(packed.data() + up_offset, up.data(), up.size());
    std::memcpy(packed.data() + down_offset, down.data(), down.size());
    ggml_backend_t cuda_backend = ggml_backend_cuda_init(0);
    if (!cuda_backend) throw std::runtime_error("cannot initialize CUDA backend");
    ggml_backend_t adapter = make_adapter(cuda_backend);
    auto *ctx = ggml_init({ggml_tensor_overhead() * 16 +
                               ggml_graph_overhead_custom(16, false) + 4096,
                           nullptr, true});
    if (!ctx) throw std::runtime_error("cannot initialize GGML context");
    auto *input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kHidden, 1);
    auto *opaque = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, packed.size());
    auto *node = ggml_map_custom2(ctx, input, opaque, custom_placeholder, 1, nullptr);
    auto *graph = ggml_new_graph_custom(ctx, 16, false);
    ggml_build_forward_expand(graph, node);
    auto *buffer = ggml_backend_alloc_ctx_tensors(ctx, cuda_backend);
    if (!buffer) throw std::runtime_error("cannot allocate CUDA GGML tensors");
    std::vector<float> input_host(kHidden);
    for (int index = 0; index < kHidden; ++index)
      input_host[index] = 0.1f * std::sin(index * 0.071f);
    ggml_backend_tensor_set(input, input_host.data(), 0,
                            input_host.size() * sizeof(float));
    ggml_backend_tensor_set(opaque, packed.data(), 0, packed.size());
    auto state = std::make_unique<FusedState>();
    state->capture = capture;
    state->stream = zeromoe_ggml_backend_cuda_get_stream(cuda_backend);
    state->offsets[0] = 0;
    state->offsets[1] = up_offset;
    state->offsets[2] = down_offset;
    std::memcpy(&state->headers[0], packed.data(), sizeof(Exl3PackedMatrixHeader));
    std::memcpy(&state->headers[1], packed.data() + up_offset, sizeof(Exl3PackedMatrixHeader));
    std::memcpy(&state->headers[2], packed.data() + down_offset, sizeof(Exl3PackedMatrixHeader));
    check(cudaMalloc(&state->activation, kIntermediate * sizeof(float)),
          "allocate fused activation");
    const size_t gate_partials = static_cast<size_t>(kIntermediate) *
                                 exl3_cuda_key_splits(kHidden);
    const size_t down_partials = static_cast<size_t>(kHidden) *
                                 exl3_cuda_key_splits(kIntermediate);
    check(cudaMalloc(&state->partials,
                     std::max(2 * gate_partials, down_partials) * sizeof(float)),
          "allocate fused partials");
    node->extra = state.get();
    const auto graph_run = [&] {
      if (ggml_backend_graph_compute(adapter, graph) != GGML_STATUS_SUCCESS)
        throw std::runtime_error("fused adapter graph failed");
    };
    graph_run();
    std::vector<float> output(kHidden);
    ggml_backend_tensor_get(node, output.data(), 0, output.size() * sizeof(float));
    for (float value : output)
      if (!std::isfinite(value)) throw std::runtime_error("non-finite fused output");
    const double enqueue_us =
        measure_enqueue(graph_run, state->stream, 10, iterations);
    const double gpu_us = measure_gpu(graph_run, state->stream, 10, iterations);
    std::printf(
        "EXL3 GGML CUDA fused FFN benchmark enqueue_us=%.2f gpu_us=%.2f "
        "tok_s=%.2f packed_bytes=%zu\n",
        enqueue_us, gpu_us, 1.0e6 / gpu_us, packed.size());
    ggml_backend_buffer_free(buffer);
    ggml_backend_free(adapter);
    ggml_free(ctx);
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "exl3_ggml_cuda_fused_ffn_bench: %s\n", error.what());
    return 1;
  }
}
