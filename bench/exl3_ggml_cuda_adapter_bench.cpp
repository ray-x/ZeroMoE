#include "exl3_cuda.hpp"
#include "exl3_format.hpp"
#include "exl3_ggml_backend.hpp"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kInput = 2560;
constexpr int kOutput = 640;
constexpr int kBits = 4;

void check(cudaError_t status, const char *operation) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

struct Exl3GpuNodeState {
  cudaStream_t stream = nullptr;
  float *partials = nullptr;
  Exl3PackedMatrixHeader header{};

  ~Exl3GpuNodeState() {
    if (partials) cudaFree(partials);
  }

  void run(const ggml_tensor *input, const ggml_tensor *opaque, ggml_tensor *output) {
    const auto *base = static_cast<const uint8_t *>(opaque->data);
    Exl3CudaMatrixView matrix{
        reinterpret_cast<const uint16_t *>(base + header.trellis_offset),
        reinterpret_cast<const uint16_t *>(base + header.suh_offset),
        reinterpret_cast<const uint16_t *>(base + header.svh_offset),
        nullptr,
        kInput,
        kOutput,
        kBits};
    check(launch_exl3_mul1_matmul(
              matrix, static_cast<const float *>(input->data),
              static_cast<float *>(output->data), 1, partials, stream),
          "launch GPU-resident EXL3 node");
  }
  static void compute(ggml_backend_t cuda_backend, ggml_tensor *node,
                      void *user_data) {
    auto *state = static_cast<Exl3GpuNodeState *>(user_data);
    state->stream = exl3_ggml_cuda_stream(cuda_backend);
    state->run(node->src[0], node->src[1], node);
  }
};

std::vector<uint8_t> make_packed_weight() {
  const size_t trellis = static_cast<size_t>(kInput) * kOutput / 8 * kBits;
  const size_t suh = static_cast<size_t>(kInput) * sizeof(uint16_t);
  const size_t svh = static_cast<size_t>(kOutput) * sizeof(uint16_t);
  const auto align64 = [](size_t value) { return (value + 63) & ~size_t(63); };
  Exl3PackedMatrixHeader header{};
  header.magic = kExl3PackedMagic;
  header.version = 1;
  header.input_features = kInput;
  header.output_features = kOutput;
  header.bits = kBits;
  header.trellis_offset = align64(sizeof(header));
  header.suh_offset = align64(header.trellis_offset + trellis);
  header.svh_offset = align64(header.suh_offset + suh);
  header.total_size = align64(header.svh_offset + svh);
  std::vector<uint8_t> packed(header.total_size);
  std::memcpy(packed.data(), &header, sizeof(header));
  std::mt19937 rng(20261006);
  for (size_t index = 0; index < trellis; ++index)
    packed[header.trellis_offset + index] = static_cast<uint8_t>(rng());
  for (int index = 0; index < kInput; ++index)
    reinterpret_cast<uint16_t *>(packed.data() + header.suh_offset)[index] = 0x3c00;
  for (int index = 0; index < kOutput; ++index)
    reinterpret_cast<uint16_t *>(packed.data() + header.svh_offset)[index] = 0x3c00;
  return packed;
}

template <typename Function>
double measure_enqueue(Function &&function, cudaStream_t stream, int warmup,
                       int iterations) {
  for (int index = 0; index < warmup; ++index) function();
  check(cudaStreamSynchronize(stream), "synchronize adapter warmup");
  const auto start = std::chrono::steady_clock::now();
  for (int index = 0; index < iterations; ++index)
    function();
  check(cudaStreamSynchronize(stream), "synchronize adapter benchmark");
  return std::chrono::duration<double, std::micro>(
             std::chrono::steady_clock::now() - start)
             .count() /
         iterations;
}

template <typename Function>
double measure_gpu(Function &&function, cudaStream_t stream, int warmup,
                   int iterations) {
  for (int index = 0; index < warmup; ++index) function();
  check(cudaStreamSynchronize(stream), "synchronize adapter GPU warmup");
  cudaEvent_t begin = nullptr;
  cudaEvent_t end = nullptr;
  check(cudaEventCreate(&begin), "create adapter begin event");
  check(cudaEventCreate(&end), "create adapter end event");
  check(cudaEventRecord(begin, stream), "record adapter begin event");
  for (int index = 0; index < iterations; ++index) function();
  check(cudaEventRecord(end, stream), "record adapter end event");
  check(cudaEventSynchronize(end), "synchronize adapter end event");
  float milliseconds = 0.0f;
  check(cudaEventElapsedTime(&milliseconds, begin, end),
        "measure adapter GPU events");
  cudaEventDestroy(begin);
  cudaEventDestroy(end);
  return static_cast<double>(milliseconds) * 1000.0 / iterations;
}

} // namespace

int main(int argc, char **argv) {
  try {
    const int iterations = argc > 1 ? std::atoi(argv[1]) : 1000;
    const bool capture = argc > 2 && std::string(argv[2]) == "--capture";
    if (iterations < 1 || iterations > 10000)
      throw std::runtime_error("iterations must be in [1, 10000]");
    const auto packed = make_packed_weight();
    ggml_backend_t cuda_backend = ggml_backend_cuda_init(0);
    if (!cuda_backend) throw std::runtime_error("cannot initialize CUDA backend");
    ggml_backend_t adapter = exl3_ggml_cuda_backend_wrap(cuda_backend, true);
    if (!adapter) throw std::runtime_error("cannot initialize EXL3 GGML adapter");
    if (!exl3_ggml_cuda_backend_set_graph_capture(adapter, capture))
      throw std::runtime_error("cannot configure EXL3 GGML graph capture");
    auto *ctx = ggml_init({ggml_tensor_overhead() * 16 +
                               ggml_graph_overhead_custom(16, false) + 4096,
                           nullptr, true});
    if (!ctx) throw std::runtime_error("cannot initialize GGML context");
    auto *input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kInput, 1);
    auto *input_activation = ggml_silu(ctx, input);
    auto *opaque = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, packed.size());
    auto state = std::make_unique<Exl3GpuNodeState>();
    Exl3GgmlCustomNode custom;
    custom.compute = Exl3GpuNodeState::compute;
    custom.user_data = state.get();
    auto *node = exl3_ggml_map_custom2(ctx, input_activation, opaque, &custom);
    if (!node) throw std::runtime_error("cannot create EXL3 GGML custom node");
    node->ne[0] = kOutput;
    node->ne[1] = 1;
    node->ne[2] = 1;
    node->nb[1] = kOutput * sizeof(float);
    auto *graph_output = ggml_silu(ctx, node);
    auto *graph = ggml_new_graph_custom(ctx, 16, false);
    ggml_build_forward_expand(graph, graph_output);
    auto *buffer = ggml_backend_alloc_ctx_tensors(ctx, cuda_backend);
    if (!buffer) throw std::runtime_error("cannot allocate CUDA GGML tensors");
    std::vector<float> input_host(kInput);
    for (int index = 0; index < kInput; ++index)
      input_host[index] = 0.1f * std::sin(index * 0.071f);
    ggml_backend_tensor_set(input, input_host.data(), 0,
                            input_host.size() * sizeof(float));
    ggml_backend_tensor_set(opaque, packed.data(), 0, packed.size());
    std::memcpy(&state->header, packed.data(), sizeof(state->header));
    check(cudaMalloc(&state->partials,
                     static_cast<size_t>(kOutput) * exl3_cuda_key_splits(kInput) *
                         sizeof(float)),
          "allocate partials");
    const auto graph_run = [&] {
      if (ggml_backend_graph_compute(adapter, graph) != GGML_STATUS_SUCCESS)
        throw std::runtime_error("adapter graph failed");
    };
    const cudaStream_t stream = exl3_ggml_cuda_stream(cuda_backend);
    graph_run();
    std::vector<float> output(kOutput);
    ggml_backend_tensor_get(node, output.data(), 0, output.size() * sizeof(float));
    for (float value : output)
      if (!std::isfinite(value)) throw std::runtime_error("non-finite adapter output");
    const double enqueue_us =
        measure_enqueue(graph_run, stream, 10, iterations);
    const double gpu_us = measure_gpu(graph_run, stream, 10, iterations);
    std::printf(
        "EXL3 GGML CUDA adapter benchmark enqueue_us=%.2f gpu_us=%.2f "
        "tok_s=%.2f capture=%d packed_bytes=%zu\n",
        enqueue_us, gpu_us, 1.0e6 / gpu_us, capture ? 1 : 0, packed.size());
    ggml_backend_buffer_free(buffer);
    ggml_backend_free(adapter);
    ggml_free(ctx);
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "exl3_ggml_cuda_adapter_bench: %s\n", error.what());
    return 1;
  }
}
