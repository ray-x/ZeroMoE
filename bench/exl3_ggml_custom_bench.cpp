#include "exl3_cuda.hpp"
#include "exl3_format.hpp"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <numeric>
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

struct DeviceBuffers {
  cudaStream_t stream = nullptr;
  uint8_t *packed = nullptr;
  float *input = nullptr;
  float *output = nullptr;
  float *partials = nullptr;
  size_t packed_bytes = 0;
  bool weight_uploaded = false;

  DeviceBuffers() {
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create stream");
    packed_bytes = packed_size();
    check(cudaMalloc(&packed, packed_bytes), "allocate packed weight");
    check(cudaMalloc(&input, kInput * sizeof(float)), "allocate input");
    check(cudaMalloc(&output, kOutput * sizeof(float)), "allocate output");
    check(cudaMalloc(&partials,
                     static_cast<size_t>(kOutput) * exl3_cuda_key_splits(kInput) *
                         sizeof(float)),
          "allocate partials");
  }

  ~DeviceBuffers() {
    if (stream) cudaStreamSynchronize(stream);
    if (partials) cudaFree(partials);
    if (output) cudaFree(output);
    if (input) cudaFree(input);
    if (packed) cudaFree(packed);
    if (stream) cudaStreamDestroy(stream);
  }

  static size_t packed_size() {
    const size_t trellis = static_cast<size_t>(kInput) * kOutput / 8 * kBits;
    const size_t suh = static_cast<size_t>(kInput) * sizeof(uint16_t);
    const size_t svh = static_cast<size_t>(kOutput) * sizeof(uint16_t);
    const auto align64 = [](size_t value) { return (value + 63) & ~size_t(63); };
    const size_t trellis_offset = align64(sizeof(Exl3PackedMatrixHeader));
    const size_t suh_offset = align64(trellis_offset + trellis);
    const size_t svh_offset = align64(suh_offset + suh);
    return align64(svh_offset + svh);
  }

  void run(const void *packed_host, const float *input_host, float *output_host) {
    if (!weight_uploaded) {
      check(cudaMemcpyAsync(packed, packed_host, packed_bytes, cudaMemcpyHostToDevice,
                            stream),
            "upload opaque weight");
      weight_uploaded = true;
    }
    Exl3PackedMatrixHeader header{};
    std::memcpy(&header, packed_host, sizeof(header));
    if (!exl3_valid_packed_header(header, packed_bytes))
      throw std::runtime_error("opaque tensor does not contain a valid EXL3 header");
    const auto *base = packed;
    Exl3CudaMatrixView matrix{
        reinterpret_cast<const uint16_t *>(base + header.trellis_offset),
        reinterpret_cast<const uint16_t *>(base + header.suh_offset),
        reinterpret_cast<const uint16_t *>(base + header.svh_offset),
        nullptr,
        static_cast<int>(header.input_features),
        static_cast<int>(header.output_features),
        static_cast<int>(header.bits)};
    check(cudaMemcpyAsync(input, input_host, kInput * sizeof(float),
                          cudaMemcpyHostToDevice, stream),
          "upload activation");
    check(launch_exl3_mul1_matmul(matrix, input, output, 1, partials, stream),
          "launch EXL3 custom kernel");
    check(cudaMemcpyAsync(output_host, output, kOutput * sizeof(float),
                          cudaMemcpyDeviceToHost, stream),
          "download custom output");
    check(cudaStreamSynchronize(stream), "synchronize custom kernel");
  }
};

struct CustomState {
  DeviceBuffers *device = nullptr;
  const void *packed_host = nullptr;
  size_t packed_bytes = 0;
};

void opaque_exl3_custom(ggml_tensor *dst, const ggml_tensor *input,
                        const ggml_tensor *opaque, int ith, int nth,
                        void *userdata) {
  if (ith != 0 || nth != 1)
    throw std::runtime_error("opaque EXL3 custom op requires one GGML task");
  auto *state = static_cast<CustomState *>(userdata);
  if (!state || !state->device || input->type != GGML_TYPE_F32 ||
      opaque->type != GGML_TYPE_I8 || dst->type != GGML_TYPE_F32 ||
      input->ne[0] != kInput || input->ne[1] != 1 || dst->ne[0] != kOutput ||
      dst->ne[1] != 1)
    throw std::runtime_error("opaque EXL3 custom op type mismatch");
  if (opaque->data != state->packed_host)
    throw std::runtime_error("GGML opaque tensor storage changed during replay");
  state->device->run(opaque->data, static_cast<const float *>(input->data),
                     static_cast<float *>(dst->data));
}

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
double measure(Function &&function, int warmup, int iterations) {
  for (int index = 0; index < warmup; ++index) function();
  const auto start = std::chrono::steady_clock::now();
  for (int index = 0; index < iterations; ++index) function();
  return std::chrono::duration<double, std::micro>(
             std::chrono::steady_clock::now() - start)
      .count() /
         iterations;
}

} // namespace

int main(int argc, char **argv) {
  try {
    const int iterations = argc > 1 ? std::atoi(argv[1]) : 100;
    if (iterations < 1 || iterations > 10000)
      throw std::runtime_error("iterations must be in [1, 10000]");
    ggml_cpu_init();
    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend) throw std::runtime_error("cannot initialize GGML CPU backend");
    auto *ctx = ggml_init({ggml_tensor_overhead() * 16 +
                               ggml_graph_overhead_custom(16, false) + 4096,
                           nullptr, true});
    if (!ctx) throw std::runtime_error("cannot initialize GGML context");
    auto *input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kInput, 1);
    auto *opaque = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, DeviceBuffers::packed_size());
    if (!input || !opaque) throw std::runtime_error("cannot create GGML tensors");
    const auto packed = make_packed_weight();
    std::vector<float> input_host(kInput);
    for (int index = 0; index < kInput; ++index)
      input_host[index] = 0.1f * std::sin(index * 0.071f);
    CustomState state;
    DeviceBuffers device;
    state.device = &device;
    state.packed_bytes = packed.size();
    auto *node = ggml_map_custom2(ctx, input, opaque, opaque_exl3_custom, 1, &state);
    node->ne[0] = kOutput;
    node->ne[1] = 1;
    node->ne[2] = 1;
    node->nb[1] = kOutput * sizeof(float);
    auto *graph = ggml_new_graph_custom(ctx, 16, false);
    ggml_build_forward_expand(graph, node);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) throw std::runtime_error("cannot allocate GGML tensors");
    ggml_backend_tensor_set(input, input_host.data(), 0,
                            input_host.size() * sizeof(float));
    ggml_backend_tensor_set(opaque, packed.data(), 0, packed.size());
    state.packed_host = opaque->data;
    const auto graph_run = [&] {
      if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS)
        throw std::runtime_error("GGML custom graph failed");
    };
    graph_run();
    for (int index = 0; index < kOutput; ++index)
      if (!std::isfinite(static_cast<float *>(node->data)[index]))
        throw std::runtime_error("non-finite custom output");
    const double graph_us = measure(graph_run, 10, iterations);
    const auto direct_run = [&] {
      device.run(opaque->data, static_cast<const float *>(input->data),
                 static_cast<float *>(node->data));
    };
    const double direct_us = measure(direct_run, 10, iterations);
    std::printf("EXL3 GGML opaque custom benchmark graph_us=%.2f direct_us=%.2f overhead_us=%.2f packed_bytes=%zu\n",
                graph_us, direct_us, graph_us - direct_us, packed.size());
    ggml_backend_buffer_free(buffer);
    ggml_backend_free(backend);
    ggml_free(ctx);
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "exl3_ggml_custom_bench: %s\n", error.what());
    return 1;
  }
}
