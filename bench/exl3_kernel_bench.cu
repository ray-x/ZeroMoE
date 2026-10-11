#include "exl3_cuda.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kHidden = 2560;
constexpr int kIntermediate = 640;

void check(cudaError_t error, const char *operation) {
  if (error != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
}

struct DeviceBuffer {
  void *data = nullptr;
  DeviceBuffer() = default;
  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;
  ~DeviceBuffer() { if (data) cudaFree(data); }
  void allocate(size_t bytes) { check(cudaMalloc(&data, bytes), "cudaMalloc"); }
  template <typename Value>
  Value *as() const { return static_cast<Value *>(data); }
};

struct Stream {
  cudaStream_t value = nullptr;
  Stream() { check(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking), "create stream"); }
  ~Stream() { if (value) cudaStreamDestroy(value); }
};

struct Events {
  cudaEvent_t start = nullptr;
  cudaEvent_t end = nullptr;
  Events() {
    check(cudaEventCreate(&start), "create start event");
    check(cudaEventCreate(&end), "create end event");
  }
  ~Events() {
    if (start) cudaEventDestroy(start);
    if (end) cudaEventDestroy(end);
  }
};

struct Graph {
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t executable = nullptr;
  ~Graph() {
    if (executable) cudaGraphExecDestroy(executable);
    if (graph) cudaGraphDestroy(graph);
  }
};

struct Options {
  int device = 0;
  int layers = 1;
  int experts = 10;
  int bits = 3;
  int warmup = 5;
  int iterations = 30;
  bool graph = true;
};

Options parse_options(int argc, char **argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    if (argument == "--no-graph") {
      options.graph = false;
      continue;
    }
    if (index + 1 >= argc)
      throw std::runtime_error("Missing value for " + argument);
    const int value = std::stoi(argv[++index]);
    if (argument == "--device") options.device = value;
    else if (argument == "--layers") options.layers = value;
    else if (argument == "--experts") options.experts = value;
    else if (argument == "--bits") options.bits = value;
    else if (argument == "--warmup") options.warmup = value;
    else if (argument == "--iterations") options.iterations = value;
    else throw std::runtime_error("Unknown option: " + argument);
  }
  if (options.device < 0 || options.layers < 1 || options.layers > 48 ||
      options.experts < 1 || options.experts > 128 ||
      options.bits < 1 || options.bits > 8 || options.warmup < 0 ||
      options.warmup > 100 || options.iterations < 1 || options.iterations > 1000)
    throw std::runtime_error("Options out of range");
  return options;
}

__global__ void fill_trellis(uint16_t *trellis, size_t words) {
  const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= words) return;
  const uint32_t value = static_cast<uint32_t>(index) * 1664525u + 1013904223u;
  trellis[index] = static_cast<uint16_t>(value ^ (value >> 16));
}

struct Benchmark {
  Options options;
  Stream stream;
  DeviceBuffer trellis;
  DeviceBuffer hidden_scales;
  DeviceBuffer intermediate_scales;
  DeviceBuffer input;
  DeviceBuffer activation;
  DeviceBuffer output;
  DeviceBuffer partials;
  std::vector<std::array<Exl3CudaMatrixView, 3>> matrices;
  size_t trellis_bytes = 0;
  size_t gate_partial_elements = 0;

  explicit Benchmark(const Options &settings) : options(settings) {
    const size_t matrix_count = static_cast<size_t>(options.layers) * options.experts * 3;
    const size_t words_per_matrix =
        static_cast<size_t>(kHidden) * kIntermediate * options.bits / 16;
    trellis_bytes = matrix_count * words_per_matrix * sizeof(uint16_t);
    gate_partial_elements = static_cast<size_t>(kIntermediate) *
                            exl3_cuda_key_splits(kHidden);
    const size_t down_partial_elements = static_cast<size_t>(kHidden) *
                                         exl3_cuda_key_splits(kIntermediate);
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    check(cudaMemGetInfo(&free_bytes, &total_bytes), "query free VRAM");
    if (free_bytes < trellis_bytes + 256ull * 1024 * 1024)
      throw std::runtime_error("Not enough free VRAM for synthetic weights");

    trellis.allocate(trellis_bytes);
    hidden_scales.allocate(kHidden * sizeof(uint16_t));
    intermediate_scales.allocate(kIntermediate * sizeof(uint16_t));
    input.allocate(kHidden * sizeof(float));
    activation.allocate(kIntermediate * sizeof(float));
    output.allocate(kHidden * sizeof(float));
    partials.allocate(std::max(2 * gate_partial_elements, down_partial_elements) * sizeof(float));

    const size_t total_words = matrix_count * words_per_matrix;
    fill_trellis<<<(total_words + 255) / 256, 256, 0, stream.value>>>(
        trellis.as<uint16_t>(), total_words);
    check(cudaGetLastError(), "fill trellis");
    std::vector<uint16_t> hidden_host(kHidden, 0x3000);
    std::vector<uint16_t> intermediate_host(kIntermediate, 0x3000);
    std::vector<float> input_host(kHidden);
    for (int index = 0; index < kHidden; ++index)
      input_host[index] = 0.1f * std::sin(index * 0.071f);
    check(cudaMemcpyAsync(hidden_scales.data, hidden_host.data(),
                          kHidden * sizeof(uint16_t), cudaMemcpyHostToDevice,
                          stream.value), "upload hidden scales");
    check(cudaMemcpyAsync(intermediate_scales.data, intermediate_host.data(),
                          kIntermediate * sizeof(uint16_t), cudaMemcpyHostToDevice,
                          stream.value), "upload intermediate scales");
    check(cudaMemcpyAsync(input.data, input_host.data(), kHidden * sizeof(float),
                          cudaMemcpyHostToDevice, stream.value), "upload input");
    check(cudaStreamSynchronize(stream.value), "initialize benchmark buffers");

    matrices.reserve(static_cast<size_t>(options.layers) * options.experts);
    for (int layer = 0; layer < options.layers; ++layer) {
      for (int expert = 0; expert < options.experts; ++expert) {
        const size_t index = static_cast<size_t>(layer * options.experts + expert) * 3;
        std::array<Exl3CudaMatrixView, 3> views{};
        for (int projection = 0; projection < 3; ++projection) {
          const bool down = projection == 2;
          views[projection] = {
              trellis.as<uint16_t>() + (index + projection) * words_per_matrix,
              down ? intermediate_scales.as<uint16_t>() : hidden_scales.as<uint16_t>(),
              down ? hidden_scales.as<uint16_t>() : intermediate_scales.as<uint16_t>(),
              nullptr,
              down ? kIntermediate : kHidden,
              down ? kHidden : kIntermediate,
              options.bits};
        }
        matrices.push_back(views);
      }
    }
  }

  void routed_iteration() {
    for (int layer = 0; layer < options.layers; ++layer) {
      check(cudaMemsetAsync(output.data, 0, kHidden * sizeof(float), stream.value),
            "clear routed output");
      for (int expert = 0; expert < options.experts; ++expert) {
        const auto &views = matrices[static_cast<size_t>(layer) * options.experts + expert];
        check(launch_exl3_mul1_gate_up(
                  views[0], views[1], input.as<float>(), activation.as<float>(),
                  1, partials.as<float>(),
                  partials.as<float>() + gate_partial_elements, stream.value),
              "launch gate/up");
        check(launch_exl3_mul1_matmul_accumulate(
                  views[2], activation.as<float>(), output.as<float>(),
                  1.0f / options.experts, 1, partials.as<float>(), stream.value),
              "launch down/accumulate");
      }
    }
  }
};

struct Timing {
  double host_ms = 0.0;
  double device_timeline_ms = 0.0;
};

template <typename Work>
Timing measure(cudaStream_t stream, int warmup, int iterations, Work work) {
  Events events;
  for (int iteration = 0; iteration < warmup; ++iteration) {
    work();
    check(cudaStreamSynchronize(stream), "warmup");
  }
  double device_ms = 0.0;
  const auto start_time = std::chrono::steady_clock::now();
  for (int iteration = 0; iteration < iterations; ++iteration) {
    check(cudaEventRecord(events.start, stream), "record start");
    work();
    check(cudaEventRecord(events.end, stream), "record end");
    check(cudaEventSynchronize(events.end), "finish iteration");
    float elapsed_ms = 0.0f;
    check(cudaEventElapsedTime(&elapsed_ms, events.start, events.end),
          "measure iteration");
    device_ms += elapsed_ms;
  }
  const double host_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - start_time).count();
  return {host_ms / iterations, device_ms / iterations};
}

void print_timing(const char *label, const Timing &timing) {
  std::cout << label << " host_ms=" << timing.host_ms
            << " device_timeline_ms=" << timing.device_timeline_ms << '\n';
}

} // namespace

int main(int argc, char **argv) {
  try {
    const Options options = parse_options(argc, argv);
    int device_count = 0;
    check(cudaGetDeviceCount(&device_count), "count devices");
    if (options.device >= device_count)
      throw std::runtime_error("Requested CUDA device is unavailable");
    check(cudaSetDevice(options.device), "select device");
    Benchmark benchmark(options);
    std::cout << std::fixed << std::setprecision(3)
              << "EXL3 synthetic GPU benchmark device=" << options.device
              << " bits=" << options.bits << " layers=" << options.layers
              << " experts=" << options.experts << " hidden=" << kHidden
              << " intermediate=" << kIntermediate
              << " packed_MiB=" << benchmark.trellis_bytes / (1024.0 * 1024.0)
              << " launches_per_iteration=" << options.layers * options.experts * 5
              << " warmup=" << options.warmup
              << " iterations=" << options.iterations << '\n';

    const auto &first = benchmark.matrices.front();
    const Timing gate = measure(benchmark.stream.value, options.warmup,
                                options.iterations, [&] {
      check(launch_exl3_mul1_matmul(
                first[0], benchmark.input.as<float>(), benchmark.activation.as<float>(),
                1, benchmark.partials.as<float>(), benchmark.stream.value),
            "launch gate matmul");
    });
    print_timing("gate_matmul", gate);
    std::vector<float> host_input(kHidden);
    std::vector<float> host_output(kIntermediate);
    for (int index = 0; index < kHidden; ++index)
      host_input[index] = 0.1f * std::sin(index * 0.071f);
    const Timing gate_with_copies = measure(
        benchmark.stream.value, options.warmup, options.iterations, [&] {
      check(cudaMemcpyAsync(benchmark.input.data, host_input.data(),
                            kHidden * sizeof(float), cudaMemcpyHostToDevice,
                            benchmark.stream.value), "copy gate input");
      check(launch_exl3_mul1_matmul(
                first[0], benchmark.input.as<float>(), benchmark.activation.as<float>(),
                1, benchmark.partials.as<float>(), benchmark.stream.value),
            "launch copied gate matmul");
      check(cudaMemcpyAsync(host_output.data(), benchmark.activation.data,
                            kIntermediate * sizeof(float), cudaMemcpyDeviceToHost,
                            benchmark.stream.value), "copy gate output");
    });
    print_timing("gate_with_copies", gate_with_copies);
    if (options.experts >= 2) {
      const std::array<Exl3CudaMatrixView, 4> grouped_views{
          benchmark.matrices[0][0], benchmark.matrices[0][1],
          benchmark.matrices[1][0], benchmark.matrices[1][1]};
      std::array<std::vector<float>, 4> grouped_outputs;
      for (auto &values : grouped_outputs) values.resize(kIntermediate);
      const auto launch_group_projection = [&](int projection) {
        check(launch_exl3_mul1_matmul(
                  grouped_views[projection], benchmark.input.as<float>(),
                  benchmark.activation.as<float>(), 1,
                  benchmark.partials.as<float>(), benchmark.stream.value),
              "launch grouped projection");
        check(cudaMemcpyAsync(grouped_outputs[projection].data(),
                              benchmark.activation.data,
                              kIntermediate * sizeof(float), cudaMemcpyDeviceToHost,
                              benchmark.stream.value), "copy grouped output");
      };
      const auto copy_group_input = [&] {
        check(cudaMemcpyAsync(benchmark.input.data, host_input.data(),
                              kHidden * sizeof(float), cudaMemcpyHostToDevice,
                              benchmark.stream.value), "copy grouped input");
      };
      const Timing separate = measure(
          benchmark.stream.value, options.warmup, options.iterations, [&] {
        for (int projection = 0; projection < 4; ++projection) {
          copy_group_input();
          launch_group_projection(projection);
          check(cudaStreamSynchronize(benchmark.stream.value),
                "finish separate projection");
        }
      });
      print_timing("four_projections_separate", separate);
      const Timing grouped = measure(
          benchmark.stream.value, options.warmup, options.iterations, [&] {
        copy_group_input();
        for (int projection = 0; projection < 4; ++projection)
          launch_group_projection(projection);
      });
      print_timing("four_projections_grouped", grouped);
    }
    const Timing down = measure(benchmark.stream.value, options.warmup,
                                options.iterations, [&] {
      check(launch_exl3_mul1_matmul(
                first[2], benchmark.activation.as<float>(), benchmark.output.as<float>(),
                1, benchmark.partials.as<float>(), benchmark.stream.value),
            "launch down matmul");
    });
    print_timing("down_matmul", down);

    const Timing direct = measure(benchmark.stream.value, options.warmup,
                                  options.iterations, [&] {
      benchmark.routed_iteration();
    });
    print_timing("routed_direct", direct);

    if (options.graph) {
      Graph captured;
      check(cudaStreamBeginCapture(benchmark.stream.value, cudaStreamCaptureModeGlobal),
            "begin graph capture");
      benchmark.routed_iteration();
      check(cudaStreamEndCapture(benchmark.stream.value, &captured.graph),
            "end graph capture");
      check(cudaGraphInstantiate(&captured.executable, captured.graph, nullptr, nullptr, 0),
            "instantiate graph");
      const Timing replay = measure(benchmark.stream.value, options.warmup,
                                    options.iterations, [&] {
        check(cudaGraphLaunch(captured.executable, benchmark.stream.value),
              "replay graph");
      });
      print_timing("routed_graph", replay);
      std::cout << "graph_speedup=" << direct.host_ms / replay.host_ms << '\n';
    }
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "EXL3 benchmark failed: " << error.what() << '\n';
    return 1;
  }
}
