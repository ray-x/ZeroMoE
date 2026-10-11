#include "exl3_cpu.hpp"

#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kHidden = 2560;
constexpr int kIntermediate = 640;

struct Options {
  int layers = 26;
  int experts = 10;
  int bits = 3;
  int threads = 6;
  int warmup = 2;
  int iterations = 10;
  bool exact = false;
  bool predecoded = false;
};

Options parse_options(int argc, char **argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    if (argument == "--exact") {
      options.exact = true;
      continue;
    }
    if (argument == "--predecoded") {
      options.predecoded = true;
      continue;
    }
    if (index + 1 >= argc)
      throw std::runtime_error("Missing value for " + argument);
    const int value = std::stoi(argv[++index]);
    if (argument == "--layers") options.layers = value;
    else if (argument == "--experts") options.experts = value;
    else if (argument == "--bits") options.bits = value;
    else if (argument == "--threads") options.threads = value;
    else if (argument == "--warmup") options.warmup = value;
    else if (argument == "--iterations") options.iterations = value;
    else throw std::runtime_error("Unknown option: " + argument);
  }
  if (options.layers < 1 || options.layers > 48 ||
      options.experts < 1 || options.experts > 32 ||
      options.bits < 1 || options.bits > 8 ||
      options.threads < 1 || options.threads > 32 ||
      options.warmup < 0 || options.warmup > 20 ||
      options.iterations < 1 || options.iterations > 1000 ||
      (options.exact && options.predecoded))
    throw std::runtime_error("Options out of range");
  return options;
}

struct Timing {
  double gate_up_ms = 0.0;
  double activation_ms = 0.0;
  double down_ms = 0.0;
  double accumulation_ms = 0.0;
};

double milliseconds(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - start).count();
}

struct Benchmark {
  Options options;
  std::vector<uint16_t> trellis;
  std::vector<uint16_t> hidden_scales;
  std::vector<uint16_t> intermediate_scales;
  std::vector<Exl3Mul1MatrixView> matrices;
  std::vector<std::vector<uint8_t>> predecoded_weights;
  std::vector<Exl3Mul1Q8MatrixView> predecoded_matrices;
  std::vector<float> input;
  std::vector<float> gate;
  std::vector<float> up;
  std::vector<float> activation;
  std::vector<float> expert_output;
  std::vector<float> output;
  std::vector<Exl3Mul1BatchItem> gate_up_items;
  std::vector<Exl3Mul1BatchItem> down_items;
  std::vector<Exl3Mul1Q8BatchItem> gate_up_q8_items;
  std::vector<Exl3Mul1Q8BatchItem> down_q8_items;
  double predecode_ms = 0.0;

  explicit Benchmark(const Options &settings)
      : options(settings),
        trellis(static_cast<size_t>(settings.layers) * settings.experts * 3 *
                kHidden * kIntermediate * settings.bits / 16),
        hidden_scales(kHidden, 0x3000),
        intermediate_scales(kIntermediate, 0x3000),
        input(kHidden),
        gate(static_cast<size_t>(settings.experts) * kIntermediate),
        up(gate.size()),
        activation(gate.size()),
        expert_output(static_cast<size_t>(settings.experts) * kHidden),
        output(kHidden),
        gate_up_items(static_cast<size_t>(settings.experts) * 2),
        down_items(settings.experts),
        gate_up_q8_items(static_cast<size_t>(settings.experts) * 2),
        down_q8_items(settings.experts) {
    for (size_t index = 0; index < trellis.size(); ++index) {
      const uint32_t value = static_cast<uint32_t>(index) * 1664525u + 1013904223u;
      trellis[index] = static_cast<uint16_t>(value ^ (value >> 16));
    }
    for (int index = 0; index < kHidden; ++index)
      input[index] = 0.1f * std::sin(index * 0.071f);
    const size_t words_per_matrix =
        static_cast<size_t>(kHidden) * kIntermediate * settings.bits / 16;
    const size_t matrix_count = static_cast<size_t>(settings.layers) * settings.experts * 3;
    matrices.reserve(matrix_count);
    for (size_t index = 0; index < matrix_count; ++index) {
      const bool down = index % 3 == 2;
      matrices.push_back({
          trellis.data() + index * words_per_matrix,
          down ? intermediate_scales.data() : hidden_scales.data(),
          down ? hidden_scales.data() : intermediate_scales.data(),
          nullptr,
          down ? kIntermediate : kHidden,
          down ? kHidden : kIntermediate,
          settings.bits});
    }
    if (options.predecoded) {
      const auto predecode_start = std::chrono::steady_clock::now();
      predecoded_weights.reserve(matrix_count);
      predecoded_matrices.reserve(matrix_count);
      for (const auto &matrix : matrices) {
        predecoded_weights.push_back(exl3_mul1_predecode_q8(matrix));
        if (predecoded_weights.back().size() !=
            static_cast<size_t>(matrix.input_features) * matrix.output_features)
          throw std::runtime_error("Predecode failed");
        predecoded_matrices.push_back({
            predecoded_weights.back().data(), matrix.suh, matrix.svh,
            matrix.bias, matrix.input_features, matrix.output_features});
      }
      predecode_ms = milliseconds(predecode_start);
    }
  }

  Timing run() {
    Timing timing;
    for (int layer = 0; layer < options.layers; ++layer) {
      for (int expert = 0; expert < options.experts; ++expert) {
        const size_t matrix_index =
            static_cast<size_t>(layer * options.experts + expert) * 3;
        gate_up_items[expert * 2] = {
            matrices[matrix_index], input.data(),
            gate.data() + static_cast<size_t>(expert) * kIntermediate};
        gate_up_items[expert * 2 + 1] = {
            matrices[matrix_index + 1], input.data(),
            up.data() + static_cast<size_t>(expert) * kIntermediate};
        down_items[expert] = {
            matrices[matrix_index + 2],
            activation.data() + static_cast<size_t>(expert) * kIntermediate,
            expert_output.data() + static_cast<size_t>(expert) * kHidden};
        if (options.predecoded) {
          gate_up_q8_items[expert * 2] = {
              predecoded_matrices[matrix_index], input.data(),
              gate.data() + static_cast<size_t>(expert) * kIntermediate};
          gate_up_q8_items[expert * 2 + 1] = {
              predecoded_matrices[matrix_index + 1], input.data(),
              up.data() + static_cast<size_t>(expert) * kIntermediate};
          down_q8_items[expert] = {
              predecoded_matrices[matrix_index + 2],
              activation.data() + static_cast<size_t>(expert) * kIntermediate,
              expert_output.data() + static_cast<size_t>(expert) * kHidden};
        }
      }

      auto start = std::chrono::steady_clock::now();
      const bool gate_success = options.predecoded
          ? exl3_mul1_q8_matmul_batch(gate_up_q8_items.data(),
                                      static_cast<int>(gate_up_q8_items.size()))
          : exl3_mul1_matmul_batch(gate_up_items.data(),
                                   static_cast<int>(gate_up_items.size()),
                                   !options.exact);
      if (!gate_success)
        throw std::runtime_error("Gate/up batch failed");
      timing.gate_up_ms += milliseconds(start);

      start = std::chrono::steady_clock::now();
      #pragma omp parallel for schedule(static)
      for (int index = 0; index < static_cast<int>(activation.size()); ++index) {
        const float value = gate[index];
        activation[index] = value / (1.0f + std::exp(-value)) * up[index];
      }
      timing.activation_ms += milliseconds(start);

      start = std::chrono::steady_clock::now();
      const bool down_success = options.predecoded
          ? exl3_mul1_q8_matmul_batch(down_q8_items.data(), options.experts)
          : exl3_mul1_matmul_batch(down_items.data(), options.experts,
                                   !options.exact);
      if (!down_success)
        throw std::runtime_error("Down batch failed");
      timing.down_ms += milliseconds(start);

      start = std::chrono::steady_clock::now();
      std::fill(output.begin(), output.end(), 0.0f);
      for (int expert = 0; expert < options.experts; ++expert) {
        const float *source = expert_output.data() +
                              static_cast<size_t>(expert) * kHidden;
        for (int column = 0; column < kHidden; ++column)
          output[column] += source[column] / options.experts;
      }
      timing.accumulation_ms += milliseconds(start);
    }
    return timing;
  }
};

} // namespace

int main(int argc, char **argv) {
  try {
    const Options options = parse_options(argc, argv);
    if (!exl3_cpu_vnni_available())
      throw std::runtime_error("AVX-512 VNNI is required for this benchmark");
    omp_set_num_threads(options.threads);
    Benchmark benchmark(options);
    std::cout << std::fixed << std::setprecision(3)
              << "EXL3 synthetic CPU benchmark bits=" << options.bits
              << " layers=" << options.layers << " experts=" << options.experts
              << " threads=" << options.threads
              << " int8_activation=" << (!options.exact)
              << " predecoded=" << options.predecoded
              << " packed_MiB=" << benchmark.trellis.size() * sizeof(uint16_t) /
                                    (1024.0 * 1024.0)
              << " predecode_ms=" << benchmark.predecode_ms
              << " warmup=" << options.warmup
              << " iterations=" << options.iterations << '\n';
    for (int iteration = 0; iteration < options.warmup; ++iteration)
      benchmark.run();
    Timing total;
    const auto wall_start = std::chrono::steady_clock::now();
    for (int iteration = 0; iteration < options.iterations; ++iteration) {
      const Timing sample = benchmark.run();
      total.gate_up_ms += sample.gate_up_ms;
      total.activation_ms += sample.activation_ms;
      total.down_ms += sample.down_ms;
      total.accumulation_ms += sample.accumulation_ms;
    }
    const double wall_ms = milliseconds(wall_start) / options.iterations;
    std::cout << "gate_up_ms=" << total.gate_up_ms / options.iterations
              << " activation_ms=" << total.activation_ms / options.iterations
              << " down_ms=" << total.down_ms / options.iterations
              << " accumulation_ms=" << total.accumulation_ms / options.iterations
              << " wall_ms=" << wall_ms
              << " checksum=" << benchmark.output[0] << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "EXL3 CPU benchmark failed: " << error.what() << '\n';
    return 1;
  }
}
