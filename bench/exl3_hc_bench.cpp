#include "exl3_gpu.hpp"

#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kWidth = 2560;
constexpr int kStreams = 4;
constexpr int kRank = 320;
constexpr int kExpanded = kWidth * kStreams;

struct Weights {
  std::vector<float> down;
  std::vector<float> up;
  std::vector<float> inject;
  Exl3GpuF32Weight down_view;
  Exl3GpuF32Weight up_view;
  Exl3GpuF32Weight inject_view;

  explicit Weights(int block)
      : down(kExpanded * kRank), up(kRank * kExpanded), inject(kExpanded * kStreams) {
    uint32_t state = 9137u + static_cast<uint32_t>(block);
    for (auto *values : {&down, &up, &inject}) {
      for (float &value : *values) {
        state = state * 1664525u + 1013904223u;
        value = (static_cast<float>(state >> 16) / 65535.0f - 0.5f) * 0.02f;
      }
    }
    const std::string prefix = "hc." + std::to_string(block) + ".";
    down_view = {prefix + "down", down.data(), kExpanded, kRank};
    up_view = {prefix + "up", up.data(), kRank, kExpanded};
    inject_view = {prefix + "inject", inject.data(), kExpanded, kStreams};
  }
};

template <typename Work>
double measure(int iterations, Work work) {
  const auto start = std::chrono::steady_clock::now();
  for (int iteration = 0; iteration < iterations; ++iteration) work();
  return std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - start).count() / iterations;
}

double relative_error(const std::vector<float> &actual, const std::vector<float> &expected) {
  double error = 0.0, energy = 0.0;
  for (size_t index = 0; index < actual.size(); ++index) {
    if (!std::isfinite(actual[index])) throw std::runtime_error("Non-finite HC output");
    const double difference = actual[index] - expected[index];
    error += difference * difference;
    energy += static_cast<double>(expected[index]) * expected[index];
  }
  return std::sqrt(error / std::max(energy, 1e-20));
}

} // namespace

int main(int argc, char **argv) {
  try {
    int device = 0, blocks = 1, iterations = 50, omp_threads = 0;
    for (int index = 1; index < argc; ++index) {
      const std::string argument = argv[index];
      if (index + 1 >= argc) throw std::runtime_error("Missing value for " + argument);
      const int value = std::stoi(argv[++index]);
      if (argument == "--device") device = value;
      else if (argument == "--blocks") blocks = value;
      else if (argument == "--iterations") iterations = value;
      else if (argument == "--omp-threads") omp_threads = value;
      else throw std::runtime_error("Unknown option: " + argument);
    }
    if (device < 0 || blocks < 1 || blocks > 128 || iterations < 1 || iterations > 1000 ||
        omp_threads < 0 || omp_threads > 32)
      throw std::runtime_error("Options out of range");
    if (omp_threads > 0) omp_set_num_threads(omp_threads);
    std::vector<Weights> weights;
    weights.reserve(blocks);
    for (int block = 0; block < blocks; ++block) weights.emplace_back(block);
    const size_t weight_bytes = static_cast<size_t>(blocks) *
        (2 * kExpanded * kRank + kExpanded * kStreams) * sizeof(float);
    Exl3GpuExecutor executor(device, 2 * weight_bytes);
    std::vector<float> normed(kExpanded), low(kRank), gate(kExpanded);
    std::vector<float> reference_hidden(kWidth), reference_injection(kStreams);
    std::vector<float> fused_hidden(kWidth), fused_injection(kStreams);
    std::vector<float> cpu_prime(6400, 0.01f);
    const auto prime_workers = [&] {
      if (omp_threads == 0) return;
      #pragma omp parallel for schedule(static)
      for (int index = 0; index < static_cast<int>(cpu_prime.size()); ++index)
        cpu_prime[index] = std::sin(cpu_prime[index] + 0.001f);
    };
    for (int index = 0; index < kExpanded; ++index)
      normed[index] = std::sin(index * 0.071f);
    const auto matmul = [&](const Exl3GpuF32Weight &weight,
                            const float *input, float *output) {
      executor.matmul(weight.key, 0, reinterpret_cast<const uint8_t *>(weight.values),
                      static_cast<size_t>(weight.input_features) * weight.output_features * sizeof(float),
                      input, output, 1, weight.input_features, weight.output_features);
    };
    const auto baseline = [&] {
      for (const auto &block : weights) {
        prime_workers();
        matmul(block.down_view, normed.data(), low.data());
        for (float &value : low) {
          value /= static_cast<float>(kStreams);
          value /= 1.0f + std::exp(-value);
        }
        matmul(block.up_view, low.data(), gate.data());
        for (float &value : gate) value = 1.0f / (1.0f + std::exp(-value));
        std::fill(reference_hidden.begin(), reference_hidden.end(), 0.0f);
        for (int stream_index = 0; stream_index < kStreams; ++stream_index)
          for (int column = 0; column < kWidth; ++column)
            reference_hidden[column] += normed[stream_index * kWidth + column] *
                gate[stream_index * kWidth + column] / static_cast<float>(kStreams);
        matmul(block.inject_view, normed.data(), reference_injection.data());
      }
    };
    const auto fused = [&] {
      for (const auto &block : weights) {
        prime_workers();
        executor.hc_mix(block.down_view, block.up_view, &block.inject_view,
                        normed.data(), fused_hidden.data(), fused_injection.data(),
                        1, kWidth, kStreams, true);
      }
    };
    baseline();
    fused();
    const double hidden_error = relative_error(fused_hidden, reference_hidden);
    const double injection_error = relative_error(fused_injection, reference_injection);
    if (hidden_error > 1e-5 || injection_error > 1e-5)
      throw std::runtime_error("HC device-chain parity failed");
    for (int warmup = 0; warmup < 3; ++warmup) {
      baseline();
      fused();
    }
    const double baseline_first = measure(iterations, baseline);
    const double fused_first = measure(iterations, fused);
    const double fused_second = measure(iterations, fused);
    const double baseline_second = measure(iterations, baseline);
    const double baseline_ms = (baseline_first + baseline_second) / 2.0;
    const double fused_ms = (fused_first + fused_second) / 2.0;
    std::cout << std::fixed << std::setprecision(3)
              << "EXL3 synthetic HC benchmark device=" << device << " blocks=" << blocks
              << " width=" << kWidth << " streams=" << kStreams << " rank=" << kRank
              << " weight_MiB=" << weight_bytes / (1024.0 * 1024.0)
              << " iterations=" << iterations << '\n'
              << "omp_threads=" << omp_threads
              << " omp_wait_policy=" << (std::getenv("OMP_WAIT_POLICY") ? std::getenv("OMP_WAIT_POLICY") : "default")
              << " cuda_schedule=" << (std::getenv("ZEROMOE_CUDA_SCHEDULE") ? std::getenv("ZEROMOE_CUDA_SCHEDULE") : "default")
              << " cpu_prime=" << cpu_prime[0] << '\n'
              << "baseline_ms=" << baseline_ms << " fused_ms=" << fused_ms
              << " speedup=" << baseline_ms / fused_ms
              << " uploads=" << executor.stats().uploads << '\n'
              << std::scientific << "hidden_relative_rms=" << hidden_error
              << " injection_relative_rms=" << injection_error << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "EXL3 HC benchmark failed: " << error.what() << '\n';
    return 1;
  }
}
