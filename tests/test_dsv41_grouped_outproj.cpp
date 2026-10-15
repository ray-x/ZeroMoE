#include <cuda_runtime.h>
#include "ggml-quants.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>

extern float dot_f32_avx512(const float*, const float*, int);
extern "C" void launch_q2_K_grouped_dot_f32(
    float*, const float*, const void*, size_t, int, int, int, cudaStream_t);

extern "C" void launch_q2_K_grouped_dot_f32_batch(
    float*, const float*, const void*, size_t, int, int, int, int, cudaStream_t);

static void check(cudaError_t err) {
  if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
}

static void run(int device, int groups, int rank, int width, int tokens = 1) {
  std::mt19937 rng(47);
  std::uniform_real_distribution<float> random(-1.0f, 1.0f);
  const size_t row_bytes = (size_t)(width / 256) * sizeof(block_q2_K);
  const size_t group_stride = (row_bytes * rank + 4095) / 4096 * 4096;
  std::vector<unsigned char> weights(group_stride * groups, 0x5a);
  std::vector<float> input((size_t)tokens * groups * width);
  std::vector<float> ref((size_t)tokens * groups * rank);
  std::vector<float> source(width), decoded(width);
  for (float& x : input) x = random(rng);
  for (int g = 0; g < groups; ++g) {
    for (int r = 0; r < rank; ++r) {
      for (float& x : source) x = random(rng) * (r % 4 == 0 ? 0.01f : 1.0f);
      auto* row = reinterpret_cast<block_q2_K*>(
          weights.data() + (size_t)g * group_stride + r * row_bytes);
      quantize_row_q2_K_ref(source.data(), row, width);
      dequantize_row_q2_K(row, decoded.data(), width);
      for (int t = 0; t < tokens; ++t)
        ref[((size_t)t * groups + g) * rank + r] = dot_f32_avx512(
            input.data() + ((size_t)t * groups + g) * width,
            decoded.data(), width);
    }
  }
  check(cudaSetDevice(device));
  float *dx = nullptr, *dy = nullptr;
  void* dw = nullptr;
  cudaStream_t stream;
  check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  check(cudaMalloc(&dx, input.size() * sizeof(float)));
  check(cudaMalloc(&dy, (ref.size() + 16) * sizeof(float)));
  check(cudaMalloc(&dw, weights.size()));
  std::vector<float> got(ref.size() + 16, -1234.5f);
  check(cudaMemcpyAsync(dx, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
  check(cudaMemcpyAsync(dw, weights.data(), weights.size(), cudaMemcpyHostToDevice, stream));
  check(cudaMemcpyAsync(dy, got.data(), got.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
  if (tokens == 1)
    launch_q2_K_grouped_dot_f32(dy, dx, dw, group_stride, groups, rank, width, stream);
  else
    launch_q2_K_grouped_dot_f32_batch(dy, dx, dw, group_stride,
                                     groups, rank, width, tokens, stream);
  check(cudaGetLastError());
  check(cudaMemcpyAsync(got.data(), dy, got.size() * sizeof(float), cudaMemcpyDeviceToHost, stream));
  check(cudaStreamSynchronize(stream));
  size_t mismatches = 0;
  float max_abs = 0;
  for (size_t i = 0; i < ref.size(); ++i) {
    if (std::memcmp(&ref[i], &got[i], sizeof(float))) ++mismatches;
    max_abs = std::max(max_abs, std::fabs(ref[i] - got[i]));
  }
  for (size_t i = ref.size(); i < got.size(); ++i)
    if (got[i] != -1234.5f) throw std::runtime_error("output guard overwritten");
  std::vector<unsigned char> after(weights.size());
  check(cudaMemcpy(after.data(), dw, after.size(), cudaMemcpyDeviceToHost));
  if (after != weights) throw std::runtime_error("weight storage modified");
  check(cudaFree(dx)); check(cudaFree(dy)); check(cudaFree(dw));
  check(cudaStreamDestroy(stream));
  std::printf("gpu=%d groups=%d rank=%d width=%d tokens=%d mismatches=%zu/%zu max_abs=%.9g\n",
      device, groups, rank, width, tokens, mismatches, ref.size(), max_abs);
  if (mismatches) throw std::runtime_error("GPU dot differs from native AVX-512 dot");
}

int main() {
  try {
    ggml_cpu_init();
    int devices = 0; check(cudaGetDeviceCount(&devices));
    if (!devices) throw std::runtime_error("no CUDA device");
    for (int device = 0; device < std::min(devices, 2); ++device) {
      run(device, 3, 7, 256);
      run(device, 2, 19, 768);
      run(device, 8, 1024, 4096);
      run(device, 3, 7, 256, 3);
      run(device, 8, 128, 4096, 17);
    }
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "FAIL: %s\n", e.what());
    return 1;
  }
}
