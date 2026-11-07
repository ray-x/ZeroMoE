// Accuracy of the GPU Q5_K F32-activation matvec (the lane8 kernel used for
// DeepSeek attn_q_b with 1024 inputs, and the shapes of the other dense Q5_K
// projections) against a float64 reference of the dequantized weights, next to
// the project's CPU AVX-512 F32 dot. The GPU result must be no less accurate
// than the CPU reference (error budget: max relative error within 2x of the CPU's
// plus a small absolute floor).
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
extern "C" void launch_q5_K_matmul(float*, const float*, const void*, int, int, int);

static void check(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }

int main() {
  int fails = 0;
  for (int I : {1024, 2048, 4096, 8192}) {
    const int O = 512;
    std::mt19937 rng(7 + I);
    std::normal_distribution<float> nd(0.f, 1.f);
    const size_t row_bytes = (size_t)(I / 256) * sizeof(block_q5_K);
    std::vector<unsigned char> w((size_t)O * row_bytes);
    std::vector<float> x(I), src(I), dec(I);
    for (float& v : x) v = nd(rng);
    std::vector<double> ref(O);
    std::vector<float> cpu(O);
    for (int o = 0; o < O; ++o) {
      for (float& v : src) v = nd(rng) * (o % 5 == 0 ? 0.05f : 1.0f);
      auto* row = reinterpret_cast<block_q5_K*>(w.data() + (size_t)o * row_bytes);
      quantize_row_q5_K_ref(src.data(), row, I);
      dequantize_row_q5_K(row, dec.data(), I);
      long double acc = 0;
      for (int i = 0; i < I; ++i) acc += (long double)dec[i] * x[i];
      ref[o] = (double)acc;
      cpu[o] = dot_f32_avx512(x.data(), dec.data(), I);
    }
    float *dx, *dy; void* dw;
    check(cudaMalloc(&dx, I * 4)); check(cudaMalloc(&dy, O * 4)); check(cudaMalloc(&dw, w.size()));
    check(cudaMemcpy(dx, x.data(), I * 4, cudaMemcpyHostToDevice));
    check(cudaMemcpy(dw, w.data(), w.size(), cudaMemcpyHostToDevice));
    launch_q5_K_matmul(dy, dx, dw, 1, I, O);
    check(cudaDeviceSynchronize());
    std::vector<float> gpu(O);
    check(cudaMemcpy(gpu.data(), dy, O * 4, cudaMemcpyDeviceToHost));
    double scale = 0; for (double r : ref) scale = std::max(scale, std::fabs(r));
    double eg = 0, ec = 0;
    for (int o = 0; o < O; ++o) {
      eg = std::max(eg, std::fabs(gpu[o] - ref[o]));
      ec = std::max(ec, std::fabs(cpu[o] - ref[o]));
    }
    const bool ok = eg <= 2.0 * ec + 1e-6 * scale;
    std::printf("I=%d O=%d  max|gpu-f64|=%.3e  max|cpu_avx512-f64|=%.3e  scale=%.3e  %s\n",
                I, O, eg, ec, scale, ok ? "ok" : "WORSE THAN CPU");
    fails += !ok;
    cudaFree(dx); cudaFree(dy); cudaFree(dw);
  }
  std::printf(fails ? "FAIL\n" : "PASS\n");
  return fails ? 1 : 0;
}
