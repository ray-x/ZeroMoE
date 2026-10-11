// Synthetic parity: MoEStream's own EXL3 mul1 matmul vs upstream ExLlamaV3's
// exl3_gemm (compiled through the torch-free stub). Random trellis words are
// valid EXL3 data, so no checkpoint is needed.
#include "exl3_cuda.hpp"
#include "exl3_stub.hpp"
#include "quant/exl3_gemm.cuh"

#include <cuda_fp16.h>

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  std::printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); return 2; } } while (0)

static std::vector<uint16_t> rand_half(std::mt19937 &g, size_t n, bool signs) {
  std::vector<uint16_t> v(n);
  std::uniform_real_distribution<float> u(0.5f, 1.5f);
  for (auto &x : v) {
    float f = signs ? ((g() & 1) ? 1.0f : -1.0f) : u(g) * 0.1f;
    x = __half_as_ushort(__float2half(f));
  }
  return v;
}

int run(int bits, int k, int n, int rows) {
  std::mt19937 g(1234 + bits * 31 + rows);
  const size_t trellis_words = static_cast<size_t>(k) / 16 * (n / 16) * 16 * bits;
  std::vector<uint16_t> trellis(trellis_words);
  for (auto &w : trellis) w = static_cast<uint16_t>(g());
  auto suh = rand_half(g, k, true), svh = rand_half(g, n, true);
  std::vector<float> in(static_cast<size_t>(rows) * k);
  std::normal_distribution<float> nd(0.f, 1.f);
  for (auto &x : in) x = nd(g);

  uint16_t *d_tr, *d_suh, *d_svh; float *d_in, *d_out_old, *d_out_new, *d_part;
  __half *d_in_h, *d_had;
  CK(cudaMalloc(&d_tr, trellis_words * 2));
  CK(cudaMalloc(&d_suh, k * 2)); CK(cudaMalloc(&d_svh, n * 2));
  CK(cudaMalloc(&d_in, in.size() * 4)); CK(cudaMalloc(&d_in_h, in.size() * 2));
  CK(cudaMalloc(&d_had, in.size() * 2));
  CK(cudaMalloc(&d_out_old, static_cast<size_t>(rows) * n * 4));
  CK(cudaMalloc(&d_out_new, static_cast<size_t>(rows) * n * 4));
  const size_t part = static_cast<size_t>(exl3_cuda_key_splits(k)) * rows * n * 4;
  CK(cudaMalloc(&d_part, part));
  CK(cudaMemcpy(d_tr, trellis.data(), trellis_words * 2, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_suh, suh.data(), k * 2, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_svh, svh.data(), n * 2, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_in, in.data(), in.size() * 4, cudaMemcpyHostToDevice));
  std::vector<__half> inh(in.size());
  for (size_t i = 0; i < in.size(); ++i) inh[i] = __float2half(in[i]);
  CK(cudaMemcpy(d_in_h, inh.data(), in.size() * 2, cudaMemcpyHostToDevice));

  Exl3CudaMatrixView mv;
  mv.trellis = d_tr; mv.suh = d_suh; mv.svh = d_svh;
  mv.input_features = k; mv.output_features = n; mv.bits = bits;
  CK(launch_exl3_mul1_matmul(mv, d_in, d_out_old, rows, d_part, nullptr));
  CK(cudaDeviceSynchronize());

  using at::Tensor;
  Tensor A(d_in_h, {rows, k}, at::kHalf, 0);
  Tensor B(d_tr, {k / 16, n / 16, 16 * bits}, at::kShort, 0);
  Tensor C(d_out_new, {rows, n}, at::kFloat, 0);
  c10::optional<Tensor> osuh = Tensor(d_suh, {k / 16}, at::kHalf, 0);
  c10::optional<Tensor> ohad = Tensor(d_had, {rows, k}, at::kHalf, 0);
  c10::optional<Tensor> osvh = Tensor(d_svh, {n / 16}, at::kHalf, 0);
  moestream_exl3::set_current_stream(nullptr);
  const int shape = exl3_gemm(A, B, C, osuh, ohad, osvh, 0, false, true, 0);
  CK(cudaDeviceSynchronize());

  std::vector<float> a(static_cast<size_t>(rows) * n), b(a.size());
  CK(cudaMemcpy(a.data(), d_out_old, a.size() * 4, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(b.data(), d_out_new, b.size() * 4, cudaMemcpyDeviceToHost));
  double num = 0, den = 0, maxabs = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    const double d = a[i] - b[i];
    num += d * d; den += static_cast<double>(a[i]) * a[i];
    maxabs = std::max(maxabs, std::fabs(d));
  }
  const double rel = std::sqrt(num / (den > 0 ? den : 1));
  std::printf("bits=%d k=%d n=%d rows=%d shape=%d rel_rms=%.3e max_abs=%.3e %s\n",
              bits, k, n, rows, shape, rel, maxabs, rel < 5e-3 ? "OK" : "MISMATCH");
  cudaFree(d_tr); cudaFree(d_suh); cudaFree(d_svh); cudaFree(d_in); cudaFree(d_in_h);
  cudaFree(d_had); cudaFree(d_out_old); cudaFree(d_out_new); cudaFree(d_part);
  return rel < 5e-3 ? 0 : 1;
}

int main() {
  int bad = 0;
  for (int bits : {2, 3, 4, 6})
    for (int rows : {1, 4, 32})
      bad += run(bits, 1024, 2048, rows);
  std::printf(bad ? "FAILED (%d)\n" : "PASSED\n", bad);
  return bad ? 1 : 0;
}
