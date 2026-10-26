#include "qwen4exp_graph_ops.hpp"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {
void check(bool ok, const char *what) {
  if (!ok) throw std::runtime_error(what);
}

void hc_test() {
  std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(
      ggml_init({2 << 20, nullptr, true}), ggml_free);
  auto *c = ctx.get();
  auto *x = ggml_new_tensor_3d(c, GGML_TYPE_F32, 4, 2, 1);
  auto *norm = ggml_new_tensor_1d(c, GGML_TYPE_F32, 8);
  auto *down = ggml_new_tensor_2d(c, GGML_TYPE_F32, 8, 2);
  auto *up = ggml_new_tensor_2d(c, GGML_TYPE_F32, 2, 8);
  auto *inject = ggml_new_tensor_2d(c, GGML_TYPE_F32, 8, 2);
  const qwen4exp_ops::HcShape shape{4, 2, 1, 1e-5f};
  ggml_tensor *injection = nullptr;
  auto *mixed = qwen4exp_ops::hc_mix(
      shape, c, x, norm, down, up, inject, &injection);
  auto *combined = qwen4exp_ops::hc_combine(shape, c, x, mixed, injection);
  auto *graph = ggml_new_graph(c);
  ggml_build_forward_expand(graph, combined);
  std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(
      ggml_backend_cpu_init(), ggml_backend_free);
  auto *buffer = ggml_backend_alloc_ctx_tensors(c, backend.get());
  check(buffer, "HC allocation");
  std::vector<float> nx(8), gains(8), dw(16), uw(16), iw(16);
  for (size_t i = 0; i < nx.size(); ++i) {
    nx[i] = (static_cast<int>(i) - 3) * 0.25f;
    gains[i] = 1.0f + i * 0.125f;
  }
  for (size_t i = 0; i < dw.size(); ++i) {
    dw[i] = (static_cast<int>(i % 5) - 2) * 0.125f;
    uw[i] = (static_cast<int>(i % 3) - 1) * 0.25f;
    iw[i] = (static_cast<int>(i % 7) - 3) * 0.125f;
  }
  const auto set = [](ggml_tensor *t, const std::vector<float> &v) {
    ggml_backend_tensor_set(t, v.data(), 0, v.size() * sizeof(float));
  };
  set(x, nx); set(norm, gains); set(down, dw); set(up, uw); set(inject, iw);
  check(ggml_backend_graph_compute(backend.get(), graph) == GGML_STATUS_SUCCESS,
        "HC compute");
  std::vector<double> normalized(8), lo(2), gate(8), inj(2), ref(4);
  for (int h = 0; h < 2; ++h) {
    double squares = 0;
    for (int d = 0; d < 4; ++d) squares += double(nx[h * 4 + d]) * nx[h * 4 + d];
    for (int d = 0; d < 4; ++d)
      normalized[h * 4 + d] =
          nx[h * 4 + d] * gains[h * 4 + d] / std::sqrt(squares / 4 + shape.eps);
  }
  for (int r = 0; r < 2; ++r) {
    for (int j = 0; j < 8; ++j) {
      lo[r] += dw[r * 8 + j] * normalized[j] / 2;
      inj[r] += iw[r * 8 + j] * normalized[j];
    }
    lo[r] /= 1 + std::exp(-lo[r]);
  }
  for (int j = 0; j < 8; ++j) {
    for (int r = 0; r < 2; ++r) gate[j] += uw[j * 2 + r] * lo[r];
    ref[j % 4] += normalized[j] / (1 + std::exp(-gate[j])) / 2;
  }
  std::vector<float> actual(8);
  ggml_backend_tensor_get(combined, actual.data(), 0, actual.size() * sizeof(float));
  for (int j = 0; j < 8; ++j) {
    const double expected = nx[j] + ref[j % 4] * 2 / (1 + std::exp(-inj[j / 4] / 2));
    check(std::abs(actual[j] - expected) < 1e-4, "HC float64 parity");
  }
  ggml_backend_buffer_free(buffer);
}

void attention_test() {
  std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(
      ggml_init({4 << 20, nullptr, true}), ggml_free);
  auto *c = ctx.get();
  const auto matrix = [c](int in, int out) {
    return ggml_new_tensor_2d(c, GGML_TYPE_F32, in, out);
  };
  auto *input = matrix(8, 1);
  auto *q = matrix(8, 16), *k = matrix(8, 4), *v = matrix(8, 4), *o = matrix(8, 8);
  auto *qn = ggml_new_tensor_1d(c, GGML_TYPE_F32, 4);
  auto *kn = ggml_dup_tensor(c, qn);
  auto *keys = matrix(4, 32), *values = matrix(4, 32);
  auto *positions = ggml_new_tensor_1d(c, GGML_TYPE_I32, 1);
  auto *indices = ggml_dup_tensor(c, positions);
  auto *mask = ggml_new_tensor_2d(c, GGML_TYPE_F16, 32, 1);
  auto *output = qwen4exp_ops::attention_decode(
      {2, 1, 4, 32, 4, 32, 1e-5f, 10000, 1, 0, 1}, c,
      input, q, k, v, qn, kn, o, keys, values, positions, indices, mask);
  auto *graph = ggml_new_graph(c);
  ggml_build_forward_expand(graph, output);
  std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(
      ggml_backend_cpu_init(), ggml_backend_free);
  auto *buffer = ggml_backend_alloc_ctx_tensors(c, backend.get());
  check(buffer, "attention allocation");
  ggml_backend_buffer_clear(buffer, 0);
  std::vector<float> vw(32), ow(64), gains(4, 1);
  for (int i = 0; i < 4; ++i) vw[i * 8 + i] = 1;
  for (int i = 0; i < 8; ++i) ow[i * 8 + i] = 1;
  ggml_backend_tensor_set(v, vw.data(), 0, vw.size() * sizeof(float));
  ggml_backend_tensor_set(o, ow.data(), 0, ow.size() * sizeof(float));
  for (auto *gain : {qn, kn})
    ggml_backend_tensor_set(gain, gains.data(), 0, gains.size() * sizeof(float));
  for (int repeat = 0; repeat < 2; ++repeat) {
    std::vector<float> zero(128);
    ggml_backend_tensor_set(keys, zero.data(), 0, zero.size() * sizeof(float));
    ggml_backend_tensor_set(values, zero.data(), 0, zero.size() * sizeof(float));
    std::vector<double> sum(4);
    for (int position = 0; position < 32; ++position) {
      std::vector<float> in(8);
      std::vector<ggml_fp16_t> causal(32);
      for (int d = 0; d < 8; ++d) in[d] = (position + d) * 0.125f;
      for (int d = 0; d < 4; ++d) sum[d] += in[d];
      for (int p = 0; p < 32; ++p)
        causal[p] = ggml_fp32_to_fp16(p <= position ? 0 : -INFINITY);
      const int32_t index = position;
      ggml_backend_tensor_set(input, in.data(), 0, in.size() * sizeof(float));
      ggml_backend_tensor_set(positions, &index, 0, sizeof(index));
      ggml_backend_tensor_set(indices, &index, 0, sizeof(index));
      ggml_backend_tensor_set(mask, causal.data(), 0, causal.size() * sizeof(ggml_fp16_t));
      check(ggml_backend_graph_compute(backend.get(), graph) == GGML_STATUS_SUCCESS,
            "attention compute");
      std::vector<float> actual(8);
      ggml_backend_tensor_get(output, actual.data(), 0, actual.size() * sizeof(float));
      for (int d = 0; d < 8; ++d)
        check(std::abs(actual[d] - sum[d % 4] / (position + 1) / 2) < 1e-4,
              "attention persistent KV/mask/reset parity");
    }
  }
  ggml_backend_buffer_free(buffer);
}
} // namespace

int main() {
  try {
    hc_test();
    attention_test();
    std::puts("Qwen shared HC/attention graph operations: PASS");
    return 0;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "Qwen graph operations: %s\n", e.what());
    return 1;
  }
}
