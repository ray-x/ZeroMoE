// Is an expert plane viewed in place at a slot stride numerically identical to the same plane in a canonical
// contiguous tensor on the CUDA backend? And how far apart are the CPU and CUDA kernels on the same quantized data?
//
// Separates two causes of a phase-cache logit mismatch: a layout/stride bug in the aliased view (the aliased and
// contiguous CUDA results differ) from ordinary CPU-versus-GPU arithmetic (backends differ for any placement).
// Real quant types and shapes: Ling 3.1 Flash gate/up IQ1_S (+ early layers' IQ2_S), down IQ2_XXS; MiMo Q2_K/Q3_K.
// The slot is [gate|up|down]-style: the plane under test sits at plane_off inside a slot of `stride` bytes.
#include <cstdlib>
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

struct Case { const char *name; ggml_type type; int64_t K, N; };

static std::vector<uint8_t> quantize_one(const Case &c, std::mt19937 &rng) {
  std::normal_distribution<float> g(0.0f, 0.5f);
  std::vector<float> w(size_t(c.K) * size_t(c.N)), imat(size_t(c.K), 1.0f);
  for (auto &x : w) x = g(rng);
  ggml_quantize_init(c.type);
  std::vector<uint8_t> q(ggml_row_size(c.type, c.K) * size_t(c.N));
  ggml_quantize_chunk(c.type, w.data(), q.data(), 0, c.N, c.K, imat.data());
  return q;
}

// mul_mat_id over `slots` experts. aliased: weights are read in place from `dev_pool` (slot stride `stride`,
// plane at `plane_off`); otherwise they are packed tightly into a canonical tensor on `be`.
static std::vector<float> run(ggml_backend_t be, const Case &c, int64_t slots, const std::vector<uint8_t> &host,
                              size_t stride, size_t plane_off, bool aliased, void *dev_pool,
                              ggml_backend_buffer_t marker, const std::vector<int32_t> &ids, const std::vector<float> &x) {
  ggml_context *ctx = ggml_init({64ull << 20, nullptr, true});
  ggml_tensor *w = ggml_new_tensor_3d(ctx, c.type, c.K, c.N, slots);
  if (aliased) { w->nb[2] = stride; w->data = static_cast<uint8_t *>(dev_pool) + plane_off; w->buffer = marker; }
  ggml_tensor *b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, c.K, int64_t(ids.size()), 1);
  ggml_tensor *i = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, int64_t(ids.size()), 1);
  ggml_tensor *o = ggml_mul_mat_id(ctx, w, b, i);
  ggml_cgraph *gr = ggml_new_graph(ctx);
  ggml_build_forward_expand(gr, o);
  ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);  // skips tensors that already have a buffer
  if (!aliased) {
    const size_t per = ggml_row_size(c.type, c.K) * size_t(c.N);
    std::vector<uint8_t> packed(per * size_t(slots));
    for (int64_t e = 0; e < slots; ++e)
      std::memcpy(packed.data() + size_t(e) * per, host.data() + size_t(e) * stride + plane_off, per);
    ggml_backend_tensor_set(w, packed.data(), 0, packed.size());
  }
  ggml_backend_tensor_set(b, x.data(), 0, x.size() * 4);
  ggml_backend_tensor_set(i, ids.data(), 0, ids.size() * 4);
  ggml_backend_graph_compute(be, gr);
  std::vector<float> out(size_t(c.N) * ids.size());
  ggml_backend_tensor_get(o, out.data(), 0, out.size() * 4);
  ggml_backend_buffer_free(buf);
  ggml_free(ctx);
  return out;
}

// float64 ground truth: dequantize each used expert's rows and multiply by the float input.
static std::vector<float> reference(const Case &c, const std::vector<uint8_t> &host, size_t stride, size_t off,
                                    const std::vector<int32_t> &ids, const std::vector<float> &x) {
  const auto *tr = ggml_get_type_traits(c.type);
  std::vector<float> row(size_t(c.K)), out(size_t(c.N) * ids.size());
  const size_t rb = ggml_row_size(c.type, c.K);
  for (size_t t = 0; t < ids.size(); ++t)
    for (int64_t n = 0; n < c.N; ++n) {
      tr->to_float(host.data() + size_t(ids[t]) * stride + off + size_t(n) * rb, row.data(), c.K);
      double acc = 0;
      for (int64_t k = 0; k < c.K; ++k) acc += double(row[size_t(k)]) * x[t * size_t(c.K) + size_t(k)];
      out[t * size_t(c.N) + size_t(n)] = float(acc);
    }
  return out;
}

static double rel_l2(const std::vector<float> &a, const std::vector<float> &b) {
  double d = 0, n = 0;
  for (size_t i = 0; i < a.size(); ++i) { d += double(a[i] - b[i]) * (a[i] - b[i]); n += double(b[i]) * b[i]; }
  return std::sqrt(d / (n > 0 ? n : 1));
}

int main() {
  ggml_backend_t cuda = ggml_backend_cuda_init(0), cpu = ggml_backend_cpu_init();
  if (!cuda || !cpu) { std::fprintf(stderr, "no CUDA/CPU backend\n"); return 2; }
  ggml_backend_cpu_set_n_threads(cpu, 8);
  const Case cases[] = {{"Ling gate/up IQ1_S", GGML_TYPE_IQ1_S, 4096, 1280}, {"Ling down IQ2_XXS", GGML_TYPE_IQ2_XXS, 1280, 4096},
                        {"Ling early IQ2_S", GGML_TYPE_IQ2_S, 4096, 1280}, {"MiMo Q2_K", GGML_TYPE_Q2_K, 4096, 1536},
                        {"MiMo Q3_K", GGML_TYPE_Q3_K, 4096, 1536}};
  std::mt19937 rng(7);
  std::normal_distribution<float> gx(0.0f, 1.0f);
  int layout_failures = 0;
  ggml_backend_buffer_t marker = ggml_backend_alloc_buffer(cuda, 256);
  ggml_backend_buffer_set_usage(marker, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
  for (const Case &c : cases) {
    const int64_t slots = 6;
    const size_t plane = ggml_row_size(c.type, c.K) * size_t(c.N), ts = ggml_type_size(c.type);
    const size_t unit = ts / std::gcd(ts, size_t(16)) * 16;   // stride granule that keeps nb[2] a multiple of the block size
    std::vector<float> x(size_t(c.K) * 4);
    for (auto &v : x) v = gx(rng);
    if (std::getenv("EXACT_INPUTS")) {   // integers in [-127,127] with +-127 in every 32-block: Q8_0/Q8_1/Q8_K rounding is lossless
      for (size_t k = 0; k < x.size(); ++k) x[k] = (k % 32 == 0) ? 127.0f : std::round(gx(rng) * 40.0f);
      for (auto &v : x) v = std::max(-127.0f, std::min(127.0f, v));
    }
    const std::vector<int32_t> ids = {1, 4, 2, 5};
    struct V { const char *what; size_t off, pad; };
    for (V v : {V{"tight, plane at 0", 0, 0}, V{"plane at +1 plane, row-LCM padded", plane, 3 * unit},
                V{"plane at +2 planes (down), padded", 2 * plane, unit}, V{"stride NOT a multiple of block size", 0, 13}}) {
      const size_t stride = v.off + plane + v.pad;
      std::vector<uint8_t> host(stride * size_t(slots), 0xA5);   // 0xA5 filler between planes
      for (int64_t e = 0; e < slots; ++e) { auto q = quantize_one(c, rng); std::memcpy(host.data() + size_t(e) * stride + v.off, q.data(), q.size()); }
      void *dev = nullptr;
      cudaMalloc(&dev, host.size() + 4096);
      cudaMemcpy(dev, host.data(), host.size(), cudaMemcpyHostToDevice);
      const auto contig = run(cuda, c, slots, host, stride, v.off, false, nullptr, marker, ids, x);
      const auto alias = run(cuda, c, slots, host, stride, v.off, true, dev, marker, ids, x);
      const auto cpuo = run(cpu, c, slots, host, stride, v.off, false, nullptr, marker, ids, x);
      if (v.off == 0 && v.pad == 0) {
        const auto ref = reference(c, host, stride, v.off, ids, x);
        const double ec = rel_l2(cpuo, ref), eg = rel_l2(contig, ref);
        std::printf("   vs float64 dequant reference: CPU %.3e  CUDA %.3e\n", ec, eg);
        // With activations that 8-bit rounding represents exactly, the kernels must match exact math; any larger
        // error is a kernel bug, not the activation-rounding difference between the backends.
        if (std::getenv("EXACT_INPUTS") && (ec > 1e-4 || eg > 1e-4)) { std::printf("   KERNEL ERROR above 1e-4\n"); ++layout_failures; }
      }
      const bool same = std::memcmp(contig.data(), alias.data(), contig.size() * 4) == 0;
      const bool granular = stride % ts == 0;
      std::printf("%-22s %-38s stride=%-9zu %%type=%zu: alias==contig(CUDA) %-3s | CPU-vs-CUDA rel L2 %.3e\n", c.name, v.what,
                  stride, stride % ts, same ? "yes" : "NO", rel_l2(cpuo, contig));
      if (!same && granular) ++layout_failures;
      cudaFree(dev);
    }
  }
  std::printf(layout_failures ? "phase_alias_numerics: LAYOUT MISMATCH\n" : "phase_alias_numerics: aliased == contiguous on every block-aligned stride\n");
  return layout_failures ? 1 : 0;
}
