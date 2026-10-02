// Bit-exactness check for MUL_MAT_ID across thread counts / env modes (real random quantized weights and activations,
// 4 quant types incl. ne11=10). Dump outputs per mode and `cmp` them, e.g.:
//   for n in 1 4 12; do MOESTREAM_MMID_PARALLEL_QUANT=1 ./mmid_exact_check $n old_$n.bin; ./mmid_exact_check $n new_$n.bin; cmp old_$n.bin new_$n.bin; done
// Build: same flags as mmid_decode_scaling_bench.cpp.
// Run a decode-shaped MUL_MAT_ID with real random quantized weights/activations, dump the output; compare across env modes.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
int main(int argc, char **argv) {
  const int K = 2560, R = 192, NE = 24, TOPK = 10; const int nth = atoi(argv[1]); const char *outp = argv[2];
  ggml_cpu_init();
  ggml_backend_t be = ggml_backend_cpu_init(); ggml_backend_cpu_set_n_threads(be, nth);
  struct Case { ggml_type t; int ne11; int k; } cases[] = {{GGML_TYPE_IQ1_M, 1, 2560}, {GGML_TYPE_IQ4_NL, 10, 640}, {GGML_TYPE_Q8_0, 1, 2560}, {GGML_TYPE_IQ2_XXS, 1, 2560}};
  FILE *f = fopen(outp, "wb");
  std::mt19937 rng(11); std::normal_distribution<float> nd(0, 1);
  for (auto &c : cases) {
    ggml_init_params ip = {ggml_tensor_overhead() * 32 + ggml_graph_overhead_custom(32, false) + 1024, nullptr, true};
    ggml_context *ctx = ggml_init(ip);
    ggml_tensor *W = ggml_new_tensor_3d(ctx, c.t, c.k, R, NE);
    ggml_tensor *x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, c.k, c.ne11, 1);
    ggml_tensor *ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, TOPK, 1);
    ggml_tensor *y = ggml_mul_mat_id(ctx, W, x, ids);
    ggml_cgraph *g = ggml_new_graph_custom(ctx, 32, false); ggml_build_forward_expand(g, y);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    ggml_quantize_init(c.t);
    std::vector<float> src((size_t)R * c.k); std::vector<float> imat(c.k, 1.f);
    size_t rs = ggml_row_size(c.t, c.k); std::vector<uint8_t> wq(rs * R);
    for (int e = 0; e < NE; ++e) {
      for (auto &v : src) v = nd(rng);
      ggml_quantize_chunk(c.t, src.data(), wq.data(), 0, R, c.k, imat.data());
      ggml_backend_tensor_set(W, wq.data(), (size_t)e * rs * R, rs * R);
    }
    std::vector<float> xv((size_t)c.k * c.ne11); for (auto &v : xv) v = nd(rng);
    ggml_backend_tensor_set(x, xv.data(), 0, xv.size() * 4);
    int32_t iv[TOPK] = {3, -1, 17, -1, -1, 8, -1, -1, -1, 21}; ggml_backend_tensor_set(ids, iv, 0, sizeof(iv));
    ggml_backend_graph_compute(be, g);
    std::vector<float> out(ggml_nelements(y)); ggml_backend_tensor_get(y, out.data(), 0, out.size() * 4);
    fwrite(out.data(), 4, out.size(), f);
    ggml_backend_buffer_free(buf); ggml_free(ctx);
  }
  fclose(f);
}
