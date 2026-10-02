// Decode-shaped MUL_MAT_ID thread scaling on the ggml CPU backend (gate/up IQ1_M-like + down IQ4_NL, 10 route slots, N valid).
// Used to find that quantizing a small src1 split across threads cost far more than the quantization itself
// (see tests/option_b_split_reverify_howto.md section 15). Env: GATE_TYPE=iq1_m|iq4_nl|iq2_xxs|q4_0|q8_0, GATE_ROWS=<rows>.
// Build: g++ -O2 -std=c++17 bench/mmid_decode_scaling_bench.cpp -I<ggml>/include -L<build>/ggml-build/src -lggml-cpu -lggml-base -lggml \
//        -Wl,-rpath,<build>/ggml-build/src -o mmid_decode_scaling_bench
// Run:   OMP_WAIT_POLICY=active GOMP_SPINCOUNT=10000000 ./mmid_decode_scaling_bench 2     (arg = number of valid experts)
// Per-op thread scaling of the CPU backend on decode-shaped MUL_MAT_ID (IQ1_M gate/up, IQ4_NL down), 2 valid experts of 10 slots.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
static double now_us() { return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
int main(int argc, char **argv) {
  const int D = 2560, F = 640, NE = 512, TOPK = 10; const int iters = 600;
  const int nvalid = argc > 1 ? atoi(argv[1]) : 2;
  ggml_backend_t be = ggml_backend_cpu_init();
  ggml_init_params ip = {ggml_tensor_overhead() * 64 + ggml_graph_overhead_custom(64, false) + 1024, nullptr, true};
  ggml_context *ctx = ggml_init(ip);
  const char *gt = getenv("GATE_TYPE");
  ggml_type gtype = GGML_TYPE_IQ1_M; int grows = F, gK = D;
  if (gt) { std::string n = gt; if (n == "iq4_nl") gtype = GGML_TYPE_IQ4_NL; else if (n == "iq2_xxs") gtype = GGML_TYPE_IQ2_XXS; else if (n == "q4_0") gtype = GGML_TYPE_Q4_0; else if (n == "q8_0") gtype = GGML_TYPE_Q8_0; }
  if (getenv("GATE_ROWS")) grows = atoi(getenv("GATE_ROWS"));
  ggml_tensor *Wg = ggml_new_tensor_3d(ctx, gtype, gK, grows, NE);
  ggml_tensor *Wd = ggml_new_tensor_3d(ctx, GGML_TYPE_IQ4_NL, F, D, NE);
  ggml_tensor *x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, 1, 1);
  ggml_tensor *xh = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, F, TOPK, 1);
  ggml_tensor *ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, TOPK, 1);
  ggml_tensor *gate = ggml_mul_mat_id(ctx, Wg, x, ids);
  ggml_tensor *down = ggml_mul_mat_id(ctx, Wd, xh, ids);
  ggml_tensor *glu = ggml_swiglu_split(ctx, gate, gate);
  const char *names[3] = {"gate mmid (640 rows, K=2560, IQ1_M)", "down mmid (2560 rows, K=640, IQ4_NL)", "swiglu"};
  ggml_tensor *outs[3] = {gate, down, glu};
  ggml_cgraph *gr[3];
  for (int i = 0; i < 3; ++i) { gr[i] = ggml_new_graph_custom(ctx, 64, false); ggml_build_forward_expand(gr[i], outs[i]); }
  ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
  ggml_backend_buffer_clear(buf, 0);
  std::vector<float> xv(D, 0.01f), xhv(F * TOPK, 0.01f); ggml_backend_tensor_set(x, xv.data(), 0, D * 4); ggml_backend_tensor_set(xh, xhv.data(), 0, F * TOPK * 4);
  std::vector<int32_t> iv(TOPK, -1); for (int i = 0; i < nvalid; ++i) iv[i] = 7 * i + 3; ggml_backend_tensor_set(ids, iv.data(), 0, TOPK * 4);
  printf("valid experts = %d\n%-40s", nvalid, "op");
  const int ths[] = {1, 2, 4, 6, 12};
  for (int t : ths) printf("  %2dT(us)", t);
  printf("   speedup12\n");
  for (int g = 0; g < 3; ++g) {
    printf("%-40s", names[g]); double t1 = 0, t12 = 0;
    for (int nth : ths) {
      ggml_backend_cpu_set_n_threads(be, nth);
      for (int w = 0; w < 30; ++w) ggml_backend_graph_compute(be, gr[g]);
      double t0 = now_us(); for (int i = 0; i < iters; ++i) ggml_backend_graph_compute(be, gr[g]); double us = (now_us() - t0) / iters;
      printf("  %8.1f", us); if (nth == 1) t1 = us; if (nth == 12) t12 = us;
    }
    printf("   %5.2fx\n", t1 / t12);
  }
}
