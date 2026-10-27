// Bit-exactness test: the fused cold-expert op must equal the stock ggml chain (get_rows remap/mask, mul,
// MUL_MAT_ID up/gate, SWIGLU, MUL_MAT_ID down, mul) bit for bit, for several token counts, thread counts,
// cold patterns (none, some, all, shared experts) and random quantized weights.
#include "cold_ffn_fused.hpp"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {
constexpr int D = 256, FF = 128, E = 24, TOPK = 10;

void fill_q2_0(std::mt19937 &rng, ggml_tensor *t) {
  const auto *tr = ggml_get_type_traits(GGML_TYPE_Q2_0);
  std::normal_distribution<float> n(0.0f, 0.4f);
  const int64_t rows = t->ne[1] * t->ne[2];
  std::vector<float> row(t->ne[0]);
  std::vector<char> buf(ggml_row_size(GGML_TYPE_Q2_0, t->ne[0]));
  for (int64_t r = 0; r < rows; ++r) {
    for (auto &v : row) v = n(rng);
    tr->from_float_ref(row.data(), buf.data(), t->ne[0]);
    ggml_backend_tensor_set(t, buf.data(), r * buf.size(), buf.size());
  }
}
}  // namespace

int main() {
  int failures = 0, cases = 0;
  std::mt19937 rng(11);
  for (int threads : {1, 2, 3, 6, 10})
    for (int S : {1, 2, 4})
      for (int pattern = 0; pattern < 4; ++pattern) {
        ggml_backend_t be = ggml_backend_cpu_init();
        ggml_backend_cpu_set_n_threads(be, threads);
        ggml_init_params ip{ggml_tensor_overhead() * 64 + ggml_graph_overhead_custom(64, false), nullptr, true};
        ggml_context *ctx = ggml_init(ip);
        auto *gate = ggml_new_tensor_3d(ctx, GGML_TYPE_Q2_0, D, FF, E);
        auto *up = ggml_new_tensor_3d(ctx, GGML_TYPE_Q2_0, D, FF, E);
        auto *down = ggml_new_tensor_3d(ctx, GGML_TYPE_Q2_0, FF, D, E);
        auto *x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, 1, S);
        auto *ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, TOPK, S);
        auto *w = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, TOPK, S);
        auto *remap = ggml_new_tensor_3d(ctx, GGML_TYPE_I32, 1, E, 1);
        auto *mask = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, E, 1);
        // stock chain
        auto *ids_flat = S == 1 ? ids : ggml_reshape_1d(ctx, ids, int64_t(TOPK) * S);
        auto *cold_ids = ggml_reshape_2d(ctx, ggml_get_rows(ctx, remap, ids_flat), TOPK, S);
        auto *mrows = ggml_get_rows(ctx, mask, ids_flat);
        auto *cold_w = ggml_mul(ctx, w, S == 1 ? mrows : ggml_reshape_3d(ctx, mrows, 1, TOPK, S));
        auto *u2 = ggml_mul_mat_id(ctx, up, x, cold_ids);
        auto *g2 = ggml_mul_mat_id(ctx, gate, x, cold_ids);
        auto *act = ggml_swiglu_split(ctx, g2, u2);
        auto *dn = ggml_mul_mat_id(ctx, down, act, cold_ids);
        auto *stock = ggml_mul(ctx, dn, cold_w);
        // fused
        std::shared_ptr<qwen4exp_cold::Context> owner;
        auto *fused = qwen4exp_cold::build(ctx, owner, x, ids, w, remap, mask, gate, up, down, threads);
        if (!fused) { std::fprintf(stderr, "FAIL: fused op refused the shapes\n"); return 1; }
        ggml_cgraph *gs = ggml_new_graph_custom(ctx, 64, false), *gf = ggml_new_graph_custom(ctx, 64, false);
        ggml_build_forward_expand(gs, stock);
        ggml_build_forward_expand(gf, fused);
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
        fill_q2_0(rng, gate); fill_q2_0(rng, up); fill_q2_0(rng, down);
        std::vector<float> xv(size_t(D) * S), wv(size_t(TOPK) * S);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        for (auto &v : xv) v = nd(rng);
        for (auto &v : wv) v = 0.02f + 0.3f * std::abs(nd(rng));
        std::vector<int32_t> iv(size_t(TOPK) * S), rm(E), cold(E);
        std::vector<float> mk(E);
        for (int e = 0; e < E; ++e) {
          bool is_cold = pattern == 0 ? false : pattern == 1 ? (e % 4 == 0) : pattern == 2 ? true : (e % 3 == 1);
          cold[e] = is_cold;
          rm[e] = is_cold ? e : -1;
          mk[e] = is_cold ? 1.0f : 0.0f;
        }
        for (int t = 0; t < S; ++t) {  // distinct experts per token, shared across tokens when pattern 3
          std::vector<int> perm(E);
          for (int e = 0; e < E; ++e) perm[e] = e;
          std::shuffle(perm.begin(), perm.end(), rng);
          for (int k = 0; k < TOPK; ++k) iv[size_t(t) * TOPK + k] = pattern == 3 ? (k * 2 + t) % E : perm[k];
        }
        ggml_backend_tensor_set(x, xv.data(), 0, ggml_nbytes(x));
        ggml_backend_tensor_set(w, wv.data(), 0, ggml_nbytes(w));
        ggml_backend_tensor_set(ids, iv.data(), 0, ggml_nbytes(ids));
        ggml_backend_tensor_set(remap, rm.data(), 0, ggml_nbytes(remap));
        ggml_backend_tensor_set(mask, mk.data(), 0, ggml_nbytes(mask));
        ggml_backend_graph_compute(be, gs);
        ggml_backend_graph_compute(be, gf);
        std::vector<float> a(size_t(D) * TOPK * S), b(a.size());
        ggml_backend_tensor_get(stock, a.data(), 0, a.size() * sizeof(float));
        ggml_backend_tensor_get(fused, b.data(), 0, b.size() * sizeof(float));
        ++cases;
        size_t bad = 0, nonzero = 0;
        for (size_t i = 0; i < a.size(); ++i) {
          uint32_t ua, ub; std::memcpy(&ua, &a[i], 4); std::memcpy(&ub, &b[i], 4);
          bad += ua != ub; nonzero += a[i] != 0.0f;
        }
        if (bad) {
          ++failures;
          std::fprintf(stderr, "FAIL threads=%d S=%d pattern=%d: %zu of %zu floats differ\n", threads, S, pattern, bad, a.size());
        }
        if (pattern == 0 && nonzero) { ++failures; std::fprintf(stderr, "FAIL: all-hot must be all zeros\n"); }
        if (pattern != 0 && !nonzero) { ++failures; std::fprintf(stderr, "FAIL: no cold output produced (test is vacuous)\n"); }
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        ggml_backend_free(be);
      }
  std::printf("cold ffn fused: %d cases, %s\n", cases, failures ? "FAIL" : "PASS (bit-identical to the stock graph)");
  return failures ? 1 : 0;
}
