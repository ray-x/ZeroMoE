#include "cold_ffn_fused.hpp"

#include "ggml-cpu.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <vector>

#if defined(__x86_64__)
#include <immintrin.h>
#define COLD_PAUSE() _mm_pause()
#else
#define COLD_PAUSE() ((void)0)
#endif

extern "C" void ggml_vec_swiglu_f32(const int n, float *y, const float *x, const float *g);

namespace qwen4exp_cold {

namespace {
constexpr int kMaxSlots = 64;       // topk * S upper bound handled on the stack
constexpr int kMaxExperts = 64;     // distinct cold experts per call
constexpr int kRows1 = 32;          // gate/up rows per phase-1 unit (a whole Q8_0 block of the activation)
constexpr int kRows2 = 16;          // down rows per phase-2 unit (ggml's blck_0)
constexpr int kMaxThreads = 64;
}  // namespace

struct Context {
  int D = 0, FF = 0, topk = 0, S = 0;
  ggml_type wtype = GGML_TYPE_Q2_0;
  ggml_type atype = GGML_TYPE_Q8_0;
  size_t xq_row = 0, aq_row = 0;
  int threads = 1;           // thread-private buffers exist for this many threads
  ggml_vec_dot_t dot = nullptr;
  ggml_from_float_t quant = nullptr;
  std::vector<uint8_t> xq;   // [thread][S][xq_row] private copies of the quantized layer input
  std::vector<uint8_t> aq;   // [slot][aq_row] quantized activation per routed slot (shared between phases)
  alignas(64) std::atomic<int> arrived{0};
  alignas(64) std::atomic<int> generation{0};

  void barrier(int nth) {
    const int gen = generation.load(std::memory_order_acquire);
    if (arrived.fetch_add(1, std::memory_order_acq_rel) + 1 == nth) {
      arrived.store(0, std::memory_order_relaxed);
      generation.fetch_add(1, std::memory_order_release);
    } else {
      while (generation.load(std::memory_order_acquire) == gen) COLD_PAUSE();
    }
  }
};

bool supported(const ggml_tensor *gate, const ggml_tensor *up, const ggml_tensor *down, int topk, int n_tokens) {
  if (!gate || !up || !down || topk <= 0 || n_tokens <= 0 || topk * n_tokens > kMaxSlots) return false;
  if (gate->type != GGML_TYPE_Q2_0 || up->type != GGML_TYPE_Q2_0 || down->type != GGML_TYPE_Q2_0) return false;
  const int64_t D = gate->ne[0], FF = gate->ne[1];
  if (up->ne[0] != D || up->ne[1] != FF || down->ne[0] != FF || down->ne[1] != D) return false;
  if (gate->ne[2] != up->ne[2] || gate->ne[2] != down->ne[2]) return false;
  if (FF % kRows1 != 0 || D % kRows2 != 0) return false;
  if (D % ggml_blck_size(GGML_TYPE_Q2_0) != 0 || FF % ggml_blck_size(GGML_TYPE_Q2_0) != 0) return false;
  const auto *t = ggml_get_type_traits_cpu(GGML_TYPE_Q2_0);
  if (!t || !t->vec_dot || t->vec_dot_type != GGML_TYPE_Q8_0) return false;
  return true;
}

namespace {

void run(ggml_tensor *dst, int ith, int nth, void *userdata) {
  auto &c = *static_cast<Context *>(userdata);
  const ggml_tensor *x = dst->src[0], *ids = dst->src[1], *w = dst->src[2], *remap = dst->src[3],
                    *mask = dst->src[4], *gate = dst->src[5], *up = dst->src[6], *down = dst->src[7];
  const int D = c.D, FF = c.FF, topk = c.topk, S = c.S, nslots = topk * S;
  nth = std::min(nth, c.threads);
  if (ith >= nth) return;  // cannot happen (ggml passes the real count) but keep the barrier arithmetic honest

  // Cold slots and the distinct cold experts, built identically by every thread (a few dozen ints).
  int slot_expert[kMaxSlots];
  float slot_w[kMaxSlots];
  int experts[kMaxExperts], n_exp = 0;
  int exp_slots[kMaxExperts][kMaxSlots], exp_count[kMaxExperts];
  const int32_t *remap_d = static_cast<const int32_t *>(remap->data);
  const float *mask_d = static_cast<const float *>(mask->data);
  for (int t = 0; t < S; ++t)
    for (int k = 0; k < topk; ++k) {
      const int j = t * topk + k;  // slot index in the output (dst row k of token t)
      const int32_t id = *reinterpret_cast<const int32_t *>(static_cast<const char *>(ids->data) +
                                                            k * ids->nb[0] + t * ids->nb[1]);
      const int32_t e = id >= 0 ? remap_d[id] : -1;
      slot_expert[j] = e;
      if (e < 0) continue;
      const float wk = *reinterpret_cast<const float *>(static_cast<const char *>(w->data) +
                                                       k * w->nb[1] + t * w->nb[2]);
      slot_w[j] = wk * mask_d[id];
      int u = 0;
      while (u < n_exp && experts[u] != e) ++u;
      if (u == n_exp) {
        if (n_exp == kMaxExperts) return;  // unreachable: topk * S <= kMaxSlots
        experts[n_exp] = e;
        exp_count[n_exp++] = 0;
      }
      exp_slots[u][exp_count[u]++] = j;
    }

  // Zero the slots that have no cold expert (round-robin over threads).
  for (int j = ith; j < nslots; j += nth)
    if (slot_expert[j] < 0) {
      const int t = j / topk, k = j % topk;
      std::memset(static_cast<char *>(dst->data) + k * dst->nb[1] + t * dst->nb[2], 0, D * sizeof(float));
    }

  if (n_exp > 0) {
    // Private quantized copy of x for the tokens that have a cold slot.
    uint8_t *xq = c.xq.data() + static_cast<size_t>(ith) * S * c.xq_row;
    bool token_cold[kMaxSlots] = {};
    for (int j = 0; j < nslots; ++j)
      if (slot_expert[j] >= 0) token_cold[j / topk] = true;
    for (int t = 0; t < S; ++t)
      if (token_cold[t])
        c.quant(reinterpret_cast<const float *>(static_cast<const char *>(x->data) + t * x->nb[2]),
                xq + static_cast<size_t>(t) * c.xq_row, D);

    // ---- phase 1: gate/up -> SwiGLU -> quantized activation, per (expert, 32 rows) ----
    const int chunks1 = FF / kRows1;
    for (int u = ith; u < n_exp * chunks1; u += nth) {
      const int ei = u / chunks1, r0 = (u % chunks1) * kRows1;
      const char *gbase = static_cast<const char *>(gate->data) + experts[ei] * gate->nb[2] + r0 * gate->nb[1];
      const char *ubase = static_cast<const char *>(up->data) + experts[ei] * up->nb[2] + r0 * up->nb[1];
      for (int n = 0; n < exp_count[ei]; ++n) {
        const int j = exp_slots[ei][n], t = j / topk;
        const uint8_t *xt = xq + static_cast<size_t>(t) * c.xq_row;
        float g[kRows1], uu[kRows1], act[kRows1];
        for (int i = 0; i + 3 < kRows1; i += 4) {
          c.dot(D, &g[i], sizeof(float), gbase + i * gate->nb[1], gate->nb[1], xt, 0, 4);
          c.dot(D, &uu[i], sizeof(float), ubase + i * up->nb[1], up->nb[1], xt, 0, 4);
        }
        ggml_vec_swiglu_f32(kRows1, act, g, uu);
        c.quant(act, c.aq.data() + static_cast<size_t>(j) * c.aq_row + (r0 / 32) * ggml_type_size(c.atype), kRows1);
      }
    }
  }

  c.barrier(nth);
  if (n_exp == 0) return;

  // ---- phase 2: down projection scaled by the routing weight, per (expert, 16 rows) ----
  const int chunks2 = D / kRows2;
  for (int u = ith; u < n_exp * chunks2; u += nth) {
    const int ei = u / chunks2, r0 = (u % chunks2) * kRows2;
    const char *dbase = static_cast<const char *>(down->data) + experts[ei] * down->nb[2] + r0 * down->nb[1];
    for (int n = 0; n < exp_count[ei]; ++n) {
      const int j = exp_slots[ei][n], t = j / topk, k = j % topk;
      const uint8_t *aq = c.aq.data() + static_cast<size_t>(j) * c.aq_row;
      float tmp[kRows2];
      for (int i = 0; i + 3 < kRows2; i += 4)
        c.dot(FF, &tmp[i], sizeof(float), dbase + i * down->nb[1], down->nb[1], aq, 0, 4);
      float *out = reinterpret_cast<float *>(static_cast<char *>(dst->data) + k * dst->nb[1] + t * dst->nb[2]) + r0;
      for (int i = 0; i < kRows2; ++i) out[i] = tmp[i] * slot_w[j];
    }
  }
}

}  // namespace

ggml_tensor *build(ggml_context *ctx, std::shared_ptr<Context> &owner, ggml_tensor *x, ggml_tensor *ids,
                   ggml_tensor *w, ggml_tensor *remap, ggml_tensor *mask, ggml_tensor *gate, ggml_tensor *up,
                   ggml_tensor *down, int max_threads) {
  const int D = int(gate->ne[0]), FF = int(gate->ne[1]);
  const int topk = int(ids->ne[0]), S = int(ids->ne[1]);
  if (!supported(gate, up, down, topk, S) || x->ne[0] != D || x->ne[2] != S) return nullptr;
  auto c = std::make_shared<Context>();
  c->D = D; c->FF = FF; c->topk = topk; c->S = S;
  const auto *traits = ggml_get_type_traits_cpu(GGML_TYPE_Q2_0);
  c->dot = traits->vec_dot;
  c->quant = ggml_get_type_traits_cpu(c->atype)->from_float;
  if (!c->quant) return nullptr;
  c->xq_row = ggml_row_size(c->atype, D);
  c->aq_row = ggml_row_size(c->atype, FF);
  const int threads = std::clamp(max_threads, 1, kMaxThreads);
  c->threads = threads;
  c->xq.assign(static_cast<size_t>(threads) * S * c->xq_row, 0);
  c->aq.assign(static_cast<size_t>(topk) * S * c->aq_row, 0);
  owner = c;
  ggml_tensor *args[8] = {x, ids, w, remap, mask, gate, up, down};
  ggml_tensor *out = ggml_custom_4d(ctx, GGML_TYPE_F32, D, topk, S, 1, args, 8, run, GGML_N_TASKS_MAX, c.get());
  return out;
}

}  // namespace qwen4exp_cold
