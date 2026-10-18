// qwen4exp_gdn.hpp — pure math for Qwen4Exp (Qwen3.8-Flash-Next) Gated
// DeltaNet layers, extracted so it is unit-testable in tests/
// test_components.cpp without CUDA/engine dependencies (same discipline as
// hyper_connections.hpp's Sinkhorn normalization and rope.hpp's RoPE).
//
// Reference: llama.cpp branch qwen4exp/qwen3.8-flash-next
// (src/models/qwen4exp.cpp's build_layer_attn_linear +
// src/models/delta-net-base.cpp's build_delta_net_autoregressive), ported
// op-for-op. See src/attention_layer.cpp's qwen4exp block comment for the
// parameterization notes (per-value-head scalar decay, tiled k/v head
// sharing, L2-normed q/k, sigmoid z-gate).
#pragma once
#include <cmath>
#include <cstring>
#include <vector>

// AVX-512 fast path for qwen4exp_gdn_delta_step below. This header is
// deliberately portable (tests/test_components.cpp compiles it on Apple
// Silicon too, per this file's own "no CUDA/engine dependencies" contract),
// so the intrinsics are guarded and the scalar implementation is kept as the
// fallback -- same discipline as kernels_cpu.cpp vs kernels_cpu_neon.cpp.
#if defined(__AVX512F__)
#include <immintrin.h>
#endif

// ggml_softplus: log1p(exp(x)), numerically stable.
inline float qwen4exp_softplus(float x) {
  return (x > 20.0f) ? x : std::log1p(std::exp(x));
}

// Per-head decay: g = softplus(alpha + dt) * a, with a = -exp(A_log) < 0
// (the "noscan" ssm_a convention), so g is a NEGATIVE log-decay and exp(g)
// is in (0, 1). alpha is the ssm_alpha projection output for this head.
inline float qwen4exp_gdn_decay_log(float alpha, float dt_bias, float ssm_a) {
  return qwen4exp_softplus(alpha + dt_bias) * ssm_a;
}

// ggml_l2_norm's exact formula: x / max(sqrt(sum(x^2)), eps) -- NOT
// x / sqrt(sum + eps) (verified in ggml-cpu/ops.cpp's
// ggml_compute_forward_l2_norm_f32).
inline void qwen4exp_l2_norm_inplace(float *v, int dim, float eps) {
  double ss = 0.0;
  for (int i = 0; i < dim; ++i)
    ss += (double)v[i] * v[i];
  const float inv = 1.0f / std::fmax(std::sqrt((float)ss), eps);
  for (int i = 0; i < dim; ++i)
    v[i] *= inv;
}

// One (token, value-head) gated-delta-rule step, matching
// build_delta_net_autoregressive op-for-op:
//   state[k][v] *= exp(g_log)               (decay, scalar for this head)
//   sk[v]        = sum_k state[k][v] * k[k]
//   state[k][v] += k[k] * (v[v] - sk[v]) * beta
//   out[v]       = sum_k state[k][v] * q[k] / sqrt(head_dim)
// st: [hd, hd] row-major ([key, value]), persistent across calls (mutated
// in place). q/k already L2-normed by the caller; beta already sigmoided.
// Tiny vector helpers shared by the chunked kernel below. Guarded like the
// rest of this header so it still compiles on Apple Silicon (test_components).
inline float gdn_dot(const float *a, const float *b, int n) {
#if defined(__AVX512F__)
  __m512 acc = _mm512_setzero_ps();
  int i = 0;
  for (; i + 16 <= n; i += 16)
    acc = _mm512_fmadd_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i), acc);
  float s = _mm512_reduce_add_ps(acc);
  for (; i < n; ++i) s += a[i] * b[i];
  return s;
#else
  float s = 0.0f;
  for (int i = 0; i < n; ++i) s += a[i] * b[i];
  return s;
#endif
}
inline void gdn_axpy(float *y, const float *x, float a, int n) {
#if defined(__AVX512F__)
  const __m512 va = _mm512_set1_ps(a);
  int i = 0;
  for (; i + 16 <= n; i += 16)
    _mm512_storeu_ps(y + i, _mm512_fmadd_ps(va, _mm512_loadu_ps(x + i),
                                            _mm512_loadu_ps(y + i)));
  for (; i < n; ++i) y[i] += a * x[i];
#else
  for (int i = 0; i < n; ++i) y[i] += a * x[i];
#endif
}
inline void gdn_scale(float *y, float a, int n) {
#if defined(__AVX512F__)
  const __m512 va = _mm512_set1_ps(a);
  int i = 0;
  for (; i + 16 <= n; i += 16)
    _mm512_storeu_ps(y + i, _mm512_mul_ps(va, _mm512_loadu_ps(y + i)));
  for (; i < n; ++i) y[i] *= a;
#else
  for (int i = 0; i < n; ++i) y[i] *= a;
#endif
}

// ---------------------------------------------------------------------------
// Chunkwise-parallel form of the SAME gated delta rule qwen4exp_gdn_delta_step
// implements token-by-token. NOT yet wired into production -- see the unit
// test in tests/test_components.cpp, which checks it against the sequential
// step (the oracle) before anything depends on it.
//
// WHY: llama.cpp runs two different algorithms for Gated DeltaNet --
// build_delta_net_autoregressive (sequential) ONLY when n_tokens == 1, and
// build_delta_net_chunking (chunk size 64) for prefill
// (buun-llama-cpp/src/models/delta-net-base.cpp:483). MoEStream uses the
// sequential form for both, so a 3,189-token prefill performs 3,189
// dependent state updates per head instead of ~50 chunk matmuls. Measured
// cost of that: 101.57s, 28% of a long prefill, the single largest bucket.
//
// The FLOP counts are nearly identical (~4.2M MACs per chunk per head either
// way). The win is memory traffic: the sequential form streams the whole
// hd*hd state (64 KB at hd=128, well past L1) four times per TOKEN -- 256
// passes per 64-token chunk -- while this form touches it about twice per
// CHUNK. It is also expressible as dense matmuls, so unlike the sequential
// recurrence it can eventually go on the GPU.
//
// DERIVATION (done from OUR sequential step, not transcribed from ggml, so
// that qwen4exp_gdn_delta_step stays the oracle and no ggml index/layout
// convention has to be guessed). With state S indexed [k][v], a_t=exp(g_t),
// A_t = prod_{j<=t} a_j = exp(cumsum(g)_t), and w_t = beta_t * k_t, the
// sequential step is exactly
//     S_t = a_t (I - beta_t k_t k_t^T) S_{t-1} + beta_t k_t v_t^T
// whose closed form over a chunk starting from S_0 is
//     S_t = A_t S_0 + sum_{i<=t} (A_t/A_i) w_i u_i^T
// Substituting that back into u_t = v_t - k_t^T (a_t S_{t-1}) gives a
// strictly-lower-triangular system in U, solved here by forward substitution:
//     (I + M) U = V - diag(A_t) (S_0^T K),  M[t][i] = (A_t/A_i) beta_i (k_t . k_i), i<t
// and the outputs / final state follow directly:
//     o_t   = q_scale [ A_t (S_0^T q_t) + sum_{i<=t} P[t][i] u_i ],  P uses i<=t and q_t.k_i
//     S_end = A_{C-1} S_0 + sum_i (A_{C-1}/A_i) beta_i k_i u_i^T
// A_t/A_i = exp(gcs[t]-gcs[i]) with t>=i and g<=0, so every exponential here
// is <= 1 -- no overflow path, unlike a naive exp(cumsum) ratio.
//
// Scalar on purpose for now: correctness first, against the oracle. The three
// hot pieces (K@S0, the P/M dot products, the rank-1 state accumulation) are
// all dense and vectorise the same way qwen4exp_gdn_delta_step just did.
inline void qwen4exp_gdn_delta_chunk(const float *Q, const float *K,
                                     const float *V, const float *beta,
                                     const float *g_log, float *st, float *out,
                                     int C, int hd) {
  if (C <= 0 || hd <= 0) return;
  const float q_scale = 1.0f / std::sqrt((float)hd);

  // Grow-only per-thread scratch: this is called once per (chunk, head,
  // layer) -- ~50 x 48 x 36 = 86,400 times per 3,189-token prefill -- so a
  // fresh allocation each time would just re-introduce a smaller version of
  // the per-token allocation problem the sequential step had.
  static thread_local std::vector<float> gcs_tls, R_tls, Qs_tls, U_tls, KK_tls, QK_tls;
  if ((int)gcs_tls.size() < C) gcs_tls.resize((size_t)C);
  const size_t need = (size_t)C * hd;
  if (R_tls.size() < need) { R_tls.resize(need); Qs_tls.resize(need); U_tls.resize(need); }
  const size_t need2 = (size_t)C * C;
  if (KK_tls.size() < need2) { KK_tls.resize(need2); QK_tls.resize(need2); }
  float *KK = KK_tls.data(), *QK = QK_tls.data();
  float *gcs = gcs_tls.data();
  {
    float acc = 0.0f;
    for (int t = 0; t < C; ++t) { acc += g_log[t]; gcs[(size_t)t] = acc; }
  }

  // exp() hoisting. The first chunked version evaluated
  // exp(gcs[t]-gcs[i]) inside BOTH O(C^2) inner loops: 354 MILLION exp calls
  // for a 3,189-token prefill versus 5.5 million for the sequential form
  // (64x more), which is a large part of why chunking showed no win at all
  // in its first three revisions.
  //
  // Since i<=t and g<=0, exp(gcs[t]-gcs[i]) = e[t]/e[i] with
  // e[x] = exp(gcs[x]-gcs[0]) in (0,1]. That is only safe while the chunk's
  // total decay span stays inside float's exponent range, so the span is
  // checked and the exact per-pair form is kept as the fallback -- a strongly
  // decaying chunk must not silently divide by an underflowed zero.
  static thread_local std::vector<float> e_tls;
  if ((int)e_tls.size() < C) e_tls.resize((size_t)C);
  float *ecum = e_tls.data();
  const float span = gcs[0] - gcs[(size_t)C - 1];
  const bool fast_exp = (span < 60.0f);
  if (fast_exp)
    for (int t = 0; t < C; ++t) ecum[t] = std::exp(gcs[(size_t)t] - gcs[0]);

  // Gram matrices, computed ONCE: KK[t][i] = k_t.k_i, QK[t][i] = q_t.k_i.
  // The first version recomputed these hd-length dots inside the forward-
  // substitution and output loops, i.e. O(C^2) dot calls; hoisting them costs
  // the same flops but turns them into two dense, cache-friendly passes.
  for (int t = 0; t < C; ++t) {
    const float *kt = K + (size_t)t * hd, *qt = Q + (size_t)t * hd;
    float *kkr = KK + (size_t)t * C, *qkr = QK + (size_t)t * C;
    for (int i = 0; i <= t; ++i) {
      const float *ki = K + (size_t)i * hd;
      kkr[i] = gdn_dot(kt, ki, hd);
      qkr[i] = gdn_dot(qt, ki, hd);
    }
  }

  // R = K @ S0, Qs = Q @ S0  (both [C, hd]); S0 is walked once, row by row.
  float *R = R_tls.data(), *Qs = Qs_tls.data();
  std::memset(R, 0, need * sizeof(float));
  std::memset(Qs, 0, need * sizeof(float));
  // Loop order matters more than vectorisation here. Iterating t outer / k
  // inner re-reads the WHOLE hd*hd state (64 KB at hd=128, past L1) once per
  // TOKEN -- C passes per chunk, which is precisely the traffic chunking
  // exists to remove, and it made the first chunked version no faster than
  // the sequential one despite identical flops. With k outer, each state row
  // (512 B) is loaded once and reused across all C tokens, so the state is
  // streamed exactly ONCE per chunk.
  for (int k = 0; k < hd; ++k) {
    const float *row = st + (size_t)k * hd;
    for (int t = 0; t < C; ++t) {
      gdn_axpy(R + (size_t)t * hd, row, K[(size_t)t * hd + k], hd);
      gdn_axpy(Qs + (size_t)t * hd, row, Q[(size_t)t * hd + k], hd);
    }
  }

  // U: forward substitution over the strictly-lower-triangular M.
  float *U = U_tls.data();
  for (int t = 0; t < C; ++t) {
    const float At = std::exp(gcs[(size_t)t]);
    float *ut = U + (size_t)t * hd;
    const float *vt = V + (size_t)t * hd, *rt = R + (size_t)t * hd;
    for (int v = 0; v < hd; ++v) ut[v] = vt[v] - At * rt[v];
    const float *kkr = KK + (size_t)t * C;
    for (int i = 0; i < t; ++i) {
      const float ratio = fast_exp ? (ecum[t] / ecum[i])
                                   : std::exp(gcs[(size_t)t] - gcs[(size_t)i]);
      const float m = ratio * beta[i] * kkr[i];
      if (m != 0.0f) gdn_axpy(ut, U + (size_t)i * hd, -m, hd);
    }
  }

  // o_t = q_scale [ A_t (S0^T q_t) + sum_{i<=t} P[t][i] u_i ]
  for (int t = 0; t < C; ++t) {
    const float At = std::exp(gcs[(size_t)t]);
    float *ot = out + (size_t)t * hd;
    const float *qst = Qs + (size_t)t * hd;
    for (int v = 0; v < hd; ++v) ot[v] = At * qst[v];
    const float *qkr = QK + (size_t)t * C;
    for (int i = 0; i <= t; ++i) {
      const float ratio = fast_exp ? (ecum[t] / ecum[i])
                                   : std::exp(gcs[(size_t)t] - gcs[(size_t)i]);
      const float pw = ratio * beta[i] * qkr[i];
      if (pw != 0.0f) gdn_axpy(ot, U + (size_t)i * hd, pw, hd);
    }
    for (int v = 0; v < hd; ++v) ot[v] *= q_scale;
  }

  // S_end = A_last S0 + sum_i (A_last/A_i) beta_i k_i u_i^T
  const float Alast = std::exp(gcs[(size_t)C - 1]);
  for (int k = 0; k < hd; ++k) gdn_scale(st + (size_t)k * hd, Alast, hd);
  // Same swap as the R/Qs loop above: k outer keeps each state row resident
  // while every chunk row of U (32 KB total, L2-resident) streams past it,
  // instead of walking the full 64 KB state once per token.
  {
    static thread_local std::vector<float> cf_tls;
    if ((int)cf_tls.size() < C) cf_tls.resize((size_t)C);
    float *cf = cf_tls.data();
    for (int i = 0; i < C; ++i)
      cf[i] = (fast_exp ? (ecum[C - 1] / ecum[i])
                        : std::exp(gcs[(size_t)C - 1] - gcs[(size_t)i])) * beta[i];
    for (int k = 0; k < hd; ++k) {
      float *row = st + (size_t)k * hd;
      for (int i = 0; i < C; ++i) {
        const float ck = cf[i] * K[(size_t)i * hd + k];
        if (ck != 0.0f) gdn_axpy(row, U + (size_t)i * hd, ck, hd);
      }
    }
  }
}

// Hot path: called once per (token, value-head, layer). A real 3,189-token
// qwen4exp prefill runs this 3189 x 48 heads x 36 GDN layers = ~5.5 MILLION
// times, and it measured 101.57s -- 28% of that prefill, the single largest
// bucket (see the "Qwen4Exp attention split" shutdown line). Three things
// were wrong with the original four-loop scalar version, fixed here:
//
//   1. `std::vector<float> sk(hd)` was constructed INSIDE the function, i.e.
//      ~5.5 million heap allocations per prefill. Now a stack buffer, with a
//      heap fallback only if hd exceeds kGdnMaxStackHd.
//   2. `(v_h[v] - sk[v])` is loop-invariant in k but was recomputed for
//      every one of the hd rows -- hd-fold redundant work. Hoisted into t[].
//   3. Four separate passes over the hd*hd state (64 KB at hd=128, so it
//      does not fit L1 and every pass is a real L2 round trip). Fused into
//      two: {decay, accumulate sk} and {rank-1 update, accumulate o}. Each
//      fusion is per-row and reads only the row it just wrote, so the
//      arithmetic order over k is unchanged.
//
// Deliberately kept bit-comparable to the original scalar form: t[] holds
// (v_h[v] - sk[v]) WITHOUT beta folded in, so the update below still
// evaluates (kk * t[v]) * beta -- the same association the original
// `kk * (v_h[v] - sk[v]) * beta` had. Folding beta into t or into kk would
// re-associate the product and perturb the last ulp for no real gain.
// Verified against tests/test_components.cpp's independent naive
// double-precision reference (`make test-components ARGS="qwen4exp"`).
inline void qwen4exp_gdn_delta_step(const float *q_h, const float *k_h,
                                    const float *v_h, float beta, float g_log,
                                    float *st, float *o_h, int hd) {
  const float q_scale = 1.0f / std::sqrt((float)hd);
  const float decay = std::exp(g_log);

  // hd is 128 for every real qwen4exp checkpoint (ssm.state_size); the heap
  // fallback exists so an unusual model can never overflow the stack.
  constexpr int kGdnMaxStackHd = 512;
  float sk_stack[kGdnMaxStackHd];
  float t_stack[kGdnMaxStackHd];
  std::vector<float> scratch_heap;
  float *sk, *t;
  if (hd <= kGdnMaxStackHd) {
    sk = sk_stack;
    t = t_stack;
  } else {
    scratch_heap.resize((size_t)hd * 2);
    sk = scratch_heap.data();
    t = scratch_heap.data() + hd;
  }
  std::memset(sk, 0, (size_t)hd * sizeof(float));
  std::memset(o_h, 0, (size_t)hd * sizeof(float));

#if defined(__AVX512F__)
  const __m512 vdecay = _mm512_set1_ps(decay);
  const __m512 vbeta = _mm512_set1_ps(beta);
  // Pass A: state *= decay, then sk = state^T @ k  (fused, one pass)
  for (int k = 0; k < hd; ++k) {
    float *row = st + (size_t)k * hd;
    const __m512 vkk = _mm512_set1_ps(k_h[k]);
    int v = 0;
    for (; v + 16 <= hd; v += 16) {
      const __m512 r = _mm512_mul_ps(_mm512_loadu_ps(row + v), vdecay);
      _mm512_storeu_ps(row + v, r);
      _mm512_storeu_ps(sk + v,
                       _mm512_fmadd_ps(r, vkk, _mm512_loadu_ps(sk + v)));
    }
    for (; v < hd; ++v) {
      row[v] *= decay;
      sk[v] += row[v] * k_h[k];
    }
  }
  // t = v - sk   (hoisted out of the k loop; beta applied below, see comment)
  {
    int v = 0;
    for (; v + 16 <= hd; v += 16)
      _mm512_storeu_ps(t + v, _mm512_sub_ps(_mm512_loadu_ps(v_h + v),
                                            _mm512_loadu_ps(sk + v)));
    for (; v < hd; ++v) t[v] = v_h[v] - sk[v];
  }
  // Pass B: state += beta * (k (x) t), then o = state^T @ q  (fused)
  for (int k = 0; k < hd; ++k) {
    float *row = st + (size_t)k * hd;
    const __m512 vkk = _mm512_set1_ps(k_h[k]);
    const __m512 vqq = _mm512_set1_ps(q_h[k] * q_scale);
    int v = 0;
    for (; v + 16 <= hd; v += 16) {
      const __m512 upd =
          _mm512_mul_ps(_mm512_mul_ps(vkk, _mm512_loadu_ps(t + v)), vbeta);
      const __m512 r = _mm512_add_ps(_mm512_loadu_ps(row + v), upd);
      _mm512_storeu_ps(row + v, r);
      _mm512_storeu_ps(o_h + v,
                       _mm512_fmadd_ps(r, vqq, _mm512_loadu_ps(o_h + v)));
    }
    for (; v < hd; ++v) {
      row[v] += (k_h[k] * t[v]) * beta;
      o_h[v] += row[v] * (q_h[k] * q_scale);
    }
  }
#else
  // Portable scalar fallback -- same two fused passes, same arithmetic order.
  for (int k = 0; k < hd; ++k) {
    float *row = st + (size_t)k * hd;
    const float kk = k_h[k];
    for (int v = 0; v < hd; ++v) {
      row[v] *= decay;
      sk[v] += row[v] * kk;
    }
  }
  for (int v = 0; v < hd; ++v) t[v] = v_h[v] - sk[v];
  for (int k = 0; k < hd; ++k) {
    float *row = st + (size_t)k * hd;
    const float kk = k_h[k];
    const float qq = q_h[k] * q_scale;
    for (int v = 0; v < hd; ++v) {
      row[v] += (kk * t[v]) * beta;
      o_h[v] += row[v] * qq;
    }
  }
#endif
}
