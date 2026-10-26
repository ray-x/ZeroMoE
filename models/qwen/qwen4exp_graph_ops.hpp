#pragma once

#include <vector>

#include "ggml.h"

#include <cmath>

namespace qwen4exp_ops {

struct HcShape {
  int D, HC, S;
  float eps;
};

inline ggml_tensor *hc_mix(const HcShape &g, ggml_context *ctx, ggml_tensor *x,
                          ggml_tensor *norm, ggml_tensor *down, ggml_tensor *up,
                          ggml_tensor *inject, ggml_tensor **inject_out) {
  ggml_tensor *xn = ggml_rms_norm(ctx, x, g.eps);
  xn = ggml_reshape_2d(ctx, xn, g.D * g.HC, g.S);
  xn = ggml_mul(ctx, xn, norm);
  ggml_tensor *lo = ggml_mul_mat(ctx, down, xn);
  lo = ggml_silu(ctx, ggml_scale(ctx, lo, 1.0f / static_cast<float>(g.HC)));
  ggml_tensor *gate_logits = ggml_mul_mat(ctx, up, lo);
  ggml_tensor *gate = ggml_sigmoid(ctx, gate_logits);
  ggml_tensor *gated = ggml_mul(ctx, xn, gate);
  ggml_tensor *g3 = ggml_reshape_3d(ctx, gated, g.D, g.HC, g.S);
  ggml_tensor *mixed = ggml_cont(
      ctx, ggml_view_2d(ctx, g3, g.D, g.S,
                       ggml_row_size(g3->type, g.D) * g.HC, 0));
  for (int h = 1; h < g.HC; ++h) {
    ggml_tensor *s = ggml_view_2d(
        ctx, g3, g.D, g.S, ggml_row_size(g3->type, g.D) * g.HC,
        ggml_row_size(g3->type, g.D) * h);
    mixed = ggml_add(ctx, mixed, s);
  }
  mixed = ggml_scale(ctx, mixed, 1.0f / static_cast<float>(g.HC));
  if (inject_out)
    *inject_out = inject ? ggml_mul_mat(ctx, inject, xn) : nullptr;
  return mixed;
}

inline ggml_tensor *hc_combine(const HcShape &g, ggml_context *ctx,
                              ggml_tensor *residual, ggml_tensor *block,
                              ggml_tensor *inject) {
  ggml_tensor *w = ggml_scale(
      ctx, ggml_sigmoid(ctx, ggml_scale(ctx, inject, 1.0f / g.HC)), 2.0f);
  w = ggml_reshape_3d(ctx, w, 1, g.HC, g.S);
  ggml_tensor *b = ggml_reshape_3d(ctx, block, g.D, 1, g.S);
  b = ggml_repeat_4d(ctx, b, g.D, g.HC, g.S, 1);
  return ggml_add(ctx, residual, ggml_mul(ctx, b, w));
}

inline ggml_tensor *shared_expert(ggml_context *ctx, ggml_tensor *cur,
                                 ggml_tensor *gate, ggml_tensor *up,
                                 ggml_tensor *down, ggml_tensor *scalar_gate) {
  if (!gate || !up || !down) return nullptr;
  ggml_tensor *gate_out = ggml_mul_mat(ctx, gate, cur);
  ggml_tensor *up_out = ggml_mul_mat(ctx, up, cur);
  ggml_tensor *act = ggml_swiglu_split(ctx, gate_out, up_out);
  ggml_tensor *out = ggml_mul_mat(ctx, down, act);
  if (scalar_gate) {
    ggml_tensor *sg = ggml_sigmoid(ctx, ggml_mul_mat(ctx, scalar_gate, cur));
    out = ggml_mul(ctx, out, sg);
  }
  return out;
}

struct AttentionShape {
  int H, Hkv, hd, capacity, n_rot, original_context;
  float eps, freq_base, freq_scale, ext_factor, attn_factor;
};

// Text-only IMRoPE uses the same NEOX substitution as the target graph.
inline ggml_tensor *attention_decode(
    const AttentionShape &g, ggml_context *ctx, ggml_tensor *cur,
    ggml_tensor *w_q, ggml_tensor *w_k, ggml_tensor *w_v, ggml_tensor *w_qn,
    ggml_tensor *w_kn, ggml_tensor *w_o, ggml_tensor *k_storage,
    ggml_tensor *v_storage, ggml_tensor *positions, ggml_tensor *indices,
    ggml_tensor *mask, std::vector<ggml_tensor *> *norm_nodes = nullptr) {
  ggml_tensor *qfull = ggml_mul_mat(ctx, w_q, cur);
  ggml_tensor *q = ggml_view_3d(ctx, qfull, g.hd, g.H, 1,
                               sizeof(float) * g.hd * 2,
                               sizeof(float) * g.hd * 2 * g.H, 0);
  ggml_tensor *q_rms = ggml_rms_norm(ctx, q, g.eps);
  q = ggml_mul(ctx, q_rms, w_qn);
  ggml_tensor *k = ggml_reshape_3d(
      ctx, ggml_mul_mat(ctx, w_k, cur), g.hd, g.Hkv, 1);
  ggml_tensor *k_rms = ggml_rms_norm(ctx, k, g.eps);
  k = ggml_mul(ctx, k_rms, w_kn);
  if (norm_nodes) norm_nodes->insert(norm_nodes->end(), {q_rms, k_rms});
  ggml_tensor *v = ggml_reshape_3d(
      ctx, ggml_mul_mat(ctx, w_v, cur), g.hd, g.Hkv, 1);
  q = ggml_rope_ext(ctx, q, positions, nullptr, g.n_rot, GGML_ROPE_TYPE_NEOX,
                    g.original_context, g.freq_base, g.freq_scale,
                    g.ext_factor, g.attn_factor, 32.0f, 1.0f);
  k = ggml_rope_ext(ctx, k, positions, nullptr, g.n_rot, GGML_ROPE_TYPE_NEOX,
                    g.original_context, g.freq_base, g.freq_scale,
                    g.ext_factor, g.attn_factor, 32.0f, 1.0f);
  ggml_tensor *k2 = ggml_reshape_2d(ctx, k, g.hd, g.Hkv);
  ggml_tensor *v2 = ggml_reshape_2d(ctx, v, g.hd, g.Hkv);
  ggml_tensor *kset = ggml_set_rows(ctx, k_storage, k2, indices);
  ggml_tensor *vset = ggml_set_rows(ctx, v_storage, v2, indices);
  ggml_tensor *kcache = ggml_reshape_4d(
      ctx, kset, g.hd, g.capacity, g.Hkv, 1);
  ggml_tensor *vcache = ggml_reshape_4d(
      ctx, vset, g.hd, g.capacity, g.Hkv, 1);
  q = ggml_permute(ctx, q, 0, 2, 1, 3);
  if (kcache->type == GGML_TYPE_F32)
    kcache = ggml_cast(ctx, kcache, GGML_TYPE_F16);
  if (vcache->type == GGML_TYPE_F32)
    vcache = ggml_cast(ctx, vcache, GGML_TYPE_F16);
  ggml_tensor *attn = ggml_flash_attn_ext(
      ctx, q, kcache, vcache, mask,
      1.0f / std::sqrt(static_cast<float>(g.hd)), 0.0f, 0.0f);
  ggml_tensor *gate = ggml_view_3d(
      ctx, qfull, g.hd, g.H, 1, sizeof(float) * g.hd * 2,
      sizeof(float) * g.hd * 2 * g.H, sizeof(float) * g.hd);
  gate = ggml_reshape_3d(ctx, ggml_cont_2d(ctx, gate, g.H * g.hd, 1),
                         g.hd, g.H, 1);
  attn = ggml_mul(ctx, attn, ggml_sigmoid(ctx, gate));
  attn = ggml_reshape_2d(ctx, attn, g.H * g.hd, 1);
  return ggml_mul_mat(ctx, w_o, attn);
}

} // namespace qwen4exp_ops
